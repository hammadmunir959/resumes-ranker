"""The Jev client: request shape, rate limiting, retries, and error mapping."""

import asyncio

import httpx

from app.config import Settings
from app.exceptions import (
    JevAuthError,
    JevCreditsError,
    JevError,
    JevProtocolError,
    JevTransientError,
)
from app.jev_client import JevClient, JevSyncClient, Questions, RateLimiter
from app.testing import MockTransport, ScriptedTransport, mock_body

QUESTION = {"g": {"type": "noul", "instructions": "Meets it?"}}
ANSWER = {
    "model": "typesafe/jev-1.13",
    "answers": {"g": {"type": "noul", "noul": 0.9, "confidence": 0.8}},
    "usage": {"input_tokens": 120, "output_tokens": 10, "cost": 0.0005},
}


def settings(**overrides) -> Settings:
    values = {"jev_max_rps": 1000.0, "jev_backoff_base": 0.01}
    values.update(overrides)
    return Settings.build(**values)


async def noop_sleep(_: float) -> None:
    """Stand-in for asyncio.sleep so retry tests do not wait."""


def client(transport, **overrides) -> JevClient:
    return JevClient(
        settings(**overrides), transport=transport,
        sleep=noop_sleep, jitter=lambda: 0.0,
    )


def _reject(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        return True
    return False


# --- Question builders ------------------------------------------------------ #

def test_noul_question_carries_both_criteria():
    question = Questions.noul("Meets it?", true_criteria="5y exp", false_criteria="nope")
    assert question["type"] == "noul"
    assert question["criteria"] == {"true": "5y exp", "false": "nope"}


def test_noul_question_without_criteria_omits_them():
    assert "criteria" not in Questions.noul("Meets it?")


def test_score_question_keeps_rubric_order():
    question = Questions.score("Level?", ["low", "mid", "high"])
    assert question["criteria"] == ["low", "mid", "high"]


def test_score_question_needs_two_levels():
    assert _reject(Questions.score, "Level?", ["only"])


def test_choice_question_requires_options():
    assert _reject(Questions.choice, "Which?", {})


# --- Request shape ---------------------------------------------------------- #

def test_request_uses_the_documented_wire_keys():
    # The Decisions API expects "questions", not "decision".
    transport = MockTransport()
    asyncio.run(
        client(transport).decide(state={"resume": "x"}, questions=QUESTION)
    )
    payload = transport.requests[0]
    assert set(payload) >= {"model", "state", "questions"}
    assert "decision" not in payload, "the API field is 'questions'"
    assert payload["questions"] == QUESTION


def test_model_defaults_to_the_configured_one():
    transport = MockTransport()
    asyncio.run(client(transport).decide(state="x", questions=QUESTION))
    assert transport.requests[0]["model"] == "typesafe/jev-1.13"


def test_model_can_be_overridden_per_call():
    transport = MockTransport()
    asyncio.run(
        client(transport).decide(state="x", questions=QUESTION, model="other/model")
    )
    assert transport.requests[0]["model"] == "other/model"


def test_at_least_one_question_is_required():
    assert _reject(asyncio.run, client(MockTransport()).decide(state="x", questions={}))


def test_authorization_header_is_sent():
    seen = {}

    class Capture(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            seen.update(request.headers)
            return httpx.Response(200, json=ANSWER)

    asyncio.run(client(Capture()).decide(state="x", questions=QUESTION))
    assert seen["authorization"] == "Bearer mock-key-not-used"
    assert seen["x-openrouter-title"] == "resumes-ranker"


def test_session_id_is_passed_through():
    transport = MockTransport()
    asyncio.run(
        client(transport).decide(state="x", questions=QUESTION, session_id="s" * 300)
    )
    assert len(transport.requests[0]["session_id"]) == 256, "should be truncated"


# --- Response parsing ------------------------------------------------------- #

def test_response_is_parsed_with_usage_and_attempts():
    response = JevClient._parse(ANSWER, attempts=2)
    assert response.noul("g") == 0.9
    assert response.model == "typesafe/jev-1.13"
    assert response.cost_usd == 0.0005
    assert response.input_tokens == 120
    assert response.output_tokens == 10
    assert response.attempts == 2


def test_a_body_without_answers_is_a_protocol_error():
    assert _reject(JevClient._parse, {"model": "m"})


def test_a_non_object_body_is_a_protocol_error():
    assert _reject(JevClient._parse, ["not", "a", "dict"])


def test_a_success_that_is_not_json_is_a_protocol_error():
    # A 200 of HTML means a proxy is broken; retrying will not parse it either.
    class Html(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, text="<html>gateway</html>")

    assert _reject(asyncio.run, client(Html()).decide(state="x", questions=QUESTION))


# --- Error classification --------------------------------------------------- #

def classify(status: int, body=None, headers=None) -> JevError:
    response = httpx.Response(status, json=body, headers=headers or {})
    return client(MockTransport()).classify(response)


def test_429_is_transient():
    assert isinstance(classify(429), JevTransientError)


def test_5xx_is_transient():
    for status in (500, 502, 503):
        assert isinstance(classify(status), JevTransientError), f"{status}"


def test_401_and_403_are_auth_errors():
    for status in (401, 403):
        error = classify(status)
        assert isinstance(error, JevAuthError), f"{status}"
        assert error.retriable is False, f"{status} must not be retried"


def test_plain_402_is_a_credits_error():
    error = classify(402, {"error": {"message": "Insufficient credits."}})
    assert isinstance(error, JevCreditsError)
    assert error.http_status == 402
    assert error.retriable is False, "an empty balance never clears itself"


def test_402_from_the_in_flight_budget_is_transient():
    # This cap clears on its own, so it is worth waiting out.
    error = classify(
        402,
        {"error": {"message": "cap", "metadata": {"limit_source": "openrouter_in_flight_budget"}}},
    )
    assert isinstance(error, JevTransientError)


def test_other_4xx_is_a_protocol_error():
    for status in (400, 404, 422):
        assert isinstance(classify(status), JevProtocolError), f"{status}"


def test_retry_after_header_is_parsed():
    assert classify(429, headers={"Retry-After": "7"}).retry_after == 7.0
    assert classify(429, headers={"Retry-After": "soon"}).retry_after is None
    assert classify(429, headers={"Retry-After": "-3"}).retry_after == 0.0


def test_upstream_message_is_surfaced():
    error = classify(402, {"error": {"message": "Insufficient credits. Buy more."}})
    assert "Insufficient credits" in str(error)


# --- Retries ---------------------------------------------------------------- #

def test_a_transient_failure_is_retried():
    transport = ScriptedTransport([(503, {"error": {"message": "down"}}), ANSWER])
    response = asyncio.run(client(transport).decide(state="x", questions=QUESTION))
    assert transport.calls == 2
    assert response.noul("g") == 0.9
    assert response.attempts == 2, "the attempt count should be reported"


def test_a_timeout_is_retried():
    transport = ScriptedTransport([httpx.TimeoutException("slow"), ANSWER])
    asyncio.run(client(transport).decide(state="x", questions=QUESTION))
    assert transport.calls == 2


def test_a_transport_error_is_retried():
    transport = ScriptedTransport([httpx.ConnectError("refused"), ANSWER])
    asyncio.run(client(transport).decide(state="x", questions=QUESTION))
    assert transport.calls == 2


def test_auth_failure_is_not_retried():
    transport = ScriptedTransport([(401, {"error": {"message": "bad key"}})])
    assert _reject(asyncio.run, client(transport).decide(state="x", questions=QUESTION))
    assert transport.calls == 1, "a bad key fails identically every time"


def test_out_of_credit_is_not_retried():
    transport = ScriptedTransport([(402, {"error": {"message": "no credits"}})])
    assert _reject(asyncio.run, client(transport).decide(state="x", questions=QUESTION))
    assert transport.calls == 1


def test_retries_are_bounded_and_then_reported():
    transport = ScriptedTransport([(503, {"error": {"message": "down"}})])
    jev = client(transport, jev_max_retries=3)
    assert _reject(asyncio.run, jev.decide(state="x", questions=QUESTION))
    assert transport.calls == 3


def test_retry_after_overrides_the_backoff():
    delays = []

    async def record(delay: float) -> None:
        delays.append(delay)

    transport = ScriptedTransport(
        [httpx.Response(429, json={"error": {"message": "slow"}},
                        headers={"Retry-After": "5"}), ANSWER]
    )
    jev = JevClient(
        settings(), transport=transport, sleep=record, jitter=lambda: 0.0
    )
    asyncio.run(jev.decide(state="x", questions=QUESTION))
    assert delays == [5.0], "Retry-After should be honoured verbatim"


def test_backoff_grows_and_is_capped():
    jev = client(MockTransport(), jev_backoff_base=1.0, jev_backoff_max=4.0)
    seen = [jev._backoff(attempt, None) for attempt in (1, 2, 3, 4, 5)]
    # jitter is pinned to 0.0, so each delay is half the nominal value
    assert seen == [0.5, 1.0, 2.0, 2.0, 2.0], seen


# --- Rate limiting ---------------------------------------------------------- #

def test_rate_limiter_waits_outside_the_lock():
    # A fake clock makes the window deterministic and instant.
    now = [0.0]
    waits = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, period=1.0, clock=lambda: now[0], sleep=sleep)

    async def scenario():
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()  # third in the same second must wait

    asyncio.run(scenario())
    assert len(waits) == 1, "only the third acquisition should have waited"
    assert waits[0] > 0


def test_rate_limiter_rejects_a_non_positive_rate():
    assert _reject(RateLimiter, 0)


def test_rate_limiter_releases_the_window_over_time():
    now = [0.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    limiter = RateLimiter(1, period=1.0, clock=lambda: now[0], sleep=sleep)

    async def scenario():
        await limiter.acquire()
        await limiter.acquire()  # waits out the window

    asyncio.run(scenario())
    assert now[0] >= 1.0


# --- Construction and cleanup ----------------------------------------------- #

def test_from_settings_defaults_to_the_process_settings():
    assert isinstance(JevClient.from_settings(Settings.build()), JevClient)


def test_context_manager_closes_the_client():
    async def scenario():
        async with JevClient(Settings.build(), transport=MockTransport()) as jev:
            await jev.decide(state="x", questions=QUESTION)

    asyncio.run(scenario())


def test_sync_client_wraps_the_async_one():
    sync = JevSyncClient(Settings.build())
    try:
        assert isinstance(sync.client, JevClient)
    finally:
        sync.close()


# --- The shared mock -------------------------------------------------------- #

def test_mock_answers_are_deterministic():
    first = mock_body({"state": {"resume": "python dev"}, "questions": QUESTION})
    second = mock_body({"state": {"resume": "python dev"}, "questions": QUESTION})
    assert first == second


def test_mock_body_is_well_formed():
    body = mock_body({"state": {"resume": "x"}, "questions": QUESTION})
    assert "answers" in body and "usage" in body
    assert body["usage"]["cost"] > 0
