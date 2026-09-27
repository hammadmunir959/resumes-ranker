"""The mock itself: its scoring heuristic, its transports, and its HTTP server.

The mock is not exercised by the rest of the suite in a way that would notice it
lying. A green ranking test only proves the pipeline agrees with the mock, so
these tests check the mock against plain expectations: that a resume with the
evidence scores higher than one without, and that a resume with none is not
promoted by noise.
"""

import asyncio
import socket
import threading
import time

import httpx

from app.config import Settings
from app.schemas import Questions
from app.services import JevClient
from app.schemas import Candidate, Criterion
from app.services import ResumeRanker
from app.utils import build_questions, build_state
from app.testing import (
    MOCK_PATH,
    MockTransport,
    ScriptedTransport,
    _Handler,
    _Server,
    coverage,
    mock_answers,
    mock_body,
    stable_jitter,
    words,
)

STRONG = "Senior engineer, 6 years Python, migrated a monolith to AWS. Authorized to work."
WEAK = "Designer, css and branding. No backend or cloud experience."
UNRELATED = "Baker, bread, ovens."


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class running_server:
    """A mock Decisions API on a real socket, for the duration of a block."""

    def __init__(self):
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}{MOCK_PATH}"
        self._server = _Server(("127.0.0.1", self.port), _Handler)

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


# --- Text helpers ----------------------------------------------------------- #

def test_words_drops_punctuation_and_short_noise():
    found = words("Senior engineer, 6 years Python (django)!")
    assert "senior" in found and "engineer" in found and "python" in found
    assert "django" in found
    assert "," not in found and "6" not in found
    assert "the" not in words("the best of the year")


def test_jitter_is_repeatable_and_in_range():
    for seed in ("a", "b", "candidate|gate|40"):
        first = stable_jitter(seed)
        assert first == stable_jitter(seed), "a demo must be reproducible"
        assert 0.0 <= first < 1.0


def test_different_seeds_give_different_jitter():
    values = {stable_jitter(f"cand_{i}|py|20") for i in range(20)}
    assert len(values) > 15, "jitter should not collapse to a few values"


# --- Coverage --------------------------------------------------------------- #

def test_coverage_is_zero_without_overlap():
    assert coverage(words(UNRELATED), "Python depth and testing") == 0.0


def test_coverage_grows_with_evidence():
    signal = "Python web frameworks testing"
    little = coverage(words("Some Python experience"), signal)
    lots = coverage(words("Python web frameworks testing all round"), signal)
    assert 0.0 < little < lots <= 1.0


def test_coverage_is_capped_at_one():
    assert coverage(words("Python web frameworks testing and more"), "Python web frameworks testing") == 1.0


def test_coverage_of_an_empty_signal_is_zero():
    assert coverage(words(STRONG), "") == 0.0


# --- Mock answers ----------------------------------------------------------- #

def answers_for(resume: str, criteria, candidate_id="c"):
    questions = build_questions(criteria)
    state = build_state(Candidate(candidate_id=candidate_id, resume_text=resume), criteria, "Backend")
    return mock_answers(state, questions), questions


def test_a_gate_passes_on_evidence_and_fails_without_it():
    criteria = [Criterion(id="auth", name="Authorized to work", required=True)]
    strong, _ = answers_for(STRONG, criteria)
    unrelated, _ = answers_for(UNRELATED, criteria)
    assert strong["auth"]["noul"] > 0.5
    assert unrelated["auth"]["noul"] < 0.5, (
        "a resume with no evidence must not pass a gate"
    )


def test_a_score_rewards_evidence():
    criteria = [Criterion(id="py", name="Python depth", weight=3.0)]
    strong, _ = answers_for(STRONG, criteria)
    unrelated, _ = answers_for(UNRELATED, criteria)
    assert strong["py"]["score"] > unrelated["py"]["score"]


def test_noise_alone_never_promotes_an_unrelated_resume():
    # Every id and resume length is tried, so a seed that happens to be kind to
    # one candidate cannot hide behind a single example.
    criteria = [Criterion(id="py", name="Python depth", weight=3.0)]
    for length in range(1, 40):
        resume = UNRELATED + " padding" * length
        answers, _ = answers_for(resume, criteria, candidate_id=f"c{length}")
        assert answers["py"]["score"] == 0.0, (
            f"{length}: zero evidence scored above 'Not met'"
        )


def test_a_strong_resume_reaches_the_top_of_the_rubric():
    criteria = [Criterion(
        id="py", name="Python depth", description="web frameworks testing", weight=3.0,
    )]
    resume = "Python depth, web frameworks, testing, django and aws."
    answers, _ = answers_for(resume, criteria)
    assert answers["py"]["score"] == 2.0


def test_the_score_legend_matches_the_requested_rubric():
    criteria = [
        Criterion(
            id="py", name="Python depth", weight=1.0,
            rubric=("none", "some", "lots", "expert"),
        )
    ]
    answers, _ = answers_for(STRONG, criteria)
    assert answers["py"]["legend"] == {
        "0": "none", "1": "some", "2": "lots", "3": "expert",
    }
    assert 0 <= answers["py"]["score"] <= 3


def test_answers_are_deterministic():
    criteria = [Criterion(id="py", name="Python depth", weight=3.0)]
    first, _ = answers_for(STRONG, criteria)
    second, _ = answers_for(STRONG, criteria)
    assert first == second


def test_a_question_with_no_matching_requirement_falls_back_to_its_instructions():
    question = Questions.score("Rate the Python experience level", ["a", "b", "c"])
    answers = mock_answers({"resume": STRONG, "candidate_id": "c"}, {"py": question})
    assert "py" in answers, "a bare state must still produce an answer"
    assert answers["py"]["type"] == "score"


def test_a_state_given_as_plain_text_still_answers():
    question = Questions.noul("Is this authorized?", true_criteria="authorized to work")
    answers = mock_answers(STRONG, {"auth": question})
    assert answers["auth"]["type"] == "noul"


def test_a_choice_question_is_answered_from_the_offered_options():
    # A criterion always becomes a noul or score question, so the choice branch
    # is only reachable by a caller building questions directly.
    question = Questions.choice("Pick a level", {"junior": "junior", "senior": "senior"})
    answers = mock_answers({"resume": STRONG, "candidate_id": "c"}, {"level": question})
    assert answers["level"]["type"] == "choice"
    assert answers["level"]["choice"] in {"junior", "senior"}


def test_an_unknown_question_type_is_skipped_rather_than_crashing():
    answers = mock_answers(
        {"resume": STRONG, "candidate_id": "c"},
        {"weird": {"type": "future", "instructions": "unclear"}},
    )
    assert answers == {}, "an unrecognized type is left unanswered, not faked"


# --- Mock body -------------------------------------------------------------- #

def test_a_mock_body_has_the_documented_shape():
    criteria = [Criterion(id="auth", name="Authorized to work", required=True)]
    payload = {
        "model": "typesafe/jev-1.13",
        "state": build_state(Candidate(candidate_id="a", resume_text=STRONG), criteria, "Backend"),
        "questions": build_questions(criteria),
    }
    body = mock_body(payload)
    assert body["model"] == "typesafe/jev-1.13"
    assert set(body["answers"]) == {"auth"}
    assert set(body["usage"]) == {"input_tokens", "output_tokens", "cost"}
    assert body["usage"]["cost"] > 0


# --- Transports ------------------------------------------------------------- #

async def decide_with(transport):
    settings = Settings.build(jev_max_rps=1000.0)
    client = JevClient(settings, transport=transport, sleep=_no_sleep, jitter=lambda: 0.0)
    try:
        return await client.decide(
            state={"candidate_id": "a", "resume": STRONG, "requirements": []},
            questions=build_questions([Criterion(id="auth", name="Authorized", required=True)]),
        )
    finally:
        await client.aclose()


async def _no_sleep(_: float) -> None:
    pass


async def test_the_mock_transport_answers_in_process():
    response = await decide_with(MockTransport())
    assert response.answers
    assert response.cost_usd is not None


async def test_the_mock_transport_records_the_payloads_it_received():
    transport = MockTransport()
    await decide_with(transport)
    assert len(transport.requests) == 1
    assert "questions" in transport.requests[0]
    assert "state" in transport.requests[0]


async def test_the_mock_transport_can_be_told_to_fail():
    transport = MockTransport(lambda payload: (503, {"error": "down"}))
    settings = Settings.build(jev_max_rps=1000.0, jev_max_retries=1, jev_backoff_base=0.01)
    client = JevClient(settings, transport=transport, sleep=_no_sleep, jitter=lambda: 0.0)
    try:
        await client.decide(state={"resume": STRONG}, questions=build_questions(
            [Criterion(id="a", name="A", required=True)]
        ))
    except Exception as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("a scripted failure should surface")


async def test_the_scripted_transport_repeats_its_last_response():
    transport = ScriptedTransport([
        httpx.Response(500, json={"error": "down"}),
        httpx.Response(200, json=mock_body({
            "state": {"candidate_id": "a", "resume": STRONG, "requirements": []},
            "questions": build_questions([Criterion(id="auth", name="Authorized", required=True)]),
        })),
    ])
    settings = Settings.build(jev_max_rps=1000.0, jev_backoff_base=0.01)
    client = JevClient(settings, transport=transport, sleep=_no_sleep, jitter=lambda: 0.0)
    try:
        response = await client.decide(
            state={"candidate_id": "a", "resume": STRONG, "requirements": []},
            questions=build_questions([Criterion(id="auth", name="Authorized", required=True)]),
        )
    finally:
        await client.aclose()
    assert transport.calls == 2, "one failure, then a success"
    assert response.attempts == 2


async def test_the_scripted_transport_can_raise_a_transport_error():
    transport = ScriptedTransport([httpx.ConnectError("refused")])
    settings = Settings.build(jev_max_rps=1000.0, jev_max_retries=1, jev_backoff_base=0.01)
    client = JevClient(settings, transport=transport, sleep=_no_sleep, jitter=lambda: 0.0)
    try:
        await client.decide(state={"resume": STRONG}, questions=build_questions(
            [Criterion(id="a", name="A", required=True)]
        ))
    except Exception as exc:
        assert "refused" in str(exc)
    else:
        raise AssertionError("a transport error should surface")


def test_an_empty_script_is_rejected():
    try:
        ScriptedTransport([])
    except ValueError:
        pass
    else:
        raise AssertionError("an empty script would replay nothing")


# --- The local HTTP server -------------------------------------------------- #

def test_the_server_answers_over_a_real_socket():
    with running_server() as server:
        response = httpx.post(server.base_url, json={
            "state": {"candidate_id": "a", "resume": STRONG, "requirements": []},
            "questions": build_questions([Criterion(id="auth", name="Authorized", required=True)]),
        }, timeout=5.0)
    assert response.status_code == 200
    assert "answers" in response.json()


def test_the_server_handles_concurrent_requests():
    # A pool is ranked concurrently, so a server that handles one connection at a
    # time deadlocks here: the second client waits until it times out. The short
    # timeout keeps that failure quick instead of hanging the suite.
    with running_server() as server:
        settings = Settings.build(
            jev_base_url=server.base_url, jev_max_rps=1000.0, jev_timeout=5.0,
        )
        ranker = ResumeRanker(settings)
        criteria = [
            Criterion(id="auth", name="Authorized to work", required=True),
            Criterion(id="py", name="Python depth", weight=3.0),
        ]
        pool = [
            Candidate(candidate_id=f"c{i}", resume_text=f"{STRONG} Candidate {i}.")
            for i in range(6)
        ]

        # Ranked and closed in one event loop: the httpx client is bound to the
        # loop that made the requests, so closing it from another one fails.
        async def rank():
            try:
                return await ranker.rank_batch("Backend", criteria, pool)
            finally:
                await ranker.aclose()

        result = asyncio.run(rank())
    assert len(result.results) == 6
    assert all(r.error is None for r in result.results), "no request should time out"
    assert result.total_attempts == 6


def test_the_server_survives_a_client_that_hangs_up():
    with running_server() as server:
        with httpx.Client(timeout=5.0) as client:
            # A dropped connection must not take the server down with it.
            for _ in range(3):
                assert client.post(server.base_url, json={
                    "state": {"candidate_id": "a", "resume": STRONG, "requirements": []},
                    "questions": build_questions(
                        [Criterion(id="auth", name="Authorized", required=True)]
                    ),
                }).status_code == 200
            with client.stream("POST", server.base_url, json={}) as abandoned:
                abandoned.close()
        assert httpx.get(f"http://127.0.0.1:{server.port}/", timeout=5.0).status_code >= 0
