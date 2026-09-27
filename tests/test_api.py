"""HTTP behavior: routing, status codes, and the shape of every response."""

import os
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

from app import config
from app.api.limiter import limiter
from app.main import create_app
from app.config import Settings, get_settings
from app.exceptions import (
    ConfigError,
    InputValidationError,
    JevAuthError,
    JevCreditsError,
    JevProtocolError,
    JevTransientError,
)
from app.services import JevClient
from app.services import ResumeRanker
from app.testing import MockTransport, ScriptedTransport

CRITERIA = [
    {"id": "gate", "name": "Authorized", "description": "authorized to work", "required": True},
    {"id": "py", "name": "Python", "description": "python web frameworks", "weight": 2.0},
]
RESUME = "Authorized to work. Six years of python with django."


async def noop_sleep(_: float) -> None:
    pass


def build(transport=None, **overrides):
    """An app wired to a transport, ready to be called without a server."""
    values = {"jev_max_rps": 1000.0, "jev_backoff_base": 0.01}
    values.update(overrides)
    settings = Settings.build(**values)
    jev = JevClient(
        settings, transport=transport or MockTransport(),
        sleep=noop_sleep, jitter=lambda: 0.0,
    )
    return create_app(ranker=ResumeRanker(settings, jev))


def single_body(**overrides):
    body = {
        "job_description": "Backend engineer",
        "criteria": CRITERIA,
        "candidate": {"candidate_id": "a", "resume_text": RESUME},
    }
    body.update(overrides)
    return body


def batch_body(count=2, **overrides):
    body = {
        "job_description": "Backend engineer",
        "criteria": CRITERIA,
        "candidates": [
            {"candidate_id": f"c{i}", "resume_text": f"{RESUME} Candidate {i}."}
            for i in range(count)
        ],
    }
    body.update(overrides)
    return body


def call(app, method, path, body=None):
    """Run one request through a complete app lifecycle, then shut it down.

    ``TestClient`` is used as a context manager because that is what enters the
    ASGI lifespan, which is where this app resolves settings and builds the
    ranker. Opening a client per call also scopes the lifespan to one request, so
    no state leaks between tests.
    """
    with TestClient(app) as client:
        return client.request(method, path, json=body)


def get(app, path):
    return call(app, "GET", path)


def post(app, path, body):
    return call(app, "POST", path, body)


@contextmanager
def _no_key_configured():
    """Hide every source of the API key, so validation genuinely fails.

    The ``.env`` lookup is swapped out rather than just uncached, because
    clearing the cache would re-read the file and hand back the real key.
    """
    saved_env = {name: os.environ.pop(name, None) for name in Settings.api_key_names}
    saved_dotenv = config._dotenv_values
    config._dotenv_values = lambda: {}
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in saved_env.items():
            if value is not None:
                os.environ[name] = value
        config._dotenv_values = saved_dotenv
        get_settings.cache_clear()


# --- Health ----------------------------------------------------------------- #

def test_health_is_ok():
    response = get(build(), "/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_the_loaded_configuration():
    body = get(build(jev_model="custom/model"), "/health").json()
    assert body["model"] == "custom/model"
    assert body["base_url"].startswith("https://openrouter.ai")
    assert body["key_configured"] is True


def test_health_flags_a_mock_key():
    assert get(build(), "/health").json()["mock_key"] is True


def test_health_never_exposes_the_key():
    body = get(build(), "/health").text
    assert "mock-key-not-used" not in body
    assert "api_key" not in body


def test_health_does_not_call_the_provider():
    transport = ScriptedTransport([httpx.Response(500, json={"error": "no"})])
    # A health check that depended on OpenRouter would turn their outage into ours.
    assert get(build(transport), "/health").status_code == 200
    assert transport.calls == 0


# --- Ranking ---------------------------------------------------------------- #

def test_single_returns_a_ranking_result():
    response = post(build(), "/rank/single", single_body())
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "typesafe/jev-1.13"
    assert len(body["results"]) == 1
    assert body["results"][0]["candidate_id"] == "a"


def test_single_and_batch_share_one_response_shape():
    single = post(build(), "/rank/single", single_body()).json()
    batch = post(build(), "/rank/batch", batch_body(1)).json()
    assert set(single) == set(batch), "clients should not handle two formats"


def test_batch_ranks_every_candidate():
    body = post(build(), "/rank/batch", batch_body(3)).json()
    assert len(body["results"]) == 3
    assert {r["candidate_id"] for r in body["results"]} == {"c0", "c1", "c2"}


def test_the_per_criterion_breakdown_is_returned():
    result = post(build(), "/rank/single", single_body()).json()["results"][0]
    assert {o["criterion_id"] for o in result["per_criterion"]} == {"gate", "py"}


def test_cost_and_timing_are_reported():
    body = post(build(), "/rank/batch", batch_body(2)).json()
    assert body["total_cost_usd"] > 0
    assert body["elapsed_seconds"] >= 0


def test_gated_candidates_are_ordered_last():
    body = post(build(), "/rank/batch", batch_body(2)).json()
    results = body["results"]
    if any(not r["gate_passed"] for r in results):
        assert results[0]["gate_passed"] is True


# --- Request validation ----------------------------------------------------- #

def test_a_malformed_body_is_rejected():
    response = post(build(), "/rank/single", {"job_description": "x"})
    assert response.status_code == 422, "FastAPI should reject the shape itself"


def test_a_blank_job_description_is_rejected():
    assert post(build(), "/rank/single", single_body(job_description="  ")).status_code == 422


def test_an_empty_criteria_list_is_rejected():
    assert post(build(), "/rank/single", single_body(criteria=[])).status_code == 422


def test_an_invalid_criterion_is_rejected():
    bad = [dict(CRITERIA[0], rubric=["only one level"])]
    assert post(build(), "/rank/single", single_body(criteria=bad)).status_code == 422


def test_an_optional_criterion_without_a_weight_is_rejected():
    bad = [{"id": "a", "name": "A", "weight": 0}]
    assert post(build(), "/rank/single", single_body(criteria=bad)).status_code == 422


def test_duplicate_candidate_ids_are_rejected():
    candidates = [{"candidate_id": "a", "resume_text": "x"}] * 2
    assert post(build(), "/rank/batch", batch_body(candidates=candidates)).status_code == 422


def test_an_unknown_field_is_rejected():
    assert post(build(), "/rank/single", single_body(extra=1)).status_code == 422


def test_a_batch_past_the_limit_is_rejected():
    body = batch_body(101)
    assert post(build(), "/rank/batch", body).status_code == 422


# --- Error mapping ---------------------------------------------------------- #

def error_client(exc: Exception):
    """An app whose ranker always fails, to check the status mapping."""
    class Broken(ResumeRanker):
        async def rank_batch(self, *args, **kwargs):
            raise exc

        async def rank_single(self, *args, **kwargs):
            raise exc

    settings = Settings.build(jev_max_rps=1000.0)
    return create_app(ranker=Broken(settings, JevClient(settings, transport=MockTransport())))


def test_out_of_credit_is_reported_as_402():
    response = post(error_client(JevCreditsError("Insufficient credits.")), "/rank/single", single_body())
    assert response.status_code == 402
    assert response.json()["error"] == "jev_out_of_credit"


def test_a_bad_key_is_reported_as_502():
    response = post(error_client(JevAuthError("bad key")), "/rank/single", single_body())
    assert response.status_code == 502
    assert response.json()["error"] == "jev_auth_failed"


def test_a_transient_failure_is_reported_as_503():
    response = post(error_client(JevTransientError("busy")), "/rank/single", single_body())
    assert response.status_code == 503
    assert response.json()["error"] == "jev_temporarily_unavailable"


def test_a_bad_response_is_reported_as_502():
    response = post(error_client(JevProtocolError("garbage")), "/rank/single", single_body())
    assert response.status_code == 502


def test_a_bad_request_is_reported_as_400():
    response = post(error_client(InputValidationError("nope")), "/rank/single", single_body())
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_every_ranker_error_maps_without_a_handler_per_class():
    # One handler covers the hierarchy because each class carries its own status.
    cases = {
        InputValidationError("x"): 400,
        ConfigError("x"): 503,
        JevAuthError("x"): 502,
        JevCreditsError("x"): 402,
        JevProtocolError("x"): 502,
        JevTransientError("x"): 503,
    }
    for error, expected in cases.items():
        response = post(error_client(error), "/rank/single", single_body())
        assert response.status_code == expected, f"{type(error).__name__}: {response.status_code}"


def test_an_upstream_status_is_included():
    error = JevCreditsError("Insufficient credits.", status_code=402)
    response = post(error_client(error), "/rank/single", single_body())
    assert response.json()["upstream_status"] == 402


# --- Wiring ----------------------------------------------------------------- #

def test_a_misconfigured_service_still_answers_health():
    # The app must boot and report, rather than crash-looping at import.
    with _no_key_configured():
        response = get(create_app(), "/health")
    assert response.status_code == 200
    assert response.json()["key_configured"] is False, (
        "a missing key must not be reported as configured"
    )


def test_a_misconfigured_service_reports_the_reason_on_ranking():
    with _no_key_configured():
        response = post(create_app(), "/rank/single", single_body())
    assert response.status_code == 503
    assert response.json()["error"] == "not_configured"


def test_an_injected_ranker_is_used():
    calls = []

    class Counting(ResumeRanker):
        async def rank_batch(self, *args, **kwargs):
            calls.append(1)
            return await super().rank_batch(*args, **kwargs)

    settings = Settings.build(jev_max_rps=1000.0)
    app = create_app(
        ranker=Counting(settings, JevClient(settings, transport=MockTransport()))
    )
    post(app, "/rank/batch", batch_body(1))
    assert calls, "the injected ranker should have been called"


def test_the_app_uses_the_configured_model():
    body = post(build(jev_model="custom/model"), "/rank/single", single_body()).json()
    assert body["model"] == "custom/model"


def test_the_batch_size_limit_is_reported_by_health():
    assert get(build(jev_max_batch_size=7), "/health").json()["max_batch_size"] == 7


def test_openapi_schema_builds():
    app = create_app()
    schema = app.openapi()
    assert "/rank/single" in schema["paths"]
    assert "/rank/batch" in schema["paths"]
    assert "/health" in schema["paths"]


def test_rate_limit_exceeded_returns_429():
    app = build()
    try:
        with TestClient(app) as client:
            statuses = [
                client.post("/rank/single", json=single_body()).status_code
                for _ in range(65)
            ]
        assert 429 in statuses
    finally:
        limiter.reset()

