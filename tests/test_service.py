"""Question building, aggregation, and the batch behavior of the ranker."""

import asyncio

import httpx

from app.config import Settings
from app.exceptions import (
    InputValidationError,
    JevAuthError,
    JevCreditsError,
    JevTransientError,
)
from app.jev_client import JevClient
from app.schemas import Candidate, CandidateScore, Criterion, JevResponse, RankingResult
from app.service import (
    GATE_MARGIN,
    GATE_THRESHOLD,
    REVIEW_CONFIDENCE_FLOOR,
    ResumeRanker,
    aggregate,
    build_questions,
    build_state,
    estimate_tokens,
    sort_scores,
)
from app.testing import MockTransport, ScriptedTransport

GATE = Criterion(id="gate", name="Authorized", description="authorized to work", required=True)
WEIGHTED = Criterion(id="py", name="Python", description="python web frameworks", weight=2.0)
OTHER = Criterion(id="cloud", name="Cloud", description="aws or gcp", weight=1.0)
CRITERIA = [GATE, WEIGHTED, OTHER]

RESUME = "Authorized to work. Six years of python with aws and gcp deployments."


async def noop_sleep(_: float) -> None:
    pass


def ranker(transport=None, **overrides) -> ResumeRanker:
    values = {"jev_max_rps": 1000.0, "jev_backoff_base": 0.01}
    values.update(overrides)
    settings = Settings.build(**values)
    return ResumeRanker(
        settings,
        JevClient(settings, transport=transport or MockTransport(),
                  sleep=noop_sleep, jitter=lambda: 0.0),
    )


def answer(**overrides) -> JevResponse:
    answers = {
        "gate": {"type": "noul", "noul": 0.95, "confidence": 0.9},
        "py": {"type": "score", "score": 2.0, "confidence": 0.8},
        "cloud": {"type": "score", "score": 0.0, "confidence": 0.8},
    }
    answers.update(overrides)
    return JevResponse(answers=answers, cost_usd=0.001, input_tokens=100)


# --- Question building ------------------------------------------------------ #

def test_a_required_criterion_becomes_a_noul_question():
    assert build_questions([GATE])["gate"]["type"] == "noul"


def test_an_optional_criterion_becomes_a_score_question():
    question = build_questions([WEIGHTED])["py"]
    assert question["type"] == "score"
    assert list(question["criteria"]) == list(WEIGHTED.rubric)


def test_a_gate_question_states_both_outcomes():
    criteria = build_questions([GATE])["gate"]["criteria"]
    assert criteria["true"] == "authorized to work"
    assert "Does not meet" in criteria["false"]


def test_questions_are_keyed_by_criterion_id():
    assert set(build_questions(CRITERIA)) == {"gate", "py", "cloud"}


def test_a_five_level_rubric_is_preserved():
    criterion = Criterion(
        id="c", name="C", rubric=("none", "low", "mid", "high", "max")
    )
    assert list(build_questions([criterion])["c"]["criteria"]) == list(criterion.rubric)


# --- State ------------------------------------------------------------------ #

def test_state_carries_the_resume_and_the_requirements():
    state = build_state(Candidate(candidate_id="a", resume_text="R"), CRITERIA, "Backend")
    assert state["candidate_id"] == "a"
    assert state["resume"] == "R"
    assert state["role"] == "Backend"
    assert len(state["requirements"]) == 3


def test_state_omits_the_form_when_absent():
    state = build_state(Candidate(candidate_id="a", resume_text="R"), CRITERIA)
    assert "application_form" not in state


def test_state_includes_the_form_when_present():
    candidate = Candidate(candidate_id="a", resume_text="R", application_form_text="F")
    assert build_state(candidate, CRITERIA)["application_form"] == "F"


def test_state_omits_a_blank_role():
    state = build_state(Candidate(candidate_id="a", resume_text="R"), CRITERIA, "   ")
    assert "role" not in state


def test_requirements_are_listed_in_order():
    state = build_state(Candidate(candidate_id="a", resume_text="R"), CRITERIA)
    assert [r["id"] for r in state["requirements"]] == ["gate", "py", "cloud"]


def test_token_estimate_grows_with_content():
    small = estimate_tokens({"resume": "x" * 40})
    large = estimate_tokens({"resume": "x" * 400})
    assert large > small


# --- Aggregation ------------------------------------------------------------ #

def test_weights_produce_a_weighted_average():
    # py scores 100 at weight 2, cloud scores 0 at weight 1 -> 200/3
    result = aggregate(CRITERIA, answer(), "a")
    assert result.overall_score == 66.7
    assert result.gate_passed is True


def test_a_failing_gate_is_recorded():
    result = aggregate(
        CRITERIA, answer(gate={"type": "noul", "noul": 0.1, "confidence": 0.9}), "a"
    )
    assert result.gate_passed is False
    assert result.failed_gates == ["gate"]


def test_a_gate_at_the_threshold_passes():
    result = aggregate(
        CRITERIA, answer(gate={"type": "noul", "noul": GATE_THRESHOLD}), "a"
    )
    assert result.per_criterion[0].passed is True


def test_a_gate_near_the_threshold_is_flagged_for_review():
    result = aggregate(
        CRITERIA,
        answer(gate={"type": "noul", "noul": GATE_THRESHOLD + GATE_MARGIN / 2}),
        "a",
    )
    assert result.needs_review is True, "a gate this close is too close to call"


def test_a_confident_gate_outside_the_margin_is_not_flagged():
    result = aggregate(
        CRITERIA, answer(gate={"type": "noul", "noul": 0.99, "confidence": 0.9}), "a"
    )
    assert result.needs_review is False


def test_low_confidence_is_flagged_for_review():
    result = aggregate(
        CRITERIA, answer(py={"type": "score", "score": 2.0, "confidence": 0.2}), "a"
    )
    assert result.needs_review is True


def test_confidence_above_the_floor_is_not_flagged():
    result = aggregate(
        CRITERIA,
        answer(py={"type": "score", "score": 2.0, "confidence": REVIEW_CONFIDENCE_FLOOR + 0.1}),
        "a",
    )
    assert result.needs_review is False


def test_an_unanswered_criterion_is_reported_not_assumed():
    result = aggregate(CRITERIA, JevResponse(answers={}), "a")
    assert all(outcome.error for outcome in result.per_criterion)
    assert result.gate_passed is False, "an unanswered gate must not pass"


def test_a_mismatched_answer_type_is_reported():
    result = aggregate(CRITERIA, answer(gate={"type": "score", "score": 1.0}), "a")
    assert "noul" in result.per_criterion[0].error


def test_a_gate_only_job_has_no_numeric_score():
    result = aggregate([GATE], answer(), "a")
    assert result.overall_score is None, "nothing was weighed"
    assert result.scored is True, "but the model did answer"
    assert result.gate_passed is True


def test_cost_and_usage_are_carried_through():
    result = aggregate(CRITERIA, answer(), "a")
    assert result.cost_usd == 0.001
    assert result.usage.input_tokens == 100


def test_a_fractional_score_lands_between_levels():
    result = aggregate(
        [WEIGHTED], JevResponse(answers={"py": {"type": "score", "score": 1.0}}), "a"
    )
    assert result.per_criterion[0].score_0_100 == 50.0
    assert result.per_criterion[0].label == "Partially met"


def test_a_score_above_the_rubric_clamps():
    result = aggregate(
        [WEIGHTED], JevResponse(answers={"py": {"type": "score", "score": 99.0}}), "a"
    )
    assert result.per_criterion[0].score_0_100 == 100.0


# --- Ordering --------------------------------------------------------------- #

def test_passing_candidates_come_first():
    ordered = sort_scores([
        CandidateScore(candidate_id="err", error="x"),
        CandidateScore(candidate_id="gated", gate_passed=False),
        CandidateScore(candidate_id="low", overall_score=10.0),
        CandidateScore(candidate_id="high", overall_score=90.0),
    ])
    assert [s.candidate_id for s in ordered] == ["high", "low", "gated", "err"]


def test_errored_candidates_come_last():
    ordered = sort_scores([
        CandidateScore(candidate_id="err", error="x", gate_passed=True),
        CandidateScore(candidate_id="ok", overall_score=50.0),
    ])
    assert ordered[-1].candidate_id == "err"


# --- Batch behavior --------------------------------------------------------- #

def test_batch_ranks_every_candidate():
    pool = [Candidate(candidate_id=f"c{i}", resume_text=RESUME) for i in range(3)]
    result = asyncio.run(ranker().rank_batch("Backend", CRITERIA, pool))
    assert len(result.results) == 3
    assert abs(result.total_cost_usd - 3 * 0.00002) < 1e-9, "one call per candidate"
    assert result.total_attempts == 3
    assert result.model == "typesafe/jev-1.13"


def test_single_returns_one_score():
    result = asyncio.run(
        ranker().rank_single(
            "Backend", CRITERIA, Candidate(candidate_id="a", resume_text=RESUME)
        )
    )
    assert isinstance(result, CandidateScore)
    assert result.candidate_id == "a"


def test_one_request_per_candidate():
    # Cost must scale with candidates, not candidates x criteria.
    transport = MockTransport()
    pool = [Candidate(candidate_id=f"c{i}", resume_text=RESUME) for i in range(4)]
    asyncio.run(ranker(transport).rank_batch("Backend", CRITERIA, pool))
    assert len(transport.requests) == 4
    assert len(transport.requests[0]["questions"]) == 3


def test_concurrency_is_capped():
    settings = Settings.build(jev_max_rps=1000.0, jev_max_concurrency=2)
    live = 0
    peak = 0

    class Counting(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            from app.testing import mock_body
            return httpx.Response(200, json=mock_body({}))

    jev = JevClient(settings, transport=Counting(), sleep=noop_sleep)
    pool = [Candidate(candidate_id=f"c{i}", resume_text=RESUME) for i in range(6)]
    asyncio.run(ResumeRanker(settings, jev).rank_batch("Backend", CRITERIA, pool))
    assert peak <= 2, f"peak concurrency was {peak}, expected at most 2"


def test_per_call_concurrency_overrides_the_setting():
    transport = MockTransport()
    pool = [Candidate(candidate_id=f"c{i}", resume_text=RESUME) for i in range(3)]
    result = asyncio.run(
        ranker(transport, jev_max_concurrency=1).rank_batch(
            "Backend", CRITERIA, pool, max_concurrency=3
        )
    )
    assert len(result.results) == 3


def test_a_transient_failure_errors_only_that_candidate():
    import json as _json

    good = {
        "model": "m",
        "answers": {
            "gate": {"type": "noul", "noul": 0.9},
            "py": {"type": "score", "score": 2.0},
            "cloud": {"type": "score", "score": 2.0},
        },
        "usage": {"cost": 0.001},
    }

    class FlakyOne(httpx.AsyncBaseTransport):
        """Fails every attempt for one candidate, succeeds for the others."""

        async def handle_async_request(self, request):
            payload = _json.loads(request.content)
            if payload["state"].get("candidate_id") == "a":
                return httpx.Response(503, json={"error": {"message": "down"}})
            return httpx.Response(200, json=good)

    pool = [
        Candidate(candidate_id="a", resume_text=RESUME),
        Candidate(candidate_id="b", resume_text=RESUME),
        Candidate(candidate_id="c", resume_text=RESUME),
    ]
    result = asyncio.run(
        ranker(FlakyOne(), jev_max_retries=2).rank_batch("Backend", CRITERIA, pool)
    )
    assert result.error_count == 1, "one candidate should carry the error"
    assert len([r for r in result.results if r.scored]) == 2, "the rest still rank"
    errored = [r for r in result.results if r.error]
    assert errored[0].candidate_id == "a"
    assert errored[0].needs_review is True


def test_exhausted_retries_become_a_candidate_error():
    transport = ScriptedTransport([httpx.Response(503, json={"error": {"message": "down"}})])
    pool = [Candidate(candidate_id="a", resume_text=RESUME)]
    result = asyncio.run(
        ranker(transport, jev_max_retries=2).rank_batch("Backend", CRITERIA, pool)
    )
    assert result.error_count == 1
    assert result.results[0].attempts == 2


def test_out_of_credit_aborts_the_whole_batch():
    transport = ScriptedTransport(
        [(402, {"error": {"message": "Insufficient credits."}})]
    )
    pool = [Candidate(candidate_id="a", resume_text=RESUME) for _ in range(3)]
    try:
        asyncio.run(ranker(transport).rank_batch("Backend", CRITERIA, pool))
    except JevCreditsError:
        return
    raise AssertionError("a systemic failure should abort, not repeat per candidate")


def test_a_bad_key_aborts_the_whole_batch():
    transport = ScriptedTransport([(401, {"error": {"message": "bad key"}})])
    try:
        asyncio.run(
            ranker(transport).rank_batch(
                "Backend", CRITERIA, [Candidate(candidate_id="a", resume_text=RESUME)]
            )
        )
    except JevAuthError:
        return
    raise AssertionError("a bad key should abort the batch")


def test_an_empty_pool_is_rejected():
    try:
        asyncio.run(ranker().rank_batch("Backend", CRITERIA, []))
    except InputValidationError:
        return
    raise AssertionError("an empty pool should be rejected")


def test_a_blank_job_description_is_rejected():
    try:
        asyncio.run(
            ranker().rank_batch(
                "  ", CRITERIA, [Candidate(candidate_id="a", resume_text=RESUME)]
            )
        )
    except InputValidationError:
        return
    raise AssertionError("a blank job description should be rejected")


def test_a_batch_past_the_configured_limit_is_rejected():
    settings = Settings.build(jev_max_rps=1000.0, jev_max_batch_size=2)
    pool = [Candidate(candidate_id=f"c{i}", resume_text=RESUME) for i in range(3)]
    try:
        asyncio.run(ResumeRanker(settings).rank_batch("Backend", CRITERIA, pool))
    except InputValidationError:
        return
    raise AssertionError("a batch past the limit should be rejected")


def test_an_oversized_request_is_caught_before_spending():
    import app.schemas as schemas

    original = schemas.MAX_RESUME_TEXT
    schemas.MAX_RESUME_TEXT = 500_000
    try:
        huge = Candidate.model_construct(
            candidate_id="huge", resume_text="x" * 200_000
        )
        transport = MockTransport()
        try:
            asyncio.run(ranker(transport).rank_batch("Backend", CRITERIA, [huge]))
        except InputValidationError:
            assert transport.requests == [], "no request should have been sent"
            return
    finally:
        schemas.MAX_RESUME_TEXT = original
    raise AssertionError("an oversized request should be caught up front")


def test_from_settings_builds_a_ranker():
    assert isinstance(ResumeRanker.from_settings(Settings.build()), ResumeRanker)


def test_results_are_ordered_with_passing_candidates_first():
    pool = [
        Candidate(candidate_id="good", resume_text=RESUME),
        Candidate(candidate_id="empty", resume_text="nothing relevant"),
    ]
    result = asyncio.run(ranker().rank_batch("Backend", CRITERIA, pool))
    assert result.results[0].gate_passed is True
    assert result.best() is not None
