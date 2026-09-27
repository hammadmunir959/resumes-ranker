"""Model validation, typed answers, and the derived values reports rely on."""

from app.exceptions import JevProtocolError
from app.schemas import (
    Candidate,
    CandidateScore,
    ChoiceAnswer,
    Criterion,
    CriterionOutcome,
    JevResponse,
    RankBatchRequest,
    RankingResult,
    RankSingleRequest,
    ScoreAnswer,
)


def _reject(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception:
        return True
    return False


# --- Criterion -------------------------------------------------------------- #

def test_criterion_trims_identifiers():
    assert Criterion(id="  py  ", name="  Python  ").id == "py"


def test_blank_identifier_is_rejected():
    assert _reject(Criterion, id="   ", name="Python")
    assert _reject(Criterion, id="py", name="  ")


def test_single_level_rubric_is_rejected():
    assert _reject(Criterion, id="a", name="A", rubric=("only",))


def test_blank_rubric_level_is_rejected():
    assert _reject(Criterion, id="a", name="A", rubric=("yes", ""))


def test_optional_criterion_needs_a_positive_weight():
    assert _reject(Criterion, id="a", name="A", weight=0)
    assert _reject(Criterion, id="a", name="A", weight=-1)
    # A gate short-circuits, so a weight on one would never be read.
    assert Criterion(id="a", name="A", required=True, weight=0).weight == 0.0


def test_criterion_is_immutable():
    criterion = Criterion(id="a", name="A")
    assert _reject(setattr, criterion, "weight", 5.0)


def test_extra_fields_are_rejected():
    assert _reject(Criterion, id="a", name="A", colour="red")


def test_max_index_is_the_top_of_the_rubric():
    assert Criterion(id="a", name="A", rubric=("x", "y", "z")).max_index == 2.0


def test_is_gate_tracks_required():
    assert Criterion(id="a", name="A", required=True).is_gate is True
    assert Criterion(id="a", name="A").is_gate is False


def test_rubric_at_clamps_to_the_range():
    criterion = Criterion(id="a", name="A", rubric=("low", "mid", "high"))
    assert criterion.rubric_at(1.0) == "mid"
    assert criterion.rubric_at(99.0) == "high", "above the top should clamp"
    assert criterion.rubric_at(-5.0) == "low", "below the bottom should clamp"


def test_weighted_score_normalizes_to_zero_one():
    criterion = Criterion(id="a", name="A", rubric=("x", "y", "z"))
    assert criterion.weighted_score(0.0) == 0.0
    assert criterion.weighted_score(1.0) == 0.5
    assert criterion.weighted_score(2.0) == 1.0
    assert criterion.weighted_score(5.0) == 1.0, "should clamp above the top"


# --- Candidate -------------------------------------------------------------- #

def test_candidate_requires_a_resume():
    assert _reject(Candidate, candidate_id="a", resume_text="")


def test_candidate_rejects_a_blank_id():
    assert _reject(Candidate, candidate_id="  ", resume_text="r")


def test_candidate_text_merges_the_application_form():
    candidate = Candidate(
        candidate_id="a", resume_text="RESUME", application_form_text="FORM"
    )
    assert "RESUME" in candidate.text_for_matching
    assert "FORM" in candidate.text_for_matching


def test_candidate_text_omits_an_empty_form():
    candidate = Candidate(candidate_id="a", resume_text="RESUME")
    assert candidate.text_for_matching == "RESUME"


def test_oversized_resume_is_rejected():
    assert _reject(
        Candidate, candidate_id="a", resume_text="x" * 40_001
    ), "a resume past the limit should be rejected by the model, not upstream"


def test_token_estimate_is_roughly_four_chars():
    assert Candidate(candidate_id="a", resume_text="x" * 400).estimated_tokens() == 100


# --- Typed answers ---------------------------------------------------------- #

def _response(**answers) -> JevResponse:
    return JevResponse(answers=answers)


def test_noul_is_read_as_a_probability():
    assert _response(g={"type": "noul", "noul": 0.93}).noul("g") == 0.93


def test_score_carries_confidence_and_distribution():
    response = _response(
        s={
            "type": "score",
            "score": 1.4,
            "confidence": 0.7,
            "probabilities": {"0": 0.3, "1": 0.4, "2": 0.3},
            "legend": {"0": "low"},
        }
    )
    answer = response.score("s")
    assert answer.score == 1.4
    assert answer.confidence == 0.7
    assert answer.legend["0"] == "low"
    assert answer.expected_score() == 1.0, "mean of 0,1,2 at those weights"


def test_expected_score_falls_back_to_the_point_estimate():
    assert ScoreAnswer(question_id="s", score=1.5).expected_score() == 1.5


def test_choice_is_read_as_a_selection():
    response = _response(c={"type": "choice", "choice": "payments", "confidence": 0.6})
    assert response.choice("c").choice == "payments"


def test_missing_answer_returns_none_rather_than_raising():
    response = _response()
    assert response.answer("nope") is None
    assert response.confidence("nope") is None


def test_asking_for_the_wrong_answer_type_raises():
    response = _response(s={"type": "score", "score": 1.0})
    assert _reject(response.noul, "s"), "a score answer is not a noul"
    assert _reject(response.choice, "s"), "a score answer is not a choice"


def test_a_boolean_is_not_a_probability():
    assert _reject(_response(g={"type": "noul", "noul": True}).noul, "g")


def test_usage_totals_tokens():
    response = JevResponse(input_tokens=900, output_tokens=40)
    assert response.usage.total_tokens == 940


# --- Results ---------------------------------------------------------------- #

def test_failed_candidate_is_marked_for_review():
    score = CandidateScore.failed("a", "boom")
    assert score.error == "boom"
    assert score.needs_review is True
    assert score.scored is False


def test_a_gate_only_job_still_counts_as_scored():
    # No weighted criterion contributed, so there is no number, but the model did
    # answer. Treating this as unscored would hide a passing candidate.
    score = CandidateScore(candidate_id="a", gate_passed=True, overall_score=None)
    assert score.scored is True


def test_best_prefers_a_passing_candidate():
    result = RankingResult(
        model="m",
        results=[
            CandidateScore(candidate_id="gated", gate_passed=False, overall_score=99.0),
            CandidateScore(candidate_id="low", overall_score=10.0),
            CandidateScore(candidate_id="high", overall_score=90.0),
        ],
    )
    assert result.best().candidate_id == "high", "a gated candidate is never best"


def test_best_falls_back_when_only_gates_were_scored():
    result = RankingResult(
        model="m",
        results=[
            CandidateScore(candidate_id="a", gate_passed=True),
            CandidateScore(candidate_id="b", gate_passed=True),
        ],
    )
    assert result.best() is not None


def test_best_is_none_when_nobody_passed():
    result = RankingResult(
        model="m",
        results=[CandidateScore(candidate_id="a", gate_passed=False)],
    )
    assert result.best() is None


def test_error_and_attempt_counts():
    result = RankingResult(
        model="m",
        results=[
            CandidateScore(candidate_id="a", attempts=2),
            CandidateScore(candidate_id="b", error="x", attempts=4),
        ],
    )
    assert result.total_attempts == 6
    assert result.error_count == 1


def test_outcome_defaults_are_report_friendly():
    outcome = CriterionOutcome(criterion_id="a", name="A", required=True)
    assert outcome.score_0_100 == 0.0
    assert outcome.confidence is None
    assert outcome.passed is None, "an unscored gate is not a failure"


# --- Requests --------------------------------------------------------------- #

def test_single_request_accepts_a_valid_body():
    request = RankSingleRequest(
        job_description="Backend",
        criteria=[Criterion(id="a", name="A")],
        candidate=Candidate(candidate_id="c", resume_text="r"),
    )
    assert request.criteria[0].id == "a"


def test_batch_request_accepts_a_valid_body():
    request = RankBatchRequest(
        job_description="Backend",
        criteria=[Criterion(id="a", name="A")],
        candidates=[Candidate(candidate_id="c", resume_text="r")],
    )
    assert request.candidates[0].candidate_id == "c"
    assert request.max_concurrency is None


def test_duplicate_criterion_ids_are_rejected():
    assert _reject(
        RankSingleRequest,
        job_description="j",
        criteria=[Criterion(id="a", name="A"), Criterion(id="a", name="B")],
        candidate=Candidate(candidate_id="c", resume_text="r"),
    )


def test_duplicate_candidate_ids_are_rejected():
    candidate = Candidate(candidate_id="c", resume_text="r")
    assert _reject(
        RankBatchRequest,
        job_description="j",
        criteria=[Criterion(id="a", name="A")],
        candidates=[candidate, candidate],
    )


def test_blank_job_description_is_rejected():
    assert _reject(
        RankSingleRequest,
        job_description="   ",
        criteria=[Criterion(id="a", name="A")],
        candidate=Candidate(candidate_id="c", resume_text="r"),
    )


def test_at_least_one_criterion_is_required():
    assert _reject(
        RankSingleRequest,
        job_description="j",
        criteria=[],
        candidate=Candidate(candidate_id="c", resume_text="r"),
    )


def test_batch_size_limit_is_enforced_by_the_model():
    candidates = [
        Candidate(candidate_id=f"c{i}", resume_text="r") for i in range(101)
    ]
    assert _reject(
        RankBatchRequest,
        job_description="j",
        criteria=[Criterion(id="a", name="A")],
        candidates=candidates,
    )


def test_shared_validation_applies_to_both_request_types():
    # RankRequestBase is the single definition; both endpoints must behave the same.
    for cls in (RankSingleRequest, RankBatchRequest):
        assert _reject(cls, job_description="", criteria=[Criterion(id="a", name="A")])


def test_unknown_request_fields_are_rejected():
    assert _reject(
        RankSingleRequest,
        job_description="j",
        criteria=[Criterion(id="a", name="A")],
        candidate=Candidate(candidate_id="c", resume_text="r"),
        extra="nope",
    )
