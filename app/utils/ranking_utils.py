"""Pure functions for the ranking pipeline, with the policy they apply.

No state and no I/O. This is the whole of "how a job becomes questions, and how
answers become scores", as functions you can call and test directly:
:meth:`ResumeRanker.rank_batch` in :mod:`app.services.ranking_service` is the
stateful wrapper that feeds them.

The judgement calls - what counts as meeting a requirement, how near the gate is
too near, when a result is too unsure to trust - are the constants at the top.
They live here because these are the only functions that read them.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

from app.exceptions import JevError
from app.schemas import (
    Candidate,
    CandidateScore,
    Criterion,
    CriterionOutcome,
    JevResponse,
    Question,
    Questions,
)

logger = logging.getLogger(__name__)

# --- Scoring policy ---------------------------------------------------------
# Declared here, beside the functions that apply them, so there is one place to
# read and one place to change. These are the judgement calls: what counts as
# meeting a requirement, and when a result is too unsure to trust.

#: Output scale for a weighted score.
SCORE_SCALE = 100.0
#: A gate passes at or above this probability that the requirement is met.
GATE_THRESHOLD = 0.5
#: A gate within this distance of the threshold is too close to call.
GATE_MARGIN = 0.15
#: Confidence below this flags a result for human review.
REVIEW_CONFIDENCE_FLOOR = 0.60

# --- Context ----------------------------------------------------------------
#: Jev's context window, in tokens.
JEV_CONTEXT_TOKENS = 32_000
#: Rough chars-per-token ratio for the pre-flight size check.
CHARS_PER_TOKEN = 4

def build_questions(criteria: Sequence[Criterion]) -> dict[str, Question]:
    """Convert criteria into Jev questions. Built once per batch.

    Gates ask a yes/no question framed by the requirement; scored criteria ask
    where the candidate sits on the rubric.
    """
    questions: dict[str, Question] = {}
    for criterion in criteria:
        requirement = criterion.description or criterion.name
        if criterion.is_gate:
            questions[criterion.id] = Questions.noul(
                f"Does this candidate meet the requirement '{criterion.name}'? "
                "Answer from evidence in the state only.",
                true_criteria=requirement,
                false_criteria=f"Does not meet: {requirement}",
            )
        else:
            questions[criterion.id] = Questions.score(
                f"Where does this candidate fall on '{criterion.name}'? "
                "Pick the single best level, using evidence in the state only.",
                criterion.rubric,
            )
    return questions


def build_state(
    candidate: Candidate,
    criteria: Sequence[Criterion],
    job_description: str = "",
) -> dict[str, Any]:
    """Assemble the ``state`` object Jev decides about.

    A structured object rather than one concatenated prompt: the model can refer
    to fields by name, and the token count stays predictable.
    """
    state: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "resume": candidate.resume_text,
    }
    if candidate.application_form_text.strip():
        state["application_form"] = candidate.application_form_text
    if job_description.strip():
        state["role"] = job_description
    state["requirements"] = [
        {
            "id": criterion.id,
            "requirement": criterion.name,
            "detail": criterion.description,
            "required": criterion.required,
        }
        for criterion in criteria
    ]
    return state


def estimate_tokens(value: Any) -> int:
    """Rough token estimate, for the pre-flight context check."""
    size = len(json.dumps(value, ensure_ascii=False, default=str))
    return size // CHARS_PER_TOKEN


def aggregate(
    criteria: Sequence[Criterion],
    response: JevResponse,
    candidate_id: str = "",
) -> CandidateScore:
    """Fold Jev's answers into a gate decision and a weighted 0-100 score.

    A criterion the model failed to answer is reported with an ``error`` and, if
    it is a gate, treated as failed. Scoring an unanswered gate as "did not
    meet" would be an assumption presented as a result.

    ``settings`` carries the gate threshold, the undecided margin, the review
    floor, and the score scale. They are read per call rather than imported as
    constants so that ranking against a stricter or laxer standard is a matter
    of configuration.
    """
    outcomes: list[CriterionOutcome] = []
    gate_passed = True
    failed_gates: list[str] = []
    needs_review = False
    weighted_sum = 0.0
    weight_total = 0.0

    for criterion in criteria:
        answer = response.answer(criterion.id)

        if answer is None:
            outcomes.append(
                CriterionOutcome(
                    criterion_id=criterion.id,
                    name=criterion.name,
                    required=criterion.required,
                    needs_review=True,
                    error="model did not answer this criterion",
                )
            )
            gate_passed = False
            needs_review = True
            if criterion.is_gate:
                failed_gates.append(criterion.id)
            continue

        if criterion.is_gate:
            outcome, weighted, weight = aggregate_gate(criterion, response)
            outcomes.append(outcome)
            if outcome.passed is False:
                gate_passed = False
                failed_gates.append(criterion.id)
            needs_review = needs_review or outcome.needs_review
            continue

        outcome, weighted, weight = aggregate_score(criterion, response)
        outcomes.append(outcome)
        needs_review = needs_review or outcome.needs_review
        weighted_sum += weighted * weight
        weight_total += weight

    return CandidateScore(
        candidate_id=candidate_id,
        model=response.model,
        gate_passed=gate_passed,
        overall_score=(
            round(weighted_sum / weight_total, 1) if weight_total else None
        ),
        failed_gates=failed_gates,
        needs_review=needs_review,
        per_criterion=outcomes,
        cost_usd=response.cost_usd,
        attempts=response.attempts,
        usage=response.usage,
    )


def aggregate_gate(
    criterion: Criterion, response: JevResponse
) -> tuple[CriterionOutcome, float, float]:
    """Evaluate one hard requirement. Carries no weight: it passes or it does not."""
    try:
        probability = response.noul(criterion.id)
    except JevError as exc:
        return (
            CriterionOutcome(
                criterion_id=criterion.id,
                name=criterion.name,
                required=True,
                needs_review=True,
                error=str(exc),
            ),
            0.0,
            0.0,
        )

    passed = probability >= GATE_THRESHOLD
    undecided = abs(probability - GATE_THRESHOLD) < GATE_MARGIN
    confidence = response.confidence(criterion.id)
    return (
        CriterionOutcome(
            criterion_id=criterion.id,
            name=criterion.name,
            required=True,
            passed=passed,
            label=criterion.rubric_at(criterion.max_index if passed else 0.0),
            raw_score=probability,
            score_0_100=round(probability * SCORE_SCALE, 1),
            confidence=confidence,
            needs_review=undecided,
        ),
        0.0,
        0.0,
    )


def aggregate_score(
    criterion: Criterion, response: JevResponse
) -> tuple[CriterionOutcome, float, float]:
    """Evaluate one weighted criterion. Returns the outcome, 0-100, and weight."""
    try:
        answer = response.score(criterion.id)
    except JevError as exc:
        return (
            CriterionOutcome(
                criterion_id=criterion.id,
                name=criterion.name,
                required=False,
                needs_review=True,
                error=str(exc),
            ),
            0.0,
            0.0,
        )

    confidence = answer.confidence
    normalized = criterion.weighted_score(answer.score) * SCORE_SCALE
    undecided = confidence is not None and confidence < REVIEW_CONFIDENCE_FLOOR
    return (
        CriterionOutcome(
            criterion_id=criterion.id,
            name=criterion.name,
            required=False,
            label=criterion.rubric_at(answer.score),
            raw_score=answer.score,
            score_0_100=round(normalized, 1),
            confidence=confidence,
            needs_review=undecided,
        ),
        normalized,
        criterion.weight,
    )


def model_that_answered(
    scores: Sequence[CandidateScore], default: str
) -> str:
    """The model that actually produced these scores.

    Read off the scores rather than taken from settings, so a report does not
    claim the primary model when the fallback is what answered. A pool split
    across both models has no single honest answer, so ``default`` is reported
    and the mix is logged.

    Module-level and shared, so the batch result and the single-candidate
    endpoint cannot report the model differently.
    """
    used = {score.model for score in scores if score.model}
    if len(used) == 1:
        return used.pop()
    if used:
        logger.warning(
            "results answered by more than one model (%s); reporting %s",
            ", ".join(sorted(used)), default,
        )
    return default


def sort_scores(scores: Sequence[CandidateScore]) -> list[CandidateScore]:
    """Order for display: passing candidates by score, then everyone else.

    Gated-out candidates sort last, and errored ones after those, so the top of
    the list is always the shortlist.
    """

    def key(score: CandidateScore) -> tuple[int, float]:
        if score.error:
            return (2, 0.0)
        if not score.gate_passed:
            return (1, 0.0)
        return (0, -(score.overall_score or 0.0))

    return sorted(scores, key=key)


# --------------------------------------------------------------------------- #
# Ranker
# --------------------------------------------------------------------------- #

__all__ = [
    "SCORE_SCALE",
    "GATE_THRESHOLD",
    "GATE_MARGIN",
    "REVIEW_CONFIDENCE_FLOOR",
    "JEV_CONTEXT_TOKENS",
    "CHARS_PER_TOKEN",
    "build_questions",
    "build_state",
    "estimate_tokens",
    "aggregate",
    "aggregate_gate",
    "aggregate_score",
    "model_that_answered",
    "sort_scores",
]
