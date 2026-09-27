"""Pure functions for the resume ranking pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

from app.config import (
    CHARS_PER_TOKEN,
    GATE_MARGIN,
    GATE_THRESHOLD,
    JEV_CONTEXT_TOKENS,
    REVIEW_CONFIDENCE_FLOOR,
    SCORE_SCALE,
)
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


def build_questions(criteria: Sequence[Criterion]) -> dict[str, Question]:
    """Convert criteria into Jev questions dict."""
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
    """Assemble structured state dict for candidate and criteria."""
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
            "id": c.id,
            "requirement": c.name,
            "detail": c.description,
            "required": c.required,
        }
        for c in criteria
    ]
    return state


def estimate_tokens(value: Any) -> int:
    """Rough token estimate for pre-flight context check."""
    size = len(json.dumps(value, ensure_ascii=False, default=str))
    return size // CHARS_PER_TOKEN


def aggregate(
    criteria: Sequence[Criterion],
    response: JevResponse,
    candidate_id: str = "",
) -> CandidateScore:
    """Fold Jev answers into candidate score and criterion outcomes."""
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
            outcome, _, _ = aggregate_gate(criterion, response)
            outcomes.append(outcome)
            if outcome.passed is False:
                gate_passed = False
                failed_gates.append(criterion.id)
            needs_review = needs_review or outcome.needs_review
        else:
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
    """Evaluate one hard requirement gate."""
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
    """Evaluate one weighted score criterion."""
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
    """Return model name that produced the scores, or default if split/none."""
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
    """Sort scores with passing candidates first, ordered by score descending."""
    def key(score: CandidateScore) -> tuple[int, float]:
        if score.error:
            return (2, 0.0)
        if not score.gate_passed:
            return (1, 0.0)
        return (0, -(score.overall_score or 0.0))

    return sorted(scores, key=key)


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
