"""Resume ranking built on the Jev decision model.

This layer adds only what :mod:`app.jev_client` does not know: how job
requirements map to Jev questions, how the answers become a gate decision and a
weighted score, and how a pool of candidates gets ranked.

How a ranking works
-------------------
1. A job's requirements are a list of weighted :class:`~app.schemas.Criterion`.
2. Each criterion becomes one Jev question, built once per batch:
   - ``required=True``  -> a ``noul`` question; the answer is a probability of
     "yes" and acts as a hard gate.
   - ``required=False`` -> a ``score`` question on an ordered rubric; the answer
     is normalized to 0-100 and combined into a weighted average.
3. Every candidate is one Decisions request containing all of its questions, so
   cost scales with the number of candidates, not criteria x candidates.

The functions that build questions, build state, and fold answers into a score
are pure and module-level, so they can be tested and reused without a client.
:class:`ResumeRanker` is the stateful wrapper that owns the HTTP client and
handles concurrency.

Use :mod:`app.jev_client` directly if you need Jev for something other than
ranking.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional, Sequence

from app.config import Settings
from app.exceptions import (
    InputValidationError,
    JevError,
    JevTransientError,
)
from app.jev_client import JevClient, Question, Questions
from app.schemas import (
    Candidate,
    CandidateScore,
    Criterion,
    CriterionOutcome,
    JevResponse,
    RankingResult,
    SCORE_SCALE,
)

# --- Thresholds -------------------------------------------------------------
#: A gate passes at or above this probability that the requirement is met.
GATE_THRESHOLD = 0.5
#: A gate within this distance of the threshold is too close to call.
GATE_MARGIN = 0.15
#: Score ``confidence`` below this flags a result for human review.
REVIEW_CONFIDENCE_FLOOR = 0.60
#: Jev's context window, in tokens.
JEV_CONTEXT_TOKENS = 32_000
#: Rough chars-per-token ratio for the pre-flight context check.
CHARS_PER_TOKEN = 4


# --------------------------------------------------------------------------- #
# Question and state construction
# --------------------------------------------------------------------------- #

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
    return len(json.dumps(value, ensure_ascii=False, default=str)) // CHARS_PER_TOKEN


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(
    criteria: Sequence[Criterion],
    response: JevResponse,
    candidate_id: str = "",
) -> CandidateScore:
    """Fold Jev's answers into a gate decision and a weighted 0-100 score.

    A criterion the model failed to answer is reported with an ``error`` and, if
    it is a gate, treated as failed. Scoring an unanswered gate as "did not
    meet" would be an assumption presented as a result.
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
            outcome, weighted, weight = _aggregate_gate(criterion, response)
            outcomes.append(outcome)
            if outcome.passed is False:
                gate_passed = False
                failed_gates.append(criterion.id)
            needs_review = needs_review or outcome.needs_review
            continue

        outcome, weighted, weight = _aggregate_score(criterion, response)
        outcomes.append(outcome)
        needs_review = needs_review or outcome.needs_review
        weighted_sum += weighted * weight
        weight_total += weight

    return CandidateScore(
        candidate_id=candidate_id,
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


def _aggregate_gate(
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


def _aggregate_score(
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

class ResumeRanker:
    """Ranks candidate pools against a job, using one Jev client."""

    def __init__(self, settings: Settings, jev: Optional[JevClient] = None):
        self.settings = settings
        self.jev = jev or JevClient(settings)

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "ResumeRanker":
        """Build a ranker, defaulting to the process-wide settings."""
        from app.config import get_settings

        return cls(settings or get_settings())

    # -- single ------------------------------------------------------------ #

    async def rank_single(
        self,
        job_description: str,
        criteria: Sequence[Criterion],
        candidate: Candidate,
    ) -> CandidateScore:
        """Score one candidate. Upstream errors are raised, never swallowed."""
        result = await self.rank_batch(job_description, criteria, [candidate])
        return result.results[0]

    # -- batch ------------------------------------------------------------- #

    async def rank_batch(
        self,
        job_description: str,
        criteria: Sequence[Criterion],
        candidates: Sequence[Candidate],
        *,
        max_concurrency: Optional[int] = None,
    ) -> RankingResult:
        """Score a pool concurrently. Gated-out candidates sort last.

        A transient failure (429, 5xx, a timeout) marks just that one candidate
        as errored and the rest of the pool still ranks. A systemic failure -
        bad key, no credit, a rejected request shape - aborts the batch, since
        retrying it per candidate would only produce the same answer N times.
        """
        if not candidates:
            raise InputValidationError("no candidates to rank")
        if len(candidates) > self.settings.jev_max_batch_size:
            raise InputValidationError(
                f"batch of {len(candidates)} exceeds the limit of "
                f"{self.settings.jev_max_batch_size}"
            )
        if not job_description.strip():
            raise InputValidationError("job_description must not be blank")
        self._check_context(criteria, candidates, job_description)

        started = time.monotonic()
        questions = build_questions(criteria)
        states = [build_state(c, criteria, job_description) for c in candidates]
        limit = max_concurrency or self.settings.jev_max_concurrency
        semaphore = asyncio.Semaphore(limit)

        async def score(state: dict[str, Any]) -> JevResponse:
            async with semaphore:
                return await self.jev.decide(state=state, questions=questions)

        outcomes = await asyncio.gather(
            *(score(state) for state in states), return_exceptions=True
        )

        scores: list[CandidateScore] = []
        for candidate, outcome in zip(candidates, outcomes):
            if isinstance(outcome, JevTransientError):
                scores.append(
                    CandidateScore.failed(
                        candidate.candidate_id,
                        f"{type(outcome).__name__}: {outcome}",
                    )
                )
                scores[-1].attempts = self.settings.jev_max_retries
            elif isinstance(outcome, BaseException):
                # Systemic: bad key, no credit, a rejected shape. Aborting beats
                # reporting the same failure once per candidate.
                raise outcome
            else:
                scores.append(aggregate(criteria, outcome, candidate.candidate_id))

        ordered = sort_scores(scores)
        return RankingResult(
            model=self.settings.jev_model,
            results=ordered,
            total_cost_usd=sum(s.cost_usd or 0.0 for s in ordered),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    # -- pre-flight -------------------------------------------------------- #

    def _check_context(
        self,
        criteria: Sequence[Criterion],
        candidates: Sequence[Candidate],
        job_description: str,
    ) -> None:
        """Reject a request that cannot fit in Jev's context window.

        Cheaper and clearer to catch here than to discover it as a 400 from the
        provider after paying for a partial run.
        """
        questions = build_questions(criteria)
        overhead = estimate_tokens(questions) + estimate_tokens(
            {"requirements": [c.name for c in criteria]}
        )
        limit = JEV_CONTEXT_TOKENS - overhead
        for candidate in candidates:
            size = estimate_tokens(build_state(candidate, criteria, job_description))
            if size > limit:
                raise InputValidationError(
                    f"candidate {candidate.candidate_id!r} needs about {size} tokens, "
                    f"which exceeds the {limit}-token budget left after the questions; "
                    f"shorten the resume or reduce the number of criteria"
                )

    async def aclose(self) -> None:
        await self.jev.aclose()


__all__ = [
    "GATE_THRESHOLD",
    "GATE_MARGIN",
    "REVIEW_CONFIDENCE_FLOOR",
    "JEV_CONTEXT_TOKENS",
    "CHARS_PER_TOKEN",
    "build_questions",
    "build_state",
    "estimate_tokens",
    "aggregate",
    "sort_scores",
    "ResumeRanker",
]
