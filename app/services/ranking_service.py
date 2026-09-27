"""Resume ranking service built on the Jev decision model."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional, Sequence

from app.config import Settings
from app.exceptions import (
    InputValidationError,
    JevTransientError,
)
from app.schemas import (
    Candidate,
    CandidateScore,
    Criterion,
    JevResponse,
    RankingResult,
)
from app.services.jev_service import JevClient
from app.utils.ranking_utils import (
    JEV_CONTEXT_TOKENS,
    aggregate,
    build_questions,
    build_state,
    estimate_tokens,
    model_that_answered,
    sort_scores,
)

logger = logging.getLogger(__name__)

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
        """Score one candidate against job criteria."""
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
        """Score a candidate pool concurrently."""
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
            model=model_that_answered(scores, self.settings.jev_model),
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
        """Reject a request that cannot fit in Jev's context window."""
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
        """Close ranker and underlying Jev client."""
        await self.jev.aclose()


__all__ = [
    "build_questions",
    "build_state",
    "estimate_tokens",
    "aggregate",
    "model_that_answered",
    "GATE_THRESHOLD",
    "GATE_MARGIN",
    "REVIEW_CONFIDENCE_FLOOR",
    "SCORE_SCALE",
    "JEV_CONTEXT_TOKENS",
    "CHARS_PER_TOKEN",
    "sort_scores",
    "ResumeRanker",
]
