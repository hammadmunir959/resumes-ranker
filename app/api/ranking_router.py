"""Ranking API endpoints."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Request

from app.api.limiter import limiter
from app.exceptions import RankerError
from app.schemas import (
    RankBatchRequest,
    RankSingleRequest,
    RankingResult,
)
from app.services import ResumeRanker
from app.utils import model_that_answered

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rank"])


def get_ranker(request: Request) -> ResumeRanker:
    """Return the shared ResumeRanker instance or raise startup error."""
    error: Optional[RankerError] = getattr(request.app.state, "error", None)
    if error is not None:
        raise error
    return request.app.state.ranker


RankerDep = Depends(get_ranker)


@router.post("/rank/single", response_model=RankingResult)
@limiter.limit("60/minute")
async def rank_single(
    request: Request,
    payload: RankSingleRequest,
    ranker: ResumeRanker = RankerDep,
) -> RankingResult:
    """Rank one candidate against job criteria."""
    started = time.monotonic()
    score = await ranker.rank_single(
        job_description=payload.job_description,
        criteria=payload.criteria,
        candidate=payload.candidate,
    )
    return RankingResult(
        model=model_that_answered([score], ranker.settings.jev_model),
        results=[score],
        total_cost_usd=score.cost_usd or 0.0,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )


@router.post("/rank/batch", response_model=RankingResult)
@limiter.limit("60/minute")
async def rank_batch(
    request: Request,
    payload: RankBatchRequest,
    ranker: ResumeRanker = RankerDep,
) -> RankingResult:
    """Rank a pool of candidates concurrently."""
    return await ranker.rank_batch(
        job_description=payload.job_description,
        criteria=payload.criteria,
        candidates=payload.candidates,
        max_concurrency=payload.max_concurrency,
    )


__all__ = ["router", "get_ranker", "RankerDep", "rank_single", "rank_batch"]
