"""The ranking endpoints, and nothing else.

Two routes, both thin: validate the payload, call the service, return the result.
No HTTP client, no retries, no scoring - those belong to
:mod:`app.services` and :mod:`app.utils`. Keeping the routes this thin is what
makes them readable as an API description rather than as the program.

Both return :class:`~app.schemas.RankingResult`. Single is a batch of one, so a
client only ever parses one response shape.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Request

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
    """The shared ranker, or the reason there isn't one.

    A missing ranker is a configuration error that the lifespan already recorded,
    so re-raising it here lets the single error handler in
    :mod:`app.main` turn it into a 503 for every ranking route at once.
    """
    error: Optional[RankerError] = getattr(request.app.state, "error", None)
    if error is not None:
        raise error
    return request.app.state.ranker


RankerDep = Depends(get_ranker)


@router.post("/rank/single", response_model=RankingResult)
async def rank_single(
    payload: RankSingleRequest,
    ranker: ResumeRanker = RankerDep,
) -> RankingResult:
    """Rank one candidate. Upstream errors are raised, not hidden."""
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
async def rank_batch(
    payload: RankBatchRequest,
    ranker: ResumeRanker = RankerDep,
) -> RankingResult:
    """Rank a pool concurrently. Gated-out candidates sort last."""
    return await ranker.rank_batch(
        job_description=payload.job_description,
        criteria=payload.criteria,
        candidates=payload.candidates,
        max_concurrency=payload.max_concurrency,
    )


__all__ = ["router", "get_ranker", "RankerDep", "rank_single", "rank_batch"]
