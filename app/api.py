"""FastAPI application exposing the ranker over HTTP.

Three endpoints:

* ``GET  /health``    liveness, plus the configuration actually loaded
* ``POST /rank/single``  rank one candidate
* ``POST /rank/batch``   rank a pool concurrently

Both rank endpoints return the same :class:`~app.schemas.RankingResult` shape;
single is just a batch of one, so clients only handle one format.

Errors are handled in one place. Every exception in :mod:`app.exceptions`
carries its own ``http_status`` and ``error_code``, so this module needs a
single handler and no mapping table that could drift away from the classes.

Settings are resolved inside the lifespan rather than at import, so
``uvicorn app.api:app`` starts even with a missing or broken key and reports it
as a 503 per request, instead of failing to import and crash-looping.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.exceptions import ConfigError, RankerError
from app.schemas import (
    HealthResponse,
    RankBatchRequest,
    RankSingleRequest,
    RankingResult,
)
from app.service import ResumeRanker

logger = logging.getLogger(__name__)

API_TITLE = "Resumes Ranker"
API_VERSION = "1.0.0"


def create_app(
    settings: Optional[Settings] = None,
    ranker: Optional[ResumeRanker] = None,
) -> FastAPI:
    """Build the application.

    Pass ``ranker`` to inject a stub in tests, or ``settings`` to override
    configuration. With neither, the process-wide settings are used.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            resolved = ranker or ResumeRanker.from_settings(settings)
        except ConfigError as exc:
            # Serve the error instead of refusing to boot, so `GET /health` can
            # report the misconfiguration and an operator can fix it without
            # reading tracebacks. Every ranking call returns 503 meanwhile.
            logger.error("not configured: %s", exc)
            app.state.settings = settings or Settings.for_reporting()
            app.state.ranker = None
            app.state.error = exc
            yield
            return

        app.state.settings = resolved.settings
        app.state.ranker = resolved
        app.state.error = None
        logger.info(
            "Resumes Ranker ready: model=%s base_url=%s key=%s",
            resolved.settings.jev_model,
            resolved.settings.jev_base_url,
            "mock" if resolved.settings.uses_mock_key else "configured",
        )
        try:
            yield
        finally:
            await resolved.aclose()

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        summary="Rank resumes against weighted job criteria with TypeSafe Jev.",
        lifespan=lifespan,
    )

    # One handler for the whole hierarchy. The class knows its own status code.
    @app.exception_handler(RankerError)
    async def handle_ranker_error(_: Request, exc: RankerError) -> JSONResponse:
        if exc.http_status >= 500:
            logger.warning("%s: %s", type(exc).__name__, exc)
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

    def get_ranker(request: Request) -> ResumeRanker:
        """The shared ranker, or the reason there isn't one."""
        error: Optional[RankerError] = getattr(request.app.state, "error", None)
        if error is not None:
            raise error
        return request.app.state.ranker

    RankerDep = Depends(get_ranker)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health(request: Request) -> HealthResponse:
        """Liveness and loaded configuration. Never calls the provider.

        Upstream is not contacted on purpose: a health check that fails when
        OpenRouter is down turns a dependency blip into an outage here, and it
        reports less than the local configuration does.
        """
        settings = request.app.state.settings
        return HealthResponse(
            status="ok",
            model=settings.jev_model,
            base_url=settings.jev_base_url,
            key_configured=bool(settings.openrouter_api_key),
            mock_key=settings.uses_mock_key,
            max_concurrency=settings.jev_max_concurrency,
            max_requests_per_second=settings.jev_max_rps,
            max_batch_size=settings.jev_max_batch_size,
        )

    @app.post("/rank/single", response_model=RankingResult, tags=["rank"])
    async def rank_single(
        payload: RankSingleRequest,
        ranker: ResumeRanker = RankerDep,
    ) -> RankingResult:
        """Rank one candidate. Upstream errors are raised, not hidden.

        Wrapped in a one-element :class:`RankingResult` rather than returned as a
        bare ``CandidateScore``, so a client only ever parses one response shape.
        """
        started = time.monotonic()
        score = await ranker.rank_single(
            job_description=payload.job_description,
            criteria=payload.criteria,
            candidate=payload.candidate,
        )
        return RankingResult(
            model=ranker.settings.jev_model,
            results=[score],
            total_cost_usd=score.cost_usd or 0.0,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    @app.post("/rank/batch", response_model=RankingResult, tags=["rank"])
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

    return app


def main() -> None:
    """Console-script entry point: run the API with uvicorn."""
    import uvicorn

    from app.config import get_settings

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    uvicorn.run(
        "app.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


# Module-level app for `uvicorn app.api:app`. Settings are resolved in the
# lifespan, so importing this module never requires a configured key.
app = create_app()


__all__ = ["app", "create_app", "main", "API_TITLE", "API_VERSION"]
