"""FastAPI application entry point, lifespan, error handling, and health check."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api import ranking_router
from app.api.limiter import limiter
from app.config import Settings
from app.exceptions import ConfigError, RankerError
from app.schemas import HealthResponse
from app.services import ResumeRanker
from app.utils import resolve_fallback

logger = logging.getLogger(__name__)

API_TITLE = "Resumes Ranker"
API_VERSION = "1.0.0"


def create_app(
    settings: Optional[Settings] = None,
    ranker: Optional[ResumeRanker] = None,
) -> FastAPI:
    """Build the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            resolved = ranker or ResumeRanker.from_settings(settings)
        except ConfigError as exc:
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
            "Resumes Ranker ready: model=%s fallback=%s base_url=%s key=%s",
            resolved.settings.jev_model,
            resolve_fallback(resolved.settings, resolved.settings.jev_model) or "none",
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
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RankerError)
    async def handle_ranker_error(_: Request, exc: RankerError) -> JSONResponse:
        """Handle all RankerError subclasses with their declared status code."""
        if exc.http_status >= 500:
            logger.warning("%s: %s", type(exc).__name__, exc)
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health(request: Request) -> HealthResponse:
        """Return liveness status and loaded configuration."""
        settings = request.app.state.settings
        return HealthResponse(
            status="ok",
            model=settings.jev_model,
            fallback_model=resolve_fallback(settings, settings.jev_model) or "",
            base_url=settings.jev_base_url,
            key_configured=bool(settings.openrouter_api_key),
            mock_key=settings.uses_mock_key,
            max_concurrency=settings.jev_max_concurrency,
            max_requests_per_second=settings.jev_max_rps,
            max_batch_size=settings.jev_max_batch_size,
        )

    app.include_router(ranking_router.router)
    return app


def main() -> None:
    """Run API server with uvicorn using loaded configuration."""
    import uvicorn

    from app.config import get_settings

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


app = create_app()

__all__ = ["app", "create_app", "main", "API_TITLE", "API_VERSION"]
