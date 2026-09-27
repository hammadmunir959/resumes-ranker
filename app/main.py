"""The process: the ASGI app, its lifespan, and the uvicorn entry point.

Everything that belongs to the process rather than to a feature lives here -
startup and shutdown, the single error handler, the health endpoint - and the
feature routers are attached to it. Adding a feature therefore means adding a
router in :mod:`app.api`, not editing this file.

This is also the module a deployment points at: ``uvicorn app.main:app``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import ranking_router
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

    # One handler for the whole hierarchy. The class knows its own status code.
    @app.exception_handler(RankerError)
    async def handle_ranker_error(_: Request, exc: RankerError) -> JSONResponse:
        if exc.http_status >= 500:
            logger.warning("%s: %s", type(exc).__name__, exc)
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

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
    """Console-script entry point: run the API with uvicorn."""
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


# Module-level app for `uvicorn app.main:app`. Settings are resolved in the
# lifespan, so importing this module never requires a configured key.
app = create_app()


__all__ = ["app", "create_app", "main", "API_TITLE", "API_VERSION"]
