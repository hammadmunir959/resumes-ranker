"""Async client for the TypeSafe Jev Decisions API on OpenRouter."""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Optional

import httpx

from app.config import Settings
from app.exceptions import (
    JevAuthError,
    JevCreditsError,
    JevError,
    JevProtocolError,
    JevTransientError,
)
from app.schemas import JevResponse, Question
from app.utils.jev_utils import (
    FALLBACK_ERRORS,
    as_number,
    json_or_raise,
    parse_retry_after,
    resolve_fallback,
)

logger = logging.getLogger(__name__)

#: Advertised to OpenRouter so the account's usage page is readable.
APP_TITLE = "resumes-ranker"


class JevClient:
    """Async client for POST /api/alpha/decisions."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        client: Optional[httpx.AsyncClient] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ):
        self.settings = settings
        self._sleep = sleep
        self._jitter = jitter
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            transport=transport, timeout=settings.jev_timeout
        )

    @classmethod
    def from_settings(
        cls, settings: Optional[Settings] = None, **kwargs: Any
    ) -> "JevClient":
        """Build a client, defaulting to the process-wide settings."""
        from app.config import get_settings

        return cls(settings or get_settings(), **kwargs)

    async def decide(
        self,
        *,
        state: Any,
        questions: Mapping[str, Question],
        metadata: Optional[Mapping[str, Any]] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> JevResponse:
        """Ask the model to answer every question against one state."""
        if not questions:
            raise ValueError("at least one question is required")
        body: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if metadata:
            body["metadata"] = dict(metadata)
        if session_id:
            body["session_id"] = session_id[:256]
        return await self._call(model=model, body=body)

    async def _call(self, *, model: Optional[str], body: dict[str, Any]) -> JevResponse:
        """Ask one model, falling back to another if this one is the problem."""
        primary = model or self.settings.jev_model
        fallback = resolve_fallback(self.settings, primary, explicit=model)
        attempts = [0]

        try:
            return await self._attempt(primary, body, attempts)
        except FALLBACK_ERRORS as exc:
            if fallback is None:
                raise
            logger.warning(
                "model %s failed (%s) - falling back to %s",
                primary, exc, fallback,
            )
            try:
                return await self._attempt(fallback, body, attempts)
            except JevError as fallback_exc:
                raise JevTransientError(
                    f"model {primary} failed ({exc}) and fallback {fallback} "
                    f"failed too ({fallback_exc})"
                ) from fallback_exc

    async def _attempt(
        self, model: str, body: dict[str, Any], attempts: list[int]
    ) -> JevResponse:
        """One model's retry loop: backoff, Retry-After, then give up."""
        payload = {"model": model, **body}
        last: Optional[JevError] = None
        max_attempts = self.settings.jev_max_retries

        for attempt in range(1, max_attempts + 1):
            attempts[0] += 1
            try:
                response = await self.client.post(
                    self.settings.jev_base_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.jev_timeout,
                )
            except httpx.TimeoutException as exc:
                last = JevTransientError(
                    f"request timed out after {self.settings.jev_timeout}s: {exc}"
                )
            except httpx.TransportError as exc:
                last = JevTransientError(f"network error: {exc}")
            else:
                if response.is_success:
                    return self._parse(json_or_raise(response), attempts[0])
                error = self.classify(response)
                if not isinstance(error, JevTransientError):
                    raise error
                last = error

            if attempt == max_attempts:
                break
            delay = self._backoff(attempt, last)
            logger.warning(
                "Jev call to %s failed (attempt %d/%d): %s - retrying in %.1fs",
                model, attempt, max_attempts, last, delay,
            )
            await self._sleep(delay)

        raise JevTransientError(
            f"Jev call to {model} failed after {max_attempts} attempts: {last}"
        )

    def _headers(self) -> dict[str, str]:
        """Return HTTP headers for Decisions API requests."""
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": APP_TITLE,
        }

    def _backoff(self, attempt: int, error: Optional[JevError]) -> float:
        """Exponential backoff with jitter, capped and honoring Retry-After."""
        if error is not None and error.retry_after is not None:
            return min(max(error.retry_after, 0.0), self.settings.jev_backoff_max)
        base = self.settings.jev_backoff_base * (2 ** (attempt - 1))
        capped = min(base, self.settings.jev_backoff_max)
        return capped * (0.5 + 0.5 * self._jitter())

    def classify(self, response: httpx.Response) -> JevError:
        """Map a failed response onto the narrowest error that fits."""
        status = response.status_code
        retry_after = parse_retry_after(response.headers.get("Retry-After"))
        message = f"Decisions API returned HTTP {status}"
        limit_source: Optional[str] = None

        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                metadata = error.get("metadata")
                if isinstance(metadata, dict):
                    limit_source = metadata.get("limit_source")

        if status == 429 or status >= 500:
            return JevTransientError(message, status_code=status, retry_after=retry_after)
        if status == 402:
            if limit_source == "openrouter_in_flight_budget":
                return JevTransientError(
                    message, status_code=status, retry_after=retry_after
                )
            return JevCreditsError(message, status_code=status)
        if status in (401, 403):
            return JevAuthError(message, status_code=status)
        return JevProtocolError(message, status_code=status)

    @staticmethod
    def _parse(body: Any, attempts: int = 1) -> JevResponse:
        """Parse Decisions API JSON response body into a JevResponse."""
        if not isinstance(body, dict):
            raise JevProtocolError("Decisions API returned a non-object body")
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise JevProtocolError("Decisions API response had no 'answers' object")
        usage = body.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return JevResponse(
            answers=answers,
            model=str(body.get("model", "")),
            cost_usd=as_number(usage.get("cost")) or 0.0,
            input_tokens=int(as_number(usage.get("input_tokens")) or 0),
            output_tokens=int(as_number(usage.get("output_tokens")) or 0),
            attempts=attempts,
            raw=body,
        )

    async def aclose(self) -> None:
        """Close the underlying client, if this instance created it."""
        if self._owns_client:
            await self.client.aclose()

    async def __aenter__(self) -> "JevClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


class JevSyncClient:
    """Synchronous wrapper for JevClient."""

    def __init__(self, settings: Optional[Settings] = None, **kwargs: Any):
        from app.config import get_settings

        self._client = JevClient(settings or get_settings(), **kwargs)
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls, settings: Optional[Settings] = None, **kwargs: Any
    ) -> "JevSyncClient":
        """Build a sync client from settings."""
        return cls(settings, **kwargs)

    @property
    def client(self) -> JevClient:
        """The underlying async client."""
        return self._client

    def decide(
        self,
        *,
        state: Any,
        questions: Mapping[str, Question],
        metadata: Optional[Mapping[str, Any]] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> JevResponse:
        """Execute decision synchronously."""
        with self._lock:
            return asyncio.run(
                self._client.decide(
                    state=state, questions=questions, metadata=metadata,
                    session_id=session_id, model=model,
                )
            )

    def close(self) -> None:
        """Close the synchronous client."""
        with self._lock:
            asyncio.run(self._client.aclose())


__all__ = [
    "APP_TITLE",
    "FALLBACK_ERRORS",
    "resolve_fallback",
    "JevClient",
    "JevSyncClient",
    "JevError",
    "JevAuthError",
    "JevCreditsError",
    "JevProtocolError",
    "JevTransientError",
]
