"""Async client for the TypeSafe Jev Decisions API on OpenRouter.

This layer knows nothing about resumes. It takes questions, asks the model, and
returns a typed :class:`~app.schemas.JevResponse`, handling rate limiting,
retries, and error classification along the way. Use it directly for anything
Jev can answer; :mod:`app.service` builds the resume-specific questions on top.

Retry policy, and why it is not simply "retry everything":

* ``429`` and ``5xx`` are retried with exponential backoff, honoring
  ``Retry-After``.
* ``402`` is retried **only** when the error says the in-flight spend cap was
  hit, because that cap clears itself. An empty balance never will, so retrying
  it just delays the real message.
* ``401``, ``403``, and other ``4xx`` fail immediately. They are deterministic,
  so four identical attempts would bury the actual cause.

The clock and sleep functions are injectable, which is what lets the tests
exercise backoff and rate limiting without real delays.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

import httpx

from app.config import Settings
from app.exceptions import (
    JevAuthError,
    JevCreditsError,
    JevError,
    JevProtocolError,
    JevTransientError,
)
from app.schemas import JevResponse

logger = logging.getLogger(__name__)

#: A question as Jev expects it. Kept as a plain dict so the wire format stays
#: visible and unmangled.
Question = dict[str, Any]

#: Advertised to OpenRouter so the account's usage page is readable.
APP_TITLE = "resumes-ranker"


# --------------------------------------------------------------------------- #
# Question builders
# --------------------------------------------------------------------------- #

class Questions:
    """Builders for the three question types Jev supports.

    Static methods, so a caller writes ``Questions.noul(...)`` with no instance
    and no import of the module-level names.
    """

    @staticmethod
    def noul(
        instructions: str,
        *,
        true_criteria: str = "",
        false_criteria: str = "",
    ) -> Question:
        """A yes/no question. The answer is the probability that it is yes."""
        question: Question = {"type": "noul", "instructions": instructions}
        if true_criteria or false_criteria:
            question["criteria"] = {
                "true": true_criteria or instructions,
                "false": false_criteria or f"Does not hold: {instructions}",
            }
        return question

    @staticmethod
    def score(instructions: str, rubric: Sequence[str]) -> Question:
        """An ordered scale, where index 0 is the lowest level.

        The order is significant, so the best level belongs last.
        """
        levels = list(rubric)
        if len(levels) < 2:
            raise ValueError("a score question needs at least 2 rubric levels")
        return {"type": "score", "instructions": instructions, "criteria": levels}

    @staticmethod
    def choice(instructions: str, options: Mapping[str, str]) -> Question:
        """Pick one of several labeled options."""
        if not options:
            raise ValueError("a choice question needs at least one option")
        return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Sliding window: at most ``rate`` acquisitions per ``period`` seconds.

    The wait happens *outside* the lock. Holding it while sleeping, the obvious
    implementation, would let one throttled caller block every other caller from
    even re-checking the window.
    """

    def __init__(
        self,
        rate: float,
        period: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = self._clock()
                self._times = [t for t in self._times if now - t < self.period]
                if len(self._times) < self.rate:
                    self._times.append(now)
                    return
                wait = self.period - (now - self._times[0])
            await self._sleep(max(wait, 0.001))


# --------------------------------------------------------------------------- #
# Async client
# --------------------------------------------------------------------------- #

class JevClient:
    """Async client for ``POST /api/alpha/decisions``."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        client: Optional[httpx.AsyncClient] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
    ):
        self.settings = settings
        self._sleep = sleep
        self._jitter = jitter
        self.limiter = RateLimiter(settings.jev_max_rps, clock=clock, sleep=sleep)
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
        """Ask the model to answer every question against one state.

        ``state`` is the subject being judged, and ``questions`` maps the answer
        key to the question itself, so the reply can be matched back up.

        Every question in one request is answered in parallel and the answers
        cannot see each other, so unrelated questions about the same state cost
        nothing extra to batch together.
        """
        if not questions:
            raise ValueError("at least one question is required")
        body: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if metadata:
            body["metadata"] = dict(metadata)
        if session_id:
            # OpenRouter groups these in Broadcast and its private logs, and
            # never forwards the value to the provider.
            body["session_id"] = session_id[:256]
        return await self._call(model=model, body=body)

    async def _call(self, *, model: Optional[str], body: dict[str, Any]) -> JevResponse:
        payload = {"model": model or self.settings.jev_model, **body}
        last: Optional[JevError] = None
        max_attempts = self.settings.jev_max_retries

        for attempt in range(1, max_attempts + 1):
            await self.limiter.acquire()
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
                    return self._parse(_json_or_raise(response), attempt)
                error = self.classify(response)
                if not isinstance(error, JevTransientError):
                    raise error
                last = error

            if attempt == max_attempts:
                break
            delay = self._backoff(attempt, last)
            logger.warning(
                "Jev call failed (attempt %d/%d): %s - retrying in %.1fs",
                attempt, max_attempts, last, delay,
            )
            await self._sleep(delay)

        raise JevTransientError(
            f"Jev call failed after {max_attempts} attempts: {last}"
        )

    def _headers(self) -> dict[str, str]:
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
        # Jitter keeps a burst of concurrent candidates from retrying in lockstep
        # and re-triggering the same rate limit.
        return capped * (0.5 + 0.5 * self._jitter())

    def classify(self, response: httpx.Response) -> JevError:
        """Map a failed response onto the narrowest error that fits."""
        status = response.status_code
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))
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
            cost_usd=_number(usage.get("cost")) or 0.0,
            input_tokens=int(_number(usage.get("input_tokens")) or 0),
            output_tokens=int(_number(usage.get("output_tokens")) or 0),
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


# --------------------------------------------------------------------------- #
# Blocking client
# --------------------------------------------------------------------------- #

class JevSyncClient:
    """Blocking client, for scripts and notebooks.

    A thin wrapper over a private event loop, so synchronous callers get the
    same retry and rate-limiting behavior as the async client rather than a
    second implementation that could drift.
    """

    def __init__(self, settings: Optional[Settings] = None, **kwargs: Any):
        from app.config import get_settings

        self._client = JevClient(settings or get_settings(), **kwargs)
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls, settings: Optional[Settings] = None, **kwargs: Any
    ) -> "JevSyncClient":
        return cls(settings, **kwargs)

    @property
    def client(self) -> JevClient:
        """The underlying async client, for parity with :class:`JevClient`."""
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
        # Serialized: one private loop cannot be entered from two threads.
        with self._lock:
            return asyncio.run(
                self._client.decide(
                    state=state, questions=questions, metadata=metadata,
                    session_id=session_id, model=model,
                )
            )

    def close(self) -> None:
        with self._lock:
            asyncio.run(self._client.aclose())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _json_or_raise(response: httpx.Response) -> Any:
    """Parse a successful body, or explain that the gateway is broken.

    A ``200`` that is not JSON is a proxy or gateway returning an HTML error
    page. It will not parse on a retry either, so this fails fast rather than
    escaping as a bare ``JSONDecodeError``.
    """
    try:
        return response.json()
    except ValueError as exc:
        raise JevProtocolError(
            f"Decisions API returned a non-JSON body: {exc}"
        ) from exc


__all__ = [
    "APP_TITLE",
    "Question",
    "Questions",
    "RateLimiter",
    "JevClient",
    "JevSyncClient",
    "JevError",
    "JevAuthError",
    "JevCreditsError",
    "JevProtocolError",
    "JevTransientError",
]
