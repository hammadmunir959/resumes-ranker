"""Pure helpers for talking to the Jev Decisions API."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.config import Settings
from app.exceptions import JevError, JevProtocolError, JevTransientError

logger = logging.getLogger(__name__)

FALLBACK_ERRORS: tuple[type[JevError], ...] = (JevTransientError, JevProtocolError)


def resolve_fallback(
    settings: Settings, primary: str, explicit: Optional[str] = None
) -> Optional[str]:
    """Return fallback model if primary fails and no explicit model was requested."""
    if explicit is not None:
        return None
    candidate = settings.jev_fallback_model.strip()
    return None if (not candidate or candidate == primary) else candidate


def as_number(value: Any) -> Optional[float]:
    """Return float value, rejecting booleans and non-numeric types."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse seconds to wait from a Retry-After header."""
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def json_or_raise(response: httpx.Response) -> Any:
    """Parse JSON response body or raise JevProtocolError."""
    try:
        return response.json()
    except ValueError as exc:
        raise JevProtocolError(
            f"Decisions API returned a non-JSON body: {exc}"
        ) from exc


__all__ = [
    "FALLBACK_ERRORS",
    "resolve_fallback",
    "as_number",
    "parse_retry_after",
    "json_or_raise",
]
