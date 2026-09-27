"""Pure helpers for talking to the Jev Decisions API.

No state and no I/O: these are the small decisions the client would otherwise
have to make inline - which model to fall back to, whether a header is a usable
delay, what to do with a ``200`` that is not JSON. Keeping them out of
:mod:`app.services.jev_service` leaves that module as the client and nothing
else, and lets the API reuse the fallback decision rather than restate it.

Import from ``app.utils`` rather than the submodules; this re-export is the
supported surface.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.config import Settings
from app.exceptions import JevError, JevProtocolError, JevTransientError

logger = logging.getLogger(__name__)

#: Failures worth retrying against a different model. Both mean the model or its
#: provider did not produce a usable answer, which a second model can fix.
#: Deliberately excludes auth and credit errors: those belong to the account, and
#: the fallback would be turned away for exactly the same reason.
FALLBACK_ERRORS: tuple[type[JevError], ...] = (JevTransientError, JevProtocolError)


def resolve_fallback(
    settings: Settings, primary: str, explicit: Optional[str] = None
) -> Optional[str]:
    """The model to try when ``primary`` is the reason a call failed.

    ``None`` means there is nothing to fall back to. That covers an empty
    ``jev_fallback_model``, a fallback naming the primary - retrying the model
    that just failed would fail identically - and an explicit ``model=`` passed
    to the client, where the caller asked for that one model and is not
    expecting a substitute.

    Module-level and shared so that ``/health`` reports the fallback that will
    actually be used rather than re-deciding it and drifting from the client.
    """
    if explicit is not None:
        return None
    candidate = settings.jev_fallback_model.strip()
    if not candidate or candidate == primary:
        return None
    return candidate


def as_number(value: Any) -> Optional[float]:
    """A float, or None. Bools are rejected: ``True`` is not a measurement."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Seconds to wait from a ``Retry-After`` header, or None if unusable."""
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def json_or_raise(response: httpx.Response) -> Any:
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
    "FALLBACK_ERRORS",
    "resolve_fallback",
    "as_number",
    "parse_retry_after",
    "json_or_raise",
]
