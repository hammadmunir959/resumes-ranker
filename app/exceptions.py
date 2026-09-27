"""Every error this project raises, in one place.

Each class declares the HTTP status and machine-readable code the API should
return, so ``app/api.py`` needs a single exception handler and a lookup table
rather than a hand-maintained mapping that can drift out of sync.

The split that matters for callers:

* :class:`InputValidationError` - the request is wrong. The caller fixes it.
* :class:`ConfigError` - this deployment is wrong. An operator fixes it.
* :class:`JevError` and subclasses - OpenRouter refused or failed. Retrying
  helps only for :class:`JevTransientError`; the rest will fail identically
  forever, which is why the retry logic keys off exactly that distinction.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional


class RankerError(RuntimeError):
    """Base class for everything this project raises."""

    #: HTTP status the API returns for this error.
    http_status: ClassVar[int] = 500
    #: Stable identifier clients can branch on.
    error_code: ClassVar[str] = "internal_error"

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 retry_after: Optional[float] = None, detail: Optional[dict[str, Any]] = None):
        super().__init__(message)
        #: HTTP status reported by the upstream provider, when there was one.
        self.status_code = status_code
        #: Seconds to wait, taken from a ``Retry-After`` header.
        self.retry_after = retry_after
        self.detail = detail or {}

    @property
    def retriable(self) -> bool:
        """Whether another attempt could plausibly succeed."""
        return False

    def to_payload(self) -> dict[str, Any]:
        """The JSON body the API returns for this error."""
        payload: dict[str, Any] = {
            "error": self.error_code,
            "detail": str(self),
        }
        if self.status_code is not None:
            payload["upstream_status"] = self.status_code
        if self.detail:
            payload["context"] = self.detail
        return payload


# --------------------------------------------------------------------------- #
# Caller and operator errors
# --------------------------------------------------------------------------- #

class InputValidationError(RankerError):
    """The request cannot produce a meaningful ranking. Retrying will not help."""

    http_status = 400
    error_code = "invalid_request"


class ConfigError(RankerError):
    """The service is not configured correctly, so it cannot serve requests."""

    http_status = 503
    error_code = "not_configured"


# --------------------------------------------------------------------------- #
# Upstream errors
# --------------------------------------------------------------------------- #

class JevError(RankerError):
    """Base class for Decisions API failures."""

    http_status = 502
    error_code = "jev_error"


class JevAuthError(JevError):
    """401/403 - the API key is missing, invalid, or lacks access.

    A bad key fails for every request, so it is never retried.
    """

    http_status = 502
    error_code = "jev_auth_failed"


class JevCreditsError(JevError):
    """402 - the account is out of credit.

    Surfaces as 402 rather than 5xx because it is unambiguous: the request was
    fine, the account just cannot be paid for.
    """

    http_status = 402
    error_code = "jev_out_of_credit"


class JevProtocolError(JevError):
    """A rejected request shape, an unreadable body, or an unparseable answer.

    Retrying would return the same rejection, so these fail fast.
    """

    http_status = 502
    error_code = "jev_bad_response"


class JevTransientError(JevError):
    """429, 5xx, a timeout, or an exhausted in-flight budget. Safe to retry."""

    http_status = 503
    error_code = "jev_temporarily_unavailable"

    @property
    def retriable(self) -> bool:
        return True


__all__ = [
    "RankerError",
    "InputValidationError",
    "ConfigError",
    "JevError",
    "JevAuthError",
    "JevCreditsError",
    "JevProtocolError",
    "JevTransientError",
]
