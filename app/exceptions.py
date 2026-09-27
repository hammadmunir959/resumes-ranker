"""Project exception hierarchy with HTTP status codes and payloads."""

from __future__ import annotations

from typing import Any, ClassVar, Optional


class RankerError(RuntimeError):
    """Base exception for all application errors."""

    http_status: ClassVar[int] = 500
    error_code: ClassVar[str] = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        detail: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.detail = detail or {}

    @property
    def retriable(self) -> bool:
        """Whether another attempt could plausibly succeed."""
        return False

    def to_payload(self) -> dict[str, Any]:
        """Serialize error to an API JSON response dict."""
        payload: dict[str, Any] = {
            "error": self.error_code,
            "detail": str(self),
        }
        if self.status_code is not None:
            payload["upstream_status"] = self.status_code
        if self.detail:
            payload["context"] = self.detail
        return payload


class InputValidationError(RankerError):
    """Request validation failure (HTTP 400)."""

    http_status = 400
    error_code = "invalid_request"


class ConfigError(RankerError):
    """Service configuration failure (HTTP 503)."""

    http_status = 503
    error_code = "not_configured"


class JevError(RankerError):
    """Base class for Decisions API failures (HTTP 502)."""

    http_status = 502
    error_code = "jev_error"


class JevAuthError(JevError):
    """Upstream authentication failure (HTTP 502)."""

    http_status = 502
    error_code = "jev_auth_failed"


class JevCreditsError(JevError):
    """Upstream out-of-credits failure (HTTP 402)."""

    http_status = 402
    error_code = "jev_out_of_credit"


class JevProtocolError(JevError):
    """Upstream malformed response or protocol failure (HTTP 502)."""

    http_status = 502
    error_code = "jev_bad_response"


class JevTransientError(JevError):
    """Transient upstream failure safe to retry (HTTP 503)."""

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
