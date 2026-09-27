"""Health check response schema."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["HealthResponse"]


class HealthResponse(BaseModel):
    """Liveness and process configuration report."""

    status: str
    model: str
    fallback_model: str = ""
    base_url: str
    key_configured: bool
    mock_key: bool = False
    max_concurrency: int
    max_requests_per_second: float
    max_batch_size: int
