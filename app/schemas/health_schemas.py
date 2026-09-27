"""The shape of the ``/health`` response.

Its own module because it is the one schema that describes this process rather
than a ranking: it reports loaded configuration so an operator can see what a
failing deployment was actually running. It must never carry the API key.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["HealthResponse"]


class HealthResponse(BaseModel):
    """Liveness plus the configuration a client would need to debug a failure.

    Reports the base URL, the primary model, and the fallback model so an
    operator can see what the process actually loaded. The key is never
    included, only whether one was found.
    """

    status: str
    model: str
    fallback_model: str = ""
    base_url: str
    key_configured: bool
    mock_key: bool = False
    max_concurrency: int
    max_requests_per_second: float
    max_batch_size: int
