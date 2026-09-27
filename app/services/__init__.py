"""The two stateful services, split by what they know about.

- :mod:`~app.services.jev_service` - talking to the Jev Decisions API. Knows
  about HTTP, retries, rate limits, and error classification. Knows nothing
  about resumes.
- :mod:`~app.services.ranking_service` - judging candidates. Knows how job
  requirements map to Jev questions and how answers become a score. Uses the Jev
  service for transport and never makes an HTTP call itself.

Both modules hold objects and nothing else. The functions they are built from
are in :mod:`app.utils`, and the shapes they pass around are in
:mod:`app.schemas`, so a question about retry behaviour, about scoring, or about
the wire format has exactly one place to be answered.

Import from ``app.services`` rather than the submodules; this re-export is the
supported surface.
"""

from __future__ import annotations

from app.services.jev_service import (
    APP_TITLE,
    JevClient,
    JevSyncClient,
    RateLimiter,
)
from app.services.ranking_service import ResumeRanker

__all__ = [
    "APP_TITLE",
    "JevClient",
    "JevSyncClient",
    "RateLimiter",
    "ResumeRanker",
]
