"""The HTTP layer: routers only.

- :mod:`~app.api.ranking_router` - the ranking routes, kept thin enough to read
  as an API description.

Application assembly is not here; it is :mod:`app.main`, so this package holds
routes and nothing else. Neither module here makes a provider call or computes a
score - both delegate to :mod:`app.services`.

Import from ``app.api`` rather than the submodule; this re-export is the
supported surface.
"""

from __future__ import annotations

from app.api import ranking_router
from app.api.ranking_router import get_ranker

__all__ = ["ranking_router", "get_ranker"]
