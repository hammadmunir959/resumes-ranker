"""Every shape in the project, split by whose API it describes.

Three modules, because they answer to different owners:

- :mod:`~app.schemas.jev_schemas` - the Jev Decisions wire format. Somebody
  else's API; changes when they change it.
- :mod:`~app.schemas.ranking_schemas` - what this program judges: criteria,
  candidates, scores, and the request envelopes the API wraps them in.
- :mod:`~app.schemas.health_schemas` - what ``/health`` reports about this
  process.

The numeric policy those shapes are bounded by lives in :mod:`app.policy`, not
here. This package is shapes only.

Import from ``app.schemas`` rather than the submodules: this re-export is the
supported surface, so the internal split can change without touching callers.
"""

from __future__ import annotations

from app.schemas.health_schemas import HealthResponse
from app.schemas.jev_schemas import (
    ChoiceAnswer,
    JevResponse,
    Question,
    Questions,
    ScoreAnswer,
    Usage,
)
from app.schemas.ranking_schemas import (
    DEFAULT_RUBRIC,
    Candidate,
    CandidateScore,
    Criterion,
    CriterionOutcome,
    RankBatchRequest,
    RankingResult,
    RankRequestBase,
    RankSingleRequest,
)

__all__ = [
    # Jev wire format
    "Question",
    "Questions",
    "ScoreAnswer",
    "ChoiceAnswer",
    "Usage",
    "JevResponse",
    # Ranking domain
    "DEFAULT_RUBRIC",
    "Criterion",
    "Candidate",
    "CriterionOutcome",
    "CandidateScore",
    "RankingResult",
    "RankRequestBase",
    "RankSingleRequest",
    "RankBatchRequest",
    # Process reporting
    "HealthResponse",
]
