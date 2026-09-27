"""Pure helper functions for Jev decisions and resume ranking."""

from __future__ import annotations

from app.utils.jev_utils import (
    FALLBACK_ERRORS,
    as_number,
    json_or_raise,
    parse_retry_after,
    resolve_fallback,
)
from app.utils.ranking_utils import (
    CHARS_PER_TOKEN,
    GATE_MARGIN,
    GATE_THRESHOLD,
    JEV_CONTEXT_TOKENS,
    REVIEW_CONFIDENCE_FLOOR,
    SCORE_SCALE,
    aggregate,
    aggregate_gate,
    aggregate_score,
    build_questions,
    build_state,
    estimate_tokens,
    model_that_answered,
    sort_scores,
)

__all__ = [
    # Jev call helpers
    "FALLBACK_ERRORS",
    "resolve_fallback",
    "as_number",
    "parse_retry_after",
    "json_or_raise",
    # Ranking pipeline
    "SCORE_SCALE",
    "GATE_THRESHOLD",
    "GATE_MARGIN",
    "REVIEW_CONFIDENCE_FLOOR",
    "JEV_CONTEXT_TOKENS",
    "CHARS_PER_TOKEN",
    "build_questions",
    "build_state",
    "estimate_tokens",
    "aggregate",
    "aggregate_gate",
    "aggregate_score",
    "model_that_answered",
    "sort_scores",
]
