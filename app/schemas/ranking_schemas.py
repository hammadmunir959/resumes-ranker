"""Domain and API schemas for resume ranking."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import (
    DEFAULT_RUBRIC,
    MAX_APPLICATION_FORM,
    MAX_CANDIDATE_ID,
    MAX_CANDIDATES,
    MAX_CRITERIA,
    MAX_CRITERION_NAME,
    MAX_JOB_DESCRIPTION,
    MAX_RESUME_TEXT,
)
from app.schemas.jev_schemas import Usage

__all__ = [
    "DEFAULT_RUBRIC",
    "MAX_CRITERIA",
    "MAX_CANDIDATES",
    "MAX_CANDIDATE_ID",
    "MAX_JOB_DESCRIPTION",
    "MAX_RESUME_TEXT",
    "MAX_APPLICATION_FORM",
    "MAX_CRITERION_NAME",
    "Criterion",
    "Candidate",
    "CriterionOutcome",
    "CandidateScore",
    "RankingResult",
    "RankRequestBase",
    "RankSingleRequest",
    "RankBatchRequest",
]


class Criterion(BaseModel):
    """One job requirement evaluated against candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=MAX_CRITERION_NAME)
    name: str = Field(min_length=1, max_length=MAX_CRITERION_NAME)
    description: str = Field(default="", max_length=MAX_JOB_DESCRIPTION)
    required: bool = False
    weight: float = Field(default=1.0, ge=0.0, le=1000.0)
    rubric: tuple[str, ...] = DEFAULT_RUBRIC

    @field_validator("id", "name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("rubric")
    @classmethod
    def _usable_rubric(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 2:
            raise ValueError("a rubric needs at least two ordered outcome levels")
        if any(not level.strip() for level in value):
            raise ValueError("rubric levels must not be blank")
        return tuple(level.strip() for level in value)

    @model_validator(mode="after")
    def _coherent(self) -> "Criterion":
        if not self.required and self.weight <= 0:
            raise ValueError("an optional criterion needs weight > 0")
        return self

    @property
    def max_index(self) -> float:
        return float(len(self.rubric) - 1)

    @property
    def is_gate(self) -> bool:
        return self.required

    def rubric_at(self, index: float) -> str:
        return self.rubric[max(0, min(round(index), len(self.rubric) - 1))]

    def weighted_score(self, index: float) -> float:
        return max(0.0, min(index / self.max_index, 1.0)) if self.max_index > 0 else 0.0


class Candidate(BaseModel):
    """Candidate resume and application details."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=MAX_CANDIDATE_ID)
    resume_text: str = Field(min_length=1, max_length=MAX_RESUME_TEXT)
    application_form_text: str = Field(default="", max_length=MAX_APPLICATION_FORM)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate_id must not be blank")
        return value.strip()

    @property
    def text_for_matching(self) -> str:
        if self.application_form_text.strip():
            return f"{self.resume_text}\n\n--- Application form ---\n{self.application_form_text}"
        return self.resume_text

    def estimated_tokens(self, chars_per_token: int = 4) -> int:
        return max(1, len(self.text_for_matching) // chars_per_token)


class CriterionOutcome(BaseModel):
    """Evaluation outcome for one criterion on a candidate."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    name: str
    required: bool
    passed: Optional[bool] = None
    label: str = ""
    raw_score: float = 0.0
    score_0_100: float = 0.0
    confidence: Optional[float] = None
    needs_review: bool = False
    error: Optional[str] = None

    @property
    def weight_used(self) -> float:
        return 0.0 if self.required else 1.0


class CandidateScore(BaseModel):
    """Evaluation results and overall score for one candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    gate_passed: bool = True
    model: str = ""
    overall_score: Optional[float] = None
    failed_gates: list[str] = Field(default_factory=list)
    needs_review: bool = False
    cost_usd: Optional[float] = None
    attempts: int = 0
    per_criterion: list[CriterionOutcome] = Field(default_factory=list)
    error: Optional[str] = None
    usage: Optional[Usage] = None

    @property
    def scored(self) -> bool:
        return self.error is None

    @classmethod
    def failed(
        cls,
        candidate_id: str,
        error: str,
        *,
        failed_gates: Optional[list[str]] = None,
    ) -> "CandidateScore":
        return cls(
            candidate_id=candidate_id,
            gate_passed=not failed_gates,
            failed_gates=failed_gates or [],
            needs_review=True,
            error=error,
        )


class RankingResult(BaseModel):
    """Overall ranking result for a pool of candidates."""

    model_config = ConfigDict(extra="forbid")

    model: str
    results: list[CandidateScore] = Field(default_factory=list)
    total_cost_usd: Optional[float] = None
    elapsed_seconds: float = 0.0
    skipped: list[str] = Field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return sum(result.attempts for result in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for result in self.results if result.error)

    def best(self) -> Optional[CandidateScore]:
        passing = [r for r in self.results if r.gate_passed and r.error is None]
        return max(passing, key=lambda r: r.overall_score or 0.0) if passing else None


class RankRequestBase(BaseModel):
    """Base request envelope with common validation."""

    model_config = ConfigDict(extra="forbid")

    job_description: str = Field(min_length=1, max_length=MAX_JOB_DESCRIPTION)
    criteria: tuple[Criterion, ...] = Field(min_length=1, max_length=MAX_CRITERIA)

    @field_validator("job_description")
    @classmethod
    def _job_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job_description must not be blank")
        return value

    @field_validator("criteria")
    @classmethod
    def _unique_ids(cls, value: tuple[Criterion, ...]) -> tuple[Criterion, ...]:
        seen: set[str] = set()
        duplicates: set[str] = {c.id for c in value if c.id in seen or seen.add(c.id)}
        if duplicates:
            raise ValueError(f"duplicate criterion ids: {sorted(duplicates)}")
        return value


class RankSingleRequest(RankRequestBase):
    """Request payload for ranking a single candidate."""

    candidate: Candidate


class RankBatchRequest(RankRequestBase):
    """Request payload for ranking a batch of candidates."""

    candidates: tuple[Candidate, ...] = Field(min_length=1, max_length=MAX_CANDIDATES)
    max_concurrency: Optional[int] = Field(default=None, ge=1, le=64)

    @model_validator(mode="after")
    def _unique_candidate_ids(self) -> "RankBatchRequest":
        seen: set[str] = set()
        duplicates: set[str] = {
            c.candidate_id for c in self.candidates if c.candidate_id in seen or seen.add(c.candidate_id)
        }
        if duplicates:
            raise ValueError(f"duplicate candidate ids: {sorted(duplicates)}")
        return self
