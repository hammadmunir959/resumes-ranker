"""What this program judges: the inputs, the per-candidate results, and the
envelopes the API wraps them in.

Separate from :mod:`app.schemas.jev_schemas` because these shapes are ours. They
describe the ranking job and its answer, and they are what the CLI, the API, and
a library caller all pass around.

Each cap below is enforced by the model that uses it.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.jev_schemas import Usage

# --- Caps --------------------------------------------------------------------
# Declared here because these models are what enforce them, and a bound baked
# into a ``Field`` cannot be varied per instance. The cost of a request scales
# with the number of criteria and candidates, so refusing an oversized payload
# here is what stops a request that would be slow and expensive rather than
# merely invalid.

MAX_CRITERIA = 50
MAX_CANDIDATES = 100
MAX_CRITERION_NAME = 120
MAX_JOB_DESCRIPTION = 20_000
MAX_RESUME_TEXT = 40_000
MAX_APPLICATION_FORM = 20_000
MAX_CANDIDATE_ID = 120

#: Ordered outcome levels, lowest to highest. Indices are what a ``score``
#: question reports, so the order is significant.
DEFAULT_RUBRIC: tuple[str, ...] = ("Not met", "Partially met", "Fully met")

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


class _Frozen(BaseModel):
    """Immutable base, so a criterion cannot change after scoring starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

class Criterion(_Frozen):
    """One thing being judged, such as "5+ years of Python experience"."""

    id: str = Field(min_length=1, max_length=MAX_CRITERION_NAME)
    name: str = Field(min_length=1, max_length=MAX_CRITERION_NAME)
    description: str = Field(default="", max_length=MAX_JOB_DESCRIPTION)
    required: bool = False
    weight: float = Field(default=1.0, ge=0.0, le=1000.0)
    #: Ordered outcomes, lowest to highest. Two or more entries.
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
        # A required criterion is a gate, and gates short-circuit the pipeline,
        # so a weight on one would be a number nothing ever reads.
        if not self.required and self.weight <= 0:
            raise ValueError("an optional criterion needs weight > 0")
        return self

    @property
    def max_index(self) -> float:
        """Highest rubric index, the value a perfect score would report."""
        return float(len(self.rubric) - 1)

    @property
    def is_gate(self) -> bool:
        """Gates are pass/fail; everything else is a weighted score."""
        return self.required

    def rubric_at(self, index: float) -> str:
        """Human label for a fractional rubric index, clamped to the range."""
        position = round(index)
        position = max(0, min(position, len(self.rubric) - 1))
        return self.rubric[position]

    def weighted_score(self, index: float) -> float:
        """Turn a rubric index into a 0.0-1.0 fraction of the maximum."""
        if self.max_index <= 0:
            return 0.0
        return max(0.0, min(index / self.max_index, 1.0))


class Candidate(_Frozen):
    """One resume, plus any application-form text for the same person."""

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
        """Resume and form together, since both can evidence a criterion."""
        if self.application_form_text.strip():
            return f"{self.resume_text}\n\n--- Application form ---\n{self.application_form_text}"
        return self.resume_text

    def estimated_tokens(self, chars_per_token: int = 4) -> int:
        """Rough prompt size, for cost projection only."""
        return max(1, len(self.text_for_matching) // chars_per_token)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

class CriterionOutcome(BaseModel):
    """The result for one criterion, as shown in the per-candidate breakdown."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    name: str
    required: bool
    #: For a gate, whether it passed. None for a weighted score, which is not
    #: pass/fail.
    passed: Optional[bool] = None
    #: The rubric level the model landed on.
    label: str = ""
    #: Rubric index, or the yes-probability for a gate.
    raw_score: float = 0.0
    #: 0-100, always populated so reports need no per-row branching.
    score_0_100: float = 0.0
    #: The model's confidence in this answer.
    confidence: Optional[float] = None
    needs_review: bool = False
    #: Set when this criterion could not be evaluated.
    error: Optional[str] = None

    @property
    def weight_used(self) -> float:
        return 0.0 if self.required else 1.0


class CandidateScore(BaseModel):
    """Everything known about one candidate after ranking."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    gate_passed: bool = True
    #: The model that actually answered. Differs from the configured primary when
    #: the Jev service fell back, so a report cannot claim the wrong model.
    model: str = ""
    #: 0-100 weighted score, or None when no weighted criterion contributed.
    #: None is different from 0.0: 0.0 means "scored and earned nothing", while
    #: None means there was nothing to score, as for a job of pure gates.
    overall_score: Optional[float] = None
    #: Ids of the required criteria that failed.
    failed_gates: list[str] = Field(default_factory=list)
    #: True when a gate failed or the result is too uncertain to trust.
    needs_review: bool = False
    cost_usd: Optional[float] = None
    #: How many HTTP attempts the upstream call took.
    attempts: int = 0
    per_criterion: list[CriterionOutcome] = Field(default_factory=list)
    #: Set when the candidate could not be scored at all.
    error: Optional[str] = None
    #: Token counts, when the provider reported them.
    usage: Optional[Usage] = None

    @property
    def scored(self) -> bool:
        """True when the model answered and a result was produced.

        Deliberately does not require ``overall_score``: a job made only of
        required criteria has nothing to weigh, so a passing candidate is fully
        scored even though there is no number to sort it by.
        """
        return self.error is None

    @classmethod
    def failed(
        cls,
        candidate_id: str,
        error: str,
        *,
        failed_gates: Optional[list[str]] = None,
    ) -> "CandidateScore":
        """A candidate that could not be scored."""
        return cls(
            candidate_id=candidate_id,
            gate_passed=not failed_gates,
            failed_gates=failed_gates or [],
            needs_review=True,
            error=error,
        )


class RankingResult(BaseModel):
    """A ranked batch, plus the cost of producing it.

    Both ``/rank/single`` and ``/rank/batch`` return this shape; single is
    simply a batch of one, which keeps clients from handling two formats.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    results: list[CandidateScore] = Field(default_factory=list)
    total_cost_usd: Optional[float] = None
    elapsed_seconds: float = 0.0
    #: Candidates excluded before scoring, e.g. duplicates.
    skipped: list[str] = Field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return sum(result.attempts for result in self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for result in self.results if result.error)

    def best(self) -> Optional[CandidateScore]:
        """The strongest candidate: passing, then highest score.

        Falls back to any passing candidate when nothing was weighed, which is
        the normal case for a job defined purely by hard requirements.
        """
        passing = [
            r for r in self.results if r.gate_passed and r.error is None
        ]
        if not passing:
            return None
        return max(passing, key=lambda r: r.overall_score or 0.0)


# --------------------------------------------------------------------------- #
# API envelopes
# --------------------------------------------------------------------------- #

class RankRequestBase(BaseModel):
    """Fields shared by the single and batch endpoints.

    Both endpoints take the same job and the same criteria, so they inherit the
    validation rather than each carrying a copy of it.
    """

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
        duplicates: set[str] = set()
        for criterion in value:
            if criterion.id in seen:
                duplicates.add(criterion.id)
            seen.add(criterion.id)
        if duplicates:
            raise ValueError(f"duplicate criterion ids: {sorted(duplicates)}")
        return value


class RankSingleRequest(RankRequestBase):
    """Rank one candidate."""

    candidate: Candidate


class RankBatchRequest(RankRequestBase):
    """Rank several candidates, concurrently, in one call."""

    candidates: tuple[Candidate, ...] = Field(min_length=1, max_length=MAX_CANDIDATES)
    #: Overrides the configured concurrency for this call only.
    max_concurrency: Optional[int] = Field(default=None, ge=1, le=64)

    @model_validator(mode="after")
    def _unique_candidate_ids(self) -> "RankBatchRequest":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for candidate in self.candidates:
            if candidate.candidate_id in seen:
                duplicates.add(candidate.candidate_id)
            seen.add(candidate.candidate_id)
        if duplicates:
            raise ValueError(f"duplicate candidate ids: {sorted(duplicates)}")
        return self
