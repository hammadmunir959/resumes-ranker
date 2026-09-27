"""Every model in the project, in one place.

The old layout had domain dataclasses in ``service.py`` and a second, parallel
set of Pydantic request/response models in ``api.py``, joined by
``to_domain()`` / ``from_domain()`` translation functions that had to be kept in
step by hand. Here there is one definition per shape, used by the service, the
API, the CLI, and the tests alike.

Validation lives on the models, so an invalid criterion or a 500-candidate batch
is rejected before any code starts ranking, and the API gets that for free from
FastAPI rather than from a hand-written handler.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.exceptions import JevProtocolError

# --- Limits -----------------------------------------------------------------
# Enforced by the models below, so they apply to the API, the CLI, and direct
# library use alike.
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

#: Default output scale for a weighted score.
SCORE_SCALE = 100.0


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
# Jev wire format
# --------------------------------------------------------------------------- #

class ScoreAnswer(BaseModel):
    """A position on an ordered rubric, with the model's full distribution.

    ``score`` is the model's chosen position, which can fall between two levels.
    ``probabilities`` and ``legend`` are the distribution behind that choice.
    """

    model_config = ConfigDict(extra="allow")

    question_id: str
    score: float
    confidence: Optional[float] = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    legend: dict[str, str] = Field(default_factory=dict)

    def expected_score(self) -> float:
        """Mean position implied by the distribution.

        Usually the same as ``score``; worth reading when the model spread its
        probability thinly and the point estimate overstates the confidence.
        """
        if not self.probabilities:
            return self.score
        total = 0.0
        for key, weight in self.probabilities.items():
            try:
                total += float(key) * float(weight)
            except (TypeError, ValueError):
                continue
        return total


class ChoiceAnswer(BaseModel):
    """One selected option out of those offered."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    choice: str
    confidence: Optional[float] = None


class Usage(BaseModel):
    """Token counts and cost as reported by the provider."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class JevResponse(BaseModel):
    """One Decisions response, normalized and typed.

    ``answers`` is keyed by question id, matching the ids the caller sent.
    """

    model_config = ConfigDict(extra="allow")

    answers: dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    cost_usd: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    #: HTTP attempts this answer took, including the successful one.
    attempts: int = 1
    #: The untouched body, kept for debugging. Not serialized in API responses.
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def usage(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )

    def answer(self, question_id: str) -> Optional[dict[str, Any]]:
        """The raw answer for one question, or None if the model omitted it."""
        value = self.answers.get(question_id)
        return value if isinstance(value, dict) else None

    def confidence(self, question_id: str) -> Optional[float]:
        answer = self.answer(question_id)
        if not answer:
            return None
        value = answer.get("confidence")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def noul(self, question_id: str) -> float:
        """Probability that a ``noul`` question is yes."""
        answer = self.answer(question_id)
        value = answer.get("noul") if answer else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JevProtocolError(
                f"question {question_id!r}: expected a noul answer, "
                f"got {answer.get('type') if answer else None!r}"
            )
        return float(value)

    def score(self, question_id: str) -> ScoreAnswer:
        answer = self.answer(question_id)
        value = answer.get("score") if answer else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JevProtocolError(
                f"question {question_id!r}: expected a score answer, "
                f"got {answer.get('type') if answer else None!r}"
            )
        probabilities = answer.get("probabilities")
        legend = answer.get("legend")
        return ScoreAnswer(
            question_id=question_id,
            score=float(value),
            confidence=self.confidence(question_id),
            probabilities=(
                {str(k): float(v) for k, v in probabilities.items()}
                if isinstance(probabilities, dict)
                else {}
            ),
            legend=(
                {str(k): str(v) for k, v in legend.items()}
                if isinstance(legend, dict)
                else {}
            ),
        )

    def choice(self, question_id: str) -> ChoiceAnswer:
        answer = self.answer(question_id)
        value = answer.get("choice") if answer else None
        if not isinstance(value, str):
            raise JevProtocolError(
                f"question {question_id!r}: expected a choice answer, "
                f"got {answer.get('type') if answer else None!r}"
            )
        return ChoiceAnswer(
            question_id=question_id, choice=value, confidence=self.confidence(question_id)
        )


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


class HealthResponse(BaseModel):
    """Liveness plus the configuration a client would need to debug a failure.

    Reports the base URL and model so an operator can see what the process
    actually loaded. The key is never included, only whether one was found.
    """

    status: str
    model: str
    base_url: str
    key_configured: bool
    mock_key: bool = False
    max_concurrency: int
    max_requests_per_second: float
    max_batch_size: int


__all__ = [
    "DEFAULT_RUBRIC",
    "SCORE_SCALE",
    "MAX_CRITERIA",
    "MAX_CANDIDATES",
    "MAX_CANDIDATE_ID",
    "MAX_JOB_DESCRIPTION",
    "MAX_RESUME_TEXT",
    "MAX_APPLICATION_FORM",
    "MAX_CRITERION_NAME",
    "Criterion",
    "Candidate",
    "ScoreAnswer",
    "ChoiceAnswer",
    "Usage",
    "JevResponse",
    "CriterionOutcome",
    "CandidateScore",
    "RankingResult",
    "RankRequestBase",
    "RankSingleRequest",
    "RankBatchRequest",
    "HealthResponse",
]
