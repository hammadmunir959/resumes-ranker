"""The Jev Decisions wire format: what a request contains and what comes back.

Kept apart from the ranking shapes in :mod:`app.schemas.ranking_schemas` because
these describe somebody else's API rather than anything this program decides.
They change when Jev changes, and the ranking layer should not have to care.

:data:`Question` and :class:`Questions` build the request side;
:class:`JevResponse` and the answer models parse the response side.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.exceptions import JevProtocolError

#: One question, keyed by its id. A plain dict because the id is chosen by the
#: caller and Jev's own schema is an open object.
Question = dict[str, Any]

__all__ = [
    "Question",
    "Questions",
    "ScoreAnswer",
    "ChoiceAnswer",
    "Usage",
    "JevResponse",
]


class Questions:
    """Builders for the three question types Jev supports.

    Static methods, so a caller writes ``Questions.noul(...)`` with no instance
    and no import of a builder type.
    """

    @staticmethod
    def noul(
        instructions: str,
        true_criteria: Optional[str] = None,
        false_criteria: Optional[str] = None,
    ) -> Question:
        """A yes/no gate answered as a probability.

        ``true_criteria`` and ``false_criteria`` describe what makes the answer
        yes and no, which is what stops the model deciding on tone.
        """
        if not instructions.strip():
            raise ValueError("instructions must not be blank")
        question: Question = {"type": "noul", "instructions": instructions}
        if true_criteria:
            question["criteria"] = {"true": true_criteria, "false": false_criteria or ""}
        elif false_criteria:
            raise ValueError("false_criteria needs true_criteria to compare against")
        return question

    @staticmethod
    def score(instructions: str, rubric: Sequence[str]) -> Question:
        """A position on an ordered scale, answered as a rubric index."""
        if not instructions.strip():
            raise ValueError("instructions must not be blank")
        levels = [str(level) for level in rubric]
        if len(levels) < 2:
            raise ValueError("a score question needs at least two rubric levels")
        return {"type": "score", "instructions": instructions, "criteria": levels}

    @staticmethod
    def choice(instructions: str, options: Mapping[str, str]) -> Question:
        """One option out of a fixed set, answered by name."""
        if not instructions.strip():
            raise ValueError("instructions must not be blank")
        if len(options) < 2:
            raise ValueError("a choice question needs at least two options")
        return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


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
