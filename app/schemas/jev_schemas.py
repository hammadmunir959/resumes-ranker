"""Jev Decisions wire format schemas."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.exceptions import JevProtocolError

#: One question definition, keyed by its id.
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
    """Builders for Jev Decisions question types."""

    @staticmethod
    def noul(
        instructions: str,
        true_criteria: Optional[str] = None,
        false_criteria: Optional[str] = None,
    ) -> Question:
        """Build a yes/no gate question answered as a probability."""
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
        """Build an ordered rubric scale question."""
        if not instructions.strip():
            raise ValueError("instructions must not be blank")
        levels = [str(level) for level in rubric]
        if len(levels) < 2:
            raise ValueError("a score question needs at least two rubric levels")
        return {"type": "score", "instructions": instructions, "criteria": levels}

    @staticmethod
    def choice(instructions: str, options: Mapping[str, str]) -> Question:
        """Build a multiple-choice question."""
        if not instructions.strip():
            raise ValueError("instructions must not be blank")
        if len(options) < 2:
            raise ValueError("a choice question needs at least two options")
        return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


class ScoreAnswer(BaseModel):
    """Ordered rubric position with probability distribution."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    score: float
    confidence: Optional[float] = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    legend: dict[str, str] = Field(default_factory=dict)

    def expected_score(self) -> float:
        """Calculate mean position from probability distribution."""
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
    """Selected option from offered choices."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    choice: str
    confidence: Optional[float] = None


class Usage(BaseModel):
    """Token usage and request cost."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


class JevResponse(BaseModel):
    """Decisions API response."""

    model_config = ConfigDict(extra="allow")

    answers: dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    cost_usd: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def usage(self) -> Usage:
        """Extract Usage object from token counts and cost."""
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )

    def answer(self, question_id: str) -> Optional[dict[str, Any]]:
        """Return raw answer mapping for a question."""
        value = self.answers.get(question_id)
        return value if isinstance(value, dict) else None

    def confidence(self, question_id: str) -> Optional[float]:
        """Return confidence score for a question."""
        answer = self.answer(question_id)
        if not answer:
            return None
        value = answer.get("confidence")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def noul(self, question_id: str) -> float:
        """Return probability for a noul question."""
        answer = self.answer(question_id)
        value = answer.get("noul") if answer else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JevProtocolError(
                f"question {question_id!r}: expected a noul answer, "
                f"got {answer.get('type') if answer else None!r}"
            )
        return float(value)

    def score(self, question_id: str) -> ScoreAnswer:
        """Return ScoreAnswer for a score question."""
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
        """Return ChoiceAnswer for a choice question."""
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
