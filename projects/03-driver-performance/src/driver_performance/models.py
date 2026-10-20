
"""Domain models for evidence-based driver performance analysis."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class ConfidenceLevel(StrEnum):
    """Qualitative confidence in a diagnosis."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendationPriority(StrEnum):
    """Priority assigned to a recommendation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _require_text(value: str, field_name: str) -> None:
    """Reject empty or whitespace-only text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_number(
    value: int | float,
    field_name: str,
    *,
    non_negative: bool = False,
) -> None:
    """Validate a finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")

    if non_negative and value < 0:
        raise ValueError(f"{field_name} must be non-negative.")


def _require_text_items(
    values: tuple[str, ...],
    field_name: str,
    *,
    require_items: bool = True,
) -> None:
    """Validate a tuple of non-empty strings."""
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple.")

    if require_items and not values:
        raise ValueError(f"{field_name} must contain at least one item.")

    for value in values:
        _require_text(value, field_name)


@dataclass(frozen=True, slots=True)
class Evidence:
    """A measured value and its optional location on the circuit."""

    metric_name: str
    value: float
    unit: str
    distance_start_m: float | None = None
    distance_end_m: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.metric_name, "metric_name")
        _require_text(self.unit, "unit")
        _require_number(self.value, "value")

        start = self.distance_start_m
        end = self.distance_end_m

        if (start is None) != (end is None):
            raise ValueError(
                "distance_start_m and distance_end_m must be "
                "provided together."
            )

        if start is not None and end is not None:
            _require_number(
                start, "distance_start_m", non_negative=True
            )
            _require_number(
                end, "distance_end_m", non_negative=True
            )

            if end < start:
                raise ValueError(
                    "distance_end_m must be greater than or equal "
                    "to distance_start_m."
                )


@dataclass(frozen=True, slots=True)
class Observation:
    """An objective statement supported by telemetry evidence."""

    id: str
    title: str
    description: str
    category: str
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        _require_text(self.id, "id")
        _require_text(self.title, "title")
        _require_text(self.description, "description")
        _require_text(self.category, "category")

        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple.")

        if not self.evidence:
            raise ValueError(
                "An observation must contain at least one evidence item."
            )

        if not all(isinstance(item, Evidence) for item in self.evidence):
            raise TypeError("Every evidence item must be an Evidence instance.")


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """A cautious interpretation linked to supporting observations."""

    id: str
    title: str
    explanation: str
    confidence: ConfidenceLevel
    supporting_observation_ids: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.id, "id")
        _require_text(self.title, "title")
        _require_text(self.explanation, "explanation")
        _require_text_items(
            self.supporting_observation_ids,
            "supporting_observation_ids",
        )
        _require_text_items(
            self.limitations,
            "limitations",
            require_items=False,
        )

        if not isinstance(self.confidence, ConfidenceLevel):
            raise TypeError("confidence must be a ConfidenceLevel value.")


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A proposed action linked to observations or diagnoses."""

    id: str
    title: str
    action: str
    rationale: str
    priority: RecommendationPriority
    related_observation_ids: tuple[str, ...] = ()
    related_diagnosis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.id, "id")
        _require_text(self.title, "title")
        _require_text(self.action, "action")
        _require_text(self.rationale, "rationale")

        _require_text_items(
            self.related_observation_ids,
            "related_observation_ids",
            require_items=False,
        )
        _require_text_items(
            self.related_diagnosis_ids,
            "related_diagnosis_ids",
            require_items=False,
        )

        if not (
            self.related_observation_ids
            or self.related_diagnosis_ids
        ):
            raise ValueError(
                "A recommendation must reference at least one "
                "observation or diagnosis."
            )

        if not isinstance(self.priority, RecommendationPriority):
            raise TypeError(
                "priority must be a RecommendationPriority value."
            )
