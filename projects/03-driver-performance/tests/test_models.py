
import pytest

from driver_performance import (
    ConfidenceLevel,
    Diagnosis,
    Evidence,
    Observation,
    Recommendation,
    RecommendationPriority,
)


def make_evidence() -> Evidence:
    return Evidence(
        metric_name="speed_delta",
        value=-8.5,
        unit="km/h",
        distance_start_m=100.0,
        distance_end_m=150.0,
    )


def make_observation() -> Observation:
    return Observation(
        id="OBS-001",
        title="Speed difference",
        description="The compared lap has a lower speed in this zone.",
        category="speed",
        evidence=(make_evidence(),),
    )


def test_evidence_accepts_valid_data():
    evidence = make_evidence()

    assert evidence.metric_name == "speed_delta"
    assert evidence.value == -8.5
    assert evidence.distance_start_m == 100.0


def test_evidence_rejects_invalid_distance_range():
    with pytest.raises(ValueError):
        Evidence(
            metric_name="speed_delta",
            value=2.0,
            unit="km/h",
            distance_start_m=200.0,
            distance_end_m=100.0,
        )


def test_evidence_requires_both_distance_boundaries():
    with pytest.raises(ValueError):
        Evidence(
            metric_name="speed_delta",
            value=2.0,
            unit="km/h",
            distance_start_m=100.0,
        )


def test_observation_requires_evidence():
    with pytest.raises(ValueError):
        Observation(
            id="OBS-002",
            title="Missing evidence",
            description="This observation has no supporting data.",
            category="speed",
            evidence=(),
        )


def test_observation_accepts_evidence():
    observation = make_observation()

    assert len(observation.evidence) == 1
    assert observation.evidence[0].metric_name == "speed_delta"


def test_diagnosis_requires_supporting_observations():
    with pytest.raises(ValueError):
        Diagnosis(
            id="DIA-001",
            title="Possible issue",
            explanation="A possible interpretation.",
            confidence=ConfidenceLevel.LOW,
            supporting_observation_ids=(),
        )


def test_diagnosis_accepts_supported_interpretation():
    diagnosis = Diagnosis(
        id="DIA-001",
        title="Speed loss in a zone",
        explanation="The available data indicates a speed difference.",
        confidence=ConfidenceLevel.MEDIUM,
        supporting_observation_ids=("OBS-001",),
        limitations=("The data does not establish the cause.",),
    )

    assert diagnosis.confidence == ConfidenceLevel.MEDIUM
    assert diagnosis.supporting_observation_ids == ("OBS-001",)


def test_recommendation_requires_a_reference():
    with pytest.raises(ValueError):
        Recommendation(
            id="REC-001",
            title="Review the zone",
            action="Review the telemetry.",
            rationale="Further analysis may clarify the difference.",
            priority=RecommendationPriority.LOW,
        )


def test_recommendation_can_reference_an_observation():
    recommendation = Recommendation(
        id="REC-001",
        title="Review the speed difference",
        action="Compare speed and braking traces in this zone.",
        rationale="The observed speed difference warrants further review.",
        priority=RecommendationPriority.MEDIUM,
        related_observation_ids=("OBS-001",),
    )

    assert recommendation.priority == RecommendationPriority.MEDIUM
    assert recommendation.related_observation_ids == ("OBS-001",)


def test_models_are_immutable():
    evidence = make_evidence()

    with pytest.raises(AttributeError):
        evidence.value = 10.0
