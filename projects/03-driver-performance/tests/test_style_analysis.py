
from types import SimpleNamespace
import math
from driver_performance.style_analysis import analyze_driving_style
from telemetry_analyzer.analysis.driver_metrics import (
    AccelerationMetrics,
    BrakingMetrics,
    CornerMetrics,
    DriverPerformanceMetrics,
    MetricComparison,
)
from telemetry_analyzer.analysis.event_performance import PerformanceEvent


def comparison(reference: float, compared: float) -> MetricComparison:
    return MetricComparison(
        reference=reference,
        compared=compared,
        delta=compared - reference,
    )


def make_braking_metrics(delta: float = -8.0) -> BrakingMetrics:
    return BrakingMetrics(
        start_distance=100.0,
        end_distance=150.0,
        distance=50.0,
        speed_at_start=comparison(200.0, 200.0),
        speed_at_end=comparison(100.0, 100.0),
        minimum_speed=comparison(90.0, 90.0 + delta),
        minimum_speed_distance=comparison(125.0, 125.0),
        max_brake=comparison(0.9, 0.9),
        average_throttle=comparison(0.2, 0.2),
        duration=comparison(2.0, 2.0),
    )


def make_event(change: float = 0.0) -> PerformanceEvent:
    return PerformanceEvent(
        event_type="braking",
        start_distance=100.0,
        end_distance=150.0,
        start_time_delta=0.0,
        end_time_delta=change,
        time_delta_change=change,
        absolute_time_delta_change=abs(change),
    )


def run_analysis(monkeypatch, metrics, events):
    monkeypatch.setattr(
        "driver_performance.style_analysis.calculate_driver_performance_metrics",
        lambda *args, **kwargs: metrics,
    )
    monkeypatch.setattr(
        "driver_performance.style_analysis.analyze_event_performance",
        lambda *args, **kwargs: events,
    )

    return analyze_driving_style(
        performance=SimpleNamespace(),
        reference_lap=SimpleNamespace(),
        compared_lap=SimpleNamespace(),
    )


def test_creates_observation_with_metric_evidence(monkeypatch):
    metrics = DriverPerformanceMetrics(
        braking=[make_braking_metrics(-8.0)],
        corners=[],
        acceleration=[],
    )

    observations = run_analysis(monkeypatch, metrics, [])

    assert len(observations) == 1

    observation = observations[0]
    assert observation.category == "braking"
    assert observation.id == "OBS-BRAKING-001"
    assert observation.evidence[0].metric_name == "minimum_speed_delta"
    assert observation.evidence[0].value == -8.0
    assert observation.evidence[0].unit == "km/h"
    assert observation.evidence[0].distance_start_m == 100.0
    assert observation.evidence[0].distance_end_m == 150.0


def test_includes_time_delta_change_as_evidence(monkeypatch):
    metrics = DriverPerformanceMetrics(
        braking=[make_braking_metrics()],
        corners=[],
        acceleration=[],
    )

    observations = run_analysis(
        monkeypatch,
        metrics,
        [make_event(change=0.35)],
    )

    observation = observations[0]
    time_evidence = [
        item
        for item in observation.evidence
        if item.metric_name == "time_delta_change"
    ]

    assert len(time_evidence) == 1
    assert time_evidence[0].value == 0.35
    assert time_evidence[0].unit == "s"


def test_does_not_create_observation_without_differences(monkeypatch):
    metrics = DriverPerformanceMetrics(
        braking=[make_braking_metrics(delta=0.0)],
        corners=[],
        acceleration=[],
    )

    observations = run_analysis(monkeypatch, metrics, [make_event(change=0.0)])

    assert observations == []


def test_negative_delta_is_preserved(monkeypatch):
    metrics = DriverPerformanceMetrics(
        braking=[make_braking_metrics(delta=-12.5)],
        corners=[],
        acceleration=[],
    )

    observations = run_analysis(monkeypatch, metrics, [])

    evidence = observations[0].evidence
    minimum_speed = next(
        item for item in evidence
        if item.metric_name == "minimum_speed_delta"
    )

    assert minimum_speed.value == -12.5


def make_acceleration_metrics() -> AccelerationMetrics:
    return AccelerationMetrics(
        start_distance=200.0,
        end_distance=260.0,
        distance=60.0,
        speed_at_start=comparison(100.0, 102.0),
        speed_at_end=comparison(180.0, 185.0),
        throttle_at_start=comparison(0.5, 0.6),
        max_throttle=comparison(0.95, 0.98),
        throttle_at_end=comparison(0.95, 0.98),
        duration=comparison(3.0, 2.9),
    )


def make_corner_metrics() -> CornerMetrics:
    return CornerMetrics(
        start_distance=300.0,
        end_distance=380.0,
        distance=80.0,
        entry_speed=comparison(200.0, 195.0),
        exit_speed=comparison(150.0, 155.0),
        minimum_speed=comparison(100.0, 95.0),
        minimum_speed_distance=comparison(340.0, 345.0),
        duration=comparison(4.0, 4.2),
    )


def make_event_for_window(
    event_type: str,
    start_distance: float,
    end_distance: float,
    change: float,
) -> PerformanceEvent:
    return PerformanceEvent(
        event_type=event_type,
        start_distance=start_distance,
        end_distance=end_distance,
        start_time_delta=0.0,
        end_time_delta=change,
        time_delta_change=change,
        absolute_time_delta_change=abs(change),
    )


def test_creates_acceleration_observation(monkeypatch):
    metrics = DriverPerformanceMetrics(
        braking=[],
        corners=[],
        acceleration=[make_acceleration_metrics()],
    )

    observations = run_analysis(monkeypatch, metrics, [])

    assert len(observations) == 1

    observation = observations[0]
    assert observation.id == "OBS-ACCELERATION-001"
    assert observation.category == "acceleration"
    assert observation.evidence
    assert any(
        item.metric_name == "throttle_at_start_delta"
        and math.isclose(item.value, 0.1, abs_tol=1e-9)
        for item in observation.evidence
    )


def test_creates_corner_observation(monkeypatch):
    metrics = DriverPerformanceMetrics(
        braking=[],
        corners=[make_corner_metrics()],
        acceleration=[],
    )

    observations = run_analysis(monkeypatch, metrics, [])

    assert len(observations) == 1

    observation = observations[0]
    assert observation.id == "OBS-CORNER-001"
    assert observation.category == "corner"
    assert observation.evidence
    assert any(
        item.metric_name == "entry_speed_delta"
        and item.value == -5.0
        for item in observation.evidence
    )


def test_creates_observation_from_time_delta_alone(monkeypatch):
    metrics = DriverPerformanceMetrics(
        braking=[make_braking_metrics(delta=0.0)],
        corners=[],
        acceleration=[],
    )
    event = make_event_for_window(
        event_type="braking",
        start_distance=100.0,
        end_distance=150.0,
        change=0.25,
    )

    observations = run_analysis(monkeypatch, metrics, [event])

    assert len(observations) == 1
    assert observations[0].category == "braking"
    assert len(observations[0].evidence) == 1
    assert observations[0].evidence[0].metric_name == "time_delta_change"
    assert observations[0].evidence[0].value == 0.25


def test_multiple_zones_receive_unique_ids(monkeypatch):
    first_zone = make_braking_metrics(delta=-8.0)
    second_zone = make_braking_metrics(delta=-12.0)

    # Give the second zone different distance boundaries.
    second_zone.start_distance = 200.0
    second_zone.end_distance = 250.0

    metrics = DriverPerformanceMetrics(
        braking=[first_zone, second_zone],
        corners=[],
        acceleration=[],
    )

    observations = run_analysis(monkeypatch, metrics, [])

    assert len(observations) == 2
    assert [item.id for item in observations] == [
        "OBS-BRAKING-001",
        "OBS-BRAKING-002",
    ]
    assert observations[0].evidence[0].distance_start_m == 100.0
    assert observations[1].evidence[0].distance_start_m == 200.0