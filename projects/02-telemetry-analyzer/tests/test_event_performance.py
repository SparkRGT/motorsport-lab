import math
from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.event_performance import (
    PerformanceEvent,
    analyze_event_performance,
)
from telemetry_analyzer.analysis.lap_performance import (
    LapPerformanceAnalysis,
    analyze_lap_performance,
)
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "telemetry_laps_sample.csv"
)

REAL_DATASET = (
    Path(__file__).resolve().parents[2]
    / "01-telemetry-collector"
    / "data"
    / "telemetry"
    / "telemetry_session.csv"
)


def _lap(samples: list[tuple[float, float, float, float, float]], lap_number: int) -> LapData:
    frame = pd.DataFrame(
        samples,
        columns=["lap_distance", "session_time", "speed", "throttle", "brake"],
    )
    return LapData(
        lap_number=lap_number,
        data=frame,
        start_time=float(frame["session_time"].iloc[0]),
        end_time=float(frame["session_time"].iloc[-1]),
        duration=float(frame["session_time"].iloc[-1] - frame["session_time"].iloc[0]),
        sample_count=len(frame),
    )


def _with_speed(samples: list[tuple[float, float, float, float]], speed: float, lap_number: int) -> LapData:
    return _lap(
        [(distance, time, speed, throttle, brake) for distance, time, throttle, brake in samples],
        lap_number,
    )


def _braking_pair(reference_speed: float = 100.0, compared_speed: float = 50.0):
    samples = [
        (0.0, 0.0, 0.2, 0.0),
        (10.0, 1.0, 0.2, 0.8),
        (20.0, 2.0, 0.2, 0.8),
        (30.0, 3.0, 0.2, 0.0),
    ]
    return (
        _with_speed(samples, reference_speed, 1),
        _with_speed(samples, compared_speed, 2),
    )


def _acceleration_pair():
    samples = [
        (0.0, 0.0, 0.2, 0.0),
        (10.0, 1.0, 0.95, 0.0),
        (20.0, 2.0, 0.95, 0.0),
        (30.0, 3.0, 0.2, 0.0),
    ]
    return _with_speed(samples, 100.0, 1), _with_speed(samples, 50.0, 2)


def _corner_pair():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 80.0, 0.2, 0.8),
            (20.0, 2.0, 70.0, 0.2, 0.0),
            (30.0, 3.0, 90.0, 0.95, 0.0),
            (40.0, 4.0, 110.0, 0.95, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 50.0, 0.2, 0.0),
            (10.0, 1.0, 50.0, 0.2, 0.0),
            (20.0, 2.0, 50.0, 0.2, 0.0),
            (30.0, 3.0, 50.0, 0.2, 0.0),
            (40.0, 4.0, 50.0, 0.2, 0.0),
        ],
        2,
    )
    return reference, compared


def _events(reference: LapData, compared: LapData, **kwargs) -> list[PerformanceEvent]:
    performance = analyze_lap_performance(reference, compared)
    return analyze_event_performance(performance, reference, compared, **kwargs)


def _one(events: list[PerformanceEvent], event_type: str) -> PerformanceEvent:
    matches = [event for event in events if event.event_type == event_type]
    assert len(matches) == 1
    return matches[0]


def test_detects_braking_event():
    reference, compared = _braking_pair()

    event = _one(_events(reference, compared), "braking")

    assert event.start_distance == pytest.approx(10.0)
    assert event.end_distance == pytest.approx(20.0)


def test_detects_acceleration_event():
    reference, compared = _acceleration_pair()

    event = _one(_events(reference, compared), "acceleration")

    assert event.start_distance == pytest.approx(10.0)
    assert event.end_distance == pytest.approx(20.0)


def test_detects_corner_event():
    reference, compared = _corner_pair()

    event = _one(_events(reference, compared), "corner")

    assert event.start_distance == pytest.approx(10.0)
    assert event.end_distance == pytest.approx(40.0)


def test_time_delta_change_is_end_minus_start():
    reference, compared = _braking_pair()

    event = _one(_events(reference, compared), "braking")

    assert event.start_time_delta == pytest.approx(0.36)
    assert event.end_time_delta == pytest.approx(0.72)
    assert event.time_delta_change == pytest.approx(
        event.end_time_delta - event.start_time_delta
    )
    assert event.time_delta_change == pytest.approx(0.36)
    assert event.absolute_time_delta_change == pytest.approx(abs(event.time_delta_change))


def test_negative_change_means_compared_accumulated_less_time():
    reference, compared = _braking_pair(reference_speed=50.0, compared_speed=100.0)

    event = _one(_events(reference, compared), "braking")

    assert event.time_delta_change == pytest.approx(-0.36)


def test_interpolates_event_distances_between_aligned_rows():
    samples = [
        (0.0, 0.0, 0.2, 0.0),
        (5.0, 1.0, 0.2, 0.8),
        (15.0, 2.0, 0.2, 0.8),
        (20.0, 3.0, 0.2, 0.0),
    ]
    reference = _with_speed(samples, 100.0, 1)
    compared = _with_speed(samples, 50.0, 2)
    performance = analyze_lap_performance(reference, compared, distance_step=10.0)

    event = _one(
        analyze_event_performance(performance, reference, compared),
        "braking",
    )

    assert 5.0 not in performance.aligned_data["distance"].tolist()
    assert 15.0 not in performance.aligned_data["distance"].tolist()
    assert event.start_time_delta == pytest.approx(0.18)
    assert event.end_time_delta == pytest.approx(0.54)
    assert event.time_delta_change == pytest.approx(0.36)


def test_events_are_ordered_by_distance_then_type():
    reference, compared = _corner_pair()

    events = _events(reference, compared)

    assert [(event.start_distance, event.event_type) for event in events] == sorted(
        (event.start_distance, event.event_type) for event in events
    )
    assert [event.event_type for event in events if event.start_distance == pytest.approx(10.0)] == [
        "braking",
        "corner",
    ]


def test_overlapping_events_are_kept():
    reference, compared = _corner_pair()

    event_types = {event.event_type for event in _events(reference, compared)}

    assert event_types == {"braking", "acceleration", "corner"}


def test_does_not_modify_laps_or_aligned_data():
    reference, compared = _corner_pair()
    performance = analyze_lap_performance(reference, compared)
    reference_before = reference.data.copy(deep=True)
    compared_before = compared.data.copy(deep=True)
    aligned_before = performance.aligned_data.copy(deep=True)
    aligned_identity = id(performance.aligned_data)

    analyze_event_performance(performance, reference, compared)

    pd.testing.assert_frame_equal(reference.data, reference_before)
    pd.testing.assert_frame_equal(compared.data, compared_before)
    pd.testing.assert_frame_equal(performance.aligned_data, aligned_before)
    assert id(performance.aligned_data) == aligned_identity


def test_lap_without_braking_zones_has_no_braking_event():
    reference, compared = _acceleration_pair()

    events = _events(reference, compared)

    assert events
    assert all(event.event_type != "braking" for event in events)


def test_lap_without_acceleration_zones_has_no_acceleration_event():
    reference, compared = _braking_pair()

    events = _events(reference, compared)

    assert all(event.event_type != "acceleration" for event in events)


def test_lap_without_corner_segments_has_no_corner_event():
    reference, compared = _braking_pair()

    events = _events(reference, compared)

    assert any(event.event_type == "braking" for event in events)
    assert all(event.event_type != "corner" for event in events)


def test_rejects_empty_aligned_data():
    reference, compared = _braking_pair()
    performance = LapPerformanceAnalysis(
        reference_lap=1,
        compared_lap=2,
        total_time_delta=0.0,
        aligned_data=pd.DataFrame(),
        performance_segments=[],
    )

    with pytest.raises(ValueError, match="empty"):
        analyze_event_performance(performance, reference, compared)


def test_analyze_event_performance_on_synthetic_sample_laps():
    dataset = load_csv(SAMPLE_DATASET)
    validate_dataset(dataset)
    laps = segment_laps(normalize_dataset(dataset))
    performance = analyze_lap_performance(laps[2], laps[1])

    events = analyze_event_performance(performance, laps[2], laps[1])

    assert events
    assert all(isinstance(event, PerformanceEvent) for event in events)
    assert all(
        event.time_delta_change == pytest.approx(event.end_time_delta - event.start_time_delta)
        for event in events
    )
    assert all(math.isfinite(event.time_delta_change) for event in events)
    assert [event.start_distance for event in events] == sorted(event.start_distance for event in events)


def test_analyze_event_performance_on_overlapping_real_laps():
    assert REAL_DATASET.is_file()

    dataset = load_csv(REAL_DATASET)
    untouched = dataset.copy(deep=True)
    validate_dataset(dataset)
    laps = list(segment_laps(normalize_dataset(dataset)).values())
    reference, compared = _first_overlapping_pair(laps)
    performance = analyze_lap_performance(reference, compared)
    aligned_before = performance.aligned_data.copy(deep=True)

    events = analyze_event_performance(performance, reference, compared)

    assert isinstance(events, list)
    assert all(event.event_type in {"braking", "acceleration", "corner"} for event in events)
    assert all(math.isfinite(event.start_time_delta) for event in events)
    assert all(math.isfinite(event.end_time_delta) for event in events)
    assert all(
        event.time_delta_change == pytest.approx(event.end_time_delta - event.start_time_delta)
        for event in events
    )
    assert [(event.start_distance, event.event_type) for event in events] == sorted(
        (event.start_distance, event.event_type) for event in events
    )
    pd.testing.assert_frame_equal(performance.aligned_data, aligned_before)
    pd.testing.assert_frame_equal(dataset, untouched)


def _first_overlapping_pair(laps: list[LapData]) -> tuple[LapData, LapData]:
    for index, reference in enumerate(laps):
        reference_start = float(reference.data["lap_distance"].iloc[0])
        reference_end = float(reference.data["lap_distance"].iloc[-1])
        for compared in laps[index + 1 :]:
            compared_start = float(compared.data["lap_distance"].iloc[0])
            compared_end = float(compared.data["lap_distance"].iloc[-1])
            if max(reference_start, compared_start) < min(reference_end, compared_end):
                return reference, compared

    raise AssertionError("The dataset has no overlapping lap pair")
