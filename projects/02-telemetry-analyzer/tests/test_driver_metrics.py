import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telemetry_analyzer.analysis.corners import detect_corner_segments
from telemetry_analyzer.analysis.driver_metrics import (
    DriverPerformanceMetrics,
    MetricComparison,
    calculate_driver_performance_metrics,
)
from telemetry_analyzer.analysis.lap_performance import (
    LapPerformanceAnalysis,
    analyze_lap_performance,
)
from telemetry_analyzer.analysis.zones import detect_acceleration_zones, detect_braking_zones
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
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
        start_time=float(frame["session_time"].iloc[0]) if len(frame) else 0.0,
        end_time=float(frame["session_time"].iloc[-1]) if len(frame) else 0.0,
        duration=float(frame["session_time"].iloc[-1] - frame["session_time"].iloc[0]) if len(frame) else 0.0,
        sample_count=len(frame),
    )


def _metrics(reference: LapData, compared: LapData, **kwargs) -> DriverPerformanceMetrics:
    performance = analyze_lap_performance(reference, compared, **kwargs)
    return calculate_driver_performance_metrics(performance, reference, compared)


def _assert_comparison_deltas(metrics) -> None:
    for field in fields(metrics):
        value = getattr(metrics, field.name)
        if isinstance(value, MetricComparison):
            assert value.delta == pytest.approx(value.compared - value.reference)
            assert math.isfinite(value.reference)
            assert math.isfinite(value.compared)
            assert math.isfinite(value.delta)


def test_braking_metrics_use_the_reference_window():
    reference = _lap(
        [
            (0.0, 0.0, 120.0, 0.5, 0.0),
            (10.0, 1.0, 110.0, 0.2, 0.30),
            (15.0, 2.0, 64.0, 0.8, 0.95),
            (20.0, 3.0, 100.0, 0.4, 0.40),
            (30.0, 4.0, 130.0, 0.2, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 90.0, 0.25, 0.60),
            (15.0, 2.0, 88.0, 0.25, 0.50),
            (20.0, 3.0, 50.0, 0.25, 0.40),
            (30.0, 4.0, 110.0, 0.2, 0.0),
        ],
        2,
    )

    braking = _metrics(reference, compared).braking
    assert len(braking) == 1
    metrics = braking[0]

    assert metrics.start_distance == pytest.approx(10.0)
    assert metrics.end_distance == pytest.approx(20.0)
    assert metrics.distance == pytest.approx(10.0)
    assert metrics.speed_at_start.reference == pytest.approx(110.0)
    assert metrics.speed_at_start.compared == pytest.approx(90.0)
    assert metrics.speed_at_end.reference == pytest.approx(100.0)
    assert metrics.speed_at_end.compared == pytest.approx(50.0)
    assert metrics.minimum_speed.reference == pytest.approx(64.0)
    assert metrics.minimum_speed_distance.reference == pytest.approx(15.0)
    assert metrics.minimum_speed.compared == pytest.approx(50.0)
    assert metrics.minimum_speed_distance.compared == pytest.approx(20.0)
    assert metrics.max_brake.reference == pytest.approx(0.95)
    assert metrics.max_brake.compared == pytest.approx(0.60)
    assert metrics.average_throttle.reference == pytest.approx(0.55)
    assert metrics.average_throttle.compared == pytest.approx(0.25)
    _assert_comparison_deltas(metrics)


def test_corner_metrics_define_entry_and_exit_on_the_segment_window():
    reference = _lap(
        [
            (0.0, 0.0, 130.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.80),
            (20.0, 2.0, 60.0, 0.2, 0.20),
            (25.0, 3.0, 80.0, 0.2, 0.0),
            (30.0, 4.0, 100.0, 0.96, 0.0),
            (40.0, 5.0, 140.0, 1.0, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 120.0, 0.2, 0.0),
            (10.0, 1.0, 90.0, 0.2, 0.0),
            (20.0, 2.0, 70.0, 0.2, 0.0),
            (25.0, 3.0, 50.0, 0.2, 0.0),
            (30.0, 4.0, 80.0, 0.2, 0.0),
            (40.0, 5.0, 110.0, 0.2, 0.0),
        ],
        2,
    )

    corner = _metrics(reference, compared).corners
    assert len(corner) == 1
    metrics = corner[0]

    assert metrics.start_distance == pytest.approx(10.0)
    assert metrics.end_distance == pytest.approx(40.0)
    assert metrics.distance == pytest.approx(30.0)
    assert metrics.entry_speed.reference == pytest.approx(100.0)
    assert metrics.entry_speed.compared == pytest.approx(90.0)
    assert metrics.exit_speed.reference == pytest.approx(140.0)
    assert metrics.exit_speed.compared == pytest.approx(110.0)
    assert metrics.minimum_speed.reference == pytest.approx(60.0)
    assert metrics.minimum_speed_distance.reference == pytest.approx(20.0)
    assert metrics.minimum_speed.compared == pytest.approx(50.0)
    assert metrics.minimum_speed_distance.compared == pytest.approx(25.0)
    _assert_comparison_deltas(metrics)


def test_acceleration_metrics_use_the_reference_window():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 80.0, 0.95, 0.0),
            (15.0, 2.0, 100.0, 1.0, 0.0),
            (20.0, 3.0, 140.0, 0.92, 0.0),
            (30.0, 4.0, 150.0, 0.2, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 90.0, 0.1, 0.0),
            (10.0, 1.0, 70.0, 0.4, 0.0),
            (15.0, 2.0, 85.0, 0.7, 0.0),
            (20.0, 3.0, 120.0, 0.5, 0.0),
            (30.0, 4.0, 130.0, 0.1, 0.0),
        ],
        2,
    )

    acceleration = _metrics(reference, compared).acceleration
    assert len(acceleration) == 1
    metrics = acceleration[0]

    assert metrics.start_distance == pytest.approx(10.0)
    assert metrics.end_distance == pytest.approx(20.0)
    assert metrics.speed_at_start.reference == pytest.approx(80.0)
    assert metrics.speed_at_start.compared == pytest.approx(70.0)
    assert metrics.speed_at_end.reference == pytest.approx(140.0)
    assert metrics.speed_at_end.compared == pytest.approx(120.0)
    assert metrics.throttle_at_start.reference == pytest.approx(0.95)
    assert metrics.throttle_at_start.compared == pytest.approx(0.4)
    assert metrics.max_throttle.reference == pytest.approx(1.0)
    assert metrics.max_throttle.compared == pytest.approx(0.7)
    assert metrics.throttle_at_end.reference == pytest.approx(0.92)
    assert metrics.throttle_at_end.compared == pytest.approx(0.5)
    _assert_comparison_deltas(metrics)


def test_interpolates_speed_when_the_window_misses_grid_rows():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (5.0, 1.0, 90.0, 0.2, 0.8),
            (15.0, 2.0, 70.0, 0.2, 0.8),
            (20.0, 3.0, 60.0, 0.2, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 80.0, 0.2, 0.0),
            (5.0, 1.0, 60.0, 0.2, 0.0),
            (15.0, 2.0, 40.0, 0.2, 0.0),
            (20.0, 3.0, 20.0, 0.2, 0.0),
        ],
        2,
    )
    performance = analyze_lap_performance(reference, compared, distance_step=10.0)

    metrics = calculate_driver_performance_metrics(performance, reference, compared).braking[0]

    assert 5.0 not in performance.aligned_data["distance"].tolist()
    assert 15.0 not in performance.aligned_data["distance"].tolist()
    assert metrics.speed_at_start.reference == pytest.approx(90.0)
    assert metrics.speed_at_start.compared == pytest.approx(65.0)
    assert metrics.speed_at_end.reference == pytest.approx(70.0)
    assert metrics.speed_at_end.compared == pytest.approx(35.0)
    assert metrics.speed_at_start.delta == pytest.approx(-25.0)
    assert metrics.speed_at_end.delta == pytest.approx(-35.0)


def test_duration_uses_the_shared_distance_window():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.8),
            (20.0, 6.0, 100.0, 0.2, 0.8),
            (30.0, 7.0, 100.0, 0.2, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 50.0, 0.2, 0.0),
            (10.0, 1.0, 50.0, 0.2, 0.0),
            (20.0, 6.0, 50.0, 0.2, 0.0),
            (30.0, 7.0, 50.0, 0.2, 0.0),
        ],
        2,
    )

    duration = _metrics(reference, compared).braking[0].duration

    assert duration.reference == pytest.approx(0.36)
    assert duration.compared == pytest.approx(0.72)
    assert duration.delta == pytest.approx(0.36)
    assert duration.reference != pytest.approx(5.0)


def test_windows_outside_the_common_distance_are_omitted():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.8),
            (20.0, 2.0, 100.0, 0.2, 0.8),
            (30.0, 3.0, 100.0, 0.95, 0.0),
            (40.0, 4.0, 100.0, 0.95, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 90.0, 0.1, 0.0),
            (10.0, 1.0, 90.0, 0.1, 0.0),
            (25.0, 2.0, 90.0, 0.1, 0.0),
        ],
        2,
    )

    metrics = _metrics(reference, compared)

    assert [(item.start_distance, item.end_distance) for item in metrics.braking] == [
        (pytest.approx(10.0), pytest.approx(20.0))
    ]
    assert metrics.corners == []
    assert metrics.acceleration == []


def test_empty_events_return_empty_groups():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.0),
            (20.0, 2.0, 100.0, 0.2, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 80.0, 0.2, 0.0),
            (10.0, 1.0, 80.0, 0.2, 0.0),
            (20.0, 2.0, 80.0, 0.2, 0.0),
        ],
        2,
    )

    metrics = _metrics(reference, compared)

    assert metrics.braking == []
    assert metrics.corners == []
    assert metrics.acceleration == []


def test_results_are_ordered_by_start_distance():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.8),
            (15.0, 2.0, 90.0, 0.2, 0.8),
            (20.0, 3.0, 110.0, 0.95, 0.0),
            (25.0, 4.0, 120.0, 0.95, 0.0),
            (30.0, 5.0, 120.0, 0.2, 0.0),
            (40.0, 6.0, 100.0, 0.2, 0.8),
            (45.0, 7.0, 90.0, 0.2, 0.8),
            (50.0, 8.0, 110.0, 0.95, 0.0),
            (55.0, 9.0, 130.0, 0.95, 0.0),
        ],
        1,
    )
    compared = _lap(
        [(distance, time, 80.0, 0.2, 0.0) for distance, time, _, _, _ in reference.data.itertuples(index=False)],
        2,
    )

    metrics = _metrics(reference, compared)

    assert [item.start_distance for item in metrics.braking] == pytest.approx([10.0, 40.0])
    assert [item.start_distance for item in metrics.corners] == pytest.approx([10.0, 40.0])
    assert [item.start_distance for item in metrics.acceleration] == pytest.approx([20.0, 50.0])


def test_single_point_window_has_zero_duration():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 88.0, 0.3, 0.7),
            (20.0, 2.0, 100.0, 0.2, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 80.0, 0.2, 0.0),
            (10.0, 1.0, 70.0, 0.4, 0.0),
            (20.0, 2.0, 80.0, 0.2, 0.0),
        ],
        2,
    )

    metrics = _metrics(reference, compared).braking[0]

    assert metrics.distance == pytest.approx(0.0)
    assert metrics.duration.reference == pytest.approx(0.0)
    assert metrics.duration.compared == pytest.approx(0.0)
    assert metrics.speed_at_start.reference == pytest.approx(metrics.speed_at_end.reference)
    assert metrics.speed_at_start.reference == pytest.approx(88.0)


def test_does_not_modify_laps_or_aligned_data():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 80.0, 0.2, 0.8),
            (20.0, 2.0, 90.0, 0.95, 0.0),
            (30.0, 3.0, 120.0, 0.95, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 90.0, 0.2, 0.0),
            (10.0, 1.0, 70.0, 0.2, 0.0),
            (20.0, 2.0, 80.0, 0.2, 0.0),
            (30.0, 3.0, 100.0, 0.2, 0.0),
        ],
        2,
    )
    performance = analyze_lap_performance(reference, compared)
    reference_before = reference.data.copy(deep=True)
    compared_before = compared.data.copy(deep=True)
    aligned_before = performance.aligned_data.copy(deep=True)
    zones_before = detect_braking_zones(reference)
    corners_before = detect_corner_segments(reference)
    acceleration_before = detect_acceleration_zones(reference)
    aligned_identity = id(performance.aligned_data)

    calculate_driver_performance_metrics(performance, reference, compared)

    pd.testing.assert_frame_equal(reference.data, reference_before)
    pd.testing.assert_frame_equal(compared.data, compared_before)
    pd.testing.assert_frame_equal(performance.aligned_data, aligned_before)
    assert id(performance.aligned_data) == aligned_identity
    assert detect_braking_zones(reference) == zones_before
    assert detect_corner_segments(reference) == corners_before
    assert detect_acceleration_zones(reference) == acceleration_before


def test_rejects_missing_required_columns():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.8),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 80.0, 0.2, 0.0),
            (10.0, 1.0, 80.0, 0.2, 0.0),
        ],
        2,
    )
    performance = analyze_lap_performance(reference, compared)
    incomplete = performance.aligned_data.drop(columns=["brake_reference"])
    broken = LapPerformanceAnalysis(
        reference_lap=performance.reference_lap,
        compared_lap=performance.compared_lap,
        total_time_delta=performance.total_time_delta,
        aligned_data=incomplete,
        performance_segments=performance.performance_segments,
    )

    with pytest.raises(ValueError, match="brake_reference"):
        calculate_driver_performance_metrics(broken, reference, compared)


def test_rejects_empty_aligned_data_and_non_finite_values():
    reference = _lap(
        [
            (0.0, 0.0, 100.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 80.0, 0.2, 0.0),
            (10.0, 1.0, 80.0, 0.2, 0.0),
        ],
        2,
    )
    empty = LapPerformanceAnalysis(1, 2, 0.0, pd.DataFrame(), [])
    with pytest.raises(ValueError, match="empty"):
        calculate_driver_performance_metrics(empty, reference, compared)

    performance = analyze_lap_performance(reference, compared)
    invalid = performance.aligned_data.copy()
    invalid.loc[invalid.index[0], "speed_reference"] = np.nan
    broken = LapPerformanceAnalysis(
        reference_lap=1,
        compared_lap=2,
        total_time_delta=performance.total_time_delta,
        aligned_data=invalid,
        performance_segments=[],
    )
    with pytest.raises(ValueError, match="speed_reference"):
        calculate_driver_performance_metrics(broken, reference, compared)


def test_rejects_empty_reference_lap():
    reference = _lap([], 1)
    compared = _lap(
        [
            (0.0, 0.0, 80.0, 0.2, 0.0),
            (10.0, 1.0, 80.0, 0.2, 0.0),
        ],
        2,
    )
    performance = LapPerformanceAnalysis(
        reference_lap=1,
        compared_lap=2,
        total_time_delta=0.0,
        aligned_data=analyze_lap_performance(
            _lap([(0.0, 0.0, 100.0, 0.2, 0.0), (10.0, 1.0, 100.0, 0.2, 0.0)], 1),
            compared,
        ).aligned_data,
        performance_segments=[],
    )

    with pytest.raises(ValueError, match="no telemetry samples"):
        calculate_driver_performance_metrics(performance, reference, compared)


def test_windows_match_existing_detectors_inside_the_aligned_range():
    reference = _lap(
        [
            (0.0, 0.0, 130.0, 0.2, 0.0),
            (10.0, 1.0, 100.0, 0.2, 0.80),
            (20.0, 2.0, 60.0, 0.2, 0.0),
            (30.0, 3.0, 100.0, 0.96, 0.0),
            (40.0, 4.0, 140.0, 1.0, 0.0),
        ],
        1,
    )
    compared = _lap(
        [
            (0.0, 0.0, 90.0, 0.2, 0.0),
            (40.0, 4.0, 90.0, 0.2, 0.0),
        ],
        2,
    )
    performance = analyze_lap_performance(reference, compared)
    metrics = calculate_driver_performance_metrics(performance, reference, compared)
    start = float(performance.aligned_data["distance"].iloc[0])
    end = float(performance.aligned_data["distance"].iloc[-1])

    def inside(zones):
        return [
            (zone.start_distance, zone.end_distance)
            for zone in zones
            if zone.start_distance >= start and zone.end_distance <= end
        ]

    assert [(item.start_distance, item.end_distance) for item in metrics.braking] == inside(
        detect_braking_zones(reference)
    )
    assert [(item.start_distance, item.end_distance) for item in metrics.corners] == inside(
        detect_corner_segments(reference)
    )
    assert [(item.start_distance, item.end_distance) for item in metrics.acceleration] == inside(
        detect_acceleration_zones(reference)
    )


def test_calculate_driver_performance_metrics_on_synthetic_sample_laps():
    dataset = load_csv(SAMPLE_DATASET)
    validate_dataset(dataset)
    laps = segment_laps(normalize_dataset(dataset))
    performance = analyze_lap_performance(laps[2], laps[1])

    metrics = calculate_driver_performance_metrics(performance, laps[2], laps[1])

    _assert_structural_metrics(metrics, performance.aligned_data)


def test_calculate_driver_performance_metrics_on_overlapping_real_laps():
    if not REAL_DATASET.is_file():
        pytest.skip("Real collector CSV is not available in this environment")

    dataset = load_csv(REAL_DATASET)
    untouched = dataset.copy(deep=True)
    validate_dataset(dataset)
    laps = list(segment_laps(normalize_dataset(dataset)).values())
    reference, compared = _first_overlapping_pair(laps)
    performance = analyze_lap_performance(reference, compared)
    aligned_before = performance.aligned_data.copy(deep=True)

    metrics = calculate_driver_performance_metrics(performance, reference, compared)

    _assert_structural_metrics(metrics, performance.aligned_data)
    pd.testing.assert_frame_equal(performance.aligned_data, aligned_before)
    pd.testing.assert_frame_equal(dataset, untouched)


def _assert_structural_metrics(metrics: DriverPerformanceMetrics, aligned: pd.DataFrame) -> None:
    assert isinstance(metrics, DriverPerformanceMetrics)
    start = float(aligned["distance"].iloc[0])
    end = float(aligned["distance"].iloc[-1])
    for group in (metrics.braking, metrics.corners, metrics.acceleration):
        distances = [item.start_distance for item in group]
        assert distances == sorted(distances)
        for item in group:
            assert start <= item.start_distance <= item.end_distance <= end
            _assert_comparison_deltas(item)


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
