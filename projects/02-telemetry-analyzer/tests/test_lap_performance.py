import math
from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.distance import align_laps_by_distance
from telemetry_analyzer.analysis.lap_performance import (
    LapPerformanceAnalysis,
    analyze_lap_performance,
)
from telemetry_analyzer.analysis.performance import detect_performance_segments
from telemetry_analyzer.analysis.time_delta import calculate_time_delta
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

_TIME_COLUMNS = (
    "time_reference",
    "time_compared",
    "time_delta",
    "cumulative_time_delta",
)


def _lap(
    samples: list[tuple[float, float, float, float]],
    lap_number: int,
) -> LapData:
    frame = pd.DataFrame(
        samples,
        columns=["lap_distance", "speed", "throttle", "brake"],
    )
    return LapData(
        lap_number=lap_number,
        data=frame,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        sample_count=len(frame),
    )


def _constant_speed_pair(
    reference_speed: float,
    compared_speed: float,
) -> tuple[LapData, LapData]:
    reference = _lap(
        [(0.0, reference_speed, 0.5, 0.0), (100.0, reference_speed, 0.5, 0.0)],
        lap_number=1,
    )
    compared = _lap(
        [(0.0, compared_speed, 0.5, 0.0), (100.0, compared_speed, 0.5, 0.0)],
        lap_number=2,
    )
    return reference, compared


def test_analyze_lap_performance_returns_analysis():
    reference, compared = _constant_speed_pair(100.0, 50.0)

    result = analyze_lap_performance(reference, compared)

    assert isinstance(result, LapPerformanceAnalysis)


def test_analyze_lap_performance_keeps_lap_numbers():
    reference, compared = _constant_speed_pair(100.0, 50.0)

    result = analyze_lap_performance(reference, compared)

    assert result.reference_lap == reference.lap_number
    assert result.compared_lap == compared.lap_number


def test_total_time_delta_matches_cumulative_compared_minus_reference():
    reference, compared = _constant_speed_pair(100.0, 50.0)

    result = analyze_lap_performance(reference, compared)

    assert result.total_time_delta == pytest.approx(
        result.aligned_data["cumulative_time_delta"].iloc[-1]
    )
    assert result.total_time_delta == pytest.approx(3.6)
    assert result.total_time_delta > 0


def test_negative_total_time_delta_means_compared_took_less_time():
    reference, compared = _constant_speed_pair(50.0, 100.0)

    result = analyze_lap_performance(reference, compared)

    assert result.total_time_delta == pytest.approx(-3.6)


def test_aligned_data_contains_distance_and_time_delta_columns():
    reference, compared = _constant_speed_pair(100.0, 50.0)
    composed = calculate_time_delta(
        align_laps_by_distance(reference, compared)
    )

    result = analyze_lap_performance(reference, compared)

    assert isinstance(result.aligned_data, pd.DataFrame)
    assert list(result.aligned_data.columns) == list(composed.columns)
    assert set(_TIME_COLUMNS).issubset(result.aligned_data.columns)
    pd.testing.assert_frame_equal(result.aligned_data, composed)


def test_performance_segments_come_from_the_existing_detector():
    reference, compared = _constant_speed_pair(100.0, 50.0)
    timed = calculate_time_delta(align_laps_by_distance(reference, compared))

    result = analyze_lap_performance(reference, compared, min_time_delta_change=0.10)

    assert result.performance_segments == detect_performance_segments(
        timed,
        min_time_delta_change=0.10,
    )
    assert result.performance_segments


def test_distance_step_is_forwarded():
    reference, compared = _constant_speed_pair(100.0, 50.0)

    result = analyze_lap_performance(reference, compared, distance_step=5.0)
    expected = align_laps_by_distance(reference, compared, distance_step=5.0)

    assert result.aligned_data["distance"].tolist() == pytest.approx(
        expected["distance"].tolist()
    )


def test_min_time_delta_change_is_forwarded():
    reference, compared = _constant_speed_pair(100.0, 50.0)

    sensitive = analyze_lap_performance(
        reference,
        compared,
        min_time_delta_change=0.10,
    )
    strict = analyze_lap_performance(
        reference,
        compared,
        min_time_delta_change=10.0,
    )

    assert sensitive.performance_segments
    assert strict.performance_segments == []


def test_analyze_lap_performance_does_not_modify_inputs():
    reference, compared = _constant_speed_pair(100.0, 50.0)
    reference_before = reference.data.copy(deep=True)
    compared_before = compared.data.copy(deep=True)

    analyze_lap_performance(reference, compared)

    pd.testing.assert_frame_equal(reference.data, reference_before)
    pd.testing.assert_frame_equal(compared.data, compared_before)


def test_analyze_lap_performance_rejects_laps_without_overlap():
    reference = _lap([(0.0, 100.0, 0.5, 0.0), (10.0, 100.0, 0.5, 0.0)], 1)
    compared = _lap([(20.0, 100.0, 0.5, 0.0), (30.0, 100.0, 0.5, 0.0)], 2)

    with pytest.raises(ValueError, match="no overlapping"):
        analyze_lap_performance(reference, compared)


def test_analyze_lap_performance_on_synthetic_sample_laps():
    dataset = load_csv(SAMPLE_DATASET)
    validate_dataset(dataset)
    laps = segment_laps(normalize_dataset(dataset))

    result = analyze_lap_performance(laps[1], laps[2])

    assert result.reference_lap == 1
    assert result.compared_lap == 2
    assert not result.aligned_data.empty
    assert math.isfinite(result.total_time_delta)
    assert isinstance(result.performance_segments, list)


def test_analyze_lap_performance_on_overlapping_real_laps():
    assert REAL_DATASET.is_file()

    dataset = load_csv(REAL_DATASET)
    untouched = dataset.copy(deep=True)
    validate_dataset(dataset)
    laps = list(segment_laps(normalize_dataset(dataset)).values())
    reference, compared = _first_overlapping_pair(laps)
    reference_before = reference.data.copy(deep=True)
    compared_before = compared.data.copy(deep=True)

    result = analyze_lap_performance(reference, compared)

    assert result.reference_lap == reference.lap_number
    assert result.compared_lap == compared.lap_number
    assert math.isfinite(result.total_time_delta)
    assert "cumulative_time_delta" in result.aligned_data.columns
    assert isinstance(result.performance_segments, list)
    pd.testing.assert_frame_equal(reference.data, reference_before)
    pd.testing.assert_frame_equal(compared.data, compared_before)
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
