from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.distance import align_laps_by_distance
from telemetry_analyzer.analysis.performance import (
    PerformanceSegment,
    detect_performance_segments,
)
from telemetry_analyzer.analysis.time_delta import calculate_time_delta
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)


def _frame(distance: list[float], cumulative_time_delta: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance": distance,
            "cumulative_time_delta": cumulative_time_delta,
            "speed_reference": [100.0] * len(distance),
        }
    )


def test_detect_performance_segments_without_bottlenecks():
    data = _frame([0.0, 10.0, 20.0], [0.00, 0.02, 0.04])

    assert detect_performance_segments(data) == []


def test_detect_performance_segments_positive_change():
    segments = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))

    assert len(segments) == 1
    assert isinstance(segments[0], PerformanceSegment)
    assert segments[0].time_delta_change == pytest.approx(0.20)
    assert segments[0].time_delta_change > 0


def test_detect_performance_segments_negative_change():
    segments = detect_performance_segments(_frame([0.0, 10.0, 20.0], [0.30, 0.20, 0.05]))

    assert len(segments) == 1
    assert segments[0].time_delta_change == pytest.approx(-0.25)
    assert segments[0].absolute_time_delta_change == pytest.approx(0.25)


def test_detect_performance_segments_finds_multiple_segments():
    data = _frame([0.0, 10.0, 20.0, 30.0, 40.0], [0.00, 0.12, 0.20, 0.05, -0.02])
    segments = detect_performance_segments(data)

    assert len(segments) == 2
    assert segments[0].time_delta_change == pytest.approx(0.20)
    assert segments[1].time_delta_change == pytest.approx(-0.22)


def test_detect_performance_segments_uses_custom_threshold():
    data = _frame([0.0, 10.0], [0.00, 0.08])

    assert detect_performance_segments(data) == []
    segments = detect_performance_segments(data, min_time_delta_change=0.05)

    assert len(segments) == 1
    assert segments[0].time_delta_change == pytest.approx(0.08)


def test_detect_performance_segments_rejects_invalid_threshold():
    data = _frame([0.0, 10.0], [0.00, 0.20])

    with pytest.raises(ValueError, match="min_time_delta_change"):
        detect_performance_segments(data, min_time_delta_change=0)
    with pytest.raises(ValueError, match="min_time_delta_change"):
        detect_performance_segments(data, min_time_delta_change=-0.1)


def test_detect_performance_segments_start_distance():
    segment = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))[0]

    assert segment.start_distance == pytest.approx(100.0)


def test_detect_performance_segments_end_distance():
    segment = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))[0]

    assert segment.end_distance == pytest.approx(130.0)


def test_detect_performance_segments_distance():
    segment = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))[0]

    assert segment.distance == pytest.approx(30.0)
    assert segment.distance == pytest.approx(segment.end_distance - segment.start_distance)


def test_detect_performance_segments_start_time_delta():
    segment = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))[0]

    assert segment.start_time_delta == pytest.approx(0.00)


def test_detect_performance_segments_end_time_delta():
    segment = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))[0]

    assert segment.end_time_delta == pytest.approx(0.20)


def test_detect_performance_segments_time_delta_change():
    segment = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))[0]

    assert segment.time_delta_change == pytest.approx(
        segment.end_time_delta - segment.start_time_delta
    )


def test_detect_performance_segments_absolute_time_delta_change():
    segment = detect_performance_segments(_frame([0.0, 10.0, 20.0], [0.30, 0.10, 0.00]))[0]

    assert segment.absolute_time_delta_change == pytest.approx(abs(segment.time_delta_change))
    assert segment.time_delta_change == pytest.approx(-0.30)


def test_detect_performance_segments_sample_count():
    segment = detect_performance_segments(_frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20]))[0]

    assert segment.sample_count == 4


def test_detect_performance_segments_keeps_distance_order():
    segments = detect_performance_segments(
        _frame([0.0, 10.0, 20.0, 30.0], [0.00, 0.15, 0.15, 0.00])
    )

    assert [segment.start_distance for segment in segments] == pytest.approx([0.0, 20.0])


def test_detect_performance_segments_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        detect_performance_segments(_frame([], []))


def test_detect_performance_segments_rejects_missing_columns():
    data = _frame([0.0, 10.0], [0.00, 0.20]).drop(columns=["cumulative_time_delta"])

    with pytest.raises(ValueError, match="cumulative_time_delta"):
        detect_performance_segments(data)


def test_detect_performance_segments_rejects_non_numeric_distance():
    data = _frame([0.0, 10.0], [0.00, 0.20])
    data["distance"] = ["start", "end"]

    with pytest.raises(ValueError, match="distance"):
        detect_performance_segments(data)


def test_detect_performance_segments_rejects_non_numeric_time_delta():
    data = _frame([0.0, 10.0], [0.00, 0.20])
    data["cumulative_time_delta"] = ["low", "high"]

    with pytest.raises(ValueError, match="cumulative_time_delta"):
        detect_performance_segments(data)


def test_detect_performance_segments_rejects_decreasing_distance():
    data = _frame([0.0, 10.0, 5.0], [0.00, 0.05, 0.20])

    with pytest.raises(ValueError, match="strictly increasing"):
        detect_performance_segments(data)


def test_detect_performance_segments_rejects_missing_values():
    data = _frame([0.0, 10.0, 20.0], [0.00, float("nan"), 0.20])

    with pytest.raises(ValueError, match="numeric"):
        detect_performance_segments(data)


def test_detect_performance_segments_rejects_non_dataframe():
    with pytest.raises(TypeError):
        detect_performance_segments([0.0, 0.2])


def test_detect_performance_segments_does_not_modify_original():
    data = _frame([100, 110, 120, 130], [0.00, 0.02, 0.15, 0.20])
    original = data.copy()

    detect_performance_segments(data)

    pd.testing.assert_frame_equal(data, original)


def test_pipeline_detects_performance_segments():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)
    aligned = align_laps_by_distance(laps[1], laps[2])
    timed = calculate_time_delta(aligned)
    segments = detect_performance_segments(timed, min_time_delta_change=0.001)

    assert isinstance(segments, list)
    assert segments
    for segment in segments:
        assert isinstance(segment, PerformanceSegment)
        assert segment.distance == pytest.approx(segment.end_distance - segment.start_distance)
        assert segment.time_delta_change == pytest.approx(
            segment.end_time_delta - segment.start_time_delta
        )
        assert segment.absolute_time_delta_change == pytest.approx(abs(segment.time_delta_change))
        assert segment.sample_count >= 2
        assert segment.absolute_time_delta_change >= 0.001
    assert [segment.start_distance for segment in segments] == sorted(
        segment.start_distance for segment in segments
    )
