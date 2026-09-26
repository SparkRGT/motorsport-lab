from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.corners import CornerSegment, detect_corner_segments
from telemetry_analyzer.analysis.zones import detect_acceleration_zones, detect_braking_zones
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)


def _lap(rows: list[tuple[float, float, float, float, float]]) -> LapData:
    frame = pd.DataFrame(
        rows,
        columns=["lap_distance", "session_time", "speed", "brake", "throttle"],
    )
    return LapData(
        lap_number=1,
        data=frame,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        sample_count=len(frame),
    )


def _one_corner() -> LapData:
    return _lap(
        [
            (90.0, 1.0, 120.0, 0.00, 0.40),
            (100.0, 1.2, 100.0, 0.20, 0.10),
            (130.0, 1.6, 70.0, 0.40, 0.00),
            (150.0, 2.0, 80.0, 0.00, 0.95),
            (190.0, 2.4, 110.0, 0.00, 1.00),
            (210.0, 2.8, 130.0, 0.00, 0.40),
        ]
    )


def _two_corners() -> LapData:
    return _lap(
        [
            (10.0, 0.1, 90.0, 0.30, 0.10),
            (20.0, 0.2, 70.0, 0.50, 0.10),
            (30.0, 0.3, 80.0, 0.00, 0.20),
            (40.0, 0.4, 95.0, 0.00, 0.95),
            (50.0, 0.5, 110.0, 0.00, 1.00),
            (60.0, 0.6, 100.0, 0.00, 0.20),
            (70.0, 0.7, 80.0, 0.40, 0.10),
            (80.0, 0.8, 60.0, 0.20, 0.10),
            (90.0, 0.9, 75.0, 0.00, 0.30),
            (100.0, 1.0, 100.0, 0.00, 0.95),
        ]
    )


def test_detect_corner_segments_without_braking():
    lap = _lap([(0.0, 0.0, 100.0, 0.0, 1.0), (10.0, 0.1, 110.0, 0.0, 1.0)])

    assert detect_corner_segments(lap) == []


def test_detect_corner_segments_without_later_acceleration():
    lap = _lap([(0.0, 0.0, 100.0, 0.40, 0.20), (10.0, 0.1, 80.0, 0.10, 0.30)])

    assert detect_corner_segments(lap) == []


def test_detect_corner_segments_finds_one_corner():
    segments = detect_corner_segments(_one_corner())

    assert len(segments) == 1
    assert isinstance(segments[0], CornerSegment)
    assert segments[0].start_distance == pytest.approx(100.0)
    assert segments[0].end_distance == pytest.approx(190.0)


def test_detect_corner_segments_finds_multiple_corners():
    segments = detect_corner_segments(_two_corners())

    assert len(segments) == 2
    assert segments[0].start_distance == pytest.approx(10.0)
    assert segments[1].start_distance == pytest.approx(70.0)


def test_detect_corner_segments_keeps_distance_order():
    segments = detect_corner_segments(_two_corners())

    assert [segment.start_distance for segment in segments] == pytest.approx([10.0, 70.0])


def test_detect_corner_segments_distance():
    segment = detect_corner_segments(_one_corner())[0]

    assert segment.distance == pytest.approx(90.0)
    assert segment.distance == pytest.approx(segment.end_distance - segment.start_distance)


def test_detect_corner_segments_time():
    segment = detect_corner_segments(_one_corner())[0]

    assert segment.start_time == pytest.approx(1.2)
    assert segment.end_time == pytest.approx(2.4)
    assert segment.duration == pytest.approx(1.2)


def test_detect_corner_segments_minimum_speed():
    segment = detect_corner_segments(_one_corner())[0]

    assert segment.minimum_speed == pytest.approx(70.0)


def test_detect_corner_segments_minimum_speed_distance():
    segment = detect_corner_segments(_one_corner())[0]

    assert segment.minimum_speed_distance == pytest.approx(130.0)


def test_detect_corner_segments_sample_count():
    segment = detect_corner_segments(_one_corner())[0]

    assert segment.sample_count == 4


def test_detect_corner_segments_uses_custom_thresholds():
    lap = _lap([(0.0, 0.0, 90.0, 0.10, 0.20), (20.0, 0.2, 70.0, 0.00, 0.80)])

    assert detect_corner_segments(lap) == []
    segments = detect_corner_segments(lap, brake_threshold=0.05, throttle_threshold=0.70)

    assert len(segments) == 1
    assert segments[0].end_distance == pytest.approx(20.0)


def test_detect_corner_segments_rejects_invalid_threshold():
    lap = _one_corner()

    with pytest.raises(ValueError, match="brake_threshold"):
        detect_corner_segments(lap, brake_threshold=-0.1)
    with pytest.raises(ValueError, match="throttle_threshold"):
        detect_corner_segments(lap, throttle_threshold=1.2)


def test_detect_corner_segments_rejects_empty_lap():
    with pytest.raises(ValueError, match="no telemetry"):
        detect_corner_segments(_lap([]))


def test_detect_corner_segments_rejects_missing_columns():
    lap = _one_corner()
    lap.data = lap.data.drop(columns=["speed"])

    with pytest.raises(ValueError, match="speed"):
        detect_corner_segments(lap)


def test_detect_corner_segments_rejects_invalid_data():
    lap = _one_corner()
    lap.data.loc[0, "brake"] = 1.5

    with pytest.raises(ValueError, match="brake"):
        detect_corner_segments(lap)
    with pytest.raises(TypeError):
        detect_corner_segments(pd.DataFrame())


def test_detect_corner_segments_does_not_modify_lap():
    lap = _one_corner()
    original = lap.data.copy()

    detect_corner_segments(lap)

    pd.testing.assert_frame_equal(lap.data, original)


def test_detect_corner_segments_uses_first_minimum_speed():
    lap = _lap(
        [
            (0.0, 0.0, 50.0, 0.20, 0.10),
            (10.0, 0.1, 40.0, 0.10, 0.10),
            (20.0, 0.2, 40.0, 0.00, 0.20),
            (30.0, 0.3, 60.0, 0.00, 0.95),
        ]
    )
    segment = detect_corner_segments(lap)[0]

    assert segment.minimum_speed == pytest.approx(40.0)
    assert segment.minimum_speed_distance == pytest.approx(10.0)


def test_detect_corner_segments_matches_braking_zone_start():
    lap = _one_corner()
    segment = detect_corner_segments(lap)[0]
    braking = detect_braking_zones(lap)[0]

    assert segment.start_distance == pytest.approx(braking.start_distance)


def test_detect_corner_segments_matches_acceleration_zone_end():
    lap = _one_corner()
    segment = detect_corner_segments(lap)[0]
    acceleration = detect_acceleration_zones(lap)[0]

    assert segment.end_distance == pytest.approx(acceleration.end_distance)


def test_pipeline_detects_corner_segments_on_sample_lap():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)
    segments = detect_corner_segments(laps[2])

    assert len(segments) == 1
    segment = segments[0]
    assert isinstance(segment, CornerSegment)
    assert segment.start_distance == pytest.approx(0.0)
    assert segment.end_distance == pytest.approx(5.0)
    assert segment.distance == pytest.approx(5.0)
    assert segment.start_time == pytest.approx(0.2)
    assert segment.end_time == pytest.approx(0.35)
    assert segment.duration == pytest.approx(0.15)
    assert segment.minimum_speed == pytest.approx(72.0)
    assert segment.minimum_speed_distance == pytest.approx(1.5)
    assert segment.sample_count == 4
    assert detect_corner_segments(laps[1]) == []
