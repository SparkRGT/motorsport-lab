from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics.basic import LapMetrics, calculate_basic_metrics
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)


def _lap(
    rows: list[dict[str, float]],
    lap_number: int = 1,
    duration: float = 0.4,
) -> LapData:
    return LapData(
        lap_number=lap_number,
        data=pd.DataFrame(rows),
        start_time=0.0,
        end_time=duration,
        duration=duration,
        sample_count=len(rows),
    )


def _sample_lap() -> LapData:
    return _lap(
        [
            {"speed": 100, "throttle": 1.00, "brake": 0.0, "rpm": 8000, "gear": 3},
            {"speed": 80, "throttle": 0.99, "brake": 0.0, "rpm": 9000, "gear": 3},
            {"speed": 120, "throttle": 0.95, "brake": 0.5, "rpm": 11000, "gear": 4},
            {"speed": 90, "throttle": 1.00, "brake": 1.0, "rpm": 10000, "gear": 4},
            {"speed": 110, "throttle": 0.80, "brake": 0.0, "rpm": 8500, "gear": 4},
        ]
    )


def test_calculate_basic_metrics_average_speed():
    metrics = calculate_basic_metrics(_sample_lap())

    assert metrics.average_speed == pytest.approx(100.0)


def test_calculate_basic_metrics_maximum_speed():
    metrics = calculate_basic_metrics(_sample_lap())

    assert metrics.maximum_speed == pytest.approx(120.0)


def test_calculate_basic_metrics_minimum_speed():
    metrics = calculate_basic_metrics(_sample_lap())

    assert metrics.minimum_speed == pytest.approx(80.0)


def test_calculate_basic_metrics_average_throttle():
    metrics = calculate_basic_metrics(_sample_lap())

    assert metrics.average_throttle == pytest.approx(0.948)


def test_calculate_basic_metrics_full_throttle_percentage():
    metrics = calculate_basic_metrics(_sample_lap())

    assert metrics.full_throttle_percentage == pytest.approx(60.0)


def test_calculate_basic_metrics_brake_percentage():
    metrics = calculate_basic_metrics(_sample_lap())

    assert metrics.brake_percentage == pytest.approx(40.0)


def test_calculate_basic_metrics_maximum_rpm():
    metrics = calculate_basic_metrics(_sample_lap())

    assert metrics.maximum_rpm == pytest.approx(11000.0)


def test_calculate_basic_metrics_gear_usage():
    lap = _lap(
        [
            {"speed": 90, "throttle": 0.5, "brake": 0.0, "rpm": 8000, "gear": gear}
            for gear in (3, 3, 4, 4, 4, 5)
        ]
    )
    metrics = calculate_basic_metrics(lap)

    assert metrics.gear_usage == {3: 2, 4: 3, 5: 1}


def test_calculate_basic_metrics_uses_lap_duration_and_number():
    lap = _sample_lap()
    metrics = calculate_basic_metrics(lap)

    assert metrics.duration == lap.duration
    assert metrics.lap_number == lap.lap_number


def test_calculate_basic_metrics_does_not_modify_lap_data():
    lap = _sample_lap()
    original = lap.data.copy()

    calculate_basic_metrics(lap)

    pd.testing.assert_frame_equal(lap.data, original)


def test_calculate_basic_metrics_single_sample():
    lap = _lap(
        [{"speed": 88, "throttle": 0.99, "brake": 0.2, "rpm": 6400, "gear": 2}],
        lap_number=7,
        duration=0.0,
    )
    metrics = calculate_basic_metrics(lap)

    assert isinstance(metrics, LapMetrics)
    assert metrics.lap_number == 7
    assert metrics.duration == 0.0
    assert metrics.average_speed == pytest.approx(88.0)
    assert metrics.maximum_speed == pytest.approx(88.0)
    assert metrics.minimum_speed == pytest.approx(88.0)
    assert metrics.average_throttle == pytest.approx(0.99)
    assert metrics.full_throttle_percentage == pytest.approx(100.0)
    assert metrics.brake_percentage == pytest.approx(100.0)
    assert metrics.maximum_rpm == pytest.approx(6400.0)
    assert metrics.gear_usage == {2: 1}


def test_calculate_basic_metrics_without_full_throttle():
    lap = _lap(
        [
            {"speed": 70, "throttle": 0.0, "brake": 0.0, "rpm": 6000, "gear": 2},
            {"speed": 75, "throttle": 0.50, "brake": 0.0, "rpm": 6500, "gear": 2},
            {"speed": 78, "throttle": 0.98, "brake": 0.0, "rpm": 7000, "gear": 3},
        ]
    )
    metrics = calculate_basic_metrics(lap)

    assert metrics.full_throttle_percentage == pytest.approx(0.0)


def test_calculate_basic_metrics_without_braking():
    lap = _lap(
        [
            {"speed": 100, "throttle": 1.0, "brake": 0.0, "rpm": 9000, "gear": 4},
            {"speed": 110, "throttle": 1.0, "brake": 0.0, "rpm": 9500, "gear": 5},
        ]
    )
    metrics = calculate_basic_metrics(lap)

    assert metrics.brake_percentage == pytest.approx(0.0)


def test_pipeline_calculates_basic_metrics():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)
    before = normalized.copy()
    metrics = {
        lap_number: calculate_basic_metrics(lap) for lap_number, lap in laps.items()
    }

    lap_one = metrics[1]
    assert isinstance(lap_one, LapMetrics)
    assert lap_one.lap_number == 1
    assert lap_one.duration == laps[1].duration
    assert lap_one.average_speed == pytest.approx(87.0)
    assert lap_one.maximum_speed == pytest.approx(110.0)
    assert lap_one.minimum_speed == pytest.approx(70.0)
    assert lap_one.average_throttle == pytest.approx(0.7125)
    assert lap_one.full_throttle_percentage == pytest.approx(25.0)
    assert lap_one.brake_percentage == pytest.approx(0.0)
    assert lap_one.maximum_rpm == pytest.approx(9800.0)
    assert lap_one.gear_usage == {2: 1, 3: 2, 4: 1}
    assert list(metrics) == [1, 2, 3]
    pd.testing.assert_frame_equal(normalized, before)
