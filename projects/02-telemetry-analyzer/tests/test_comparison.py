from dataclasses import replace
from pathlib import Path

import pytest

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics.basic import LapMetrics, calculate_basic_metrics
from telemetry_analyzer.metrics.comparison import LapComparison, compare_laps
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)


def _metrics(**overrides: float | int | dict[int, int]) -> LapMetrics:
    values: dict[str, float | int | dict[int, int]] = {
        "lap_number": 1,
        "duration": 90.500,
        "average_speed": 180.0,
        "maximum_speed": 300.0,
        "minimum_speed": 90.0,
        "average_throttle": 0.80,
        "full_throttle_percentage": 55.0,
        "brake_percentage": 12.0,
        "maximum_rpm": 12000.0,
        "gear_usage": {3: 2, 4: 3},
    }
    values.update(overrides)
    return LapMetrics(**values)


def _example_pair() -> tuple[LapMetrics, LapMetrics]:
    reference = _metrics(lap_number=1)
    compared = _metrics(
        lap_number=2,
        duration=89.800,
        average_speed=182.0,
        maximum_speed=305.0,
        minimum_speed=92.0,
        average_throttle=0.84,
        full_throttle_percentage=60.0,
        brake_percentage=10.0,
        maximum_rpm=12100.0,
        gear_usage={2: 4, 5: 1},
    )
    return reference, compared


def test_compare_laps_time_delta():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).time_delta == pytest.approx(-0.7)


def test_compare_laps_average_speed_delta():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).average_speed_delta == pytest.approx(2.0)


def test_compare_laps_maximum_speed_delta():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).maximum_speed_delta == pytest.approx(5.0)


def test_compare_laps_minimum_speed_delta():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).minimum_speed_delta == pytest.approx(2.0)


def test_compare_laps_average_throttle_delta():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).average_throttle_delta == pytest.approx(
        0.04
    )


def test_compare_laps_full_throttle_percentage_delta():
    reference, compared = _example_pair()
    comparison = compare_laps(reference, compared)

    assert comparison.full_throttle_percentage_delta == pytest.approx(5.0)


def test_compare_laps_brake_percentage_delta():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).brake_percentage_delta == pytest.approx(
        -2.0
    )


def test_compare_laps_maximum_rpm_delta():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).maximum_rpm_delta == pytest.approx(100.0)


def test_compare_laps_reference_lap():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).reference_lap == 1


def test_compare_laps_compared_lap():
    reference, compared = _example_pair()

    assert compare_laps(reference, compared).compared_lap == 2


def test_compare_laps_negative_deltas():
    reference = _metrics(lap_number=1)
    compared = _metrics(
        lap_number=3,
        duration=89.0,
        average_speed=170.0,
        maximum_speed=290.0,
        minimum_speed=80.0,
        average_throttle=0.70,
        full_throttle_percentage=40.0,
        brake_percentage=8.0,
        maximum_rpm=11000.0,
    )
    comparison = compare_laps(reference, compared)

    assert comparison.time_delta == pytest.approx(-1.5)
    assert comparison.average_speed_delta == pytest.approx(-10.0)
    assert comparison.maximum_speed_delta == pytest.approx(-10.0)
    assert comparison.minimum_speed_delta == pytest.approx(-10.0)
    assert comparison.average_throttle_delta == pytest.approx(-0.10)
    assert comparison.full_throttle_percentage_delta == pytest.approx(-15.0)
    assert comparison.brake_percentage_delta == pytest.approx(-4.0)
    assert comparison.maximum_rpm_delta == pytest.approx(-1000.0)


def test_compare_laps_positive_deltas():
    reference = _metrics(lap_number=1, duration=90.0)
    compared = _metrics(
        lap_number=4,
        duration=91.2,
        average_speed=190.0,
        maximum_speed=310.0,
        minimum_speed=100.0,
        average_throttle=0.90,
        full_throttle_percentage=70.0,
        brake_percentage=18.0,
        maximum_rpm=12500.0,
    )
    comparison = compare_laps(reference, compared)

    assert comparison.time_delta == pytest.approx(1.2)
    assert comparison.average_speed_delta == pytest.approx(10.0)
    assert comparison.maximum_speed_delta == pytest.approx(10.0)
    assert comparison.minimum_speed_delta == pytest.approx(10.0)
    assert comparison.average_throttle_delta == pytest.approx(0.10)
    assert comparison.full_throttle_percentage_delta == pytest.approx(15.0)
    assert comparison.brake_percentage_delta == pytest.approx(6.0)
    assert comparison.maximum_rpm_delta == pytest.approx(500.0)


def test_compare_laps_identical_metrics_are_zero():
    reference = _metrics(lap_number=1)
    compared = _metrics(lap_number=2)
    comparison = compare_laps(reference, compared)

    assert comparison.time_delta == pytest.approx(0.0)
    assert comparison.average_speed_delta == pytest.approx(0.0)
    assert comparison.maximum_speed_delta == pytest.approx(0.0)
    assert comparison.minimum_speed_delta == pytest.approx(0.0)
    assert comparison.average_throttle_delta == pytest.approx(0.0)
    assert comparison.full_throttle_percentage_delta == pytest.approx(0.0)
    assert comparison.brake_percentage_delta == pytest.approx(0.0)
    assert comparison.maximum_rpm_delta == pytest.approx(0.0)


def test_compare_laps_does_not_modify_inputs():
    reference, compared = _example_pair()
    reference_before = replace(reference, gear_usage=dict(reference.gear_usage))
    compared_before = replace(compared, gear_usage=dict(compared.gear_usage))

    comparison = compare_laps(reference, compared)

    assert isinstance(comparison, LapComparison)
    assert reference == reference_before
    assert compared == compared_before
    assert comparison is not reference
    assert comparison is not compared


def test_pipeline_compares_sample_laps():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)
    reference = calculate_basic_metrics(laps[1])
    compared = calculate_basic_metrics(laps[2])
    comparison = compare_laps(reference, compared)

    assert comparison.reference_lap == 1
    assert comparison.compared_lap == 2
    assert comparison.time_delta == pytest.approx(0.0)
    assert comparison.average_speed_delta == pytest.approx(6.75)
    assert comparison.maximum_speed_delta == pytest.approx(10.0)
    assert comparison.minimum_speed_delta == pytest.approx(2.0)
    assert comparison.average_throttle_delta == pytest.approx(-0.2125)
    assert comparison.full_throttle_percentage_delta == pytest.approx(0.0)
    assert comparison.brake_percentage_delta == pytest.approx(50.0)
    assert comparison.maximum_rpm_delta == pytest.approx(600.0)
    assert comparison.time_delta == pytest.approx(compared.duration - reference.duration)
    assert comparison.average_speed_delta == pytest.approx(
        compared.average_speed - reference.average_speed
    )
