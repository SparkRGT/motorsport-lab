from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.zones import (
    AccelerationZone,
    BrakingZone,
    detect_acceleration_zones,
    detect_braking_zones,
)
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)


def _lap(
    brake: list[float],
    throttle: list[float] | None = None,
    distance: list[float] | None = None,
    time: list[float] | None = None,
) -> LapData:
    count = len(brake)
    frame = pd.DataFrame(
        {
            "lap_distance": distance if distance is not None else [float(index) for index in range(count)],
            "session_time": time if time is not None else [index * 0.1 for index in range(count)],
            "brake": brake,
            "throttle": throttle if throttle is not None else [0.0] * count,
        }
    )
    return LapData(
        lap_number=1,
        data=frame,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        sample_count=count,
    )


def test_detect_braking_zones_returns_empty_when_absent():
    zones = detect_braking_zones(_lap([0.0, 0.0, 0.04]))

    assert zones == []


def test_detect_braking_zones_groups_one_zone():
    zones = detect_braking_zones(_lap([0.00, 0.00, 0.12, 0.35, 0.80, 0.50, 0.20, 0.00]))

    assert len(zones) == 1
    assert isinstance(zones[0], BrakingZone)
    assert zones[0].sample_count == 5


def test_detect_braking_zones_finds_several_zones():
    zones = detect_braking_zones(_lap([0.0, 0.2, 0.3, 0.0, 0.0, 0.4, 0.5, 0.0]))

    assert len(zones) == 2
    assert zones[0].start_distance < zones[1].start_distance


def test_detect_braking_zones_accepts_isolated_sample():
    zones = detect_braking_zones(_lap([0.0, 0.4, 0.0]))

    assert len(zones) == 1
    assert zones[0].start_distance == pytest.approx(zones[0].end_distance)
    assert zones[0].distance == pytest.approx(0.0)
    assert zones[0].start_time == pytest.approx(zones[0].end_time)
    assert zones[0].duration == pytest.approx(0.0)
    assert zones[0].sample_count == 1


def test_detect_braking_zones_uses_custom_threshold():
    lap = _lap([0.10, 0.20, 0.0])

    assert detect_braking_zones(lap, brake_threshold=0.15)[0].sample_count == 1
    assert detect_braking_zones(lap, brake_threshold=0.05)[0].sample_count == 2


def test_detect_braking_zones_threshold_zero_includes_any_positive_brake():
    zones = detect_braking_zones(_lap([0.0, 0.01, 0.0]), brake_threshold=0)

    assert len(zones) == 1
    assert zones[0].sample_count == 1


def test_detect_braking_zones_threshold_one_excludes_full_brake():
    zones = detect_braking_zones(_lap([1.0, 1.0]), brake_threshold=1)

    assert zones == []


def test_detect_braking_zones_rejects_invalid_threshold():
    lap = _lap([0.2, 0.0])

    with pytest.raises(ValueError, match="brake_threshold"):
        detect_braking_zones(lap, brake_threshold=-0.1)
    with pytest.raises(ValueError, match="brake_threshold"):
        detect_braking_zones(lap, brake_threshold=1.1)


def test_detect_braking_zones_groups_consecutive_samples():
    zones = detect_braking_zones(_lap([0.0, 0.2, 0.4, 0.6, 0.0]))

    assert len(zones) == 1
    assert zones[0].sample_count == 3


def test_detect_braking_zones_distance():
    zones = detect_braking_zones(
        _lap([0.0, 0.2, 0.4, 0.0], distance=[0.0, 10.0, 25.0, 40.0])
    )

    assert zones[0].start_distance == pytest.approx(10.0)
    assert zones[0].end_distance == pytest.approx(25.0)
    assert zones[0].distance == pytest.approx(15.0)


def test_detect_braking_zones_duration():
    zones = detect_braking_zones(
        _lap([0.0, 0.2, 0.4, 0.0], time=[1.0, 1.2, 1.5, 1.8])
    )

    assert zones[0].start_time == pytest.approx(1.2)
    assert zones[0].end_time == pytest.approx(1.5)
    assert zones[0].duration == pytest.approx(0.3)


def test_detect_braking_zones_sample_count():
    zones = detect_braking_zones(_lap([0.2, 0.3, 0.4, 0.0]))

    assert zones[0].sample_count == 3


def test_detect_acceleration_zones_returns_empty_when_absent():
    zones = detect_acceleration_zones(_lap([0.0, 0.0], throttle=[0.5, 0.89]))

    assert zones == []


def test_detect_acceleration_zones_groups_one_zone():
    zones = detect_acceleration_zones(
        _lap([0.0] * 7, throttle=[0.50, 0.70, 0.91, 0.97, 1.00, 0.95, 0.70])
    )

    assert len(zones) == 1
    assert isinstance(zones[0], AccelerationZone)
    assert zones[0].sample_count == 4


def test_detect_acceleration_zones_finds_several_zones():
    zones = detect_acceleration_zones(
        _lap([0.0] * 6, throttle=[0.2, 0.95, 1.0, 0.2, 0.91, 0.4])
    )

    assert len(zones) == 2
    assert zones[0].start_distance < zones[1].start_distance


def test_detect_acceleration_zones_accepts_isolated_sample():
    zones = detect_acceleration_zones(_lap([0.0, 0.0, 0.0], throttle=[0.2, 1.0, 0.2]))

    assert len(zones) == 1
    assert zones[0].distance == pytest.approx(0.0)
    assert zones[0].duration == pytest.approx(0.0)
    assert zones[0].sample_count == 1


def test_detect_acceleration_zones_uses_custom_threshold():
    lap = _lap([0.0, 0.0], throttle=[0.80, 0.95])

    assert len(detect_acceleration_zones(lap, throttle_threshold=0.90)) == 1
    assert detect_acceleration_zones(lap, throttle_threshold=0.70)[0].sample_count == 2


def test_detect_acceleration_zones_rejects_invalid_threshold():
    lap = _lap([0.0], throttle=[1.0])

    with pytest.raises(ValueError, match="throttle_threshold"):
        detect_acceleration_zones(lap, throttle_threshold=1.5)


def test_detect_acceleration_zones_groups_consecutive_samples():
    zones = detect_acceleration_zones(
        _lap([0.0] * 5, throttle=[0.2, 0.91, 0.95, 1.0, 0.2])
    )

    assert len(zones) == 1
    assert zones[0].sample_count == 3


def test_detect_acceleration_zones_distance():
    zones = detect_acceleration_zones(
        _lap([0.0, 0.0, 0.0], throttle=[0.2, 0.95, 1.0], distance=[0.0, 8.0, 14.0])
    )

    assert zones[0].start_distance == pytest.approx(8.0)
    assert zones[0].end_distance == pytest.approx(14.0)
    assert zones[0].distance == pytest.approx(6.0)


def test_detect_acceleration_zones_duration():
    zones = detect_acceleration_zones(
        _lap([0.0, 0.0], throttle=[0.95, 1.0], time=[2.0, 2.4])
    )

    assert zones[0].start_time == pytest.approx(2.0)
    assert zones[0].end_time == pytest.approx(2.4)
    assert zones[0].duration == pytest.approx(0.4)


def test_detect_acceleration_zones_sample_count():
    zones = detect_acceleration_zones(_lap([0.0, 0.0, 0.0], throttle=[0.91, 0.95, 1.0]))

    assert zones[0].sample_count == 3


def test_zone_detection_rejects_empty_lap():
    lap = _lap([])

    with pytest.raises(ValueError, match="no telemetry"):
        detect_braking_zones(lap)
    with pytest.raises(ValueError, match="no telemetry"):
        detect_acceleration_zones(lap)


def test_zone_detection_rejects_missing_columns():
    lap = _lap([0.2], throttle=[0.95])
    lap.data = lap.data.drop(columns=["brake"])

    with pytest.raises(ValueError, match="brake"):
        detect_braking_zones(lap)


def test_zone_detection_rejects_invalid_values():
    high_brake = _lap([1.2])
    text_distance = _lap([0.2])
    text_distance.data["lap_distance"] = ["start"]
    backwards_time = _lap([0.2, 0.3], time=[1.0, 0.5])

    with pytest.raises(ValueError, match="brake"):
        detect_braking_zones(high_brake)
    with pytest.raises(ValueError, match="lap_distance"):
        detect_braking_zones(text_distance)
    with pytest.raises(ValueError, match="session_time"):
        detect_braking_zones(backwards_time)
    with pytest.raises(TypeError):
        detect_acceleration_zones(pd.DataFrame())


def test_zone_detection_does_not_modify_lap():
    lap = _lap([0.0, 0.4, 0.0], throttle=[0.2, 1.0, 0.2])
    original = lap.data.copy()

    detect_braking_zones(lap)
    detect_acceleration_zones(lap)

    pd.testing.assert_frame_equal(lap.data, original)


def test_pipeline_detects_zones_on_sample_laps():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)

    braking = detect_braking_zones(laps[2])
    acceleration = detect_acceleration_zones(laps[1])

    assert len(braking) == 1
    assert isinstance(braking[0], BrakingZone)
    assert braking[0].sample_count == 2
    assert braking[0].start_distance == pytest.approx(0.0)
    assert braking[0].end_distance == pytest.approx(1.5)
    assert braking[0].distance == pytest.approx(1.5)
    assert detect_braking_zones(laps[1]) == []

    assert len(acceleration) == 1
    assert isinstance(acceleration[0], AccelerationZone)
    assert acceleration[0].sample_count == 1
    assert acceleration[0].distance == pytest.approx(0.0)
    assert acceleration[0].duration == pytest.approx(0.0)
