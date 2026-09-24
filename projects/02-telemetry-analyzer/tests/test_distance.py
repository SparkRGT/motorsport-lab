from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.distance import align_laps_by_distance
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)

EXPECTED_COLUMNS = [
    "distance",
    "speed_reference",
    "speed_compared",
    "speed_delta",
    "throttle_reference",
    "throttle_compared",
    "throttle_delta",
    "brake_reference",
    "brake_compared",
    "brake_delta",
]


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


def _linear_pair() -> tuple[LapData, LapData]:
    reference = _lap(
        [
            (0.0, 0.0, 0.0, 0.0),
            (10.0, 100.0, 1.0, 0.5),
        ],
        lap_number=1,
    )
    compared = _lap(
        [
            (0.0, 10.0, 0.2, 0.0),
            (10.0, 20.0, 0.4, 1.0),
        ],
        lap_number=2,
    )
    return reference, compared


def _aligned(distance_step: float = 5.0) -> pd.DataFrame:
    reference, compared = _linear_pair()
    return align_laps_by_distance(reference, compared, distance_step=distance_step)


def test_align_laps_by_distance_returns_dataframe():
    assert isinstance(_aligned(), pd.DataFrame)


def test_align_laps_by_distance_includes_distance_column():
    assert "distance" in _aligned().columns
    assert list(_aligned().columns) == EXPECTED_COLUMNS


def test_align_laps_by_distance_orders_distance():
    distance = _aligned()["distance"]

    assert distance.is_monotonic_increasing
    assert list(distance) == sorted(distance)


def test_align_laps_by_distance_has_unique_distances():
    distance = _aligned()["distance"]

    assert distance.is_unique


def test_align_laps_by_distance_stays_inside_common_range():
    reference, compared = _linear_pair()
    aligned = align_laps_by_distance(reference, compared, distance_step=5.0)
    common_start = 0.0
    common_end = 10.0

    assert aligned["distance"].min() >= common_start
    assert aligned["distance"].max() <= common_end


def test_align_laps_by_distance_default_step():
    reference, compared = _linear_pair()
    aligned = align_laps_by_distance(reference, compared)

    assert aligned["distance"].tolist() == pytest.approx([float(value) for value in range(11)])


def test_align_laps_by_distance_custom_step():
    aligned = _aligned(distance_step=2.0)

    assert aligned["distance"].tolist() == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


def test_align_laps_by_distance_speed_reference():
    aligned = _aligned()

    assert aligned["speed_reference"].tolist() == pytest.approx([0.0, 50.0, 100.0])


def test_align_laps_by_distance_speed_compared():
    aligned = _aligned()

    assert aligned["speed_compared"].tolist() == pytest.approx([10.0, 15.0, 20.0])


def test_align_laps_by_distance_speed_delta():
    aligned = _aligned()

    assert aligned["speed_delta"].tolist() == pytest.approx(
        aligned["speed_compared"] - aligned["speed_reference"]
    )
    assert aligned["speed_delta"].tolist() == pytest.approx([10.0, -35.0, -80.0])


def test_align_laps_by_distance_throttle_reference():
    aligned = _aligned()

    assert aligned["throttle_reference"].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_align_laps_by_distance_throttle_compared():
    aligned = _aligned()

    assert aligned["throttle_compared"].tolist() == pytest.approx([0.2, 0.3, 0.4])


def test_align_laps_by_distance_throttle_delta():
    aligned = _aligned()

    assert aligned["throttle_delta"].tolist() == pytest.approx(
        aligned["throttle_compared"] - aligned["throttle_reference"]
    )


def test_align_laps_by_distance_brake_reference():
    aligned = _aligned()

    assert aligned["brake_reference"].tolist() == pytest.approx([0.0, 0.25, 0.5])


def test_align_laps_by_distance_brake_compared():
    aligned = _aligned()

    assert aligned["brake_compared"].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_align_laps_by_distance_brake_delta():
    aligned = _aligned()

    assert aligned["brake_delta"].tolist() == pytest.approx(
        aligned["brake_compared"] - aligned["brake_reference"]
    )


def test_align_laps_by_distance_does_not_extrapolate():
    reference = _lap([(0.0, 0.0, 0.0, 0.0), (10.0, 100.0, 1.0, 0.0)], lap_number=1)
    compared = _lap([(2.0, 20.0, 0.2, 0.1), (12.0, 80.0, 0.8, 0.4)], lap_number=2)
    aligned = align_laps_by_distance(reference, compared, distance_step=1.0)

    assert aligned["distance"].min() == pytest.approx(2.0)
    assert aligned["distance"].max() == pytest.approx(10.0)
    assert 0.0 not in aligned["distance"].tolist()
    assert 12.0 not in aligned["distance"].tolist()


def test_align_laps_by_distance_uses_overlap_only():
    reference = _lap([(0.0, 0.0, 0.0, 0.0), (8.0, 80.0, 0.8, 0.2)], lap_number=1)
    compared = _lap([(3.0, 30.0, 0.3, 0.1), (15.0, 90.0, 0.9, 0.5)], lap_number=2)
    aligned = align_laps_by_distance(reference, compared, distance_step=1.0)

    assert aligned["distance"].min() == pytest.approx(3.0)
    assert aligned["distance"].max() == pytest.approx(8.0)


def test_align_laps_by_distance_rejects_non_positive_step():
    reference, compared = _linear_pair()

    with pytest.raises(ValueError, match="distance_step"):
        align_laps_by_distance(reference, compared, distance_step=0.0)

    with pytest.raises(ValueError, match="distance_step"):
        align_laps_by_distance(reference, compared, distance_step=-1.0)


def test_align_laps_by_distance_rejects_missing_overlap():
    reference = _lap([(0.0, 10.0, 0.2, 0.0), (5.0, 20.0, 0.4, 0.1)], lap_number=1)
    compared = _lap([(6.0, 30.0, 0.5, 0.2), (10.0, 40.0, 0.6, 0.3)], lap_number=2)

    with pytest.raises(ValueError, match="no overlapping"):
        align_laps_by_distance(reference, compared)


def test_align_laps_by_distance_rejects_empty_lap():
    empty = _lap([], lap_number=1)
    populated = _linear_pair()[0]

    with pytest.raises(ValueError, match="Reference lap has no telemetry"):
        align_laps_by_distance(empty, populated)

    with pytest.raises(ValueError, match="Compared lap has no telemetry"):
        align_laps_by_distance(populated, empty)


def test_align_laps_by_distance_rejects_insufficient_points():
    single = _lap([(0.0, 10.0, 0.2, 0.0)], lap_number=1)
    populated = _linear_pair()[0]

    with pytest.raises(ValueError, match="not have enough"):
        align_laps_by_distance(single, populated)


def test_align_laps_by_distance_does_not_modify_inputs():
    reference, compared = _linear_pair()
    reference_before = reference.data.copy()
    compared_before = compared.data.copy()

    align_laps_by_distance(reference, compared, distance_step=2.0)

    pd.testing.assert_frame_equal(reference.data, reference_before)
    pd.testing.assert_frame_equal(compared.data, compared_before)


def test_align_laps_by_distance_with_identical_sample_positions():
    reference = _lap(
        [(0.0, 100.0, 0.5, 0.0), (2.0, 120.0, 0.8, 0.2), (4.0, 110.0, 0.4, 0.0)],
        lap_number=1,
    )
    compared = _lap(
        [(0.0, 90.0, 0.4, 0.1), (2.0, 130.0, 1.0, 0.0), (4.0, 100.0, 0.2, 0.3)],
        lap_number=2,
    )
    aligned = align_laps_by_distance(reference, compared, distance_step=2.0)

    assert aligned["distance"].tolist() == pytest.approx([0.0, 2.0, 4.0])
    assert aligned["speed_reference"].tolist() == pytest.approx([100.0, 120.0, 110.0])
    assert aligned["speed_compared"].tolist() == pytest.approx([90.0, 130.0, 100.0])
    assert aligned["speed_delta"].tolist() == pytest.approx([-10.0, 10.0, -10.0])


def test_align_laps_by_distance_with_different_sample_positions():
    reference = _lap([(0.0, 0.0, 0.0, 0.0), (10.0, 100.0, 1.0, 0.0)], lap_number=1)
    compared = _lap(
        [(0.0, 0.0, 0.0, 0.0), (4.0, 40.0, 0.4, 0.4), (10.0, 100.0, 1.0, 1.0)],
        lap_number=2,
    )
    aligned = align_laps_by_distance(reference, compared, distance_step=4.0)
    row = aligned.iloc[1]

    assert row["distance"] == pytest.approx(4.0)

    assert row["speed_reference"] == pytest.approx(40.0)
    assert row["speed_compared"] == pytest.approx(40.0)
    assert row["throttle_compared"] == pytest.approx(0.4)
    assert row["brake_compared"] == pytest.approx(0.4)


def test_pipeline_aligns_sample_laps_by_distance():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)
    aligned = align_laps_by_distance(laps[1], laps[2], distance_step=1.0)

    assert isinstance(aligned, pd.DataFrame)
    assert list(aligned.columns) == EXPECTED_COLUMNS
    assert aligned["distance"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert aligned["distance"].is_monotonic_increasing
    assert aligned["distance"].max() <= 3.8
    assert not aligned.isna().any().any()
