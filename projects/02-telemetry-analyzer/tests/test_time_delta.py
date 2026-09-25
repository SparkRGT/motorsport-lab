from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.distance import align_laps_by_distance
from telemetry_analyzer.analysis.time_delta import calculate_time_delta
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)


def _frame(
    distance: list[float],
    speed_reference: list[float],
    speed_compared: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance": distance,
            "speed_reference": speed_reference,
            "speed_compared": speed_compared,
            "throttle_reference": [0.5] * len(distance),
        }
    )


def test_calculate_time_delta_basic_interval():
    result = calculate_time_delta(_frame([0.0, 36.0], [36.0, 36.0], [36.0, 36.0]))

    assert result["time_reference"].tolist() == pytest.approx([0.0, 3.6])
    assert result["time_compared"].tolist() == pytest.approx([0.0, 3.6])


def test_calculate_time_delta_converts_kmh_to_mps():
    result = calculate_time_delta(_frame([0.0, 36.0], [72.0, 72.0], [36.0, 36.0]))

    assert result.loc[1, "time_reference"] == pytest.approx(1.8)
    assert result.loc[1, "time_compared"] == pytest.approx(3.6)


def test_calculate_time_delta_accumulates_time():
    result = calculate_time_delta(
        _frame([0.0, 36.0, 72.0], [36.0, 36.0, 36.0], [72.0, 72.0, 72.0])
    )

    assert result["cumulative_time_delta"].tolist() == pytest.approx([0.0, -1.8, -3.6])


def test_calculate_time_delta_compares_different_speeds():
    result = calculate_time_delta(_frame([0.0, 36.0], [36.0, 36.0], [72.0, 72.0]))

    assert result.loc[1, "time_reference"] == pytest.approx(3.6)
    assert result.loc[1, "time_compared"] == pytest.approx(1.8)


def test_calculate_time_delta_interval_delta():
    result = calculate_time_delta(_frame([0.0, 36.0], [36.0, 36.0], [72.0, 72.0]))

    assert result["time_delta"].tolist() == pytest.approx(
        result["time_compared"] - result["time_reference"]
    )
    assert result.loc[1, "time_delta"] == pytest.approx(-1.8)


def test_calculate_time_delta_cumulative_delta():
    result = calculate_time_delta(
        _frame([0.0, 36.0, 72.0], [36.0, 36.0, 36.0], [72.0, 72.0, 72.0])
    )

    assert result["cumulative_time_delta"].tolist() == pytest.approx(
        result["time_delta"].cumsum()
    )


def test_calculate_time_delta_starts_at_zero():
    result = calculate_time_delta(_frame([10.0, 46.0], [36.0, 36.0], [72.0, 72.0]))
    first = result.iloc[0]

    assert first["time_reference"] == pytest.approx(0.0)
    assert first["time_compared"] == pytest.approx(0.0)
    assert first["time_delta"] == pytest.approx(0.0)
    assert first["cumulative_time_delta"] == pytest.approx(0.0)


def test_calculate_time_delta_single_point():
    result = calculate_time_delta(_frame([0.0], [0.0], [120.0]))

    assert result["time_reference"].tolist() == pytest.approx([0.0])
    assert result["time_compared"].tolist() == pytest.approx([0.0])
    assert result["time_delta"].tolist() == pytest.approx([0.0])
    assert result["cumulative_time_delta"].tolist() == pytest.approx([0.0])


def test_calculate_time_delta_rejects_empty_dataframe():
    empty = _frame([], [], []).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        calculate_time_delta(empty)


def test_calculate_time_delta_rejects_missing_columns():
    data = _frame([0.0, 36.0], [36.0, 36.0], [36.0, 36.0]).drop(columns=["speed_compared"])

    with pytest.raises(ValueError, match="speed_compared"):
        calculate_time_delta(data)


def test_calculate_time_delta_rejects_non_dataframe():
    with pytest.raises(TypeError):
        calculate_time_delta([0.0, 36.0])


def test_calculate_time_delta_rejects_decreasing_distance():
    data = _frame([0.0, 36.0, 10.0], [36.0, 36.0, 36.0], [36.0, 36.0, 36.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        calculate_time_delta(data)


def test_calculate_time_delta_rejects_duplicate_distance():
    data = _frame([0.0, 36.0, 36.0], [36.0, 36.0, 36.0], [36.0, 36.0, 36.0])

    with pytest.raises(ValueError, match="duplicate"):
        calculate_time_delta(data)


def test_calculate_time_delta_rejects_negative_speed():
    data = _frame([0.0, 36.0], [36.0, -1.0], [36.0, 36.0])

    with pytest.raises(ValueError, match="speed_reference"):
        calculate_time_delta(data)


def test_calculate_time_delta_rejects_zero_speed_interval():
    data = _frame([0.0, 36.0], [36.0, 0.0], [36.0, 36.0])

    with pytest.raises(ValueError, match="greater than 0"):
        calculate_time_delta(data)


def test_calculate_time_delta_rejects_non_numeric_distance():
    data = _frame([0.0, 36.0], [36.0, 36.0], [36.0, 36.0])
    data["distance"] = ["start", "end"]

    with pytest.raises(ValueError, match="numeric"):
        calculate_time_delta(data)


def test_calculate_time_delta_does_not_modify_original():
    data = _frame([0.0, 36.0], [36.0, 36.0], [72.0, 72.0])
    original = data.copy()

    calculate_time_delta(data)

    pd.testing.assert_frame_equal(data, original)


def test_calculate_time_delta_keeps_original_columns():
    data = _frame([0.0, 36.0], [36.0, 36.0], [72.0, 72.0])
    result = calculate_time_delta(data)

    assert list(result.columns[: len(data.columns)]) == list(data.columns)
    assert "time_reference" in result.columns
    assert "time_compared" in result.columns
    assert "time_delta" in result.columns
    assert "cumulative_time_delta" in result.columns


def test_pipeline_calculates_time_delta_from_aligned_laps():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)
    aligned = align_laps_by_distance(laps[1], laps[2], distance_step=1.0)
    result = calculate_time_delta(aligned)

    assert list(result.columns[: len(aligned.columns)]) == list(aligned.columns)
    assert {"time_reference", "time_compared", "time_delta", "cumulative_time_delta"} <= set(
        result.columns
    )
    assert result.iloc[0]["time_reference"] == pytest.approx(0.0)
    assert result.iloc[0]["time_compared"] == pytest.approx(0.0)
    assert result.iloc[0]["cumulative_time_delta"] == pytest.approx(0.0)
    assert result["time_delta"].tolist() == pytest.approx(
        (result["time_compared"] - result["time_reference"]).tolist()
    )
    assert result["cumulative_time_delta"].tolist() == pytest.approx(
        result["time_delta"].cumsum().tolist()
    )
    assert result["time_reference"].iloc[1] > 0
    assert result["time_compared"].iloc[1] > 0
