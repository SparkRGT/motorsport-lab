from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.validation import validate_dataset


SAMPLE_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_sample.csv"
)


def _valid_dataset() -> pd.DataFrame:
    return load_csv(SAMPLE_DATASET)


def test_validate_dataset_accepts_sample():
    assert validate_dataset(_valid_dataset()) is None


def test_validate_dataset_rejects_empty_dataframe():
    empty = _valid_dataset().iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        validate_dataset(empty)


def test_validate_dataset_rejects_missing_column():
    frame = _valid_dataset().drop(columns=["throttle"])

    with pytest.raises(ValueError, match="throttle"):
        validate_dataset(frame)


def test_validate_dataset_rejects_missing_values():
    frame = _valid_dataset()
    frame.loc[1, "rpm"] = float("nan")

    with pytest.raises(ValueError, match="rpm at rows"):
        validate_dataset(frame)


def test_validate_dataset_rejects_non_numeric_values():
    frame = _valid_dataset()
    frame["gear"] = "third"

    with pytest.raises(ValueError, match="numeric"):
        validate_dataset(frame)


def test_validate_dataset_rejects_throttle_out_of_range():
    frame = _valid_dataset()
    frame.loc[0, "throttle"] = 1.5

    with pytest.raises(ValueError, match="throttle"):
        validate_dataset(frame)


def test_validate_dataset_rejects_brake_out_of_range():
    frame = _valid_dataset()
    frame.loc[0, "brake"] = -0.1

    with pytest.raises(ValueError, match="brake"):
        validate_dataset(frame)


def test_validate_dataset_rejects_negative_session_time():
    frame = _valid_dataset()
    frame.loc[0, "session_time"] = -0.1

    with pytest.raises(ValueError, match="session_time must be >= 0"):
        validate_dataset(frame)


def test_validate_dataset_rejects_negative_lap_distance():
    frame = _valid_dataset()
    frame.loc[0, "lap_distance"] = -1.0

    with pytest.raises(ValueError, match="lap_distance"):
        validate_dataset(frame)


def test_validate_dataset_rejects_negative_speed():
    frame = _valid_dataset()
    frame.loc[0, "speed"] = -1

    with pytest.raises(ValueError, match="speed"):
        validate_dataset(frame)


@pytest.mark.parametrize("column", ["lap_number", "gear", "rpm"])
def test_validate_dataset_rejects_negative_non_negative_columns(column: str):
    frame = _valid_dataset()
    frame.loc[0, column] = -1

    with pytest.raises(ValueError, match=column):
        validate_dataset(frame)


def test_validate_dataset_rejects_invalid_drs():
    frame = _valid_dataset()
    frame.loc[0, "drs"] = 2

    with pytest.raises(ValueError, match="drs"):
        validate_dataset(frame)


def test_validate_dataset_rejects_unordered_session_time():
    frame = _valid_dataset().iloc[:4].copy()
    frame["session_time"] = [0.00, 0.05, 0.03, 0.10]

    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_dataset(frame)


def test_validate_dataset_accepts_repeated_session_time():
    frame = _valid_dataset().iloc[:4].copy()
    frame["session_time"] = [0.00, 0.05, 0.05, 0.10]

    assert validate_dataset(frame) is None


def test_validate_dataset_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_dataset([0.0, 0.05, 0.10])
