from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import REQUIRED_COLUMNS, validate_dataset


SAMPLE_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_sample.csv"
)

FLOAT_COLUMNS = [
    "session_time",
    "lap_distance",
    "speed",
    "throttle",
    "brake",
    "steering",
]

INTEGER_COLUMNS = [
    "frame",
    "lap_number",
    "gear",
    "rpm",
    "drs",
]


def _sample_dataset() -> pd.DataFrame:
    return load_csv(SAMPLE_DATASET)


def test_normalize_dataset_returns_dataframe():
    normalized = normalize_dataset(_sample_dataset())

    assert isinstance(normalized, pd.DataFrame)


def test_normalize_dataset_returns_a_copy():
    frame = _sample_dataset()
    normalized = normalize_dataset(frame)

    assert normalized is not frame


def test_normalize_dataset_does_not_modify_original():
    frame = _sample_dataset()
    original = frame.copy()

    normalize_dataset(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_normalize_dataset_converts_expected_dtypes():
    normalized = normalize_dataset(_sample_dataset())

    for column in FLOAT_COLUMNS:
        assert pd.api.types.is_float_dtype(normalized[column])

    for column in INTEGER_COLUMNS:
        assert pd.api.types.is_integer_dtype(normalized[column])


def test_normalize_dataset_keeps_rows_and_columns():
    frame = _sample_dataset()
    normalized = normalize_dataset(frame)

    assert len(normalized) == len(frame)
    assert list(normalized.columns) == list(frame.columns)
    assert list(normalized.columns) == REQUIRED_COLUMNS


def test_normalize_dataset_keeps_values():
    frame = _sample_dataset()
    normalized = normalize_dataset(frame)

    pd.testing.assert_frame_equal(normalized, frame, check_dtype=False)


def test_normalize_dataset_preserves_missing_values():
    frame = _sample_dataset()
    frame["gear"] = frame["gear"].astype("float64")
    frame.loc[2, "gear"] = float("nan")
    frame.loc[3, "throttle"] = float("nan")

    normalized = normalize_dataset(frame)

    assert len(normalized) == len(frame)
    assert pd.isna(normalized.loc[2, "gear"])
    assert pd.isna(normalized.loc[3, "throttle"])
    assert int(normalized["gear"].isna().sum()) == 1
    assert pd.api.types.is_integer_dtype(normalized["gear"])
    assert pd.api.types.is_float_dtype(normalized["throttle"])


def test_normalize_dataset_rejects_fractional_integer_values():
    frame = _sample_dataset()
    frame["gear"] = frame["gear"].astype("float64")
    frame.loc[0, "gear"] = 3.5

    with pytest.raises(ValueError, match="non-integer"):
        normalize_dataset(frame)


def test_sample_dataset_pipeline():
    frame = _sample_dataset()

    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    validate_dataset(normalized)

    assert normalized is not frame
