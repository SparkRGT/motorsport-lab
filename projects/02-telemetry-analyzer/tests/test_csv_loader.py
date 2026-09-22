from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.ingestion.csv_loader import load_csv


SAMPLE_DATASET = (
    Path(__file__).parent.parent
    / "data"
    / "sample"
    / "telemetry_sample.csv"
)

EXPECTED_COLUMNS = [
    "session_time",
    "frame",
    "lap_number",
    "lap_distance",
    "speed",
    "throttle",
    "brake",
    "steering",
    "gear",
    "rpm",
    "drs",
]


def test_load_csv_returns_dataframe():
    frame = load_csv(SAMPLE_DATASET)

    assert isinstance(frame, pd.DataFrame)


def test_load_csv_loads_expected_rows():
    frame = load_csv(SAMPLE_DATASET)

    assert len(frame) == 10


def test_load_csv_loads_expected_columns():
    frame = load_csv(SAMPLE_DATASET)

    assert list(frame.columns) == EXPECTED_COLUMNS


def test_load_csv_missing_file():
    missing = SAMPLE_DATASET.parent / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing)
