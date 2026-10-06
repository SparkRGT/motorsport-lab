import math
from pathlib import Path

import pandas as pd

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics.basic import LapMetrics, calculate_basic_metrics
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import (
    REQUIRED_COLUMNS,
    validate_dataset,
)


REAL_DATASET = (
    Path(__file__).resolve().parents[2]
    / "01-telemetry-collector"
    / "data"
    / "telemetry"
    / "telemetry_session.csv"
)

_METRIC_VALUES = (
    "duration",
    "average_speed",
    "maximum_speed",
    "minimum_speed",
    "average_throttle",
    "full_throttle_percentage",
    "brake_percentage",
    "maximum_rpm",
)


def test_real_f1_dataset_runs_through_the_existing_pipeline():
    assert REAL_DATASET.is_file()

    original = load_csv(REAL_DATASET)
    untouched = original.copy(deep=True)

    assert isinstance(original, pd.DataFrame)
    assert not original.empty
    assert list(original.columns) == REQUIRED_COLUMNS

    validate_dataset(original)

    normalized = normalize_dataset(original)

    assert isinstance(normalized, pd.DataFrame)
    assert len(normalized) == len(original)
    assert list(normalized.columns) == REQUIRED_COLUMNS

    laps = segment_laps(normalized)

    assert laps
    assert any(
        lap.sample_count > 1 and lap.duration > 0
        for lap in laps.values()
    )

    for lap in laps.values():
        assert lap.lap_number >= 0
        assert lap.duration >= 0
        assert lap.sample_count > 0

        metrics = calculate_basic_metrics(lap)

        assert isinstance(metrics, LapMetrics)
        assert metrics.lap_number == lap.lap_number
        assert metrics.duration == lap.duration

        for name in _METRIC_VALUES:
            value = getattr(metrics, name)
            assert isinstance(value, float)
            assert math.isfinite(value)

        assert metrics.gear_usage
        assert all(
            isinstance(gear, int) and count > 0
            for gear, count in metrics.gear_usage.items()
        )

    pd.testing.assert_frame_equal(original, untouched)
