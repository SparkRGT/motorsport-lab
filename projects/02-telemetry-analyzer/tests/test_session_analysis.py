from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.analysis.session import SessionSummary, analyze_session
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics.basic import LapMetrics
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "telemetry_laps_sample.csv"
)

REAL_DATASET = (
    Path(__file__).resolve().parents[2]
    / "01-telemetry-collector"
    / "data"
    / "telemetry"
    / "telemetry_session.csv"
)


def _write_session(
    path: Path,
    laps: list[tuple[int, float, float]],
) -> None:
    """Write two samples per lap: (lap number, duration, maximum distance)."""
    rows: list[dict[str, float]] = []
    session_time = 0.0
    frame = 1

    for lap_number, duration, distance in laps:
        for lap_distance, time_offset in ((0.0, 0.0), (distance, duration)):
            rows.append(
                {
                    "session_time": session_time + time_offset,
                    "frame": frame,
                    "lap_number": lap_number,
                    "lap_distance": lap_distance,
                    "speed": 100,
                    "throttle": 0.5,
                    "brake": 0.0,
                    "steering": 0.0,
                    "gear": 3,
                    "rpm": 8000,
                    "drs": 0,
                }
            )
            frame += 1
        session_time += duration + 0.05

    pd.DataFrame(rows).to_csv(path, index=False)


def test_analyze_session_returns_summary(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 1000.0), (2, 12.0, 1000.0)])

    summary = analyze_session(path)

    assert isinstance(summary, SessionSummary)


def test_analyze_session_counts_samples(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 1000.0), (2, 11.0, 1000.0)])

    dataset = load_csv(path)
    summary = analyze_session(path)

    assert summary.total_samples == len(dataset)


def test_analyze_session_reports_segmented_lap_numbers(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(
        path,
        [(1, 10.0, 1000.0), (2, 12.0, 1000.0), (3, 30.0, 40.0)],
    )

    laps = segment_laps(normalize_dataset(load_csv(path)))
    summary = analyze_session(path)

    assert summary.lap_numbers == tuple(laps)
    assert summary.lap_count == len(summary.lap_numbers)


def test_analyze_session_includes_metrics_for_every_segmented_lap(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(
        path,
        [(1, 10.0, 1000.0), (2, 12.0, 1000.0), (3, 30.0, 40.0)],
    )

    summary = analyze_session(path)

    assert set(summary.lap_metrics) == set(summary.lap_numbers)
    assert all(
        isinstance(metrics, LapMetrics)
        for metrics in summary.lap_metrics.values()
    )


def test_analyze_session_selects_fastest_and_slowest_complete_laps(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(
        path,
        [(1, 10.0, 1000.0), (2, 12.0, 990.0), (4, 11.0, 1000.0)],
    )

    summary = analyze_session(path)

    assert summary.fastest_lap == 1
    assert summary.slowest_lap == 2
    assert summary.fastest_lap_time == summary.lap_metrics[1].duration
    assert summary.slowest_lap_time == summary.lap_metrics[2].duration
    assert summary.fastest_lap_time == pytest.approx(10.0)
    assert summary.slowest_lap_time == pytest.approx(12.0)


def test_incomplete_lap_is_not_selected_as_slowest(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(
        path,
        [(1, 10.0, 1000.0), (2, 12.0, 1000.0), (3, 40.0, 50.0)],
    )

    summary = analyze_session(path)

    assert 3 in summary.lap_metrics
    assert summary.slowest_lap == 2
    assert summary.slowest_lap != 3
    assert summary.lap_metrics[3].duration > summary.slowest_lap_time


def test_single_complete_lap_does_not_invent_fastest_or_slowest(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 1000.0), (2, 40.0, 30.0)])

    summary = analyze_session(path)

    assert summary.lap_count == 2
    assert summary.fastest_lap is None
    assert summary.slowest_lap is None
    assert summary.fastest_lap_time is None
    assert summary.slowest_lap_time is None


def test_analyze_session_rejects_invalid_values_through_validation(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 1000.0), (2, 11.0, 1000.0)])
    dataset = pd.read_csv(path)
    dataset.loc[0, "speed"] = -1
    dataset.to_csv(path, index=False)

    with pytest.raises(ValueError, match="speed"):
        analyze_session(path)


def test_analyze_session_does_not_modify_the_dataset_file(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 1000.0), (2, 12.0, 1000.0)])
    before = load_csv(path)

    analyze_session(path)

    after = load_csv(path)
    pd.testing.assert_frame_equal(before, after)


def test_analyze_session_reads_the_synthetic_sample():
    dataset = load_csv(SAMPLE_DATASET)
    validate_dataset(dataset)

    summary = analyze_session(SAMPLE_DATASET)

    assert summary.total_samples == len(dataset)
    assert summary.lap_numbers == tuple(
        segment_laps(normalize_dataset(dataset))
    )
    assert summary.lap_count == len(summary.lap_numbers)
    assert set(summary.lap_metrics) == set(summary.lap_numbers)


def test_analyze_session_summarizes_the_real_collector_dataset():
    assert REAL_DATASET.is_file()

    dataset = load_csv(REAL_DATASET)
    untouched = dataset.copy(deep=True)
    summary = analyze_session(REAL_DATASET)

    assert isinstance(summary, SessionSummary)
    assert summary.total_samples == len(dataset)
    assert summary.lap_count == len(summary.lap_numbers)
    assert summary.lap_count >= 1
    assert set(summary.lap_metrics) == set(summary.lap_numbers)
    assert summary.fastest_lap in summary.lap_metrics
    assert summary.slowest_lap in summary.lap_metrics
    assert summary.fastest_lap_time == pytest.approx(
        summary.lap_metrics[summary.fastest_lap].duration
    )
    assert summary.slowest_lap_time == pytest.approx(
        summary.lap_metrics[summary.slowest_lap].duration
    )
    assert summary.fastest_lap_time <= summary.slowest_lap_time

    peaks = dataset.groupby("lap_number")["lap_distance"].max()
    shortest = int(peaks.idxmin())
    if peaks.loc[shortest] < peaks.max() * 0.90:
        assert summary.slowest_lap != shortest
        assert summary.fastest_lap != shortest

    pd.testing.assert_frame_equal(dataset, untouched)
