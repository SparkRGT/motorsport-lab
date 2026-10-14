import json
import math
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from telemetry_analyzer.analysis.driver_metrics import calculate_driver_performance_metrics
from telemetry_analyzer.analysis.event_performance import analyze_event_performance
from telemetry_analyzer.analysis.lap_performance import analyze_lap_performance
from telemetry_analyzer.analysis.session import analyze_session
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics.basic import calculate_basic_metrics
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset
from telemetry_analyzer.reporting import (
    SessionReport,
    generate_session_report,
    session_report_to_json,
)
from telemetry_analyzer.reporting.session_report import _json_value


SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "telemetry_laps_sample.csv"
)

REAL_DATASET = (
    Path(__file__).resolve().parents[2]
    / "01-telemetry-collector"
    / "data"
    / "telemetry"
    / "telemetry_session.csv"
)


class _Source(Enum):
    COLLECTOR = "collector"


def _write_session(path: Path, laps: list[tuple[int, float, float]]) -> None:
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


def _write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(session_time: float, frame: int, lap_number: int, lap_distance: float) -> dict[str, float]:
    return {
        "session_time": session_time,
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


def test_generate_session_report_on_synthetic_sample(tmp_path):
    output = tmp_path / "report"

    report = generate_session_report(
        SAMPLE_DATASET,
        reference_lap=2,
        compared_lap=1,
        output_dir=output,
    )

    assert isinstance(report, SessionReport)
    assert report.session.lap_numbers == (1, 2, 3)
    assert report.comparison is not None
    assert report.comparison.reference_lap == 2
    assert report.comparison.compared_lap == 1
    assert report.comparison.lap_comparison.time_delta == pytest.approx(
        report.session.lap_metrics[1].duration - report.session.lap_metrics[2].duration
    )
    assert report.comparison_note is None
    assert output.joinpath("session_report.json").is_file()


def test_session_report_without_comparison(tmp_path):
    path = tmp_path / "session.csv"
    output = tmp_path / "out"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])

    report = generate_session_report(path, output_dir=output)

    assert report.comparison is None
    assert report.comparison_note == "Lap comparison was not requested."
    assert report.plots == ()
    assert report.session.lap_count == 2
    assert not output.joinpath("plots").exists()


def test_session_report_with_two_selected_laps(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])

    report = generate_session_report(path, reference_lap=1, compared_lap=2)

    assert report.comparison is not None
    assert report.comparison.reference_lap == 1
    assert report.comparison.compared_lap == 2
    assert report.comparison.lap_comparison.time_delta == pytest.approx(2.0)
    assert report.comparison.total_time_delta == pytest.approx(0.0)
    assert isinstance(report.comparison.performance_segments, list)
    assert isinstance(report.comparison.events, list)
    assert isinstance(report.comparison.driver_metrics.braking, list)
    assert report.comparison_note is None


def test_missing_lap_raises(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])

    with pytest.raises(ValueError, match="not in the session"):
        generate_session_report(path, reference_lap=1, compared_lap=9)


def test_one_selected_lap_raises(tmp_path):
    path = tmp_path / "session.csv"
    output = tmp_path / "out"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])

    with pytest.raises(ValueError, match="Both reference_lap and compared_lap"):
        generate_session_report(path, reference_lap=1, output_dir=output)

    assert not output.exists()


def test_json_export_is_valid_and_readable(tmp_path):
    path = tmp_path / "session.csv"
    output = tmp_path / "nested" / "out"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])

    generate_session_report(path, reference_lap=1, compared_lap=2, output_dir=output)

    text = output.joinpath("session_report.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["session"]["lap_count"] == 2
    assert payload["session"]["lap_metrics"]["1"]["gear_usage"] == {"3": 2}
    assert payload["comparison"]["reference_lap"] == 1
    assert "total_time_delta" in payload["comparison"]
    assert "\n" in text


def test_serializer_handles_dataclasses_numpy_pandas_and_enum():
    path_value = _json_value(Path("plots/speed_comparison.png"))
    number = _json_value(pd.Series([1.5], dtype="float64").iloc[0])
    count = _json_value(np.int64(4))
    label = _json_value(_Source.COLLECTOR)

    assert path_value == "plots/speed_comparison.png"
    assert number == pytest.approx(1.5)
    assert isinstance(number, float)
    assert count == 4
    assert isinstance(count, int)
    assert label == "collector"

    report = generate_session_report(SAMPLE_DATASET)
    report.session.lap_metrics[1].average_speed = np.float64(
        report.session.lap_metrics[1].average_speed
    )
    report.session.lap_metrics[1].gear_usage = {np.int64(4): np.int64(1)}
    payload = json.loads(session_report_to_json(report))
    assert isinstance(payload["session"]["lap_metrics"]["1"]["average_speed"], float)
    assert payload["session"]["lap_metrics"]["1"]["gear_usage"] == {"4": 1}


def test_serializer_rejects_non_finite_values_and_dataframes(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])
    report = generate_session_report(path, reference_lap=1, compared_lap=2)
    report.comparison.total_time_delta = math.nan

    with pytest.raises(ValueError, match="Non-finite"):
        session_report_to_json(report)

    report.comparison.total_time_delta = 0.0
    report.comparison.events = pd.DataFrame({"distance": [1.0]})
    with pytest.raises(TypeError, match="DataFrame"):
        session_report_to_json(report)


def test_output_directory_and_plots_are_created_for_a_comparison(tmp_path):
    path = tmp_path / "session.csv"
    output = tmp_path / "fresh" / "report"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 80.0)])
    figures_before = plt.get_fignums()

    report = generate_session_report(path, reference_lap=1, compared_lap=2, output_dir=output)

    assert output.is_dir()
    assert report.plots == (
        "plots/speed_comparison.png",
        "plots/speed_delta.png",
        "plots/throttle_comparison.png",
        "plots/brake_comparison.png",
    )
    for relative in report.plots:
        assert (output / relative).is_file()
        assert Path(relative).is_absolute() is False
    assert plt.get_fignums() == figures_before


def test_json_plot_paths_are_relative_and_not_absolute(tmp_path):
    path = tmp_path / "session.csv"
    output = tmp_path / "report"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])

    generate_session_report(path, reference_lap=1, compared_lap=2, output_dir=output)

    text = output.joinpath("session_report.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["plots"] == [
        "plots/speed_comparison.png",
        "plots/speed_delta.png",
        "plots/throttle_comparison.png",
        "plots/brake_comparison.png",
    ]
    assert str(output.resolve()) not in text
    assert ":\\" not in text
    with pytest.raises(ValueError, match="relative"):
        _json_value(output.resolve())


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        generate_session_report(tmp_path / "missing.csv")


def test_invalid_dataset_raises(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("session_time,frame\n0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        generate_session_report(path)


def test_output_directory_rejects_a_file(tmp_path):
    path = tmp_path / "session.csv"
    output = tmp_path / "not-a-directory"
    _write_session(path, [(1, 10.0, 100.0)])
    output.write_text("reserved", encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir must be a directory"):
        generate_session_report(path, output_dir=output)


def test_laps_without_overlap_keep_the_session_summary(tmp_path):
    path = tmp_path / "session.csv"
    output = tmp_path / "out"
    _write_rows(
        path,
        [
            _row(0.0, 1, 1, 0.0),
            _row(1.0, 2, 1, 10.0),
            _row(2.0, 3, 2, 100.0),
            _row(3.0, 4, 2, 110.0),
        ],
    )

    report = generate_session_report(path, reference_lap=1, compared_lap=2, output_dir=output)

    assert report.session.lap_count == 2
    assert report.comparison is None
    assert "no overlapping" in report.comparison_note
    assert report.plots == ()
    assert not output.joinpath("plots").exists()
    payload = json.loads(output.joinpath("session_report.json").read_text(encoding="utf-8"))
    assert payload["comparison"] is None
    assert payload["session"]["total_samples"] == 4


def test_insufficient_distance_samples_raise(tmp_path):
    path = tmp_path / "session.csv"
    _write_rows(
        path,
        [
            _row(0.0, 1, 1, 0.0),
            _row(1.0, 2, 1, 10.0),
            _row(2.0, 3, 2, 0.0),
        ],
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        generate_session_report(path, reference_lap=1, compared_lap=2)


def test_report_does_not_modify_the_csv_or_loaded_frame(tmp_path):
    before = SAMPLE_DATASET.read_bytes()
    original = load_csv(SAMPLE_DATASET)
    snapshot = original.copy(deep=True)

    generate_session_report(
        SAMPLE_DATASET,
        reference_lap=2,
        compared_lap=1,
        output_dir=tmp_path / "out",
    )

    pd.testing.assert_frame_equal(original, snapshot)
    assert SAMPLE_DATASET.read_bytes() == before


def test_comparison_reuses_one_lap_performance_analysis(tmp_path):
    path = tmp_path / "session.csv"
    _write_session(path, [(1, 10.0, 100.0), (2, 12.0, 100.0)])
    seen: dict[str, object] = {}

    def performance_wrapper(*args, **kwargs):
        result = analyze_lap_performance(*args, **kwargs)
        seen["performance"] = result
        return result

    def event_wrapper(performance, *args, **kwargs):
        seen["event"] = performance
        return analyze_event_performance(performance, *args, **kwargs)

    def driver_wrapper(performance, *args, **kwargs):
        seen["driver"] = performance
        return calculate_driver_performance_metrics(performance, *args, **kwargs)

    with (
        patch(
            "telemetry_analyzer.analysis.session.calculate_basic_metrics",
            wraps=calculate_basic_metrics,
        ) as basic_metrics,
        patch(
            "telemetry_analyzer.reporting.session_report.analyze_lap_performance",
            side_effect=performance_wrapper,
        ) as performance,
        patch(
            "telemetry_analyzer.reporting.session_report.analyze_event_performance",
            side_effect=event_wrapper,
        ),
        patch(
            "telemetry_analyzer.reporting.session_report.calculate_driver_performance_metrics",
            side_effect=driver_wrapper,
        ),
        patch(
            "telemetry_analyzer.analysis.zones.detect_braking_zones",
        ) as braking_zones,
    ):
        generate_session_report(path, reference_lap=1, compared_lap=2)

    assert basic_metrics.call_count == 2
    assert performance.call_count == 1
    assert seen["event"] is seen["performance"]
    assert seen["driver"] is seen["performance"]
    assert braking_zones.call_count == 0


def test_real_collector_csv_produces_a_session_report(tmp_path):
    if not REAL_DATASET.is_file():
        pytest.skip("Real collector CSV is not available in this environment")

    untouched = REAL_DATASET.read_bytes()
    summary = generate_session_report(REAL_DATASET, output_dir=tmp_path / "summary")
    dataset = load_csv(REAL_DATASET)
    validate_dataset(dataset)
    reference, compared = _first_overlapping_pair(
        list(segment_laps(normalize_dataset(dataset)).values())
    )

    report = generate_session_report(
        REAL_DATASET,
        reference_lap=reference.lap_number,
        compared_lap=compared.lap_number,
        output_dir=tmp_path / "comparison",
    )

    assert summary.comparison is None
    assert summary.session.total_samples == len(dataset)
    assert summary.session.lap_count == len(summary.session.lap_numbers)
    assert report.comparison is not None
    assert math.isfinite(report.comparison.total_time_delta)
    assert report.comparison.lap_comparison.time_delta == pytest.approx(
        report.session.lap_metrics[compared.lap_number].duration
        - report.session.lap_metrics[reference.lap_number].duration
    )
    assert (tmp_path / "comparison" / "session_report.json").is_file()
    assert (tmp_path / "comparison" / "plots" / "speed_comparison.png").is_file()
    text = (tmp_path / "comparison" / "session_report.json").read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in text
    assert REAL_DATASET.read_bytes() == untouched


def _first_overlapping_pair(laps: list[LapData]) -> tuple[LapData, LapData]:
    for index, reference in enumerate(laps):
        reference_start = float(reference.data["lap_distance"].iloc[0])
        reference_end = float(reference.data["lap_distance"].iloc[-1])
        for compared in laps[index + 1 :]:
            compared_start = float(compared.data["lap_distance"].iloc[0])
            compared_end = float(compared.data["lap_distance"].iloc[-1])
            if max(reference_start, compared_start) < min(reference_end, compared_end):
                return reference, compared

    raise AssertionError("The dataset has no overlapping lap pair")
