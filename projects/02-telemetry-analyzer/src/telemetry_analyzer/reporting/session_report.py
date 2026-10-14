"""Build a session report from the existing analysis functions and export it."""

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telemetry_analyzer.analysis.driver_metrics import (
    DriverPerformanceMetrics,
    calculate_driver_performance_metrics,
)
from telemetry_analyzer.analysis.event_performance import (
    PerformanceEvent,
    analyze_event_performance,
)
from telemetry_analyzer.analysis.lap_performance import analyze_lap_performance
from telemetry_analyzer.analysis.performance import PerformanceSegment
from telemetry_analyzer.analysis.session import SessionSummary, analyze_session
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics.comparison import LapComparison, compare_laps
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset
from telemetry_analyzer.visualization.telemetry import (
    plot_brake_comparison,
    plot_speed_comparison,
    plot_speed_delta,
    plot_throttle_comparison,
    save_figure,
)

_NOT_REQUESTED = "Lap comparison was not requested."
_NO_OVERLAP = (
    "Lap comparison was not generated: the selected laps have no overlapping lap_distance."
)

_PLOT_BUILDERS = (
    ("speed_comparison.png", plot_speed_comparison),
    ("speed_delta.png", plot_speed_delta),
    ("throttle_comparison.png", plot_throttle_comparison),
    ("brake_comparison.png", plot_brake_comparison),
)


@dataclass
class LapComparisonReport:
    """Comparison produced from one existing lap-performance analysis."""

    reference_lap: int
    compared_lap: int
    lap_comparison: LapComparison
    total_time_delta: float
    performance_segments: list[PerformanceSegment]
    events: list[PerformanceEvent]
    driver_metrics: DriverPerformanceMetrics


@dataclass
class SessionReport:
    """Session summary and, when requested and possible, one lap comparison."""

    session: SessionSummary
    comparison: LapComparisonReport | None
    comparison_note: str | None
    plots: tuple[str, ...]


def generate_session_report(
    csv_path: str | Path,
    reference_lap: int | None = None,
    compared_lap: int | None = None,
    output_dir: str | Path | None = None,
) -> SessionReport:
    """
    Analyze one telemetry CSV and optionally export the report.

    Parameters
    ----------
    csv_path:
        Path to a telemetry CSV with the analyzer columns.
    reference_lap:
        Baseline lap number. Must be given together with ``compared_lap``.
    compared_lap:
        Lap number subtracted from the reference. Must be given together
        with ``reference_lap``.
    output_dir:
        Directory for ``session_report.json`` and, when a comparison exists,
        the PNG files under ``plots/``. Omitted results are not written.

    Returns
    -------
    SessionReport
        Structured session summary. ``comparison`` is present only when both
        lap numbers were requested and the laps share a distance range.
        ``comparison_note`` explains a missing comparison. ``plots`` contains
        paths relative to ``output_dir``.

    Notes
    -----
    The session summary comes from ``analyze_session``. Its complete-lap rule
    is unchanged. Basic lap metrics are taken from that summary.

    A requested comparison loads the lap samples once, builds one
    ``LapPerformanceAnalysis``, and passes that same object to the event and
    driver-metric functions. When the laps do not overlap, the summary is kept
    and the comparison is left empty with an explicit note. Other comparison
    failures propagate.

    The JSON report does not include the aligned distance table. Plot files
    use a copy of that table. There is no command-line entry point.
    """
    _require_lap_selection(reference_lap, compared_lap)
    destination = _require_output_directory(output_dir)
    summary = analyze_session(csv_path)
    comparison, aligned, note = _comparison(
        csv_path,
        summary,
        reference_lap,
        compared_lap,
    )
    plots: tuple[str, ...] = ()
    if destination is not None and aligned is not None:
        plots = _export_plots(aligned, destination)

    report = SessionReport(
        session=summary,
        comparison=comparison,
        comparison_note=note,
        plots=plots,
    )
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "session_report.json").write_text(
            session_report_to_json(report),
            encoding="utf-8",
        )
    return report


def session_report_to_json(report: SessionReport) -> str:
    """
    Serialize a session report as strict, readable JSON.

    Parameters
    ----------
    report:
        Report returned by ``generate_session_report``.

    Returns
    -------
    str
        JSON text. Object keys are sorted. Missing values are JSON null.
        Non-finite numbers and DataFrames are rejected.
    """
    return json.dumps(
        _json_value(report),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _require_lap_selection(reference_lap: int | None, compared_lap: int | None) -> None:
    if reference_lap is None and compared_lap is None:
        return
    if reference_lap is None or compared_lap is None:
        raise ValueError("Both reference_lap and compared_lap are required to compare laps")
    _require_lap_number(reference_lap, "reference_lap")
    _require_lap_number(compared_lap, "compared_lap")


def _require_lap_number(lap_number: int, name: str) -> None:
    if isinstance(lap_number, bool) or not isinstance(lap_number, int):
        raise TypeError(f"{name} must be an integer lap number, got {lap_number!r}")


def _require_output_directory(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    path = Path(output_dir)
    if path.exists() and not path.is_dir():
        raise ValueError(f"output_dir must be a directory, got a file: {path}")
    return path


def _comparison(
    csv_path: str | Path,
    summary: SessionSummary,
    reference_lap: int | None,
    compared_lap: int | None,
) -> tuple[LapComparisonReport | None, pd.DataFrame | None, str | None]:
    if reference_lap is None or compared_lap is None:
        return None, None, _NOT_REQUESTED

    missing = [
        lap_number
        for lap_number in (reference_lap, compared_lap)
        if lap_number not in summary.lap_numbers
    ]
    if missing:
        raise ValueError(f"Selected laps are not in the session: {missing}")

    dataset = load_csv(csv_path)
    validate_dataset(dataset)
    laps = segment_laps(normalize_dataset(dataset))
    reference = laps[reference_lap]
    compared = laps[compared_lap]
    try:
        performance = analyze_lap_performance(reference, compared)
    except ValueError as error:
        if "no overlapping" not in str(error):
            raise
        return None, None, _NO_OVERLAP

    return (
        LapComparisonReport(
            reference_lap=reference_lap,
            compared_lap=compared_lap,
            lap_comparison=compare_laps(
                summary.lap_metrics[reference_lap],
                summary.lap_metrics[compared_lap],
            ),
            total_time_delta=performance.total_time_delta,
            performance_segments=list(performance.performance_segments),
            events=list(analyze_event_performance(performance, reference, compared)),
            driver_metrics=calculate_driver_performance_metrics(
                performance,
                reference,
                compared,
            ),
        ),
        performance.aligned_data,
        None,
    )


def _export_plots(aligned: pd.DataFrame, output_dir: Path) -> tuple[str, ...]:
    matplotlib.use("Agg", force=True)
    plotting = aligned.copy(deep=True)
    relative_paths: list[str] = []
    for filename, plot in _PLOT_BUILDERS:
        figure = plot(plotting)
        try:
            save_figure(figure, output_dir / "plots" / filename)
        finally:
            plt.close(figure)
        relative_paths.append(f"plots/{filename}")
    return tuple(relative_paths)


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _finite_number(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        if value.is_absolute():
            raise ValueError("Session report paths must be relative")
        return value.as_posix()
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_number(float(value))
    if value is pd.NA or value is pd.NaT:
        raise ValueError("Missing values cannot be written to the session report")
    if isinstance(value, pd.DataFrame):
        raise TypeError("DataFrames are not included in the session report JSON")
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {_json_key(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported session report value: {type(value).__name__}")


def _json_key(key) -> str:
    if isinstance(key, str):
        if Path(key).is_absolute():
            raise ValueError("Session report paths must be relative")
        return key
    if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
        raise TypeError(f"Unsupported session report key: {type(key).__name__}")
    return str(int(key))


def _finite_number(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite values cannot be written to the session report: {value}")
    return value
