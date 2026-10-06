"""Summarize one telemetry session using the existing processing pipeline."""

from dataclasses import dataclass
from pathlib import Path

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.metrics.basic import LapMetrics, calculate_basic_metrics
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


# A lap is complete enough for fastest/slowest when its furthest sample
# reaches this fraction of the longest lap in the same file.
_COMPLETE_DISTANCE_RATIO = 0.90


@dataclass
class SessionSummary:
    """Session totals and the basic metrics already calculated per lap."""

    total_samples: int
    lap_count: int
    lap_numbers: tuple[int, ...]
    lap_metrics: dict[int, LapMetrics]
    fastest_lap: int | None
    slowest_lap: int | None
    fastest_lap_time: float | None
    slowest_lap_time: float | None


def analyze_session(path: str | Path) -> SessionSummary:
    """
    Load one CSV session and summarize the laps it contains.

    Parameters
    ----------
    path:
        Path to a telemetry CSV with the analyzer columns.

    Returns
    -------
    SessionSummary
        Sample count, every segmented lap, and fastest/slowest among the
        laps that cover most of the longest distance in the file.

    Notes
    -----
    Segmentation stays neutral: a short lap is still reported. Fastest and
    slowest use only laps whose maximum ``lap_distance`` is at least 90% of
    the longest lap in this session. The reference is that session, not a
    fixed track length. If the longest lap is itself incomplete, shorter laps
    are judged against that partial distance. Fewer than two qualifying laps
    leaves fastest and slowest empty.
    """
    dataset = load_csv(path)
    validate_dataset(dataset)
    normalized = normalize_dataset(dataset)
    laps = segment_laps(normalized)
    metrics = {
        lap_number: calculate_basic_metrics(lap)
        for lap_number, lap in laps.items()
    }
    complete = _complete_lap_numbers(laps)
    fastest, slowest = _fastest_and_slowest(complete, metrics)
    lap_numbers = tuple(laps)

    return SessionSummary(
        total_samples=len(dataset),
        lap_count=len(lap_numbers),
        lap_numbers=lap_numbers,
        lap_metrics=metrics,
        fastest_lap=None if fastest is None else fastest.lap_number,
        slowest_lap=None if slowest is None else slowest.lap_number,
        fastest_lap_time=None if fastest is None else fastest.duration,
        slowest_lap_time=None if slowest is None else slowest.duration,
    )


def _complete_lap_numbers(laps: dict[int, LapData]) -> list[int]:
    peaks = {
        lap_number: float(lap.data["lap_distance"].max())
        for lap_number, lap in laps.items()
    }
    if not peaks:
        return []

    reference = max(peaks.values())
    if reference <= 0:
        return []

    threshold = reference * _COMPLETE_DISTANCE_RATIO
    return [
        lap_number
        for lap_number, peak in peaks.items()
        if peak >= threshold
    ]


def _fastest_and_slowest(
    complete: list[int],
    metrics: dict[int, LapMetrics],
) -> tuple[LapMetrics | None, LapMetrics | None]:
    if len(complete) < 2:
        return None, None

    fastest = min(
        complete,
        key=lambda lap_number: (metrics[lap_number].duration, lap_number),
    )
    slowest = max(
        complete,
        key=lambda lap_number: (metrics[lap_number].duration, lap_number),
    )
    return metrics[fastest], metrics[slowest]
