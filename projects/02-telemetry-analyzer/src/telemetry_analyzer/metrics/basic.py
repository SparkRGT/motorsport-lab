"""Basic per-lap telemetry metrics."""

from dataclasses import dataclass

import pandas as pd

from telemetry_analyzer.processing.laps import LapData

_FULL_THROTTLE_THRESHOLD = 0.99


@dataclass
class LapMetrics:
    """Basic telemetry characteristics of one lap."""

    lap_number: int
    duration: float
    average_speed: float
    maximum_speed: float
    minimum_speed: float
    average_throttle: float
    full_throttle_percentage: float
    brake_percentage: float
    maximum_rpm: float
    gear_usage: dict[int, int]


def calculate_basic_metrics(lap: LapData) -> LapMetrics:
    """
    Calculate basic telemetry metrics for one lap.

    Parameters
    ----------
    lap:
        Segmented lap produced by the processing pipeline.

    Returns
    -------
    LapMetrics
        Summary metrics. The lap and its samples are left unchanged.
    """
    samples = lap.data
    sample_count = len(samples)
    full_throttle_count = int((samples["throttle"] >= _FULL_THROTTLE_THRESHOLD).sum())
    brake_count = int((samples["brake"] > 0).sum())

    return LapMetrics(
        lap_number=lap.lap_number,
        duration=lap.duration,
        average_speed=float(samples["speed"].mean()),
        maximum_speed=float(samples["speed"].max()),
        minimum_speed=float(samples["speed"].min()),
        average_throttle=float(samples["throttle"].mean()),
        full_throttle_percentage=full_throttle_count / sample_count * 100,
        brake_percentage=brake_count / sample_count * 100,
        maximum_rpm=float(samples["rpm"].max()),
        gear_usage=_gear_usage(samples["gear"]),
    )


def _gear_usage(gears: pd.Series) -> dict[int, int]:
    counts = gears.value_counts().sort_index()
    return {int(gear): int(count) for gear, count in counts.items()}
