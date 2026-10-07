"""Compare two laps by composing distance alignment, time delta, and segments."""

from dataclasses import dataclass

import pandas as pd

from telemetry_analyzer.analysis.distance import align_laps_by_distance
from telemetry_analyzer.analysis.performance import (
    PerformanceSegment,
    detect_performance_segments,
)
from telemetry_analyzer.analysis.time_delta import calculate_time_delta
from telemetry_analyzer.processing.laps import LapData


@dataclass
class LapPerformanceAnalysis:
    """Where the time difference between two laps changes along the distance."""

    reference_lap: int
    compared_lap: int
    total_time_delta: float
    aligned_data: pd.DataFrame
    performance_segments: list[PerformanceSegment]


def analyze_lap_performance(
    reference: LapData,
    compared: LapData,
    distance_step: float = 1.0,
    min_time_delta_change: float = 0.10,
) -> LapPerformanceAnalysis:
    """
    Compare two laps on their common distance.

    Parameters
    ----------
    reference:
        Baseline lap.
    compared:
        Lap subtracted from the reference. A negative total means this lap
        took less time than ``reference``.
    distance_step:
        Spacing passed to ``align_laps_by_distance``.
    min_time_delta_change:
        Threshold passed to ``detect_performance_segments``.

    Returns
    -------
    LapPerformanceAnalysis
        Aligned samples with cumulative time delta, and the significant
        segments already detected from that delta.
    """
    aligned = align_laps_by_distance(
        reference,
        compared,
        distance_step=distance_step,
    )
    timed = calculate_time_delta(aligned)
    segments = detect_performance_segments(
        timed,
        min_time_delta_change=min_time_delta_change,
    )

    return LapPerformanceAnalysis(
        reference_lap=reference.lap_number,
        compared_lap=compared.lap_number,
        total_time_delta=float(timed["cumulative_time_delta"].iloc[-1]),
        aligned_data=timed,
        performance_segments=segments,
    )
