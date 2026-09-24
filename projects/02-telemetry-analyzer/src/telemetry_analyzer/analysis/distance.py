"""Distance-based alignment of telemetry from two laps."""

import numpy as np
import pandas as pd

from telemetry_analyzer.processing.laps import LapData

_SIGNALS = ("speed", "throttle", "brake")

_COLUMNS = [
    "distance",
    "speed_reference",
    "speed_compared",
    "speed_delta",
    "throttle_reference",
    "throttle_compared",
    "throttle_delta",
    "brake_reference",
    "brake_compared",
    "brake_delta",
]


def align_laps_by_distance(
    reference: LapData,
    compared: LapData,
    distance_step: float = 1.0,
) -> pd.DataFrame:
    """
    Interpolate two laps onto a shared lap-distance grid.

    Parameters
    ----------
    reference:
        Baseline lap.
    compared:
        Lap subtracted from the reference when computing deltas.
    distance_step:
        Spacing of the shared distance grid. Must be greater than zero.

    Returns
    -------
    pandas.DataFrame
        Aligned signals inside the overlapping distance range.
        Each delta is ``compared - reference``.
    """
    if distance_step <= 0:
        raise ValueError(f"distance_step must be greater than 0, got {distance_step}")

    reference_distance = _distance_axis(reference, "Reference")
    compared_distance = _distance_axis(compared, "Compared")
    common_start = float(max(reference_distance[0], compared_distance[0]))
    common_end = float(min(reference_distance[-1], compared_distance[-1]))
    if common_start > common_end:
        raise ValueError(
            "Laps have no overlapping lap_distance range: "
            f"reference [{reference_distance[0]}, {reference_distance[-1]}], "
            f"compared [{compared_distance[0]}, {compared_distance[-1]}]"
        )

    distance = _distance_grid(common_start, common_end, distance_step)
    aligned = {"distance": distance}
    for signal in _SIGNALS:
        reference_values = _interpolate(distance, reference_distance, reference, signal)
        compared_values = _interpolate(distance, compared_distance, compared, signal)
        aligned[f"{signal}_reference"] = reference_values
        aligned[f"{signal}_compared"] = compared_values
        aligned[f"{signal}_delta"] = compared_values - reference_values

    return pd.DataFrame(aligned, columns=_COLUMNS)


def _distance_axis(lap: LapData, name: str) -> np.ndarray:
    if lap.data.empty:
        raise ValueError(f"{name} lap has no telemetry samples")

    distance = np.array(lap.data["lap_distance"].to_numpy(), dtype=float, copy=True)
    if len(distance) < 2 or not np.all(np.diff(distance) > 0):
        raise ValueError(
            f"{name} lap does not have enough strictly increasing "
            "lap_distance points to interpolate"
        )
    return distance


def _distance_grid(start: float, end: float, step: float) -> np.ndarray:
    if end == start:
        return np.array([start], dtype=float)

    count = int(np.floor((end - start) / step + 1e-9))
    grid = start + np.arange(count + 1, dtype=float) * step
    grid = grid[grid <= end + 1e-9]
    near_end = np.abs(grid - end) <= 1e-9
    grid[near_end] = end
    return np.unique(grid)


def _interpolate(
    distance: np.ndarray,
    source_distance: np.ndarray,
    lap: LapData,
    signal: str,
) -> np.ndarray:
    source_values = np.array(lap.data[signal].to_numpy(), dtype=float, copy=True)
    return np.interp(distance, source_distance, source_values, left=np.nan, right=np.nan)
