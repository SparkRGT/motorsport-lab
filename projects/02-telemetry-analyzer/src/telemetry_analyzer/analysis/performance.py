"""Significant cumulative time-delta changes between two aligned laps."""

from dataclasses import dataclass

import pandas as pd

_REQUIRED_COLUMNS = ("distance", "cumulative_time_delta")


@dataclass
class PerformanceSegment:
    """A contiguous distance range where cumulative time delta changes past a threshold."""

    start_distance: float
    end_distance: float
    distance: float
    start_time_delta: float
    end_time_delta: float
    time_delta_change: float
    absolute_time_delta_change: float
    sample_count: int


def detect_performance_segments(
    data: pd.DataFrame,
    min_time_delta_change: float = 0.10,
) -> list[PerformanceSegment]:
    """
    Find distance ranges where cumulative time delta changes significantly.

    ``cumulative_time_delta`` is read as already calculated. Consecutive samples
    that move in the same direction form one segment. A segment is kept when
    the absolute change reaches ``min_time_delta_change``.

    Parameters
    ----------
    data:
        Distance-aligned telemetry that already includes ``cumulative_time_delta``.
    min_time_delta_change:
        Minimum absolute change, in seconds. Must be greater than zero.

    Returns
    -------
    list[PerformanceSegment]
        Segments in ascending start distance. An empty list means no significant change.
    """
    _require_threshold(min_time_delta_change)
    _require_frame(data)

    distance = data["distance"].tolist()
    time_delta = data["cumulative_time_delta"].tolist()
    segments: list[PerformanceSegment] = []
    index = 0
    last = len(time_delta) - 1

    while index < last:
        direction = _direction(time_delta[index + 1] - time_delta[index])
        if direction == 0:
            index += 1
            continue

        end = index
        while end < last and _direction(time_delta[end + 1] - time_delta[end]) == direction:
            end += 1

        change = time_delta[end] - time_delta[index]
        if abs(change) >= min_time_delta_change:
            segments.append(_segment(distance, time_delta, index, end, change))
        index = end

    return segments


def _require_threshold(min_time_delta_change: float) -> None:
    if isinstance(min_time_delta_change, bool) or not isinstance(min_time_delta_change, (int, float)):
        raise ValueError(
            "min_time_delta_change must be greater than 0, "
            f"got {min_time_delta_change!r}"
        )
    if min_time_delta_change <= 0:
        raise ValueError(
            f"min_time_delta_change must be greater than 0, got {min_time_delta_change}"
        )


def _require_frame(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, got {type(data).__name__}")

    missing = [column for column in _REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if data.empty:
        raise ValueError("DataFrame is empty")

    _require_numeric(data["distance"], "distance")
    _require_numeric(data["cumulative_time_delta"], "cumulative_time_delta")
    steps = data["distance"].diff().iloc[1:]
    if (steps < 0).any():
        raise ValueError("distance must be strictly increasing")
    if (steps == 0).any():
        raise ValueError("distance contains duplicate values")


def _require_numeric(values: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(values) or values.isna().any():
        raise ValueError(f"{name} must contain numeric data")


def _direction(step: float) -> int:
    if step > 0:
        return 1
    if step < 0:
        return -1
    return 0


def _segment(
    distance: list[float],
    time_delta: list[float],
    start: int,
    end: int,
    change: float,
) -> PerformanceSegment:
    start_distance = float(distance[start])
    end_distance = float(distance[end])
    return PerformanceSegment(
        start_distance=start_distance,
        end_distance=end_distance,
        distance=end_distance - start_distance,
        start_time_delta=float(time_delta[start]),
        end_time_delta=float(time_delta[end]),
        time_delta_change=float(change),
        absolute_time_delta_change=abs(float(change)),
        sample_count=end - start + 1,
    )
