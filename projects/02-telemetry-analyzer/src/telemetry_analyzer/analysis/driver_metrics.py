"""Objective telemetry indicators inside reference-lap distance windows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from telemetry_analyzer.analysis.corners import detect_corner_segments
from telemetry_analyzer.analysis.lap_performance import LapPerformanceAnalysis
from telemetry_analyzer.analysis.zones import detect_acceleration_zones, detect_braking_zones
from telemetry_analyzer.processing.laps import LapData

_REQUIRED_COLUMNS = (
    "distance",
    "speed_reference",
    "speed_compared",
    "throttle_reference",
    "throttle_compared",
    "brake_reference",
    "brake_compared",
    "time_reference",
    "time_compared",
)


@dataclass
class MetricComparison:
    """One indicator measured on both laps over the same distance window."""

    reference: float
    compared: float
    delta: float


@dataclass
class BrakingMetrics:
    """Telemetry inside one reference-lap braking window."""

    start_distance: float
    end_distance: float
    distance: float
    speed_at_start: MetricComparison
    speed_at_end: MetricComparison
    minimum_speed: MetricComparison
    minimum_speed_distance: MetricComparison
    max_brake: MetricComparison
    average_throttle: MetricComparison
    duration: MetricComparison


@dataclass
class CornerMetrics:
    """Telemetry inside one reference-lap corner window."""

    start_distance: float
    end_distance: float
    distance: float
    entry_speed: MetricComparison
    exit_speed: MetricComparison
    minimum_speed: MetricComparison
    minimum_speed_distance: MetricComparison
    duration: MetricComparison


@dataclass
class AccelerationMetrics:
    """Telemetry inside one reference-lap acceleration window."""

    start_distance: float
    end_distance: float
    distance: float
    speed_at_start: MetricComparison
    speed_at_end: MetricComparison
    throttle_at_start: MetricComparison
    max_throttle: MetricComparison
    throttle_at_end: MetricComparison
    duration: MetricComparison


@dataclass
class DriverPerformanceMetrics:
    """Braking, corner, and acceleration indicators for one lap pair."""

    braking: list[BrakingMetrics]
    corners: list[CornerMetrics]
    acceleration: list[AccelerationMetrics]


def calculate_driver_performance_metrics(
    performance: LapPerformanceAnalysis,
    reference_lap: LapData,
    compared_lap: LapData,
    brake_threshold: float = 0.05,
    throttle_threshold: float = 0.90,
) -> DriverPerformanceMetrics:
    """
    Measure both laps inside the reference lap's driving windows.

    Parameters
    ----------
    performance:
        Distance-aligned comparison. It is not recalculated.
    reference_lap:
        Lap whose braking zones, acceleration zones, and corner segments
        define the distance windows.
    compared_lap:
        The other lap in ``performance``. Its signals are read from
        ``performance.aligned_data``, not from a second event detection.
    brake_threshold:
        Forwarded to braking and corner detection.
    throttle_threshold:
        Forwarded to acceleration and corner detection.

    Returns
    -------
    DriverPerformanceMetrics
        Indicators grouped by braking, corner, and acceleration. Each delta
        is ``compared - reference``. Lists are ordered by start distance.

    Notes
    -----
    A window that starts before or ends after the aligned distance is omitted.
    Entry speed is the speed at the corner's start distance. Exit speed is
    the speed at its end distance. Those distances come from the existing
    corner segment: the reference braking-zone start through the following
    acceleration-zone end.

    Start and end values are linearly interpolated on the aligned grid.
    Minimum, maximum, and average values use the piecewise-linear signal on
    that same window: the interpolated endpoints plus every grid sample
    strictly inside the window. Ties use the smallest distance. The average
    is the distance-weighted mean. A window whose start and end distances
    are equal uses the interpolated value at that point.

    Duration is the time taken to cover the shared window. It is the
    difference of cumulative interval time between the window's end and
    start, using ``time_reference`` and ``time_compared``. It is not the
    session-time length of a zone detected separately on each lap.

    Speed is km/h, distance is metres, duration is seconds, and throttle
    and brake stay in the range recorded by the telemetry. These values
    describe the window. They do not explain the driver's actions.
    """
    _require_matching_laps(performance, reference_lap, compared_lap)
    grid = _aligned_grid(performance.aligned_data)

    braking = [
        _braking_metrics(zone.start_distance, zone.end_distance, grid)
        for zone in detect_braking_zones(reference_lap, brake_threshold)
        if _window_is_inside(grid.distance, zone.start_distance, zone.end_distance)
    ]
    corners = [
        _corner_metrics(segment.start_distance, segment.end_distance, grid)
        for segment in detect_corner_segments(
            reference_lap,
            brake_threshold=brake_threshold,
            throttle_threshold=throttle_threshold,
        )
        if _window_is_inside(grid.distance, segment.start_distance, segment.end_distance)
    ]
    acceleration = [
        _acceleration_metrics(zone.start_distance, zone.end_distance, grid)
        for zone in detect_acceleration_zones(reference_lap, throttle_threshold)
        if _window_is_inside(grid.distance, zone.start_distance, zone.end_distance)
    ]
    braking.sort(key=lambda item: item.start_distance)
    corners.sort(key=lambda item: item.start_distance)
    acceleration.sort(key=lambda item: item.start_distance)
    return DriverPerformanceMetrics(
        braking=braking,
        corners=corners,
        acceleration=acceleration,
    )


@dataclass
class _AlignedGrid:
    distance: np.ndarray
    speed_reference: np.ndarray
    speed_compared: np.ndarray
    throttle_reference: np.ndarray
    throttle_compared: np.ndarray
    brake_reference: np.ndarray
    brake_compared: np.ndarray
    elapsed_reference: np.ndarray
    elapsed_compared: np.ndarray


def _require_matching_laps(
    performance: LapPerformanceAnalysis,
    reference_lap: LapData,
    compared_lap: LapData,
) -> None:
    if reference_lap.lap_number != performance.reference_lap:
        raise ValueError(
            "reference_lap does not match the lap performance analysis: "
            f"{reference_lap.lap_number} != {performance.reference_lap}"
        )
    if compared_lap.lap_number != performance.compared_lap:
        raise ValueError(
            "compared_lap does not match the lap performance analysis: "
            f"{compared_lap.lap_number} != {performance.compared_lap}"
        )


def _aligned_grid(data: pd.DataFrame) -> _AlignedGrid:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected aligned data to be a DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("Aligned performance data is empty")

    missing = [column for column in _REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Aligned performance data is missing required columns: {missing}")

    columns = {name: _finite_column(data[name], name) for name in _REQUIRED_COLUMNS}
    distance = columns["distance"]
    if len(distance) < 1 or (len(distance) > 1 and np.any(np.diff(distance) <= 0)):
        raise ValueError("distance must be strictly increasing")

    return _AlignedGrid(
        distance=distance,
        speed_reference=columns["speed_reference"],
        speed_compared=columns["speed_compared"],
        throttle_reference=columns["throttle_reference"],
        throttle_compared=columns["throttle_compared"],
        brake_reference=columns["brake_reference"],
        brake_compared=columns["brake_compared"],
        elapsed_reference=np.cumsum(columns["time_reference"]),
        elapsed_compared=np.cumsum(columns["time_compared"]),
    )


def _finite_column(values: pd.Series, name: str) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(values) or values.isna().any():
        raise ValueError(f"{name} must contain numeric data")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _window_is_inside(distance: np.ndarray, start: float, end: float) -> bool:
    if not np.isfinite(start) or not np.isfinite(end) or end < start:
        raise ValueError(
            "Event distances must be finite and the end must not precede the start: "
            f"[{start}, {end}]"
        )
    return bool(start >= distance[0] and end <= distance[-1])


def _braking_metrics(start: float, end: float, grid: _AlignedGrid) -> BrakingMetrics:
    reference_min, reference_min_distance = _minimum(grid.distance, grid.speed_reference, start, end)
    compared_min, compared_min_distance = _minimum(grid.distance, grid.speed_compared, start, end)
    return BrakingMetrics(
        start_distance=float(start),
        end_distance=float(end),
        distance=float(end - start),
        speed_at_start=_signal_at(grid.distance, grid.speed_reference, grid.speed_compared, start),
        speed_at_end=_signal_at(grid.distance, grid.speed_reference, grid.speed_compared, end),
        minimum_speed=_comparison(reference_min, compared_min),
        minimum_speed_distance=_comparison(reference_min_distance, compared_min_distance),
        max_brake=_comparison(
            _maximum(grid.distance, grid.brake_reference, start, end),
            _maximum(grid.distance, grid.brake_compared, start, end),
        ),
        average_throttle=_comparison(
            _average(grid.distance, grid.throttle_reference, start, end),
            _average(grid.distance, grid.throttle_compared, start, end),
        ),
        duration=_duration(grid, start, end),
    )


def _corner_metrics(start: float, end: float, grid: _AlignedGrid) -> CornerMetrics:
    reference_min, reference_min_distance = _minimum(grid.distance, grid.speed_reference, start, end)
    compared_min, compared_min_distance = _minimum(grid.distance, grid.speed_compared, start, end)
    return CornerMetrics(
        start_distance=float(start),
        end_distance=float(end),
        distance=float(end - start),
        entry_speed=_signal_at(grid.distance, grid.speed_reference, grid.speed_compared, start),
        exit_speed=_signal_at(grid.distance, grid.speed_reference, grid.speed_compared, end),
        minimum_speed=_comparison(reference_min, compared_min),
        minimum_speed_distance=_comparison(reference_min_distance, compared_min_distance),
        duration=_duration(grid, start, end),
    )


def _acceleration_metrics(start: float, end: float, grid: _AlignedGrid) -> AccelerationMetrics:
    return AccelerationMetrics(
        start_distance=float(start),
        end_distance=float(end),
        distance=float(end - start),
        speed_at_start=_signal_at(grid.distance, grid.speed_reference, grid.speed_compared, start),
        speed_at_end=_signal_at(grid.distance, grid.speed_reference, grid.speed_compared, end),
        throttle_at_start=_signal_at(
            grid.distance,
            grid.throttle_reference,
            grid.throttle_compared,
            start,
        ),
        max_throttle=_comparison(
            _maximum(grid.distance, grid.throttle_reference, start, end),
            _maximum(grid.distance, grid.throttle_compared, start, end),
        ),
        throttle_at_end=_signal_at(
            grid.distance,
            grid.throttle_reference,
            grid.throttle_compared,
            end,
        ),
        duration=_duration(grid, start, end),
    )


def _comparison(reference: float, compared: float) -> MetricComparison:
    return MetricComparison(
        reference=float(reference),
        compared=float(compared),
        delta=float(compared - reference),
    )


def _signal_at(
    distance: np.ndarray,
    reference: np.ndarray,
    compared: np.ndarray,
    point: float,
) -> MetricComparison:
    return _comparison(
        float(np.interp(point, distance, reference)),
        float(np.interp(point, distance, compared)),
    )


def _knots(
    distance: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_value = float(np.interp(start, distance, values))
    if start == end:
        return np.array([start], dtype=float), np.array([start_value], dtype=float)

    end_value = float(np.interp(end, distance, values))
    interior = (distance > start) & (distance < end)
    return (
        np.concatenate(([start], distance[interior], [end])),
        np.concatenate(([start_value], values[interior], [end_value])),
    )


def _minimum(
    distance: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
) -> tuple[float, float]:
    knots_distance, knots_values = _knots(distance, values, start, end)
    index = int(np.argmin(knots_values))
    return float(knots_values[index]), float(knots_distance[index])


def _maximum(distance: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    _, knots_values = _knots(distance, values, start, end)
    return float(knots_values[int(np.argmax(knots_values))])


def _average(distance: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    knots_distance, knots_values = _knots(distance, values, start, end)
    span = float(knots_distance[-1] - knots_distance[0])
    if span == 0:
        return float(knots_values[0])
    widths = np.diff(knots_distance)
    heights = (knots_values[:-1] + knots_values[1:]) * 0.5
    return float(np.sum(widths * heights) / span)


def _duration(grid: _AlignedGrid, start: float, end: float) -> MetricComparison:
    reference = _elapsed(grid.distance, grid.elapsed_reference, start, end)
    compared = _elapsed(grid.distance, grid.elapsed_compared, start, end)
    return _comparison(reference, compared)


def _elapsed(distance: np.ndarray, elapsed: np.ndarray, start: float, end: float) -> float:
    return float(np.interp(end, distance, elapsed) - np.interp(start, distance, elapsed))
