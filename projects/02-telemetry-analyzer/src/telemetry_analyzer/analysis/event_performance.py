"""Measure cumulative time-delta change inside existing driving events."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from telemetry_analyzer.analysis.corners import detect_corner_segments
from telemetry_analyzer.analysis.lap_performance import LapPerformanceAnalysis
from telemetry_analyzer.analysis.zones import detect_acceleration_zones, detect_braking_zones
from telemetry_analyzer.processing.laps import LapData


@dataclass
class PerformanceEvent:
    """Change in cumulative time delta across one existing driving event."""

    event_type: str
    start_distance: float
    end_distance: float
    start_time_delta: float
    end_time_delta: float
    time_delta_change: float
    absolute_time_delta_change: float


def analyze_event_performance(
    performance: LapPerformanceAnalysis,
    reference_lap: LapData,
    compared_lap: LapData,
    brake_threshold: float = 0.05,
    throttle_threshold: float = 0.90,
) -> list[PerformanceEvent]:
    """
    Measure how cumulative time delta changes inside reference-lap events.

    Parameters
    ----------
    performance:
        Distance-aligned comparison that already includes
        ``cumulative_time_delta``. It is not recalculated.
    reference_lap:
        Lap whose braking zones, acceleration zones, and corner segments
        define the distance windows.
    compared_lap:
        The other lap in ``performance``. It is not used to detect events.
    brake_threshold:
        Forwarded to braking and corner detection.
    throttle_threshold:
        Forwarded to acceleration and corner detection.

    Returns
    -------
    list[PerformanceEvent]
        Events ordered by start distance, then event type. A positive
        ``time_delta_change`` means the compared lap accumulated more time
        than the reference inside that window.

    Notes
    -----
    Braking, corner, and acceleration windows are different views of the same
    lap. A corner can overlap the braking and acceleration that formed it, and
    all three are kept. This does not explain why the time difference changed.
    An event that falls outside the aligned distance is omitted because the
    comparison has no time delta there.
    """
    _require_matching_laps(performance, reference_lap, compared_lap)
    distance, cumulative = _delta_axis(performance.aligned_data)
    events = [
        *_events("braking", detect_braking_zones(reference_lap, brake_threshold), distance, cumulative),
        *_events(
            "acceleration",
            detect_acceleration_zones(reference_lap, throttle_threshold),
            distance,
            cumulative,
        ),
        *_events(
            "corner",
            detect_corner_segments(
                reference_lap,
                brake_threshold=brake_threshold,
                throttle_threshold=throttle_threshold,
            ),
            distance,
            cumulative,
        ),
    ]
    events.sort(key=lambda event: (event.start_distance, event.event_type))
    return events


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


def _delta_axis(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected aligned data to be a DataFrame, got {type(data).__name__}")
    if data.empty:
        raise ValueError("Aligned performance data is empty")

    missing = [
        column
        for column in ("distance", "cumulative_time_delta")
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Aligned performance data is missing required columns: {missing}")

    distance = data["distance"].to_numpy(dtype=float)
    cumulative = data["cumulative_time_delta"].to_numpy(dtype=float)
    if np.isnan(distance).any() or np.isnan(cumulative).any():
        raise ValueError("distance and cumulative_time_delta must be numeric")
    if len(distance) < 2 or np.any(np.diff(distance) <= 0):
        raise ValueError("distance must be strictly increasing")
    return distance, cumulative


def _events(event_type: str, zones, distance: np.ndarray, cumulative: np.ndarray):
    events: list[PerformanceEvent] = []
    for zone in zones:
        event = _event(event_type, zone.start_distance, zone.end_distance, distance, cumulative)
        if event is not None:
            events.append(event)
    return events


def _event(
    event_type: str,
    start_distance: float,
    end_distance: float,
    distance: np.ndarray,
    cumulative: np.ndarray,
) -> PerformanceEvent | None:
    if start_distance < distance[0] or end_distance > distance[-1]:
        return None
    if end_distance < start_distance:
        return None

    start_time_delta = float(np.interp(start_distance, distance, cumulative))
    end_time_delta = float(np.interp(end_distance, distance, cumulative))
    change = end_time_delta - start_time_delta
    return PerformanceEvent(
        event_type=event_type,
        start_distance=float(start_distance),
        end_distance=float(end_distance),
        start_time_delta=start_time_delta,
        end_time_delta=end_time_delta,
        time_delta_change=change,
        absolute_time_delta_change=abs(change),
    )
