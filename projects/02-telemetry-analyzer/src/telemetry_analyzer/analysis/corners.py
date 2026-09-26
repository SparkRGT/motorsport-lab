"""Telemetry-derived corner segments from braking and acceleration zones."""

from dataclasses import dataclass

import pandas as pd

from telemetry_analyzer.analysis.zones import detect_acceleration_zones, detect_braking_zones
from telemetry_analyzer.processing.laps import LapData


@dataclass
class CornerSegment:
    """A corner candidate inferred from braking and the following acceleration."""

    start_distance: float
    end_distance: float
    distance: float
    start_time: float
    end_time: float
    duration: float
    minimum_speed: float
    minimum_speed_distance: float
    sample_count: int


def detect_corner_segments(
    lap: LapData,
    brake_threshold: float = 0.05,
    throttle_threshold: float = 0.90,
) -> list[CornerSegment]:
    """
    Build telemetry-derived corner segments for one lap.

    Each segment runs from a braking zone to the first later acceleration zone.
    This does not identify a physical corner on a circuit.

    Parameters
    ----------
    lap:
        One segmented lap.
    brake_threshold:
        Forwarded to braking-zone detection. Must be between 0 and 1.
    throttle_threshold:
        Forwarded to acceleration-zone detection. Must be between 0 and 1.

    Returns
    -------
    list[CornerSegment]
        Segments in ascending start distance. An empty list means no pairing.
    """
    braking_zones = detect_braking_zones(lap, brake_threshold)
    acceleration_zones = detect_acceleration_zones(lap, throttle_threshold)
    _require_speed(lap)

    segments: list[CornerSegment] = []
    next_acceleration = 0
    for braking_zone in braking_zones:
        acceleration_index = _next_acceleration_index(
            acceleration_zones,
            next_acceleration,
            braking_zone.start_distance,
        )
        if acceleration_index is None:
            continue
        next_acceleration = acceleration_index + 1
        segments.append(
            _segment(
                lap,
                braking_zone.start_distance,
                acceleration_zones[acceleration_index].end_distance,
            )
        )
    return segments


def _next_acceleration_index(
    zones,
    start_index: int,
    braking_start: float,
) -> int | None:
    for offset, zone in enumerate(zones[start_index:]):
        if zone.start_distance > braking_start:
            return start_index + offset
    return None


def _require_speed(lap: LapData) -> None:
    if "speed" not in lap.data.columns:
        raise ValueError("Lap is missing required columns: ['speed']")
    speed = lap.data["speed"]
    if not pd.api.types.is_numeric_dtype(speed) or speed.isna().any():
        raise ValueError("speed must contain numeric data")
    if (speed < 0).any():
        raise ValueError("speed must be greater than or equal to 0")


def _segment(lap: LapData, start_distance: float, end_distance: float) -> CornerSegment:
    distance = lap.data["lap_distance"]
    included = lap.data.loc[(distance >= start_distance) & (distance <= end_distance)]
    start_time = float(included["session_time"].iloc[0])
    end_time = float(included["session_time"].iloc[-1])
    speeds = included["speed"]
    minimum_index = speeds.idxmin()
    return CornerSegment(
        start_distance=start_distance,
        end_distance=end_distance,
        distance=end_distance - start_distance,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        minimum_speed=float(speeds.loc[minimum_index]),
        minimum_speed_distance=float(included.loc[minimum_index, "lap_distance"]),
        sample_count=len(included),
    )
