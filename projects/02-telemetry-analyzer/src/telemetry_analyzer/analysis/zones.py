"""Detect braking and acceleration zones on a single lap."""

from dataclasses import dataclass

import pandas as pd

from telemetry_analyzer.processing.laps import LapData

_BRAKE_COLUMNS = ("lap_distance", "session_time", "brake")
_THROTTLE_COLUMNS = ("lap_distance", "session_time", "throttle")


@dataclass
class BrakingZone:
    """Consecutive samples where brake pressure is above a threshold."""

    start_distance: float
    end_distance: float
    distance: float
    start_time: float
    end_time: float
    duration: float
    sample_count: int


@dataclass
class AccelerationZone:
    """Consecutive samples where throttle is at or above a threshold."""

    start_distance: float
    end_distance: float
    distance: float
    start_time: float
    end_time: float
    duration: float
    sample_count: int


def detect_braking_zones(
    lap: LapData,
    brake_threshold: float = 0.05,
) -> list[BrakingZone]:
    """
    Group consecutive braking samples into zones.

    A sample belongs to a zone when ``brake > brake_threshold``.

    Parameters
    ----------
    lap:
        One segmented lap.
    brake_threshold:
        Minimum brake value, exclusive. Must be between 0 and 1.

    Returns
    -------
    list[BrakingZone]
        Zones in ascending lap distance. An empty list means no braking event.
    """
    _require_threshold(brake_threshold, "brake_threshold")
    samples = _lap_samples(lap, _BRAKE_COLUMNS)
    _require_signal(samples["brake"], "brake")
    return [
        BrakingZone(*fields)
        for fields in _zone_fields(samples, samples["brake"] > brake_threshold)
    ]


def detect_acceleration_zones(
    lap: LapData,
    throttle_threshold: float = 0.90,
) -> list[AccelerationZone]:
    """
    Group consecutive acceleration samples into zones.

    A sample belongs to a zone when ``throttle >= throttle_threshold``.

    Parameters
    ----------
    lap:
        One segmented lap.
    throttle_threshold:
        Minimum throttle value, inclusive. Must be between 0 and 1.

    Returns
    -------
    list[AccelerationZone]
        Zones in ascending lap distance. An empty list means no acceleration event.
    """
    _require_threshold(throttle_threshold, "throttle_threshold")
    samples = _lap_samples(lap, _THROTTLE_COLUMNS)
    _require_signal(samples["throttle"], "throttle")
    return [
        AccelerationZone(*fields)
        for fields in _zone_fields(samples, samples["throttle"] >= throttle_threshold)
    ]


def _require_threshold(threshold: float, name: str) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"{name} must be between 0 and 1, got {threshold!r}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {threshold}")


def _lap_samples(lap: LapData, columns: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(lap, LapData):
        raise TypeError(f"Expected a LapData, got {type(lap).__name__}")
    if lap.data.empty:
        raise ValueError("Lap has no telemetry samples")

    missing = [column for column in columns if column not in lap.data.columns]
    if missing:
        raise ValueError(f"Lap is missing required columns: {missing}")

    samples = lap.data.loc[:, list(columns)].copy()
    _require_axis(samples["lap_distance"], "lap_distance")
    _require_axis(samples["session_time"], "session_time")
    return samples


def _require_axis(values: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(values) or values.isna().any():
        raise ValueError(f"{name} must contain numeric data")
    steps = values.diff().iloc[1:]
    if (steps < 0).any():
        raise ValueError(f"{name} must be non-decreasing")


def _require_signal(values: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(values) or values.isna().any():
        raise ValueError(f"{name} must contain numeric data")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} must be between 0 and 1")


def _zone_fields(samples: pd.DataFrame, active: pd.Series) -> list[tuple[float, ...]]:
    zones: list[tuple[float, ...]] = []
    start: int | None = None
    flags = active.tolist()
    for index, is_active in enumerate(flags):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            zones.append(_zone_bounds(samples, start, index - 1))
            start = None
    if start is not None:
        zones.append(_zone_bounds(samples, start, len(flags) - 1))
    return zones


def _zone_bounds(samples: pd.DataFrame, start: int, end: int) -> tuple[float, ...]:
    start_distance = float(samples["lap_distance"].iloc[start])
    end_distance = float(samples["lap_distance"].iloc[end])
    start_time = float(samples["session_time"].iloc[start])
    end_time = float(samples["session_time"].iloc[end])
    return (
        start_distance,
        end_distance,
        end_distance - start_distance,
        start_time,
        end_time,
        end_time - start_time,
        end - start + 1,
    )
