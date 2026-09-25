"""Distance-based time delta between two aligned laps."""

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = ("distance", "speed_reference", "speed_compared")
_KMH_TO_MPS = 3.6


def calculate_time_delta(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate elapsed time for two laps from distance and speed.

    Speed is interpreted as km/h. Each delta is ``compared - reference``.
    The input DataFrame is not modified.

    Parameters
    ----------
    data:
        Distance-aligned telemetry, as returned by ``align_laps_by_distance``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``data`` with interval times and the cumulative time delta.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, got {type(data).__name__}")

    missing = [column for column in _REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if data.empty:
        raise ValueError("DataFrame is empty")

    _require_distance(data["distance"])
    _require_speed(data["speed_reference"], "speed_reference")
    _require_speed(data["speed_compared"], "speed_compared")

    result = data.copy()
    result["time_reference"] = _interval_times(result["distance"], result["speed_reference"])
    result["time_compared"] = _interval_times(result["distance"], result["speed_compared"])
    result["time_delta"] = result["time_compared"] - result["time_reference"]
    result["cumulative_time_delta"] = result["time_delta"].cumsum()
    return result


def _require_distance(distance: pd.Series) -> None:
    if not pd.api.types.is_numeric_dtype(distance) or distance.isna().any():
        raise ValueError("distance must contain numeric data")

    steps = np.diff(distance.to_numpy(dtype=float))
    if np.any(steps < 0):
        raise ValueError("distance must be strictly increasing")
    if np.any(steps == 0):
        raise ValueError("distance contains duplicate values")


def _require_speed(speed: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(speed) or speed.isna().any():
        raise ValueError(f"{name} must contain numeric data")
    if (speed < 0).any():
        raise ValueError(f"{name} must be greater than or equal to 0")
    if len(speed) > 1 and (speed.iloc[1:] == 0).any():
        raise ValueError(f"{name} must be greater than 0 to calculate a time interval")


def _interval_times(distance: pd.Series, speed_kmh: pd.Series) -> np.ndarray:
    times = np.zeros(len(distance), dtype=float)
    if len(distance) == 1:
        return times

    delta_distance = np.diff(distance.to_numpy(dtype=float))
    speed_mps = speed_kmh.iloc[1:].to_numpy(dtype=float) / _KMH_TO_MPS
    times[1:] = delta_distance / speed_mps
    return times
