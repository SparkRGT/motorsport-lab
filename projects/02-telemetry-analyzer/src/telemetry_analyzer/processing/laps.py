"""Segment telemetry into laps using the recorded lap number."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class LapData:
    """Telemetry samples recorded for a single lap number."""

    lap_number: int
    data: pd.DataFrame
    start_time: float
    end_time: float
    duration: float
    sample_count: int


def segment_laps(df: pd.DataFrame) -> dict[int, LapData]:
    """
    Group telemetry samples by the recorded lap number.

    Parameters
    ----------
    df:
        Telemetry dataset ordered by session time.

    Returns
    -------
    dict[int, LapData]
        One entry for each lap number present in ``df``, in appearance order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, got {type(df).__name__}")

    missing = [
        column for column in ("lap_number", "session_time") if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    laps: dict[int, LapData] = {}
    for lap_number, group in df.groupby("lap_number", sort=False):
        lap_df = group.copy()
        start_time = float(lap_df["session_time"].iloc[0])
        end_time = float(lap_df["session_time"].iloc[-1])
        number = int(lap_number)
        laps[number] = LapData(
            lap_number=number,
            data=lap_df,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            sample_count=len(lap_df),
        )

    return laps
