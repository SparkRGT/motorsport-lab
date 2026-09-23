"""Validate telemetry datasets before analysis."""

import pandas as pd

REQUIRED_COLUMNS = [
    "session_time",
    "frame",
    "lap_number",
    "lap_distance",
    "speed",
    "throttle",
    "brake",
    "steering",
    "gear",
    "rpm",
    "drs",
]


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the minimum structure and values of a telemetry dataset.

    Parameters
    ----------
    df:
        Telemetry dataset to validate.

    Returns
    -------
    None
        The dataset is valid.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    ValueError
        If required columns, values, or time order are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, got {type(df).__name__}")

    _require_columns(df)
    _require_rows(df)
    _require_numeric(df)
    _require_no_missing_values(df)
    _require_ranges(df)
    _require_monotonic_session_time(df)


def _require_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def _require_rows(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty")


def _require_numeric(df: pd.DataFrame) -> None:
    invalid = [
        f"{column} ({df[column].dtype})"
        for column in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if invalid:
        raise ValueError(
            "Required columns must contain numeric data: " + ", ".join(invalid)
        )


def _require_no_missing_values(df: pd.DataFrame) -> None:
    problems: list[str] = []
    for column in REQUIRED_COLUMNS:
        rows = df.index[df[column].isna()].tolist()
        if rows:
            problems.append(f"{column} at rows {rows}")

    if problems:
        raise ValueError(
            "Required columns contain missing values: " + "; ".join(problems)
        )


def _require_ranges(df: pd.DataFrame) -> None:
    rules = [
        ("session_time", df["session_time"] < 0, "session_time must be >= 0"),
        ("lap_number", df["lap_number"] < 0, "lap_number must be >= 0"),
        ("lap_distance", df["lap_distance"] < 0, "lap_distance must be >= 0"),
        ("speed", df["speed"] < 0, "speed must be >= 0"),
        (
            "throttle",
            (df["throttle"] < 0.0) | (df["throttle"] > 1.0),
            "throttle must be between 0.0 and 1.0 inclusive",
        ),
        (
            "brake",
            (df["brake"] < 0.0) | (df["brake"] > 1.0),
            "brake must be between 0.0 and 1.0 inclusive",
        ),
        ("gear", df["gear"] < 0, "gear must be >= 0"),
        ("rpm", df["rpm"] < 0, "rpm must be >= 0"),
        ("drs", ~df["drs"].isin([0, 1]), "drs must be 0 or 1"),
    ]

    problems: list[str] = []
    for _column, mask, message in rules:
        rows = df.index[mask].tolist()
        if rows:
            problems.append(f"{message} (rows {rows})")

    if problems:
        raise ValueError("Dataset values are out of range: " + "; ".join(problems))


def _require_monotonic_session_time(df: pd.DataFrame) -> None:
    if not df["session_time"].is_monotonic_increasing:
        raise ValueError("session_time must be monotonically increasing")
