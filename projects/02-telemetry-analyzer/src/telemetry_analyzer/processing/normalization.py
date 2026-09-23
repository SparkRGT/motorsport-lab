"""Normalize telemetry dataset dtypes without changing measured values."""

import pandas as pd

from telemetry_analyzer.processing.validation import REQUIRED_COLUMNS

# Integer columns may arrive as float when a source file contains missing values.
_INTEGER_COLUMNS = {
    "frame",
    "lap_number",
    "gear",
    "rpm",
    "drs",
}


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataset with the expected numeric dtypes.

    The original DataFrame is not modified. Missing values are preserved.
    Invalid tokens are not coerced to NaN, and no rows are dropped or filled.

    Parameters
    ----------
    df:
        Telemetry dataset to normalize.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with expected numeric dtypes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, got {type(df).__name__}")

    normalized = df.copy()

    for column in REQUIRED_COLUMNS:
        if column in _INTEGER_COLUMNS:
            normalized[column] = _as_integer(normalized[column])
        else:
            normalized[column] = _as_float(normalized[column])

    return normalized


def _as_float(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    return numeric.astype("float64")


def _as_integer(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    present = numeric.dropna()
    fractional_rows = present.index[present.mod(1).ne(0)].tolist()
    if fractional_rows:
        raise ValueError(
            f"Column '{series.name}' contains non-integer values at rows {fractional_rows}"
        )

    if numeric.isna().any():
        return numeric.astype("Int64")

    return numeric.astype("int64")
