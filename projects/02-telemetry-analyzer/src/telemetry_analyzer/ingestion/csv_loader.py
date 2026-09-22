from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a telemetry dataset from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded telemetry data.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    if not csv_path.is_file():
        raise ValueError(f"Dataset path is not a file: {csv_path}")

    return pd.read_csv(csv_path)
