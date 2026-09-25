"""Plot distance-aligned telemetry comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_speed_comparison(data: pd.DataFrame) -> Figure:
    """Plot reference and compared speed against distance."""
    _require_columns(data, ["distance", "speed_reference", "speed_compared"])
    figure, axes = plt.subplots()
    axes.plot(data["distance"], data["speed_reference"], label="Reference Lap")
    axes.plot(data["distance"], data["speed_compared"], label="Compared Lap")
    _decorate(axes, "Speed comparison", "Speed")
    return figure


def plot_speed_delta(data: pd.DataFrame) -> Figure:
    """Plot speed delta against distance."""
    _require_columns(data, ["distance", "speed_delta"])
    figure, axes = plt.subplots()
    axes.plot(data["distance"], data["speed_delta"], label="Speed Delta")
    _decorate(axes, "Speed delta", "Speed Delta")
    return figure


def plot_throttle_comparison(data: pd.DataFrame) -> Figure:
    """Plot reference and compared throttle against distance."""
    _require_columns(data, ["distance", "throttle_reference", "throttle_compared"])
    figure, axes = plt.subplots()
    axes.plot(data["distance"], data["throttle_reference"] * 100, label="Reference Lap")
    axes.plot(data["distance"], data["throttle_compared"] * 100, label="Compared Lap")
    _decorate(axes, "Throttle comparison", "Throttle (%)")
    return figure


def plot_brake_comparison(data: pd.DataFrame) -> Figure:
    """Plot reference and compared brake against distance."""
    _require_columns(data, ["distance", "brake_reference", "brake_compared"])
    figure, axes = plt.subplots()
    axes.plot(data["distance"], data["brake_reference"] * 100, label="Reference Lap")
    axes.plot(data["distance"], data["brake_compared"] * 100, label="Compared Lap")
    _decorate(axes, "Brake comparison", "Brake (%)")
    return figure


def save_figure(figure: Figure, path: str | Path) -> None:
    """Save a figure to disk without displaying it."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def _require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _decorate(axes: plt.Axes, title: str, ylabel: str) -> None:
    axes.set_title(title)
    axes.set_xlabel("Distance (m)")
    axes.set_ylabel(ylabel)
    axes.legend()
    axes.grid(True)
