import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from telemetry_analyzer.analysis.distance import align_laps_by_distance
from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset
from telemetry_analyzer.visualization.telemetry import (
    plot_brake_comparison,
    plot_speed_comparison,
    plot_speed_delta,
    plot_throttle_comparison,
    save_figure,
)


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)


def _aligned() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance": [0.0, 1.0, 2.0],
            "speed_reference": [100.0, 110.0, 120.0],
            "speed_compared": [102.0, 108.0, 125.0],
            "speed_delta": [2.0, -2.0, 5.0],
            "throttle_reference": [0.5, 1.0, 0.8],
            "throttle_compared": [0.4, 0.9, 1.0],
            "throttle_delta": [-0.1, -0.1, 0.2],
            "brake_reference": [0.0, 0.2, 0.5],
            "brake_compared": [0.0, 0.0, 0.4],
            "brake_delta": [0.0, -0.2, -0.1],
        }
    )


def _line(figure: Figure, label: str):
    for line in figure.axes[0].get_lines():
        if line.get_label() == label:
            return line
    raise AssertionError(f"Missing line: {label}")


def test_plot_speed_comparison_returns_figure():
    figure = plot_speed_comparison(_aligned())

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_speed_comparison_creates_axes():
    figure = plot_speed_comparison(_aligned())

    assert len(figure.axes) == 1
    plt.close(figure)


def test_plot_speed_comparison_creates_two_lines():
    figure = plot_speed_comparison(_aligned())

    assert len(figure.axes[0].get_lines()) == 2
    plt.close(figure)


def test_plot_speed_comparison_uses_expected_data():
    data = _aligned()
    figure = plot_speed_comparison(data)
    reference = _line(figure, "Reference Lap")
    compared = _line(figure, "Compared Lap")

    assert list(reference.get_xdata()) == pytest.approx(data["distance"].tolist())
    assert list(reference.get_ydata()) == pytest.approx(data["speed_reference"].tolist())
    assert list(compared.get_ydata()) == pytest.approx(data["speed_compared"].tolist())
    plt.close(figure)


def test_plot_speed_delta_returns_figure():
    figure = plot_speed_delta(_aligned())

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_speed_delta_creates_one_line():
    figure = plot_speed_delta(_aligned())

    assert len(figure.axes[0].get_lines()) == 1
    plt.close(figure)


def test_plot_speed_delta_uses_speed_delta():
    data = _aligned()
    figure = plot_speed_delta(data)

    assert list(_line(figure, "Speed Delta").get_ydata()) == pytest.approx(
        data["speed_delta"].tolist()
    )
    plt.close(figure)


def test_plot_throttle_comparison_returns_figure():
    figure = plot_throttle_comparison(_aligned())

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_throttle_comparison_creates_two_lines():
    figure = plot_throttle_comparison(_aligned())

    assert len(figure.axes[0].get_lines()) == 2
    plt.close(figure)


def test_plot_throttle_comparison_uses_throttle_reference():
    data = _aligned()
    figure = plot_throttle_comparison(data)

    assert list(_line(figure, "Reference Lap").get_ydata()) == pytest.approx(
        (data["throttle_reference"] * 100).tolist()
    )
    plt.close(figure)


def test_plot_throttle_comparison_uses_throttle_compared():
    data = _aligned()
    figure = plot_throttle_comparison(data)

    assert list(_line(figure, "Compared Lap").get_ydata()) == pytest.approx(
        (data["throttle_compared"] * 100).tolist()
    )
    plt.close(figure)


def test_plot_brake_comparison_returns_figure():
    figure = plot_brake_comparison(_aligned())

    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_brake_comparison_creates_two_lines():
    figure = plot_brake_comparison(_aligned())

    assert len(figure.axes[0].get_lines()) == 2
    plt.close(figure)


def test_plot_brake_comparison_uses_brake_reference():
    data = _aligned()
    figure = plot_brake_comparison(data)

    assert list(_line(figure, "Reference Lap").get_ydata()) == pytest.approx(
        (data["brake_reference"] * 100).tolist()
    )
    plt.close(figure)


def test_plot_brake_comparison_uses_brake_compared():
    data = _aligned()
    figure = plot_brake_comparison(data)

    assert list(_line(figure, "Compared Lap").get_ydata()) == pytest.approx(
        (data["brake_compared"] * 100).tolist()
    )
    plt.close(figure)


def test_plot_functions_do_not_modify_dataframe():
    data = _aligned()
    original = data.copy()

    figures = [
        plot_speed_comparison(data),
        plot_speed_delta(data),
        plot_throttle_comparison(data),
        plot_brake_comparison(data),
    ]

    pd.testing.assert_frame_equal(data, original)
    for figure in figures:
        plt.close(figure)


def test_plot_functions_reject_missing_columns():
    data = _aligned().drop(columns=["speed_compared", "speed_delta", "throttle_reference", "brake_compared"])

    with pytest.raises(ValueError, match="speed_compared"):
        plot_speed_comparison(data)
    with pytest.raises(ValueError, match="speed_delta"):
        plot_speed_delta(data)
    with pytest.raises(ValueError, match="throttle_reference"):
        plot_throttle_comparison(data)
    with pytest.raises(ValueError, match="brake_compared"):
        plot_brake_comparison(data)


def test_save_figure_writes_png(tmp_path: Path):
    figure = plot_speed_comparison(_aligned())
    output = tmp_path / "plots" / "speed.png"

    save_figure(figure, output)

    assert output.is_file()
    assert output.stat().st_size > 0
    plt.close(figure)


def test_save_figure_accepts_path(tmp_path: Path):
    figure = plot_speed_delta(_aligned())
    output = Path(tmp_path / "delta.png")

    save_figure(figure, output)

    assert output.is_file()
    plt.close(figure)


def test_figures_save_without_show(tmp_path: Path):
    figure = plot_brake_comparison(_aligned())
    output = tmp_path / "brake.png"

    figure.savefig(output)

    assert output.is_file()
    plt.close(figure)


def test_pipeline_plots_aligned_speed():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)
    aligned = align_laps_by_distance(laps[1], laps[2])
    figure = plot_speed_comparison(aligned)

    assert isinstance(figure, Figure)
    assert len(figure.axes[0].get_lines()) == 2
    plt.close(figure)
