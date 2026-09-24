"""Numeric comparison of two laps from their basic metrics."""

from dataclasses import dataclass

from telemetry_analyzer.metrics.basic import LapMetrics


@dataclass
class LapComparison:
    """Numeric differences between a compared lap and a reference lap."""

    reference_lap: int
    compared_lap: int
    time_delta: float
    average_speed_delta: float
    maximum_speed_delta: float
    minimum_speed_delta: float
    average_throttle_delta: float
    full_throttle_percentage_delta: float
    brake_percentage_delta: float
    maximum_rpm_delta: float


def compare_laps(reference: LapMetrics, compared: LapMetrics) -> LapComparison:
    """
    Compare two laps using basic metrics.

    Each delta is ``compared - reference``. The input metrics are not modified.

    Parameters
    ----------
    reference:
        Baseline lap metrics.
    compared:
        Lap metrics to subtract the reference from.

    Returns
    -------
    LapComparison
        Numeric differences between the two laps.
    """
    return LapComparison(
        reference_lap=reference.lap_number,
        compared_lap=compared.lap_number,
        time_delta=compared.duration - reference.duration,
        average_speed_delta=compared.average_speed - reference.average_speed,
        maximum_speed_delta=compared.maximum_speed - reference.maximum_speed,
        minimum_speed_delta=compared.minimum_speed - reference.minimum_speed,
        average_throttle_delta=compared.average_throttle - reference.average_throttle,
        full_throttle_percentage_delta=(
            compared.full_throttle_percentage - reference.full_throttle_percentage
        ),
        brake_percentage_delta=compared.brake_percentage - reference.brake_percentage,
        maximum_rpm_delta=compared.maximum_rpm - reference.maximum_rpm,
    )
