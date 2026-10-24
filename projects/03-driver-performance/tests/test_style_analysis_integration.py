
import numpy as np
import pandas as pd

from driver_performance.style_analysis import analyze_driving_style
from telemetry_analyzer.analysis.lap_performance import analyze_lap_performance
from telemetry_analyzer.processing.laps import LapData


def make_lap(lap_number: int, compared: bool = False) -> LapData:
    """Build synthetic telemetry for an integration test."""
    distance = np.arange(0.0, 501.0, 10.0)

    speed = np.interp(
        distance,
        [0, 80, 100, 150, 180, 220, 260, 300, 400, 500],
        [210, 220, 210, 115, 130, 180, 220, 225, 215, 220],
    )

    throttle = np.full(len(distance), 0.5)
    throttle[(distance >= 180) & (distance <= 260)] = 0.95

    brake = np.zeros(len(distance))
    brake[(distance >= 100) & (distance <= 150)] = 0.8

    if compared:
        # Introduce measurable differences in the compared lap.
        braking_zone = (distance >= 100) & (distance <= 150)
        acceleration_zone = (distance >= 180) & (distance <= 260)

        speed[braking_zone] -= 8.0
        speed[acceleration_zone] += 5.0
        throttle[acceleration_zone] = 0.98
        brake[braking_zone] = 0.9

    # Calculate elapsed time from distance and speed.
    interval_times = np.zeros(len(distance))
    interval_times[1:] = (
        np.diff(distance) / (speed[1:] / 3.6)
    )
    session_time = np.cumsum(interval_times)

    data = pd.DataFrame(
        {
            "lap_number": lap_number,
            "lap_distance": distance,
            "session_time": session_time,
            "speed": speed,
            "throttle": throttle,
            "brake": brake,
        }
    )

    return LapData(
        lap_number=lap_number,
        data=data,
        start_time=float(session_time[0]),
        end_time=float(session_time[-1]),
        duration=float(session_time[-1] - session_time[0]),
        sample_count=len(data),
    )


def test_real_project_02_pipeline_creates_evidence_based_observations():
    reference_lap = make_lap(lap_number=1)
    compared_lap = make_lap(lap_number=2, compared=True)

    performance = analyze_lap_performance(
        reference=reference_lap,
        compared=compared_lap,
    )

    observations = analyze_driving_style(
        performance=performance,
        reference_lap=reference_lap,
        compared_lap=compared_lap,
    )

    assert observations

    categories = {observation.category for observation in observations}
    assert "braking" in categories
    assert "corner" in categories
    assert "acceleration" in categories

    for observation in observations:
        assert observation.id
        assert observation.title
        assert observation.description
        assert observation.evidence

        for evidence in observation.evidence:
            assert evidence.metric_name
            assert np.isfinite(evidence.value)
            assert evidence.unit
            assert evidence.distance_start_m is not None
            assert evidence.distance_end_m is not None
            assert evidence.distance_start_m <= evidence.distance_end_m
