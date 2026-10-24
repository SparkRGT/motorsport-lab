
"""Evidence-based observations from driving telemetry comparisons."""

from telemetry_analyzer.analysis.driver_metrics import (
    AccelerationMetrics,
    BrakingMetrics,
    CornerMetrics,
    calculate_driver_performance_metrics,
)
from telemetry_analyzer.analysis.event_performance import (
    PerformanceEvent,
    analyze_event_performance,
)
from telemetry_analyzer.analysis.lap_performance import LapPerformanceAnalysis
from telemetry_analyzer.processing.laps import LapData

from driver_performance.models import Evidence, Observation


def analyze_driving_style(
    performance: LapPerformanceAnalysis,
    reference_lap: LapData,
    compared_lap: LapData,
) -> list[Observation]:
    """Create objective observations from a pair of analyzed laps.

    The reference lap defines the telemetry windows. Metric deltas use
    compared minus reference. This function does not infer causes.
    """
    metrics = calculate_driver_performance_metrics(
        performance,
        reference_lap,
        compared_lap,
    )
    events = analyze_event_performance(
        performance,
        reference_lap,
        compared_lap,
    )

    observations: list[Observation] = []

    groups = (
        ("braking", "Braking", metrics.braking),
        ("corner", "Corner segment", metrics.corners),
        ("acceleration", "Acceleration", metrics.acceleration),
    )

    for category, label, windows in groups:
        category_events = [
            event for event in events if event.event_type == category
        ]

        for index, window in enumerate(windows, start=1):
            evidence = _window_evidence(window)
            matching_event = _matching_event(window, category_events)

            if matching_event is not None:
                _append_event_evidence(evidence, matching_event)

            if not evidence:
                continue

            start = window.start_distance
            end = window.end_distance
            observations.append(
                Observation(
                    id=f"OBS-{category.upper()}-{index:03d}",
                    title=f"{label} telemetry differences {index}",
                    description=(
                        f"The compared lap differs from the reference lap "
                        f"within the {category} window from {start:.1f} m "
                        f"to {end:.1f} m. Evidence values are deltas "
                        f"(compared minus reference). These differences "
                        f"do not establish their cause."
                    ),
                    category=category,
                    evidence=tuple(evidence),
                )
            )

    return observations


def _window_evidence(
    window: BrakingMetrics | CornerMetrics | AccelerationMetrics,
) -> list[Evidence]:
    """Convert non-zero metric deltas into evidence records."""
    evidence: list[Evidence] = []
    start = window.start_distance
    end = window.end_distance

    for metric_name, comparison, unit in _comparisons(window):
        if comparison.delta == 0:
            continue

        evidence.append(
            Evidence(
                metric_name=metric_name,
                value=float(comparison.delta),
                unit=unit,
                distance_start_m=float(start),
                distance_end_m=float(end),
            )
        )

    return evidence


def _comparisons(window):
    """Return the comparable measurements for a telemetry window."""
    if isinstance(window, BrakingMetrics):
        return [
            ("speed_at_start_delta", window.speed_at_start, "km/h"),
            ("speed_at_end_delta", window.speed_at_end, "km/h"),
            ("minimum_speed_delta", window.minimum_speed, "km/h"),
            (
                "minimum_speed_distance_delta",
                window.minimum_speed_distance,
                "m",
            ),
            ("max_brake_delta", window.max_brake, "telemetry units"),
            (
                "average_throttle_delta",
                window.average_throttle,
                "telemetry units",
            ),
            ("duration_delta", window.duration, "s"),
        ]

    if isinstance(window, CornerMetrics):
        return [
            ("entry_speed_delta", window.entry_speed, "km/h"),
            ("exit_speed_delta", window.exit_speed, "km/h"),
            ("minimum_speed_delta", window.minimum_speed, "km/h"),
            (
                "minimum_speed_distance_delta",
                window.minimum_speed_distance,
                "m",
            ),
            ("duration_delta", window.duration, "s"),
        ]

    if isinstance(window, AccelerationMetrics):
        return [
            ("speed_at_start_delta", window.speed_at_start, "km/h"),
            ("speed_at_end_delta", window.speed_at_end, "km/h"),
            (
                "throttle_at_start_delta",
                window.throttle_at_start,
                "telemetry units",
            ),
            ("max_throttle_delta", window.max_throttle, "telemetry units"),
            (
                "throttle_at_end_delta",
                window.throttle_at_end,
                "telemetry units",
            ),
            ("duration_delta", window.duration, "s"),
        ]

    raise TypeError(f"Unsupported telemetry window: {type(window).__name__}")


def _matching_event(
    window: BrakingMetrics | CornerMetrics | AccelerationMetrics,
    events: list[PerformanceEvent],
) -> PerformanceEvent | None:
    """Find the event with the same distance boundaries."""
    for event in events:
        if (
            event.start_distance == window.start_distance
            and event.end_distance == window.end_distance
        ):
            return event
    return None


def _append_event_evidence(
    evidence: list[Evidence],
    event: PerformanceEvent,
) -> None:
    """Attach the time-delta change when it is non-zero."""
    if event.time_delta_change == 0:
        return

    evidence.append(
        Evidence(
            metric_name="time_delta_change",
            value=float(event.time_delta_change),
            unit="s",
            distance_start_m=event.start_distance,
            distance_end_m=event.end_distance,
        )
    )
