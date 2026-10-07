"""Lap analysis, session summary, zones, corners, and time-delta changes."""

from telemetry_analyzer.analysis.corners import CornerSegment, detect_corner_segments
from telemetry_analyzer.analysis.distance import align_laps_by_distance
from telemetry_analyzer.analysis.lap_performance import (
    LapPerformanceAnalysis,
    analyze_lap_performance,
)
from telemetry_analyzer.analysis.performance import (
    PerformanceSegment,
    detect_performance_segments,
)
from telemetry_analyzer.analysis.session import SessionSummary, analyze_session
from telemetry_analyzer.analysis.time_delta import calculate_time_delta
from telemetry_analyzer.analysis.zones import (
    AccelerationZone,
    BrakingZone,
    detect_acceleration_zones,
    detect_braking_zones,
)

__all__ = [
    "AccelerationZone",
    "BrakingZone",
    "CornerSegment",
    "LapPerformanceAnalysis",
    "PerformanceSegment",
    "SessionSummary",
    "align_laps_by_distance",
    "analyze_lap_performance",
    "analyze_session",
    "calculate_time_delta",
    "detect_acceleration_zones",
    "detect_braking_zones",
    "detect_corner_segments",
    "detect_performance_segments",
]
