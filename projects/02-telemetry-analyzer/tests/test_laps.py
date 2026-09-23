from pathlib import Path

import pandas as pd
import pytest

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import LapData, segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import validate_dataset


LAPS_DATASET = (
    Path(__file__).parent.parent / "data" / "sample" / "telemetry_laps_sample.csv"
)

COLUMNS = [
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


def _telemetry(samples: list[tuple[float, int, int, float]]) -> pd.DataFrame:
    rows = []
    for index, (session_time, lap_number, frame, lap_distance) in enumerate(samples):
        rows.append(
            {
                "session_time": session_time,
                "frame": frame,
                "lap_number": lap_number,
                "lap_distance": lap_distance,
                "speed": 80 + index,
                "throttle": 0.5,
                "brake": 0.0,
                "steering": 0.0,
                "gear": 3,
                "rpm": 8000,
                "drs": 0,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _timed_laps() -> pd.DataFrame:
    return _telemetry(
        [
            (10.0, 1, 1, 0.0),
            (10.1, 1, 2, 1.0),
            (10.2, 1, 3, 2.0),
            (10.3, 1, 4, 3.0),
            (10.4, 2, 5, 0.0),
            (10.8, 2, 6, 4.0),
        ]
    )


def test_segment_laps_detects_existing_laps():
    frame = load_csv(LAPS_DATASET)
    laps = segment_laps(frame)

    assert list(laps) == [1, 2, 3]
    assert all(isinstance(lap, LapData) for lap in laps.values())


def test_segment_laps_returns_single_lap():
    frame = _telemetry(
        [
            (0.0, 1, 1, 0.0),
            (0.1, 1, 2, 1.0),
            (0.2, 1, 3, 2.0),
        ]
    )
    laps = segment_laps(frame)

    assert list(laps) == [1]
    assert isinstance(laps[1], LapData)


def test_segment_laps_keeps_all_samples():
    frame = _timed_laps()
    laps = segment_laps(frame)

    assert sum(lap.sample_count for lap in laps.values()) == len(frame)
    for lap_number, lap in laps.items():
        expected = frame.loc[frame["lap_number"] == lap_number]
        pd.testing.assert_frame_equal(lap.data, expected)


def test_segment_laps_preserves_temporal_order():
    frame = _timed_laps()
    laps = segment_laps(frame)

    for lap_number, lap in laps.items():
        expected = frame.loc[frame["lap_number"] == lap_number, "session_time"]
        assert lap.data["session_time"].tolist() == expected.tolist()
        assert lap.data["session_time"].is_monotonic_increasing


def test_segment_laps_preserves_lap_number():
    frame = _timed_laps()
    laps = segment_laps(frame)

    for lap_number, lap in laps.items():
        assert lap.lap_number == lap_number
        assert set(lap.data["lap_number"]) == {lap_number}


def test_segment_laps_start_time_is_first_sample():
    laps = segment_laps(_timed_laps())

    assert laps[1].start_time == 10.0
    assert laps[2].start_time == 10.4


def test_segment_laps_end_time_is_last_sample():
    laps = segment_laps(_timed_laps())

    assert laps[1].end_time == 10.3
    assert laps[2].end_time == 10.8


def test_segment_laps_duration_is_end_minus_start():
    laps = segment_laps(_timed_laps())

    assert laps[1].duration == laps[1].end_time - laps[1].start_time
    assert laps[1].duration == pytest.approx(0.3)
    assert laps[2].duration == laps[2].end_time - laps[2].start_time
    assert laps[2].duration == pytest.approx(0.4)


def test_segment_laps_sample_count_matches_rows():
    laps = segment_laps(_timed_laps())

    assert laps[1].sample_count == 4
    assert laps[1].sample_count == len(laps[1].data)
    assert laps[2].sample_count == 2
    assert laps[2].sample_count == len(laps[2].data)


def test_segment_laps_single_sample_has_zero_duration():
    frame = _telemetry([(5.0, 4, 10, 0.0)])
    laps = segment_laps(frame)

    assert list(laps) == [4]
    assert laps[4].sample_count == 1
    assert laps[4].start_time == 5.0
    assert laps[4].end_time == 5.0
    assert laps[4].duration == 0.0


def test_segment_laps_does_not_invent_missing_laps():
    frame = _telemetry(
        [
            (0.0, 1, 1, 0.0),
            (0.1, 1, 2, 1.0),
            (0.2, 3, 3, 0.0),
            (0.3, 3, 4, 1.0),
        ]
    )
    laps = segment_laps(frame)

    assert list(laps) == [1, 3]
    assert 2 not in laps


def test_segment_laps_does_not_modify_original():
    frame = _timed_laps()
    original = frame.copy()

    segment_laps(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_lap_data_is_independent_copy():
    frame = _timed_laps()
    original = frame.copy()
    laps = segment_laps(frame)
    target = laps[1].data.index[0]

    laps[1].data.loc[target, "speed"] = 9999

    pd.testing.assert_frame_equal(frame, original)
    assert laps[1].data is not frame


def test_pipeline_segments_multi_lap_dataset():
    frame = load_csv(LAPS_DATASET)
    validate_dataset(frame)
    normalized = normalize_dataset(frame)
    laps = segment_laps(normalized)

    assert list(laps) == [1, 2, 3]
    assert [lap.sample_count for lap in laps.values()] == [4, 4, 4]
    assert sum(lap.sample_count for lap in laps.values()) == len(normalized)

    for lap_number, lap in laps.items():
        expected = normalized.loc[normalized["lap_number"] == lap_number]
        assert isinstance(lap, LapData)
        assert lap.lap_number == lap_number
        assert lap.sample_count == len(lap.data) == len(expected)
        assert lap.start_time == float(expected["session_time"].iloc[0])
        assert lap.end_time == float(expected["session_time"].iloc[-1])
        assert lap.duration == lap.end_time - lap.start_time
        assert lap.data["session_time"].tolist() == expected["session_time"].tolist()
