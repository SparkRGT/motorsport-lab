import csv

from packets import (
    build_car_telemetry_packet,
    build_lap_data_packet,
)
from telemetry_collector.logger import TelemetryLogger
from telemetry_collector.models import AnalysisRow
from telemetry_collector.sync import FrameSynchronizer
from test_sync import (
    _add_lap,
    _add_telemetry,
    _pair,
)


def _frames_of(rows: list[AnalysisRow]) -> list[int]:
    return [row.frame for row in rows]


def _emit(
    sync: FrameSynchronizer,
    frame: int,
    *,
    session_uid: int = 1000,
    lap_number: int = 1,
    lap_distance: float | None = None,
    telemetry_first: bool = True,
    speed: int = 200,
) -> list[AnalysisRow]:
    if lap_distance is None:
        lap_distance = float(frame)

    first, second = _pair(
        sync,
        session_uid=session_uid,
        overall_frame_identifier=frame,
        frame_identifier=frame + 50,
        session_time=frame * 0.05,
        lap_number=lap_number,
        lap_distance=lap_distance,
        telemetry_first=telemetry_first,
        speed=speed,
    )

    return [*first, *second]


def _feed_sequence(
    sync: FrameSynchronizer,
    frames: list[int],
    **kwargs,
) -> list[AnalysisRow]:
    rows: list[AnalysisRow] = []

    for frame in frames:
        rows.extend(_emit(sync, frame, **kwargs))

    return rows


def test_reordered_frame_is_written_in_frame_order() -> None:
    sync = FrameSynchronizer()

    rows = _feed_sequence(
        sync,
        [1282, 1283, 1285, 1284, 1286],
    )

    assert _frames_of(rows) == [1282, 1283, 1284, 1285, 1286]
    assert all(
        later.session_time >= earlier.session_time
        for earlier, later in zip(rows, rows[1:])
    )
    assert all(
        later.lap_distance >= earlier.lap_distance
        for earlier, later in zip(rows, rows[1:])
    )


def test_reordered_frame_accepts_lap_data_before_telemetry() -> None:
    sync = FrameSynchronizer()

    rows = _feed_sequence(sync, [1282, 1283, 1285])
    rows.extend(
        _emit(
            sync,
            1284,
            telemetry_first=False,
        )
    )

    assert _frames_of(rows) == [1282, 1283, 1284, 1285]


def test_reordered_frame_accepts_telemetry_before_lap_data() -> None:
    sync = FrameSynchronizer()

    rows = _feed_sequence(sync, [10, 11])
    telemetry = build_car_telemetry_packet(
        overall_frame_identifier=13,
        session_time=13 * 0.05,
    )
    assert _add_telemetry(sync, telemetry) == []

    lap = build_lap_data_packet(
        overall_frame_identifier=13,
        session_time=13 * 0.05,
        lap_number=1,
        lap_distance=13.0,
    )
    held = _add_lap(sync, lap)
    released = _emit(sync, 12, telemetry_first=False)

    assert held == []
    assert _frames_of([*rows, *released]) == [10, 11, 12, 13]


def test_two_frame_reorder_matches_the_real_session_pattern() -> None:
    sync = FrameSynchronizer()

    rows = _feed_sequence(
        sync,
        [1756, 1757, 1760, 1758, 1759, 1761, 1762],
    )

    assert _frames_of(rows) == [
        1756,
        1757,
        1758,
        1759,
        1760,
        1761,
        1762,
    ]


def test_incomplete_frame_is_not_emitted() -> None:
    sync = FrameSynchronizer()

    rows = _emit(sync, 20)
    telemetry = build_car_telemetry_packet(
        overall_frame_identifier=21,
        session_time=21 * 0.05,
    )

    assert _add_telemetry(sync, telemetry) == []
    assert _frames_of(rows) == [20]
    assert _frames_of(sync.flush()) == []


def test_lost_frame_remains_a_gap_after_the_reorder_window() -> None:
    sync = FrameSynchronizer(reorder_window=8)
    rows = _emit(sync, 30)

    for frame in range(32, 40):
        rows.extend(_emit(sync, frame))

    assert _frames_of(rows) == [30, *range(32, 40)]
    assert 31 not in _frames_of(rows)

    late = _emit(sync, 31)

    assert late == []
    assert _frames_of(rows) == [30, *range(32, 40)]


def test_forward_frame_jump_is_preserved() -> None:
    sync = FrameSynchronizer(reorder_window=2)

    rows = _emit(sync, 7)
    rows.extend(_emit(sync, 9))
    rows.extend(_emit(sync, 10))

    assert _frames_of(rows) == [7, 9, 10]


def test_duplicate_frame_is_not_written_twice() -> None:
    sync = FrameSynchronizer()

    first = _emit(sync, 4)
    second = _emit(sync, 4)

    assert _frames_of(first) == [4]
    assert second == []


def test_full_pending_buffer_drops_the_old_incomplete_frame() -> None:
    sync = FrameSynchronizer(max_pending=1)
    rows = _emit(sync, 1)

    assert _add_telemetry(
        sync,
        build_car_telemetry_packet(
            overall_frame_identifier=2,
            session_time=2 * 0.05,
        ),
    ) == []

    held = _emit(sync, 3)
    rows.extend(held)
    rows.extend(
        _add_telemetry(
            sync,
            build_car_telemetry_packet(
                overall_frame_identifier=4,
                session_time=4 * 0.05,
            ),
        )
    )

    assert 2 not in _frames_of(rows)
    assert _frames_of(rows) == [1, 3]
    assert _emit(sync, 2) == []


def test_flush_releases_a_held_frame_without_inventing_the_gap() -> None:
    sync = FrameSynchronizer(reorder_window=8)

    rows = _emit(sync, 50)
    assert _emit(sync, 52) == []

    released = sync.flush()

    assert _frames_of([*rows, *released]) == [50, 52]
    assert _emit(sync, 51) == []


def test_lap_change_keeps_frame_order_and_allows_a_new_lap_distance() -> None:
    sync = FrameSynchronizer()

    rows = [
        *_emit(sync, 100, lap_number=1, lap_distance=500.0),
        *_emit(sync, 101, lap_number=1, lap_distance=520.0),
        *_emit(sync, 102, lap_number=2, lap_distance=3.0),
    ]

    assert _frames_of(rows) == [100, 101, 102]
    assert [row.lap_number for row in rows] == [1, 1, 2]
    assert rows[2].lap_distance < rows[1].lap_distance
    lap_one = [row.lap_distance for row in rows if row.lap_number == 1]
    assert lap_one == sorted(lap_one)


def test_new_session_uid_does_not_mix_or_duplicate_rows() -> None:
    sync = FrameSynchronizer()

    old_session = [
        *_emit(sync, 5, session_uid=1, speed=100, lap_number=1),
        *_emit(sync, 6, session_uid=1, speed=110, lap_number=1),
    ]
    new_session = [
        *_emit(sync, 1, session_uid=2, speed=50, lap_number=4),
        *_emit(sync, 2, session_uid=2, speed=55, lap_number=4),
    ]

    assert _frames_of(old_session) == [5, 6]
    assert [row.speed for row in old_session] == [100, 110]
    assert _frames_of(new_session) == [1, 2]
    assert [row.speed for row in new_session] == [50, 55]
    assert _emit(sync, 5, session_uid=1, speed=999) == []


def test_emitted_rows_reach_the_csv_in_frame_order(tmp_path) -> None:
    sync = FrameSynchronizer()
    logger = TelemetryLogger(
        output_directory=tmp_path,
        filename="telemetry_session",
    )

    for row in _feed_sequence(sync, [1282, 1283, 1285, 1284, 1286]):
        logger.write_row(row)

    for row in sync.flush():
        logger.write_row(row)

    with logger.save_csv().open(
        encoding="utf-8",
        newline="",
    ) as file:
        stored = list(csv.DictReader(file))

    frames = [int(row["frame"]) for row in stored]
    times = [float(row["session_time"]) for row in stored]
    distances = [float(row["lap_distance"]) for row in stored]

    assert frames == [1282, 1283, 1284, 1285, 1286]
    assert all(later >= earlier for earlier, later in zip(times, times[1:]))
    assert all(
        later >= earlier
        for earlier, later in zip(distances, distances[1:])
    )


def test_reorder_window_must_be_positive() -> None:
    try:
        FrameSynchronizer(reorder_window=0)
    except ValueError as error:
        assert "reorder_window" in str(error)
    else:
        raise AssertionError("reorder_window=0 debería rechazarse")
