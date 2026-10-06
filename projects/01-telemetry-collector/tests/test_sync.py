import pytest

from packets import (
    build_car_telemetry_packet,
    build_lap_data_packet,
)
from telemetry_collector.car_telemetry import (
    get_player_car_telemetry,
    parse_car_telemetry_packet,
)
from telemetry_collector.header import parse_packet_header
from telemetry_collector.lap_data import parse_player_lap_data
from telemetry_collector.sync import FrameSynchronizer


def _add_telemetry(sync: FrameSynchronizer, packet: bytes):
    header = parse_packet_header(packet)
    telemetry = parse_car_telemetry_packet(packet)
    car = get_player_car_telemetry(
        telemetry,
        header.player_car_index,
    )

    return sync.add_car_telemetry(header, car)


def _add_lap(sync: FrameSynchronizer, packet: bytes):
    return sync.add_lap_data(parse_player_lap_data(packet))


def _pair(
    sync: FrameSynchronizer,
    *,
    session_uid: int = 1000,
    overall_frame_identifier: int = 500,
    frame_identifier: int = 10,
    player_car_index: int = 0,
    session_time: float = 12.5,
    lap_distance: float = 40.0,
    lap_number: int = 2,
    gear: int = 5,
    engine_rpm: int = 12000,
    drs: int = 1,
    speed: int = 280,
    telemetry_first: bool = True,
):
    telemetry = build_car_telemetry_packet(
        session_uid=session_uid,
        session_time=session_time,
        frame_identifier=frame_identifier,
        overall_frame_identifier=overall_frame_identifier,
        player_car_index=player_car_index,
        speed=speed,
        gear=gear,
        engine_rpm=engine_rpm,
        drs=drs,
    )
    lap = build_lap_data_packet(
        session_uid=session_uid,
        session_time=session_time,
        frame_identifier=frame_identifier,
        overall_frame_identifier=overall_frame_identifier,
        player_car_index=player_car_index,
        lap_distance=lap_distance,
        lap_number=lap_number,
    )

    if telemetry_first:
        first = _add_telemetry(sync, telemetry)
        second = _add_lap(sync, lap)
    else:
        first = _add_lap(sync, lap)
        second = _add_telemetry(sync, telemetry)

    return first, second


def test_matching_packets_create_one_complete_row() -> None:
    sync = FrameSynchronizer()

    first, second = _pair(
        sync,
        frame_identifier=15,
        overall_frame_identifier=800,
        engine_rpm=11500,
        drs=1,
        gear=-1,
    )

    assert first == []
    assert len(second) == 1

    row = second[0]

    assert row.frame == 800
    assert row.frame != 15
    assert row.rpm == 11500
    assert row.drs == 1
    assert row.gear == -1
    assert row.lap_number == 2
    assert row.lap_distance == pytest.approx(40.0)
    assert row.session_time == pytest.approx(12.5)


def test_lap_packet_before_telemetry_still_creates_the_row() -> None:
    sync = FrameSynchronizer()

    first, second = _pair(sync, telemetry_first=False)

    assert first == []
    assert len(second) == 1


def test_telemetry_without_lap_data_creates_no_row() -> None:
    sync = FrameSynchronizer()
    packet = build_car_telemetry_packet()

    assert _add_telemetry(sync, packet) == []


def test_lap_data_without_telemetry_creates_no_row() -> None:
    sync = FrameSynchronizer()
    packet = build_lap_data_packet()

    assert _add_lap(sync, packet) == []


def test_different_session_uid_values_are_not_mixed() -> None:
    sync = FrameSynchronizer()

    telemetry = build_car_telemetry_packet(session_uid=1)
    lap = build_lap_data_packet(session_uid=2)

    assert _add_telemetry(sync, telemetry) == []
    assert _add_lap(sync, lap) == []


def test_different_overall_frames_are_not_mixed() -> None:
    sync = FrameSynchronizer()

    telemetry = build_car_telemetry_packet(
        overall_frame_identifier=10,
        frame_identifier=10,
    )
    lap = build_lap_data_packet(
        overall_frame_identifier=11,
        frame_identifier=10,
    )

    assert _add_telemetry(sync, telemetry) == []
    assert _add_lap(sync, lap) == []


def test_same_overall_frame_matches_with_different_frame_identifier() -> None:
    sync = FrameSynchronizer()

    telemetry = build_car_telemetry_packet(
        overall_frame_identifier=70,
        frame_identifier=1,
    )
    lap = build_lap_data_packet(
        overall_frame_identifier=70,
        frame_identifier=99,
        lap_number=3,
    )

    assert _add_telemetry(sync, telemetry) == []
    rows = _add_lap(sync, lap)

    assert len(rows) == 1
    assert rows[0].frame == 70
    assert rows[0].lap_number == 3


def test_different_player_car_index_values_are_not_mixed() -> None:
    sync = FrameSynchronizer()

    telemetry = build_car_telemetry_packet(player_car_index=1)
    lap = build_lap_data_packet(player_car_index=2)

    assert _add_telemetry(sync, telemetry) == []
    assert _add_lap(sync, lap) == []


def test_negative_lap_distance_is_not_emitted() -> None:
    sync = FrameSynchronizer()

    _first, second = _pair(sync, lap_distance=-0.5)

    assert second == []


def test_negative_lap_distance_does_not_block_the_next_frame() -> None:
    sync = FrameSynchronizer()

    _pair(sync, overall_frame_identifier=1, lap_distance=-2.0)
    _first, second = _pair(
        sync,
        overall_frame_identifier=2,
        lap_distance=5.0,
        session_time=2.0,
    )

    assert len(second) == 1
    assert second[0].frame == 2
    assert second[0].lap_distance == pytest.approx(5.0)


def test_newer_frame_waits_for_an_older_pending_frame() -> None:
    sync = FrameSynchronizer()

    older_telemetry = build_car_telemetry_packet(
        overall_frame_identifier=1,
        session_time=1.0,
    )
    assert _add_telemetry(sync, older_telemetry) == []

    _first, newer = _pair(
        sync,
        overall_frame_identifier=2,
        session_time=2.0,
        lap_number=1,
    )

    assert newer == []

    older_lap = build_lap_data_packet(
        overall_frame_identifier=1,
        session_time=1.0,
        lap_number=1,
        lap_distance=8.0,
    )
    rows = _add_lap(sync, older_lap)

    assert [row.frame for row in rows] == [1, 2]


def test_drs_is_exported_as_zero_or_one() -> None:
    sync = FrameSynchronizer()

    _first, disabled = _pair(
        sync,
        overall_frame_identifier=1,
        drs=0,
    )
    _first, enabled = _pair(
        sync,
        overall_frame_identifier=2,
        drs=1,
        session_time=2.0,
    )

    assert disabled[0].drs == 0
    assert enabled[0].drs == 1


def test_pending_buffer_drops_incomplete_frames_without_inventing_rows() -> None:
    sync = FrameSynchronizer(max_pending=1)

    first = build_car_telemetry_packet(overall_frame_identifier=1)
    second = build_car_telemetry_packet(overall_frame_identifier=2)

    assert _add_telemetry(sync, first) == []
    assert _add_telemetry(sync, second) == []

    late_lap = build_lap_data_packet(overall_frame_identifier=1)
    matching_lap = build_lap_data_packet(
        overall_frame_identifier=2,
        lap_distance=12.0,
        lap_number=1,
    )

    assert _add_lap(sync, late_lap) == []
    rows = _add_lap(sync, matching_lap)

    assert len(rows) == 1
    assert rows[0].frame == 2
