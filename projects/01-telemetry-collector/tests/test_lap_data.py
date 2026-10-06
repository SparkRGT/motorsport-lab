import struct

import pytest

from packets import (
    LAP_CAR_FORMAT,
    LAP_DATA_PACKET_SIZE,
    build_lap_data_packet,
)
from telemetry_collector.lap_data import (
    NUM_CARS,
    PACKET_ID_LAP_DATA,
    PACKET_SIZE,
    parse_player_lap_data,
)


def test_lap_car_record_is_57_bytes() -> None:
    assert struct.calcsize(LAP_CAR_FORMAT) == 57


def test_valid_lap_data_packet_is_decoded() -> None:
    packet = build_lap_data_packet(
        session_uid=42,
        overall_frame_identifier=900,
        player_car_index=3,
        lap_distance=123.5,
        lap_number=4,
    )

    assert len(packet) == PACKET_SIZE
    assert PACKET_SIZE == LAP_DATA_PACKET_SIZE
    assert PACKET_ID_LAP_DATA == 2
    assert NUM_CARS == 22

    lap = parse_player_lap_data(packet)

    assert lap.lap_distance == pytest.approx(123.5)
    assert lap.lap_number == 4
    assert lap.session_uid == 42
    assert lap.overall_frame_identifier == 900
    assert lap.player_car_index == 3


def test_wrong_packet_id_is_rejected() -> None:
    packet = build_lap_data_packet(packet_id=6)

    with pytest.raises(ValueError, match="no es Lap Data"):
        parse_player_lap_data(packet)


def test_truncated_packet_is_rejected() -> None:
    packet = build_lap_data_packet()[:100]

    with pytest.raises(ValueError, match="Tamaño de Lap Data Packet"):
        parse_player_lap_data(packet)


def test_unexpected_packet_format_is_rejected() -> None:
    packet = build_lap_data_packet(packet_format=2024)

    with pytest.raises(ValueError, match="Formato de Lap Data Packet"):
        parse_player_lap_data(packet)


def test_player_car_index_selects_the_matching_car() -> None:
    packet = build_lap_data_packet(
        player_car_index=7,
        per_car={
            0: (10.0, 1),
            7: (88.25, 6),
            21: (400.0, 9),
        },
    )

    lap = parse_player_lap_data(packet)

    assert lap.player_car_index == 7
    assert lap.lap_distance == pytest.approx(88.25)
    assert lap.lap_number == 6


def test_player_car_index_out_of_range_is_rejected() -> None:
    packet = build_lap_data_packet(player_car_index=22)

    with pytest.raises(ValueError, match="fuera de rango"):
        parse_player_lap_data(packet)
