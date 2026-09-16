from pathlib import Path

import pytest

from telemetry_collector.header import (
    HEADER_SIZE,
    parse_packet_header,
)


def test_header_size() -> None:
    assert HEADER_SIZE == 29

def test_parse_header_rejects_short_packet() -> None:
    data = b"\x00" * 10

    with pytest.raises(ValueError):
        parse_packet_header(data)

def test_parse_real_f1_25_packet() -> None:
    packet_path = Path("first_packet.bin")

    if not packet_path.exists():
        pytest.skip("first_packet.bin no está disponible.")

    data = packet_path.read_bytes()

    header = parse_packet_header(data)

    assert header.packet_format == 2025
    assert header.game_year == 25
    assert header.game_major_version == 1
    assert header.game_minor_version == 25
    assert header.packet_version == 1

    assert header.packet_id == 3

    assert header.frame_identifier == 1265
    assert header.overall_frame_identifier == 1265

    assert header.player_car_index == 19
    assert header.secondary_player_car_index == 255