from pathlib import Path

import pytest

from telemetry_collector.car_telemetry import (
    CAR_TELEMETRY_DATA_SIZE,
    NUM_CARS,
    PACKET_ID_CAR_TELEMETRY,
    PACKET_SIZE,
    get_player_car_telemetry,
    parse_car_telemetry_data,
    parse_car_telemetry_packet,
)


PACKET_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "car_telemetry_packet.bin"
)


def test_car_telemetry_constants() -> None:
    assert NUM_CARS == 22
    assert CAR_TELEMETRY_DATA_SIZE == 60
    assert PACKET_ID_CAR_TELEMETRY == 6
    assert PACKET_SIZE == 1352


def test_parse_car_telemetry_packet_size() -> None:
    if not PACKET_PATH.exists():
        pytest.skip(
            "car_telemetry_packet.bin no está disponible."
        )

    data = PACKET_PATH.read_bytes()

    assert len(data) == PACKET_SIZE


def test_parse_car_telemetry_packet() -> None:
    if not PACKET_PATH.exists():
        pytest.skip(
            "car_telemetry_packet.bin no está disponible."
        )

    data = PACKET_PATH.read_bytes()

    packet = parse_car_telemetry_packet(data)

    assert len(packet.cars) == NUM_CARS


def test_parse_first_car_from_real_packet() -> None:
    if not PACKET_PATH.exists():
        pytest.skip(
            "car_telemetry_packet.bin no está disponible."
        )

    data = PACKET_PATH.read_bytes()

    first_car_data = data[29:89]

    car = parse_car_telemetry_data(
        first_car_data
    )

    assert car.speed == 0
    assert car.engine_rpm == 3920

    assert car.drs == 0
    assert car.rev_lights_percent == 0

    assert car.brake_temperatures == (
        21,
        21,
        21,
        21,
    )

    assert car.tyre_surface_temperatures == (
        70,
        70,
        70,
        70,
    )

    assert car.tyre_inner_temperatures == (
        69,
        69,
        69,
        69,
    )

    assert car.engine_temperature == 110

    assert car.tyre_pressures[0] == pytest.approx(
        21.3,
        abs=0.01,
    )

    assert car.tyre_pressures[1] == pytest.approx(
        21.3,
        abs=0.01,
    )

    assert car.tyre_pressures[2] == pytest.approx(
        24.2,
        abs=0.01,
    )

    assert car.tyre_pressures[3] == pytest.approx(
        24.2,
        abs=0.01,
    )

def test_get_player_car_telemetry() -> None:
    if not PACKET_PATH.exists():
        pytest.skip(
            "car_telemetry_packet.bin no está disponible."
        )

    data = PACKET_PATH.read_bytes()

    packet = parse_car_telemetry_packet(data)

    player_car = get_player_car_telemetry(
        packet,
        player_car_index=19,
    )

    assert player_car.speed == 0
    assert player_car.engine_rpm == 3920


def test_get_player_car_telemetry_rejects_invalid_index() -> None:
    if not PACKET_PATH.exists():
        pytest.skip(
            "car_telemetry_packet.bin no está disponible."
        )

    data = PACKET_PATH.read_bytes()

    packet = parse_car_telemetry_packet(data)

    with pytest.raises(ValueError):
        get_player_car_telemetry(
            packet,
            player_car_index=-1,
        )

    with pytest.raises(ValueError):
        get_player_car_telemetry(
            packet,
            player_car_index=NUM_CARS,
        )