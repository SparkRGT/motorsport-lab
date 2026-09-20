import struct
from dataclasses import dataclass

from telemetry_collector.header import HEADER_SIZE


NUM_CARS = 22
CAR_TELEMETRY_DATA_SIZE = 60

PACKET_ID_CAR_TELEMETRY = 6

PACKET_SIZE = (
    HEADER_SIZE
    + (NUM_CARS * CAR_TELEMETRY_DATA_SIZE)
    + 3
)


@dataclass(frozen=True)
class CarTelemetryData:
    """Telemetría de un coche en F1 25."""

    speed: int
    throttle: float
    steering: float
    brake: float
    clutch: int
    gear: int
    engine_rpm: int
    drs: int
    rev_lights_percent: int
    rev_lights_bit_value: int
    brake_temperatures: tuple[int, int, int, int]
    tyre_surface_temperatures: tuple[int, int, int, int]
    tyre_inner_temperatures: tuple[int, int, int, int]
    engine_temperature: int
    tyre_pressures: tuple[float, float, float, float]
    surface_types: tuple[int, int, int, int]


@dataclass(frozen=True)
class CarTelemetryPacket:
    """Representa los datos principales de un Car Telemetry Packet."""

    cars: tuple[CarTelemetryData, ...]
    mfd_panel_index: int
    mfd_panel_index_secondary_player: int
    suggested_gear: int


_CAR_TELEMETRY_FORMAT = "<HfffBbHBBH4H4B4BH4f4B"


def parse_car_telemetry_data(data: bytes) -> CarTelemetryData:
    """
    Interpreta los 60 bytes correspondientes a un coche.

    Los datos utilizan Little Endian según la especificación
    UDP de F1 25.
    """

    if len(data) < CAR_TELEMETRY_DATA_SIZE:
        raise ValueError(
            "Los datos del coche son demasiado pequeños: "
            f"{len(data)} bytes."
        )

    values = struct.unpack_from(
        _CAR_TELEMETRY_FORMAT,
        data,
        0,
    )

    return CarTelemetryData(
        speed=values[0],
        throttle=values[1],
        steering=values[2],
        brake=values[3],
        clutch=values[4],
        gear=values[5],
        engine_rpm=values[6],
        drs=values[7],
        rev_lights_percent=values[8],
        rev_lights_bit_value=values[9],
        brake_temperatures=(
            values[10],
            values[11],
            values[12],
            values[13],
        ),
        tyre_surface_temperatures=(
            values[14],
            values[15],
            values[16],
            values[17],
        ),
        tyre_inner_temperatures=(
            values[18],
            values[19],
            values[20],
            values[21],
        ),
        engine_temperature=values[22],
        tyre_pressures=(
            values[23],
            values[24],
            values[25],
            values[26],
        ),
        surface_types=(
            values[27],
            values[28],
            values[29],
            values[30],
        ),
    )


def parse_car_telemetry_packet(
    data: bytes,
) -> CarTelemetryPacket:
    """
    Interpreta un Car Telemetry Packet completo de F1 25.
    """

    if len(data) != PACKET_SIZE:
        raise ValueError(
            "Tamaño de Car Telemetry Packet inesperado: "
            f"{len(data)} bytes. "
            f"Se esperaban {PACKET_SIZE} bytes."
        )

    cars: list[CarTelemetryData] = []

    for car_index in range(NUM_CARS):
        offset = (
            HEADER_SIZE
            + car_index * CAR_TELEMETRY_DATA_SIZE
        )

        car_data = parse_car_telemetry_data(
            data[offset:]
        )

        cars.append(car_data)

    mfd_offset = (
        HEADER_SIZE
        + NUM_CARS * CAR_TELEMETRY_DATA_SIZE
    )

    return CarTelemetryPacket(
        cars=tuple(cars),
        mfd_panel_index=data[mfd_offset],
        mfd_panel_index_secondary_player=data[mfd_offset + 1],
        suggested_gear=struct.unpack_from(
            "<b",
            data,
            mfd_offset + 2,
        )[0],
    )

def get_player_car_telemetry(
    packet: CarTelemetryPacket,
    player_car_index: int,
) -> CarTelemetryData:
    """
    Obtiene la telemetría del coche del jugador.

    El índice del coche proviene del header del paquete F1 25.
    """

    if not 0 <= player_car_index < NUM_CARS:
        raise ValueError(
            "Índice de coche del jugador fuera de rango: "
            f"{player_car_index}. "
            f"Debe estar entre 0 y {NUM_CARS - 1}."
        )

    return packet.cars[player_car_index]