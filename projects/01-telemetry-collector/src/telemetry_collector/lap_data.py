import struct
from dataclasses import dataclass

from telemetry_collector.header import (
    HEADER_SIZE,
    parse_packet_header,
)


NUM_CARS = 22
LAP_DATA_SIZE = 57
TIME_TRIAL_DATA_SIZE = 2

PACKET_ID_LAP_DATA = 2
PACKET_FORMAT_F1_25 = 2025

PACKET_SIZE = (
    HEADER_SIZE
    + (NUM_CARS * LAP_DATA_SIZE)
    + TIME_TRIAL_DATA_SIZE
)

LAP_DISTANCE_OFFSET = 20
CURRENT_LAP_NUM_OFFSET = 33


@dataclass(frozen=True)
class PlayerLapData:
    """Vuelta del coche del jugador, con la clave de sincronización."""

    lap_distance: float
    lap_number: int
    session_uid: int
    overall_frame_identifier: int
    player_car_index: int


def parse_player_lap_data(data: bytes) -> PlayerLapData:
    """
    Interpreta el Lap Data del coche del jugador en un paquete F1 25.

    El paquete completo mide 1285 bytes: header de 29 bytes, 22 coches
    de 57 bytes y 2 bytes finales de Time Trial. Solo se extraen
    ``m_lapDistance`` y ``m_currentLapNum`` del jugador.
    """

    if len(data) != PACKET_SIZE:
        raise ValueError(
            "Tamaño de Lap Data Packet inesperado: "
            f"{len(data)} bytes. "
            f"Se esperaban {PACKET_SIZE} bytes."
        )

    header = parse_packet_header(data)

    if header.packet_format != PACKET_FORMAT_F1_25:
        raise ValueError(
            "Formato de Lap Data Packet inesperado: "
            f"{header.packet_format}. "
            f"Se esperaba {PACKET_FORMAT_F1_25}."
        )

    if header.packet_id != PACKET_ID_LAP_DATA:
        raise ValueError(
            "El paquete no es Lap Data: "
            f"packet id {header.packet_id}."
        )

    if not 0 <= header.player_car_index < NUM_CARS:
        raise ValueError(
            "Índice de coche del jugador fuera de rango: "
            f"{header.player_car_index}. "
            f"Debe estar entre 0 y {NUM_CARS - 1}."
        )

    offset = (
        HEADER_SIZE
        + header.player_car_index * LAP_DATA_SIZE
    )
    record_end = offset + LAP_DATA_SIZE

    if record_end > PACKET_SIZE - TIME_TRIAL_DATA_SIZE:
        raise ValueError(
            "El registro de lap data del jugador no cabe "
            "en el paquete."
        )

    lap_distance = struct.unpack_from(
        "<f",
        data,
        offset + LAP_DISTANCE_OFFSET,
    )[0]
    lap_number = struct.unpack_from(
        "<B",
        data,
        offset + CURRENT_LAP_NUM_OFFSET,
    )[0]

    return PlayerLapData(
        lap_distance=lap_distance,
        lap_number=lap_number,
        session_uid=header.session_uid,
        overall_frame_identifier=header.overall_frame_identifier,
        player_car_index=header.player_car_index,
    )
