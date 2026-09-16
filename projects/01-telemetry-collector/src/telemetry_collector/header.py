from dataclasses import dataclass
import struct


HEADER_SIZE = 29


@dataclass(frozen=True)
class PacketHeader:
    """Representa el header de un paquete UDP de F1 25."""

    packet_format: int
    game_year: int
    game_major_version: int
    game_minor_version: int
    packet_version: int
    packet_id: int
    session_uid: int
    session_time: float
    frame_identifier: int
    overall_frame_identifier: int
    player_car_index: int
    secondary_player_car_index: int


def parse_packet_header(data: bytes) -> PacketHeader:
    """
    Extrae el header de 29 bytes de un paquete F1 25.

    F1 25 utiliza Little Endian para los datos del protocolo.
    """

    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"El paquete es demasiado pequeño para contener "
            f"el header de F1 25: {len(data)} bytes."
        )

    (
        packet_format,
        game_year,
        game_major_version,
        game_minor_version,
        packet_version,
        packet_id,
        session_uid,
        session_time,
        frame_identifier,
        overall_frame_identifier,
        player_car_index,
        secondary_player_car_index,
    ) = struct.unpack_from(
        "<HBBBBBQfIIBB",
        data,
        0,
    )

    return PacketHeader(
        packet_format=packet_format,
        game_year=game_year,
        game_major_version=game_major_version,
        game_minor_version=game_minor_version,
        packet_version=packet_version,
        packet_id=packet_id,
        session_uid=session_uid,
        session_time=session_time,
        frame_identifier=frame_identifier,
        overall_frame_identifier=overall_frame_identifier,
        player_car_index=player_car_index,
        secondary_player_car_index=secondary_player_car_index,
    )