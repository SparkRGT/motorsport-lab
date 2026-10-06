import struct


HEADER_FORMAT = "<HBBBBBQfIIBB"
CAR_FORMAT = "<HfffBbHBBH4H4B4BH4f4B"
LAP_CAR_FORMAT = "<" + "II" + ("HB" * 4) + "fff" + ("B" * 15) + "HHBfB"

CAR_TELEMETRY_PACKET_SIZE = 1352
LAP_DATA_PACKET_SIZE = 1285
NUM_CARS = 22


def build_header(
    *,
    packet_id: int,
    session_uid: int = 1000,
    session_time: float = 1.0,
    frame_identifier: int = 10,
    overall_frame_identifier: int = 500,
    player_car_index: int = 0,
    packet_format: int = 2025,
) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        packet_format,
        25,
        1,
        25,
        1,
        packet_id,
        session_uid,
        session_time,
        frame_identifier,
        overall_frame_identifier,
        player_car_index,
        255,
    )


def build_car_telemetry_packet(
    *,
    session_uid: int = 1000,
    session_time: float = 1.0,
    frame_identifier: int = 10,
    overall_frame_identifier: int = 500,
    player_car_index: int = 0,
    speed: int = 250,
    throttle: float = 0.8,
    brake: float = 0.0,
    steering: float = 0.1,
    gear: int = 4,
    engine_rpm: int = 11000,
    drs: int = 0,
    per_car: dict[int, dict[str, object]] | None = None,
) -> bytes:
    cars = bytearray()

    for car_index in range(NUM_CARS):
        values = {
            "speed": 0,
            "throttle": 0.0,
            "brake": 0.0,
            "steering": 0.0,
            "gear": 0,
            "engine_rpm": 0,
            "drs": 0,
        }

        if car_index == player_car_index:
            values.update(
                {
                    "speed": speed,
                    "throttle": throttle,
                    "brake": brake,
                    "steering": steering,
                    "gear": gear,
                    "engine_rpm": engine_rpm,
                    "drs": drs,
                }
            )

        if per_car is not None and car_index in per_car:
            values.update(per_car[car_index])

        cars.extend(_pack_car(values))

    packet = build_header(
        packet_id=6,
        session_uid=session_uid,
        session_time=session_time,
        frame_identifier=frame_identifier,
        overall_frame_identifier=overall_frame_identifier,
        player_car_index=player_car_index,
    )
    packet += bytes(cars)
    packet += bytes(3)

    return packet


def build_lap_data_packet(
    *,
    session_uid: int = 1000,
    session_time: float = 1.0,
    frame_identifier: int = 10,
    overall_frame_identifier: int = 500,
    player_car_index: int = 0,
    lap_distance: float = 25.0,
    lap_number: int = 1,
    packet_format: int = 2025,
    packet_id: int = 2,
    per_car: dict[int, tuple[float, int]] | None = None,
) -> bytes:
    cars = bytearray()

    for car_index in range(NUM_CARS):
        distance = 0.0
        number = 0

        if car_index == player_car_index:
            distance = lap_distance
            number = lap_number

        if per_car is not None and car_index in per_car:
            distance, number = per_car[car_index]

        cars.extend(
            _pack_lap_car(
                lap_distance=distance,
                lap_number=number,
            )
        )

    packet = build_header(
        packet_id=packet_id,
        session_uid=session_uid,
        session_time=session_time,
        frame_identifier=frame_identifier,
        overall_frame_identifier=overall_frame_identifier,
        player_car_index=player_car_index,
        packet_format=packet_format,
    )
    packet += bytes(cars)
    packet += bytes(2)

    return packet


def _pack_car(values: dict[str, object]) -> bytes:
    return struct.pack(
        CAR_FORMAT,
        values["speed"],
        values["throttle"],
        values["steering"],
        values["brake"],
        0,
        values["gear"],
        values["engine_rpm"],
        values["drs"],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0,
    )


def _pack_lap_car(
    *,
    lap_distance: float,
    lap_number: int,
) -> bytes:
    values: list[object] = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        lap_distance,
        0.0,
        0.0,
        1,
        lap_number,
        *([0] * 13),
        0,
        0,
        0,
        0.0,
        0,
    ]

    return struct.pack(LAP_CAR_FORMAT, *values)
