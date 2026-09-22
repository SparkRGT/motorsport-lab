from dataclasses import dataclass

from telemetry_collector.car_telemetry import CarTelemetryData
from telemetry_collector.header import PacketHeader


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Representa una muestra de telemetría de un coche."""

    frame: int
    session_time: float
    car_index: int

    speed: int
    throttle: float
    brake: float
    steering: float

    gear: int
    engine_rpm: int

    drs: bool

    def __post_init__(self) -> None:
        if self.car_index < 0:
            raise ValueError(
                "El índice del coche no puede ser negativo."
            )

        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError(
                "Throttle debe estar entre 0.0 y 1.0."
            )

        if not 0.0 <= self.brake <= 1.0:
            raise ValueError(
                "Brake debe estar entre 0.0 y 1.0."
            )

        if not -1.0 <= self.steering <= 1.0:
            raise ValueError(
                "Steering debe estar entre -1.0 y 1.0."
            )

def create_telemetry_snapshot(
    header: PacketHeader,
    car: CarTelemetryData,
) -> TelemetrySnapshot:
    """Construye un snapshot de telemetría a partir de los datos parseados."""

    return TelemetrySnapshot(
        frame=header.frame_identifier,
        session_time=header.session_time,
        car_index=header.player_car_index,
        speed=car.speed,
        throttle=car.throttle,
        brake=car.brake,
        steering=car.steering,
        gear=car.gear,
        engine_rpm=car.engine_rpm,
        drs=bool(car.drs),
    )