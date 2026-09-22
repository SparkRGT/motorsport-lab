import pytest

from telemetry_collector.car_telemetry import CarTelemetryData
from telemetry_collector.header import PacketHeader
from telemetry_collector.models import (
    TelemetrySnapshot,
    create_telemetry_snapshot,
)


def create_test_header() -> PacketHeader:
    return PacketHeader(
        packet_format=2025,
        game_year=25,
        game_major_version=1,
        game_minor_version=25,
        packet_version=1,
        packet_id=6,
        session_uid=123,
        session_time=42.5,
        frame_identifier=1000,
        overall_frame_identifier=1000,
        player_car_index=19,
        secondary_player_car_index=255,
    )


def create_test_car() -> CarTelemetryData:
    return CarTelemetryData(
        speed=250,
        throttle=1.0,
        steering=0.25,
        brake=0.0,
        clutch=0,
        gear=8,
        engine_rpm=11000,
        drs=1,
        rev_lights_percent=80,
        rev_lights_bit_value=0,
        brake_temperatures=(500, 500, 500, 500),
        tyre_surface_temperatures=(90, 90, 90, 90),
        tyre_inner_temperatures=(85, 85, 85, 85),
        engine_temperature=100,
        tyre_pressures=(22.0, 22.0, 24.0, 24.0),
        surface_types=(0, 0, 0, 0),
    )


def test_create_telemetry_snapshot() -> None:
    header = create_test_header()
    car = create_test_car()

    snapshot = create_telemetry_snapshot(
        header,
        car,
    )

    assert isinstance(snapshot, TelemetrySnapshot)
    assert snapshot.frame == 1000
    assert snapshot.session_time == 42.5
    assert snapshot.car_index == 19
    assert snapshot.speed == 250
    assert snapshot.throttle == 1.0
    assert snapshot.brake == 0.0
    assert snapshot.steering == 0.25
    assert snapshot.gear == 8
    assert snapshot.engine_rpm == 11000
    assert snapshot.drs is True


def test_telemetry_snapshot_accepts_boundary_values() -> None:
    snapshot = TelemetrySnapshot(
        frame=0,
        session_time=0.0,
        car_index=0,
        speed=0,
        throttle=0.0,
        brake=1.0,
        steering=-1.0,
        gear=-1,
        engine_rpm=0,
        drs=False,
    )

    assert snapshot.throttle == 0.0
    assert snapshot.brake == 1.0
    assert snapshot.steering == -1.0


def test_telemetry_snapshot_accepts_maximum_steering() -> None:
    snapshot = TelemetrySnapshot(
        frame=1,
        session_time=1.0,
        car_index=1,
        speed=100,
        throttle=1.0,
        brake=0.0,
        steering=1.0,
        gear=3,
        engine_rpm=8000,
        drs=False,
    )

    assert snapshot.steering == 1.0


def test_telemetry_snapshot_rejects_negative_car_index() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(
            frame=1,
            session_time=1.0,
            car_index=-1,
            speed=100,
            throttle=0.5,
            brake=0.0,
            steering=0.0,
            gear=3,
            engine_rpm=8000,
            drs=False,
        )


def test_telemetry_snapshot_rejects_invalid_throttle() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(
            frame=1,
            session_time=1.0,
            car_index=19,
            speed=100,
            throttle=1.5,
            brake=0.0,
            steering=0.0,
            gear=3,
            engine_rpm=8000,
            drs=False,
        )


def test_telemetry_snapshot_rejects_negative_throttle() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(
            frame=1,
            session_time=1.0,
            car_index=19,
            speed=100,
            throttle=-0.1,
            brake=0.0,
            steering=0.0,
            gear=3,
            engine_rpm=8000,
            drs=False,
        )


def test_telemetry_snapshot_rejects_invalid_brake() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(
            frame=1,
            session_time=1.0,
            car_index=19,
            speed=100,
            throttle=0.5,
            brake=1.5,
            steering=0.0,
            gear=3,
            engine_rpm=8000,
            drs=False,
        )


def test_telemetry_snapshot_rejects_negative_brake() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(
            frame=1,
            session_time=1.0,
            car_index=19,
            speed=100,
            throttle=0.5,
            brake=-0.1,
            steering=0.0,
            gear=3,
            engine_rpm=8000,
            drs=False,
        )


def test_telemetry_snapshot_rejects_invalid_steering() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(
            frame=1,
            session_time=1.0,
            car_index=19,
            speed=100,
            throttle=0.5,
            brake=0.0,
            steering=2.0,
            gear=3,
            engine_rpm=8000,
            drs=False,
        )


def test_telemetry_snapshot_rejects_steering_below_minimum() -> None:
    with pytest.raises(ValueError):
        TelemetrySnapshot(
            frame=1,
            session_time=1.0,
            car_index=19,
            speed=100,
            throttle=0.5,
            brake=0.0,
            steering=-2.0,
            gear=3,
            engine_rpm=8000,
            drs=False,
        )