from pathlib import Path

from telemetry_collector.car_telemetry import (
    get_player_car_telemetry,
    parse_car_telemetry_packet,
)
from telemetry_collector.header import parse_packet_header
from telemetry_collector.logger import TelemetryLogger
from telemetry_collector.models import create_telemetry_snapshot
from telemetry_collector.receiver import UDPReceiver


UDP_HOST = "0.0.0.0"
UDP_PORT = 20777

PACKET_ID_CAR_TELEMETRY = 6

DATA_DIRECTORY = Path("data") / "telemetry"


def main() -> None:
    receiver = UDPReceiver(
        host=UDP_HOST,
        port=UDP_PORT,
    )

    telemetry_logger = TelemetryLogger(
        output_directory=DATA_DIRECTORY,
        filename="telemetry_session",
    )

    print("F1 25 Telemetry Collector")
    print("==========================")
    print("UDP Receiver + Telemetry Logger")
    print(f"Escuchando en {UDP_HOST}:{UDP_PORT}")
    print("Esperando telemetría de F1 25...")
    print("Los datos se guardarán al detener el programa.")
    print("Presiona Ctrl+C para detener.\n")

    try:
        receiver.start()

        while True:
            received_packet = receiver.receive()

            data = received_packet.data

            if len(data) < 29:
                print(
                    "Paquete ignorado: tamaño insuficiente "
                    "para el header."
                )
                continue

            header = parse_packet_header(data)

            if header.packet_id != PACKET_ID_CAR_TELEMETRY:
                continue

            try:
                telemetry_packet = parse_car_telemetry_packet(
                    data
                )

                player_car = get_player_car_telemetry(
                    telemetry_packet,
                    header.player_car_index,
                )

                snapshot = create_telemetry_snapshot(
                    header,
                    player_car,
                )

                telemetry_logger.add(snapshot)

            except ValueError as error:
                print(
                    "Error al interpretar "
                    "Car Telemetry Packet: "
                    f"{error}"
                )
                continue

            print(
                f"[Telemetry] "
                f"Frame: {snapshot.frame} | "
                f"Time: {snapshot.session_time:.2f}s | "
                f"Car: {snapshot.car_index} | "
                f"Speed: {snapshot.speed} km/h | "
                f"Throttle: {snapshot.throttle * 100:.1f}% | "
                f"Brake: {snapshot.brake * 100:.1f}% | "
                f"Steering: {snapshot.steering:.2f} | "
                f"Gear: {snapshot.gear} | "
                f"RPM: {snapshot.engine_rpm} | "
                f"DRS: {'ON' if snapshot.drs else 'OFF'}"
            )

    except KeyboardInterrupt:
        print("\nDeteniendo receptor...")

    finally:
        receiver.close()

        if telemetry_logger.snapshot_count > 0:
            json_path, csv_path = telemetry_logger.save()

            print(
                f"Snapshots guardados: "
                f"{telemetry_logger.snapshot_count}"
            )
            print(f"JSON: {json_path}")
            print(f"CSV:  {csv_path}")
        else:
            print("No se recibieron datos de telemetría.")


if __name__ == "__main__":
    main()