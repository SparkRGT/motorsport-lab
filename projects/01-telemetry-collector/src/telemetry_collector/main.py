from pathlib import Path

from telemetry_collector.car_telemetry import (
    PACKET_ID_CAR_TELEMETRY,
    get_player_car_telemetry,
    parse_car_telemetry_packet,
)
from telemetry_collector.header import parse_packet_header
from telemetry_collector.lap_data import (
    PACKET_ID_LAP_DATA,
    parse_player_lap_data,
)
from telemetry_collector.logger import TelemetryLogger
from telemetry_collector.models import AnalysisRow
from telemetry_collector.receiver import UDPReceiver
from telemetry_collector.sync import FrameSynchronizer


UDP_HOST = "0.0.0.0"
UDP_PORT = 20777

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
    synchronizer = FrameSynchronizer()

    print("F1 25 Telemetry Collector")
    print("==========================")
    print("UDP Receiver + Telemetry Logger")
    print(f"Escuchando en {UDP_HOST}:{UDP_PORT}")
    print("Esperando telemetría de F1 25...")
    print("El CSV se escribe durante la sesión.")
    print("El JSON se guarda al detener el programa.")
    print("Presiona Ctrl+C para detener.\n")

    telemetry_logger.start()

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

            try:
                if header.packet_id == PACKET_ID_CAR_TELEMETRY:
                    telemetry_packet = parse_car_telemetry_packet(
                        data
                    )
                    player_car = get_player_car_telemetry(
                        telemetry_packet,
                        header.player_car_index,
                    )
                    rows = synchronizer.add_car_telemetry(
                        header,
                        player_car,
                    )
                elif header.packet_id == PACKET_ID_LAP_DATA:
                    lap_data = parse_player_lap_data(data)
                    rows = synchronizer.add_lap_data(lap_data)
                else:
                    continue
            except ValueError as error:
                print(
                    "Error al interpretar el paquete "
                    f"{header.packet_id}: {error}"
                )
                continue

            _record_rows(telemetry_logger, rows)

    except KeyboardInterrupt:
        print("\nDeteniendo receptor...")

    finally:
        receiver.close()
        _record_rows(
            telemetry_logger,
            synchronizer.flush(),
        )

        if telemetry_logger.row_count > 0:
            json_path, csv_path = telemetry_logger.save()

            print(
                f"Filas guardadas: "
                f"{telemetry_logger.row_count}"
            )
            print(f"JSON: {json_path}")
            print(f"CSV:  {csv_path}")
        else:
            telemetry_logger.close()
            print("No se recibieron datos de telemetría.")


def _record_rows(
    telemetry_logger: TelemetryLogger,
    rows: list[AnalysisRow],
) -> None:
    for row in rows:
        telemetry_logger.write_row(row)

        print(
            f"[Telemetry] "
            f"Frame: {row.frame} | "
            f"Time: {row.session_time:.2f}s | "
            f"Lap: {row.lap_number} | "
            f"Dist: {row.lap_distance:.1f} m | "
            f"Speed: {row.speed} km/h | "
            f"Throttle: {row.throttle * 100:.1f}% | "
            f"Brake: {row.brake * 100:.1f}% | "
            f"Steering: {row.steering:.2f} | "
            f"Gear: {row.gear} | "
            f"RPM: {row.rpm} | "
            f"DRS: {'ON' if row.drs else 'OFF'}"
        )


if __name__ == "__main__":
    main()
