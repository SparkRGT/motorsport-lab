from telemetry_collector.car_telemetry import (
    get_player_car_telemetry,
    parse_car_telemetry_packet,
)
from telemetry_collector.header import parse_packet_header
from telemetry_collector.receiver import UDPReceiver


UDP_HOST = "0.0.0.0"
UDP_PORT = 20777

PACKET_ID_CAR_TELEMETRY = 6


def main() -> None:
    receiver = UDPReceiver(
        host=UDP_HOST,
        port=UDP_PORT,
    )

    print("F1 25 Telemetry Collector")
    print("==========================")
    print("UDP Receiver + Car Telemetry Parser")
    print(f"Escuchando en {UDP_HOST}:{UDP_PORT}")
    print("Esperando telemetría de F1 25...")
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

            except ValueError as error:
                print(
                    "Error al interpretar Car Telemetry Packet: "
                    f"{error}"
                )
                continue

            print(
                f"[Telemetry] "
                f"Frame: {header.frame_identifier} | "
                f"Car: {header.player_car_index} | "
                f"Speed: {player_car.speed} km/h | "
                f"Throttle: {player_car.throttle * 100:.1f}% | "
                f"Brake: {player_car.brake * 100:.1f}% | "
                f"Gear: {player_car.gear} | "
                f"RPM: {player_car.engine_rpm} | "
                f"DRS: {'ON' if player_car.drs else 'OFF'}"
            )
            

    except KeyboardInterrupt:
        print("\nReceptor detenido.")

    finally:
        receiver.close()


if __name__ == "__main__":
    main()