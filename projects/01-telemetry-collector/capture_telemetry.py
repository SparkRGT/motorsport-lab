import socket


HOST = "0.0.0.0"
PORT = 20777
BUFFER_SIZE = 4096


def main() -> None:
    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    udp_socket.bind((HOST, PORT))

    print("F1 25 Telemetry Capture")
    print("=======================")
    print(f"Escuchando en {HOST}:{PORT}")
    print("Esperando un Car Telemetry Packet (ID 6)...")
    print("Presiona Ctrl+C para detener.\n")

    try:
        while True:
            data, address = udp_socket.recvfrom(BUFFER_SIZE)

            if len(data) < 7:
                continue

            packet_id = data[6]

            print(
                f"Paquete recibido | "
                f"ID: {packet_id} | "
                f"bytes: {len(data)} | "
                f"origen: {address[0]}:{address[1]}"
            )

            if packet_id == 6:
                with open("car_telemetry_packet.bin", "wb") as file:
                    file.write(data)

                print()
                print("========================================")
                print("Car Telemetry Packet encontrado.")
                print("Guardado como:")
                print("car_telemetry_packet.bin")
                print("========================================")

                break

    except KeyboardInterrupt:
        print("\nCaptura detenida.")

    finally:
        udp_socket.close()


if __name__ == "__main__":
    main()