import socket


HOST = "0.0.0.0"
PORT = 20777
BUFFER_SIZE = 4096


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.bind((HOST, PORT))

        print(f"Escuchando UDP en {HOST}:{PORT}")
        print("Esperando el primer paquete de F1 25...")
        print("Presiona Ctrl+C para detener.\n")

        data, address = sock.recvfrom(BUFFER_SIZE)

        print(f"Paquete recibido de {address[0]}:{address[1]}")
        print(f"Tamaño: {len(data)} bytes")
        print("\nPrimeros 32 bytes:")

        print(data[:32].hex(" "))

        with open("first_packet.bin", "wb") as file:
            file.write(data)

        print("\nPaquete guardado como: first_packet.bin")

    except KeyboardInterrupt:
        print("\nReceptor detenido.")

    finally:
        sock.close()


if __name__ == "__main__":
    main()