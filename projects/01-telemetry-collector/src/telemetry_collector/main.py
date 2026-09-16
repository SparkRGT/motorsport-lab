from telemetry_collector.receiver import UDPReceiver


def main() -> None:
    receiver = UDPReceiver(
        host="0.0.0.0",
        port=20777,
    )

    print("F1 25 Telemetry Collector")
    print("==========================")
    print("UDP Receiver")
    print("Escuchando en 0.0.0.0:20777")
    print("Esperando paquetes de F1 25...")
    print("Presiona Ctrl+C para detener.\n")

    try:
        receiver.start()

        while True:
            packet = receiver.receive()

            print(
                f"Paquete recibido de "
                f"{packet.address}:{packet.port} "
                f"| bytes: {len(packet.data)}"
            )

    except KeyboardInterrupt:
        print("\nReceptor detenido.")

    finally:
        receiver.close()


if __name__ == "__main__":
    main()