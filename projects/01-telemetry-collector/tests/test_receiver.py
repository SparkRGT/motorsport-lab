import socket

from src.telemetry_collector.receiver import UDPReceiver


def test_receiver_receives_udp_packet() -> None:
    receiver = UDPReceiver(
        host="127.0.0.1",
        port=0,
    )

    receiver.start()

    try:
        assert receiver._socket is not None

        receiver_port = receiver._socket.getsockname()[1]

        sender = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        try:
            message = b"test-packet"

            sender.sendto(
                message,
                ("127.0.0.1", receiver_port),
            )

            packet = receiver.receive()

            assert packet.data == message
            assert packet.address == "127.0.0.1"

        finally:
            sender.close()

    finally:
        receiver.close()