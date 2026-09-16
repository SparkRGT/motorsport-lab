import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceivedPacket:
    """Representa un paquete UDP recibido."""

    data: bytes
    address: str
    port: int


class UDPReceiver:
    """Recibe datagramas UDP desde la red."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 20777,
        buffer_size: int = 4096,
    ) -> None:
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self._socket: socket.socket | None = None

    def start(self) -> None:
        """Crea y configura el socket UDP."""

        if self._socket is not None:
            raise RuntimeError("El receptor ya está iniciado.")

        udp_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        udp_socket.bind((self.host, self.port))

        self._socket = udp_socket

    def receive(self) -> ReceivedPacket:
        """Espera y recibe un único paquete UDP."""

        if self._socket is None:
            raise RuntimeError(
                "El receptor no está iniciado. "
                "Ejecuta start() antes de receive()."
            )

        data, address = self._socket.recvfrom(self.buffer_size)

        return ReceivedPacket(
            data=data,
            address=address[0],
            port=address[1],
        )

    def close(self) -> None:
        """Cierra el socket UDP."""

        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "UDPReceiver":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()