from dataclasses import dataclass

from telemetry_collector.car_telemetry import CarTelemetryData
from telemetry_collector.header import PacketHeader
from telemetry_collector.lap_data import PlayerLapData
from telemetry_collector.models import AnalysisRow


SyncKey = tuple[int, int, int]


@dataclass
class _PartialFrame:
    session_time: float | None = None
    speed: int | None = None
    throttle: float | None = None
    brake: float | None = None
    steering: float | None = None
    gear: int | None = None
    engine_rpm: int | None = None
    drs: int | None = None
    lap_distance: float | None = None
    lap_number: int | None = None

    def is_complete(self) -> bool:
        return (
            self.session_time is not None
            and self.speed is not None
            and self.throttle is not None
            and self.brake is not None
            and self.steering is not None
            and self.gear is not None
            and self.engine_rpm is not None
            and self.drs is not None
            and self.lap_distance is not None
            and self.lap_number is not None
        )


@dataclass(frozen=True)
class _ReadyRow:
    session_uid: int
    player_car_index: int
    row: AnalysisRow


class FrameSynchronizer:
    """
    Asocia Car Telemetry y Lap Data del mismo frame.

    La clave es ``session_uid``, ``overall_frame_identifier`` y
    ``player_car_index``. Si falta uno de los dos paquetes, no hay fila.

    Un frame completo no se escribe mientras un frame anterior de la
    misma sesión pueda llegar dentro de ``reorder_window``. Ese margen
    absorbe el reorden de UDP. Un hueco que sigue vacío al salir de la
    ventana se conserva: no se inventa la muestra perdida.
    """

    def __init__(
        self,
        max_pending: int = 64,
        reorder_window: int = 8,
    ) -> None:
        if max_pending < 1:
            raise ValueError(
                "max_pending debe ser al menos 1."
            )

        if reorder_window < 1:
            raise ValueError(
                "reorder_window debe ser al menos 1."
            )

        self.max_pending = max_pending
        self.reorder_window = reorder_window
        self._pending: dict[SyncKey, _PartialFrame] = {}
        self._ready: list[_ReadyRow] = []
        self._closed: set[SyncKey] = set()
        self._last_emitted: dict[tuple[int, int], int] = {}
        self._newest: dict[tuple[int, int], int] = {}
        self._flushing = False

    def add_car_telemetry(
        self,
        header: PacketHeader,
        car: CarTelemetryData,
    ) -> list[AnalysisRow]:
        """Guarda el Packet 6 y devuelve las filas que ya pueden escribirse."""

        key = _sync_key(
            header.session_uid,
            header.overall_frame_identifier,
            header.player_car_index,
        )

        if not self._accept(key):
            return self._drain()

        partial = self._pending.setdefault(
            key,
            _PartialFrame(),
        )
        partial.session_time = header.session_time
        partial.speed = car.speed
        partial.throttle = car.throttle
        partial.brake = car.brake
        partial.steering = car.steering
        partial.gear = car.gear
        partial.engine_rpm = car.engine_rpm
        partial.drs = 1 if car.drs else 0

        self._complete(key, partial)
        self._evict_oldest()

        return self._drain()

    def add_lap_data(
        self,
        lap: PlayerLapData,
    ) -> list[AnalysisRow]:
        """Guarda el Packet 2 y devuelve las filas que ya pueden escribirse."""

        key = _sync_key(
            lap.session_uid,
            lap.overall_frame_identifier,
            lap.player_car_index,
        )

        if not self._accept(key):
            return self._drain()

        partial = self._pending.setdefault(
            key,
            _PartialFrame(),
        )
        partial.lap_distance = lap.lap_distance
        partial.lap_number = lap.lap_number

        self._complete(key, partial)
        self._evict_oldest()

        return self._drain()

    def flush(self) -> list[AnalysisRow]:
        """
        Emite las filas completas que seguían esperando un hueco.

        Los frames que todavía tienen un solo paquete no generan fila.
        """

        for key in list(self._pending):
            del self._pending[key]
            self._closed.add(key)

        self._flushing = True

        try:
            return self._drain()
        finally:
            self._flushing = False

    def _accept(self, key: SyncKey) -> bool:
        session_uid, frame, player_car_index = key
        group = (session_uid, player_car_index)
        newest = self._newest.get(group)

        if newest is None or frame > newest:
            self._newest[group] = frame

        last_emitted = self._last_emitted.get(group)

        if last_emitted is not None and frame <= last_emitted:
            return False

        return key not in self._closed

    def _complete(
        self,
        key: SyncKey,
        partial: _PartialFrame,
    ) -> None:
        if not partial.is_complete():
            return

        del self._pending[key]
        self._closed.add(key)

        lap_distance = partial.lap_distance
        if lap_distance is None or lap_distance < 0:
            return

        session_uid, overall_frame, player_car_index = key

        self._ready.append(
            _ReadyRow(
                session_uid=session_uid,
                player_car_index=player_car_index,
                row=AnalysisRow(
                    session_time=partial.session_time,
                    frame=overall_frame,
                    lap_number=partial.lap_number,
                    lap_distance=lap_distance,
                    speed=partial.speed,
                    throttle=partial.throttle,
                    brake=partial.brake,
                    steering=partial.steering,
                    gear=partial.gear,
                    rpm=partial.engine_rpm,
                    drs=partial.drs,
                ),
            )
        )

    def _evict_oldest(self) -> None:
        while len(self._pending) > self.max_pending:
            oldest = min(
                self._pending,
                key=lambda item: (item[1], item[0], item[2]),
            )
            del self._pending[oldest]
            self._closed.add(oldest)

    def _drain(self) -> list[AnalysisRow]:
        if not self._ready:
            return []

        pending_frames: dict[tuple[int, int], int] = {}

        for session_uid, frame, player_car_index in self._pending:
            group = (session_uid, player_car_index)
            current = pending_frames.get(group)

            if current is None or frame < current:
                pending_frames[group] = frame

        released: list[AnalysisRow] = []
        held: list[_ReadyRow] = []
        blocked: set[tuple[int, int]] = set()

        ordered = sorted(
            self._ready,
            key=lambda item: (
                item.session_uid,
                item.player_car_index,
                item.row.frame,
            ),
        )

        for item in ordered:
            group = (item.session_uid, item.player_car_index)
            last_emitted = self._last_emitted.get(group)

            if (
                last_emitted is not None
                and item.row.frame <= last_emitted
            ):
                continue

            older_pending = pending_frames.get(group)
            waiting_for_gap = self._gap_is_open(
                item.session_uid,
                item.player_car_index,
                item.row.frame,
            )

            if group in blocked or (
                older_pending is not None
                and older_pending < item.row.frame
            ) or waiting_for_gap:
                blocked.add(group)
                held.append(item)
                continue

            self._last_emitted[group] = item.row.frame
            released.append(item.row)

        self._ready = held

        return released

    def _gap_is_open(
        self,
        session_uid: int,
        player_car_index: int,
        frame: int,
    ) -> bool:
        """
        Indica si falta un frame anterior que todavía puede llegar.

        La comparación usa ``overall_frame_identifier``. ``session_time``
        no participa.
        """

        if self._flushing:
            return False

        group = (session_uid, player_car_index)
        last_emitted = self._last_emitted.get(group)

        if last_emitted is None or frame <= last_emitted + 1:
            return False

        newest = self._newest[group]
        oldest_open = newest - self.reorder_window + 1
        start = max(last_emitted + 1, oldest_open)

        for missing in range(start, frame):
            key = (session_uid, missing, player_car_index)

            if key in self._closed or key in self._pending:
                continue

            return True

        return False


def _sync_key(
    session_uid: int,
    overall_frame_identifier: int,
    player_car_index: int,
) -> SyncKey:
    return (
        session_uid,
        overall_frame_identifier,
        player_car_index,
    )
