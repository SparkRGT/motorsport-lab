import csv
import json
from pathlib import Path

from telemetry_collector.models import TelemetrySnapshot


CSV_FIELDS = (
    "frame",
    "session_time",
    "car_index",
    "speed",
    "throttle",
    "brake",
    "steering",
    "gear",
    "engine_rpm",
    "drs",
)


class TelemetryLogger:
    """Guarda snapshots de telemetría en JSON y CSV."""

    def __init__(
        self,
        output_directory: Path,
        filename: str = "telemetry",
    ) -> None:
        self.output_directory = output_directory
        self.filename = filename

        self.json_path = (
            self.output_directory / f"{self.filename}.json"
        )

        self.csv_path = (
            self.output_directory / f"{self.filename}.csv"
        )

        self._snapshots: list[TelemetrySnapshot] = []

    def add(self, snapshot: TelemetrySnapshot) -> None:
        """Agrega un snapshot al registro actual."""

        self._snapshots.append(snapshot)

    def save_json(self) -> Path:
        """Guarda todos los snapshots en formato JSON."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = [
            {
                "frame": snapshot.frame,
                "session_time": snapshot.session_time,
                "car_index": snapshot.car_index,
                "speed": snapshot.speed,
                "throttle": snapshot.throttle,
                "brake": snapshot.brake,
                "steering": snapshot.steering,
                "gear": snapshot.gear,
                "engine_rpm": snapshot.engine_rpm,
                "drs": snapshot.drs,
            }
            for snapshot in self._snapshots
        ]

        with self.json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return self.json_path

    def save_csv(self) -> Path:
        """Guarda todos los snapshots en formato CSV."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS,
            )

            writer.writeheader()

            for snapshot in self._snapshots:
                writer.writerow(
                    {
                        "frame": snapshot.frame,
                        "session_time": snapshot.session_time,
                        "car_index": snapshot.car_index,
                        "speed": snapshot.speed,
                        "throttle": snapshot.throttle,
                        "brake": snapshot.brake,
                        "steering": snapshot.steering,
                        "gear": snapshot.gear,
                        "engine_rpm": snapshot.engine_rpm,
                        "drs": snapshot.drs,
                    }
                )

        return self.csv_path

    def save(self) -> tuple[Path, Path]:
        """Guarda los datos en JSON y CSV."""

        json_path = self.save_json()
        csv_path = self.save_csv()

        return json_path, csv_path

    @property
    def snapshot_count(self) -> int:
        """Cantidad de snapshots almacenados."""

        return len(self._snapshots)