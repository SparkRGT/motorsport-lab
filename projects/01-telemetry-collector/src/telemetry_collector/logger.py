import csv
import json
from pathlib import Path

from telemetry_collector.models import AnalysisRow


CSV_FIELDS = (
    "session_time",
    "frame",
    "lap_number",
    "lap_distance",
    "speed",
    "throttle",
    "brake",
    "steering",
    "gear",
    "rpm",
    "drs",
)


class TelemetryLogger:
    """Escribe el CSV analítico al vuelo y el JSON al cierre."""

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

        self._rows: list[AnalysisRow] = []
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None

    def start(self) -> Path:
        """Crea el CSV y escribe el encabezado una sola vez."""

        if self._csv_writer is not None:
            return self.csv_path

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._csv_file = self.csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=CSV_FIELDS,
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

        return self.csv_path

    def write_row(self, row: AnalysisRow) -> None:
        """Agrega una fila al CSV y la conserva para el JSON final."""

        if self._csv_writer is None or self._csv_file is None:
            self.start()

        assert self._csv_writer is not None
        assert self._csv_file is not None

        self._csv_writer.writerow(
            _csv_values(row)
        )
        self._csv_file.flush()
        self._rows.append(row)

    def close(self) -> None:
        """Cierra el CSV incremental."""

        if self._csv_file is None:
            return

        self._csv_file.flush()
        self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None

    def save_json(self) -> Path:
        """Guarda las filas ya escritas en JSON."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = [
            _json_values(row)
            for row in self._rows
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
        """Devuelve el CSV ya escrito de forma incremental."""

        if self._csv_writer is None:
            self.start()

        if self._csv_file is not None:
            self._csv_file.flush()

        return self.csv_path

    def save(self) -> tuple[Path, Path]:
        """Cierra el CSV y guarda el JSON de la sesión."""

        csv_path = self.save_csv()
        self.close()
        json_path = self.save_json()

        return json_path, csv_path

    @property
    def row_count(self) -> int:
        """Cantidad de filas analíticas escritas."""

        return len(self._rows)

    @property
    def snapshot_count(self) -> int:
        """Alias de ``row_count`` usado por el flujo de cierre."""

        return self.row_count


def _csv_values(row: AnalysisRow) -> dict[str, object]:
    return {
        "session_time": row.session_time,
        "frame": row.frame,
        "lap_number": row.lap_number,
        "lap_distance": row.lap_distance,
        "speed": row.speed,
        "throttle": row.throttle,
        "brake": row.brake,
        "steering": row.steering,
        "gear": row.gear,
        "rpm": row.rpm,
        "drs": row.drs,
    }


def _json_values(row: AnalysisRow) -> dict[str, object]:
    return _csv_values(row)
