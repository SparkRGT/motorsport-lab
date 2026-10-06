import csv
import json

from telemetry_collector.logger import (
    CSV_FIELDS,
    TelemetryLogger,
)
from telemetry_collector.models import AnalysisRow


def create_test_row(
    frame: int,
    *,
    gear: int = 6,
    drs: int = 1,
    rpm: int = 9500,
) -> AnalysisRow:
    return AnalysisRow(
        session_time=float(frame),
        frame=frame,
        lap_number=1,
        lap_distance=100.0 + frame,
        speed=200,
        throttle=0.75,
        brake=0.0,
        steering=-0.25,
        gear=gear,
        rpm=rpm,
        drs=drs,
    )


def test_logger_adds_rows(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.write_row(create_test_row(100))
    logger.write_row(create_test_row(101))

    assert logger.row_count == 2
    assert logger.snapshot_count == 2


def test_save_json(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.write_row(create_test_row(100))

    json_path = logger.save_json()

    assert json_path.exists()

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert len(data) == 1

    assert data[0]["frame"] == 100
    assert data[0]["speed"] == 200
    assert data[0]["throttle"] == 0.75
    assert data[0]["rpm"] == 9500
    assert data[0]["drs"] == 1
    assert "engine_rpm" not in data[0]
    assert "car_index" not in data[0]


def test_save_csv(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.write_row(
        create_test_row(
            100,
            gear=-1,
            drs=0,
        )
    )

    csv_path = logger.save_csv()

    assert csv_path.exists()

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == list(CSV_FIELDS)
    assert list(CSV_FIELDS) == [
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
    ]

    assert len(rows) == 1

    assert rows[0]["frame"] == "100"
    assert rows[0]["speed"] == "200"
    assert rows[0]["throttle"] == "0.75"
    assert rows[0]["gear"] == "-1"
    assert rows[0]["rpm"] == "9500"
    assert rows[0]["drs"] == "0"


def test_csv_is_written_incrementally(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    csv_path = logger.start()

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == list(CSV_FIELDS)
        assert list(reader) == []

    logger.write_row(create_test_row(1, drs=0))
    logger.write_row(create_test_row(2, drs=1))

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        content = file.read()

    assert content.count("session_time") == 1

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert [row["frame"] for row in rows] == ["1", "2"]
    assert [row["drs"] for row in rows] == ["0", "1"]
    assert [row["rpm"] for row in rows] == ["9500", "9500"]

    logger.close()


def test_save_creates_both_formats(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.write_row(create_test_row(100))

    json_path, csv_path = logger.save()

    assert json_path.exists()
    assert csv_path.exists()


def test_logger_starts_empty(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    assert logger.row_count == 0


def test_logger_creates_output_directory(tmp_path) -> None:
    output_directory = tmp_path / "telemetry"

    logger = TelemetryLogger(
        output_directory=output_directory,
    )

    logger.write_row(create_test_row(100))
    logger.save()

    assert output_directory.exists()
    assert output_directory.is_dir()
