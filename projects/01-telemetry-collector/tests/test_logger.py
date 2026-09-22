import csv
import json

from telemetry_collector.logger import (
    CSV_FIELDS,
    TelemetryLogger,
)
from telemetry_collector.models import TelemetrySnapshot


def create_test_snapshot(
    frame: int,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        frame=frame,
        session_time=float(frame),
        car_index=19,
        speed=200,
        throttle=0.75,
        brake=0.0,
        steering=-0.25,
        gear=6,
        engine_rpm=9500,
        drs=True,
    )


def test_logger_adds_snapshots(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.add(create_test_snapshot(100))
    logger.add(create_test_snapshot(101))

    assert logger.snapshot_count == 2


def test_save_json(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.add(create_test_snapshot(100))

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
    assert data[0]["drs"] is True


def test_save_csv(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.add(create_test_snapshot(100))

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

    assert len(rows) == 1

    assert rows[0]["frame"] == "100"
    assert rows[0]["speed"] == "200"
    assert rows[0]["throttle"] == "0.75"
    assert rows[0]["drs"] == "True"


def test_save_creates_both_formats(tmp_path) -> None:
    logger = TelemetryLogger(
        output_directory=tmp_path,
    )

    logger.add(create_test_snapshot(100))

    json_path, csv_path = logger.save()

    assert json_path.exists()
    assert csv_path.exists()