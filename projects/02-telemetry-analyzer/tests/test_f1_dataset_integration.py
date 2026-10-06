import sys
from pathlib import Path

import pytest

from telemetry_analyzer.ingestion.csv_loader import load_csv
from telemetry_analyzer.processing.laps import segment_laps
from telemetry_analyzer.processing.normalization import normalize_dataset
from telemetry_analyzer.processing.validation import (
    REQUIRED_COLUMNS,
    validate_dataset,
)


COLLECTOR_ROOT = (
    Path(__file__).resolve().parents[2] / "01-telemetry-collector"
)
sys.path.insert(0, str(COLLECTOR_ROOT / "src"))
sys.path.insert(0, str(COLLECTOR_ROOT / "tests"))

from packets import (  # noqa: E402
    build_car_telemetry_packet,
    build_lap_data_packet,
)
from telemetry_collector.car_telemetry import (  # noqa: E402
    get_player_car_telemetry,
    parse_car_telemetry_packet,
)
from telemetry_collector.header import parse_packet_header  # noqa: E402
from telemetry_collector.lap_data import parse_player_lap_data  # noqa: E402
from telemetry_collector.logger import TelemetryLogger  # noqa: E402
from telemetry_collector.sync import FrameSynchronizer  # noqa: E402


def _add_telemetry(sync: FrameSynchronizer, packet: bytes):
    header = parse_packet_header(packet)
    telemetry = parse_car_telemetry_packet(packet)
    car = get_player_car_telemetry(
        telemetry,
        header.player_car_index,
    )
    return sync.add_car_telemetry(header, car)


def _add_lap(sync: FrameSynchronizer, packet: bytes):
    return sync.add_lap_data(parse_player_lap_data(packet))


def test_synthetic_f1_packets_feed_the_analyzer_pipeline(tmp_path):
    sync = FrameSynchronizer()
    logger = TelemetryLogger(
        output_directory=tmp_path,
        filename="telemetry_session",
    )
    logger.start()

    samples = [
        {
            "overall_frame_identifier": 1000,
            "frame_identifier": 10,
            "session_time": 10.0,
            "lap_number": 1,
            "lap_distance": 10.0,
            "gear": 3,
            "drs": 0,
            "engine_rpm": 9000,
        },
        {
            "overall_frame_identifier": 1001,
            "frame_identifier": 11,
            "session_time": 10.1,
            "lap_number": 1,
            "lap_distance": 25.0,
            "gear": -1,
            "drs": 1,
            "engine_rpm": 4000,
        },
        {
            "overall_frame_identifier": 1002,
            "frame_identifier": 12,
            "session_time": 20.0,
            "lap_number": 2,
            "lap_distance": 5.0,
            "gear": 4,
            "drs": 0,
            "engine_rpm": 10000,
        },
    ]

    for sample in samples:
        common = {
            "session_uid": 77,
            "session_time": sample["session_time"],
            "frame_identifier": sample["frame_identifier"],
            "overall_frame_identifier": sample["overall_frame_identifier"],
            "player_car_index": 19,
        }
        telemetry = build_car_telemetry_packet(
            **common,
            gear=sample["gear"],
            drs=sample["drs"],
            engine_rpm=sample["engine_rpm"],
        )
        lap = build_lap_data_packet(
            **common,
            lap_number=sample["lap_number"],
            lap_distance=sample["lap_distance"],
        )
        assert _add_telemetry(sync, telemetry) == []
        for row in _add_lap(sync, lap):
            logger.write_row(row)

    missing_lap = build_car_telemetry_packet(
        session_uid=77,
        overall_frame_identifier=1003,
        frame_identifier=13,
        session_time=20.1,
        player_car_index=19,
    )
    assert _add_telemetry(sync, missing_lap) == []

    negative = {
        "session_uid": 77,
        "session_time": 20.2,
        "frame_identifier": 14,
        "overall_frame_identifier": 1004,
        "player_car_index": 19,
    }
    assert _add_telemetry(
        sync,
        build_car_telemetry_packet(**negative, gear=2, engine_rpm=8000),
    ) == []
    assert _add_lap(
        sync,
        build_lap_data_packet(
            **negative,
            lap_number=2,
            lap_distance=-1.0,
        ),
    ) == []

    csv_path = logger.save_csv()
    logger.close()

    dataset = load_csv(csv_path)

    assert list(dataset.columns) == REQUIRED_COLUMNS
    assert len(dataset) == 3
    assert dataset["frame"].tolist() == [1000, 1001, 1002]
    assert dataset["lap_number"].tolist() == [1, 1, 2]
    assert dataset["lap_distance"].tolist() == pytest.approx([10.0, 25.0, 5.0])
    assert dataset["rpm"].tolist() == [9000, 4000, 10000]
    assert dataset["drs"].tolist() == [0, 1, 0]
    assert dataset["gear"].tolist() == [3, -1, 4]

    validate_dataset(dataset)
    normalized = normalize_dataset(dataset)
    laps = segment_laps(normalized)

    assert set(laps) == {1, 2}
    assert laps[1].sample_count == 2
    assert laps[2].sample_count == 1
    assert int(normalized.loc[1, "gear"]) == -1
