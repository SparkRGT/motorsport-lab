from pathlib import Path

from telemetry_collector.header import parse_packet_header


def main() -> None:
    packet_path = Path("first_packet.bin")

    if not packet_path.exists():
        raise FileNotFoundError(
            "No se encontró first_packet.bin."
        )

    data = packet_path.read_bytes()

    header = parse_packet_header(data)

    print("F1 25 Packet Header")
    print("===================")
    print(f"Packet format:             {header.packet_format}")
    print(f"Game year:                 {header.game_year}")
    print(f"Game major version:        {header.game_major_version}")
    print(f"Game minor version:        {header.game_minor_version}")
    print(f"Packet version:            {header.packet_version}")
    print(f"Packet ID:                 {header.packet_id}")
    print(f"Session UID:               {header.session_uid}")
    print(f"Session time:              {header.session_time}")
    print(f"Frame identifier:          {header.frame_identifier}")
    print(f"Overall frame identifier:  {header.overall_frame_identifier}")
    print(f"Player car index:          {header.player_car_index}")
    print(
        "Secondary player car index:"
        f" {header.secondary_player_car_index}"
    )


if __name__ == "__main__":
    main()