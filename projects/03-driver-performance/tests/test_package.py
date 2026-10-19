
from driver_performance import __version__


def test_package_has_version():
    assert __version__ == "0.1.0"


def test_telemetry_analyzer_is_importable():
    import telemetry_analyzer

    assert telemetry_analyzer.__file__ is not None
