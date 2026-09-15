import json
from pathlib import Path

import pytest

from lidar_shield import __version__
from lidar_shield.cli import build_parser, main

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def test_help_lists_only_implemented_configuration_surface() -> None:
    help_text = build_parser().format_help()
    assert "config" in help_text
    for future_command in ("data", "attack", "trust", "replay", "report"):
        assert f"{{{future_command}}}" not in help_text


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert capsys.readouterr().out.strip() == f"lidar-shield {__version__}"


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "M0 foundation" in capsys.readouterr().out


def test_config_validate_prints_machine_readable_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "config",
            "validate",
            "--config",
            str(FIXTURE_ROOT / "config" / "valid.yaml"),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert result == 0
    assert summary["status"] == "valid"
    assert summary["schema_version"] == "0.1.0"
    assert len(summary["config_hash"]) == 64


def test_config_validate_reports_error(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(
        [
            "config",
            "validate",
            "--config",
            str(FIXTURE_ROOT / "config" / "invalid.yaml"),
        ]
    )
    assert result == 2
    assert "configuration error:" in capsys.readouterr().err
