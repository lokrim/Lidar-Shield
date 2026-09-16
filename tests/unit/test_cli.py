import json
from pathlib import Path

import pytest

from lidar_shield import __version__
from lidar_shield.attacks.config import VelocityAttackConfig
from lidar_shield.cli import build_parser, main
from lidar_shield.data.cache import CacheBuildResult, CacheError

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"
PROJECT_ROOT = Path(__file__).parents[2]


def test_help_lists_only_implemented_configuration_surface() -> None:
    help_text = build_parser().format_help()
    assert "config" in help_text
    for future_command in ("data", "trust", "replay", "report"):
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
    assert summary["schema_version"] == "1.0.0"
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


def test_d0_demo_is_stable_and_accounts_for_every_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "demo",
        "d0",
        "--agents",
        str(PROJECT_ROOT / "configs" / "agents" / "mixed_signals.yaml"),
    ]
    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert main(arguments) == 0
    second = capsys.readouterr().out
    assert first == second
    payload = json.loads(first)
    assert payload["event_count"] == 9
    assert set(payload["reason_counts"]) == {
        "empty_scene",
        "invalid_pose",
        "late",
        "matched",
        "missing_message",
        "not_member",
        "out_of_range",
        "stale",
        "unmatched",
    }


def test_cache_and_d1_cli_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = CacheBuildResult(
        clean_dir=tmp_path,
        output_hashes={"frame.parquet": "a" * 64},
        manifest_sha256="b" * 64,
        resumed=False,
    )
    captured = []

    def fake_build(config: object) -> CacheBuildResult:
        captured.append(config)
        return result

    monkeypatch.setattr("lidar_shield.cli.build_clean_cache", fake_build)
    arguments = [
        "cache",
        "build",
        "--data-root",
        "data",
        "--artifact-root",
        str(tmp_path),
        "--experiment-id",
        "cli-fixture",
        "--frames",
        "0:2",
    ]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "verified"
    assert payload["resumed"] is False
    assert captured

    monkeypatch.setattr(
        "lidar_shield.cli.build_d1_payload", lambda value: {"demo": "D1"}
    )
    arguments[0:2] = ["demo", "d1"]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"demo": "D1"}


def test_cache_cli_verification_and_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = CacheBuildResult(tmp_path, {}, "c" * 64, True)
    monkeypatch.setattr("lidar_shield.cli.verify_clean_cache", lambda value: result)
    assert main(["cache", "verify", "--cache-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"

    def fail(_value: object) -> CacheBuildResult:
        raise CacheError("broken")

    monkeypatch.setattr("lidar_shield.cli.verify_clean_cache", fail)
    assert main(["cache", "verify", "--cache-dir", str(tmp_path)]) == 2
    assert "cache error: broken" in capsys.readouterr().err
    assert (
        main(
            [
                "cache",
                "build",
                "--data-root",
                "data",
                "--experiment-id",
                "x",
                "--frames",
                "bad",
            ]
        )
        == 2
    )
    assert "--frames" in capsys.readouterr().err


def test_d2_cli_is_stable_and_reports_fixture_errors(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "m3" / "integration_fixture.json"
    arguments = ["demo", "d2", "--fixture", str(fixture)]
    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert main(arguments) == 0
    second = capsys.readouterr().out
    assert first == second
    payload = json.loads(first)
    assert payload["demo"] == "D2"
    assert payload["scientific_eligibility"] == "none"

    assert main(["demo", "d2", "--fixture", str(tmp_path / "missing")]) == 2
    assert "integration fixture error:" in capsys.readouterr().err


def test_attack_run_and_d3_cli_surfaces(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeManifest:
        episode_id = "episode"
        variant_id = "variant"

    fake_manifest = FakeManifest()
    monkeypatch.setattr(
        "lidar_shield.cli.load_attack_manifest", lambda _: fake_manifest
    )
    monkeypatch.setattr(
        "lidar_shield.cli.run_kinematic_attack", lambda *args, **kwargs: tmp_path
    )
    assert (
        main(
            [
                "attack",
                "run",
                "--manifest",
                "manifest.json",
                "--data-root",
                "data",
                "--clean-dir",
                "clean",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["variant_id"] == "variant"

    declaration = VelocityAttackConfig.model_validate(
        {
            "schema_version": "1.0.0",
            "experiment_id": "d3",
            "episode_id": "episode",
            "variant_id": "variant",
            "source_id": "source",
            "sequence_id": "mini_7",
            "session_id": "session",
            "sync_frame_id": 1,
            "sender_id": "003",
            "severity": "high",
            "attack_kind": "velocity_spike",
            "delta_v_mps": [1.0, 0.0, 0.0],
            "duration_samples": 1,
            "seed": 1,
            "attacker_knowledge": "source_only",
        },
        strict=False,
    )
    cache = CacheBuildResult(tmp_path / "clean", {}, "a" * 64, False)
    monkeypatch.setattr(
        "lidar_shield.cli.load_velocity_attack_config", lambda _: declaration
    )
    monkeypatch.setattr("lidar_shield.cli.build_clean_cache", lambda _: cache)
    monkeypatch.setattr(
        "lidar_shield.cli.build_velocity_manifest",
        lambda *args, **kwargs: fake_manifest,
    )
    monkeypatch.setattr(
        "lidar_shield.cli.build_d3_payload", lambda *args, **kwargs: {"demo": "D3"}
    )
    assert main(["demo", "d3", "--config", "attack.yaml"]) == 0
    assert json.loads(capsys.readouterr().out) == {"demo": "D3"}

    def fail_attack(*args: object, **kwargs: object) -> Path:
        raise CacheError("blocked")

    monkeypatch.setattr("lidar_shield.cli.run_kinematic_attack", fail_attack)
    assert (
        main(
            [
                "attack",
                "run",
                "--manifest",
                "manifest.json",
                "--data-root",
                "data",
                "--clean-dir",
                "clean",
            ]
        )
        == 2
    )
    assert "attack error:" in capsys.readouterr().err
