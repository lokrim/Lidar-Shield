from pathlib import Path

import pytest

from lidar_shield.config import ConfigurationError, config_hash, load_config

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def test_valid_config_resolves_explicit_paths() -> None:
    config_path = FIXTURE_ROOT / "config" / "valid.yaml"
    config = load_config(config_path)

    assert config.project.distribution == "lidar-shield"
    assert config.reproducibility.seed == 1729
    assert config.paths.fixture_root == FIXTURE_ROOT.parent.resolve()
    assert config_hash(config_path) == (
        "4ed582c0f5383a9607165b4f3889b35d9a17dab6ecc6d4dbd9cb95b4f32e7d9a"
    )


def test_invalid_config_has_meaningful_errors() -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_config(FIXTURE_ROOT / "config" / "invalid.yaml")

    message = str(caught.value)
    assert "invalid configuration" in message
    assert "seed" in message
    assert "unexpected_m0_field" in message


def test_missing_config_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("content", ["- not\n- a\n- mapping\n", "key: [unterminated"])
def test_unreadable_config_shapes_are_reported(tmp_path: Path, content: str) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(config_path)


def test_fixture_root_must_exist(tmp_path: Path) -> None:
    source = (FIXTURE_ROOT / "config" / "valid.yaml").read_text(encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(source.replace("fixture_root: ../../", "fixture_root: missing"))
    with pytest.raises(ConfigurationError, match="fixture_root"):
        load_config(path)
