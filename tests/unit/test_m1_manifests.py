import json
from pathlib import Path

from lidar_shield.contracts.schemas import ExperimentManifest, SourceManifest
from lidar_shield.manifest import canonical_json_bytes, sha256_bytes, sha256_file

ROOT = Path(__file__).parents[2]


def test_committed_source_manifests_bind_exact_fixture_and_config_hashes() -> None:
    for name in ("m1_synthetic_sequence_a.json", "m1_synthetic_sequence_b.json"):
        path = ROOT / "manifests" / "datasets" / name
        manifest = SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
        records = []
        for item in manifest.files:
            target = ROOT / item.path
            assert target.stat().st_size == item.bytes
            assert sha256_file(target) == item.sha256
            records.append({"path": item.path, "sha256": item.sha256})
        assert sha256_bytes(canonical_json_bytes(records)) == manifest.source_sha256
        assert sha256_file(ROOT / "configs" / "default.yaml") == manifest.config_sha256
        assert (
            sha256_file(ROOT / "configs" / "agents" / "mixed_signals.yaml")
            == manifest.agent_registry_sha256
        )
        assert (
            sha256_file(ROOT / "configs" / "data" / "stage1_split.yaml")
            == manifest.split_registry_sha256
        )


def test_experiment_manifest_binds_schema_and_source_manifests() -> None:
    path = ROOT / "manifests" / "experiments" / "m1_contract_fixtures.json"
    manifest = ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))
    assert manifest.schema_registry_sha256 == sha256_file(
        ROOT / "manifests" / "contracts" / "m1_schema_registry.json"
    )
    actual_sources = tuple(
        sorted(
            sha256_file(source)
            for source in (ROOT / "manifests" / "datasets").glob("m1_*.json")
        )
    )
    assert manifest.source_manifest_sha256 == actual_sources
    registry = json.loads(
        (ROOT / "manifests" / "contracts" / "m1_schema_registry.json").read_text()
    )
    assert registry["signed_age_convention"] == ("decision_time_ns - source_time_ns")
