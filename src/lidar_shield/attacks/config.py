"""Authored M4 configuration to fully bound immutable attack manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lidar_shield import __version__
from lidar_shield.attacks.manifests import (
    AttackManifest,
    AttackTarget,
    AttackVersions,
    KinematicParameters,
    SourceHashes,
)
from lidar_shield.contracts.schemas import CONTRACT_SCHEMA_VERSION
from lidar_shield.data.adapters.mixed_signals import (
    FractionalTimeConvention,
    parse_cloud_filename,
)
from lidar_shield.data.index import load_agent_registry
from lidar_shield.evaluation.splits import SplitRole, load_split_registry
from lidar_shield.manifest import production_code_sha256, sha256_file


class AttackConfigurationError(ValueError):
    """A declarative M4 attack configuration cannot bind a source manifest."""


class VelocityAttackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0.0"]
    experiment_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    sequence_id: Literal["mini_7"]
    session_id: str = Field(min_length=1)
    sync_frame_id: int = Field(ge=0)
    sender_id: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    attack_kind: Literal["velocity_spike", "velocity_drift"]
    delta_v_mps: tuple[float, float, float] = Field(strict=False)
    duration_samples: int = Field(gt=0)
    seed: int = Field(ge=0)
    attacker_knowledge: Literal["source_only"] = "source_only"


def load_velocity_attack_config(path: str | Path) -> VelocityAttackConfig:
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        return VelocityAttackConfig.model_validate(raw, strict=False)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise AttackConfigurationError(
            f"invalid attack config {source}: {exc}"
        ) from exc


def build_velocity_manifest(
    config: VelocityAttackConfig,
    *,
    config_path: str | Path,
    data_root: str | Path,
    clean_dir: str | Path,
    split_registry_path: str | Path,
    agent_registry_path: str | Path,
) -> AttackManifest:
    """Resolve exact source times and all hashes before variant generation."""

    clean = Path(clean_dir).expanduser().resolve()
    try:
        clean_manifest = json.loads(
            (clean / "source_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttackConfigurationError(f"cannot read clean manifest: {exc}") from exc
    registry, split_hash = load_split_registry(split_registry_path)
    assignment = registry.assignment_for(config.sequence_id)
    if assignment.source_id != config.source_id:
        raise AttackConfigurationError("configured source ID is not registered")
    capabilities = {
        item.agent_id: item for item in load_agent_registry(agent_registry_path).agents
    }
    if config.sender_id not in capabilities:
        raise AttackConfigurationError("configured sender is absent from registry")
    cloud_root = (
        Path(data_root).expanduser().resolve() / "PointClouds" / config.sequence_id
    )

    def cloud_time(agent: str, frame_id: int) -> int:
        matches = tuple(cloud_root.glob(f"{agent}_{frame_id}_*.pcd"))
        if len(matches) != 1:
            raise AttackConfigurationError(
                f"expected one cloud for {agent}/{frame_id}, found {len(matches)}"
            )
        return parse_cloud_filename(
            matches[0],
            convention=FractionalTimeConvention.OMITTED_LEADING_ZEROES,
        ).source_time_ns

    frame_ids = tuple(
        config.sync_frame_id + offset for offset in range(config.duration_samples)
    )
    sample_times = tuple(
        cloud_time(config.sender_id, frame_id) for frame_id in frame_ids
    )
    raw_relative = f"Odometry/{config.sequence_id}/odometry_{config.sender_id}.csv"
    raw_path = Path(data_root).expanduser().resolve() / raw_relative
    outputs = clean_manifest.get("outputs", {})
    try:
        source_hashes = SourceHashes(
            archive_sha256=clean_manifest["archive"]["sha256"],
            selected_source_sha256=clean_manifest["selected_source_sha256"],
            clean_manifest_sha256=sha256_file(clean / "source_manifest.json"),
            clean_frame_sha256=outputs["frame.parquet"]["sha256"],
            clean_object_sha256=outputs["object.parquet"]["sha256"],
            clean_evidence_sha256=outputs["evidence.parquet"]["sha256"],
            target_raw_sha256=sha256_file(raw_path),
            split_registry_sha256=split_hash,
        )
    except (KeyError, TypeError, ValidationError) as exc:
        raise AttackConfigurationError("clean manifest lacks required hashes") from exc
    return AttackManifest(
        experiment_id=config.experiment_id,
        episode_id=config.episode_id,
        variant_id=config.variant_id,
        origin="official_dataset",
        split_role=SplitRole(assignment.role),
        target=AttackTarget(
            source_id=config.source_id,
            sequence_id=config.sequence_id,
            session_id=config.session_id,
            sync_frame_id=config.sync_frame_id,
            decision_time_ns=cloud_time("top", config.sync_frame_id),
            source_time_ns=cloud_time(config.sender_id, config.sync_frame_id),
            sender_id=config.sender_id,
            agent_type=capabilities[config.sender_id].agent_type.value,
            interval_sample_times_ns=sample_times,
            raw_relative_path=raw_relative,
        ),
        source_hashes=source_hashes,
        family="kinematic",
        severity=config.severity,
        parameters=KinematicParameters(
            attack_kind=config.attack_kind,
            delta_v_mps=config.delta_v_mps,
            duration_samples=config.duration_samples,
        ),
        seed=config.seed,
        attacker_knowledge=config.attacker_knowledge,
        expected_modality="kinematic",
        versions=AttackVersions(
            contract_schema=CONTRACT_SCHEMA_VERSION,
            package_version=__version__,
            code_sha256=production_code_sha256(),
            config_sha256=sha256_file(config_path),
        ),
    )
