"""Incremental dependency expansion and immutable M4 artifact publication."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lidar_shield.attacks.manifests import AttackManifest
from lidar_shield.manifest import canonical_json_bytes, sha256_bytes, sha256_file


class OverlayError(ValueError):
    """An overlay cannot be generated or published immutably."""


EVIDENCE_DELTA_KEY = (
    "experiment_id",
    "variant_id",
    "sequence_id",
    "session_id",
    "sync_frame_id",
    "decision_time_ns",
    "source_time_ns",
    "agent_id",
    "agent_type",
    "object_id_or_local_index",
    "evidence_level",
    "evidence_name",
)


@dataclass(frozen=True, order=True)
class EvidenceKey:
    sequence_id: str
    session_id: str
    sync_frame_id: int
    agent_id: str
    evidence_level: Literal["frame", "object"]
    evidence_name: str
    object_id_or_local_index: str | int | None = None


@dataclass(frozen=True)
class IncrementalDependencyMap:
    changed_observations: tuple[tuple[int, str], ...]
    affected_kinematic_intervals: tuple[tuple[int, str], ...]
    affected_object_groups: tuple[tuple[int, str | int], ...]
    affected_evidence_keys: tuple[EvidenceKey, ...]
    honest_companion_agents: tuple[str, ...]


@dataclass(frozen=True)
class AttemptOutcome:
    status: Literal["succeeded", "zero_effect", "low_effect", "failed", "interrupted"]
    reason_code: str
    injection_succeeded: bool
    intended_target_labels: tuple[str, ...]
    actual_affected_evidence_keys: tuple[EvidenceKey, ...]
    requested_physical_effect: dict[str, Any]
    realized_physical_effect: dict[str, Any]
    realized_evidence_effect: dict[str, Any]


def kinematic_dependency_map(
    manifest: AttackManifest,
    *,
    affected_interval_frames: tuple[int, ...],
) -> IncrementalDependencyMap:
    sender = manifest.target.sender_id
    keys = tuple(
        EvidenceKey(
            manifest.target.sequence_id,
            manifest.target.session_id,
            frame,
            sender,
            "frame",
            name,
        )
        for frame in sorted(set(affected_interval_frames))
        for name in (
            "kinematic_dt_s",
            "kinematic_speed_mps",
            "kinematic_acceleration_mps2",
            "kinematic_yaw_change_rad",
            "kinematic_residual_m",
            "unknown_aware_fixed_risk_uncalibrated",
        )
    )
    return IncrementalDependencyMap(
        tuple((frame, sender) for frame in sorted(set(affected_interval_frames))),
        tuple((frame, sender) for frame in sorted(set(affected_interval_frames))),
        (),
        keys,
        (),
    )


def spatial_dependency_map(
    manifest: AttackManifest,
    *,
    affected_object_ids: tuple[str | int, ...],
    all_agents: tuple[str, ...],
) -> IncrementalDependencyMap:
    """Expand overlap and leave-one-sender-out consequences to all companions."""

    frame = manifest.target.sync_frame_id
    sender = manifest.target.sender_id
    spatial_names = (
        "point_count",
        "count_deficit_residual_development",
        "count_surplus_residual_development",
    )
    peer_names = (
        "eligible_peer_count",
        "positive_peer_count",
        "peer_count_deficit_residual",
        "peer_count_surplus_residual",
        "peer_count_disagreement",
        "peer_corroboration_score",
    )
    keys = [
        EvidenceKey(
            manifest.target.sequence_id,
            manifest.target.session_id,
            frame,
            sender,
            "object",
            name,
            object_id,
        )
        for object_id in affected_object_ids
        for name in spatial_names
    ]
    keys.extend(
        EvidenceKey(
            manifest.target.sequence_id,
            manifest.target.session_id,
            frame,
            agent,
            "object",
            name,
            object_id,
        )
        for object_id in affected_object_ids
        for agent in all_agents
        for name in peer_names
    )
    keys.extend(
        EvidenceKey(
            manifest.target.sequence_id,
            manifest.target.session_id,
            frame,
            agent,
            "frame",
            name,
        )
        for agent in all_agents
        for name in (
            "maximum_object_suspicion",
            "median_object_suspicion",
            "unknown_aware_fixed_risk_uncalibrated",
        )
    )
    return IncrementalDependencyMap(
        ((frame, sender),),
        (),
        tuple((frame, item) for item in affected_object_ids),
        tuple(sorted(set(keys))),
        tuple(sorted(agent for agent in all_agents if agent != sender)),
    )


def evidence_row_hash(row: pd.Series) -> str:
    values = {
        key: (None if pd.isna(value) else value) for key, value in row.to_dict().items()
    }
    return sha256_bytes(canonical_json_bytes(values))


def evidence_table_hash(table: pd.DataFrame) -> str:
    """Hash ordered logical rows independently from Parquet container bytes."""

    return sha256_bytes(
        canonical_json_bytes([evidence_row_hash(row) for _, row in table.iterrows()])
    )


def apply_evidence_delta(
    clean: pd.DataFrame, delta: pd.DataFrame, *, key_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Overlay exact replacement rows while preserving untouched row bytes logically."""

    if delta.empty:
        return clean.copy()
    if (
        clean.duplicated(list(key_columns)).any()
        or delta.duplicated(list(key_columns)).any()
    ):
        raise OverlayError("clean and delta evidence keys must be unique")
    clean_indexed = clean.set_index(list(key_columns), drop=False)
    delta_indexed = delta.set_index(list(key_columns), drop=False)
    unknown = delta_indexed.index.difference(clean_indexed.index)
    if len(unknown):
        raise OverlayError(f"evidence delta contains unknown keys: {list(unknown)}")
    merged = clean_indexed.copy()
    merged.loc[delta_indexed.index, clean.columns] = delta_indexed[clean.columns]
    return merged.reset_index(drop=True)[clean.columns]


class VariantArtifactWriter:
    """Publish exactly one immutable variant directory without touching clean data."""

    def __init__(self, artifact_root: str | Path, manifest: AttackManifest) -> None:
        self.manifest = manifest
        self.directory = (
            Path(artifact_root).expanduser().resolve()
            / manifest.experiment_id
            / "sequences"
            / manifest.target.sequence_id
            / "variants"
            / manifest.variant_id
        )

    def begin(self) -> Path:
        if self.directory.exists():
            raise OverlayError(f"immutable variant already exists: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "attack_manifest.json").write_bytes(
            self.manifest.canonical_bytes()
        )
        return self.directory

    def write_json(self, name: str, value: object) -> Path:
        if not name.endswith(".json") or Path(name).name != name:
            raise OverlayError("artifact JSON name must be a local .json filename")
        path = self.directory / name
        if path.exists():
            raise OverlayError(f"artifact is immutable: {path}")
        temporary = self.directory / f".{name}.tmp"
        temporary.write_bytes(canonical_json_bytes(value))
        os.replace(temporary, path)
        return path

    def write_evidence_delta(self, table: pd.DataFrame) -> Path:
        path = self.directory / "evidence_delta.parquet"
        if path.exists():
            raise OverlayError(f"artifact is immutable: {path}")
        temporary = self.directory / ".evidence_delta.parquet.tmp"
        arrow = pa.Table.from_pandas(table, preserve_index=False)
        arrow = arrow.replace_schema_metadata({b"lidar_shield": b"attack-delta-v1"})
        pq.write_table(
            arrow,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
        )
        if pq.read_table(temporary).num_rows != len(table):
            raise OverlayError("evidence delta failed reload verification")
        os.replace(temporary, path)
        return path

    def finalize(self, outcome: AttemptOutcome, *, overlay_name: str) -> dict[str, str]:
        if not (self.directory / overlay_name).is_file():
            raise OverlayError("variant cannot finalize without its overlay artifact")
        self.write_json(
            "realized_effect.json",
            {
                "schema_version": "1.0.0",
                "episode_id": self.manifest.episode_id,
                "variant_id": self.manifest.variant_id,
                "status": outcome.status,
                "reason_code": outcome.reason_code,
                "injection_succeeded": outcome.injection_succeeded,
                "intended_target_labels": list(outcome.intended_target_labels),
                "actual_affected_evidence_keys": [
                    item.__dict__ for item in outcome.actual_affected_evidence_keys
                ],
                "requested_physical_effect": outcome.requested_physical_effect,
                "realized_physical_effect": outcome.realized_physical_effect,
                "realized_evidence_effect": outcome.realized_evidence_effect,
            },
        )
        names = (
            "attack_manifest.json",
            overlay_name,
            "evidence_delta.parquet",
            "realized_effect.json",
        )
        return {name: sha256_file(self.directory / name) for name in names}


def load_realized_effect(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OverlayError(f"cannot read realized effect: {exc}") from exc
    if not isinstance(value, dict):
        raise OverlayError("realized effect must be a JSON object")
    return value
