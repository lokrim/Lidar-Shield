"""Validated generation of immutable M4 variant artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from lidar_shield.attacks.kinematic import apply_velocity_overlay
from lidar_shield.attacks.manifests import (
    AttackManifest,
    AttackManifestError,
    KinematicParameters,
    SpatialParameters,
    validate_manifest_source,
)
from lidar_shield.attacks.overlays import (
    EVIDENCE_DELTA_KEY,
    AttemptOutcome,
    EvidenceKey,
    OverlayError,
    VariantArtifactWriter,
    apply_evidence_delta,
    evidence_table_hash,
    kinematic_dependency_map,
    spatial_dependency_map,
)
from lidar_shield.attacks.spatial import (
    add_points_inside_box,
    remove_points_inside_box,
)
from lidar_shield.data.adapters.mixed_signals import (
    load_production_point_cloud,
    read_odometry,
    save_production_point_cloud,
)
from lidar_shield.evaluation.splits import load_split_registry
from lidar_shield.evidence.corroboration import leave_one_sender_out_counts
from lidar_shield.evidence.kinematics import displacement_self_consistency
from lidar_shield.evidence.spatial import (
    DevelopmentCountReference,
    measure_spatial_support,
)
from lidar_shield.geometry.boxes import YawBox
from lidar_shield.geometry.frames import RigidTransform
from lidar_shield.manifest import production_code_sha256, sha256_file
from lidar_shield.trust.fixed_baseline import unknown_aware_fixed_baseline


def _validate_clean_manifest_bindings(
    manifest: AttackManifest, clean_dir: Path
) -> dict[str, Any]:
    try:
        clean_manifest = json.loads(
            (clean_dir / "source_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AttackManifestError(f"cannot read clean source manifest: {exc}") from exc
    if not isinstance(clean_manifest, dict):
        raise AttackManifestError("clean source manifest must be a JSON object")
    if clean_manifest.get("archive", {}).get("sha256") != (
        manifest.source_hashes.archive_sha256
    ):
        raise AttackManifestError("archive hash conflicts with clean manifest")
    if clean_manifest.get("selected_source_sha256") != (
        manifest.source_hashes.selected_source_sha256
    ):
        raise AttackManifestError("selected-source hash conflicts with clean manifest")
    return clean_manifest


def _validate(
    manifest: AttackManifest,
    *,
    clean_dir: Path,
    data_root: Path,
    split_registry_path: Path,
) -> None:
    if manifest.versions.code_sha256 != production_code_sha256():
        raise AttackManifestError("manifest production-code hash is stale")
    registry, registry_hash = load_split_registry(split_registry_path)
    validate_manifest_source(
        manifest,
        split_registry=registry,
        split_registry_sha256=registry_hash,
        clean_dir=clean_dir,
        data_root=data_root,
    )
    clean_manifest = _validate_clean_manifest_bindings(manifest, clean_dir)
    archive_path = data_root / "train" / f"{manifest.target.sequence_id}.tar"
    if (
        archive_path.exists()
        and sha256_file(archive_path) != clean_manifest["archive"]["sha256"]
    ):
        raise AttackManifestError("official source archive hash is stale")
    frame_table = pd.read_parquet(clean_dir / "frame.parquet")
    frame_match = frame_table.loc[
        frame_table["sequence_id"].eq(manifest.target.sequence_id)
        & frame_table["session_id"].eq(manifest.target.session_id)
        & frame_table["sync_frame_id"].eq(manifest.target.sync_frame_id)
        & frame_table["decision_time_ns"].eq(manifest.target.decision_time_ns)
        & frame_table["source_time_ns"].eq(manifest.target.source_time_ns)
        & frame_table["agent_id"].eq(manifest.target.sender_id)
        & frame_table["agent_type"].eq(manifest.target.agent_type)
    ]
    if len(frame_match) != 1:
        raise AttackManifestError("manifest target does not match one clean frame")
    if manifest.target.object_id_or_local_index is not None:
        object_table = pd.read_parquet(clean_dir / "object.parquet")
        object_match = object_table.loc[
            object_table["sync_frame_id"].eq(manifest.target.sync_frame_id)
            & object_table["agent_id"].eq(manifest.target.sender_id)
            & object_table["object_id_or_local_index"].eq(
                manifest.target.object_id_or_local_index
            )
        ]
        if len(object_match) != 1:
            raise AttackManifestError(
                "manifest spatial target does not match one clean object"
            )


def _clean_value(
    evidence: pd.DataFrame,
    *,
    frame: int,
    agent: str,
    name: str,
    object_id: str | int | None = None,
) -> float | None:
    mask = (
        evidence["sync_frame_id"].eq(frame)
        & evidence["agent_id"].eq(agent)
        & evidence["evidence_name"].eq(name)
    )
    if object_id is None:
        mask &= evidence["object_id_or_local_index"].isna()
    else:
        mask &= evidence["object_id_or_local_index"].eq(object_id)
    rows = evidence.loc[mask, "value"]
    if len(rows) != 1:
        return None
    value = rows.iloc[0]
    return None if pd.isna(value) else float(value)


def _replacement_row(
    evidence: pd.DataFrame,
    *,
    frame: int,
    agent: str,
    name: str,
    value: float | int | bool | None,
    object_id: str | int | None = None,
) -> pd.Series | None:
    mask = (
        evidence["sync_frame_id"].eq(frame)
        & evidence["agent_id"].eq(agent)
        & evidence["evidence_name"].eq(name)
    )
    if object_id is None:
        mask &= evidence["object_id_or_local_index"].isna()
    else:
        mask &= evidence["object_id_or_local_index"].eq(object_id)
    indexes = evidence.index[mask]
    if len(indexes) != 1:
        return None
    row = evidence.loc[indexes[0]].copy()
    row["value"] = value
    row["available"] = value is not None
    row["reason_code"] = "matched" if value is not None else "incomplete_evidence"
    return row


def _key_from_row(row: pd.Series) -> EvidenceKey:
    object_id = row["object_id_or_local_index"]
    return EvidenceKey(
        str(row["sequence_id"]),
        str(row["session_id"]),
        int(row["sync_frame_id"]),
        str(row["agent_id"]),
        str(row["evidence_level"]),  # type: ignore[arg-type]
        str(row["evidence_name"]),
        None if pd.isna(object_id) else object_id,
    )


def _assert_sources_unchanged(
    manifest: AttackManifest, *, clean_dir: Path, raw_path: Path, data_root: Path
) -> None:
    checks = {
        raw_path: manifest.source_hashes.target_raw_sha256,
        clean_dir
        / "source_manifest.json": manifest.source_hashes.clean_manifest_sha256,
        clean_dir / "frame.parquet": manifest.source_hashes.clean_frame_sha256,
        clean_dir / "object.parquet": manifest.source_hashes.clean_object_sha256,
        clean_dir / "evidence.parquet": manifest.source_hashes.clean_evidence_sha256,
    }
    archive_path = data_root / "train" / f"{manifest.target.sequence_id}.tar"
    if archive_path.exists():
        checks[archive_path] = manifest.source_hashes.archive_sha256
    changed = [path for path, digest in checks.items() if sha256_file(path) != digest]
    if changed:
        raise OverlayError(f"source or clean cache changed during attack: {changed}")


def run_kinematic_attack(
    manifest: AttackManifest,
    *,
    data_root: str | Path,
    clean_dir: str | Path,
    artifact_root: str | Path,
    split_registry_path: str | Path,
    max_gap_s: float = 0.25,
) -> Path:
    """Generate a sparse velocity overlay and incrementally recompute intervals."""

    if not isinstance(manifest.parameters, KinematicParameters):
        raise AttackManifestError("kinematic runner requires kinematic parameters")
    data = Path(data_root).expanduser().resolve()
    clean = Path(clean_dir).expanduser().resolve()
    split = Path(split_registry_path).expanduser().resolve()
    _validate(manifest, clean_dir=clean, data_root=data, split_registry_path=split)
    raw_path = data / manifest.target.raw_relative_path
    writer = VariantArtifactWriter(artifact_root, manifest)
    directory = writer.begin()
    try:
        samples = read_odometry(raw_path)
        overlay = apply_velocity_overlay(
            samples,
            selected_sample_times_ns=manifest.target.interval_sample_times_ns,
            delta_v_mps=manifest.parameters.delta_v_mps,
            mode=manifest.parameters.attack_kind,
        )
        frame_table = pd.read_parquet(clean / "frame.parquet")
        evidence = pd.read_parquet(clean / "evidence.parquet")
        sender_frames = frame_table.loc[
            frame_table["agent_id"].eq(manifest.target.sender_id),
            ["sync_frame_id", "source_time_ns"],
        ]
        frame_by_time = {
            int(row.source_time_ns): int(row.sync_frame_id)
            for row in sender_frames.itertuples()
            if not pd.isna(row.source_time_ns)
        }
        positions = {sample.time_ns: index for index, sample in enumerate(samples)}
        affected_frames = tuple(
            sorted(
                frame_by_time[time_ns]
                for time_ns in overlay.affected_interval_end_times_ns
                if time_ns in frame_by_time
            )
        )
        dependency = kinematic_dependency_map(
            manifest, affected_interval_frames=affected_frames
        )
        replacement_rows: list[pd.Series] = []
        affected_keys: list[EvidenceKey] = []
        residual_deltas: dict[str, float] = {}
        for endpoint_time in overlay.affected_interval_end_times_ns:
            frame = frame_by_time.get(endpoint_time)
            index = positions[endpoint_time]
            if frame is None:
                continue
            clean_result = displacement_self_consistency(
                samples[index - 1], samples[index], max_gap_s=max_gap_s
            )
            attacked_result = displacement_self_consistency(
                overlay.samples[index - 1], overlay.samples[index], max_gap_s=max_gap_s
            )
            values = {
                "kinematic_dt_s": attacked_result.dt_s,
                "kinematic_speed_mps": attacked_result.speed_mps,
                "kinematic_acceleration_mps2": attacked_result.acceleration_mps2,
                "kinematic_yaw_change_rad": attacked_result.yaw_change_rad,
                "kinematic_residual_m": attacked_result.residual_m,
            }
            for name, value in values.items():
                row = _replacement_row(
                    evidence,
                    frame=frame,
                    agent=manifest.target.sender_id,
                    name=name,
                    value=value,
                )
                if row is not None:
                    replacement_rows.append(row)
                    if (
                        _clean_value(
                            evidence,
                            frame=frame,
                            agent=manifest.target.sender_id,
                            name=name,
                        )
                        != value
                    ):
                        affected_keys.append(_key_from_row(row))
            if (
                clean_result.residual_m is not None
                and attacked_result.residual_m is not None
            ):
                residual_deltas[str(frame)] = (
                    attacked_result.residual_m - clean_result.residual_m
                )
            count_residual = _clean_value(
                evidence,
                frame=frame,
                agent=manifest.target.sender_id,
                name="maximum_object_suspicion",
            )
            peer_rows = evidence.loc[
                evidence["sync_frame_id"].eq(frame)
                & evidence["agent_id"].eq(manifest.target.sender_id)
                & evidence["evidence_name"].eq("peer_corroboration_score"),
                "value",
            ].dropna()
            peer_score = float(peer_rows.min()) if len(peer_rows) else None
            risk = unknown_aware_fixed_baseline(
                kinematic_error_m=attacked_result.residual_m,
                two_sided_count_residual=count_residual,
                peer_corroboration_score=peer_score,
                kinematic_applicable=True,
            )
            row = _replacement_row(
                evidence,
                frame=frame,
                agent=manifest.target.sender_id,
                name="unknown_aware_fixed_risk_uncalibrated",
                value=risk.risk,
            )
            if row is not None:
                replacement_rows.append(row)
                affected_keys.append(_key_from_row(row))
        delta = pd.DataFrame(replacement_rows, columns=evidence.columns)
        attacked_evidence_hash = evidence_table_hash(
            apply_evidence_delta(evidence, delta, key_columns=EVIDENCE_DELTA_KEY)
        )
        writer.write_json(
            "overlay.json",
            {
                "schema_version": "1.0.0",
                "kind": manifest.parameters.attack_kind,
                "changes": [item.__dict__ for item in overlay.changes],
                "affected_interval_end_times_ns": list(
                    overlay.affected_interval_end_times_ns
                ),
                "dependency_map": {
                    "changed_observations": dependency.changed_observations,
                    "affected_kinematic_intervals": (
                        dependency.affected_kinematic_intervals
                    ),
                    "affected_evidence_keys": [
                        item.__dict__ for item in dependency.affected_evidence_keys
                    ],
                },
            },
        )
        writer.write_evidence_delta(delta)
        maximum = max((abs(value) for value in residual_deltas.values()), default=0.0)
        status: Literal[
            "succeeded", "zero_effect", "low_effect", "failed", "interrupted"
        ]
        if maximum == 0.0:
            status = "zero_effect"
        elif maximum < 0.01:
            status = "low_effect"
        else:
            status = "succeeded"
        outcome = AttemptOutcome(
            status,
            "no_evidence_change" if status == "zero_effect" else "applied",
            True,
            tuple(
                f"sample:{item}" for item in manifest.target.interval_sample_times_ns
            ),
            tuple(sorted(set(affected_keys))),
            {
                "delta_v_mps": list(manifest.parameters.delta_v_mps),
                "duration_samples": manifest.parameters.duration_samples,
            },
            {"modified_sample_count": len(overlay.changes)},
            {
                "kinematic_residual_delta_m_by_frame": residual_deltas,
                "maximum_absolute_delta_m": maximum,
                "attacked_evidence_sha256": attacked_evidence_hash,
            },
        )
        writer.finalize(outcome, overlay_name="overlay.json")
        _assert_sources_unchanged(
            manifest, clean_dir=clean, raw_path=raw_path, data_root=data
        )
        return directory
    except Exception as exc:
        _preserve_failed_attempt(writer, exc, overlay_name="overlay.json")
        _assert_sources_unchanged(
            manifest, clean_dir=clean, raw_path=raw_path, data_root=data
        )
        raise


def run_spatial_attack(
    manifest: AttackManifest,
    *,
    data_root: str | Path,
    clean_dir: str | Path,
    artifact_root: str | Path,
    split_registry_path: str | Path,
    raw_to_common: RigidTransform | None,
    count_reference: DevelopmentCountReference | None = None,
) -> Path:
    """Generate/reload a derived PCD and recompute overlap/peer dependencies."""

    if not isinstance(manifest.parameters, SpatialParameters):
        raise AttackManifestError("spatial runner requires spatial parameters")
    data = Path(data_root).expanduser().resolve()
    clean = Path(clean_dir).expanduser().resolve()
    split = Path(split_registry_path).expanduser().resolve()
    _validate(manifest, clean_dir=clean, data_root=data, split_registry_path=split)
    raw_path = data / manifest.target.raw_relative_path
    writer = VariantArtifactWriter(artifact_root, manifest)
    directory = writer.begin()
    try:
        if raw_to_common is None:
            raise OverlayError("unavailable_target:unresolved_transform")
        box_spec = manifest.parameters.box
        box = YawBox(
            box_spec.center_m, box_spec.size_m, box_spec.yaw_rad, box_spec.frame
        )
        source_cloud = load_production_point_cloud(raw_path)
        if manifest.parameters.attack_kind == "point_addition":
            assert manifest.parameters.n_add is not None
            mutation = add_points_inside_box(
                source_cloud,
                box,
                n_add=manifest.parameters.n_add,
                seed=manifest.seed,
                raw_to_common=raw_to_common,
                boundary_tolerance_m=manifest.parameters.boundary_tolerance_m,
            )
        else:
            assert manifest.parameters.removal_fraction is not None
            mutation = remove_points_inside_box(
                source_cloud,
                box,
                removal_fraction=manifest.parameters.removal_fraction,
                seed=manifest.seed,
                raw_to_common=raw_to_common,
                boundary_tolerance_m=manifest.parameters.boundary_tolerance_m,
            )
        overlay_path = directory / "overlay.pcd"
        save_production_point_cloud(mutation.cloud, overlay_path)
        reloaded = load_production_point_cloud(overlay_path)
        if (
            reloaded.metadata.fields != source_cloud.metadata.fields
            or reloaded.metadata.sizes != source_cloud.metadata.sizes
            or reloaded.metadata.types != source_cloud.metadata.types
            or reloaded.metadata.counts != source_cloud.metadata.counts
            or reloaded.metadata.encoding != source_cloud.metadata.encoding
            or reloaded.pc_data.dtype != source_cloud.pc_data.dtype
        ):
            raise OverlayError("production reload changed PCD field/dtype/encoding")
        objects = pd.read_parquet(clean / "object.parquet")
        evidence = pd.read_parquet(clean / "evidence.parquet")
        frame_objects = objects.loc[
            objects["sync_frame_id"].eq(manifest.target.sync_frame_id)
            & objects["agent_id"].eq(manifest.target.sender_id)
        ]
        common_before = raw_to_common.apply(
            np.column_stack(
                (
                    source_cloud.pc_data["x"],
                    source_cloud.pc_data["y"],
                    source_cloud.pc_data["z"],
                )
            )
        )
        common_after = raw_to_common.apply(
            np.column_stack(
                (
                    reloaded.pc_data["x"],
                    reloaded.pc_data["y"],
                    reloaded.pc_data["z"],
                )
            )
        )
        changed_counts: dict[str | int, int] = {}
        changed_spatial: dict[str | int, tuple[float | None, float | None]] = {}
        for row in frame_objects.itertuples():
            candidate = YawBox(
                (row.center_x_m, row.center_y_m, row.center_z_m),
                (row.size_x_m, row.size_y_m, row.size_z_m),
                row.yaw_rad,
                row.box_frame,
            )
            before_result = measure_spatial_support(
                common_before,
                candidate,
                class_id=int(row.class_id),
                visibility_eligible=True,
                transform_quality="verified",
                count_reference=count_reference,
                boundary_tolerance_m=manifest.parameters.boundary_tolerance_m,
            )
            after_result = measure_spatial_support(
                common_after,
                candidate,
                class_id=int(row.class_id),
                visibility_eligible=True,
                transform_quality="verified",
                count_reference=count_reference,
                boundary_tolerance_m=manifest.parameters.boundary_tolerance_m,
            )
            before = before_result.point_count
            after = after_result.point_count
            assert before is not None and after is not None
            if before != after:
                changed_counts[row.object_id_or_local_index] = after
                changed_spatial[row.object_id_or_local_index] = (
                    after_result.deficit_residual,
                    after_result.surplus_residual,
                )
        all_agents = tuple(sorted(evidence["agent_id"].dropna().unique().tolist()))
        dependency = spatial_dependency_map(
            manifest,
            affected_object_ids=tuple(sorted(changed_counts, key=str)),
            all_agents=all_agents,
        )
        replacement_rows: list[pd.Series] = []
        affected_keys: list[EvidenceKey] = []
        for object_id, attacked_count in changed_counts.items():
            row = _replacement_row(
                evidence,
                frame=manifest.target.sync_frame_id,
                agent=manifest.target.sender_id,
                name="point_count",
                object_id=object_id,
                value=attacked_count,
            )
            if row is not None:
                replacement_rows.append(row)
                affected_keys.append(_key_from_row(row))
            deficit, surplus = changed_spatial[object_id]
            for name, value in (
                ("count_deficit_residual_development", deficit),
                ("count_surplus_residual_development", surplus),
            ):
                replacement = _replacement_row(
                    evidence,
                    frame=manifest.target.sync_frame_id,
                    agent=manifest.target.sender_id,
                    name=name,
                    object_id=object_id,
                    value=value,
                )
                if replacement is not None:
                    replacement_rows.append(replacement)
                    affected_keys.append(_key_from_row(replacement))
            counts = {
                agent: int(
                    attacked_count
                    if agent == manifest.target.sender_id
                    else (
                        _clean_value(
                            evidence,
                            frame=manifest.target.sync_frame_id,
                            agent=agent,
                            name="point_count",
                            object_id=object_id,
                        )
                        or 0
                    )
                )
                for agent in all_agents
            }
            peer = leave_one_sender_out_counts(
                counts, {agent: True for agent in all_agents}
            )
            for agent, result in peer.items():
                peer_values = {
                    "eligible_peer_count": result.eligible_peer_count,
                    "positive_peer_count": result.positive_peer_count,
                    "peer_count_deficit_residual": result.deficit_residual,
                    "peer_count_surplus_residual": result.surplus_residual,
                    "peer_count_disagreement": result.disagreement,
                    "peer_corroboration_score": result.peer_corroboration_score,
                }
                for name, value in peer_values.items():
                    replacement = _replacement_row(
                        evidence,
                        frame=manifest.target.sync_frame_id,
                        agent=agent,
                        name=name,
                        object_id=object_id,
                        value=value,
                    )
                    if replacement is not None:
                        replacement_rows.append(replacement)
                        affected_keys.append(_key_from_row(replacement))
        delta = pd.DataFrame(replacement_rows, columns=evidence.columns)
        attacked_evidence_hash = evidence_table_hash(
            apply_evidence_delta(evidence, delta, key_columns=EVIDENCE_DELTA_KEY)
        )
        writer.write_evidence_delta(delta)
        status: Literal[
            "succeeded", "zero_effect", "low_effect", "failed", "interrupted"
        ]
        if not changed_counts:
            status = "zero_effect"
        elif mutation.realized_count <= 1:
            status = "low_effect"
        else:
            status = "succeeded"
        outcome = AttemptOutcome(
            status,
            "no_evidence_change" if not changed_counts else "applied",
            True,
            (f"object:{manifest.target.object_id_or_local_index}",),
            tuple(sorted(set(affected_keys))),
            {
                "requested_count": mutation.requested_count,
                "attack_kind": manifest.parameters.attack_kind,
            },
            {
                "realized_count": mutation.realized_count,
                "before_target_count": mutation.before_inside_count,
                "after_target_count": mutation.after_inside_count,
            },
            {
                "changed_object_counts": {
                    str(key): value for key, value in changed_counts.items()
                },
                "honest_companion_agents": list(dependency.honest_companion_agents),
                "attacked_evidence_sha256": attacked_evidence_hash,
            },
        )
        writer.finalize(outcome, overlay_name="overlay.pcd")
        _assert_sources_unchanged(
            manifest, clean_dir=clean, raw_path=raw_path, data_root=data
        )
        return directory
    except Exception as exc:
        _preserve_failed_attempt(writer, exc, overlay_name="overlay.pcd")
        _assert_sources_unchanged(
            manifest, clean_dir=clean, raw_path=raw_path, data_root=data
        )
        raise


def _preserve_failed_attempt(
    writer: VariantArtifactWriter, exc: Exception, *, overlay_name: str
) -> None:
    overlay = writer.directory / overlay_name
    if not overlay.exists():
        if overlay_name.endswith(".json"):
            writer.write_json(overlay_name, {"status": "not_applied"})
        else:
            # A failed spatial attempt still has an explicit sparse overlay record.
            (writer.directory / "overlay.failure.json").write_bytes(
                json.dumps(
                    {"status": "not_applied", "reason": str(exc)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            overlay_name = "overlay.failure.json"
    delta = writer.directory / "evidence_delta.parquet"
    if not delta.exists():
        writer.write_evidence_delta(
            pd.DataFrame({"reason_code": pd.Series(dtype="string")})
        )
    if not (writer.directory / "realized_effect.json").exists():
        writer.finalize(
            AttemptOutcome(
                "failed",
                str(exc),
                False,
                (),
                (),
                {},
                {},
                {},
            ),
            overlay_name=overlay_name,
        )


def record_interrupted_attempt(
    writer: VariantArtifactWriter, *, reason_code: str = "interrupted"
) -> None:
    """Persist a caller-observed interruption in the same coverage denominator."""

    writer.begin()
    writer.write_json("overlay.json", {"status": "not_applied"})
    writer.write_evidence_delta(
        pd.DataFrame({"reason_code": pd.Series(dtype="string")})
    )
    writer.finalize(
        AttemptOutcome("interrupted", reason_code, False, (), (), {}, {}, {}),
        overlay_name="overlay.json",
    )
