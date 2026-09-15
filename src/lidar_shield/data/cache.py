"""Verified, hash-stable per-sequence clean evidence caches."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from lidar_shield import __version__
from lidar_shield.contracts.reasons import Applicability, MatchPolicy, ReasonCode
from lidar_shield.contracts.schemas import (
    CONTRACT_SCHEMA_VERSION,
    EvidenceEnvelope,
    FrameRecord,
    ObjectRecord,
    evidence_records_to_table,
    frame_records_to_table,
    object_records_to_table,
)
from lidar_shield.data.adapters.mixed_signals import (
    MixedSignalsConfig,
    SequenceIndex,
    audit_filename_odometry_agreement,
    index_sequence,
    load_point_measurements,
    read_odometry,
)
from lidar_shield.data.adapters.oracle_objects import parse_oracle_labels
from lidar_shield.data.index import load_agent_registry
from lidar_shield.evidence.aggregation import summarize_object_suspicion
from lidar_shield.evidence.corroboration import leave_one_sender_out_counts
from lidar_shield.evidence.kinematics import (
    KinematicResult,
    displacement_self_consistency,
)
from lidar_shield.evidence.spatial import (
    DevelopmentCountReference,
    SpatialResult,
    measure_spatial_support,
)
from lidar_shield.geometry.boxes import YawBox
from lidar_shield.geometry.frames import resolve_extrinsics
from lidar_shield.geometry.visibility import visibility_eligibility
from lidar_shield.manifest import canonical_json_bytes, sha256_bytes, sha256_file
from lidar_shield.trust.fixed_baseline import (
    historical_fixed_baseline,
    raw_point_share,
    unknown_aware_fixed_baseline,
)


class CacheError(ValueError):
    """A clean cache could not be built or verified."""


class StaleCacheError(CacheError):
    """A completed cache does not match its declared request or hashes."""


class IncompleteCacheError(CacheError):
    """Partial outputs exist without a verified completion marker."""


@dataclass(frozen=True)
class CacheBuildConfig:
    data_root: Path
    artifact_root: Path
    experiment_id: str
    sequence_id: str
    selected_frames: tuple[int, ...]
    selected_agents: tuple[str, ...]
    agent_registry_path: Path
    split_registry_path: Path
    dataset_config_path: Path
    producer_command: str
    variant_id: str = "clean"
    session_id: str = "mini_7-session"
    split_role: str = "regression_development"
    max_gap_s: float = 0.25

    def __post_init__(self) -> None:
        for name in ("experiment_id", "sequence_id", "variant_id", "session_id"):
            if not getattr(self, name):
                raise CacheError(f"{name} must be non-empty")
        if not self.selected_frames or any(value < 0 for value in self.selected_frames):
            raise CacheError("selected frames must be non-empty and nonnegative")
        if tuple(sorted(set(self.selected_frames))) != self.selected_frames:
            raise CacheError("selected frames must be sorted and unique")
        if not self.selected_agents or len(set(self.selected_agents)) != len(
            self.selected_agents
        ):
            raise CacheError("selected agents must be non-empty and unique")
        if not math.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise CacheError("max_gap_s must be finite and positive")

    @property
    def clean_dir(self) -> Path:
        return (
            Path(self.artifact_root).expanduser().resolve()
            / self.experiment_id
            / "sequences"
            / self.sequence_id
            / "clean"
        )


@dataclass(frozen=True)
class CacheBuildResult:
    clean_dir: Path
    output_hashes: dict[str, str]
    manifest_sha256: str
    resumed: bool


OUTPUT_NAMES = ("evidence.parquet", "frame.parquet", "object.parquet")
MANIFEST_NAME = "source_manifest.json"
MARKER_NAME = "resume.complete.json"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CacheError(f"cannot read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheError(f"configuration root must be a mapping: {path}")
    return value


def _adapter_config(
    config: CacheBuildConfig,
) -> tuple[MixedSignalsConfig, dict[str, Any]]:
    declaration = _load_yaml_mapping(Path(config.dataset_config_path).resolve())
    try:
        layout = declaration["layout"]
        time = declaration["time"]
        point = declaration["point_cloud"]
    except KeyError as exc:
        raise CacheError(f"dataset configuration lacks {exc.args[0]!r}") from exc
    if declaration.get("sequence_id") != config.sequence_id:
        raise CacheError("dataset configuration sequence does not match request")
    from lidar_shield.data.adapters.mixed_signals import FractionalTimeConvention

    adapter = MixedSignalsConfig(
        data_root=Path(config.data_root),
        sequence_id=config.sequence_id,
        agents=tuple(str(value) for value in layout["agents"]),
        top_anchor_agent=str(layout["top_anchor_agent"]),
        label_file_index_offset=int(layout["label_file_index_offset"]),
        fractional_time_convention=FractionalTimeConvention(
            time["filename_fractional_convention"]
        ),
        intensity_divisor=float(point["intensity_normalization"]["divisor"]),
        expected_sync_start=int(layout["expected_sync_start"]),
        expected_sync_stop=int(layout["expected_sync_stop"]),
        expected_label_stop=int(layout["expected_label_stop"]),
    )
    return adapter, declaration


def _source_file_records(
    config: CacheBuildConfig, index: SequenceIndex
) -> tuple[dict[str, Any], ...]:
    paths = [
        index.cloud(agent, frame).path
        for frame in config.selected_frames
        for agent in config.selected_agents
    ]
    paths.extend(index.odometry.values())
    paths.extend(
        index.labels[frame] for frame in config.selected_frames if frame in index.labels
    )
    root = index.config.data_root
    records = []
    for path in sorted(
        set(paths), key=lambda value: value.relative_to(root).as_posix()
    ):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(records)


def _request_manifest(
    config: CacheBuildConfig,
    index: SequenceIndex,
    declaration: dict[str, Any],
    source_files: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    archive = index.config.archive_path
    registry = load_agent_registry(config.agent_registry_path)
    transform_records = {
        agent.agent_id: {
            "from_frame": agent.extrinsics.from_frame,
            "to_frame": agent.extrinsics.to_frame,
            "status": agent.extrinsics.status,
            "matrix_4x4": agent.extrinsics.matrix_4x4,
            "note": agent.extrinsics.note,
        }
        for agent in registry.agents
        if agent.agent_id in config.selected_agents
    }
    source = declaration.get("source", {})
    source_digest = sha256_bytes(canonical_json_bytes(source_files))
    return {
        "schema_version": "2.0.0",
        "experiment_id": config.experiment_id,
        "variant_id": config.variant_id,
        "sequence_id": config.sequence_id,
        "session_id": config.session_id,
        "split_role": config.split_role,
        "configured_source_root": str(config.data_root),
        "resolved_source_root": index.config.data_root.as_posix(),
        "archive": {
            "path": archive.relative_to(index.config.data_root).as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        },
        "official_source": {
            "hugging_face_dataset": source.get("hugging_face_dataset"),
            "hugging_face_revision": source.get("hugging_face_revision"),
            "xet_object_id": source.get("xet_object_id"),
        },
        "selected_frames": list(config.selected_frames),
        "selected_agents": list(config.selected_agents),
        "selected_source_files": list(source_files),
        "selected_source_sha256": source_digest,
        "transforms": transform_records,
        "versions": {
            "package": __version__,
            "production_code_sha256": _production_code_sha256(),
            "contract_schema": CONTRACT_SCHEMA_VERSION,
            "cache_schema": "2.0.0",
            "dataset_config_schema": str(declaration.get("schema_version")),
        },
        "configuration_hashes": {
            "dataset": sha256_file(config.dataset_config_path),
            "agents": sha256_file(config.agent_registry_path),
            "split": sha256_file(config.split_registry_path),
        },
        "producer_command": config.producer_command,
    }


def _fingerprint(request: dict[str, Any]) -> str:
    compatible_request = {
        key: value for key, value in request.items() if key != "producer_command"
    }
    return sha256_bytes(canonical_json_bytes(compatible_request))


def _production_code_sha256() -> str:
    package_root = Path(__file__).parents[1]
    records = [
        {
            "path": path.relative_to(package_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(package_root.rglob("*.py"))
    ]
    return sha256_bytes(canonical_json_bytes(records))


def _frame_record(
    config: CacheBuildConfig,
    index: SequenceIndex,
    agent_type: Any,
    agent: str,
    frame: int,
    *,
    lookahead_ns: int,
) -> FrameRecord:
    cloud = index.cloud(agent, frame)
    decision = index.cloud(index.config.top_anchor_agent, frame).source_time_ns
    age = decision - cloud.source_time_ns
    if age < -lookahead_ns:
        raise CacheError(f"source exceeds configured look-ahead: {agent}/{frame}")
    return FrameRecord(
        experiment_id=config.experiment_id,
        variant_id=config.variant_id,
        sequence_id=config.sequence_id,
        session_id=config.session_id,
        sync_frame_id=frame,
        decision_time_ns=decision,
        source_time_ns=cloud.source_time_ns,
        agent_id=agent,
        agent_type=agent_type,
        member=True,
        message_present=True,
        source_matched=True,
        evidence_available=True,
        reason_code=ReasonCode.MATCHED,
        match_policy=MatchPolicy.BUFFERED_NEAREST,
        match_age_ns=age,
        lookahead_buffer_ns=lookahead_ns,
        pose_source_time_ns=(cloud.source_time_ns if agent in index.odometry else None),
        pose_age_ns=(age if agent in index.odometry else None),
        oracle_gt=True,
    )


def _evidence(
    frame: FrameRecord,
    name: str,
    value: float | int | bool | None,
    *,
    object_id: int | None = None,
    applicable: bool = True,
    reason: ReasonCode = ReasonCode.MATCHED,
) -> EvidenceEnvelope:
    available = value is not None
    return EvidenceEnvelope(
        experiment_id=frame.experiment_id,
        variant_id=frame.variant_id,
        sequence_id=frame.sequence_id,
        session_id=frame.session_id,
        sync_frame_id=frame.sync_frame_id,
        decision_time_ns=frame.decision_time_ns,
        source_time_ns=frame.source_time_ns,
        agent_id=frame.agent_id,
        agent_type=frame.agent_type,
        object_id_or_local_index=object_id,
        match_policy=frame.match_policy,
        match_age_ns=frame.match_age_ns,
        lookahead_buffer_ns=frame.lookahead_buffer_ns,
        pose_source_time_ns=frame.pose_source_time_ns,
        pose_age_ns=frame.pose_age_ns,
        evidence_level="object" if object_id is not None else "frame",
        evidence_name=name,
        value=value,
        applicable=applicable,
        available=available,
        reason_code=ReasonCode.MATCHED if available else reason,
        oracle_gt=True,
    )


def _kinematics_for_frames(
    config: CacheBuildConfig,
    index: SequenceIndex,
) -> dict[tuple[str, int], KinematicResult]:
    result: dict[tuple[str, int], KinematicResult] = {}
    for agent, path in index.odometry.items():
        samples = read_odometry(path)
        by_time = {sample.time_ns: position for position, sample in enumerate(samples)}
        for frame in config.selected_frames:
            source_time = index.cloud(agent, frame).source_time_ns
            position = by_time.get(source_time)
            if position is None or position == 0:
                result[(agent, frame)] = KinematicResult(
                    False,
                    ReasonCode.OUT_OF_RANGE,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    10,
                )
            else:
                result[(agent, frame)] = displacement_self_consistency(
                    samples[position - 1],
                    samples[position],
                    max_gap_s=config.max_gap_s,
                )
    return result


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    arrow = arrow.replace_schema_metadata({b"lidar_shield": b"clean-cache-v2"})
    pq.write_table(
        arrow,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )


def _build_tables(
    config: CacheBuildConfig,
    index: SequenceIndex,
    declaration: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = load_agent_registry(config.agent_registry_path)
    capabilities = {agent.agent_id: agent for agent in registry.agents}
    if set(config.selected_agents) - set(capabilities):
        raise CacheError("selected agents are absent from the registry")
    if set(config.selected_frames) - set(index.sync_frame_ids):
        raise CacheError("selected frames are absent from the top-anchor index")
    if set(config.selected_agents) != set(index.config.agents):
        raise CacheError(
            "D1/M2 clean cache requires the complete five-agent membership"
        )
    time_config = declaration["time"]
    lookahead_ns = int(time_config["lookahead_buffer_ns"])
    max_age_ns = int(time_config["maximum_absolute_age_ns"])
    frame_records: list[FrameRecord] = []
    frame_extras: list[dict[str, Any]] = []
    object_records: list[ObjectRecord] = []
    object_extras: list[dict[str, Any]] = []
    evidence: list[EvidenceEnvelope] = []
    odometry = {agent: read_odometry(path) for agent, path in index.odometry.items()}
    if not all(audit_filename_odometry_agreement(index, odometry).values()):
        raise CacheError("vehicle filename timestamps do not agree with odometry")
    kinematics = _kinematics_for_frames(config, index)
    reference_config = declaration["development_count_reference"]
    reference = DevelopmentCountReference(
        expected_log1p_count=float(reference_config["expected_log1p_count"]),
        scale=float(reference_config["scale"]),
        reference_id=str(reference_config["reference_id"]),
    )

    for frame_id in config.selected_frames:
        labels = (
            parse_oracle_labels(index.labels[frame_id])
            if frame_id in index.labels
            else ()
        )
        frame_spatial: dict[str, list[SpatialResult]] = {}
        frames_by_agent: dict[str, FrameRecord] = {}
        points_by_agent: dict[str, np.ndarray | None] = {}
        for agent in config.selected_agents:
            capability = capabilities[agent]
            frame = _frame_record(
                config,
                index,
                capability.agent_type,
                agent,
                frame_id,
                lookahead_ns=lookahead_ns,
            )
            assert frame.match_age_ns is not None
            if abs(frame.match_age_ns) > max_age_ns:
                raise CacheError(f"source exceeds maximum age: {agent}/{frame_id}")
            frames_by_agent[agent] = frame
            frame_records.append(frame)
            measurement = load_point_measurements(
                index.cloud(agent, frame_id).path,
                intensity_divisor=index.config.intensity_divisor,
            )
            transform = resolve_extrinsics(capability.extrinsics)
            points_top = (
                transform.transform.apply(measurement.xyzi[:, :3])
                if transform.transform is not None
                else None
            )
            points_by_agent[agent] = points_top
            finite_intensity = measurement.normalized_intensity[
                np.isfinite(measurement.normalized_intensity)
            ]
            frame_extras.append(
                {
                    "cloud_path": index.cloud(agent, frame_id)
                    .path.relative_to(index.config.data_root)
                    .as_posix(),
                    "cloud_timestamp_text": index.cloud(agent, frame_id).timestamp_text,
                    "pcd_fields": ",".join(measurement.metadata.fields),
                    "pcd_types": ",".join(measurement.metadata.types),
                    "pcd_encoding": measurement.metadata.encoding,
                    "pcd_point_count": measurement.metadata.points,
                    "intensity_normalization_divisor": index.config.intensity_divisor,
                    "normalized_intensity_min": (
                        float(finite_intensity.min()) if finite_intensity.size else None
                    ),
                    "normalized_intensity_max": (
                        float(finite_intensity.max()) if finite_intensity.size else None
                    ),
                    "label_available": frame_id in index.labels,
                    "label_reason_code": (
                        ReasonCode.MATCHED.value
                        if frame_id in index.labels
                        else ReasonCode.LABEL_UNAVAILABLE.value
                    ),
                    "transform_quality": transform.quality,
                }
            )
            evidence.extend(
                (
                    _evidence(frame, "message_present", True),
                    _evidence(frame, "source_age_ns", frame.match_age_ns),
                    _evidence(
                        frame,
                        "pose_age_ns",
                        frame.pose_age_ns,
                        applicable=capability.odometry
                        is not Applicability.NOT_APPLICABLE,
                        reason=(
                            ReasonCode.NOT_APPLICABLE
                            if capability.odometry is Applicability.NOT_APPLICABLE
                            else ReasonCode.OUT_OF_RANGE
                        ),
                    ),
                    _evidence(
                        frame,
                        "transform_verified",
                        transform.transform is not None,
                    ),
                    _evidence(
                        frame, "raw_cloud_point_count", measurement.metadata.points
                    ),
                    _evidence(
                        frame,
                        "kinematic_modality_applicable",
                        capability.odometry is not Applicability.NOT_APPLICABLE,
                    ),
                )
            )
            if capability.odometry is Applicability.NOT_APPLICABLE:
                kin = None
                kin_reason = ReasonCode.NOT_APPLICABLE
            else:
                kin = kinematics[(agent, frame_id)]
                kin_reason = kin.reason_code
            for name, value in (
                ("kinematic_dt_s", kin.dt_s if kin else None),
                ("kinematic_speed_mps", kin.speed_mps if kin else None),
                ("kinematic_acceleration_mps2", kin.acceleration_mps2 if kin else None),
                ("kinematic_yaw_change_rad", kin.yaw_change_rad if kin else None),
                ("kinematic_residual_m", kin.residual_m if kin else None),
            ):
                evidence.append(
                    _evidence(
                        frame,
                        name,
                        value,
                        applicable=capability.odometry
                        is not Applicability.NOT_APPLICABLE,
                        reason=kin_reason,
                    )
                )

        for agent in config.selected_agents:
            frame = frames_by_agent[agent]
            assert frame.source_time_ns is not None
            assert frame.match_age_ns is not None
            capability = capabilities[agent]
            transform = resolve_extrinsics(capability.extrinsics)
            spatial_values: list[SpatialResult] = []
            for label in labels:
                record = ObjectRecord(
                    experiment_id=frame.experiment_id,
                    variant_id=frame.variant_id,
                    sequence_id=frame.sequence_id,
                    session_id=frame.session_id,
                    sync_frame_id=frame.sync_frame_id,
                    decision_time_ns=frame.decision_time_ns,
                    source_time_ns=frame.source_time_ns,
                    agent_id=frame.agent_id,
                    agent_type=frame.agent_type,
                    object_id_or_local_index=label.local_index,
                    match_policy=frame.match_policy,
                    match_age_ns=frame.match_age_ns,
                    lookahead_buffer_ns=frame.lookahead_buffer_ns,
                    pose_source_time_ns=frame.pose_source_time_ns,
                    pose_age_ns=frame.pose_age_ns,
                    object_key_kind="frame_local_index",
                    source_kind="oracle_gt",
                    oracle_gt=True,
                    class_id=label.class_id,
                )
                object_records.append(record)
                object_extras.append(
                    {
                        "center_x_m": label.center_m[0],
                        "center_y_m": label.center_m[1],
                        "center_z_m": label.center_m[2],
                        "size_x_m": label.size_m[0],
                        "size_y_m": label.size_m[1],
                        "size_z_m": label.size_m[2],
                        "yaw_rad": label.yaw_rad,
                        "box_frame": "top",
                    }
                )
                box = YawBox(label.center_m, label.size_m, label.yaw_rad)
                if transform.transform is not None:
                    transformed_center = transform.transform.inverse().apply(
                        np.asarray(label.center_m)
                    )
                    center_sensor = (
                        float(transformed_center[0]),
                        float(transformed_center[1]),
                        float(transformed_center[2]),
                    )
                else:
                    center_sensor = None
                visibility = visibility_eligibility(
                    center_sensor, capability.nominal_coverage
                )
                spatial = measure_spatial_support(
                    points_by_agent[agent],
                    box,
                    class_id=label.class_id,
                    visibility_eligible=visibility.eligible,
                    transform_quality=transform.quality,
                    count_reference=reference,
                )
                spatial_values.append(spatial)
                raw_values = {
                    "point_count": spatial.point_count,
                    "box_range_m": spatial.box_range_m,
                    "box_volume_m3": spatial.box_volume_m3,
                    "class_id": spatial.class_id,
                    "visibility_eligible": spatial.visibility_eligible,
                    "count_deficit_residual_development": spatial.deficit_residual,
                    "count_surplus_residual_development": spatial.surplus_residual,
                }
                for name, value in raw_values.items():
                    if name in {"box_range_m", "box_volume_m3", "class_id"}:
                        reason = ReasonCode.MATCHED
                    elif name == "visibility_eligible":
                        reason = visibility.reason_code
                    else:
                        reason = spatial.reason_code
                    evidence.append(
                        _evidence(
                            frame,
                            name,
                            value,
                            object_id=label.local_index,
                            reason=reason,
                        )
                    )
            frame_spatial[agent] = spatial_values

        for object_index, _label in enumerate(labels):
            counts = {
                agent: frame_spatial[agent][object_index].point_count
                for agent in config.selected_agents
            }
            eligible: dict[str, bool | None] = {
                agent: frame_spatial[agent][object_index].transform_quality
                == "verified"
                for agent in config.selected_agents
            }
            corroboration = leave_one_sender_out_counts(counts, eligible)
            for agent, result in corroboration.items():
                frame = frames_by_agent[agent]
                for name, value in (
                    ("eligible_peer_count", result.eligible_peer_count),
                    ("positive_peer_count", result.positive_peer_count),
                    ("peer_count_deficit_residual", result.deficit_residual),
                    ("peer_count_surplus_residual", result.surplus_residual),
                    ("peer_count_disagreement", result.disagreement),
                    ("peer_corroboration_score", result.peer_corroboration_score),
                ):
                    evidence.append(
                        _evidence(
                            frame,
                            name,
                            value,
                            object_id=object_index,
                            reason=result.reason_code,
                        )
                    )

        for agent in config.selected_agents:
            frame = frames_by_agent[agent]
            spatial_values = frame_spatial[agent]
            suspicious = tuple(
                max(value.deficit_residual or 0.0, value.surplus_residual or 0.0)
                if value.deficit_residual is not None
                and value.surplus_residual is not None
                else None
                for value in spatial_values
            )
            summary = summarize_object_suspicion(suspicious)
            for name, value in (
                ("object_count", summary.object_count),
                ("available_object_count", summary.available_object_count),
                ("maximum_object_suspicion", summary.maximum_suspicion),
                ("median_object_suspicion", summary.median_suspicion),
                ("any_object_evidence_unavailable", summary.any_unavailable),
            ):
                evidence.append(
                    _evidence(
                        frame,
                        name,
                        value,
                        reason=(
                            ReasonCode.UNRESOLVED_TRANSFORM
                            if value is None
                            else ReasonCode.MATCHED
                        ),
                    )
                )
            kin_result = kinematics.get((agent, frame_id))
            mean_count = (
                float(np.mean([value.point_count for value in spatial_values]))
                if spatial_values
                and all(value.point_count is not None for value in spatial_values)
                else None
            )
            shares: list[float] = []
            for object_index in range(len(labels)):
                object_counts = [
                    frame_spatial[peer][object_index].point_count
                    for peer in config.selected_agents
                ]
                if all(value is not None for value in object_counts):
                    own_position = config.selected_agents.index(agent)
                    own_value = object_counts[own_position]
                    assert own_value is not None
                    own = own_value
                    peers = tuple(
                        value
                        for position, value in enumerate(object_counts)
                        if position != own_position and value is not None
                    )
                    share = raw_point_share(own, peers)
                    if share is not None:
                        shares.append(share)
            mean_share = float(np.mean(shares)) if shares else None
            historical = historical_fixed_baseline(
                kinematic_error_m=(
                    kin_result.residual_m
                    if kin_result and kin_result.available
                    else None
                ),
                mean_point_count=mean_count,
                raw_share=mean_share,
            )
            peer_scores = [
                leave_one_sender_out_counts(
                    {
                        peer: frame_spatial[peer][object_index].point_count
                        for peer in config.selected_agents
                    },
                    {
                        peer: frame_spatial[peer][object_index].transform_quality
                        == "verified"
                        for peer in config.selected_agents
                    },
                )[agent].peer_corroboration_score
                for object_index in range(len(labels))
            ]
            available_peer_scores = [
                value for value in peer_scores if value is not None
            ]
            peer_score = min(available_peer_scores) if available_peer_scores else None
            unknown = unknown_aware_fixed_baseline(
                kinematic_error_m=(
                    kin_result.residual_m
                    if kin_result and kin_result.available
                    else None
                ),
                two_sided_count_residual=summary.maximum_suspicion,
                peer_corroboration_score=peer_score,
                kinematic_applicable=agent in index.odometry,
            )
            evidence.extend(
                (
                    _evidence(
                        frame,
                        "raw_point_share_diagnostic",
                        mean_share,
                        reason=ReasonCode.INCOMPLETE_EVIDENCE,
                    ),
                    _evidence(
                        frame, "historical_fixed_risk_diagnostic", historical.risk
                    ),
                    _evidence(
                        frame,
                        "unknown_aware_fixed_risk_uncalibrated",
                        unknown.risk,
                        reason=unknown.reason_code,
                    ),
                    _evidence(
                        frame, "unknown_aware_fixed_abstained", unknown.abstained
                    ),
                )
            )

    frame_table = frame_records_to_table(tuple(frame_records))
    frame_table = pd.concat(
        [frame_table.reset_index(drop=True), pd.DataFrame(frame_extras)], axis=1
    )
    object_table = object_records_to_table(tuple(object_records))
    if object_records:
        object_table = pd.concat(
            [object_table.reset_index(drop=True), pd.DataFrame(object_extras)], axis=1
        )
    evidence_table = evidence_records_to_table(tuple(evidence))
    evidence_table["value"] = pd.to_numeric(
        evidence_table["value"], errors="coerce"
    ).astype("Float64")
    evidence_table = evidence_table.sort_values(
        [
            "sync_frame_id",
            "agent_id",
            "evidence_level",
            "object_id_or_local_index",
            "evidence_name",
        ],
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)
    return frame_table, object_table, evidence_table


def verify_clean_cache(
    clean_dir: str | Path,
    *,
    expected_request_fingerprint: str | None = None,
) -> CacheBuildResult:
    directory = Path(clean_dir).expanduser().resolve()
    marker_path = directory / MARKER_NAME
    manifest_path = directory / MANIFEST_NAME
    if not marker_path.is_file():
        raise IncompleteCacheError(f"verified resume marker is absent: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StaleCacheError(f"cannot read cache metadata: {exc}") from exc
    manifest_hash = sha256_bytes(manifest_bytes)
    if marker.get("manifest_sha256") != manifest_hash:
        raise StaleCacheError("resume marker does not match source manifest")
    if (
        expected_request_fingerprint is not None
        and manifest.get("request_fingerprint") != expected_request_fingerprint
    ):
        raise StaleCacheError("cache request/configuration is stale or incompatible")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_NAMES):
        raise StaleCacheError("manifest output set is invalid")
    for name in OUTPUT_NAMES:
        path = directory / name
        if not path.is_file() or sha256_file(path) != outputs[name]["sha256"]:
            raise StaleCacheError(f"cache output hash mismatch: {name}")
        try:
            pq.read_table(path)
        except (OSError, pa.ArrowException) as exc:
            raise StaleCacheError(f"cache output is unreadable: {name}") from exc
    if marker.get("output_hashes") != {
        name: outputs[name]["sha256"] for name in OUTPUT_NAMES
    }:
        raise StaleCacheError("resume marker output hashes disagree with manifest")
    return CacheBuildResult(
        directory,
        {name: outputs[name]["sha256"] for name in OUTPUT_NAMES},
        manifest_hash,
        resumed=True,
    )


def build_clean_cache(config: CacheBuildConfig) -> CacheBuildResult:
    """Build once, verify outputs, then atomically publish the resume marker."""

    adapter, declaration = _adapter_config(config)
    index = index_sequence(adapter)
    source_files = _source_file_records(config, index)
    request = _request_manifest(config, index, declaration, source_files)
    request_fingerprint = _fingerprint(request)
    directory = config.clean_dir
    marker = directory / MARKER_NAME
    if marker.exists():
        return verify_clean_cache(
            directory, expected_request_fingerprint=request_fingerprint
        )
    if directory.exists() and any(directory.iterdir()):
        raise IncompleteCacheError(
            f"partial clean cache exists without verified marker: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    frame_table, object_table, evidence_table = _build_tables(
        config, index, declaration
    )
    tables = {
        "frame.parquet": frame_table,
        "object.parquet": object_table,
        "evidence.parquet": evidence_table,
    }
    outputs: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        temporary = directory / f".{name}.tmp"
        _write_parquet(tables[name], temporary)
        loaded = pq.read_table(temporary)
        if loaded.num_rows != len(tables[name]):
            raise CacheError(f"row-count verification failed for {name}")
        final = directory / name
        os.replace(temporary, final)
        outputs[name] = {
            "bytes": final.stat().st_size,
            "rows": len(tables[name]),
            "sha256": sha256_file(final),
        }
    manifest = {
        **request,
        "request_fingerprint": request_fingerprint,
        "outputs": outputs,
    }
    manifest_path = directory / MANIFEST_NAME
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_hash = sha256_file(manifest_path)
    marker_payload = {
        "schema_version": "1.0.0",
        "status": "verified_complete",
        "manifest_sha256": manifest_hash,
        "output_hashes": {name: outputs[name]["sha256"] for name in OUTPUT_NAMES},
    }
    marker.write_bytes(canonical_json_bytes(marker_payload))
    verified = verify_clean_cache(
        directory, expected_request_fingerprint=request_fingerprint
    )
    return CacheBuildResult(
        verified.clean_dir,
        verified.output_hashes,
        verified.manifest_sha256,
        resumed=False,
    )
