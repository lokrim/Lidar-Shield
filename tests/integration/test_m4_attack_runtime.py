from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from pypcd4 import Encoding, PointCloud  # type: ignore[attr-defined]

from lidar_shield.attacks.config import (
    AttackConfigurationError,
    build_velocity_manifest,
    load_velocity_attack_config,
)
from lidar_shield.attacks.kinematic import apply_velocity_overlay
from lidar_shield.attacks.manifests import (
    AttackManifest,
    AttackManifestError,
    AttackTarget,
    AttackVersions,
    BoxTarget,
    KinematicParameters,
    SourceHashes,
    SpatialParameters,
    load_attack_manifest,
)
from lidar_shield.attacks.overlays import (
    EVIDENCE_DELTA_KEY,
    OverlayError,
    VariantArtifactWriter,
    apply_evidence_delta,
    evidence_row_hash,
    evidence_table_hash,
    load_realized_effect,
)
from lidar_shield.attacks.runtime import (
    record_interrupted_attempt,
    run_kinematic_attack,
    run_spatial_attack,
)
from lidar_shield.attacks.spatial import (
    add_points_inside_box,
    remove_points_inside_box,
    sample_box_local_points,
)
from lidar_shield.data.adapters.mixed_signals import (
    OdometrySample,
    load_production_point_cloud,
    read_odometry,
)
from lidar_shield.data.cache import CacheBuildConfig, build_clean_cache
from lidar_shield.demos.d3 import build_d3_payload
from lidar_shield.evaluation.splits import SplitRole
from lidar_shield.evidence.kinematics import displacement_self_consistency
from lidar_shield.geometry.frames import RigidTransform
from lidar_shield.manifest import production_code_sha256, sha256_file

AGENTS = ("003", "004", "dome", "laser", "top")
VEHICLES = {"003", "004", "laser"}


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[CacheBuildConfig, dict[tuple[str, int], int]]:
    root = tmp_path / "source"
    data = root / "data"
    clouds = data / "PointClouds" / "mini_7"
    odometry = data / "Odometry" / "mini_7"
    labels = data / "labels" / "mini_7"
    archive = data / "train" / "mini_7.tar"
    for directory in (clouds, odometry, labels, archive.parent):
        directory.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"newly authored M4 source fixture")
    base = 1_700_000_000_000_000_000
    offsets = {agent: index * 1_000_000 for index, agent in enumerate(AGENTS)}
    times: dict[tuple[str, int], int] = {}
    for frame in range(3):
        for agent_index, agent in enumerate(AGENTS):
            time_ns = base + frame * 100_000_000 + offsets[agent]
            times[(agent, frame)] = time_ns
            seconds, fraction = divmod(time_ns, 1_000_000_000)
            path = clouds / f"{agent}_{frame}_{seconds}.{fraction:09d}.pcd"
            count = agent_index + 3
            x = np.linspace(-0.75, 0.75, count, dtype=np.float32)
            y = np.zeros(count, dtype=np.float32)
            z = np.zeros(count, dtype=np.float32)
            intensity = np.full(count, 100 + agent_index, dtype=np.float32)
            ring = np.arange(count, dtype=np.uint16) + agent_index
            PointCloud.from_points(
                [x, y, z, intensity, ring],
                ("x", "y", "z", "intensity", "ring"),
                (np.float32, np.float32, np.float32, np.float32, np.uint16),
            ).save(path, encoding=Encoding.BINARY)
        (labels / f"mini_7_{frame + 1}.txt").write_text(
            "1 0 0 0 2 2 2 0\n1 0.5 0 0 2 2 2 0\n", encoding="utf-8"
        )
    columns = [
        "%time",
        "field.header.stamp",
        "field.pose.pose.position.x",
        "field.pose.pose.position.y",
        "field.pose.pose.position.z",
        "field.pose.pose.orientation.x",
        "field.pose.pose.orientation.y",
        "field.pose.pose.orientation.z",
        "field.pose.pose.orientation.w",
        "field.twist.twist.linear.x",
        "field.twist.twist.linear.y",
        "field.twist.twist.linear.z",
    ]
    for agent in VEHICLES:
        rows = [
            [
                times[(agent, frame)],
                times[(agent, frame)],
                frame * 0.1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
            ]
            for frame in range(3)
        ]
        pd.DataFrame(rows, columns=columns).to_csv(
            odometry / f"odometry_{agent}.csv", index=False
        )
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    agent_path = root / "agents.yaml"
    _write_yaml(
        agent_path,
        {
            "schema_version": "1.0.0",
            "registry_id": "m4-agents",
            "dataset": "mixed_signals",
            "agents": [
                {
                    "agent_id": agent,
                    "agent_type": (
                        "vehicle" if agent in VEHICLES else "infrastructure"
                    ),
                    "sensor_type": "lidar",
                    "odometry": "required" if agent in VEHICLES else "not_applicable",
                    "extrinsics": {
                        "from_frame": f"sensor_{agent}",
                        "to_frame": "top",
                        "matrix_4x4": identity,
                        "status": "verified",
                        "note": "identity transform for new M4 fixture",
                    },
                    "nominal_coverage": {
                        "horizontal_fov_deg": 360.0,
                        "vertical_fov_deg": 180.0,
                        "range_m": 100.0,
                        "status": "verified",
                    },
                    "expected_point_fields": [
                        "x",
                        "y",
                        "z",
                        "intensity",
                        "ring",
                    ],
                    "expected_odometry_fields": (columns if agent in VEHICLES else []),
                }
                for agent in AGENTS
            ],
        },
    )
    split_path = root / "split.yaml"
    _write_yaml(
        split_path,
        {
            "schema_version": "1.0.0",
            "registry_id": "m4-split",
            "status": "M4 regression development only",
            "assignments": [
                {
                    "source_id": "mixed-signals-train-mini_7",
                    "sequence_id": "mini_7",
                    "role": "regression_development",
                    "resolved": True,
                    "frozen": True,
                }
            ],
            "unresolved_slots": {"training": 4, "calibration": 1, "final_test": 2},
        },
    )
    dataset_path = root / "mini_7.yaml"
    _write_yaml(
        dataset_path,
        {
            "schema_version": "1.0.0",
            "dataset": "mixed_signals",
            "sequence_id": "mini_7",
            "source": {
                "hugging_face_dataset": "fixture/Mixed-Signals-V2X",
                "hugging_face_revision": "fixture",
                "xet_object_id": "fixture",
            },
            "layout": {
                "agents": list(AGENTS),
                "top_anchor_agent": "top",
                "expected_sync_start": 0,
                "expected_sync_stop": 2,
                "label_file_index_offset": 1,
                "expected_label_stop": 2,
            },
            "time": {
                "filename_fractional_convention": "omitted_leading_zeroes",
                "maximum_absolute_age_ns": 75_000_000,
                "lookahead_buffer_ns": 5_000_000,
            },
            "point_cloud": {"intensity_normalization": {"divisor": 3500.0}},
            "development_count_reference": {
                "reference_id": "m4-fixed",
                "expected_log1p_count": float(np.log1p(4)),
                "scale": 1.0,
            },
        },
    )
    config = CacheBuildConfig(
        data_root=data,
        artifact_root=tmp_path / "clean-artifacts",
        experiment_id="m4-fixture",
        sequence_id="mini_7",
        selected_frames=(0, 1, 2),
        selected_agents=AGENTS,
        agent_registry_path=agent_path,
        split_registry_path=split_path,
        dataset_config_path=dataset_path,
        producer_command="lidar-shield cache build m4-fixture",
    )
    return config, times


def _source_hashes(config: CacheBuildConfig, raw_relative: str) -> SourceHashes:
    clean = config.clean_dir
    source_manifest = json.loads((clean / "source_manifest.json").read_text())
    return SourceHashes(
        archive_sha256=source_manifest["archive"]["sha256"],
        selected_source_sha256=source_manifest["selected_source_sha256"],
        clean_manifest_sha256=sha256_file(clean / "source_manifest.json"),
        clean_frame_sha256=sha256_file(clean / "frame.parquet"),
        clean_object_sha256=sha256_file(clean / "object.parquet"),
        clean_evidence_sha256=sha256_file(clean / "evidence.parquet"),
        target_raw_sha256=sha256_file(config.data_root / raw_relative),
        split_registry_sha256=sha256_file(config.split_registry_path),
    )


def _kinematic_manifest(
    config: CacheBuildConfig,
    times: dict[tuple[str, int], int],
    *,
    variant: str = "kin-spike-severe-seed-17",
    selected_times: tuple[int, ...] | None = None,
) -> AttackManifest:
    raw = "Odometry/mini_7/odometry_003.csv"
    chosen = selected_times or (times[("003", 1)],)
    return AttackManifest(
        experiment_id="m4-fixture",
        episode_id="episode-kin-frame-1",
        variant_id=variant,
        origin="synthetic",
        split_role=SplitRole.REGRESSION_DEVELOPMENT,
        target=AttackTarget(
            source_id="mixed-signals-train-mini_7",
            sequence_id="mini_7",
            session_id="mini_7-session",
            sync_frame_id=1,
            decision_time_ns=times[("top", 1)],
            source_time_ns=times[("003", 1)],
            sender_id="003",
            agent_type="vehicle",
            interval_sample_times_ns=chosen,
            raw_relative_path=raw,
        ),
        source_hashes=_source_hashes(config, raw),
        family="kinematic",
        severity="severe",
        parameters=KinematicParameters(
            attack_kind="velocity_spike",
            delta_v_mps=(3.0, -1.0, 0.5),
            duration_samples=len(chosen),
        ),
        seed=17,
        attacker_knowledge="source_only",
        expected_modality="kinematic",
        versions=AttackVersions(
            contract_schema="1.0.0",
            package_version="0.1.0",
            code_sha256=production_code_sha256(),
            config_sha256="b" * 64,
        ),
    )


def _spatial_manifest(
    config: CacheBuildConfig,
    times: dict[tuple[str, int], int],
    *,
    sender: str,
    variant: str,
    attack_kind: str = "point_addition",
) -> AttackManifest:
    cloud_name = next(
        path.name
        for path in (config.data_root / "PointClouds" / "mini_7").glob(
            f"{sender}_1_*.pcd"
        )
    )
    raw = f"PointClouds/mini_7/{cloud_name}"
    parameters = (
        SpatialParameters(
            attack_kind="point_addition",
            box=BoxTarget(
                center_m=(0.0, 0.0, 0.0),
                size_m=(2.0, 2.0, 2.0),
                yaw_rad=0.0,
                frame="top",
            ),
            n_add=5,
        )
        if attack_kind == "point_addition"
        else SpatialParameters(
            attack_kind="point_removal",
            box=BoxTarget(
                center_m=(0.0, 0.0, 0.0),
                size_m=(2.0, 2.0, 2.0),
                yaw_rad=0.0,
                frame="top",
            ),
            removal_fraction=0.5,
        )
    )
    return AttackManifest(
        experiment_id="m4-fixture",
        episode_id=f"episode-spatial-{sender}-frame-1",
        variant_id=variant,
        origin="synthetic",
        split_role=SplitRole.REGRESSION_DEVELOPMENT,
        target=AttackTarget(
            source_id="mixed-signals-train-mini_7",
            sequence_id="mini_7",
            session_id="mini_7-session",
            sync_frame_id=1,
            decision_time_ns=times[("top", 1)],
            source_time_ns=times[(sender, 1)],
            sender_id=sender,
            agent_type=("vehicle" if sender in VEHICLES else "infrastructure"),
            object_id_or_local_index=0,
            raw_relative_path=raw,
        ),
        source_hashes=_source_hashes(config, raw),
        family="spatial_support",
        severity="medium",
        parameters=parameters,
        seed=23,
        attacker_knowledge="box_aware",
        expected_modality="spatial",
        versions=AttackVersions(
            contract_schema="1.0.0",
            package_version="0.1.0",
            code_sha256=production_code_sha256(),
            config_sha256="b" * 64,
        ),
    )


def test_kinematic_overlay_spike_drift_and_spillover() -> None:
    samples = tuple(
        OdometrySample(
            time_ns=index + 1,
            header_time_ns=index + 1,
            position_m=(float(index), 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            body_velocity_mps=(1.0, 2.0, 3.0),
        )
        for index in range(4)
    )
    spike = apply_velocity_overlay(
        samples,
        selected_sample_times_ns=(2, 3),
        delta_v_mps=(2.0, -1.0, 0.0),
        mode="velocity_spike",
    )
    assert [item.applied_delta_v_mps for item in spike.changes] == [
        (2.0, -1.0, 0.0),
        (2.0, -1.0, 0.0),
    ]
    assert spike.affected_interval_end_times_ns == (2, 3, 4)
    drift = apply_velocity_overlay(
        samples,
        selected_sample_times_ns=(2, 3),
        delta_v_mps=(2.0, 0.0, 0.0),
        mode="velocity_drift",
    )
    assert [item.applied_delta_v_mps[0] for item in drift.changes] == [1.0, 2.0]
    assert samples[1].body_velocity_mps == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="unsupported"):
        apply_velocity_overlay(
            samples,
            selected_sample_times_ns=(2,),
            delta_v_mps=(1.0, 0.0, 0.0),
            mode="bad",
        )
    with pytest.raises(ValueError, match="consecutive"):
        apply_velocity_overlay(
            samples,
            selected_sample_times_ns=(1, 3),
            delta_v_mps=(1.0, 0.0, 0.0),
            mode="velocity_spike",
        )


def test_kinematic_runtime_is_immutable_deterministic_and_collision_safe(
    tmp_path: Path,
) -> None:
    config, times = _fixture(tmp_path)
    cache = build_clean_cache(config)
    manifest = _kinematic_manifest(config, times)
    before = dict(cache.output_hashes)
    directory = run_kinematic_attack(
        manifest,
        data_root=config.data_root,
        clean_dir=config.clean_dir,
        artifact_root=tmp_path / "attack-artifacts-a",
        split_registry_path=config.split_registry_path,
    )
    assert {path.name for path in directory.iterdir()} == {
        "attack_manifest.json",
        "overlay.json",
        "evidence_delta.parquet",
        "realized_effect.json",
    }
    effect = load_realized_effect(directory / "realized_effect.json")
    assert effect["status"] == "succeeded"
    delta = pd.read_parquet(directory / "evidence_delta.parquet")
    assert set(delta["sync_frame_id"]) == {1, 2}
    assert before == {name: sha256_file(config.clean_dir / name) for name in before}
    with pytest.raises(OverlayError, match="already exists"):
        run_kinematic_attack(
            manifest,
            data_root=config.data_root,
            clean_dir=config.clean_dir,
            artifact_root=tmp_path / "attack-artifacts-a",
            split_registry_path=config.split_registry_path,
        )
    second = run_kinematic_attack(
        manifest,
        data_root=config.data_root,
        clean_dir=config.clean_dir,
        artifact_root=tmp_path / "attack-artifacts-b",
        split_registry_path=config.split_registry_path,
    )
    assert sha256_file(directory / "evidence_delta.parquet") == sha256_file(
        second / "evidence_delta.parquet"
    )
    clean_evidence = pd.read_parquet(config.clean_dir / "evidence.parquet")
    attacked = apply_evidence_delta(
        clean_evidence, delta, key_columns=EVIDENCE_DELTA_KEY
    )
    unchanged = clean_evidence.loc[
        ~clean_evidence.set_index(list(EVIDENCE_DELTA_KEY)).index.isin(
            delta.set_index(list(EVIDENCE_DELTA_KEY)).index
        )
    ]
    attacked_unchanged = attacked.loc[unchanged.index]
    assert [evidence_row_hash(row) for _, row in unchanged.iterrows()] == [
        evidence_row_hash(row) for _, row in attacked_unchanged.iterrows()
    ]
    assert effect["realized_evidence_effect"]["attacked_evidence_sha256"] == (
        evidence_table_hash(attacked)
    )
    samples = read_odometry(config.data_root / manifest.target.raw_relative_path)
    assert isinstance(manifest.parameters, KinematicParameters)
    full_overlay = apply_velocity_overlay(
        samples,
        selected_sample_times_ns=manifest.target.interval_sample_times_ns,
        delta_v_mps=manifest.parameters.delta_v_mps,
        mode=manifest.parameters.attack_kind,
    )
    by_time = {sample.time_ns: index for index, sample in enumerate(samples)}
    frame_table = pd.read_parquet(config.clean_dir / "frame.parquet")
    for endpoint_time in full_overlay.affected_interval_end_times_ns:
        index = by_time[endpoint_time]
        full = displacement_self_consistency(
            full_overlay.samples[index - 1],
            full_overlay.samples[index],
            max_gap_s=config.max_gap_s,
        )
        frame = int(
            frame_table.loc[
                frame_table["agent_id"].eq("003")
                & frame_table["source_time_ns"].eq(endpoint_time),
                "sync_frame_id",
            ].iloc[0]
        )
        incremental = delta.loc[
            delta["sync_frame_id"].eq(frame)
            & delta["agent_id"].eq("003")
            & delta["evidence_name"].eq("kinematic_residual_m"),
            "value",
        ].iloc[0]
        assert incremental == pytest.approx(full.residual_m)


def test_spatial_runtime_reload_overlap_dome_and_failure_retention(
    tmp_path: Path,
) -> None:
    config, times = _fixture(tmp_path)
    build_clean_cache(config)
    identity = RigidTransform.from_matrix("sensor", "top", np.eye(4))
    top_manifest = _spatial_manifest(
        config, times, sender="top", variant="top-add-medium-seed-23"
    )
    directory = run_spatial_attack(
        top_manifest,
        data_root=config.data_root,
        clean_dir=config.clean_dir,
        artifact_root=tmp_path / "spatial",
        split_registry_path=config.split_registry_path,
        raw_to_common=identity,
    )
    derived = load_production_point_cloud(directory / "overlay.pcd")
    assert derived.metadata.fields == ("x", "y", "z", "intensity", "ring")
    assert derived.metadata.encoding == "binary"
    assert derived.pc_data.dtype["ring"] == np.dtype("uint16")
    effect = load_realized_effect(directory / "realized_effect.json")
    assert set(effect["realized_evidence_effect"]["changed_object_counts"]) == {
        "0",
        "1",
    }
    assert set(effect["realized_evidence_effect"]["honest_companion_agents"]) == (
        set(AGENTS) - {"top"}
    )

    dome_manifest = _spatial_manifest(
        config,
        times,
        sender="dome",
        variant="dome-remove-medium-seed-23",
        attack_kind="point_removal",
    )
    dome_dir = run_spatial_attack(
        dome_manifest,
        data_root=config.data_root,
        clean_dir=config.clean_dir,
        artifact_root=tmp_path / "spatial",
        split_registry_path=config.split_registry_path,
        raw_to_common=identity,
    )
    assert load_realized_effect(dome_dir / "realized_effect.json")[
        "injection_succeeded"
    ]

    vehicle_manifest = _spatial_manifest(
        config, times, sender="003", variant="vehicle-add-success"
    )
    vehicle_dir = run_spatial_attack(
        vehicle_manifest,
        data_root=config.data_root,
        clean_dir=config.clean_dir,
        artifact_root=tmp_path / "spatial",
        split_registry_path=config.split_registry_path,
        raw_to_common=identity,
    )
    assert load_production_point_cloud(vehicle_dir / "overlay.pcd").metadata.fields == (
        "x",
        "y",
        "z",
        "intensity",
        "ring",
    )

    failed_manifest = _spatial_manifest(
        config, times, sender="003", variant="vehicle-unavailable-transform"
    )
    with pytest.raises(OverlayError, match="unresolved_transform"):
        run_spatial_attack(
            failed_manifest,
            data_root=config.data_root,
            clean_dir=config.clean_dir,
            artifact_root=tmp_path / "spatial",
            split_registry_path=config.split_registry_path,
            raw_to_common=None,
        )
    failed = (
        tmp_path
        / "spatial"
        / "m4-fixture"
        / "sequences"
        / "mini_7"
        / "variants"
        / failed_manifest.variant_id
    )
    assert load_realized_effect(failed / "realized_effect.json")["status"] == "failed"
    assert (failed / "evidence_delta.parquet").is_file()
    assert sha256_file(config.data_root / failed_manifest.target.raw_relative_path) == (
        failed_manifest.source_hashes.target_raw_sha256
    )


def test_spatial_primitives_manifest_validation_and_interrupted_attempt(
    tmp_path: Path,
) -> None:
    config, times = _fixture(tmp_path)
    build_clean_cache(config)
    manifest = _spatial_manifest(
        config, times, sender="top", variant="spatial-primitives"
    )
    raw = load_production_point_cloud(
        config.data_root / manifest.target.raw_relative_path
    )
    transform = RigidTransform.from_matrix("raw", "top", np.eye(4))
    box = BoxTarget(
        center_m=(0.0, 0.0, 0.0),
        size_m=(2.0, 2.0, 2.0),
        yaw_rad=0.0,
        frame="top",
    )
    from lidar_shield.geometry.boxes import YawBox

    yaw_box = YawBox(box.center_m, box.size_m, box.yaw_rad)
    points_a = sample_box_local_points(yaw_box, n_add=4, seed=99)
    points_b = sample_box_local_points(yaw_box, n_add=4, seed=99)
    assert np.array_equal(points_a, points_b)
    assert yaw_box.contains(points_a).all()
    added = add_points_inside_box(
        raw, yaw_box, n_add=4, seed=99, raw_to_common=transform
    )
    assert added.realized_count == 4
    removed = remove_points_inside_box(
        raw,
        yaw_box,
        removal_fraction=0.5,
        seed=99,
        raw_to_common=transform,
    )
    assert removed.realized_count == int(0.5 * removed.before_inside_count)
    with pytest.raises(ValueError):
        sample_box_local_points(yaw_box, n_add=-1, seed=0)
    with pytest.raises(ValueError):
        remove_points_inside_box(
            raw,
            yaw_box,
            removal_fraction=2.0,
            seed=0,
            raw_to_common=transform,
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest.canonical_bytes())
    assert load_attack_manifest(manifest_path) == manifest
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(AttackManifestError):
        load_attack_manifest(manifest_path)
    with pytest.raises(ValidationError):
        SpatialParameters(
            attack_kind="point_addition",
            box=box,
            n_add=1,
            removal_fraction=0.2,
        )
    with pytest.raises(ValidationError):
        _kinematic_manifest(config, times).model_copy(
            update={"expected_modality": "spatial"}
        ).model_validate(
            {
                **_kinematic_manifest(config, times).model_dump(mode="json"),
                "expected_modality": "spatial",
            }
        )

    interrupted = _kinematic_manifest(config, times, variant="kin-interrupted-seed-17")
    writer = VariantArtifactWriter(tmp_path / "interrupted", interrupted)
    record_interrupted_attempt(writer)
    effect = load_realized_effect(writer.directory / "realized_effect.json")
    assert effect["status"] == "interrupted"


def test_unknown_sample_failure_is_preserved_but_stale_hash_is_rejected(
    tmp_path: Path,
) -> None:
    config, times = _fixture(tmp_path)
    build_clean_cache(config)
    unavailable = _kinematic_manifest(
        config,
        times,
        variant="kin-unavailable-sample",
        selected_times=(times[("003", 2)] + 1,),
    )
    with pytest.raises(ValueError, match="unavailable"):
        run_kinematic_attack(
            unavailable,
            data_root=config.data_root,
            clean_dir=config.clean_dir,
            artifact_root=tmp_path / "attempts",
            split_registry_path=config.split_registry_path,
        )
    failed_dir = (
        tmp_path
        / "attempts"
        / "m4-fixture"
        / "sequences"
        / "mini_7"
        / "variants"
        / unavailable.variant_id
    )
    assert load_realized_effect(failed_dir / "realized_effect.json")["status"] == (
        "failed"
    )

    stale = _kinematic_manifest(config, times, variant="stale").model_copy(
        update={
            "source_hashes": _kinematic_manifest(
                config, times
            ).source_hashes.model_copy(update={"target_raw_sha256": "0" * 64})
        }
    )
    with pytest.raises(AttackManifestError, match="source hash mismatch"):
        run_kinematic_attack(
            stale,
            data_root=config.data_root,
            clean_dir=config.clean_dir,
            artifact_root=tmp_path / "attempts",
            split_registry_path=config.split_registry_path,
        )


def test_authored_config_builds_manifest_and_d3_final_pipeline(tmp_path: Path) -> None:
    config, times = _fixture(tmp_path)
    build_clean_cache(config)
    authored = tmp_path / "velocity.yaml"
    _write_yaml(
        authored,
        {
            "schema_version": "1.0.0",
            "experiment_id": "m4-fixture",
            "episode_id": "episode-configured",
            "variant_id": "configured-drift-seed-31",
            "source_id": "mixed-signals-train-mini_7",
            "sequence_id": "mini_7",
            "session_id": "mini_7-session",
            "sync_frame_id": 1,
            "sender_id": "003",
            "severity": "medium",
            "attack_kind": "velocity_drift",
            "delta_v_mps": [2.0, 0.0, 0.0],
            "duration_samples": 1,
            "seed": 31,
            "attacker_knowledge": "source_only",
        },
    )
    declaration = load_velocity_attack_config(authored)
    manifest = build_velocity_manifest(
        declaration,
        config_path=authored,
        data_root=config.data_root,
        clean_dir=config.clean_dir,
        split_registry_path=config.split_registry_path,
        agent_registry_path=config.agent_registry_path,
    )
    assert manifest.target.source_time_ns == times[("003", 1)]
    assert manifest.target.interval_sample_times_ns == (times[("003", 1)],)
    directory = run_kinematic_attack(
        manifest,
        data_root=config.data_root,
        clean_dir=config.clean_dir,
        artifact_root=tmp_path / "configured-artifacts",
        split_registry_path=config.split_registry_path,
    )
    payload = build_d3_payload(
        manifest, clean_dir=config.clean_dir, variant_dir=directory
    )
    assert payload["demo"] == "D3"
    assert payload["manifest_sha256"] == manifest.sha256
    assert payload["coverage"]["attempt_count"] == 1
    assert payload["clean"]["predictions"]
    assert payload["attacked"]["fusion"]
    assert {item["variant_id"] for item in payload["attacked"]["predictions"]} == {
        manifest.variant_id
    }

    authored.write_text("[]", encoding="utf-8")
    with pytest.raises(AttackConfigurationError):
        load_velocity_attack_config(authored)
