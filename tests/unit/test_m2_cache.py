from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pypcd4 import Encoding, PointCloud  # type: ignore[attr-defined]

from lidar_shield.data.cache import (
    CacheBuildConfig,
    IncompleteCacheError,
    StaleCacheError,
    build_clean_cache,
    verify_clean_cache,
)
from lidar_shield.demos.d1 import build_d1_payload

AGENTS = ("003", "004", "dome", "laser", "top")
VEHICLES = {"003", "004", "laser"}


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _build_source(root: Path) -> tuple[Path, Path, Path]:
    data = root / "data"
    clouds = data / "PointClouds" / "mini_7"
    odometry = data / "Odometry" / "mini_7"
    labels = data / "labels" / "mini_7"
    archive = data / "train" / "mini_7.tar"
    for directory in (clouds, odometry, labels, archive.parent):
        directory.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"independent official-input fixture")
    times: dict[tuple[str, int], int] = {}
    base = 1_700_000_000_000_000_000
    offsets = {
        "003": 0,
        "004": 10_000_000,
        "dome": 20_000_000,
        "laser": 30_000_000,
        "top": 40_000_000,
    }
    for frame in range(2):
        for agent_index, agent in enumerate(AGENTS):
            time_ns = base + frame * 100_000_000 + offsets[agent]
            times[(agent, frame)] = time_ns
            seconds, fraction = divmod(time_ns, 1_000_000_000)
            path = clouds / f"{agent}_{frame}_{seconds}.{fraction:09d}.pcd"
            inside = np.zeros((agent_index + 1, 4), dtype=np.float32)
            inside[:, 3] = 1750.0
            outside = np.asarray([[10.0, 10.0, 10.0, 7000.0]], dtype=np.float32)
            PointCloud.from_xyzi_points(np.vstack((inside, outside))).save(
                path, encoding=Encoding.ASCII
            )
        (labels / f"mini_7_{frame + 1}.txt").write_text(
            "1 0 0 0 2 2 2 0\n", encoding="utf-8"
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
        rows = []
        for frame in range(2):
            time_ns = times[(agent, frame)]
            rows.append([time_ns, time_ns, frame * 0.1, 0, 0, 0, 0, 0, 1, 1, 0, 0])
        pd.DataFrame(rows, columns=columns).to_csv(
            odometry / f"odometry_{agent}.csv", index=False
        )

    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    agent_rows = []
    for agent in AGENTS:
        agent_rows.append(
            {
                "agent_id": agent,
                "agent_type": "vehicle" if agent in VEHICLES else "infrastructure",
                "sensor_type": "lidar",
                "odometry": "required" if agent in VEHICLES else "not_applicable",
                "extrinsics": {
                    "from_frame": f"sensor_{agent}",
                    "to_frame": "top",
                    "matrix_4x4": identity,
                    "status": "verified",
                    "note": "independently authored identity fixture",
                },
                "nominal_coverage": {
                    "horizontal_fov_deg": 360.0,
                    "vertical_fov_deg": 180.0,
                    "range_m": 100.0,
                    "status": "verified",
                },
                "expected_point_fields": ["x", "y", "z", "intensity"],
                "expected_odometry_fields": columns if agent in VEHICLES else [],
            }
        )
    agent_path = root / "agents.yaml"
    _write_yaml(
        agent_path,
        {
            "schema_version": "1.0.0",
            "registry_id": "fixture-agents",
            "dataset": "mixed_signals",
            "agents": agent_rows,
        },
    )
    split_path = root / "split.yaml"
    _write_yaml(split_path, {"role": "regression_development"})
    dataset_path = root / "mini_7.yaml"
    _write_yaml(
        dataset_path,
        {
            "schema_version": "1.0.0",
            "dataset": "mixed_signals",
            "sequence_id": "mini_7",
            "source": {
                "configured_data_root": "data",
                "hugging_face_dataset": "fixture/Mixed-Signals-V2X",
                "hugging_face_revision": "fixture-revision",
                "xet_object_id": "fixture-object",
            },
            "layout": {
                "agents": list(AGENTS),
                "top_anchor_agent": "top",
                "expected_sync_start": 0,
                "expected_sync_stop": 1,
                "label_file_index_offset": 1,
                "expected_label_stop": 1,
            },
            "time": {
                "filename_fractional_convention": "omitted_leading_zeroes",
                "decision_anchor": "top_cloud_source_time",
                "match_policy": "buffered_nearest",
                "maximum_absolute_age_ns": 75_000_000,
                "lookahead_buffer_ns": 5_000_000,
            },
            "point_cloud": {"intensity_normalization": {"divisor": 3500.0}},
            "development_count_reference": {
                "reference_id": "fixture-fixed",
                "expected_log1p_count": float(np.log1p(3)),
                "scale": 1.0,
            },
        },
    )
    return agent_path, split_path, dataset_path


def _config(
    root: Path, artifact_root: Path, *, frames: tuple[int, ...] = (0, 1)
) -> CacheBuildConfig:
    agents, split, dataset = _build_source(root)
    return CacheBuildConfig(
        data_root=root / "data",
        artifact_root=artifact_root,
        experiment_id="m2-fixture",
        sequence_id="mini_7",
        selected_frames=frames,
        selected_agents=AGENTS,
        agent_registry_path=agents,
        split_registry_path=split,
        dataset_config_path=dataset,
        producer_command="lidar-shield cache build fixture",
    )


def test_cache_is_hash_stable_resumable_and_auditable(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    first = build_clean_cache(_config(source_root, tmp_path / "artifacts-a"))
    assert not first.resumed
    resumed = build_clean_cache(_config(source_root, tmp_path / "artifacts-a"))
    assert resumed.resumed
    assert resumed.output_hashes == first.output_hashes
    second = build_clean_cache(_config(source_root, tmp_path / "artifacts-b"))
    assert second.output_hashes == first.output_hashes
    manifest = json.loads((first.clean_dir / "source_manifest.json").read_text())
    assert manifest["archive"]["sha256"]
    assert manifest["selected_frames"] == [0, 1]
    assert manifest["selected_agents"] == list(AGENTS)
    assert manifest["official_source"]["hugging_face_revision"] == "fixture-revision"
    assert set(manifest["outputs"]) == {
        "frame.parquet",
        "object.parquet",
        "evidence.parquet",
    }
    frames = pd.read_parquet(first.clean_dir / "frame.parquet")
    objects = pd.read_parquet(first.clean_dir / "object.parquet")
    evidence = pd.read_parquet(first.clean_dir / "evidence.parquet")
    assert len(frames) == 10
    assert len(objects) == 10
    assert objects["object_id_or_local_index"].eq(0).all()
    assert evidence["evidence_name"].eq("maximum_object_suspicion").sum() == 10
    point_counts = evidence.loc[
        evidence["evidence_name"].eq("point_count") & evidence["sync_frame_id"].eq(0)
    ].set_index("agent_id")["value"]
    assert point_counts.to_dict() == {
        agent: float(position + 1) for position, agent in enumerate(AGENTS)
    }
    payload = build_d1_payload(first)
    assert len(payload["playback"]) == 10
    assert {row["agent_id"] for row in payload["playback"]} == set(AGENTS)


def test_cache_rejects_stale_incomplete_and_tampered_outputs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    config = _config(source_root, tmp_path / "artifacts")
    result = build_clean_cache(config)
    with pytest.raises(StaleCacheError):
        build_clean_cache(_config(source_root, tmp_path / "artifacts", frames=(0,)))
    (result.clean_dir / "frame.parquet").write_bytes(b"tampered")
    with pytest.raises(StaleCacheError):
        verify_clean_cache(result.clean_dir)
    incomplete = tmp_path / "partial"
    incomplete.mkdir()
    (incomplete / "frame.parquet").write_bytes(b"partial")
    with pytest.raises(IncompleteCacheError):
        verify_clean_cache(incomplete)
