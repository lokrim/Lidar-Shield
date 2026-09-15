#!/usr/bin/env python3
"""Regenerate the small M0 fixtures from their independently written contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VALID_CONFIG = """schema_version: "1.0.0"
project:
  distribution: lidar-shield
  package: lidar_shield
versions:
  software: "0.1.0"
  config_schema: "1.0.0"
  manifest_schema: "1.0.0"
  fixture_schema: "1.0.0"
  contract_schema: "1.0.0"
paths:
  fixture_root: ../../
  artifact_root: ../../../../artifacts
  report_root: ../../../../reports
reproducibility:
  seed: 1729
  hash_algorithm: sha256
  stable_serialization: json-sort-keys-utf8-v1
  stable_ordering: lexicographic-utf8-v1
  deterministic_algorithms: true
"""

INVALID_CONFIG = VALID_CONFIG.replace("seed: 1729", "seed: -1") + (
    "unexpected_m0_field: rejected\n"
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


FILES = {
    "config/invalid.yaml": INVALID_CONFIG.encode(),
    "config/valid.yaml": VALID_CONFIG.encode(),
    "labels/empty_000001.txt": b"",
    "labels/frame_000000.txt": (
        b"1 1.0 2.0 0.5 4.0 2.0 1.5 0.0\n"
        b"3 -1.0 0.5 0.9 0.8 0.6 1.7 1.5707963267948966\n"
    ),
    "odometry/odometry_003.csv": (
        b"%time,field.header.stamp,field.pose.pose.position.x,"
        b"field.pose.pose.position.y,field.pose.pose.position.z,"
        b"field.pose.pose.orientation.x,field.pose.pose.orientation.y,"
        b"field.pose.pose.orientation.z,field.pose.pose.orientation.w,"
        b"field.twist.twist.linear.x,field.twist.twist.linear.y,"
        b"field.twist.twist.linear.z\n"
        b"1700000000000000001,1700000000000000001,10.0,20.0,0.0,0.0,0.0,"
        b"0.0,1.0,2.0,0.0,0.0\n"
        b"1700000000100000001,1700000000100000001,10.2,20.0,0.0,0.0,0.0,"
        b"0.0,1.0,2.0,0.0,0.0\n"
    ),
    "points/003_000000_1700000000.000000001.pcd": (
        b"# .PCD v0.7 - Point Cloud Data file format\n"
        b"VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
        b"COUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        b"POINTS 3\nDATA ascii\n0.0 0.0 0.0 3500.0\n"
        b"1.0 -2.0 0.5 1750.0\n-1.0 2.0 3.0 0.0\n"
    ),
    "m1/contracts/identities.json": json_bytes(
        {
            "schema_version": "1.0.0",
            "identities": [
                {
                    "agent_id": "003",
                    "decision_time_ns": 1700000000000000003,
                    "experiment_id": "m1-fixtures",
                    "sequence_id": "synthetic-sequence-a",
                    "session_id": "synthetic-session-a",
                    "source_time_ns": 1700000000000000001,
                    "sync_frame_id": 0,
                    "variant_id": "clean",
                },
                {
                    "agent_id": "003-suffix",
                    "decision_time_ns": 1700000001000000003,
                    "experiment_id": "m1-fixtures",
                    "sequence_id": "synthetic-sequence-b",
                    "session_id": "synthetic-session-b",
                    "source_time_ns": None,
                    "sync_frame_id": 0,
                    "variant_id": "clean",
                },
            ],
        }
    ),
    "m1/membership/two_sequences.json": json_bytes(
        {
            "schema_version": "1.0.0",
            "frames": [
                {
                    "agents": {
                        "003": True,
                        "004": True,
                        "dome": True,
                        "laser": True,
                        "top": True,
                    },
                    "sequence_id": "synthetic-sequence-a",
                    "session_id": "synthetic-session-a",
                    "sync_frame_id": 0,
                },
                {
                    "agents": {
                        "003": True,
                        "004": False,
                        "dome": True,
                        "laser": False,
                        "top": True,
                    },
                    "sequence_id": "synthetic-sequence-b",
                    "session_id": "synthetic-session-b",
                    "sync_frame_id": 0,
                },
            ],
        }
    ),
    "m1/poses/boundaries.json": json_bytes(
        {
            "invalid": {
                "nonpositive_interval_ns": [200, 200],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 0.0],
            },
            "samples": [
                {
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "time_ns": 100,
                    "translation_m": [0.0, 0.0, 0.0],
                },
                {
                    "quaternion_xyzw": [
                        0.0,
                        0.0,
                        0.7071067811865475,
                        0.7071067811865476,
                    ],
                    "time_ns": 200,
                    "translation_m": [10.0, 0.0, 0.0],
                },
            ],
            "schema_version": "1.0.0",
        }
    ),
}

EXPECTED = {
    "config/invalid.yaml": {"validation": "fails: negative seed and extra field"},
    "config/valid.yaml": {"seed": 1729, "validation": "passes"},
    "labels/empty_000001.txt": {"row_count": 0},
    "labels/frame_000000.txt": {
        "classes": [1, 3],
        "field_count": 8,
        "row_count": 2,
    },
    "odometry/odometry_003.csv": {
        "agent_id": "003",
        "row_count": 2,
        "time_ns": ["1700000000000000001", "1700000000100000001"],
    },
    "points/003_000000_1700000000.000000001.pcd": {
        "fields": ["x", "y", "z", "intensity"],
        "intensity": [3500.0, 1750.0, 0.0],
        "point_count": 3,
    },
    "m1/contracts/identities.json": {
        "integer_nanoseconds": True,
        "sequences": 2,
        "string_ids": ["003", "003-suffix"],
    },
    "m1/membership/two_sequences.json": {
        "dynamic_membership": True,
        "sequences": 2,
    },
    "m1/poses/boundaries.json": {
        "includes_first_last": True,
        "includes_invalid_cases": True,
    },
}

MEDIA_TYPES = {
    ".csv": "text/csv",
    ".pcd": "application/vnd.pointcloud.pcd",
    ".txt": "text/plain",
    ".yaml": "application/yaml",
    ".json": "application/json",
}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_manifest() -> dict[str, object]:
    records = []
    for relative_path in sorted(FILES):
        content = FILES[relative_path]
        records.append(
            {
                "bytes": len(content),
                "expected": EXPECTED[relative_path],
                "media_type": MEDIA_TYPES[Path(relative_path).suffix],
                "path": relative_path,
                "sha256": digest(content),
            }
        )
    return {
        "artifact_kind": "fixture_set",
        "config_sha256": digest(FILES["config/valid.yaml"]),
        "created_at_utc": "2026-09-15T00:00:00Z",
        "experiment_id": "m1-synthetic-fixtures",
        "files": records,
        "fixture_schema_version": "1.0.0",
        "manifest_schema_version": "1.0.0",
        "origin": "synthetic",
        "origin_description": (
            "New tiny examples authored from the M0/M1 written contracts; no "
            "historical source, dataset row, or precomputed output was used."
        ),
        "producer": {
            "command": (
                "uv run python scripts/generate_synthetic_fixtures.py "
                "--output tests/fixtures/synthetic"
            ),
            "package_version": "0.1.0",
        },
        "schema_versions": {
            "config": "1.0.0",
            "contract": "1.0.0",
            "fixture": "1.0.0",
            "manifest": "1.0.0",
        },
        "seed": 1729,
        "software_versions": {
            "lidar-shield": "0.1.0",
            "python": ">=3.10,<3.14",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output: Path = args.output.resolve()

    for relative_path, content in FILES.items():
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    manifest = json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
