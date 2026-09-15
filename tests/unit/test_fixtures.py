import csv
import json
from pathlib import Path

import numpy as np
from pypcd4 import PointCloud  # type: ignore[attr-defined]

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic"


def test_label_examples_match_declared_eight_field_contract() -> None:
    rows = [
        line.split()
        for line in (FIXTURE_ROOT / "labels" / "frame_000000.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert all(len(row) == 8 for row in rows)
    assert [int(row[0]) for row in rows] == [1, 3]
    assert (FIXTURE_ROOT / "labels" / "empty_000001.txt").read_bytes() == b""


def test_odometry_keeps_nanoseconds_as_exact_integer_text() -> None:
    with (FIXTURE_ROOT / "odometry" / "odometry_003.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert [row["%time"] for row in rows] == [
        "1700000000000000001",
        "1700000000100000001",
    ]
    assert all(row["%time"] == row["field.header.stamp"] for row in rows)
    assert "003" in "odometry_003.csv"


def test_point_cloud_loads_through_verified_pypcd4_api() -> None:
    path = FIXTURE_ROOT / "points" / "003_000000_1700000000.000000001.pcd"
    points = PointCloud.from_path(path).numpy(("x", "y", "z", "intensity"))

    assert points.shape == (3, 4)
    np.testing.assert_allclose(points[:, 3], [3500.0, 1750.0, 0.0])


def test_m1_fixture_identities_and_membership_cover_two_isolated_sequences() -> None:
    identities = json.loads(
        (FIXTURE_ROOT / "m1" / "contracts" / "identities.json").read_text()
    )["identities"]
    assert [item["sequence_id"] for item in identities] == [
        "synthetic-sequence-a",
        "synthetic-sequence-b",
    ]
    assert identities[0]["agent_id"] == "003"
    assert isinstance(identities[0]["decision_time_ns"], int)
    assert identities[1]["source_time_ns"] is None

    frames = json.loads(
        (FIXTURE_ROOT / "m1" / "membership" / "two_sequences.json").read_text()
    )["frames"]
    assert frames[0]["agents"] != frames[1]["agents"]
