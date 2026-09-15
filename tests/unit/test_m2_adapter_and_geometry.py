from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pypcd4 import Encoding, PointCloud  # type: ignore[attr-defined]

from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.contracts.schemas import Extrinsics, NominalCoverage
from lidar_shield.data.adapters.mixed_signals import (
    AdapterError,
    FractionalTimeConvention,
    MixedSignalsConfig,
    audit_filename_odometry_agreement,
    index_sequence,
    load_point_measurements,
    parse_cloud_filename,
    parse_decimal_time_ns,
    read_odometry,
)
from lidar_shield.data.adapters.oracle_objects import (
    OracleBoxLabel,
    OracleLabelError,
    parse_oracle_labels,
)
from lidar_shield.geometry.boxes import YawBox
from lidar_shield.geometry.frames import RigidTransform, resolve_extrinsics
from lidar_shield.geometry.visibility import visibility_eligibility


def test_filename_time_conventions_preserve_integer_nanoseconds() -> None:
    expected = 1_712_121_167_080_167_605
    assert (
        parse_decimal_time_ns(
            "1712121167",
            "80167605",
            convention=FractionalTimeConvention.OMITTED_LEADING_ZEROES,
        )
        == expected
    )
    assert (
        parse_decimal_time_ns(
            "1", "2", convention=FractionalTimeConvention.DECIMAL_DIGITS
        )
        == 1_200_000_000
    )
    cloud = parse_cloud_filename(
        "003_0_1712121167.80167605.pcd",
        convention=FractionalTimeConvention.OMITTED_LEADING_ZEROES,
    )
    assert (cloud.agent_id, cloud.sync_frame_id, cloud.source_time_ns) == (
        "003",
        0,
        expected,
    )
    with pytest.raises(AdapterError):
        parse_decimal_time_ns(
            "1", "1234567890", convention=FractionalTimeConvention.DECIMAL_DIGITS
        )
    with pytest.raises(AdapterError):
        parse_cloud_filename(
            "bad.pcd",
            convention=FractionalTimeConvention.OMITTED_LEADING_ZEROES,
        )


def test_point_loader_validates_metadata_and_normalizes_intensity(
    tmp_path: Path,
) -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0, -5.0], [1.0, 2.0, 3.0, 1750.0], [2, 3, 4, 7000]],
        dtype=np.float32,
    )
    for agent in ("top", "dome"):
        path = tmp_path / f"{agent}_0_1.000000001.pcd"
        PointCloud.from_xyzi_points(points).save(path, encoding=Encoding.ASCII)
        loaded = load_point_measurements(path)
        assert loaded.metadata.fields == ("x", "y", "z", "intensity")
        assert loaded.metadata.types == ("F", "F", "F", "F")
        assert loaded.metadata.sizes == (4, 4, 4, 4)
        assert loaded.metadata.encoding == "ascii"
        assert loaded.xyzi.dtype == np.float32
        assert loaded.normalized_intensity.tolist() == [0.0, 0.5, 1.0]
    with pytest.raises(AdapterError):
        load_point_measurements(path, intensity_divisor=0)


def test_oracle_label_parser_accepts_empty_and_rejects_invalid(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    assert parse_oracle_labels(empty) == ()
    valid = tmp_path / "valid.txt"
    valid.write_text("1 1 2 3 4 5 6 0.5\n", encoding="utf-8")
    assert parse_oracle_labels(valid) == (
        OracleBoxLabel(0, 1, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), 0.5),
    )
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("4 1 2 3 4 5 6 0\n", encoding="utf-8")
    with pytest.raises(OracleLabelError):
        parse_oracle_labels(invalid)
    invalid.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(OracleLabelError):
        parse_oracle_labels(invalid)


def test_yaw_box_and_matrix_round_trip() -> None:
    box = YawBox((1.0, 2.0, 0.0), (2.0, 4.0, 2.0), np.pi / 2)
    points = np.asarray([[1, 2, 0], [3, 2, 0], [1, 3.0000005, 0], [1, 4.1, 0]])
    assert box.contains(points).tolist() == [True, True, True, False]
    assert box.count_points(points) == 3
    assert box.volume_m3 == 16.0
    matrix = np.asarray(
        [[0, -1, 0, 3], [1, 0, 0, -2], [0, 0, 1, 1], [0, 0, 0, 1]],
        dtype=float,
    )
    transform = RigidTransform.from_matrix("dome", "top", matrix)
    transformed = transform.apply(points[:2])
    assert np.allclose(transform.inverse().apply(transformed), points[:2])
    assert np.allclose(transform.as_matrix(), matrix)
    with pytest.raises(ValueError):
        RigidTransform.from_matrix("a", "b", np.ones((4, 4)))


def test_transform_and_visibility_unknowns_are_explicit() -> None:
    unresolved = Extrinsics(
        from_frame="dome",
        to_frame="top",
        matrix_4x4=None,
        status="unresolved",
        note="awaiting calibration",
    )
    resolution = resolve_extrinsics(unresolved)
    assert resolution.transform is None
    assert resolution.reason_code is ReasonCode.UNRESOLVED_TRANSFORM
    coverage = NominalCoverage(
        horizontal_fov_deg=None,
        vertical_fov_deg=None,
        range_m=None,
        status="unresolved",
    )
    assert visibility_eligibility((1.0, 0.0, 0.0), coverage).reason_code is (
        ReasonCode.VISIBILITY_UNKNOWN
    )
    assert visibility_eligibility(None, coverage).reason_code is (
        ReasonCode.UNRESOLVED_TRANSFORM
    )
    verified = NominalCoverage(
        horizontal_fov_deg=90.0,
        vertical_fov_deg=30.0,
        range_m=10.0,
        status="verified",
    )
    assert visibility_eligibility((5.0, 0.0, 0.0), verified).eligible is True
    assert visibility_eligibility((0.0, 5.0, 0.0), verified).eligible is False


@pytest.mark.skipif(not Path("data/train/mini_7.tar").is_file(), reason="mini_7 absent")
def test_official_mini7_layout_timestamps_and_pcd_spot_checks() -> None:
    index = index_sequence(MixedSignalsConfig(Path("data"), "mini_7"))
    assert index.sync_frame_ids == tuple(range(299))
    assert set(index.clouds) == {"003", "004", "dome", "laser", "top"}
    assert set(index.labels) == set(range(290))
    odometry = {agent: read_odometry(path) for agent, path in index.odometry.items()}
    assert audit_filename_odometry_agreement(index, odometry) == {
        "003": True,
        "004": True,
        "laser": True,
    }
    assert index.cloud("003", 0).source_time_ns == odometry["003"][0].time_ns
    for agent in ("003", "top", "dome"):
        loaded = load_point_measurements(index.cloud(agent, 0).path)
        assert loaded.metadata.points > 0
        assert loaded.metadata.encoding == "ascii"
        assert loaded.metadata.fields == ("x", "y", "z", "intensity")
        assert loaded.metadata.sizes == (4, 4, 4, 4)
        assert loaded.metadata.types == ("F", "F", "F", "F")
        assert loaded.xyzi.shape[1] == 4
        assert np.nanmin(loaded.normalized_intensity) >= 0.0
        assert np.nanmax(loaded.normalized_intensity) <= 1.0
