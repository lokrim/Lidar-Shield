"""Deterministic box-local point addition/removal with schema preservation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lidar_shield.data.adapters.mixed_signals import ProductionPointCloud
from lidar_shield.geometry.boxes import YawBox
from lidar_shield.geometry.frames import RigidTransform


@dataclass(frozen=True)
class SpatialMutation:
    cloud: ProductionPointCloud
    requested_count: int
    realized_count: int
    before_inside_count: int
    after_inside_count: int
    selected_source_indices: tuple[int, ...]


def _box_local_to_frame(local: np.ndarray, box: YawBox) -> np.ndarray:
    cosine = math.cos(box.yaw_rad)
    sine = math.sin(box.yaw_rad)
    x = cosine * local[:, 0] - sine * local[:, 1] + box.center_m[0]
    y = sine * local[:, 0] + cosine * local[:, 1] + box.center_m[1]
    z = local[:, 2] + box.center_m[2]
    return np.column_stack((x, y, z))


def sample_box_local_points(box: YawBox, *, n_add: int, seed: int) -> np.ndarray:
    if n_add < 0 or seed < 0:
        raise ValueError("point count and seed must be nonnegative")
    generator = np.random.default_rng(seed)
    half = np.asarray(box.size_m, dtype=np.float64) / 2.0
    local = generator.uniform(-half, half, size=(n_add, 3))
    return _box_local_to_frame(local, box)


def _common_xyz(data: np.ndarray, raw_to_common: RigidTransform) -> np.ndarray:
    raw = np.column_stack((data["x"], data["y"], data["z"]))
    return raw_to_common.apply(raw)


def add_points_inside_box(
    cloud: ProductionPointCloud,
    box: YawBox,
    *,
    n_add: int,
    seed: int,
    raw_to_common: RigidTransform,
    boundary_tolerance_m: float = 1e-6,
) -> SpatialMutation:
    """Add seeded uniform volume points and median-fill every extra field."""

    common_before = _common_xyz(cloud.pc_data, raw_to_common)
    before_count = box.count_points(common_before, tolerance_m=boundary_tolerance_m)
    common_added = sample_box_local_points(box, n_add=n_add, seed=seed)
    raw_added = raw_to_common.inverse().apply(common_added)
    additions = np.zeros(n_add, dtype=cloud.pc_data.dtype)
    for coordinate, name in enumerate(("x", "y", "z")):
        additions[name] = raw_added[:, coordinate]
    for name in cloud.metadata.fields:
        if name in {"x", "y", "z"}:
            continue
        dtype_fields = cloud.pc_data.dtype.fields
        if dtype_fields is None:
            raise ValueError("production point cloud requires structured fields")
        target_dtype = dtype_fields[name][0]
        if len(cloud.pc_data):
            median = np.median(cloud.pc_data[name], axis=0)
            additions[name] = np.asarray(median, dtype=target_dtype)
        else:
            additions[name] = np.zeros((), dtype=target_dtype)
    combined = np.concatenate((cloud.pc_data, additions))
    derived = ProductionPointCloud(
        combined,
        cloud.metadata.__class__(
            fields=cloud.metadata.fields,
            sizes=cloud.metadata.sizes,
            types=cloud.metadata.types,
            counts=cloud.metadata.counts,
            points=len(combined),
            width=len(combined),
            height=1,
            encoding=cloud.metadata.encoding,
        ),
    )
    after_count = box.count_points(
        _common_xyz(derived.pc_data, raw_to_common),
        tolerance_m=boundary_tolerance_m,
    )
    return SpatialMutation(
        derived,
        n_add,
        after_count - before_count,
        before_count,
        after_count,
        (),
    )


def remove_points_inside_box(
    cloud: ProductionPointCloud,
    box: YawBox,
    *,
    removal_fraction: float,
    seed: int,
    raw_to_common: RigidTransform,
    boundary_tolerance_m: float = 1e-6,
) -> SpatialMutation:
    if not 0.0 <= removal_fraction <= 1.0 or seed < 0:
        raise ValueError("removal fraction must be in [0, 1] and seed nonnegative")
    common_before = _common_xyz(cloud.pc_data, raw_to_common)
    eligible = np.flatnonzero(
        box.contains(common_before, tolerance_m=boundary_tolerance_m)
    )
    requested = math.floor(removal_fraction * len(eligible))
    generator = np.random.default_rng(seed)
    selected = np.sort(
        generator.choice(eligible, size=requested, replace=False)
        if requested
        else np.asarray([], dtype=np.int64)
    )
    keep = np.ones(len(cloud.pc_data), dtype=bool)
    keep[selected] = False
    remaining = cloud.pc_data[keep].copy()
    derived = ProductionPointCloud(
        remaining,
        cloud.metadata.__class__(
            fields=cloud.metadata.fields,
            sizes=cloud.metadata.sizes,
            types=cloud.metadata.types,
            counts=cloud.metadata.counts,
            points=len(remaining),
            width=len(remaining),
            height=1,
            encoding=cloud.metadata.encoding,
        ),
    )
    after_count = box.count_points(
        _common_xyz(derived.pc_data, raw_to_common),
        tolerance_m=boundary_tolerance_m,
    )
    return SpatialMutation(
        derived,
        requested,
        len(selected),
        len(eligible),
        after_count,
        tuple(int(item) for item in selected),
    )
