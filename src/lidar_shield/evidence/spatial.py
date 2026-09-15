"""Raw point-in-box measurements and development-only count surprise."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.geometry.boxes import YawBox


@dataclass(frozen=True)
class DevelopmentCountReference:
    """Fixed, non-fitted smoke reference; never a scientific M5 normalizer."""

    expected_log1p_count: float
    scale: float
    reference_id: str = "development-fixed-v1"
    development_only: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.expected_log1p_count):
            raise ValueError("expected log count must be finite")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("count-reference scale must be finite and positive")
        if not self.reference_id:
            raise ValueError("reference ID must be non-empty")


@dataclass(frozen=True)
class SpatialResult:
    available: bool
    reason_code: ReasonCode
    point_count: int | None
    box_range_m: float
    box_volume_m3: float
    class_id: int
    visibility_eligible: bool | None
    deficit_residual: float | None
    surplus_residual: float | None
    transform_quality: str
    development_reference_id: str | None


def measure_spatial_support(
    points_in_box_frame: ArrayLike | None,
    box: YawBox,
    *,
    class_id: int,
    visibility_eligible: bool | None,
    transform_quality: str,
    count_reference: DevelopmentCountReference | None = None,
    boundary_tolerance_m: float = 1e-6,
) -> SpatialResult:
    """Measure support; unresolved transforms remain an explicit unknown."""

    if class_id not in {1, 2, 3}:
        raise ValueError("class_id must be 1, 2, or 3")
    if points_in_box_frame is None or transform_quality != "verified":
        return SpatialResult(
            False,
            ReasonCode.UNRESOLVED_TRANSFORM,
            None,
            box.center_range_m,
            box.volume_m3,
            class_id,
            visibility_eligible,
            None,
            None,
            transform_quality,
            None,
        )
    points = np.asarray(points_in_box_frame, dtype=np.float64)
    if not np.isfinite(points).all():
        return SpatialResult(
            False,
            ReasonCode.NONFINITE,
            None,
            box.center_range_m,
            box.volume_m3,
            class_id,
            visibility_eligible,
            None,
            None,
            transform_quality,
            None,
        )
    count = box.count_points(points, tolerance_m=boundary_tolerance_m)
    deficit: float | None = None
    surplus: float | None = None
    reference_id: str | None = None
    if count_reference is not None:
        signed = (
            math.log1p(count) - count_reference.expected_log1p_count
        ) / count_reference.scale
        deficit = max(-signed, 0.0)
        surplus = max(signed, 0.0)
        reference_id = count_reference.reference_id
    return SpatialResult(
        True,
        ReasonCode.MATCHED,
        count,
        box.center_range_m,
        box.volume_m3,
        class_id,
        visibility_eligible,
        deficit,
        surplus,
        transform_quality,
        reference_id,
    )
