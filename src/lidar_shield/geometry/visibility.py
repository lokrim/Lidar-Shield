"""Conservative visibility eligibility with explicit unknown outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.contracts.schemas import NominalCoverage


@dataclass(frozen=True)
class VisibilityResult:
    eligible: bool | None
    reason_code: ReasonCode
    range_m: float | None
    azimuth_deg: float | None
    elevation_deg: float | None


def visibility_eligibility(
    center_in_sensor_m: tuple[float, float, float] | None,
    coverage: NominalCoverage,
) -> VisibilityResult:
    """Return visible/not-visible only when transform and coverage are verified."""

    if center_in_sensor_m is None:
        return VisibilityResult(None, ReasonCode.UNRESOLVED_TRANSFORM, None, None, None)
    if coverage.status != "verified":
        return VisibilityResult(None, ReasonCode.VISIBILITY_UNKNOWN, None, None, None)
    x, y, z = center_in_sensor_m
    if not all(math.isfinite(value) for value in center_in_sensor_m):
        return VisibilityResult(None, ReasonCode.NONFINITE, None, None, None)
    horizontal = math.hypot(x, y)
    distance = math.sqrt(horizontal * horizontal + z * z)
    azimuth = math.degrees(math.atan2(y, x))
    elevation = math.degrees(math.atan2(z, horizontal))
    assert coverage.range_m is not None
    assert coverage.horizontal_fov_deg is not None
    assert coverage.vertical_fov_deg is not None
    eligible = (
        distance <= coverage.range_m
        and abs(azimuth) <= coverage.horizontal_fov_deg / 2.0
        and abs(elevation) <= coverage.vertical_fov_deg / 2.0
    )
    return VisibilityResult(eligible, ReasonCode.MATCHED, distance, azimuth, elevation)
