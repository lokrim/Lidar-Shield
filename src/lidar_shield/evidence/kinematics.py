"""Body-velocity versus map-displacement self-consistency evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.data.adapters.mixed_signals import OdometrySample


@dataclass(frozen=True)
class KinematicResult:
    available: bool
    reason_code: ReasonCode
    dt_s: float | None
    speed_mps: float | None
    acceleration_mps2: float | None
    yaw_change_rad: float | None
    actual_displacement_m: tuple[float, float, float] | None
    integrated_displacement_m: tuple[float, float, float] | None
    residual_vector_m: tuple[float, float, float] | None
    residual_m: float | None
    integration_substeps: int
    self_consistency_only: bool = True


def _unknown(reason: ReasonCode, substeps: int) -> KinematicResult:
    return KinematicResult(
        False, reason, None, None, None, None, None, None, None, None, substeps
    )


def _finite_sample(sample: OdometrySample) -> bool:
    values = sample.position_m + sample.quaternion_xyzw + sample.body_velocity_mps
    return all(math.isfinite(value) for value in values)


def displacement_self_consistency(
    previous: OdometrySample,
    current: OdometrySample,
    *,
    max_gap_s: float,
    integration_substeps: int = 10,
    quaternion_norm_tolerance: float = 1e-3,
) -> KinematicResult:
    """Implement the specified endpoint-average/ten-midpoint-SLERP definition."""

    if integration_substeps <= 0:
        raise ValueError("integration_substeps must be positive")
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    if not _finite_sample(previous) or not _finite_sample(current):
        return _unknown(ReasonCode.NONFINITE, integration_substeps)
    delta_ns = current.time_ns - previous.time_ns
    if delta_ns <= 0:
        return _unknown(ReasonCode.NONPOSITIVE_DT, integration_substeps)
    dt_s = delta_ns / 1_000_000_000.0
    if dt_s > max_gap_s:
        return _unknown(ReasonCode.EXCESSIVE_GAP, integration_substeps)

    quaternion_array = np.asarray(
        [previous.quaternion_xyzw, current.quaternion_xyzw], dtype=np.float64
    )
    norms = np.linalg.norm(quaternion_array, axis=1)
    if (
        not np.isfinite(norms).all()
        or np.any(norms == 0)
        or np.any(np.abs(norms - 1.0) > quaternion_norm_tolerance)
    ):
        return _unknown(ReasonCode.INVALID_QUATERNION, integration_substeps)
    quaternion_array = quaternion_array / norms[:, None]
    rotations = Rotation.from_quat(quaternion_array)
    interpolator = Slerp([0.0, 1.0], rotations)
    midpoint_fractions = (
        np.arange(integration_substeps, dtype=np.float64) + 0.5
    ) / integration_substeps
    midpoint_rotations = interpolator(midpoint_fractions)
    endpoint_average_velocity = 0.5 * (
        np.asarray(previous.body_velocity_mps, dtype=np.float64)
        + np.asarray(current.body_velocity_mps, dtype=np.float64)
    )
    rotated_velocities = midpoint_rotations.apply(
        np.repeat(endpoint_average_velocity[None, :], integration_substeps, axis=0)
    )
    integrated = rotated_velocities.sum(axis=0) * (dt_s / integration_substeps)
    actual = np.asarray(current.position_m) - np.asarray(previous.position_m)
    residual = actual - integrated
    relative_rotation = rotations[1] * rotations[0].inv()
    yaw_change = float(relative_rotation.as_euler("zyx", degrees=False)[0])
    speed = float(
        0.5
        * (
            np.linalg.norm(previous.body_velocity_mps)
            + np.linalg.norm(current.body_velocity_mps)
        )
    )
    acceleration = float(
        np.linalg.norm(
            np.asarray(current.body_velocity_mps)
            - np.asarray(previous.body_velocity_mps)
        )
        / dt_s
    )
    return KinematicResult(
        available=True,
        reason_code=ReasonCode.MATCHED,
        dt_s=dt_s,
        speed_mps=speed,
        acceleration_mps2=acceleration,
        yaw_change_rad=yaw_change,
        actual_displacement_m=(float(actual[0]), float(actual[1]), float(actual[2])),
        integrated_displacement_m=(
            float(integrated[0]),
            float(integrated[1]),
            float(integrated[2]),
        ),
        residual_vector_m=(
            float(residual[0]),
            float(residual[1]),
            float(residual[2]),
        ),
        residual_m=float(np.linalg.norm(residual)),
        integration_substeps=integration_substeps,
    )
