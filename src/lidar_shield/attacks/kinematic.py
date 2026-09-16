"""Deterministic sparse body-velocity spike and drift overlays."""

from __future__ import annotations

from dataclasses import dataclass

from lidar_shield.data.adapters.mixed_signals import OdometrySample


@dataclass(frozen=True)
class VelocityChange:
    sample_time_ns: int
    original_body_velocity_mps: tuple[float, float, float]
    attacked_body_velocity_mps: tuple[float, float, float]
    applied_delta_v_mps: tuple[float, float, float]


@dataclass(frozen=True)
class KinematicOverlay:
    samples: tuple[OdometrySample, ...]
    changes: tuple[VelocityChange, ...]
    affected_interval_end_times_ns: tuple[int, ...]


def apply_velocity_overlay(
    samples: tuple[OdometrySample, ...],
    *,
    selected_sample_times_ns: tuple[int, ...],
    delta_v_mps: tuple[float, float, float],
    mode: str,
) -> KinematicOverlay:
    """Apply exactly K declared offsets and expand both endpoint intervals."""

    if mode not in {"velocity_spike", "velocity_drift"}:
        raise ValueError(f"unsupported kinematic overlay mode {mode!r}")
    if not selected_sample_times_ns:
        raise ValueError("a velocity overlay requires at least one sample")
    positions = {sample.time_ns: index for index, sample in enumerate(samples)}
    try:
        indexes = tuple(positions[item] for item in selected_sample_times_ns)
    except KeyError as exc:
        raise ValueError(
            f"target odometry sample is unavailable: {exc.args[0]}"
        ) from exc
    if any(
        right != left + 1 for left, right in zip(indexes, indexes[1:], strict=False)
    ):
        raise ValueError("selected odometry samples must be consecutive")
    count = len(indexes)
    changed = list(samples)
    changes: list[VelocityChange] = []
    for order, index in enumerate(indexes):
        factor = 1.0 if mode == "velocity_spike" else (order + 1) / count
        applied = (
            float(factor * delta_v_mps[0]),
            float(factor * delta_v_mps[1]),
            float(factor * delta_v_mps[2]),
        )
        original = samples[index]
        attacked_velocity = (
            float(original.body_velocity_mps[0] + applied[0]),
            float(original.body_velocity_mps[1] + applied[1]),
            float(original.body_velocity_mps[2] + applied[2]),
        )
        changed[index] = OdometrySample(
            time_ns=original.time_ns,
            header_time_ns=original.header_time_ns,
            position_m=original.position_m,
            quaternion_xyzw=original.quaternion_xyzw,
            body_velocity_mps=attacked_velocity,
        )
        changes.append(
            VelocityChange(
                original.time_ns,
                original.body_velocity_mps,
                attacked_velocity,
                applied,
            )
        )
    affected_indexes = {
        endpoint
        for index in indexes
        for endpoint in (index, index + 1)
        if 0 < endpoint < len(samples)
    }
    return KinematicOverlay(
        tuple(changed),
        tuple(changes),
        tuple(samples[index].time_ns for index in sorted(affected_indexes)),
    )
