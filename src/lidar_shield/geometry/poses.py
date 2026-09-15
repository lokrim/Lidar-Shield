"""Finite, unit-quaternion poses and availability-bounded interpolation."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from lidar_shield.contracts.reasons import MatchPolicy, ReasonCode


def _strict_ns(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer nanosecond value")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class Pose:
    """Translation in metres and quaternion in SciPy's ``x,y,z,w`` order."""

    translation_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        values = self.translation_m + self.quaternion_xyzw
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose values must be finite")
        norm = math.sqrt(sum(value * value for value in self.quaternion_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("pose quaternion must be unit length")


@dataclass(frozen=True)
class PoseSample:
    time_ns: int
    pose: Pose

    def __post_init__(self) -> None:
        _strict_ns(self.time_ns, "time_ns")


@dataclass(frozen=True)
class PoseLookup:
    pose: Pose | None
    reason_code: ReasonCode
    previous_time_ns: int | None
    next_time_ns: int | None
    pose_source_time_ns: int | None
    pose_age_ns: int | None


def validate_pose_series(samples: tuple[PoseSample, ...]) -> None:
    if not samples:
        raise ValueError("pose series must not be empty")
    times = [sample.time_ns for sample in samples]
    if any(later <= earlier for earlier, later in pairwise(times)):
        raise ValueError("pose sample times must be strictly increasing")


def interpolate_pose(
    samples: tuple[PoseSample, ...],
    target_time_ns: int,
    *,
    decision_time_ns: int,
    policy: MatchPolicy,
    max_bracketing_gap_ns: int,
    lookahead_buffer_ns: int = 0,
) -> PoseLookup:
    """Find or interpolate a pose without exceeding the availability contract.

    An interpolated pose records the newer bracket as ``pose_source_time_ns``;
    its signed age therefore exposes any buffered future bracket. No
    extrapolation occurs before the first or after the last sample.
    """

    validate_pose_series(samples)
    target_time_ns = _strict_ns(target_time_ns, "target_time_ns")
    decision_time_ns = _strict_ns(decision_time_ns, "decision_time_ns")
    max_bracketing_gap_ns = _strict_ns(max_bracketing_gap_ns, "max_bracketing_gap_ns")
    lookahead_buffer_ns = _strict_ns(lookahead_buffer_ns, "lookahead_buffer_ns")
    if max_bracketing_gap_ns == 0:
        raise ValueError("max_bracketing_gap_ns must be positive")
    if policy is MatchPolicy.CAUSAL_BACKWARD and lookahead_buffer_ns != 0:
        raise ValueError("causal pose lookup cannot declare look-ahead")
    if policy is MatchPolicy.BUFFERED_NEAREST and lookahead_buffer_ns == 0:
        raise ValueError("buffered pose lookup requires positive look-ahead")
    if policy is MatchPolicy.NONE:
        raise ValueError("pose lookup requires a matching policy")

    times = [sample.time_ns for sample in samples]
    position = bisect.bisect_left(times, target_time_ns)
    if position < len(samples) and samples[position].time_ns == target_time_ns:
        source_time = target_time_ns
        availability_limit = decision_time_ns + lookahead_buffer_ns
        if source_time > availability_limit:
            return PoseLookup(
                None,
                ReasonCode.FUTURE_POSE_UNAVAILABLE,
                source_time,
                source_time,
                None,
                None,
            )
        return PoseLookup(
            samples[position].pose,
            ReasonCode.MATCHED,
            source_time,
            source_time,
            source_time,
            decision_time_ns - source_time,
        )
    if position == 0 or position == len(samples):
        previous_time = samples[position - 1].time_ns if position else None
        following_time = samples[position].time_ns if position < len(samples) else None
        return PoseLookup(
            None, ReasonCode.OUT_OF_RANGE, previous_time, following_time, None, None
        )

    previous_sample = samples[position - 1]
    following_sample = samples[position]
    if following_sample.time_ns - previous_sample.time_ns > max_bracketing_gap_ns:
        return PoseLookup(
            None,
            ReasonCode.INVALID_POSE,
            previous_sample.time_ns,
            following_sample.time_ns,
            None,
            None,
        )
    availability_limit = decision_time_ns + lookahead_buffer_ns
    if following_sample.time_ns > availability_limit:
        return PoseLookup(
            None,
            ReasonCode.FUTURE_POSE_UNAVAILABLE,
            previous_sample.time_ns,
            following_sample.time_ns,
            None,
            None,
        )

    fraction_time = (target_time_ns - previous_sample.time_ns) / (
        following_sample.time_ns - previous_sample.time_ns
    )
    interpolated_translation = np.asarray(
        previous_sample.pose.translation_m
    ) + fraction_time * (
        np.asarray(following_sample.pose.translation_m)
        - np.asarray(previous_sample.pose.translation_m)
    )
    translation = (
        float(interpolated_translation[0]),
        float(interpolated_translation[1]),
        float(interpolated_translation[2]),
    )
    rotations = Rotation.from_quat(
        np.asarray(
            [
                previous_sample.pose.quaternion_xyzw,
                following_sample.pose.quaternion_xyzw,
            ],
            dtype=np.float64,
        )
    )
    interpolated_quaternion = Slerp(
        [previous_sample.time_ns, following_sample.time_ns], rotations
    )(target_time_ns).as_quat()
    quaternion = (
        float(interpolated_quaternion[0]),
        float(interpolated_quaternion[1]),
        float(interpolated_quaternion[2]),
        float(interpolated_quaternion[3]),
    )
    pose = Pose(translation, quaternion)
    return PoseLookup(
        pose,
        ReasonCode.MATCHED,
        previous_sample.time_ns,
        following_sample.time_ns,
        following_sample.time_ns,
        decision_time_ns - following_sample.time_ns,
    )
