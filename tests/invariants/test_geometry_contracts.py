import math

import numpy as np
import pytest

from lidar_shield.contracts.reasons import MatchPolicy, ReasonCode
from lidar_shield.geometry.frames import RigidTransform
from lidar_shield.geometry.poses import Pose, PoseLookup, PoseSample, interpolate_pose


def pose(x: float, yaw_degrees: float = 0.0) -> Pose:
    half = math.radians(yaw_degrees) / 2
    return Pose((x, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half)))


SAMPLES = (
    PoseSample(100, pose(0.0, 0.0)),
    PoseSample(200, pose(10.0, 90.0)),
    PoseSample(300, pose(20.0, 180.0)),
)


def lookup(target: int, decision: int = 300, gap: int = 100) -> PoseLookup:
    return interpolate_pose(
        SAMPLES,
        target,
        decision_time_ns=decision,
        policy=MatchPolicy.CAUSAL_BACKWARD,
        max_bracketing_gap_ns=gap,
    )


def test_first_last_and_midpoint_pose_behavior() -> None:
    assert lookup(100).pose == SAMPLES[0].pose
    assert lookup(300).pose == SAMPLES[-1].pose
    midpoint = lookup(150)
    assert midpoint.pose is not None
    np.testing.assert_allclose(midpoint.pose.translation_m, [5.0, 0.0, 0.0])
    np.testing.assert_allclose(
        RigidTransform("a", "b", midpoint.pose).apply([1.0, 0.0, 0.0]),
        [5 + math.sqrt(0.5), math.sqrt(0.5), 0.0],
        atol=1e-9,
    )


def test_pose_interpolation_is_bounded_and_never_extrapolates() -> None:
    assert lookup(99).reason_code is ReasonCode.OUT_OF_RANGE
    assert lookup(301).reason_code is ReasonCode.OUT_OF_RANGE
    assert lookup(150, gap=99).reason_code is ReasonCode.INVALID_POSE


def test_future_bracket_must_respect_availability_contract() -> None:
    causal = lookup(150, decision=150)
    assert causal.reason_code is ReasonCode.FUTURE_POSE_UNAVAILABLE
    buffered = interpolate_pose(
        SAMPLES,
        150,
        decision_time_ns=150,
        policy=MatchPolicy.BUFFERED_NEAREST,
        max_bracketing_gap_ns=100,
        lookahead_buffer_ns=50,
    )
    assert buffered.reason_code is ReasonCode.MATCHED
    assert buffered.pose_source_time_ns == 200
    assert buffered.pose_age_ns == -50


def test_invalid_quaternions_and_nonpositive_intervals_are_rejected() -> None:
    with pytest.raises(ValueError, match="unit length"):
        Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        Pose((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    duplicate = (PoseSample(100, pose(0)), PoseSample(100, pose(1)))
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_pose(
            duplicate,
            100,
            decision_time_ns=100,
            policy=MatchPolicy.CAUSAL_BACKWARD,
            max_bracketing_gap_ns=100,
        )


def test_transform_round_trip_and_composition() -> None:
    transform = RigidTransform("vehicle", "map", pose(2.0, 90.0))
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        transform.inverse().apply(transform.apply(points)), points, atol=1e-12
    )

    following = RigidTransform("map", "top", Pose((0.0, 1.0, 0.0), (0, 0, 0, 1)))
    composed = transform.then(following)
    np.testing.assert_allclose(
        composed.apply(points), following.apply(transform.apply(points))
    )
    with pytest.raises(ValueError, match="cannot compose"):
        transform.then(transform)


def test_pose_and_transform_argument_guards() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        interpolate_pose(
            (),
            1,
            decision_time_ns=1,
            policy=MatchPolicy.CAUSAL_BACKWARD,
            max_bracketing_gap_ns=1,
        )
    with pytest.raises(TypeError, match="integer nanosecond"):
        PoseSample(True, pose(0))
    with pytest.raises(ValueError, match="must be positive"):
        interpolate_pose(
            SAMPLES,
            150,
            decision_time_ns=200,
            policy=MatchPolicy.CAUSAL_BACKWARD,
            max_bracketing_gap_ns=0,
        )
    with pytest.raises(ValueError, match="cannot declare"):
        interpolate_pose(
            SAMPLES,
            150,
            decision_time_ns=200,
            policy=MatchPolicy.CAUSAL_BACKWARD,
            max_bracketing_gap_ns=100,
            lookahead_buffer_ns=1,
        )
    with pytest.raises(ValueError, match="requires a matching policy"):
        interpolate_pose(
            SAMPLES,
            150,
            decision_time_ns=200,
            policy=MatchPolicy.NONE,
            max_bracketing_gap_ns=100,
        )

    future_exact = interpolate_pose(
        SAMPLES,
        200,
        decision_time_ns=150,
        policy=MatchPolicy.CAUSAL_BACKWARD,
        max_bracketing_gap_ns=100,
    )
    assert future_exact.reason_code is ReasonCode.FUTURE_POSE_UNAVAILABLE

    with pytest.raises(ValueError, match="frame names"):
        RigidTransform("", "map", pose(0))
    transform = RigidTransform("a", "b", pose(0))
    with pytest.raises(ValueError, match="shape"):
        transform.apply([1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        transform.apply([math.inf, 0.0, 0.0])
