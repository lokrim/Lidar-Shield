from __future__ import annotations

import math

import numpy as np
import pytest

from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.data.adapters.mixed_signals import OdometrySample
from lidar_shield.evidence.aggregation import summarize_object_suspicion
from lidar_shield.evidence.corroboration import leave_one_sender_out_counts
from lidar_shield.evidence.kinematics import displacement_self_consistency
from lidar_shield.evidence.spatial import (
    DevelopmentCountReference,
    measure_spatial_support,
)
from lidar_shield.geometry.boxes import YawBox
from lidar_shield.trust.fixed_baseline import (
    FixedWeights,
    historical_fixed_baseline,
    raw_point_share,
    unknown_aware_fixed_baseline,
)


def _sample(
    time_ns: int,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> OdometrySample:
    return OdometrySample(time_ns, time_ns, position, quaternion, velocity)


def test_kinematic_constant_motion_and_unknown_conditions() -> None:
    result = displacement_self_consistency(
        _sample(0, (0.0, 0.0, 0.0)),
        _sample(1_000_000_000, (1.0, 0.0, 0.0)),
        max_gap_s=2.0,
    )
    assert result.available
    assert result.dt_s == 1.0
    assert result.residual_m == pytest.approx(0.0, abs=1e-12)
    assert result.integration_substeps == 10
    invalid_quaternion = displacement_self_consistency(
        _sample(0, (0, 0, 0), quaternion=(0, 0, 0, 0)),
        _sample(1, (0, 0, 0)),
        max_gap_s=1,
    )
    assert invalid_quaternion.reason_code is ReasonCode.INVALID_QUATERNION
    nonpositive = displacement_self_consistency(
        _sample(1, (0, 0, 0)), _sample(1, (0, 0, 0)), max_gap_s=1
    )
    assert nonpositive.reason_code is ReasonCode.NONPOSITIVE_DT
    gap = displacement_self_consistency(
        _sample(0, (0, 0, 0)), _sample(2_000_000_000, (2, 0, 0)), max_gap_s=1
    )
    assert gap.reason_code is ReasonCode.EXCESSIVE_GAP


def test_kinematic_rotating_motion_matches_midpoint_slerp_definition() -> None:
    half_turn = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    previous = _sample(0, (0, 0, 0))
    current = _sample(1_000_000_000, (0, 0, 0), quaternion=half_turn)
    result = displacement_self_consistency(previous, current, max_gap_s=2)
    angles = (np.arange(10) + 0.5) * (math.pi / 2) / 10
    expected = (float(np.cos(angles).mean()), float(np.sin(angles).mean()), 0.0)
    assert result.integrated_displacement_m == pytest.approx(expected)


def test_two_sided_spatial_surprise_moves_correct_component() -> None:
    box = YawBox((0, 0, 0), (2, 2, 2), 0)
    reference = DevelopmentCountReference(math.log1p(2), 1.0)
    deficit = measure_spatial_support(
        np.empty((0, 3)),
        box,
        class_id=1,
        visibility_eligible=True,
        transform_quality="verified",
        count_reference=reference,
    )
    surplus = measure_spatial_support(
        np.zeros((8, 3)),
        box,
        class_id=1,
        visibility_eligible=True,
        transform_quality="verified",
        count_reference=reference,
    )
    assert deficit.deficit_residual and deficit.deficit_residual > 0
    assert deficit.surplus_residual == 0
    assert surplus.surplus_residual and surplus.surplus_residual > 0
    assert surplus.deficit_residual == 0
    unknown = measure_spatial_support(
        None,
        box,
        class_id=1,
        visibility_eligible=None,
        transform_quality="unresolved",
        count_reference=reference,
    )
    assert unknown.reason_code is ReasonCode.UNRESOLVED_TRANSFORM


def test_leave_one_out_invariants_and_insufficient_peers() -> None:
    eligibility = {agent: True for agent in ("a", "b", "c", "d")}
    baseline = leave_one_sender_out_counts(
        {"a": 10, "b": 10, "c": 10, "d": 10}, eligibility
    )
    added = leave_one_sender_out_counts(
        {"a": 100, "b": 10, "c": 10, "d": 10}, eligibility
    )
    assert added["a"].peer_corroboration_score == (
        baseline["a"].peer_corroboration_score
    )
    assert added["a"].surplus_residual is not None
    assert baseline["a"].surplus_residual is not None
    assert added["a"].surplus_residual > baseline["a"].surplus_residual
    three = leave_one_sender_out_counts(
        {"a": 10, "b": 10, "c": 10}, {"a": True, "b": True, "c": True}
    )
    assert three["a"].peer_corroboration_score == baseline["a"].peer_corroboration_score
    insufficient = leave_one_sender_out_counts(
        {"a": 1, "b": None}, {"a": True, "b": False}
    )
    assert insufficient["a"].reason_code is ReasonCode.INSUFFICIENT_PEERS


def test_fixed_baselines_keep_historical_share_separate_and_abstain() -> None:
    assert raw_point_share(10, (10, 10)) == pytest.approx(1 / 3)
    historical = historical_fixed_baseline(
        kinematic_error_m=None, mean_point_count=None, raw_share=None
    )
    assert historical.trust == pytest.approx(0.4)
    assert historical.diagnostic_only and historical.uncalibrated
    missing_kinematics = unknown_aware_fixed_baseline(
        kinematic_error_m=None,
        two_sided_count_residual=0.0,
        peer_corroboration_score=1.0,
        kinematic_applicable=True,
    )
    assert missing_kinematics.abstained
    assert missing_kinematics.trust is None
    infrastructure = unknown_aware_fixed_baseline(
        kinematic_error_m=None,
        two_sided_count_residual=0.0,
        peer_corroboration_score=1.0,
        kinematic_applicable=False,
    )
    assert infrastructure.trust == pytest.approx(1.0)
    with pytest.raises(ValueError):
        FixedWeights(0.5, 0.5, 0.5)


def test_object_summary_preserves_single_box_anomaly() -> None:
    summary = summarize_object_suspicion((0.0, 0.0, 9.0, None))
    assert summary.maximum_suspicion == 9.0
    assert summary.median_suspicion == 0.0
    assert summary.any_unavailable
