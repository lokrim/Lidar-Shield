"""Independently implemented fixed mathematical diagnostic baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lidar_shield.contracts.reasons import ReasonCode


@dataclass(frozen=True)
class FixedWeights:
    kinematic: float = 0.40
    support: float = 0.30
    corroboration: float = 0.30

    def __post_init__(self) -> None:
        values = (self.kinematic, self.support, self.corroboration)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("fixed weights must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
            raise ValueError("fixed weights must sum to one")


DEFAULT_FIXED_WEIGHTS = FixedWeights()


@dataclass(frozen=True)
class HistoricalFixedResult:
    trust: float
    risk: float
    kinematic_score: float
    support_score: float
    raw_share_score: float
    diagnostic_only: bool = True
    uncalibrated: bool = True


@dataclass(frozen=True)
class UnknownAwareFixedResult:
    trust: float | None
    risk: float | None
    available_weight: float
    abstained: bool
    reason_code: ReasonCode
    component_scores: dict[str, float | None]
    policy: str = "renormalize-if-weight>=0.60; require-kinematics-if-applicable"
    uncalibrated: bool = True


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("fixed-baseline input must be finite")
    return min(max(value, 0.0), 1.0)


def raw_point_share(own_count: int, peer_counts: tuple[int, ...]) -> float | None:
    """Historical diagnostic only; it is not peer corroboration."""

    if own_count < 0 or any(value < 0 for value in peer_counts):
        raise ValueError("point counts must be nonnegative")
    total = own_count + sum(peer_counts)
    return own_count / total if total else None


def historical_fixed_baseline(
    *,
    kinematic_error_m: float | None,
    mean_point_count: float | None,
    raw_share: float | None,
    weights: FixedWeights = DEFAULT_FIXED_WEIGHTS,
    kinematic_cap_m: float = 0.057,
    support_cap_points: float = 350.0,
) -> HistoricalFixedResult:
    """Reproduce the documented 40/30/30 score as a labeled ablation.

    Its historical missing choices are deliberately isolated here:
    missing kinematics maps to one; other missing components map to zero.
    """

    if kinematic_cap_m <= 0 or support_cap_points <= 0:
        raise ValueError("fixed-baseline caps must be positive")
    kinematic = (
        1.0
        if kinematic_error_m is None
        else 1.0 - _clip01(kinematic_error_m / kinematic_cap_m)
    )
    support = (
        0.0
        if mean_point_count is None
        else _clip01(mean_point_count / support_cap_points)
    )
    share = 0.0 if raw_share is None else _clip01(raw_share)
    trust = (
        weights.kinematic * kinematic
        + weights.support * support
        + weights.corroboration * share
    )
    return HistoricalFixedResult(
        trust=trust,
        risk=1.0 - trust,
        kinematic_score=kinematic,
        support_score=support,
        raw_share_score=share,
    )


def unknown_aware_fixed_baseline(
    *,
    kinematic_error_m: float | None,
    two_sided_count_residual: float | None,
    peer_corroboration_score: float | None,
    kinematic_applicable: bool,
    weights: FixedWeights = DEFAULT_FIXED_WEIGHTS,
    kinematic_cap_m: float = 0.057,
    count_residual_cap: float = 3.0,
    minimum_available_weight: float = 0.60,
) -> UnknownAwareFixedResult:
    """Combine available new evidence without manufacturing a trust floor.

    Applicable missing kinematics always abstains.  Otherwise components are
    renormalized only when their original total weight is at least 0.60.
    Infrastructure may therefore combine spatial and peer evidence when
    kinematics is explicitly not applicable, rather than pretending it is
    perfect.
    """

    if kinematic_cap_m <= 0 or count_residual_cap <= 0:
        raise ValueError("unknown-aware caps must be positive")
    if not 0 < minimum_available_weight <= 1:
        raise ValueError("minimum available weight must lie in (0, 1]")
    if kinematic_applicable and kinematic_error_m is None:
        return UnknownAwareFixedResult(
            None,
            None,
            0.0,
            True,
            ReasonCode.INCOMPLETE_EVIDENCE,
            {"kinematic": None, "support": None, "corroboration": None},
        )
    kinematic_score = (
        None
        if kinematic_error_m is None
        else 1.0 - _clip01(kinematic_error_m / kinematic_cap_m)
    )
    support_score = (
        None
        if two_sided_count_residual is None
        else 1.0 - _clip01(two_sided_count_residual / count_residual_cap)
    )
    corroboration_score = (
        None if peer_corroboration_score is None else _clip01(peer_corroboration_score)
    )
    scores = {
        "kinematic": kinematic_score,
        "support": support_score,
        "corroboration": corroboration_score,
    }
    weighted = (
        (weights.kinematic * kinematic_score if kinematic_score is not None else 0.0)
        + (weights.support * support_score if support_score is not None else 0.0)
        + (
            weights.corroboration * corroboration_score
            if corroboration_score is not None
            else 0.0
        )
    )
    available_weight = (
        (weights.kinematic if kinematic_score is not None else 0.0)
        + (weights.support if support_score is not None else 0.0)
        + (weights.corroboration if corroboration_score is not None else 0.0)
    )
    if available_weight + 1e-12 < minimum_available_weight:
        return UnknownAwareFixedResult(
            None,
            None,
            available_weight,
            True,
            ReasonCode.INCOMPLETE_EVIDENCE,
            scores,
        )
    trust = weighted / available_weight
    return UnknownAwareFixedResult(
        trust,
        1.0 - trust,
        available_weight,
        False,
        ReasonCode.MATCHED,
        scores,
    )
