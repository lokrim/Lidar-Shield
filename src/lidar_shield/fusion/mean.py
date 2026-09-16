"""Ego-anchored arithmetic mean with explicit fallback and abstention."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from lidar_shield.contracts.schemas import ContributorSource, FusionResult
from lidar_shield.fusion.interfaces import AlignedContribution, FusionFrameKey


def mean_fusion(
    key: FusionFrameKey,
    contributions: Sequence[AlignedContribution],
    *,
    policy: Literal["full_share", "hard_gate", "ego_only"],
) -> FusionResult:
    """Average valid aligned contributors while always anchoring on ego."""

    ego_candidates = [item for item in contributions if item.is_ego]
    if len(ego_candidates) > 1:
        raise ValueError("proxy fusion permits at most one ego contribution")
    ego = ego_candidates[0] if ego_candidates else None
    if ego is None or not ego.valid or not ego.aligned:
        return _result(
            key,
            policy=policy,
            vector=None,
            contributions=(),
            degraded=True,
            reason="ego_absent",
            quality=None,
            uncertainty=None,
        )
    remotes = [
        item
        for item in contributions
        if not item.is_ego and item.valid and item.aligned
    ]
    selected = [ego, *remotes]
    width = len(ego.vector)
    if any(len(item.vector) != width for item in selected):
        raise ValueError("all proxy vectors must have the same dimension")
    vector = tuple(
        sum(item.vector[index] for item in selected) / len(selected)
        for index in range(width)
    )
    uncertainties = [item.predictive_uncertainty for item in selected]
    uncertainty = (
        sum(value for value in uncertainties if value is not None) / len(selected)
        if all(value is not None for value in uncertainties)
        else None
    )
    return _result(
        key,
        policy=policy,
        vector=vector,
        contributions=tuple(selected),
        degraded=not remotes,
        reason="cooperative" if remotes else "ego_only_fallback",
        quality=sum(item.quality_score for item in selected) / len(selected),
        uncertainty=uncertainty,
    )


def ego_only_fusion(
    key: FusionFrameKey, ego: AlignedContribution | None
) -> FusionResult:
    """Return the explicit ego-only baseline, or abstain when ego is absent."""

    contributions = () if ego is None else (ego,)
    return mean_fusion(key, contributions, policy="ego_only")


def _result(
    key: FusionFrameKey,
    *,
    policy: Literal["full_share", "hard_gate", "ego_only"],
    vector: tuple[float, ...] | None,
    contributions: tuple[AlignedContribution, ...],
    degraded: bool,
    reason: Literal["cooperative", "ego_only_fallback", "ego_absent"],
    quality: float | None,
    uncertainty: float | None,
) -> FusionResult:
    contributor_sources = tuple(
        ContributorSource(
            fixture_run_id=key.fixture_run_id,
            fixture_case=key.fixture_case,
            source_id=key.source_id,
            sequence_id=key.sequence_id,
            session_id=key.session_id,
            sync_frame_id=key.sync_frame_id,
            decision_time_ns=key.decision_time_ns,
            agent_id=item.agent_id,
            source_time_ns=item.source_time_ns,
            is_ego=item.is_ego,
        )
        for item in contributions
    )
    remote_count = sum(not item.is_ego for item in contributions)
    total_count = len(contributions)
    return FusionResult(
        fixture_run_id=key.fixture_run_id,
        fixture_case=key.fixture_case,
        source_id=key.source_id,
        sequence_id=key.sequence_id,
        session_id=key.session_id,
        sync_frame_id=key.sync_frame_id,
        decision_time_ns=key.decision_time_ns,
        policy=policy,
        fused_vector=vector,
        abstained=vector is None,
        degraded=degraded,
        reason=reason,
        contributors=contributor_sources,
        contributor_count=total_count,
        remote_contributor_count=remote_count,
        total_remote_weight=remote_count / total_count if total_count else 0.0,
        quality_score=quality,
        predictive_uncertainty=uncertainty,
        uncertainty_available=uncertainty is not None,
    )
