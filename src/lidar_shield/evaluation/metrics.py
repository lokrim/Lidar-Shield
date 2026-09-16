"""Minimal identity-checked metrics for the non-scientific M3 fixture."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence

from lidar_shield.contracts.schemas import (
    ContributorSource,
    EvaluationRunMetrics,
    FusionResult,
    PredictionRecord,
    ScheduleRecord,
    StateTransitionRecord,
)
from lidar_shield.fusion.proxy import proxy_reconstruction_error


def _frame_key(record: object) -> tuple[object, ...]:
    return tuple(
        getattr(record, field)
        for field in (
            "fixture_run_id",
            "fixture_case",
            "source_id",
            "sequence_id",
            "session_id",
            "sync_frame_id",
            "decision_time_ns",
        )
    )


def evaluate_integration_frame(
    predictions: Sequence[PredictionRecord],
    states: Sequence[StateTransitionRecord],
    schedules: Sequence[ScheduleRecord],
    fusion: FusionResult,
    *,
    clean_reference: tuple[float, ...],
    integration_corruption_sources: Collection[tuple[int, str]] = (),
    ego_agent_id: str = "ego",
) -> EvaluationRunMetrics:
    """Join one complete frame without filtering missing or companion senders."""

    if not predictions:
        raise ValueError("integration metrics require expected observation rows")
    expected_key = _frame_key(fusion)
    records: tuple[object, ...] = (*predictions, *states, *schedules)
    if any(_frame_key(item) != expected_key for item in records):
        raise ValueError("integration metric inputs have mismatched fixture identities")
    prediction_agents = [item.agent_id for item in predictions]
    if len(prediction_agents) != len(set(prediction_agents)):
        raise ValueError("prediction rows must be unique by sender")
    if {item.agent_id for item in states} != set(prediction_agents):
        raise ValueError("every expected observation requires a state row")
    risks = [item.risk_score for item in predictions if item.risk_score is not None]
    error, utility = proxy_reconstruction_error(fusion, clean_reference)
    first = predictions[0]
    sources = tuple(
        ContributorSource(
            fixture_run_id=item.fixture_run_id,
            fixture_case=item.fixture_case,
            source_id=item.source_id,
            sequence_id=item.sequence_id,
            session_id=item.session_id,
            sync_frame_id=item.sync_frame_id,
            decision_time_ns=item.decision_time_ns,
            agent_id=item.agent_id,
            source_time_ns=item.source_time_ns,
            is_ego=item.agent_id == ego_agent_id,
        )
        for item in predictions
    )
    state_counts = Counter(item.current_state for item in states)
    return EvaluationRunMetrics(
        fixture_run_id=first.fixture_run_id,
        fixture_case=first.fixture_case,
        source_id=first.source_id,
        sequence_id=first.sequence_id,
        session_id=first.session_id,
        sync_frame_id=first.sync_frame_id,
        decision_time_ns=first.decision_time_ns,
        policy=fusion.policy,
        observation_sources=sources,
        observation_count=len(predictions),
        missing_observation_count=sum(
            not item.evidence_available for item in predictions
        ),
        prediction_available_count=len(risks),
        abstention_count=sum(item.abstained for item in predictions),
        maximum_uncalibrated_risk=max(risks) if risks else None,
        state_counts=dict(sorted(state_counts.items())),
        admitted_bytes=sum(item.admitted_bytes for item in schedules),
        corrupted_source_bytes=sum(
            item.admitted_bytes
            for item in schedules
            if (item.sync_frame_id, item.agent_id) in integration_corruption_sources
        ),
        proxy_reconstruction_error=error,
        proxy_utility=utility,
        fallback=fusion.degraded,
        fusion_abstained=fusion.abstained,
        reference_scope="evaluation_only_clean_fixture",
    )
