"""M3 thin evidence-to-metrics runtime over a disposable integration fixture."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lidar_shield.communication.baselines import full_share, hard_gate
from lidar_shield.communication.messages import PacketBuildInput, build_proxy_packet
from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.contracts.schemas import (
    EvaluationRunMetrics,
    FusionResult,
    PoseDescriptor,
    PredictionRecord,
    ProxyPacket,
    ScheduleRecord,
    StateTransitionRecord,
)
from lidar_shield.evaluation.metrics import evaluate_integration_frame
from lidar_shield.fusion.interfaces import AlignedContribution, FusionFrameKey
from lidar_shield.fusion.proxy import fuse_admitted_proxy_packets
from lidar_shield.runtime.state_store import BoundedStateStore, StateKey
from lidar_shield.temporal.ewma import EWMAConfig, update_ewma
from lidar_shield.temporal.state_machine import BaselineState, label_baseline_state
from lidar_shield.trust.fixed_baseline import unknown_aware_fixed_baseline

FixtureCase = Literal["clean", "integration_corruption"]


class IntegrationFixtureError(ValueError):
    """The disposable M3 fixture is malformed or inconsistent."""


class _FixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FixtureObservation(_FixtureModel):
    agent_id: str = Field(min_length=1)
    source_time_ns: int | None = Field(default=None, ge=0)
    is_ego: bool
    message_present: bool
    kinematic_error_m: float | None = Field(default=None, ge=0.0)
    two_sided_count_residual: float | None = Field(default=None, ge=0.0)
    peer_corroboration_score: float | None = Field(default=None, ge=0.0, le=1.0)
    kinematic_applicable: bool
    vector: tuple[float, ...] | None = Field(default=None, strict=False)
    pose: PoseDescriptor
    region_or_object_key: str | int
    local_confidence: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    reason_code: ReasonCode

    @model_validator(mode="after")
    def validate_presence(self) -> FixtureObservation:
        if self.message_present != (self.source_time_ns is not None):
            raise ValueError("message presence must match source time")
        if self.vector is not None and not self.message_present:
            raise ValueError("unavailable fixture observations cannot carry vectors")
        if self.message_present and self.reason_code is not ReasonCode.MATCHED:
            raise ValueError("present fixture observations must be matched")
        return self


class FixtureFrame(_FixtureModel):
    sync_frame_id: int = Field(ge=0)
    decision_time_ns: int = Field(ge=0)
    clean_reference: tuple[float, ...] = Field(min_length=1, strict=False)
    observations: tuple[FixtureObservation, ...] = Field(min_length=1, strict=False)

    @model_validator(mode="after")
    def unique_agents(self) -> FixtureFrame:
        agents = [item.agent_id for item in self.observations]
        if len(agents) != len(set(agents)):
            raise ValueError("fixture frame agents must be unique")
        return self


class CorruptionReplacement(_FixtureModel):
    kinematic_error_m: float = Field(ge=0.0)
    two_sided_count_residual: float = Field(ge=0.0)
    peer_corroboration_score: float = Field(ge=0.0, le=1.0)
    vector: tuple[float, ...] = Field(min_length=1, strict=False)


class IntegrationCorruption(_FixtureModel):
    label: Literal["integration_corruption_fixture"]
    seed: int = Field(ge=0)
    target_sync_frame_id: int = Field(ge=0)
    target_agent_id: str = Field(min_length=1)
    replacement: CorruptionReplacement
    scientific_eligibility: Literal["none"]


class AnalyticalExpectations(_FixtureModel):
    target_clean_risk: float
    target_corrupted_risk: float
    target_corrupted_ewma: float
    target_corrupted_state: Literal["quarantined"]
    hard_gate_corrupted_source_bytes: Literal[0]


class IntegrationFixture(_FixtureModel):
    fixture_schema_version: Literal["1.0.0"]
    fixture_run_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    ego_agent_id: str = Field(min_length=1)
    ewma: EWMAConfig
    max_source_age_ns: int = Field(ge=0)
    frames: tuple[FixtureFrame, ...] = Field(min_length=1, strict=False)
    integration_corruption: IntegrationCorruption
    analytical_expectations: AnalyticalExpectations

    @model_validator(mode="after")
    def validate_fixture(self) -> IntegrationFixture:
        frame_ids = [item.sync_frame_id for item in self.frames]
        if frame_ids != sorted(frame_ids) or len(frame_ids) != len(set(frame_ids)):
            raise ValueError("fixture frames must use unique chronological IDs")
        if self.seed != self.integration_corruption.seed:
            raise ValueError("fixture and integration-corruption seeds must match")
        for frame in self.frames:
            ego_rows = [
                item
                for item in frame.observations
                if item.is_ego and item.agent_id == self.ego_agent_id
            ]
            if len(ego_rows) != 1 or any(
                item.is_ego != (item.agent_id == self.ego_agent_id)
                for item in frame.observations
            ):
                raise ValueError(
                    "each frame requires one consistently identified ego row"
                )
        target = [
            observation
            for frame in self.frames
            if frame.sync_frame_id == self.integration_corruption.target_sync_frame_id
            for observation in frame.observations
            if observation.agent_id == self.integration_corruption.target_agent_id
        ]
        if len(target) != 1 or not target[0].message_present:
            raise ValueError(
                "integration corruption must target one present observation"
            )
        return self


@dataclass(frozen=True)
class _StoredTemporalState:
    ewma_value: float
    label: BaselineState


@dataclass(frozen=True)
class IntegrationRun:
    fixture_run_id: str
    fixture_case: FixtureCase
    predictions: tuple[PredictionRecord, ...]
    states: tuple[StateTransitionRecord, ...]
    packets: tuple[ProxyPacket, ...]
    schedules: tuple[ScheduleRecord, ...]
    fusion: tuple[FusionResult, ...]
    metrics: tuple[EvaluationRunMetrics, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "fixture_run_id": self.fixture_run_id,
            "fixture_case": self.fixture_case,
            "predictions": [item.model_dump(mode="json") for item in self.predictions],
            "states": [item.model_dump(mode="json") for item in self.states],
            "packets": [item.model_dump(mode="json") for item in self.packets],
            "schedules": [item.model_dump(mode="json") for item in self.schedules],
            "fusion": [item.model_dump(mode="json") for item in self.fusion],
            "metrics": [item.model_dump(mode="json") for item in self.metrics],
        }


def load_integration_fixture(path: str | Path) -> IntegrationFixture:
    fixture_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        return IntegrationFixture.model_validate(raw, strict=False)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise IntegrationFixtureError(
            f"invalid M3 fixture {fixture_path}: {exc}"
        ) from exc


def _identity(
    fixture: IntegrationFixture,
    case: FixtureCase,
    frame: FixtureFrame,
    observation: FixtureObservation,
) -> dict[str, Any]:
    return {
        "fixture_run_id": fixture.fixture_run_id,
        "fixture_case": case,
        "source_id": fixture.source_id,
        "sequence_id": fixture.sequence_id,
        "session_id": fixture.session_id,
        "sync_frame_id": frame.sync_frame_id,
        "decision_time_ns": frame.decision_time_ns,
        "agent_id": observation.agent_id,
        "source_time_ns": observation.source_time_ns,
    }


def _observation_for_case(
    fixture: IntegrationFixture,
    frame: FixtureFrame,
    observation: FixtureObservation,
    case: FixtureCase,
) -> FixtureObservation:
    corruption = fixture.integration_corruption
    if (
        case == "integration_corruption"
        and frame.sync_frame_id == corruption.target_sync_frame_id
        and observation.agent_id == corruption.target_agent_id
    ):
        return observation.model_copy(update=corruption.replacement.model_dump())
    return observation


def _predict(
    fixture: IntegrationFixture,
    case: FixtureCase,
    frame: FixtureFrame,
    observation: FixtureObservation,
) -> PredictionRecord:
    fixed = unknown_aware_fixed_baseline(
        kinematic_error_m=observation.kinematic_error_m,
        two_sided_count_residual=observation.two_sided_count_residual,
        peer_corroboration_score=observation.peer_corroboration_score,
        kinematic_applicable=observation.kinematic_applicable,
    )
    prediction_available = observation.message_present and not fixed.abstained
    risk = fixed.risk if prediction_available else None
    reason = (
        fixed.reason_code if observation.message_present else observation.reason_code
    )
    return PredictionRecord(
        **_identity(fixture, case, frame, observation),
        model_id="unknown-aware-fixed-v1",
        risk_score=risk,
        risk_semantics="uncalibrated_attack_risk_score",
        calibration_status="uncalibrated",
        calibrated_p_attack=None,
        operational_reliability=None if risk is None else 1.0 - risk,
        predictive_uncertainty=None,
        uncertainty_available=False,
        evidence_available=observation.message_present,
        abstained=not prediction_available,
        reason_code=reason,
    )


def _transition(
    fixture: IntegrationFixture,
    case: FixtureCase,
    frame: FixtureFrame,
    observation: FixtureObservation,
    prediction: PredictionRecord,
    store: BoundedStateStore[_StoredTemporalState],
) -> StateTransitionRecord:
    key = StateKey(fixture.sequence_id, fixture.session_id, observation.agent_id)
    previous = store.get(key)
    update = update_ewma(
        None if previous is None else previous.ewma_value,
        prediction.risk_score,
        fixture.ewma,
    )
    state = label_baseline_state(
        update, "unknown" if previous is None else previous.label
    )
    if update.value is not None:
        store.set(key, _StoredTemporalState(update.value, state.current_state))
    return StateTransitionRecord(
        **_identity(fixture, case, frame, observation),
        input_risk_score=prediction.risk_score,
        input_available=prediction.risk_score is not None,
        ewma_risk_score=update.value,
        alpha=fixture.ewma.alpha,
        threshold=fixture.ewma.threshold,
        initialization=fixture.ewma.initialization,
        missing_evidence_policy=fixture.ewma.missing_evidence_policy,
        previous_state=state.previous_state,
        current_state=state.current_state,
        transition_reason=state.transition_reason,
    )


def _packet(
    fixture: IntegrationFixture,
    case: FixtureCase,
    frame: FixtureFrame,
    observation: FixtureObservation,
) -> ProxyPacket | None:
    if (
        observation.is_ego
        or not observation.message_present
        or observation.source_time_ns is None
        or observation.vector is None
    ):
        return None
    return build_proxy_packet(
        PacketBuildInput(
            fixture_run_id=fixture.fixture_run_id,
            fixture_case=case,
            source_id=fixture.source_id,
            sequence_id=fixture.sequence_id,
            session_id=fixture.session_id,
            sync_frame_id=frame.sync_frame_id,
            decision_time_ns=frame.decision_time_ns,
            agent_id=observation.agent_id,
            source_time_ns=observation.source_time_ns,
            pose=observation.pose,
            region_or_object_key=observation.region_or_object_key,
            local_confidence=observation.local_confidence,
            payload=observation.vector,
        )
    )


def _ego_contribution(
    observations: tuple[FixtureObservation, ...],
) -> AlignedContribution | None:
    ego = [item for item in observations if item.is_ego]
    if len(ego) > 1:
        raise IntegrationFixtureError("fixture frame contains multiple ego rows")
    if not ego or not ego[0].message_present or ego[0].vector is None:
        return None
    item = ego[0]
    if item.source_time_ns is None:
        return None
    vector = item.vector
    if vector is None:  # narrowed above; defensive for static analysis
        return None
    return AlignedContribution(
        agent_id=item.agent_id,
        source_time_ns=item.source_time_ns,
        vector=vector,
        is_ego=True,
        quality_score=item.quality_score,
        predictive_uncertainty=None,
    )


def run_integration_fixture(
    fixture: IntegrationFixture, case: FixtureCase
) -> IntegrationRun:
    """Execute one clean or integration-corruption case without writing artifacts."""

    store: BoundedStateStore[_StoredTemporalState] = BoundedStateStore(max_entries=64)
    all_predictions: list[PredictionRecord] = []
    all_states: list[StateTransitionRecord] = []
    all_packets: list[ProxyPacket] = []
    all_schedules: list[ScheduleRecord] = []
    all_fusion: list[FusionResult] = []
    all_metrics: list[EvaluationRunMetrics] = []
    corruption_sources = (
        {
            (
                fixture.integration_corruption.target_sync_frame_id,
                fixture.integration_corruption.target_agent_id,
            )
        }
        if case == "integration_corruption"
        else set()
    )
    for frame in fixture.frames:
        observations = tuple(
            _observation_for_case(fixture, frame, item, case)
            for item in frame.observations
        )
        predictions = tuple(
            _predict(fixture, case, frame, item) for item in observations
        )
        states = tuple(
            _transition(fixture, case, frame, item, prediction, store)
            for item, prediction in zip(observations, predictions, strict=True)
        )
        packets = tuple(
            packet
            for item in observations
            if (packet := _packet(fixture, case, frame, item)) is not None
        )
        state_by_sender = {item.agent_id: item.current_state for item in states}
        schedules_by_policy: dict[
            Literal["full_share", "hard_gate"], tuple[ScheduleRecord, ...]
        ] = {
            "full_share": full_share(
                packets, max_source_age_ns=fixture.max_source_age_ns
            ),
            "hard_gate": hard_gate(
                packets,
                states=state_by_sender,
                max_source_age_ns=fixture.max_source_age_ns,
            ),
        }
        key = FusionFrameKey(
            fixture.fixture_run_id,
            case,
            fixture.source_id,
            fixture.sequence_id,
            fixture.session_id,
            frame.sync_frame_id,
            frame.decision_time_ns,
        )
        ego = _ego_contribution(observations)
        for policy, schedule in schedules_by_policy.items():
            fusion = fuse_admitted_proxy_packets(
                key,
                ego=ego,
                packets=packets,
                schedule=schedule,
                policy=policy,
            )
            metric = evaluate_integration_frame(
                predictions,
                states,
                schedule,
                fusion,
                clean_reference=frame.clean_reference,
                integration_corruption_sources=corruption_sources,
                ego_agent_id=fixture.ego_agent_id,
            )
            all_schedules.extend(schedule)
            all_fusion.append(fusion)
            all_metrics.append(metric)
        all_predictions.extend(predictions)
        all_states.extend(states)
        all_packets.extend(packets)
    return IntegrationRun(
        fixture.fixture_run_id,
        case,
        tuple(all_predictions),
        tuple(all_states),
        tuple(all_packets),
        tuple(all_schedules),
        tuple(all_fusion),
        tuple(all_metrics),
    )


def build_d2_payload(path: str | Path) -> dict[str, Any]:
    """Run both fixture cases and return a deterministic, non-scientific report."""

    fixture = load_integration_fixture(path)
    cases: tuple[FixtureCase, ...] = ("clean", "integration_corruption")
    runs = {case: run_integration_fixture(fixture, case) for case in cases}
    summaries: dict[str, dict[str, Any]] = {}
    target_frame = fixture.integration_corruption.target_sync_frame_id
    target_agent = fixture.integration_corruption.target_agent_id
    for case, run in runs.items():
        target_prediction = next(
            item
            for item in run.predictions
            if item.sync_frame_id == target_frame and item.agent_id == target_agent
        )
        target_state = next(
            item
            for item in run.states
            if item.sync_frame_id == target_frame and item.agent_id == target_agent
        )
        policy_metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "admitted_bytes": 0,
                "corrupted_source_bytes": 0,
                "proxy_reconstruction_error": 0.0,
                "fallback_count": 0,
                "fusion_abstention_count": 0,
            }
        )
        for metric in run.metrics:
            summary = policy_metrics[metric.policy]
            summary["admitted_bytes"] += metric.admitted_bytes
            summary["corrupted_source_bytes"] += metric.corrupted_source_bytes
            if metric.proxy_reconstruction_error is not None:
                summary["proxy_reconstruction_error"] += (
                    metric.proxy_reconstruction_error
                )
            summary["fallback_count"] += int(metric.fallback)
            summary["fusion_abstention_count"] += int(metric.fusion_abstained)
        summaries[case] = {
            "target_uncalibrated_risk": target_prediction.risk_score,
            "target_calibrated_p_attack": target_prediction.calibrated_p_attack,
            "target_uncertainty_available": target_prediction.uncertainty_available,
            "target_ewma_risk": target_state.ewma_risk_score,
            "target_state": target_state.current_state,
            "policies": dict(sorted(policy_metrics.items())),
        }
    return {
        "demo": "D2",
        "fixture_run_id": fixture.fixture_run_id,
        "fixture_seed": fixture.seed,
        "result_scope": "disposable integration corruption fixture only",
        "scientific_eligibility": "none",
        "reference_scope": "evaluation only; never risk or scheduling input",
        "summaries": summaries,
        "analytical_expectations": fixture.analytical_expectations.model_dump(
            mode="json"
        ),
        "runs": {case: run.model_dump() for case, run in runs.items()},
    }
