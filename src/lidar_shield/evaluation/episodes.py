"""M4 scientific episode replay through the complete M3 baseline chain."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lidar_shield.manifest import canonical_json_bytes, sha256_bytes
from lidar_shield.temporal.ewma import DEFAULT_EWMA_CONFIG, EWMAConfig, update_ewma
from lidar_shield.temporal.state_machine import BaselineState, label_baseline_state
from lidar_shield.trust.fixed_baseline import unknown_aware_fixed_baseline

EpisodeCase = Literal["clean", "attacked"]
Policy = Literal["full_share", "hard_gate"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EpisodeObservation(_StrictModel):
    episode_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sync_frame_id: int = Field(ge=0)
    decision_time_ns: int = Field(ge=0)
    source_time_ns: int | None = Field(default=None, ge=0)
    agent_id: str = Field(min_length=1)
    agent_type: Literal["vehicle", "infrastructure"]
    is_ego: bool
    intended_target: bool
    actually_affected: bool
    message_present: bool
    kinematic_error_m: float | None = Field(default=None, ge=0.0)
    two_sided_count_residual: float | None = Field(default=None, ge=0.0)
    peer_corroboration_score: float | None = Field(default=None, ge=0.0, le=1.0)
    kinematic_applicable: bool
    proxy_vector: tuple[float, ...] | None = Field(default=None, strict=False)
    local_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_presence(self) -> EpisodeObservation:
        if self.message_present != (self.source_time_ns is not None):
            raise ValueError("message presence must match source time")
        if self.proxy_vector is not None:
            if not self.message_present or not all(
                math.isfinite(item) for item in self.proxy_vector
            ):
                raise ValueError("proxy vectors require finite present messages")
        return self


@dataclass(frozen=True)
class EpisodeReplay:
    case: EpisodeCase
    predictions: tuple[dict[str, Any], ...]
    states: tuple[dict[str, Any], ...]
    schedules: tuple[dict[str, Any], ...]
    fusion: tuple[dict[str, Any], ...]
    metrics: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "predictions": list(self.predictions),
            "states": list(self.states),
            "schedules": list(self.schedules),
            "fusion": list(self.fusion),
            "metrics": list(self.metrics),
        }


def _identity(item: EpisodeObservation, case: EpisodeCase) -> dict[str, Any]:
    return {
        "episode_id": item.episode_id,
        "variant_id": item.variant_id,
        "case": case,
        "source_id": item.source_id,
        "sequence_id": item.sequence_id,
        "session_id": item.session_id,
        "sync_frame_id": item.sync_frame_id,
        "decision_time_ns": item.decision_time_ns,
        "agent_id": item.agent_id,
        "source_time_ns": item.source_time_ns,
    }


def _packet(item: EpisodeObservation, case: EpisodeCase) -> dict[str, Any] | None:
    if item.is_ego or item.proxy_vector is None or item.source_time_ns is None:
        return None
    core = {
        **_identity(item, case),
        "payload": list(item.proxy_vector),
        "local_confidence": item.local_confidence,
    }
    core["payload_sha256"] = sha256_bytes(canonical_json_bytes(core["payload"]))
    size = 1
    for _ in range(8):
        wire = {**core, "measured_size_bytes": size}
        measured = len(canonical_json_bytes(wire))
        if measured == size:
            wire["packet_sha256"] = sha256_bytes(canonical_json_bytes(wire))
            return wire
        size = measured
    raise RuntimeError("episode packet size failed to converge")


def replay_attack_episode(
    observations: tuple[EpisodeObservation, ...],
    *,
    case: EpisodeCase,
    ewma: EWMAConfig = DEFAULT_EWMA_CONFIG,
    max_source_age_ns: int = 100_000_000,
) -> EpisodeReplay:
    """Replay all expected agents; companions are never filtered from metrics."""

    if not observations:
        raise ValueError("episode replay requires observations")
    identities = {
        (item.episode_id, item.variant_id, item.source_id, item.sequence_id)
        for item in observations
    }
    if len(identities) != 1:
        raise ValueError("episode observations must share full source/variant identity")
    ordered = tuple(
        sorted(observations, key=lambda item: (item.decision_time_ns, item.agent_id))
    )
    temporal: dict[str, tuple[float, BaselineState]] = {}
    predictions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    frame_ids = sorted({item.sync_frame_id for item in ordered})
    for frame_id in frame_ids:
        frame = tuple(item for item in ordered if item.sync_frame_id == frame_id)
        frame_predictions: dict[str, float | None] = {}
        frame_states: dict[str, BaselineState] = {}
        for item in frame:
            result = unknown_aware_fixed_baseline(
                kinematic_error_m=item.kinematic_error_m,
                two_sided_count_residual=item.two_sided_count_residual,
                peer_corroboration_score=item.peer_corroboration_score,
                kinematic_applicable=item.kinematic_applicable,
            )
            risk = (
                result.risk if item.message_present and not result.abstained else None
            )
            frame_predictions[item.agent_id] = risk
            predictions.append(
                {
                    **_identity(item, case),
                    "risk_score": risk,
                    "abstained": risk is None,
                    "evidence_available": item.message_present,
                    "intended_target": item.intended_target,
                    "actually_affected": item.actually_affected,
                }
            )
            previous = temporal.get(item.agent_id)
            update = update_ewma(None if previous is None else previous[0], risk, ewma)
            label = label_baseline_state(
                update, "unknown" if previous is None else previous[1]
            )
            if update.value is not None:
                temporal[item.agent_id] = (update.value, label.current_state)
            frame_states[item.agent_id] = label.current_state
            states.append(
                {
                    **_identity(item, case),
                    "input_risk_score": risk,
                    "ewma_risk_score": update.value,
                    "state": label.current_state,
                    "transition_reason": label.transition_reason,
                    "intended_target": item.intended_target,
                    "actually_affected": item.actually_affected,
                }
            )
        packets = tuple(
            packet for item in frame if (packet := _packet(item, case)) is not None
        )
        packet_by_agent = {str(item["agent_id"]): item for item in packets}
        ego = next((item for item in frame if item.is_ego), None)
        for policy in ("full_share", "hard_gate"):
            admitted_vectors: list[tuple[float, ...]] = []
            admitted_bytes = 0
            affected_bytes = 0
            for item in frame:
                packet = packet_by_agent.get(item.agent_id)
                if packet is None:
                    continue
                assert item.source_time_ns is not None
                age = item.decision_time_ns - item.source_time_ns
                admitted = 0 <= age <= max_source_age_ns and (
                    policy == "full_share" or frame_states[item.agent_id] == "normal"
                )
                size = int(packet["measured_size_bytes"])
                schedules.append(
                    {
                        **_identity(item, case),
                        "policy": policy,
                        "packet_sha256": packet["packet_sha256"],
                        "measured_size_bytes": size,
                        "admitted": admitted,
                        "admitted_bytes": size if admitted else 0,
                        "intended_target": item.intended_target,
                        "actually_affected": item.actually_affected,
                    }
                )
                if admitted:
                    assert item.proxy_vector is not None
                    admitted_vectors.append(item.proxy_vector)
                    admitted_bytes += size
                    affected_bytes += size if item.actually_affected else 0
            vectors = []
            if ego is not None and ego.proxy_vector is not None:
                vectors.append(ego.proxy_vector)
            vectors.extend(admitted_vectors)
            if vectors:
                width = len(vectors[0])
                if any(len(item) != width for item in vectors):
                    raise ValueError("episode proxy vectors must have one width")
                fused = tuple(
                    sum(item[index] for item in vectors) / len(vectors)
                    for index in range(width)
                )
            else:
                fused = None
            first = frame[0]
            frame_identity = {
                key: value
                for key, value in _identity(first, case).items()
                if key not in {"agent_id", "source_time_ns"}
            }
            fusion_rows.append(
                {
                    **frame_identity,
                    "policy": policy,
                    "fused_vector": fused,
                    "remote_contributor_count": len(admitted_vectors),
                    "fallback": bool(vectors and not admitted_vectors),
                    "abstained": fused is None,
                }
            )
            metric_rows.append(
                {
                    **frame_identity,
                    "policy": policy,
                    "observation_count": len(frame),
                    "honest_companion_count": sum(
                        not item.intended_target for item in frame
                    ),
                    "actually_affected_honest_companion_count": sum(
                        not item.intended_target and item.actually_affected
                        for item in frame
                    ),
                    "admitted_bytes": admitted_bytes,
                    "affected_source_bytes": affected_bytes,
                    "fallback": bool(vectors and not admitted_vectors),
                    "fusion_abstained": fused is None,
                }
            )
    return EpisodeReplay(
        case,
        tuple(predictions),
        tuple(states),
        tuple(schedules),
        tuple(fusion_rows),
        tuple(metric_rows),
    )


def summarize_attack_coverage(attempts: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Coverage denominators never screen failed, zero-, or low-effect rows."""

    dimensions = (
        "family",
        "severity",
        "agent_type",
        "status",
        "realized_effect_band",
    )
    result: dict[str, Any] = {"attempt_count": len(attempts)}
    for dimension in dimensions:
        counts: dict[str, int] = {}
        for item in attempts:
            key = str(item[dimension])
            counts[key] = counts.get(key, 0) + 1
        result[f"by_{dimension}"] = dict(sorted(counts.items()))
    return result
