from __future__ import annotations

from typing import Literal, cast

import pytest
from pydantic import ValidationError

from lidar_shield.evaluation.episodes import (
    EpisodeObservation,
    replay_attack_episode,
    summarize_attack_coverage,
)
from lidar_shield.temporal.ewma import EWMAConfig


def _observations(*, attacked: bool) -> tuple[EpisodeObservation, ...]:
    rows = []
    for frame in (0, 1):
        for agent, agent_type, ego in (
            ("top", "infrastructure", True),
            ("003", "vehicle", False),
            ("dome", "infrastructure", False),
        ):
            target = agent == "003" and frame == 1
            collateral = agent == "dome" and frame == 1 and attacked
            rows.append(
                EpisodeObservation(
                    episode_id="episode-001",
                    variant_id="velocity-spike-severe-seed-17",
                    source_id="mixed-signals-train-mini_7",
                    sequence_id="mini_7",
                    session_id="mini_7-session",
                    sync_frame_id=frame,
                    decision_time_ns=1_000_000_000 + frame * 100_000_000,
                    source_time_ns=1_000_000_000 + frame * 100_000_000,
                    agent_id=agent,
                    agent_type=cast(Literal["vehicle", "infrastructure"], agent_type),
                    is_ego=ego,
                    intended_target=target,
                    actually_affected=(target or collateral) and attacked,
                    message_present=True,
                    kinematic_error_m=(0.057 if target and attacked else 0.0),
                    two_sided_count_residual=(3.0 if collateral else 0.0),
                    peer_corroboration_score=(0.0 if collateral else 1.0),
                    kinematic_applicable=agent_type == "vehicle",
                    proxy_vector=(
                        (8.0, 8.0)
                        if target and attacked
                        else ((2.0, 2.0) if collateral else (1.0, 1.0))
                    ),
                )
            )
    return tuple(rows)


def test_episode_replay_keeps_identity_honest_companions_and_final_outputs() -> None:
    clean = replay_attack_episode(
        _observations(attacked=False),
        case="clean",
        ewma=EWMAConfig(alpha=0.3, threshold=0.1),
    )
    attacked = replay_attack_episode(
        _observations(attacked=True),
        case="attacked",
        ewma=EWMAConfig(alpha=0.3, threshold=0.1),
    )
    assert clean.as_dict()["case"] == "clean"
    target = next(
        item
        for item in attacked.predictions
        if item["sync_frame_id"] == 1 and item["agent_id"] == "003"
    )
    assert target["variant_id"] == "velocity-spike-severe-seed-17"
    assert target["episode_id"] == "episode-001"
    assert target["risk_score"] == pytest.approx(0.4)
    companion = next(
        item
        for item in attacked.predictions
        if item["sync_frame_id"] == 1 and item["agent_id"] == "dome"
    )
    assert not companion["intended_target"] and companion["actually_affected"]
    attacked_metric = next(
        item
        for item in attacked.metrics
        if item["sync_frame_id"] == 1 and item["policy"] == "full_share"
    )
    assert attacked_metric["observation_count"] == 3
    assert attacked_metric["honest_companion_count"] == 2
    assert attacked_metric["actually_affected_honest_companion_count"] == 1
    hard = next(
        item
        for item in attacked.metrics
        if item["sync_frame_id"] == 1 and item["policy"] == "hard_gate"
    )
    assert hard["affected_source_bytes"] == 0
    assert attacked.fusion != clean.fusion
    for collection in (
        attacked.predictions,
        attacked.states,
        attacked.schedules,
        attacked.fusion,
        attacked.metrics,
    ):
        assert all(item["episode_id"] == "episode-001" for item in collection)
        assert all(
            item["variant_id"] == "velocity-spike-severe-seed-17" for item in collection
        )


def test_episode_replay_missing_fallback_abstention_and_validation() -> None:
    base = _observations(attacked=False)
    unavailable = tuple(
        item.model_copy(
            update={
                "source_time_ns": None,
                "message_present": False,
                "proxy_vector": None,
            }
        )
        if item.sync_frame_id == 1 and not item.is_ego
        else item
        for item in base
    )
    replay = replay_attack_episode(unavailable, case="attacked")
    frame_one = [item for item in replay.fusion if item["sync_frame_id"] == 1]
    assert all(item["fallback"] for item in frame_one)
    no_ego = tuple(
        item.model_copy(
            update={
                "source_time_ns": None,
                "message_present": False,
                "proxy_vector": None,
            }
        )
        if item.sync_frame_id == 1
        else item
        for item in base
    )
    abstained = replay_attack_episode(no_ego, case="attacked")
    assert all(
        item["abstained"] for item in abstained.fusion if item["sync_frame_id"] == 1
    )
    with pytest.raises(ValueError, match="requires observations"):
        replay_attack_episode((), case="clean")
    with pytest.raises(ValueError, match="share full"):
        replay_attack_episode(
            (base[0], base[1].model_copy(update={"variant_id": "collision"})),
            case="clean",
        )
    with pytest.raises(ValidationError):
        EpisodeObservation(
            **{
                **base[0].model_dump(),
                "message_present": False,
                "source_time_ns": base[0].source_time_ns,
            }
        )


def test_coverage_retains_every_attempt_and_deduplicates_nothing() -> None:
    attempts = (
        {
            "family": "kinematic",
            "severity": "low",
            "agent_type": "vehicle",
            "status": "succeeded",
            "realized_effect_band": "low",
            "variant_id": "seed-1",
        },
        {
            "family": "kinematic",
            "severity": "low",
            "agent_type": "vehicle",
            "status": "zero_effect",
            "realized_effect_band": "zero",
            "variant_id": "seed-2",
        },
        {
            "family": "spatial_support",
            "severity": "high",
            "agent_type": "infrastructure",
            "status": "failed",
            "realized_effect_band": "unavailable",
            "variant_id": "seed-3",
        },
        {
            "family": "spatial_support",
            "severity": "medium",
            "agent_type": "vehicle",
            "status": "interrupted",
            "realized_effect_band": "unknown",
            "variant_id": "seed-4",
        },
    )
    coverage = summarize_attack_coverage(attempts)
    assert coverage["attempt_count"] == 4
    assert coverage["by_status"] == {
        "failed": 1,
        "interrupted": 1,
        "succeeded": 1,
        "zero_effect": 1,
    }
    assert coverage["by_family"] == {
        "kinematic": 2,
        "spatial_support": 2,
    }
