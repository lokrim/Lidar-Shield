from __future__ import annotations

import pytest
from pydantic import ValidationError

from lidar_shield.communication.baselines import full_share, hard_gate
from lidar_shield.communication.byte_cost import (
    measured_packet_size,
    serialize_proxy_packet,
)
from lidar_shield.communication.messages import (
    PacketBuildInput,
    build_proxy_packet,
    verify_proxy_packet,
)
from lidar_shield.contracts.schemas import PoseDescriptor, ProxyPacket
from lidar_shield.fusion.interfaces import AlignedContribution, FusionFrameKey
from lidar_shield.fusion.mean import mean_fusion
from lidar_shield.fusion.proxy import (
    fuse_admitted_proxy_packets,
    proxy_reconstruction_error,
)


def packet(*, source_time_ns: int = 990) -> ProxyPacket:
    return build_proxy_packet(
        PacketBuildInput(
            fixture_run_id="run",
            fixture_case="clean",
            source_id="source",
            sequence_id="sequence",
            session_id="session",
            sync_frame_id=0,
            decision_time_ns=1000,
            agent_id="remote",
            source_time_ns=source_time_ns,
            pose=PoseDescriptor(frame_id="top", xyz_m=(1.0, 2.0, 0.0), yaw_rad=0.1),
            region_or_object_key="region",
            local_confidence=0.8,
            payload=(1.0, 3.0),
        )
    )


def key() -> FusionFrameKey:
    return FusionFrameKey("run", "clean", "source", "sequence", "session", 0, 1000)


def test_packet_serialization_is_deterministic_measured_and_integrity_checked() -> None:
    first = packet()
    second = packet()
    assert serialize_proxy_packet(first) == serialize_proxy_packet(second)
    assert first.packet_sha256 == second.packet_sha256
    assert measured_packet_size(first) == first.descriptor.measured_size_bytes
    assert verify_proxy_packet(first)
    assert first.descriptor.sender_id == first.descriptor.agent_id

    changed = first.model_copy(update={"payload": (9.0, 9.0)})
    assert not verify_proxy_packet(changed)


def test_policies_share_packet_cost_deadline_and_invalid_handling() -> None:
    on_time = packet()
    full = full_share((on_time,), max_source_age_ns=20)
    gated = hard_gate(
        (on_time,), states={"remote": "quarantined"}, max_source_age_ns=20
    )
    assert full[0].packet_sha256 == gated[0].packet_sha256
    assert full[0].measured_size_bytes == gated[0].measured_size_bytes
    assert full[0].admitted
    assert not gated[0].admitted
    assert gated[0].reason_code == "hard_gated"

    late = packet(source_time_ns=900)
    assert full_share((late,), max_source_age_ns=20)[0].reason_code == "late"
    assert (
        hard_gate((late,), states={"remote": "normal"}, max_source_age_ns=20)[
            0
        ].reason_code
        == "late"
    )

    invalid = on_time.model_copy(update={"payload": (7.0, 7.0)})
    assert full_share((invalid,), max_source_age_ns=20)[0].reason_code == (
        "invalid_integrity"
    )
    assert (
        hard_gate((invalid,), states={"remote": "normal"}, max_source_age_ns=20)[
            0
        ].reason_code
        == "invalid_integrity"
    )
    with pytest.raises(ValueError):
        full_share((on_time,), max_source_age_ns=-1)


def test_mean_fusion_normalizes_and_preserves_ego_fallback() -> None:
    ego = AlignedContribution("ego", 1000, (0.0, 2.0), True, quality_score=1.0)
    remote = AlignedContribution("remote", 990, (2.0, 4.0), False, quality_score=0.5)
    result = mean_fusion(key(), (ego, remote), policy="full_share")
    assert result.fused_vector == (1.0, 3.0)
    assert result.contributor_count == 2
    assert result.total_remote_weight == 0.5
    assert result.quality_score == 0.75
    assert not result.uncertainty_available

    fallback = mean_fusion(key(), (ego,), policy="ego_only")
    assert fallback.fused_vector == ego.vector
    assert fallback.degraded
    assert fallback.reason == "ego_only_fallback"
    absent = mean_fusion(key(), (remote,), policy="full_share")
    assert absent.abstained and absent.fused_vector is None
    assert absent.reason == "ego_absent"


def test_mean_fusion_validates_inputs_and_propagates_known_uncertainty() -> None:
    ego = AlignedContribution("ego", 1000, (0.0,), True, predictive_uncertainty=0.2)
    remote = AlignedContribution(
        "remote", 990, (2.0,), False, predictive_uncertainty=0.4
    )
    result = mean_fusion(key(), (ego, remote), policy="hard_gate")
    assert result.predictive_uncertainty == pytest.approx(0.3)
    with pytest.raises(ValueError, match="at most one ego"):
        mean_fusion(key(), (ego, ego), policy="full_share")
    with pytest.raises(ValueError, match="same dimension"):
        mean_fusion(
            key(),
            (ego, AlignedContribution("remote", 990, (1.0, 2.0), False)),
            policy="full_share",
        )
    with pytest.raises(ValueError):
        AlignedContribution("bad", 1, (float("nan"),), False)
    with pytest.raises(ValueError):
        AlignedContribution("bad", 1, (0.0,), False, quality_score=2.0)
    with pytest.raises(ValueError):
        AlignedContribution("bad", 1, (0.0,), False, predictive_uncertainty=-1.0)


def test_proxy_adapter_uses_only_admitted_packets_and_reference_is_evaluator_only() -> (
    None
):
    remote = packet()
    schedule = hard_gate(
        (remote,), states={"remote": "quarantined"}, max_source_age_ns=20
    )
    ego = AlignedContribution("ego", 1000, (5.0, 6.0), True)
    result = fuse_admitted_proxy_packets(
        key(), ego=ego, packets=(remote,), schedule=schedule, policy="hard_gate"
    )
    assert result.fused_vector == (5.0, 6.0)
    assert result.degraded
    assert proxy_reconstruction_error(result, (5.0, 6.0)) == (0.0, 1.0)
    abstained = mean_fusion(key(), (), policy="ego_only")
    assert proxy_reconstruction_error(abstained, (0.0,)) == (None, None)
    with pytest.raises(ValueError, match="finite non-empty"):
        proxy_reconstruction_error(result, ())
    with pytest.raises(ValueError, match="matching dimensions"):
        proxy_reconstruction_error(result, (1.0,))


def test_packet_and_pose_contract_validation() -> None:
    with pytest.raises(ValidationError, match="finite"):
        PoseDescriptor(frame_id="top", xyz_m=(0.0, float("inf"), 0.0), yaw_rad=0.0)
    valid = packet()
    descriptor = valid.descriptor.model_dump()
    descriptor["sender_id"] = "different"
    with pytest.raises(ValidationError, match="match agent_id"):
        type(valid.descriptor).model_validate(descriptor)
