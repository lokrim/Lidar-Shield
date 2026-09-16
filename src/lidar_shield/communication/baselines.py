"""Full-share and hard-gate baselines over the same immutable packets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from lidar_shield.communication.messages import verify_proxy_packet
from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.contracts.schemas import ProxyPacket, ScheduleRecord
from lidar_shield.temporal.state_machine import BaselineState

PolicyName = Literal["full_share", "hard_gate"]


def _schedule(
    packets: Sequence[ProxyPacket],
    *,
    policy: PolicyName,
    states: Mapping[str, BaselineState],
    max_source_age_ns: int,
) -> tuple[ScheduleRecord, ...]:
    if max_source_age_ns < 0:
        raise ValueError("maximum source age must be nonnegative")
    decisions: list[ScheduleRecord] = []
    for packet in packets:
        descriptor = packet.descriptor
        source_time_ns = descriptor.source_time_ns
        if source_time_ns is None:  # guarded by MessageDescriptor; defensive only
            raise ValueError("scheduled packet lacks source time")
        age = descriptor.decision_time_ns - source_time_ns
        integrity_valid = verify_proxy_packet(packet)
        if not integrity_valid:
            admitted, reason = False, ReasonCode.INVALID_INTEGRITY
        elif age < 0 or age > max_source_age_ns:
            admitted, reason = False, ReasonCode.LATE
        elif policy == "hard_gate" and states.get(descriptor.sender_id) != "normal":
            admitted, reason = False, ReasonCode.HARD_GATED
        else:
            admitted, reason = True, ReasonCode.ADMITTED
        decisions.append(
            ScheduleRecord(
                **descriptor.model_dump(
                    include={
                        "schema_version",
                        "fixture_run_id",
                        "fixture_case",
                        "source_id",
                        "sequence_id",
                        "session_id",
                        "sync_frame_id",
                        "decision_time_ns",
                        "agent_id",
                        "source_time_ns",
                    }
                ),
                policy=policy,
                packet_sha256=packet.packet_sha256,
                measured_size_bytes=descriptor.measured_size_bytes,
                max_source_age_ns=max_source_age_ns,
                source_age_ns=age,
                integrity_valid=integrity_valid,
                admitted=admitted,
                admitted_bytes=descriptor.measured_size_bytes if admitted else 0,
                reason_code=reason,
            )
        )
    return tuple(decisions)


def full_share(
    packets: Sequence[ProxyPacket], *, max_source_age_ns: int
) -> tuple[ScheduleRecord, ...]:
    return _schedule(
        packets,
        policy="full_share",
        states={},
        max_source_age_ns=max_source_age_ns,
    )


def hard_gate(
    packets: Sequence[ProxyPacket],
    *,
    states: Mapping[str, BaselineState],
    max_source_age_ns: int,
) -> tuple[ScheduleRecord, ...]:
    return _schedule(
        packets,
        policy="hard_gate",
        states=states,
        max_source_age_ns=max_source_age_ns,
    )
