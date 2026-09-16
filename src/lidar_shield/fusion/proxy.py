"""Adapters from admitted proxy packets to the framework-neutral mean."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from lidar_shield.contracts.schemas import FusionResult, ProxyPacket, ScheduleRecord
from lidar_shield.fusion.interfaces import AlignedContribution, FusionFrameKey
from lidar_shield.fusion.mean import mean_fusion


def _packet_contribution(packet: ProxyPacket) -> AlignedContribution:
    source_time_ns = packet.descriptor.source_time_ns
    if source_time_ns is None:  # prohibited by the packet contract
        raise ValueError("admitted proxy packet lacks source time")
    return AlignedContribution(
        agent_id=packet.descriptor.agent_id,
        source_time_ns=source_time_ns,
        vector=packet.payload,
        is_ego=False,
        quality_score=packet.descriptor.local_confidence,
        predictive_uncertainty=None,
    )


def fuse_admitted_proxy_packets(
    key: FusionFrameKey,
    *,
    ego: AlignedContribution | None,
    packets: Sequence[ProxyPacket],
    schedule: Sequence[ScheduleRecord],
    policy: Literal["full_share", "hard_gate"],
) -> FusionResult:
    admitted = {item.packet_sha256 for item in schedule if item.admitted}
    remote_contributions = tuple(
        _packet_contribution(packet)
        for packet in packets
        if packet.packet_sha256 in admitted
    )
    contributions = (() if ego is None else (ego,)) + remote_contributions
    return mean_fusion(key, contributions, policy=policy)


def proxy_reconstruction_error(
    result: FusionResult, clean_reference: tuple[float, ...]
) -> tuple[float | None, float | None]:
    """Evaluate against a clean fixture reference unavailable to runtime inputs."""

    if not clean_reference or not all(
        math.isfinite(value) for value in clean_reference
    ):
        raise ValueError("clean reference must be a finite non-empty vector")
    if result.fused_vector is None:
        return None, None
    if len(result.fused_vector) != len(clean_reference):
        raise ValueError("fused and reference vectors must have matching dimensions")
    error = math.sqrt(
        sum(
            (actual - expected) ** 2
            for actual, expected in zip(
                result.fused_vector, clean_reference, strict=True
            )
        )
    )
    return error, 1.0 / (1.0 + error)
