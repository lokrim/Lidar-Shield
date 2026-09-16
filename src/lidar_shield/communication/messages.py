"""Construction and integrity validation for deterministic proxy packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lidar_shield.communication.byte_cost import serialize_proxy_packet
from lidar_shield.contracts.schemas import (
    IntegrityMetadata,
    MessageDescriptor,
    PoseDescriptor,
    ProxyPacket,
)
from lidar_shield.manifest import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True)
class PacketBuildInput:
    fixture_run_id: str
    fixture_case: Literal["clean", "integration_corruption"]
    source_id: str
    sequence_id: str
    session_id: str
    sync_frame_id: int
    decision_time_ns: int
    agent_id: str
    source_time_ns: int
    pose: PoseDescriptor
    region_or_object_key: str | int
    local_confidence: float
    payload: tuple[float, ...]


def _descriptor(
    values: PacketBuildInput, payload_sha256: str, size: int
) -> MessageDescriptor:
    return MessageDescriptor(
        fixture_run_id=values.fixture_run_id,
        fixture_case=values.fixture_case,
        source_id=values.source_id,
        sequence_id=values.sequence_id,
        session_id=values.session_id,
        sync_frame_id=values.sync_frame_id,
        decision_time_ns=values.decision_time_ns,
        agent_id=values.agent_id,
        source_time_ns=values.source_time_ns,
        sender_id=values.agent_id,
        pose=values.pose,
        region_or_object_key=values.region_or_object_key,
        payload_type="aligned_proxy_vector",
        local_confidence=values.local_confidence,
        integrity=IntegrityMetadata(
            algorithm="sha256",
            payload_sha256=payload_sha256,
            serialization="canonical-json-sort-keys-utf8-v1",
        ),
        measured_size_bytes=size,
    )


def build_proxy_packet(values: PacketBuildInput) -> ProxyPacket:
    """Build a packet whose embedded measured size equals its exact wire size."""

    payload_sha256 = sha256_bytes(canonical_json_bytes(list(values.payload)))
    size = 1
    for _ in range(8):
        provisional = ProxyPacket(
            descriptor=_descriptor(values, payload_sha256, size),
            payload=values.payload,
            packet_sha256="0" * 64,
        )
        measured = len(serialize_proxy_packet(provisional))
        if measured == size:
            encoded = serialize_proxy_packet(provisional)
            return provisional.model_copy(
                update={"packet_sha256": sha256_bytes(encoded)}
            )
        size = measured
    raise RuntimeError("packet measured-size field did not reach a fixed point")


def verify_proxy_packet(packet: ProxyPacket) -> bool:
    payload_hash = sha256_bytes(canonical_json_bytes(list(packet.payload)))
    encoded = serialize_proxy_packet(packet)
    return (
        payload_hash == packet.descriptor.integrity.payload_sha256
        and len(encoded) == packet.descriptor.measured_size_bytes
        and sha256_bytes(encoded) == packet.packet_sha256
    )
