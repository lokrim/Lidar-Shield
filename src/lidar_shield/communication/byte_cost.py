"""One exact serialization and byte-cost model shared by all M3 policies."""

from __future__ import annotations

from typing import Any

from lidar_shield.contracts.schemas import ProxyPacket
from lidar_shield.manifest import canonical_json_bytes


def packet_wire_value(packet: ProxyPacket) -> dict[str, Any]:
    return {
        "descriptor": packet.descriptor.model_dump(mode="json"),
        "payload": list(packet.payload),
    }


def serialize_proxy_packet(packet: ProxyPacket) -> bytes:
    """Return the canonical on-wire bytes (including one trailing newline)."""

    return canonical_json_bytes(packet_wire_value(packet))


def measured_packet_size(packet: ProxyPacket) -> int:
    return len(serialize_proxy_packet(packet))
