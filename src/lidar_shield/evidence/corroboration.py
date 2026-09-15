"""Development-only leave-one-sender-out peer corroboration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median

from lidar_shield.contracts.reasons import ReasonCode


@dataclass(frozen=True)
class CorroborationResult:
    sender_id: str
    available: bool
    reason_code: ReasonCode
    eligible_peer_count: int
    positive_peer_count: int
    peer_location_log1p: float | None
    peer_scale_log1p: float | None
    deficit_residual: float | None
    surplus_residual: float | None
    disagreement: float | None
    peer_corroboration_score: float | None
    development_only: bool = True


def leave_one_sender_out_counts(
    counts: Mapping[str, int | None],
    eligibility: Mapping[str, bool | None],
    *,
    minimum_peers: int = 2,
    scale_floor: float = 0.25,
) -> dict[str, CorroborationResult]:
    """Compare each sender to eligible peers without using its own count.

    ``peer_corroboration_score`` is a witness-availability diagnostic based only
    on peers. Consequently, changing the sender's own points cannot improve it.
    Deficit/surplus and disagreement retain the sender-versus-peer comparison.
    """

    if minimum_peers < 1:
        raise ValueError("minimum_peers must be positive")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be finite and positive")
    if set(counts) != set(eligibility):
        raise ValueError("count and eligibility agent sets must match")
    if any(value is not None and value < 0 for value in counts.values()):
        raise ValueError("point counts must be nonnegative")

    results: dict[str, CorroborationResult] = {}
    for sender, own_count in counts.items():
        peers = [
            count
            for agent, count in counts.items()
            if agent != sender and eligibility[agent] is True and count is not None
        ]
        peer_count = len(peers)
        positive_count = sum(value > 0 for value in peers)
        if (
            peer_count < minimum_peers
            or eligibility[sender] is not True
            or own_count is None
        ):
            reason = (
                ReasonCode.INSUFFICIENT_PEERS
                if peer_count < minimum_peers
                else ReasonCode.VISIBILITY_UNKNOWN
            )
            results[sender] = CorroborationResult(
                sender,
                False,
                reason,
                peer_count,
                positive_count,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            continue
        peer_logs = [math.log1p(value) for value in peers]
        location = float(median(peer_logs))
        raw_mad = float(median(abs(value - location) for value in peer_logs))
        scale = max(1.4826 * raw_mad, scale_floor)
        signed = (math.log1p(own_count) - location) / scale
        deficit = max(-signed, 0.0)
        surplus = max(signed, 0.0)
        results[sender] = CorroborationResult(
            sender,
            True,
            ReasonCode.MATCHED,
            peer_count,
            positive_count,
            location,
            scale,
            deficit,
            surplus,
            abs(signed),
            positive_count / peer_count,
        )
    return results
