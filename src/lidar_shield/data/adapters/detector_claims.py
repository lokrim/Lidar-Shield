"""Stage 2 received-claim boundary; intentionally no implementation in M1."""

from __future__ import annotations

from typing import Protocol

from lidar_shield.contracts.schemas import FrameRecord, ObjectRecord


class DetectorClaimSource(Protocol):
    """Future detector/received-claim implementations must satisfy this API."""

    oracle_gt: bool

    def objects_for_frame(self, frame: FrameRecord) -> tuple[ObjectRecord, ...]:
        """Return sender proposals without using GT for proposal association."""
