"""Stage 1 oracle GT-indexed object source for controlled measurement regions."""

from __future__ import annotations

from dataclasses import dataclass, field

from lidar_shield.contracts.schemas import FrameRecord, ObjectRecord


@dataclass(frozen=True)
class InMemoryOracleObjectSource:
    """Small contract implementation for synthetic fixtures only."""

    rows: tuple[ObjectRecord, ...]
    oracle_gt: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if any(not row.oracle_gt for row in self.rows):
            raise ValueError("oracle object source accepts only oracle_gt rows")

    def objects_for_frame(self, frame: FrameRecord) -> tuple[ObjectRecord, ...]:
        if not frame.source_matched:
            return ()
        key = (
            frame.experiment_id,
            frame.variant_id,
            frame.sequence_id,
            frame.session_id,
            frame.sync_frame_id,
            frame.agent_id,
        )
        return tuple(
            row
            for row in self.rows
            if (
                row.experiment_id,
                row.variant_id,
                row.sequence_id,
                row.session_id,
                row.sync_frame_id,
                row.agent_id,
            )
            == key
        )
