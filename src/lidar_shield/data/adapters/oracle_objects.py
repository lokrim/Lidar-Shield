"""Stage 1 oracle GT-indexed object source and Mixed Signals TXT parser."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from lidar_shield.contracts.schemas import FrameRecord, ObjectRecord


class OracleLabelError(ValueError):
    """A label file violates the eight-field oracle-object contract."""


@dataclass(frozen=True)
class OracleBoxLabel:
    """One frame-local GT box expressed in the dataset's common top frame."""

    local_index: int
    class_id: int
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    yaw_rad: float

    def __post_init__(self) -> None:
        if self.local_index < 0:
            raise OracleLabelError("local object index must be nonnegative")
        if self.class_id not in {1, 2, 3}:
            raise OracleLabelError("Mixed Signals class must be 1, 2, or 3")
        values = self.center_m + self.size_m + (self.yaw_rad,)
        if not all(math.isfinite(value) for value in values):
            raise OracleLabelError("label values must be finite")
        if any(value <= 0 for value in self.size_m):
            raise OracleLabelError("box sizes must be positive")


def parse_oracle_labels(path: str | Path) -> tuple[OracleBoxLabel, ...]:
    """Parse a TXT label file; a zero-byte/whitespace-only file is valid."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OracleLabelError(f"cannot read label file {source}: {exc}") from exc
    labels: list[OracleBoxLabel] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 8:
            raise OracleLabelError(
                f"expected eight fields at {source}:{line_number}, got {len(fields)}"
            )
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise OracleLabelError(
                f"invalid numeric label at {source}:{line_number}"
            ) from exc
        labels.append(
            OracleBoxLabel(
                local_index=len(labels),
                class_id=class_id,
                center_m=(values[0], values[1], values[2]),
                size_m=(values[3], values[4], values[5]),
                yaw_rad=values[6],
            )
        )
    return tuple(labels)


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
