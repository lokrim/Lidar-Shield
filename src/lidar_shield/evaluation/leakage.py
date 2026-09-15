"""Assertions that keep all derivatives of a source episode in one partition."""

from __future__ import annotations

from dataclasses import dataclass

from lidar_shield.evaluation.splits import SplitError, SplitRegistry, SplitRole


@dataclass(frozen=True)
class ProvenanceRecord:
    sequence_id: str
    session_id: str
    source_episode_id: str
    neighboring_frame_block_id: str
    clean_counterpart_id: str
    variant_id: str
    role: SplitRole
    agent_id: str
    object_id_or_local_index: str | int | None


def assert_no_source_leakage(
    records: tuple[ProvenanceRecord, ...], registry: SplitRegistry
) -> None:
    """Require sequence, episode, neighbor-block, and clean lineage role unity."""

    role_by_group: dict[tuple[str, str], SplitRole] = {}
    for record in records:
        assignment = registry.assignment_for(record.sequence_id)
        if record.role is not assignment.role:
            raise SplitError(
                f"record role {record.role.value} conflicts with source role "
                f"{assignment.role.value} for {record.sequence_id}"
            )
        groups = (
            ("sequence_session", f"{record.sequence_id}\0{record.session_id}"),
            ("source_episode", record.source_episode_id),
            ("neighboring_frame_block", record.neighboring_frame_block_id),
            ("clean_counterpart", record.clean_counterpart_id),
        )
        for group in groups:
            existing = role_by_group.setdefault(group, record.role)
            if existing is not record.role:
                raise SplitError(
                    f"leakage: {group[0]} {group[1]!r} spans "
                    f"{existing.value} and {record.role.value}"
                )


def frame_local_object_group(
    sequence_id: str,
    session_id: str,
    sync_frame_id: int,
    object_id_or_local_index: str | int,
) -> str:
    """Build a frame-scoped key; this deliberately is not a track identity."""

    return (
        f"{sequence_id}\0{session_id}\0frame={sync_frame_id}\0"
        f"local-object={object_id_or_local_index}"
    )
