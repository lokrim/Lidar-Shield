"""Configured agent registry and explicit per-frame membership indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lidar_shield.contracts.schemas import AgentRegistry


class IndexError(ValueError):
    """Agent registry or frame membership is invalid."""


class MembershipDeclaration(BaseModel):
    """One explicit agent membership decision for one frame."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequence_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sync_frame_id: int = Field(ge=0)
    decision_time_ns: int = Field(ge=0)
    agent_id: str = Field(min_length=1)
    member: bool
    message_expected: bool

    @model_validator(mode="after")
    def nonmembers_do_not_expect_messages(self) -> MembershipDeclaration:
        if not self.member and self.message_expected:
            raise ValueError("a non-member cannot have an expected message")
        return self


def load_agent_registry(path: str | Path) -> AgentRegistry:
    registry_path = Path(path).expanduser().resolve()
    try:
        raw: Any = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        return AgentRegistry.model_validate(raw, strict=False)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise IndexError(f"invalid agent registry {registry_path}: {exc}") from exc


def validate_membership(
    declarations: tuple[MembershipDeclaration, ...], registry: AgentRegistry
) -> tuple[MembershipDeclaration, ...]:
    """Validate and stably order explicit frame membership rows."""

    known_agents = {agent.agent_id for agent in registry.agents}
    keys: set[tuple[str, str, int, str]] = set()
    for row in declarations:
        if row.agent_id not in known_agents:
            raise IndexError(f"membership names unknown agent {row.agent_id!r}")
        key = (row.sequence_id, row.session_id, row.sync_frame_id, row.agent_id)
        if key in keys:
            raise IndexError(f"duplicate membership key: {key}")
        keys.add(key)
    return tuple(
        sorted(
            declarations,
            key=lambda row: (
                row.sequence_id,
                row.session_id,
                row.sync_frame_id,
                row.agent_id,
            ),
        )
    )
