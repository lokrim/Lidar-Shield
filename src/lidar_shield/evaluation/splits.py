"""Frozen source-role registry established before scientific attack APIs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lidar_shield.manifest import sha256_file


class SplitError(ValueError):
    """A split registry or attempted use violates the frozen-role contract."""


class SplitRole(str, Enum):
    REGRESSION_DEVELOPMENT = "regression_development"
    TRAINING = "training"
    CALIBRATION = "calibration"
    FINAL_TEST = "final_test"


class ScientificAction(str, Enum):
    ATTACK_GENERATION = "attack_generation"
    NORMALIZER_FIT = "normalizer_fit"
    SUPERVISED_FIT = "supervised_fit"
    CALIBRATION = "calibration"
    FINAL_EVALUATION = "final_evaluation"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SourceAssignment(_StrictModel):
    source_id: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    role: SplitRole
    resolved: bool
    frozen: bool

    @model_validator(mode="after")
    def assigned_sources_are_resolved_and_frozen(self) -> SourceAssignment:
        if not self.resolved:
            raise ValueError("assigned source rows must be resolved")
        if not self.frozen:
            raise ValueError("assigned source roles must be frozen")
        return self


class UnresolvedSlots(_StrictModel):
    training: int = Field(ge=0)
    calibration: int = Field(ge=0)
    final_test: int = Field(ge=0)


class SplitRegistry(_StrictModel):
    schema_version: str = Field(pattern=r"^1\.[0-9]+\.[0-9]+$")
    registry_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    assignments: tuple[SourceAssignment, ...] = Field(strict=False)
    unresolved_slots: UnresolvedSlots

    @model_validator(mode="after")
    def validate_roles(self) -> SplitRegistry:
        source_ids = [item.source_id for item in self.assignments]
        sequence_ids = [item.sequence_id for item in self.assignments]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must have exactly one split role")
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("sequence IDs must have exactly one split role")
        regression = [
            item
            for item in self.assignments
            if item.role is SplitRole.REGRESSION_DEVELOPMENT
        ]
        if len(regression) != 1 or regression[0].sequence_id != "mini_7":
            raise ValueError("mini_7 must be the sole regression/development source")
        return self

    def assignment_for(self, sequence_id: str) -> SourceAssignment:
        for assignment in self.assignments:
            if assignment.sequence_id == sequence_id:
                return assignment
        raise SplitError(f"source sequence {sequence_id!r} is unresolved/unassigned")

    def assert_allowed(self, sequence_id: str, action: ScientificAction) -> None:
        assignment = self.assignment_for(sequence_id)
        allowed: dict[ScientificAction, frozenset[SplitRole]] = {
            ScientificAction.ATTACK_GENERATION: frozenset(
                {SplitRole.TRAINING, SplitRole.CALIBRATION, SplitRole.FINAL_TEST}
            ),
            ScientificAction.NORMALIZER_FIT: frozenset({SplitRole.TRAINING}),
            ScientificAction.SUPERVISED_FIT: frozenset({SplitRole.TRAINING}),
            ScientificAction.CALIBRATION: frozenset({SplitRole.CALIBRATION}),
            ScientificAction.FINAL_EVALUATION: frozenset({SplitRole.FINAL_TEST}),
        }
        if not assignment.resolved or not assignment.frozen:
            raise SplitError(f"source sequence {sequence_id!r} is not frozen")
        if assignment.role not in allowed[action]:
            raise SplitError(
                f"{action.value} is not allowed for {assignment.role.value} source "
                f"{sequence_id!r}"
            )

    def assert_variant_role(self, sequence_id: str, role: SplitRole) -> None:
        assignment = self.assignment_for(sequence_id)
        if not assignment.frozen:
            raise SplitError("source role must be frozen before variant generation")
        if assignment.role is not role:
            raise SplitError(
                f"variant role {role.value} conflicts with frozen source role "
                f"{assignment.role.value}"
            )


def load_split_registry(path: str | Path) -> tuple[SplitRegistry, str]:
    registry_path = Path(path).expanduser().resolve()
    try:
        raw: Any = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        registry = SplitRegistry.model_validate(raw, strict=False)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise SplitError(f"invalid split registry {registry_path}: {exc}") from exc
    return registry, sha256_file(registry_path)
