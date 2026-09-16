"""Framework-neutral aligned-vector interfaces for Stage 1 proxy fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

FixtureCase = Literal["clean", "integration_corruption"]


@dataclass(frozen=True)
class FusionFrameKey:
    fixture_run_id: str
    fixture_case: FixtureCase
    source_id: str
    sequence_id: str
    session_id: str
    sync_frame_id: int
    decision_time_ns: int


@dataclass(frozen=True)
class AlignedContribution:
    agent_id: str
    source_time_ns: int
    vector: tuple[float, ...]
    is_ego: bool
    valid: bool = True
    aligned: bool = True
    quality_score: float = 1.0
    predictive_uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not self.agent_id or not self.vector:
            raise ValueError("fusion contributors require an agent and vector")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("fusion vectors must be finite")
        if not math.isfinite(self.quality_score) or not 0 <= self.quality_score <= 1:
            raise ValueError("fusion quality must lie in [0, 1]")
        if self.predictive_uncertainty is not None and (
            not math.isfinite(self.predictive_uncertainty)
            or self.predictive_uncertainty < 0
        ):
            raise ValueError("fusion uncertainty must be finite and nonnegative")
