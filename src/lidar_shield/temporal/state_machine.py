"""Threshold-only M3 state labels; calibrated hysteresis is deferred to M7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lidar_shield.temporal.ewma import EWMAUpdate

BaselineState = Literal["normal", "quarantined", "unknown"]


@dataclass(frozen=True)
class BaselineStateUpdate:
    previous_state: BaselineState
    current_state: BaselineState
    transition_reason: str
    full_hysteretic_controller_deferred: bool = True


def label_baseline_state(
    update: EWMAUpdate,
    previous_state: BaselineState = "unknown",
) -> BaselineStateUpdate:
    """Label one EWMA decision without recovery dwell time or hysteresis."""

    if not update.observation_available:
        current: BaselineState = "unknown"
    elif update.threshold_exceeded:
        current = "quarantined"
    else:
        current = "normal"
    movement = "remained" if current == previous_state else "transitioned"
    return BaselineStateUpdate(
        previous_state,
        current,
        f"{movement}:{previous_state}->{current}:{update.reason}",
    )
