"""Unknown-aware implementation of the documented EWMA baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EWMAConfig:
    alpha: float = 0.30
    threshold: float = 0.25
    initial_value: float | None = None
    missing_evidence_policy: Literal["hold_prior_mark_unknown"] = (
        "hold_prior_mark_unknown"
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("EWMA alpha must lie in (0, 1]")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("EWMA threshold must lie in [0, 1]")
        if self.initial_value is not None and (
            not math.isfinite(self.initial_value) or not 0 <= self.initial_value <= 1
        ):
            raise ValueError("EWMA initial value must lie in [0, 1]")

    @property
    def initialization(self) -> Literal["first_available", "configured_value"]:
        return "first_available" if self.initial_value is None else "configured_value"


@dataclass(frozen=True)
class EWMAUpdate:
    value: float | None
    observation_available: bool
    initialized_now: bool
    threshold_exceeded: bool | None
    reason: Literal[
        "initialized_from_first_available",
        "initialized_from_configured_value",
        "updated",
        "missing_evidence_no_prior",
        "missing_evidence_prior_held_marked_unknown",
    ]


DEFAULT_EWMA_CONFIG = EWMAConfig()


def update_ewma(
    previous: float | None,
    observation: float | None,
    config: EWMAConfig = DEFAULT_EWMA_CONFIG,
) -> EWMAUpdate:
    """Apply ``E_t = alpha*T_t + (1-alpha)*E_previous`` once.

    A missing observation never becomes a benign value. An existing numeric
    prior is retained only for the next update while this decision is marked
    unavailable.
    """

    for name, value in (("previous", previous), ("observation", observation)):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"EWMA {name} must lie in [0, 1]")
    if observation is None:
        if previous is None:
            return EWMAUpdate(previous, False, False, None, "missing_evidence_no_prior")
        return EWMAUpdate(
            previous,
            False,
            False,
            None,
            "missing_evidence_prior_held_marked_unknown",
        )
    if previous is None:
        if config.initial_value is None:
            value = observation
            return EWMAUpdate(
                value,
                True,
                True,
                value >= config.threshold,
                "initialized_from_first_available",
            )
        else:
            value = (
                config.alpha * observation + (1.0 - config.alpha) * config.initial_value
            )
            return EWMAUpdate(
                value,
                True,
                True,
                value >= config.threshold,
                "initialized_from_configured_value",
            )
    value = config.alpha * observation + (1.0 - config.alpha) * previous
    return EWMAUpdate(value, True, False, value >= config.threshold, "updated")
