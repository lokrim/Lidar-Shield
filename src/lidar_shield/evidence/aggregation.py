"""Object-preserving evidence summaries for frame-level consumers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class FrameEvidenceSummary:
    object_count: int
    available_object_count: int
    unavailable_object_count: int
    maximum_suspicion: float | None
    median_suspicion: float | None
    any_unavailable: bool


def summarize_object_suspicion(
    values: tuple[float | None, ...],
) -> FrameEvidenceSummary:
    """Retain max and median; never collapse a single-box anomaly into a mean."""

    available = [float(value) for value in values if value is not None]
    if any(value < 0 for value in available):
        raise ValueError("suspicion values must be nonnegative")
    return FrameEvidenceSummary(
        object_count=len(values),
        available_object_count=len(available),
        unavailable_object_count=len(values) - len(available),
        maximum_suspicion=max(available) if available else None,
        median_suspicion=float(median(available)) if available else None,
        any_unavailable=len(available) != len(values),
    )
