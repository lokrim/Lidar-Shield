"""Stable availability, matching, and missingness reason codes.

Signed age always means ``decision_time_ns - matched_source_time_ns``. Positive
values are past samples, zero is an exact-time sample, and negative values are
future samples admitted by a declared buffered policy.
"""

from __future__ import annotations

from enum import Enum


class AgentType(str, Enum):
    """Configured agent categories; names never imply capabilities."""

    VEHICLE = "vehicle"
    INFRASTRUCTURE = "infrastructure"


class Applicability(str, Enum):
    """Whether a configured modality applies to an agent."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_APPLICABLE = "not_applicable"


class MatchPolicy(str, Enum):
    """Declared source-time selection policy."""

    CAUSAL_BACKWARD = "causal_backward"
    BUFFERED_NEAREST = "buffered_nearest"
    NONE = "none"


class ReasonCode(str, Enum):
    """Why a frame/evidence row is available or unavailable."""

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    LATE = "late"
    MISSING_MESSAGE = "missing_message"
    EMPTY_SCENE = "empty_scene"
    INVALID_POSE = "invalid_pose"
    FUTURE_POSE_UNAVAILABLE = "future_pose_unavailable"
    OUT_OF_RANGE = "out_of_range"
    STALE = "stale"
    NOT_MEMBER = "not_member"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_PEERS = "insufficient_peers"


UNAVAILABLE_SOURCE_REASONS = frozenset(
    {
        ReasonCode.UNMATCHED,
        ReasonCode.LATE,
        ReasonCode.MISSING_MESSAGE,
        ReasonCode.OUT_OF_RANGE,
        ReasonCode.STALE,
        ReasonCode.NOT_MEMBER,
    }
)

MATCHED_SOURCE_REASONS = frozenset(
    {ReasonCode.MATCHED, ReasonCode.EMPTY_SCENE, ReasonCode.INVALID_POSE}
)
