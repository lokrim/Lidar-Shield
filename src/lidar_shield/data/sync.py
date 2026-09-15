"""Causal and explicitly buffered source matching without event attrition."""

from __future__ import annotations

from dataclasses import dataclass

from lidar_shield.contracts.reasons import MatchPolicy, ReasonCode
from lidar_shield.contracts.schemas import AgentRegistry, FrameRecord
from lidar_shield.data.index import MembershipDeclaration, validate_membership


@dataclass(frozen=True)
class SyncConfig:
    policy: MatchPolicy
    max_age_ns: int
    lookahead_buffer_ns: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_age_ns, bool) or not isinstance(self.max_age_ns, int):
            raise TypeError("max_age_ns must be an integer")
        if self.max_age_ns < 0:
            raise ValueError("max_age_ns must be nonnegative")
        if isinstance(self.lookahead_buffer_ns, bool) or not isinstance(
            self.lookahead_buffer_ns, int
        ):
            raise TypeError("lookahead_buffer_ns must be an integer")
        if self.policy is MatchPolicy.CAUSAL_BACKWARD:
            if self.lookahead_buffer_ns != 0:
                raise ValueError("causal matching cannot declare a look-ahead")
        elif self.policy is MatchPolicy.BUFFERED_NEAREST:
            if self.lookahead_buffer_ns <= 0:
                raise ValueError("buffered matching requires a positive look-ahead")
        else:
            raise ValueError("synchronization requires causal or buffered policy")


@dataclass(frozen=True)
class SourceMessage:
    sequence_id: str
    session_id: str
    agent_id: str
    source_time_ns: int
    arrival_time_ns: int
    empty_scene: bool = False
    pose_valid: bool = True
    pose_source_time_ns: int | None = None

    def __post_init__(self) -> None:
        for name in ("sequence_id", "session_id", "agent_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        for name in ("source_time_ns", "arrival_time_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.arrival_time_ns < self.source_time_ns:
            raise ValueError("arrival_time_ns cannot precede source_time_ns")
        if self.pose_source_time_ns is not None:
            if isinstance(self.pose_source_time_ns, bool) or not isinstance(
                self.pose_source_time_ns, int
            ):
                raise TypeError("pose_source_time_ns must be an integer")
            if self.pose_source_time_ns < 0:
                raise ValueError("pose_source_time_ns must be nonnegative")


class Synchronizer:
    """Match explicit membership events to immutable source messages."""

    def __init__(self, registry: AgentRegistry, config: SyncConfig) -> None:
        self._registry = registry
        self._config = config
        self._agents = {agent.agent_id: agent for agent in registry.agents}

    def synchronize(
        self,
        memberships: tuple[MembershipDeclaration, ...],
        messages: tuple[SourceMessage, ...],
        *,
        experiment_id: str,
        variant_id: str,
        oracle_gt: bool,
    ) -> tuple[FrameRecord, ...]:
        rows = validate_membership(memberships, self._registry)
        message_keys: set[tuple[str, str, str, int]] = set()
        grouped: dict[tuple[str, str, str], list[SourceMessage]] = {}
        for message in messages:
            if message.agent_id not in self._agents:
                raise ValueError(f"message names unknown agent {message.agent_id!r}")
            identity = (
                message.sequence_id,
                message.session_id,
                message.agent_id,
                message.source_time_ns,
            )
            if identity in message_keys:
                raise ValueError(f"duplicate source message: {identity}")
            message_keys.add(identity)
            grouped.setdefault(identity[:3], []).append(message)
        for values in grouped.values():
            values.sort(key=lambda item: (item.source_time_ns, item.arrival_time_ns))

        return tuple(
            self._row_for_membership(
                membership,
                grouped.get(
                    (
                        membership.sequence_id,
                        membership.session_id,
                        membership.agent_id,
                    ),
                    [],
                ),
                experiment_id=experiment_id,
                variant_id=variant_id,
                oracle_gt=oracle_gt,
            )
            for membership in rows
        )

    def _row_for_membership(
        self,
        membership: MembershipDeclaration,
        messages: list[SourceMessage],
        *,
        experiment_id: str,
        variant_id: str,
        oracle_gt: bool,
    ) -> FrameRecord:
        if not membership.member:
            return self._unmatched_row(
                membership,
                experiment_id,
                variant_id,
                oracle_gt,
                ReasonCode.NOT_MEMBER,
                message_present=False,
            )
        if not membership.message_expected or not messages:
            return self._unmatched_row(
                membership,
                experiment_id,
                variant_id,
                oracle_gt,
                ReasonCode.MISSING_MESSAGE,
                message_present=False,
            )

        decision = membership.decision_time_ns
        arrival_limit = decision + self._config.lookahead_buffer_ns
        if self._config.policy is MatchPolicy.CAUSAL_BACKWARD:
            available = [
                item
                for item in messages
                if item.source_time_ns <= decision and item.arrival_time_ns <= decision
            ]
            candidate = max(
                available, key=lambda item: item.source_time_ns, default=None
            )
        else:
            available = [
                item
                for item in messages
                if item.source_time_ns <= arrival_limit
                and item.arrival_time_ns <= arrival_limit
            ]
            candidate = min(
                available,
                key=lambda item: (
                    abs(decision - item.source_time_ns),
                    item.source_time_ns > decision,
                    -item.source_time_ns,
                ),
                default=None,
            )

        if candidate is not None:
            age = decision - candidate.source_time_ns
            if abs(age) <= self._config.max_age_ns:
                return self._matched_row(
                    membership,
                    candidate,
                    experiment_id,
                    variant_id,
                    oracle_gt,
                )
            reason = ReasonCode.STALE if age > 0 else ReasonCode.OUT_OF_RANGE
            return self._unmatched_row(
                membership,
                experiment_id,
                variant_id,
                oracle_gt,
                reason,
                message_present=True,
            )

        time_eligible = [
            item
            for item in messages
            if (
                item.source_time_ns <= decision
                if self._config.policy is MatchPolicy.CAUSAL_BACKWARD
                else item.source_time_ns <= arrival_limit
            )
            and abs(decision - item.source_time_ns) <= self._config.max_age_ns
        ]
        if any(item.arrival_time_ns > arrival_limit for item in time_eligible):
            reason = ReasonCode.LATE
        elif all(item.source_time_ns > arrival_limit for item in messages):
            reason = ReasonCode.UNMATCHED
        else:
            past = [item for item in messages if item.source_time_ns < decision]
            reason = ReasonCode.STALE if past else ReasonCode.OUT_OF_RANGE
        return self._unmatched_row(
            membership,
            experiment_id,
            variant_id,
            oracle_gt,
            reason,
            message_present=True,
        )

    def _matched_row(
        self,
        membership: MembershipDeclaration,
        message: SourceMessage,
        experiment_id: str,
        variant_id: str,
        oracle_gt: bool,
    ) -> FrameRecord:
        if message.empty_scene:
            reason = ReasonCode.EMPTY_SCENE
        elif not message.pose_valid:
            reason = ReasonCode.INVALID_POSE
        else:
            reason = ReasonCode.MATCHED
        pose_time = message.pose_source_time_ns if message.pose_valid else None
        return FrameRecord(
            experiment_id=experiment_id,
            variant_id=variant_id,
            sequence_id=membership.sequence_id,
            session_id=membership.session_id,
            sync_frame_id=membership.sync_frame_id,
            decision_time_ns=membership.decision_time_ns,
            source_time_ns=message.source_time_ns,
            agent_id=membership.agent_id,
            agent_type=self._agents[membership.agent_id].agent_type,
            member=True,
            message_present=True,
            source_matched=True,
            evidence_available=reason is ReasonCode.MATCHED,
            reason_code=reason,
            match_policy=self._config.policy,
            match_age_ns=membership.decision_time_ns - message.source_time_ns,
            lookahead_buffer_ns=self._config.lookahead_buffer_ns,
            pose_source_time_ns=pose_time,
            pose_age_ns=(
                membership.decision_time_ns - pose_time
                if pose_time is not None
                else None
            ),
            oracle_gt=oracle_gt,
        )

    def _unmatched_row(
        self,
        membership: MembershipDeclaration,
        experiment_id: str,
        variant_id: str,
        oracle_gt: bool,
        reason: ReasonCode,
        *,
        message_present: bool,
    ) -> FrameRecord:
        return FrameRecord(
            experiment_id=experiment_id,
            variant_id=variant_id,
            sequence_id=membership.sequence_id,
            session_id=membership.session_id,
            sync_frame_id=membership.sync_frame_id,
            decision_time_ns=membership.decision_time_ns,
            source_time_ns=None,
            agent_id=membership.agent_id,
            agent_type=self._agents[membership.agent_id].agent_type,
            member=membership.member,
            message_present=message_present,
            source_matched=False,
            evidence_available=False,
            reason_code=reason,
            match_policy=self._config.policy,
            match_age_ns=None,
            lookahead_buffer_ns=self._config.lookahead_buffer_ns,
            pose_source_time_ns=None,
            pose_age_ns=None,
            oracle_gt=oracle_gt,
        )
