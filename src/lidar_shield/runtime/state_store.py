"""Bounded in-memory state keyed by sequence, session, and agent."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

StateValue = TypeVar("StateValue")


@dataclass(frozen=True, order=True)
class StateKey:
    sequence_id: str
    session_id: str
    agent_id: str

    def __post_init__(self) -> None:
        if not self.sequence_id or not self.session_id or not self.agent_id:
            raise ValueError("state key fields must be non-empty")


class BoundedStateStore(Generic[StateValue]):
    """Small LRU store with exact-key and scoped reset operations."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("state store capacity must be positive")
        self.max_entries = max_entries
        self._values: OrderedDict[StateKey, StateValue] = OrderedDict()

    def get(self, key: StateKey) -> StateValue | None:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def set(self, key: StateKey, value: StateValue) -> StateKey | None:
        self._values[key] = value
        self._values.move_to_end(key)
        if len(self._values) <= self.max_entries:
            return None
        evicted, _ = self._values.popitem(last=False)
        return evicted

    def reset(
        self,
        *,
        sequence_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Remove matching state; with no filters, reset the entire store."""

        selected = [
            key
            for key in self._values
            if (sequence_id is None or key.sequence_id == sequence_id)
            and (session_id is None or key.session_id == session_id)
            and (agent_id is None or key.agent_id == agent_id)
        ]
        for key in selected:
            del self._values[key]
        return len(selected)

    def __len__(self) -> int:
        return len(self._values)
