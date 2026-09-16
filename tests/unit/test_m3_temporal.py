import pytest

from lidar_shield.runtime.state_store import BoundedStateStore, StateKey
from lidar_shield.temporal.ewma import EWMAConfig, update_ewma
from lidar_shield.temporal.state_machine import label_baseline_state


def test_ewma_equation_initialization_and_threshold_metadata() -> None:
    config = EWMAConfig(alpha=0.3, threshold=0.25)
    first = update_ewma(None, 0.0, config)
    second = update_ewma(first.value, 1.0, config)
    assert first.value == 0.0
    assert first.reason == "initialized_from_first_available"
    assert first.initialized_now
    assert second.value == pytest.approx(0.3)
    assert second.threshold_exceeded is True
    assert config.initialization == "first_available"

    configured = EWMAConfig(alpha=0.5, threshold=0.8, initial_value=0.4)
    initialized = update_ewma(None, 0.8, configured)
    assert initialized.value == pytest.approx(0.6)
    assert initialized.reason == "initialized_from_configured_value"
    assert configured.initialization == "configured_value"


def test_missing_evidence_is_marked_unknown_without_erasing_prior() -> None:
    no_prior = update_ewma(None, None)
    held = update_ewma(0.7, None)
    assert no_prior.value is None
    assert no_prior.reason == "missing_evidence_no_prior"
    assert held.value == 0.7
    assert held.observation_available is False
    assert held.threshold_exceeded is None
    assert "marked_unknown" in held.reason
    state = label_baseline_state(held, "quarantined")
    assert state.current_state == "unknown"
    assert state.full_hysteretic_controller_deferred


def test_state_labels_expose_threshold_transitions() -> None:
    normal = label_baseline_state(update_ewma(None, 0.1), "unknown")
    gated = label_baseline_state(update_ewma(0.1, 1.0), "normal")
    remained = label_baseline_state(update_ewma(0.8, 0.8), "quarantined")
    assert normal.current_state == "normal"
    assert normal.transition_reason.startswith("transitioned")
    assert gated.current_state == "quarantined"
    assert remained.transition_reason.startswith("remained")


def test_ewma_validation() -> None:
    with pytest.raises(ValueError):
        EWMAConfig(alpha=0.0)
    with pytest.raises(ValueError):
        EWMAConfig(alpha=float("nan"))
    with pytest.raises(ValueError):
        EWMAConfig(threshold=1.1)
    with pytest.raises(ValueError):
        EWMAConfig(initial_value=-0.1)
    for previous, observation in ((-1.0, 0.0), (0.0, float("inf"))):
        with pytest.raises(ValueError):
            update_ewma(previous, observation)


def test_bounded_state_store_isolates_keys_evicts_and_resets_scopes() -> None:
    store: BoundedStateStore[int] = BoundedStateStore(max_entries=2)
    a = StateKey("sequence-a", "session-a", "agent")
    b = StateKey("sequence-a", "session-b", "agent")
    c = StateKey("sequence-b", "session-a", "agent")
    assert store.set(a, 1) is None
    assert store.set(b, 2) is None
    assert store.get(a) == 1
    assert store.set(c, 3) == b
    assert store.get(b) is None
    assert len(store) == 2
    assert store.reset(session_id="session-a", agent_id="agent") == 2
    assert len(store) == 0
    assert store.reset() == 0


def test_state_store_rejects_invalid_capacity_and_keys() -> None:
    with pytest.raises(ValueError):
        BoundedStateStore[float](0)
    with pytest.raises(ValueError):
        StateKey("", "session", "agent")
