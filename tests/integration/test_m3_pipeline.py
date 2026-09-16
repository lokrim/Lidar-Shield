from __future__ import annotations

import json
from pathlib import Path

import pytest

from lidar_shield.communication.byte_cost import measured_packet_size
from lidar_shield.runtime.pipeline import (
    IntegrationFixtureError,
    build_d2_payload,
    load_integration_fixture,
    run_integration_fixture,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "m3" / "integration_fixture.json"


def test_d2_clean_corrupted_chain_matches_analytical_expectations() -> None:
    fixture = load_integration_fixture(FIXTURE)
    payload = build_d2_payload(FIXTURE)
    expected = fixture.analytical_expectations
    clean = payload["summaries"]["clean"]
    corrupted = payload["summaries"]["integration_corruption"]

    assert payload["fixture_run_id"] == fixture.fixture_run_id
    assert payload["scientific_eligibility"] == "none"
    assert clean["target_uncalibrated_risk"] == expected.target_clean_risk
    assert corrupted["target_uncalibrated_risk"] == expected.target_corrupted_risk
    assert corrupted["target_ewma_risk"] == pytest.approx(
        expected.target_corrupted_ewma
    )
    assert corrupted["target_state"] == expected.target_corrupted_state
    assert corrupted["target_calibrated_p_attack"] is None
    assert corrupted["target_uncertainty_available"] is False
    assert corrupted["policies"]["full_share"]["corrupted_source_bytes"] > 0
    assert corrupted["policies"]["hard_gate"]["corrupted_source_bytes"] == (
        expected.hard_gate_corrupted_source_bytes
    )
    assert corrupted["policies"]["full_share"]["proxy_reconstruction_error"] > 0
    assert corrupted["policies"]["hard_gate"]["proxy_reconstruction_error"] == 0


def test_every_record_keeps_fixture_source_sequence_and_session_identity() -> None:
    fixture = load_integration_fixture(FIXTURE)
    for case in ("clean", "integration_corruption"):
        run = run_integration_fixture(fixture, case)
        collections = (
            run.predictions,
            run.states,
            (packet.descriptor for packet in run.packets),
            run.schedules,
            run.fusion,
            run.metrics,
        )
        for records in collections:
            for record in records:
                assert record.fixture_run_id == fixture.fixture_run_id
                assert record.fixture_case == case
                assert record.source_id == fixture.source_id
                assert record.sequence_id == fixture.sequence_id
                assert record.session_id == fixture.session_id


def test_missing_and_companion_rows_survive_metrics_join() -> None:
    fixture = load_integration_fixture(FIXTURE)
    run = run_integration_fixture(fixture, "integration_corruption")
    frame_two = [item for item in run.metrics if item.sync_frame_id == 2]
    assert len(frame_two) == 2
    for metric in frame_two:
        assert metric.observation_count == 3
        assert metric.missing_observation_count == 2
        assert metric.abstention_count == 2
        assert {item.agent_id for item in metric.observation_sources} == {
            "ego",
            "remote-companion",
            "remote-corrupted",
        }


def test_policies_consume_identical_packets_and_exact_costs() -> None:
    fixture = load_integration_fixture(FIXTURE)
    run = run_integration_fixture(fixture, "integration_corruption")
    for packet in run.packets:
        decisions = [
            item for item in run.schedules if item.packet_sha256 == packet.packet_sha256
        ]
        assert {item.policy for item in decisions} == {"full_share", "hard_gate"}
        assert all(
            item.measured_size_bytes == measured_packet_size(packet)
            for item in decisions
        )
        assert len({item.max_source_age_ns for item in decisions}) == 1


def test_exact_ego_fallback_and_ego_absent_abstention() -> None:
    fixture = load_integration_fixture(FIXTURE)
    run = run_integration_fixture(fixture, "integration_corruption")
    fallbacks = [item for item in run.fusion if item.sync_frame_id == 2]
    assert len(fallbacks) == 2
    for result in fallbacks:
        assert result.fused_vector == (0.5, -0.5)
        assert result.degraded
        assert result.remote_contributor_count == 0
        assert result.reason == "ego_only_fallback"
    abstentions = [item for item in run.fusion if item.sync_frame_id == 3]
    assert all(item.abstained and item.reason == "ego_absent" for item in abstentions)


def test_d2_is_stable_and_contains_no_scientific_episode_or_variant_identity() -> None:
    first = build_d2_payload(FIXTURE)
    second = build_d2_payload(FIXTURE)
    assert first == second
    serialized = json.dumps(first, sort_keys=True)
    assert "episode_id" not in serialized
    assert "variant_id" not in serialized
    assert "attack_manifest" not in serialized
    assert "coverage_denominator" not in serialized


def test_clean_reference_is_evaluation_only_not_runtime_input() -> None:
    fixture = load_integration_fixture(FIXTURE)
    changed = fixture.model_copy(
        update={
            "frames": tuple(
                frame.model_copy(update={"clean_reference": (99.0, 99.0)})
                for frame in fixture.frames
            )
        }
    )
    original_run = run_integration_fixture(fixture, "clean")
    changed_run = run_integration_fixture(changed, "clean")
    assert original_run.predictions == changed_run.predictions
    assert original_run.states == changed_run.states
    assert original_run.packets == changed_run.packets
    assert original_run.schedules == changed_run.schedules
    assert original_run.fusion == changed_run.fusion
    assert original_run.metrics != changed_run.metrics


def test_fixture_loader_rejects_malformed_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(IntegrationFixtureError):
        load_integration_fixture(missing)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(IntegrationFixtureError):
        load_integration_fixture(invalid)
