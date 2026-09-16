from pathlib import Path

import pytest
from pydantic import ValidationError

from lidar_shield.contracts.schemas import (
    EvaluationRunMetrics,
    FusionResult,
    PredictionRecord,
    ScheduleRecord,
    StateTransitionRecord,
)
from lidar_shield.evaluation.metrics import evaluate_integration_frame
from lidar_shield.runtime.pipeline import (
    IntegrationRun,
    load_integration_fixture,
    run_integration_fixture,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "m3" / "integration_fixture.json"


def run() -> IntegrationRun:
    return run_integration_fixture(
        load_integration_fixture(FIXTURE), "integration_corruption"
    )


def test_prediction_contract_keeps_risk_uncertainty_and_abstention_separate() -> None:
    raw = run().predictions[0].model_dump()
    for update, message in (
        ({"operational_reliability": 0.5}, "one minus risk"),
        ({"uncertainty_available": True}, "uncertainty availability"),
        ({"abstained": True}, "abstention"),
        ({"calibrated_p_attack": 0.2}, "cannot expose"),
    ):
        with pytest.raises(ValidationError, match=message):
            PredictionRecord.model_validate({**raw, **update})

    calibrated = {
        **raw,
        "risk_score": 0.2,
        "operational_reliability": 0.8,
        "risk_semantics": "calibrated_attack_probability",
        "calibration_status": "calibrated",
        "calibrated_p_attack": 0.2,
    }
    assert PredictionRecord.model_validate(calibrated).calibrated_p_attack == 0.2
    with pytest.raises(ValidationError, match="probability semantics"):
        PredictionRecord.model_validate({**calibrated, "calibrated_p_attack": None})
    with pytest.raises(ValidationError, match="must equal"):
        PredictionRecord.model_validate({**calibrated, "calibrated_p_attack": 0.3})


def test_state_schedule_and_fusion_consistency_guards() -> None:
    result = run()
    state = result.states[0].model_dump()
    with pytest.raises(ValidationError, match="input availability"):
        StateTransitionRecord.model_validate({**state, "input_available": False})

    schedule = result.schedules[0].model_dump()
    with pytest.raises(ValidationError, match="admitted bytes"):
        ScheduleRecord.model_validate({**schedule, "admitted_bytes": 1})
    with pytest.raises(ValidationError, match="source_time"):
        ScheduleRecord.model_validate({**schedule, "source_time_ns": None})

    fusion = result.fusion[0].model_dump()
    for update, message in (
        ({"abstained": True}, "abstention"),
        ({"contributor_count": 99}, "contributor count"),
        ({"remote_contributor_count": 99}, "remote contributor count"),
        ({"uncertainty_available": True}, "uncertainty availability"),
        ({"reason": "ego_absent"}, "ego absence"),
        ({"reason": "ego_only_fallback", "degraded": False}, "must be degraded"),
    ):
        with pytest.raises(ValidationError, match=message):
            FusionResult.model_validate({**fusion, **update})


def test_metric_contract_and_identity_checked_join_guards() -> None:
    result = run()
    metric = result.metrics[0].model_dump()
    for update, message in (
        ({"observation_count": 99}, "preserve every source"),
        ({"prediction_available_count": 0}, "cover all observations"),
        ({"corrupted_source_bytes": metric["admitted_bytes"] + 1}, "cannot exceed"),
        ({"proxy_utility": None}, "must coexist"),
    ):
        with pytest.raises(ValidationError, match=message):
            EvaluationRunMetrics.model_validate({**metric, **update})

    frame = 0
    predictions = tuple(
        item for item in result.predictions if item.sync_frame_id == frame
    )
    states = tuple(item for item in result.states if item.sync_frame_id == frame)
    schedules = tuple(
        item
        for item in result.schedules
        if item.sync_frame_id == frame and item.policy == "full_share"
    )
    fusion = next(
        item
        for item in result.fusion
        if item.sync_frame_id == frame and item.policy == "full_share"
    )
    with pytest.raises(ValueError, match="expected observation"):
        evaluate_integration_frame((), (), (), fusion, clean_reference=(1.0, 1.0))
    mismatched = states[0].model_copy(update={"fixture_run_id": "other"})
    with pytest.raises(ValueError, match="mismatched fixture"):
        evaluate_integration_frame(
            predictions,
            (mismatched, *states[1:]),
            schedules,
            fusion,
            clean_reference=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="unique by sender"):
        evaluate_integration_frame(
            (*predictions, predictions[0]),
            states,
            schedules,
            fusion,
            clean_reference=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="requires a state row"):
        evaluate_integration_frame(
            predictions,
            states[:-1],
            schedules,
            fusion,
            clean_reference=(1.0, 1.0),
        )
