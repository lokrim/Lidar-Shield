from pathlib import Path

import pytest
from pydantic import ValidationError

from lidar_shield.evaluation.leakage import (
    ProvenanceRecord,
    assert_no_source_leakage,
    frame_local_object_group,
)
from lidar_shield.evaluation.splits import (
    ScientificAction,
    SourceAssignment,
    SplitError,
    SplitRegistry,
    SplitRole,
    UnresolvedSlots,
    load_split_registry,
)

ROOT = Path(__file__).parents[2]
SPLITS = ROOT / "configs" / "data" / "stage1_split.yaml"


def full_registry() -> SplitRegistry:
    return SplitRegistry(
        schema_version="1.0.0",
        registry_id="test",
        status="frozen test registry",
        assignments=(
            SourceAssignment(
                source_id="mini",
                sequence_id="mini_7",
                role=SplitRole.REGRESSION_DEVELOPMENT,
                resolved=True,
                frozen=True,
            ),
            SourceAssignment(
                source_id="train",
                sequence_id="train-a",
                role=SplitRole.TRAINING,
                resolved=True,
                frozen=True,
            ),
            SourceAssignment(
                source_id="test",
                sequence_id="test-a",
                role=SplitRole.FINAL_TEST,
                resolved=True,
                frozen=True,
            ),
        ),
        unresolved_slots=UnresolvedSlots(training=3, calibration=1, final_test=1),
    )


def provenance(sequence: str, role: SplitRole, variant: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        sequence_id=sequence,
        session_id=f"session-{sequence}",
        source_episode_id=f"episode-{sequence}",
        neighboring_frame_block_id=f"block-{sequence}",
        clean_counterpart_id=f"clean-{sequence}",
        variant_id=variant,
        role=role,
        agent_id="003",
        object_id_or_local_index=0,
    )


def test_split_skeleton_registers_only_mini_7_and_hashes_exact_bytes() -> None:
    registry, digest = load_split_registry(SPLITS)
    assert registry.assignment_for("mini_7").role is SplitRole.REGRESSION_DEVELOPMENT
    assert registry.unresolved_slots.model_dump() == {
        "training": 4,
        "calibration": 1,
        "final_test": 2,
    }
    assert len(digest) == 64
    with pytest.raises(SplitError, match="unresolved"):
        registry.assignment_for("not-selected")
    with pytest.raises(SplitError, match="not allowed"):
        registry.assert_allowed("mini_7", ScientificAction.ATTACK_GENERATION)


def test_scientific_actions_follow_frozen_roles() -> None:
    registry = full_registry()
    registry.assert_allowed("train-a", ScientificAction.NORMALIZER_FIT)
    registry.assert_allowed("train-a", ScientificAction.SUPERVISED_FIT)
    registry.assert_allowed("test-a", ScientificAction.FINAL_EVALUATION)
    registry.assert_variant_role("train-a", SplitRole.TRAINING)
    with pytest.raises(SplitError, match="conflicts"):
        registry.assert_variant_role("train-a", SplitRole.CALIBRATION)


def test_every_derivative_of_source_episode_stays_in_one_partition() -> None:
    registry = full_registry()
    clean = provenance("train-a", SplitRole.TRAINING, "clean")
    attacked = provenance("train-a", SplitRole.TRAINING, "attack-a")
    assert_no_source_leakage((clean, attacked), registry)
    bad = ProvenanceRecord(
        **{
            **attacked.__dict__,
            "role": SplitRole.FINAL_TEST,
        }
    )
    with pytest.raises(SplitError, match="conflicts"):
        assert_no_source_leakage((clean, bad), registry)


def test_frame_local_indexes_do_not_become_track_ids() -> None:
    first = frame_local_object_group("s", "session", 1, 0)
    second = frame_local_object_group("s", "session", 2, 0)
    assert first != second


def test_split_registry_rejects_unfrozen_and_duplicate_assignments(
    tmp_path: Path,
) -> None:
    base = full_registry()
    assignment = base.assignments[0].model_dump()
    with pytest.raises(ValidationError, match="resolved"):
        SourceAssignment.model_validate({**assignment, "resolved": False})
    with pytest.raises(ValidationError, match="frozen"):
        SourceAssignment.model_validate({**assignment, "frozen": False})

    raw = base.model_dump()
    raw["assignments"] = base.assignments + (base.assignments[0],)
    with pytest.raises(ValidationError, match="source IDs"):
        SplitRegistry.model_validate(raw)
    duplicated_sequence = base.assignments[1].model_copy(
        update={"source_id": "other-source"}
    )
    raw["assignments"] = base.assignments + (duplicated_sequence,)
    with pytest.raises(ValidationError, match="sequence IDs"):
        SplitRegistry.model_validate(raw)

    raw["assignments"] = base.assignments[1:]
    with pytest.raises(ValidationError, match="mini_7"):
        SplitRegistry.model_validate(raw)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not: [valid", encoding="utf-8")
    with pytest.raises(SplitError, match="invalid split registry"):
        load_split_registry(invalid)
