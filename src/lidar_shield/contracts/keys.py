"""Canonical identity validation and overlap-safe tabular joins."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from enum import Enum
from typing import Literal

import pandas as pd
from pandas.api.types import is_integer_dtype

CANONICAL_IDENTITY_FIELDS = (
    "experiment_id",
    "variant_id",
    "sequence_id",
    "session_id",
    "sync_frame_id",
    "decision_time_ns",
    "source_time_ns",
    "agent_id",
    "agent_type",
    "object_id_or_local_index",
)

FRAME_UNIQUE_KEY = (
    "experiment_id",
    "variant_id",
    "sequence_id",
    "session_id",
    "sync_frame_id",
    "decision_time_ns",
    "agent_id",
)

OBJECT_UNIQUE_KEY = FRAME_UNIQUE_KEY + ("object_id_or_local_index",)


class ContractError(ValueError):
    """A canonical table or join violates the declared contract."""


class TableLevel(str, Enum):
    FRAME = "frame"
    OBJECT = "object"
    EVIDENCE = "evidence"


class JoinCardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


def _require_string_ids(table: pd.DataFrame, columns: Collection[str]) -> None:
    for column in columns:
        invalid = table[column].map(
            lambda value: not isinstance(value, str) or not value
        )
        if bool(invalid.any()):
            raise ContractError(f"{column} must contain non-empty strings")


def _require_integer_column(
    table: pd.DataFrame, column: str, *, nullable: bool
) -> None:
    values = table[column]
    if not nullable and bool(values.isna().any()):
        raise ContractError(f"{column} must not be null")
    non_null = values.dropna()
    if not is_integer_dtype(non_null.dtype):
        raise ContractError(
            f"{column} must use an integer dtype; float nanoseconds are forbidden"
        )


def validate_canonical_table(table: pd.DataFrame, *, level: TableLevel) -> None:
    """Validate identity presence, types, nullability, and row uniqueness.

    Frame rows deliberately require a null object key. Their source time may be
    null for unavailable-source reasons. Object rows require both source and
    object keys. Evidence rows may be frame- or object-level and are validated
    by their ``evidence_level`` column.
    """

    missing = sorted(set(CANONICAL_IDENTITY_FIELDS) - set(table.columns))
    if missing:
        raise ContractError(f"missing canonical columns: {missing}")
    if table.empty:
        return

    _require_string_ids(
        table,
        (
            "experiment_id",
            "variant_id",
            "sequence_id",
            "session_id",
            "agent_id",
            "agent_type",
        ),
    )
    _require_integer_column(table, "sync_frame_id", nullable=False)
    _require_integer_column(table, "decision_time_ns", nullable=False)
    _require_integer_column(table, "source_time_ns", nullable=True)

    unique_key: tuple[str, ...]
    if level is TableLevel.FRAME:
        if bool(table["object_id_or_local_index"].notna().any()):
            raise ContractError("frame rows must have a null object key")
        unique_key = FRAME_UNIQUE_KEY
    elif level is TableLevel.OBJECT:
        if bool(table["object_id_or_local_index"].isna().any()):
            raise ContractError("object rows require an object/local-index key")
        if bool(table["source_time_ns"].isna().any()):
            raise ContractError("object rows require source_time_ns")
        unique_key = OBJECT_UNIQUE_KEY
    else:
        evidence_columns = {"evidence_level", "evidence_name"}
        missing_evidence = sorted(evidence_columns - set(table.columns))
        if missing_evidence:
            raise ContractError(f"evidence rows require columns: {missing_evidence}")
        frame_mask = table["evidence_level"].eq("frame")
        object_mask = table["evidence_level"].eq("object")
        if bool((~(frame_mask | object_mask)).any()):
            raise ContractError("evidence_level must be frame or object")
        if bool(table.loc[frame_mask, "object_id_or_local_index"].notna().any()):
            raise ContractError("frame evidence must have a null object key")
        if bool(table.loc[object_mask, "object_id_or_local_index"].isna().any()):
            raise ContractError("object evidence requires an object key")
        unique_key = OBJECT_UNIQUE_KEY + ("evidence_name",)

    duplicated = table.duplicated(list(unique_key), keep=False)
    if bool(duplicated.any()):
        raise ContractError(
            f"duplicate {level.value} keys: "
            f"{table.loc[duplicated, list(unique_key)].to_dict('records')}"
        )


def merge_checked(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: Sequence[str],
    cardinality: JoinCardinality,
    how: Literal["left", "right", "inner", "outer"] = "left",
    right_rename: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Merge only after cardinality and every overlapping column are explicit."""

    join_columns = tuple(on)
    if not join_columns:
        raise ContractError("join columns must be declared")
    absent = sorted(
        column
        for column in join_columns
        if column not in left.columns or column not in right.columns
    )
    if absent:
        raise ContractError(f"join columns absent from an input: {absent}")

    overlap = (set(left.columns) & set(right.columns)) - set(join_columns)
    rename = dict(right_rename or {})
    if overlap != set(rename):
        raise ContractError(
            "every overlapping non-key column must be deliberately renamed; "
            f"overlap={sorted(overlap)}, mappings={sorted(rename)}"
        )
    if len(set(rename.values())) != len(rename):
        raise ContractError("right-side rename targets must be unique")
    conflicting_targets = (set(rename.values()) & set(left.columns)) - set(join_columns)
    if conflicting_targets:
        raise ContractError(
            "right-side rename targets collide with left columns: "
            f"{sorted(conflicting_targets)}"
        )

    prepared = right.rename(columns=rename)
    try:
        result = left.merge(
            prepared,
            how=how,
            on=list(join_columns),
            validate=cardinality.value,
            suffixes=("", "__forbidden_overlap"),
            sort=False,
        )
    except pd.errors.MergeError as exc:
        raise ContractError(
            f"join violates declared {cardinality.value} cardinality: {exc}"
        ) from exc
    forbidden = [column for column in result if column.endswith("__forbidden_overlap")]
    if forbidden:
        raise ContractError(f"join produced undeclared overlap columns: {forbidden}")
    return result
