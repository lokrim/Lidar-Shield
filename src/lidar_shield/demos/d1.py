"""Compact real-data playback payload for the M2 development slice."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lidar_shield.data.cache import CacheBuildResult


def _evidence_lookup(rows: pd.DataFrame, name: str) -> dict[str, Any]:
    selected = rows.loc[rows["evidence_name"].eq(name)]
    if selected.empty:
        return {"value": None, "reason": "absent"}
    row = selected.iloc[0]
    value = row["value"]
    return {
        "value": None if pd.isna(value) else float(value),
        "reason": str(row["reason_code"]),
    }


def build_d1_payload(result: CacheBuildResult) -> dict[str, Any]:
    """Read a verified cache and expose auditable five-agent evidence."""

    frame_table = pd.read_parquet(result.clean_dir / "frame.parquet")
    evidence = pd.read_parquet(result.clean_dir / "evidence.parquet")
    playback: list[dict[str, Any]] = []
    for (frame_id, agent_id), rows in evidence.groupby(
        ["sync_frame_id", "agent_id"], sort=True
    ):
        frame = frame_table.loc[
            frame_table["sync_frame_id"].eq(frame_id)
            & frame_table["agent_id"].eq(agent_id)
        ].iloc[0]
        object_rows = rows.loc[rows["evidence_level"].eq("object")]
        deficits = object_rows.loc[
            object_rows["evidence_name"].eq("count_deficit_residual_development"),
            "value",
        ].dropna()
        surpluses = object_rows.loc[
            object_rows["evidence_name"].eq("count_surplus_residual_development"),
            "value",
        ].dropna()
        deficit_status = _evidence_lookup(
            object_rows, "count_deficit_residual_development"
        )
        surplus_status = _evidence_lookup(
            object_rows, "count_surplus_residual_development"
        )
        playback.append(
            {
                "sync_frame_id": int(frame_id),
                "decision_time_ns": int(frame["decision_time_ns"]),
                "agent_id": str(agent_id),
                "source_time_ns": int(frame["source_time_ns"]),
                "source_age_ns": int(frame["match_age_ns"]),
                "transform_quality": str(frame["transform_quality"]),
                "label_available": bool(frame["label_available"]),
                "kinematic_residual_m": _evidence_lookup(rows, "kinematic_residual_m"),
                "max_development_deficit": (
                    float(deficits.max()) if not deficits.empty else None
                ),
                "development_deficit_reason": deficit_status["reason"],
                "max_development_surplus": (
                    float(surpluses.max()) if not surpluses.empty else None
                ),
                "development_surplus_reason": surplus_status["reason"],
                "peer_corroboration": _evidence_lookup(
                    object_rows, "peer_corroboration_score"
                ),
                "historical_fixed_risk_diagnostic": _evidence_lookup(
                    rows, "historical_fixed_risk_diagnostic"
                ),
                "unknown_aware_fixed_risk_uncalibrated": _evidence_lookup(
                    rows, "unknown_aware_fixed_risk_uncalibrated"
                ),
                "unknown_aware_abstained": bool(
                    _evidence_lookup(rows, "unknown_aware_fixed_abstained")["value"]
                ),
            }
        )
    return {
        "demo": "D1",
        "sequence_id": str(frame_table["sequence_id"].iloc[0]),
        "oracle_gt": True,
        "development_only": True,
        "calibration_gate": "unresolved_non_top_extrinsics_and_nominal_coverage",
        "fixed_risk_calibration": "uncalibrated",
        "cache": {
            "path": result.clean_dir.as_posix(),
            "manifest_sha256": result.manifest_sha256,
            "output_hashes": result.output_hashes,
            "resumed": result.resumed,
        },
        "playback": playback,
    }
