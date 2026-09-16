"""D3 manifest preview and clean-versus-attacked final-pipeline replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lidar_shield.attacks.manifests import AttackManifest
from lidar_shield.attacks.overlays import (
    EVIDENCE_DELTA_KEY,
    apply_evidence_delta,
    load_realized_effect,
)
from lidar_shield.evaluation.episodes import (
    EpisodeObservation,
    replay_attack_episode,
    summarize_attack_coverage,
)


def _value(evidence: pd.DataFrame, frame: int, agent: str, name: str) -> float | None:
    values = evidence.loc[
        evidence["sync_frame_id"].eq(frame)
        & evidence["agent_id"].eq(agent)
        & evidence["evidence_name"].eq(name),
        "value",
    ].dropna()
    if not len(values):
        return None
    return float(values.max() if name == "maximum_object_suspicion" else values.min())


def _episode_observations(
    manifest: AttackManifest,
    frames: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    variant_id: str,
    actually_affected: frozenset[tuple[int, str]],
) -> tuple[EpisodeObservation, ...]:
    affected_frames = sorted(
        set(
            evidence.loc[
                evidence["agent_id"].eq(manifest.target.sender_id), "sync_frame_id"
            ]
        )
    )
    if manifest.target.sync_frame_id not in affected_frames:
        affected_frames.append(manifest.target.sync_frame_id)
        affected_frames.sort()
    rows: list[EpisodeObservation] = []
    for frame_id in affected_frames:
        frame_rows = frames.loc[frames["sync_frame_id"].eq(frame_id)]
        for frame in frame_rows.itertuples():
            kinematic = _value(
                evidence, frame_id, frame.agent_id, "kinematic_residual_m"
            )
            count = _value(
                evidence, frame_id, frame.agent_id, "maximum_object_suspicion"
            )
            peer = _value(
                evidence, frame_id, frame.agent_id, "peer_corroboration_score"
            )
            applicable = bool(
                _value(
                    evidence,
                    frame_id,
                    frame.agent_id,
                    "kinematic_modality_applicable",
                )
            )
            rows.append(
                EpisodeObservation(
                    episode_id=manifest.episode_id,
                    variant_id=variant_id,
                    source_id=manifest.target.source_id,
                    sequence_id=manifest.target.sequence_id,
                    session_id=manifest.target.session_id,
                    sync_frame_id=frame_id,
                    decision_time_ns=int(frame.decision_time_ns),
                    source_time_ns=(
                        None
                        if pd.isna(frame.source_time_ns)
                        else int(frame.source_time_ns)
                    ),
                    agent_id=str(frame.agent_id),
                    agent_type=str(frame.agent_type),  # type: ignore[arg-type]
                    is_ego=str(frame.agent_id) == "top",
                    intended_target=(
                        frame_id == manifest.target.sync_frame_id
                        and str(frame.agent_id) == manifest.target.sender_id
                    ),
                    actually_affected=(frame_id, str(frame.agent_id))
                    in actually_affected,
                    message_present=bool(frame.message_present),
                    kinematic_error_m=kinematic,
                    two_sided_count_residual=count,
                    peer_corroboration_score=peer,
                    kinematic_applicable=applicable,
                    proxy_vector=(
                        kinematic or 0.0,
                        count or 0.0,
                        1.0 - peer if peer is not None else 0.0,
                    ),
                    local_confidence=1.0,
                )
            )
    return tuple(rows)


def build_d3_payload(
    manifest: AttackManifest,
    *,
    clean_dir: str | Path,
    variant_dir: str | Path,
) -> dict[str, Any]:
    clean = Path(clean_dir).expanduser().resolve()
    variant = Path(variant_dir).expanduser().resolve()
    frames = pd.read_parquet(clean / "frame.parquet")
    clean_evidence = pd.read_parquet(clean / "evidence.parquet")
    delta = pd.read_parquet(variant / "evidence_delta.parquet")
    attacked_evidence = apply_evidence_delta(
        clean_evidence, delta, key_columns=EVIDENCE_DELTA_KEY
    )
    changed = frozenset(
        (int(row.sync_frame_id), str(row.agent_id)) for row in delta.itertuples()
    )
    clean_observations = _episode_observations(
        manifest,
        frames,
        clean_evidence,
        variant_id="clean",
        actually_affected=frozenset(),
    )
    attacked_observations = _episode_observations(
        manifest,
        frames,
        attacked_evidence,
        variant_id=manifest.variant_id,
        actually_affected=changed,
    )
    clean_run = replay_attack_episode(clean_observations, case="clean")
    attacked_run = replay_attack_episode(attacked_observations, case="attacked")
    effect = load_realized_effect(variant / "realized_effect.json")
    maximum_effect = float(
        effect.get("realized_evidence_effect", {}).get("maximum_absolute_delta_m", 0.0)
    )
    if maximum_effect == 0:
        band = "zero"
    elif maximum_effect < 0.1:
        band = "low"
    else:
        band = "material"
    coverage_row = {
        "family": manifest.family,
        "severity": manifest.severity,
        "agent_type": manifest.target.agent_type,
        "status": effect["status"],
        "realized_effect_band": band,
    }
    return {
        "demo": "D3",
        "scope": "M4 development/regression scientific attack episode",
        "manifest_sha256": manifest.sha256,
        "manifest_preview": manifest.model_dump(mode="json"),
        "realized_effect": effect,
        "coverage": summarize_attack_coverage((coverage_row,)),
        "clean": clean_run.as_dict(),
        "attacked": attacked_run.as_dict(),
    }
