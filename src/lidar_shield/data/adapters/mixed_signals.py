"""Clean-room adapter for the documented Mixed Signals archive layout.

Dataset-specific ambiguities are explicit configuration values.  In particular,
``mini_7`` uses short decimal timestamp fractions whose omitted digits are
*leading* zeroes, and its one-based label filenames map to zero-based sync IDs.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pypcd4 import PointCloud  # type: ignore[attr-defined]


class AdapterError(ValueError):
    """A source path or record violates the declared dataset contract."""


class FractionalTimeConvention(str, Enum):
    """How a filename fraction shorter than nine digits is interpreted."""

    OMITTED_LEADING_ZEROES = "omitted_leading_zeroes"
    DECIMAL_DIGITS = "decimal_digits"


@dataclass(frozen=True)
class MixedSignalsConfig:
    data_root: Path
    sequence_id: str
    agents: tuple[str, ...] = ("003", "004", "dome", "laser", "top")
    top_anchor_agent: str = "top"
    label_file_index_offset: int = 1
    fractional_time_convention: FractionalTimeConvention = (
        FractionalTimeConvention.OMITTED_LEADING_ZEROES
    )
    intensity_divisor: float = 3500.0
    expected_sync_start: int | None = 0
    expected_sync_stop: int | None = 298
    expected_label_stop: int | None = 289

    def __post_init__(self) -> None:
        root = Path(self.data_root).expanduser().resolve()
        object.__setattr__(self, "data_root", root)
        if not self.sequence_id or not self.agents:
            raise AdapterError("sequence_id and agents must be non-empty")
        if len(set(self.agents)) != len(self.agents):
            raise AdapterError("agent IDs must be unique")
        if self.top_anchor_agent not in self.agents:
            raise AdapterError("top anchor must be a configured agent")
        if not math.isfinite(self.intensity_divisor) or self.intensity_divisor <= 0:
            raise AdapterError("intensity divisor must be finite and positive")

    @property
    def cloud_dir(self) -> Path:
        return self.data_root / "PointClouds" / self.sequence_id

    @property
    def odometry_dir(self) -> Path:
        return self.data_root / "Odometry" / self.sequence_id

    @property
    def label_dir(self) -> Path:
        return self.data_root / "labels" / self.sequence_id

    @property
    def archive_path(self) -> Path:
        return self.data_root / "train" / f"{self.sequence_id}.tar"


@dataclass(frozen=True)
class CloudFile:
    agent_id: str
    sync_frame_id: int
    source_time_ns: int
    timestamp_text: str
    path: Path


@dataclass(frozen=True)
class PcdMetadata:
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    points: int
    width: int
    height: int
    encoding: str


@dataclass(frozen=True)
class PointMeasurements:
    xyzi: NDArray[np.float32]
    normalized_intensity: NDArray[np.float32]
    metadata: PcdMetadata


@dataclass(frozen=True)
class OdometrySample:
    time_ns: int
    header_time_ns: int
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    body_velocity_mps: tuple[float, float, float]


@dataclass(frozen=True)
class SequenceIndex:
    config: MixedSignalsConfig
    clouds: dict[str, dict[int, CloudFile]]
    labels: dict[int, Path]
    odometry: dict[str, Path]

    @property
    def sync_frame_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.clouds[self.config.top_anchor_agent]))

    def cloud(self, agent_id: str, sync_frame_id: int) -> CloudFile:
        try:
            return self.clouds[agent_id][sync_frame_id]
        except KeyError as exc:
            raise AdapterError(
                f"no cloud for agent={agent_id!r}, sync={sync_frame_id}"
            ) from exc


_CLOUD_NAME = re.compile(
    r"^(?P<agent>.+)_(?P<sync>[0-9]+)_"
    r"(?P<seconds>[0-9]+)\.(?P<fraction>[0-9]{1,9})\.pcd$"
)
_LABEL_NAME = re.compile(r"^.+_(?P<index>[0-9]+)\.txt$")

ODOMETRY_COLUMNS = (
    "%time",
    "field.header.stamp",
    "field.pose.pose.position.x",
    "field.pose.pose.position.y",
    "field.pose.pose.position.z",
    "field.pose.pose.orientation.x",
    "field.pose.pose.orientation.y",
    "field.pose.pose.orientation.z",
    "field.pose.pose.orientation.w",
    "field.twist.twist.linear.x",
    "field.twist.twist.linear.y",
    "field.twist.twist.linear.z",
)


def parse_decimal_time_ns(
    seconds: str,
    fraction: str,
    *,
    convention: FractionalTimeConvention,
) -> int:
    """Parse a filename timestamp without ever passing nanoseconds via float."""

    if not seconds.isdecimal() or not fraction.isdecimal() or len(fraction) > 9:
        raise AdapterError("timestamp must contain decimal seconds and 1-9 digits")
    if convention is FractionalTimeConvention.OMITTED_LEADING_ZEROES:
        nanoseconds = int(fraction.zfill(9))
    else:
        nanoseconds = int(fraction.ljust(9, "0"))
    return int(seconds) * 1_000_000_000 + nanoseconds


def parse_cloud_filename(
    path: str | Path,
    *,
    convention: FractionalTimeConvention,
) -> CloudFile:
    cloud_path = Path(path)
    match = _CLOUD_NAME.fullmatch(cloud_path.name)
    if match is None:
        raise AdapterError(f"invalid Mixed Signals cloud filename: {cloud_path.name}")
    timestamp_text = f"{match['seconds']}.{match['fraction']}"
    return CloudFile(
        agent_id=match["agent"],
        sync_frame_id=int(match["sync"]),
        source_time_ns=parse_decimal_time_ns(
            match["seconds"], match["fraction"], convention=convention
        ),
        timestamp_text=timestamp_text,
        path=cloud_path.resolve(),
    )


def _require_directories(config: MixedSignalsConfig) -> None:
    for path in (config.cloud_dir, config.odometry_dir, config.label_dir):
        if not path.is_dir():
            raise AdapterError(f"required dataset directory is absent: {path}")
    if not config.archive_path.is_file():
        raise AdapterError(f"required source archive is absent: {config.archive_path}")


def index_sequence(config: MixedSignalsConfig) -> SequenceIndex:
    """Index and validate the exact configured five-agent sequence layout."""

    _require_directories(config)
    clouds: dict[str, dict[int, CloudFile]] = {agent: {} for agent in config.agents}
    for path in sorted(config.cloud_dir.glob("*.pcd"), key=lambda item: item.name):
        record = parse_cloud_filename(
            path, convention=config.fractional_time_convention
        )
        if record.agent_id not in clouds:
            raise AdapterError(f"unexpected cloud agent {record.agent_id!r}")
        if record.sync_frame_id in clouds[record.agent_id]:
            raise AdapterError(
                f"duplicate cloud sync ID {record.sync_frame_id} for {record.agent_id}"
            )
        clouds[record.agent_id][record.sync_frame_id] = record

    anchor_ids = set(clouds[config.top_anchor_agent])
    if not anchor_ids:
        raise AdapterError("top-anchor agent has no point clouds")
    for agent, rows in clouds.items():
        if set(rows) != anchor_ids:
            missing = sorted(anchor_ids - set(rows))
            extra = sorted(set(rows) - anchor_ids)
            raise AdapterError(
                f"agent {agent!r} cloud membership differs from top anchor; "
                f"missing={missing}, extra={extra}"
            )

    if (
        config.expected_sync_start is not None
        and min(anchor_ids) != config.expected_sync_start
    ):
        raise AdapterError("observed sync start differs from configured expectation")
    if (
        config.expected_sync_stop is not None
        and max(anchor_ids) != config.expected_sync_stop
    ):
        raise AdapterError("observed sync stop differs from configured expectation")
    if anchor_ids != set(range(min(anchor_ids), max(anchor_ids) + 1)):
        raise AdapterError("cloud sync IDs must be contiguous")

    labels: dict[int, Path] = {}
    for path in sorted(config.label_dir.glob("*.txt"), key=lambda item: item.name):
        match = _LABEL_NAME.fullmatch(path.name)
        if match is None:
            raise AdapterError(f"invalid Mixed Signals label filename: {path.name}")
        sync_id = int(match["index"]) - config.label_file_index_offset
        if sync_id < 0 or sync_id in labels:
            raise AdapterError(f"invalid or duplicate semantic label sync ID {sync_id}")
        labels[sync_id] = path.resolve()
    if labels and config.expected_label_stop is not None:
        if max(labels) != config.expected_label_stop:
            raise AdapterError(
                "observed label stop differs from configured expectation"
            )
        if set(labels) != set(range(min(anchor_ids), config.expected_label_stop + 1)):
            raise AdapterError(
                "semantic label sync IDs are not contiguous from anchor start"
            )

    odometry: dict[str, Path] = {}
    for agent in config.agents:
        path = config.odometry_dir / f"odometry_{agent}.csv"
        if path.is_file():
            odometry[agent] = path.resolve()
    if set(odometry) != {"003", "004", "laser"}:
        raise AdapterError(
            "mini_7 requires odometry for exactly vehicles 003, 004, and laser"
        )
    return SequenceIndex(config=config, clouds=clouds, labels=labels, odometry=odometry)


def read_odometry(path: str | Path) -> tuple[OdometrySample, ...]:
    """Read odometry with integer timestamp agreement and finite-value audits."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(set(ODOMETRY_COLUMNS) - set(reader.fieldnames or ()))
            if missing:
                raise AdapterError(f"odometry is missing columns: {missing}")
            samples: list[OdometrySample] = []
            previous_time: int | None = None
            for line_number, row in enumerate(reader, start=2):
                try:
                    time_ns = int(row["%time"])
                    header_time_ns = int(row["field.header.stamp"])
                    numeric = tuple(float(row[name]) for name in ODOMETRY_COLUMNS[2:])
                except (TypeError, ValueError) as exc:
                    raise AdapterError(
                        f"invalid odometry value at {source}:{line_number}"
                    ) from exc
                if time_ns != header_time_ns:
                    raise AdapterError(
                        f"odometry timestamp disagreement at {source}:{line_number}"
                    )
                if time_ns < 0 or not all(math.isfinite(value) for value in numeric):
                    raise AdapterError(
                        f"nonfinite or negative-time odometry at {source}:{line_number}"
                    )
                if previous_time is not None and time_ns <= previous_time:
                    raise AdapterError(
                        "odometry timestamps must be strictly increasing"
                    )
                previous_time = time_ns
                samples.append(
                    OdometrySample(
                        time_ns=time_ns,
                        header_time_ns=header_time_ns,
                        position_m=(numeric[0], numeric[1], numeric[2]),
                        quaternion_xyzw=(
                            numeric[3],
                            numeric[4],
                            numeric[5],
                            numeric[6],
                        ),
                        body_velocity_mps=(numeric[7], numeric[8], numeric[9]),
                    )
                )
    except (OSError, UnicodeError) as exc:
        raise AdapterError(f"cannot read odometry {source}: {exc}") from exc
    if not samples:
        raise AdapterError(f"odometry is empty: {source}")
    return tuple(samples)


def audit_filename_odometry_agreement(
    index: SequenceIndex,
    odometry_by_agent: dict[str, tuple[OdometrySample, ...]],
) -> dict[str, bool]:
    """Audit whether every vehicle cloud filename time occurs in its CSV."""

    result: dict[str, bool] = {}
    for agent, samples in odometry_by_agent.items():
        times = {sample.time_ns for sample in samples}
        result[agent] = all(
            row.source_time_ns in times for row in index.clouds[agent].values()
        )
    return result


def _pcd_metadata(cloud: PointCloud) -> PcdMetadata:
    metadata = cloud.metadata
    return PcdMetadata(
        fields=tuple(metadata.fields),
        sizes=tuple(int(value) for value in metadata.size),
        types=tuple(metadata.type),
        counts=tuple(int(value) for value in metadata.count),
        points=int(metadata.points),
        width=int(metadata.width),
        height=int(metadata.height),
        encoding=metadata.data.value,
    )


def load_point_measurements(
    path: str | Path,
    *,
    expected_fields: tuple[str, ...] = ("x", "y", "z", "intensity"),
    intensity_divisor: float = 3500.0,
) -> PointMeasurements:
    """Load validated XYZI data and retain both raw and normalized intensity."""

    if not math.isfinite(intensity_divisor) or intensity_divisor <= 0:
        raise AdapterError("intensity divisor must be finite and positive")
    source = Path(path)
    try:
        cloud = PointCloud.from_path(source)
    except (OSError, ValueError, TypeError) as exc:
        raise AdapterError(f"cannot decode PCD {source}: {exc}") from exc
    metadata = _pcd_metadata(cloud)
    if not set(expected_fields).issubset(metadata.fields):
        raise AdapterError(
            f"PCD fields {metadata.fields} do not contain {expected_fields}"
        )
    by_name = {
        name: (size, kind, count)
        for name, size, kind, count in zip(
            metadata.fields,
            metadata.sizes,
            metadata.types,
            metadata.counts,
            strict=True,
        )
    }
    if any(by_name[name] != (4, "F", 1) for name in expected_fields):
        raise AdapterError("XYZI fields must be scalar float32 PCD fields")
    if metadata.encoding not in {"ascii", "binary", "binary_compressed"}:
        raise AdapterError(f"unsupported PCD encoding {metadata.encoding!r}")
    xyzi = np.asarray(cloud.numpy(expected_fields), dtype=np.float32)
    if xyzi.shape != (metadata.points, 4):
        raise AdapterError("decoded XYZI shape disagrees with PCD metadata")
    normalized = np.clip(xyzi[:, 3] / intensity_divisor, 0.0, 1.0).astype(
        np.float32, copy=False
    )
    return PointMeasurements(
        xyzi=xyzi, normalized_intensity=normalized, metadata=metadata
    )
