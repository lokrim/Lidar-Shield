"""Command-line entry point for configuration and contract verification."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from lidar_shield import __version__
from lidar_shield.config import ConfigurationError, config_hash, load_config
from lidar_shield.data.cache import (
    CacheBuildConfig,
    CacheError,
    build_clean_cache,
    verify_clean_cache,
)
from lidar_shield.data.index import IndexError, load_agent_registry
from lidar_shield.demos.d0 import build_d0_payload
from lidar_shield.demos.d1 import build_d1_payload
from lidar_shield.manifest import canonical_json_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lidar-shield",
        description="Independent lidar-shield research tooling (M0 foundation–M2)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command")

    config_parser = commands.add_parser("config", help="configuration operations")
    config_commands = config_parser.add_subparsers(dest="config_command")
    validate_parser = config_commands.add_parser(
        "validate", help="load and strictly validate one YAML configuration"
    )
    validate_parser.add_argument(
        "--config",
        required=True,
        help="explicit path to the YAML configuration",
    )

    demo_parser = commands.add_parser("demo", help="dataset-free demonstrations")
    demo_commands = demo_parser.add_subparsers(dest="demo_command")
    d0_parser = demo_commands.add_parser(
        "d0", help="replay the deterministic five-agent contract timeline"
    )
    d0_parser.add_argument(
        "--agents",
        required=True,
        help="explicit path to the agent registry YAML",
    )

    cache_parser = commands.add_parser("cache", help="clean-cache operations")
    cache_commands = cache_parser.add_subparsers(dest="cache_command")
    cache_build = cache_commands.add_parser(
        "build", help="build or verify one immutable sequence cache"
    )
    _add_cache_build_arguments(cache_build)
    cache_verify = cache_commands.add_parser(
        "verify", help="verify hashes and Parquet readability"
    )
    cache_verify.add_argument("--cache-dir", required=True)

    d1_parser = demo_commands.add_parser(
        "d1", help="build and display a small real mini_7 evidence window"
    )
    _add_cache_build_arguments(d1_parser)
    return parser


def _add_cache_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--sequence", default="mini_7")
    parser.add_argument(
        "--frames", required=True, help="half-open sync-frame range, e.g. 0:2"
    )
    parser.add_argument(
        "--agents", default="003,004,dome,laser,top", help="comma-separated IDs"
    )
    parser.add_argument("--agent-registry", default="configs/agents/mixed_signals.yaml")
    parser.add_argument("--split-registry", default="configs/data/stage1_split.yaml")
    parser.add_argument("--dataset-config", default="configs/data/mini_7.yaml")


def _parse_frames(value: str) -> tuple[int, ...]:
    try:
        start_text, stop_text = value.split(":", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise CacheError("--frames must be a half-open START:STOP range") from exc
    if start < 0 or stop <= start:
        raise CacheError("--frames requires 0 <= START < STOP")
    return tuple(range(start, stop))


def _cache_config(args: argparse.Namespace) -> CacheBuildConfig:
    command = shlex.join(
        [
            "lidar-shield",
            args.command,
            args.cache_command if args.command == "cache" else args.demo_command,
            "--data-root",
            args.data_root,
            "--artifact-root",
            args.artifact_root,
            "--experiment-id",
            args.experiment_id,
            "--sequence",
            args.sequence,
            "--frames",
            args.frames,
            "--agents",
            args.agents,
        ]
    )
    return CacheBuildConfig(
        data_root=Path(args.data_root),
        artifact_root=Path(args.artifact_root),
        experiment_id=args.experiment_id,
        sequence_id=args.sequence,
        selected_frames=_parse_frames(args.frames),
        selected_agents=tuple(args.agents.split(",")),
        agent_registry_path=Path(args.agent_registry),
        split_registry_path=Path(args.split_registry),
        dataset_config_path=Path(args.dataset_config),
        producer_command=command,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "config" and args.config_command is None:
        parser.parse_args(["config", "--help"])
    if args.command == "demo" and args.demo_command is None:
        parser.parse_args(["demo", "--help"])
    if args.command == "cache" and args.cache_command is None:
        parser.parse_args(["cache", "--help"])

    if args.command == "config" and args.config_command == "validate":
        try:
            config = load_config(args.config)
        except ConfigurationError as exc:
            print(f"configuration error: {exc}", file=sys.stderr)
            return 2
        summary = {
            "artifact_root": str(config.paths.artifact_root),
            "config_hash": config_hash(args.config),
            "schema_version": config.schema_version,
            "seed": config.reproducibility.seed,
            "status": "valid",
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0

    if args.command == "demo" and args.demo_command == "d0":
        try:
            registry = load_agent_registry(args.agents)
        except IndexError as exc:
            print(f"agent registry error: {exc}", file=sys.stderr)
            return 2
        sys.stdout.buffer.write(canonical_json_bytes(build_d0_payload(registry)))
        return 0

    if args.command == "cache" and args.cache_command == "verify":
        try:
            result = verify_clean_cache(args.cache_dir)
        except CacheError as exc:
            print(f"cache error: {exc}", file=sys.stderr)
            return 2
        sys.stdout.buffer.write(
            canonical_json_bytes(
                {
                    "status": "verified",
                    "cache_dir": result.clean_dir.as_posix(),
                    "manifest_sha256": result.manifest_sha256,
                    "output_hashes": result.output_hashes,
                }
            )
        )
        return 0

    if (args.command == "cache" and args.cache_command == "build") or (
        args.command == "demo" and args.demo_command == "d1"
    ):
        try:
            result = build_clean_cache(_cache_config(args))
        except (CacheError, IndexError) as exc:
            print(f"cache error: {exc}", file=sys.stderr)
            return 2
        payload = (
            build_d1_payload(result)
            if args.command == "demo"
            else {
                "status": "verified",
                "cache_dir": result.clean_dir.as_posix(),
                "manifest_sha256": result.manifest_sha256,
                "output_hashes": result.output_hashes,
                "resumed": result.resumed,
            }
        )
        sys.stdout.buffer.write(canonical_json_bytes(payload))
        return 0

    parser.error("unsupported command")


def entrypoint() -> None:
    raise SystemExit(main())
