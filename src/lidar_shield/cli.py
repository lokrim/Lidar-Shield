"""Command-line entry point for configuration and contract verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from lidar_shield import __version__
from lidar_shield.config import ConfigurationError, config_hash, load_config
from lidar_shield.data.index import IndexError, load_agent_registry
from lidar_shield.demos.d0 import build_d0_payload
from lidar_shield.manifest import canonical_json_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lidar-shield",
        description=(
            "Independent lidar-shield research tooling (M0 foundation + M1 contracts)"
        ),
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
    return parser


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

    parser.error("unsupported command")


def entrypoint() -> None:
    raise SystemExit(main())
