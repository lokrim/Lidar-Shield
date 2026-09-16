#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

uv sync --frozen
uv pip check
uv run lidar-shield demo d2 \
  --fixture tests/fixtures/m3/integration_fixture.json
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
