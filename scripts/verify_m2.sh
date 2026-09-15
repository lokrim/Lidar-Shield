#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

uv sync --frozen
uv pip check
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy

if [[ -f data/train/mini_7.tar ]]; then
  uv run lidar-shield demo d1 \
    --data-root data \
    --artifact-root artifacts \
    --experiment-id m2-mini7-d1 \
    --frames 0:2
  uv run lidar-shield cache verify \
    --cache-dir artifacts/m2-mini7-d1/sequences/mini_7/clean
else
  printf '%s\n' 'official mini_7 archive absent; real D1 gate not executed' >&2
  exit 2
fi
