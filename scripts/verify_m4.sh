#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

uv sync --frozen
uv pip check
if [[ -f data/train/mini_7.tar ]]; then
  d3_root="$(mktemp -d)"
  uv run lidar-shield demo d3 \
    --config configs/attacks/mini7_d3_velocity_spike.yaml \
    --data-root data \
    --artifact-root "$d3_root" \
    --frames 0:3
elif [[ -f artifacts/m2-mini7-d1/sequences/mini_7/clean/resume.complete.json ]]; then
  d3_root="$(mktemp -d)"
  uv run lidar-shield demo d3 \
    --config configs/attacks/mini7_d3_velocity_spike.yaml \
    --data-root data \
    --artifact-root "$d3_root" \
    --clean-dir artifacts/m2-mini7-d1/sequences/mini_7/clean
fi
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
