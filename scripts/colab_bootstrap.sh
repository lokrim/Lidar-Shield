#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Pin the environment manager as well as all project packages in uv.lock.
python -m pip install --quiet "uv==0.11.17"
uv sync --frozen
uv run lidar-shield --version
uv run lidar-shield config validate --config configs/default.yaml
uv run pytest

echo "M0 Colab bootstrap smoke passed; no dataset or GPU was used."
