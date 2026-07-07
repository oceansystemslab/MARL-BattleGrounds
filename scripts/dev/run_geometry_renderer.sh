#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/dev/run_geometry_renderer.sh [--manual|--static] [options]

Options:
  --controlled-slot N     Global agent slot controlled in manual mode.
  --step-interval-ms N    Manual-control timestep interval.

Environment:
  MODE=manual|static       Renderer mode. Defaults to manual.
  CONTROLLED_SLOT=0        Global agent slot controlled in manual mode.
  STEP_INTERVAL_MS=50     Manual-control timestep interval.

Controls:
  WASD move cardinally, QEZC move diagonally, no input means stay.
EOF
  exit 0
fi

cd "${REPO_ROOT}"
exec uv run --extra viz python scripts/dev/geometry_debug_renderer.py "$@"
