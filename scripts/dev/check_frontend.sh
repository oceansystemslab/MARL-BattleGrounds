#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FRONTEND_ROOT="${REPO_ROOT}/web/visual_debugger"

if ! command -v npm >/dev/null 2>&1; then
  echo "error: Node.js 24 and npm are required for frontend contributor checks." >&2
  exit 127
fi

run_style() {
  npm run format:check --prefix "${FRONTEND_ROOT}"
  npm run lint --prefix "${FRONTEND_ROOT}"
  npm run typecheck --prefix "${FRONTEND_ROOT}"
}

run_unit() {
  npm run test:unit --prefix "${FRONTEND_ROOT}"
}

run_static() {
  run_style
  run_unit
}

run_e2e() {
  npm run test:e2e --prefix "${FRONTEND_ROOT}" -- "$@"
}

run_e2e_shard() {
  local shard="$1"
  shift
  node "${FRONTEND_ROOT}/e2e/support/run-ci-shard.js" "${shard}" "$@"
}

case "${1:-}" in
  "")
    run_static
    run_e2e
    ;;
  --static-only)
    shift
    if (( $# != 0 )); then
      echo "error: --static-only does not accept additional arguments." >&2
      exit 2
    fi
    run_static
    ;;
  --style-only)
    shift
    if (( $# != 0 )); then
      echo "error: --style-only does not accept additional arguments." >&2
      exit 2
    fi
    run_style
    ;;
  --unit-only)
    shift
    if (( $# != 0 )); then
      echo "error: --unit-only does not accept additional arguments." >&2
      exit 2
    fi
    run_unit
    ;;
  --e2e-only)
    shift
    run_e2e "$@"
    ;;
  --e2e-shard)
    shift
    if (( $# < 1 )); then
      echo "error: --e2e-shard requires N/8." >&2
      exit 2
    fi
    shard="$1"
    shift
    run_e2e_shard "${shard}" "$@"
    ;;
  *)
    echo "usage: scripts/dev/check_frontend.sh [--static-only | --style-only | --unit-only | --e2e-only [playwright arguments...] | --e2e-shard N/8 [playwright arguments...]]" >&2
    exit 2
    ;;
esac
