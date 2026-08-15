#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FRONTEND_ROOT="${REPO_ROOT}/web/visual_debugger"

if ! command -v npm >/dev/null 2>&1; then
  echo "error: Node.js 24 and npm are required for frontend contributor checks." >&2
  exit 127
fi

run_static() {
  npm run format:check --prefix "${FRONTEND_ROOT}"
  npm run lint --prefix "${FRONTEND_ROOT}"
  npm run typecheck --prefix "${FRONTEND_ROOT}"
  npm run test:unit --prefix "${FRONTEND_ROOT}"
}

run_e2e() {
  npm run test:e2e --prefix "${FRONTEND_ROOT}" -- "$@"
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
  --e2e-only)
    shift
    run_e2e "$@"
    ;;
  *)
    echo "usage: scripts/dev/check_frontend.sh [--static-only | --e2e-only [playwright arguments...]]" >&2
    exit 2
    ;;
esac
