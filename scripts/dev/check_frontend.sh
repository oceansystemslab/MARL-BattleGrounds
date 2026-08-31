#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FRONTEND_ROOT="${REPO_ROOT}/web/visual_debugger"

# shellcheck source=scripts/dev/validation_parallel.sh
source "${SCRIPT_DIR}/validation_parallel.sh"

export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
export UV_NO_SYNC=1

require_canonical_frontend_environment() {
  local forbidden_variable=""

  for forbidden_variable in \
    JAX_PLATFORM_NAME \
    JAX_XLA_BACKEND \
    JAX_DISABLE_JIT \
    MARL_CP4_C3_SHIELD_ONLY \
    MARL_CP4_E_CAPTURE_DIR \
    MARL_CP5_C_SLICE_ONLY \
    MARL_CP5_SLICE_5_ONLY; do
    if [[ -v "${forbidden_variable}" ]]; then
      echo "error: unset ${forbidden_variable} before the canonical frontend gate." >&2
      return 2
    fi
  done
}

usage() {
  echo "usage: scripts/dev/check_frontend.sh [--static-only | --style-only | --unit-only | --e2e-only [playwright arguments...] | --e2e-shard N/8 [playwright arguments...] | --help]" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

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

run_complete_frontend_gate() {
  local output_root=""
  local shard_number=""
  local status=0

  require_canonical_frontend_environment
  export CI=1
  output_root="$(mktemp -d "${TMPDIR:-/tmp}/marl-browser-validation.XXXXXX")"
  marl_validation_init 8 frontend-validation
  for shard_number in {1..8}; do
    marl_validation_start \
      "Browser profile ${shard_number}/8" \
      run_e2e_shard \
      "${shard_number}/8" \
      --max-failures=1 \
      --output "${output_root}/profile-${shard_number}"
  done
  marl_validation_start "Frontend static and unit gates" run_static
  if ! marl_validation_finish; then
    status=1
  fi
  marl_validation_cleanup

  if (( status == 0 )); then
    rm -rf -- "${output_root}"
  else
    echo "Browser failure artifacts retained at ${output_root}" >&2
  fi
  return "${status}"
}

case "${1:-}" in
  "")
    run_complete_frontend_gate
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
    require_canonical_frontend_environment
    export CI=1
    run_e2e_shard "${shard}" "$@"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
