#!/usr/bin/env bash

# Run browser-client style, type, unit and browser checks.
# Usage: scripts/dev/check_frontend.sh --help. With no options, run eight browser
# profiles plus static/unit checks with at most eight workers. The profile file
# owns browser membership. --static-only includes unit tests; --style-only does
# not. --e2e-only and --e2e-shard forward extra Playwright arguments.
# --timings DIR runs the same full gate and also saves each browser profile's
# per-test times as DIR/browser-profile-N.json for scripts/dev/shard_costs.py.
# Requires Node.js 24/npm, installed frontend/browser dependencies and the prepared
# uv environment for local servers. Python uses CPU; this is a correctness gate.
# Full runs remove temporary browser outputs on success and retain them on failure.
# Every failing task makes the full command fail.
set -euo pipefail

CALLER_DIR="${PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FRONTEND_ROOT="${REPO_ROOT}/web/visual_debugger"

# shellcheck source=scripts/dev/validation_parallel.sh
source "${SCRIPT_DIR}/validation_parallel.sh"

export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
export UV_NO_SYNC=1

# Reject backend, JIT or legacy slice/capture environment overrides that would
# change the canonical browser check. No arguments; return 2 on the first error.
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

# Print frontend check options to stderr. No arguments or file changes.
usage() {
  echo "usage: scripts/dev/check_frontend.sh [--static-only | --style-only | --unit-only | --timings DIR | --e2e-only [playwright arguments...] | --e2e-shard N/8 [playwright arguments...] | --help]" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "error: Node.js 24 and npm are required for frontend contributor checks." >&2
  exit 127
fi

# Run frontend format-check, lint and typecheck in order, without fixes. No
# arguments. A failed npm command stops this strict-mode script or worker.
run_style() {
  npm run format:check --prefix "${FRONTEND_ROOT}"
  npm run lint --prefix "${FRONTEND_ROOT}"
  npm run typecheck --prefix "${FRONTEND_ROOT}"
}

# Run the frontend unit-test npm command. No arguments. Return npm/test status.
run_unit() {
  npm run test:unit --prefix "${FRONTEND_ROOT}"
}

# Run style/type checks followed by unit tests. No arguments. Preserve failures
# through the caller strict-mode script or worker.
run_static() {
  run_style
  run_unit
}

# Run Playwright through npm, forwarding every argument unchanged. The test
# configuration owns server startup, browser outputs and test selection.
run_e2e() {
  npm run test:e2e --prefix "${FRONTEND_ROOT}" -- "$@"
}

# Run browser profile $1 through the shared profile runner; pass remaining
# arguments to it. That runner validates membership and selector syntax.
run_e2e_shard() {
  local shard="$1"
  shift
  node "${FRONTEND_ROOT}/e2e/support/run-ci-shard.js" "${shard}" "$@"
}

# Run browser profile $1 like run_e2e_shard and write its Playwright JSON report,
# which holds per-test times, to file $2. Remaining arguments go to Playwright.
run_e2e_shard_timed() {
  local shard="$1"
  local report="$2"
  shift 2
  PLAYWRIGHT_JSON_OUTPUT_FILE="${report}" run_e2e_shard "${shard}" \
    --reporter=line,json "$@"
}

# Run eight browser profiles plus static/unit checks in an eight-worker pool.
# Optional $1 is an existing directory; when given, each profile also writes its
# per-test times to $1/browser-profile-N.json. Set CI=1, reject noncanonical
# overrides and isolate browser output directories. Remove outputs on success;
# retain failures for inspection. Return 1 when any task fails. Pool logs are
# printed before their temporary removal.
run_complete_frontend_gate() {
  local timings_dir="${1:-}"
  local output_root=""
  local shard_number=""
  local status=0

  require_canonical_frontend_environment
  export CI=1
  output_root="$(mktemp -d "${TMPDIR:-/tmp}/marl-browser-validation.XXXXXX")"
  marl_validation_init 8 frontend-validation
  for shard_number in {1..8}; do
    if [[ -n "${timings_dir}" ]]; then
      marl_validation_start \
        "Browser profile ${shard_number}/8" \
        run_e2e_shard_timed \
        "${shard_number}/8" \
        "${timings_dir}/browser-profile-${shard_number}.json" \
        --max-failures=1 \
        --output "${output_root}/profile-${shard_number}"
    else
      marl_validation_start \
        "Browser profile ${shard_number}/8" \
        run_e2e_shard \
        "${shard_number}/8" \
        --max-failures=1 \
        --output "${output_root}/profile-${shard_number}"
    fi
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
  --timings)
    shift
    if (( $# != 1 )) || [[ -z "$1" ]]; then
      echo "error: --timings requires exactly one output directory." >&2
      exit 2
    fi
    timings_dir="$1"
    if [[ "${timings_dir}" != /* ]]; then
      timings_dir="${CALLER_DIR}/${timings_dir}"
    fi
    mkdir -p -- "${timings_dir}"
    if ! timings_dir="$(cd -- "${timings_dir}" && pwd -P)"; then
      echo "error: cannot enter the --timings directory." >&2
      exit 2
    fi
    run_complete_frontend_gate "${timings_dir}"
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
