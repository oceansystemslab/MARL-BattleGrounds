#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# shellcheck source=scripts/dev/validation_parallel.sh
source "${SCRIPT_DIR}/validation_parallel.sh"

export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
export UV_NO_SYNC=1

require_canonical_python_environment() {
  if [[ -n "${JAX_PLATFORM_NAME+x}" ]]; then
    echo "error: unset deprecated JAX_PLATFORM_NAME before the canonical Python gate." >&2
    return 2
  fi
  if [[ -n "${JAX_XLA_BACKEND+x}" ]]; then
    echo "error: unset deprecated JAX_XLA_BACKEND before the canonical Python gate." >&2
    return 2
  fi
  if [[ -n "${PYTEST_ADDOPTS+x}" ]]; then
    echo "error: unset PYTEST_ADDOPTS before running the canonical Python gate." >&2
    return 2
  fi
  if [[ -n "${JAX_DISABLE_JIT+x}" ]]; then
    echo "error: unset JAX_DISABLE_JIT before running the canonical Python gate." >&2
    return 2
  fi
}

run_python_shard() {
  local shard="$1"
  shift
  uv run --no-sync pytest \
    -p scripts.dev.pytest_shard \
    "--ci-shard=${shard}" \
    -p no:cacheprovider \
    --maxfail=1 \
    "$@"
}

run_all_python_tests() {
  local shard_number=""
  local status=0

  marl_validation_init 12 python-validation
  for shard_number in {1..12}; do
    marl_validation_start \
      "Python tests ${shard_number}/12" \
      run_python_shard "${shard_number}/12"
  done
  if ! marl_validation_finish; then
    status=1
  fi
  marl_validation_cleanup
  return "${status}"
}

run_python_static() {
  local status=0

  marl_validation_init 3 python-static
  marl_validation_start "Ruff format" uv run --no-sync ruff format --check .
  marl_validation_start "Ruff lint" uv run --no-sync ruff check .
  marl_validation_start "Pyright" uv run --no-sync pyright
  if ! marl_validation_finish; then
    status=1
  fi
  marl_validation_cleanup
  return "${status}"
}

run_complete_python_gate() {
  local shard_number=""
  local status=0

  marl_validation_init 12 python-validation
  for shard_number in {1..12}; do
    marl_validation_start \
      "Python tests ${shard_number}/12" \
      run_python_shard "${shard_number}/12"
  done
  marl_validation_start "Ruff format" uv run --no-sync ruff format --check .
  marl_validation_start "Ruff lint" uv run --no-sync ruff check .
  marl_validation_start "Pyright" uv run --no-sync pyright
  if ! marl_validation_finish; then
    status=1
  fi
  marl_validation_cleanup
  return "${status}"
}

usage() {
  cat >&2 <<'EOF'
usage: scripts/dev/check.sh [--tests-only | --static-only | --shard N/12 [pytest arguments...] | --help]
EOF
}

cd -- "${REPO_ROOT}"
case "${1:-}" in
  "")
    require_canonical_python_environment
    run_complete_python_gate
    ;;
  --tests-only)
    shift
    if (( $# != 0 )); then
      echo "error: --tests-only does not accept additional arguments." >&2
      exit 2
    fi
    require_canonical_python_environment
    run_all_python_tests
    ;;
  --static-only)
    shift
    if (( $# != 0 )); then
      echo "error: --static-only does not accept additional arguments." >&2
      exit 2
    fi
    run_python_static
    ;;
  --shard)
    shift
    if (( $# < 1 )) || [[ ! "$1" =~ ^([1-9]|1[0-2])/12$ ]]; then
      echo "error: --shard requires N/12 with 1 <= N <= 12." >&2
      exit 2
    fi
    shard="$1"
    shift
    require_canonical_python_environment
    run_python_shard "${shard}" "$@"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
