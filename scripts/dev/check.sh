#!/usr/bin/env bash

# Run the repository's Python correctness and static checks on the CPU.
# Usage: scripts/dev/check.sh --help. With no options, run all twelve Python
# shards, Ruff format/lint and Pyright with at most twelve workers. --tests-only
# and --static-only select those groups; --shard N/12 accepts extra pytest args.
# Run after preparing the locked uv environment; this command never syncs it.
# The shard plugin owns test assignment. Output includes each task's exit code and
# elapsed seconds. Failures return nonzero. No commit or GPU speed claim is made.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# shellcheck source=scripts/dev/validation_parallel.sh
source "${SCRIPT_DIR}/validation_parallel.sh"

export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
export UV_NO_SYNC=1

# Reject environment variables that could disable JIT, alter collection or
# select a deprecated JAX backend. No arguments; print the first error and return 2.
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

# Run shard selector $1 through the sole pytest assignment plugin; forward the
# remaining arguments to pytest. Disable its cache and stop after the first failure.
# The outer command validates the selector. Return pytest/uv status.
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

# Run all twelve Python shards in parallel, print their results and clean worker
# logs. No arguments. Return 1 if any shard fails; otherwise return 0.
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

# Run Ruff format-check, Ruff lint and Pyright concurrently without fixing files.
# No arguments. Print results, remove worker logs and return 1 if any check fails.
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

# Run all twelve shards and three static checks through the shared twelve-slot
# pool. No arguments. Print every result, clean logs and return 1 on any failure.
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

# Print accepted command forms to stderr. No arguments or state changes.
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
