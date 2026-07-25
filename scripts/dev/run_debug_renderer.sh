#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PYPROJECT_PATH="${REPO_ROOT}/pyproject.toml"
ENTRYPOINT_PATH="${REPO_ROOT}/scripts/dev/debug_renderer.py"

if [[ ! -f "${PYPROJECT_PATH}" ]]; then
  echo "error: repository root does not contain pyproject.toml: ${REPO_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${ENTRYPOINT_PATH}" ]]; then
  echo "error: visual debugger entry point is missing: ${ENTRYPOINT_PATH}" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; install uv and run 'uv sync --extra viz --extra dev'." >&2
  exit 127
fi

cd "${REPO_ROOT}"
exec uv run --project "${REPO_ROOT}" --extra viz python "${ENTRYPOINT_PATH}" "$@"
