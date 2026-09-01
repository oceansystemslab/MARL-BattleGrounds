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
  echo "error: DevClient entry point is missing: ${ENTRYPOINT_PATH}" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to run the DevClient." >&2
  exit 127
fi

UV_EXTRA_ARGS=()
for argument in "$@"; do
  if [[ "${argument}" == "--static" ]]; then
    UV_EXTRA_ARGS=(--extra viz)
    break
  fi
done

cd "${REPO_ROOT}"
exec uv run --project "${REPO_ROOT}" "${UV_EXTRA_ARGS[@]}" python "${ENTRYPOINT_PATH}" "$@"
