#!/usr/bin/env bash

# Launch the DevClient from this checkout, from any working directory.
# Usage: scripts/dev/run_dev_client.sh --help. Forward all arguments to
# scripts/dev/debug_renderer.py. A literal --static argument asks uv to include
# the viz extra. Requires uv; uv may install/sync dependencies. The Python entry
# point owns server, file and rendering effects. Missing project/entrypoint files
# exit 2; missing uv exits 127. exec passes the Python process's final status back.
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
