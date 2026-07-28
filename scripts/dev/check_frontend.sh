#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
FRONTEND_ROOT="${REPO_ROOT}/web/visual_debugger"

if ! command -v npm >/dev/null 2>&1; then
  echo "error: Node.js 24 and npm are required for frontend contributor checks." >&2
  exit 127
fi

npm run format:check --prefix "${FRONTEND_ROOT}"
npm run lint --prefix "${FRONTEND_ROOT}"
npm run typecheck --prefix "${FRONTEND_ROOT}"
npm run test:unit --prefix "${FRONTEND_ROOT}"
npm run test:e2e --prefix "${FRONTEND_ROOT}"
