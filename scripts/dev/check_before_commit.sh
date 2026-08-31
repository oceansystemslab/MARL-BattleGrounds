#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# shellcheck source=scripts/dev/validation_parallel.sh
source "${SCRIPT_DIR}/validation_parallel.sh"

candidate_fingerprint() {
  local head_revision=""
  local staged_tree=""

  head_revision="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)" || return 1
  staged_tree="$(git -C "${REPO_ROOT}" write-tree)" || return 1
  printf '%s:%s\n' "${head_revision}" "${staged_tree}"
}

nonignored_untracked_files() {
  git -C "${REPO_ROOT}" ls-files --others --exclude-standard
}

require_frozen_staged_candidate() {
  local untracked=""

  if git -C "${REPO_ROOT}" diff --cached --quiet --exit-code HEAD --; then
    echo "error: the Codex pre-commit gate requires a nonempty staged candidate." >&2
    return 2
  fi
  if ! git -C "${REPO_ROOT}" diff --quiet --exit-code --; then
    echo "error: tracked unstaged changes must be staged or reverted first." >&2
    return 2
  fi
  if ! untracked="$(nonignored_untracked_files)"; then
    echo "error: could not inspect nonignored untracked files." >&2
    return 1
  fi
  if [[ -n "${untracked}" ]]; then
    echo "error: nonignored untracked files must be staged or removed first:" >&2
    printf '%s\n' "${untracked}" >&2
    return 2
  fi
  git -C "${REPO_ROOT}" diff --cached --check
}

if (( $# != 0 )); then
  echo "usage: scripts/dev/check_before_commit.sh" >&2
  exit 2
fi

require_frozen_staged_candidate
if ! before="$(candidate_fingerprint)"; then
  echo "error: could not fingerprint the staged candidate." >&2
  exit 1
fi

status=0
marl_validation_init 2 before-commit
marl_validation_start "Complete Python validation" "${SCRIPT_DIR}/check.sh"
marl_validation_start "Complete frontend validation" "${SCRIPT_DIR}/check_frontend.sh"
if ! marl_validation_finish; then
  status=1
fi
marl_validation_cleanup

if ! require_frozen_staged_candidate; then
  echo "error: validation changed the candidate worktree or index." >&2
  exit 1
fi
if ! after="$(candidate_fingerprint)"; then
  echo "error: could not fingerprint the validated candidate." >&2
  exit 1
fi
if [[ "${before}" != "${after}" ]]; then
  echo "error: staged candidate bytes changed during validation; rerun the gate." >&2
  exit 1
fi
if (( status != 0 )); then
  echo "error: the complete local pre-commit validation failed." >&2
  exit "${status}"
fi

echo "Pre-commit validation passed for ${after}."
