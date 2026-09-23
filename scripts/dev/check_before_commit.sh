#!/usr/bin/env bash

# Check an already frozen, nonempty staged candidate before a Codex commit.
# Usage: scripts/dev/check_before_commit.sh (no arguments). Requires Git, the
# prepared uv environment, Node/npm and installed browser dependencies. Reject
# tracked unstaged changes and nonignored untracked files. Run the complete Python
# and frontend gates, then recheck HEAD, the index tree and worktree cleanliness.
# The script reports success only for that unchanged candidate. It never stages
# or commits. git write-tree may write Git tree objects while fingerprinting.
# A caller's GIT_INDEX_FILE selects this gate's candidate only. Child validation
# uses its own Git indexes so tests can safely create separate repositories.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

# shellcheck source=scripts/dev/validation_parallel.sh
source "${SCRIPT_DIR}/validation_parallel.sh"

# Print HEAD:index-tree for this repository. No arguments. Return nonzero if
# Git cannot read HEAD or write the index tree. Tree-object creation is allowed;
# this function does not stage files or create a commit.
candidate_fingerprint() {
  local head_revision=""
  local staged_tree=""

  head_revision="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)" || return 1
  staged_tree="$(git -C "${REPO_ROOT}" write-tree)" || return 1
  printf '%s:%s\n' "${head_revision}" "${staged_tree}"
}

# Print untracked paths that Git does not ignore, one per line. No arguments.
# Return Git status; ignored private milestone files are excluded.
nonignored_untracked_files() {
  git -C "${REPO_ROOT}" ls-files --others --exclude-standard
}

# Require staged changes, no tracked unstaged changes, no nonignored untracked
# files and no staged whitespace errors. No arguments. Print a reason and return
# nonzero on failure; inspect only, with no staging or worktree changes.
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
marl_validation_start "Complete Python validation" \
  env -u GIT_INDEX_FILE "${SCRIPT_DIR}/check.sh"
marl_validation_start "Complete frontend validation" \
  env -u GIT_INDEX_FILE "${SCRIPT_DIR}/check_frontend.sh"
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
