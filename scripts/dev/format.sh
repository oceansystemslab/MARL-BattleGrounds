#!/usr/bin/env bash

# Format Python files and apply Ruff's safe lint fixes in the current directory.
# Usage from the repository root: scripts/dev/format.sh. Requires uv and the
# project's Ruff dependency. Both commands may edit files; inspect the diff after
# running them. uv may prepare the environment. Extra shell arguments are unused.
# Stop on the first failing command. This script does not stage or commit.
set -euo pipefail

uv run ruff format .
uv run ruff check --fix .
