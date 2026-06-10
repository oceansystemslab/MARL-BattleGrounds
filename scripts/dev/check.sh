#!/usr/bin/env bash
set -euo pipefail

uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright
