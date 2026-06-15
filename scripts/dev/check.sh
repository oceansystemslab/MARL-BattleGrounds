#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v
uv run ruff format --check .
uv run ruff check .
uv run pyright
