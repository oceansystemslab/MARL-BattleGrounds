# Quality Gates

Before commit, run:

uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright

All gates must pass before changes are committed.
