# Quality Gates

Before commit, run:

uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright

Pre-commit hooks run fast local hygiene checks before each commit.

GitHub Actions runs CPU and GPU CI on branch pushes and pull requests.
