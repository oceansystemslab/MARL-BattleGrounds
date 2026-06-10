# Quality Gates

Before commit, run the full non-mutating review gate:

```bash
scripts/dev/check.sh
```

This is equivalent to:

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

To apply local formatting and auto-fix safe lint issues, run:

```bash
scripts/dev/format.sh
```

Pre-commit hooks run fast local hygiene checks before each commit.

GitHub Actions runs CPU and GPU CI on branch pushes and pull requests.
