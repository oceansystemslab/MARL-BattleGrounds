# Milestone 2

Milestone 2 establishes the repository, development environment, and CI
substrate for MARL-BattleGrounds.

Completed:

- src-layout Python package skeleton
- uv dependency management
- Python 3.14 lockfile
- local quality gates
- GitHub-hosted CPU CI
- self-hosted RTX 5090 GPU CI
- persistent GitHub Actions GPU runner
- JAX CUDA smoke validation
- pre-commit hooks
- VSCode development environment
- minimal development documentation

Deferred until real consumers exist:

- configuration files and config resolution
- run manifest writer
- task registry

These deferred items remain required project capabilities, but they should be
introduced alongside the first code paths that consume them instead of as
placeholder abstractions.
