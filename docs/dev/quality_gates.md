# Quality Gates

Use the smallest check that can disprove the change being made. A green command
is not rerun until a later edit creates a plausible impact path to it.

## Development selection

| Change | Smallest justified proof |
| --- | --- |
| Core or Python semantics | Nearest Python unit/integration tests, then targeted Ruff/Pyright |
| Debugger scene/event schema | Scene/Event V2, live-frame, audience-boundary, and choreography tests |
| Command/service/server behavior | Protocol, input, service, server, and affected real-browser case |
| Scenario trajectory | Scenario preflight/reference tests |
| Replay/POV/scenario artifact | Focused semantic, canonical-I/O, tamper, privacy, and import-isolation tests |
| Read-only replay viewer | Replay protocol/service/server/launcher tests, strict browser normalizer/controller units, and a real canonical-artifact Playwright flow |
| Evaluation-to-scene projection | Researcher/POV adapter tests plus static replay launcher smoke |
| Static Matplotlib path | Launcher plus relevant renderer smoke/scene-painter tests with `viz` |
| SVG/CSS/layout | Affected JavaScript unit test and selected Playwright case |
| Choreography/effects | Effect/animation unit tests and relevant scenario browser case |
| Tracked prose only | Link, command, and stale-text inspection; no simulator tests |

Do not run the complete Python or browser suite after every local edit. Run both
complete suites on the frozen commit candidate before committing or pushing;
focused checks are the fast feedback loop that gets the tree to that point. Do
not rerun an unchanged visual comparison merely for reassurance.

## Focused commands

Python formatting, linting, and types can target the changed paths:

```bash
uv run ruff format --check <paths>
uv run ruff check <paths>
uv run pyright <paths>
```

Frontend source checks:

```bash
npm run format:check --prefix web/visual_debugger
npm run lint --prefix web/visual_debugger
npm run typecheck --prefix web/visual_debugger
npm run test:unit --prefix web/visual_debugger
```

Select one browser case with the Playwright CLI or `--grep` when only one
interaction/layout path changed.

Replay changes should include a real canonical-artifact browser case rather
than only synthetic JavaScript objects. That case must exercise the injected
replay HTTP routes, audience-matching frame/timeline roots, settled seek and
reconnect behavior, exact-next animation, endpoint pause, and actor-POV
non-disclosure. Keep the replay subprocess on the CPU/import-isolation path so
an accidental simulator or JAX import fails the test.

## Complete closeout gates

After Python changes have stopped, prepare the locked environment and run the
complete Python gate once:

```bash
uv sync --locked --extra dev --extra viz
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

`scripts/dev/check.sh` runs the four Python review commands after the
environment is prepared.

After frontend changes have stopped, install the locked contributor toolchain
and pinned Chromium, then run the complete frontend/browser gate once:

```bash
npm ci --prefix web/visual_debugger
npm run install:browser --prefix web/visual_debugger
scripts/dev/check_frontend.sh
```

The frontend script runs format check, lint, typecheck, unit tests, and the
complete Playwright E2E/visual suite. It does not install dependencies or update
snapshots. Run it from the exact frozen commit candidate. When a changed helper
spawns a package manager, interpreter, generated-artifact exporter, or browser,
also exercise that path once from a clean worktree with cold local environment
state; a warm developer environment can suppress first-run output and setup
behavior that CI will encounter.

If a later fix occurs, rerun only the affected gate:

- CSS-only fix: frontend static checks and affected browser cases;
- protocol fix: focused Python protocol/service tests, frontend contract
  checks, and affected E2E;
- documentation fix: inspection only;
- cross-cutting fix: repeat every gate it can invalidate.

## Visual baselines

Normal comparison:

```bash
npm run test:visual --prefix web/visual_debugger
```

Intentional update:

```bash
npm run test:visual:update --prefix web/visual_debugger
```

Snapshot update is never an automatic repair. Review each diff at original
resolution, confirm the paired semantic assertions, and commit only deliberate
changes. CI uploads Playwright failure artifacts.

## Automation

GitHub Actions runs:

- Python tests, Ruff, and Pyright with locked `dev` and `viz` extras so
  Matplotlib coverage cannot silently skip; and
- a separate Node 24 frontend/browser job with `npm ci`, pinned Playwright
  Chromium, all frontend checks, and failure artifacts; and
- the existing self-hosted GPU sanity workflow for the CUDA environment and
  JAX GPU backend.

Pre-commit hooks remain fast hygiene, not a substitute for the affected
behavioral proof or final closeout gates.
