# Quality Gates

Use the smallest check that can disprove the change being made. A green command
is not rerun until a later edit creates a plausible impact path to it.

## Lean validation contract

Excellence and efficiency are complementary requirements. Every review, test,
or audit must answer a distinct question that could change an engineering
decision. Do not add ceremony merely to increase the volume of evidence.

- Define each checkpoint by one observable user or system outcome.
- During implementation, run one focused regression and one adjacent
  compatibility check. Run the literal CI gate once, after the candidate is
  frozen.
- Use one independent reviewer at the checkpoint boundary. Do not commission
  overlapping audits, repeat hash freezes, or rerun unchanged evidence.
- Stop a broad gate at its first actionable failure when possible. Diagnose and
  repair that failure before spending time on the remainder.
- Do not call a checkpoint complete until the exact applicable CI commands pass
  on the candidate bytes. A candidate push may exercise hosted CI, but it is
  evidence collection—not acceptance—until every required aggregate is green.
- Prefer a smaller proof with a precise failure signal over a larger proof that
  obscures failures or consumes time without covering a new risk.
- Treat every test as an ongoing maintenance and runtime liability as well as
  an asset. If it has no serious purpose, duplicates cheaper evidence, or its
  marginal protection does not justify its cost, delete it. Historical effort,
  incidental uniqueness, and fear of reducing a test count are not reasons to
  keep it.

Evaluate every product checkpoint against the four North Stars, briefly and
concretely:

- **Researcher Usability:** can a researcher understand and complete the public
  task without misleading controls or hidden prerequisites?
- **Sample Efficiency:** does the policy receive the necessary authorized,
  same-epoch information without avoidable ambiguity or noise?
- **Tactical Depth:** are all advertised and unmasked decisions meaningful,
  reachable, and strategically distinct?
- **Software Engineering:** is the result correct, maintainable, reproducible,
  and validated with the least costly sufficient evidence?

Efficiency is part of the Software Engineering North Star. Duplicate proof and
unbounded serialized test time are process defects, not signs of rigor.

## Evidence economics and CI runtime budget

The required push/PR pipeline has a 10–15 minute normal critical-path budget.
Sharding is explicitly permitted—and expected—when it preserves an exact,
non-overlapping test inventory. Aggregate jobs retain the stable required-check
names; the implementation jobs may run in parallel underneath them.

Put each assertion at the cheapest layer that can genuinely disprove the risk:

- Python owns simulator, trajectory, replay, schema, and scientific semantics.
- Node unit tests own deterministic normalization, planning, rendering inputs,
  formatting, control state, and DOM-independent accessibility contracts.
- Playwright owns only facts that require a real browser or real browser/server
  boundary: native focus, hit testing, responsive geometry, authority clearing,
  public causal flows, privacy across rendered surfaces, exact-once recovery,
  and a small representative visual baseline.
- GPU CI owns backend discovery plus focused compiled JAX compatibility. The
  complete CPU suite is not rerun on the GPU for every push; a complete GPU
  regression is scheduled or explicitly requested.

Do not keep a browser test merely because one incidental CSS value or catalog
member is unique. Exhaustive mechanic cross-products, repeated viewport tours,
wall-clock animation thresholds, and raw-only synthetic main-path fixtures are
not required-gate evidence when a lower layer proves the contract more directly.
A browser case must protect a named North Star, require browser-native behavior,
and justify its process/setup cost. Otherwise move it down or delete it.

Required gates fail fast on the first actionable test failure. Missing-element
actions use a short timeout instead of consuming the whole test timeout. Green
coverage remains complete because fail-fast changes only red-run work, not the
test inventory executed on a passing candidate.

## Development selection

| Change | Smallest justified proof |
| --- | --- |
| Core or Python semantics | Nearest Python unit/integration tests, then targeted Ruff/Pyright |
| Debugger scene/event schema | Scene/Event V2, live-frame, audience-boundary, and choreography tests |
| Command/service/server behavior | Protocol, input, service, server, and affected real-browser case |
| Scenario trajectory | Scenario preflight/reference tests |
| Replay/POV/scenario artifact | Focused semantic, canonical-I/O, tamper, privacy, and import-isolation tests |
| Read-only replay viewer | Replay protocol/service/server/launcher tests, strict browser normalizer/controller units, and a real canonical-artifact Playwright flow |
| Live replay recording/handoff | Recording/replay-I/O/service/router/launcher tests, strict lifecycle controls, and real T0/prefix/endpoint/recovery/Exit/Ctrl-C/two-tab/POV Playwright flows |
| Evaluation-to-scene projection | Researcher/POV adapter tests plus static replay launcher smoke |
| Static Matplotlib path | Launcher plus relevant renderer smoke/scene-painter tests with `viz` |
| SVG/CSS/layout | Affected JavaScript unit test and selected Playwright case |
| Choreography/effects | Effect/animation unit tests and relevant scenario browser case |
| Tracked prose only | Link, command, and stale-text inspection; no simulator tests |

Do not run the complete Python or browser suite after every local edit. Focused
checks are the fast feedback loop. Execute the complete required inventory once
for the frozen candidate, using the parallel CI shards when a local monolithic
run would only duplicate coverage and delay feedback. A candidate commit may be
pushed to exercise those isolated runners, but the checkpoint is not accepted
until every aggregate gate is green. Do not repeat an unchanged suite or visual
comparison merely for reassurance.

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

Recording changes must exercise the production `--record-replay` launcher, not
only an in-memory recorder fixture. The real-browser gate must load the saved
replay and metric sidecar through public contracts and prove frame-zero handoff,
complete versus open-prefix closeout, restart/discard fencing, immutable-byte
Retry/Save As recovery, durable Exit and Ctrl-C, two-tab stale authority, POV
non-disclosure, strict console/page-error collection, and subprocess/temp-file
cleanup. A mocked provenance test does not replace one real host discovery run;
runtime strings may differ across CPU/CUDA/PJRT installations.

## Complete local closeout gates

These commands are available for release closeout, manual full verification,
or environments without the CI shard matrix. They are not an instruction to
duplicate the same complete inventory locally before every push.

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

With no arguments, the frontend script runs format check, lint, typecheck, unit
tests, and the required Playwright E2E/visual inventory. CI runs a Node-only
style/type job and a separate unit job with the minimal Python environment its
fixture exporters require, then distributes the exact browser inventory across
eight validated, nonempty file groups. Each browser group retains one worker
and file-local ordering. Required CI does not retry deterministic failures,
stops a red shard after its first failure, and targets a 15-minute job ceiling.
The script does not install dependencies or update snapshots. Run it from the
exact frozen commit candidate. When a changed helper spawns a package manager,
interpreter, generated-artifact exporter, or browser, also exercise that path
once from a clean worktree with cold local environment state; a warm developer
environment can suppress first-run output and setup behavior that CI will
encounter.

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

- three deterministic, whole-file Python test shards plus a parallel Ruff and
  Pyright job with locked `dev` and `viz` extras;
- a Node-only style/type job, a Python-backed frontend unit job, and eight
  isolated browser groups with pinned Playwright Chromium and shard-qualified
  failure artifacts; and
- focused self-hosted GPU sanity for the CUDA backend and compiled environment
  contracts, with the complete GPU regression scheduled or manually requested.

Pre-commit hooks remain fast hygiene, not a substitute for the affected
behavioral proof or final closeout gates.
