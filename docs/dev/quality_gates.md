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

## Mandatory Codex candidate qualification

Immediately before every Codex-authored commit, stage the exact final
candidate and run:

```bash
scripts/dev/check_before_commit.sh
```

The wrapper requires a fully staged candidate with no tracked unstaged changes
or nonignored untracked candidate files. It fingerprints those bytes, runs the
complete CPU Python and frontend/browser inventories, and rejects any candidate
mutation during validation. Any byte change after a pass invalidates that pass
and requires the complete command to be run again before committing.

Dependency installation remains a separate preparation step. The validation
scripts use the existing Python sharder and browser profile manifest as their
only test-inventory authorities; the wrapper does not define another scheduler
or duplicate test membership. Both ordinary gates force the CPU backend so a
GPU-equipped workstation cannot accidentally execute broad contributor tests
through CUDA. Both canonical gates reject deprecated platform selectors and JIT
disablement that could conflict with CPU selection or replace compiled proofs
with eager execution. The Python gate also rejects ambient pytest options that
could omit tests. The frontend gate forces Playwright's CI-only checks and rejects
ambient selector or capture variables that could reduce or alter the executed
browser inventory.

Before Codex pushes a commit, opens a PR, merges, or qualifies a release, the
exact clean commit must also pass the maintainer-only local GPU gate:

```bash
scripts/dev/check_gpu.sh
```

Ordinary contributors do not need an NVIDIA GPU. `--allow-dirty` is useful for
diagnosis but is explicitly not release-qualification evidence. A local pass
is strong CI-parity evidence, not a guarantee that GitHub-hosted runners or the
service itself will succeed; after publication, every required hosted aggregate
must still be checked.

## Evidence economics and CI runtime budget

The required push/PR pipeline initially targets an approximately six-minute
repository-controlled critical path, measured from the first hosted job's
`started_at` timestamp through the later stable aggregate's `completed_at`
timestamp. GitHub queue delay before runner start is external and reported
separately. Sharding is explicitly permitted—and expected—when it preserves an
exact, non-overlapping test inventory. Aggregate jobs retain the stable
required-check names; twenty implementation jobs may run concurrently under
the current GitHub-hosted account limit. Local validation may use the developer
workstation's additional CPU concurrency.

The twelve-minute Python and eight-minute browser matrix-job timeouts are runaway-
job safety ceilings, not performance targets. Downstream aggregates
necessarily run after their dependencies. Hosted runs `33329934258`,
`33330400572`, and `33330879589` showed that repeatedly expiring an unchanged
distribution at seven, eight, and nine minutes did not correct its load
imbalance. The final run left only Python shards 1 and 6 unfinished, with no
test failure.

The scheduler therefore smooths eight exact, fixture-safe work units from
measured overloaded shards into shards with measured headroom. It fails closed
if a future collection changes an expected source owner. The resulting local
12-way proof selects all 3,403 tests exactly once: pytest time ranges from 3:38
to 4:34 and whole-command wall time from 3:50 to 4:50. Every publication
candidate must still be checked against hosted timestamps. A material
regression beyond the approximately six-minute target requires profiling and
an actionable CI-only follow-up; ordinary timing variance does not. Rebalance
intact work units within the current twelve-Python/eight-browser, twenty-job
ceiling before changing a timeout. Once that split is exhausted, let valid
work finish under a generous safety ceiling. Do not cancel and rerun unchanged
work; adjust the ceiling only with measured headroom or another concrete
correction. Never omit tests or weaken assertions to satisfy the timing target.

The twelve Python shards and eight browser profiles are continuing ownership
obligations, not a one-time optimization. Any change that adds, removes,
renames, moves, or parameterizes tests must re-prove that the relevant inventory
is an exact, disjoint cover and review measured shard/profile elapsed times.
Rebalance intact work units first; if one test file is the irreducible hotspot,
split that file mechanically into coherent test-family files without weakening
or changing its assertions. Keep parameterized families and fixture-affinity
boundaries intact.

Hosted main run `33541558614` established why the safety ceilings require
headroom. Across three attempts, browser profile 5 completed all 30 assertions
in about 5.5 minutes but crossed the former six-minute whole-job ceiling during
process shutdown, while Python shard 4 repeatedly reached 97 percent before
crossing the former nine-minute ceiling. The DevClient authoring file moved
intact from browser profile 5 to the lighter profile 4; exact-cover tests still
forbid omission or duplication. The 12/8-minute ceilings leave ordinary runner,
setup, and teardown variance outside the six-minute performance target instead
of terminating otherwise valid work.

Main-branch CI runs use `github.run_id` in their concurrency key, so consecutive
merges cannot replace an older pending or running main check. Non-main refs
remain grouped by ref with `cancel-in-progress: true`, preserving deliberate
supersession of stale branch and pull-request work.

If measured evidence proves the current six-minute target unattainable after
safe balancing at the twelve-plus-eight, twenty-job ceiling, increase the
documented target by exactly one minute. Record the measurements and the reason
no further safe redistribution exists. Never omit tests, duplicate execution,
weaken assertions, or cancel a valid run merely because it crossed the target.
Cancel only when a concrete corrective change is ready to apply before rerun.

## Authoritative Replay and DevClient integration baseline

Commit `82077d275caef8bc3d08322e6c9f55c8d5242aec` is the accepted Replay Viewer
and Combat Debugger product baseline beneath the DevClient. A later integration
into `main` must keep this commit as an ancestor and run the Replay/DevClient
presentation, control, Oracle/Agent parity, privacy, and real-browser gates
against the integrated tree. Conflict resolution must not replace these product
bytes wholesale or silently restore older behavior from another branch.

This is a regression guard, not a permanent feature freeze. Changes explicitly
requested by the user, or deliberate additions required by later `main`
features, remain allowed. Any intentional departure from the baseline behavior
must be identified, reviewed, tested, and approved; ancestry alone is not
evidence that the accepted behavior survived the merge.

Put each assertion at the cheapest layer that can genuinely disprove the risk:

- Python owns simulator, trajectory, replay, schema, and scientific semantics.
- Node unit tests own deterministic normalization, planning, rendering inputs,
  formatting, control state, and DOM-independent accessibility contracts.
- Playwright owns only facts that require a real browser or real browser/server
  boundary: native focus, hit testing, responsive geometry, authority clearing,
  public causal flows, privacy across rendered surfaces, exact-once recovery,
  and a small representative visual baseline.
- Local maintainer GPU qualification owns CUDA backend discovery plus focused
  compiled JAX compatibility. The complete CPU suite is never rerun on the GPU
  as part of this gate, and no full GPU regression is required.

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
| Team Deathmatch task semantics | Configuration/state validation, score/termination/reward transition tests, evaluation capture/replay/event tests, rollout tests, and scripted SharedObs/NoSharedObs policy integration |
| DevClient map/scenario authoring | Strict draft parsing, compile/validation joins, digest/persistence/tamper tests, browser authoring units, and the smallest persisted reopen/load Playwright flow |
| DevClient controller or information mode | Protocol/input/service/frame tests, same-start causal proofs, truthful capture/provenance tests, and the affected real-browser selector flow |
| Actor projection version change | Projection/capture tests, explicit older-version rejection, exact actor-input export tests for the new version, and Oracle/Agent privacy parity before enabling that version in Replay Agent POV |
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

These scripts are the canonical complete local inventories. Codex runs both
through `check_before_commit.sh` immediately before every commit; a human may
also invoke either gate independently for release closeout or manual full
verification.

After Python changes have stopped, prepare the locked environment and run the
complete Python gate once:

```bash
uv sync --locked --extra dev --extra viz
scripts/dev/check.sh
```

`scripts/dev/check.sh` forces `JAX_PLATFORMS=cpu`, runs all twelve Python shards
as an exact cover, and runs Ruff format checking, Ruff lint, and Pyright once
after the environment is prepared. It does not install or update dependencies.

CI performs ordinary pytest collection in every Python shard and identifies
each atomic test family by test path, exact parent collector node ID, and
pytest's unparameterized `originalname`. Ordinary files remain single affinity
units so file-local fixtures and compiled execution paths are not repeated
across workers. A small, explicit runtime profile may extract a measured slow
family from a declared hotspot file while keeping that parameterized family
indivisible and all residual families together. A measured whole-file hotspot
may instead split at function-family boundaries only when every parameterized
family remains indivisible. Module-scoped fixtures remain affinity barriers
unless the runtime profile names one exact fixture as safe to reconstruct: that
exception requires fixed inputs, deterministic output, immutable returned
data leaves, read-only consumers, and no I/O, dynamic fixture selection, or
global mutation. Collection
rejects stale exceptions, same-name fixtures from another definition site, and
every other shared module fixture. Pytest's dynamic `request` fixture API is
prohibited inside profiled split files; collection rejects a split if a test or
any non-pytest fixture in its resolved chain declares `request`.
The hosted semantic-inventory and service-preflight
families, plus the sample-replay fixture residual, have deliberately dominant
scheduling weights so each intact proof receives a dedicated worker. Twelve
nonempty workers use deterministic weighted longest-processing-time packing,
including reserved capacity for a static gate. Collection fails closed if a
configured file or family disappears, moves, or would be assigned twice. The
profile changes CI scheduling only; it never selects a smaller test inventory.

The integrated profile was measured after the M7 merge. The scripted Team
Deathmatch NoSharedObs file is split across its 73 intact function families;
its fixed-key `class_rows` fixture is the sole repeatable module-fixture
exception. That fixture is pure, deterministic, I/O-free, and returns fresh
JAX-array values inside a read-only consumer contract. Every other module-
scoped fixture retains ordinary affinity. Reprofiling must use per-work-unit
and hosted job timings. A timing
change may adjust only measured affinity weights or intact family membership;
it must not omit a test, split a parameterized family, exceed twelve Python
workers, or displace a static gate.

After frontend changes have stopped, install the locked contributor toolchain
and pinned Chromium, then run the complete frontend/browser gate once:

```bash
npm ci --prefix web/visual_debugger
npm run install:browser --prefix web/visual_debugger
scripts/dev/check_frontend.sh
```

With no arguments, the frontend script runs format check, lint, typecheck, unit
tests, and the required Playwright E2E/visual inventory. CI runs the combined
frontend format/lint/type/unit gate inside one short browser profile, then
distributes the exact browser inventory across eight validated, nonempty
profiles. The long serial authorized-presentation install suite is split by
exact collected test title without changing its serial mode. Its causal/privacy
proof is divided only at existing scenario boundaries into three collected
tests; every original assertion remains, and each slice independently guards
the checked sample bytes. The three slices are distributed across existing
profiles and use the same setup-isolation flag. An executable list-only proof
requires the eight profiles to be a disjoint, complete cover. Each profile
retains one worker and file-local ordering.
Required CI does not retry deterministic failures, stops a red shard after its
first failure, and enforces matrix-job safety ceilings that are deliberately
larger than the current performance target. Rebalance measured work before
changing a ceiling, and never cancel unchanged valid work merely to enforce the
performance target. When safe balancing is exhausted, raise the target by one
minute with recorded evidence rather than repeatedly terminating the same valid
workload. The script does not install dependencies or update snapshots. Run it
from the exact frozen commit candidate. When a changed helper spawns a package
manager, interpreter,
generated-artifact exporter, or browser, also exercise that path once from a
clean worktree with cold local environment state; a warm developer
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

- twelve deterministic weighted-affinity Python shards with atomic
  parameterized families, fail-closed runtime-profile validation, and Ruff
  formatting, Ruff lint, and Pyright distributed across the matrix using
  locked `dev` and `viz` extras;
- eight isolated browser profiles with pinned Playwright Chromium,
  shard-qualified failure artifacts, and the combined frontend
  format/lint/type/unit gate folded into a short profile.

GitHub Actions does not target a self-hosted GPU. GPU qualification is a
deliberate local maintainer gate described in `docs/dev/gpu_sanity.md`.

Pre-commit hooks remain fast hygiene, not a substitute for the affected
behavioral proof or final closeout gates.
