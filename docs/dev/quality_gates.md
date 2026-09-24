# Quality Gates

Use focused checks while editing and complete gates when qualifying a finished
candidate. Each check must answer a concrete question: does the behavior work,
do callers remain compatible, is the information boundary intact, or is the GPU
workflow efficient? A passing syntax check proves none of the other results.

## Lean validation contract

Pair each implementation checkpoint with its nearest correctness proof, affected
cost measurements and [documentation review](documentation_standard.md). Avoid
rerunning unchanged evidence unless a later edit can invalidate it. Follow the
active packet's review requirements; M8 requires two independent component
reviews and a separate complete-workflow review. Reviewers inspect fresh source
and raw evidence before reading each other's verdicts.

Check all four North Stars:

- **Researcher Usability:** the complete public task has clear names, sensible
  defaults, accurate hover text and no unnecessary setup.
- **Sample Efficiency:** actors receive their permitted information at the right
  decision time, without avoidable input or action ambiguity. Actual learning
  curves are still needed to prove sample efficiency.
- **Tactical Depth:** supported choices can express meaningful team behavior.
  A successful scripted example does not prove that a policy learned it.
- **Software Engineering:** behavior, ownership, compatibility and evidence are
  clear. The implementation avoids unnecessary work and storage.

GPU execution with JAX is the performance target. CPU tests check correctness,
compatibility and contributor CI; their elapsed times guide shard balancing,
not simulator speed claims. GPU cost evidence separates setup, compilation and
synchronized warmed execution and includes affected memory, transfer, recording
and serialization costs. Use the same inputs and retained outputs for a fair
comparison. Future GPU environment batches are 32, 64, 128, 512 and 1024 only;
standalone scalar and odd-batch edge checks remain on CPU. Follow the shared
[GPU efficiency protocol](gpu_sanity.md#gpu-efficiency-protocol), including its
required alignment of older GPU test routes before their next run.
Report absolute numbers as well as changes from the reference. A speedup alone
does not resolve a known material waste of computation, memory, transfers or
storage; use proportionate review and measurements to resolve it before acceptance.
The user's 2026-09-16 clarification rejects automatic multi-hour GPU campaigns
at each step. Reuse unaffected evidence, choose the smallest useful workload and
profile only when a concrete unresolved cost warrants it. The five GPU sizes are
allowed choices, not a compulsory matrix for every packet. Separate commit and
release gates remain required.
Every future closeout must give an honest interpretation: what the tests and
absolute numbers establish, practical strengths and concerns, missing evidence
and the next useful check. Counts, PASS labels and percentage changes alone are
not an assessment of the project's goals.

## Mandatory Codex candidate qualification

Do not commit while implementing or reviewing an unfinished packet. When a commit
is explicitly authorized, finish all mutating fixes, stage exactly the intended
candidate and leave no tracked unstaged or nonignored untracked candidate files.
Private milestone documents remain ignored and unstaged. Then run:

```bash
scripts/dev/check_before_commit.sh
```

This wrapper fingerprints the frozen candidate and runs the full Python and
frontend gates. It rejects missing prerequisites, failures and candidate changes.
If the caller selects a separate Git index with `GIT_INDEX_FILE`, only the
candidate checks use it. Python and frontend checks do not inherit that setting,
so tests that create their own repositories use their own indexes. The two raw
upstream value-normalization fixtures retain their exact bytes; `.gitattributes`
exempts only their trailing spaces from Git's whitespace check.
Run the complete gate once for unchanged candidate files and relevant test inputs.
Before committing, verify that the staged files match the tested files, the base
revision and dependencies still match, and no intended file was omitted. Staging
or committing the same files does not require another run. Changed candidate files
or relevant test inputs require a new complete gate. Private notes and evidence
alone do not invalidate it. Never substitute a partial check or `--no-verify`.

Before pushing, opening a PR, merging or qualifying a release, the exact clean
commit must also pass:

```bash
scripts/dev/check_gpu.sh
```

`--allow-dirty` is a diagnostic route, not clean-commit qualification. Ordinary
contributors need no NVIDIA GPU. Hosted aggregate checks must still pass after
publication; a local pass cannot guarantee a remote service's result.

## Evidence economics and CI runtime budget

Keep twelve nonempty Python shards and eight nonempty browser profiles. Their
sole assignment authorities are [pytest_shard.py](../../scripts/dev/pytest_shard.py)
and [ci-shards.json](../../web/visual_debugger/e2e/ci-shards.json). Do not copy
membership into another scheduler.

Every test addition, removal, rename, move, split or parametrization change must
include an exact-cover audit and review of measured per-shard/profile times.
Preserve deterministic assignment, complete disjoint coverage, parameter-family
atomicity, fixture affinity and browser file/serial order. Rebalance measured
intact units first. Split a proven oversized file only into coherent groups,
without changing or weakening its assertions.

The current hosted target is about six minutes across the repository-controlled
critical path at the twenty-job ceiling. Measure from the first job start to the
last required aggregate completion; report queue delay separately. If safe
balancing cannot meet that target, document the evidence and raise it by exactly
one minute. Never omit tests, duplicate them or cancel unchanged valid work to
meet a timing target. Cancel only when a concrete correction is ready for rerun.

The following historical timing and CI records explain the scheduler's evolution.
Their test counts and individual distributions are historical, not the current
collection. Current candidate qualification must collect and measure again.

CI jobs deliberately configure no explicit job timeout. Hosted runner setup,
dependency installation, cache handling, and teardown are variable and count
toward `timeout-minutes`; finite repository-configured ceilings repeatedly
cancelled otherwise valid jobs without reporting a test failure. Omitting the
key leaves GitHub's default and platform-enforced maximum in effect. Downstream
aggregates necessarily run after their dependencies. Hosted runs `33329934258`,
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
ceiling while allowing valid work to finish. Do not cancel and rerun unchanged
work. Never omit tests or weaken assertions to satisfy the timing target.
Since 22 September 2026 the scheduler uses measured per-file costs instead of
those hand-picked moves; see [Rebalancing the test shards](#rebalancing-the-test-shards).

The twelve Python shards and eight browser profiles are continuing ownership
obligations, not a one-time optimization. Any change that adds, removes,
renames, moves, or parameterizes tests must re-prove that the relevant inventory
is an exact, disjoint cover and review measured shard/profile elapsed times.
Rebalance intact work units first; if one test file is the irreducible hotspot,
split that file mechanically into coherent test-family files without weakening
or changing its assertions. Keep parameterized families and fixture-affinity
boundaries intact.

Hosted main run `33541558614` established why workflow-level timeouts are an
unsuitable performance control. Across three attempts, browser profile 5
completed all 30 assertions in about 5.5 minutes but crossed the former
six-minute whole-job ceiling during process shutdown, while Python shard 4
repeatedly reached 97 percent before crossing the former nine-minute ceiling.
The DevClient authoring file moved intact from browser profile 5 to the lighter
profile 4; exact-cover tests still forbid omission or duplication. Current CI
uses measured timings and shard balancing to enforce the performance target;
it does not terminate jobs through repository-configured timeouts.

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


## Rebalancing the test shards

The Python scheduler weighs each test file by its measured cost in seconds. The
costs live in one table, [pytest_shard_costs.json](../../scripts/dev/pytest_shard_costs.json),
which [pytest_shard.py](../../scripts/dev/pytest_shard.py) loads. A file that is
too big for one shard may be split at test-function boundaries: the table
lists its slow functions under `split_families` and the rest of the file under
`split_residuals`. A file missing from the table, such as a new one, costs one
unit per collected test until the next refresh. Entries for files or functions
that no longer exist are ignored, so renaming or deleting tests never stops a
shard. `reserved_seconds` keeps 50 seconds free on shard 12 because hosted CI
runs Pyright there before its tests. The numbers come from the tool below; do
not edit them by hand.

Refresh the table when shard times drift apart or after adding or moving heavy
tests. Use timings from a gate where every shard passed; the tool refuses
missing or failed reports, duplicate cases, invalid durations and any difference
from the exact current test collection, including parameter cases. It checks
the new table with the scheduler before replacing the old one. JUnit files
prove which tests ran; they do not prove that the gate's static checks passed:

```bash
scripts/dev/check.sh --timings /tmp/shard-timings            # full Python gate, saves per-test times
uv run --no-sync python scripts/dev/shard_costs.py update /tmp/shard-timings
uv run --no-sync python scripts/dev/shard_costs.py plan       # predicted seconds per shard
```

`update` sums each file's measured seconds, splits files larger than half an
average shard where the scheduler allows it, rewrites the table and prints the
predicted seconds of every shard. It warns about a file whose test functions
share module fixtures, which the scheduler cannot split (move some tests into
a new file), about one test function that alone is too large (split its
parameter cases into two functions or make it faster). After adding or renaming
tests, collect fresh passing timings before updating; old reports cannot prove
the new collection. If a split file later gains a shared module fixture,
every shard stops at collection; run `update` again with the last good timings
folder, or delete that file's `split_families` and `split_residuals` entries.
A malformed table also stops every shard; restore it with
`git checkout -- scripts/dev/pytest_shard_costs.json`.

Browser profiles are listed by hand in
[ci-shards.json](../../web/visual_debugger/e2e/ci-shards.json). A profile runs
its files whole, or only the tests whose titles it lists, across all of its
files; adding or renaming a test in a title-selected file needs a manifest
edit. Measure and read the profiles with:

```bash
scripts/dev/check_frontend.sh --timings /tmp/shard-timings
uv run --no-sync python scripts/dev/shard_costs.py browser /tmp/shard-timings
```

The browser report sums per-test time only. It leaves out each file's shared
`beforeAll` setup and worker start-up, which cost some profiles one to two
minutes, so also compare the whole-profile seconds the gate prints. Keep tests
that depend on each other's server or viewer state in one profile. A file whose
tests reset shared state may be split, but each profile then repeats its setup:
control-parity is split this way, and profile 7 pays about 65 seconds for the
replay viewer's setup. Keep the environment-setting profile 3 unchanged. Hosted
CI runs the frontend static and unit checks with profile 3, the lightest. The
exact-cover test in `web/visual_debugger/tests/ci-shards.test.js` proves every
browser test runs exactly once; update its layout checks with the manifest.

Measurement of 22 September 2026 on a 32-CPU workstation, Python and browser
gates running together. All times are whole-command seconds. "Before" is the
old hand-kept layout; "after" is the measured layout in the first full gate
that used it. Three review agents were using the machine during that gate, so
every shard ran about 8 percent slower than in the measuring run.

| Shard | Python before | Python after | Browser before | Browser after |
| --- | --- | --- | --- | --- |
| 1 | 744 | 1,216 | 340 | 423 |
| 2 | 794 | 1,126 | 295 | 392 |
| 3 | 741 | 1,212 | 225 | 223 |
| 4 | 1,425 | 1,203 | 552 | 401 |
| 5 | 941 | 1,078 | 259 | 264 |
| 6 | 831 | 1,088 | 581 | 364 |
| 7 | 1,477 | 1,104 | 221 | 441 |
| 8 | 742 | 1,102 | 288 | 400 |
| 9 | 1,029 | 955 | | |
| 10 | 1,247 | 1,174 | | |
| 11 | 1,234 | 958 | | |
| 12 | 1,012 | 992 | | |

The gate waits for its slowest shard: 1,477 seconds before and 1,216 seconds
after, about 20 minutes instead of about 25. The slowest shard went from 45
percent above the average to 10 percent above it. The average itself is the
floor: balancing cannot go below it, only removing work or adding machines can.
Each Python shard also spends about 60 to 90 seconds collecting and starting
before its tests.

## Authoritative Replay and DevClient integration baseline

Commit `82077d275caef8bc3d08322e6c9f55c8d5242aec` is the accepted product baseline
beneath DevClient. Keep it as an ancestor during integration and verify that its
supported presentation, control, audience and replay behavior survives. This is
a regression guard, not a ban on approved later changes. Ancestry alone is not a
behavioral test.

Place tests at the lowest layer that can prove the requirement:

| Layer | Responsibility |
| --- | --- |
| Python | Simulator trajectories, numerical wrappers, information boundaries, schemas, recording and scientific calculations |
| Node unit tests | Normalization, planning, display inputs, formatting, controls and DOM-independent accessibility |
| Playwright | Real focus/hit testing, browser geometry, complete client/server flows, authority clearing, recovery and selected visual baselines |
| Local GPU | Focused compiled JAX correctness plus separately declared workflow cost measurements |

Do not duplicate an exhaustive numerical cross-product in browser tests. Each
browser case must protect behavior that needs a real browser. The GPU correctness
gate does not run the complete CPU suite again.

Tests have no time limits. Playwright uses `timeout: 0` for whole tests,
actions, navigation and assertions, and Python tests wait without `timeout=`
or deadlines. A slow step on a busy machine then waits instead of failing. A
hung test stays hung: find and fix its cause. Only three kinds of limit stay:
waits that are meant to expire (for example, "nothing happens within 0.1 s"),
product timing settings under test, and checks of the product's own speed.
Canonical gates stop a red shard at its first failure; successful runs still
execute the complete inventory.

No time limit does not mean waiting for something that can never come. A wait
for a result that a child process, thread or server must produce also watches
that producer. While the producer lives, the test keeps waiting. If the
producer dies, the test checks for the result once more, since it may have
arrived just before the death. If it is still missing, the test fails at once
and reports how the producer ended, such as its exit code and output. Cleanup
after such a failure signals only processes that the test has proven it owns:
for example, by an identity recorded while the process was surely running, or
by a private token given to the process when it started. It never signals a
bare process ID or group number, because the system may have reused that number
for an unrelated process. The report names the original failure first, even
when cleanup meets errors of its own.

## Development selection

Choose the proof from the changed contract and its callers. Preserve generic
custom research even when official eligibility is stricter. For example,
[Amendment A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution)
requires official SharedObs availability to equal the configured active,
same-team, off-diagonal roster matrix in every frame. Dead configured teammates
remain authorized sources, with lifecycle-zeroed sensor material.

The scenario builder and validator must reach one replay-level authority.
Prove rejection of forbidden modes/projections, stable unauthorized subsets and
later-frame changes, with the offending frame identified. A 1v1 all-false matrix
is valid when it exactly matches the roster. Generic allowed subsets and
NoSharedObs remain supported custom/historical paths, not official evidence.
Use recorded projection versions; do not reinterpret old feature columns.

### Reactive controller replacement gates

The supported controllers are Reactive TDM ALPHA and BETA under the recorded
versions of [A30](../design/specification_amendments.md#a30-reactive-tdm-and-specialist-scenario-controllers),
[A34](../design/specification_amendments.md#a34-reactive-tdm-alpha-and-beta),
[A35](../design/specification_amendments.md#a35-reactive-tdm-wall-steering) and
[A42](../design/specification_amendments.md#a42-reactive-tdm-fixes-at-blocked-walls).
Older version-specific witness results remain historical evidence. Use the
current implementation and descriptors when qualifying a new candidate.

Check each class's choices, tie handling, exact action masks, permitted inputs
and RNG use. BETA preserves non-Rogue ALPHA behavior and uses its declared
observed-class pursuit priority; generic behavior must not branch on scenario ID.
Test wall steering, boundary/corner behavior, slowed movement, body screening,
prey/self exclusion, overlap escape and stationary/moving blockers through real
trajectories. Scenario 5's accepted glancing contact includes the 45-degree
boundary and geometry-tolerance cases. Disclose remaining congestion; these
checks do not prove unbeatable pursuit or deadlock-free navigation.

Keep selector rejection, exact resets, current input/action epochs and saved
controller identities coherent through service, HTTP and browser callers.
Reject unsupported reactive NoSharedObs choices and retired live literals.
Existing witness divergence requires investigation, not automatic replacement
of expected results. Measure affected policy cost on GPU with representative
obstacles. Preserve separately qualified geometry and replay semantics.

### Scenario-ablation activation gates

[A26](../design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations)
and [A36](../design/specification_amendments.md#a36-submission-roadmap-approved-tdm-content-and-m7-closeout)
separate accepted TDM content from later learning evidence. The accepted content
is eight scenarios; older twelve-scenario wording is historical. Scenario schema
validity must not hard-code one suite's horizon or population.

Before a confirmatory ablation, freeze one behavior claim, primary endpoint,
at most two supporting margins, full method and one ablation, scenario revision,
exact start/roster, controller, actor-input contract, seeds, spawn conditions,
training budget and checkpoint-selection rule. Use independent trained pairs;
agents, steps and episodes are nested observations, not independent learners.
The same immutable deterministic reactive pressure controller and permitted
inputs must apply to both treatments. A recorded action tape is insufficient.

Protect the complete scenario content closure from adaptive training and model
selection: maps, starts, configurations, controllers, seeds, endpoints, replays
and feedback. Check content-addressed manifests, including embedded maps. This
restriction covers gradients, imitation, curricula, prompts, tuning, checkpoint
selection and other adaptation. Scenarios are public, not secret. Reproduction
and honest provenance enforce eligibility; failed reproduction alone does not
prove misconduct. Scenario results never supply tournament Elo credit.

### Big 12 activation gates

The accepted future design uses a monthly immutable snapshot containing exactly
twelve controller versions, or those twelve plus one challenger. A growing
Baseline Library does not enlarge the official tournament. Larger populations
belong to custom configurations. The older weekly/100-game plan in
[A27](../design/specification_amendments.md#a27-rolling-big-12-and-baseline-library-governance)
is historical where it conflicts with this direction. This design is not a
claim that the official runner or a qualified released bundle exists today.

Prove 66 unordered incumbent matchups and, with a challenger, 12 additional
matchups. Verify compatible saved games and requested report coverage before
execution. Resolve one uniform budget for every matchup, balanced over maps
and complete spawn pairs. The official numerical budget remains an approval
gate. A different research budget cannot qualify promotion. Reuse selection
must be declared independently of outcomes; insufficient coverage stops with
an explicit fresh-run route.

Recompute ratings jointly from qualifying game records; old Elo values are not
credit. Preserve original game identities/seeds/roles and count each once.
Use unrounded ratings for shared competition ranks. Test analysis equality from
identical records; fresh random draws need not reproduce the same games.

Admission runs collect full metrics for every challenger game. Public calls keep
priority metrics by default. Promotion publishes only after all 66 retained
matchups have complete outcomes, priority and full records; selected replays
remain separate. Failed/incomplete admission leaves membership unchanged.
Refit the retained twelve after promotion, retain removed entrants historically,
and keep active researcher runs bound to their resolved snapshot.

Keep tied-incumbent eviction, revised-method admission, scientific eligibility,
resource limits and the official numerical budget as explicit later gates.
Do not invent their values during implementation or treat fixture snapshots as
released qualification. Frozen learned systems require validation-only model
selection and independent learning evidence; provider feasibility has its own
measurement gate. Local execution cannot publish or admit a system.

## Focused commands

After preparing dependencies, run the relevant Python files, static checks or
one existing shard:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_environment.py
uv run --no-sync ruff format --check src tests scripts examples
uv run --no-sync ruff check src tests scripts examples
uv run --no-sync pyright
scripts/dev/check.sh --shard 1/12
```

For browser source changes, use the existing commands:

```bash
npm run format:check --prefix web/visual_debugger
npm run lint --prefix web/visual_debugger
npm run typecheck --prefix web/visual_debugger
npm run test:unit --prefix web/visual_debugger
```

Select an affected Playwright case for a browser-specific change. Replay or
recording changes need a real saved-artifact flow through public routes, including
relevant initial/terminal/prefix behavior, exact identity, recovery, audience
limits and cleanup. Synthetic objects alone do not prove that boundary. Preserve
historical sidecar readers; current V3 replay saving does not require a metric
sidecar. Cold launcher/subprocess checks catch setup assumptions hidden by a
warm development environment.

## Complete local closeout gates

Prepare the locked dependencies separately, then run the complete inventories:

```bash
uv sync --locked --extra dev --extra viz --extra training
scripts/dev/check.sh
npm ci --prefix web/visual_debugger
npm run install:browser --prefix web/visual_debugger
scripts/dev/check_frontend.sh
```

The scripts do not install dependencies, change snapshots or commit. Python
forces CPU and rejects ambient selectors/JIT settings that can weaken coverage.
Frontend forces its CI inventory and rejects conflicting selectors/capture
settings. Both retain per-task status and elapsed time. A missing prerequisite
or failed shard blocks qualification.

On a machine with limited RAM, set `MARL_PYTHON_GATE_JOBS=6` when running
`check.sh` or `check_before_commit.sh`. The supported range is 1–12, with 12
as the default. This limits concurrent Python workers, not test membership:
all twelve shards and all static checks still run. Use the same worker count
when comparing shard timings. Heavy swap use is a reason to lower concurrency,
not to omit tests or call an unfinished gate a pass.

The Python plugin collects ordinary pytest nodes and groups parameter families
by path, parent collector and original function name. Files normally remain
intact. Explicit measured exceptions may split a hotspot only while keeping
families and required fixture affinity intact. Pure fixture-reconstruction
exceptions require exact source identity, immutable deterministic data, read-only
consumers and no hidden I/O or mutable state. Stale exceptions fail collection.
Do not use dynamic fixture lookup to bypass these checks. The sharder alone owns
weights and reservations for static checks.

Browser profiles retain one worker and declared ordering. Split serial tests
only at approved complete scenario boundaries, keeping their full assertions and
setup isolation. Exact list-only coverage must show every required test once.
Do not use retries or whole-job timeouts to conceal deterministic failures or
poor balance. Keep the stable hosted aggregate names.

After a later edit, repeat the checks it can invalidate. A documentation-only
change needs source-equivalence, example/link and reflection checks; it may also
need schema/help tests when documentation is consumed at runtime. It does not
justify an unrelated benchmark campaign. Any candidate-byte change still
invalidates the separate frozen pre-commit qualification.

## Visual baselines

Compare first:

```bash
npm run test:visual --prefix web/visual_debugger
```

Only after approving an intentional visual change, update:

```bash
npm run test:visual:update --prefix web/visual_debugger
```

Review each image at its original resolution and retain its semantic assertions.
Updating snapshots is not an automatic test repair. Keep failure artifacts in
ignored outputs and tracked baselines within the repository file-size limit.

## Automation

[CI](../../.github/workflows/ci.yml) runs twelve Python shards with the static
checks distributed across the matrix, plus eight browser profiles containing
the frontend unit/static inventory and real Chromium tests. It uses locked
Python `dev`/`viz` and frontend dependencies. Hosted jobs do not qualify CUDA;
that is the separate [local GPU gate](gpu_sanity.md).

Pre-commit hooks provide fast hygiene and may mutate files. Run them before
freezing a candidate. They never replace behavioral, documentation, full
candidate or clean GPU qualification.
