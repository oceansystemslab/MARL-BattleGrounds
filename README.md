# MARL-BattleGrounds

MARL-BattleGrounds lets researchers train and compare teams of agents in
competitive games. Agents have different classes and abilities, so a team
must combine movement, attacks, healing and support. The simulator uses JAX
for compiled, batched GPU execution.

The project is under development. Team Deathmatch, the native researcher API,
evaluation, recording and canonical tournament machinery are implemented.
Trained baselines, a qualified official Big 12 bundle and manuscript results
remain future work.

The installed package also has terminal commands:

```bash
python -m marl_battlegrounds --help
python -m marl_battlegrounds evaluate --system random --opponent tdm-alpha --episodes 32
python -m marl_battlegrounds replay /path/to/saved-replay.json
```

Evaluation prints a short result preview. Add `--output-dir runs/my-evaluation`
to keep the complete results. Replay viewing needs an existing replay file; it
does not require this checkout or Node. See the [complete workflows](docs/evaluation/workflows.md)
for factories, saved-first resume, explicit asset preparation and analysis.

## Take a First Step

The package includes 52 Team Deathmatch maps and eight fixed scenarios. Prepare
128 games, sample legal actions and advance them together:

```python
import jax
import marl_battlegrounds as marl_bgs

reset_key, action_key, step_key, next_reset_key = jax.random.split(jax.random.key(42), 4)
env = marl_bgs.make("tdm", map_id=0, num_envs=128)
observation, state = env.reset(reset_key)
actions = env.sample_actions(action_key, state)
observation, state, reward, done, info = env.step(step_key, state, actions)
observation, state = env.reset_done(next_reset_key, state)
```

Keep the returned state for the next call. `reset_done` replaces only completed
games; continuing games keep their exact state. The environment supplies episode
IDs and legal action sampling. It does not reset inside `step`.

Map-based setup balances spawn locations by default: half the lanes keep the
map's ordered spawn banks and half exchange both banks. Team identities, rosters
and world directions stay fixed. Automatic balance needs an even positive batch.
For a scalar or odd batch, set `balance_spawn_locations=False`. An explicit
`env_config` remains exact and overrides automatic preparation. Equal episode
counts alone do not establish equal training steps from both spawn locations.

See [examples/environment.py](examples/environment.py) for a runnable loop and
[Ubuntu setup](docs/dev/setup_ubuntu.md) for the locked GPU environment.

## Choose Maps and Rosters

`marl_bgs.list_tdm_maps()` and `marl_bgs.list_tdm_scenarios()` list names and source
identities. `marl_bgs.load_tdm_scenario(1)` returns a configuration, exact initial
state and notes. Pass that object to `env.reset(key, scenario=scenario)`.
Each team may contain one through five agents, including repeated classes.
Supplied roster order fills the first team-local slots; it does not reserve a
fixed slot for each class. `marl_bgs.canonical_tournament_rosters()` returns the
shared default roster order.

The factory
`marl_battlegrounds.tasks.make_canonical_team_deathmatch_evaluation_config(map_id=47)`
fixes mirrored Mage/Warrior/Hunter/Rogue/Priest teams, first to 20, a 300-transition
horizon, five-transition respawn waves and canonical shields. Canonical map IDs
are 47–51. Training maps are 0–41, including curriculum maps 0–11; validation
maps are 42–46. Numbered map names, authored IDs and saved-map folders use the
same IDs. See the
[map ID change guide](docs/evaluation/workflows.md#map-id-change--2026-09-12)
when using saved IDs. Training distributions belong to the later curriculum
milestone.

## Train, Evaluate and Read Results

Researchers own their models, sampling and learning update. Use JAX `jit`,
`vmap` and `lax.scan` around the same `reset`/`step` contract. Configuration,
current legal-action masks and episode counters travel in environment state.
Use `reset_done` for ordinary terminal resets, or explicit `reset` arguments
for custom episode starts. A mask describes current legal actions; a structural
action space alone cannot express every coupled action constraint.

The optional [training package](docs/training/README.md) adds recurrent and
feedforward MAPPO and IPPO, recurrent QMIX with compact replay, complete
save/resume, frozen actor loading and learning reports. Choose the method in
one `TrainConfig`; the workflow stays the same. The
[baseline System example](examples/baseline_system.py) uses initialized,
**untrained** weights through the same public workflow. The
[complete training example](examples/mappo_training.py) exercises the public
Python route; the [QMIX example](examples/qmix_training.py) does the same for
QMIX. Working execution and useful learned behavior need separate evidence.

Use one callable for frozen-System validation and evaluation:

```python
import marl_battlegrounds as marl_bgs

result = marl_bgs.evaluate(
    "tdm-alpha", "tdm-beta", maps=[47, 48, 49, 50, 51],
    num_episodes=100, num_envs=128, seed=42,
    metrics="full", replay_episodes=range(1, 11),
    output_dir="runs/evaluation",
)
print(result.paths)
print(result.table("episodes")["system_game_score"])
```

`Policy(name, apply, variables, initial_carry)` adapts a learned policy through
`apply(variables, carry, actor_input, action_mask, key) -> (action, next_carry)`.
Variables stay frozen during evaluation; recurrent memory resets per actor and
episode. The authorized SharedObs input remains structured. Custom encoders and
training algorithms remain the researcher's choice.
Shared, independent, recurrent and host `System` methods use the same evaluator.
The researcher stays Team A. The default runs complete pairs with exchanged
spawn locations; `spawn_mode="default"` or `"swapped"` runs a fixed choice.
Omitted maps use validation maps for `phase="validation"` and test maps for
`phase="evaluation"`. Custom phases require explicit maps.
Actor inputs use self/ally/enemy roles and a local self row, with no simulator
team ID or global slot. See the [input contract](docs/evaluation/workflows.md#policy-inputs)
for shapes and the recording-version change.

Metrics use `"none"`, `"priority"` (default), or `"full"`. Select extra full episodes
with `full_metrics_episodes=range(1000, 50_001, 1000)` and replays independently
with `replay_episodes=range(49_951, 50_001)` when those IDs exist in the run.
No replay is needed for metrics.
Without an output directory, results are column arrays suitable for
`pandas.DataFrame(result.full_metrics)`, and no files are created. Each output
request creates a unique child run; `resume_from` explicitly resumes a run.
`load_results(run_dir)` reads saved tables without loading the simulator or
changing files. Use `iter_table("full_metrics", rows=128)` for bounded reading.
See the [complete evaluation examples](examples/evaluation_results.py) for
validation, exact authored episodes, resume and tournament analysis.

The [metric specification and data dictionary](docs/evaluation/metric_specification.md)
explain scalar columns, missing values, attribution and recording. The
[evaluation protocol](docs/evaluation/protocol.md) defines held-out evaluation,
paired tournaments, ratings and uncertainty.
The [workflow guide](docs/evaluation/workflows.md) includes runnable evaluation,
shared-writer validation, replay and tournament commands.

## DevClient

Launch the live development tool:

```bash
./scripts/dev/run_dev_client.sh
```

Its **Combat Debugger** stages simultaneous actions, shows current legality and
switches between Oracle and authorized Agent POV views. **Maps** and **Scenarios**
author reusable local assets. Save is explicit: numbered revisions persist under
ignored `artifacts/dev_client/` storage. Validation uses existing simulator
rules. Deletion requires confirmation. A saved map opens as a labelled standard
5v5 preview; loading it does not change the saved map.

Each team may be Manual, Random or Reactive TDM ALPHA. Team B also offers BETA,
whose Rogues prioritize observed living Priests, then Mages, then Hunters and
try body-avoidance routes. Other classes use ALPHA behavior. Reactive controllers
require SharedObs; Random supports both observation modes and samples exact
current masks. These are diagnostic/pressure controllers, not official learned
baselines. Their limitations and versioned rules remain explicit.

Record one episode to a new replay file:

```bash
mkdir -p recordings
./scripts/dev/run_dev_client.sh \
  --record-replay recordings/episode.marlbg-replay.json
```

The [DevClient guide](docs/dev/combat_debugger.md) covers arguments, asset IDs,
controls, exact starts, recording/recovery and troubleshooting. It explains
which fields belong to researcher inspection and which may reach an actor.
Existing replay files belong in the separate Viewer.

## Replay Viewer

Open an immutable replay, discover checked samples or inspect a scripted demo:

```bash
./scripts/dev/run_replay_viewer.sh --replay episode.marlbg-replay.json
./scripts/dev/run_replay_viewer.sh --list-sample-replays
./scripts/dev/run_replay_viewer.sh --sample-replay death-respawn-shield
./scripts/dev/run_replay_viewer.sh --list-scenarios
./scripts/dev/run_replay_viewer.sh --scenario stacked_team_auras
```

The Viewer validates saved evidence before serving it. A scripted-demo request
first materializes its replay in a separate preparation step. The read-only
Viewer then navigates exact frames, plays recorded transitions and provides
participant/score summaries, configurable effects and provenance-bearing PNG
exports. It cannot submit simulator actions.

Open **TDM Evaluation Metrics** for **Up to Current Tick** or **Entire Episode**.
**Download Metrics CSV** exports the selected boundary through the same scalar
metric authority as evaluation. Current replay files need no metric sidecar;
historical files keep their readers. Metrics are computed on request. A visual
POV changes the battlefield view, not the researcher-level metric ownership.

See the [Viewer guide](docs/dev/replay_viewer.md) for transport, keyboard controls,
audience limits, exports and recovery. The
[migration guide](docs/dev/visual_debugger.md) explains the former combined tool.
Both clients use local HTML, CSS, SVG and JavaScript served by Python. Researchers
need neither Node.js nor npm. SharedObs visual union is a rendering contract,
separate from the structured actor input, as described in
[A17](docs/design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).

## Actor Information Regimes

Paper 1 uses canonical SharedObs: each actor receives its own permitted view
plus authorized teammate sensor rows. Official recordings require the complete
configured-active, same-team, off-diagonal availability matrix at every frame.
A configured dead teammate remains an authorized source, with no live sensor
material. The input adapter does not reveal simulator team IDs or global slots.

Current recording binds SharedObs actor projection 2; historical projection 1
files retain their original meaning. Always use the recorded version rather
than treating equal tensor widths as compatibility. Base observations and exact
availability are recorded without storing a second materialized source bank.

Generic execution and historical readers also support homogeneous NoSharedObs
for custom research and diagnostics. It is not the official Paper 1 comparison
axis. Assets are regime-independent; official eligibility is checked after
loading. [A25](docs/design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution)
records the original canonical decision and historical version context.

Researchers choose their networks, encoders and learning algorithms. The API
fixes permitted information, shapes, categorical meanings, masks and provenance;
it does not require one-hot features, embeddings, attention or a particular
model framework.

## Planned Scenario Evaluations and Big 12

The approved TDM suite contains eight scenarios. Seven use five-transition
horizons; Scenario 3 uses ten transitions to examine sustained body blocking,
while retaining the canonical five-transition respawn period. The scenario and
map designs are approved. Packaged definitions and a bounded complete-suite
qualification use the shared episode executor, metrics and replay APIs.
These plumbing controls do not replace learned-policy behavioral ablations.

The public scenario suite is primarily a controlled behavioral-ablation
instrument. Each manuscript comparison will compare a complete method
with its matched ablation under the same scenario revision, embedded map and
initial state, deterministic reactive pressure controller, canonical SharedObs
contract, seeds and sides, training budget, checkpoint-selection rule, and
primary endpoint. Scenario pressure controllers follow versioned rules and
authoritative action masks rather than replaying hard-coded action tapes. An
evaluation definition may bind Reactive TDM or a specialist; it remains
separate from the saved, controller-independent DevClient scenario. Diagnostic
controller availability does not qualify an official evaluation. Scenario results
provide evidence for a specific behavioral claim under frozen conditions; they
do not contribute to Elo or establish general strength. See
[specification amendment A26](docs/design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations).

The current sequence is M7 TDM Benchmark and Researcher Tools → M8 Policy
Execution and Evaluation → M9 Training Distributions and Curriculum → M10
Learning Platform and Baselines → M11 LLM-Agent Integration → M12 Manuscript
Experiments and Release. After manuscript completion, a dedicated profiler-driven
optimization audit will cover runtime, VRAM, RAM and disk costs across the research
workflow. KOTH (M13) and CTF (M14) follow manuscript submission and that audit.
[Amendment A36](docs/design/specification_amendments.md#a36-submission-roadmap-approved-tdm-content-and-m7-closeout)
records the executive override and historical milestone-number mapping.

M8's native API scope is complete. M9 will define training distributions and
curricula; M10 will add learners and learning experiments. The growing-pool
example shows how to use the API, not an approved training curriculum.
Gymnasium/PettingZoo adapters are deferred until a specific learner integration
shows a need. The optional `interop` dependencies do not provide those adapters.
The native JAX workflow is the supported route; see the
[researcher workflows](docs/evaluation/workflows.md).

Public evaluation scenarios and their complete content closure must not inform
training, checkpoint selection, early stopping, hyperparameters, prompts,
curricula, population weights, or any other adaptive choice. Future M9/M10
pipelines will enforce content-addressed training, validation, and evaluation
manifest separation, while official systems retain complete provenance for
maintainer reproduction. This is a reproducibility and eligibility boundary,
not a claim that open-source software can make deliberate misconduct
impossible.

The planned Big 12 is a rolling ladder of exactly twelve method-level entrants,
each represented by one validation-selected fixed tournament system and one
Elo. The tentative initial roster is:

1. RNN-IPPO, parameter-shared
2. RNN-MAPPO, parameter-shared
3. RNN-MAPPO, class-specific actors
4. RNN-HAPPO
5. HyperMARL-PPO
6. RNN-QMIX
7. RNN-PQN-VDN
8. MAPPO-PFSP League
9. MAPPO-PSRO
10. S\*-Curriculum
11. S\*-Curriculum-Shaped
12. Qwen-Five

These learned systems and the manuscript training campaign are planned; the
shared evaluation and tournament tools are implemented. Rows 1–11 will each
retain three independently trained runs and their checkpoint
histories, but only the fixed checkpoint selected by the frozen validation-only
rule enters the tournament. Qwen-Five remains tentative until a measured
throughput and resource gate is passed. Twelve systems yield 66 unordered
pairings. For illustration, 100 episodes per pairing would mean 6,600 games;
the final official numerical budget is not settled. The raw
win/draw/loss matrix remains authoritative. The compact ladder presentation is
Policy, Elo with uncertainty, Expected Score with uncertainty, Win %, Draw % and
Loss %, accompanied by full matchup and per-map results. Full tactical metrics
are optional when running a local tournament. The headline report identifies tournament entries, rather than calling
them teams. Its K/D ratio divides total kills by total deaths and stays blank
when deaths are zero. This is descriptive evidence, not a rating input.

The implemented canonical machinery uses immutable monthly snapshots with exactly twelve
controller versions. A challenger comparison contains those twelve plus one
challenger; the growing Baseline Library does not enlarge that population.
Custom tournaments may use other populations. Local runs produce results and
do not admit or publish a controller. Numerical budgets and several admission
rules still need approval. The snapshot/reuse runner and separate maintainer
admission machinery are implemented and tested with fixtures. Those fixtures
do not supply trained controllers or a qualified official bundle. See
[Canonical Tournaments](docs/evaluation/canonical_tournaments.md).

The Paper 1 snapshot stays frozen. Pool-centred Elo values from different
populations are not directly comparable over time. Current and former entrants
retain immutable identities and supporting records in the Baseline Library.
The earlier weekly update wording is superseded by this monthly direction.
Reactive TDM, specialist scenario controllers,
Random, and internal training-population members are not Big 12 entrants.
See
[specification amendment A27](docs/design/specification_amendments.md#a27-rolling-big-12-and-baseline-library-governance).

## Static Snapshots

The optional `viz` dependency is activated automatically by either shell
launcher when `--static` is present:

```bash
# One manual arena reset snapshot.
./scripts/dev/run_dev_client.sh --static

# One exact frame from an immutable replay.
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json --static --frame-index 0
```

Static rendering starts no browser server. For direct Python calls, prepare the
optional painter with `uv sync --locked --extra viz`. Browser launch does not
require Matplotlib.

## Design and Evaluation

- [Specification amendments](docs/design/specification_amendments.md) are the
  controlling public authority for accepted departures from the historical
  design PDF.
- [Evaluation metric specification](docs/evaluation/metric_specification.md)
  defines metric meanings, attribution limits, scorecard surfaces, and
  candidate dispositions.
- [Evaluation protocol](docs/evaluation/protocol.md) defines evaluation cells,
  aggregation, inference, cross-play, scenarios, failure handling, and
  canonical actor-information provenance.
- [Standard replay format](docs/evaluation/replay_format.md) defines the
  versioned semantic replay normal form, whole-artifact validation boundary,
  companion artifacts, and canonical bounded local persistence.
- [Dependency policy](docs/dev/dependency_policy.md) separates researcher
  runtime requirements from optional and contributor-only tooling.

## Contribute

Use the locked dependencies and the [quality gates](docs/dev/quality_gates.md).
GPU/JAX speed, memory, compilation reuse and data movement are the performance
targets. CPU checks establish correctness and compatibility; their timings help
balance CI jobs and are not simulator performance claims.

Documentation is part of every change. The shared
[Documentation Standard](docs/dev/documentation_standard.md)
requires clear file descriptions and complete, source-checked function contracts.
Python uses NumPy-style docstrings; browser code uses JSDoc. Test files keep one
file-level description and necessary type directives, with no function or class
docstrings. Examples must show working public calls and label future designs.

## Author

MARL-BattleGrounds is developed by Ulixes Tariq Hawili as part of the SPADS CDT,
University of Edinburgh School of Engineering, and the Ocean Systems Lab at HWU.

## License

Licensed under the Apache License, Version 2.0.
