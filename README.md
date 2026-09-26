# MARL-BattleGrounds

MARL-BattleGrounds lets researchers train and compare teams of agents in
competitive games. Agents have different classes and abilities, so a team
must combine movement, attacks, healing and support. The simulator uses JAX
for compiled, batched GPU execution.

The project is under development. Team Deathmatch, the native researcher API,
evaluation, recording and canonical tournament machinery are implemented.
Trained baseline qualification, a released official tournament field and
manuscript results remain future work. Six learner implementations are available;
software checks do not establish learned skill.

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

Team Deathmatch counts points, and one enemy death is not always one point.
Each team's Red Zone is the full-height strip at its own spawn edge, 5.0 map
units deep by default. When an agent dies with its centre inside its own team's Red Zone, the
enemy team gets 2 points instead of 1. It is still one kill and one death, so
kill counts and K/D do not double. Set the depth with `red_zone_depth`:

```python
deeper = marl_bgs.make("tdm", map_id=0, num_envs=128, red_zone_depth=6.0)
one_point = marl_bgs.make("tdm", map_id=0, num_envs=128, red_zone_depth=0.0)
```

`0.0` turns the rule off, so every death gives 1 point. `evaluate` and
`run_tournament` take the same keyword, the `evaluate` command takes
`--red-zone-depth`, and training sets `red_zone_depth` in its config. An exact
`env_config` keeps its own depth. Results saved before this rule keep their
original one-point meaning. See
[A44](docs/design/specification_amendments.md#a44-team-deathmatch-red-zone-scoring).

See [examples/environment.py](examples/environment.py) for a runnable loop
with both depth choices and [Ubuntu setup](docs/dev/setup_ubuntu.md) for the
locked GPU environment.

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
fixes mirrored Mage/Warrior/Hunter/Rogue/Priest teams, first to 20 points, the
default Red Zone depth of 5.0 (pass `red_zone_depth` to change it), a
300-transition horizon, five-transition respawn waves and canonical shields. Canonical map IDs
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
feedforward MAPPO and IPPO, recurrent QMIX with compact replay, recurrent
PQN-VDN, complete save/resume, frozen actor loading and learning reports.
Choose the method in one `TrainConfig`; the workflow stays the same. The
[baseline System example](examples/baseline_system.py) uses initialized,
**untrained** weights through the same public workflow. The
[complete training example](examples/mappo_training.py) exercises the public
Python route; the [QMIX example](examples/qmix_training.py) and the
[PQN-VDN example](examples/pqn_training.py) do the same for those methods.
Working execution and useful learned behavior need separate evidence.

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

Each team may be Manual, Random, Reactive TDM ALPHA, BETA or GAMMA. BETA's
Rogues prioritize observed living Priests, then Mages, then Hunters and try
body-avoidance routes; its other classes use ALPHA behavior. GAMMA
(`tdm-gamma`) is BETA plus three rules for its Warriors, Mages, Hunters and
Rogues: with no enemy in view they walk toward the middle of the enemy spawn
pads; they never damage an enemy with 2 or more Hunter Trap ticks left; and
Hunters start a new Trap only on an untrapped enemy they can reach now,
choosing Priest, Mage, Rogue, Warrior, Hunter in that order (see
[A43](docs/design/specification_amendments.md#a43-reactive-tdm-gamma)).
Reactive controllers require SharedObs; Random supports both observation modes
and samples exact current masks. These are diagnostic/pressure controllers, not
official learned baselines. Their limitations and versioned rules remain explicit.

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

<a id="planned-scenario-evaluations-and-big-12"></a>

## Scenario Evaluations And The Official Tournament

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
Experiments and Release. M12 starts by cleaning up the whole repository: a
refactor, an audit of all documentation, and a profiler-driven optimization audit
of runtime, VRAM, RAM and disk costs across the research workflow. The manuscript
experiments then run on the cleaned code. M12 ends with the public release,
including a documentation website and tutorials. KOTH (M13) and CTF (M14) follow
manuscript submission.
[Amendment A36](docs/design/specification_amendments.md#a36-submission-roadmap-approved-tdm-content-and-m7-closeout)
records the roadmap and the historical milestone-number mapping.

M8's native API scope is complete. Training distributions, curricula and six
learners are implemented. Their integration checks establish software behavior,
not scientific qualification. Finish the software before running the scientific
campaign. The growing-pool example shows how to use the API; it does not choose
an approved experimental curriculum.
Gymnasium/PettingZoo adapters are deferred until a specific learner integration
shows a need. The optional `interop` dependencies do not provide those adapters.
The native JAX workflow is the supported route; see the
[researcher workflows](docs/evaluation/workflows.md).

Public evaluation scenarios and their complete content closure must not inform
training, checkpoint selection, early stopping, hyperparameters, prompts,
curricula, population weights, or any other adaptive choice. The shared training
content checks keep declared training, validation and evaluation content separate.
Official systems must also retain complete provenance for maintainer
reproduction. This is a reproducibility and eligibility boundary,
not a claim that open-source software can make deliberate misconduct
impossible.

The official ladder uses a frozen population, called **Big N**. Each entrant
is one fixed, validation-selected System with one Elo value. Version-2 releases
take N from their participant list, with at least two entrants; no separate size
setting can disagree with that list. Historical version-1 official snapshots
keep exactly twelve entrants and their original rules and hashes.

The initial scientific plan covers eighteen settings from six learners:

| Learner | Planned Settings |
| --- | --- |
| Recurrent MAPPO, recurrent IPPO, QMIX and PQN-VDN | Plain, curriculum, reward shaping, and curriculum plus reward shaping for each learner |
| Feedforward MAPPO and feedforward IPPO | Plain only |

These are candidate settings, not eighteen qualified Systems. The initial
campaign plan uses three independent training seeds per setting. That is a
starting scientific plan, not evidence that three seeds are enough. Declare
comparable search and extension effort, or disclose differences. Each setting
supplies one fixed candidate through a rule declared before training; keep all
run, checkpoint and selection records.

Freeze the candidate field and selection rule before its validation games. Fit
joint Elo on the complete eligible field, take the top N, and break cutoff ties
by expected score against that same field, then by declared entrant order. Freeze
membership before any selected-field refit or separate test-map evaluation. A
refit may change ratings and order, but cannot choose different members. Weak
valid candidates stay eligible; failed work stays visible and missing evidence
must not turn into invented results or replacement seeds.

Scripted controllers and LLMs, including the former Qwen-Five proposal, are
excluded from the official candidate field and ladder. Valid scripted and host
Systems remain usable in custom tournaments and local comparisons against the
released field. Their diagnostic, scenario and declared competence-evaluation
roles remain separate. Loading a valid actor does not require official training
provenance; loading it does not grant official eligibility.

A field has N*(N-1)/2 unordered matchups; one challenger adds N. Twelve entrants
at 100 games per pairing give 6,600 games. That is a historical workload example,
not a selected official N or budget. The final official field, game budget and
resource rules still need a scientific declaration and qualification.

The raw win/draw/loss matrix remains authoritative. The compact report is Policy,
Elo with uncertainty, Expected Score with uncertainty, Win %, Draw % and Loss %,
with full matchup and per-map results beside it. Tactical metrics are optional
for local tournaments. K/D divides total kills by total deaths and stays blank
when deaths are zero; it is descriptive evidence, not a rating input.

The shared runner resolves one config, reuses compatible incumbent games and
runs the required new games. Saved-first resume keeps the run's original field.
Local runs neither admit nor publish a controller. Separate maintainer admission
works against a pinned field; fixture checks do not create an official release.
See [Canonical Tournaments](docs/evaluation/canonical_tournaments.md).

The Paper 1 field stays frozen. Later releases follow the accepted monthly
process; a promotion replaces one incumbent and keeps that release's N. Ratings
from different populations are not directly comparable over time. The Baseline
Library retains current and former entrants with their identities and evidence;
its growth does not enlarge the active field. See
[amendment A27](docs/design/specification_amendments.md#a27-rolling-big-n-and-baseline-library-governance)
and the [tournament protocol](docs/evaluation/protocol.md#big-n-tournament-and-baseline-library).

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

- [LLM Systems](docs/llm/README.md) shows default and custom formats, actor
  history, evaluation, tournaments and the qualified local Qwen recipe.

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
