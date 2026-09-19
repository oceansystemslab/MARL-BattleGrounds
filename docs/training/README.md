# Training Setup And Baseline Components

The `marl_battlegrounds.training` package prepares verified maps, samples
configurations and collects exact experience budgets with optional curriculum,
score shaping and self-play history. The `marl_battlegrounds.baselines` package
provides input encoders, exact action helpers and recurrent MAPPO calculations.
The collector supplies data for a learner; it does not apply optimizer updates
or provide a complete trainer or trained policy. Training helpers and ordinary
environment use need only base dependencies; the MAPPO actor needs `training`.

Install the existing `training` extra to use PPO. In a prepared contributor
checkout, run the example with:

```bash
uv sync --locked --extra dev --extra training
JAX_PLATFORMS=cpu uv run --no-sync python examples/baseline_system.py
```

For the qualified local GPU setup, follow the [GPU guide](../dev/gpu_sanity.md).
For the distribution example and its recording benchmark, use
`JAX_PLATFORMS=cuda,cpu`. The numerical loop uses the GPU; the existing host
configuration validator uses the CPU. Selecting only `cuda` makes that validator
fall back to many small GPU operations, which adds recording cost.
The [example](../../examples/baseline_system.py) initializes **untrained** actor
weights, wraps them in an existing M8 `System`, chooses actions and steps the
environment. Its results say nothing about learned skill. Methods that do not
use these baseline components can continue using ordinary `reset` and `step`
or their own Systems.

## Verified Training Content

`marl_battlegrounds.training.prepare_training_content()` checks the installed
content before collection starts. It returns an immutable `binding` and one
`source_configs` bank with 42 rows. Each row is a canonical 5v5 source, with its
original ordered spawn pads. This setup uses base dependencies and runs no
learner. Importing the `training` package alone does not import JAX or PPO.

Each preparation reads all 52 installed maps and eight protected scenarios once
through their shared loaders. It validates each scenario once and validates both
spawn arrangements for each training source. Maps 0–41 are training, 42–46
validation and 47–51 test.
Eligibility depends on resolved content and its declared role. Renaming a
protected scenario cannot make its layout eligible for training or validation.
Shared class catalogs, schemas and simulator rules remain permitted shared
dependencies. External maps, controllers, datasets and feedback are outside
this check's supported content set.

Save `prepared.binding.model_dump(mode="json")` with experiment state. On
resume, call `prepare_training_content(expected=saved_binding)` **before**
constructing a resumed `RunWriter`: writer recovery can change durable files.
This check compares scientific content, roles, ordering and supported versions.
Harmless labels and relocated authored paths may differ. Resource-byte evidence
and authored-file metadata remain separate from resolved scientific identities.
The protected scenario specifications describe the existing two-coordinate
recording check; they are not a complete evaluation schedule.

Keep the binding on the host and the numerical bank shared across compiled
calls. Hashing, file reads and physical validation belong to preparation, not
individual sampling decisions. The bank is immutable experiment input: do not
mutate or donate it while sampling or recording uses it.

## Sample Maps And Rosters At Reset

`sample_training_configs` receives the prepared bank, one Threefry root key,
int32 reset generations `(B,)`, a Boolean `eligible_maps` mask `(42,)`, and an
int32 scalar `team_size`. B must be positive and even. It returns a numerical
`SampledTrainingConfigs` containing `config`, int32 `source_indices` `(B,)`,
and int32 `source_class_ids` `(B,10)`.

Eligible maps are equally likely. Both teams have the same size, but draw their
classes independently and without replacement. Size one excludes Priest;
sizes two through four allow every class. Slots stay compact and follow the
canonical relative class order. Size five uses the exact canonical roster.
The first half of lanes retain the source spawn banks; the second half exchange
them. Pass the resulting configurations directly to reset. Do not apply spawn
balancing a second time.

This complete setup prepares 32 games with independently drawn 2v2 rosters:

```python
import jax
import jax.numpy as jnp
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training

prepared = training.prepare_training_content()
root_key = jax.random.key(42, impl="threefry2x32")
generations = jnp.zeros(32, dtype=jnp.int32)
eligible_maps = jnp.ones(42, dtype=jnp.bool_)
team_size = jnp.asarray(2, dtype=jnp.int32)
training.validate_training_distribution(
    eligible_maps=eligible_maps, team_size=team_size,
)
sample = jax.jit(training.sample_training_configs)
chosen = sample(
    prepared.source_configs, root_key, generations,
    eligible_maps=eligible_maps, team_size=team_size,
)
env = marl_bgs.make("tdm", num_envs=32, metrics="none")
observations, state = env.reset(
    training.training_keys(root_key, generations, stream="reset"),
    env_config=chosen.config,
)
tracking = marl_bgs.init_episode_tracking(
    env, state, source_configs=prepared.source_configs,
    source_indices=chosen.source_indices,
    source_class_ids=chosen.source_class_ids,
)
```

`validate_training_distribution` runs on the host. Call it whenever eligibility
or team size changes, before passing those controls into a compiled function.
It rejects an empty map mask or sizes outside one through five. The sampler
checks shapes and dtypes under JIT, but invalid traced values are unsupported;
there is no fallback or hidden host check. Changing valid values with the same
shapes does not require a fresh compiled sampler.

During a rollout, sample only when at least one lane needs reset. Use the next
reset generation for those lanes, and keep continuing lanes' generations fixed.
Adopt new configurations, source indices and class rows only in reset lanes.
Then pass those declarations to `track_episode_step` on the new episode's first
real transition. Continuing games keep their old configuration, declaration and
System memory. An episode reset clears memory on the next action decision;
death, respawn and a rollout boundary do not.

Keep one shared source bank and the current per-lane declarations. Do not retain
full sampled configuration histories. The sampler changes no score threshold,
horizon or class capability. Curriculum schedules, shaping and opponent history
remain outside these helpers.

## Own The Training Random Keys

`training_keys` keeps separate version-1 Threefry streams. It folds the stream
tag, fixed lane index and reset generation into the root, in that order. Action
and step streams then fold in the local decision step.

| Stream | Tag | Decision Step |
| --- | ---: | --- |
| `map` | 0 | Omitted |
| `roster` | 1 | Omitted |
| `opponent` | 2 | Omitted; used when a new game selects its opponent |
| `action` | 3 | Required |
| `reset` | 4 | Omitted |
| `step` | 5 | Required |
| `initialization` | 6 | Omitted |

Keys are typed `(B,)` or legacy uint32 `(B,2)`. Typed roots must use Threefry;
legacy roots require JAX's default Threefry implementation. RBG keys are rejected.
Counters use nonnegative int32 values. The existing environment owns checked
reset generations; do not advance them by an unchecked counter that can wrap.
The sampler derives separate Team A and Team B roster keys internally.

Pass action keys directly to `apply_systems`; M8 still owns team keys and each
System owns its actor keys. The MAPPO System derives five actor keys per supplied
environment key. The training helpers do not replace M8's reset or memory rules.
Another lane's resets cannot change these training stream keys. A custom
stochastic System initializer may also use M8's episode IDs, so this does not
promise identical initialization under every reset schedule.

For repeatable continuation, save the root bits, Threefry implementation,
`TRAINING_KEY_SCHEMA_VERSION`, batch size, lane order, reset generations and
local decision counters with the numerical carry. The carry must also retain
current configurations, source/class declarations, environment and tracker
state, and System memory. A content binding alone is not an execution checkpoint.

## Run The Distribution Example

The [sampled-episode example](../../examples/training_distributions.py) uses the
public environment, Systems, tracker and collector. Random plays both teams by
default. It needs only base dependencies, creates 32 games, and runs 320 decisions
per lane so the ordinary 300-transition horizon reaches a reset. It runs no
learner and writes no files unless recording is requested.

In a prepared checkout:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/training_distributions.py
JAX_PLATFORMS=cpu uv run --no-sync python examples/training_distributions.py \
  --output-dir artifacts/training-distribution-example
```

With `--output-dir`, the example saves a continuation in memory and obtains an
M8 recording token. It runs ahead, verifies the saved content before opening the
writer, restores the recorded boundary, and repeats the suffix. This checks
recording and numerical continuation. It creates no learner checkpoint format
and does not demonstrate recovery after losing that in-memory continuation.

After installing the `training` extra shown above, `--mappo` replaces Team A with
**untrained** recurrent MAPPO weights. Team B remains Random. Weights remain
dynamic action inputs; the critic never enters the System. Rewards and episode
outcomes are wiring checks, not evidence of learned skill. `--steps N` changes
the example length; N must be at least two, and short runs may contain no reset.

For recorded sampled episodes, set `record_starts=True` when initializing the
tracker. The writer persists explicit class rows; absent or all-`-1` declarations
remain absent in saved metadata. It reconstructs the declared roster from the
class catalog and checks the entire resolved configuration and ordered spawn
banks. Source-bank identity and actual episode configuration identity stay
separate. Adding or removing a roster claim later is a conflicting declaration,
even after an earlier successful verification was cached.

Both ordinary writer resume and recording-token restore validate saved start
claims before changing output. Verified and custom starts require saved resolved
configuration evidence. Ordinary resume can retain a pending declaration that
has no transition evidence yet. Historical records without a class field keep
their original meaning; the reader does not invent a roster claim.
See [tracking and recording](../evaluation/workflows.md#track-experience-and-reset-finished-games)
for the optional low-level route.

## Collect An Exact Experience Budget

The [collection example](../../examples/training_collection.py) uses the existing
untrained recurrent MAPPO actor. It prepares content once, uses current self-play
and collects the requested number of real transitions. Defaults are 32 games,
128 rounds per returned block and 4,096 total transitions. A round advances each
game once. These are collection examples; no optimizer updates are performed.

```bash
# Small CPU check, including a padded final block.
JAX_PLATFORMS=cpu uv run --no-sync python examples/training_collection.py \
  --mode Plain --num-envs 2 --total-env-steps 4 --length 4

# Ordinary batch and block sizes; select the GPU as described in the GPU guide.
JAX_PLATFORMS=cuda,cpu uv run --no-sync python examples/training_collection.py \
  --mode C-RS
```

`--mode Plain` enables neither curriculum nor shaping. `C` enables curriculum,
`RS` enables shaping, and `C-RS` enables both. All four keep the installed K20
score threshold and H300 horizon. `--total-env-steps` must divide exactly by the
positive even `--num-envs`. A curriculum budget must give each of its 17 requested
stages at least one round. The library rejects insufficient budgets.

The example's `run(...)` function is also callable from Python. Its CLI delegates
to that function. Add `--output-dir artifacts/training-collection` to save a
training run through the existing `RunWriter`. The default creates no files.
Saved outcomes remain canonical; shaping does not change CSV scores or metrics.
This example creates no learner checkpoint and offers no restart command.

The library route uses four calls:

1. `make_training_schedule(total_env_steps=..., num_envs=..., curriculum=...)`
   resolves exact stage budgets on the host.
2. `init_training_collection(actor, actor_variables, schedule=schedule, ...)`
   returns a stable `TrainingCollection` and its numerical `TrainingCarry`.
   Supply a native JAX `System` whose lanes are independent and whose memory
   has a leading game axis. Actor-only weights stay dynamic. Policy adapters,
   host methods, custom memory-reset hooks and frozen checkpoint labels are
   unsupported by this collector; ordinary environment/System loops remain
   available for those methods.
3. `collect_training_rollout(collection, carry, length=128, writer=None)`
   returns the next carry and a `TrainingRollout`. Reuse the same collection
   descriptor and block length. The final block stops at the exact total and
   returns invalid padding without extra actions, resets or writer records.
4. `training_summary(collection, carry)` reads small counters on the host. It
   reports requested rounding, games started, played transitions, active living
   Team A decisions, maps, opponent assignments and unfinished games.

Setup defaults to `shaping=False`, `discount=0.99`, `coefficient=0.01`,
`collect_training_state=False`, `metrics="priority"` and `recording=False`.
Pass `prepared=prepared` to reuse verified content. Public setup keeps that bank
exact; a synthetic horizon override is not a supported training input.
With recording enabled, supply the same healthy writer on every collection
call, starting before the first action. Register the descriptors as
`policies={"team_a": collection.actor, "team_b": collection.opponent}` with
`phase="training"`. Training policy versions are evolving weights, not frozen
checkpoint labels. The example saves the content binding and requested schedule.

`scan_training_rollout(collection, carry, length=...)` is the pure JAX route
for a larger compiled loop. Keep the descriptor and length fixed outside the
dynamic carry. It sets sticky failure flags and stops further actions; check
the returned carry on the host before learning. The host collection helper does
this check automatically. `advance_training_step` is the shared one-round
building block and requires remaining budget. Neither path applies a learner
update or owns critic memory.

### Requested Stages And Played Exposure

Plain and RS request canonical 5v5 on all 42 training maps. C and C-RS request:

| Stages, Zero-Based | Requested Budget | Games Chosen At Reset |
| --- | --- | --- |
| 0–4 | 4% each | Map 0, sizes 1v1 through 5v5 |
| 5–15 | 20/11% each | 5v5, pools 0–1 through 0–11 |
| 16 | 60% | 5v5, maps 0–41 |

The schedule assigns whole rounds by largest remainders; earlier stages win
exact ties. It never increases the total budget. The tracker checks exact real
advances and equal use of the two fixed spawn arrangements at each stage end.
Continuing games keep the distribution chosen at their reset, even after the
requested stage changes. Finished games reset immediately before their next
real action, using the latest requested stage and opponent bank.

The summary therefore separates `steps_by_episode_stage` from requested stage
budgets. A stage may complete its accounting budget while receiving no new
games and no played exposure of its own. The short example can show this under
H300. A reset alone is not a played game; `starts_by_episode_stage` counts the
first real decision. Actor decisions are active, living Team A decisions.
`learner_samples` stays `None`: collection counts do not establish which samples
a future learner used, sample efficiency or learned tactics.

### Optional Score Shaping

`team_potential_shaping(before_scores, info, discount=..., coefficient=0.01)`
returns float32 `(B,2)` adjustments in Team A/Team B order. The potential is the
coefficient times own score minus opponent score. The adjustment is the exact
learner discount times the next potential, minus the previous potential.
Wins, losses and horizon draws set the next potential to zero. A rollout or
stage cutoff, death or respawn does not. Padding returns zero.

Read int32 `(B,2)` `before_scores` before the action. Use the matching producing
`EpisodeInfo`, whose score remains correct even after AutoReset returns a new
game. `validate_shaping` checks finite host settings: discount in `[0,1]` and a
nonnegative coefficient, excluding Boolean values. Numerical execution uses
dynamic scalar float32 settings and checks static shapes/dtypes.

Complete discounted adjustments sum to minus the starting potential, including
authored nonzero scores. They sum to zero from a tied start. This preserves the
declared discounted objective for fixed starts. It does not prove faster
learning or preservation of an undiscounted win-rate objective. The coefficient
0.01 is a starting setting, not a qualified learning choice.

The collector stores one Team A `shaping_reward` per game and keeps native
`task_rewards` separately. A learner may add the team signal to each active
Team A value target; dead active agents retain that signal and inactive slots
remain excluded. Do not sum repeated actor copies into a larger team bonus.
Disabled shaping skips the calculation entirely. Shaping never enters actor
inputs or changes official rewards, scores or metric definitions.

### Current And Historical Self-Play

Each new opponent game uses current actor weights until history exists. After
that, its assignment is 80% current and 20% uniformly sampled from stored
snapshots. Assignment probabilities are not promises about observed episode or
transition shares. A historical game keeps its chosen weights until it ends.
Current-policy games use newly published actor weights while retaining their
own recurrent memory. Team A and Team B never share memory.

Call `refresh_opponents` once after each completed learner update and before
the next block. Supply already-updated actor-only variables, the exact next
`update_index`, the carry's completed real rounds and its numerical schedule.
Keep the returned history in the carry. The helper performs no learning.
It captures one immutable actor when an unmet 5%, 10%, ..., 100% threshold has
been reached. Several thresholds crossed by one update share one stored
snapshot. The bank has room for 20 snapshots, with no eviction. Record actual
capture rounds/updates from `SnapshotEvent`; a final snapshot may see no play.

Opponent slots use `-1` for current and zero-based snapshot indices otherwise.
Invalid update notifications or assignments set a sticky error and block further
collection. The actor must keep its variable tree, shapes, dtypes and information
rights. Critic and optimizer state stay outside the bank. The supplied example
does not refresh weights, so it creates no historical snapshots or update claims.

### Compact Data For A Learner

`TrainingRollout.transitions` uses `(T,B,...)` axes. It keeps compact permitted
observations, Team A masks, submitted joint actions, same-call Team A learning
outputs, task/shaping rewards, active/alive flags and exact producing identities.
Completion outcomes, lengths, scores and optional priority values are present
only on completed rows. Team B learning outputs are discarded.

`valid` excludes padding. Padding has zero payloads, `-1` identities and a
neutral-only action mask. `real_steps` counts rounds, not total transitions.
The rollout keeps Team A's starting memory and the true final observations,
masks and ending flags before a pending reset. Use that successor for a cutoff's
value estimate. A real task ending, including H300, stops the value estimate.
With `collect_training_state=True`, the separate 919-value physical view is
stored once per game, with a matching final view. It never enters the actor.
Critic values, GAE, complete `PPOBatch` construction, optimization, checkpoint
selection and durable learner restart remain the learner's responsibility.

## Inputs And Information Limits

| Encoder | Output Per Record | Purpose |
| --- | --- | --- |
| `encode_actor_inputs(ActorInput)` | 5,164 float32 values | Only the receiving actor's permitted information. |
| `encode_training_state(EnvState, EnvConfig)` | 919 float32 values | Privileged physical state for the training critic. |

Both schemas start at version 1. `ACTOR_FEATURE_OFFSETS` and
`TRAINING_STATE_FEATURE_OFFSETS` in [inputs.py](../../src/marl_battlegrounds/baselines/inputs.py)
map explicit field names to fixed Python slices. Those ordered definitions own
the layouts; dictionary or PyTree traversal does not choose field order.

Actor fields include all permitted observations, the source-specific sensor
bank, visibility and source permissions. Class IDs use six categories;
obstacles use three. Hidden or unused categorical rows stay all zero. Public
roster membership, life status and class facts remain distinct from private
sensing. Missing previous actions stay missing; they do not become neutral
actions. The encoder uses existing masks. It does not recalculate visibility
or grant information to another actor.

The training view contains the physical state, then physical configuration in
declaration order. Task IDs use four categories, including Neutral and reserved
modes. Team IDs use three, including unused. Membership is 0 or 1. Ordered
spawn pads retain their order. Seeds, administrative map IDs, opponent names
and checkpoint identities are absent. Configuration may broadcast over matching
state axes; incompatible static shapes raise an error.

Values keep their original units. Float32 rounds some integers above 16,777,216.
Large timers therefore cannot always be recovered exactly from these features.
This representation is a model input, not a lossless saved state.

Keep compact `Observations` in rollout storage. Expand actor features only when
applying a model. Store the critic view once per game. Its temporary broadcast
across actors belongs inside the critic application or current minibatch.

## Actions And Random Keys

[actions.py](../../src/marl_battlegrounds/baselines/actions.py) defines 198 choices
in `(movement, target, Ultimate)` order, with Ultimate changing fastest. Index
zero is native neutral `(0, 0, 0)`. `decode_actions` owns conversion to
`ActorAction`; M8 owns joining Team A and Team B.

`categorical_action_mask` combines movement legality with the existing **joint**
target/Ultimate mask. Separate marginal masks do not capture that constraint.
`sample_actions` takes one explicit key per sample and returns int32 indices.
`action_log_prob` and `action_entropy` use float32 and natural logarithms.
Leading sample axes are preserved. Illegal choices have zero probability.
An empty mask is invalid; no helper invents a neutral fallback for it.

Use typed Threefry keys or legacy uint32 keys with JAX's default Threefry
implementation. RBG keys are rejected because their `vmap` behavior can ignore
individual row keys. The System derives five actor keys from each environment
key supplied by M8. The caller must supply fresh decision keys.

Scores must be finite, and differences between legal scores must fit float32.
Helpers check static shapes and dtypes. They do not copy arrays to the host to
check numerical preconditions. Dead and inactive actors use Core's existing
neutral-only masks. The helpers do not repair sampled actions.

## Recurrent MAPPO Boundary

[ppo.py](../../src/marl_battlegrounds/baselines/ppo.py) provides these entry points:

- `initialize_ppo` creates separate untrained actor and critic parameters and
  Adam states. `PPOConfig` holds immutable update settings.
- `make_recurrent_mappo_system` receives actor parameters only and returns an
  ordinary M8 `System`. Its learning outputs contain sampled indices and their
  action-time log probabilities. Critic parameters, inputs and memory stay out.
- `critic_values` applies the separate critic to one physical view per game.
- `calculate_gae` calculates advantages and value targets.
- `update_recurrent_ppo` consumes a compact `PPOBatch` and returns updated
  parameters, optimizer states and small loss/sample-count arrays.
  `update_minibatch` is its lower-level grouped numerical boundary.

Actor and critic each use Dense(128)/ReLU, GRU(128), Dense(128)/ReLU, then their
own output head. There is no running normalization or added agent-ID wrapper.
The [source ledger](source_reuse.md) gives the exact donor, initialization,
settings, masks and independent offline reference.

`PPOBatch` uses time first: `(T, B, ...)`. Actions, values, rewards and sample
masks use five Team A slots. `ended[t]` describes the transition produced by
decision `t`; `episode_start[t]` describes the reset before that decision.
Store masks, actions, log probabilities and values from that same decision.
Keep actor and critic memories from immediately before the first decision.

Memory resets on an episode reset. It continues through death, respawn and
ordinary rollout boundaries. Padding preserves memory and contributes no
samples. Inactive actors contribute no samples. Dead active actors keep value
learning, while their forced-neutral actions contribute no policy samples.
Task endings, including horizon endings, stop bootstrap. A collection cutoff
does not: provide the value at its true successor in `final_values`.

B32 means 32 total games split into two fixed groups of sixteen. Each group
shuffles complete game sequences and normalizes its own selected advantages.
No shorter recurrent chunks are supported. Each network averages gradients
over its nonempty groups, clips that average and applies its one shared Adam
state. If a network has no samples in any group, skip its forward/gradient
calculation and preserve all its parameters, moments and counters.

Pass changing weights as `variables_a` or `variables_b` to `apply_systems`
inside compiled workflows. Keep model structure and `PPOConfig` static. Ordinary
same-shaped value changes need no new compiled program. Changed shapes or
settings may compile again. These numerical functions perform no file I/O,
training loop, checkpoint save or hidden host callback.

## What The Checks Establish

Offline source tests check networks, memory, GAE, losses, gradients and Adam
updates against preserved Mava calculations. Separate tests cover BG information
rights, masks, reset timing and public System integration. GPU measurements use
fixed permitted inputs and retained actor outputs. They measure actor-decision
costs, not training throughput.

Training-content tests check verified resources, protected-content separation,
scientific resume compatibility and rejection before writer recovery. Sampling
and tracking tests check map/roster distributions, independent keys, reset-only
ownership and exact source declarations. Recording tests check reconstruction
and recovery without weakening historical claims.

The [distribution benchmark](../../scripts/dev/benchmark_training_distributions.py)
checks the same public loop against identical preselected configurations and
keys. A bounded RTX 5090 check used B32/T128, changing observations, Random
Systems, priority metrics and matching retained outputs. All outputs matched
exactly. Five synchronized warm samples gave these median times:

| Case | Sampled Loop | Preselected Loop | Real Transitions Per Second | Host Writer Drains |
| --- | ---: | ---: | ---: | ---: |
| No reset | 0.888 s | 0.898 s | 4,611 | 1.39 s |
| Partial reset, synthetic horizons | 0.897 s | 0.895 s | 4,565 | 35.44 s |
| All reset, synthetic horizon | 0.899 s | 0.899 s | 4,555 | 137.67 s |

Each loop completed 4,096 real transitions. The two stress cases reset 1,032 and
4,096 episodes. Writer times are additional host work, including exact physical
configuration checks; they are not included in transition rates. Many new
recorded rosters are expensive because the existing host clearance validator
checks each distinct actual configuration. Caching avoids repeated checks of
the same configuration. Disabling recording avoids those writer costs.

Sampled-loop compilation took 7.52–7.90 seconds; first execution took
0.90–0.92 seconds. The retained output was 4.59 MB. Peak live GPU allocations
were 572.5 MB, with a 1.08 GB allocator pool; whole-process peak RAM was
3.29–3.35 GB. These peaks include setup, compilation, reference inputs and writing.
Warmed loops allowed no host/device transfers or callbacks. Changed same-shaped
inputs reused compiled work. These are bounded workflow measurements, not a
general speed guarantee. The benchmark writes setup, compilation, all five
samples, transfers, memory, output sizes and source hashes to `measurement.json`.

The [collection benchmark](../../scripts/dev/benchmark_training_collection.py)
adds untrained recurrent MAPPO actors, compact learning outputs and current or
historical self-play. Its internal RTX 5090 checks also used B32/T128 and five
synchronized warm samples. Each case matched its equivalent direct public-call
loop exactly:

| Case | Collector Median | Direct Median | Real Transitions Per Second |
| --- | ---: | ---: | ---: |
| Current, no reset | 1.010 s | 1.002 s | 4,055 |
| Mixed history, no reset | 1.033 s | 1.026 s | 3,965 |
| Mixed history, native episode continuation | 1.030 s | 1.024 s | 3,977 |
| Mixed history, synthetic reset/stage stress | 1.041 s | 1.033 s | 3,936 |

Native continuation reset 32 games. The labelled stress case reset 1,590 games;
its shortened horizons are test inputs, not training settings. Current and mixed
cases start at different states, so their difference is not isolated opponent
cost. A separate comparison holding inputs, weights, memory and keys fixed gave
0.117 ms for shared current inference and 0.179 ms for mapped history inference.
Those small calls are noisy: the five samples span 0.101–0.180 ms and
0.173–0.560 ms respectively. No optimizer runs in any of these checks.

The actor has 3.21 MB of variables and the twenty-slot bank uses 64.16 MB.
The bank is not stored along the rollout time axis. Each compact rollout uses
214.67 MB; a fresh device-to-host copy took 23.42 ms. Collection compilation
took 10.37–13.58 s and temporary executable storage was about 86.79 MB.
Sampled process GPU memory reached 3.12 GB; process RAM reached 5.05–5.52 GB.
These peaks include setup and both comparison paths, not a learner's memory
requirement. Warm execution forbade host/device transfers and callbacks;
same-shaped changing values reused the compiled program.

One canonical 5v5 recorded block took 16.11 s including its compilation,
with 1.18 s in writer drains and 0.21 MB of durable output. It recorded starts,
not completed games. It does not resolve the new-roster recording costs above.
Earlier cached transfer timings were excluded and replaced with the fresh-copy
measurement. Failed measurement-probe attempts remain in the local evidence.

Shared-batch and per-lane GPU matrix calls can round differently. In the matched
probe, actions agreed while recurrent memory differed by at most 0.000165
(root-mean-square difference 0.00000213). A separate highest-precision check
reduced the maximum to 0.00000891. Production precision stays unchanged. This
supports numerical agreement for these inputs, not bitwise equality or identical
future trajectories across every execution layout.

These checks establish a numerical and integration foundation. They do not
prove sample efficiency, learned team behavior, complete training throughput
or a competent policy within one GPU day. Those need later learning trials.
