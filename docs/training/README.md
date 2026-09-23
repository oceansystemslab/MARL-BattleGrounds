# Training And Baseline Components

For saved models, results and manuscript evidence, start with the
[development checkpoint population](checkpoint_population.md). For tuning
decisions and their limits, use the [baseline methods record](baseline_methods.md).

The `marl_battlegrounds.training` package prepares verified maps, samples
configurations and collects exact experience budgets with optional curriculum,
score shaping and self-play history. The `marl_battlegrounds.baselines` package
provides input encoders, exact action helpers and the PPO and QMIX calculations.
The optional trainer joins these components into recurrent and feedforward
MAPPO and IPPO runs and recurrent QMIX runs with compact replay,
with learner checkpoints, frozen actor loading, validation and analysis.
Collection alone still performs no optimizer update. Ordinary environment and
collection helpers need only base dependencies; PPO and QMIX need `training`
(which includes Flashbax for QMIX replay) and automatic plots need `viz`. A
working trainer is not a claim of learned skill.

Install the existing `training` extra to use PPO or QMIX. In a prepared contributor
checkout, run the example with:

```bash
uv sync --locked --extra dev --extra training
JAX_PLATFORMS=cpu uv run --no-sync python examples/baseline_system.py
```

For the qualified local GPU setup, follow the [GPU guide](../dev/gpu_sanity.md).
The [baseline development record](baseline_methods.md#current-mappo-standard)
records **512 parallel environments and rollout length 32** as our current
MAPPO starting configuration, with the measured results and their limits.
Those learning results, and every result from models trained on the old maps
(maps 0 to 41 with revision 6 of Map 39), are superseded as of 22 September
2026 and are historical only.
The completed five-minute screen's launcher is described under
[Configuration Screen](#configuration-screen).
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
full sampled configuration histories. By default, the sampler keeps K20. A bank
prepared with declared score thresholds can select another exact source at
reset. The sampler does not change the horizon or class capabilities.
Curriculum schedules, shaping and opponent history remain outside this helper.

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
   resolves exact stage budgets on the host. Pass `early_history_capture=True`
   only together with a positive `pinned_opponent_share` in step 2; the
   collection refuses a mismatched pair.
2. `init_training_collection(actor, actor_variables, schedule=schedule, ...)`
   returns a stable `TrainingCollection` and its numerical `TrainingCarry`.
   Supply a native JAX `System` whose lanes are independent and whose memory
   has a leading game axis. Actor-only weights stay dynamic. Policy adapters,
   host methods, custom memory-reset hooks and frozen checkpoint labels are
   unsupported for this trained actor; ordinary environment/System loops remain
   available for those methods. The optional `pinned_opponent=` argument is
   different: it names the opponent that plays the pinned share of games, and
   it accepts any valid System or Policy, including host methods and methods
   with custom reset hooks (see "Current And Historical Self-Play").
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

The small-team stages keep K20/H300. Some sampled class pairs cannot reach that
threshold. For example, Hunter versus Hunter needs at least 2,000 damage for
20 kills, but even 300 Basics plus ten Ultimates deliver at most 1,900 damage,
before travel, shields and healing. Hunter+Priest mirrors also have this limit.
Such games must draw. Played exposure therefore does not establish a useful
win/loss learning signal; report this limitation when using the curriculum.

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

An explicit alternative is `TrainConfig(shaping=True, shaping_mode="score_delta")`.
It adds `shaping_coefficient` for each new team kill and subtracts the same amount
for each new team death. The default coefficient is 0.01. Native win/loss reward
is still added. Simultaneous kills and deaths offset each other. Terminal kills
count, and there is no cancellation at a real ending. Low-level collection uses
the same `shaping_mode` argument and `team_score_delta_shaping` helper.

**Score-delta shaping changes the training objective.** It can reward combat
progress in games that still draw under K20/H300. This is a separate experimental
baseline option, not a proven improvement or a substitute definition of winning.
The default remains `"potential"`; old settings without a mode keep that behavior.
Both modes log task reward and shaping separately, save the mode for exact
resume, and evaluate with unmodified native rewards. Compare them with declared
seeds and budgets. A higher shaped return alone does not show better play.

### Current And Historical Self-Play

Each new opponent game uses current actor weights until history exists. After
that, its assignment is 80% current and 20% uniformly sampled from stored
snapshots. Assignment probabilities are not promises about observed episode or
transition shares. A historical game keeps its chosen weights until it ends.
Current-policy games use newly published actor weights while retaining their
own recurrent memory. Team A and Team B never share memory.

An optional pinned near-start opponent is available through
`TrainConfig(pinned_opponent_share=0.1)`. A positive share moves the first
history capture to the first completed update, so slot 0 holds the actor after
one update for the whole run, and drops the never-played 100% capture. Each new
game then meets slot 0 with that probability, one of the other stored snapshots
with total probability 0.2 once any exist, and current weights otherwise. While
only slot 0 exists, that 0.2 goes to current weights. The share must lie within
[0, 0.8]. Leaving it at zero keeps the recipe above with exactly today's random
draws. Evaluation opponents are unaffected. The realized share of game starts
and steps that met slot 0 appears in `exposure.json`, row 1 of the opponent
lists. A pinned opponent is a development setting for the self-play recipe; it
is not evidence of learning until a declared comparison shows it.

`TrainConfig(pinned_opponent=...)` makes the pinned share play a named System
instead of that first-update actor:

```python
from marl_battlegrounds import training

config = training.TrainConfig(
    pinned_opponent="tdm-alpha",  # or "/abs/path/run/actors/<id>", or "pkg.agents:make_team"
    pinned_opponent_share=0.1,
)
```

The reference is a built-in name (`"random"`, `"tdm-alpha"`, `"tdm-beta"`), an
absolute path to one of our exported actor directories, or a `module:function`
factory that returns a `System` or `Policy`; the library route's
`init_training_collection(..., pinned_opponent=...)` also takes the object
itself. Slot 0 still marks the assignment, so row 1 of the opponent lists in
`exposure.json` counts games against the named System, and `exposure.json`
labels every row. Learner rows record `opponent_update=-2` for those games. The
named System is frozen once, registered the way evaluation registers it, and
follows the same rules as a Team B method in evaluation: only its own games are
valid, it starts each new game with fresh memory, and it may use its own
`reset_memory` hook. A Policy acts per actor, so it is computed on its own
games only: inside the compiled step those games use the smallest fitting
quarter, half or full batch, rounded up. Unused rows are padding, and their
results are ignored.
A generic System is given the whole Team B batch with the other games
marked invalid, and it must ignore them, as M8 requires of every System. A JAX
method runs inside the compiled step and is skipped on steps where no game
uses it; its initializer still runs when one of its games starts. A host
method, such as an LLM agent, runs once per step on the host through the same
helper evaluation uses. A step with no pinned game copies no policy inputs to the host;
otherwise Team B's inputs are copied, about 94 KB per game: only the pinned
games for a Policy, all games for a generic System. The method's own time comes
on top. `collect_training_rollout` runs that route;
`scan_training_rollout` and `advance_training_step` refuse such a collection.
A host collection serves each training round once, and after a method error it
refuses reuse, so a stale carry never meets newer opponent memory.

Resume checks the pinned opponent identity and supported saved memory.
Opaque provider state and remote behavior are not guaranteed to repeat.
JAX methods keep their
memory in the saved state. A host method's memory is never saved: a host method
with no initializer, no reset hook and no memory template has none and resumes
normally, and any other host method can resume only from a checkpoint where
none of its games is unfinished. Use absolute export paths; a moved export
fails the resume check. A named pinned System is a training opponent of that
run and of any run that later pins an export of it. Results against it are
familiar-opponent results, and a different name is not proof of an unfamiliar
opponent: Beta's rules include Alpha's. The run's record says what is known
about the opponent's own training history: `exposure` is `none`, `known` (which
scripted scenario controllers) or `unknown`. Pinning `tdm-alpha` or `tdm-beta`,
directly or through an export trained that way, makes all eight protected
scenarios familiar, so their results from that run are not protected-scenario
evidence. The validation panel is declared separately and does not change
because a System was pinned.

Call `refresh_opponents` once after each completed learner update and before
the next block. Supply already-updated actor-only variables, the exact next
`update_index`, the carry's completed real rounds and its numerical schedule.
Keep the returned history in the carry. The helper performs no learning.
It captures one immutable actor when an unmet 5%, 10%, ..., 100% threshold has
been reached; with a positive pinned share the thresholds are round 1, then 5%
through 95%. Several thresholds crossed by one update share one stored
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

Core observations stay in world coordinates. The optional helpers
`team_on_right`, `mirror_team_view` and `mirror_move` in
[input.py](../../src/marl_battlegrounds/policies/input.py) let any method
reflect a team's permitted view and move mask about the vertical centerline
and map a chosen move back, using only that team's own spawn pads and the map
width. The four PPO baselines call them before encoding when their spawn
frame is `"left"`, its default; the environment, evaluator and tournament never
apply them.

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

These existing MAPPO calls retain their defaults and saved identities. For all
four methods, use the method-aware calls described below.

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
own output head. There is no running input normalization or added agent-ID
wrapper. New learners normalize critic targets as described below.
The [source ledger](source_reuse.md) gives the exact donor, initialization,
settings, masks and independent offline reference.

### PPO Method Choices

Use `TrainConfig(method=...)` to choose the full workflow. The default is
`"mappo"`.

| Method | Actor | Critic |
| --- | --- | --- |
| `mappo` | 128-wide recurrent network | Physical training state, separate memory |
| `ippo` | Same recurrent network | Each actor's permitted inputs, separate memory |
| `ff_mappo` | Two 128-wide ReLU layers | Physical training state, no memory |
| `ff_ippo` | Two 128-wide ReLU layers | Each actor's permitted inputs, no memory |

The actor and critic always have separate parameters and optimizers. IPPO
collects no physical-state features. Its critic expands compact permitted
observations one time row at a time and reuses encoded actor inputs inside
each learning minibatch. It cannot read another actor's private row or memory.
Feedforward System and critic memory are empty tuples; there are no unused
recurrent arrays or GRU calls.

Low-level users pass the same static `method` to `initialize_ppo`,
`make_ppo_system`, `init_learner`, `build_ppo_batch`, `update_learner`,
`validate_learner` and
checkpoint save/export/restore calls. `update_ppo` selects the complete numerical
update. Existing `make_recurrent_mappo_system` and `update_recurrent_ppo` calls
keep their MAPPO meaning. Ordinary researchers need only the saved method in
`TrainConfig`; the runner passes it to these owners.

Returns and advantages always use time order. Recurrent minibatches keep whole
game sequences. Feedforward minibatches shuffle time/game rows within each
group, keeping five actor slots together. With G groups and M minibatches,
recurrent B must be divisible by G times M. Feedforward B must be divisible by
G, and T times B must be divisible by G times M. Here B is the environment
batch and T is the rollout length. Training needs an even B for paired spawn
ends. For example,
B6/T4/G2/M2 works for feedforward training but cannot form recurrent minibatches.

All four methods share ValueNorm, input-scale and spawn-frame settings below.
Recurrent IPPO uses the existing curriculum and shaping options. The initial
feedforward experiment recipes are plain; reusable flags remain available but
do not establish feedforward treatment results. Parameter counts are listed in
the [source ledger](source_reuse.md#ppo-model-choices). These architectures have
different parameter counts, so their comparison is not a pure test of memory.

`PPOConfig(input_scale=0.01)` explicitly scales actor and critic inputs before
their first layer. The default `1.0` preserves the donor and old models. Use a
different scale only as a declared numerical experiment; reduced memory-gate
saturation does not prove better learning. The runner carries the setting into
collection, value targets, updates, checkpoints and loaded actors. Low-level
users must pass the same `input_scale` to `make_recurrent_mappo_system` and
`critic_values`; pass the original `ppo` config to `build_ppo_batch`.
The [source ledger](source_reuse.md#optional-input-scale) explains compatibility.

`PPOConfig.spawn_frame` defaults to `"left"`: the actor always sees the game as
if its team started on the left bank. Whenever its own team starts on the
right, its permitted view is reflected about the map's vertical centerline
before the network and the chosen move is reflected back before the game
receives it. `spawn_frame="world"` keeps raw coordinates. Historical runs also
need their saved numerical settings and source. The default changed from
`"world"` on 22 September 2026,
and the value `"right"` was removed. Saved checkpoints, exported actors and run
records without a frame still mean `"world"`, and resuming such a run keeps
it. An old training config file, however, names no frame, so starting a new
run from it now trains in `"left"`; add `"spawn_frame": "world"` to its `ppo`
block to reproduce the old run. The JSON examples in this guide name no frame
and therefore mean `"left"`. Every
built-in Team Deathmatch map is its own mirror image, so the reflected view is
exact up to float rounding; the game itself reproduces a mirrored match only
approximately after contact, as the reflection diagnostics in amendment A37
record. The setting changes no rule and reads nothing beyond the actor's own
view. It is a Team Deathmatch adapter: objective positions for later game
modes are not reflected. Measured cost on an RTX 5090 at 512 games and
32-step rollouts, same session, alternating frames: one collection block and
one update take the same time under `"left"` as under `"world"` within
measurement noise (about 433 ms and 93 ms), because the only quadratic part,
the obstacle mirror decision, is made once per game and shared by a team's
five actors (`team_obstacle_partners`); the update keeps about 2 ms of that
decision, 0.4 percent of a block. (Measured 2026-09-22, training on the old
maps with revision 6 of Map 39; these costs describe that original workload.)
The adapter gives both spawn ends a consistent horizontal orientation. It
does not guarantee equal actions or outcomes. Whether it learns faster or
better is a separate measured question. The runner carries the setting into
collection, updates, checkpoints, exported actors and their inference identity;
a loaded actor plays in the frame it was trained in, never a guessed one.
Low-level users pass the same `spawn_frame` to `make_recurrent_mappo_system`,
whose default follows `PPOConfig` (so old world weights need
`spawn_frame="world"`), and the original `ppo` config to `update_recurrent_ppo`.
`export_system` has no default frame: it writes a durable identity, so the
caller must name the frame the weights were trained in. The
[source ledger](source_reuse.md#optional-spawn-frame) gives the precedent and
compatibility rules.

`PPOConfig.value_normalization=True` is the default for new runs. It scales
critic targets with one learner-owned running mean and variance. Rewards and
GAE keep their original units. Dead active agents contribute to critic targets;
inactive slots and padding do not. Statistics update once per nonempty
optimizer minibatch and roll back if the complete update is rejected.

`PPOBatch.old_values` and `final_values` remain in reward units. Enabled
normalization also requires `old_normalized_values`: the exact network outputs
from collection-time critic evaluation, before statistics change. This keeps
value clipping anchored to the predictions that generated the batch.
`build_ppo_batch` supplies both forms from the same forward pass. Low-level
`critic_values` callers pass the learner's `value_norm` to obtain reward units.

Checkpoints carry the three scalar statistics; actor exports do not. Restored
historical settings that omit the flag mean `False`, preserving the old array
layout. To start a fresh run with the former numerical recipe, set
`value_normalization=False` explicitly. See the
[source ledger](source_reuse.md#critic-target-normalization) for the fixed
constants, donor source and deliberate adaptations.

PPO metrics report approximate policy KL (change in action probabilities),
policy and value clipping fractions, entropy, actor and critic gradient norms
before clipping, raw target mean and standard deviation, raw predicted mean
and root-mean-square value error, and the normalization mean and scale.
Value loss uses normalized network units when normalization is enabled.
Diagnostics reuse the loss forward passes. These values help diagnose a run;
there is no universal threshold that proves or disproves useful learning.

When investigating weak learning, inspect actual task outcomes and fixed-opponent
validation before increasing the budget. Under native K20/H300, a game without
a winner at the horizon is a draw even when its kill scores differ. Potential
shaping preserves that objective; it does not turn such draws into wins. Low
value loss and a negative actor loss do not prove progress: the actor loss also
includes its entropy bonus. Compare input scales, spawn frames and update
settings on declared development seeds, and keep test-map outcomes out of
those choices.

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

## Recurrent QMIX

Choose QMIX with `TrainConfig(method="qmix")`. QMIX runs use their own
settings object, `TrainConfig.qmix` (a `QMIXConfig`), and leave `ppo` at its
default; saved configs hold a `"qmix"` block and no `"ppo"` block. Everything
else is the same workflow: train, resume, validate, select, export, load,
evaluate and analyze. The [QMIX example](../../examples/qmix_training.py) runs
a tiny complete case:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training
from marl_battlegrounds.baselines.qmix import QMIXConfig

config = training.TrainConfig(
    method="qmix",
    num_envs=4,
    total_env_steps=192,
    qmix=QMIXConfig(
        rollout_length=8,
        buffer_size=64,
        min_buffer_size=32,
        sample_sequence_length=20,
        sample_batch_size=4,
        epochs=1,
    ),
)
result = training.train(config, output_dir="runs/qmix-example")
actor = training.load_system(result.final_actor)
marl_bgs.evaluate(actor, "random", num_episodes=2, maps=[42], num_envs=2,
                  phase="validation", output_dir="runs/qmix-example/evaluation")
training.analyze([result.run_dir], output_dir="runs/qmix-example/analysis")
```

Curriculum and reward shaping are the usual one-line changes:
`curriculum=True`, `shaping=True`, or both. Shaping uses `qmix.gamma`.

**Team reward.** QMIX learns from one team reward per decision: the mean task
reward over the Team A slots that are in the game, plus the shaping value when
shaping is on. The donor averages over all five slots; the two differ only
when a curriculum stage has fewer than five players per team.

**Model and actors.** One shared local Q-network (256 wide, with a 256-wide
GRU) turns each actor's permitted inputs and own memory into 198 action
values. The mixer combines the five chosen values into one team value from the
919-value physical state; it never reaches an actor. Loaded and exported QMIX
Systems play greedily: epsilon 0, and among equal best values the first legal
action in the network's own order (mirrored for left-frame games). Details,
parameter counts and the donor are in the
[source ledger](source_reuse.md#recurrent-qmix).

**Exploration.** Before every decision the collection sets the current actor's
epsilon from a clock of real transitions: it falls from 1 to `eps_min` over
`eps_decay` transitions (all games counted) and then stays at `eps_min`. Both
current self-play teams share it. A historical opponent keeps the epsilon it
had when it was captured, so an early history snapshot or pinned first-update
actor can stay very exploratory for the whole run. Warmup publishes nothing,
so a history snapshot due during warmup is taken at the first learning block,
with that block's weights and epsilon.

**Replay and warmup.** Here "game" means one of the `num_envs` parallel
games; each plays many episodes. Replay keeps the last `buffer_size` decisions
of each parallel game; it is never flushed, including across curriculum
stages. A block adds its real decisions, then learns only once at least
`min_buffer_size` rows per game are stored. Earlier blocks are warmup blocks:
their decisions are stored, and no optimizer step runs. Each ready block takes
`epochs` optimizer steps; each step draws `sample_batch_size` sequences of
`sample_sequence_length` consecutive decisions from one parallel game, with
replacement, and starts their memory at zero (no burn-in). A sequence may
cross an episode ending; the learning target stops at the ending and the next
episode starts with fresh memory. The newest decision of each game only ever
appears as the last row of a sequence, because its successor is not stored
yet. Targets copy the online networks every `update_period` optimizer steps
(or blend with `tau`).

**Counts.** `completed_updates` counts optimizer steps, not blocks. The
training log adds blocks, learning blocks, sampled sequences, used TD pairs and
used actor utilities. A TD pair is two consecutive stored decisions of one
parallel game, the unit of one learning target; each active actor in a used
pair adds one actor utility. Used counts include repeated draws of the same
stored rows; they are not new experience. `exposure.json` adds the used TD pairs by
curriculum stage, source row and opponent. The runner keeps these totals as
exact whole numbers; low-level callers of `update_qmix_learner` receive
per-block counts and must add them up themselves.

**Memory and disk.** One replay row is 29,664 bytes per game: 949,248,000
bytes for 32 games and 1,000 rows. While a block is checked the learner keeps
the previous replay (for rollback) and the replay with the new rows, plus
working space, so plan for up to about three times that; at those sizes XLA
estimates about 4.0 GB for the whole update on CPU, and the GPU run used about
3.3 GB. Each complete
checkpoint stores the replay, the networks, targets, optimizer and history, so
QMIX saves hold about 1.1 GB of arrays at those sizes. Files are compressed,
so disk use is lower and depends on the contents; a save with 88 of 1,000 rows
filled took about 80 MB. Plan disk for the uncompressed size. `checkpoint_interval_updates`
counts `completed_updates`: PPO updates for PPO, optimizer steps for QMIX. Its
QMIX default of 1,600 gives the same spacing as PPO's 25 at the defaults:
102,400 transitions between saves at 32 games (PPO: 25 updates of 128 rounds;
QMIX: 1,600 steps at 4 per block of 8 rounds). An explicit value is always
kept. The number is fixed when the config is built, so
`dataclasses.replace(ppo_config, method="qmix")` keeps PPO's 25 unless that
call also passes `checkpoint_interval_updates=None`. Exports hold only the
Q-network, about 7 MB.

A run keeps the complete checkpoint, replay included, at every boundary with an
exported actor (initialization, each validation fraction, each extra save point
and the final boundary), plus the two newest recovery saves. A ten-fraction
panel run at 32 games therefore keeps about 11 complete checkpoints, up to
roughly 12 GB before compression. Memory and disk grow with
`num_envs × buffer_size`: at 256 games one replay is about 7.6 GB and a block
briefly holds three (about 23 GB), and 512 games with 1,000 rows do not fit a
32 GB card. Lower `buffer_size` or `num_envs` for large batches.

**Recovery.** A resume restores into a shape-only template, so restoring never
holds a second replay. During training a block holds the previous replay and
the new one plus working space, as described under Memory and disk. A resume
rejects impossible counters, the wrong
method, changed QMIX settings or a changed replay layout before reading arrays.
CPU tests show that a resumed run continues bit for bit, including a learner
continued in a fresh process; the matching GPU check is a separate
qualification.

**Tournaments.** A loaded QMIX actor is an ordinary `System`, so evaluation and
tournaments use it unchanged:

```python
selected = training.load_system(result.selected_actor or result.final_actor)
tournament = marl_bgs.run_tournament(
    [selected, "tdm-alpha", "tdm-beta"],
    episodes_per_pair=20,
    num_envs=32,
    output_dir="runs/qmix-tournament",
)
print(tournament.table("tournament_rankings"))
```

Read the rankings and uncertainty as described in
[Run a tournament](../evaluation/workflows.md#run-a-tournament); the
[canonical tournament guide](../evaluation/canonical_tournaments.md) covers the
fixed Big 12 comparison, and [examples/evaluation.py](../../examples/evaluation.py)
runs the same call from the command line.

Low-level users import `from marl_battlegrounds.training import qmix_learner`
and call `qmix_learner.init_qmix_learner`, `update_qmix_learner` and
`validate_qmix_learner`, with the numerical pieces in
[baselines.qmix](../../src/marl_battlegrounds/baselines/qmix.py). A short run
proves software wiring only; it does not show learning or sample efficiency.

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

## Train, Resume, Load And Analyze

Install both `training` and `viz` extras for the complete workflow. The package
keeps one training loop for Python and the CLI. External methods may still use
ordinary environment calls and the collection helpers without this trainer.

From a contributor checkout, install the needed dependencies with:

```bash
uv sync --locked --extra dev --extra training --extra viz
```

This small CPU example checks training, saving, loading and reporting. It is too
short to show useful learning. With no validation panel, `selected_actor` is `None`:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/mappo_training.py \
  --output-dir artifacts/mappo-example
```

The same route from Python is:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training
from marl_battlegrounds.baselines.ppo import PPOConfig

result = training.train(
    training.TrainConfig(
        seed=42, num_envs=4, total_env_steps=32,
        ppo=PPOConfig(rollout_length=4, epochs=1),
    ),
    output_dir="artifacts/mappo-python-example",
)
system = training.load_system(result.selected_actor or result.final_actor)
reports = training.analyze([result.run_dir], output_dir="artifacts/mappo-report")
```

Set `JAX_PLATFORMS=cpu` before starting Python for this four-environment example.
For new MAPPO development runs, explicitly set `num_envs=512` and
`PPOConfig(rollout_length=32)`. The [standard recipe record](
baseline_methods.md#current-mappo-standard) describes the supporting reward and
input settings and the limits of the evidence. The original Packet 4 example
below preserves its historical B32/T128 recipe. This small CPU example does not
select a GPU or launch either experiment.

`TrainConfig` defaults to plain recurrent MAPPO, 32 environments, rollout length
128 and 10,000,000 real transitions. `curriculum` and `shaping` are independent
Boolean choices. `recording=False` skips episode-table drains; update records,
learner checkpoints and final reports still save. Set `method="ippo"`,
`method="ff_mappo"` or `method="ff_ippo"` to use another PPO method with the same
workflow. Unknown method values fail before output or recovery work.

For a small complete example with saved selection and evaluation, use:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training
from marl_battlegrounds.baselines.ppo import PPOConfig

result = training.train(
    training.TrainConfig(
        method="ippo", seed=19047101, num_envs=4, total_env_steps=32,
        ppo=PPOConfig(rollout_length=4, epochs=1),
        validation_opponents=("tdm-alpha",), purpose="development",
        validation_fractions=(0.5, 1.0),
        routine_seed_pairs=1, confirmation_seed_pairs=1,
    ),
    output_dir="artifacts/ippo-workflow",
)
actor = training.load_system(result.selected_actor or result.final_actor)
games = marl_bgs.evaluate(
    actor, "tdm-alpha", num_episodes=2, maps=[42], seed=19047103,
    num_envs=2, phase="validation", output_dir="artifacts/ippo-evaluation",
)
reports = training.analyze(
    [result.run_dir], output_dir="artifacts/ippo-analysis",
)
```

Run this on CPU. Change only `method` to choose any of the four implementations;
use a fresh output directory for each run. These tiny budgets check the workflow,
not useful play. For recurrent IPPO curriculum/shaping wiring, use 256 transitions
and `curriculum=True`, `shaping=True`, or both in the same config. Set
`validation_opponents=None` for those small wiring checks. A short check
does not prove that every curriculum stage received useful experience.

A new run requires a new or empty exact output directory. Resume requires the
path of a complete learner checkpoint, not an actor export or run directory:

```python
resumed = training.train(resume_from="RUN/checkpoints/CHECKPOINT_ID")
```

Omit config to inherit it. Supplying one asserts exact equality; resume cannot
change the seed, budget, panel or method. Checkpoint and content validation
happen before any writer rewind. Failures preserve the last published checkpoint
and report an error; no run silently replaces a seed or increases its budget.

CLI configuration is a JSON object with `schema_version: 1` and the same
`TrainConfig` field names. PPO options belong in a nested `ppo` object; a
QMIX config has `"method": "qmix"` and a nested `qmix` object instead, for
example `{"method": "qmix", "num_envs": 4, "total_env_steps": 192, "qmix":
{"rollout_length": 8, "buffer_size": 64, "min_buffer_size": 32}}`. Omitted
fields use the same defaults; unknown fields fail. These commands call the same
Python functions. For a small CPU development run, save this as `CONFIG.json`:

```json
{
  "schema_version": 1,
  "seed": 42,
  "num_envs": 4,
  "total_env_steps": 32,
  "ppo": {"rollout_length": 4, "epochs": 1}
}
```

Choose a new `RUN` directory for the first command. The resume command needs an
actual checkpoint path from that run's `status.json` or `latest_checkpoint.json`;
replace `CHECKPOINT_ID` with its saved identity. `REPORT` holds derived reports:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python -m marl_battlegrounds train \
  --config CONFIG.json --output-dir RUN
JAX_PLATFORMS=cpu uv run --no-sync python -m marl_battlegrounds train \
  --resume-from RUN/checkpoints/CHECKPOINT_ID
uv run --no-sync python -m marl_battlegrounds analyze-training RUN --output-dir REPORT
```

The [complete example](../../examples/mappo_training.py) also accepts
`--method ippo`, `--method ff_mappo` or `--method ff_ippo` for its tiny new run. Omitting
`--method` keeps MAPPO; `--method mappo` makes it explicit. For example:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/mappo_training.py \
  --method ff_ippo --output-dir artifacts/ff-ippo-example --evaluate
```

The example also accepts `--config`
or `--resume-from`. Adding `--evaluate` loads the saved actor, plays two CPU
diagnostic games against Random and saves ordinary M8 results. These are wiring
checks, not useful-learning evidence or checkpoint-selection results. An explicit
`--method` cannot accompany `--config` or `--resume-from`; those settings already
name the saved method. The package CLI reads method from the JSON config.

### Progress And Costs

With `verbose=True`, readable progress appears at completed updates, at most once
per 10 seconds, plus phase changes and errors. It reports real transitions,
updates, percentage, elapsed time, recent/average training throughput, estimated
training time remaining and available learning summaries. Whole-run ETA also
includes estimated pending validation, saving and final diagnostics when their
measured costs are available. Unknown estimates say **Estimating**. Estimates
are not promised finish times. Training throughput excludes validation and saving.

Training Speed is the measured average, including compilation. Training ETA
uses a separate estimate for the current attempt: it skips the first update,
waits for two warm updates, then uses up to the last five completed updates.
It divides their real transitions by their measured collection and update time.
Resume resets this estimate, so it shows Estimating again until enough warm
samples exist. Once the training budget is complete, Training ETA is zero;
whole-run work can still remain.

The display uses host records already required by training. It adds no model
calls, GPU synchronization, device transfers or evaluation games. Its full-run
use is conditional on the matched reporting-cost check showing no measurable
slowdown; otherwise use `verbose=False` and the existing logs/status command.
Quiet mode skips terminal display and the extra ETA calculations. Required
counters, raw timings and durable status still update. The status command shows
**Estimating** for forecasts without timing estimates; completed work shows zero
remaining time.
Loss, entropy and changing self-play reward help diagnose learning, but fixed
opponent task scores are the main progress evidence.

Reports label `elapsed_seconds` as Recorded Active Time. Resume carries forward
the larger saved checkpoint or same-run status value, then adds the current
attempt. Interrupted work without a durable timing record may be missing, so
this is not a complete total of every attempt's active cost. `wall_seconds`
measures time since run creation, including pauses and recovery gaps. Reports
keep both fields separate. The elapsed-time plot uses Recorded Active Time;
training throughput still excludes validation and saving. The phase-cost table
uses completed event records from all attempts and also names missing costs.

### Saved State And Reports

`run_details.json` holds immutable settings and source, dependency, content and
panel identities, plus schema versions and the initial M8 runtime record. That
runtime record names the backend, device, precision and batch. Declared environment
selectors are saved too; they do not verify the physical GPU's UUID or PCI bus.
The frozen launch package owns that hardware check.
The saved execution identity also records the training compiler policy, backend,
device kind, numerical settings and declared `XLA_FLAGS`. Resume checks it before
rewinding recordings or logs. A different or missing identity rejects learner
continuation; older actor exports and reports remain readable.
`training_updates.jsonl` holds accepted updates; `run_events.jsonl` retains each
attempt's runtime, failures and evaluation execution segments.
`status.json` is the current view, not the scientific source of truth. A missing
process is not proof of successful completion.

Completion and failure record a best-effort memory snapshot in status and the
terminal event. Process peak RAM includes earlier work in the same Python
process. JAX allocator counters exclude driver/display memory and CPU worker
processes; they are not guaranteed whole-process VRAM. Missing counters and read
errors remain explicit. Reports use these saved records without making new
measurements. Earlier terminal snapshots remain historical and their peaks must
not be added together. Older records without runtime, schema or memory fields
remain unknown.

Recovery preserves abandoned update records and report bytes under `attempts/`.
It then rebuilds the active summaries from the restored boundary. A torn event
line stays preserved and is reported as unknown work; it never becomes a
successful update or validation result. Training-update prefixes remain strict.

Full checkpoints retain actor/critic parameters, both optimizers, random streams,
game state, separate memories, tracking, curriculum, opponent history and host
selection state. They bind exact log prefixes and optional recording tokens.
They publish at initialization, whenever `completed_updates` crosses a multiple
of `checkpoint_interval_updates` (25 PPO updates or 1,600 QMIX optimizer steps
by default), declared capture/validation points and the final update. An immutable checkpoint ID includes its actual
payload and continuation ancestry; an update number alone is not an identity.
Ordinary local filesystem rename/fsync semantics are required.

Setup and restore verify the installed training content. Periodic saves reuse
that checked descriptor instead of rebuilding the installed content each time.
Every save still checks the actual source bank carried by the learner, its
numerical state and its counters. Restore places every array on the selected
device, including empty recording and metric IDs.

Built-in PPO fixes GPU kernel selection with `xla_gpu_autotune_level=0` on its
outer collection and update compilations, including recorded collection. This
avoids choosing different kernels from fresh timing trials after a restart.
It changes no process-wide JAX setting. Raw numerical helpers remain usable in
researchers' own compiled loops; the M8 evaluator keeps its existing settings.
See [the compiler policy and its limits](source_reuse.md#training-compiler-policy).
Fresh-process public training checks matched continued state, actions and actor
exports exactly, with and without recording, using this built-in policy and no
global autotune flag. See the [measured engineering checks](#measured-mappo-engineering-checks).
This evidence does not promise equality across hardware or library changes.

Frozen actor exports contain only deployment data and provenance. PPO exports
load as sampled M8 Systems and QMIX exports load as greedy ones, without
critic or mixer state, optimizer state or training maps. Death
and respawn preserve actor memory; episode resets clear it. The training-only
critic view never enters actor decisions or exported actor memory.

Saved model tags distinguish `recurrent_mappo_128`, `recurrent_ippo_128`,
`feedforward_mappo_128x128`, `feedforward_ippo_128x128` and
`recurrent_qmix_256`. The loader reads the
validated schema; it never guesses the method from array shapes or filenames.
Equal actor weights from different methods have different inference identities.
New variant identities include the model, weights, scale and frame. Historical
MAPPO schema and inference identities remain unchanged. Missing fields in an
old saved config mean MAPPO, input scale 1.0, world frame and ValueNorm off.
These meanings apply to loading saved work; new configs use today's defaults.
Exact learner resume still requires the original source, dependencies and
execution settings. Actor-only loading supports independent deployment.

Automatic reports include `run_summary.md`, `learning_curve.csv`,
`learning_curves.png`, `learning_curves.svg` and validation cell results; a
report with a QMIX run adds `qmix_curves.png` and `qmix_curves.svg` (loss,
replay rows, used TD pairs per transition and exploration rate). Analysis
reads saved evidence without changing original scores or checkpoints. Interrupted
final validation, selection, export and reporting can resume after training has
used its exact budget, without collecting another transition.

The summary links actual stage/map/opponent exposure in `exposure.json`. It
separates task reward per active-agent sample from shaping per real environment
transition. Missing validation stays missing, and plots identify their frozen
panel. Failure updates the Markdown status without rerunning games or plots.

### Validation And Checkpoint Selection

Validation accepts any valid `System` or `Policy`: scripted, learned, compiled,
Python, external, or a mixture. Each method receives only its permitted inputs.
A panel fixes the opponents before training; it does not require MAPPO opponents.
A development run may omit validation and then returns no selected actor. A
`purpose="demonstration"` run requires a panel and completes checkpoint selection.

For ordinary Alpha validation, declare its existing built-in name:

```python
from marl_battlegrounds import training

config = training.TrainConfig(
    validation_opponents=("tdm-alpha",),
    routine_seed_pairs=4,
    confirmation_seed_pairs=10,
)
result = training.train(config, output_dir="artifacts/alpha-validation")
print(result.selected_actor)
```

A reference may also be an absolute exported-actor path or an installed
`module:function` factory returning a `System` or `Policy`. The factory runs once
when the panel is loaded. Python callers may instead pass their live methods to
`train(config, ..., validation_opponents=(my_system,))`. Live clients stay in the
current process. No client object is written to JSON. On resume, supply the same
live bindings again; the saved M8 identities must match before recovery changes.
External services may have state that MARL-BGs cannot freeze or reproduce. Their
registration keeps that evidence unknown rather than claiming exact behavior.

For shared fixed panels or explicitly declared validation seeds:

```python
from marl_battlegrounds.training.validation import create_panel, validate_checkpoint

panel = create_panel(
    opponents=("tdm-alpha",),
    roots={"routine": 19046300, "confirmation": 19046400},
    output_dir="artifacts/alpha-panel",
)
validation = validate_checkpoint(
    "RUN/actors/CHECKPOINT_ID", panel,
    output_dir="artifacts/standalone-validation", seed_pairs=4,
)
print(validation["score"], validation["mean_kill_difference"])
```

`TrainConfig(validation_panel=".../panel.json")` reuses this saved panel. Declare
opponents once, either in config or in `train`; explicit bindings alongside a
saved panel may only restore its existing members. `load_panel(path, bindings=...)`
also restores live-only methods. Changing frozen membership, roots or identities
requires a new panel. Repeating `validate_checkpoint` with the same task directory
resumes only its unfinished games. It never updates an actor or resumes training.

Validation uses maps 42–46, mirrored canonical 5v5, K20/H300, and both spawn ends.
Each seed pair means ten games per opponent across those five maps. Defaults are
10 routine pairs and 50 confirmation pairs. Initialization cannot be selected.
Requested progress fractions round up to completed blocks (PPO updates or QMIX
blocks); final is included. QMIX warmup actors, with no optimizer step, are
never selected.
The best two routine checkpoints plus final, when distinct, receive fresh
confirmation. The highest native score wins: win 1, draw 0.5, loss 0. Exact ties
use mean kill difference, then earlier training step, then checkpoint ID.
Incomplete cells cannot select a checkpoint. Maps and opponents have equal weight.

New panels give each opponent its own deterministic root derived from the declared
purpose root and saved method identity. Routine games stay matched across
checkpoints. Confirmation uses a separate root. Intervals keep the two spawn ends
of each game seed together and resample opponents independently. They describe
game sampling for these fixed actors, not variation across training seeds.
For a separate fixed-actor assessment, use `purpose="assessment"`, an explicit
fresh `root_seed`, and the desired `seed_pairs`. Assessment does not select again.

New panels use fixed 32-lane GPU evaluation, including short initial and resumed
tails. Unused lanes are inactive; they produce no games, records or provider calls.
Opaque host clients stay in process. The learner is paused during validation;
validation has its own state and random keys.

An existing development tournament may select a panel with
`create_panel(opponents=candidate_methods, ranking=result, size=2, output_dir=...)`.
`ranking` may be a complete `TournamentResult` or its saved run directory. It must
cover the whole declared pool on current canonical maps 42–46, 5v5, K20/H300 and
paired spawn ends. The existing M8 owner verifies actual configurations, method
identities and completed pairs. Highest saved Elo wins; registration IDs break
ties. This uses an existing ranking and does not start another tournament.

The old positional `create_panel(halfway, final, ...)` route remains readable and
keeps its original hashes, Random qualification, shared opponent seeds, earlier
step tie rule and CPU tail recovery. Those historical rules do not silently
change when an old run resumes. The optional slot diagnostic compares two
actor exports of the run's method. New panels require `slot_diagnostic_actor` to name that
comparison explicitly; ordinary validation works without it.


## A Panel-Backed MAPPO Demonstration

This reusable example trains plain recurrent MAPPO with 32 environments,
128 transitions per environment per rollout and 10,000,000 real environment
transitions. It uses seed 19042001, the default PPO settings, priority metrics
and no episode-table recording. Curriculum and shaping are off. The declared
slot diagnostic runs after training finishes.

This example uses the historical two-actor panel and slot diagnostic. New
experiments can use the generic System panels shown above. For this example,
you need an existing qualified two-member `panel.json`. Its fixed members must
have passed their declared Random checks. A test-only panel is not accepted for
a demonstration. Qualification does not mean the panel proves broad competence.
If the declared panel fails its checks, the scientific launch is not qualified.
Keep the failed results. Choosing different members or changing development
settings is a separate development decision; do not replace the panel silently.

These are alternative ways to run the same experiment. Choose one route. The
prepared package is the route for an unattended run whose source and dependencies
must stay fixed while the development checkout changes.

### Save The Settings

Save the following as `/absolute/path/mappo-demo.json`. Replace the panel path
with the absolute path of the existing qualified panel. The other values below
fully state this example's settings. Changing a value creates a different
experiment and may need new qualification. The spawn frame is written out
because its default changed on 22 September 2026; a demonstration declared
before then used `"world"`.

```json
{
  "schema_version": 1,
  "seed": 19042001,
  "method": "mappo",
  "num_envs": 32,
  "total_env_steps": 10000000,
  "curriculum": false,
  "shaping": false,
  "shaping_coefficient": 0.01,
  "ppo": {
    "actor_lr": 0.00025,
    "critic_lr": 0.00025,
    "rollout_length": 128,
    "epochs": 4,
    "minibatches": 2,
    "groups": 2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
    "max_grad_norm": 0.5,
    "adam_epsilon": 0.00001,
    "spawn_frame": "left"
  },
  "metrics": "priority",
  "recording": false,
  "validation_panel": "/absolute/path/qualified-panel/panel.json",
  "validation_fractions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  "routine_seed_pairs": 10,
  "confirmation_seed_pairs": 50,
  "checkpoint_interval_updates": 25,
  "checkpoint_env_steps": [],
  "slot_diagnostic": true,
  "purpose": "demonstration",
  "verbose": true
}
```

The trainer keeps native K20/H300 games. Initialization is a diagnostic check.
Progress checks round up to completed blocks, and final is always included.
Routine checks use 200 games; fresh confirmation uses 1,000 games per candidate.
The best two eligible routine actors plus final, when different, receive
confirmation. The slot diagnostic adds its declared 3,200 games after training.
These games are separate from the 10,000,000 training transitions.

Use `verbose: true` only when the reporting-cost check supports it for the
qualified workflow. Progress reuses existing host records; the cost check cannot
prove literally zero overhead. Automatic saved reports remain available with
`verbose: false`.

### Prepare An Unattended Run

Use an already qualified local commit and the verified internal RTX 5090 UUID.
The [preparation tool](../../scripts/dev/prepare_mappo_run.py) reads that commit,
copies its source and panel into a new package, and installs a separate
environment from that source's lockfile. It does not start training or create a
commit. Installation may use the package cache or download locked dependencies.

From the contributor checkout, install the tool's dependencies if needed:

```bash
uv sync --locked --extra dev --extra training --extra viz
```

Replace every `/absolute/path/...` value below. `QUALIFIED_COMMIT_SHA` must be the
reviewed implementation's commit. `GPU-INTERNAL-UUID` must be the verified full
hardware UUID, not a device ordinal such as `0`. The package destination must
not already exist.

```bash
uv run --no-sync python scripts/dev/prepare_mappo_run.py \
  --repository /absolute/path/MARL-BattleGrounds \
  --commit QUALIFIED_COMMIT_SHA \
  --config /absolute/path/mappo-demo.json \
  --destination /absolute/path/mappo-run-package \
  --gpu-uuid GPU-INTERNAL-UUID
```

Preparation prints exact absolute launch, status and resume commands. For the
example destination above, they are:

```bash
bash /absolute/path/mappo-run-package/launch.sh
bash /absolute/path/mappo-run-package/status.sh
bash /absolute/path/mappo-run-package/resume.sh
```

Run `launch.sh` once to start a new experiment. It detaches training, closes its
input and sends output to the package's log. The terminal or conversation may
close while the run continues. Use `status.sh` to read progress and process
state. To follow the saved output, use:

```bash
tail -f /absolute/path/mappo-run-package/logs/process.log
```

Press Ctrl+C in that log-following terminal to stop following the log; this does
not stop training. The run writes results under
`/absolute/path/mappo-run-package/run`. It completes validation, selection,
exports, the declared slot diagnostic and reports before reporting success.
A process disappearing by itself is not proof of success.

After an interruption, use `resume.sh`. It uses the last published complete
checkpoint, or the exact checkpoint named by an unfinished recovery marker.
It does not silently substitute a different checkpoint after a failed check.
To request a particular complete checkpoint, use its absolute path:

```bash
bash /absolute/path/mappo-run-package/resume.sh \
  --checkpoint /absolute/path/mappo-run-package/run/checkpoints/CHECKPOINT_ID
```

Resume keeps the same source, environment, settings, panel and run identity.
It finishes pending output work without adding training transitions after the
budget. Launch and resume check package files, imports and hardware identity.
Keep the package in its prepared location, and keep its files unchanged. Its
commands and copied configuration contain absolute paths.

The final handoff must give the actual checked commands, output path and cost
estimate. These reusable examples do not qualify a current scientific launch.
Training-only throughput cannot predict total cost without measured setup,
checkpoint, validation, diagnostic and report costs.

### Use The Same Settings From Python

For an ordinary foreground run in a qualified environment, save this as
`mappo_demo.py`. Replace the paths. Set the verified GPU selection before starting
Python, and use a new output directory. This is an alternative to the prepared
launch above; do not start both for the same declared experiment.

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training
from marl_battlegrounds.training.runner import read_config

config = read_config("/absolute/path/mappo-demo.json")
result = training.train(config, output_dir="/absolute/path/mappo-direct-run")
if result.selected_actor is None:
    raise RuntimeError("This demonstration did not produce a selected actor")
system = training.load_system(result.selected_actor)
reports = training.analyze(
    [result.run_dir], output_dir="/absolute/path/mappo-direct-report"
)
print(result.selected_actor)
print(reports["artifacts"]["summary"])
```

The returned `system` is an ordinary frozen M8 System that can be passed to
`marl_bgs.evaluate` or other compatible evaluation tools. Loading performs no
training or games. To resume from Python, replace the new-run call with:

```python
result = training.train(
    resume_from="/absolute/path/mappo-direct-run/checkpoints/CHECKPOINT_ID"
)
```

No config is needed on resume; it inherits the saved settings. Passing a config
asserts exact equality and cannot change the experiment.

### Use The Same Settings From The CLI

The direct GPU route needs the locked `training`, `viz` and `cuda13` extras.
Install them in the qualified checkout before running it:

```bash
uv sync --locked --extra dev --extra training --extra viz --extra cuda13
```

The CLI calls the same trainer and reads the same JSON. Replace the hardware
UUID and paths. The Python example above can be launched with the same environment
settings by replacing `-m marl_battlegrounds train ...` with `mappo_demo.py`.

```bash
CUDA_VISIBLE_DEVICES=GPU-INTERNAL-UUID JAX_PLATFORMS=cuda,cpu \
  uv run --no-sync python -m marl_battlegrounds train \
  --config /absolute/path/mappo-demo.json \
  --output-dir /absolute/path/mappo-direct-run

CUDA_VISIBLE_DEVICES=GPU-INTERNAL-UUID JAX_PLATFORMS=cuda,cpu \
  uv run --no-sync python -m marl_battlegrounds train \
  --resume-from /absolute/path/mappo-direct-run/checkpoints/CHECKPOINT_ID

uv run --no-sync python -m marl_battlegrounds analyze-training \
  /absolute/path/mappo-direct-run \
  --output-dir /absolute/path/mappo-direct-report
```

Use the resume command only for an existing run. For the prepared package, use
its own interpreter when making a new analysis report:

```bash
/absolute/path/mappo-run-package/.venv/bin/python -I \
  -m marl_battlegrounds analyze-training \
  /absolute/path/mappo-run-package/run \
  --output-dir /absolute/path/mappo-run-package/report
```

There is no separate model-loading CLI. Use the public Python loader. This
terminal command reads the completed run's saved selected path and loads it
through the prepared package's own interpreter; it does not run games:

```bash
JAX_PLATFORMS=cpu /absolute/path/mappo-run-package/.venv/bin/python -I - <<'PY'
import json
from pathlib import Path

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training

run_dir = Path("/absolute/path/mappo-run-package/run")
if (run_dir / "checkpoint_recovery.json").exists():
    raise RuntimeError("Finish checkpoint recovery before loading a selected actor")
status = json.loads((run_dir / "status.json").read_text())
if status.get("status") != "complete" or not status.get("selected_actor"):
    raise RuntimeError("This run has no completed selected actor")
system = training.load_system(status["selected_actor"])
print(status["selected_actor"])
PY
```

Start with `run/run_summary.md`, `run/learning_curve.csv` and the learning-curve
figures. The summary links identities, validation, actor paths, exposure and
measured costs. Final and selected actors can differ. Results describe the
declared fixed panel; neither successful execution nor a single seed establishes
general competence, sample efficiency or learned team tactics.

### Measured MAPPO Engineering Checks

**Workload note, 22 September 2026.** These checks used training on the old maps
(maps 0 to 41 with revision 6 of Map 39). They remain valid measurements of the
code's correctness and cost. They are not learning results.

The final compiler-policy check used the internal RTX 5090, B32/T128, four PPO
epochs, native K20/H300 and priority metrics, with episode recording disabled.
The production update and independent reference matched all 354 output leaves
exactly for initial, continued and changed-seed inputs. Changing
ordinary weights, keys, maps and recurrent values reused the compiled programs.

| Measured Work | Time |
| --- | ---: |
| Content Preparation | 14.78 seconds |
| Learner Initialization | 11.77 seconds |
| Collection Compilation | 12.64 seconds, plus 0.82 seconds lowering |
| Update Compilation | 10.18 seconds, plus 0.68 seconds lowering |
| Warm Collection | 1.00491 seconds per block |
| Warm Update | 0.17172 seconds per block |
| Warm Collection Plus Update | 1.17672 seconds per block; 3,480.87 real transitions per second |

Warm figures are medians of five samples. Collection and update were synchronized
separately. In three reversed-order pairs on identical fixed inputs, the fixed
compiler policy took 1.17888 seconds per block versus 1.17619 with ordinary
compiler settings, about 0.23% more. This small measured difference supports the
simpler fixed policy. The two settings produced different floating-point values;
this comparison does not establish equal sampled trajectories.

Peak process RAM was 5.17 GB. GPU process polling observed 5.30 GB; sampling every
0.2 seconds can miss short peaks and began after the initial production/reference
compilations. JAX recorded 2.70 GB peak live allocations and a 4.43 GB pool. These
figures include reference and compiler-comparison states/programs, so they are not
the memory requirement of a lone trainer. The logical rollout held 229.84 MB and
the learner 80.37 MB; logical sizes may count shared buffers more than once.
The required compact result transfer was 695 bytes and took about 0.67 milliseconds.

Two reversed-order quiet/verbose pairs used real output files and 12 updates
each. Each verbose pass emitted a periodic progress line with a warm ETA at the
real ten-second cadence. Verbose elapsed time differed by +13.26 milliseconds and
-0.054 milliseconds over roughly 14.56-second passes. The differences were within
observed timing variation; no repeatable slowdown was measured. This is not a
zero-cost guarantee. Final states passed the declared numerical comparison.

The actual panel ETA helper separately took median batch means of 0.413–0.571
microseconds per call over 20,000-call batches. The slowest observed batch mean
projects to 1.71 milliseconds over 2,442 updates plus 512 phase records. That is
an illustrative allowance, not a limit on retries or worst-case delays. Its
0.38-second CPU import/setup cost was outside timing; the measured helper made
no JAX calls. This small host calculation does not measure full reporting or
validation cost.

Separate public-trainer checks restarted from update 8 and completed updates 9
and 10 in a fresh process, adding exactly 8,192 transitions each. All 332 state
leaves without recording, and 333 with recording, matched bit for bit. Actor
arrays, action/log-probability witnesses, scientific logs, exposure and exported
actor digests matched too. With recording, original registrations and every CSV
byte matched, including 32 newly completed episode rows. No global `XLA_FLAGS`
was set. The earlier default-compiler failure and an attempt rejected after a
source change remain preserved; neither is counted as successful continuation.

These are bounded engineering checks on the recorded hardware and software.
The cost benchmark excludes checkpoint save/restore, loaded validation and
plotting; it does not estimate a complete scientific run or prove useful learning.
Generated evidence is in `artifacts/m9-m10/packet-4/gpu-final-policy-costs/` and
`artifacts/m9-m10/packet-4/support/`, including
`final-fresh-process-resume-results.json` and `pending_eta_host_cost.json`.

### Measured Integration Results

**Superseded, 22 September 2026.** These runs trained on the old maps (maps 0 to
41 with revision 6 of Map 39). Their learning results are historical only; the
costs they report remain valid measurements of that code.

The four declared local development runs used the internal RTX 5090, B32/T128,
four PPO epochs and native K20/H300. They completed their exact budgets:

| Setting | Real Transitions | Recorded Active Time |
| --- | ---: | ---: |
| Plain | 1,048,576 | 364.26 seconds |
| Curriculum | 655,360 | 244.29 seconds |
| Reward Shaping | 131,072 | 93.72 seconds |
| Curriculum And Reward Shaping | 655,360 | 242.75 seconds |

These separate seeds and budgets check integration; they are not a controlled
treatment comparison. All 8,256 completed training games drew, and every logged
terminal task-reward mean was zero. Both curriculum runs reached all 17 stages.
The fixed Plain initialization, halfway and final actors then drew all 300
declared Random games. Each scored 0.5, so the fixed panel is **not qualified**.
No full scientific run was launched, and no mini budget was extended.

Warm training measured about 3,479–3,513 real transitions per second. Repeated
Random checks took about 13.5 seconds per 100 games after the first call and
reused the measured compiled evaluator cache. This does not measure two-learner
panel validation or the full experiment's cost. These measurements predate the
fixed training compiler policy above; they retain their original source and
execution settings. Later host-only progress and process-ownership fixes leave
that measured numerical path unchanged. Successful execution and these costs
do not establish useful learning.

Generated local evidence lives under `artifacts/m9-m10/packet-4/`:
`mini_qualification.json`, `curriculum-evidence/curriculum_evidence.json`, and
`early-panel-check/panel_attempt.json` with `validation_costs.jsonl`. These are
local run outputs, not files required to use the public API.

## Configuration Screen

`training.prepare_screen` prepares the declared twelve-case MAPPO screen without
starting it. It copies current reviewed public source, including uncommitted
changes, records file hashes and builds an isolated environment from `uv.lock`.
Private milestone files are excluded. The source directory stays reusable code;
all generated experiment files live under the chosen `artifacts/` directory.

From the repository, prepare a new package using the explicitly authorized GPU:

```bash
JAX_PLATFORMS=cpu uv run --no-sync python -m marl_battlegrounds.training.screen prepare \
  "$PWD/artifacts/m9-m10/packet-4/bt-screen-5min" \
  --repository "$PWD" \
  --gpu-uuid GPU-6b11a0c4-14e8-6782-8932-1df56d599796
```

This UUID identifies the development machine's internal card; use your own
explicitly verified card when preparing another machine. Preparation prints
absolute commands. It rejects an existing destination instead of replacing work.
For a package prepared at the path above:

```bash
bash artifacts/m9-m10/packet-4/bt-screen-5min/launch.sh
tail -F artifacts/m9-m10/packet-4/bt-screen-5min/logs/experiment.log
bash artifacts/m9-m10/packet-4/bt-screen-5min/status.sh
```

Launch detaches the worker. Closing `tail` with Ctrl+C stops log viewing only.
Use `stop.sh` to stop the experiment deliberately. Use `resume.sh` after an
interruption; it checks source, dependencies, GPU and saved identities first.
Resume preserves the first launch's deadline and every frozen step budget.
No active assistant session is needed. Status reads files and process identities;
it does not initialize JAX. The log prints UTC timestamps, steps completed, readable times, training speed
and estimated time left. Speed checks and comparison trials have separate counts.
The latest Random check shows kills and deaths per game, the kill difference,
its change since initialization, and wins/draws/losses. Wins are not expected or
required in this short screen. An all-draw check may still show combat learning.
Estimates say `Estimating` until measured costs support them. Raw losses and
signed training rewards stay in `training_updates.jsonl`; they are not learning
scores. Process details stay in `process.json`, and screen error tracebacks go
to `logs/errors.log`. Add `--json` to `status.sh` for machine-readable output.

The screen uses B=32/512/1024 and T=16/32/64/128, curriculum off, score-delta
shaping 0.01 and input scale 0.01. Its five-minute target covers collection and
updates. Calibration converts that target into fixed steps before scientific
training; other work costs extra. One GPU worker runs at a time. The two-hour
ceiling includes all launched work, with numerical work stopped at 117 minutes.
Read the [full method record](baseline_methods.md) for seeds, rounding, the shared
experience checkpoint, validation and the rule for stopping an infeasible run.

Reports live in the package's `reports/` directory. `baseline_trials.csv` retains
all cases, including failed and unstarted ones. `learning_curve.csv` and
`validation_cells.csv` link measurements to their original records. PNG/SVG plots
show combat improvement against Random versus actual elapsed minutes first,
with native task scores shown separately. Training time and step counts are
additional views. `run_summary.md` includes kills, deaths, completion state and
elapsed time. This one-seed screen can suggest promising settings; it cannot
establish the final best baseline. An all-draw result is not a failed run.
To regenerate reports without collecting experience:

```python
from marl_battlegrounds import training

report = training.analyze_screen(
    "artifacts/m9-m10/packet-4/bt-screen-5min"
)
print(report["artifacts"]["summary"])
```

The optional Random hook is also available through ordinary `TrainConfig`:
`random_diagnostic_seed_pairs=4` evaluates 40 games at initialization, each
`checkpoint_env_steps` capture, and final. It is separate from any frozen
opponent panel. Leaving it unset preserves ordinary training behavior. The
screen alone manages the shared initialization reference after verifying matching
inference weights and settings. Interrupted games resume through the existing
M8 evaluator before more training. Native scores always keep K20/H300 and exclude
shaping. These Random checks are diagnostics, not a claim of competence.

## Kill-Threshold Curriculum

The optional trainer can start with one kill needed to win, then raise the
threshold while keeping 5v5 teams and all 42 training maps throughout:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.training import TrainConfig, train

result = train(
    TrainConfig(
        seed=19_044_701,
        num_envs=512,
        total_env_steps=19_999_744,
        score_threshold_curriculum=True,
        shaping=True,
        shaping_mode="score_delta",
        shaping_coefficient=0.01,
        ppo=PPOConfig(rollout_length=32, input_scale=0.01),
        metrics="none",
        recording=False,
    ),
    output_dir="artifacts/my-threshold-run",
)
```

K1 receives the first 10% of the requested experience. K2 through K10 share the
next 30% equally. K12 and K15 receive 5% each. K20 receives the final 50%.
Rounding preserves the exact total. Thresholds change only at episode reset;
continuing games keep their old rules. Actors already see the current threshold.
`exposure.json` records actual starts and steps under each K, plus outcomes by
episode stage. A requested stage is not proof that games actually played it.

This option cannot be combined with `curriculum=True`, which selects the older
team-size/map schedule. False retains the ordinary K20 path. The trainer checks
and records each threshold/map source using the existing content authority.
Direct users of the content helper may pass `score_thresholds=(1, 2, 20)` to
`prepare_training_content`; default calls retain their original 42-source bank
and identity. A saved extended binding carries its declared threshold order.
Learner resume uses its original frozen source package. Historical frozen actors
remain usable through `load_system`.

Validation always has its own fixed rules. `validate_random` keeps maps 42–46,
canonical 5v5 and K20/H300 even when training uses K1. Its optional `root_seed`
argument selects the actual paired evaluation streams and is part of the saved
task identity. The default is 19,043,001. For example, a declared fresh check can
use `seed_pairs=20, root_seed=19_044_791` for 200 games. Merely raising
`seed_pairs` with the old root would reuse the earlier games' streams.

For the complete 48-run tuning recipe, read the
[baseline methods record](baseline_methods.md#twenty-million-step-mappo-tuning-study).
The recipe still applies; that study's results are superseded and historical only.
The study's artifact directory owns its launch/watch/status/resume scripts and
reports. It checks every 10% boundary, selects a model, confirms it on fresh
games and writes plots without an active assistant session. It has no automatic
time limit and starts no longer follow-up run.
