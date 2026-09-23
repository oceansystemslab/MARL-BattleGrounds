# Baseline Source Reuse

The four PPO baselines and recurrent QMIX start from Mava's reviewed learning
code. BG keeps its own environment, permitted actor inputs, action rules,
evaluation, recording and checkpoint ownership. This page records the source and the
independent numerical reference. A source match is not evidence of learning.

## PPO Model Choices

One shared PPO implementation serves four method settings:

| Method | Actor | Critic Input | Memory |
| --- | --- | --- | --- |
| `mappo` | 128-wide recurrent network | Physical training state | Separate actor and critic carries |
| `ippo` | Same recurrent network | Each actor's own permitted view | Separate actor and critic carries |
| `ff_mappo` | Two 128-wide ReLU layers | Physical training state | None |
| `ff_ippo` | Two 128-wide ReLU layers | Each actor's own permitted view | None |

Feedforward networks keep the donor's orthogonal gains: square root of two
for torso layers, 0.01 for the action head and 1.0 for the value head. They use
zero biases, no layer normalization and no recurrent cell. Their System memory
is the empty tuple, not an unused recurrent array. Recurrent and feedforward
models differ in both architecture and sample ordering; comparing them is not
a pure test of memory.

All four use the shared optimizer, masking and optional ValueNorm rules below.
The public defaults remain input scale 1.0, left spawn frame and ValueNorm on.
These BG adaptations are checked separately from raw donor comparisons, which
use scale 1.0, world frame and ValueNorm off. Search recipes do not silently
change constructor defaults or historical saved settings.

With the current 5,164 actor features, 919 physical-state features and 198
actions, the parameter counts are:

| Method | Actor Parameters | Critic Parameters |
| --- | ---: | ---: |
| `mappo` | 801,990 | 233,217 |
| `ippo` | 801,990 | 776,577 |
| `ff_mappo` | 703,174 | 134,401 |
| `ff_ippo` | 703,174 | 677,761 |

Each parameter is float32, so weights alone use four bytes per parameter.
Optimizer state, gradients, temporary arrays and opponent history add storage.
Each recurrent actor or critic carry adds 640 float32 values per game;
feedforward methods allocate neither carry. These exact layout counts are not
measurements of peak process or GPU memory.

An IPPO critic uses the same permitted view, input scale and spawn frame as its
actor. It has separate weights and, for recurrent IPPO, separate memory. It
cannot use another actor's private row, the actor's carry or physical training
state. Collection stores compact observations and action-time outputs. Preparing
the learner batch computes fixed old values and cutoff values. The local critic
expands one time row at a time, and selected update minibatches share the
already encoded local features with the actor. This avoids retaining a full
expanded local-feature rollout.

Returns and advantages are calculated in time order for every method. Recurrent
updates then shuffle whole game sequences within each data group. Feedforward
updates shuffle complete time/game rows within each group; the five actors stay
together. Only the chosen minibatch is expanded into network inputs. Feedforward
MAPPO computes one physical critic prediction per game row when preparing
rollout values and during learning, then shares it across the actor slots. Each
actor retains its own loss mask, old prediction and target. Neither feedforward
method runs a GRU or resets a hidden carry. Feedforward IPPO collects no
physical-state feature payload.

The recurrent batch must be divisible by groups times minibatches. For
feedforward methods, the batch must be divisible by groups, and rollout length
times batch must be divisible by groups times minibatches. Training also needs
an even batch for paired spawn ends. These rules change sample ordering; they do not change
the experience budget, loss masks or optimizer settings.

## Source Identity And License

The source is [Mava commit
`9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93`](https://github.com/instadeepai/Mava/tree/9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93).
Its Git tree is `f47f19aa124c460ce9dc23e63288ed765d5cdd47`. The source was
checked out at that exact commit and verified clean before extraction.

Mava uses Apache License 2.0. The fixture retains the complete license and the
original notices. The [source manifest](../../tests/fixtures/baseline_donor/source_manifest.json)
records each retained file's SHA256 and Git blob identity. Files ending in
`.txt` under that fixture are unchanged source evidence, not BG modules.
Do not reformat them or replace their expected results to hide a failed port.

The [PPO variants manifest](../../tests/fixtures/baseline_donor/variants_source_manifest.json)
adds the pinned recurrent IPPO and feedforward MAPPO/IPPO update files, their
system/default settings and the feedforward network settings. Shared source
files keep their original bytes. The [variant reference record](../../tests/fixtures/baseline_donor/variants_reference.json)
binds the additional fixed arrays to those exact sources and the isolated donor
runtime. The reference helper retains MAPPO as its default; explicit variant
generation writes separate outputs and does not replace the old MAPPO record.

| Original File | Reused Calculation Or Checked Contract |
| --- | --- |
| `mava/networks/torsos.py` | `MLPTorso` and its activation lookup. |
| `mava/networks/base.py` | `ScannedRNN`, `RecurrentActor` and `RecurrentValueNet`. |
| `mava/networks/heads.py` | `DiscreteActionHead`, including its initializer and masked logits. |
| `mava/networks/distributions.py` | `IdentityTransformation`; the reference uses the real TensorFlow Probability categorical distribution. |
| `mava/utils/multistep.py` | `calculate_gae`, including the successor-ending convention. |
| `mava/systems/ppo/anakin/rec_mappo.py` | PPO losses, group averaging, minibatch/epoch order and optimizer construction. |
| `mava/types.py`, `mava/systems/ppo/types.py` | Original observation, transition, parameter, optimizer and memory containers. |
| `mava/utils/training.py`, `mava/utils/config.py`, `mava/utils/network_utils.py` | Learning-rate, experience-count and action-head selection rules. |
| `mava/configs/default/rec_mappo.yaml` | Selects the Anakin execution settings, recurrent MAPPO settings and recurrent network. |
| `mava/configs/arch/anakin.yaml` | Sixteen environments in each of two data groups; sampled PPO evaluation. |
| `mava/configs/network/rnn.yaml`, `mava/configs/system/ppo/rec_mappo.yaml` | The network and learning settings below. |

The later PQN-VDN port is reserved for JaxMARL commit
`976aeb152cb184a5095021968bba94da96eb6394`. This reference does not verify that
later port. Mava's environment wrappers, experiment logger, evaluation loop,
checkpoint loader and distributed experiment harness are not copied into BG.

## Resolved Recurrent MAPPO Settings

Actor and critic each have a 128-wide Dense layer with ReLU, a 128-wide GRU,
and another 128-wide Dense layer with ReLU. They have separate parameters,
optimizer state and recurrent memory. There is no layer normalization in this
configuration. Torso kernel gain is square root of two. The actor head gain is
0.01 and the scalar value-head gain is 1. Dense biases start at zero. The
reference uses Flax 0.10.3's unchanged GRU defaults. Cross-version random
initialization is not claimed to produce identical values.

| Setting | Value |
| --- | --- |
| Actor And Critic Learning Rates | 0.00025 each; no decay. |
| Rollout Length | 128 transitions per environment. |
| PPO Epochs And Minibatches | Four epochs, two minibatches per epoch. |
| Discount And GAE Lambda | 0.99 and 0.95. |
| Policy And Value Clip | 0.2. |
| Entropy And Value Coefficients | 0.01 and 0.5. |
| Optimizers | Separate global-norm clip at 0.5, then Adam with epsilon 0.00001. |
| Recurrent Chunks | Full rollout; the unspecified donor setting resolves to rollout length. |
| Data Groups | Two groups; gradients are averaged before clipping and Adam. |

The two groups are parts of one learner. They are not independent runs. At
the initial BG batch of 32 environments, each group has 16 environments and
each minibatch has eight complete environment sequences. Advantage statistics
are computed separately within each group's minibatch. Time and agent axes
are not shuffled into independent recurrent samples.

The donor value loss has a factor of 0.5 before the maximum of clipped and
unclipped squared errors, followed by the configured value coefficient of 0.5.
Both factors belong to the checked calculation. Actor and critic gradient
norms are clipped separately.

### Optional Input Scale

`PPOConfig.input_scale` defaults to `1.0`, preserving the donor calculations and
historical models. A positive finite value multiplies actor and critic features
before their first Dense layer. For example, `0.01` makes an input value of 200
enter that layer as 2. The raw encoder, information limits and network size stay
the same. This input scale has no learned or running statistics. The critic's
separate target normalization is described below.

This is an explicit numerical adaptation. It can reduce saturation, where a
memory gate sits near its limit and responds weakly to input changes. Better
conditioning alone does not establish better learning. Keep the scale fixed
for a run and compare against the default with declared seeds and budgets.
Collection, value targets, PPO updates, historical opponents and exported actors
must all use the saved scale. Old exports without this field use `1.0`; loading
them never applies a new scale. Equal weights with different inference scales
have different policy identities.

### Optional Spawn Frame

`PPOConfig.spawn_frame` defaults to `"left"` since 22 September 2026. `"left"`
reflects the actor's permitted view about the map's vertical centerline
whenever its own team starts on the right bank, so its spawn end appears on
the left, and reflects the chosen move back before the game receives it.
The built-in maps place that bank at x = 0.5. Custom valid maps need not.
`"world"` preserves the raw coordinate convention. Reproducing a historical
run also needs its original settings and frozen source. The value `"right"`
existed only to rescue world-trained models that had
learned the right bank; it was removed with those models' supersession, and
records that used it stay reproducible only through their frozen packages. For example, an actor at x = 19.5 that sees an enemy at
x = 3 is shown itself at x = 0.5 and the enemy at x = 17; when it says East the
game receives West. The reflection changes the x of unit rows, spawn pads and
obstacle rows that have no mirror partner in their own table (with wall angles
negated; a row whose mirror image is already present stays as authored, so the
built-in maps encode identically from both ends), and swaps East with West,
Northeast
with Northwest and Southeast with Southwest in the previous-move one-hots and
the move mask. Targets, Ultimates, memory and every other field are frame-free.
The raw encoder, information limits and network size stay the same.

This is a representation choice, not a donor setting. Google Research Football
flips the live observation and translates the action back for right-side
players (`observation_rotation.flip_observation` and `flip_single_action` in
its environment); its documentation says "even if you control players on the
right team, observations are mirrored". RLGym inverts physics for the orange
team "to avoid re-learning the same strategy on both sides of the pitch".
AlphaZero orients the board to the perspective of the current player, and
warns that chess and shogi permit no reflection augmentation because their
rules are asymmetric; here the left-right symmetry of all 52 maps was checked
by the mirror check's test over every map's obstacle table and spawn pads,
recorded in the baseline methods record (mirror-check section), the up-down
one was not, so only the left-right flip is used. The design claim, verbatim:
"An optional team-relative coordinate adapter presents permitted observations
in a consistent orientation and maps chosen actions back to native
coordinates. It leaves game rules and information access unchanged."

A policy trained in the left frame receives a consistent horizontal orientation.
Physics, other inputs and recurrent history can still differ between games;
the adapter does not prove equal play or outcomes at both ends. Keep the frame
fixed for a run and compare against `"world"` with declared seeds and budgets.
Collection, PPO updates, historical opponents and exported actors must all use
the saved frame: the actor samples in its frame and stores world-frame indices
and probabilities computed in its frame, and the update reflects the rebuilt
view, mask and stored index the same way. Old exports without this field use
`"world"`; loading never applies a new frame and never infers one from a
model's results; adapting an old model means re-exporting it with an explicit
frame, which is a new identity. Equal weights with different spawn frames have
different policy identities. A run in the left frame cannot reuse a
world-frame run's shared random initialization result, because that identity
includes the frame.

### Critic Target Normalization

New runs use `PPOConfig.value_normalization=True`. This adds the official
MAPPO ValueNorm calculation to the Mava learner. The reference is
[on-policy commit `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac).
The unchanged source and MIT license are retained in
[`tests/fixtures/value_norm_donor`](../../tests/fixtures/value_norm_donor/source_manifest.json).
The manifest records each file's SHA256. The existing Mava reference remains
unchanged and is exercised with target normalization disabled.

The learner carries three float32 scalars: the running target mean, running
mean-square and accumulated averaging weight. Each nonempty optimizer
minibatch updates these with decay 0.99999. Dividing by the accumulated weight
corrects their initial zero values; that denominator has floor 0.00001.
The corrected variance has floor 0.01. Constant targets therefore remain safe.
The network learns normalized targets. Rewards, GAE and reported value error
remain in reward units. Statistics are fixed data outside differentiation.

The JAX adaptation pools eligible critic rows across the two gradient groups.
This uses one shared normalizer because there is one shared critic. Padding
and inactive slots are excluded; dead active agents remain critic samples.
The official update runs once per optimizer minibatch, including repeated
PPO epochs. Empty minibatches preserve statistics. A failed complete learner
update rolls back statistics with weights, optimizer state and memory.
These masking and atomic-update rules are BG adaptations, not new settings
in the hyperparameter search.

One critic pass provides both raw values for GAE and the exact normalized
network outputs saved for value clipping. Old clipping anchors are never
reconstructed using newer statistics. The only extra retained value array
uses 320 KiB at B512/T32, or 640 KiB at T64; the scalar state uses 12 bytes.
This is a storage calculation, not a measured peak-memory claim. The update
adds small reductions and arithmetic without another critic forward pass.
Runtime and learning effects need their own evidence.

Saved settings without `value_normalization` mean `False`. Disabled state is
`None` and adds no serialized numerical leaves. Enabled checkpoints save all
three scalars. Actor exports contain no critic or normalization state, so the
actor's inference contract and identity are unchanged by critic statistics.
Source and runtime compatibility checks remain required; support for the old
array layout does not bypass those checks.

### Optional Dense Training Reward

The default `shaping_mode="potential"` keeps the original score-potential
adjustment and terminal cancellation. The explicit `"score_delta"` alternative
adds coefficient times new team kills minus new team deaths to native reward.
It retains terminal kills and has no terminal cancellation. Both use actual
scores from the producing game and keep padding at zero.

This changes the training objective; it is not a donor calculation or a claim
of policy-invariant shaping. It addresses the absence of lasting feedback in
drawn games, including some small-team rosters that cannot reach K20 within
H300. Evaluation keeps native task reward and win rules. Saved configuration
binds the chosen mode; old configurations continue to mean potential shaping.
Learning benefit requires a controlled comparison on development seeds.

## Deliberate BG Adaptations

BG supplies separate, versioned actor and training-only physical-state views.
The actor never receives critic state or critic memory. BG's encoder already
contains permitted class and self-row identity; the donor's `add_agent_id`
wrapper is not copied. The action head has 198 outputs for BG's exact native
action combinations. BG uses a small JAX categorical helper instead of adding
TensorFlow Probability to the runtime dependency set.

Recurrent memory resets at a real episode start. Death, respawn and collection
cutoffs do not reset it. Invalid padding keeps the prior memory. A real ending,
including a horizon draw, blocks value bootstrap; a collection cutoff does not.
The donor stores the reset before an observation and uses the carried next
reset during its reverse GAE scan. Tests must check the mapping to BG's
produced-transition ending flags.

BG removes inactive actors from learning statistics and removes forced-neutral
dead-actor choices from policy statistics while retaining their value learning.
The user selected equal averaging over **nonempty groups**, separately for
actor and critic. An empty group contributes neither a gradient nor a denominator
count. When every corresponding group is empty, skip that optimizer entirely,
including its counters and stored momentum. These are explicit adaptations;
the unchanged donor comparison uses all-valid samples.

## Independent CPU Reference

[The reference support module](../../tests/baseline_donor_reference.py) extracts
the named original syntax-tree bodies without rewriting their calculations.
It supplies imports, fixed configuration and ordinary-array packaging instead
of loading Mava's environment framework. It executes the original networks,
real categorical distribution, GAE, loss functions, two-group averaging and
four-epoch update. The original optimizer construction is extracted too.

The reference environment uses Python 3.12 and the donor's own locked versions:
JAX/JAXlib 0.5.3, Flax 0.10.3, Optax 0.2.4, TensorFlow Probability 0.25.0,
NumPy 1.26.4 and SciPy 1.12.0. Every installed dependency is fixed in
[reference-requirements.txt](../../tests/fixtures/baseline_donor/reference-requirements.txt).
This environment is separate from BG's Python 3.14/JAX 0.10.1 environment.
The repository does not acquire a TensorFlow Probability dependency.

The synthetic case uses four time steps, two groups of four environments,
five actors, 13 actor features and 17 critic features. It retains the real
128-wide networks and 198 actions. It includes nonzero memory, a reset inside
the sequence, both terminal and continuing final states, masked action categories and
old-policy offsets that exercise PPO clipping. It is a CPU numerical test,
not a simulation or learning run.

The archive stores fixed parameters and inputs, raw and masked logits, values,
memory, action log probabilities, advantages, targets, group losses and
gradients, one optimizer update, and the full original four-epoch result.
Stored parameters isolate the port from differences in random initialization
between library versions. The fixture metadata binds the archive, source
manifest, shapes, settings and reference package versions. Ordinary tests read
these NumPy arrays without loading any historical training library.

For larger current-stack checks, `build_same_stack_reference()` returns the
same original networks, GAE and loss bodies on BG's installed libraries. A
small test-only categorical wrapper supplies log probability and entropy. The
historical fixture checks this wrapper against the actual TensorFlow Probability
distribution before it is used as a reference for BG's larger input shapes.
It does not call the production categorical helper. Keep that bridge check;
two consumers sharing an unchecked replacement would not be independent proof.

To rebuild in a fresh temporary environment, run these commands from the
repository root. Use `--no-config` so BG's dependency constraints do not change
the independent historical environment. The output stays separate until its
changes have been reviewed.

```bash
UV_CACHE_DIR=/tmp/mava-reference-cache uv --no-config venv --python /usr/bin/python3.12 /tmp/mava-reference
UV_CACHE_DIR=/tmp/mava-reference-cache uv --no-config pip install --python /tmp/mava-reference/bin/python -r tests/fixtures/baseline_donor/reference-requirements.txt
JAX_PLATFORMS=cpu /tmp/mava-reference/bin/python tests/baseline_donor_reference.py --generate /tmp/mava-reference-output
```

The generator checks source hashes and exact package versions before running.
Compare each generated array with the checked archive; matching filenames are
not proof. Never regenerate expectations as a fix for a production mismatch.

## Evidence And Limits

The original source calculations execute successfully in the isolated CPU
environment. Repeated fresh processes produced the same 325 arrays and the
same 10,096,759-byte archive. Its SHA256 is
`94aa9f1424c124fa753935ec2c88c678bd263af189f7aca815dc6a2c6491322b`.
The current-stack bridge checked 278 forward, GAE, loss, gradient, parameter
and optimizer arrays, including all four epochs. The largest absolute
difference was 0.0000019074 in entropy. The comparison used absolute tolerance
0.000002 and relative tolerance 0.0002. These are measured agreement bounds for
the declared synthetic case, not universal numerical-error guarantees.

The production comparison in `tests/test_baseline_ppo_reference.py` also passes
on CPU. Current-stack initialization matches the donor exactly, including
parameter names, shapes and dtypes. Against the historical fixture, GAE matches
exactly; maximum absolute differences are 0.0000019074 for forward outputs,
0.0000004769 for losses and gradients, 0.0000000503 for the first parameter and
optimizer update, and 0.0000004769 after four epochs of grouped updates. These
production checks use absolute and relative tolerances of 0.000002 each and
require exact integer optimizer counters. The tighter relative bound is based
on the measured differences; the earlier bridge result keeps its original bound.

BG masking, information, episode-memory and System integration checks are
separate from these all-valid comparisons. A fixed-input donor match does not
establish training throughput, complete save/resume, sample efficiency, learned
tactics or competence. The smaller synthetic tensor shapes do not measure the
memory cost of BG's complete input encoder.

## Complete MAPPO Integration

The complete trainer reuses these verified networks and update equations.
Its learner adapter reconstructs compact permitted actor inputs only where
needed by the existing PPO update. It reads old values with the fixed rollout
critic, keeps separate critic memory and discards bootstrap-only memory advance.
Action-time categorical choices and log probabilities come from collection;
the adapter makes no second behavior-policy call or duplicate GAE calculation.

Model initialization folds tag `0x4D415050` into the run's Threefry seed. The
separate learner shuffle root uses `0x50504F55` and folds in the next accepted
update index. Empty updates consume no shuffle step. This is learner-key schema
version 1; collection's existing independent streams retain their own version.

Synchronous Orbax checkpoints replace donor checkpoint orchestration. The pinned
handler cannot store zero-element arrays; their paths/shapes/dtypes remain in
the checked schema and known code reconstructs those empty leaves on restore.
They contain no numerical bytes. Typed keys use Orbax's native handling.
This adaptation changes neither an optimizer equation nor a learned parameter.

Validation, recording, deployment loading and analysis reuse MARL-BGs owners.
A named pinned opponent reuses them too: the M8 evaluator's freezing,
preparation, registration and one-team execution helpers, including its host
helper for host methods, which run outside compiled code in a host loop of the
collector. No donor code is involved.
The early two-member panel is provisional. Complete-workflow tests and measured
costs supplement the original numerical comparisons; actual learning quality
still needs the declared development and demonstration evidence.

## Recurrent QMIX

[`baselines.qmix`](../../src/marl_battlegrounds/baselines/qmix.py) adapts
Mava's recurrent QMIX (`rec_qmix`) from the same pinned commit. The
[QMIX source manifest](../../tests/fixtures/baseline_donor/qmix_source_manifest.json)
is bound to the MAPPO manifest by hash. It adds these original files and
reuses the shared ones above byte for byte:

| Original File | Reused Calculation Or Checked Contract |
| --- | --- |
| `mava/systems/q_learning/anakin/rec_qmix.py` | Epsilon-greedy action selection, sample preparation, the Double-Q loss, `update_q` and the optimizer construction. |
| `mava/systems/q_learning/types.py` | The transition, action-selection and four-network parameter containers. |
| `mava/networks/base.py` | `ScannedRNN`, `RecQNetwork` and `QMixingNetwork`. |
| `mava/networks/distributions.py` | `MaskedEpsGreedyDistribution`; the historical reference uses the real TensorFlow Probability categorical distribution. |
| `mava/utils/jax_utils.py` | `add_batch_dim` and `switch_leading_axes`. |
| `mava/configs/default/rec_qmix.yaml`, `mava/configs/network/qmix_rnn.yaml`, `mava/configs/system/q_learning/rec_qmix.yaml` | The network and learning settings below. |

### QMIX Model And Settings

One local Q-network is shared by every Team A actor; each actor keeps its own
256-wide memory. The layers are the donor's: Dense 256 with ReLU, a 256-wide
GRU that resets before a new episode, Dense 256 with ReLU and a Dense 198 head
with orthogonal gain 0.01. Parameter paths match the donor, so the same key
gives the same starting weights on the current stack. The mixer reads the
layer-normalized 919-value physical state. Its hypernetworks are
919→64→160 and 919→64→32 for the two weight layers (absolute value keeps the
mix monotonic) and 919→32 and 919→32→ReLU→1 for the biases; the hidden layer
uses ELU. With the current input sizes, the Q-network has 1,833,158 parameters
and the mixer 190,991: 8,096,596 bytes of float32 weights for one copy of each.
Target copies and Adam's two moments add three more copies.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `rollout_length` | 8 | Rounds collected per block |
| `buffer_size` | 1000 | Replay rows kept per game |
| `min_buffer_size` | 32 | Rows per game before any learning |
| `sample_sequence_length` | 20 | Rows per sampled sequence, giving 19 TD pairs |
| `sample_batch_size` | 128 | Sequences per optimizer step |
| `epochs` | 4 | Fresh samples and optimizer steps per ready block |
| `q_lr` | 0.00003 | Adam learning rate for the Q-network and mixer together |
| `hard_update`, `update_period` | True, 200 | Copy online to target when the step count before a step is a multiple of 200 |
| `tau` | 0.01 | Blend weight when `hard_update` is False |
| `gamma` | 0.99 | Discount per transition |
| `eps_min`, `eps_decay` | 0.05, 100000 | Exploration falls from 1 to 0.05 over 100,000 real transitions |

The donor YAML also names `max_grad_norm: 10`, but its optimizer is
`optax.chain(optax.adam(q_lr))` and never clips, so BG copies no clipping. The
donor's `add_agent_id` is not copied either: the shared permitted encoder
already gives each actor its own class and self features, and any later change
to that encoder applies to QMIX as well. No burn-in, extra clipping, Q or
return normalization, or tuning is added.

### Deliberate QMIX Adaptations

- **Inputs, frame and masks.** Actors read BG's permitted SharedObs encoding in
  the chosen spawn frame (default "left"), with BG's 198-way legality masks.
  Chosen actions are mapped back to world directions.
- **Legal greedy choice.** The donor masks illegal actions with
  `finfo(float32).min`. If every legal score were exactly that value, its
  argmax would pick an illegal action. BG masks with negative infinity through
  the shared action helper, so the greedy choice is always legal; ties go to
  the lowest legal index in the network's frame (mirrored for left-frame
  games). For ordinary finite scores both rules choose the same
  action. Double-Q next-action selection uses the same rule.
- **Epsilon clock.** Exploration follows the donor's formula on a clock of real
  transitions. The clock stops counting at the decay end, so it never
  overflows 32-bit integers, and the rate is then exactly `eps_min`. Before the
  decay end it matches the donor up to float32 rounding.
- **Two-stage exploration draw.** Each actor first draws whether to explore and
  then, if so, draws a uniform legal action. This is the donor's
  `eps·uniform + (1−eps)·greedy` distribution exactly; the random stream
  differs, so sampled actions are not compared with the donor.
- **Team reward.** The donor learns from the mean reward over all agents. BG
  stores one team task reward per decision: the mean task reward over the
  Team A slots that are in the game, and adds the shaping value, stored
  separately, when the sample is built. The two rules agree for full 5v5
  teams and differ only in smaller curriculum rosters.
- **Inactive slots and padding.** Smaller rosters pad Team A to five slots.
  Inactive slots add nothing to the team value or its gradient. Invalid padding
  rows are replaced by neutral values before any calculation, keep recurrent
  memory unchanged and never form a TD pair; the loss averages over eligible
  pairs only. A sample with no eligible pair moves neither the optimizer nor
  the targets.
- **Endings.** BG marks the row whose transition ended the episode; the donor
  marks the next row as an episode start. The TD target stops bootstrapping at
  a real ending, including a horizon draw, but not at a collection cutoff.
- **Compact Team A replay.** The donor stores whole observations twice (the
  observation and the next observation). BG's replay row keeps only Team A's
  five observer rows, their 5x5 source permissions, Team A masks and chosen
  actions, the team task reward and the shaping reward separately, lifecycle
  flags, the 919-value physical state once, and eight identity fields: 29,664
  bytes per game row, or 949,248,000 bytes for 32 games and 1,000 rows each.
  The next row of a sequence supplies the successor, so the newest stored row
  of a game is only ever used as a successor. A sample is rebuilt into network
  inputs with the same permitted-input builder the live actor uses; the
  rebuilt features are bitwise equal to the live ones.
- **Replay storage and readiness.** Storage starts as explicit zeros with a
  strong 32-bit write index. Flashbax's own initializer uses a weak index that
  becomes strong after a checkpoint restore, so its type would differ before
  and after a restore; the explicit index keeps it the same.
  Only the real prefix of a block is added, one round at a time; padding is
  never stored. BG checks that `min_buffer_size` is at least
  `sample_sequence_length` instead of letting Flashbax raise it silently, and
  checks readiness itself because an unready Flashbax sample returns zeros.
  The donor has no readiness check.
- **Exploration hook.** The donor keeps epsilon in its own acting loop. BG's
  shared collection calls an optional hook before every decision; QMIX's hook
  sets epsilon on the shared current actor, so current self-play teams share
  it, while historical opponents keep the epsilon they had when captured.
  Without a hook the collection program is unchanged.
- **One learner.** BG runs one gradient step per sample instead of averaging
  over the donor's replicated device and update-batch axes.
- **Warmup and keys.** The donor samples from its first block. BG accepts
  blocks before readiness as warmup blocks with no sample, key use or optimizer
  step, then learns on every later block. Each optimizer step draws its sample
  with a key folded from a fixed sampling root and the step count before it,
  instead of the donor's split key chain, so a resumed run draws the same
  samples. Each ready block publishes the new actor to self-play history once.
- **Separate orchestration.** Collection, curriculum, shaping, history,
  pinned opponents, recording, checkpoints, validation and reports are BG's
  existing owners, not the donor's loop.
- **One online pass.** BG unrolls the online Q-network once over all sample
  rows and uses it both for the chosen-action values and, without gradient, for
  the next greedy actions. The donor makes two online passes; the recurrent
  network is causal, so the values and gradients are the same.

### QMIX CPU Reference

`build_same_stack_qmix_reference()` and `load_qmix_reference()` in the
[reference support module](../../tests/baseline_donor_reference.py) form a
separate route; the PPO functions and fixtures are unchanged. The historical
generator runs the original Q-network, mixer, epsilon-greedy distribution,
action selection, sample preparation, loss and `update_q` bodies in the same
isolated Python 3.12 environment as the PPO reference. Its synthetic case uses
four sequences of six rows, five agents, 13 actor features, 17 state features,
the real 256-wide networks and 198 actions. It includes nonzero starting
memory, episode starts on an interior, a first and a last row, and a
neutral-only actor row. A distinct target network makes the Double-Q choice
visible. Gradients are captured by running `update_q` with a zero-step
optimizer whose new state is the gradient.

```bash
JAX_PLATFORMS=cpu /tmp/mava-reference/bin/python tests/baseline_donor_reference.py --qmix --generate /tmp/mava-qmix-reference-output
```

Two fresh processes produced the same 294 arrays and the same
13,668,558-byte archive, SHA256
`c744458bf3553f1f7ad4f2d7a67a68406f4eb2d3ba3409619fe9bf38dc8f006f`.
`tests/test_baseline_qmix_reference.py` compares BG with both the archive and
the current-stack donor, using absolute and relative tolerances of 0.000002.
Initialization matches the current-stack donor exactly for both Q-networks,
both mixers and the Adam state. Measured largest absolute differences were
0.00000018 for forward values, memory and the epsilon clock; 0.0000029 for
mixer values; 0.0000057 for gradients (largest gradient magnitude about 9.8);
0.0000019 for one and two Adam steps; and 0.00000003 for the soft target rule.
Probabilities and greedy choices matched exactly. Hard target copies happen at
step counts 0 and 200 and not at 1 or 199, as in the donor. At exactly
100,000 transitions the donor's float32 formula gives 0.05000001; BG gives
exactly 0.05.

These all-valid comparisons prove agreement for the declared synthetic case
only. They do not establish replay behavior, training throughput, save/resume,
sample efficiency or learned tactics; those have their own checks.

## Training Compiler Policy

Built-in PPO passes `xla_gpu_autotune_level=0` to its outer collection and update
compilations, including collection with recording. The shared owner is
[`training._compilation`](../../src/marl_battlegrounds/training/_compilation.py).
It changes no process-wide environment or JAX setting. The raw scan and learner
functions remain usable inside callers' own compiled loops. Generic collection
keeps JAX defaults unless its caller supplies compiler options. Evaluation keeps
its existing compiler settings.

GPU autotuning times several kernel choices while compiling. Separate timing
trials can select different choices, whose floating-point calculations round
differently. Turning autotuning off fixes these compilation choices; it does
not remove every possible source of runtime variation. This distinction follows
[OpenXLA's determinism guide](https://openxla.org/xla/determinism).
JAX requires these options on the outermost compiled call, so inner numerical
helpers do not set them. See [JAX compiler controls](https://docs.jax.dev/en/latest/201/controlling-xla.html).

The fixed choice removes a restart dependency on live timing trials without a
saved compiler-cache lifecycle. A persisted autotuning cache is an alternative,
but it needs version, hardware and completeness checks. Missing entries can
otherwise trigger new timing trials. See [OpenXLA's cache rules](https://openxla.org/xla/persisted_autotuning).
The policy changes GPU execution choices, not the donor's equations, learning
settings, actor information or random-key rules. Different rounding can still
change a later sampled trajectory, so results from an earlier policy keep their
original identity.

New learner checkpoints bind this policy and the active backend, device kind,
runtime version, JAX precision, random-number and JIT settings, the profile-guided
compilation switch, and declared `XLA_FLAGS`. Resume requires
the saved identity to match before recording or log recovery. The environment
string records what was declared; changing it after JAX starts does not prove
the compiler changed. Historical descriptions, frozen actor loading and reports
remain readable; a missing old execution identity is not treated as the new one.

**Workload note, 22 September 2026.** The checks in the next two paragraphs used
training on the old maps (maps 0 to 41 with revision 6 of Map 39). They remain
valid measurements of the code's correctness and cost. They are not learning
results.

The first fresh-process GPU check restored all 332 array leaves exactly but
later chose different actions under ordinary compiler settings. Setting the
global autotune flag to zero produced exact continuation in two fresh child
processes on the internal RTX 5090. The final per-compilation policy then passed
the actual public trainer's fresh-process checks with global `XLA_FLAGS` absent.
From update 8 through updates 9 and 10, all 332 state leaves without recording
and 333 with recording matched bit for bit, as did actor arrays, action-time
witnesses, scientific logs, exposure and export digests. The recorded case also
matched original registrations and every CSV byte, with 32 new completed rows.
The original failure and the separate attempt rejected for changing source bytes
remain preserved.

The final B32/T128, four-epoch numerical comparison matched all 354 output leaves
exactly in initial, continued and changed-seed cases. A fixed-input compiler
comparison measured 1.17888 seconds per block for this policy versus 1.17619 for
ordinary settings, about 0.23% more. It preserved their floating-point differences
and makes no trajectory-equivalence claim. Full costs and memory limits are in
[the measured engineering checks](README.md#measured-mappo-engineering-checks).
The evidence concerns the declared workload and software stack, not equality
across devices, library upgrades, arbitrary compiler flags or all JAX programs.
