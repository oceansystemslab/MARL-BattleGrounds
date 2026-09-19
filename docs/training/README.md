# Baseline Components

The `marl_battlegrounds.baselines` package provides input encoders, exact action
helpers and recurrent MAPPO calculations. It does not yet provide a training
loop or a trained policy. Ordinary environment use needs no training packages.

Install the existing `training` extra to use PPO. In a prepared contributor
checkout, run the example with:

```bash
uv sync --locked --extra dev --extra training
JAX_PLATFORMS=cpu uv run --no-sync python examples/baseline_system.py
```

For the qualified local GPU setup, follow the [GPU guide](../dev/gpu_sanity.md).
The [example](../../examples/baseline_system.py) initializes **untrained** actor
weights, wraps them in an existing M8 `System`, chooses actions and steps the
environment. Its results say nothing about learned skill. Methods that do not
use these baseline components can continue using ordinary `reset` and `step`
or their own Systems.

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

These checks establish a numerical and integration foundation. They do not
prove sample efficiency, learned team behavior, complete training throughput
or a competent policy within one GPU day. Those need later learning trials.
