"""Join compact training collection to the existing recurrent MAPPO update.

init_learner owns host setup. build_ppo_batch applies the separate collection-time
critic, and update_learner accepts a completed collection block or returns a
sticky failure. Current actor variables have one persistent owner in collection
history. Critic variables, optimizers and memory never enter an actor System.
The numerical helpers support jit and write no files. Checkpoint validation is
host-only; this module starts no training loop, validation run or recording.
"""

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    categorical_action_mask,
    encode_actions,
)
from marl_battlegrounds.baselines.inputs import TRAINING_STATE_FEATURE_SIZE
from marl_battlegrounds.baselines.ppo import (
    DEFAULT_PPO_CONFIG,
    HIDDEN_SIZE,
    PPOBatch,
    PPOConfig,
    PPOLearningOutputs,
    PPOMetrics,
    PPOTrainState,
    critic_values,
    initialize_ppo,
    make_recurrent_mappo_system,
    update_recurrent_ppo,
)
from marl_battlegrounds.core.types import (
    TASK_MODE_OUTCOME_DRAW,
    TASK_MODE_OUTCOME_TEAM_A_WIN,
    TASK_MODE_OUTCOME_TEAM_B_WIN,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.training._content import PreparedTrainingContent

# These package-private guards keep collection's lifecycle and restore rules
# with their existing owner.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    _failed,
    _validate_training_continuation,
    init_training_collection,
)
from marl_battlegrounds.training.curriculum import TrainingSchedule
from marl_battlegrounds.training.opponents import SnapshotEvent, refresh_opponents

type Tree = Any

LEARNER_KEY_SCHEMA_VERSION = 1
MODEL_INITIALIZATION_TAG = 0x4D415050
SHUFFLE_ROOT_TAG = 0x50504F55

LEARNER_ERROR_NONE = 0
LEARNER_ERROR_COLLECTION = 1
LEARNER_ERROR_BOUNDARY = 2
LEARNER_ERROR_NONFINITE_BATCH = 3
LEARNER_ERROR_ACTION = 4
LEARNER_ERROR_NONFINITE_UPDATE = 5
LEARNER_ERROR_HISTORY = 6
_MAX_COUNT = int(np.iinfo(np.int32).max)


class LearnerState(NamedTuple):
    """Keep one complete numerical MAPPO boundary with one current actor tree.

    Attributes
    ----------
    carry : TrainingCarry
        Complete collection state. carry.history.current_variables is the only
        current actor tree. Its environment is the true successor before reset.
    critic_params, actor_opt_state, critic_opt_state : PyTree
        Separate critic Flax variables and actor/critic Optax states. Their
        shapes and dtypes match initialize_ppo for the configured update.
    critic_memory : Array
        Float32 (B,5,128) carry before the next real decision. A bootstrap-only
        critic call never advances this stored memory.
    shuffle_root : Array
        Scalar typed Threefry key derived from the collection root with
        SHUFFLE_ROOT_TAG. Each real update folds in its next update index.
    completed_updates : Array
        Nonnegative int32 scalar, equal to carry.history.current_update. At an
        accepted boundary, history.last_refresh_rounds equals progress.rounds.
    failed, failure_reason : Array
        Sticky scalar bool and int32 reason. Zero means no failure; 1 collection,
        2 boundary/count mismatch, 3 nonfinite batch, 4 invalid behavior action,
        5 nonfinite update, and 6 rejected history publication. Failed states
        must not be used for further collection or saved as usable checkpoints.

    Notes
    -----
    This immutable PyTree contains no host descriptor, saved rollout, expanded
    actor features or duplicate actor variables. Arrays stay dynamic under jit.
    """

    carry: TrainingCarry
    critic_params: Tree
    actor_opt_state: Tree
    critic_opt_state: Tree
    critic_memory: Array
    shuffle_root: Array
    completed_updates: Array
    failed: Array
    failure_reason: Array


class UpdateSummary(NamedTuple):
    """Hold small sums and counts from one successfully learned real prefix.

    task_reward_sum is float32 summed over valid active Team A rows;
    active_samples is its int32 denominator. shaping_reward_sum is float32
    summed once per valid game; real_transitions is its int32 denominator.
    live_actor_decisions counts valid active living Team A rows once. These
    counts do not include repeated PPO epochs. These five fields are scalar.
    stage_completed is int32 (17,3) finished-game counts in Team A win/draw/loss
    order, grouped by the producing game's reset-time stage. stage_score_sums is
    int32 (17,2) summed Team A/B final scores; stage_length_sum is int32 (17,)
    summed completed-game lengths. stage_k20_count is int32 (17,) counting
    completed games where either team reached 20 points. Continuing games and
    padding contribute nothing to these four completion tables. Existing
    collection progress owns actual live decisions and exposure per stage.

    Empty or rejected updates return zeros; UpdateResult.performed decides
    whether these are observations. Accumulate whole-run counts as host integers.
    """

    task_reward_sum: Array
    shaping_reward_sum: Array
    active_samples: Array
    live_actor_decisions: Array
    real_transitions: Array
    stage_completed: Array
    stage_score_sums: Array
    stage_length_sum: Array
    stage_k20_count: Array


class UpdateResult(NamedTuple):
    """Return one fixed-shape update result for success, empty and failure paths.

    performed is scalar bool and is True only after a nonempty accepted update.
    metrics is PPOMetrics with (epochs,minibatches) leaves. failed/reason mirror
    the returned LearnerState. snapshot is the existing fixed SnapshotEvent;
    its created flag describes a history capture, not whether learning occurred.
    summary contains unrepeated experience sums/counts. Absent updates use zero
    metric/summary payloads and an absent snapshot. Report these measurements as
    missing when performed is False. All branches retain identical JAX structure.
    """

    performed: Array
    metrics: PPOMetrics
    failed: Array
    failure_reason: Array
    snapshot: SnapshotEvent
    summary: UpdateSummary


def _cond[T](predicate: Array, yes: Callable[[None], T], no: Callable[[None], T]) -> T:
    """Keep static type information around JAX's same-tree conditional branches."""
    return cast(T, jax.lax.cond(predicate, yes, no, None))


def _array(value: Array, shape: tuple[int, ...], dtype: object, name: str) -> None:
    """Reject a leaf whose static shape or dtype differs from its named contract."""
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")


def _state_shapes(state: LearnerState) -> int:
    """Check fixed learner memory/key/count shapes and return its game count.

    This uses only static metadata and supports jit. Numerical values and the
    network/optimizer trees are checked at their separate setup/update boundaries.
    """
    games = state.carry.state.episode_id.shape[0]
    for name, memory in (
        ("Actor memory", state.carry.memory.team_a),
        ("Critic memory", state.critic_memory),
    ):
        _array(memory, (games, 5, HIDDEN_SIZE), jnp.float32, name)
    for name in ("completed_updates", "failure_reason"):
        _array(getattr(state, name), (), jnp.int32, name)
    _array(state.failed, (), jnp.bool_, "failed")
    if (
        state.shuffle_root.shape != ()
        or str(jax.random.key_impl(state.shuffle_root)) != "threefry2x32"
    ):
        raise ValueError("Learner shuffle_root must be one typed Threefry key")
    return games


def _rollout_shapes(state: LearnerState, rollout: TrainingRollout) -> tuple[int, int]:
    """Check static MAPPO rollout leaves before tracing critic or update work.

    Require positive output capacity, compact physical features and the existing
    PPOLearningOutputs record. Real-prefix length remains a numerical guard.
    Return (capacity, games); these are structural Python integers.
    """
    games = _state_shapes(state)
    rows = rollout.transitions
    if rows.valid.ndim != 2 or rows.valid.shape[1] != games or rows.valid.shape[0] < 1:
        raise ValueError("MAPPO rollout needs positive (T,B) capacity")
    length = rows.valid.shape[0]
    if not isinstance(rows.learning_outputs, PPOLearningOutputs):
        raise TypeError("MAPPO collection must return PPOLearningOutputs")
    if rows.training_state is None or rollout.final_training_state is None:
        raise ValueError("MAPPO collection must enable collect_training_state")
    _array(
        rows.training_state,
        (length, games, TRAINING_STATE_FEATURE_SIZE),
        jnp.float32,
        "training_state",
    )
    _array(
        rollout.final_training_state,
        (games, TRAINING_STATE_FEATURE_SIZE),
        jnp.float32,
        "final_training_state",
    )
    _array(
        rollout.initial_memory, (games, 5, HIDDEN_SIZE), jnp.float32, "initial_memory"
    )
    _array(rollout.real_steps, (), jnp.int32, "real_steps")
    for name in ("valid", "episode_start", "ended"):
        _array(getattr(rows, name), (length, games), jnp.bool_, name)
    for name in ("active", "alive"):
        _array(getattr(rows, name), (length, games, 5), jnp.bool_, name)
    for name in ("final_active", "final_alive"):
        _array(getattr(rollout, name), (games, 5), jnp.bool_, name)
    _array(rollout.final_ended, (games,), jnp.bool_, "final_ended")
    _array(rows.task_rewards, (length, games, 5), jnp.float32, "task_rewards")
    _array(rows.shaping_reward, (length, games), jnp.float32, "shaping_reward")
    _array(
        rows.learning_outputs.action_indices, (length, games, 5), jnp.int32, "actions"
    )
    _array(
        rows.learning_outputs.log_prob, (length, games, 5), jnp.float32, "old_log_prob"
    )
    _array(rows.learner_update, (length, games), jnp.int32, "learner_update")
    for head in rows.actions:
        _array(head, (length, games, 10), jnp.int32, "native action")
    return length, games


def _finite(tree: Tree) -> Array:
    """Reduce inexact numerical leaves to one on-device finite flag.

    Integer, Boolean and key leaves need no finite check. No arrays are copied
    to the host; callers separately check their shapes and count/key contracts.
    """
    checks = [
        jnp.all(jnp.isfinite(value))
        for value in jax.tree.leaves(tree)
        if jnp.issubdtype(value.dtype, jnp.inexact)
    ]
    return jnp.all(jnp.stack(checks)) if checks else jnp.bool_(True)


_boundary_finite = cast(Callable[[LearnerState], Array], jax.jit(_finite))


def _ppo_state(state: LearnerState) -> PPOTrainState:
    """Borrow the single current actor tree for one temporary numerical update."""
    return PPOTrainState(
        state.carry.history.current_variables,
        state.critic_params,
        state.actor_opt_state,
        state.critic_opt_state,
    )


@lru_cache(maxsize=16)
def _ppo_spec(ppo: PPOConfig) -> PPOTrainState:
    """Trace the known initializer once per static config to obtain safe leaf specs.

    This host helper executes no network initialization or optimizer step. The
    cached tree contains only shapes/dtypes, not weights or experiment arrays.
    """
    return cast(
        PPOTrainState,
        jax.eval_shape(partial(initialize_ppo, config=ppo), jax.random.key(0)),
    )


def init_learner(
    *,
    schedule: TrainingSchedule,
    seed: int = 42,
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
    prepared: PreparedTrainingContent | None = None,
    shaping: bool = False,
    shaping_coefficient: float = 0.01,
    shaping_mode: str = "potential",
    metrics: str = "priority",
    recording: bool = False,
    pinned_opponent_share: float = 0.0,
) -> tuple[TrainingCollection, LearnerState]:
    """Initialize one untrained MAPPO learner and its verified collection setup.

    Parameters
    ----------
    schedule : TrainingSchedule
        Checked exact budget and positive even batch from make_training_schedule.
        B must be divisible by ppo.groups*ppo.minibatches.
    seed : int, default=42
        Python run seed, excluding bool. Collection retains its own unchanged
        streams. Version-1 model/shuffle tags derive separate learner keys.
    ppo : PPOConfig, default=DEFAULT_PPO_CONFIG
        Static donor update settings. Counters and per-minibatch sample counts
        must fit int32 for the complete scheduled run.
    prepared : PreparedTrainingContent or None, default=None
        Existing verified content, or None to perform the collection preflight.
    shaping : bool, default=False
        Enable the chosen team reward adjustment. Potential uses ppo.gamma.
    shaping_coefficient : float, default=0.01
        Nonnegative finite weight checked by the existing shaping authority.
    shaping_mode : {"potential", "score_delta"}, default="potential"
        Fixed training reward method, used when shaping is enabled. Potential
        preserves the discounted task objective. Score_delta rewards new team
        kills minus deaths, including at a real ending, and changes that objective.
    metrics : str, default="priority"
        Collection metrics mode: "priority" or "none".
    recording : bool, default=False
        Retain numerical recording starts for a later caller-owned writer.
    pinned_opponent_share : float, default=0.0
        Static probability, within [0, 0.8], that a reset lane meets the pinned
        first-update actor in history slot 0. Zero keeps the existing 80/20
        self-play recipe. A positive share needs a schedule built with
        early_history_capture=True; the collection setup checks that pairing.

    Returns
    -------
    tuple[TrainingCollection, LearnerState]
        Stable host descriptor and dynamic numerical update-0 state. Critic
        memory and optimizer counters are zero; the history has no snapshots.

    Raises
    ------
    TypeError, ValueError
        Setup types, batch divisibility, counter capacity, verified content or
        the existing collection/shaping input contracts are invalid.

    Notes
    -----
    Requires the training extra. Host setup performs content checks, allocation,
    network initialization and reset, but no actor decision or learner update.
    It opens no writer and saves no artifact. Reuse the returned descriptor.
    """
    if not isinstance(cast(object, schedule), TrainingSchedule) or not isinstance(
        cast(object, ppo), PPOConfig
    ):
        raise TypeError("Learner setup requires TrainingSchedule and PPOConfig")
    if type(seed) is not int:
        raise TypeError("Learner seed must be a Python integer")
    games = schedule.num_envs
    if games % (ppo.groups * ppo.minibatches):
        raise ValueError("Learner B must be divisible by groups*minibatches")
    updates = (
        int(schedule.arrays.total_rounds) + ppo.rollout_length - 1
    ) // ppo.rollout_length
    if (
        updates * ppo.epochs * ppo.minibatches > _MAX_COUNT
        or ppo.rollout_length * games * 5 > _MAX_COUNT
    ):
        raise ValueError("Learner optimizer or block sample counts exceed int32")
    root = jax.random.key(seed, impl="threefry2x32")
    initialized = initialize_ppo(
        jax.random.fold_in(root, MODEL_INITIALIZATION_TAG), ppo
    )
    actor = make_recurrent_mappo_system(
        initialized.actor_params, input_scale=ppo.input_scale
    )
    collection, carry = init_training_collection(
        actor,
        initialized.actor_params,
        schedule=schedule,
        seed=seed,
        prepared=prepared,
        shaping=shaping,
        discount=ppo.gamma,
        coefficient=shaping_coefficient,
        shaping_mode=shaping_mode,
        collect_training_state=True,
        metrics=metrics,
        recording=recording,
        pinned_opponent_share=pinned_opponent_share,
    )
    state = LearnerState(
        carry,
        initialized.critic_params,
        initialized.actor_opt_state,
        initialized.critic_opt_state,
        jnp.zeros((games, 5, HIDDEN_SIZE), jnp.float32),
        jax.random.fold_in(root, SHUFFLE_ROOT_TAG),
        jnp.int32(0),
        jnp.bool_(False),
        jnp.int32(LEARNER_ERROR_NONE),
    )
    _state_shapes(state)
    return collection, state


def build_ppo_batch(
    state: LearnerState,
    rollout: TrainingRollout,
    *,
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
) -> tuple[PPOBatch, Array]:
    """Pair same-call behavior with fixed pre-update critic values and memory.

    Parameters
    ----------
    state : LearnerState
        Accepted learner state immediately before collecting this block. Only
        its critic parameters and pre-sequence critic memory enter value work.
    rollout : TrainingRollout
        Compact MAPPO rollout with positive T capacity, optional suffix padding,
        PPOLearningOutputs and physical-state collection enabled. Its prefix and
        same-epoch fields must obey the existing collector contract.
    ppo : PPOConfig, default=DEFAULT_PPO_CONFIG
        Original learner settings. The critic uses its fixed input scale for
        both sequence values and the final bootstrap value.

    Returns
    -------
    tuple[PPOBatch, Array]
        Complete fixed-length PPO batch and float32 (B,5,128) critic carry after
        real pre-action rows. The bootstrap-only call's memory is discarded.
        First-row start lanes have zero temporary initial actor/critic memory;
        continuing/dead lanes preserve it. Old values and final values use the
        same supplied critic. Padding values and terminal bootstraps are zero.

    Raises
    ------
    TypeError, ValueError
        Static MAPPO output, memory, feature or decision-field shapes/dtypes fail.

    Notes
    -----
    Pure device work supports jit and changes no inputs. An empty real prefix
    skips both critic calls and retains memory; update_learner skips this whole
    helper for empty updates. No actor call, sampling, GAE, update or reset occurs.
    """
    length, games = _rollout_shapes(state, rollout)
    rows = rollout.transitions
    physical = cast(Array, rows.training_state)
    final_physical = cast(Array, rollout.final_training_state)
    reset = (rows.valid[0] & rows.episode_start[0])[:, None, None]
    actor_memory = jnp.where(reset, 0.0, rollout.initial_memory)
    critic_memory = jnp.where(reset, 0.0, state.critic_memory)

    def values(_: None) -> tuple[Array, Array, Array]:
        """Read the sequence and value its successor without retaining that carry."""
        memory, old = critic_values(
            state.critic_params,
            critic_memory,
            physical,
            rows.episode_start,
            rows.valid,
            input_scale=ppo.input_scale,
        )
        continues = ~rollout.final_ended

        def bootstrap(_: None) -> Array:
            """Value continuing final lanes without retaining bootstrap memory."""
            _unused_memory, final = critic_values(
                state.critic_params,
                memory,
                final_physical[None],
                jnp.zeros((1, games), jnp.bool_),
                continues[None],
                input_scale=ppo.input_scale,
            )
            return jnp.where(continues[:, None] & rollout.final_active, final[0], 0.0)

        final = _cond(
            jnp.any(continues),
            bootstrap,
            lambda _: jnp.zeros((games, 5), jnp.float32),
        )
        return memory, jnp.where(rows.valid[..., None], old, 0.0), final

    def empty(_: None) -> tuple[Array, Array, Array]:
        """Keep the input critic carry and emit finite values without a network call."""
        return (
            state.critic_memory,
            jnp.zeros((length, games, 5), jnp.float32),
            jnp.zeros((games, 5), jnp.float32),
        )

    memory, old_values, final_values = _cond(rollout.real_steps > 0, values, empty)
    return PPOBatch(
        rows.observations,
        physical,
        rows.action_mask,
        rows.learning_outputs.action_indices,
        rows.learning_outputs.log_prob,
        old_values,
        rows.task_rewards + jnp.where(rows.active, rows.shaping_reward[..., None], 0.0),
        rows.ended,
        rows.episode_start,
        rows.valid,
        rows.active,
        rows.alive,
        final_values,
        actor_memory,
        critic_memory,
    ), memory


def _absent_result(state: LearnerState, ppo: PPOConfig) -> UpdateResult:
    """Create fixed absent payloads and preserve the state's failure reason."""
    floats = jnp.zeros((ppo.epochs, ppo.minibatches), jnp.float32)
    counts = jnp.zeros((ppo.epochs, ppo.minibatches), jnp.int32)
    return UpdateResult(
        jnp.bool_(False),
        PPOMetrics(floats, floats, floats, counts, counts),
        state.failed,
        state.failure_reason,
        SnapshotEvent(
            jnp.bool_(False),
            jnp.int32(-1),
            jnp.int32(-1),
            jnp.int32(-1),
            jnp.zeros(20, jnp.bool_),
        ),
        UpdateSummary(
            jnp.float32(0),
            jnp.float32(0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.zeros((17, 3), jnp.int32),
            jnp.zeros((17, 2), jnp.int32),
            jnp.zeros(17, jnp.int32),
            jnp.zeros(17, jnp.int32),
        ),
    )


def _reject(
    state: LearnerState, reason: Array, ppo: PPOConfig
) -> tuple[LearnerState, UpdateResult]:
    """Keep the previous boundary and make its first failure sticky."""
    failed = state._replace(
        failed=jnp.bool_(True),
        failure_reason=jnp.where(state.failed, state.failure_reason, reason),
    )
    return failed, _absent_result(failed, ppo)


def _boundary_valid(
    state: LearnerState, collected: TrainingCarry, rollout: TrainingRollout
) -> Array:
    """Check round/update pairing and exact full-lane prefix before any value work."""
    rows = rollout.transitions
    length = rows.valid.shape[0]
    prefix = jnp.arange(length)[:, None] < rollout.real_steps
    prior = state.carry
    return (
        (state.completed_updates >= 0)
        & (state.completed_updates < _MAX_COUNT)
        & (state.completed_updates == prior.history.current_update)
        & (prior.history.last_refresh_rounds == prior.progress.rounds)
        & (collected.history.current_update == state.completed_updates)
        & (collected.history.last_refresh_rounds == prior.progress.rounds)
        & (rollout.real_steps >= 0)
        & (rollout.real_steps <= length)
        & (collected.progress.rounds >= prior.progress.rounds)
        & (collected.progress.rounds - prior.progress.rounds == rollout.real_steps)
        & jnp.all(rows.valid == prefix)
        & jnp.all(
            jnp.where(rows.valid, rows.learner_update == state.completed_updates, True)
        )
        & jnp.all(rollout.initial_memory == prior.memory.team_a)
        & jnp.all(rollout.final_ended == collected.state.done.done)
    )


def _behavior_valid(rollout: TrainingRollout) -> Array:
    """Check probabilities, legality and agreement with submitted Team A actions."""
    rows = rollout.transitions
    actions = rows.learning_outputs.action_indices
    mask = categorical_action_mask(rows.action_mask)
    selected = jnp.take_along_axis(mask, actions[..., None], axis=-1)[..., 0]
    native = ActorAction(*(head[..., :5] for head in rows.actions))
    return (
        jnp.all((actions >= 0) & (actions < NUM_ACTIONS))
        & jnp.all(selected)
        & jnp.all(actions == encode_actions(native))
        & jnp.all(jnp.isfinite(rows.learning_outputs.log_prob))
    )


def _summary(rollout: TrainingRollout) -> UpdateSummary:
    """Reduce only real transition rows to compact unrepeated learning-log facts."""
    rows = rollout.transitions
    active = rows.valid[..., None] & rows.active
    completed = rows.valid & rows.ended
    stages = jnp.where(completed, rows.episode_stage, 0).reshape(-1)

    def stage_sum(value: Array) -> Array:
        """Group masked completion facts without retaining a stage/time cube."""
        tail = value.shape[2:]
        return (
            jnp.zeros((17, *tail), jnp.int32).at[stages].add(value.reshape((-1, *tail)))
        )

    outcomes = jnp.stack(
        tuple(
            completed & (rows.outcome == code)
            for code in (
                TASK_MODE_OUTCOME_TEAM_A_WIN,
                TASK_MODE_OUTCOME_DRAW,
                TASK_MODE_OUTCOME_TEAM_B_WIN,
            )
        ),
        axis=-1,
    ).astype(jnp.int32)
    return UpdateSummary(
        jnp.sum(jnp.where(active, rows.task_rewards, 0.0)),
        jnp.sum(jnp.where(rows.valid, rows.shaping_reward, 0.0)),
        jnp.sum(active, dtype=jnp.int32),
        jnp.sum(active & rows.alive, dtype=jnp.int32),
        jnp.sum(rows.valid, dtype=jnp.int32),
        stage_sum(outcomes),
        stage_sum(jnp.where(completed[..., None], rows.final_scores, 0)),
        stage_sum(jnp.where(completed, rows.episode_length, 0)),
        stage_sum(
            (completed & jnp.any(rows.final_scores >= 20, axis=-1)).astype(jnp.int32)
        ),
    )


def update_learner(
    state: LearnerState,
    collected: TrainingCarry,
    rollout: TrainingRollout,
    *,
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
) -> tuple[LearnerState, UpdateResult]:
    """Accept one real collection block, update PPO and publish its actor once.

    Parameters
    ----------
    state : LearnerState
        Last accepted pre-collection boundary. Keep this unchanged while calling
        collection with state.carry. A failed state remains failed.
    collected : TrainingCarry
        Successor returned by that collection call. It must retain the same
        current actor/history version. No intervening update is permitted.
    rollout : TrainingRollout
        Matching compact rollout with T=ppo.rollout_length, including any final
        invalid suffix. B must divide into the configured groups/minibatches.
    ppo : PPOConfig, default=DEFAULT_PPO_CONFIG
        Same static settings used at initialization. Capture these in the jit
        wrapper; changing ordinary state/data values does not make them static.

    Returns
    -------
    tuple[LearnerState, UpdateResult]
        Accepted successor and small fixed-shape result. A nonempty successful
        update advances the version once and refreshes opponents once. An empty
        block returns state unchanged. Invalid boundaries/actions, collection
        errors or nonfinite values reject the entire candidate and retain the
        prior numerical boundary with a sticky failure. Its recording may have
        progressed: the host must stop and recover from a saved checkpoint.

    Raises
    ------
    TypeError, ValueError
        Static rollout, behavior, memory or configured batch shapes are invalid.

    Notes
    -----
    Wrap in jit with fixed ppo. No host callbacks, extra actor call, reset, I/O
    or duplicated GAE calculation occurs. Finite checks reduce on device. Never
    collect from a returned failed state. Host code must check result.failed
    before logging an accepted update, saving it or scheduling another action.
    """
    length, games = _rollout_shapes(state, rollout)
    if length != ppo.rollout_length or games % (ppo.groups * ppo.minibatches):
        raise ValueError(
            "MAPPO needs T=rollout_length and B divisible by groups*minibatches"
        )
    if collected.state.episode_id.shape != (games,):
        raise ValueError("Collected boundary must retain the learner batch")
    entry_reason = jnp.where(
        _failed(state.carry) | _failed(collected),
        LEARNER_ERROR_COLLECTION,
        jnp.where(
            _boundary_valid(state, collected, rollout),
            LEARNER_ERROR_NONE,
            LEARNER_ERROR_BOUNDARY,
        ),
    ).astype(jnp.int32)

    def nonempty(_: None) -> tuple[LearnerState, UpdateResult]:
        """Build fixed old-value data and conditionally accept the update."""
        batch, memory = build_ppo_batch(state, rollout, ppo=ppo)
        batch_reason = jnp.where(
            ~_finite((batch, memory)),
            LEARNER_ERROR_NONFINITE_BATCH,
            jnp.where(
                _behavior_valid(rollout), LEARNER_ERROR_NONE, LEARNER_ERROR_ACTION
            ),
        ).astype(jnp.int32)

        def learn(_: None) -> tuple[LearnerState, UpdateResult]:
            """Apply the existing PPO once; publish only a finite complete candidate."""
            index = state.completed_updates + jnp.int32(1)
            updated, metrics = update_recurrent_ppo(
                _ppo_state(state),
                batch,
                jax.random.fold_in(state.shuffle_root, index),
                ppo,
            )

            def publish(_: None) -> tuple[LearnerState, UpdateResult]:
                """Publish actor variables without changing either team's memory."""
                history, event = refresh_opponents(
                    collected.history,
                    updated.actor_params,
                    completed_rounds=collected.progress.rounds,
                    update_index=index,
                    schedule=collected.schedule,
                )
                successor = LearnerState(
                    collected._replace(history=history),
                    updated.critic_params,
                    updated.actor_opt_state,
                    updated.critic_opt_state,
                    memory,
                    state.shuffle_root,
                    index,
                    jnp.bool_(False),
                    jnp.int32(LEARNER_ERROR_NONE),
                )
                result = UpdateResult(
                    jnp.bool_(True),
                    metrics,
                    successor.failed,
                    successor.failure_reason,
                    event,
                    _summary(rollout),
                )
                return _cond(
                    history.error,
                    lambda _: _reject(state, jnp.int32(LEARNER_ERROR_HISTORY), ppo),
                    lambda _: (successor, result),
                )

            return _cond(
                _finite((updated, metrics)),
                publish,
                lambda _: _reject(
                    state, jnp.int32(LEARNER_ERROR_NONFINITE_UPDATE), ppo
                ),
            )

        return _cond(
            batch_reason == LEARNER_ERROR_NONE,
            learn,
            lambda _: _reject(state, batch_reason, ppo),
        )

    def accepted(_: None) -> tuple[LearnerState, UpdateResult]:
        """Skip all value/update/key/publication work for a genuinely empty prefix."""
        return _cond(
            rollout.real_steps > 0,
            nonempty,
            lambda _: (state, _absent_result(state, ppo)),
        )

    return _cond(
        state.failed | (entry_reason != LEARNER_ERROR_NONE),
        lambda _: _reject(state, entry_reason, ppo),
        accepted,
    )


def validate_learner(
    collection: TrainingCollection,
    state: LearnerState,
    *,
    ppo: PPOConfig = DEFAULT_PPO_CONFIG,
    recheck_installed_content: bool = True,
) -> None:
    """Validate a complete accepted learner before any mutable recording recovery.

    Parameters
    ----------
    collection : TrainingCollection
        Reconstructed matching host descriptor with verified content/settings.
    state : LearnerState
        Fully restored numerical boundary, including initialization at update 0.
    ppo : PPOConfig, default=DEFAULT_PPO_CONFIG
        Original static learner settings; optimizer/network shapes must agree.
    recheck_installed_content : bool, default=True
        Python bool. True reruns installed-content preflight, as restore requires.
        Checkpoint saves use False to reuse the immutable descriptor verified
        at setup or restore. Both routes check the actual carried source bank,
        numerical state, keys and counters.

    Raises
    ------
    TypeError, ValueError
        Structure, shapes, dtypes, content, collection bindings, learner keys,
        counters, finite arrays or optimizer states do not describe a valid
        completed-update boundary. A sticky failure is never resumable here.
        The content-recheck switch must be a Python bool.

    Notes
    -----
    Host-only. Reuses collection's continuation check, with installed-content
    preflight enabled by default. It performs reductions/transfers and shape-only
    network tracing; no actor decision, reset, optimizer update or writer change occurs.
    Saved source/dependency/file hashes and configuration identity belong to the
    checkpoint owner and must be checked separately before invoking this helper.
    """
    if type(recheck_installed_content) is not bool:
        raise TypeError("recheck_installed_content must be a Python bool")
    games = _state_shapes(state)
    if games % (ppo.groups * ppo.minibatches) or not collection.collect_training_state:
        raise ValueError("Learner descriptor or PPO batch is incompatible")
    if bool(state.failed) or int(state.failure_reason) != LEARNER_ERROR_NONE:
        raise ValueError("A failed learner boundary cannot be restored")
    _validate_training_continuation(
        collection,
        state.carry,
        recheck_installed_content=recheck_installed_content,
    )
    count = int(state.completed_updates)
    history = state.carry.history
    if (
        count < 0
        or count != int(history.current_update)
        or int(history.last_refresh_rounds) != int(state.carry.progress.rounds)
    ):
        raise ValueError("Learner update and collection boundary counts disagree")
    expected_key = jax.random.fold_in(state.carry.root_key, SHUFFLE_ROOT_TAG)
    if not np.array_equal(
        jax.random.key_data(state.shuffle_root), jax.random.key_data(expected_key)
    ):
        raise ValueError("Learner shuffle key does not match the run root")
    numerical = _ppo_state(state)
    spec = _ppo_spec(ppo)
    if jax.tree.structure(numerical) != jax.tree.structure(spec):
        raise ValueError("Learner network or optimizer structure is incompatible")
    for actual, expected in zip(
        jax.tree.leaves(numerical), jax.tree.leaves(spec), strict=True
    ):
        _array(actual, expected.shape, expected.dtype, "Learner network/optimizer leaf")
    # Save/restore admission checks the whole numerical boundary, including
    # frozen opponents and game/observation state, in one device reduction.
    if not bool(_boundary_finite(state)):
        raise ValueError(
            "Learner contains nonfinite network, optimizer, memory or collection values"
        )
    for optimizer in (state.actor_opt_state, state.critic_opt_state):
        for leaf in jax.tree.leaves(optimizer):
            if jnp.issubdtype(leaf.dtype, jnp.integer) and (
                np.any(np.asarray(leaf) < 0)
                or np.any(np.asarray(leaf) > count * ppo.epochs * ppo.minibatches)
            ):
                raise ValueError("Learner optimizer counter exceeds completed updates")
