"""Join shared training collection to recurrent QMIX through compact replay.

This module owns QMIX's replay and learner lifecycle. Replay: the compact Team A
row stored for every real decision, the Flashbax trajectory buffer that holds
those rows per game, guarded insertion of a collected block, the readiness
check, sequence sampling, and the expansion of a sample into the network-frame
QMIXBatch that ``baselines.qmix.update_qmix`` reads. Lifecycle:
``init_qmix_learner`` builds the collection and untrained networks,
``update_qmix_learner`` accepts one collected block (a warmup block only stores
rows; a ready block runs ``epochs`` sampled optimizer steps and publishes the
actor once) and ``validate_qmix_learner`` checks a saved boundary. Collection,
curriculum, shaping, self-play history, pinned opponents and recording stay with
their existing owners; PPO never imports this module or Flashbax.

Replay stores only permitted Team A material: the five Team A observation rows,
their 5x5 source permissions, Team A masks and chosen actions (world frame),
the team task reward and shaping reward separately, lifecycle flags, the
920-value physical state once per game row, and eight int32 identity fields.
Rebuilding the actor inputs of a sample uses the same builder as live action
selection, so the learner sees exactly what the actor saw. Every numerical
helper here is pure JAX for jit and scan; replay descriptors are host setup.
Requires the training extra.
"""

import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

try:
    import flashbax as fbx  # pyright: ignore[reportMissingTypeStubs]
    from flashbax.buffers.trajectory_buffer import (  # pyright: ignore[reportMissingTypeStubs]
        TrajectoryBufferState,
    )
except ImportError as error:
    raise ImportError(
        "QMIX replay needs the training extra. Install marl-battlegrounds[training]."
    ) from error

import numpy as np

from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    categorical_action_mask,
    encode_actions,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import (
    encode_actor_inputs,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.baselines.qmix import (
    DEFAULT_QMIX_CONFIG,
    QMIX_HIDDEN_SIZE,
    TEAM_SLOTS,
    QMIXActorVariables,
    QMIXBatch,
    QMIXConfig,
    QMIXExploration,
    QMIXMetrics,
    QMIXTrainState,
    epsilon_at,
    epsilon_reference,
    initialize_qmix,
    make_qmix_system,
    team_task_reward,
    update_qmix,
    validate_qmix_batch_size,
)
from marl_battlegrounds.core.types import ActionMask, Observation
from marl_battlegrounds.evaluation.policy_execution import Policy, System
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import (
    Observations,
    build_team_actor_input,
    mirror_team_view,
)
from marl_battlegrounds.training._content import PreparedTrainingContent
from marl_battlegrounds.training._continuation_schedules import (
    ContinuationExploration,
    LearnerContinuation,
    continuation_config,
    schedule_continuation,
)

# Collection's padding template, failure and continuation checks, and the PPO
# learner's summary and error codes keep one owner each.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    TrainingTransition,
    _failed,
    _padding,
    _validate_training_continuation,
    init_training_collection,
)
from marl_battlegrounds.training.curriculum import TrainingSchedule
from marl_battlegrounds.training.learner import (
    LEARNER_ERROR_ACTION,
    LEARNER_ERROR_BOUNDARY,
    LEARNER_ERROR_COLLECTION,
    LEARNER_ERROR_HISTORY,
    LEARNER_ERROR_NONE,
    LEARNER_ERROR_NONFINITE_BATCH,
    LEARNER_ERROR_NONFINITE_UPDATE,
    UpdateSummary,
    _array,
    _cond,
    _finite,
    _summary,
)
from marl_battlegrounds.training.opponents import SnapshotEvent, refresh_opponents

type Tree = Any


class QMIXReplayRow(NamedTuple):
    """Hold one game's compact Team A facts from one real decision.

    Leaves carry the leading axes of their source: (B,) for one collected
    round, (B,C) inside the buffer and (M,S) in a sample. The shapes below are
    per game row.

    Attributes
    ----------
    observation : Observation
        Team A's five observer rows of the pre-action observation.
    source_availability : Array
        Bool (5,5) Team A source permissions, recipient by source.
    action_mask : ActionMask
        Team A's native move, target, Ultimate and joint masks for this epoch.
    actions : Array
        Int32 (5,) categorical indices of the submitted Team A actions, world
        frame.
    task_reward : Array
        Float32 team task reward: the mean native reward over configured slots.
    shaping_reward : Array
        Float32 team shaping reward (0 when shaping is off).
    active, alive : Array
        Bool (5,) configured and living Team A slots before the action.
    episode_start, ended, valid : Array
        Bool. First decision of an episode; this decision ended the episode (a
        real ending, including a horizon draw, not a collection cutoff); real
        decision. Stored rows are always valid.
    training_state : Array
        Float32 (920,) world-frame physical state for the mixer only.
    episode_id, decision_step, requested_stage, episode_stage, source_index,
    learner_update, opponent_update, opponent_snapshot : Array
        Int32 identities copied from the transition, for sample checks and
        exposure counts. They never enter a network.

    Notes
    -----
    One row is 29,688 bytes with the current encoder: 25,715 bytes of Team A
    observation and permissions and 3,973 bytes of the rest. No Team B row,
    opposite-team permission, learning output or expanded feature is stored.
    Replay schema version: ``baselines.qmix.QMIX_REPLAY_SCHEMA_VERSION``.
    """

    observation: Observation
    source_availability: Array
    action_mask: ActionMask
    actions: Array
    task_reward: Array
    shaping_reward: Array
    active: Array
    alive: Array
    episode_start: Array
    ended: Array
    valid: Array
    training_state: Array
    episode_id: Array
    decision_step: Array
    requested_stage: Array
    episode_stage: Array
    source_index: Array
    learner_update: Array
    opponent_update: Array
    opponent_snapshot: Array


type ReplayState = TrajectoryBufferState[QMIXReplayRow]
"""Flashbax trajectory storage of QMIXReplayRow leaves shaped (B, C, ...)."""


def _replay_rows(rows: TrainingTransition) -> QMIXReplayRow:
    """Map transition rows with leading axes (..., B) to compact replay rows.

    rows must carry physical training state (collect_training_state=True).
    Team A is routing rows 0-4 of every observation leaf and permission axis.
    Raises ValueError when training_state is None. Pure JAX; it copies only
    the selected Team A slices.
    """
    if rows.training_state is None:
        raise ValueError("QMIX replay needs collect_training_state=True")
    observer_axis = rows.valid.ndim

    def team_a(value: Array) -> Array:
        """Keep the five Team A observer rows of one observation leaf."""
        return jax.lax.slice_in_dim(value, 0, TEAM_SLOTS, axis=observer_axis)

    native = ActorAction(*(head[..., :TEAM_SLOTS] for head in rows.actions))
    return QMIXReplayRow(
        jax.tree.map(team_a, rows.observations.observation),
        rows.observations.source_availability[..., :TEAM_SLOTS, :TEAM_SLOTS],
        rows.action_mask,
        encode_actions(native),
        team_task_reward(rows.task_rewards, rows.active),
        rows.shaping_reward,
        rows.active,
        rows.alive,
        rows.episode_start,
        rows.ended,
        rows.valid,
        rows.training_state,
        rows.episode_id,
        rows.decision_step,
        rows.requested_stage,
        rows.episode_stage,
        rows.source_index,
        rows.learner_update,
        rows.opponent_update,
        rows.opponent_snapshot,
    )


def _check_replay_order(qmix: QMIXConfig) -> None:
    """Repeat the buffer-size ordering check before Flashbax sees the settings.

    Flashbax silently raises a minimum below the sequence length; BG refuses
    instead. QMIXConfig already enforces this; the repeat guards direct calls.
    Raises ValueError.
    """
    if (
        not (
            qmix.buffer_size >= qmix.min_buffer_size >= qmix.sample_sequence_length >= 2
        )
        or not 1 <= qmix.rollout_length <= qmix.buffer_size
    ):
        raise ValueError("QMIX replay sizes are out of order")


@lru_cache(maxsize=16)
def _replay_buffer(qmix: QMIXConfig, num_envs: int) -> Any:  # noqa: ANN401
    """Return the cached Flashbax trajectory buffer for these static settings.

    qmix supplies capacity C, minimum, sequence length S and batch M; num_envs
    is the number of game lanes B. Sampling period is 1 (any start). Any
    Flashbax warning, such as its silent minimum raise, becomes an error.
    Host-only; the returned object holds pure functions, never arrays.
    """
    _check_replay_order(qmix)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return cast(
            Any,
            fbx.make_trajectory_buffer(
                add_batch_size=num_envs,
                sample_batch_size=qmix.sample_batch_size,
                sample_sequence_length=qmix.sample_sequence_length,
                period=1,
                min_length_time_axis=qmix.min_buffer_size,
                max_length_time_axis=qmix.buffer_size,
            ),
        )


def _row_spec(collection: TrainingCollection, carry: TrainingCarry) -> QMIXReplayRow:
    """Trace one compact row's (B, ...) shapes from collection's padding row.

    Shape-only: no array is created. Raises ValueError when the collection
    does not keep physical training state.
    """
    if not collection.collect_training_state:
        raise ValueError("QMIX replay needs collect_training_state=True")

    def row(values: TrainingCarry) -> QMIXReplayRow:
        """Build the compact form of collection's finite padding row."""
        return _replay_rows(_padding(collection, values))

    return cast(QMIXReplayRow, jax.eval_shape(row, carry))


def _init_replay(
    collection: TrainingCollection, carry: TrainingCarry, qmix: QMIXConfig
) -> ReplayState:
    """Allocate empty replay storage with explicit zeros and a strong index.

    Every leaf is zero-filled with shape (B, buffer_size, ...), the write
    index is a strong int32 0 and is_full is False. Flashbax's own init uses
    empty storage and a weak index, which would change after a checkpoint
    restore and force a new compilation. At B=32 and C=1000 this allocates
    950,016,000 bytes on the default device.
    """
    spec = _row_spec(collection, carry)
    capacity = qmix.buffer_size

    def zeros(value: jax.ShapeDtypeStruct) -> Array:
        """Allocate one zero leaf with the capacity axis after the game axis."""
        return jnp.zeros((value.shape[0], capacity, *value.shape[1:]), value.dtype)

    return TrajectoryBufferState[QMIXReplayRow](
        experience=jax.tree.map(zeros, spec),
        current_index=jnp.asarray(0, jnp.int32),
        is_full=jnp.asarray(False),
    )


def _insert_rollout(
    replay: ReplayState, rollout: TrainingRollout, qmix: QMIXConfig
) -> ReplayState:
    """Append the real prefix of one collected block, one round at a time.

    rollout has static capacity T and a dynamic real prefix
    ``rollout.real_steps``. Round t is added only when ``t < real_steps``, so
    padding rows are never stored and the final successor observation is not a
    row. Each add writes one row per game at the ring index and advances it.
    Pure JAX; the caller keeps the input replay for rollback.
    """
    buffer = _replay_buffer(qmix, rollout.transitions.valid.shape[1])
    rows = _replay_rows(rollout.transitions)
    for index in range(rollout.transitions.valid.shape[0]):

        def one(value: Array, step: int = index) -> Array:
            """Select round `step` as a (B, 1, ...) Flashbax batch."""
            return value[step][:, None]

        row = cast(QMIXReplayRow, jax.tree.map(one, rows))

        def add(state: ReplayState, values: QMIXReplayRow = row) -> ReplayState:
            """Write this round's row for every game and advance the index."""
            return cast(ReplayState, buffer.add(state, values))

        def keep(state: ReplayState) -> ReplayState:
            """Leave the replay unchanged for a padding round."""
            return state

        replay = cast(
            ReplayState, jax.lax.cond(index < rollout.real_steps, add, keep, replay)
        )
    return replay


def _replay_ready(replay: ReplayState, qmix: QMIXConfig) -> Array:
    """Return bool True once min_buffer_size rows per game are stored."""
    return replay.is_full | (replay.current_index >= qmix.min_buffer_size)


def _sample_rows(
    replay: ReplayState, key: Array, qmix: QMIXConfig, num_envs: int
) -> QMIXReplayRow:
    """Draw M sequences of S consecutive stored rows, with replacement.

    key is one Threefry key. Each sequence picks a game uniformly and a start
    uniformly among the starts whose S rows are stored in time order. Call
    only when ready: Flashbax returns zero rows otherwise. Leaves are (M,S,...).
    Sampling never changes the replay.
    """
    buffer = _replay_buffer(qmix, num_envs)
    return cast(QMIXReplayRow, buffer.sample(replay, key).experience)


def _expand_sample(sample: QMIXReplayRow, qmix: QMIXConfig) -> QMIXBatch:
    """Rebuild network-frame inputs for an (M,S) replay sample.

    The five stored Team A rows and their permissions are padded back to ten
    observer rows with zero Team B rows and a zero 10x10 permission matrix, then
    passed through ``build_team_actor_input`` for Team A, the same builder as
    live action selection. In the "left" frame, flagged rows' views, masks and
    stored world actions are reflected. Rewards are task plus shaping.
    Returns float32 (M,S,5,5165) features among the QMIXBatch leaves; about
    264 MB for M=128, S=20. Pure JAX.
    """
    m, s = sample.valid.shape

    def flatten(value: Array) -> Array:
        """Merge the (M,S) sample axes into one row axis."""
        return value.reshape(m * s, *value.shape[2:])

    flat = cast(QMIXReplayRow, jax.tree.map(flatten, sample))

    def ten_rows(value: Array) -> Array:
        """Restore the routing axis to ten rows with zero Team B rows."""
        return jnp.concatenate((value, jnp.zeros_like(value)), axis=1)

    permissions = (
        jnp.zeros((m * s, 2 * TEAM_SLOTS, 2 * TEAM_SLOTS), jnp.bool_)
        .at[:, :TEAM_SLOTS, :TEAM_SLOTS]
        .set(flat.source_availability)
    )
    observations = Observations(jax.tree.map(ten_rows, flat.observation), permissions)
    actors = jax.vmap(build_team_actor_input, in_axes=(0, None))(observations, 0)
    mask, actions = flat.action_mask, flat.actions
    if qmix.spawn_frame != "world":
        flag = spawn_frame_flag(actors, qmix.spawn_frame)
        actors, mask = mirror_team_view(
            actors,
            mask,
            flag,
            obstacle_partners=team_obstacle_partners(actors),
        )
        # Mirroring is its own inverse, so it also maps world to network frame.
        actions = mirror_action_indices(actions, flag)
    features = encode_actor_inputs(actors)

    def rows(value: Array) -> Array:
        """Restore the (M,S) sample axes."""
        return value.reshape(m, s, *value.shape[1:])

    return QMIXBatch(
        rows(features),
        rows(categorical_action_mask(mask)),
        rows(actions),
        sample.task_reward + sample.shaping_reward,
        sample.episode_start,
        sample.ended,
        sample.valid,
        sample.active,
        sample.training_state,
    )


QMIX_MODEL_INITIALIZATION_TAG = 0x514D4958
"""Tag folded into the run root to derive the one QMIX model key ("QMIX")."""
QMIX_SAMPLING_ROOT_TAG = 0x51534D50
"""Tag folded into the run root to derive the replay sampling root ("QSMP")."""
LEARNER_ERROR_SEQUENCE = 7
"""Failure reason: a replay sample broke episode or decision-step order."""
_MAX_COUNT = int(np.iinfo(np.int32).max)
_STAGES = 17
_OPPONENT_ROWS = 21


class QMIXLearnerState(NamedTuple):
    """Keep one complete numerical QMIX boundary with one current actor tree.

    Attributes
    ----------
    carry : TrainingCarry
        Complete collection state. ``carry.history.current_variables`` is the
        only current actor: QMIXActorVariables with the online Q-network and
        the epsilon of the clock at this boundary.
    mixer_params, target_q_params, target_mixer_params : PyTree
        Online mixer and the two target networks.
    opt_state : PyTree
        Chained Adam state over ``(online Q, online mixer)``.
    replay : ReplayState
        Compact per-game replay, never flushed.
    sampling_root : Array
        Scalar typed Threefry key, ``fold_in(root, QMIX_SAMPLING_ROOT_TAG)``.
        Optimizer step n draws its sample with ``fold_in(sampling_root, n)``,
        n being the count before the step.
    completed_blocks : Array
        Int32 accepted nonempty blocks, warmup included.
    completed_updates : Array
        Int32 optimizer steps; always ``learning_blocks * epochs``.
    learning_blocks : Array
        Int32 published actor versions; equals ``history.current_update``.
    failed, failure_reason : Array
        Sticky bool and int32 reason: 0 none, 1 collection, 2 boundary, 3
        nonfinite rows, 4 illegal action, 5 rejected update, 6 rejected
        publication, 7 broken sample sequence.

    Notes
    -----
    Cumulative sampled counts are not stored here; each result reports its
    block's counts and callers keep exact host totals. Arrays stay dynamic
    under jit. A failed state must not be collected from or saved.
    """

    carry: TrainingCarry
    mixer_params: Tree
    target_q_params: Tree
    target_mixer_params: Tree
    opt_state: Tree
    replay: ReplayState
    sampling_root: Array
    completed_blocks: Array
    completed_updates: Array
    learning_blocks: Array
    failed: Array
    failure_reason: Array


class QMIXSampledCounts(NamedTuple):
    """Count what one block's optimizer steps used, summed over its epochs.

    Attributes
    ----------
    sampled_sequences, used_td_pairs, used_agent_utilities : Array
        Int32 sums of the per-step QMIXMetrics counts.
    exposure_by_stage : Array
        Int32 (17,) TD pairs by the left row's reset-time curriculum stage.
    exposure_by_source : Array
        Int32 (bank,) TD pairs by the left row's source-bank row.
    exposure_by_opponent : Array
        Int32 (21,) TD pairs by opponent: row 0 current self-play, row k the
        historical (or pinned, for slot 0) opponent in slot k-1.

    Notes
    -----
    Each exposure marginal sums to used_td_pairs. Repeated draws count again;
    these are used samples, not new experience. Warmup and absent blocks
    report zeros.
    """

    sampled_sequences: Array
    used_td_pairs: Array
    used_agent_utilities: Array
    exposure_by_stage: Array
    exposure_by_source: Array
    exposure_by_opponent: Array


class QMIXUpdateResult(NamedTuple):
    """Return one fixed-shape QMIX block result for every path.

    Attributes
    ----------
    accepted : Array
        Bool. True for an accepted nonempty block, warmup included.
    performed : Array
        Bool. True only when the block ran optimizer steps and published.
    failed, failure_reason : Array
        Mirror the returned state's sticky failure.
    metrics : QMIXMetrics
        Leaves with leading (epochs,); zeros unless performed.
    snapshot : SnapshotEvent
        History capture made by the publication; absent (-1) otherwise.
    summary : UpdateSummary
        Unrepeated experience sums of the block's real rows; also for warmup.
    sampled : QMIXSampledCounts
        What the optimizer steps used; zeros unless performed.

    Notes
    -----
    Empty input gives accepted, performed and failed all False and the state
    unchanged. All branches share one JAX structure.
    """

    accepted: Array
    performed: Array
    failed: Array
    failure_reason: Array
    metrics: QMIXMetrics
    snapshot: SnapshotEvent
    summary: UpdateSummary
    sampled: QMIXSampledCounts


def _bank_size(carry: TrainingCarry) -> int:
    """Return the static number of source-bank rows in the carried bank."""
    assert carry.tracking.source_configs is not None
    return int(jax.tree.leaves(carry.tracking.source_configs)[0].shape[0])


def _block_counts_possible(
    rounds: int, blocks: int, learning: int, *, rollout_length: int, minimum: int
) -> bool:
    """Say whether some sequence of accepted blocks reaches these counters.

    Parameters
    ----------
    rounds : int
        Real rounds collected (rows stored per game).
    blocks, learning : int
        Accepted nonempty blocks and learning blocks among them.
    rollout_length : int
        Largest block (T); every block holds 1 to T rounds.
    minimum : int
        min_buffer_size: a block learns exactly when the rows after its
        insertion reach it.

    Returns
    -------
    bool
        True when the first ``blocks - learning`` blocks can all stay below the
        minimum and the rest can all learn. The search tries every possible
        warmup row total, at most ``minimum`` values. Plain Python; no arrays.

    Examples
    --------
    With T=8 and minimum 32, 32 rounds in four blocks cannot all learn: four
    blocks of 32 rounds are eight rounds each, so only the fourth reaches 32.

    >>> _block_counts_possible(32, 4, 4, rollout_length=8, minimum=32)
    False
    >>> _block_counts_possible(32, 4, 1, rollout_length=8, minimum=32)
    True
    """
    if min(rounds, blocks, learning) < 0 or learning > blocks:
        return False
    warm, top = blocks - learning, rollout_length
    if learning == 0:
        return rounds < minimum and warm <= rounds <= warm * top
    low = max(warm, minimum - top)
    high = min(warm * top, minimum - 1)
    for warm_rows in range(low, high + 1):
        rest = rounds - warm_rows
        if max(1, minimum - warm_rows) + learning - 1 <= rest <= learning * top:
            return True
    return False


def _state_shapes(state: QMIXLearnerState, qmix: QMIXConfig) -> int:
    """Check static learner memory, counter, key and replay shapes; return B.

    Uses only static metadata and works under jit. Raises ValueError.
    """
    games = state.carry.state.episode_id.shape[0]
    _array(
        state.carry.memory.team_a,
        (games, TEAM_SLOTS, QMIX_HIDDEN_SIZE),
        jnp.float32,
        "Actor memory",
    )
    for name in (
        "completed_blocks",
        "completed_updates",
        "learning_blocks",
        "failure_reason",
    ):
        _array(getattr(state, name), (), jnp.int32, name)
    _array(state.failed, (), jnp.bool_, "failed")
    if (
        state.sampling_root.shape != ()
        or str(jax.random.key_impl(state.sampling_root)) != "threefry2x32"
    ):
        raise ValueError("QMIX sampling_root must be one typed Threefry key")
    _array(state.replay.current_index, (), jnp.int32, "replay current_index")
    _array(state.replay.is_full, (), jnp.bool_, "replay is_full")
    for leaf in jax.tree.leaves(state.replay.experience):
        if leaf.shape[:2] != (games, qmix.buffer_size):
            raise ValueError("QMIX replay must hold buffer_size rows per game")
    return games


def init_qmix_learner(
    *,
    schedule: TrainingSchedule,
    seed: int = 42,
    prepared: PreparedTrainingContent | None = None,
    shaping: bool = False,
    shaping_coefficient: float = 0.01,
    shaping_mode: str = "potential",
    metrics: str = "priority",
    recording: bool = False,
    pinned_opponent_share: float = 0.0,
    pinned_opponent: System | Policy | str | None = None,
    qmix: QMIXConfig = DEFAULT_QMIX_CONFIG,
) -> tuple[TrainingCollection, QMIXLearnerState]:
    """Initialize one untrained QMIX learner and its verified collection setup.

    Parameters
    ----------
    schedule : TrainingSchedule
        Checked exact budget and positive even batch from make_training_schedule.
    seed : int, default=42
        Python run seed, excluding bool. Collection keeps its own streams; the
        model key and sampling root come from version-1 QMIX tags.
    prepared : PreparedTrainingContent or None, default=None
        Existing verified content, or None to verify installed content now.
    shaping : bool, default=False
        Enable the chosen team reward adjustment; potential shaping uses
        qmix.gamma.
    shaping_coefficient : float, default=0.01
        Nonnegative finite weight, checked by the shaping owner.
    shaping_mode : {"potential", "score_delta"}, default="potential"
        Fixed shaping method, used when shaping is enabled.
    metrics : {"priority", "none"}, default="priority"
        Collection metrics mode.
    recording : bool, default=False
        Keep numerical recording starts for a later caller-owned writer.
    pinned_opponent_share, pinned_opponent
        Passed unchanged to init_training_collection, which documents them.
    qmix : QMIXConfig, default=DEFAULT_QMIX_CONFIG
        Static QMIX settings shared by the actor, replay and every update.

    Returns
    -------
    tuple[TrainingCollection, QMIXLearnerState]
        Stable host descriptor, with the QMIX exploration hook and physical
        state collection, and the numerical block-0 state: epsilon from the
        clock at round 0 (always exactly 1.0), targets equal to the
        online networks, empty zero replay and all counters zero.

    Raises
    ------
    TypeError, ValueError
        Setup types, the collected-block or whole-run count guards, verified
        content, or the collection and shaping contracts fail.

    Notes
    -----
    Host setup: content checks, allocation of the replay (950,016,000 bytes at
    B=32 and C=1000), network initialization and reset. No actor decision,
    optimizer step, writer or file write happens. Reuse the descriptor.
    """
    if not isinstance(cast(object, schedule), TrainingSchedule) or not isinstance(
        cast(object, qmix), QMIXConfig
    ):
        raise TypeError("QMIX setup requires TrainingSchedule and QMIXConfig")
    if type(seed) is not int:
        raise TypeError("Learner seed must be a Python integer")
    games = schedule.num_envs
    validate_qmix_batch_size(games, qmix)
    if int(schedule.arrays.total_rounds) * qmix.epochs > _MAX_COUNT:
        raise ValueError("QMIX optimizer counts for this budget exceed int32")
    root = jax.random.key(seed, impl="threefry2x32")
    initialized = initialize_qmix(
        jax.random.fold_in(root, QMIX_MODEL_INITIALIZATION_TAG), qmix
    )
    variables = QMIXActorVariables(
        initialized.online_q,
        epsilon_at(jnp.int32(0), games, qmix.eps_min, qmix.eps_decay),
    )
    actor = make_qmix_system(
        initialized.online_q,
        epsilon=float(variables.epsilon),
        input_scale=qmix.input_scale,
        spawn_frame=qmix.spawn_frame,
    )
    collection, carry = init_training_collection(
        actor,
        variables,
        schedule=schedule,
        seed=seed,
        prepared=prepared,
        shaping=shaping,
        shaping_mode=shaping_mode,
        discount=qmix.gamma,
        coefficient=shaping_coefficient,
        collect_training_state=True,
        metrics=metrics,
        recording=recording,
        pinned_opponent_share=pinned_opponent_share,
        pinned_opponent=pinned_opponent,
        actor_variables_at_step=QMIXExploration(qmix.eps_min, qmix.eps_decay, games),
    )
    state = QMIXLearnerState(
        carry,
        initialized.online_mixer,
        initialized.target_q,
        initialized.target_mixer,
        initialized.opt_state,
        _init_replay(collection, carry, qmix),
        jax.random.fold_in(root, QMIX_SAMPLING_ROOT_TAG),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.bool_(False),
        jnp.int32(LEARNER_ERROR_NONE),
    )
    _state_shapes(state, qmix)
    return collection, state


def _train_state(state: QMIXLearnerState) -> QMIXTrainState:
    """Borrow the single current actor tree for one temporary update view."""
    return QMIXTrainState(
        state.carry.history.current_variables.params,
        state.target_q_params,
        state.mixer_params,
        state.target_mixer_params,
        state.opt_state,
        state.completed_updates,
    )


def _rollout_shapes(
    state: QMIXLearnerState, rollout: TrainingRollout, qmix: QMIXConfig
) -> tuple[int, int]:
    """Check static QMIX rollout leaves; return (capacity T, games B).

    T may be 1 to rollout_length; each capacity compiles its own program.
    QMIX actors return no learning outputs and collection must keep the
    physical state. Raises ValueError or TypeError.
    """
    games = _state_shapes(state, qmix)
    rows = rollout.transitions
    if rows.valid.ndim != 2 or rows.valid.shape[1] != games:
        raise ValueError("QMIX rollout needs (T,B) rows for the learner batch")
    length = rows.valid.shape[0]
    if not 1 <= length <= qmix.rollout_length:
        raise ValueError("QMIX rollout capacity must be 1 to rollout_length")
    if rows.learning_outputs != ():
        raise TypeError("QMIX collection must return no learning outputs")
    if rows.training_state is None or rollout.final_training_state is None:
        raise ValueError("QMIX collection must enable collect_training_state")
    _array(
        rollout.initial_memory,
        (games, TEAM_SLOTS, QMIX_HIDDEN_SIZE),
        jnp.float32,
        "initial_memory",
    )
    _array(rollout.real_steps, (), jnp.int32, "real_steps")
    for name in ("valid", "episode_start", "ended"):
        _array(getattr(rows, name), (length, games), jnp.bool_, name)
    for name in ("active", "alive"):
        _array(getattr(rows, name), (length, games, TEAM_SLOTS), jnp.bool_, name)
    _array(rows.task_rewards, (length, games, TEAM_SLOTS), jnp.float32, "task_rewards")
    _array(rows.shaping_reward, (length, games), jnp.float32, "shaping_reward")
    _array(rows.learner_update, (length, games), jnp.int32, "learner_update")
    _array(rollout.final_ended, (games,), jnp.bool_, "final_ended")
    for head in rows.actions:
        _array(head, (length, games, 10), jnp.int32, "native action")
    return length, games


def _boundary_valid(
    state: QMIXLearnerState,
    collected: TrainingCarry,
    rollout: TrainingRollout,
    qmix: QMIXConfig,
) -> Array:
    """Check counters, replay cursor and the exact real prefix in one bool.

    Before this block, learning has started exactly when the stored rows reach
    the minimum; the write index and full flag follow the round count; the
    collected carry kept the same actor version; and the rows are an exact
    valid prefix of this learner version with matching memory and endings.
    """
    rows = rollout.transitions
    prefix = jnp.arange(rows.valid.shape[0])[:, None] < rollout.real_steps
    prior = state.carry
    rounds = prior.progress.rounds
    learning = state.learning_blocks
    refreshed = prior.history.last_refresh_rounds
    return (
        (learning >= 0)
        & (state.completed_blocks >= learning)
        & (state.completed_blocks < _MAX_COUNT)
        & (state.completed_updates == learning * qmix.epochs)
        & (prior.history.current_update == learning)
        & (refreshed == jnp.where(learning > 0, rounds, 0))
        & ((learning > 0) == (rounds >= qmix.min_buffer_size))
        & (state.replay.current_index == rounds % qmix.buffer_size)
        & (state.replay.is_full == (rounds >= qmix.buffer_size))
        & (collected.history.current_update == learning)
        & (collected.history.last_refresh_rounds == refreshed)
        & (rollout.real_steps >= 0)
        & (rollout.real_steps <= rows.valid.shape[0])
        & (collected.progress.rounds - rounds == rollout.real_steps)
        & jnp.all(rows.valid == prefix)
        & jnp.all(jnp.where(rows.valid, rows.learner_update == learning, True))
        & jnp.all(rollout.initial_memory == prior.memory.team_a)
        & jnp.all(rollout.final_ended == collected.state.done.done)
    )


def _behavior_valid(rollout: TrainingRollout) -> Array:
    """Check that every real Team A action is a legal categorical choice."""
    rows = rollout.transitions
    native = ActorAction(*(head[..., :TEAM_SLOTS] for head in rows.actions))
    indices = encode_actions(native)
    in_range = (indices >= 0) & (indices < NUM_ACTIONS)
    legal = jnp.take_along_axis(
        categorical_action_mask(rows.action_mask),
        jnp.clip(indices, 0, NUM_ACTIONS - 1)[..., None],
        axis=-1,
    )[..., 0]
    return jnp.all(jnp.where(rows.valid[..., None], in_range & legal, True))


def _sequence_valid(sample: QMIXReplayRow) -> Array:
    """Check a sample's rows follow one another in each game's real order.

    Within an episode the next row has the same episode ID and one more
    decision step; a new episode follows only an ended row and starts at
    decision step 0 with a new ID. All rows must be valid.
    """
    ended = sample.ended[:, :-1]
    same = (
        (sample.episode_id[:, 1:] == sample.episode_id[:, :-1])
        & (sample.decision_step[:, 1:] == sample.decision_step[:, :-1] + 1)
        & ~sample.episode_start[:, 1:]
    )
    new = (
        sample.episode_start[:, 1:]
        & (sample.decision_step[:, 1:] == 0)
        & (sample.episode_id[:, 1:] != sample.episode_id[:, :-1])
    )
    return jnp.all(sample.valid) & jnp.all(jnp.where(ended, new, same))


def _step_counts(
    sample: QMIXReplayRow, metrics: QMIXMetrics, bank: int
) -> QMIXSampledCounts:
    """Count one step's TD pairs by stage, source row and opponent row."""
    pair = (sample.valid[:, :-1] & sample.valid[:, 1:]).reshape(-1).astype(jnp.int32)

    def count(index: Array, size: int) -> Array:
        """Add each pair to the bin of its left row."""
        return jnp.zeros(size, jnp.int32).at[index[:, :-1].reshape(-1)].add(pair)

    return QMIXSampledCounts(
        metrics.sampled_sequences,
        metrics.used_td_pairs,
        metrics.used_agent_utilities,
        count(sample.episode_stage, _STAGES),
        count(sample.source_index, bank),
        count(sample.opponent_snapshot + 1, _OPPONENT_ROWS),
    )


def _epoch_sum(value: Array) -> Array:
    """Sum one per-epoch count over the leading epoch axis."""
    return jnp.sum(value, axis=0)


def _absent_counts(bank: int) -> QMIXSampledCounts:
    """Return zero sampled counts with the fixed shapes."""
    zero = jnp.int32(0)
    return QMIXSampledCounts(
        zero,
        zero,
        zero,
        jnp.zeros(_STAGES, jnp.int32),
        jnp.zeros(bank, jnp.int32),
        jnp.zeros(_OPPONENT_ROWS, jnp.int32),
    )


def _absent_metrics(epochs: int) -> QMIXMetrics:
    """Return zero (epochs,) metrics for blocks without optimizer steps."""
    floats = jnp.zeros(epochs, jnp.float32)
    counts = jnp.zeros(epochs, jnp.int32)
    return QMIXMetrics(
        floats, floats, floats, counts, counts, counts, jnp.zeros(epochs, jnp.bool_)
    )


def _absent_snapshot() -> SnapshotEvent:
    """Return the existing absent history-capture record."""
    return SnapshotEvent(
        jnp.bool_(False),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.zeros(20, jnp.bool_),
    )


def _absent_summary() -> UpdateSummary:
    """Return a zero experience summary for empty or rejected blocks."""
    return UpdateSummary(
        jnp.float32(0),
        jnp.float32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.zeros((_STAGES, 3), jnp.int32),
        jnp.zeros((_STAGES, 2), jnp.int32),
        jnp.zeros(_STAGES, jnp.int32),
        jnp.zeros(_STAGES, jnp.int32),
    )


def _absent_result(
    state: QMIXLearnerState, qmix: QMIXConfig, bank: int
) -> QMIXUpdateResult:
    """Build the not-accepted result, keeping the state's failure reason."""
    return QMIXUpdateResult(
        jnp.bool_(False),
        jnp.bool_(False),
        state.failed,
        state.failure_reason,
        _absent_metrics(qmix.epochs),
        _absent_snapshot(),
        _absent_summary(),
        _absent_counts(bank),
    )


def _reject(
    state: QMIXLearnerState, reason: Array, qmix: QMIXConfig, bank: int
) -> tuple[QMIXLearnerState, QMIXUpdateResult]:
    """Keep the previous boundary and make its first failure sticky."""
    failed = state._replace(
        failed=jnp.bool_(True),
        failure_reason=jnp.where(state.failed, state.failure_reason, reason),
    )
    return failed, _absent_result(failed, qmix, bank)


type _Outcome = tuple[QMIXLearnerState, QMIXUpdateResult, Array]
"""A branch result inside the update: state without replay, result, keep flag."""


def _without_replay(state: QMIXLearnerState) -> QMIXLearnerState:
    """Return state with its replay removed, for branch results inside the update."""
    return state._replace(replay=cast(Any, None))


def update_qmix_learner(
    state: QMIXLearnerState,
    collected: TrainingCarry,
    rollout: TrainingRollout,
    *,
    qmix: QMIXConfig = DEFAULT_QMIX_CONFIG,
    continuation: LearnerContinuation | None = None,
) -> tuple[QMIXLearnerState, QMIXUpdateResult]:
    """Accept one collected block: store it, and learn once replay is ready.

    Parameters
    ----------
    state : QMIXLearnerState
        Last accepted boundary before this block's collection. Keep it
        unchanged while collecting with ``state.carry``. A failed state stays
        failed.
    collected : TrainingCarry
        Successor returned by that collection call, with the same actor version.
    rollout : TrainingRollout
        Matching rollout of capacity 1 to rollout_length and a real prefix of
        any length up to that capacity. Each capacity compiles its own program.
    qmix : QMIXConfig, default=DEFAULT_QMIX_CONFIG
        Same static settings used at initialization; capture them in the jit
        wrapper.

    continuation : LearnerContinuation or None, default=None
        Checked static child rule for future rate and exploration changes.
        Replay, target, optimizer and sample clocks keep their saved counts.

    Returns
    -------
    tuple[QMIXLearnerState, QMIXUpdateResult]
        The real rows are appended to replay. If replay is not yet ready, the
        block is a warmup: accepted, no optimizer step, no key use, no target
        or version change; only the stored epsilon moves to the new clock. If
        ready, ``epochs`` fresh samples each drive one optimizer step with key
        ``fold_in(sampling_root, count before the step)``; then the new actor
        (with the epsilon of the new clock) is published once through
        ``refresh_opponents``. An empty block returns the state unchanged.

    Raises
    ------
    TypeError, ValueError
        Static rollout, memory, counter or batch shapes are invalid.

    Notes
    -----
    Any failure (collection error, boundary or counter mismatch, nonfinite
    rows, illegal action, nonfinite or incomplete update, broken sample order,
    rejected publication) rejects the whole block and keeps the previous
    numerical boundary with a sticky reason. Its recording may have
    progressed; the host must stop and recover from a checkpoint. The input
    replay is not donated, so rollback keeps it while the block's rows are
    inserted into a new replay. The rows are inserted once, before any
    branch; branches return the state without its replay plus a keep flag,
    and one select at the end chooses the inserted or the previous replay, so
    no branch level adds a replay copy. At B=32 and C=1000 XLA's CPU memory
    analysis gives 1.30 GB inputs, 1.13 GB outputs and 1.57 GB temporaries
    (2.92 GB before this layout). Pure JAX; wrap in jit with a fixed qmix.
    """
    qmix = continuation_config(qmix, continuation)
    length, games = _rollout_shapes(state, rollout, qmix)
    validate_qmix_batch_size(games, qmix)
    del length
    if collected.state.episode_id.shape != (games,):
        raise ValueError("Collected boundary must retain the learner batch")
    bank = _bank_size(state.carry)
    entry_reason = jnp.where(
        _failed(state.carry) | _failed(collected),
        LEARNER_ERROR_COLLECTION,
        jnp.where(
            _boundary_valid(state, collected, rollout, qmix),
            LEARNER_ERROR_NONE,
            LEARNER_ERROR_BOUNDARY,
        ),
    ).astype(jnp.int32)
    rounds = collected.progress.rounds
    epsilon = epsilon_at(rounds, games, qmix.eps_min, qmix.eps_decay)
    if continuation is not None and continuation.exploration is not None:
        epsilon = ContinuationExploration(continuation.exploration, games).rate(rounds)
    # Insert once, outside every branch. Branches return the state without its
    # replay plus whether to keep the inserted rows; one select at the end
    # chooses the replay, so XLA keeps no extra replay copy per branch level.
    replay = _insert_rollout(state.replay, rollout, qmix)

    def keep(pair: tuple[QMIXLearnerState, QMIXUpdateResult], new: bool) -> _Outcome:
        """Drop the replay from a branch result and say which replay to keep."""
        return _without_replay(pair[0]), pair[1], jnp.bool_(new)

    def nonempty(_: None) -> _Outcome:
        """Check the rows, then warm up or learn from the inserted replay."""
        rows_reason = jnp.where(
            ~_finite(rollout.transitions),
            LEARNER_ERROR_NONFINITE_BATCH,
            jnp.where(
                _behavior_valid(rollout), LEARNER_ERROR_NONE, LEARNER_ERROR_ACTION
            ),
        ).astype(jnp.int32)

        def proceed(_: None) -> _Outcome:
            """Choose warmup or learning; the rows were already inserted."""
            summary = _summary(rollout)

            def warmup(_: None) -> _Outcome:
                """Accept the rows without any optimizer or version change."""
                current = collected.history.current_variables
                history = collected.history._replace(
                    current_variables=current._replace(epsilon=epsilon)
                )
                successor = state._replace(
                    carry=collected._replace(history=history),
                    completed_blocks=state.completed_blocks + 1,
                )
                return keep(
                    (
                        successor,
                        QMIXUpdateResult(
                            jnp.bool_(True),
                            jnp.bool_(False),
                            jnp.bool_(False),
                            jnp.int32(LEARNER_ERROR_NONE),
                            _absent_metrics(qmix.epochs),
                            _absent_snapshot(),
                            summary,
                            _absent_counts(bank),
                        ),
                    ),
                    True,
                )

            def learn(_: None) -> _Outcome:
                """Run the sampled epochs and publish one finite candidate."""

                def epoch(
                    carry: tuple[QMIXTrainState, Array, Array], _unused: None
                ) -> tuple[
                    tuple[QMIXTrainState, Array, Array],
                    tuple[QMIXMetrics, QMIXSampledCounts],
                ]:
                    """Draw one fresh sample and take one optimizer step."""
                    train, bad_update, bad_order = carry
                    key = jax.random.fold_in(state.sampling_root, train.optimizer_steps)
                    sample = _sample_rows(replay, key, qmix, games)
                    candidate, metrics = update_qmix(
                        train, _expand_sample(sample, qmix), config=qmix
                    )
                    return (
                        candidate,
                        bad_update | ~metrics.finite,
                        bad_order | ~_sequence_valid(sample),
                    ), (metrics, _step_counts(sample, metrics, bank))

                (train, bad_update, bad_order), (metrics, counts) = jax.lax.scan(
                    epoch,
                    (_train_state(state), jnp.bool_(False), jnp.bool_(False)),
                    None,
                    length=qmix.epochs,
                )
                sampled = cast(QMIXSampledCounts, jax.tree.map(_epoch_sum, counts))
                complete = (
                    train.optimizer_steps == state.completed_updates + qmix.epochs
                )
                reason = jnp.where(
                    bad_order,
                    LEARNER_ERROR_SEQUENCE,
                    jnp.where(
                        bad_update | ~complete | ~_finite(train),
                        LEARNER_ERROR_NONFINITE_UPDATE,
                        LEARNER_ERROR_NONE,
                    ),
                ).astype(jnp.int32)

                def publish(_: None) -> _Outcome:
                    """Publish the new actor once; memory stays unchanged."""
                    index = state.learning_blocks + jnp.int32(1)
                    history, event = refresh_opponents(
                        collected.history,
                        QMIXActorVariables(train.online_q, epsilon),
                        completed_rounds=rounds,
                        update_index=index,
                        schedule=collected.schedule,
                    )
                    successor = QMIXLearnerState(
                        collected._replace(history=history),
                        train.online_mixer,
                        train.target_q,
                        train.target_mixer,
                        train.opt_state,
                        replay,
                        state.sampling_root,
                        state.completed_blocks + 1,
                        train.optimizer_steps,
                        index,
                        jnp.bool_(False),
                        jnp.int32(LEARNER_ERROR_NONE),
                    )
                    result = QMIXUpdateResult(
                        jnp.bool_(True),
                        jnp.bool_(True),
                        jnp.bool_(False),
                        jnp.int32(LEARNER_ERROR_NONE),
                        metrics,
                        event,
                        summary,
                        sampled,
                    )
                    return _cond(
                        history.error,
                        lambda _: keep(
                            _reject(
                                state, jnp.int32(LEARNER_ERROR_HISTORY), qmix, bank
                            ),
                            False,
                        ),
                        lambda _: keep((successor, result), True),
                    )

                return _cond(
                    reason == LEARNER_ERROR_NONE,
                    publish,
                    lambda _: keep(_reject(state, reason, qmix, bank), False),
                )

            return _cond(_replay_ready(replay, qmix), learn, warmup)

        return _cond(
            rows_reason == LEARNER_ERROR_NONE,
            proceed,
            lambda _: keep(_reject(state, rows_reason, qmix, bank), False),
        )

    def accepted(_: None) -> _Outcome:
        """Skip key and optimizer work for an empty prefix; keep the old replay."""
        return _cond(
            rollout.real_steps > 0,
            nonempty,
            lambda _: keep((state, _absent_result(state, qmix, bank)), False),
        )

    successor, result, use_new = _cond(
        state.failed | (entry_reason != LEARNER_ERROR_NONE),
        lambda _: keep(_reject(state, entry_reason, qmix, bank), False),
        accepted,
    )

    def choose(new: Array, old: Array) -> Array:
        """Keep one replay leaf's inserted rows or its pre-block value."""
        return jnp.where(use_new, new, old)

    chosen = cast(ReplayState, jax.tree.map(choose, replay, state.replay))
    return successor._replace(replay=chosen), result


@lru_cache(maxsize=16)
def _qmix_spec(qmix: QMIXConfig) -> QMIXTrainState:
    """Trace initialize_qmix once per config for shape-only leaf specs."""

    def build(key: Array) -> QMIXTrainState:
        """Initialize with this config; only shapes are kept."""
        return initialize_qmix(key, qmix)

    return cast(QMIXTrainState, jax.eval_shape(build, jax.random.key(0)))


_saved_boundary_valid = cast(Callable[[QMIXLearnerState], Array], jax.jit(_finite))


def _epsilon_matches(value: float, rounds: int, games: int, qmix: QMIXConfig) -> bool:
    """Compare a stored float32 rate with the float64 clock value.

    The rate must be within 2e-6 of epsilon_reference, and exactly eps_min
    once the integer clock has reached the decay end.
    """
    expected = epsilon_reference(rounds, games, qmix.eps_min, qmix.eps_decay)
    plateau = games * min(rounds, -(-qmix.eps_decay // games)) >= qmix.eps_decay
    if plateau:
        return value == float(np.float32(qmix.eps_min))
    return abs(value - expected) <= 2e-6


def validate_qmix_learner(
    collection: TrainingCollection,
    state: QMIXLearnerState,
    *,
    qmix: QMIXConfig = DEFAULT_QMIX_CONFIG,
    recheck_installed_content: bool = True,
) -> None:
    """Validate a complete accepted QMIX boundary before any recording recovery.

    Parameters
    ----------
    collection : TrainingCollection
        Matching host descriptor with the QMIX exploration hook.
    state : QMIXLearnerState
        Fully restored numerical boundary, including block 0.
    qmix : QMIXConfig, default=DEFAULT_QMIX_CONFIG
        Original static settings.
    recheck_installed_content : bool, default=True
        Python bool. True reruns the installed-content preflight, as restore
        requires; saves pass False to reuse the verified descriptor.

    Raises
    ------
    TypeError, ValueError
        The descriptor, shapes, content, keys, counters, replay cursor,
        epsilon values, target timing, finite values or optimizer count do not
        describe a reachable accepted boundary. A failed state is never valid.

    Notes
    -----
    Host-only. Counters must satisfy ``completed_updates == learning_blocks *
    epochs``, ``history.current_update == learning_blocks``, the last refresh
    at the current rounds once learning has started (0 before), and the
    readiness-aware block rule (``_block_counts_possible``). The replay index
    must be ``rounds % buffer_size`` and full exactly once rounds reach it.
    Every stored epsilon (current and each occupied history slot) must match
    the clock at its round count. A child reads its declared future rule from
    collection.schedule.continuation['learner']; inherited slots keep the
    verified parent rates. With hard updates, targets must equal the
    online networks at count 0 and when a block's last step was a copy step.
    Because a block takes epochs steps, the second case needs a copy step to
    land at the end of a block; with the defaults (4 epochs, period 200) it
    never does, so after the first block wrong targets pass this check. The
    replay's contents are not inspected either; a corrupted but finite replay
    is caught by the sample-order check when it is first sampled. This check
    reads the whole state once on the device; it changes nothing.
    """
    if type(recheck_installed_content) is not bool:
        raise TypeError("recheck_installed_content must be a Python bool")
    if not isinstance(cast(object, qmix), QMIXConfig):
        raise TypeError("qmix must be a QMIXConfig")
    continuation = schedule_continuation(collection.schedule)
    qmix = continuation_config(qmix, continuation)
    games = _state_shapes(state, qmix)
    validate_qmix_batch_size(games, qmix)
    hook = collection.actor_variables_at_step
    expected_hook = (
        ContinuationExploration(continuation.exploration, games)
        if continuation is not None and continuation.exploration is not None
        else QMIXExploration(qmix.eps_min, qmix.eps_decay, games)
    )
    if (
        not collection.collect_training_state
        or hook is None
        or hook.identity != expected_hook.identity
    ):
        raise ValueError("Learner descriptor does not match these QMIX settings")
    if bool(state.failed) or int(state.failure_reason) != LEARNER_ERROR_NONE:
        raise ValueError("A failed learner boundary cannot be restored")
    _validate_training_continuation(
        collection, state.carry, recheck_installed_content=recheck_installed_content
    )
    rounds = int(state.carry.progress.rounds)
    blocks = int(state.completed_blocks)
    learning = int(state.learning_blocks)
    updates = int(state.completed_updates)
    history = state.carry.history
    if (
        updates != learning * qmix.epochs
        or int(history.current_update) != learning
        or int(history.last_refresh_rounds) != (rounds if learning else 0)
        or not _block_counts_possible(
            rounds,
            blocks,
            learning,
            rollout_length=qmix.rollout_length,
            minimum=qmix.min_buffer_size,
        )
    ):
        raise ValueError("QMIX block, update and collection counts disagree")
    expected_key = jax.random.fold_in(state.carry.root_key, QMIX_SAMPLING_ROOT_TAG)
    if not np.array_equal(
        jax.random.key_data(state.sampling_root), jax.random.key_data(expected_key)
    ):
        raise ValueError("QMIX sampling key does not match the run root")
    if int(state.replay.current_index) != rounds % qmix.buffer_size or bool(
        state.replay.is_full
    ) != (rounds >= qmix.buffer_size):
        raise ValueError("QMIX replay cursor disagrees with collected rounds")

    def empty_replay(values: TrainingCarry) -> ReplayState:
        """Trace the replay layout this collection and config would allocate."""
        return _init_replay(collection, values, qmix)

    spec = jax.eval_shape(empty_replay, state.carry)
    if jax.tree.structure(state.replay) != jax.tree.structure(spec) or any(
        a.shape != b.shape or a.dtype != b.dtype
        for a, b in zip(
            jax.tree.leaves(state.replay), jax.tree.leaves(spec), strict=True
        )
    ):
        raise ValueError("QMIX replay layout is incompatible")
    numerical = _train_state(state)
    template = _qmix_spec(qmix)
    if jax.tree.structure(numerical) != jax.tree.structure(template):
        raise ValueError("QMIX network or optimizer structure is incompatible")
    for actual, expected in zip(
        jax.tree.leaves(numerical), jax.tree.leaves(template), strict=True
    ):
        _array(actual, expected.shape, expected.dtype, "QMIX network/optimizer leaf")
    for leaf in cast(list[Array], jax.tree.leaves(state.opt_state)):
        if jnp.issubdtype(leaf.dtype, jnp.integer) and int(leaf) != updates:
            raise ValueError("QMIX optimizer count differs from completed updates")
    if not bool(_saved_boundary_valid(state)):
        raise ValueError("QMIX learner contains nonfinite values")

    def epsilon_matches(value: float, at_round: int) -> bool:
        """Use the active future curve, or the unchanged original QMIX curve."""
        if continuation is not None and continuation.exploration is not None:
            expected = continuation.exploration.at(at_round * games)
            return abs(value - expected) <= 2e-6
        return _epsilon_matches(value, at_round, games, qmix)

    if not epsilon_matches(float(history.current_variables.epsilon), rounds):
        raise ValueError("Current exploration rate disagrees with the clock")
    captured = np.asarray(history.captured_rounds)
    rates = np.asarray(history.historical_variables.epsilon)
    for slot in range(int(history.count)):
        if continuation is not None and slot < len(continuation.frozen_epsilon):
            matches = float(rates[slot]) == continuation.frozen_epsilon[slot]
        else:
            matches = epsilon_matches(float(rates[slot]), int(captured[slot]))
        if not matches:
            raise ValueError("A frozen opponent's exploration rate disagrees")
    copied = updates == 0 or (
        qmix.hard_update and (updates - 1) % qmix.update_period == 0
    )
    if copied:
        pairs = (
            (numerical.target_q, numerical.online_q),
            (numerical.target_mixer, numerical.online_mixer),
        )
        for target, online in pairs:
            for a, b in zip(
                jax.tree.leaves(target), jax.tree.leaves(online), strict=True
            ):
                if not np.array_equal(np.asarray(a), np.asarray(b)):
                    raise ValueError(
                        "QMIX targets differ from online at a copy boundary"
                    )
