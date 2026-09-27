"""Join shared training collection to recurrent PQN-VDN through a recent window.

This module owns PQN-VDN's recent data and learner lifecycle. Recent data: the
compact Team A row stored for every real decision, the last ``memory_window``
real rows kept per game with the memory each actor held before acting, the
learning window that joins those rows to a new block, and the expansion of the
selected games into the network-frame ``PQNBatch`` that
``baselines.pqn.update_pqn`` reads. Lifecycle: ``init_pqn_learner`` builds the
collection and the untrained network, ``update_pqn_learner`` accepts one
collected block (an initial random chunk only updates the kept rows; a
learning block runs ``epochs`` passes of ``num_minibatches`` optimizer steps
over the window and publishes the actor once) and ``validate_pqn_learner``
checks a saved boundary. Collection, curriculum, shaping, self-play history,
pinned opponents and recording stay with their existing owners. This module
imports nothing from the QMIX learner and never loads Flashbax.

Stored rows hold only permitted Team A material: the five Team A observation
rows, their 5x5 source permissions, Team A masks and chosen actions (world
frame), the team task reward and shaping reward separately, lifecycle flags,
and eight int32 identity fields; no physical state and no Q values. Rebuilding
the actor inputs of a minibatch uses the same builder as live action
selection, so the learner sees exactly what the actor saw. Every numerical
helper is pure JAX for jit and scan; the checks in ``validate_pqn_learner``
are host-only. Requires the training extra.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
from marl_battlegrounds.baselines.ppo import _actor_groups
from marl_battlegrounds.baselines.pqn import (
    DEFAULT_PQN_CONFIG,
    PQN_HIDDEN_SIZE,
    PQNActorVariables,
    PQNBatch,
    PQNConfig,
    PQNLearningOutputs,
    PQNMetrics,
    PQNTrainState,
    _nonnegative_variances,
    epsilon_at,
    initialize_pqn,
    make_pqn_system,
    pqn_epsilon_reference,
    pqn_planned_learning_blocks,
    update_pqn,
    validate_pqn_batch_size,
)
from marl_battlegrounds.baselines.qmix import TEAM_SLOTS, team_task_reward
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
    LearnerContinuation,
    continuation_boundary,
    continuation_config,
    reward_refill_end,
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
    learner_memory,
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
from marl_battlegrounds.training.opponents import (
    OpponentHistory,
    SnapshotEvent,
    opponent_counter_row,
    publication_valid,
    refresh_opponents,
)
from marl_battlegrounds.training.shaping import RewardFunction

type Tree = Any

PQN_MODEL_INITIALIZATION_TAG = 0x50514E49
"""Tag folded into the run root to derive the one PQN model key ("PQNI")."""
PQN_SHUFFLE_ROOT_TAG = 0x50514E53
"""Tag folded into the run root to derive the minibatch shuffle root ("PQNS")."""
LEARNER_ERROR_SEQUENCE = 7
"""Failure reason: a learning window broke episode or decision-step order.

The same number and meaning as the QMIX learner's code. It is kept here
because the QMIX learner module imports Flashbax, which PQN never loads."""
_MAX_COUNT = int(np.iinfo(np.int32).max)
_STAGES = 17
_SNAPSHOT_SLOTS = 20


class PQNRow(NamedTuple):
    """Hold one game's compact Team A facts from one real decision.

    Leaves carry the leading axes of their source: (B,) for one collected
    round, (T,B) for a block and (H,B) or (C,B) in kept rows and windows. The
    shapes below are per game row.

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
        Float32 native team reward plus the mean custom adjustment over owned
        active slots. The native team reward enters once, not once per actor.
    shaping_reward : Array
        Float32 team shaping reward (0 when shaping is off).
    learner_active : Array or None, default=None
        Bool (5,) slots owned by this learner. None owns all active slots.
        Stored only when partners are enabled; adds 5 bytes per game row.
    active, alive : Array
        Bool (5,) configured and living Team A slots before the action.
    episode_start, ended, valid : Array
        Bool. First decision of an episode; this decision ended the episode (a
        real ending, including a horizon draw, not a collection cutoff); real
        decision. Padding rows are invalid.
    episode_id, decision_step, requested_stage, episode_stage, source_index,
    learner_update, opponent_update, opponent_snapshot : Array
        Int32 identities copied from the transition, for order checks and
        exposure counts. They never enter a network.

    Notes
    -----
    One row is 26,008 bytes with the current encoder: QMIX's compact row
    without its 920-value physical state. No Team B row, opposite-team
    permission, Q value or expanded feature is stored. Padding rows have
    identities -1, neutral-only masks and finite zeros. Layout version:
    ``baselines.pqn.PQN_RECENT_WINDOW_SCHEMA_VERSION``.
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
    episode_id: Array
    decision_step: Array
    requested_stage: Array
    episode_stage: Array
    source_index: Array
    learner_update: Array
    opponent_update: Array
    opponent_snapshot: Array
    learner_active: Array | None = None


class PQNRecent(NamedTuple):
    """Keep the last real rows of every game for the next learning window.

    Attributes
    ----------
    rows : PQNRow
        Leaves with leading (H,B): the most recent real rows in time order,
        then neutral padding when fewer than H real rows exist.
    pre_memory : Array
        Float32 (H,B,5,512) memory each actor held immediately before acting on
        that row, exactly as stored at collection time. It is never recomputed
        with newer weights.
    size : Array
        Int32 number of real rows at the front: ``min(H, rounds)``.

    Notes
    -----
    4,639,744 bytes plus the size scalar at B=32 and H=4. Only this suffix is
    kept between blocks; the full learning window is rebuilt from it and the
    new block.
    """

    rows: PQNRow
    pre_memory: Array
    size: Array


class PQNUsedCounts(NamedTuple):
    """Count what one learning block's optimizer steps used, over all epochs.

    Attributes
    ----------
    used_sequences : Array
        Int32 game sequences used: epochs times games for a learned block.
    used_td_pairs : Array
        Int32 eligible adjacent TD pairs over every epoch's use:
        ``E * B * (H + m - 1)`` for a block with m new rounds.
    used_agent_utilities : Array
        Int32 configured Team A slots on the left rows of those pairs (dead
        slots included), between 1 and 5 per pair.
    used_prefix_td_pairs : Array
        Int32 pairs whose left row is one of the H kept rows: ``E * B * H``.
        The newest kept row may be learned for the first time, so these are
        not all repeats.
    exposure_by_stage : Array
        Int32 (17,) pairs by the left row's reset-time curriculum stage.
    exposure_by_source : Array
        Int32 (bank,) pairs by the left row's source-bank row.
    exposure_by_opponent : Array
        Int32 (C+2,) pairs by stable opponent: row 0 self, row 1 pin,
        then capture_id+2. C is the configured capture-count bound.

    Notes
    -----
    Each exposure marginal sums to used_td_pairs. These are block deltas of
    used rows, not new experience or random replay samples. Initial chunks and
    absent blocks report zeros. Low-level callers add the deltas themselves;
    the runner keeps exact Python-int totals.
    """

    used_sequences: Array
    used_td_pairs: Array
    used_agent_utilities: Array
    used_prefix_td_pairs: Array
    exposure_by_stage: Array
    exposure_by_source: Array
    exposure_by_opponent: Array


class PQNLearnerState(NamedTuple):
    """Keep one complete numerical PQN boundary with one current actor tree.

    Attributes
    ----------
    carry : TrainingCarry
        Complete collection state. ``carry.history.current_variables`` is the
        only current actor: PQNActorVariables with the network, its BatchNorm
        statistics and the stored exploration rate.
    opt_state : PyTree
        Clipped RAdam state over the network parameters only.
    recent : PQNRecent
        The last H real rows per game and their stored memories.
    shuffle_root : Array
        Scalar typed Threefry key, ``fold_in(root, PQN_SHUFFLE_ROOT_TAG)``.
        Epoch e of learning block k shuffles games with
        ``fold_in(fold_in(shuffle_root, k), e)``.
    completed_blocks : Array
        Int32 accepted nonempty collection blocks, initial chunks included.
    completed_updates : Array
        Int32 optimizer steps; always ``learning_blocks * epochs *
        num_minibatches``.
    learning_blocks : Array
        Int32 published actor versions; equals ``history.current_update``.
    failed, failure_reason : Array
        Sticky bool and int32 reason: 0 none, 1 collection, 2 boundary, 3
        nonfinite rows, 4 illegal action, 5 rejected update, 6 rejected
        publication, 7 broken window order.

    Notes
    -----
    No target network, replay, critic, second network copy or cumulative
    sample totals. Arrays stay dynamic under jit. A failed state must not be
    collected from or saved.
    """

    carry: TrainingCarry
    opt_state: Tree
    recent: PQNRecent
    shuffle_root: Array
    completed_blocks: Array
    completed_updates: Array
    learning_blocks: Array
    failed: Array
    failure_reason: Array


class PQNUpdateResult(NamedTuple):
    """Return one fixed-shape PQN block result for every path.

    Attributes
    ----------
    accepted : Array
        Bool. True for an accepted nonempty block, initial chunks included.
    performed : Array
        Bool. True only when the block ran its optimizer steps and published.
    failed, failure_reason : Array
        Mirror the returned state's sticky failure.
    reward, reward_identity : callable and dict or None, defaults=None
        Optional pure training reward and its saved identity. Forwarded to
        init_training_collection; None skips callback work and extra storage.
    metrics : PQNMetrics
        Leaves with leading (epochs, num_minibatches). Zeros unless a learning
        block ran; a rejected learning block keeps its attempted values.
    snapshot : SnapshotEvent
        History capture made by the publication; absent (-1) otherwise.
    summary : UpdateSummary
        Unrepeated experience sums of the block's real rows, for initial
        chunks, learned blocks and rejected nonempty blocks alike.
    used : PQNUsedCounts
        What the optimizer steps used; zeros unless performed.

    Notes
    -----
    Empty input gives accepted, performed and failed all False and the state
    unchanged. A rejected block keeps its generated summary and attempted
    metrics as diagnostics; the state returns to the previous boundary. All
    branches share one JAX structure.
    """

    accepted: Array
    performed: Array
    failed: Array
    failure_reason: Array
    metrics: PQNMetrics
    snapshot: SnapshotEvent
    summary: UpdateSummary
    used: PQNUsedCounts


def _pqn_rows(rows: TrainingTransition) -> tuple[PQNRow, Array]:
    """Pack transition rows with leading axes (..., B) into compact PQN rows.

    rows must come from a PQN collection: no physical state and
    ``PQNLearningOutputs`` learning outputs. Team A is routing rows 0-4 of
    every observation leaf and permission axis. Returns ``(rows, pre_memory)``
    with pre_memory float32 (..., B, 5, 512). Raises ValueError or TypeError
    for another collection. Pure JAX; it copies only the selected Team A
    slices.
    """
    if rows.training_state is not None:
        raise ValueError("PQN collection must not keep physical training state")
    outputs = rows.learning_outputs
    if not isinstance(outputs, PQNLearningOutputs):
        raise TypeError("PQN collection must return PQNLearningOutputs")
    observer_axis = rows.valid.ndim

    def team_a(value: Array) -> Array:
        """Keep the five Team A observer rows of one observation leaf."""
        return jax.lax.slice_in_dim(value, 0, TEAM_SLOTS, axis=observer_axis)

    native = ActorAction(*(head[..., :TEAM_SLOTS] for head in rows.actions))
    packed = PQNRow(
        jax.tree.map(team_a, rows.observations.observation),
        rows.observations.source_availability[..., :TEAM_SLOTS, :TEAM_SLOTS],
        rows.action_mask,
        encode_actions(native),
        team_task_reward(
            rows.task_rewards
            if rows.custom_rewards is None
            else rows.task_rewards + rows.custom_rewards,
            rows.active
            if rows.learner_active is None
            else rows.active & rows.learner_active,
        ),
        rows.shaping_reward,
        rows.active,
        rows.alive,
        rows.episode_start,
        rows.ended,
        rows.valid,
        rows.episode_id,
        rows.decision_step,
        rows.requested_stage,
        rows.episode_stage,
        rows.source_index,
        rows.learner_update,
        rows.opponent_update,
        rows.opponent_snapshot,
        rows.learner_active,
    )
    return packed, outputs.pre_memory


def _empty_recent(
    collection: TrainingCollection, carry: TrainingCarry, pqn: PQNConfig
) -> PQNRecent:
    """Return the size-0 kept rows: H neutral padding rows and zero memories.

    Padding comes from collection's own padding row (identities -1,
    neutral-only masks, invalid, finite zeros), repeated H times. Pure JAX.
    """
    padding, memory = _pqn_rows(_padding(collection, carry))
    window = pqn.memory_window

    def repeat(value: Array) -> Array:
        """Repeat one (B,...) padding leaf H times along a new leading axis."""
        return jnp.broadcast_to(value, (window, *value.shape))

    return PQNRecent(
        cast(PQNRow, jax.tree.map(repeat, padding)),
        repeat(memory),
        jnp.int32(0),
    )


def _learning_window(
    recent: PQNRecent, rows: PQNRow, memory: Array, real_steps: Array
) -> tuple[PQNRow, Array, Array]:
    """Join the kept rows and a new block into one gap-free time window.

    Parameters
    ----------
    recent : PQNRecent
        Kept rows (H,B) with ``size`` real rows at the front.
    rows : PQNRow
        The new block's compact rows (capacity,B) with ``real_steps`` real rows
        at the front.
    memory : Array
        Float32 (capacity,B,5,512) stored pre-action memories of the new rows.
    real_steps : Array
        Int32 real rounds in the new block.

    Returns
    -------
    tuple[PQNRow, Array, Array]
        Window rows and memories with leading (H + capacity, B) and the int32
        count of real rows ``size + real_steps``. Real kept rows come first,
        then the new real rows, then every padding row, so no padding row sits
        between real rows and padding never becomes a successor.

    Notes
    -----
    A stable reordering of the joined rows; no row is copied twice and no old
    memory is recomputed. Pure JAX.
    """
    window = recent.pre_memory.shape[0]
    capacity = memory.shape[0]
    index = jnp.arange(window + capacity)
    kept = index < window
    group = jnp.where(
        kept,
        jnp.where(index < recent.size, 0, 2),
        jnp.where(index - window < real_steps, 1, 3),
    )
    order = jnp.argsort(group * (window + capacity) + index, stable=True)

    def join(old: Array, new: Array) -> Array:
        """Concatenate one leaf and apply the window order."""
        return jnp.take(jnp.concatenate([old, new], axis=0), order, axis=0)

    joined = cast(PQNRow, jax.tree.map(join, recent.rows, rows))
    return joined, join(recent.pre_memory, memory), recent.size + real_steps


def _retain_recent(
    rows: PQNRow, memory: Array, real_rows: Array, pqn: PQNConfig
) -> PQNRecent:
    """Keep the last H real rows of a window and their original memories.

    rows and memory have leading (C,B) with ``real_rows`` real rows first.
    Returns the rows ``max(real_rows - H, 0)`` onward, H of them; when fewer
    than H real rows exist the tail is the window's own padding. size is
    ``min(H, real_rows)``. Pure JAX.
    """
    window = pqn.memory_window
    start = jnp.maximum(real_rows - window, 0)
    picks = start + jnp.arange(window)

    def take(value: Array) -> Array:
        """Select the kept rows of one leaf."""
        return jnp.take(value, picks, axis=0)

    return PQNRecent(
        cast(PQNRow, jax.tree.map(take, rows)),
        take(memory),
        jnp.minimum(real_rows, window).astype(jnp.int32),
    )


def _expand_minibatch(rows: PQNRow, initial_memory: Array, pqn: PQNConfig) -> PQNBatch:
    """Rebuild network-frame inputs for the selected games of a window.

    rows has leading (C,D) for D selected games; initial_memory is float32
    (D,5,512), the oldest row's stored memory. The five stored Team A rows
    and their permissions are padded back to ten observer rows with zero
    Team B rows and a zero 10x10 permission matrix, then passed through
    ``build_team_actor_input`` for Team A, the same builder as live action
    selection. In the "left" frame, flagged rows' views, masks and stored
    world actions are reflected. Actor groups come from raw self classes or
    physical slots before encoding. They are temporary batch fields, not stored
    data. Rewards are task plus shaping. Returns float32
    (C,D,5,5165) features among the PQNBatch leaves: 27,271,200 bytes for
    C=132 and D=2. Pure JAX.
    """
    steps, games = rows.valid.shape

    def flatten(value: Array) -> Array:
        """Merge the (C,D) axes into one row axis."""
        return value.reshape(steps * games, *value.shape[2:])

    flat = cast(PQNRow, jax.tree.map(flatten, rows))

    def ten_rows(value: Array) -> Array:
        """Restore the routing axis to ten rows with zero Team B rows."""
        return jnp.concatenate((value, jnp.zeros_like(value)), axis=1)

    permissions = (
        jnp.zeros((steps * games, 2 * TEAM_SLOTS, 2 * TEAM_SLOTS), jnp.bool_)
        .at[:, :TEAM_SLOTS, :TEAM_SLOTS]
        .set(flat.source_availability)
    )
    observations = Observations(jax.tree.map(ten_rows, flat.observation), permissions)
    actors = jax.vmap(build_team_actor_input, in_axes=(0, None))(observations, 0)
    groups = (
        None
        if pqn.parameter_sharing == "all"
        else _actor_groups(actors, pqn.parameter_sharing)
    )
    mask, actions = flat.action_mask, flat.actions
    if pqn.spawn_frame != "world":
        flag = spawn_frame_flag(actors, pqn.spawn_frame)
        actors, mask = mirror_team_view(
            actors,
            mask,
            flag,
            obstacle_partners=team_obstacle_partners(actors),
        )
        # Mirroring is its own inverse, so it also maps world to network frame.
        actions = mirror_action_indices(actions, flag)
    features = encode_actor_inputs(actors)

    def restore(value: Array) -> Array:
        """Restore the (C,D) axes."""
        return value.reshape(steps, games, *value.shape[1:])

    return PQNBatch(
        restore(features),
        restore(categorical_action_mask(mask)),
        restore(actions),
        rows.task_reward + rows.shaping_reward,
        rows.episode_start,
        rows.ended,
        rows.valid,
        rows.active,
        initial_memory,
        actor_group=None if groups is None else restore(groups),
        learner_active=rows.learner_active,
    )


def _bank_size(carry: TrainingCarry) -> int:
    """Return the static number of source-bank rows in the carried bank."""
    assert carry.tracking.source_configs is not None
    return int(jax.tree.leaves(carry.tracking.source_configs)[0].shape[0])


def _state_shapes(state: PQNLearnerState, pqn: PQNConfig) -> int:
    """Check static learner memory, counter, key and kept-row shapes; return B.

    Uses only static metadata and works under jit. Raises ValueError.
    """
    games = state.carry.state.episode_id.shape[0]
    _array(
        learner_memory(state.carry),
        (games, TEAM_SLOTS, PQN_HIDDEN_SIZE),
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
        state.shuffle_root.shape != ()
        or not jnp.issubdtype(state.shuffle_root.dtype, jax.dtypes.prng_key)
        or str(jax.random.key_impl(state.shuffle_root)) != "threefry2x32"
    ):
        raise ValueError("PQN shuffle_root must be one typed Threefry key")
    window = pqn.memory_window
    _array(
        state.recent.pre_memory,
        (window, games, TEAM_SLOTS, PQN_HIDDEN_SIZE),
        jnp.float32,
        "kept pre_memory",
    )
    _array(state.recent.size, (), jnp.int32, "kept size")
    if state.recent.rows.learner_active is not None:
        _array(
            state.recent.rows.learner_active,
            (window, games, TEAM_SLOTS),
            jnp.bool_,
            "stored learner_active",
        )
    for leaf in jax.tree.leaves(state.recent.rows):
        if leaf.shape[:2] != (window, games):
            raise ValueError("PQN kept rows must hold memory_window rows per game")
    return games


def init_pqn_learner(
    *,
    schedule: TrainingSchedule,
    seed: int = 42,
    initial_actor: Tree | None = None,
    prepared: PreparedTrainingContent | None = None,
    shaping: bool = False,
    shaping_coefficient: float = 0.01,
    shaping_mode: str = "potential",
    reward: RewardFunction | None = None,
    reward_identity: dict[str, object] | None = None,
    metrics: str = "priority",
    recording: bool = False,
    pinned_opponent_share: float = 0.0,
    pinned_opponent: System | Policy | str | None = None,
    keep_past: int = 20,
    history_capture_capacity: int = 20,
    minimum_capture_rounds: int = 1,
    capture_interval_rounds: int = 0,
    opponent_population: Mapping[str, System | Policy | str] | None = None,
    opponent_selection: Mapping[str, float] | Sequence[str] | None = None,
    learner_slots: tuple[int, ...] | None = None,
    partner_population: Mapping[str, System | Policy | str] | None = None,
    partner_selection: Mapping[str, float] | Sequence[str] | None = None,
    pqn: PQNConfig = DEFAULT_PQN_CONFIG,
) -> tuple[TrainingCollection, PQNLearnerState]:
    """Initialize one untrained PQN-VDN learner and its verified collection setup.

    Parameters
    ----------
    schedule : TrainingSchedule
        Checked exact budget and positive even batch from make_training_schedule.
        Its total rounds R must be at least ``memory_window + rollout_length +
        1`` and fit the int32 guards of ``pqn_planned_learning_blocks``.
    seed : int, default=42
        Python run seed, excluding bool. Collection keeps its own streams; the
        model key and shuffle root come from version-1 PQN tags.
    initial_actor : numerical actor tree or None, default=None
        Compatible actor weights copied into a fresh learner before history is
        initialized. PPO takes actor parameters; QMIX takes online Q parameters
        (also used for its targets); PQN takes parameters and running statistics.
        Critic, mixer, optimizer, memory and counters start fresh. Initial
        exploration and collection keep this method's normal rules.
    prepared : PreparedTrainingContent or None, default=None
        Existing verified content, or None to verify installed content now.
    shaping : bool, default=False
        Enable the chosen team reward adjustment; potential shaping uses
        pqn.gamma.
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
    pqn : PQNConfig, default=DEFAULT_PQN_CONFIG
        Static PQN settings shared by the actor, kept rows and every update.

    keep_past, history_capture_capacity : int, defaults=20, 20
        Rolling copy count and maximum stable capture count. See
        init_training_collection for storage and counter meanings.
    minimum_capture_rounds, capture_interval_rounds : int, defaults=1, 0
        Minimum actual capture gap and optional recurring gap. The public runner
        checks the maximum game horizon and passes it as the minimum.

    opponent_population, opponent_selection : mapping, sequence or None
        Named frozen Systems and future-game shares or exact repeating order.
        Forwarded to init_training_collection; None keeps the existing recipe.
        Open member resources around setup and training, as public train does.

    learner_slots : tuple[int, ...] or None, default=None
        Physical Team A slots trained by this learner. None owns all slots.
    partner_population, partner_selection : mapping, sequence or None
        Frozen named partners and their future-game shares or repeating order.
        Forwarded to init_training_collection with learner_slots.

    Returns
    -------
    tuple[TrainingCollection, PQNLearnerState]
        Stable host descriptor (no physical state, no step hook) and the
        numerical zero-experience boundary: exploration rate exactly 1.0 for
        initial random collection, empty kept rows and all counters zero.
        Random initialization starts with running mean 0 and variance 1;
        a warm start keeps the imported running statistics.

    Raises
    ------
    TypeError, ValueError
        Setup types, the batch or budget guards (including R < H + T + 1),
        verified content, or the collection and shaping contracts fail. An
        initial actor has incompatible arrays, nonfinite values or negative
        running variance. Batch and budget checks run before model allocation;
        imported actor checks run before collection setup.

    Notes
    -----
    Host setup: content checks, network initialization, reset and the kept
    rows (4,639,744 bytes at B=32, H=4). No actor decision, collection,
    optimizer step, writer or file write happens; the caller makes every real
    collection call, starting with the initial random chunks.
    """
    if not isinstance(cast(object, schedule), TrainingSchedule) or not isinstance(
        cast(object, pqn), PQNConfig
    ):
        raise TypeError("PQN setup requires TrainingSchedule and PQNConfig")
    if type(seed) is not int:
        raise TypeError("Learner seed must be a Python integer")
    games = schedule.num_envs
    validate_pqn_batch_size(games, pqn)
    planned = pqn_planned_learning_blocks(int(schedule.arrays.total_rounds), pqn)
    root = jax.random.key(seed, impl="threefry2x32")
    initialized = initialize_pqn(
        jax.random.fold_in(root, PQN_MODEL_INITIALIZATION_TAG),
        pqn=pqn,
        planned_learning_blocks=planned,
    )
    if initial_actor is not None:
        from marl_battlegrounds.training.checkpoints import initial_actor_variables

        initialized = initialized._replace(
            network=initial_actor_variables(initial_actor, initialized.network)
        )
        if not _nonnegative_variances(initialized.network.batch_stats):
            raise ValueError("Initial actor running variance must be nonnegative")
    variables = PQNActorVariables(initialized.network, jnp.float32(1.0))
    actor = make_pqn_system(
        initialized.network,
        epsilon=1.0,
        input_scale=pqn.input_scale,
        spawn_frame=pqn.spawn_frame,
        parameter_sharing=pqn.parameter_sharing,
    )
    collection, carry = init_training_collection(
        actor,
        variables,
        schedule=schedule,
        seed=seed,
        prepared=prepared,
        shaping=shaping,
        shaping_mode=shaping_mode,
        reward=reward,
        reward_identity=reward_identity,
        discount=pqn.gamma,
        coefficient=shaping_coefficient,
        collect_training_state=False,
        metrics=metrics,
        recording=recording,
        pinned_opponent_share=pinned_opponent_share,
        pinned_opponent=pinned_opponent,
        keep_past=keep_past,
        history_capture_capacity=history_capture_capacity,
        minimum_capture_rounds=minimum_capture_rounds,
        capture_interval_rounds=capture_interval_rounds,
        opponent_population=opponent_population,
        opponent_selection=opponent_selection,
        learner_slots=learner_slots,
        partner_population=partner_population,
        partner_selection=partner_selection,
        vectorize_opponent_lanes=pqn.parameter_sharing == "all",
        actor_variables_at_step=None,
    )
    state = PQNLearnerState(
        carry,
        initialized.opt_state,
        _empty_recent(collection, carry, pqn),
        jax.random.fold_in(root, PQN_SHUFFLE_ROOT_TAG),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
        jnp.bool_(False),
        jnp.int32(LEARNER_ERROR_NONE),
    )
    _state_shapes(state, pqn)
    return collection, state


def _rollout_shapes(
    state: PQNLearnerState, rollout: TrainingRollout, pqn: PQNConfig
) -> tuple[int, int]:
    """Check static PQN rollout leaves; return (capacity, games B).

    Capacity may be 1 to rollout_length; each capacity compiles its own
    program and only capacity rollout_length can learn. PQN actors return
    PQNLearningOutputs and collection keeps no physical state. Raises
    ValueError or TypeError.
    """
    games = _state_shapes(state, pqn)
    rows = rollout.transitions
    if rows.valid.ndim != 2 or rows.valid.shape[1] != games:
        raise ValueError("PQN rollout needs (T,B) rows for the learner batch")
    length = rows.valid.shape[0]
    if not 1 <= length <= pqn.rollout_length:
        raise ValueError("PQN rollout capacity must be 1 to rollout_length")
    if not isinstance(rows.learning_outputs, PQNLearningOutputs):
        raise TypeError("PQN collection must return PQNLearningOutputs")
    _array(
        rows.learning_outputs.pre_memory,
        (length, games, TEAM_SLOTS, PQN_HIDDEN_SIZE),
        jnp.float32,
        "pre_memory",
    )
    if rows.training_state is not None or rollout.final_training_state is not None:
        raise ValueError("PQN collection must not keep physical training state")
    _array(
        rollout.initial_memory,
        (games, TEAM_SLOTS, PQN_HIDDEN_SIZE),
        jnp.float32,
        "initial_memory",
    )
    _array(rollout.real_steps, (), jnp.int32, "real_steps")
    for name in ("valid", "episode_start", "ended"):
        _array(getattr(rows, name), (length, games), jnp.bool_, name)
    for name in ("active", "alive"):
        _array(getattr(rows, name), (length, games, TEAM_SLOTS), jnp.bool_, name)
    _array(rows.task_rewards, (length, games, TEAM_SLOTS), jnp.float32, "task_rewards")
    if rows.learner_active is not None:
        _array(
            rows.learner_active,
            (length, games, TEAM_SLOTS),
            jnp.bool_,
            "learner_active",
        )
    if rows.custom_rewards is not None:
        _array(
            rows.custom_rewards,
            (length, games, TEAM_SLOTS),
            jnp.float32,
            "custom_rewards",
        )
    _array(rows.shaping_reward, (length, games), jnp.float32, "shaping_reward")
    for name in ("learner_update", "opponent_update", "opponent_snapshot"):
        _array(getattr(rows, name), (length, games), jnp.int32, name)
    _array(rollout.final_ended, (games,), jnp.bool_, "final_ended")
    for head in rows.actions:
        _array(head, (length, games, 10), jnp.int32, "native action")
    return length, games


def _ceil(value: Array, divisor: int) -> Array:
    """Integer ceil division of a nonnegative int32 array by a positive int."""
    return (value + (divisor - 1)) // divisor


def _boundary_valid(
    state: PQNLearnerState,
    collected: TrainingCarry,
    rollout: TrainingRollout,
    pqn: PQNConfig,
    planned_learning_blocks: int,
    continuation: LearnerContinuation | None = None,
) -> Array:
    """Check counters, the chunk rule and the exact real prefix in one bool.

    The prior counters, version and refresh round must match the reachable
    boundary table for the prior rounds; the kept-row size must be
    ``min(H, rounds)``; the static planned count must equal
    ``ceil((total_rounds - W) / T)``; a nonempty block must hold exactly the
    expected chunk (``min(T, W - r)`` before W, ``min(T, R - r)`` after) and
    may learn only at capacity T; the collected carry must keep the same actor
    version; and the rows must be an exact valid prefix of this learner
    version with matching memory, opponent versions and endings. A checked
    continuation counts later blocks from its saved origin and keeps the
    original planned horizon; remaining initial collection is still capped
    at W. The child end may also cap an initial chunk.
    """
    rows = rollout.transitions
    capacity = rows.valid.shape[0]
    prefix = jnp.arange(capacity)[:, None] < rollout.real_steps
    prior = state.carry
    rounds = prior.progress.rounds
    learning = state.learning_blocks
    refreshed = prior.history.last_refresh_rounds
    initial, length = pqn.initial_rounds, pqn.rollout_length
    refill_end = reward_refill_end(
        continuation, initial_rounds=initial, memory_window=pqn.memory_window
    )
    reset = None if continuation is None else continuation.reward_reset
    elapsed = rounds - (0 if reset is None else reset.rounds)
    learned_before = 0 if reset is None else reset.learning_blocks
    refreshed_before = 0 if reset is None else reset.last_refresh_rounds
    total = collected.schedule.total_rounds
    after = rounds > initial
    expected_learning = jnp.where(after, _ceil(rounds - initial, length), 0)
    expected_blocks = jnp.where(
        after, -(-initial // length) + expected_learning, _ceil(rounds, length)
    )
    expected_chunk = jnp.where(
        rounds < refill_end,
        jnp.minimum(length, refill_end - rounds),
        jnp.minimum(length, total - rounds),
    )
    learning_now = rounds >= refill_end
    snapshot = rows.opponent_snapshot
    history = collected.history
    matched = (snapshot[..., None] == history.captured_ids) & (snapshot[..., None] >= 0)
    captured = jnp.sum(jnp.where(matched, history.captured_updates, 0), axis=-1)
    pin_update = jnp.where(history.external_pin, -2, history.pinned_update)
    opponent_version = jnp.where(
        snapshot == -1,
        rows.opponent_update == learning,
        jnp.where(
            snapshot == -2,
            rows.opponent_update == pin_update,
            jnp.any(matched, axis=-1) & (rows.opponent_update == captured),
        ),
    )
    if collected.opponent_selection is not None:
        members = collected.opponent_selection.shares.shape[0] - 2
        opponent_version |= (
            (snapshot <= -3) & (snapshot >= -2 - members) & (rows.opponent_update == -2)
        )
    planned = _ceil(jnp.maximum(total - initial, 0), length)
    if continuation is not None:
        start = continuation.start_rounds
        warm = jnp.maximum(0, jnp.minimum(rounds, refill_end) - start)
        learned_rounds = jnp.maximum(0, rounds - max(start, refill_end))
        new_learning = _ceil(learned_rounds, length)
        expected_learning = continuation.start_learning_blocks + new_learning
        expected_blocks = continuation.start_blocks + _ceil(warm, length) + new_learning
        expected_chunk = jnp.minimum(expected_chunk, total - rounds)
        planned = jnp.int32(continuation.pqn_planned_learning_blocks)
    return (
        (learning >= 0)
        & (state.completed_blocks < _MAX_COUNT)
        & (state.completed_blocks == expected_blocks)
        & (learning == expected_learning)
        & (state.completed_updates == learning * pqn.epochs * pqn.num_minibatches)
        & (prior.history.current_update == learning)
        & (refreshed == jnp.where(learning > learned_before, rounds, refreshed_before))
        & (elapsed >= 0)
        & (state.recent.size == jnp.minimum(elapsed, pqn.memory_window))
        & (planned == planned_learning_blocks)
        & (collected.history.current_update == learning)
        & (collected.history.last_refresh_rounds == refreshed)
        & (rollout.real_steps >= 0)
        & (rollout.real_steps <= capacity)
        & ((rollout.real_steps == 0) | (rollout.real_steps == expected_chunk))
        & ((rollout.real_steps == 0) | ~learning_now | jnp.bool_(capacity == length))
        & (collected.progress.rounds - rounds == rollout.real_steps)
        & jnp.all(rows.valid == prefix)
        & jnp.all(jnp.where(rows.valid, rows.learner_update == learning, True))
        & jnp.all(jnp.where(rows.valid, opponent_version, True))
        & jnp.all(rollout.initial_memory == learner_memory(prior))
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


def _sequence_valid(rows: PQNRow, learning: Array) -> Array:
    """Check a window's real rows follow one another in each game's order.

    rows has leading (C,B) with a chronological valid prefix. Within an
    episode the next row has the same episode ID and one more decision step;
    a new episode follows only an ended row and starts at decision step 0
    with a new ID. Kept rows keep their original learner version: versions
    never decrease along the window and never exceed ``learning``.
    """
    pair = rows.valid[:-1] & rows.valid[1:]
    ended = rows.ended[:-1]
    same = (
        (rows.episode_id[1:] == rows.episode_id[:-1])
        & (rows.decision_step[1:] == rows.decision_step[:-1] + 1)
        & ~rows.episode_start[1:]
    )
    new = (
        rows.episode_start[1:]
        & (rows.decision_step[1:] == 0)
        & (rows.episode_id[1:] != rows.episode_id[:-1])
    )
    order = jnp.where(ended, new, same)
    versions = (rows.learner_update[1:] >= rows.learner_update[:-1]) & (
        rows.learner_update[1:] <= learning
    )
    return jnp.all(jnp.where(pair, order & versions, True))


def _minibatch_counts(
    rows: PQNRow,
    metrics: PQNMetrics,
    bank: int,
    window: int,
    opponent_rows: int,
    capture_capacity: int | Array = 20,
) -> PQNUsedCounts:
    """Count one minibatch step's pairs by stage, source row and opponent row.

    rows has leading (C,D). Only performed steps count. window is H, so left
    rows 0..H-1 are kept-row pairs.
    """
    step = metrics.performed.astype(jnp.int32)
    pair = rows.valid[:-1] & rows.valid[1:]
    if rows.learner_active is not None:
        pair &= jnp.any((rows.active & rows.learner_active)[:-1], axis=-1)
    pair = pair.astype(jnp.int32) * step
    kept = (jnp.arange(pair.shape[0]) < window)[:, None]

    def count(index: Array, size: int) -> Array:
        """Add each pair to the bin of its left row."""
        return (
            jnp.zeros(size, jnp.int32)
            .at[index[:-1].reshape(-1)]
            .add(pair.reshape(-1), mode="drop")
        )

    games = jnp.int32(rows.valid.shape[1]) * step
    return PQNUsedCounts(
        games,
        metrics.used_td_pairs,
        metrics.used_agent_utilities,
        jnp.sum(pair * kept, dtype=jnp.int32),
        count(rows.episode_stage, _STAGES),
        count(rows.source_index, bank),
        count(
            opponent_counter_row(rows.opponent_snapshot, capture_capacity),
            opponent_rows,
        ),
    )


def _absent_counts(bank: int, opponent_rows: int) -> PQNUsedCounts:
    """Return zero used counts with the fixed shapes."""
    zero = jnp.int32(0)
    return PQNUsedCounts(
        zero,
        zero,
        zero,
        zero,
        jnp.zeros(_STAGES, jnp.int32),
        jnp.zeros(bank, jnp.int32),
        jnp.zeros(opponent_rows, jnp.int32),
    )


def _absent_metrics(pqn: PQNConfig) -> PQNMetrics:
    """Return zero (epochs, num_minibatches) metrics for blocks without steps."""
    shape = (pqn.epochs, pqn.num_minibatches)
    floats = jnp.zeros(shape, jnp.float32)
    counts = jnp.zeros(shape, jnp.int32)
    return PQNMetrics(
        floats,
        floats,
        floats,
        floats,
        counts,
        counts,
        jnp.ones(shape, jnp.bool_),
        jnp.zeros(shape, jnp.bool_),
    )


def _absent_snapshot() -> SnapshotEvent:
    """Return the existing absent history-capture record."""
    return SnapshotEvent(
        jnp.bool_(False),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.zeros(_SNAPSHOT_SLOTS, jnp.bool_),
        jnp.int32(-1),
    )


def epoch_permutations(
    shuffle_root: Array, learning_block: Array, pqn: PQNConfig, num_envs: int
) -> Array:
    """Return the game order of every epoch of one learning block.

    Parameters
    ----------
    shuffle_root : Array
        The learner's typed Threefry shuffle root.
    learning_block : Array
        Int32 index k of the learning block (learning blocks completed before
        it).
    pqn : PQNConfig
        Settings giving the number of epochs.
    num_envs : int
        Number of games B.

    Returns
    -------
    Array
        Int32 (epochs, B): row e is ``permutation(fold_in(fold_in(root, k),
        e), B)``. After the permutation, minibatch j takes games
        ``[j * D, (j + 1) * D)`` with D = B / num_minibatches.

    Notes
    -----
    Pure JAX. Rejected and empty blocks never call it, so they consume no
    persistent key.
    """
    block_key = jax.random.fold_in(shuffle_root, learning_block)
    return jnp.stack(
        [
            jax.random.permutation(jax.random.fold_in(block_key, epoch), num_envs)
            for epoch in range(pqn.epochs)
        ]
    ).astype(jnp.int32)


def run_epochs[T](
    train: PQNTrainState,
    permutations: Array,
    step: Callable[[PQNTrainState, Array], tuple[PQNTrainState, T]],
    *,
    num_minibatches: int,
) -> tuple[PQNTrainState, T]:
    """Run the epoch and minibatch loop of one learning block.

    Parameters
    ----------
    train : PQNTrainState
        Network, statistics, optimizer and step count before the block.
    permutations : Array
        Int32 (epochs, B) game orders, one row per epoch; normally from
        :func:`epoch_permutations`. Reference tests pass the donor's own.
    step : callable
        ``step(state, chosen) -> (state, output)`` takes one optimizer step on
        the games ``chosen`` (int32 (D,), D = B / num_minibatches). It owns
        building that minibatch.
    num_minibatches : int
        Static M; must divide B.

    Returns
    -------
    tuple[PQNTrainState, T]
        The state after epochs * M steps and every step's output with
        leading (epochs, M) axes.

    Notes
    -----
    Within epoch e, minibatch j takes the contiguous games
    ``permutations[e, j * D:(j + 1) * D]`` and minibatches run in order, the
    donor's reshape. Each step's candidate parameters, statistics and
    optimizer state feed the next step, even a nonfinite one: acceptance is
    decided once for the whole block by the caller, which rejects the block if
    any step was not performed or not finite. Time is never shuffled and
    teammates stay together. Pure JAX; both loops are compiled scans.
    """
    games = permutations.shape[1]
    if games % num_minibatches:
        raise ValueError("num_minibatches must divide the number of games")

    def epoch(state: PQNTrainState, order: Array) -> tuple[PQNTrainState, T]:
        """Split one epoch's game order into minibatches and scan them."""
        return jax.lax.scan(
            step, state, order.reshape(num_minibatches, games // num_minibatches)
        )

    return jax.lax.scan(epoch, train, permutations)


def learn_window(
    train: PQNTrainState,
    rows: PQNRow,
    memory: Array,
    permutations: Array,
    *,
    pqn: PQNConfig,
    planned_learning_blocks: int,
    bank: int,
    opponent_rows: int = 22,
    capture_capacity: int | Array = 20,
    continuation: LearnerContinuation | None = None,
) -> tuple[PQNTrainState, PQNMetrics, PQNUsedCounts]:
    """Run every epoch and minibatch of one learning window.

    Parameters
    ----------
    train : PQNTrainState
        Network, statistics, optimizer and step count before the block.
    rows : PQNRow
        Window rows with leading (C,B) and a chronological real prefix.
    memory : Array
        Float32 (C,B,5,512) stored pre-action memories; row 0's memory starts
        each game's unroll.
    permutations : Array
        Int32 (epochs, B) game orders from :func:`epoch_permutations`.
    pqn : PQNConfig
        Static settings.
    planned_learning_blocks : int
        Static positive N for the learning-rate schedule.
    bank : int
        Static source-bank size for exposure bins.

    continuation : LearnerContinuation or None, default=None
        Checked child schedule. Its optional future rate uses absolute optimizer
        counts; the saved optimizer tree and every counter stay intact.

    Returns
    -------
    tuple[PQNTrainState, PQNMetrics, PQNUsedCounts]
        The state after E * M steps, metrics with leading (E, M), and the
        block's used counts summed over every step. Targets are recomputed at
        every step with that step's parameters.

    Notes
    -----
    Uses :func:`run_epochs`. Only each minibatch's selected games are
    expanded to network inputs (27,271,200 bytes at the defaults), never the
    whole window. Pure JAX.
    """
    start = memory[0]

    def minibatch(
        state: PQNTrainState, chosen: Array
    ) -> tuple[PQNTrainState, tuple[PQNMetrics, PQNUsedCounts]]:
        """Expand the chosen games and take one optimizer step."""

        def select(value: Array) -> Array:
            """Keep the chosen games of one (C,B,...) leaf."""
            return jnp.take(value, chosen, axis=1)

        selected = cast(PQNRow, jax.tree.map(select, rows))
        batch = _expand_minibatch(selected, jnp.take(start, chosen, axis=0), pqn)
        rate = None if continuation is None else continuation.pqn_rate
        candidate, metrics = update_pqn(
            state,
            batch,
            pqn=pqn,
            planned_learning_blocks=planned_learning_blocks,
            learning_rate=None if rate is None else rate.at_array,
        )
        return candidate, (
            metrics,
            _minibatch_counts(
                selected,
                metrics,
                bank,
                pqn.memory_window,
                opponent_rows,
                capture_capacity,
            ),
        )

    final, (metrics, counts) = run_epochs(
        train, permutations, minibatch, num_minibatches=pqn.num_minibatches
    )

    def total(value: Array) -> Array:
        """Sum one count over the (E, M) step axes."""
        return jnp.sum(value, axis=(0, 1), dtype=jnp.int32)

    return final, metrics, cast(PQNUsedCounts, jax.tree.map(total, counts))


def _count_leaves_equal(
    opt_state: Tree, steps: Array, *, grouped: bool = False
) -> Array:
    """Check complete optimizer clocks against the run-wide int32 step count.

    Shared mode requires every scalar count to equal steps. Grouped mode lets
    absent actors keep local counts: all five counts must be in [0, steps],
    and RAdam/schedule counts must agree for each group. A nonzero run count
    requires at least one used group. This check works under jit.
    """
    counts = [
        leaf
        for leaf in jax.tree.leaves(opt_state)
        if jnp.issubdtype(leaf.dtype, jnp.integer)
    ]
    if not counts:
        return jnp.bool_(True)
    if not grouped:
        return jnp.all(jnp.stack([jnp.all(leaf == steps) for leaf in counts]))
    first = counts[0]
    valid = jnp.all((first >= 0) & (first <= steps)) & (
        (steps == 0) | jnp.any(first > 0)
    )
    for count in counts[1:]:
        valid = valid & jnp.all(count == first)
    return valid


def _choose[T](flag: Array, new: T, old: T) -> T:
    """Pick every leaf of new where the scalar flag is true, else old."""

    def pick(a: Array, b: Array) -> Array:
        """Select one leaf."""
        return jnp.where(flag, a, b)

    return cast(T, jax.tree.map(pick, new, old))


def update_pqn_learner(
    state: PQNLearnerState,
    collected: TrainingCarry,
    rollout: TrainingRollout,
    *,
    pqn: PQNConfig = DEFAULT_PQN_CONFIG,
    planned_learning_blocks: int,
    continuation: LearnerContinuation | None = None,
) -> tuple[PQNLearnerState, PQNUpdateResult]:
    """Accept one collected block: keep its rows, and learn once W is reached.

    Parameters
    ----------
    state : PQNLearnerState
        Last accepted boundary before this block's collection. Keep it
        unchanged while collecting with ``state.carry``. A failed state stays
        failed.
    collected : TrainingCarry
        Successor returned by that collection call, with the same actor
        version.
    rollout : TrainingRollout
        Matching rollout. Before W rounds it must hold exactly the next
        initial chunk: ``min(T, W - r)`` real rounds, so T then H. From W on,
        capacity T with T real rounds, or fewer only at the declared total.
        Each capacity compiles its own program; a capacity below T never
        learns.
    pqn : PQNConfig, default=DEFAULT_PQN_CONFIG
        Same static settings used at initialization; capture them in the jit
        wrapper.
    planned_learning_blocks : int
        Static positive N from ``pqn_planned_learning_blocks`` for the
        declared total. The block is rejected when it differs from the count
        computed from the carried schedule.

    continuation : LearnerContinuation or None, default=None
        Checked static child boundary and future schedules. The original PQN
        horizon, saved blocks and optimizer clocks continue. If stored rewards
        changed, only kept rows restart; learning waits for a clean H-row prefix.

    Returns
    -------
    tuple[PQNLearnerState, PQNUpdateResult]
        An initial chunk keeps the latest rows and counts the block only: no
        optimizer step, statistics change, key use or version change. The chunk
        that reaches W also stores ``epsilon_at(0)`` as the current rate. A
        learning block joins the H kept rows and the new rows into one window,
        runs E epochs of M minibatch steps, publishes the new network, its
        statistics and ``epsilon_at(k + 1)`` once through
        ``refresh_opponents``, and keeps the last H real rows with their
        original memories. An empty block (no real round) returns the state
        unchanged with accepted, performed and failed all False.

    Raises
    ------
    TypeError, ValueError
        Static rollout, memory, counter or batch shapes are invalid.

    Notes
    -----
    Any failure rejects the whole block and keeps every leaf of the previous
    boundary with a sticky reason: 1 collection error, 2 boundary, chunk or
    counter mismatch, 3 nonfinite rows, 4 illegal recorded action, 5
    nonfinite or incomplete update, 6 a publication the shared history would
    refuse, 7 broken window order. A collection or boundary failure fails the
    state even for an empty block. The result of a rejected block keeps its
    experience summary and attempted (E, M) metrics for diagnosis. Its
    recording may have progressed; the host must stop and recover from a
    checkpoint.

    Inputs are not donated, so the previous boundary survives a rejection.
    The learning branch returns only the network, optimizer state, metrics
    and used counts; the opponent bank never passes through it. One
    three-way switch then makes the new history: publish, keep (initial
    chunks, with the rate write at W) or restore the previous one. Pure JAX;
    wrap in jit with fixed settings.
    """
    pqn = continuation_config(pqn, continuation)
    capacity, games = _rollout_shapes(state, rollout, pqn)
    validate_pqn_batch_size(games, pqn)
    if collected.state.episode_id.shape != (games,):
        raise ValueError("Collected boundary must retain the learner batch")
    bank = _bank_size(state.carry)
    can_learn = jnp.bool_(capacity == pqn.rollout_length)
    rounds = collected.progress.rounds
    learning = state.learning_blocks
    nonempty = rollout.real_steps > 0
    learn_now = state.carry.progress.rounds >= reward_refill_end(
        continuation, initial_rounds=pqn.initial_rounds, memory_window=pqn.memory_window
    )
    entry_reason = jnp.where(
        _failed(state.carry) | _failed(collected),
        LEARNER_ERROR_COLLECTION,
        jnp.where(
            _boundary_valid(
                state, collected, rollout, pqn, planned_learning_blocks, continuation
            ),
            LEARNER_ERROR_NONE,
            LEARNER_ERROR_BOUNDARY,
        ),
    ).astype(jnp.int32)
    rows, memory = _pqn_rows(rollout.transitions)
    window_rows, window_memory, real_rows = _learning_window(
        state.recent, rows, memory, rollout.real_steps
    )
    rows_reason = jnp.where(
        ~_finite(rollout.transitions),
        LEARNER_ERROR_NONFINITE_BATCH,
        jnp.where(
            ~_behavior_valid(rollout),
            LEARNER_ERROR_ACTION,
            jnp.where(
                _sequence_valid(window_rows, learning),
                LEARNER_ERROR_NONE,
                LEARNER_ERROR_SEQUENCE,
            ),
        ),
    ).astype(jnp.int32)
    ready = (
        ~state.failed
        & (entry_reason == LEARNER_ERROR_NONE)
        & (rows_reason == LEARNER_ERROR_NONE)
        & nonempty
    )
    attempted = ready & learn_now & can_learn
    train = PQNTrainState(
        state.carry.history.current_variables.network,
        state.opt_state,
        state.completed_updates,
    )
    steps_per_block = pqn.epochs * pqn.num_minibatches

    def skip(_: None) -> tuple[PQNTrainState, PQNMetrics, PQNUsedCounts, Array]:
        """Leave the network and optimizer untouched."""
        return (
            train,
            _absent_metrics(pqn),
            _absent_counts(bank, state.carry.progress.opponent_steps.shape[0]),
            jnp.bool_(True),
        )

    if capacity == pqn.rollout_length:

        def learn(_: None) -> tuple[PQNTrainState, PQNMetrics, PQNUsedCounts, Array]:
            """Run every epoch over the window; report whether all steps held."""
            final, metrics, used = learn_window(
                train,
                window_rows,
                window_memory,
                epoch_permutations(state.shuffle_root, learning, pqn, games),
                pqn=pqn,
                planned_learning_blocks=planned_learning_blocks,
                bank=bank,
                opponent_rows=state.carry.progress.opponent_steps.shape[0],
                capture_capacity=state.carry.history.capture_capacity,
                continuation=continuation,
            )
            complete = (
                jnp.all(metrics.performed)
                & jnp.all(metrics.finite)
                & (final.optimizer_steps == state.completed_updates + steps_per_block)
                & _count_leaves_equal(
                    final.opt_state,
                    final.optimizer_steps,
                    grouped=pqn.parameter_sharing != "all",
                )
                & _finite(final)
            )
            return final, metrics, used, complete

        learned, metrics, used, complete = _cond(attempted, learn, skip)
    else:
        learned, metrics, used, complete = skip(None)
    history = collected.history
    index = learning + jnp.int32(1)
    publishable = publication_valid(
        history,
        completed_rounds=rounds,
        update_index=index,
        schedule=collected.schedule,
    )
    later_reason = jnp.where(
        attempted & ~complete,
        LEARNER_ERROR_NONFINITE_UPDATE,
        jnp.where(attempted & ~publishable, LEARNER_ERROR_HISTORY, LEARNER_ERROR_NONE),
    )
    reason = jnp.where(
        state.failed,
        state.failure_reason,
        jnp.where(
            entry_reason != LEARNER_ERROR_NONE,
            entry_reason,
            jnp.where(
                nonempty & (rows_reason != LEARNER_ERROR_NONE),
                rows_reason,
                later_reason,
            ),
        ),
    ).astype(jnp.int32)
    failed = state.failed | (reason != LEARNER_ERROR_NONE)
    accepted = ~failed & nonempty
    publish = accepted & attempted
    next_epsilon = epsilon_at(index, planned_learning_blocks, pqn)
    initial_epsilon = epsilon_at(jnp.int32(0), planned_learning_blocks, pqn)
    if continuation is not None and continuation.exploration is not None:
        next_epsilon = continuation.exploration.at_array(index)
        initial_epsilon = continuation.exploration.at_array(jnp.int32(0))
    kept_epsilon = jnp.where(
        rounds == pqn.initial_rounds,
        initial_epsilon,
        history.current_variables.epsilon,
    )

    def published(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
        """Publish the learned network with the next exploration rate once."""
        return refresh_opponents(
            history,
            PQNActorVariables(learned.network, next_epsilon),
            completed_rounds=rounds,
            update_index=index,
            schedule=collected.schedule,
        )

    def kept(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
        """Keep the collected history; only the stored rate may change."""
        current = history.current_variables._replace(epsilon=kept_epsilon)
        return history._replace(current_variables=current), _absent_snapshot()

    def previous(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
        """Restore the history of the previous boundary."""
        return state.carry.history, _absent_snapshot()

    branch = jnp.where(publish, 0, jnp.where(accepted, 1, 2))
    new_history, event = cast(
        tuple[OpponentHistory, SnapshotEvent],
        jax.lax.switch(branch, (published, kept, previous), None),
    )
    carry = _choose(
        accepted,
        collected._replace(history=None),
        state.carry._replace(history=None),
    )._replace(history=new_history)
    final_state = PQNLearnerState(
        carry,
        _choose(publish, learned.opt_state, state.opt_state),
        _choose(
            accepted,
            _retain_recent(window_rows, window_memory, real_rows, pqn),
            state.recent,
        ),
        state.shuffle_root,
        jnp.where(accepted, state.completed_blocks + 1, state.completed_blocks),
        jnp.where(publish, learned.optimizer_steps, state.completed_updates),
        jnp.where(publish, index, learning),
        failed,
        jnp.where(failed, reason, LEARNER_ERROR_NONE).astype(jnp.int32),
    )
    result = PQNUpdateResult(
        accepted,
        publish,
        failed,
        final_state.failure_reason,
        metrics,
        event,
        _summary(
            rollout,
            collected.progress.opponent_steps.shape[0],
            collected.history.capture_capacity,
        ),
        _choose(
            publish,
            used,
            _absent_counts(bank, state.carry.progress.opponent_steps.shape[0]),
        ),
    )
    return final_state, result


@lru_cache(maxsize=16)
def _pqn_spec(pqn: PQNConfig, planned_learning_blocks: int) -> PQNTrainState:
    """Trace initialize_pqn once per settings and N for shape-only leaf specs."""

    def build(key: Array) -> PQNTrainState:
        """Initialize with these settings; only shapes are kept."""
        return initialize_pqn(
            key, pqn=pqn, planned_learning_blocks=planned_learning_blocks
        )

    return cast(PQNTrainState, jax.eval_shape(build, jax.random.key(0)))


_saved_boundary_valid = cast(Callable[[PQNLearnerState], Array], jax.jit(_finite))


def _epsilon_matches(
    value: float, learning_blocks: int, planned: int, pqn: PQNConfig
) -> bool:
    """Compare a stored float32 rate with the float64 block-clock value.

    The rate must be within 2e-6 of pqn_epsilon_reference, and exactly
    eps_finish once the integer block clock reaches the decay span.
    """
    if learning_blocks >= pqn.eps_decay_fraction * planned:
        return value == float(np.float32(pqn.eps_finish))
    expected = pqn_epsilon_reference(learning_blocks, planned, pqn)
    return abs(value - expected) <= 2e-6


def _check_recent(
    state: PQNLearnerState,
    pqn: PQNConfig,
    continuation: LearnerContinuation | None = None,
) -> None:
    """Check the kept rows are the latest real suffix of every game.

    Raises ValueError when the size, the valid prefix, in-game order, learner
    versions, the link to the carry's last successor or the stored memories
    (zero at episode starts and in inactive slots) are wrong. A reward reset
    begins an empty suffix at its saved round offset. Host-only.
    """
    recent = state.recent
    rounds = int(state.carry.progress.rounds)
    size = int(recent.size)
    reset = None if continuation is None else continuation.reward_reset
    elapsed = rounds - (0 if reset is None else reset.rounds)
    if elapsed < 0 or size != min(pqn.memory_window, elapsed):
        raise ValueError("PQN kept rows hold the wrong number of real rows")
    rows = cast(PQNRow, jax.tree.map(np.asarray, recent.rows))
    if not (rows.valid[:size].all() and not rows.valid[size:].any()):
        raise ValueError("PQN kept rows are not a real prefix")
    if not bool(_sequence_valid(recent.rows, state.learning_blocks)):
        raise ValueError("PQN kept rows break game order or learner versions")
    memory = np.asarray(recent.pre_memory)[:size]
    if np.any(memory[rows.episode_start[:size]] != 0) or np.any(
        memory[~rows.active[:size]] != 0
    ):
        raise ValueError("PQN kept memory is not zero at a start or inactive slot")
    if size:
        last = size - 1
        info = state.carry.state
        same = np.asarray(info.episode_id) == np.asarray(rows.episode_id[last])
        done = np.asarray(info.done.done, bool)
        step = np.asarray(info.core_state.step_count - info.initial_step_count)
        ended = np.asarray(rows.ended[last], bool)
        following = step == np.asarray(rows.decision_step[last]) + 1
        # The carry holds the true last successor before any pending reset.
        if not np.all(same & np.where(ended, done, ~done & following)):
            raise ValueError("PQN kept rows do not lead into the carried successor")


def validate_pqn_learner(
    collection: TrainingCollection,
    state: PQNLearnerState,
    *,
    pqn: PQNConfig = DEFAULT_PQN_CONFIG,
    recheck_installed_content: bool = True,
) -> None:
    """Validate a complete accepted PQN boundary before any recording recovery.

    Parameters
    ----------
    collection : TrainingCollection
        Matching host descriptor: no physical state and no step hook.
    state : PQNLearnerState
        Fully restored numerical boundary, including the zero boundary and
        the initial-collection boundaries at T and W rounds.
    pqn : PQNConfig, default=DEFAULT_PQN_CONFIG
        Original static settings.
    recheck_installed_content : bool, default=True
        Python bool. True reruns the installed-content preflight, as restore
        requires; saves pass False to reuse the verified descriptor.

    Raises
    ------
    TypeError, ValueError
        The descriptor, shapes, content, keys, counters, kept rows, exploration
        rates, finite values, running variances or optimizer counts do not
        describe a reachable accepted boundary. A failed state is never valid.

    Notes
    -----
    Host-only. A fresh run obtains N from its declared schedule through
    ``pqn_planned_learning_blocks``. A child keeps N and its block origin in
    ``collection.schedule.continuation['learner']``. The rounds must be a
    reachable boundary under that exact declaration
    (``validation._collection_boundary`` with W initial rounds) and every
    counter must match it: completed blocks equal its ordinal, learning blocks
    k, ``completed_updates == k * epochs * num_minibatches``,
    ``history.current_update == k`` and the last refresh at the current
    rounds once learning has started (0 before). The shuffle root must be
    re-derived from the run root. Shared optimizer counts equal completed_updates.
    Grouped actor counts may be lower when that group was absent; RAdam and
    schedule counts must agree within each group. All local counts are bounded
    by completed_updates, and every running variance is at least zero. The kept
    rows must hold ``min(H, rounds)`` real rows in game order that lead into
    the carried successor, with zero memory at episode starts and in inactive
    slots. The current rate is exactly 1 before W and follows the block clock
    from W on; each history slot's rate follows its captured update. A child
    checks inherited slots against its verified parent proof and later slots
    against its active future rule. This check reads the whole state once
    from the device; it changes nothing.
    """
    if type(recheck_installed_content) is not bool:
        raise TypeError("recheck_installed_content must be a Python bool")
    if not isinstance(cast(object, pqn), PQNConfig):
        raise TypeError("pqn must be a PQNConfig")
    continuation = schedule_continuation(collection.schedule)
    pqn = continuation_config(pqn, continuation)
    games = _state_shapes(state, pqn)
    validate_pqn_batch_size(games, pqn)
    if (
        collection.collect_training_state
        or collection.actor_variables_at_step is not None
    ):
        raise ValueError("Learner descriptor does not match PQN collection")
    if bool(state.failed) or int(state.failure_reason) != LEARNER_ERROR_NONE:
        raise ValueError("A failed learner boundary cannot be restored")
    _validate_training_continuation(
        collection, state.carry, recheck_installed_content=recheck_installed_content
    )
    from marl_battlegrounds.training.validation import _collection_boundary

    total = int(state.carry.schedule.total_rounds)
    planned = (
        pqn_planned_learning_blocks(total, pqn)
        if continuation is None
        else cast(int, continuation.pqn_planned_learning_blocks)
    )
    rounds = int(state.carry.progress.rounds)
    learning = int(state.learning_blocks)
    updates = int(state.completed_updates)
    history = state.carry.history
    initial_blocks = -(-pqn.initial_rounds // pqn.rollout_length)
    if continuation is None:
        reachable, ordinal = _collection_boundary(
            rounds,
            total_env_steps=total,
            num_envs=1,
            rollout_length=pqn.rollout_length,
            initial_rounds=pqn.initial_rounds,
        )
        expected_learning = max(0, ordinal - initial_blocks)
    else:
        reachable, ordinal, expected_learning = continuation_boundary(
            rounds,
            total_rounds=total,
            continuation=continuation,
            rollout_length=pqn.rollout_length,
            initial_rounds=pqn.initial_rounds,
        )
    reset = None if continuation is None else continuation.reward_reset
    learned_before = 0 if reset is None else reset.learning_blocks
    refreshed_before = 0 if reset is None else reset.last_refresh_rounds
    if (
        reachable != rounds
        or int(state.completed_blocks) != ordinal
        or learning != expected_learning
        or updates != learning * pqn.epochs * pqn.num_minibatches
        or int(history.current_update) != learning
        or int(history.last_refresh_rounds)
        != (rounds if learning > learned_before else refreshed_before)
    ):
        raise ValueError("PQN block, update and collection counts disagree")
    expected_key = jax.random.fold_in(state.carry.root_key, PQN_SHUFFLE_ROOT_TAG)
    if not np.array_equal(
        jax.random.key_data(state.shuffle_root), jax.random.key_data(expected_key)
    ):
        raise ValueError("PQN shuffle key does not match the run root")
    numerical = PQNTrainState(
        history.current_variables.network, state.opt_state, state.completed_updates
    )
    template = _pqn_spec(pqn, planned)
    if jax.tree.structure(numerical) != jax.tree.structure(template):
        raise ValueError("PQN network or optimizer structure is incompatible")
    for actual, expected in zip(
        jax.tree.leaves(numerical), jax.tree.leaves(template), strict=True
    ):
        _array(actual, expected.shape, expected.dtype, "PQN network/optimizer leaf")
    if not bool(
        _count_leaves_equal(
            state.opt_state,
            state.completed_updates,
            grouped=pqn.parameter_sharing != "all",
        )
    ):
        raise ValueError("PQN optimizer count differs from completed updates")

    def empty_recent(values: TrainingCarry) -> PQNRecent:
        """Trace the kept-row layout this collection and settings allocate."""
        return _empty_recent(collection, values, pqn)

    spec = jax.eval_shape(empty_recent, state.carry)
    if jax.tree.structure(state.recent) != jax.tree.structure(spec) or any(
        a.shape != b.shape or a.dtype != b.dtype
        for a, b in zip(
            jax.tree.leaves(state.recent), jax.tree.leaves(spec), strict=True
        )
    ):
        raise ValueError("PQN kept-row layout is incompatible")
    if not bool(_saved_boundary_valid(state)):
        raise ValueError("PQN learner contains nonfinite values")
    if not (
        _nonnegative_variances(history.current_variables.network.batch_stats)
        and _nonnegative_variances(history.historical_variables.network.batch_stats)
    ):
        raise ValueError("A PQN running variance is negative")
    _check_recent(state, pqn, continuation)

    def epsilon_matches(value: float, block: int) -> bool:
        """Check the active future rule without changing inherited opponents."""
        if continuation is not None and continuation.exploration is not None:
            return abs(value - continuation.exploration.at(block)) <= 2e-6
        return _epsilon_matches(value, block, planned, pqn)

    rate = float(history.current_variables.epsilon)
    if rounds < pqn.initial_rounds:
        if rate != 1.0:
            raise ValueError("Initial random collection must explore with rate 1")
    elif not epsilon_matches(rate, learning):
        raise ValueError("Current exploration rate disagrees with the block clock")
    inherited = (
        {}
        if continuation is None
        else dict(
            zip(
                continuation.frozen_capture_ids,
                continuation.frozen_epsilon,
                strict=True,
            )
        )
    )
    captured = np.asarray(history.captured_updates)
    ids = np.asarray(history.captured_ids)
    rates = np.asarray(history.historical_variables.epsilon)
    for slot in np.flatnonzero(ids >= 0):
        expected = inherited.get(int(ids[slot]))
        matches = (
            float(rates[slot]) == expected
            if expected is not None
            else epsilon_matches(float(rates[slot]), int(captured[slot]))
        )
        if not matches:
            raise ValueError("A frozen opponent's exploration rate disagrees")
    if history.pinned_variables is not None and int(history.pinned_update) >= 0:
        rate = float(history.pinned_variables.epsilon)
        expected = None if continuation is None else continuation.pinned_epsilon
        if expected is not None:
            matches = rate == expected
        else:
            matches = epsilon_matches(rate, int(history.pinned_update))
        if not matches:
            raise ValueError("The permanent pin's exploration rate disagrees")
