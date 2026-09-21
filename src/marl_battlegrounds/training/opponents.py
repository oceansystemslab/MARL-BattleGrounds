"""Keep current self-play and frozen actor versions for one training run.

Initialize a capacity-20 bank on the host, draw assignments only at game resets,
and publish updated actor variables after learning with refresh_opponents.
make_opponent_system applies those choices through the actor's existing System.
Recurrent memory belongs to SystemState.team_b, never to this bank. These helpers
do not train, load checkpoints, call a critic, or change simulator rules.
"""

import math
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import DTypeLike

from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
    _normal_output,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.training.curriculum import ScheduleArrays

type Tree = Any
_CAPACITY = 20


class OpponentHistory(NamedTuple):
    """Carry one run's actor variables and reset-time opponent choices.

    Attributes
    ----------
    current_variables : PyTree
        Actor-only array tree, including every changing inference value. Shared
        by current opponents and the learner. No critic or optimizer belongs here.
    historical_variables : PyTree
        Same leaves with a leading capacity axis of 20. Occupied slots never
        change. Unused slots are zero. No rollout-time axis or lane copy is stored.
    count : Array
        Int32 scalar number of occupied slots, from 0 through 20.
    lane_snapshot : Array
        Int32 (B,) choices. -1 means current; other values index occupied slots.
        An assignment stays fixed through death, respawn and learner updates.
    current_update : Array
        Int32 scalar current learner version; initialization is version zero.
    captured_rounds, captured_updates : Array
        Int32 (20,) actual capture boundaries. Unused entries are -1.
    threshold_to_snapshot : Array
        Int32 (20,) slot satisfying each requested history threshold, or -1 if
        unmet. Thresholds are the 5% steps, or round 1 followed by 5% through
        95% when the schedule requests early capture of a pinned first-update
        actor. Several thresholds can point to one slot after one update.
    last_refresh_rounds : Array
        Int32 scalar last accepted real-round count, initially zero. One round
        means one decision across the fixed environment batch.
    error : Array
        Sticky Boolean scalar. A bad notification or assignment preserves prior
        values and sets this flag. Collection must stop while it is true.

    Notes
    -----
    This immutable numerical PyTree supports jit/scan. Initialization reserves
    20 times the actor tree's bytes for history, plus one current tree. Current
    leaves may share immutable storage with learner variables. Mixed application
    can need temporary selected weights; those are never retained in this record.
    The bank is evolving run state, not an immutable content declaration.
    """

    current_variables: Tree
    historical_variables: Tree
    count: Array  # pyright: ignore[reportIncompatibleMethodOverride]
    lane_snapshot: Array
    current_update: Array
    captured_rounds: Array
    captured_updates: Array
    threshold_to_snapshot: Array
    last_refresh_rounds: Array
    error: Array


class SnapshotEvent(NamedTuple):
    """Describe the one physical snapshot made by a completed learner update.

    created is a Boolean scalar. slot, rounds and update_index are int32 scalars;
    all three are -1 when no snapshot was made. threshold_mask is Boolean (20,)
    and marks every requested threshold satisfied by this capture. Report actual
    environment transitions by multiplying rounds by the fixed batch size on the
    host. A final snapshot can have no later training exposure.
    """

    created: Array
    slot: Array
    rounds: Array
    update_index: Array
    threshold_mask: Array


def _array_tree(tree: Tree, *, name: str) -> None:
    """Reject nonarray or nonnumeric leaves in an actor or memory tree.

    Empty trees are allowed; None leaves, strings, objects and random-key dtypes
    are not. Arrays may be Boolean, integer, real or complex. This checks declared
    structure, not scientific ownership of arbitrary caller-named variables.
    """
    for leaf in jax.tree.leaves(tree, is_leaf=lambda value: value is None):
        if not isinstance(leaf, (Array, np.ndarray, Tracer)):
            raise TypeError(f"{name} must contain only numerical array leaves")
        dtype = cast(DTypeLike, leaf.dtype)
        if not (jnp.issubdtype(dtype, jnp.number) or jnp.issubdtype(dtype, jnp.bool_)):
            raise TypeError(f"{name} must contain only numerical array leaves")


def _schema(tree: Tree) -> tuple[Any, tuple[tuple[tuple[int, ...], Any], ...]]:
    """Describe tree structure, leaf shapes and dtypes without retaining arrays."""
    structure = cast(Any, jax.tree.structure(tree))
    return structure, tuple((leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(tree))


def _matching_variables(variables: Tree, expected: Tree) -> None:
    """Require an actor-only array tree to match a fixed structure and leaf schema."""
    _array_tree(variables, name="Actor variables")
    if _schema(variables) != expected:
        raise ValueError("Actor variables must keep the same tree, shapes and dtypes")


def _pinned_share(value: object) -> None:
    """Require a static Python pinned-opponent share within [0, 0.8].

    Parameters
    ----------
    value : object
        The share to check. Zero means the 80/20 recipe is unchanged.

    Raises
    ------
    TypeError
        value is bool, a traced array or another non-number.
    ValueError
        value is not finite or lies outside [0, 0.8].
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "The pinned opponent share must be a Python number, not bool, string or "
            "traced"
        )
    if not math.isfinite(value) or not 0 <= value <= 0.8:
        raise ValueError("The pinned opponent share must be finite within [0, 0.8]")


def _field(value: Array, shape: tuple[int, ...], dtype: DTypeLike, name: str) -> None:
    """Check one numerical field's static shape and dtype, including under jit."""
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{name} must have {jnp.dtype(dtype).name} dtype")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")


def _history_shapes(history: OpponentHistory) -> int:
    """Check history structure and return B without reading device values."""
    if not isinstance(cast(object, history), OpponentHistory):
        raise TypeError("Opponent variables must be an OpponentHistory")
    if history.lane_snapshot.ndim != 1 or not history.lane_snapshot.shape[0]:
        raise ValueError("lane_snapshot must have a nonempty lane axis")
    size = history.lane_snapshot.shape[0]
    _field(history.lane_snapshot, (size,), jnp.int32, "lane_snapshot")
    for name in ("count", "current_update", "last_refresh_rounds"):
        _field(getattr(history, name), (), jnp.int32, name)
    _field(history.error, (), jnp.bool_, "error")
    for name in ("captured_rounds", "captured_updates", "threshold_to_snapshot"):
        _field(getattr(history, name), (_CAPACITY,), jnp.int32, name)
    _array_tree(history.current_variables, name="Actor variables")
    _array_tree(history.historical_variables, name="Historical variables")
    if jax.tree.structure(history.current_variables) != jax.tree.structure(
        history.historical_variables
    ):
        raise ValueError("Historical variables must match the actor tree")
    for current, bank in zip(
        jax.tree.leaves(history.current_variables),
        jax.tree.leaves(history.historical_variables),
        strict=True,
    ):
        if bank.shape != (_CAPACITY, *current.shape) or bank.dtype != current.dtype:
            raise ValueError("Historical leaves need capacity 20 and the actor schema")
    return size


def _history_invalid(history: OpponentHistory) -> Array:
    """Return sticky failure or invalid count, version, round or occupied-slot IDs.

    This small numerical guard supports jit. Static structure must already have
    passed _history_shapes. Collection may check it before memory initialization
    or actor application; it never clips or repairs a bad snapshot ID.
    """
    return (
        history.error
        | (history.count < 0)
        | (history.count > _CAPACITY)
        | (history.current_update < 0)
        | (history.last_refresh_rounds < 0)
        | jnp.any(history.lane_snapshot < -1)
        | jnp.any(history.lane_snapshot >= history.count)
    )


def _keys(keys: Array, size: int) -> None:
    """Require B typed Threefry keys or legacy uint32 keys shaped (B,2)."""
    if str(jax.random.key_impl(keys)) != "threefry2x32":
        raise ValueError("Opponent selection requires Threefry keys")
    if jax.random.key_data(keys).shape != (size, 2):
        raise ValueError("Opponent keys must contain one key per lane")


def init_opponent_history(actor_variables: Tree, *, num_envs: int) -> OpponentHistory:
    """Reserve an empty frozen-actor bank and assign every lane to current play.

    Parameters
    ----------
    actor_variables : PyTree
        Actor-only numerical arrays, including all changing inference state.
        Critic, optimizer and recurrent memory must stay outside this tree.
        Leaves retain their shapes/dtypes for the entire run. Empty () is valid.
    num_envs : int
        Positive Python batch size, excluding bool, within the int32 limit.

    Returns
    -------
    OpponentHistory
        Zero-filled capacity-20 bank, no snapshots, B current assignments (-1),
        version/round zero and error=False. Current JAX arrays are retained by
        reference; NumPy arrays become device arrays. No random draw is made.

    Raises
    ------
    TypeError, ValueError
        Batch size or array tree is invalid, or setup is called with traced arrays.

    Notes
    -----
    Host setup allocates the full bank once. It reads no files and does not
    initialize System memory. Actor-only ownership is a caller precondition;
    arbitrary array names cannot prove that an external tree excludes a critic.
    """
    if type(num_envs) is not int:
        raise TypeError("num_envs must be a Python integer")
    if not 1 <= num_envs <= np.iinfo(np.int32).max:
        raise ValueError("num_envs must be positive and fit int32")
    _array_tree(actor_variables, name="Actor variables")
    if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(actor_variables)):
        raise TypeError("Initialize opponent history on the host before compiled use")

    def asarray(leaf: Array) -> Array:
        """Keep existing device arrays or copy host arrays to numerical storage."""
        return jnp.asarray(leaf)

    def empty_bank(leaf: Array) -> Array:
        """Reserve twenty zero slots with this inference leaf's shape and dtype."""
        return jnp.zeros((_CAPACITY, *leaf.shape), leaf.dtype)

    current = jax.tree.map(asarray, actor_variables)
    return OpponentHistory(
        current,
        jax.tree.map(empty_bank, current),
        jnp.int32(0),
        jnp.full((num_envs,), -1, jnp.int32),
        jnp.int32(0),
        jnp.full((_CAPACITY,), -1, jnp.int32),
        jnp.full((_CAPACITY,), -1, jnp.int32),
        jnp.full((_CAPACITY,), -1, jnp.int32),
        jnp.int32(0),
        jnp.bool_(False),
    )


def assign_opponents(
    history: OpponentHistory,
    reset_mask: Array,
    keys: Array,
    *,
    pinned_share: float = 0.0,
) -> OpponentHistory:
    """Draw current or frozen opponents only for lanes starting another game.

    Parameters
    ----------
    history : OpponentHistory
        Latest numerical bank. Existing valid assignments remain until reset.
    reset_mask : Array
        Boolean (B,) lanes starting games before their next real decision.
    keys : Array
        Independent opponent-stream Threefry keys, typed (B,) or uint32 (B,2).
        Each selected lane splits its key for the 20% historical choice and a
        uniform occupied-slot draw. Other sampling/action streams are untouched.
    pinned_share : float, default=0.0
        Static Python probability, within [0, 0.8], that a reset lane meets
        history slot 0, the pinned first-update actor. Zero keeps the 80%
        current and 20% uniform-history recipe with exactly today's random
        draws. A positive share requires a schedule whose first history
        threshold is round 1, so slot 0 holds the actor after the first completed
        update. Each reset then picks slot 0 with this probability, one of slots
        1 through count-1 uniformly with total probability 0.2 while at least two
        slots exist, and current weights otherwise. While only slot 0 exists,
        that 0.2 goes to current weights. bool and traced values are rejected.

    Returns
    -------
    OpponentHistory
        New assignments for selected lanes only. Empty history selects current
        without drawing. Otherwise each reset has 80% current probability and
        20% historical probability, or the pinned split above; finite counts
        and transition shares differ.
        Invalid numerical history preserves every old value and sets error=True.

    Raises
    ------
    TypeError, ValueError
        History, reset-mask or key shapes/dtypes are incompatible, or
        pinned_share is not a Python number within [0, 0.8].

    Notes
    -----
    Pure JAX work supports jit/scan. No memory is reset here. The collector must
    stop on error before applying a System, including its memory initializer.
    The sticky flag cannot be repaired by a later assignment or refresh. The
    pinned branch is chosen in Python, so a zero share traces today's program.
    """
    size = _history_shapes(history)
    _field(reset_mask, (size,), jnp.bool_, "reset_mask")
    _keys(keys, size)
    _pinned_share(pinned_share)
    invalid = _history_invalid(history)

    def draw(_: None) -> Array:
        """Draw one choice per lane from two independent parts of its own key."""

        def one(key: Array) -> Array:
            """Choose current or one occupied slot using only this lane's key."""
            choose_key, slot_key = jax.random.split(key)
            historical = jax.random.bernoulli(choose_key, 0.2)
            slot = jax.random.randint(slot_key, (), 0, history.count, dtype=jnp.int32)
            return jnp.where(historical, slot, jnp.int32(-1))

        def one_pinned(key: Array) -> Array:
            """Choose the pinned slot, another occupied slot or current weights."""
            choose_key, slot_key = jax.random.split(key)
            chance = jax.random.uniform(choose_key)
            pinned = chance < pinned_share
            historical = ~pinned & (chance < pinned_share + 0.2) & (history.count > 1)
            slot = jax.random.randint(
                slot_key, (), 1, jnp.maximum(history.count, 2), dtype=jnp.int32
            )
            return jnp.where(
                pinned, jnp.int32(0), jnp.where(historical, slot, jnp.int32(-1))
            )

        chooser = one if pinned_share == 0 else one_pinned
        return jnp.where(reset_mask, jax.vmap(chooser)(keys), history.lane_snapshot)

    def unchanged(_: None) -> Array:
        """Keep every assignment when no draw is needed or history is invalid."""
        return history.lane_snapshot

    assignments = cast(
        Array,
        jax.lax.cond(
            ~invalid & (history.count > 0) & jnp.any(reset_mask),
            draw,
            unchanged,
            operand=None,
        ),
    )
    return history._replace(lane_snapshot=assignments, error=invalid)


def refresh_opponents(
    history: OpponentHistory,
    actor_variables: Tree,
    *,
    completed_rounds: Array,
    update_index: Array,
    schedule: ScheduleArrays,
) -> tuple[OpponentHistory, SnapshotEvent]:
    """Publish one completed learner update and capture a due frozen actor once.

    Parameters
    ----------
    history : OpponentHistory
        Latest bank and assignments; this function never resets game memory.
    actor_variables : PyTree
        Already-updated actor-only arrays with exactly the original tree, leaf
        shapes and dtypes. Includes all changing inference state. No learning is
        performed here; critic and optimizer state remain with the learner.
    completed_rounds : Array
        Int32 scalar real rounds after this update. Must be strictly greater
        than last_refresh_rounds and no greater than schedule.total_rounds.
    update_index : Array
        Int32 scalar exactly one greater than current_update, without overflow.
    schedule : ScheduleArrays
        Validated numerical schedule with int32 total_rounds and
        history_threshold_rounds shaped (20,) as built by make_training_schedule:
        the upward-rounded 5% steps, or round 1 then 5% through 95% when early
        history capture is requested. Structure stays fixed under jit.

    Returns
    -------
    tuple[OpponentHistory, SnapshotEvent]
        Refreshed current variables and at most one new slot. Every newly met
        threshold points to that slot. Existing slots and assignments stay fixed.
        Invalid numerical notifications preserve old weights and timing, set the
        sticky error flag and return an absent event (all scalar IDs/counts -1).

    Raises
    ------
    TypeError, ValueError
        A field's static shape/dtype or the actor-variable schema is incompatible.

    Notes
    -----
    Pure JAX work supports jit/scan. Call once after each completed learner update,
    before the next collection block, including a final partial block. Several
    crossed thresholds share one physical snapshot; there is no eviction. The
    final captured actor may receive no later game. All inference leaves freeze
    together; current games use the new values with their continuing memory.
    """
    _history_shapes(history)
    _matching_variables(actor_variables, _schema(history.current_variables))
    _field(completed_rounds, (), jnp.int32, "completed_rounds")
    _field(update_index, (), jnp.int32, "update_index")
    _field(schedule.total_rounds, (), jnp.int32, "total_rounds")
    _field(schedule.history_threshold_rounds, (_CAPACITY,), jnp.int32, "thresholds")
    due = (history.threshold_to_snapshot == -1) & (
        completed_rounds >= schedule.history_threshold_rounds
    )
    capture = jnp.any(due)
    valid = (
        ~_history_invalid(history)
        & (history.current_update < np.iinfo(np.int32).max)
        & (update_index == history.current_update + 1)
        & (completed_rounds > history.last_refresh_rounds)
        & (completed_rounds <= schedule.total_rounds)
        & (~capture | (history.count < _CAPACITY))
    )
    absent = SnapshotEvent(
        jnp.bool_(False),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.zeros((_CAPACITY,), jnp.bool_),
    )

    def publish(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
        """Replace current variables, then append one slot if any threshold is due."""
        refreshed = history._replace(
            current_variables=actor_variables,
            current_update=update_index,
            last_refresh_rounds=completed_rounds,
        )

        def append(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
            """Freeze every inference leaf in the next unused slot."""
            slot = history.count

            def capture_leaf(bank: Array, current: Array) -> Array:
                """Write one actor leaf to the unused slot, retaining older slots."""
                return bank.at[slot].set(current)

            return refreshed._replace(
                historical_variables=jax.tree.map(
                    capture_leaf,
                    history.historical_variables,
                    actor_variables,
                ),
                count=slot + 1,
                captured_rounds=history.captured_rounds.at[slot].set(completed_rounds),
                captured_updates=history.captured_updates.at[slot].set(update_index),
                threshold_to_snapshot=jnp.where(
                    due, slot, history.threshold_to_snapshot
                ),
            ), SnapshotEvent(jnp.bool_(True), slot, completed_rounds, update_index, due)

        def current_only(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
            """Publish new current weights without creating an early snapshot."""
            return refreshed, absent

        return cast(
            tuple[OpponentHistory, SnapshotEvent],
            jax.lax.cond(capture, append, current_only, None),
        )

    def reject(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
        """Keep prior numerical evidence and make the failure sticky."""
        return history._replace(error=jnp.bool_(True)), absent

    return cast(
        tuple[OpponentHistory, SnapshotEvent],
        jax.lax.cond(valid, publish, reject, None),
    )


def _memory(memory: Tree, size: int) -> Tree:
    """Validate numerical memory with a leading lane axis and return it unchanged."""
    _array_tree(memory, name="Opponent memory")
    if any(leaf.ndim < 1 or leaf.shape[0] != size for leaf in jax.tree.leaves(memory)):
        raise ValueError("Opponent memory leaves must have the leading lane axis")
    return memory


def _chosen_variables(history: OpponentHistory, slot: Array) -> Tree:
    """Select one lane's current or occupied frozen tree without a persistent copy.

    slot must be -1 or an occupied slot. The collector checks that precondition
    before entering application. A mixed vmap may materialize temporary selected
    leaves; no selected variables are stored in the returned memory or history.
    """

    def current(_: None) -> Tree:
        """Use this update's shared actor variables."""
        return history.current_variables

    def historical(_: None) -> Tree:
        """Read this lane's occupied immutable snapshot."""

        def select(bank: Array) -> Array:
            """Read one leaf from the assigned slot."""
            return bank[slot]

        return jax.tree.map(select, history.historical_variables)

    return cast(Tree, jax.lax.cond(slot == -1, current, historical, None))


def _add_batch(leaf: Array) -> Array:
    """Give one lane the base System's required leading batch axis."""
    return leaf[None]


def _remove_batch(leaf: Array) -> Array:
    """Remove the length-one batch axis before vmap restores full lane order."""
    return leaf[0]


def make_opponent_system(actor: System) -> System:
    """Wrap a native JAX actor for current or frozen per-game opponent variables.

    Parameters
    ----------
    actor : System
        Lane-independent native JAX System with actor-only array variables,
        ordinary leading-B numerical memory, and no custom reset hook. Applying
        or initializing a lane alone must equal its row in a batch. The actor's
        architecture, input/action schemas and inference mode stay fixed. Only
        changing numerical inference state belongs in the variables tree.

    Returns
    -------
    System
        Stable apply/init callbacks with variables=(). Supply an OpponentHistory
        explicitly as variables_b to init_systems and apply_systems. All-current
        batches call the base actor once on the full batch. Mixed batches map
        independent length-one calls in lane order. Initialization selects each
        lane's assigned variables too. Learning outputs are empty and policy IDs
        are unreported (-1); training records own exact version identities.

    Raises
    ------
    TypeError, ValueError
        actor is not a native JAX System, uses a Policy adapter/custom reset hook,
        or has nonnumeric variables. The returned callbacks reject incompatible
        variable/input/key/memory shapes and concrete invalid snapshot IDs.

    Notes
    -----
    Construct once, outside compiled collection. Callbacks retain stable base
    callables and array schemas, not actor weights. Team B memory remains owned
    by M8's SystemState; current-weight updates, death and respawn do not reset
    it. M8 resets selected memory on the next decision of a new game. Its random
    initializer keys also include allocated episode IDs, so arbitrary stochastic
    initializers are not independent of other lanes' global reset schedules.
    Lane independence and actor-only information ownership are caller contracts,
    not properties that inspecting an arbitrary callback can prove. Traced calls
    require a valid history and the collector's pre-application failure guard.
    Policy adapters and host methods remain supported through their existing M8
    routes, outside this wrapper. No critic, checkpoint loader or learner is called.
    """
    if not isinstance(cast(object, actor), System):
        raise TypeError("actor must be a System")
    if actor.execution != "jax":
        raise ValueError("Opponent history requires a native JAX System")
    if actor._policies:  # pyright: ignore[reportPrivateUsage]
        raise ValueError("Policy adapters must use their existing apply_systems route")
    if actor.reset_memory is not None:
        raise ValueError("Opponent history does not support custom memory reset hooks")
    _array_tree(actor.variables, name="Actor variables")
    expected = _schema(actor.variables)
    base_apply, base_init = actor.apply, actor.init
    components = len(actor.components or ())

    def check(history: OpponentHistory, inputs: SystemInput, keys: Array) -> int:
        """Check numerical structure and concrete IDs before calling the actor."""
        size = _history_shapes(history)
        _matching_variables(history.current_variables, expected)
        if inputs.valid.shape != (size,):
            raise ValueError(
                "Opponent history and inputs must have the same lane count"
            )
        _keys(keys, size)
        invalid = _history_invalid(history)
        if not isinstance(invalid, Tracer) and bool(invalid):
            raise ValueError(
                "Opponent history contains an error or invalid snapshot ID"
            )
        return size

    def initialize(history: OpponentHistory, inputs: SystemInput, keys: Array) -> Tree:
        """Create lane-leading memory using each lane's assigned actor variables."""
        size = check(history, inputs, keys)
        if base_init is None:
            return ()
        initializer = base_init

        def current(_: None) -> Tree:
            """Initialize the full batch once when every lane uses current weights."""
            return _memory(initializer(history.current_variables, inputs, keys), size)

        def mixed(_: None) -> Tree:
            """Initialize chosen variables with a length-one batch per lane."""

            def one(slot: Array, row: SystemInput, key: Array) -> Tree:
                """Initialize one assigned lane with its own input and key."""
                fresh = _memory(
                    initializer(
                        _chosen_variables(history, slot),
                        jax.tree.map(_add_batch, row),
                        key[None],
                    ),
                    1,
                )
                return jax.tree.map(_remove_batch, fresh)

            return jax.vmap(one)(history.lane_snapshot, inputs, keys)

        return cast(
            Tree,
            jax.lax.cond(jnp.all(history.lane_snapshot == -1), current, mixed, None),
        )

    def apply(
        history: OpponentHistory, memory: Tree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        """Apply each lane's assigned actor and return no opponent learning values."""
        size = check(history, inputs, keys)
        _memory(memory, size)

        def normalized(
            variables: Tree, prior: Tree, view: SystemInput, rng: Array
        ) -> SystemOutput:
            """Reuse M8 result meanings and drop opponent learner/component outputs."""
            output = _normal_output(
                base_apply(variables, prior, view, rng),
                view.valid.shape[0],
                component_count=components,
                check_ids=True,
            )
            return SystemOutput(
                output.actions, _memory(output.next_memory, view.valid.shape[0])
            )

        def current(_: None) -> SystemOutput:
            """Apply shared current variables once for the full batch."""
            return normalized(history.current_variables, memory, inputs, keys)

        def mixed(_: None) -> SystemOutput:
            """Map only the assigned actor for each lane."""

            def one(
                slot: Array, prior: Tree, row: SystemInput, key: Array
            ) -> SystemOutput:
                """Apply one assigned actor with a length-one batch."""
                result = normalized(
                    _chosen_variables(history, slot),
                    jax.tree.map(_add_batch, prior),
                    jax.tree.map(_add_batch, row),
                    key[None],
                )
                return jax.tree.map(_remove_batch, result)

            return jax.vmap(one)(history.lane_snapshot, memory, inputs, keys)

        return cast(
            SystemOutput,
            jax.lax.cond(jnp.all(history.lane_snapshot == -1), current, mixed, None),
        )

    return System(f"{actor.name} Training Opponent", apply, init=initialize)
