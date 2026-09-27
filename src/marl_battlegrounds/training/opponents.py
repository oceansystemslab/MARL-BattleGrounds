"""Keep current self-play and frozen actor versions for one training run.

Initialize a bounded rolling bank on the host, draw assignments at game starts,
and publish updated actor variables after learning with refresh_opponents.
make_opponent_system applies those choices through the actor's existing System,
and can play a named pinned System on the lanes assigned to -2 through M8's
own one-team helpers. A pinned host method runs outside compiled code through
HostOpponent, and HOST_ACTIONS_SYSTEM hands its actions to the compiled step.
Recurrent memory belongs to SystemState.team_b, never to this bank. These helpers
do not train, load checkpoints, call a critic, or change simulator rules.
"""

import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import DTypeLike

# One owner for per-team execution: the pinned opponent reuses M8's helpers.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
    _CompositionVariables,
    _default_reset,
    _execution,
    _host_apply,
    _initial_memory,
    _jax_apply,
    _normal_output,
    _SystemExecution,
    select_policy_carry,
)
from marl_battlegrounds.evaluation.system_evaluation import prepare_evaluation_system
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.training.curriculum import ScheduleArrays

type Tree = Any
_CAPACITY = 20


class OpponentHistory(NamedTuple):
    """Keep actor copies and game assignments as a numerical JAX tree.

    Attributes
    ----------
    current_variables, historical_variables : PyTree
        Current actor-only inference leaves and a bank with leading axis C.
        C is keep_past + 1, or zero when keep_past is zero. Copies include
        exploration values. No critic, optimizer or game memory belongs here.
    count : Array
        Int32 scalar resident slot count, including any retired copy. At most
        C - 1 copies are eligible for new games. Slots may be reused only after
        retirement and after all games using them end.
    lane_snapshot : Array
        Int32 (B,) physical choices: -1 current, -2 permanent pin, or bank slot.
        A game keeps its choice until its next reset.
    current_update, last_refresh_rounds : Array
        Int32 scalar most recent accepted learner version and round, initially
        zero. One round means one transition in every environment lane.
    captured_rounds, captured_updates, captured_ids : Array
        Int32 (C,) actual capture boundaries and stable capture IDs. Unused
        slots contain -1. IDs never repeat even when physical slots are reused.
    threshold_to_snapshot : Array
        Int32 (20,) stable capture IDs for explicit thresholds, or -1 if unmet.
        Several crossed thresholds may share one ID. IDs survive slot eviction.
    eligible : Array
        Bool (C,) copies available for new games, always the newest keep_past.
        A retired resident copy remains readable by a continuing game.
    next_capture_id, last_capture_rounds : Array
        Int32 scalars: next unused stable ID and last capture round (-1 if none).
    minimum_capture_rounds, capture_interval_rounds : Array
        Int32 scalars. Captures must be at least minimum_capture_rounds apart.
        A positive interval requests the next capture that many rounds after the
        last actual capture (or run start); zero disables recurring requests.
    pinned_variables, pinned_update : PyTree or None, Array
        A separate permanent first-update actor tree, or None when disabled.
        pinned_update is int32 -1 before capture or when disabled. The pin uses
        no rolling slot and remains available when keep_past is zero.
    external_pin : Array
        Bool scalar: a separate named System owns the permanent-pin route.
        Its variables and memory stay with collection, outside this actor bank.
    capture_capacity : Array
        Int32 scalar total stable capture bound. Counter rows reserve this many IDs.
    error : Array
        Sticky bool scalar. Invalid operations preserve state and set this flag.

    Notes
    -----
    Weights and metadata have fixed shapes during a run. Only current weights,
    the C bank copies and an explicitly requested pin consume actor storage.
    The caller saves capture records and exports before a slot can be reused.
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
    eligible: Array
    captured_ids: Array
    next_capture_id: Array
    last_capture_rounds: Array
    minimum_capture_rounds: Array
    capture_interval_rounds: Array
    pinned_variables: Tree
    pinned_update: Array
    external_pin: Array
    capture_capacity: Array


class SnapshotEvent(NamedTuple):
    """Report one exported-copy boundary without copying actor weights.

    created is bool scalar. slot, rounds, update_index and capture_id are int32
    scalars, all -1 for an absent event. threshold_mask is bool (20,) and marks
    explicit thresholds satisfied by this capture. A recurring capture may
    have an empty mask. Multiply rounds by num_envs for actual transitions.
    Read the weights from slot before processing another learner update.
    """

    created: Array
    slot: Array
    rounds: Array
    update_index: Array
    threshold_mask: Array
    capture_id: Array


def opponent_selection_names(selection: object) -> tuple[str, ...]:
    """Validate training shares or a repeating order and return distinct names.

    None keeps the existing self/history recipe. A mapping gives finite,
    nonnegative relative shares with a positive total. A nonempty sequence
    gives an exact repeating game-start order; repeated names are allowed.
    Names must be nonempty strings. This does not load or resolve any member.
    """
    if selection is None:
        return ()
    if isinstance(selection, Mapping):
        if not selection:
            raise ValueError("Opponent shares must not be empty")
        values = list(cast(Mapping[Any, Any], selection).values())
        if any(
            isinstance(v, bool) or not isinstance(cast(object, v), (int, float))
            for v in values
        ):
            raise TypeError("Opponent shares must be real numbers")
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Opponent shares must be finite and nonnegative")
        if not math.isfinite(sum(values)) or sum(values) <= 0:
            raise ValueError("Opponent shares must have a finite positive total")
        names = tuple(cast(Mapping[Any, Any], selection))
    elif isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        names = tuple(cast(Sequence[Any], selection))
        if not names:
            raise ValueError("Opponent order must not be empty")
    else:
        raise TypeError("Opponent selection must be a share mapping or a name sequence")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Opponent names must be nonempty strings")
    return tuple(dict.fromkeys(names))


class OpponentSelection(NamedTuple):
    """Keep numerical matchmaking settings and the next dense game-start index.

    shares follows self, past, then declared named members. order contains those
    same category indices and repeats from order_start. ordered chooses between
    them. game_starts counts real initial/reset games in ascending lane order;
    the environment's existing int32 episode allocator bounds it. choices keeps
    each lane's external member index. past_ids binds per-slot past_weights to
    the captured IDs they describe; a newly reused slot gets uniform weight.
    """

    shares: Array
    order: Array
    ordered: Array
    game_starts: Array
    order_start: Array
    choices: Array
    past_ids: Array
    past_weights: Array


def configure_opponent_selection(
    names: tuple[str, ...],
    history: OpponentHistory,
    selection: object,
    *,
    previous: OpponentSelection | None = None,
    past: Mapping[int, float] | None = None,
) -> OpponentSelection:
    """Validate host settings and preserve every live game's member and memory.

    names is the fixed external-member declaration order, excluding self/past.
    selection is a share mapping or repeating name sequence. None means 80%
    self and 20% past. A changed sequence starts from its first entry at the
    next game start. Same-shaped share changes keep compiled programs reusable.
    A different sequence length changes shape and may need compilation.
    past optionally gives relative weights by currently eligible capture ID;
    omitted IDs receive zero. None keeps the last weights; newly captured copies
    default to one. All-zero eligible weights fall back to self. No game starts.
    """
    selected = opponent_selection_names(selection)
    if len(set(names)) != len(names) or any(n in ("self", "past") for n in names):
        raise ValueError("Opponent member names must be unique and exclude self/past")
    categories = ("self", "past", *names)
    if any(name not in categories for name in selected):
        raise ValueError("Opponent selection names must refer to declared members")
    if selection is None:
        selection = {"self": 0.8, "past": 0.2}
    ordered = not isinstance(selection, Mapping)
    if ordered:
        order = np.asarray(
            [categories.index(n) for n in cast(Sequence[str], selection)], np.int32
        )
        shares = np.zeros(len(categories), np.float32)
    else:
        mapping = cast(Mapping[str, float], selection)
        values = np.asarray([mapping.get(n, 0.0) for n in categories], np.float64)
        shares = (values / values.sum()).astype(np.float32)
        order = np.zeros(1, np.int32)
    starts = jnp.int32(0) if previous is None else previous.game_starts
    origin = starts
    if (
        previous is not None
        and bool(previous.ordered) == ordered
        and np.array_equal(previous.order, order)
    ):
        origin = previous.order_start
    ids = np.asarray(history.captured_ids)
    weights = np.ones(ids.shape, np.float32)
    if previous is not None:
        old = dict(
            zip(
                np.asarray(previous.past_ids).tolist(),
                np.asarray(previous.past_weights).tolist(),
                strict=True,
            )
        )
        weights = np.asarray(
            [old.get(int(i), 1.0) if i >= 0 else 1.0 for i in ids], np.float32
        )
    if past is not None:
        eligible = set(ids[np.asarray(history.eligible)].tolist())
        if any(type(i) is not int or i not in eligible for i in past):
            raise ValueError("Past weights must name currently eligible capture IDs")
        if any(
            isinstance(v, bool)
            or not isinstance(cast(object, v), (int, float))
            or not math.isfinite(v)
            or v < 0
            for v in past.values()
        ):
            raise ValueError("Past weights must be finite nonnegative numbers")
        weights = np.asarray([past.get(int(i), 0.0) for i in ids], np.float32)
        if not np.all(np.isfinite(weights)):
            raise ValueError("Past weights must fit float32")
    return OpponentSelection(
        jnp.asarray(shares),
        jnp.asarray(order),
        jnp.bool_(ordered),
        starts,
        origin,
        jnp.zeros(history.lane_snapshot.shape, jnp.int32)
        if previous is None
        else previous.choices,
        jnp.asarray(ids),
        jnp.asarray(weights),
    )


def select_opponents(
    history: OpponentHistory,
    settings: OpponentSelection,
    reset_mask: Array,
    keys: Array,
) -> tuple[OpponentHistory, OpponentSelection]:
    """Assign only real game starts, reusing the pool's full-batch member memory.

    reset_mask is bool (B,), keys are B Threefry keys. Return history with
    physical self/past/external choices and settings with external member indices
    and the advanced dense count. Continuing lanes are unchanged. Missing past
    copies or zero eligible past weight falls back to self; padding calls none.
    """
    offsets = jnp.cumsum(reset_mask.astype(jnp.int32)) - 1
    indices = settings.game_starts - settings.order_start + offsets
    order = settings.order[jnp.mod(indices, settings.order.shape[0])]
    split = jax.vmap(partial(jax.random.split, num=2))(keys)

    def draw(_: None) -> Array:
        """Draw categories only for weighted selection."""
        return jax.vmap(
            partial(jax.random.categorical, logits=jnp.log(settings.shares))
        )(split[:, 0]).astype(jnp.int32)

    def in_order(_: None) -> Array:
        """Read the repeating sequence without a category draw."""
        return order

    category = cast(Array, jax.lax.cond(settings.ordered, in_order, draw, None))
    slots = jnp.full(reset_mask.shape, -1, jnp.int32)
    if history.captured_ids.shape[0]:
        weights = jnp.where(
            history.captured_ids == settings.past_ids, settings.past_weights, 1.0
        )
        weights = jnp.where(history.eligible, weights, 0.0)
        available = jnp.sum(weights) > 0
        safe = jnp.where(available, weights, jnp.ones_like(weights))
        past_slots = jax.vmap(partial(jax.random.categorical, logits=jnp.log(safe)))(
            split[:, 1]
        ).astype(jnp.int32)
        slots = jnp.where((category == 1) & available, past_slots, slots)
    slots = jnp.where(category >= 2, -2, slots)
    return history._replace(
        lane_snapshot=jnp.where(reset_mask, slots, history.lane_snapshot)
    ), settings._replace(
        game_starts=settings.game_starts + jnp.sum(reset_mask, dtype=jnp.int32),
        choices=jnp.where(reset_mask & (category >= 2), category - 2, settings.choices),
    )


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
        The share to check. Zero reserves no probability for a permanent pin.

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
    """Check fixed field shapes and return B without reading device values."""
    if not isinstance(cast(object, history), OpponentHistory):
        raise TypeError("Opponent variables must be an OpponentHistory")
    if history.lane_snapshot.ndim != 1 or not history.lane_snapshot.shape[0]:
        raise ValueError("lane_snapshot must have a nonempty lane axis")
    size = history.lane_snapshot.shape[0]
    if history.captured_ids.ndim != 1:
        raise ValueError("captured_ids must have a one-dimensional slot axis")
    capacity = history.captured_ids.shape[0]
    if capacity == 1:
        raise ValueError("A rolling bank needs keep_past + 1 slots, or zero slots")
    _field(history.lane_snapshot, (size,), jnp.int32, "lane_snapshot")
    for name in (
        "count",
        "current_update",
        "last_refresh_rounds",
        "next_capture_id",
        "last_capture_rounds",
        "minimum_capture_rounds",
        "capture_interval_rounds",
        "pinned_update",
        "capture_capacity",
    ):
        _field(getattr(history, name), (), jnp.int32, name)
    _field(history.error, (), jnp.bool_, "error")
    _field(history.external_pin, (), jnp.bool_, "external_pin")
    _field(history.eligible, (capacity,), jnp.bool_, "eligible")
    for name in ("captured_rounds", "captured_updates", "captured_ids"):
        _field(getattr(history, name), (capacity,), jnp.int32, name)
    _field(
        history.threshold_to_snapshot, (_CAPACITY,), jnp.int32, "threshold_to_snapshot"
    )
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
        if bank.shape != (capacity, *current.shape) or bank.dtype != current.dtype:
            raise ValueError(
                "Historical leaves need the bank capacity and actor schema"
            )
    if history.pinned_variables is not None:
        _matching_variables(
            history.pinned_variables, _schema(history.current_variables)
        )
    return size


def _history_invalid(history: OpponentHistory) -> Array:
    """Reject malformed slot ownership and counters after static shape checks.

    Retired copies remain valid for live games. This compiled guard never clips
    or repairs an invalid assignment. A permanent pin requires a captured tree.
    """
    capacity = history.captured_ids.shape[0]
    occupied = history.captured_ids >= 0
    slots = history.lane_snapshot
    pin_ready = history.external_pin | (
        (history.pinned_variables is not None) & (history.pinned_update >= 0)
    )
    invalid_slot = (slots < -2) | ((slots == -2) & ~pin_ready) | (slots >= capacity)
    if capacity:
        invalid_slot |= (slots >= 0) & ~occupied[jnp.clip(slots, 0, capacity - 1)]
    return (
        history.error
        | (history.count != jnp.sum(occupied))
        | (jnp.sum(history.eligible) > max(capacity - 1, 0))
        | jnp.any(history.eligible & ~occupied)
        | jnp.any(history.captured_ids < -1)
        | jnp.any(history.captured_ids >= history.next_capture_id)
        | (history.next_capture_id < 0)
        | (history.next_capture_id > history.capture_capacity)
        | (history.capture_capacity < 0)
        | (history.current_update < 0)
        | (history.last_refresh_rounds < 0)
        | (history.minimum_capture_rounds < 1)
        | (history.capture_interval_rounds < 0)
        | jnp.any(invalid_slot)
    )


def _keys(keys: Array, size: int) -> None:
    """Require B typed Threefry keys or legacy uint32 keys shaped (B,2)."""
    if str(jax.random.key_impl(keys)) != "threefry2x32":
        raise ValueError("Opponent selection requires Threefry keys")
    if jax.random.key_data(keys).shape != (size, 2):
        raise ValueError("Opponent keys must contain one key per lane")


def init_opponent_history(
    actor_variables: Tree,
    *,
    num_envs: int,
    keep_past: int = 20,
    minimum_capture_rounds: int = 1,
    capture_interval_rounds: int = 0,
    pin_first_update: bool = False,
    external_pin: bool = False,
    capture_capacity: int = 2147483646,
) -> OpponentHistory:
    """Reserve a rolling actor bank on the host without drawing assignments.

    Parameters
    ----------
    actor_variables : PyTree
        Numerical actor-only inference arrays with fixed shapes and dtypes.
        Empty () is valid. Critic, optimizer and memory must stay outside.
    num_envs : int
        Positive fixed lane count B, excluding bool and fitting int32.
    keep_past : int, default=20
        Number of newest copies offered to new games. Reserve one extra slot
        for a retired copy still playing. Zero allocates no bank and makes past
        selection fall back to current weights. Must be nonnegative and fit int32.
    minimum_capture_rounds : int, default=1
        Minimum distance between actual captures. Collection must supply the
        maximum game horizon, not the learner rollout length. Positive int32.
    capture_interval_rounds : int, default=0
        Recurring capture interval in rounds, alongside explicit thresholds.
        Zero disables recurring requests. Positive values must be at least the
        minimum distance. The public runner converts total transitions to rounds
        and validates actual learner publication boundaries before this call.
    pin_first_update : bool, default=False
        Reserve a separate permanent actor copy at the first accepted update.
        It does not count against keep_past or the rolling capture interval.
    external_pin : bool, default=False
        A separate named System owns assignment -2 from the first game. Cannot
        be combined with pin_first_update; collection owns its values and memory.

    capture_capacity : int, default=2147483646
        Maximum total captures. Collection supplies its fixed counter capacity.

    Returns
    -------
    OpponentHistory
        Empty bank, B current assignments, zero counters and error=False.
        No file access, random draws or System memory initialization occurs.

    Raises
    ------
    TypeError, ValueError
        Invalid integer controls, array leaves or traced setup inputs.
    """
    for name, value, lower in (
        ("num_envs", num_envs, 1),
        ("keep_past", keep_past, 0),
        ("capture_capacity", capture_capacity, 0),
        ("minimum_capture_rounds", minimum_capture_rounds, 1),
        ("capture_interval_rounds", capture_interval_rounds, 0),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be a Python integer")
        if not lower <= value < np.iinfo(np.int32).max:
            raise ValueError(
                f"{name} must be {'positive' if lower else 'nonnegative'} and fit int32"
            )
    if type(pin_first_update) is not bool or type(external_pin) is not bool:
        raise TypeError("pin_first_update and external_pin must be Python bools")
    if pin_first_update and external_pin:
        raise ValueError("A permanent pin must use one actor or one named System")
    if capture_interval_rounds and capture_interval_rounds < minimum_capture_rounds:
        raise ValueError("Capture interval must be at least the maximum game horizon")
    _array_tree(actor_variables, name="Actor variables")
    if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(actor_variables)):
        raise TypeError("Initialize opponent history on the host before compiled use")
    capacity = keep_past + 1 if keep_past else 0

    def asarray(leaf: Array) -> Array:
        """Keep device leaves or copy host leaves into numerical arrays."""
        return jnp.asarray(leaf)

    current = jax.tree.map(asarray, actor_variables)

    def empty_bank(leaf: Array) -> Array:
        """Reserve the rolling slots for one inference leaf."""
        return jnp.zeros((capacity, *leaf.shape), leaf.dtype)

    return OpponentHistory(
        current,
        jax.tree.map(empty_bank, current),
        jnp.int32(0),
        jnp.full((num_envs,), -1, jnp.int32),
        jnp.int32(0),
        jnp.full((capacity,), -1, jnp.int32),
        jnp.full((capacity,), -1, jnp.int32),
        jnp.full((_CAPACITY,), -1, jnp.int32),
        jnp.int32(0),
        jnp.bool_(False),
        jnp.zeros((capacity,), jnp.bool_),
        jnp.full((capacity,), -1, jnp.int32),
        jnp.int32(0),
        jnp.int32(-1),
        jnp.int32(minimum_capture_rounds),
        jnp.int32(capture_interval_rounds),
        current if pin_first_update else None,
        jnp.int32(-1),
        jnp.bool_(external_pin),
        jnp.int32(capture_capacity),
    )


def assign_opponents(
    history: OpponentHistory,
    reset_mask: Array,
    keys: Array,
    *,
    pinned_share: float = 0.0,
    past_share: float | Array = 0.2,
    past_weights: Array | None = None,
) -> OpponentHistory:
    """Choose current, permanent pin or eligible past copies for new games.

    Parameters
    ----------
    history : OpponentHistory
        Latest bank. Continuing lanes keep their physical slot unchanged.
    reset_mask : Array
        Bool (B,) lanes starting a game, including initial games.
    keys : Array
        Independent Threefry keys, typed (B,) or legacy uint32 (B,2).
    pinned_share : float, default=0.0
        Static Python probability in [0, 0.8] for the permanent first-update
        actor at assignment -2. Before that capture its share goes to current.
    past_share : float or Array, default=0.2
        Scalar probability in [0, 1 - pinned_share]. Empty past selection falls
        back to current. Numeric values remain dynamic under jit.
    past_weights : Array or None, default=None
        Optional floating (C,) nonnegative finite weights indexed by physical
        slot. Only eligible slots participate. None selects them uniformly.
        All eligible weights zero sends the past share to current. The caller
        updates slot weights from captured_ids when slots are reused.

    Returns
    -------
    OpponentHistory
        Reset-lane assignments, with no memory reset or weight changes. Bad
        numeric state or probabilities preserve assignments and set error=True.
        Bad static shapes, dtypes or key types raise TypeError or ValueError.
    """
    size = _history_shapes(history)
    capacity = history.captured_ids.shape[0]
    _field(reset_mask, (size,), jnp.bool_, "reset_mask")
    _keys(keys, size)
    _pinned_share(pinned_share)
    share = jnp.asarray(past_share, jnp.float32)
    _field(share, (), jnp.float32, "past_share")
    invalid = (
        _history_invalid(history)
        | ~jnp.isfinite(share)
        | (share < 0)
        | (share + pinned_share > 1)
    )
    weights = history.eligible.astype(jnp.float32)
    if past_weights is not None:
        if past_weights.shape != (capacity,) or not jnp.issubdtype(
            past_weights.dtype, jnp.floating
        ):
            raise ValueError(
                "past_weights must be floating values with one entry per physical slot"
            )
        invalid |= jnp.any(~jnp.isfinite(past_weights) | (past_weights < 0))
        weights = jnp.where(history.eligible, past_weights, 0)
    available = jnp.sum(weights) > 0
    pin_ready = history.external_pin | (
        (history.pinned_variables is not None) & (history.pinned_update >= 0)
    )

    def draw(_: None) -> Array:
        """Draw independently per reset lane without changing other RNG streams."""

        def one(key: Array) -> Array:
            """Split this game's key between member choice and past-copy choice."""
            choose_key, slot_key = jax.random.split(key)
            if pinned_share:
                chance = jax.random.uniform(choose_key)
                pinned = (chance < pinned_share) & pin_ready
                historical = (chance >= pinned_share) & (chance < pinned_share + share)
            else:
                pinned = jnp.bool_(False)
                historical = jax.random.bernoulli(choose_key, share)
            slot = jnp.int32(-1)
            if capacity:
                if past_weights is None:
                    eligible = jnp.nonzero(
                        history.eligible, size=capacity, fill_value=0
                    )[0]
                    index = jax.random.randint(
                        slot_key, (), 0, jnp.maximum(jnp.sum(history.eligible), 1)
                    )
                    slot = eligible[index]
                else:
                    slot = jax.random.categorical(slot_key, jnp.log(weights)).astype(
                        jnp.int32
                    )
            return jnp.where(
                pinned,
                jnp.int32(-2),
                jnp.where(historical & available, slot, jnp.int32(-1)),
            )

        return jnp.where(reset_mask, jax.vmap(one)(keys), history.lane_snapshot)

    def unchanged(_: None) -> Array:
        """Preserve game assignments when input values are invalid."""
        return history.lane_snapshot

    assignments = cast(
        Array, jax.lax.cond(~invalid & jnp.any(reset_mask), draw, unchanged, None)
    )
    return history._replace(lane_snapshot=assignments, error=invalid)


def opponent_counter_row(snapshot: Array, capture_capacity: int | Array = 20) -> Array:
    """Map stable opponent identities to counter rows without changing identity.

    snapshot is an int32 array: -1 current, -2 permanent pin, or a nonnegative
    capture ID. Values -3-index name declared external members. Return int32
    rows: 0 current, 1 pin, ID+2 history, then the named members after the
    capture_capacity reserved rows.
    The caller owns the counter capacity and validates all supplied identities.
    """
    return jnp.where(
        snapshot < -2,
        capture_capacity + 2 + (-snapshot - 3),
        jnp.where(snapshot == -1, 0, jnp.where(snapshot == -2, 1, snapshot + 2)),
    )


def opponent_identity(history: OpponentHistory) -> tuple[Array, Array]:
    """Return each lane's stable snapshot ID and actual actor update index.

    Physical slots are replaced by captured_ids. -1 names current weights and
    -2 the permanent pin. A named external pin has update -2; a first-update
    pin keeps its captured update. Both outputs are int32 (B,). Empty banks
    work without indexing an empty leaf. This does not read host values.
    """
    slot = history.lane_snapshot
    snapshot = slot
    update = jnp.full_like(slot, history.current_update)
    if history.captured_ids.shape[0]:
        index = jnp.maximum(slot, 0)
        snapshot = jnp.where(slot >= 0, history.captured_ids[index], slot)
        update = jnp.where(slot >= 0, history.captured_updates[index], update)
    return snapshot, jnp.where(
        slot == -2, jnp.where(history.external_pin, -2, history.pinned_update), update
    )


def _capture_request(
    history: OpponentHistory, rounds: Array, schedule: ScheduleArrays
) -> tuple[Array, Array]:
    """Return explicit due thresholds and whether either capture route is due."""
    due = (history.threshold_to_snapshot == -1) & (
        rounds >= schedule.history_threshold_rounds
    )
    if schedule.history_threshold_count is not None:
        due &= jnp.arange(_CAPACITY) < schedule.history_threshold_count
    interval = history.capture_interval_rounds
    recurring = (interval > 0) & (
        rounds - jnp.maximum(history.last_capture_rounds, 0) >= interval
    )
    return due, (jnp.any(due) | recurring) & (history.captured_ids.shape[0] > 0)


def _free_history_slots(history: OpponentHistory, live_slots: Array | None) -> Array:
    """Find retired or unused slots with no opponent or extra partner references.

    live_slots is optional bool (C,) usage from every other team or owner. The
    built-in opponent assignments are always included, even when it is supplied.
    """
    capacity = history.captured_ids.shape[0]
    used = (
        jnp.bincount(
            jnp.maximum(history.lane_snapshot, 0),
            weights=(history.lane_snapshot >= 0).astype(jnp.int32),
            length=capacity,
        )
        > 0
    )
    if live_slots is not None:
        _field(live_slots, (capacity,), jnp.bool_, "live_slots")
        used |= live_slots
    return ~history.eligible & ~used


def _capture_ready(
    history: OpponentHistory,
    rounds: Array,
    schedule: ScheduleArrays,
    live_slots: Array | None,
) -> tuple[Array, Array]:
    """Keep requests pending until spacing and both-team slot safety allow capture."""
    due, requested = _capture_request(history, rounds, schedule)
    spaced = (history.last_capture_rounds < 0) | (
        rounds - history.last_capture_rounds >= history.minimum_capture_rounds
    )
    return due, requested & spaced & jnp.any(_free_history_slots(history, live_slots))


def publication_valid(
    history: OpponentHistory,
    *,
    completed_rounds: Array,
    update_index: Array,
    schedule: ScheduleArrays,
    live_slots: Array | None = None,
) -> Array:
    """Check the shared publisher rule before committing a learner update.

    history and schedule have already passed host setup. completed_rounds and
    update_index are int32 scalars: a strictly later real round within the
    schedule and exactly the next learner version. live_slots is optional bool
    (C,) usage by partners or other owners; opponent references are included.
    Returns a bool scalar under jit. Capture timing and occupied retired slots
    may delay a copy, but never reject a valid learner update. An exhausted
    stable-ID counter rejects a ready capture. Shape errors raise TypeError
    or ValueError. No state changes.
    """
    _history_shapes(history)
    _field(completed_rounds, (), jnp.int32, "completed_rounds")
    _field(update_index, (), jnp.int32, "update_index")
    _field(schedule.total_rounds, (), jnp.int32, "total_rounds")
    _field(schedule.history_threshold_rounds, (_CAPACITY,), jnp.int32, "thresholds")
    _, capture = _capture_ready(history, completed_rounds, schedule, live_slots)
    return (
        ~_history_invalid(history)
        & (history.current_update < np.iinfo(np.int32).max)
        & (update_index == history.current_update + 1)
        & (completed_rounds > history.last_refresh_rounds)
        & (completed_rounds <= schedule.total_rounds)
        & (~capture | (history.next_capture_id < history.capture_capacity))
    )


def refresh_opponents(
    history: OpponentHistory,
    actor_variables: Tree,
    *,
    completed_rounds: Array,
    update_index: Array,
    schedule: ScheduleArrays,
    live_slots: Array | None = None,
) -> tuple[OpponentHistory, SnapshotEvent]:
    """Publish one actor update and freeze a due copy without changing live games.

    Parameters
    ----------
    history : OpponentHistory
        Latest rolling bank. Its slot assignments and game memory stay fixed.
    actor_variables : PyTree
        Updated actor-only inference arrays with the same leaf schema.
    completed_rounds, update_index : Array
        Int32 scalar actual publication round and consecutive learner version.
    schedule : ScheduleArrays
        Validated total and explicit capture thresholds. Crossed thresholds
        share one capture. Recurring requests use the interval stored in history.
    live_slots : Array or None, default=None
        Bool (C,) extra references, including partners. Opponent assignments
        are always protected. The caller must include every other live owner.

    Returns
    -------
    tuple[OpponentHistory, SnapshotEvent]
        New current weights and at most one rolling capture. It retires the
        oldest eligible copy when needed. All still-referenced weights remain
        unchanged. Every event needs exporting before the next publication.
        Invalid publication preserves all prior values and sets error=True.
        See publication_valid for the single shared acceptance rule.

    Notes
    -----
    Call after each accepted learner update, including a final partial block.
    Pure JAX supports jit/scan. Setup checks explicit capture spacing. At run
    time, a due request waits for the minimum actual gap and a free safe slot.
    Current weights still publish, and the next eligible update captures its
    actor. Recurring cadence starts again at that actual capture round.
    """
    _matching_variables(actor_variables, _schema(history.current_variables))
    valid = publication_valid(
        history,
        completed_rounds=completed_rounds,
        update_index=update_index,
        schedule=schedule,
        live_slots=live_slots,
    )
    due, capture = _capture_ready(history, completed_rounds, schedule, live_slots)
    absent = SnapshotEvent(
        jnp.bool_(False),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.int32(-1),
        jnp.zeros((_CAPACITY,), jnp.bool_),
        jnp.int32(-1),
    )

    def publish(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
        """Publish current and the separate first-update pin before a rolling copy."""
        pinned = history.pinned_variables
        if pinned is not None:

            def first_pin(new: Array, old: Array) -> Array:
                """Freeze this inference leaf only at the first publication."""
                return jnp.where(history.pinned_update < 0, new, old)

            pinned = jax.tree.map(first_pin, actor_variables, pinned)
        refreshed = history._replace(
            current_variables=actor_variables,
            current_update=update_index,
            last_refresh_rounds=completed_rounds,
            pinned_variables=pinned,
            pinned_update=jnp.where(
                (pinned is not None) & (history.pinned_update < 0),
                update_index,
                history.pinned_update,
            ),
        )
        capacity = history.captured_ids.shape[0]
        if not capacity:
            return refreshed, absent

        def append(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
            """Use a free slot and retire the oldest copy when the window is full."""
            slot = jnp.argmax(_free_history_slots(history, live_slots)).astype(
                jnp.int32
            )
            oldest = jnp.argmin(
                jnp.where(
                    history.eligible, history.captured_ids, np.iinfo(np.int32).max
                )
            )
            eligible = (
                history.eligible.at[oldest]
                .set(
                    history.eligible[oldest]
                    & (jnp.sum(history.eligible) < capacity - 1)
                )
                .at[slot]
                .set(True)
            )

            def capture_leaf(bank: Array, current: Array) -> Array:
                """Replace one unused or safely retired inference leaf."""
                return bank.at[slot].set(current)

            copied = jax.tree.map(
                capture_leaf, history.historical_variables, actor_variables
            )
            return refreshed._replace(
                historical_variables=copied,
                count=history.count
                + (history.captured_ids[slot] < 0).astype(jnp.int32),
                eligible=eligible,
                captured_ids=history.captured_ids.at[slot].set(history.next_capture_id),
                captured_rounds=history.captured_rounds.at[slot].set(completed_rounds),
                captured_updates=history.captured_updates.at[slot].set(update_index),
                threshold_to_snapshot=jnp.where(
                    due, history.next_capture_id, history.threshold_to_snapshot
                ),
                next_capture_id=history.next_capture_id + 1,
                last_capture_rounds=completed_rounds,
            ), SnapshotEvent(
                jnp.bool_(True),
                slot,
                completed_rounds,
                update_index,
                due,
                history.next_capture_id,
            )

        def current_only(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
            """Publish current weights while no copy is ready."""
            return refreshed, absent

        return cast(
            tuple[OpponentHistory, SnapshotEvent],
            jax.lax.cond(capture, append, current_only, None),
        )

    def reject(_: None) -> tuple[OpponentHistory, SnapshotEvent]:
        """Keep prior state and make a bad publication sticky."""
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

    slot must be -1, a ready permanent pin (-2), or an occupied slot. Collection
    checks that precondition before application. A mixed vmap may materialize
    temporary selected leaves. No selected variables are stored in returned memory
    or history.
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

    def unpinned(_: None) -> Tree:
        """Read current or rolling weights without indexing an empty bank."""
        if not history.captured_ids.shape[0]:
            return history.current_variables
        return cast(Tree, jax.lax.cond(slot == -1, current, historical, None))

    def pinned(_: None) -> Tree:
        """Read the separate permanent actor copy."""
        return history.pinned_variables

    if history.pinned_variables is not None:
        return cast(Tree, jax.lax.cond(slot == -2, pinned, unpinned, None))
    return unpinned(None)


def _add_batch(leaf: Array) -> Array:
    """Give one lane the base System's required leading batch axis."""
    return leaf[None]


def _remove_batch(leaf: Array) -> Array:
    """Remove the length-one batch axis before vmap restores full lane order."""
    return leaf[0]


def _compact_capacity(size: int) -> int:
    """Return how many pinned games the compact route holds: a quarter, at least one.

    Parameters
    ----------
    size : int
        Static number of games B, positive.

    Returns
    -------
    int
        ceil(size / 4), at least 1. A pinned share of 0.1 at B=512 puts about
        51 games on the pinned System, well inside 128. Larger pin counts
        use the next distinct capacity: half the batch, then the full batch.
    """
    return max(1, -(-size // 4))


def _gather_rows(tree: Tree, index: Array, size: int) -> Tree:
    """Take the rows named by index from every leading-B leaf of a tree.

    Parameters
    ----------
    tree : Any
        Pytree whose leaves all have leading axis B; typed key arrays work too.
    index : Array
        Int32 (C,) row numbers. Entries equal to size mark unused slots.
    size : int
        Static B.

    Returns
    -------
    Any
        The same tree with leading axis C. Unused slots read row B - 1; the
        caller marks them invalid. Pure; no host transfer.
    """
    safe = jnp.minimum(index, size - 1)

    def take(leaf: Array) -> Array:
        """Read the selected rows of one leaf."""
        return leaf[safe]

    return jax.tree.map(take, tree)


def _scatter_rows(full: Tree, part: Tree, index: Array) -> Tree:
    """Write rows of part back into full at index; unused slots are dropped.

    Parameters
    ----------
    full : Any
        Pytree with leading-B leaves.
    part : Any
        The same tree structure with leading-C leaves of matching dtypes.
    index : Array
        Int32 (C,) target rows; entries equal to B are out of range.

    Returns
    -------
    Any
        full with the indexed rows replaced. Out-of-range entries are dropped,
        so padding never overwrites a real game. Pure.
    """

    def put(whole: Array, rows: Array) -> Array:
        """Place one leaf's rows."""
        return whole.at[index].set(rows, mode="drop")

    return jax.tree.map(put, full, part)


def make_opponent_system(
    actor: System, *, pinned: System | None = None, vectorize_lanes: bool = True
) -> System:
    """Wrap a native JAX actor for current or frozen per-game opponent variables.

    Parameters
    ----------
    actor : System
        Lane-independent native JAX System with actor-only array variables,
        ordinary leading-B numerical memory, and no custom reset hook. Applying
        or initializing a lane alone must equal its row in a batch. The actor's
        architecture, input/action schemas and inference mode stay fixed. Only
        changing numerical inference state belongs in the variables tree.
    pinned : System or None, default None
        Keyword-only. None returns exactly today's wrapper. A frozen JAX-execution
        System, or a Policy adapted with shared_policy, plays the lanes whose
        assignment is -2 instead of the network. Its memory may have any
        layout M8 allows, with or without its own reset_memory hook. A host
        method is not passed here: the collection passes HOST_ACTIONS_SYSTEM
        and runs the host method through HostOpponent.

    vectorize_lanes : bool, default=True
        Map mixed history lanes with vmap. Built-in grouped models use False
        because their grouped dot primitive lacks this extra batching rule;
        lax.map then preserves their results. All-current batches still call the
        actor once. The shared-model path keeps its original compiled program.

    Returns
    -------
    System
        Stable apply/init callbacks with variables=(). Without a pinned System,
        supply an OpponentHistory explicitly as variables_b to init_systems and
        apply_systems; with one, supply (history, pinned variables, pinned memory
        template), and the memory is (network memory, pinned memory).
        All-current batches call the base actor once on the full batch. Mixed
        batches map independent length-one calls in lane order; with a pinned
        System, permanent-pin lanes count as current for the network, whose output
        there is discarded. Initialization selects each lane's assigned
        variables too. Learning outputs are empty and policy IDs are unreported
        (-1); training records own exact version identities.

    Raises
    ------
    TypeError, ValueError
        actor is not a native JAX System, uses a Policy adapter/custom reset hook,
        or has nonnumeric variables; pinned is not a System or uses host
        execution. The returned callbacks reject incompatible variable/input/
        key/memory shapes and concrete invalid snapshot IDs.

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
    A pinned System follows M8's rules for one team: it sees only its own lanes
    as valid (other lanes are padding to it), its memory is initialized by M8's
    outer reset through this wrapper's init for pinned lanes of new games, and
    its step runs through M8's _jax_apply with the same keys the network gets.
    A whole-batch lax.cond skips either actor when no valid lane uses it. A Policy
    adapter, which acts per actor, then runs on the pinned games only, gathered
    into the smallest fitting quarter, half or full batch (rounded up);
    a generic System, which may look across the batch, runs on the full batch
    with the other games marked invalid. The actions are merged per game across
    all five actors. A wrapper reset hook exists only when the pinned System has
    its own; that hook receives M8's reset mask for every new game, and for
    games it does not play the fresh memory is its idle placeholder.
    No critic, checkpoint loader or learner is called.
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

            if vectorize_lanes:
                return jax.vmap(one)(history.lane_snapshot, inputs, keys)

            def row(values: tuple[Array, SystemInput, Array]) -> Tree:
                """Keep grouped primitives outside an extra vmap axis."""
                return one(*values)

            return cast(Tree, jax.lax.map(row, (history.lane_snapshot, inputs, keys)))

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

            if vectorize_lanes:
                return jax.vmap(one)(history.lane_snapshot, memory, inputs, keys)

            def row(values: tuple[Array, Tree, SystemInput, Array]) -> SystemOutput:
                """Keep grouped primitives outside an extra vmap axis."""
                return one(*values)

            return cast(
                SystemOutput,
                jax.lax.map(row, (history.lane_snapshot, memory, inputs, keys)),
            )

        return cast(
            SystemOutput,
            jax.lax.cond(jnp.all(history.lane_snapshot == -1), current, mixed, None),
        )

    if pinned is None:
        return System(f"{actor.name} Training Opponent", apply, init=initialize)
    if not isinstance(cast(object, pinned), System):
        raise TypeError("pinned must be a System")
    if pinned.execution != "jax":
        raise ValueError(
            "A pinned System inside this wrapper must use JAX execution; host "
            "methods run through collect_training_rollout's host route"
        )
    pinned_execution = _execution(pinned)
    pinned_reset = pinned.reset_memory
    # A Policy acts per actor, so running it on the pinned games alone gives
    # exactly the actions it gives them inside the full batch. A generic
    # System may legally look across the batch, so it keeps the full batch.
    pinned_compact = bool(pinned._policies)  # pyright: ignore[reportPrivateUsage]

    def network_view(history: OpponentHistory) -> OpponentHistory:
        """Give pinned lanes current weights; their network output is discarded."""
        snapshot = jnp.where(history.lane_snapshot == -2, -1, history.lane_snapshot)
        return history._replace(lane_snapshot=snapshot)

    def unpack(variables: Tree) -> tuple[OpponentHistory, Tree, Tree]:
        """Split (history, pinned variables, pinned memory template)."""
        if (
            not isinstance(variables, tuple)
            or len(cast(tuple[Tree, ...], variables)) != 3
        ):
            raise ValueError(
                "A pinned opponent wrapper needs variables_b=(history, pinned "
                "variables, pinned memory template)"
            )
        history, pinned_variables, template = cast(tuple[Tree, Tree, Tree], variables)
        return cast(OpponentHistory, history), pinned_variables, template

    def initialize_pinned(variables: Tree, inputs: SystemInput, keys: Array) -> Tree:
        """Create (network memory, pinned memory), the pinned half for pinned lanes."""
        history, pinned_variables, template = unpack(variables)
        check(history, inputs, keys)
        network = initialize(network_view(history), inputs, keys)
        use = inputs.valid & (history.lane_snapshot == -2)
        fresh = _initial_memory(
            pinned_execution,
            pinned_variables,
            template,
            inputs._replace(valid=use),
            keys,
        )
        return (network, fresh)

    def apply_pinned(
        variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        """Apply the network, then the pinned System on its lanes, per game."""
        history, pinned_variables, template = unpack(variables)
        size = check(history, inputs, keys)
        if not isinstance(memory, tuple) or len(cast(tuple[Tree, ...], memory)) != 2:
            raise ValueError(
                "A pinned opponent's memory must be (network memory, pinned memory)"
            )
        network_memory, pinned_memory = cast(tuple[Tree, Tree], memory)
        use = inputs.valid & (history.lane_snapshot == -2)
        native_inputs = inputs._replace(valid=inputs.valid & ~use)

        def native(_: None) -> SystemOutput:
            """Apply the learning actor only when a native opponent needs it."""
            return apply(network_view(history), network_memory, native_inputs, keys)

        def no_native(_: None) -> SystemOutput:
            """Keep unused native memory and skip the actor for all-external games."""
            zero = jnp.zeros((size, 5), jnp.int32)
            return SystemOutput(ActorAction(zero, zero, zero), network_memory)

        network = cast(
            SystemOutput,
            jax.lax.cond(jnp.any(native_inputs.valid), native, no_native, None),
        )
        next_network = network.next_memory
        if pinned_reset is not None:
            # A wrapper reset hook turns off M8's padding protection, so keep
            # invalid lanes of the network half here.
            next_network = select_policy_carry(
                native_inputs.valid, next_network, network_memory
            )
        use = inputs.valid & (history.lane_snapshot == -2)

        def run(_: None) -> tuple[ActorAction, Tree]:
            """Apply the pinned System to its lanes with M8's one-team rules."""
            output = _jax_apply(
                pinned_execution,
                pinned_variables,
                template,
                pinned_memory,
                inputs._replace(valid=use),
                keys,
                keys,
                jnp.zeros_like(use),
                keep_learning_outputs=False,
            )
            heads = ActorAction(
                *(jnp.asarray(head, jnp.int32) for head in output.actions)
            )
            return heads, output.next_memory

        def skip(_: None) -> tuple[ActorAction, Tree]:
            """No lane meets the pinned System this step, so it is not called."""
            zero = jnp.zeros((size, 5), jnp.int32)
            return ActorAction(zero, zero, zero), pinned_memory

        chosen_run = run
        if pinned_compact:
            capacity = _compact_capacity(size)
            middle_capacity = max(capacity, -(-size // 2))
            count = jnp.sum(use)

            def run_compact(capacity: int, _: None) -> tuple[ActorAction, Tree]:
                """Apply a Policy adapter to the pinned games only, then put back."""
                index = jnp.nonzero(use, size=capacity, fill_value=size)[0]
                rows = _gather_rows(inputs, index, size)._replace(
                    valid=jnp.arange(capacity) < count
                )
                row_keys = _gather_rows(keys, index, size)
                output = _jax_apply(
                    pinned_execution,
                    pinned_variables,
                    template,
                    _gather_rows(pinned_memory, index, size),
                    rows,
                    row_keys,
                    row_keys,
                    jnp.zeros((capacity,), jnp.bool_),
                    keep_learning_outputs=False,
                )
                heads = ActorAction(
                    *(
                        _scatter_rows(
                            jnp.zeros((size, 5), jnp.int32),
                            jnp.asarray(head, jnp.int32),
                            index,
                        )
                        for head in output.actions
                    )
                )
                return heads, _scatter_rows(pinned_memory, output.next_memory, index)

            def run_larger(_: None) -> tuple[ActorAction, Tree]:
                """Use half when it fits; skip repeated sizes in tiny batches."""
                if middle_capacity in (capacity, size):
                    return run(None)
                return cast(
                    tuple[ActorAction, Tree],
                    jax.lax.cond(
                        count <= middle_capacity,
                        partial(run_compact, middle_capacity),
                        run,
                        None,
                    ),
                )

            def run_fitting_batch(_: None) -> tuple[ActorAction, Tree]:
                """Keep quarter-sized calls for small pin counts, then try larger."""
                return cast(
                    tuple[ActorAction, Tree],
                    jax.lax.cond(
                        count <= capacity,
                        partial(run_compact, capacity),
                        run_larger,
                        None,
                    ),
                )

            chosen_run = run_fitting_batch if capacity < size else run

        pinned_actions, next_pinned = cast(
            tuple[ActorAction, Tree],
            jax.lax.cond(jnp.any(use), chosen_run, skip, None),
        )
        actions = ActorAction(
            *(
                jnp.where(use[:, None], chosen, own)
                for chosen, own in zip(pinned_actions, network.actions, strict=True)
            )
        )
        return SystemOutput(actions, (next_network, next_pinned))

    hook = None
    if pinned_reset is not None:
        reset_pinned_half = pinned_reset

        def reset_both(old: Tree, fresh: Tree, mask: Array) -> Tree:
            """Reset the network half by M8's default, the pinned half by its hook."""
            old_pair = cast(tuple[Tree, Tree], old)
            fresh_pair = cast(tuple[Tree, Tree], fresh)
            return (
                _default_reset(old_pair[0], fresh_pair[0], mask, host=False),
                reset_pinned_half(old_pair[1], fresh_pair[1], mask),
            )

        hook = reset_both
    return System(
        f"{actor.name} Training Opponent With {pinned.name}",
        apply_pinned,
        init=initialize_pinned,
        reset_memory=hook,
    )


def _replay_host_actions(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Tree]:
    """Return actions a host method already chose, supplied as this step's data.

    Parameters
    ----------
    variables : ActorAction
        This step's host actions, three int32 arrays of shape (B, 5).
    memory : Any
        Returned unchanged; the stand-in keeps no memory of its own.
    inputs, keys
        Unused. The host method saw the same inputs and keys before this step.

    Returns
    -------
    tuple[ActorAction, Any]
        The supplied actions and the unchanged memory.
    """
    del inputs, keys
    return ActorAction(*cast(tuple[Array, Array, Array], variables)), memory


HOST_ACTIONS_SYSTEM = System("Host Actions", _replay_host_actions)
"""JAX stand-in for a pinned host method inside the compiled step.

The host route in ``collect_training_rollout`` asks the host method for its
actions, puts them in ``TrainingCarry.pinned_opponent``, and this System returns
them for the pinned lanes. It has no memory; the host method's memory lives on
its ``HostOpponent`` holder.
"""


class HostOpponent:
    """Hold one pinned host method, its memory, and the round it may serve next.

    Parameters
    ----------
    system : System
        Frozen host-execution System (a Policy is adapted with shared_policy
        before it arrives here).
    num_envs : int
        Fixed number of games B.

    Attributes
    ----------
    execution : _SystemExecution
        Stable M8 execution descriptor of the method.
    variables, template : Any
        Frozen parameters and memory template from prepare_evaluation_system.
    memory : Any
        The method's current memory for all B lanes, in M8's host layout.
        Only pinned lanes ever change it.
    generations : numpy.ndarray
        Int64 (B,) reset generation each lane's memory belongs to; -1 means
        never served, so the lane initializes on its first pinned decision.
    stateful : bool
        False only when the method has no init, no reset_memory hook and an
        empty memory template; M8's reset rule then keeps its memory empty and
        refuses anything else. Every other host method counts as stateful,
        because it may build memory while it plays.
    next_round : int or None
        The training round this holder serves next; None until the first
        call, so a holder built for a resumed run accepts the restored round.
    failure : BaseException or None
        The first error raised by the method. Once set, the holder refuses
        every later call.
    failed_round : int or None
        The round whose call failed, named in the refusal message; None until
        a failure.

    Notes
    -----
    Host-only and mutable. One holder belongs to one TrainingCollection. It
    serves each training round once, in order: a stale carry, such as the
    block's starting carry after a failure, is refused rather than combined
    with newer memory. The memory is never saved; see restore_checkpoint for
    the resume rule.
    """

    def __init__(
        self, system: System, num_envs: int, *, external_selection: bool = False
    ) -> None:
        """Split a frozen method; external_selection marks a training-owned pool.

        Only that outer pool's choices are supplied by the saved training
        selector. Nested researcher parameters keep their ordinary meanings.
        Memory is built by start(); no numerical action occurs here.
        """
        self.external_selection = external_selection
        self.execution: _SystemExecution
        self.execution, self.variables, self.template = prepare_evaluation_system(
            system
        )
        self.stateful = (
            self.execution.init is not None
            or self.execution.reset is not None
            or bool(jax.tree.leaves(self.template))
        )
        self.memory: Tree = ()
        self.generations = np.full(num_envs, -1, np.int64)
        self.next_round: int | None = None
        self.failure: BaseException | None = None
        self.failed_round: int | None = None

    def _frozen_tree(self) -> tuple[Tree, Tree]:
        """Omit only the training selector's outer pool choice from frozen values."""
        variables = self.variables
        if self.external_selection:
            if not isinstance(variables, _CompositionVariables):
                raise ValueError("External selection requires an ordinary pool")
            variables = variables._replace(choices=None)
        return variables, self.template

    def checkpoint_values(self) -> tuple[Array, ...]:
        """Return accessible frozen array leaves for the existing checkpoint tree.

        Numerical parameters and initial-memory templates are saved in their
        existing PyTree order. Opaque clients and other host objects stay bound
        to this holder; no serializer is invented for them. This does not save
        current game memory or make an unfinished host game resumable. Array
        values retain their device placement; no numerical policy call occurs.
        """
        return tuple(
            jnp.asarray(value)
            for value in jax.tree.leaves(self._frozen_tree())
            if isinstance(value, (Array, np.ndarray))
        )

    def restore_values(self, saved: tuple[Array, ...]) -> None:
        """Restore checked frozen arrays without replacing opaque live bindings.

        saved is checkpoint_values() restored by the normal learner array loader.
        Its count, shapes and dtypes must match the rebound method. NumPy leaves
        stay on the host and JAX leaves stay on their restored device. Current
        memory and round guards are unchanged. Call only during checkpoint
        preflight, before any resumed action or output write.
        """
        leaves, structure = cast(
            tuple[list[Any], Any], jax.tree.flatten(self._frozen_tree())
        )
        indices = [
            index
            for index, value in enumerate(leaves)
            if isinstance(value, (Array, np.ndarray))
        ]
        if len(indices) != len(saved):
            raise ValueError("Restored host method's frozen array count differs")
        for index, value in zip(indices, saved, strict=True):
            old = leaves[index]
            if old.shape != value.shape or old.dtype != value.dtype:
                raise ValueError("Restored host method's frozen array schema differs")
            leaves[index] = np.asarray(value) if isinstance(old, np.ndarray) else value
        restored, self.template = jax.tree.unflatten(structure, leaves)
        self.variables = (
            restored._replace(choices=self.variables.choices)
            if self.external_selection
            else restored
        )

    def start(self, inputs: SystemInput, init_keys: Array) -> None:
        """Build placeholder memory with every lane invalid, so nothing opens.

        Parameters
        ----------
        inputs : SystemInput
            Team B inputs of the setup reset, already on the host.
        init_keys : Array
            Team B initialization keys, as M8 derives them.
        """
        idle = inputs._replace(valid=np.zeros_like(np.asarray(inputs.valid)))
        self.memory = _initial_memory(
            self.execution, self.variables, self.template, idle, init_keys
        )

    def act(
        self,
        round_index: int,
        inputs: SystemInput,
        pinned: np.ndarray,
        keys: Array,
        init_keys: Array,
        generations: np.ndarray,
    ) -> ActorAction:
        """Choose the pinned lanes' actions for one round through M8's host helper.

        Parameters
        ----------
        round_index : int
            The carry's completed-round count before this decision.
        inputs : SystemInput
            Team B inputs for all B lanes, after finished games were reset, still
            on the device. Only what the method needs is copied to the host:
            nothing when no game is pinned, the pinned games' rows for a Policy
            adapter, all rows for a generic host System.
        pinned : numpy.ndarray
            Bool (B,) lanes assigned to the pinned System.
        keys, init_keys : Array
            Team B action and initialization keys, as M8 derives them.
        generations : numpy.ndarray
            Int (B,) current reset generation of each lane.

        Returns
        -------
        ActorAction
            Three (B, 5) int32 JAX arrays on the default device. Lanes that are
            not pinned are zero, whatever the method returned for them.

        Raises
        ------
        RuntimeError
            An earlier call failed; the holder is unusable.
        ValueError
            round_index is not the next round this holder serves.
        TypeError
            A method classified as memory-free returned memory: M8's own reset
            rule refuses memory that does not keep the structure its init
            established (no init means empty memory).
        BaseException
            Any error of the method itself propagates with its own type, and
            the holder becomes unusable.
        """
        if self.failure is not None:
            raise RuntimeError(
                f"This host-pinned collection failed at round {self.failed_round}; "
                "build a new collection or resume from a checkpoint"
            ) from self.failure
        if self.next_round is not None and round_index != self.next_round:
            raise ValueError(
                f"The host opponent's memory is at round {self.next_round}, but the "
                f"carry is at round {round_index}; a host-pinned collection serves "
                "each round once, so build a new collection or resume from a "
                "checkpoint"
            )
        valid = np.asarray(jax.device_get(inputs.valid)) & np.asarray(pinned)
        count = len(valid)
        if not valid.any():
            # No game meets the method this round: no input transfer, no call.
            self.next_round = round_index + 1
            zero = jnp.zeros((count, 5), jnp.int32)
            return ActorAction(zero, zero, zero)
        reset = (np.asarray(generations) != self.generations) & valid
        try:
            if self.execution.policies:
                # A Policy acts per actor, so only the pinned games' rows are
                # sent to the host and computed; the result is exact.
                rows = np.flatnonzero(valid)

                def pick(leaf: Any) -> Any:  # noqa: ANN401
                    """Take the pinned games' rows of one leaf."""
                    return leaf[rows]

                output = _host_apply(
                    self.execution,
                    self.variables,
                    self.template,
                    jax.tree.map(pick, self.memory),
                    jax.tree.map(pick, inputs)._replace(
                        valid=np.ones(len(rows), np.bool_)
                    ),
                    keys[rows],
                    init_keys[rows],
                    reset[rows],
                    keep_learning_outputs=False,
                )

                def put(whole: Any, part: Any) -> np.ndarray:  # noqa: ANN401
                    """Write the pinned games' rows back into one leaf."""
                    result = np.array(whole)
                    result[rows] = np.asarray(part)
                    return result

                memory = jax.tree.map(put, self.memory, output.next_memory)
                heads: list[np.ndarray] = []
                for head in output.actions:
                    full = np.zeros((count, 5), np.int32)
                    full[rows] = np.asarray(head, np.int32)
                    heads.append(full)
            else:
                output = _host_apply(
                    self.execution,
                    self.variables,
                    self.template,
                    self.memory,
                    inputs._replace(valid=valid),
                    keys,
                    init_keys,
                    reset,
                    keep_learning_outputs=False,
                )
                memory = output.next_memory
                heads = [
                    np.where(valid[:, None], np.asarray(head, np.int32), 0)
                    for head in output.actions
                ]
        except BaseException as error:
            self.failure = error
            self.failed_round = round_index
            raise
        self.memory = memory
        self.generations = np.where(valid, np.asarray(generations), self.generations)
        self.next_round = round_index + 1
        return ActorAction(*(jnp.asarray(head) for head in heads))
