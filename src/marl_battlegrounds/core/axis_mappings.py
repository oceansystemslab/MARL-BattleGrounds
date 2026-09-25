"""Keep global slots, actor-relative targets and observation rows in agreement.

Team A occupies global slots 0..4 and Team B occupies 5..9. Each actor uses
target zero for None, 1..5 for own-team rows and 6..10 for opposing rows.
These mappings include unused slots and self; they do not decide legality.
Movement categories retain world directions for both teams.

Python helpers validate concrete IDs on the host. Constant tuples support
host consumers, while the JAX array tables support compiled indexing.

spawn_bank_on_right is the one Core rule for which side of the map a team's
spawn bank is on. Team Deathmatch Red Zone scoring uses it, and so does the
public policy helper team_on_right. It is exact: the answer does not depend
on pad order or on float32 rounding of a mean."""

from typing import Literal

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    NUM_TARGET_ACTIONS,
)

type _ObservationRelation = Literal["ally", "enemy"]

TEAM_A_START = 0
TEAM_A_END = MAX_AGENTS_PER_TEAM
TEAM_B_START = MAX_AGENTS_PER_TEAM
TEAM_B_END = MAX_AGENT_SLOTS

MOVEMENT_ACTION_NAME_BY_ID: tuple[str, ...] = (
    "Stay",
    "North",
    "South",
    "East",
    "West",
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest",
)
TARGET_ACTION_NAME_BY_ID: tuple[str, ...] = (
    "Target None",
    "Ally 0",
    "Ally 1",
    "Ally 2",
    "Ally 3",
    "Ally 4",
    "Enemy 0",
    "Enemy 1",
    "Enemy 2",
    "Enemy 3",
    "Enemy 4",
)

# These are exact current float32 unit directions, not realized displacement.
_INVERSE_SQUARE_ROOT_OF_TWO = 0.7071067690849304
UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (_INVERSE_SQUARE_ROOT_OF_TWO, _INVERSE_SQUARE_ROOT_OF_TWO),
    (-_INVERSE_SQUARE_ROOT_OF_TWO, _INVERSE_SQUARE_ROOT_OF_TWO),
    (_INVERSE_SQUARE_ROOT_OF_TWO, -_INVERSE_SQUARE_ROOT_OF_TWO),
    (-_INVERSE_SQUARE_ROOT_OF_TWO, -_INVERSE_SQUARE_ROOT_OF_TWO),
)
UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY: Array = jnp.asarray(
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION,
    dtype=jnp.float32,
)

_TEAM_A_RECIPIENT_ROW: tuple[int | None, ...] = (
    None,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
)
_TEAM_B_RECIPIENT_ROW: tuple[int | None, ...] = (
    None,
    5,
    6,
    7,
    8,
    9,
    0,
    1,
    2,
    3,
    4,
)

# Target action zero is target-none. Categories 1..5 are ally rows and 6..10
# are enemy rows for every actor in the corresponding fixed team block.
GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION: tuple[tuple[int | None, ...], ...] = (
    *(_TEAM_A_RECIPIENT_ROW for _ in range(MAX_AGENTS_PER_TEAM)),
    *(_TEAM_B_RECIPIENT_ROW for _ in range(MAX_AGENTS_PER_TEAM)),
)

GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW: tuple[tuple[int, ...], ...] = tuple(
    tuple(recipient for recipient in row[1 : 1 + MAX_AGENTS_PER_TEAM])
    for row in GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION
)
GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW: tuple[tuple[int, ...], ...] = tuple(
    tuple(recipient for recipient in row[1 + MAX_AGENTS_PER_TEAM :])
    for row in GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION
)

# The sentinel preserves target-none as an all-zero row under ``jax.nn.one_hot``.
GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION: Array = jnp.asarray(
    tuple(
        tuple(-1 if recipient is None else recipient for recipient in row)
        for row in GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION
    ),
    dtype=jnp.int32,
)


def _validate_global_slot(global_slot: object, *, name: str) -> int:
    """Check a host slot ID before indexing a fixed mapping table.

    ``name`` labels errors. Accept Python ints other than bool in 0..9;
    raise TypeError for another type and ValueError outside that range.
    This uses Python control flow and is not a traced-array validator.
    """
    if isinstance(global_slot, bool) or not isinstance(global_slot, int):
        raise TypeError(f"{name} must be an int; got {type(global_slot).__name__}.")
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError(
            f"{name} must be a fixed global slot in [0, {MAX_AGENT_SLOTS}); "
            f"got {global_slot}."
        )
    return global_slot


def _validate_target_action(target_action: object) -> int:
    """Check a host target category before indexing a mapping table.

    Accept Python ints other than bool in 0..10. Raise TypeError for another
    type and ValueError outside that range. Zero means Target None.
    """
    if isinstance(target_action, bool) or not isinstance(target_action, int):
        raise TypeError(
            f"target_action must be an int; got {type(target_action).__name__}."
        )
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        raise ValueError(
            f"target_action must be in [0, {NUM_TARGET_ACTIONS}); got {target_action}."
        )
    return target_action


def global_slot_to_target_action(
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    """Express one global recipient as an actor-relative target category.

    Parameters
    ----------
    actor_global_slot : int
        Acting slot in 0..9. Slots 0..4 are Team A; 5..9 are Team B.
    target_global_slot : int or None
        Recipient slot in 0..9, or None for Target None.

    Returns
    -------
    int
        Zero for None; 1..5 for an own-team row; 6..10 for an opposing row.
        Own-team rows include the actor itself.

    Raises
    ------
    TypeError
        A supplied slot is not a Python int, or is a bool.
    ValueError
        A supplied slot is outside 0..9.

    Notes
    -----
    This host helper checks identity only, not participation or legality.
    Use the fixed array tables for traced JAX indexing.
    """
    actor_global_slot = _validate_global_slot(
        actor_global_slot,
        name="actor_global_slot",
    )
    if target_global_slot is None:
        return 0
    target_global_slot = _validate_global_slot(
        target_global_slot,
        name="target_global_slot",
    )
    return GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION[actor_global_slot].index(
        target_global_slot
    )


def target_action_to_global_slot(
    actor_global_slot: int,
    target_action: int,
) -> int | None:
    """Decode an actor-relative target category to its global recipient.

    Parameters
    ----------
    actor_global_slot : int
        Acting slot in 0..9, with Team A before Team B.
    target_action : int
        Category in 0..10: none, five own-team rows, then five opposing rows.

    Returns
    -------
    int or None
        Global recipient in 0..9, or None for target category zero.

    Raises
    ------
    TypeError
        Either input is not a Python int, or is a bool.
    ValueError
        A slot or category is outside its range.

    Notes
    -----
    This is a host identity lookup. It does not inspect masks, visibility,
    class, alive status or configured participation.
    """
    actor_global_slot = _validate_global_slot(
        actor_global_slot,
        name="actor_global_slot",
    )
    target_action = _validate_target_action(target_action)
    return GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION[actor_global_slot][
        target_action
    ]


def observation_relation_and_row(
    observer_global_slot: int,
    candidate_global_slot: int,
) -> tuple[_ObservationRelation, int]:
    """Locate a candidate in one observer's stable ally or enemy rows.

    Parameters
    ----------
    observer_global_slot : int
        Observer slot in 0..9, with Team A before Team B.
    candidate_global_slot : int
        Candidate slot in 0..9, including self or an unused slot.

    Returns
    -------
    tuple of str and int
        ``("ally", row)`` or ``("enemy", row)`` with row in 0..4.
        The result follows roster order and does not depend on visibility.

    Raises
    ------
    TypeError
        A slot is not a Python int, or is a bool.
    ValueError
        A slot is outside 0..9.

    Notes
    -----
    This host helper does not read a simulator state or transform positions.
    """
    observer_global_slot = _validate_global_slot(
        observer_global_slot,
        name="observer_global_slot",
    )
    candidate_global_slot = _validate_global_slot(
        candidate_global_slot,
        name="candidate_global_slot",
    )

    ally_slots = GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW[observer_global_slot]
    if candidate_global_slot in ally_slots:
        return "ally", ally_slots.index(candidate_global_slot)

    enemy_slots = GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW[observer_global_slot]
    return "enemy", enemy_slots.index(candidate_global_slot)


# Exact power of two applied to pads and the half-width before the exact sum.
# It keeps every intermediate value finite on any map Core accepts (width up
# to the float32 maximum, pads inside the map); see spawn_bank_on_right Notes.
_SPAWN_SIDE_SCALE = 0.125


def _two_sum(a: Array, b: Array) -> tuple[Array, Array]:
    """Split a + b into its rounded float32 sum and the exact rounding error.

    Parameters
    ----------
    a, b : Array
        Float32 arrays of one broadcastable shape.

    Returns
    -------
    tuple of Array
        (s, e) with s = fl(a + b) and s + e == a + b exactly (Knuth's
        error-free sum), provided the adds are evaluated as written with
        round-to-nearest and nothing overflows.
    """
    s = a + b
    b_virtual = s - a
    e = (a - (s - b_virtual)) + (b - b_virtual)
    return s, e


def _spawn_side_expansion(pads: Array, map_width: Array | float) -> list[Array]:
    """Return float32 components whose exact sum is (sum(x_i) - 5 h) / 8.

    Parameters
    ----------
    pads : Array
        Float32 spawn pads with shape (..., 5, 2); x is column 0.
    map_width : Array or float
        Map width broadcastable to pads.shape[:-2]; converted to float32.
        h is float32(map_width) * 0.5, which is exact.

    Returns
    -------
    list of Array
        Ten float32 arrays of shape pads.shape[:-2] forming a nonoverlapping
        expansion in rising magnitude (Shewchuk's grow-expansion over Knuth
        two-sums). Their exact sum is (sum of the five pad x values minus
        5 h) / 8. A non-finite component means an input was outside the
        validated domain described in spawn_bank_on_right.
    """
    x = pads[..., 0] * _SPAWN_SIDE_SCALE
    half = jnp.asarray(map_width, jnp.float32) * (0.5 * _SPAWN_SIDE_SCALE)
    terms: list[Array] = []
    for pad in range(MAX_AGENTS_PER_TEAM):
        terms.extend(_two_sum(x[..., pad], -half))
    expansion: list[Array] = []
    for term in terms:
        carry, grown = term, []
        for component in expansion:
            carry, low = _two_sum(carry, component)
            grown.append(low)
        expansion = [*grown, carry]
    return expansion


def spawn_bank_on_right(
    team_spawn_pad_positions: Array, map_width: Array | float
) -> Array:
    """Tell whether one team's spawn bank is on the right half of the map.

    Parameters
    ----------
    team_spawn_pad_positions : Array
        One team's five ordered spawn pads, float32 with shape (..., 5, 2) as
        [x, y] world positions; any leading shape L, for example (2,) for
        both teams of one game or (B, 2) for a batch. Unused pad rows count.
    map_width : Array or float
        Map width in world units, broadcastable to L; converted to float32.

    Returns
    -------
    Array
        Boolean array of shape L. True exactly when the mean x of the five
        pads is greater than half the float32 map width, that is when
        sum(x_i) - 5 * (float32(map_width) / 2) is positive in exact
        arithmetic. An exactly centred bank reads False (left). The answer is
        the same for every order of the five pads.

    Raises
    ------
    ValueError
        The pads do not end with shape (5, 2). This is a static shape check.

    Notes
    -----
    Pure JAX; works eagerly and under jit and vmap, on CPU and GPU. The pads
    and half-width are first scaled by exactly 1/8, then the five exact
    differences are summed without error (Knuth two-sums grown into a
    Shewchuk expansion); the largest nonzero component gives the sign. For
    this to be exact, each float32 add and subtract must be evaluated as
    written with round-to-nearest; the tests check eager, jit and vmap
    results against an exact reference.

    Validated domain: Core accepts 0 < map_width <= the float32 maximum and
    pads with radius <= x <= map_width - radius, so 0 <= x_i <= w32. After
    the 1/8 scale every value in the expansion stays below 0.75 of the
    float32 maximum, so nothing overflows. In Team Deathmatch every pad is at
    least one body radius (0.5) inside the map, so every scaled value and
    rounding error is a whole multiple of 2**-27 and never subnormal. The
    one remaining limit is outside Team Deathmatch scoring: in a neutral
    layout an empty team has radius-0 pads, and pad x values or widths below
    about 1e-30 could flush to zero.

    Cost: 6 exact multiplies and about 55 two-sums per bank.
    """
    pads = jnp.asarray(team_spawn_pad_positions, jnp.float32)
    if pads.shape[-2:] != (MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS):
        raise ValueError("team_spawn_pad_positions must end with shape (5, 2).")
    expansion = _spawn_side_expansion(pads, map_width)
    sign = jnp.zeros_like(expansion[-1])
    for component in expansion:
        # Components rise in magnitude, so the last nonzero one decides.
        sign = jnp.where(component != 0.0, jnp.sign(component), sign)
    return sign > 0.0


__all__ = [
    "GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION",
    "GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION",
    "GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW",
    "GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW",
    "MOVEMENT_ACTION_NAME_BY_ID",
    "TARGET_ACTION_NAME_BY_ID",
    "TEAM_A_END",
    "TEAM_A_START",
    "TEAM_B_END",
    "TEAM_B_START",
    "UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION",
    "UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY",
    "global_slot_to_target_action",
    "observation_relation_and_row",
    "spawn_bank_on_right",
    "target_action_to_global_slot",
]
