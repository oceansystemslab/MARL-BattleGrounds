"""Canonical fixed-axis mappings shared by simulator and host consumers."""

from typing import Literal

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
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
    if isinstance(global_slot, bool) or not isinstance(global_slot, int):
        raise TypeError(f"{name} must be an int; got {type(global_slot).__name__}.")
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError(
            f"{name} must be a fixed global slot in [0, {MAX_AGENT_SLOTS}); "
            f"got {global_slot}."
        )
    return global_slot


def _validate_target_action(target_action: object) -> int:
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
    """Return the actor-relative target category for one fixed global slot."""
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
    """Return the fixed global recipient represented by one target category."""
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
    """Return the candidate's ally/enemy relation and stable observation row."""
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
    "target_action_to_global_slot",
]
