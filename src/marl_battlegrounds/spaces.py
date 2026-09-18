"""Describe one actor's existing observation and action without flattening them.

Use ``env.observation_space(agent)`` and ``env.action_space(agent)`` to inspect
named fields, array shapes, exact dtypes and inclusive bounds. These descriptions
have no environment-batch or actor axis. Select one actor from one game before
checking it; a native batch needs both a game index and an actor index.

``contains`` is a host inspection helper. It may copy device arrays to the host
and wait for device work. Keep it outside ``jit``, ``vmap`` and rollout loops.
Structural membership does not establish physical validity, information rights
or current action legality. Core's ActionMask owns legal choices, including the
joint target/Ultimate constraint. These spaces do not sample actions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import cast

import numpy as np

from marl_battlegrounds.core import types as core_types


@dataclass(frozen=True)
class ArraySpace:
    """Describe one array field without allocating an example value.

    Parameters
    ----------
    shape : tuple[int, ...]
        Exact dimensions; ``()`` describes one scalar value.
    dtype : np.dtype[np.generic]
        Exact NumPy dtype, such as ``np.dtype(np.int32)``. Membership
        does not cast a candidate to this dtype.
    low : int | float
        Inclusive lower bound, applied to every array element.
    high : int | float
        Inclusive upper bound, applied to every array element.

    Notes
    -----
    Attributes are frozen after construction. Callers creating custom descriptors
    supply valid dimensions, a dtype and ordered bounds; construction does not
    validate those choices. Use ``contains`` for host-side value checks.
    """

    shape: tuple[int, ...]
    """Required array dimensions, with no implicit batch or actor axis."""

    dtype: np.dtype[np.generic]
    """Required exact dtype; candidate values are never cast to match it."""

    low: int | float
    """Inclusive lower bound shared by every element."""

    high: int | float
    """Inclusive upper bound shared by every element."""

    def contains(self, value: object) -> bool:
        """Return whether a value has this shape, dtype and inclusive bounds.

        Parameters
        ----------
        value : object
            A NumPy-compatible scalar, array or sequence. A sequence's
            inferred dtype must already match; no dtype conversion is made.

        Returns
        -------
        bool
            A Python bool. False means conversion failed with TypeError or
            ValueError, a shape/dtype differs, or an element is outside the
            bounds. NaN fails the bounds check; infinity passes only when the
            corresponding bound permits it. Other conversion errors propagate.

        Notes
        -----
        This call reads values on the host and may synchronize device arrays.
        Do not call it from a compiled function or use it as an action mask.
        """
        try:
            array = np.asarray(value)
            return bool(
                array.shape == self.shape
                and array.dtype == self.dtype
                and np.all((array >= self.low) & (array <= self.high))
            )
        except TypeError, ValueError:
            return False


@dataclass(frozen=True)
class StructuredSpace:
    """Describe named fields, including nested observation families.

    Parameters
    ----------
    fields : Mapping[str, ArraySpace | StructuredSpace]
        Mapping from each field name to an ArraySpace or StructuredSpace.
        Its order defines the expected NamedTuple field order. The public
        space factories return read-only mappings and cache their descriptors.

    Notes
    -----
    The descriptor is frozen. A custom mapping passed directly to this constructor
    is retained, not copied or frozen; its owner must keep it stable. Construction
    does not validate custom field descriptors. Membership checks only structure
    and bounds, not Core validity or actor information rights.
    """

    fields: Mapping[str, ArraySpace | StructuredSpace]
    """Named child spaces in the same order as the matching NamedTuple fields."""

    def contains(self, value: object) -> bool:
        """Check a mapping or NamedTuple recursively on the host.

        Parameters
        ----------
        value : object
            A mapping with exactly these keys, in any mapping order, or a
            NamedTuple with exactly these field names in descriptor order.
            Nested values must meet their corresponding child spaces.

        Returns
        -------
        bool
            A Python bool. False means a plain tuple or another unsupported
            container, missing/extra field, different NamedTuple field order,
            or failed child membership. This does not require one particular
            NamedTuple class. Exceptions from custom mapping access, malformed
            tuple objects or invalid custom descriptors are not suppressed.

        Notes
        -----
        Array checks may copy device data to the host. Keep this outside JAX
        transforms and the hot rollout path. Membership is not action legality.
        """
        if isinstance(value, Mapping):
            values = cast(Mapping[str, object], value)
            return values.keys() == self.fields.keys() and all(
                space.contains(values[name]) for name, space in self.fields.items()
            )
        if isinstance(value, tuple):
            items = cast(tuple[object, ...], value)
            if getattr(type(items), "_fields", ()) != tuple(self.fields):
                return False
            return all(
                space.contains(item)
                for space, item in zip(self.fields.values(), items, strict=True)
            )
        return False


def _structure(**fields: ArraySpace | StructuredSpace) -> StructuredSpace:
    """Keep supplied field order in a read-only mapping for a shared descriptor."""
    return StructuredSpace(MappingProxyType(fields))


def _float(*shape: int) -> ArraySpace:
    """Describe one float32 field with exact shape and unbounded numeric range."""
    return ArraySpace(shape, np.dtype(np.float32), -np.inf, np.inf)


def _integer(*shape: int, low: int = 0, high: int = 2_147_483_647) -> ArraySpace:
    """Describe one int32 field with exact shape and inclusive scalar bounds."""
    return ArraySpace(shape, np.dtype(np.int32), low, high)


def _boolean(*shape: int) -> ArraySpace:
    """Describe one bool field with exact shape and the values False or True."""
    return ArraySpace(shape, np.dtype(np.bool_), 0, 1)


@cache
def action_space() -> StructuredSpace:
    """Return the cached descriptor for one ActorAction.

    Fields ``move``, ``select_target`` and ``use_ultimate`` each have shape ``()``
    and dtype int32. Their inclusive ranges are currently 0..8, 0..10 and 0..1,
    taken from Core's category constants. There is no actor or native-batch axis.

    ``contains`` accepts an ActorAction or a matching field mapping. It checks
    category ranges only; the current joint target/Ultimate mask may still
    forbid a structurally valid pair. Use ``env.sample_actions`` for legal joint
    sampling. Reading this descriptor allocates no example action or device data.
    """
    return _structure(
        move=_integer(high=core_types.NUM_MOVE_ACTIONS - 1),
        select_target=_integer(high=core_types.NUM_TARGET_ACTIONS - 1),
        use_ultimate=_integer(high=core_types.NUM_ULTIMATE_ACTIONS - 1),
    )


@cache
def observation_space() -> StructuredSpace:
    """Return the cached descriptor for one actor's Core Observation.

    Select one actor from one game before using ``contains``. A scalar environment
    stores ten actor rows; a native environment adds a leading batch axis, so
    select both a game index and an actor index.
    The descriptor itself has neither axis. Field names and nested families
    match Observation, including previous actions and spawn lifecycle values.

    Feature arrays are float32 with unbounded ranges. Visibility and lifecycle
    masks are bool. Lifecycle counts/class IDs are nonnegative int32; those broad
    bounds do not replace Core's field-specific validity checks. The scalar
    ``self_ally_index`` is int32 in 0..4. All dimensions come from Core constants
    and can be inspected through ``fields`` and each ArraySpace.shape.

    No fields are flattened, no identity feature is added, and no example
    observation is allocated. Membership is a host check, not a JAX operation
    or a guarantee that an observation came from an authorized simulator state.
    """
    team = core_types.MAX_AGENTS_PER_TEAM
    teams = core_types.NUM_TEAMS
    return _structure(
        self_features=_float(core_types.SELF_FEATURES),
        ally_unit_features=_float(team, core_types.UNIT_FEATURES),
        enemy_unit_features=_float(team, core_types.UNIT_FEATURES),
        map_obstacle_features=_float(
            core_types.MAX_OBSTACLE_SLOTS, core_types.OBSTACLE_FEATURES
        ),
        objective_features=_float(
            core_types.MAX_OBJECTIVE_SLOTS, core_types.OBJECTIVE_FEATURES
        ),
        context_features=_float(core_types.CONTEXT_FEATURES),
        ally_visibility_mask=_boolean(team),
        enemy_visibility_mask=_boolean(team),
        previous_timestep_actions=_structure(
            ally_previous_timestep_move_actions_one_hot=_float(
                team, core_types.NUM_MOVE_ACTIONS
            ),
            enemy_previous_timestep_move_actions_one_hot=_float(
                team, core_types.NUM_MOVE_ACTIONS
            ),
            ally_previous_timestep_select_target_actions_one_hot=_float(
                team, core_types.NUM_TARGET_ACTIONS
            ),
            enemy_previous_timestep_select_target_actions_one_hot=_float(
                team, core_types.NUM_TARGET_ACTIONS
            ),
            ally_previous_timestep_use_ultimate_actions_one_hot=_float(
                team, core_types.NUM_ULTIMATE_ACTIONS
            ),
            enemy_previous_timestep_use_ultimate_actions_one_hot=_float(
                team, core_types.NUM_ULTIMATE_ACTIONS
            ),
        ),
        spawn_lifecycle=_structure(
            spawn_pad_positions_by_agent_by_team=_float(
                teams, team, core_types.ENVIRONMENT_DIMENSIONS
            ),
            spawn_shield_actual_durations_by_agent_by_team=_integer(teams, team),
            spawn_shield_configured_duration_by_agent=_integer(),
            spawn_shield_speed_by_agent=_float(),
            respawn_wave_period_step_count_by_agent_by_team=_integer(teams),
            respawn_wave_countdowns_by_agent_by_team=_integer(teams),
            active_mask_by_agent_by_team=_boolean(teams, team),
            alive_mask_by_agent_by_team=_boolean(teams, team),
            class_ids_by_agent_by_team=_integer(teams, team),
        ),
        self_ally_index=_integer(high=team - 1),
    )
