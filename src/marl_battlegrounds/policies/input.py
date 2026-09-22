"""Keep rollout observations compact and expand permitted actor inputs on demand.

Observations stores the ten base observation rows and a source-permission matrix.
build_team_actor_input expands one team's permitted inputs when needed.
build_actor_input provides the compatible all-actor route with default permissions.
A source is another actor's sensor row; it is not access to hidden Core state.
Policy masks and random keys remain separate arguments.

The optional mirror helpers reflect one team's permitted view and its move
choices about the map's vertical centerline, so a method can always see the
game as if its team started on one bank. They are an input convention a method
may adopt; Core, the evaluator and the tournament never apply them.
obstacle_mirror_partners is the one quadratic piece, the per-table check of
which obstacle rows already have their mirror image present; a caller whose
rows share one table per game computes it once per game and passes it in.
"""

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_X,
    CONTEXT_FEATURE_MAP_WIDTH,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_WALL,
    ActionMask,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV2,
    build_default_shared_obs_information_availability,
    build_shared_obs_team_source_bank_from_base_rows,
    mask_source_bank_for_recipient,
)


class ActorInput(NamedTuple):
    """Bundle an actor's observation with the sensor rows it may receive.

    Attributes
    ----------
    observation : Observation
        The actor's own current observation. The team builder adds five recipient
        rows; build_actor_input adds ten. Select one row before a scalar policy
        call. An outer vmap adds the environment axis.
    source_bank : SharedObsSensorSourceBankV2
        Shared sensor data with unavailable source rows cleared. One actor's
        feature, visibility and objective arrays have shapes (5, 10, 58),
        (5, 10) and (5, 8, 12). Builders add their leading recipient axis.
    source_availability : Array
        Boolean permission for five own-team sources. Shape (5,) for one actor,
        (5, 5) for one team, or (10, 5) for all actors. This is permission, not
        a visibility claim: an admitted source can have no current sensor data.

    Notes
    -----
    This immutable tuple is JAX-compatible. It contains no action masks or random
    keys. Returning actions for several actors does not authorize sharing their
    private observation rows with one another.
    """

    observation: Observation
    source_bank: SharedObsSensorSourceBankV2
    source_availability: Array


class Observations(NamedTuple):
    """Store compact observations and source permissions for one game or batch.

    Attributes
    ----------
    observation : Observation
        Core's current observation rows. One game has a leading ten-actor axis.
        A native environment batch adds a leading game axis to every leaf.
    source_availability : Array
        Boolean array of shape (10, 10), or (B, 10, 10) for B games. Axes are
        recipient and source in the simulator's routing order. Only permitted
        active same-team sources other than self may be enabled.

    Notes
    -----
    This immutable tuple avoids storing a separate expanded source bank for every
    actor at every rollout step. It is data for policy routing, not an authorization
    for a scalar actor to inspect every row. The policy executor selects and clears
    rows before delivering them to each actor.
    """

    observation: Observation
    source_availability: Array


def build_observations(observation: Observation, config: EnvConfig) -> Observations:
    """Pair base observation rows with the default source permissions.

    Parameters
    ----------
    observation : Observation
        Current observation for all ten actor slots of one game.
    config : EnvConfig
        Configuration for that game. Its active mask and team IDs determine
        which sources may be shared.

    Returns
    -------
    Observations
        The supplied observation and a Boolean (10, 10) permission matrix.
        Active same-team sources are allowed, excluding the recipient itself.

    Notes
    -----
    This function does not expand source banks or inspect privileged state. It
    does not check that observation and config belong to the same game; the caller
    must keep them paired. Use vmap for several games. Inputs are unchanged.
    """
    return Observations(
        observation,
        build_default_shared_obs_information_availability(
            config.agent_profile.active_mask, config.agent_profile.team_ids
        ),
    )


def build_team_actor_input(observations: Observations, team: int) -> ActorInput:
    """Expand one team's supplied permissions into separate actor inputs.

    Parameters
    ----------
    observations : Observations
        One game's ten current observation rows and Boolean source permissions
        shaped (10, 10). Permissions must already be a valid same-team, non-self
        subset. This helper never replaces them with broader defaults.
    team : int
        Static routing choice: 0 selects Team A and 1 selects Team B. Pass a
        Python integer, not a traced array. Booleans are not accepted.

    Returns
    -------
    ActorInput
        Five recipient observations, Boolean permissions (5, 5), and redacted
        source banks. Bank feature, visibility and objective shapes are
        (5, 5, 10, 58), (5, 5, 10), and (5, 5, 8, 12). Features are float32;
        visibility is Boolean. Dead, inactive, hidden and forbidden source
        material is cleared by the existing sensor and redaction authorities.

    Raises
    ------
    ValueError
        team is not an integer 0 or 1, or is a Boolean.

    Notes
    -----
    Only the requested team's bank is built. The five recipient rows retain
    separate information rights; they do not form a shared private observation.
    Inputs stay unchanged and numerical values stay on device. Shapes and
    permission validity are caller preconditions. Use jit with a static team
    and vmap to add environment batches. Keep the expanded result transient
    instead of storing it for every rollout step.
    """
    if isinstance(team, bool) or not isinstance(team, Integral) or team not in (0, 1):
        raise ValueError("team must be the integer 0 or 1")
    start = int(team) * 5

    def take(value: Array) -> Array:
        """Select this team's five routing rows without changing feature axes."""
        return value[start : start + 5]

    observation = jax.tree.map(take, observations.observation)
    source_is_living = (observation.self_features[:, AGENT_FEATURE_ACTIVE] > 0.0) & (
        observation.self_features[:, AGENT_FEATURE_ALIVE] > 0.0
    )
    bank = build_shared_obs_team_source_bank_from_base_rows(
        observation.ally_unit_features,
        observation.enemy_unit_features,
        observation.objective_features,
        observation.ally_visibility_mask,
        observation.enemy_visibility_mask,
        source_is_living,
    )
    availability = observations.source_availability[
        start : start + 5, start : start + 5
    ]
    return ActorInput(
        observation,
        jax.vmap(mask_source_bank_for_recipient, in_axes=(None, 0))(bank, availability),
        availability,
    )


def build_actor_input(observation: Observation, config: EnvConfig) -> ActorInput:
    """Build permitted SharedObs inputs for all ten actors in one game.

    Parameters
    ----------
    observation : Observation
        Current ten-actor observations. Each actor's feature rows are already
        expressed in its observation order.
    config : EnvConfig
        Matching game configuration. Active slots and team IDs supply the
        default source permissions.

    Returns
    -------
    ActorInput
        A ten-actor bundle. Source features have shape (10, 5, 10, 58),
        visibility (10, 5, 10), objectives (10, 5, 8, 12), and source permission
        (10, 5). Unavailable source data is zeroed before actor delivery.

    Notes
    -----
    The fixed Team A/Team B routing blocks must match the supplied configuration.
    This helper expands banks for policy application; retaining its output at every
    rollout step costs more space than retaining Observations. Inputs stay dynamic
    under jit. Use vmap to add a game axis. No action mask or hidden simulator state
    is constructed here.
    """
    observations = build_observations(observation, config)
    team_a = build_team_actor_input(observations, 0)
    team_b = build_team_actor_input(observations, 1)

    def join(a: Array, b: Array) -> Array:
        """Join Team A's five recipients followed by Team B's five recipients."""
        return jnp.concatenate((a, b), axis=0)

    return ActorInput(
        observation=observation,
        source_bank=jax.tree.map(join, team_a.source_bank, team_b.source_bank),
        source_availability=join(
            team_a.source_availability, team_b.source_availability
        ),
    )


# Move category m becomes MOVE_MIRROR[m] when the world is reflected about the
# vertical centerline: Stay, North and South are fixed; East and West,
# Northeast and Northwest, Southeast and Southwest change places.
MOVE_MIRROR: Array = jnp.asarray((0, 1, 2, 4, 3, 6, 5, 8, 7), jnp.int32)


def _trail(value: Array, ndim: int) -> Array:
    """Append size-one axes so value broadcasts against an array of rank ndim."""
    return value.reshape(value.shape + (1,) * (ndim - value.ndim))


def _check_flag(flag: Array, leading: tuple[int, ...]) -> None:
    """Require a Boolean flag whose shape is exactly the actors' leading shape."""
    if flag.dtype != jnp.bool_:
        raise TypeError("The mirror flag must be a Boolean array.")
    if flag.shape != leading:
        raise ValueError("The mirror flag must match the actors' leading shape.")


def team_on_right(actors: ActorInput) -> Array:
    """Tell, per actor, whether its own team spawns past the map's vertical centerline.

    Parameters
    ----------
    actors : ActorInput
        Permitted inputs with any leading shape L, for example (B, 5). The own
        team's spawn pads are the first bank of
        observation.spawn_lifecycle.spawn_pad_positions_by_agent_by_team and
        the map width is column CONTEXT_FEATURE_MAP_WIDTH of
        observation.context_features.

    Returns
    -------
    Array
        Boolean array of shape L. True where the mean x of the own team's five
        pads exceeds half the map width. All-zero padding rows read False.

    Notes
    -----
    Pure JAX; works under jit and vmap. The answer is constant for a game
    because pads and width change only at reset. It reads only the actor's own
    permitted view, so it widens no information right.
    """
    pads = actors.observation.spawn_lifecycle.spawn_pad_positions_by_agent_by_team
    width = actors.observation.context_features[..., CONTEXT_FEATURE_MAP_WIDTH]
    own_x = jnp.mean(pads[..., 0, :, 0], axis=-1)
    return own_x > width / 2


def _reflect_unit_rows(rows: Array, width: Array, flag: Array) -> Array:
    """Reflect the x column of active 58-wide unit rows in flagged actors' views."""
    x = rows[..., AGENT_FEATURE_X]
    present = rows[..., AGENT_FEATURE_ACTIVE] > 0
    select = _trail(flag, x.ndim) & present
    return rows.at[..., AGENT_FEATURE_X].set(
        jnp.where(select, _trail(width, x.ndim) - x, x)
    )


_OBSTACLE_MATCH_TOLERANCE = 1e-3


def _same_angle(first: Array, second: Array) -> Array:
    """Tell whether two wall angles in radians agree up to a half turn."""
    difference = jnp.mod(first - second, jnp.pi)
    return jnp.minimum(difference, jnp.pi - difference) < _OBSTACLE_MATCH_TOLERANCE


def _same_obstacle(candidate: Array, authored: Array) -> Array:
    """Tell, pairwise, whether reflected rows describe shapes already in the table.

    candidate has shape (..., K, 1, 8) and authored (..., 1, J, 8), both Core
    obstacle rows. Return a Boolean (..., K, J) array that is True where the two
    rows have the same type, centre and radius and, for walls, the same rectangle
    up to a half turn, or the same rectangle with width and height swapped up to
    a quarter turn. Pillars ignore the angle. Coordinates and angles agree within
    _OBSTACLE_MATCH_TOLERANCE.
    """

    def close(column: int, other: Array | None = None) -> Array:
        source = authored[..., column] if other is None else other
        return jnp.abs(candidate[..., column] - source) < _OBSTACLE_MATCH_TOLERANCE

    same_place = (
        close(OBSTACLE_FEATURE_TYPE)
        & close(OBSTACLE_FEATURE_X)
        & close(OBSTACLE_FEATURE_Y)
        & close(OBSTACLE_FEATURE_RADIUS)
    )
    straight = close(OBSTACLE_FEATURE_WIDTH) & close(OBSTACLE_FEATURE_HEIGHT)
    turned = close(
        OBSTACLE_FEATURE_WIDTH, authored[..., OBSTACLE_FEATURE_HEIGHT]
    ) & close(OBSTACLE_FEATURE_HEIGHT, authored[..., OBSTACLE_FEATURE_WIDTH])
    theta, other_theta = (
        candidate[..., OBSTACLE_FEATURE_THETA],
        authored[..., OBSTACLE_FEATURE_THETA],
    )
    wall_match = (straight & _same_angle(theta, other_theta)) | (
        turned & _same_angle(theta, other_theta + jnp.pi / 2)
    )
    is_wall = authored[..., OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL
    return same_place & jnp.where(is_wall, wall_match, straight)


def _reflected_obstacle_rows(rows: Array, width: Array) -> Array:
    """Return every obstacle row with x mirrored about the centerline, angle negated."""
    x = rows[..., OBSTACLE_FEATURE_X]
    reflected = rows.at[..., OBSTACLE_FEATURE_X].set(_trail(width, x.ndim) - x)
    return reflected.at[..., OBSTACLE_FEATURE_THETA].set(
        -rows[..., OBSTACLE_FEATURE_THETA]
    )


def obstacle_mirror_partners(table: Array, width: Array) -> Array:
    """Tell, per obstacle row, whether its mirror image is already in its own table.

    Parameters
    ----------
    table : Array
        Float32 Core obstacle rows ending in (32, 8), with any leading shape L,
        for example (B,) for one table per game or (B, 5) for the same table
        repeated per actor. Inactive rows are all zero.
    width : Array
        Map width per table, shape L; the reflection line is half of it.

    Returns
    -------
    Array
        Boolean array of shape L + (32,). True where an active row's reflection
        (x to width minus x, angle negated) matches an active row of the same
        table, itself included: same type, centre and radius within 1e-3 and,
        for walls, the same rectangle up to a half turn or with width and height
        swapped up to a quarter turn. Inactive rows are False.

    Notes
    -----
    Pure JAX; works under jit and vmap. This is the decision mirror_team_view
    makes for obstacle rows. It compares every row with every row of its own
    table, about 32 x 32 pairs per table, so a caller whose rows share one
    table per game (for example the five actors of one team) should compute it
    once per game and pass it to mirror_team_view as obstacle_partners rather
    than let every row repeat the same comparison. On the built-in maps, which
    are their own mirror images, every active row has a partner.
    """
    present = table[..., OBSTACLE_FEATURE_ACTIVE] > 0
    reflected = _reflected_obstacle_rows(table, width)
    partners = (
        _same_obstacle(reflected[..., :, None, :], table[..., None, :, :])
        & present[..., :, None]
        & present[..., None, :]
    )
    return jnp.any(partners, axis=-1)


def _reflect_obstacles(
    rows: Array, width: Array, flag: Array, partners: Array
) -> Array:
    """Reflect flagged views' obstacle rows that have no mirror partner.

    rows ends in (32, 8) Core obstacle rows; width and flag carry the leading
    shape; partners is obstacle_mirror_partners for these rows, shape leading +
    (32,). In flagged views a row whose reflection already exists in the table
    is returned as authored, because reflecting it would only move that shape
    into another slot; a row with no partner is reflected in place. On a map
    that is its own mirror image the table therefore comes back unchanged and
    the flattened encoding is identical from both spawn ends; on an asymmetric
    layout the geometry is still mirrored. Unflagged views and inactive rows
    are returned as given.
    """
    present = rows[..., OBSTACLE_FEATURE_ACTIVE] > 0
    select = _trail(flag, present.ndim) & present & ~partners
    return jnp.where(select[..., None], _reflected_obstacle_rows(rows, width), rows)


def _reflect_pads(pads: Array, width: Array, flag: Array) -> Array:
    """Reflect the x of every spawn pad in both banks for flagged views."""
    x = pads[..., 0]
    select = _trail(flag, x.ndim)
    return pads.at[..., 0].set(jnp.where(select, _trail(width, x.ndim) - x, x))


def _permute_moves(values: Array, flag: Array) -> Array:
    """Reorder a trailing 9-way move axis with MOVE_MIRROR in flagged views."""
    return jnp.where(
        _trail(flag, values.ndim), jnp.take(values, MOVE_MIRROR, axis=-1), values
    )


def mirror_team_view(
    actors: ActorInput,
    mask: ActionMask,
    flag: Array,
    *,
    obstacle_partners: Array | None = None,
) -> tuple[ActorInput, ActionMask]:
    """Reflect flagged actors' Team Deathmatch view and move mask about the centerline.

    Parameters
    ----------
    actors : ActorInput
        One team's permitted inputs with any leading shape L, for example
        (), (B,), (B, 5) or (G, T, E, 5). Rows are treated independently; no
        grouping of rows into games is assumed.
    mask : ActionMask
        The same actors' Core action masks with leading shape L.
    flag : Array
        Boolean array of shape L. True selects an actor whose view is reflected;
        False rows are returned unchanged.
    obstacle_partners : Array or None, default=None
        Optional Boolean array of shape L + (32,) from obstacle_mirror_partners
        for these rows' own obstacle tables. None computes it here for every
        row, which is exact for any leading shape. A caller that knows its
        rows share one table per game may compute the mask once per game and
        pass it broadcast to L; the output is the same, and the 32 x 32 row
        comparison is not repeated per row.

    Returns
    -------
    tuple[ActorInput, ActionMask]
        Reflected copies. In flagged rows the x column of every active unit row
        (self, allies, enemies and shared-sensor rows) becomes map width minus
        x, every spawn pad's x is reflected, an active obstacle row whose
        mirror image is not already in the table has x reflected and its angle
        negated while a row whose mirror image is present (itself included)
        stays as authored, and the ally and enemy previous-move one-hots and
        the move mask swap East with West, Northeast with Northwest and
        Southeast with Southwest. On the built-in maps, which are their own
        mirror images, the obstacle table is therefore unchanged and both
        spawn ends encode identically. Every other field is returned
        by reference: targets and Ultimates are roster relations, the joint
        combat mask is frame-free, context and lifecycle fields carry no x, and
        objective features are zero in Team Deathmatch. All-zero hidden rows,
        unavailable sources and unused obstacles stay zero.

    Raises
    ------
    TypeError
        flag or obstacle_partners is not a Boolean array.
    ValueError
        flag's shape differs from the actors' leading shape, or
        obstacle_partners' shape is not that leading shape plus (32,).

    Notes
    -----
    Pure JAX; works under jit and vmap with no host work. The map width comes
    from each actor's own context features and the reflection line is half of
    it. Reflected coordinates carry float32 rounding; the move permutations are
    exact. Rows keep their slots; obstacle rows are matched against their own
    table within 1e-3 world units and radians, so the same shape is never moved
    to another slot. Reflecting twice returns the input within rounding. A
    method that reflects its view must map its chosen move back with
    mirror_move before the game receives it. Cost: the reflection is
    elementwise over the permitted view; the obstacle matching is the only
    quadratic part, and obstacle_partners lets a caller pay it once per game.
    """
    observation = actors.observation
    leading = observation.self_features.shape[:-1]
    _check_flag(flag, leading)
    width = observation.context_features[..., CONTEXT_FEATURE_MAP_WIDTH]
    table = observation.map_obstacle_features
    if obstacle_partners is None:
        partners = obstacle_mirror_partners(table, width)
    else:
        if obstacle_partners.dtype != jnp.bool_:
            raise TypeError("obstacle_partners must be a Boolean array.")
        if obstacle_partners.shape != table.shape[:-1]:
            raise ValueError(
                "obstacle_partners must have the actors' leading shape plus (32,)."
            )
        partners = obstacle_partners
    previous = observation.previous_timestep_actions
    observation = observation._replace(
        self_features=_reflect_unit_rows(observation.self_features, width, flag),
        ally_unit_features=_reflect_unit_rows(
            observation.ally_unit_features, width, flag
        ),
        enemy_unit_features=_reflect_unit_rows(
            observation.enemy_unit_features, width, flag
        ),
        map_obstacle_features=_reflect_obstacles(table, width, flag, partners),
        previous_timestep_actions=previous._replace(
            ally_previous_timestep_move_actions_one_hot=_permute_moves(
                previous.ally_previous_timestep_move_actions_one_hot, flag
            ),
            enemy_previous_timestep_move_actions_one_hot=_permute_moves(
                previous.enemy_previous_timestep_move_actions_one_hot, flag
            ),
        ),
        spawn_lifecycle=observation.spawn_lifecycle._replace(
            spawn_pad_positions_by_agent_by_team=_reflect_pads(
                observation.spawn_lifecycle.spawn_pad_positions_by_agent_by_team,
                width,
                flag,
            )
        ),
    )
    bank = actors.source_bank._replace(
        unit_features_by_source_and_candidate=_reflect_unit_rows(
            actors.source_bank.unit_features_by_source_and_candidate, width, flag
        )
    )
    reflected = ActorInput(observation, bank, actors.source_availability)
    return reflected, mask._replace(move_mask=_permute_moves(mask.move_mask, flag))


def mirror_move(move: Array, flag: Array) -> Array:
    """Map move categories chosen in a reflected view back to world directions.

    Parameters
    ----------
    move : Array
        Integer move categories 0..8 with any shape L.
    flag : Array
        Boolean array of shape L, or broadcastable to it. True rows are mapped
        through MOVE_MIRROR; False rows are returned unchanged.

    Returns
    -------
    Array
        Integer array of the same shape and dtype as move. The mapping is an
        involution: applying it twice returns the input.

    Raises
    ------
    TypeError
        move is not an integer array or flag is not Boolean.
    """
    if not jnp.issubdtype(move.dtype, jnp.integer):
        raise TypeError("Move categories must be an integer array.")
    if flag.dtype != jnp.bool_:
        raise TypeError("The mirror flag must be a Boolean array.")
    return jnp.where(flag, MOVE_MIRROR[move].astype(move.dtype), move)


__all__ = (
    "MOVE_MIRROR",
    "ActorInput",
    "Observations",
    "build_actor_input",
    "build_observations",
    "build_team_actor_input",
    "mirror_move",
    "mirror_team_view",
    "obstacle_mirror_partners",
    "team_on_right",
)
