"""Core type contracts for the JAX-native simulator spine."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

NUM_MOVE_ACTIONS = 9
NUM_ULTIMATE_ACTIONS = 2
NUM_TEAMS = 2
MAX_AGENTS_PER_TEAM = 5
MAX_AGENT_SLOTS = NUM_TEAMS * MAX_AGENTS_PER_TEAM
NUM_TARGET_ACTIONS = MAX_AGENT_SLOTS + 1
ENVIRONMENT_DIMENSIONS = 2
MAX_OBSTACLE_SLOTS = 16
OBSTACLE_FEATURES = 8
OBSTACLE_TYPE_NONE = 0
OBSTACLE_TYPE_PILLAR = 1
OBSTACLE_TYPE_WALL = 2
OBSTACLE_FEATURE_TYPE = 0
OBSTACLE_FEATURE_X = 1
OBSTACLE_FEATURE_Y = 2
OBSTACLE_FEATURE_RADIUS = 3
OBSTACLE_FEATURE_WIDTH = 4
OBSTACLE_FEATURE_HEIGHT = 5
OBSTACLE_FEATURE_THETA = 6
OBSTACLE_FEATURE_ACTIVE = 7
MOVE_STAY = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_EAST = 3
MOVE_WEST = 4
MOVE_NORTHEAST = 5
MOVE_NORTHWEST = 6
MOVE_SOUTHEAST = 7
MOVE_SOUTHWEST = 8
CLASS_NEUTRAL = 0
SELF_FEATURES = 16
UNIT_FEATURES = 16
MAX_OBJECTIVE_SLOTS = 8
OBJECTIVE_FEATURES = 12
CONTEXT_FEATURES = 8

# Self rows and unit-candidate rows use one shared agent-feature schema.
# self_features exists only for convenient actor conditioning; ally/enemy unit
# rows are relation-indexed candidate uses of the same AGENT_FEATURE_* contract.
# Keep SELF_FEATURES == UNIT_FEATURES unless a future schema decision explicitly
# splits these families.
AGENT_FEATURE_X = 0
AGENT_FEATURE_Y = 1
AGENT_FEATURE_RADIUS = 2
AGENT_FEATURE_TEAM_ID = 3
AGENT_FEATURE_ACTIVE = 4
AGENT_FEATURE_ALIVE = 5
AGENT_FEATURE_CLASS_ID = 6
AGENT_FEATURE_MOVEMENT_SPEED = 7
AGENT_FEATURE_OBSERVATION_RADIUS = 8
AGENT_FEATURE_BASIC_INTERACTION_RADIUS = 9
AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS = 10


class EnvConfig(NamedTuple):
    """Static episode settings and reset defaults.

    Static map geometry belongs here because Milestone 4 obstacles do not change
    during an episode. ``default_*`` stat values are construction defaults for
    reset and scenario loading only; after ``EnvState`` exists, per-slot state
    arrays are the simulator truth for transition, observation, and masks.
    """

    team_size: int
    max_steps: int
    map_width: float
    map_height: float
    default_agent_radius: float
    default_movement_speed: float
    default_observation_radius: float
    default_basic_interaction_radius: float
    default_ultimate_interaction_radius: float
    obstacles: Array


class EnvState(NamedTuple):
    """Dynamic slot-aligned simulator state carried through transitions.

    Current stat arrays are authoritative per-slot effective values. Future
    class catalogs and status systems may derive or update these values, but
    step, observation, and masks consume the arrays here rather than config
    defaults.
    """

    step_count: Array
    agent_positions: Array
    agent_radii: Array
    team_ids: Array
    class_ids: Array
    movement_speeds: Array
    observation_radii: Array
    basic_interaction_radii: Array
    ultimate_interaction_radii: Array
    active_mask: Array
    alive_mask: Array


class Action(NamedTuple):
    """Factored joint action supplied by policies for every agent slot."""

    move: Array
    target: Array
    use_ultimate: Array


class ActionMask(NamedTuple):
    """Slot-aligned validity masks for each factored action head."""

    move: Array
    target: Array
    use_ultimate: Array


class Observation(NamedTuple):
    """Structured per-slot observations emitted by reset and step."""

    self_features: Array
    ally_unit_features: Array
    enemy_unit_features: Array
    map_obstacle_features: Array
    objective_features: Array
    context_features: Array
    ally_visibility_mask: Array
    enemy_visibility_mask: Array
    ally_targetability_mask: Array
    enemy_targetability_mask: Array


class Reward(NamedTuple):
    """Slot-aligned scalar rewards emitted by the core simulator."""

    rewards: Array


class DoneFlags(NamedTuple):
    """Episode termination and truncation signals from the core simulator."""

    terminated: Array
    truncated: Array

    @property
    def done(self) -> Array:
        """Whether rollout control should stop for this episode."""

        return jnp.logical_or(self.terminated, self.truncated)


class Info(NamedTuple):
    """Lightweight auxiliary diagnostics placeholder for reset and step."""
