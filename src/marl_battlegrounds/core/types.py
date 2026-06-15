"""Core type contracts for the minimal JAX simulator spine."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

NUM_MOVE_ACTIONS = 9
NUM_ULTIMATE_ACTIONS = 2
NUM_TEAMS = 2
NUM_OBSERVATION_FEATURES = 1
MAX_AGENTS_PER_TEAM = 5
MAX_AGENT_SLOTS = NUM_TEAMS * MAX_AGENTS_PER_TEAM
NUM_TARGET_ACTIONS = MAX_AGENT_SLOTS + 1
ENVIRONMENT_DIMENSIONS = 2


class EnvConfig(NamedTuple):
    """Static episode settings consumed by reset and step functions."""

    team_size: int
    max_steps: int


class EnvState(NamedTuple):
    """Dynamic simulator state carried through functional transitions."""

    step_count: Array
    agent_positions: Array
    team_ids: Array
    active_mask: Array
    alive_mask: Array


class Action(NamedTuple):
    """Factored joint action supplied by policies for all agent slots."""

    move: Array
    target: Array
    use_ultimate: Array


class ActionMask(NamedTuple):
    """Joint validity masks for each factored action head."""

    move: Array
    target: Array
    use_ultimate: Array


class Observation(NamedTuple):
    """Per-slot observation vectors emitted by reset and step."""

    observation_vectors: Array


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
