# pyright: reportUnknownMemberType=false

import jax.numpy as jnp
import pytest
from jax import Array


def test_simulator_spine_invariants() -> None:
    from marl_battlegrounds.core.types import (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        NUM_TEAMS,
    )

    assert MAX_AGENT_SLOTS == 10
    assert MAX_AGENTS_PER_TEAM == 5
    assert NUM_TEAMS == 2


def test_env_config_construction() -> None:

    from marl_battlegrounds.core.types import EnvConfig

    env_conf = EnvConfig(team_size=5, max_steps=10000)

    assert env_conf.max_steps == 10000
    assert env_conf.team_size == 5


def test_env_state_construction() -> None:

    from marl_battlegrounds.core.types import MAX_AGENT_SLOTS, EnvState

    env_state = EnvState(
        step_count=jnp.array(1),
        agent_positions=jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        team_ids=jnp.ones(shape=(MAX_AGENT_SLOTS,)),
        active_mask=jnp.ones(shape=(MAX_AGENT_SLOTS,)),
        alive_mask=jnp.ones(shape=(MAX_AGENT_SLOTS,)),
    )

    assert env_state.step_count.shape == ()
    assert env_state.agent_positions.shape == (MAX_AGENT_SLOTS, 2)
    assert env_state.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert env_state.active_mask.shape == (MAX_AGENT_SLOTS,)
    assert env_state.alive_mask.shape == (MAX_AGENT_SLOTS,)


def test_action_construction() -> None:

    from marl_battlegrounds.core.types import MAX_AGENT_SLOTS, Action

    joint_action = Action(
        move=jnp.ones(shape=(MAX_AGENT_SLOTS,)),
        target=jnp.ones(shape=(MAX_AGENT_SLOTS,)),
        use_ultimate=jnp.ones(shape=(MAX_AGENT_SLOTS,)),
    )

    assert joint_action.move.shape == (MAX_AGENT_SLOTS,)
    assert joint_action.target.shape == (MAX_AGENT_SLOTS,)
    assert joint_action.use_ultimate.shape == (MAX_AGENT_SLOTS,)


def test_observation_construction() -> None:

    from marl_battlegrounds.core.types import (
        MAX_AGENT_SLOTS,
        NUM_OBSERVATION_FEATURES,
        Observation,
    )

    obs_vecs = Observation(
        observation_vectors=jnp.array(
            [
                jnp.ones(shape=(NUM_OBSERVATION_FEATURES,))
                for _ in range(MAX_AGENT_SLOTS)
            ]
        ),
    )
    assert obs_vecs.observation_vectors.shape == (
        MAX_AGENT_SLOTS,
        NUM_OBSERVATION_FEATURES,
    )


def test_action_mask_construction() -> None:

    from marl_battlegrounds.core.types import (
        MAX_AGENT_SLOTS,
        NUM_MOVE_ACTIONS,
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
        ActionMask,
    )

    action_mask = ActionMask(
        move=jnp.ones(shape=(MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS), dtype=bool),
        target=jnp.ones(shape=(MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS), dtype=bool),
        use_ultimate=jnp.ones(
            shape=(MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS), dtype=bool
        ),
    )

    assert action_mask.move.shape == (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert action_mask.move.dtype == bool
    assert action_mask.target.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.target.dtype == bool
    assert action_mask.use_ultimate.shape == (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS)
    assert action_mask.use_ultimate.dtype == bool


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected"),
    [
        (jnp.array(0, dtype=bool), jnp.array(0, dtype=bool), jnp.array(0, dtype=bool)),
        (jnp.array(1, dtype=bool), jnp.array(0, dtype=bool), jnp.array(1, dtype=bool)),
        (jnp.array(0, dtype=bool), jnp.array(1, dtype=bool), jnp.array(1, dtype=bool)),
        (jnp.array(1, dtype=bool), jnp.array(1, dtype=bool), jnp.array(1, dtype=bool)),
    ],
)
def test_termination_and_truncation_return_correct_done_flag_construction(
    terminated: Array, truncated: Array, expected: Array
) -> None:

    from marl_battlegrounds.core.types import DoneFlags

    current_done_flag = DoneFlags(terminated=terminated, truncated=truncated)
    done_flag = current_done_flag.done
    assert jnp.array_equal(done_flag, expected)


def test_info_construction() -> None:

    from marl_battlegrounds.core.types import Info

    info_object = Info()

    assert info_object is not None
