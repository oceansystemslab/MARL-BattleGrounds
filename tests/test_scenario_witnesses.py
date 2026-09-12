"""Saved per-turn results for every packaged solution beyond Scenario 3.

Scenario 3 has its own eight-turn and draw tests using the same comparison.
Older authoring-revision witness and ablation tests remain separate.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.scenario_witness_assertions import assert_witness_tick

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    TEAM_B_ID,
    ActionMask,
    DoneFlags,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
    reactive_tdm_alpha_policy,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
    reactive_tdm_beta_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)
from marl_battlegrounds.tasks import load_tdm_scenario

_TEAM = cast(
    Callable[..., ActorAction],
    jax.jit(execute_shared_obs_team_policy, static_argnums=(5, 6)),
)
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)
_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("scenario_id", "case_name"),
    [
        (1, "scenario_1_heal_hunter"),
        (1, "scenario_1_heal_rogue"),
        (2, "scenario_2"),
        (4, "scenario_4"),
        (5, "scenario_5"),
        (6, "scenario_6"),
        (7, "scenario_7"),
        (8, "scenario_8"),
    ],
)
def test_packaged_solution_preserves_every_turn(
    scenario_id: int, case_name: str
) -> None:
    """Replay only Team A's chosen commands; Team B must still choose for itself."""
    expected = json.loads((_FIXTURES / f"{case_name}_witness.json").read_text())
    scenario = load_tdm_scenario(scenario_id)
    assert expected["scenario_id"] == scenario_id
    assert scenario.info.approved_source.revision == expected["approved_revision"]
    assert scenario.info.source.revision == expected["source_revision"]
    assert scenario.info.source.semantic_digest == expected["semantic_digest"]
    assert (
        scenario.info.resolved_configuration_digest == expected["configuration_digest"]
    )
    assert (
        scenario.info.resolved_initial_state_digest == expected["initial_state_digest"]
    )
    beta = scenario_id in (5, 8)
    descriptor = (
        reactive_tdm_beta_controller_descriptor()
        if beta
        else reactive_tdm_alpha_controller_descriptor()
    )
    assert expected["opponent"] == ("tdm-beta" if beta else "tdm-alpha")
    assert expected["opponent_version"] == descriptor["version"]
    constant_key = scenario_id in (1, 2, 4)
    assert expected["key_schedule"] == (
        "constant_zero" if constant_key else "turn_index"
    )
    opponent = reactive_tdm_beta_policy if beta else reactive_tdm_alpha_policy

    state, observation, mask, _ = initialize_scenario_state(
        scenario.initial_state, scenario.config
    )
    profile = scenario.config.agent_profile
    availability = build_default_shared_obs_information_availability(
        profile.active_mask, profile.team_ids
    )
    rows = expected["ticks"]
    assert rows, f"{case_name}: missing expected turns"
    slots = np.arange(10)
    for tick, row in enumerate(rows, start=1):
        assert row["tick"] == tick
        key = jax.random.key(0 if constant_key else tick - 1)
        # Expected opponent actions are assertions only, never policy inputs.
        team_b = _TEAM(
            observation,
            mask,
            jax.random.split(key, 10),
            build_shared_obs_sensor_source_bank(observation),
            availability,
            opponent,
            TEAM_B_ID,
        )
        team_a = ActorAction(
            *(
                jnp.asarray(row[head][:5], dtype=jnp.int32)
                for head in ActorAction._fields
            )
        )
        action = build_joint_action_from_actor_actions(team_a, team_b)
        assert np.asarray(mask.move_mask)[slots, np.asarray(action.move)].all()
        assert np.asarray(mask.select_target_use_ultimate_joint_mask)[
            slots, np.asarray(action.select_target), np.asarray(action.use_ultimate)
        ].all()
        before_step = int(state.step_count)
        state, observation, reward, done, mask, info = _STEP(
            scenario.config, state, mask, action, key
        )
        assert int(state.step_count) == before_step + 1
        assert_witness_tick(
            row,
            state=state,
            action=action,
            reward=reward,
            done=done,
            info=info,
            case_name=case_name,
        )
        # No padding or extra turns may hide an early or late end.
        assert bool(done.done) == (tick == len(rows))

    np.testing.assert_array_equal(state.team_deathmatch_scores, [20, 19])
