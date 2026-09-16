"""Check SharedObs input delivery across death, reset and JAX batching."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_IS_ENEMY,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    OBJECTIVE_FEATURES,
    UNIT_FEATURES,
    Action,
    ActionMask,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    build_actor_input,
    build_observations,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    SharedObsSensorSourceBankV2,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
    mask_source_bank_for_recipient,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


def _assert_tree_exact(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.shape == expected_array.shape
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_array_equal(actual_array, expected_array)


def _echo_delivered_input(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
    source_bank: SharedObsSensorSourceBankV2,
    source_availability: Array,
) -> ActorInput:
    del action_mask, key
    return ActorInput(observation, source_bank, source_availability)


def _dead_teammate_observation(config: EnvConfig) -> tuple[Observation, ActionMask]:
    state, _, _, _ = reset(config, jax.random.key(0))
    state = state._replace(
        alive_mask=state.alive_mask.at[1].set(False),
        current_health=state.current_health.at[1].set(0.0),
    )
    _, observation, action_mask, _ = initialize_scenario_state(state, config)
    return observation, action_mask


def test_actor_input_preserves_closed_fields_shapes_and_dtypes() -> None:
    config = evaluation_env_config()
    _, observation, _, _ = reset(config, jax.random.key(0))
    actual = build_actor_input(observation, config)
    assert ActorInput._fields == (
        "observation",
        "source_bank",
        "source_availability",
    )
    assert actual.observation is observation
    bank = actual.source_bank
    assert bank.unit_features_by_source_and_candidate.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        MAX_AGENT_SLOTS,
        UNIT_FEATURES,
    )
    assert bank.unit_features_by_source_and_candidate.dtype == jnp.float32
    assert bank.unit_visibility_by_source_and_candidate.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        MAX_AGENT_SLOTS,
    )
    assert bank.unit_visibility_by_source_and_candidate.dtype == jnp.bool_
    assert bank.objective_features_by_source.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        MAX_OBJECTIVE_SLOTS,
        OBJECTIVE_FEATURES,
    )
    assert bank.objective_features_by_source.dtype == jnp.float32
    assert actual.source_availability.shape == (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM)
    assert actual.source_availability.dtype == jnp.bool_
    assert actual.observation.self_ally_index.dtype == jnp.int32
    np.testing.assert_array_equal(
        actual.observation.self_ally_index, [0, 1, 2, 0, 0, 0, 1, 0, 0, 0]
    )


def test_actor_input_matches_direct_delivery_for_both_teams_and_lifecycles() -> None:
    config = evaluation_env_config()
    _, living_observation, living_mask, _ = reset(config, jax.random.key(0))
    dead_observation, dead_mask = _dead_teammate_observation(config)
    keys = jax.random.split(jax.random.key(31), MAX_AGENT_SLOTS)
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask, config.agent_profile.team_ids
    )
    for observation, action_mask in (
        (living_observation, living_mask),
        (dead_observation, dead_mask),
    ):
        actual = build_actor_input(observation, config)
        bank = build_shared_obs_sensor_source_bank(observation)
        for team in (1, 2):
            # The executor only transports policy outputs. A test-only echo
            # lets us compare every delivered leaf, without lossy checksums.
            expected = cast(
                ActorInput,
                execute_shared_obs_team_policy(
                    observation,
                    action_mask,
                    keys,
                    bank,
                    availability,
                    cast(SharedObsPolicy, _echo_delivered_input),
                    team,
                ),
            )
            start = (team - 1) * MAX_AGENTS_PER_TEAM

            def take_team(leaf: Array, start: int = start) -> Array:
                return leaf[start : start + MAX_AGENTS_PER_TEAM]

            _assert_tree_exact(jax.tree.map(take_team, actual), expected)


def test_actor_input_removes_unavailable_and_dead_source_material() -> None:
    config = evaluation_env_config()
    observation, _ = _dead_teammate_observation(config)
    actual = build_actor_input(observation, config)
    availability = np.asarray(actual.source_availability)
    assert availability[0, 1]  # Dead teammate remains authorized, with no sensing.
    assert not availability[0, 0]
    assert availability.shape[1] == MAX_AGENTS_PER_TEAM  # No opponent sources.
    assert not availability[3].any()  # Inactive recipient.
    for leaf in jax.tree.leaves(actual.source_bank):
        values = np.asarray(leaf)
        assert not np.any(values[~availability])
        assert not np.any(values[:5, 1])  # Dead source on the first team.
        assert not np.any(values[:, 3])  # Inactive source.

    poisoned = observation._replace(
        ally_unit_features=observation.ally_unit_features.at[5].set(73.0),
        enemy_unit_features=observation.enemy_unit_features.at[5].set(91.0),
        ally_visibility_mask=observation.ally_visibility_mask.at[5].set(True),
        enemy_visibility_mask=observation.enemy_visibility_mask.at[5].set(True),
        objective_features=observation.objective_features.at[5].set(41.0),
    )
    poisoned_input = build_actor_input(poisoned, config)

    def first_actor(leaf: Array) -> Array:
        return leaf[0]

    _assert_tree_exact(
        jax.tree.map(first_actor, poisoned_input), jax.tree.map(first_actor, actual)
    )


def test_actor_input_matches_compilation_and_external_batching() -> None:
    configs = (
        evaluation_env_config(team_sizes=(3, 2)),
        evaluation_env_config(team_sizes=(1, 5)),
    )
    observations = tuple(reset(config, jax.random.key(0))[1] for config in configs)
    compiled = jax.jit(build_actor_input).lower(observations[0], configs[0]).compile()
    for observation, config in zip(observations, configs, strict=True):
        _assert_tree_exact(
            compiled(observation, config), build_actor_input(observation, config)
        )

    def stack(*leaves: Array) -> Array:
        return jnp.stack(leaves)

    batched_observation = jax.tree.map(stack, *observations)
    batched_config = jax.tree.map(stack, *configs)
    actual = cast(
        ActorInput,
        jax.jit(jax.vmap(build_actor_input))(batched_observation, batched_config),
    )
    expected = jax.tree.map(
        stack,
        *(
            build_actor_input(obs, cfg)
            for obs, cfg in zip(observations, configs, strict=True)
        ),
    )
    _assert_tree_exact(actual, expected)


def test_compact_observations_reconstruct_authorized_inputs_without_stored_banks() -> (
    None
):
    config = evaluation_env_config()
    observation, _ = _dead_teammate_observation(config)
    compact = build_observations(observation, config)
    assert Observations._fields == ("observation", "source_availability")
    assert compact.observation is observation
    assert compact.source_availability.shape == (10, 10)
    assert compact.source_availability.dtype == jnp.bool_
    shared_bank = build_shared_obs_sensor_source_bank(compact.observation)
    local_availability = jnp.stack(
        (compact.source_availability[:5, :5], compact.source_availability[5:, 5:])
    )
    team_banks = jax.vmap(jax.vmap(mask_source_bank_for_recipient, in_axes=(None, 0)))(
        shared_bank, local_availability
    )

    def flatten_actors(value: Array) -> Array:
        return value.reshape((10, *value.shape[2:]))

    banks = jax.tree.map(flatten_actors, team_banks)
    reconstructed = ActorInput(
        compact.observation,
        banks,
        local_availability.reshape((10, 5)),
    )
    expected = build_actor_input(observation, config)
    _assert_tree_exact(reconstructed, expected)

    def byte_count(tree: object) -> int:
        return sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(tree))

    assert byte_count(compact) == byte_count(observation) + 100
    assert byte_count(expected) == byte_count(observation) + 10 * (
        5 * 10 * 58 * 4 + 5 * 10 + 5 * 8 * 12 * 4 + 5
    )
    assert compact.source_availability[0, 1]  # Authorized dead source remains empty.
    assert not compact.source_availability[
        0, 5
    ]  # Opponent source never becomes visible.
    for leaf in jax.tree.leaves(reconstructed.source_bank):
        assert not np.asarray(leaf)[:5, 1].any()
        assert np.asarray(leaf).shape[1] == 5


@pytest.mark.parametrize("sizes", [(5, 5), (1, 5), (3, 2)])
def test_same_physical_actors_receive_equal_inputs_after_team_block_exchange(
    sizes: tuple[int, int],
) -> None:
    config = evaluation_env_config(team_sizes=sizes)._replace(
        spawn_shield_duration_steps=0
    )
    order = jnp.asarray([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=jnp.int32)

    def exchange(value: Array) -> Array:
        return value[order]

    profile = jax.tree.map(exchange, config.agent_profile)
    profile = profile._replace(
        team_ids=jnp.where(profile.active_mask, jnp.repeat(jnp.asarray([1, 2]), 5), 0)
    )
    exchanged = config._replace(
        agent_profile=profile,
        team_spawn_pad_positions=config.team_spawn_pad_positions[::-1],
        team_respawn_wave_period_step_count=config.team_respawn_wave_period_step_count[
            ::-1
        ],
    )
    _, original, original_masks, _ = reset(config, jax.random.key(9))
    _, swapped, swapped_masks, _ = reset(exchanged, jax.random.key(9))
    original_inputs = build_actor_input(original, config)
    swapped_inputs = build_actor_input(swapped, exchanged)
    _assert_tree_exact(original_inputs, jax.tree.map(exchange, swapped_inputs))
    _assert_tree_exact(original_masks, jax.tree.map(exchange, swapped_masks))


def test_relation_flags_and_self_lookup_survive_hidden_rows_death_and_respawn() -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("mage", "mage", "priest"),
        team_b_roster=("hunter", "hunter"),
        max_steps=20,
    )
    state, _, _, _ = reset(config, jax.random.key(0))
    dead = jnp.asarray([1, 6], dtype=jnp.int32)
    state = state._replace(
        alive_mask=state.alive_mask.at[dead].set(False),
        current_health=state.current_health.at[dead].set(0.0),
    )
    state, observation, mask, _ = initialize_scenario_state(state, config)
    no_op = Action(*(jnp.zeros((10,), dtype=jnp.int32) for _ in range(3)))
    expected_index = [0, 1, 2, 0, 0, 0, 1, 0, 0, 0]
    for _ in range(6):
        np.testing.assert_array_equal(observation.self_ally_index, expected_index)
        np.testing.assert_array_equal(
            observation.self_features[:, AGENT_FEATURE_IS_ENEMY], 0
        )
        np.testing.assert_array_equal(
            observation.ally_unit_features[:, :, AGENT_FEATURE_IS_ENEMY], 0
        )
        np.testing.assert_array_equal(
            observation.enemy_unit_features[:, :, AGENT_FEATURE_IS_ENEMY],
            observation.enemy_visibility_mask.astype(jnp.float32),
        )
        assert not np.asarray(observation.enemy_unit_features)[
            ~np.asarray(observation.enemy_visibility_mask)
        ].any()
        assert observation.spawn_lifecycle.active_mask_by_agent_by_team[0, 1, 1]
        assert not observation.spawn_lifecycle.active_mask_by_agent_by_team[0, 1, 2]
        assert observation.self_features[1, AGENT_FEATURE_ACTIVE] == 1
        state, observation, _, _, mask, _ = step(
            config, state, mask, no_op, jax.random.key(1)
        )
    assert bool(jnp.all(state.alive_mask[dead]))
