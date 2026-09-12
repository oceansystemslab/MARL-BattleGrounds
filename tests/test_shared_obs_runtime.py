"""Structured SharedObs composition, adapter, and provenance proofs."""

from operator import itemgetter
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import (
    evaluation_context,
    evaluation_env_config,
    historical_observation,
)

from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MOVE_STAY,
    OBJECTIVE_FEATURES,
    TASK_MODE_TDM,
    UNIT_FEATURES,
    ActionMask,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.evaluation.actor_projection import (
    SHARED_OBS_ACTOR_PROJECTION_ID,
    SHARED_OBS_ACTOR_PROJECTION_V1,
    SHARED_OBS_ACTOR_PROJECTION_VERSION,
    reconstruct_shared_obs_sensor_source_bank_v1,
)
from marl_battlegrounds.evaluation.capture import capture_initial_evaluation_frame_v1
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    StaticMechanicsCatalogV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.no_shared_obs import (
    execute_no_shared_obs_team_policy,
)
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.reactive_tdm_alpha import reactive_tdm_alpha_policy
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV1,
    SharedObsSensorSourceBankV2,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    build_shared_obs_sensor_source_bank_from_base_rows,
    compose_shared_obs_unit_features,
    execute_shared_obs_team_policy,
)


def _shared_random_policy(
    observation: Observation,
    action_mask: ActionMask,
    actor_key: Array,
    source_bank: SharedObsSensorSourceBankV2,
    recipient_source_availability: Array,
) -> ActorAction:
    """Test-only SharedObs ABI wrapper retaining canonical Random key sensitivity."""
    del source_bank, recipient_source_availability
    return random_policy(observation, action_mask, actor_key)


def _tdm_config(
    *,
    team_sizes: tuple[int, int] = (3, 2),
    max_steps: int = 1,
) -> EnvConfig:
    """Return an unshielded TDM configuration for policy execution."""
    return evaluation_env_config(
        team_sizes=team_sizes,
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=100,
        max_steps=max_steps,
    )._replace(spawn_shield_duration_steps=0)


def _scalar_actor(tree: object, global_slot: int) -> object:
    """Select one global-slot row from every leaf in a fixed actor PyTree."""

    def _take_actor_row(leaf: Array) -> Array:
        return leaf[global_slot]

    return jax.tree.map(_take_actor_row, tree)


def _team_bank(observation: Observation, team: int = 0) -> SharedObsSensorSourceBankV2:
    """Select the own-team source rows before calling an actor policy."""
    return jax.tree.map(
        itemgetter(team), build_shared_obs_sensor_source_bank(observation)
    )


def _historical_bank(observation: Observation) -> SharedObsSensorSourceBankV1:
    """Build the explicitly historical Team-ID fixture's original source layout."""
    return build_shared_obs_sensor_source_bank_from_base_rows(
        observation.ally_unit_features,
        observation.enemy_unit_features,
        observation.objective_features,
        observation.ally_visibility_mask,
        observation.enemy_visibility_mask,
        (observation.self_features[:, AGENT_FEATURE_ACTIVE] > 0)
        & (observation.self_features[:, AGENT_FEATURE_ALIVE] > 0),
    )


def _assert_tree_exact(actual: object, expected: object) -> None:
    """Require exact structure, shape, dtype, and values."""
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.shape == expected_array.shape
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_array_equal(actual_array, expected_array)


def _bank_with_one_sighting(
    source_bank: SharedObsSensorSourceBankV2,
    *,
    sensor_source: int,
    candidate: int,
    candidate_features: Array,
) -> SharedObsSensorSourceBankV2:
    """Return a bank whose selected source carries one explicit candidate row."""
    return source_bank._replace(
        unit_features_by_source_and_candidate=(
            source_bank.unit_features_by_source_and_candidate.at[
                sensor_source, candidate
            ].set(candidate_features)
        ),
        unit_visibility_by_source_and_candidate=(
            source_bank.unit_visibility_by_source_and_candidate.at[
                sensor_source, candidate
            ].set(True)
        ),
    )


def test_source_bank_has_exact_closed_fields_shapes_dtypes_and_relative_joins() -> None:
    """Every team/source/candidate row preserves the authored relative sensing."""
    config = _tdm_config()
    _, observation, _, _ = reset(config, jax.random.key(0))
    bank = build_shared_obs_sensor_source_bank(observation)

    assert SharedObsSensorSourceBankV2._fields == (
        "unit_features_by_source_and_candidate",
        "unit_visibility_by_source_and_candidate",
        "objective_features_by_source",
    )
    assert bank.unit_features_by_source_and_candidate.shape == (
        2,
        MAX_AGENTS_PER_TEAM,
        MAX_AGENT_SLOTS,
        UNIT_FEATURES,
    )
    assert bank.unit_features_by_source_and_candidate.dtype == jnp.float32
    assert bank.unit_visibility_by_source_and_candidate.shape == (
        2,
        MAX_AGENTS_PER_TEAM,
        MAX_AGENT_SLOTS,
    )
    assert bank.unit_visibility_by_source_and_candidate.dtype == jnp.bool_
    assert bank.objective_features_by_source.shape == (
        2,
        MAX_AGENTS_PER_TEAM,
        MAX_OBJECTIVE_SLOTS,
        OBJECTIVE_FEATURES,
    )
    assert bank.objective_features_by_source.dtype == jnp.float32

    for source in range(MAX_AGENT_SLOTS):
        source_living = bool(
            (observation.self_features[source, AGENT_FEATURE_ACTIVE] > 0.0)
            & (observation.self_features[source, AGENT_FEATURE_ALIVE] > 0.0)
        )
        for candidate in range(MAX_AGENT_SLOTS):
            row = candidate % MAX_AGENTS_PER_TEAM
            if candidate < MAX_AGENTS_PER_TEAM:
                expected_visible = bool(observation.ally_visibility_mask[source, row])
                expected_features = observation.ally_unit_features[source, row]
            else:
                expected_visible = bool(observation.enemy_visibility_mask[source, row])
                expected_features = observation.enemy_unit_features[source, row]
            expected_visible = source_living and expected_visible
            assert (
                bool(
                    bank.unit_visibility_by_source_and_candidate[
                        source // 5, source % 5, candidate
                    ]
                )
                is expected_visible
            )
            if expected_visible:
                np.testing.assert_array_equal(
                    np.asarray(
                        bank.unit_features_by_source_and_candidate[
                            source // 5, source % 5, candidate
                        ]
                    ),
                    np.asarray(expected_features),
                )
            else:
                assert bool(
                    jnp.all(
                        bank.unit_features_by_source_and_candidate[
                            source // 5, source % 5, candidate
                        ]
                        == 0.0
                    )
                )


def test_default_availability_is_static_same_team_active_and_off_diagonal() -> None:
    """Asymmetric rosters retain dead-source authorization but exclude padding."""
    config = _tdm_config(team_sizes=(3, 2))
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )

    expected = np.zeros((MAX_AGENT_SLOTS, MAX_AGENT_SLOTS), dtype=np.bool_)
    expected[:3, :3] = True
    expected[5:7, 5:7] = True
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(np.asarray(availability), expected)
    assert availability.dtype == jnp.bool_


def test_dead_source_is_authorized_but_contributes_no_sensor_or_objective_rows() -> (
    None
):
    """Source lifecycle redaction is independent from static authorization."""
    config = _tdm_config(team_sizes=(2, 1))
    _, observation, _, _ = reset(config, jax.random.key(0))
    poisoned = observation._replace(
        self_features=observation.self_features.at[1, AGENT_FEATURE_ALIVE].set(0.0),
        ally_unit_features=observation.ally_unit_features.at[1].set(7.0),
        enemy_unit_features=observation.enemy_unit_features.at[1].set(9.0),
        ally_visibility_mask=observation.ally_visibility_mask.at[1].set(True),
        enemy_visibility_mask=observation.enemy_visibility_mask.at[1].set(True),
        objective_features=observation.objective_features.at[1].set(11.0),
    )
    bank = build_shared_obs_sensor_source_bank(poisoned)
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )

    assert bool(availability[0, 1])
    assert not bool(jnp.any(bank.unit_visibility_by_source_and_candidate[0, 1]))
    assert bool(jnp.all(bank.unit_features_by_source_and_candidate[0, 1] == 0.0))
    assert bool(jnp.all(bank.objective_features_by_source[0, 1] == 0.0))


def test_compositor_preserves_base_rows_and_ignores_every_unavailable_mutation() -> (
    None
):
    """Only an admitted source can add a hidden globally joined candidate row."""
    config = _tdm_config(team_sizes=(2, 1))
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    recipient = cast(Observation, _scalar_actor(observation, 0))
    recipient = recipient._replace(
        enemy_unit_features=jnp.zeros_like(recipient.enemy_unit_features),
        enemy_visibility_mask=jnp.zeros_like(recipient.enemy_visibility_mask),
    )
    recipient_mask = cast(ActionMask, _scalar_actor(action_mask, 0))
    bank = _team_bank(observation)
    bank = _bank_with_one_sighting(
        bank,
        sensor_source=1,
        candidate=MAX_AGENTS_PER_TEAM,
        candidate_features=observation.self_features[MAX_AGENTS_PER_TEAM],
    )

    unavailable = jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.bool_)
    base_composite = compose_shared_obs_unit_features(
        recipient,
        bank,
        unavailable,
    )
    np.testing.assert_array_equal(
        np.asarray(base_composite[0]), np.asarray(recipient.ally_unit_features)
    )
    np.testing.assert_array_equal(
        np.asarray(base_composite[1]), np.asarray(recipient.enemy_unit_features)
    )
    np.testing.assert_array_equal(
        np.asarray(base_composite[2]), np.asarray(recipient.ally_visibility_mask)
    )
    np.testing.assert_array_equal(
        np.asarray(base_composite[3]), np.asarray(recipient.enemy_visibility_mask)
    )

    poisoned_unavailable = _bank_with_one_sighting(
        bank,
        sensor_source=4,
        candidate=MAX_AGENTS_PER_TEAM,
        candidate_features=jnp.full((UNIT_FEATURES,), 123.0, dtype=jnp.float32),
    )
    _assert_tree_exact(
        compose_shared_obs_unit_features(
            recipient,
            poisoned_unavailable,
            unavailable,
        ),
        base_composite,
    )

    admitted = unavailable.at[1].set(True)
    ally, enemy, ally_visible, enemy_visible = compose_shared_obs_unit_features(
        recipient,
        bank,
        admitted,
    )
    del ally, ally_visible, recipient_mask
    assert bool(enemy_visible[0])
    np.testing.assert_array_equal(
        np.asarray(enemy[0]),
        np.asarray(observation.self_features[MAX_AGENTS_PER_TEAM]),
    )


def test_random_adapter_preserves_local_policy_actions_and_key_sensitivity() -> None:
    """A test-only ABI wrapper preserves the unchanged policy's per-key output."""
    config = _tdm_config()
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    bank = _team_bank(observation)
    recipient = cast(Observation, _scalar_actor(observation, 0))
    recipient_mask = cast(ActionMask, _scalar_actor(action_mask, 0))
    unavailable = jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.bool_)
    moves: list[int] = []
    for seed in range(8):
        key = jax.random.key(seed)
        local = random_policy(recipient, recipient_mask, key)
        shared = _shared_random_policy(
            recipient, recipient_mask, key, bank, unavailable
        )
        _assert_tree_exact(shared, local)
        moves.append(int(shared.move))
    assert len(set(moves)) > 1


def test_shared_adapter_ignores_cross_team_inactive_and_unavailable_mutations() -> None:
    """Excluded source columns cannot perturb reactive action bytes."""
    config = _tdm_config(team_sizes=(2, 1))
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    bank = _team_bank(observation)
    recipient = cast(Observation, _scalar_actor(observation, 0))
    recipient_mask = cast(ActionMask, _scalar_actor(action_mask, 0))
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )[0, :5]
    key = jax.random.key(29)
    baseline = reactive_tdm_alpha_policy(
        recipient,
        recipient_mask,
        key,
        bank,
        availability,
    )

    changed = bank
    for excluded_source in (3, 4):
        changed = _bank_with_one_sighting(
            changed,
            sensor_source=excluded_source,
            candidate=MAX_AGENTS_PER_TEAM,
            candidate_features=jnp.full(
                (UNIT_FEATURES,),
                10_000.0 + excluded_source,
                dtype=jnp.float32,
            ),
        )
    changed = changed._replace(
        objective_features_by_source=jnp.full_like(
            changed.objective_features_by_source,
            99_999.0,
        )
    )
    actual = reactive_tdm_alpha_policy(
        recipient,
        recipient_mask,
        key,
        changed,
        availability,
    )
    _assert_tree_exact(actual, baseline)


def test_bank_ignores_non_sensor_base_fields_and_previous_action_history() -> None:
    """Maps, context, lifecycle metadata, and action history cannot enter the bank."""
    config = _tdm_config()
    _, observation, _, _ = reset(config, jax.random.key(0))
    baseline = build_shared_obs_sensor_source_bank(observation)

    def _fill_history(leaf: Array) -> Array:
        return jnp.full_like(leaf, 1.0)

    changed = observation._replace(
        map_obstacle_features=jnp.full_like(
            observation.map_obstacle_features,
            321.0,
        ),
        context_features=jnp.full_like(observation.context_features, 654.0),
        previous_timestep_actions=jax.tree.map(
            _fill_history,
            observation.previous_timestep_actions,
        ),
        spawn_lifecycle=jax.tree.map(
            _fill_history,
            observation.spawn_lifecycle,
        ),
    )
    _assert_tree_exact(build_shared_obs_sensor_source_bank(changed), baseline)


def test_source_bank_and_shared_scalar_adapter_match_eager_jit_and_vmap() -> None:
    """Structured composition preserves exact values under supported transforms."""
    config = _tdm_config()
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    eager_bank = build_shared_obs_sensor_source_bank(observation)
    compiled_bank = cast(
        SharedObsSensorSourceBankV2,
        jax.jit(build_shared_obs_sensor_source_bank)(observation),
    )
    _assert_tree_exact(compiled_bank, eager_bank)

    def _stack_twice(leaf: Array) -> Array:
        return jnp.stack((leaf, leaf))

    def _take_first(leaf: Array) -> Array:
        return leaf[0]

    def _take_second(leaf: Array) -> Array:
        return leaf[1]

    stacked_observation = jax.tree.map(_stack_twice, observation)
    mapped_banks = jax.vmap(build_shared_obs_sensor_source_bank)(stacked_observation)
    _assert_tree_exact(
        jax.tree.map(_take_first, mapped_banks),
        eager_bank,
    )
    _assert_tree_exact(
        jax.tree.map(_take_second, mapped_banks),
        eager_bank,
    )

    recipient = cast(Observation, _scalar_actor(observation, 0))
    recipient_mask = cast(ActionMask, _scalar_actor(action_mask, 0))
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )[0, :5]
    key = jax.random.key(37)
    actor_bank = jax.tree.map(itemgetter(0), eager_bank)
    eager_decision = reactive_tdm_alpha_policy(
        recipient,
        recipient_mask,
        key,
        actor_bank,
        availability,
    )
    compiled_decision = cast(
        object,
        jax.jit(reactive_tdm_alpha_policy)(
            recipient,
            recipient_mask,
            key,
            actor_bank,
            availability,
        ),
    )
    _assert_tree_exact(compiled_decision, eager_decision)
    closed_jaxpr = jax.make_jaxpr(reactive_tdm_alpha_policy)(
        recipient,
        recipient_mask,
        key,
        actor_bank,
        availability,
    )
    jaxpr_text = str(closed_jaxpr)
    assert closed_jaxpr.jaxpr.eqns
    assert "io_callback" not in jaxpr_text
    assert "pure_callback" not in jaxpr_text
    assert "f64[" not in jaxpr_text
    assert "i64[" not in jaxpr_text


def test_shared_adapter_cannot_bypass_the_recipient_exact_action_mask() -> None:
    """Arbitrary admitted source material cannot create an unsupported action."""
    config = _tdm_config()
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    bank = _team_bank(observation)
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )
    recipient = cast(Observation, _scalar_actor(observation, 0))
    original_mask = cast(ActionMask, _scalar_actor(action_mask, 0))
    stay_only = original_mask._replace(
        move_mask=jnp.arange(original_mask.move_mask.shape[0]) == MOVE_STAY,
        select_target_mask=jnp.arange(original_mask.select_target_mask.shape[0]) == 0,
        use_ultimate_mask=jnp.arange(original_mask.use_ultimate_mask.shape[0]) == 0,
        select_target_use_ultimate_joint_mask=(
            jnp.zeros_like(original_mask.select_target_use_ultimate_joint_mask)
            .at[0, 0]
            .set(True)
        ),
    )
    action = reactive_tdm_alpha_policy(
        recipient,
        stay_only,
        jax.random.key(23),
        bank,
        availability[0, :5],
    )
    assert int(action.move) == MOVE_STAY
    assert int(action.select_target) == 0
    assert int(action.use_ultimate) == 0


def _forbidden_bank_reader_policy(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    key: Array,
    source_bank: SharedObsSensorSourceBankV2,
    recipient_source_availability: Array,
) -> ActorAction:
    """Ignore availability and inspect the actor's forbidden shared self source."""
    del key, recipient_source_availability
    forbidden_source = recipient_observation.self_ally_index
    leaked = jnp.logical_or(
        jnp.any(
            source_bank.unit_features_by_source_and_candidate[forbidden_source] != 0.0
        ),
        jnp.logical_or(
            jnp.any(
                source_bank.unit_visibility_by_source_and_candidate[forbidden_source]
            ),
            jnp.any(source_bank.objective_features_by_source[forbidden_source] != 0.0),
        ),
    )
    requested = jnp.where(leaked, 3, MOVE_STAY).astype(jnp.int32)
    move = jnp.where(recipient_action_mask.move_mask[requested], requested, MOVE_STAY)
    return ActorAction(
        move=move.astype(jnp.int32),
        select_target=jnp.asarray(0, dtype=jnp.int32),
        use_ultimate=jnp.asarray(0, dtype=jnp.int32),
    )


def test_teammate_only_same_epoch_sighting_can_change_movement_intent() -> None:
    """Reactive TDM acts on an admitted current teammate sighting."""
    config = _tdm_config(team_sizes=(2, 1))
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    recipient = cast(Observation, _scalar_actor(observation, 0))._replace(
        enemy_unit_features=jnp.zeros(
            (MAX_AGENTS_PER_TEAM, UNIT_FEATURES), jnp.float32
        ),
        enemy_visibility_mask=jnp.zeros((MAX_AGENTS_PER_TEAM,), jnp.bool_),
    )
    recipient_mask = cast(ActionMask, _scalar_actor(action_mask, 0))
    candidate_features = (
        observation.self_features[MAX_AGENTS_PER_TEAM]
        .at[AGENT_FEATURE_X]
        .set(1.5)
        .at[AGENT_FEATURE_Y]
        .set(5.0)
    )
    bank = _bank_with_one_sighting(
        _team_bank(observation),
        sensor_source=1,
        candidate=MAX_AGENTS_PER_TEAM,
        candidate_features=candidate_features,
    )
    unavailable = jnp.zeros((MAX_AGENTS_PER_TEAM,), dtype=jnp.bool_)
    admitted = unavailable.at[1].set(True)

    without_source = reactive_tdm_alpha_policy(
        recipient,
        recipient_mask,
        jax.random.key(0),
        bank,
        unavailable,
    )
    with_source = reactive_tdm_alpha_policy(
        recipient,
        recipient_mask,
        jax.random.key(0),
        bank,
        admitted,
    )
    without_move = int(without_source.move)
    with_move = int(with_source.move)
    assert without_move != with_move


def test_shared_team_executor_uses_global_slot_keys_and_homogeneous_source_bank() -> (
    None
):
    """Both fixed team blocks receive aligned actor rows under JIT and vmap."""
    config = _tdm_config(team_sizes=(2, 1))
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    keys = jax.random.split(jax.random.key(31), MAX_AGENT_SLOTS)
    bank = build_shared_obs_sensor_source_bank(observation)
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )

    shared_a = cast(
        ActorAction,
        execute_shared_obs_team_policy(
            observation,
            action_mask,
            keys,
            bank,
            availability,
            _shared_random_policy,
            1,
        ),
    )
    shared_b = cast(
        ActorAction,
        execute_shared_obs_team_policy(
            observation,
            action_mask,
            keys,
            bank,
            availability,
            _shared_random_policy,
            2,
        ),
    )
    for team_offset, team_action in ((0, shared_a), (MAX_AGENTS_PER_TEAM, shared_b)):
        for local_slot in range(MAX_AGENTS_PER_TEAM):
            global_slot = team_offset + local_slot
            expected = random_policy(
                cast(Observation, _scalar_actor(observation, global_slot)),
                cast(ActionMask, _scalar_actor(action_mask, global_slot)),
                keys[global_slot],
            )
            _assert_tree_exact(_scalar_actor(team_action, local_slot), expected)

        def _take_team_rows(leaf: Array, offset: int = team_offset) -> Array:
            return leaf[offset : offset + MAX_AGENTS_PER_TEAM]

        team_mask = jax.tree.map(
            _take_team_rows,
            action_mask,
        )
        actor = jnp.arange(MAX_AGENTS_PER_TEAM)
        assert bool(jnp.all(team_mask.move_mask[actor, team_action.move]))
        assert bool(
            jnp.all(
                team_mask.select_target_use_ultimate_joint_mask[
                    actor,
                    team_action.select_target,
                    team_action.use_ultimate,
                ]
            )
        )

    no_shared = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            keys,
            random_policy,
            1,
        ),
    )
    zero_availability = jnp.zeros_like(availability)
    shared_without_extra = cast(
        ActorAction,
        execute_shared_obs_team_policy(
            observation,
            action_mask,
            keys,
            bank,
            zero_availability,
            _shared_random_policy,
            1,
        ),
    )
    _assert_tree_exact(shared_without_extra, no_shared)


def test_executor_masks_all_unavailable_bank_fields_before_arbitrary_policy() -> None:
    """The executor enforces availability even when a policy ignores its mask row."""
    config = _tdm_config(team_sizes=(2, 1))
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    keys = jax.random.split(jax.random.key(43), MAX_AGENT_SLOTS)
    bank = build_shared_obs_sensor_source_bank(observation)
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )
    poisoned = bank._replace(
        unit_features_by_source_and_candidate=(
            jnp.full_like(bank.unit_features_by_source_and_candidate, 77.0)
        ),
        unit_visibility_by_source_and_candidate=(
            jnp.ones_like(bank.unit_visibility_by_source_and_candidate)
        ),
        objective_features_by_source=(
            jnp.full_like(bank.objective_features_by_source, 99.0)
        ),
    )

    baseline = cast(
        ActorAction,
        execute_shared_obs_team_policy(
            observation,
            action_mask,
            keys,
            bank,
            availability,
            _forbidden_bank_reader_policy,
            1,
        ),
    )
    actual = cast(
        ActorAction,
        execute_shared_obs_team_policy(
            observation,
            action_mask,
            keys,
            poisoned,
            availability,
            _forbidden_bank_reader_policy,
            1,
        ),
    )
    assert bool(jnp.all(baseline.move == MOVE_STAY))
    _assert_tree_exact(actual, baseline)


def test_rollout_availability_and_host_reconstruction_preserve_exact_provenance() -> (
    None
):
    """Capture stores base rows and matrix, then reconstructs no second copy."""
    config = _tdm_config()
    state, observation, action_mask, _ = reset(config, jax.random.key(0))
    observation = historical_observation(config, observation)
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )
    context = evaluation_context(
        execution_information_mode="shared_obs",
        config=config,
        expected_horizon=1,
    ).model_copy(update={"actor_projection": SHARED_OBS_ACTOR_PROJECTION_V1})
    frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        observation,
        action_mask,
        availability,
    )
    reconstructed = reconstruct_shared_obs_sensor_source_bank_v1(context, frame)
    live = _historical_bank(observation)
    _assert_tree_exact(reconstructed, live)
    assert frame.shared_obs_information_availability_by_recipient_and_sensor_source == (
        tuple(tuple(bool(value) for value in row) for row in np.asarray(availability))
    )
    assert not hasattr(frame, "shared_obs_sensor_source_bank")


def test_host_reconstruction_uses_recorded_relation_mapping() -> None:
    """A valid recorded row permutation remains the reconstruction authority."""
    config = _tdm_config()
    state, observation, action_mask, _ = reset(config, jax.random.key(0))
    observation = historical_observation(config, observation)
    base_context = evaluation_context(
        execution_information_mode="shared_obs",
        config=config,
        expected_horizon=1,
    ).model_copy(update={"actor_projection": SHARED_OBS_ACTOR_PROJECTION_V1})
    catalog = base_context.static_mechanics_catalog
    ally_mapping = tuple(
        tuple(reversed(row))
        for row in catalog.global_slot_by_actor_and_ally_observation_row
    )
    enemy_mapping = tuple(
        tuple(reversed(row))
        for row in catalog.global_slot_by_actor_and_enemy_observation_row
    )
    target_mapping = tuple(
        (None, *ally_row, *enemy_row)
        for ally_row, enemy_row in zip(ally_mapping, enemy_mapping, strict=True)
    )
    catalog_payload = {
        **catalog.model_dump(mode="python"),
        "global_recipient_slot_by_actor_and_target_action": target_mapping,
        "global_slot_by_actor_and_ally_observation_row": ally_mapping,
        "global_slot_by_actor_and_enemy_observation_row": enemy_mapping,
    }
    catalog_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        catalog_payload,
        exclude={"canonical_digest_sha256"},
    )
    permuted_catalog = StaticMechanicsCatalogV1.model_validate(catalog_payload)
    context = EvaluationEpisodeContextV1.model_validate(
        {
            **base_context.model_dump(mode="python"),
            "static_mechanics_catalog": permuted_catalog.model_dump(mode="python"),
        }
    )
    reverse_rows = jnp.arange(MAX_AGENTS_PER_TEAM - 1, -1, -1)
    permuted_observation = observation._replace(
        ally_unit_features=observation.ally_unit_features[:, reverse_rows],
        enemy_unit_features=observation.enemy_unit_features[:, reverse_rows],
        ally_visibility_mask=observation.ally_visibility_mask[:, reverse_rows],
        enemy_visibility_mask=observation.enemy_visibility_mask[:, reverse_rows],
        spawn_lifecycle=observation.spawn_lifecycle._replace(
            class_ids_by_agent_by_team=(
                observation.spawn_lifecycle.class_ids_by_agent_by_team[
                    :, :, reverse_rows
                ]
            )
        ),
    )
    target_order = jnp.asarray(
        (0, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6),
        dtype=jnp.int32,
    )
    permuted_action_mask = action_mask._replace(
        select_target_mask=action_mask.select_target_mask[:, target_order],
        select_target_use_ultimate_joint_mask=(
            action_mask.select_target_use_ultimate_joint_mask[:, target_order]
        ),
    )
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )
    frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        permuted_observation,
        permuted_action_mask,
        availability,
    )

    reconstructed = reconstruct_shared_obs_sensor_source_bank_v1(context, frame)
    _assert_tree_exact(
        reconstructed,
        _historical_bank(observation),
    )


def test_shared_projection_identity_and_reconstruction_fail_closed() -> None:
    """Mode, projection, frame identity, and availability all fail closed."""
    assert (
        SHARED_OBS_ACTOR_PROJECTION_ID
        == "base-observation-plus-authorized-sensor-source-bank"
    )
    assert SHARED_OBS_ACTOR_PROJECTION_VERSION == 2
    assert (
        VersionedIdentityV1(
            identifier="base-observation-plus-authorized-sensor-source-bank",
            version=1,
        )
        == SHARED_OBS_ACTOR_PROJECTION_V1
    )

    no_shared_context = evaluation_context().model_copy(
        update={"actor_projection": SHARED_OBS_ACTOR_PROJECTION_V1}
    )
    with pytest.raises(ValueError, match="requires shared_obs"):
        reconstruct_shared_obs_sensor_source_bank_v1(
            no_shared_context,
            cast(object, None),  # type: ignore[arg-type]
        )

    config = _tdm_config()
    state, observation, action_mask, _ = reset(config, jax.random.key(0))
    observation = historical_observation(config, observation)
    context = evaluation_context(
        execution_information_mode="shared_obs",
        config=config,
        expected_horizon=1,
    ).model_copy(update={"actor_projection": SHARED_OBS_ACTOR_PROJECTION_V1})
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask,
        config.agent_profile.team_ids,
    )
    frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        observation,
        action_mask,
        availability,
    )

    for invalid_projection in (
        VersionedIdentityV1(identifier="wrong-projection", version=1),
        VersionedIdentityV1(
            identifier=SHARED_OBS_ACTOR_PROJECTION_ID,
            version=2,
        ),
    ):
        invalid_context = context.model_copy(
            update={"actor_projection": invalid_projection}
        )
        with pytest.raises(ValueError, match=r"requires.*version 1"):
            reconstruct_shared_obs_sensor_source_bank_v1(invalid_context, frame)

    with pytest.raises(TypeError, match="requires EvaluationFrameV1"):
        reconstruct_shared_obs_sensor_source_bank_v1(
            context,
            cast(object, None),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="episode IDs must match"):
        reconstruct_shared_obs_sensor_source_bank_v1(
            context,
            frame.model_copy(update={"episode_id": "different-episode"}),
        )
    without_availability = frame.model_copy(
        update={
            "shared_obs_information_availability_by_recipient_and_sensor_source": None
        }
    )
    with pytest.raises(ValueError, match="requires recorded information availability"):
        reconstruct_shared_obs_sensor_source_bank_v1(
            context,
            without_availability,
        )
