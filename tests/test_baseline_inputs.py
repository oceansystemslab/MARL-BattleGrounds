"""Check the fixed baseline actor and privileged training-state encodings.

These tests cover every input field, named offsets, categorical absence, source
permissions, public lifecycle facts, accepted-action history and float32 limits.
They compare scalar, batched and compiled calls without broadening actor rights
or treating encoding checks as learning or GPU-speed evidence.
"""

from collections.abc import Iterator
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_OFFSETS,
    ACTOR_FEATURE_SIZE,
    ACTOR_INPUT_SCHEMA_VERSION,
    TRAINING_STATE_FEATURE_OFFSETS,
    TRAINING_STATE_FEATURE_SIZE,
    TRAINING_STATE_SCHEMA_VERSION,
    encode_actor_inputs,
    encode_training_state,
)
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_X,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_TYPE,
    EnvConfig,
    EnvState,
    Observation,
)
from marl_battlegrounds.policies.input import (
    ActorInput,
    build_actor_input,
    build_observations,
    build_team_actor_input,
)


def _row[T](value: T, index: int) -> T:
    def take(leaf: Array) -> Array:
        return leaf[index]

    return jax.tree.map(take, value)


def _batch[T](value: T, count: int) -> T:
    def expand(leaf: ArrayLike) -> Array:
        array = jnp.asarray(leaf)
        return jnp.broadcast_to(array, (count, *array.shape))

    return jax.tree.map(expand, value)


def _leaves(value: object, prefix: str = "") -> Iterator[tuple[str, Array]]:
    fields = cast(tuple[str, ...] | None, getattr(value, "_fields", None))
    if fields is not None:
        for name in fields:
            yield from _leaves(
                getattr(value, name), f"{prefix}.{name}" if prefix else name
            )
    else:
        yield prefix, jnp.asarray(cast(ArrayLike, value))


@pytest.fixture(scope="module")
def snapshot() -> tuple[EnvConfig, EnvState, Observation, ActorInput]:
    config = evaluation_env_config(team_sizes=(3, 2))
    state, observation, _, _ = core.reset(config, jax.random.key(1))
    return config, state, observation, _row(build_actor_input(observation, config), 0)


def _distinct_actor(actor: ActorInput) -> ActorInput:
    count = 0

    def fill(value: Array) -> Array:
        nonlocal count
        count += 1
        if value.dtype == jnp.bool_:
            return jnp.ones_like(value)
        return (
            jnp.arange(value.size, dtype=value.dtype).reshape(value.shape) + count * 7
        )

    def units(value: Array) -> Array:
        classes = (jnp.arange(value[..., 0].size).reshape(value.shape[:-1]) % 5) + 1
        return (
            value.at[..., AGENT_FEATURE_CLASS_ID]
            .set(classes)
            .at[..., AGENT_FEATURE_ACTIVE]
            .set(1)
        )

    actor = jax.tree.map(fill, actor)
    observation = actor.observation
    obstacles = observation.map_obstacle_features.at[:, OBSTACLE_FEATURE_TYPE].set(
        jnp.arange(32) % 2 + 1
    )
    history = type(observation.previous_timestep_actions)(
        *(
            jax.nn.one_hot(jnp.arange(5) % field.shape[-1], field.shape[-1])
            for field in observation.previous_timestep_actions
        )
    )
    lifecycle = observation.spawn_lifecycle._replace(
        class_ids_by_agent_by_team=(jnp.arange(10, dtype=jnp.int32).reshape(2, 5) % 5)
        + 1
    )
    return actor._replace(
        observation=observation._replace(
            self_features=units(observation.self_features),
            ally_unit_features=units(observation.ally_unit_features),
            enemy_unit_features=units(observation.enemy_unit_features),
            map_obstacle_features=obstacles.at[:, OBSTACLE_FEATURE_ACTIVE].set(1),
            previous_timestep_actions=history,
            spawn_lifecycle=lifecycle,
            self_ally_index=jnp.int32(3),
        ),
        source_bank=actor.source_bank._replace(
            unit_features_by_source_and_candidate=units(
                actor.source_bank.unit_features_by_source_and_candidate
            )
        ),
    )


def _class_rows(value: Array, present: Array) -> np.ndarray:
    rows = np.asarray(value)
    included = np.asarray(present) & (rows[..., AGENT_FEATURE_ACTIVE] > 0)
    result = np.zeros((*rows.shape[:-1], 63), dtype=np.float32)
    result[..., :6] = rows[..., :6]
    result[..., 6:12] = np.eye(6)[rows[..., 6].astype(int)]
    result[..., 12:] = rows[..., 7:]
    result[~included] = 0
    return result


def test_actor_schema_covers_every_field_with_distinct_values(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    actor = _distinct_actor(snapshot[3])
    actual = np.asarray(encode_actor_inputs(actor))
    expected: list[np.ndarray] = []
    widths = (
        63,
        315,
        315,
        320,
        96,
        19,
        5,
        5,
        45,
        45,
        55,
        55,
        10,
        10,
        20,
        10,
        1,
        1,
        2,
        2,
        10,
        10,
        60,
        5,
        3150,
        50,
        480,
        5,
    )
    fields = tuple(_leaves(actor))
    assert tuple(ACTOR_FEATURE_OFFSETS) == tuple(name for name, _ in fields)
    offset = 0
    for (name, value), width in zip(fields, widths, strict=True):
        assert ACTOR_FEATURE_OFFSETS[name] == slice(offset, offset + width)
        if name.endswith(("self_features", "unit_features", "and_candidate")) and (
            value.shape[-1] == 58
        ):
            encoded = _class_rows(value, jnp.ones(value.shape[:-1], dtype=jnp.bool_))
        elif name.endswith("map_obstacle_features"):
            rows = np.asarray(value)
            encoded = np.concatenate(
                (np.eye(3)[rows[:, 0].astype(int)], rows[:, 1:]), -1
            )
        elif name.endswith("class_ids_by_agent_by_team"):
            encoded = np.eye(6)[np.asarray(value)]
        elif name.endswith("self_ally_index"):
            encoded = np.eye(5)[np.asarray(value)]
        else:
            encoded = np.asarray(value)
        expected.append(encoded.reshape(-1))
        offset += width
    assert ACTOR_INPUT_SCHEMA_VERSION == 1
    assert offset == ACTOR_FEATURE_SIZE == 5164
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.concatenate(expected).astype(np.float32))
    with pytest.raises(TypeError):
        cast(dict[str, slice], ACTOR_FEATURE_OFFSETS)["new"] = slice(0, 1)


def test_actor_redaction_keeps_missing_categories_absent_and_public_rosters_visible(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    config, state, _, _ = snapshot
    state = state._replace(
        alive_mask=state.alive_mask.at[0].set(False),
        current_health=state.current_health.at[0].set(0.0),
    )
    _, observation, _, _ = core.initialize_scenario_state(state, config)
    actors = build_actor_input(observation, config)
    actual = encode_actor_inputs(actors)
    class_slice = ACTOR_FEATURE_OFFSETS[
        "observation.spawn_lifecycle.class_ids_by_agent_by_team"
    ]
    roster = np.asarray(actual[0, class_slice]).reshape(2, 5, 6)
    np.testing.assert_array_equal(
        roster[0, 0], np.eye(6)[config.agent_profile.class_ids[0]]
    )
    assert (
        np.asarray(actual[0, ACTOR_FEATURE_OFFSETS["observation.self_features"]])[4]
        == 1
    )
    assert (
        np.asarray(actual[0, ACTOR_FEATURE_OFFSETS["observation.self_features"]])[5]
        == 0
    )
    assert not np.any(roster[0, 3:])
    unit_slice = ACTOR_FEATURE_OFFSETS["observation.enemy_unit_features"]
    enemy = np.asarray(actual[0, unit_slice]).reshape(5, 63)
    assert not np.any(enemy[~np.asarray(observation.enemy_visibility_mask[0])])
    index_slice = ACTOR_FEATURE_OFFSETS["observation.self_ally_index"]
    np.testing.assert_array_equal(actual[0, index_slice], (1, 0, 0, 0, 0))
    assert not np.any(actual[3, index_slice])
    assert not np.any(actual[3, class_slice])
    obstacle_slice = ACTOR_FEATURE_OFFSETS["observation.map_obstacle_features"]
    np.testing.assert_array_equal(actual[3, obstacle_slice], actual[0, obstacle_slice])
    unused_obstacles = np.asarray(config.obstacles[:, OBSTACLE_FEATURE_ACTIVE]) == 0
    assert not np.any(
        np.asarray(actual[0, obstacle_slice]).reshape(32, 10)[unused_obstacles]
    )


def test_forbidden_sources_cannot_change_actor_features_and_allowed_sources_can(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    config, _, observation, _ = snapshot
    compact = build_observations(observation, config)
    restricted = compact._replace(source_availability=jnp.zeros((10, 10), jnp.bool_))
    baseline = encode_actor_inputs(_row(build_team_actor_input(restricted, 0), 0))
    changed = observation._replace(
        objective_features=observation.objective_features.at[1].set(73),
        ally_unit_features=observation.ally_unit_features.at[1, :, AGENT_FEATURE_X].add(
            41
        ),
        enemy_unit_features=observation.enemy_unit_features.at[
            1, :, AGENT_FEATURE_X
        ].add(42),
    )
    changed_inputs = _row(
        build_team_actor_input(restricted._replace(observation=changed), 0), 0
    )
    np.testing.assert_array_equal(encode_actor_inputs(changed_inputs), baseline)
    allowed = restricted.source_availability.at[0, 1].set(True)
    before = _row(
        build_team_actor_input(restricted._replace(source_availability=allowed), 0), 0
    )
    after = _row(
        build_team_actor_input(
            restricted._replace(source_availability=allowed, observation=changed), 0
        ),
        0,
    )
    objective_slice = ACTOR_FEATURE_OFFSETS["source_bank.objective_features_by_source"]
    assert not np.array_equal(
        encode_actor_inputs(before)[objective_slice],
        encode_actor_inputs(after)[objective_slice],
    )
    assert not np.any(
        np.asarray(encode_actor_inputs(after)[objective_slice]).reshape(5, 8, 12)[
            [0, 2, 3, 4]
        ]
    )


def test_source_rows_and_visibility_are_separate_and_forbidden_bytes_are_cleared(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    actor = _distinct_actor(snapshot[3])
    visibility = actor.source_bank.unit_visibility_by_source_and_candidate.at[1, 3].set(
        False
    )
    actor = actor._replace(
        source_availability=actor.source_availability.at[2].set(False),
        source_bank=actor.source_bank._replace(
            unit_visibility_by_source_and_candidate=visibility
        ),
    )
    actual = encode_actor_inputs(actor)
    units = np.asarray(
        actual[
            ACTOR_FEATURE_OFFSETS["source_bank.unit_features_by_source_and_candidate"]
        ]
    ).reshape(5, 10, 63)
    visible = np.asarray(
        actual[
            ACTOR_FEATURE_OFFSETS["source_bank.unit_visibility_by_source_and_candidate"]
        ]
    ).reshape(5, 10)
    assert not np.any(units[1, 3])
    assert not np.any(units[2])
    assert not np.any(visible[2])
    assert visible[0, 3] and not visible[1, 3]
    assert np.any(units[0, 3])
    np.testing.assert_array_equal(actual[-5:], (1, 1, 0, 1, 1))


def test_actor_history_distinguishes_reset_absence_from_real_neutral_action(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    actor = snapshot[3]
    absent = encode_actor_inputs(actor)
    history = type(actor.observation.previous_timestep_actions)(
        *(
            field.at[0, 0].set(1)
            for field in actor.observation.previous_timestep_actions
        )
    )
    present = encode_actor_inputs(
        actor._replace(
            observation=actor.observation._replace(previous_timestep_actions=history)
        )
    )
    for name, region in ACTOR_FEATURE_OFFSETS.items():
        if "previous_timestep_actions" in name:
            assert not np.any(absent[region])
            assert present[region][0] == 1
            assert np.sum(present[region]) == 1


def test_training_schema_covers_every_state_and_config_field(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    config, state, _, _ = snapshot
    state = state._replace(
        step_count=jnp.int32(17),
        has_previous_timestep_joint_action=jnp.bool_(True),
        previous_timestep_move_actions=jnp.arange(10, dtype=jnp.int32) % 9,
        previous_timestep_select_target_actions=jnp.arange(10, dtype=jnp.int32),
        previous_timestep_use_ultimate_actions=jnp.arange(10, dtype=jnp.int32) % 2,
    )
    count = 0

    def distinct_state(value: Array) -> Array:
        nonlocal count
        count += 1
        return value + count if value.dtype != jnp.bool_ else value

    physical = jax.tree.map(distinct_state, state)
    state = physical._replace(
        previous_timestep_move_actions=state.previous_timestep_move_actions,
        previous_timestep_select_target_actions=state.previous_timestep_select_target_actions,
        previous_timestep_use_ultimate_actions=state.previous_timestep_use_ultimate_actions,
    )
    actual = np.asarray(encode_training_state(state, config))
    fields = (*_leaves(state, "state"), *_leaves(config, "config"))
    assert tuple(TRAINING_STATE_FEATURE_OFFSETS) == tuple(name for name, _ in fields)
    expected: list[np.ndarray] = []
    active = np.asarray(config.agent_profile.active_mask)
    offset = 0
    for name, value in fields:
        row = np.asarray(value)
        if row.shape and row.shape[0] == 10:
            row = row.copy()
            row[~active] = 0
        if name == "config.task_mode":
            row = np.eye(4)[row]
        elif name == "config.obstacles":
            row = np.concatenate((np.eye(3)[row[:, 0].astype(int)], row[:, 1:]), -1)
            row[np.asarray(config.obstacles[:, 7]) == 0] = 0
        elif name in (
            "config.agent_profile.class_ids",
            "config.agent_profile.team_ids",
        ):
            row = np.eye(6 if name.endswith("class_ids") else 3)[row]
            row[~active] = 0
        elif name.startswith("state.previous_timestep_"):
            categories = 9 if "move_" in name else 11 if "select_target_" in name else 2
            row = np.eye(categories)[row]
            row[~active] = 0
        flat = row.reshape(-1)
        assert TRAINING_STATE_FEATURE_OFFSETS[name] == slice(offset, offset + flat.size)
        expected.append(flat)
        offset += flat.size
    assert TRAINING_STATE_SCHEMA_VERSION == 1
    assert offset == TRAINING_STATE_FEATURE_SIZE == 919
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.concatenate(expected).astype(np.float32))


def test_training_history_preserves_dead_members_and_hides_unused_slots(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    config, state, _, actor = snapshot
    initial_actor = encode_actor_inputs(actor)
    absent = encode_training_state(state, config)
    altered = state._replace(
        alive_mask=state.alive_mask.at[0].set(False),
        current_health=state.current_health.at[0].set(0).at[3].set(999),
        agent_positions=state.agent_positions.at[0].add(2).at[3].set(999),
        has_previous_timestep_joint_action=jnp.bool_(True),
    )
    present = encode_training_state(altered, config)
    for field, count in (("move", 9), ("select_target", 11), ("use_ultimate", 2)):
        region = TRAINING_STATE_FEATURE_OFFSETS[
            f"state.previous_timestep_{field}_actions"
        ]
        assert not np.any(absent[region])
        rows = np.asarray(present[region]).reshape(10, count)
        np.testing.assert_array_equal(rows[0], np.eye(count)[0])
        assert not np.any(rows[3:5])
    positions = np.asarray(
        present[TRAINING_STATE_FEATURE_OFFSETS["state.agent_positions"]]
    ).reshape(10, 2)
    assert not np.any(positions[3])
    assert not np.array_equal(absent, present)
    np.testing.assert_array_equal(encode_actor_inputs(actor), initial_actor)
    pads = TRAINING_STATE_FEATURE_OFFSETS["config.team_spawn_pad_positions"]
    np.testing.assert_array_equal(
        present[pads], config.team_spawn_pad_positions.reshape(-1)
    )


def test_float32_rounding_is_explicit_for_large_lifecycle_counts(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    config, state, _, actor = snapshot
    values = (16_777_216, 16_777_217, 16_777_218, 2_147_483_647)
    for value in values:
        lifecycle = actor.observation.spawn_lifecycle._replace(
            spawn_shield_configured_duration_by_agent=jnp.int32(value)
        )
        encoded_actor = encode_actor_inputs(
            actor._replace(
                observation=actor.observation._replace(spawn_lifecycle=lifecycle)
            )
        )
        region = ACTOR_FEATURE_OFFSETS[
            "observation.spawn_lifecycle.spawn_shield_configured_duration_by_agent"
        ]
        np.testing.assert_array_equal(
            encoded_actor[region], np.asarray([value], np.float32)
        )
        encoded_state = encode_training_state(
            state, config._replace(spawn_shield_duration_steps=value)
        )
        region = TRAINING_STATE_FEATURE_OFFSETS["config.spawn_shield_duration_steps"]
        np.testing.assert_array_equal(
            encoded_state[region], np.asarray([value], np.float32)
        )


def test_actor_arbitrary_leading_axes_match_mapped_calls_and_reuse_compilation(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    actor = _distinct_actor(snapshot[3])
    inputs = _batch(_batch(_batch(actor, 5), 3), 2)
    traces: list[int] = []

    @jax.jit
    def encode(value: ActorInput) -> Array:
        traces.append(1)
        return encode_actor_inputs(value)

    first = cast(Array, encode(inputs))
    changed = inputs._replace(
        observation=inputs.observation._replace(
            context_features=inputs.observation.context_features.at[1, 2, 4, 0].add(1)
        )
    )
    second = cast(Array, encode(changed))
    np.testing.assert_array_equal(
        first, jax.vmap(jax.vmap(jax.vmap(encode_actor_inputs)))(inputs)
    )
    assert first.shape == (2, 3, 5, 5164)
    assert traces == [1]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1, 2, :4], second[1, 2, :4])
    assert not np.array_equal(first[1, 2, 4], second[1, 2, 4])
    empty = _batch(actor, 0)
    assert encode_actor_inputs(empty).shape == (0, 5164)


def test_training_config_broadcast_matches_mapped_calls_and_reuses_compilation(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput],
) -> None:
    config, state, _, _ = snapshot
    states = _batch(_batch(state, 3), 2)
    configs = _batch(config, 3)._replace(
        map_width=jnp.asarray((30, 31, 32), jnp.float32)
    )
    traces: list[int] = []

    @jax.jit
    def encode(value: EnvState, settings: EnvConfig) -> Array:
        traces.append(1)
        return encode_training_state(value, settings)

    first = cast(Array, encode(states, configs))
    expected = jax.vmap(jax.vmap(encode_training_state), in_axes=(0, None))(
        states, configs
    )
    np.testing.assert_array_equal(first, expected)
    second = cast(
        Array,
        encode(
            states._replace(step_count=states.step_count + 1),
            configs._replace(map_width=configs.map_width + 2),
        ),
    )
    assert traces == [1]
    assert first.shape == (2, 3, 919)
    assert not np.array_equal(first, second)
    constant = encode_training_state(states, config)
    np.testing.assert_array_equal(
        constant, np.broadcast_to(encode_training_state(state, config), (2, 3, 919))
    )


@pytest.mark.parametrize("field", ("source", "self", "index", "visibility"))
def test_actor_shape_mismatches_fail_before_broadcasting_private_rows(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput], field: str
) -> None:
    actor = snapshot[3]
    if field == "source":
        actor = actor._replace(source_availability=actor.source_availability[None])
    elif field == "self":
        actor = actor._replace(
            observation=actor.observation._replace(
                self_features=actor.observation.self_features[:-1]
            )
        )
    elif field == "index":
        actor = actor._replace(
            observation=actor.observation._replace(
                self_ally_index=jnp.zeros((1,), jnp.int32)
            )
        )
    else:
        actor = actor._replace(
            source_bank=actor.source_bank._replace(
                unit_visibility_by_source_and_candidate=jnp.ones((1, 10), jnp.bool_)
            )
        )
    with pytest.raises(ValueError, match="must have shape"):
        jax.jit(encode_actor_inputs)(actor)


@pytest.mark.parametrize("field", ("state", "config_suffix", "config_prefix"))
def test_training_shapes_require_fixed_suffixes_and_broadcastable_config(
    snapshot: tuple[EnvConfig, EnvState, Observation, ActorInput], field: str
) -> None:
    config, state, _, _ = snapshot
    if field == "state":
        state = state._replace(alive_mask=jnp.zeros((11,), jnp.bool_))
    elif field == "config_suffix":
        config = config._replace(obstacles=jnp.zeros((32, 7), jnp.float32))
    else:
        state = _batch(state, 3)
        config = _batch(config, 2)
    with pytest.raises(ValueError, match="must have shape"):
        jax.jit(encode_training_state)(state, config)
