"""Prove faithful actor text, exact numbers, source privacy and frame consistency.

All inputs come from development map 0 or explicitly synthetic field changes.
Tests retain every permitted field, distinguish hidden/absent history from zero,
and use the existing delivery and reflection helpers without a model/tokenizer.
"""

# pyright: reportPrivateUsage=false

from collections.abc import Iterator
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_X,
    ActionMask,
)
from marl_battlegrounds.environment import Environment, EnvironmentState
from marl_battlegrounds.llm import format_actor_view
from marl_battlegrounds.llm.text import _number
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    build_team_actor_input,
    mirror_team_view,
    team_on_right,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


def _row[T](tree: T, index: int) -> T:
    def take(value: Array) -> Array:
        return value[index]

    return jax.tree.map(take, tree)


@pytest.fixture(scope="module")
def game() -> tuple[Environment, Observations, EnvironmentState]:
    config = make_standard_team_deathmatch_config(
        map_id=0,
        team_a_roster=("warrior", "mage", "hunter", "rogue", "priest"),
        team_b_roster=("warrior", "mage", "hunter", "rogue", "priest"),
        max_steps=20,
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(711))
    return env, observations, state


@pytest.fixture(scope="module")
def actor(
    game: tuple[Environment, Observations, EnvironmentState],
) -> tuple[ActorInput, ActionMask]:
    env, observations, state = game
    inputs = env.policy_inputs(observations, state)
    return _row(_row(inputs.actors, 0), 0), _row(_row(inputs.action_mask, 0), 0)


def test_float32_numbers_round_trip_bits_including_signed_zero() -> None:
    rng = np.random.default_rng(173)
    values = rng.integers(0, 2**32, 512, dtype=np.uint32).view(np.float32)
    edge = np.asarray(
        [
            0.0,
            -0.0,
            np.finfo(np.float32).max,
            np.finfo(np.float32).smallest_subnormal,
            9.292893,
        ],
        np.float32,
    )
    for value in np.concatenate((values[np.isfinite(values)], edge)):
        assert np.float32(_number(value)).tobytes() == value.tobytes()
    assert _number(np.float32(9.292893)) == "9.292893"
    assert _number(np.int32(2147483647)) == "2147483647"
    for value in (np.float32(np.nan), np.float32(np.inf), np.float64(1.5)):
        with pytest.raises(ValueError):
            _number(value)


def test_all_input_leaves_remain_detectable_without_mutation(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    host = jax.device_get(source)
    baseline = format_actor_view(host, masks, frame="world")
    leaves = jax.tree.leaves(host)
    before = [np.asarray(leaf).copy() for leaf in leaves]
    for index, leaf in enumerate(leaves):
        changed = np.asarray(leaf).copy()
        flat = changed.reshape(-1)
        if changed.dtype == np.bool_:
            flat[0] = not flat[0]
        elif changed.dtype.kind == "i":
            flat[0] += 1
        else:
            flat[0] = np.float32(123.125)
        inputs = list(leaves)
        inputs[index] = changed
        replacements = iter(inputs)

        def replace_leaf(
            value: Array, replacements: Iterator[Array] = replacements
        ) -> Array:
            del value
            return next(replacements)

        variant = jax.tree.map(replace_leaf, host)
        assert format_actor_view(variant, masks, frame="world") != baseline, index
    for original, actual in zip(before, jax.tree.leaves(host), strict=True):
        np.testing.assert_array_equal(original, actual)


def test_inactive_padding_and_reserved_nonzero_values_are_not_silently_lost(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    obs = source.observation
    changed = source._replace(
        observation=obs._replace(
            objective_features=obs.objective_features.at[7, 11].set(0.125),
            map_obstacle_features=obs.map_obstacle_features.at[31, 6].set(-0.0),
        )
    )
    text = format_actor_view(changed, masks)
    assert "Objectives: 7:" in text and "0.125]" in text
    assert "Obstacle 31:" in text and "theta=-0" in text


def test_absent_and_neutral_previous_actions_are_different(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    before = format_actor_view(source, masks)
    assert "ally_0: Not seen" in before
    old = source.observation.previous_timestep_actions
    observed = old._replace(
        ally_previous_timestep_move_actions_one_hot=old[0].at[0, 0].set(1),
        ally_previous_timestep_select_target_actions_one_hot=old[2].at[0, 0].set(1),
        ally_previous_timestep_use_ultimate_actions_one_hot=old[4].at[0, 0].set(1),
    )
    after = format_actor_view(
        source._replace(
            observation=source.observation._replace(previous_timestep_actions=observed)
        ),
        masks,
    )
    assert "ally_0: move=stay target=target_none ultimate=No" in after
    assert before != after


def test_previous_enemy_targets_keep_observer_names_when_target_is_hidden(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    old = source.observation.previous_timestep_actions
    observed = old._replace(
        enemy_previous_timestep_move_actions_one_hot=old[1].at[0, 0].set(1),
        enemy_previous_timestep_select_target_actions_one_hot=old[3].at[0, 8].set(1),
        enemy_previous_timestep_use_ultimate_actions_one_hot=old[5].at[0, 1].set(1),
    )
    obs = source.observation._replace(
        previous_timestep_actions=observed,
        enemy_visibility_mask=source.observation.enemy_visibility_mask.at[2].set(False),
    )
    text = format_actor_view(source._replace(observation=obs), masks)
    assert "enemy_0: move=stay target=enemy_2 ultimate=Yes" in text
    assert "enemy_2: Hidden" in text


def test_source_permissions_and_visibility_are_separate(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor

    def clear(value: Array) -> Array:
        return jnp.zeros_like(value)

    zeros = jax.tree.map(clear, source.source_bank)
    no_sources = source._replace(
        source_bank=zeros, source_availability=jnp.zeros(5, bool)
    )
    permission = no_sources._replace(
        source_availability=jnp.asarray([False, True, False, False, False])
    )
    a, b = format_actor_view(no_sources, masks), format_actor_view(permission, masks)
    assert "Source ally_1: Unavailable; No visible rows" in a
    assert "Source ally_1: Allowed; No visible rows" in b
    visible_zero = permission._replace(
        source_bank=zeros._replace(
            unit_visibility_by_source_and_candidate=zeros.unit_visibility_by_source_and_candidate.at[
                1, 7
            ].set(True)
        )
    )
    assert "Source ally_1: Allowed; enemy_2=Visible Zero" in format_actor_view(
        visible_zero, masks
    )


def test_forbidden_source_material_cannot_change_delivered_text(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    restricted = observations._replace(
        source_availability=jnp.zeros_like(observations.source_availability)
    )
    original = env.policy_inputs(restricted, state)

    def poison(value: Array) -> Array:
        selector = (jnp.arange(10) != 0).reshape((10,) + (1,) * (value.ndim - 1))
        return jnp.where(selector, jnp.full_like(value, 73), value)

    poisoned = restricted._replace(
        observation=jax.tree.map(poison, restricted.observation)
    )
    changed = env.policy_inputs(poisoned, state)
    masks = _row(_row(original.action_mask, 0), 0)
    assert format_actor_view(
        _row(_row(original.actors, 0), 0), masks
    ) == format_actor_view(_row(_row(changed.actors, 0), 0), masks)


def test_team_labels_on_same_physical_side_do_not_change_text(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    permutation = jnp.concatenate((jnp.arange(5, 10), jnp.arange(5)))

    def reorder(value: Array) -> Array:
        return value[permutation]

    swapped = Observations(
        jax.tree.map(reorder, observations.observation),
        observations.source_availability[permutation][:, permutation],
    )
    own = _row(build_team_actor_input(observations, 0), 0)
    relabelled = _row(build_team_actor_input(swapped, 1), 0)
    masks = _row(_row(env.policy_inputs(observations, state).action_mask, 0), 0)
    assert format_actor_view(own, masks, frame="world") == format_actor_view(
        relabelled, masks, frame="world"
    )


def test_frame_label_does_not_transform_coordinates(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    world = format_actor_view(source, masks, frame="world")
    left = format_actor_view(source, masks, frame="left")
    assert world.splitlines()[1:] == left.splitlines()[1:]
    with pytest.raises(ValueError, match="Frame"):
        format_actor_view(source, masks, frame=cast(Literal["left", "world"], "other"))


def test_existing_mirror_handles_asymmetric_map_before_formatting(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    obstacle = (
        jnp.zeros((32, 8), jnp.float32)
        .at[0]
        .set(jnp.asarray([2, 4, 3, 0, 2, 1, 0.25, 1], jnp.float32))
    )
    source = source._replace(
        observation=source.observation._replace(map_obstacle_features=obstacle)
    )
    mirrored, mask = mirror_team_view(source, masks, jnp.asarray(True))
    text = format_actor_view(mirrored, mask)
    assert "Obstacle 0:" in text and "theta=-0.25" in text
    width = source.observation.context_features[2]
    assert f"x={_number(np.float32(width - 4))}" in text
    assert float(mirrored.observation.self_features[AGENT_FEATURE_X]) != float(
        source.observation.self_features[AGENT_FEATURE_X]
    )
    assert bool(team_on_right(source)) == bool(
        team_on_right(
            source._replace(
                observation=source.observation._replace(
                    self_features=source.observation.self_features.at[0].set(100)
                )
            )
        )
    )


def test_rejects_batched_or_nonfinite_inputs(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor

    def batch(value: Array) -> Array:
        return value[None]

    with pytest.raises(ValueError, match="shape"):
        format_actor_view(jax.tree.map(batch, source), masks)
    bad = source._replace(
        observation=source.observation._replace(
            self_features=source.observation.self_features.at[0].set(jnp.nan)
        )
    )
    with pytest.raises(ValueError, match="finite"):
        format_actor_view(bad, masks)


def test_every_unit_context_and_obstacle_column_is_preserved(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = jax.device_get(actor)
    baseline = format_actor_view(source, masks)
    for field, width in (
        ("self_features", 58),
        ("context_features", 20),
        ("map_obstacle_features", 8),
    ):
        for column in range(width):
            values = np.asarray(getattr(source.observation, field)).copy()
            if values.ndim == 2:
                values[0, column] = np.float32(123.125)
            else:
                values[column] = np.float32(123.125)
            changed = source._replace(
                observation=source.observation._replace(**{field: cast(Array, values)})
            )
            text = format_actor_view(changed, masks)
            assert text != baseline and "123.125" in text, (field, column)


def test_identical_rows_share_a_definition_without_losing_sources(
    actor: tuple[ActorInput, ActionMask],
) -> None:
    source, masks = actor
    own = source.observation.self_features
    observation = source.observation._replace(
        ally_unit_features=jnp.broadcast_to(own, (5, 58)),
        enemy_unit_features=jnp.zeros((5, 58), jnp.float32),
        ally_visibility_mask=jnp.ones(5, bool),
        enemy_visibility_mask=jnp.zeros(5, bool),
    )
    rows = jnp.zeros((5, 10, 58), jnp.float32).at[:, 0].set(own)
    bank = source.source_bank._replace(
        unit_features_by_source_and_candidate=rows,
        unit_visibility_by_source_and_candidate=jnp.zeros((5, 10), bool)
        .at[:, 0]
        .set(True),
    )
    view = source._replace(
        observation=observation, source_bank=bank, source_availability=jnp.ones(5, bool)
    )
    text = format_actor_view(view, masks)
    definitions = [line for line in text.splitlines() if line.startswith("unit_")]
    assert len(definitions) == 1
    for index in range(5):
        assert f"Source ally_{index}: Allowed; ally_0=Visible unit_0" in text
    different = view._replace(
        source_bank=bank._replace(
            unit_features_by_source_and_candidate=rows.at[2, 0, 0].set(
                jnp.nextafter(own[0], jnp.inf)
            )
        )
    )
    changed = format_actor_view(different, masks)
    assert len([line for line in changed.splitlines() if line.startswith("unit_")]) == 2
    assert "Source ally_2: Allowed; ally_0=Visible unit_1" in changed
