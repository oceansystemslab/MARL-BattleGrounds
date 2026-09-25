"""Check stable Core action names, strict replies and exact joint-mask legality.

These CPU-only tests require no provider or tokenizer. They check every native
category, reject malformed/custom actions before integer narrowing, and keep
actor masks unchanged. Public transition proofs are added beside these checks.
"""

import json
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.axis_mappings import (
    MOVEMENT_ACTION_NAME_BY_ID,
    TARGET_ACTION_NAME_BY_ID,
)
from marl_battlegrounds.core.env import initialize_scenario_state
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.llm import (
    IllegalActionError,
    ReplyFormatError,
    format_actor_view,
    legal_action_names,
    parse_action_reply,
    validate_actor_action,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import (
    mirror_move,
    mirror_team_view,
    team_on_right,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


def _masks(joint: np.ndarray | None = None) -> ActionMask:
    pairs = np.ones((11, 2), bool) if joint is None else joint
    return ActionMask(
        *(cast(Array, a) for a in (np.ones(9, bool), pairs.any(1), pairs.any(0), pairs))
    )


def _action(move: object = 0, target: object = 0, ultimate: object = 0) -> ActorAction:
    return ActorAction(*(cast(Array, np.asarray(x)) for x in (move, target, ultimate)))


def test_names_match_core_and_every_category_decodes() -> None:
    masks = _masks()
    menu = legal_action_names(masks)
    assert menu["move"] == tuple(name.lower() for name in MOVEMENT_ACTION_NAME_BY_ID)
    expected_combat: list[str] = []
    for target, name in enumerate(TARGET_ACTION_NAME_BY_ID):
        for ultimate in range(2):
            expected = (
                ("no_combat", "ultimate")[ultimate]
                if target == 0
                else name.lower().replace(" ", "_") + ("_basic", "_ultimate")[ultimate]
            )
            expected_combat.append(expected)
            for move, move_name in enumerate(menu["move"]):
                actual = parse_action_reply(
                    json.dumps({"move": move_name, "combat": expected}), masks
                )
                assert tuple(int(x) for x in actual) == (move, target, ultimate)
                assert all(x.shape == () and x.dtype == np.int32 for x in actual)
    assert menu["combat"] == tuple(expected_combat)
    assert (
        int(
            parse_action_reply(
                '{"move":"north","combat":"enemy_2_ultimate"}', masks
            ).select_target
        )
        == 8
    )


@pytest.mark.parametrize(
    "reply",
    [
        "{}",
        "[]",
        "null",
        '{"move":"north"}',
        '{"move":"north","combat":"no_combat","extra":0}',
        '{"move":"north","move":"south","combat":"no_combat"}',
        '{"move":1,"combat":"no_combat"}',
        '{"move":"North","combat":"no_combat"}',
        '{"move":"north","combat":"enemy_5_basic"}',
        '```json\n{"move":"north","combat":"no_combat"}\n```',
        '{"move":"north","combat":"no_combat"} extra',
        '{"move":"north","combat":null}',
        '{"move":"north","combat":NaN}',
    ],
)
def test_malformed_reply_is_not_repaired(reply: str) -> None:
    with pytest.raises(ReplyFormatError):
        parse_action_reply(reply, _masks())


def test_joint_pair_is_checked_even_when_both_margins_are_true() -> None:
    pairs = np.zeros((11, 2), bool)
    pairs[0, 0] = True
    pairs[8, 1] = True
    masks = _masks(pairs)
    before = [x.copy() for x in masks]
    assert bool(masks.select_target_mask[8]) and bool(masks.use_ultimate_mask[0])
    with pytest.raises(IllegalActionError, match="Combat"):
        parse_action_reply('{"move":"stay","combat":"enemy_2_basic"}', masks)
    with pytest.raises(IllegalActionError):
        validate_actor_action(_action(target=8), masks)
    assert legal_action_names(masks)["combat"] == ("no_combat", "enemy_2_ultimate")
    for actual, original in zip(masks, before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("bad", [True, 1.0, -1, 9, 2**40, [0]])
def test_custom_action_cannot_coerce_invalid_move(bad: object) -> None:
    with pytest.raises(IllegalActionError):
        validate_actor_action(_action(move=bad), _masks())


@pytest.mark.parametrize("target,ultimate", [(11, 0), (0, 2), (-1, 0), (0, True)])
def test_custom_action_checks_other_heads(target: object, ultimate: object) -> None:
    with pytest.raises(IllegalActionError):
        validate_actor_action(_action(target=target, ultimate=ultimate), _masks())


def test_masked_move_and_singleton_menu() -> None:
    pairs = np.zeros((11, 2), bool)
    pairs[0, 0] = True
    masks = _masks(pairs)._replace(move_mask=cast(Array, np.arange(9) == 0))
    assert legal_action_names(masks) == {"move": ("stay",), "combat": ("no_combat",)}
    with pytest.raises(IllegalActionError, match="Movement"):
        parse_action_reply('{"move":"north","combat":"no_combat"}', masks)
    assert tuple(
        int(x)
        for x in parse_action_reply(' {"combat":"no_combat", "move":"stay"}\n', masks)
    ) == (0, 0, 0)


@pytest.mark.parametrize("bad", [np.ones(9, np.int32), np.ones((1, 9), bool)])
def test_invalid_masks_fail_clearly(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="move_mask"):
        legal_action_names(_masks()._replace(move_mask=cast(Array, bad)))


def _row[T](tree: T, index: int) -> T:
    def take(value: Array) -> Array:
        return value[index]

    return jax.tree.map(take, tree)


@pytest.mark.parametrize("right", (False, True))
def test_named_charge_and_precommitted_move_match_native_public_trajectory(
    right: bool,
) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=0,
        team_a_roster=("warrior",),
        team_b_roster=("mage",),
        max_steps=20,
    )
    pads = config.team_spawn_pad_positions
    pads = pads.at[0, 0].set(jnp.asarray([2, 2], jnp.float32))
    pads = pads.at[1, 0].set(jnp.asarray([5, 2], jnp.float32))
    if right:
        pads = pads.at[:, :, 0].set(config.map_width - pads[:, :, 0])
    config = config._replace(
        team_spawn_pad_positions=pads, obstacles=jnp.zeros_like(config.obstacles)
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(75))
    inputs = env.policy_inputs(observations, state)
    actor, world_masks = (
        _row(_row(inputs.actors, 0), 0),
        _row(_row(inputs.action_mask, 0), 0),
    )
    flag = team_on_right(actor)
    assert bool(flag) == right
    view, masks = mirror_team_view(actor, world_masks, flag)
    prompt = format_actor_view(view, masks)
    assert "enemy_0_ultimate" in prompt
    reply = parse_action_reply('{"move":"east","combat":"enemy_0_ultimate"}', masks)
    world = validate_actor_action(
        reply._replace(move=mirror_move(reply.move, flag)), world_masks
    )
    idle = ActorAction(*(jnp.zeros(5, jnp.int32) for _ in range(3)))
    chosen = ActorAction(
        *(zero.at[0].set(value) for zero, value in zip(idle, world, strict=True))
    )
    native = idle._replace(
        move=idle.move.at[0].set(4 if right else 3),
        select_target=idle.select_target.at[0].set(6),
        use_ultimate=idle.use_ultimate.at[0].set(1),
    )
    action = env.join_actions(chosen, idle)
    actual = env.step(jax.random.key(76), state, action)
    expected = env.step(jax.random.key(76), state, env.join_actions(native, idle))
    for left, other in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, other)
    next_observations, next_state, _, _, _ = actual
    assert int(next_state.core_state.previous_timestep_move_actions[0]) == (
        4 if right else 3
    )
    assert int(next_state.core_state.previous_timestep_select_target_actions[0]) == 6
    assert int(next_state.core_state.previous_timestep_use_ultimate_actions[0]) == 1
    next_input = env.policy_inputs(next_observations, next_state)
    next_actor, next_mask = mirror_team_view(
        _row(_row(next_input.actors, 0), 0),
        _row(_row(next_input.action_mask, 0), 0),
        flag,
    )
    next_text = format_actor_view(next_actor, next_mask)
    assert "ally_0: move=east target=enemy_0 ultimate=Yes" in next_text
    assert "ultimate_cooldown_remaining=" in next_text
    opponent = env.policy_inputs(next_observations, next_state, team=1)
    target = _row(_row(opponent.actors, 0), 0)
    assert float(target.observation.self_features[21]) > 0
    duration = int(target.observation.self_features[21])
    for remaining in range(duration, -1, -1):
        opponent = env.policy_inputs(next_observations, next_state, team=1)
        target = _row(_row(opponent.actors, 0), 0)
        target_masks = _row(_row(opponent.action_mask, 0), 0)
        text = format_actor_view(target, target_masks, frame="world")
        assert float(target.observation.self_features[21]) == remaining
        if remaining:
            assert f"stun_warrior_charge_duration={remaining}" in text
            assert legal_action_names(target_masks) == {
                "move": ("stay",),
                "combat": ("no_combat",),
            }
        else:
            self_line = next(
                line for line in text.splitlines() if line.startswith("unit_0:")
            )
            assert "stun_warrior_charge_duration=" not in self_line
            assert len(legal_action_names(target_masks)["move"]) > 1
            break
        next_observations, next_state, _, _, _ = env.step(
            jax.random.key(80 + remaining), next_state, env.join_actions(idle, idle)
        )


@pytest.mark.parametrize("ticks", (0, 1, 3))
def test_shield_text_and_masks_follow_public_expiration(ticks: int) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("mage",), team_b_roster=("mage",), max_steps=20
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    _, base = env.reset(jax.random.key(87))
    core = base.core_state._replace(
        spawn_shield_durations=base.core_state.spawn_shield_durations.at[0].set(ticks)
    )
    core, obs, masks, _ = initialize_scenario_state(core, config)
    observations, state = env.reset(jax.random.key(88), initial=(core, obs, masks))
    idle = ActorAction(*(jnp.zeros(5, jnp.int32) for _ in range(3)))
    for remaining in range(ticks, -1, -1):
        inputs = env.policy_inputs(observations, state)
        actor = _row(_row(inputs.actors, 0), 0)
        masks = _row(_row(inputs.action_mask, 0), 0)
        text = format_actor_view(actor, masks, frame="world")
        assert f"Shield ticks: [[{remaining}," in text
        assert (
            int(
                actor.observation.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team[
                    0, 0
                ]
            )
            == remaining
        )
        if remaining:
            assert legal_action_names(masks)["combat"] == ("no_combat",)
        if remaining == 0:
            break
        observations, state, _, _, _ = env.step(
            jax.random.key(90 + remaining), state, env.join_actions(idle, idle)
        )
