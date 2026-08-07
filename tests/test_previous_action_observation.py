"""Previous accepted joint-action state and observation contract tests.

These tests cover the Milestone 5 Step 8 public trajectory: pre-state action
acceptance, one-step state history, post-state actor visibility, and
observer-relative policy-facing action categories.
"""

# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    _build_observation_and_action_mask,
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_STAY,
    MOVE_WEST,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Observation,
    PreviousTimestepActionObservation,
)

_TEAM_A_ACTOR_0 = 0
_TEAM_A_ACTOR_1 = 1
_TEAM_B_ACTOR_0 = MAX_AGENTS_PER_TEAM
_TEAM_B_ACTOR_1 = MAX_AGENTS_PER_TEAM + 1

_TARGET_NONE = 0
_FIRST_ALLY_TARGET = 1
_SECOND_ALLY_TARGET = 2
_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM
_SECOND_ENEMY_TARGET = 2 + MAX_AGENTS_PER_TEAM


def _requested_roster(
    team_a_first_class: int = MAGE_CLASS_ID,
    team_b_first_class: int = MAGE_CLASS_ID,
) -> Array:
    """Return deterministic class IDs for both fixed team blocks."""
    return jnp.asarray(
        (
            team_a_first_class,
            MAGE_CLASS_ID,
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
            team_b_first_class,
            MAGE_CLASS_ID,
            MAGE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
        ),
        dtype=jnp.int32,
    )


def _positions() -> Array:
    """Return nonoverlapping deterministic positions for every fixed slot."""
    return jnp.asarray(
        (
            (2.0, 2.0),
            (4.0, 2.0),
            (6.0, 2.0),
            (8.0, 2.0),
            (10.0, 2.0),
            (2.0, 8.0),
            (4.0, 8.0),
            (6.0, 8.0),
            (8.0, 8.0),
            (10.0, 8.0),
        ),
        dtype=jnp.float32,
    )


def _config(
    *,
    team_sizes: tuple[int, int] = (2, 2),
    positions: Array | None = None,
    observation_radius: float = 30.0,
    basic_interaction_radius: float = 30.0,
    ultimate_interaction_radius: float = 30.0,
    team_a_first_class: int = MAGE_CLASS_ID,
    team_b_first_class: int = MAGE_CLASS_ID,
) -> EnvConfig:
    """Return a deterministic config with explicit policy-relevant ranges."""
    profile = resolve_agent_profile(
        _requested_roster(team_a_first_class, team_b_first_class),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    profile = profile._replace(
        observation_radii=jnp.where(
            profile.active_mask, observation_radius, 0.0
        ).astype(jnp.float32),
        basic_interaction_radii=jnp.where(
            profile.active_mask, basic_interaction_radius, 0.0
        ).astype(jnp.float32),
        ultimate_interaction_radii=jnp.where(
            profile.active_mask, ultimate_interaction_radius, 0.0
        ).astype(jnp.float32),
    )
    configured_positions = _positions() if positions is None else positions
    return EnvConfig(
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=profile,
        ordinary_movement_distance_scale=0.1,
        team_spawn_pad_positions=configured_positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
    )


def _neutral_action() -> Action:
    """Return the canonical neutral submission for every fixed slot."""
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.full((MAX_AGENT_SLOTS,), _TARGET_NONE, dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _action(
    *,
    move_rows: tuple[tuple[int, int], ...] = (),
    target_rows: tuple[tuple[int, int], ...] = (),
    ultimate_rows: tuple[tuple[int, int], ...] = (),
) -> Action:
    """Return a neutral joint action with selected per-head overrides."""
    action = _neutral_action()
    move = action.move
    select_target = action.select_target
    use_ultimate = action.use_ultimate
    for slot, value in move_rows:
        move = move.at[slot].set(value)
    for slot, value in target_rows:
        select_target = select_target.at[slot].set(value)
    for slot, value in ultimate_rows:
        use_ultimate = use_ultimate.at[slot].set(value)
    return Action(
        move=move,
        select_target=select_target,
        use_ultimate=use_ultimate,
    )


def _fully_permissive_action_mask() -> ActionMask:
    """Return an all-true fixed-shape mask for acceptance-boundary tests."""
    return ActionMask(
        move_mask=jnp.ones((MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS), dtype=jnp.bool_),
        select_target_mask=jnp.ones(
            (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS), dtype=jnp.bool_
        ),
        use_ultimate_mask=jnp.ones(
            (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS), dtype=jnp.bool_
        ),
        select_target_use_ultimate_joint_mask=jnp.ones(
            (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS),
            dtype=jnp.bool_,
        ),
    )


def _canonical_only_action_mask() -> ActionMask:
    """Return a mask that permits only the canonical neutral tuple."""
    move_mask = (
        jnp.zeros((MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS), dtype=jnp.bool_)
        .at[:, MOVE_STAY]
        .set(True)
    )
    joint_mask = (
        jnp.zeros(
            (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS),
            dtype=jnp.bool_,
        )
        .at[:, _TARGET_NONE, 0]
        .set(True)
    )
    return ActionMask(
        move_mask=move_mask,
        select_target_mask=jnp.any(joint_mask, axis=-1),
        use_ultimate_mask=jnp.any(joint_mask, axis=-2),
        select_target_use_ultimate_joint_mask=joint_mask,
    )


def _step(
    config: EnvConfig,
    state: EnvState,
    action_mask: ActionMask,
    action: Action,
) -> tuple[EnvState, Observation, ActionMask]:
    """Advance one transition and return its next public snapshot."""
    next_state, observation, _, _, next_action_mask, _ = step(
        config,
        state,
        action_mask,
        action,
        jax.random.key(7),
    )
    return next_state, observation, next_action_mask


def _previous_action_leaves(
    observation: Observation,
) -> tuple[Array, ...]:
    """Return the six policy-facing previous-action tensors."""
    return tuple(observation.previous_timestep_actions)


def _assert_tree_equal(left: object, right: object) -> None:
    """Assert exact equality for two JAX PyTrees."""
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        assert bool(jnp.array_equal(left_leaf, right_leaf))


def _global_target_slot(actor_slot: int, target_action: int) -> int | None:
    """Decode one actor-relative target category to stable global identity."""
    if target_action == _TARGET_NONE:
        return None
    actor_team_start = 0 if actor_slot < MAX_AGENTS_PER_TEAM else MAX_AGENTS_PER_TEAM
    opposing_team_start = MAX_AGENTS_PER_TEAM if actor_team_start == 0 else 0
    if target_action <= MAX_AGENTS_PER_TEAM:
        return actor_team_start + target_action - 1
    return opposing_team_start + target_action - MAX_AGENTS_PER_TEAM - 1


def _observer_relative_target_action(
    observer_slot: int,
    global_target_slot: int | None,
) -> int:
    """Encode stable target identity in one observer's relation convention."""
    if global_target_slot is None:
        return _TARGET_NONE
    observer_team_start = (
        0 if observer_slot < MAX_AGENTS_PER_TEAM else MAX_AGENTS_PER_TEAM
    )
    target_team_start = (
        0 if global_target_slot < MAX_AGENTS_PER_TEAM else MAX_AGENTS_PER_TEAM
    )
    target_row = global_target_slot - target_team_start
    if target_team_start == observer_team_start:
        return 1 + target_row
    return 1 + MAX_AGENTS_PER_TEAM + target_row


def _observed_actor_target_row(
    observation: Observation,
    observer_slot: int,
    actor_slot: int,
) -> Array:
    """Return one observed actor's target-category row."""
    observer_is_team_a = observer_slot < MAX_AGENTS_PER_TEAM
    actor_is_team_a = actor_slot < MAX_AGENTS_PER_TEAM
    actor_relation_row = actor_slot % MAX_AGENTS_PER_TEAM
    previous_actions = observation.previous_timestep_actions
    if observer_is_team_a == actor_is_team_a:
        return previous_actions.ally_previous_timestep_select_target_actions_one_hot[
            observer_slot, actor_relation_row
        ]
    return previous_actions.enemy_previous_timestep_select_target_actions_one_hot[
        observer_slot, actor_relation_row
    ]


def test_reset_exposes_exact_zero_history_schema() -> None:
    """Reset must distinguish absent history from a real neutral action."""
    state, observation, _, _ = reset(_config(), jax.random.key(0))

    assert state.previous_timestep_move_actions.shape == (MAX_AGENT_SLOTS,)
    assert state.previous_timestep_move_actions.dtype == jnp.int32
    assert state.previous_timestep_select_target_actions.shape == (MAX_AGENT_SLOTS,)
    assert state.previous_timestep_select_target_actions.dtype == jnp.int32
    assert state.previous_timestep_use_ultimate_actions.shape == (MAX_AGENT_SLOTS,)
    assert state.previous_timestep_use_ultimate_actions.dtype == jnp.int32
    assert state.has_previous_timestep_joint_action.shape == ()
    assert state.has_previous_timestep_joint_action.dtype == jnp.bool_
    assert not bool(state.has_previous_timestep_joint_action)

    previous_actions = observation.previous_timestep_actions
    assert isinstance(previous_actions, PreviousTimestepActionObservation)
    assert previous_actions.ally_previous_timestep_move_actions_one_hot.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
        NUM_MOVE_ACTIONS,
    )
    assert (
        previous_actions.ally_previous_timestep_select_target_actions_one_hot.shape
        == (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_TARGET_ACTIONS)
    )
    assert (
        previous_actions.ally_previous_timestep_use_ultimate_actions_one_hot.shape
        == (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, NUM_ULTIMATE_ACTIONS)
    )
    for leaf in _previous_action_leaves(observation):
        assert leaf.dtype == jnp.float32
        assert bool(jnp.all(leaf == 0.0))


def test_first_neutral_transition_exposes_valid_neutral_one_hots() -> None:
    """A real neutral action must not be represented as absent history."""
    config = _config()
    state, _, action_mask, _ = reset(config, jax.random.key(0))

    next_state, observation, _ = _step(config, state, action_mask, _neutral_action())

    assert bool(next_state.has_previous_timestep_joint_action)
    previous_actions = observation.previous_timestep_actions
    assert (
        previous_actions.ally_previous_timestep_move_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, MOVE_STAY
        ]
        == 1.0
    )
    assert (
        previous_actions.ally_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, _TARGET_NONE
        ]
        == 1.0
    )
    assert (
        previous_actions.ally_previous_timestep_use_ultimate_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, 0
        ]
        == 1.0
    )


@pytest.mark.parametrize(
    ("head", "malformed_value"),
    (
        pytest.param("move", -1, id="move-lower-bound"),
        pytest.param("move", NUM_MOVE_ACTIONS, id="move-upper-bound"),
        pytest.param("select_target", -1, id="target-lower-bound"),
        pytest.param("select_target", NUM_TARGET_ACTIONS, id="target-upper-bound"),
        pytest.param("use_ultimate", -1, id="ultimate-lower-bound"),
        pytest.param("use_ultimate", NUM_ULTIMATE_ACTIONS, id="ultimate-upper-bound"),
    ),
)
def test_malformed_head_neutralizes_only_that_actors_complete_tuple(
    head: str,
    malformed_value: int,
) -> None:
    """Every malformed head must trigger per-actor whole-tuple containment."""
    config = _config()
    state, _, _, _ = reset(config, jax.random.key(0))
    submitted = _action(
        move_rows=(
            (_TEAM_A_ACTOR_0, MOVE_NORTH),
            (_TEAM_B_ACTOR_0, MOVE_EAST),
        ),
        target_rows=(
            (_TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET),
            (_TEAM_B_ACTOR_0, _FIRST_ENEMY_TARGET),
        ),
        ultimate_rows=(
            (_TEAM_A_ACTOR_0, 1),
            (_TEAM_B_ACTOR_0, 1),
        ),
    )
    submitted = submitted._replace(
        **{head: getattr(submitted, head).at[_TEAM_A_ACTOR_0].set(malformed_value)}
    )

    next_state, _, _ = _step(
        config,
        state,
        _fully_permissive_action_mask(),
        submitted,
    )

    assert next_state.previous_timestep_move_actions[_TEAM_A_ACTOR_0] == MOVE_STAY
    assert (
        next_state.previous_timestep_select_target_actions[_TEAM_A_ACTOR_0]
        == _TARGET_NONE
    )
    assert next_state.previous_timestep_use_ultimate_actions[_TEAM_A_ACTOR_0] == 0
    assert next_state.previous_timestep_move_actions[_TEAM_B_ACTOR_0] == MOVE_EAST
    assert (
        next_state.previous_timestep_select_target_actions[_TEAM_B_ACTOR_0]
        == _FIRST_ENEMY_TARGET
    )
    assert next_state.previous_timestep_use_ultimate_actions[_TEAM_B_ACTOR_0] == 1


def test_multiple_malformed_heads_remain_isolated_to_one_actor() -> None:
    """Multiple malformed heads must still produce one local neutral tuple."""
    config = _config()
    state, _, _, _ = reset(config, jax.random.key(0))
    submitted = _action(
        move_rows=(
            (_TEAM_A_ACTOR_0, -1),
            (_TEAM_B_ACTOR_0, MOVE_EAST),
        ),
        target_rows=(
            (_TEAM_A_ACTOR_0, NUM_TARGET_ACTIONS),
            (_TEAM_B_ACTOR_0, _FIRST_ENEMY_TARGET),
        ),
        ultimate_rows=(
            (_TEAM_A_ACTOR_0, NUM_ULTIMATE_ACTIONS),
            (_TEAM_B_ACTOR_0, 1),
        ),
    )

    next_state, _, _ = _step(config, state, _fully_permissive_action_mask(), submitted)

    assert next_state.previous_timestep_move_actions[_TEAM_A_ACTOR_0] == MOVE_STAY
    assert (
        next_state.previous_timestep_select_target_actions[_TEAM_A_ACTOR_0]
        == _TARGET_NONE
    )
    assert next_state.previous_timestep_use_ultimate_actions[_TEAM_A_ACTOR_0] == 0
    assert next_state.previous_timestep_move_actions[_TEAM_B_ACTOR_0] == MOVE_EAST
    assert (
        next_state.previous_timestep_select_target_actions[_TEAM_B_ACTOR_0]
        == _FIRST_ENEMY_TARGET
    )
    assert next_state.previous_timestep_use_ultimate_actions[_TEAM_B_ACTOR_0] == 1


def test_in_domain_mask_fallback_preserves_movement_combat_independence() -> None:
    """Mask illegality must preserve the other in-domain acceptance group."""
    config = _config()
    state, _, _, _ = reset(config, jax.random.key(0))
    action = _action(
        move_rows=((_TEAM_A_ACTOR_0, MOVE_EAST),),
        target_rows=((_TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET),),
    )

    combat_legal_mask = _canonical_only_action_mask()
    combat_legal_mask = combat_legal_mask._replace(
        select_target_use_ultimate_joint_mask=(
            combat_legal_mask.select_target_use_ultimate_joint_mask.at[
                _TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET, 0
            ].set(True)
        )
    )
    combat_legal_mask = combat_legal_mask._replace(
        select_target_mask=jnp.any(
            combat_legal_mask.select_target_use_ultimate_joint_mask, axis=-1
        ),
        use_ultimate_mask=jnp.any(
            combat_legal_mask.select_target_use_ultimate_joint_mask, axis=-2
        ),
    )
    state_after_illegal_movement, _, _ = _step(config, state, combat_legal_mask, action)
    assert (
        state_after_illegal_movement.previous_timestep_move_actions[_TEAM_A_ACTOR_0]
        == MOVE_STAY
    )
    assert (
        state_after_illegal_movement.previous_timestep_select_target_actions[
            _TEAM_A_ACTOR_0
        ]
        == _FIRST_ENEMY_TARGET
    )

    movement_legal_mask = _canonical_only_action_mask()
    movement_legal_mask = movement_legal_mask._replace(
        move_mask=movement_legal_mask.move_mask.at[_TEAM_A_ACTOR_0, MOVE_EAST].set(True)
    )
    state_after_illegal_combat, _, _ = _step(config, state, movement_legal_mask, action)
    assert (
        state_after_illegal_combat.previous_timestep_move_actions[_TEAM_A_ACTOR_0]
        == MOVE_EAST
    )
    assert (
        state_after_illegal_combat.previous_timestep_select_target_actions[
            _TEAM_A_ACTOR_0
        ]
        == _TARGET_NONE
    )
    assert (
        state_after_illegal_combat.previous_timestep_use_ultimate_actions[
            _TEAM_A_ACTOR_0
        ]
        == 0
    )


def test_second_transition_completely_overwrites_first_history() -> None:
    """One-step history must never retain an older non-neutral action."""
    config = _config()
    state, _, action_mask, _ = reset(config, jax.random.key(0))
    non_neutral_action = _action(
        move_rows=((_TEAM_A_ACTOR_0, MOVE_EAST),),
        target_rows=((_TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET),),
    )

    first_state, _, first_next_mask = _step(
        config, state, action_mask, non_neutral_action
    )
    second_state, second_observation, _ = _step(
        config, first_state, first_next_mask, _neutral_action()
    )

    assert bool(jnp.all(second_state.previous_timestep_move_actions == MOVE_STAY))
    assert bool(
        jnp.all(second_state.previous_timestep_select_target_actions == _TARGET_NONE)
    )
    assert bool(jnp.all(second_state.previous_timestep_use_ultimate_actions == 0))
    previous_actions = second_observation.previous_timestep_actions
    self_move_row = previous_actions.ally_previous_timestep_move_actions_one_hot[
        _TEAM_A_ACTOR_0, 0
    ]
    assert self_move_row[MOVE_STAY] == 1.0
    assert jnp.sum(self_move_row) == 1.0


def test_target_categories_are_reinterpreted_for_each_observers_team() -> None:
    """Every observer-relative category must preserve one stable target identity."""
    config = _config()
    state, _, _, _ = reset(config, jax.random.key(0))
    state = state._replace(
        previous_timestep_select_target_actions=(
            state.previous_timestep_select_target_actions.at[_TEAM_A_ACTOR_0]
            .set(_SECOND_ALLY_TARGET)
            .at[_TEAM_A_ACTOR_1]
            .set(_SECOND_ENEMY_TARGET)
            .at[_TEAM_B_ACTOR_0]
            .set(_SECOND_ALLY_TARGET)
            .at[_TEAM_B_ACTOR_1]
            .set(_SECOND_ENEMY_TARGET)
        ),
        has_previous_timestep_joint_action=jnp.asarray(True),
    )

    observation, _ = _build_observation_and_action_mask(state, config)
    previous_actions = observation.previous_timestep_actions

    assert (
        previous_actions.ally_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, _SECOND_ALLY_TARGET
        ]
        == 1.0
    )
    assert (
        previous_actions.enemy_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, _SECOND_ENEMY_TARGET
        ]
        == 1.0
    )
    assert (
        previous_actions.enemy_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_ACTOR_0, 1, _SECOND_ALLY_TARGET
        ]
        == 1.0
    )
    assert (
        previous_actions.enemy_previous_timestep_select_target_actions_one_hot[
            _TEAM_B_ACTOR_0, 0, _SECOND_ENEMY_TARGET
        ]
        == 1.0
    )
    assert (
        previous_actions.enemy_previous_timestep_select_target_actions_one_hot[
            _TEAM_B_ACTOR_0, 1, _SECOND_ALLY_TARGET
        ]
        == 1.0
    )


@pytest.mark.parametrize(
    "actor_slot",
    (
        pytest.param(_TEAM_A_ACTOR_0, id="team-a-first-actor"),
        pytest.param(_TEAM_A_ACTOR_1, id="team-a-second-actor"),
        pytest.param(_TEAM_B_ACTOR_0, id="team-b-first-actor"),
        pytest.param(_TEAM_B_ACTOR_1, id="team-b-second-actor"),
    ),
)
@pytest.mark.parametrize("target_action", range(NUM_TARGET_ACTIONS))
def test_every_target_category_preserves_identity_for_both_observer_teams(
    actor_slot: int,
    target_action: int,
) -> None:
    """Exhaustively prove target-category conversion without mirrored formulas."""
    config = _config()
    state, _, _, _ = reset(config, jax.random.key(0))
    state = state._replace(
        previous_timestep_select_target_actions=(
            state.previous_timestep_select_target_actions.at[actor_slot].set(
                target_action
            )
        ),
        has_previous_timestep_joint_action=jnp.asarray(True),
    )

    observation, _ = _build_observation_and_action_mask(state, config)
    global_target_slot = _global_target_slot(actor_slot, target_action)
    for observer_slot in (
        _TEAM_A_ACTOR_0,
        _TEAM_A_ACTOR_1,
        _TEAM_B_ACTOR_0,
        _TEAM_B_ACTOR_1,
    ):
        expected_action = _observer_relative_target_action(
            observer_slot, global_target_slot
        )
        observed_row = _observed_actor_target_row(
            observation, observer_slot, actor_slot
        )
        assert observed_row.shape == (NUM_TARGET_ACTIONS,)
        assert observed_row.dtype == jnp.float32
        assert jnp.sum(observed_row) == 1.0
        assert observed_row[expected_action] == 1.0


def test_hidden_actor_rows_are_zero_but_hidden_target_identity_remains_public() -> None:
    """Actor visibility gates rows while target visibility does not rewrite identity."""
    positions = _positions()
    positions = positions.at[_TEAM_A_ACTOR_0].set(
        jnp.asarray((2.0, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_A_ACTOR_1].set(
        jnp.asarray((3.0, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_ACTOR_0].set(
        jnp.asarray((18.0, 10.0), dtype=jnp.float32)
    )
    config = _config(positions=positions)
    profile = config.agent_profile._replace(
        observation_radii=config.agent_profile.observation_radii.at[
            _TEAM_A_ACTOR_0
        ].set(2.0)
    )
    config = config._replace(agent_profile=profile)
    state, _, action_mask, _ = reset(config, jax.random.key(0))
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_ACTOR_0].set(1.0)
    )
    _, action_mask = _build_observation_and_action_mask(state, config)
    action = _action(
        target_rows=((_TEAM_A_ACTOR_1, _FIRST_ENEMY_TARGET),),
    )

    next_state, observation, _ = _step(config, state, action_mask, action)
    previous_actions = observation.previous_timestep_actions

    assert next_state.current_health[_TEAM_B_ACTOR_0] == 0.0
    assert not bool(next_state.alive_mask[_TEAM_B_ACTOR_0])
    assert bool(observation.ally_visibility_mask[_TEAM_A_ACTOR_0, 1])
    assert not bool(observation.enemy_visibility_mask[_TEAM_A_ACTOR_0, 0])
    assert bool(
        jnp.all(
            previous_actions.enemy_previous_timestep_move_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ]
            == 0.0
        )
    )
    assert bool(
        jnp.all(
            previous_actions.enemy_previous_timestep_select_target_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ]
            == 0.0
        )
    )
    assert bool(
        jnp.all(
            previous_actions.enemy_previous_timestep_use_ultimate_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ]
            == 0.0
        )
    )
    assert (
        previous_actions.ally_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_ACTOR_0, 1, _FIRST_ENEMY_TARGET
        ]
        == 1.0
    )
    assert bool(jnp.all(observation.enemy_unit_features[_TEAM_A_ACTOR_0, 0] == 0.0))


def test_spawn_shield_hides_actor_history_without_rewriting_target_history() -> None:
    """Conceal actor rows while preserving a visible actor's target provenance."""
    config = _config()
    state, _, _, _ = reset(config, jax.random.key(0))
    state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[_TEAM_B_ACTOR_0].set(3),
        previous_timestep_move_actions=state.previous_timestep_move_actions.at[
            _TEAM_B_ACTOR_0
        ].set(MOVE_EAST),
        previous_timestep_select_target_actions=(
            state.previous_timestep_select_target_actions.at[_TEAM_A_ACTOR_1].set(
                _FIRST_ENEMY_TARGET
            )
        ),
        has_previous_timestep_joint_action=jnp.asarray(True),
    )

    observation, _ = _build_observation_and_action_mask(state, config)
    previous_actions = observation.previous_timestep_actions

    assert bool(observation.ally_visibility_mask[_TEAM_A_ACTOR_0, 1])
    assert not bool(observation.enemy_visibility_mask[_TEAM_A_ACTOR_0, 0])
    assert bool(
        jnp.all(
            previous_actions.enemy_previous_timestep_move_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ]
            == 0.0
        )
    )
    assert (
        previous_actions.ally_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_ACTOR_0, 1, _FIRST_ENEMY_TARGET
        ]
        == 1.0
    )


def test_spawn_shield_concealment_is_invariant_to_hidden_position_and_move() -> None:
    """Hidden spatial and actor-history rows reveal neither position nor movement."""
    config = _config(team_b_first_class=PRIEST_CLASS_ID)
    state, _, _, _ = reset(config, jax.random.key(0))
    shielded_slot = _TEAM_B_ACTOR_0
    first_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[shielded_slot].set(2),
        previous_timestep_move_actions=state.previous_timestep_move_actions.at[
            shielded_slot
        ].set(MOVE_EAST),
        has_previous_timestep_joint_action=jnp.asarray(True),
    )
    second_state = first_state._replace(
        agent_positions=first_state.agent_positions.at[shielded_slot].set(
            jnp.asarray((3.0, 8.0), dtype=jnp.float32)
        ),
        previous_timestep_move_actions=first_state.previous_timestep_move_actions.at[
            shielded_slot
        ].set(MOVE_WEST),
    )

    first_observation, _ = _build_observation_and_action_mask(first_state, config)
    second_observation, _ = _build_observation_and_action_mask(second_state, config)
    first_history = first_observation.previous_timestep_actions
    second_history = second_observation.previous_timestep_actions

    assert not bool(first_observation.enemy_visibility_mask[_TEAM_A_ACTOR_0, 0])
    assert not bool(second_observation.enemy_visibility_mask[_TEAM_A_ACTOR_0, 0])
    assert bool(
        jnp.array_equal(
            first_observation.enemy_unit_features[_TEAM_A_ACTOR_0, 0],
            second_observation.enemy_unit_features[_TEAM_A_ACTOR_0, 0],
        )
    )
    assert bool(
        jnp.array_equal(
            first_history.enemy_previous_timestep_move_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ],
            second_history.enemy_previous_timestep_move_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ],
        )
    )
    assert bool(
        jnp.array_equal(
            first_history.enemy_previous_timestep_select_target_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ],
            second_history.enemy_previous_timestep_select_target_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ],
        )
    )
    assert bool(
        jnp.array_equal(
            first_history.enemy_previous_timestep_use_ultimate_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ],
            second_history.enemy_previous_timestep_use_ultimate_actions_one_hot[
                _TEAM_A_ACTOR_0, 0
            ],
        )
    )


def test_spawn_shield_rejected_combat_pair_preserves_legal_movement() -> None:
    """Official shield masks reject combat without discarding legal movement."""
    config = _config(team_sizes=(1, 1))
    state, _, _, _ = reset(config, jax.random.key(0))
    shielded_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[_TEAM_A_ACTOR_0].set(3)
    )
    _, action_mask = _build_observation_and_action_mask(shielded_state, config)
    submitted_action = _action(
        move_rows=((_TEAM_A_ACTOR_0, MOVE_EAST),),
        target_rows=((_TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET),),
        ultimate_rows=((_TEAM_A_ACTOR_0, 1),),
    )

    next_state, observation, _ = _step(
        config,
        shielded_state,
        action_mask,
        submitted_action,
    )

    assert next_state.previous_timestep_move_actions[_TEAM_A_ACTOR_0] == MOVE_EAST
    assert (
        next_state.previous_timestep_select_target_actions[_TEAM_A_ACTOR_0]
        == _TARGET_NONE
    )
    assert next_state.previous_timestep_use_ultimate_actions[_TEAM_A_ACTOR_0] == 0
    previous_actions = observation.previous_timestep_actions
    assert (
        previous_actions.ally_previous_timestep_move_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, MOVE_EAST
        ]
        == 1.0
    )
    assert (
        previous_actions.ally_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, _TARGET_NONE
        ]
        == 1.0
    )
    assert (
        previous_actions.ally_previous_timestep_use_ultimate_actions_one_hot[
            _TEAM_A_ACTOR_0, 0, 0
        ]
        == 1.0
    )


@pytest.mark.parametrize(
    ("actor_start_x", "move_action", "expected_visible"),
    (
        pytest.param(4.1, MOVE_WEST, True, id="enters-range"),
        pytest.param(4.0, MOVE_EAST, False, id="leaves-range"),
    ),
)
def test_successor_visibility_controls_previous_action_exposure(
    actor_start_x: float,
    move_action: int,
    expected_visible: bool,
) -> None:
    """The successor snapshot, not the pre-state, gates action-history rows."""
    positions = _positions()
    positions = positions.at[_TEAM_A_ACTOR_0].set(
        jnp.asarray((2.0, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_ACTOR_0].set(
        jnp.asarray((actor_start_x, 2.0), dtype=jnp.float32)
    )
    config = _config(
        team_sizes=(1, 1),
        positions=positions,
        observation_radius=2.0,
    )
    state, _, action_mask, _ = reset(config, jax.random.key(0))

    next_state, observation, _ = _step(
        config,
        state,
        action_mask,
        _action(move_rows=((_TEAM_B_ACTOR_0, move_action),)),
    )
    previous_actions = observation.previous_timestep_actions
    observed_move_row = previous_actions.enemy_previous_timestep_move_actions_one_hot[
        _TEAM_A_ACTOR_0, 0
    ]

    assert (
        bool(observation.enemy_visibility_mask[_TEAM_A_ACTOR_0, 0]) is expected_visible
    )
    assert next_state.previous_timestep_move_actions[_TEAM_B_ACTOR_0] == move_action
    if expected_visible:
        assert observed_move_row[move_action] == 1.0
        assert jnp.sum(observed_move_row) == 1.0
    else:
        assert bool(jnp.all(observed_move_row == 0.0))


def test_collision_limited_movement_records_accepted_category() -> None:
    """History records accepted intent rather than realized displacement."""
    config = _config(team_sizes=(1, 1))
    state, _, action_mask, _ = reset(config, jax.random.key(0))
    boundary_position = jnp.asarray((19.5, 2.0), dtype=jnp.float32)
    state = state._replace(
        agent_positions=state.agent_positions.at[_TEAM_A_ACTOR_0].set(boundary_position)
    )
    _, action_mask = _build_observation_and_action_mask(state, config)

    next_state, _, _ = _step(
        config,
        state,
        action_mask,
        _action(move_rows=((_TEAM_A_ACTOR_0, MOVE_EAST),)),
    )

    assert bool(
        jnp.array_equal(next_state.agent_positions[_TEAM_A_ACTOR_0], boundary_position)
    )
    assert next_state.previous_timestep_move_actions[_TEAM_A_ACTOR_0] == MOVE_EAST


def test_charge_and_precommitted_movement_are_both_recorded() -> None:
    """Accepted Charge must not erase its independently accepted movement head."""
    positions = _positions()
    positions = positions.at[_TEAM_A_ACTOR_0].set(
        jnp.asarray((2.0, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_ACTOR_0].set(
        jnp.asarray((5.0, 2.0), dtype=jnp.float32)
    )
    config = _config(
        positions=positions,
        team_sizes=(1, 1),
        team_a_first_class=WARRIOR_CLASS_ID,
    )
    state, _, action_mask, _ = reset(config, jax.random.key(0))
    charge_action = _action(
        move_rows=((_TEAM_A_ACTOR_0, MOVE_EAST),),
        target_rows=((_TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET),),
        ultimate_rows=((_TEAM_A_ACTOR_0, 1),),
    )
    assert bool(
        action_mask.select_target_use_ultimate_joint_mask[
            _TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET, 1
        ]
    )

    next_state, _, _ = _step(config, state, action_mask, charge_action)

    assert next_state.previous_timestep_move_actions[_TEAM_A_ACTOR_0] == MOVE_EAST
    assert (
        next_state.previous_timestep_select_target_actions[_TEAM_A_ACTOR_0]
        == _FIRST_ENEMY_TARGET
    )
    assert next_state.previous_timestep_use_ultimate_actions[_TEAM_A_ACTOR_0] == 1


def test_jitted_step_matches_complete_eager_previous_action_outputs() -> None:
    """Compiled and eager transitions must agree on every new state/output leaf."""
    config = _config()
    state, _, action_mask, _ = reset(config, jax.random.key(0))
    action = _action(
        move_rows=((_TEAM_A_ACTOR_0, MOVE_EAST),),
        target_rows=((_TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET),),
    )

    eager_outputs = step(config, state, action_mask, action, jax.random.key(1))
    jitted_outputs = cast(
        object,
        jax.jit(step)(config, state, action_mask, action, jax.random.key(1)),
    )

    _assert_tree_equal(eager_outputs, jitted_outputs)


def test_compiled_scan_establishes_and_overwrites_fixed_history() -> None:
    """Scanned carry must retain fixed history leaves and replacement timing."""
    config = _config()
    initial_state, _, initial_mask, _ = reset(config, jax.random.key(0))
    first_action = _action(
        move_rows=((_TEAM_A_ACTOR_0, MOVE_EAST),),
        target_rows=((_TEAM_A_ACTOR_0, _FIRST_ENEMY_TARGET),),
    )
    second_action = _neutral_action()
    actions = Action(
        move=jnp.stack((first_action.move, second_action.move)),
        select_target=jnp.stack(
            (first_action.select_target, second_action.select_target)
        ),
        use_ultimate=jnp.stack((first_action.use_ultimate, second_action.use_ultimate)),
    )

    def scan_rollout(
        state: EnvState,
        action_mask: ActionMask,
        rollout_actions: Action,
    ) -> tuple[EnvState, tuple[Array, Array, Array, Array]]:
        def scan_step(
            carry: tuple[EnvState, ActionMask],
            action: Action,
        ) -> tuple[
            tuple[EnvState, ActionMask],
            tuple[Array, Array, Array, Array],
        ]:
            current_state, current_mask = carry
            next_state, _, _, _, next_mask, _ = step(
                config,
                current_state,
                current_mask,
                action,
                jax.random.key(2),
            )
            history = (
                next_state.previous_timestep_move_actions,
                next_state.previous_timestep_select_target_actions,
                next_state.previous_timestep_use_ultimate_actions,
                next_state.has_previous_timestep_joint_action,
            )
            return (next_state, next_mask), history

        (final_state, _), histories = jax.lax.scan(
            scan_step, (state, action_mask), rollout_actions
        )
        return final_state, histories

    final_state, histories = cast(
        tuple[EnvState, tuple[Array, Array, Array, Array]],
        jax.jit(scan_rollout)(initial_state, initial_mask, actions),
    )
    move_history, target_history, ultimate_history, validity_history = histories

    assert move_history.shape == (2, MAX_AGENT_SLOTS)
    assert target_history.shape == (2, MAX_AGENT_SLOTS)
    assert ultimate_history.shape == (2, MAX_AGENT_SLOTS)
    assert validity_history.shape == (2,)
    assert move_history.dtype == jnp.int32
    assert validity_history.dtype == jnp.bool_
    assert move_history[0, _TEAM_A_ACTOR_0] == MOVE_EAST
    assert target_history[0, _TEAM_A_ACTOR_0] == _FIRST_ENEMY_TARGET
    assert move_history[1, _TEAM_A_ACTOR_0] == MOVE_STAY
    assert target_history[1, _TEAM_A_ACTOR_0] == _TARGET_NONE
    assert bool(jnp.all(validity_history))
    assert final_state.previous_timestep_move_actions[_TEAM_A_ACTOR_0] == MOVE_STAY
