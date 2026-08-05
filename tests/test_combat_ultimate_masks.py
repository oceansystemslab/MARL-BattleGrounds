"""Semantic and transformation proofs for target-conditioned ultimate masks."""
# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import marl_battlegrounds.core.combat as combat
from marl_battlegrounds.core.combat import (
    MAGE_DAMAGE_AURA_MULTIPLIER,
    ULTIMATE_COOLDOWN_BY_CLASS,
    ULTIMATE_DAMAGE_BY_CLASS,
)
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import _build_observation_and_action_mask, step
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    NO_TEAM_ID,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Observation,
)

_ACTOR_SLOT = 0
_ALLY_SLOT = 1
_ENEMY_SLOT = MAX_AGENTS_PER_TEAM
_NONE_TARGET = 0
_SELF_TARGET = 1
_ALLY_TARGET = 2
_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM
_PADDED_SLOT = 2


def _requested_roster(actor_class_id: int) -> Array:
    """Return a fixed roster with one actor, one ally, and one enemy."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    roster = roster.at[_ACTOR_SLOT].set(actor_class_id)
    roster = roster.at[_ALLY_SLOT].set(MAGE_CLASS_ID)
    roster = roster.at[_ENEMY_SLOT].set(MAGE_CLASS_ID)
    return roster


def _empty_obstacles() -> Array:
    """Return an inactive fixed-size obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _blocking_pillar() -> Array:
    """Return one pillar crossing the actor-to-enemy line segment."""
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_X].set(4.0)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_Y].set(2.0)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_RADIUS].set(0.75)
    return obstacles.at[0, OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _stay_action(*, target: int = 0, use_ultimate: int = 0) -> Action:
    """Return a joint action with an optional combat choice for the actor."""
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
        .at[_ACTOR_SLOT]
        .set(target),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
        .at[_ACTOR_SLOT]
        .set(use_ultimate),
    )


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the action mask paired with an explicitly built test state."""
    _, action_mask = _build_observation_and_action_mask(state, config)
    return action_mask


def _scenario(
    actor_class_id: int,
    *,
    enemy_x: float = 6.0,
    observation_radius: float = 10.0,
    basic_radius: float = 10.0,
    ultimate_radius: float = 10.0,
    obstacles: Array | None = None,
) -> tuple[EnvConfig, EnvState]:
    """Build a deterministic 2v1 scenario with stable relation-local rows."""
    profile = resolve_agent_profile(
        _requested_roster(actor_class_id),
        jnp.asarray((2, 1), dtype=jnp.int32),
    )
    profile = profile._replace(
        base_movement_speeds=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        observation_radii=profile.observation_radii.at[_ACTOR_SLOT].set(
            observation_radius
        ),
        basic_interaction_radii=profile.basic_interaction_radii.at[_ACTOR_SLOT].set(
            basic_radius
        ),
        ultimate_interaction_radii=profile.ultimate_interaction_radii.at[
            _ACTOR_SLOT
        ].set(ultimate_radius),
    )
    initial_positions = (
        jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
        .at[_ACTOR_SLOT]
        .set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
        .at[_ALLY_SLOT]
        .set(jnp.asarray((3.0, 2.0), dtype=jnp.float32))
        .at[_ENEMY_SLOT]
        .set(jnp.asarray((enemy_x, 2.0), dtype=jnp.float32))
    )
    config = EnvConfig(
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=initial_positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
    )

    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=initial_positions,
        alive_mask=profile.active_mask,
        current_health=profile.max_health,
        ultimate_cooldowns=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        slow_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
        stun_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32),
        rogue_poison_anti_heal_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        mage_burst_damage_amplification_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        priest_blessing_of_freedom_slow_floor_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        spawn_shield_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        previous_timestep_move_actions=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        previous_timestep_select_target_actions=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        previous_timestep_use_ultimate_actions=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        has_previous_timestep_joint_action=jnp.asarray(False),
    )
    return config, state


def _outputs(
    config: EnvConfig,
    state: EnvState,
    action: Action | None = None,
) -> tuple[EnvState, Observation, ActionMask]:
    """Advance one inert tick and return the Step 4 public surfaces."""
    next_state, observation, _, _, action_mask, _ = step(
        config,
        state,
        _current_action_mask(config, state),
        _stay_action() if action is None else action,
        jax.random.key(7),
    )
    return next_state, observation, action_mask


def _assert_exact_marginals(action_mask: ActionMask) -> None:
    """Assert both flat masks are exact views of the authoritative pair mask."""
    joint_mask = action_mask.select_target_use_ultimate_joint_mask
    assert joint_mask.shape == (
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert joint_mask.dtype == jnp.bool_
    assert bool(
        jnp.array_equal(action_mask.select_target_mask, jnp.any(joint_mask, axis=-1))
    )
    assert bool(
        jnp.array_equal(action_mask.use_ultimate_mask, jnp.any(joint_mask, axis=1))
    )


def _assert_only_first_category_is_valid(mask_row: Array) -> None:
    """Assert one categorical row exposes exactly its canonical first entry."""
    flattened_row = mask_row.reshape(-1)

    assert flattened_row.dtype == jnp.bool_
    assert bool(flattened_row[0])
    assert not bool(jnp.any(flattened_row[1:]))
    assert int(jnp.sum(flattened_row)) == 1


def _masked_categorical_statistics(mask: Array) -> tuple[Array, Array, Array]:
    """Return probabilities, log-probabilities, and entropy for finite logits."""
    logits = jnp.linspace(-2.5, 3.5, mask.shape[-1], dtype=jnp.float32)
    masked_logits = jnp.where(mask, logits[None, :], -jnp.inf)
    probabilities = jax.nn.softmax(masked_logits, axis=-1)
    log_probabilities = jax.nn.log_softmax(masked_logits, axis=-1)
    safe_log_probabilities = jnp.where(mask, log_probabilities, 0.0)
    entropy = -jnp.sum(probabilities * safe_log_probabilities, axis=-1)
    return probabilities, log_probabilities, entropy


@pytest.mark.parametrize(
    ("actor_class_id", "legal_ultimate_targets"),
    [
        pytest.param(MAGE_CLASS_ID, (_NONE_TARGET,), id="mage-no-target"),
        pytest.param(WARRIOR_CLASS_ID, (_ENEMY_TARGET,), id="warrior-enemy"),
        pytest.param(HUNTER_CLASS_ID, (_ENEMY_TARGET,), id="hunter-enemy"),
        pytest.param(ROGUE_CLASS_ID, (_ENEMY_TARGET,), id="rogue-enemy"),
        pytest.param(
            PRIEST_CLASS_ID,
            (_SELF_TARGET, _ALLY_TARGET),
            id="priest-allies-including-self",
        ),
        pytest.param(NEUTRAL_CLASS_ID, (), id="neutral-unavailable"),
    ],
)
def test_canonical_ultimate_target_relations(
    actor_class_id: int,
    legal_ultimate_targets: tuple[int, ...],
) -> None:
    """Prove every class exposes only its canonical ultimate target relation."""
    config, state = _scenario(actor_class_id)
    _, observation, action_mask = _outputs(config, state)

    expected_ultimate_lane = jnp.zeros((NUM_TARGET_ACTIONS,), dtype=bool)
    if legal_ultimate_targets:
        expected_ultimate_lane = expected_ultimate_lane.at[
            jnp.asarray(legal_ultimate_targets)
        ].set(True)

    assert bool(
        jnp.array_equal(
            action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, :, 1],
            expected_ultimate_lane,
        )
    )
    assert "ally_targetability_mask" not in Observation._fields
    assert "enemy_targetability_mask" not in Observation._fields
    assert observation.ally_visibility_mask.dtype == jnp.bool_
    assert observation.enemy_visibility_mask.dtype == jnp.bool_
    _assert_exact_marginals(action_mask)


@pytest.mark.parametrize(
    ("actor_class_id", "enemy_x", "expects_basic", "expects_ultimate"),
    [
        pytest.param(
            WARRIOR_CLASS_ID,
            6.0,
            False,
            True,
            id="warrior-ultimate-only",
        ),
        pytest.param(
            HUNTER_CLASS_ID,
            7.0,
            True,
            False,
            id="hunter-basic-only",
        ),
    ],
)
def test_basic_and_ultimate_ranges_are_independent(
    actor_class_id: int,
    enemy_x: float,
    expects_basic: bool,
    expects_ultimate: bool,
) -> None:
    """Prove canonical radius asymmetry survives the shared target head."""
    config, state = _scenario(
        actor_class_id,
        enemy_x=enemy_x,
        basic_radius=0.5 if actor_class_id == WARRIOR_CLASS_ID else 5.5,
        ultimate_radius=6.0 if actor_class_id == WARRIOR_CLASS_ID else 4.0,
    )
    _, _, action_mask = _outputs(config, state)
    pair = action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, _ENEMY_TARGET]

    assert bool(pair[0]) is expects_basic
    assert bool(pair[1]) is expects_ultimate
    assert bool(action_mask.select_target_mask[_ACTOR_SLOT, _ENEMY_TARGET])
    _assert_exact_marginals(action_mask)


def test_positive_cooldown_disables_only_the_ultimate_lane() -> None:
    """Prove remaining cooldown gates ultimate use without narrowing basics."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=2.25)
    _, _, available_mask = _outputs(config, state)
    cooled_down_state = state._replace(
        ultimate_cooldowns=state.ultimate_cooldowns.at[_ACTOR_SLOT].set(3)
    )
    _, _, unavailable_mask = _outputs(config, cooled_down_state)

    assert bool(jnp.any(available_mask.select_target_use_ultimate_joint_mask[0, :, 1]))
    assert not bool(
        jnp.any(unavailable_mask.select_target_use_ultimate_joint_mask[0, :, 1])
    )
    assert bool(
        jnp.array_equal(
            available_mask.select_target_use_ultimate_joint_mask[0, :, 0],
            unavailable_mask.select_target_use_ultimate_joint_mask[0, :, 0],
        )
    )
    _assert_exact_marginals(unavailable_mask)


@pytest.mark.parametrize(
    "stun_channels",
    [
        pytest.param((STUN_CHANNEL_WARRIOR_CHARGE,), id="warrior-charge"),
        pytest.param((STUN_CHANNEL_HUNTER_TRAP,), id="hunter-trap"),
        pytest.param((STUN_CHANNEL_ROGUE_POISON,), id="rogue-poison"),
        pytest.param(
            (
                STUN_CHANNEL_WARRIOR_CHARGE,
                STUN_CHANNEL_HUNTER_TRAP,
                STUN_CHANNEL_ROGUE_POISON,
            ),
            id="simultaneous",
        ),
    ],
)
def test_actor_stun_removes_nonempty_combat_control(
    stun_channels: tuple[int, ...],
) -> None:
    """Prove every actor stun source preserves only target-none/no-ultimate."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=2.25)
    stun_maxima = (
        combat.WARRIOR_CHARGE_STUN_DURATION_TICKS,
        combat.HUNTER_TRAP_STUN_DURATION_TICKS,
        combat.ROGUE_POISON_STUN_DURATION_TICKS,
    )
    for channel in stun_channels:
        state = state._replace(
            stun_durations=state.stun_durations.at[_ACTOR_SLOT, channel].set(
                stun_maxima[channel]
            )
        )

    _, action_mask = _build_observation_and_action_mask(state, config)
    actor_pair = action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT]

    assert bool(actor_pair[_NONE_TARGET, 0])
    assert not bool(jnp.any(actor_pair[1:, 0]))
    assert not bool(jnp.any(actor_pair[:, 1]))
    _assert_exact_marginals(action_mask)


def test_candidate_stun_changes_its_own_row_but_not_actor_legality() -> None:
    """Prove stun is actor-side control and rows remain actor-local."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=2.25)
    _, _, control_mask = _outputs(config, state)
    candidate_stunned = state._replace(
        stun_durations=state.stun_durations.at[
            _ENEMY_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ].set(2)
    )
    _, _, changed_mask = _outputs(config, candidate_stunned)

    assert bool(
        jnp.array_equal(
            changed_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT],
            control_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT],
        )
    )
    assert not bool(
        jnp.array_equal(
            changed_mask.select_target_use_ultimate_joint_mask[_ENEMY_SLOT],
            control_mask.select_target_use_ultimate_joint_mask[_ENEMY_SLOT],
        )
    )


@pytest.mark.parametrize(
    ("actor_active", "actor_alive"),
    [
        pytest.param(False, False, id="inactive"),
        pytest.param(True, False, id="dead"),
    ],
)
def test_inactive_or_dead_actor_exposes_only_canonical_no_op(
    actor_active: bool,
    actor_alive: bool,
) -> None:
    """Prove every nonacting actor mask has exact singleton no-op support."""
    config, state = _scenario(MAGE_CLASS_ID)
    # The inactive case deliberately mutates only the identity gates so this
    # low-level mask test can isolate redaction behavior. It is not an
    # official host-valid configuration/state pair.
    profile = config.agent_profile._replace(
        active_mask=config.agent_profile.active_mask.at[_ACTOR_SLOT].set(actor_active),
        team_ids=config.agent_profile.team_ids.at[_ACTOR_SLOT].set(
            config.agent_profile.team_ids[_ACTOR_SLOT] if actor_active else NO_TEAM_ID
        ),
    )
    config = config._replace(agent_profile=profile)
    state = state._replace(
        alive_mask=state.alive_mask.at[_ACTOR_SLOT].set(actor_alive),
        current_health=state.current_health.at[_ACTOR_SLOT].set(
            state.current_health[_ACTOR_SLOT] if actor_alive else 0.0
        ),
    )

    _, _, action_mask = _outputs(config, state)

    _assert_only_first_category_is_valid(action_mask.move_mask[_ACTOR_SLOT])
    _assert_only_first_category_is_valid(
        action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT]
    )
    _assert_only_first_category_is_valid(action_mask.select_target_mask[_ACTOR_SLOT])
    _assert_only_first_category_is_valid(action_mask.use_ultimate_mask[_ACTOR_SLOT])
    _assert_exact_marginals(action_mask)


def test_every_categorical_mask_row_is_numerically_sampleable() -> None:
    """Prove ordinary hard masking is normalized and NaN-free for every slot."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=2.25)
    _, _, action_mask = _outputs(config, state)
    active_and_alive = jnp.logical_and(
        config.agent_profile.active_mask,
        state.alive_mask,
    )
    nonacting = jnp.logical_not(active_and_alive)

    categorical_masks = (
        action_mask.move_mask,
        action_mask.select_target_mask,
        action_mask.use_ultimate_mask,
        action_mask.select_target_use_ultimate_joint_mask.reshape(
            MAX_AGENT_SLOTS,
            NUM_TARGET_ACTIONS * NUM_ULTIMATE_ACTIONS,
        ),
    )

    for mask in categorical_masks:
        probabilities, log_probabilities, entropy = _masked_categorical_statistics(mask)

        assert bool(jnp.all(jnp.any(mask, axis=-1)))
        assert not bool(jnp.any(jnp.isnan(probabilities)))
        assert not bool(jnp.any(jnp.isnan(log_probabilities)))
        assert bool(jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0))
        assert bool(jnp.all(probabilities[jnp.logical_not(mask)] == 0.0))
        assert bool(jnp.all(jnp.isfinite(log_probabilities[mask])))
        assert bool(jnp.all(jnp.isneginf(log_probabilities[jnp.logical_not(mask)])))
        assert bool(jnp.all(jnp.isfinite(entropy)))
        assert bool(jnp.all(probabilities[nonacting, 0] == 1.0))
        assert bool(jnp.all(entropy[nonacting] == 0.0))


@pytest.mark.parametrize(
    ("candidate_slot", "target_action", "candidate_active", "candidate_alive"),
    [
        pytest.param(_ENEMY_SLOT, _ENEMY_TARGET, False, False, id="inactive"),
        pytest.param(_ENEMY_SLOT, _ENEMY_TARGET, True, False, id="dead"),
        pytest.param(_PADDED_SLOT, 1 + _PADDED_SLOT, False, False, id="padded"),
    ],
)
def test_inactive_dead_or_padded_candidate_cannot_be_targeted(
    candidate_slot: int,
    target_action: int,
    candidate_active: bool,
    candidate_alive: bool,
) -> None:
    """Prove candidate participation gates both basic and ultimate lanes."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=2.25)
    profile = config.agent_profile._replace(
        active_mask=config.agent_profile.active_mask.at[candidate_slot].set(
            candidate_active
        ),
        team_ids=config.agent_profile.team_ids.at[candidate_slot].set(
            config.agent_profile.team_ids[candidate_slot]
            if candidate_active
            else NO_TEAM_ID
        ),
    )
    config = config._replace(agent_profile=profile)
    state = state._replace(
        alive_mask=state.alive_mask.at[candidate_slot].set(candidate_alive),
        current_health=state.current_health.at[candidate_slot].set(
            state.current_health[candidate_slot] if candidate_alive else 0.0
        ),
    )

    _, _, action_mask = _outputs(config, state)

    assert not bool(
        jnp.any(
            action_mask.select_target_use_ultimate_joint_mask[
                _ACTOR_SLOT, target_action
            ]
        )
    )


def test_no_team_sentinels_never_form_combat_relations() -> None:
    """Prove equal padding sentinels are neither allies nor enemies."""
    config, state = _scenario(PRIEST_CLASS_ID)
    no_team_ids = config.agent_profile.team_ids.at[_ACTOR_SLOT].set(NO_TEAM_ID)
    no_team_ids = no_team_ids.at[_ALLY_SLOT].set(NO_TEAM_ID)
    config = config._replace(
        agent_profile=config.agent_profile._replace(team_ids=no_team_ids)
    )

    _, _, action_mask = _outputs(config, state)
    actor_pair = action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT]

    assert bool(actor_pair[_NONE_TARGET, 0])
    assert not bool(jnp.any(actor_pair[1:]))


@pytest.mark.parametrize(
    "health_fraction",
    [0.01, 1.0],
    ids=["low-positive", "full"],
)
def test_priest_ultimate_legality_is_independent_of_ally_health(
    health_fraction: float,
) -> None:
    """Prove health magnitude does not replace the liveness contract."""
    config, state = _scenario(PRIEST_CLASS_ID)
    state = state._replace(
        current_health=state.current_health.at[_ALLY_SLOT].set(
            config.agent_profile.max_health[_ALLY_SLOT] * health_fraction
        )
    )

    _, _, action_mask = _outputs(config, state)

    assert bool(
        action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, _ALLY_TARGET, 1]
    )


def test_unrelated_statuses_do_not_gate_ultimate_legality() -> None:
    """Prove only stun and cooldown affect actor-side ultimate availability."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=2.25)
    _, control_mask = _build_observation_and_action_mask(state, config)
    changed_state = state._replace(
        slow_durations=state.slow_durations.at[_ACTOR_SLOT, 0].set(4),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[_ACTOR_SLOT].set(4)
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[_ALLY_SLOT].set(4)
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[_ACTOR_SLOT].set(
                combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS
            )
        ),
    )
    _, changed_mask = _build_observation_and_action_mask(changed_state, config)

    assert bool(
        jnp.array_equal(
            changed_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT],
            control_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT],
        )
    )


@pytest.mark.parametrize(
    ("enemy_x", "expects_ultimate"),
    [
        pytest.param(6.0, True, id="inclusive-boundary"),
        pytest.param(6.001, False, id="outside-boundary"),
    ],
)
def test_ultimate_range_boundary_is_inclusive(
    enemy_x: float,
    expects_ultimate: bool,
) -> None:
    """Prove targeted-ultimate range uses the documented inclusive boundary."""
    config, state = _scenario(
        WARRIOR_CLASS_ID,
        enemy_x=enemy_x,
        basic_radius=0.5,
        ultimate_radius=4.0,
    )
    _, _, action_mask = _outputs(config, state)

    assert (
        bool(
            action_mask.select_target_use_ultimate_joint_mask[
                _ACTOR_SLOT, _ENEMY_TARGET, 1
            ]
        )
        is expects_ultimate
    )


def test_blocked_los_disables_targeted_ultimate() -> None:
    """Prove targeted ultimates consume the established LOS visibility truth."""
    config, state = _scenario(
        WARRIOR_CLASS_ID,
        obstacles=_blocking_pillar(),
    )
    _, observation, action_mask = _outputs(config, state)

    assert not bool(observation.enemy_visibility_mask[_ACTOR_SLOT, 0])
    assert not bool(
        jnp.any(
            action_mask.select_target_use_ultimate_joint_mask[
                _ACTOR_SLOT, _ENEMY_TARGET
            ]
        )
    )


def test_hidden_candidate_changes_do_not_leak_through_actor_outputs() -> None:
    """Prove hidden state and hidden-preserving movement are observationally equal."""
    config, state = _scenario(
        WARRIOR_CLASS_ID,
        enemy_x=8.0,
        observation_radius=3.0,
    )
    control_observation, control_mask = _build_observation_and_action_mask(
        state,
        config,
    )
    hidden_changed_state = state._replace(
        agent_positions=state.agent_positions.at[_ENEMY_SLOT].set(
            jnp.asarray((10.0, 2.0), dtype=jnp.float32)
        ),
        current_health=state.current_health.at[_ENEMY_SLOT].set(1.0),
        ultimate_cooldowns=state.ultimate_cooldowns.at[_ENEMY_SLOT].set(9),
        slow_durations=state.slow_durations.at[
            _ENEMY_SLOT, SLOW_CHANNEL_WARRIOR_CHARGE
        ].set(combat.WARRIOR_CHARGE_SLOW_DURATION_TICKS),
        stun_durations=state.stun_durations.at[
            _ENEMY_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ].set(combat.HUNTER_TRAP_STUN_DURATION_TICKS),
    )
    changed_observation, changed_mask = _build_observation_and_action_mask(
        hidden_changed_state,
        config,
    )

    assert not bool(control_observation.enemy_visibility_mask[_ACTOR_SLOT, 0])
    assert not bool(changed_observation.enemy_visibility_mask[_ACTOR_SLOT, 0])
    assert bool(
        jnp.array_equal(
            control_observation.enemy_unit_features[_ACTOR_SLOT],
            changed_observation.enemy_unit_features[_ACTOR_SLOT],
        )
    )
    assert bool(
        jnp.array_equal(
            control_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT],
            changed_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT],
        )
    )


def test_accepted_ultimate_updates_health_cooldown_and_returned_mask() -> None:
    """Prove accepted use updates state before the returned mask is built."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=2.25)
    current_action_mask = _current_action_mask(config, state)
    next_state, _, _, _, next_action_mask, _ = step(
        config,
        state,
        current_action_mask,
        _stay_action(target=_ENEMY_TARGET, use_ultimate=1),
        jax.random.key(8),
    )

    assert (
        next_state.current_health[_ENEMY_SLOT]
        == state.current_health[_ENEMY_SLOT]
        - ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID] * MAGE_DAMAGE_AURA_MULTIPLIER
    )
    assert (
        next_state.ultimate_cooldowns[_ACTOR_SLOT]
        == ULTIMATE_COOLDOWN_BY_CLASS[WARRIOR_CLASS_ID]
    )
    assert not bool(
        next_action_mask.select_target_use_ultimate_joint_mask[
            _ACTOR_SLOT, _ENEMY_TARGET, 1
        ]
    )


def test_jitted_step_matches_eager_pair_semantics() -> None:
    """Prove compiled outputs preserve meaningful pair and marginal values."""
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=6.0, basic_radius=0.5)
    eager = _outputs(config, state)
    compiled_state, compiled_observation, _, _, compiled_mask, _ = cast(
        tuple[EnvState, Observation, object, object, ActionMask, object],
        jax.jit(step)(
            config,
            state,
            _current_action_mask(config, state),
            _stay_action(),
            jax.random.key(7),
        ),
    )

    assert bool(jnp.array_equal(compiled_state.step_count, eager[0].step_count))
    assert bool(
        jnp.array_equal(
            compiled_observation.enemy_visibility_mask,
            eager[1].enemy_visibility_mask,
        )
    )
    assert bool(
        jnp.array_equal(
            compiled_mask.select_target_use_ultimate_joint_mask,
            eager[2].select_target_use_ultimate_joint_mask,
        )
    )
    assert not bool(
        compiled_mask.select_target_use_ultimate_joint_mask[
            _ACTOR_SLOT, _ENEMY_TARGET, 0
        ]
    )
    assert bool(
        compiled_mask.select_target_use_ultimate_joint_mask[
            _ACTOR_SLOT, _ENEMY_TARGET, 1
        ]
    )
    _assert_only_first_category_is_valid(compiled_mask.move_mask[_PADDED_SLOT])
    _assert_only_first_category_is_valid(
        compiled_mask.select_target_use_ultimate_joint_mask[_PADDED_SLOT]
    )
    _assert_only_first_category_is_valid(compiled_mask.select_target_mask[_PADDED_SLOT])
    _assert_only_first_category_is_valid(compiled_mask.use_ultimate_mask[_PADDED_SLOT])
    _assert_exact_marginals(compiled_mask)


def test_scanned_rollout_preserves_meaningful_pair_history() -> None:
    """Prove a compiled scan retains fixed shapes and ultimate-only semantics."""
    horizon = 3
    config, state = _scenario(WARRIOR_CLASS_ID, enemy_x=6.0, basic_radius=0.5)
    keys = jax.random.split(jax.random.key(10), horizon)
    initial_action_mask = _current_action_mask(config, state)

    def _scan_step(
        carry: tuple[EnvState, ActionMask],
        step_key: Array,
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array, Array]]:
        current_state, current_action_mask = carry
        next_state, _, _, _, action_mask, _ = step(
            config,
            current_state,
            current_action_mask,
            _stay_action(),
            step_key,
        )
        return (next_state, action_mask), (
            action_mask.move_mask,
            action_mask.select_target_use_ultimate_joint_mask,
            action_mask.select_target_mask,
            action_mask.use_ultimate_mask,
        )

    def _rollout(
        initial_state: EnvState,
        initial_mask: ActionMask,
        scan_keys: Array,
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array, Array]]:
        """Run the fixed-horizon transition under one compiled scan."""
        return jax.lax.scan(_scan_step, (initial_state, initial_mask), scan_keys)

    (
        (final_state, _),
        (
            move_history,
            joint_history,
            target_history,
            ultimate_history,
        ),
    ) = cast(
        tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array, Array]],
        jax.jit(_rollout)(state, initial_action_mask, keys),
    )

    assert final_state.step_count == horizon
    assert move_history.shape == (horizon, MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)
    assert move_history.dtype == jnp.bool_
    assert joint_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert joint_history.dtype == jnp.bool_
    assert target_history.shape == (horizon, MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert ultimate_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert not bool(jnp.any(joint_history[:, _ACTOR_SLOT, _ENEMY_TARGET, 0]))
    assert bool(jnp.all(joint_history[:, _ACTOR_SLOT, _ENEMY_TARGET, 1]))
    assert bool(jnp.all(move_history[:, _PADDED_SLOT, MOVE_STAY]))
    assert bool(jnp.all(jnp.sum(move_history[:, _PADDED_SLOT], axis=-1) == 1))
    assert bool(jnp.all(joint_history[:, _PADDED_SLOT, _NONE_TARGET, 0]))
    assert bool(jnp.all(jnp.sum(joint_history[:, _PADDED_SLOT], axis=(-2, -1)) == 1))
    assert bool(jnp.all(target_history[:, _PADDED_SLOT, _NONE_TARGET]))
    assert bool(jnp.all(jnp.sum(target_history[:, _PADDED_SLOT], axis=-1) == 1))
    assert bool(jnp.all(ultimate_history[:, _PADDED_SLOT, 0]))
    assert bool(jnp.all(jnp.sum(ultimate_history[:, _PADDED_SLOT], axis=-1) == 1))
    assert bool(jnp.array_equal(target_history, jnp.any(joint_history, axis=-1)))
    assert bool(jnp.array_equal(ultimate_history, jnp.any(joint_history, axis=2)))
