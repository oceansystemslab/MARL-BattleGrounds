"""Focused semantic proofs for Milestone 5 basic target legality."""
# pyright: reportPrivateUsage=false

from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import marl_battlegrounds.core.combat as combat
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    _build_observation_and_action_mask,
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
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
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
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
_ALLY_TARGET_START = 1
_ENEMY_TARGET_START = 1 + MAX_AGENTS_PER_TEAM


def _requested_roster(actor_class_id: int) -> Array:
    """Return a fixed roster with one actor, ally, and enemy candidate."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    roster = roster.at[_ACTOR_SLOT].set(actor_class_id)
    roster = roster.at[_ALLY_SLOT].set(MAGE_CLASS_ID)
    roster = roster.at[_ENEMY_SLOT].set(MAGE_CLASS_ID)
    return roster


def _stay_action() -> Action:
    """Return a slot-aligned joint action that preserves scenario geometry."""
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the action mask paired with an explicitly built test state."""
    _, action_mask = _build_observation_and_action_mask(state, config)
    return action_mask


def _target_scenario(actor_class_id: int = MAGE_CLASS_ID) -> tuple[EnvConfig, EnvState]:
    """Build a clear-LOS 2v1 scenario with every candidate inside basic range."""
    profile = resolve_agent_profile(
        _requested_roster(actor_class_id),
        jnp.asarray((2, 1), dtype=jnp.int32),
    )
    # Neutral has zero catalog radii, so explicit test radii keep the spatial
    # preconditions identical while capability is varied independently.
    profile = profile._replace(
        agent_radii=jnp.where(profile.active_mask, 0.5, 0.0).astype(jnp.float32),
        base_movement_speeds=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        observation_radii=jnp.where(profile.active_mask, 10.0, 0.0).astype(jnp.float32),
        basic_interaction_radii=jnp.where(profile.active_mask, 10.0, 0.0).astype(
            jnp.float32
        ),
        ultimate_interaction_radii=jnp.where(profile.active_mask, 10.0, 0.0).astype(
            jnp.float32
        ),
    )
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    positions = positions.at[_ACTOR_SLOT].set(
        jnp.asarray((2.0, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[_ALLY_SLOT].set(jnp.asarray((4.0, 2.0), dtype=jnp.float32))
    positions = positions.at[_ENEMY_SLOT].set(
        jnp.asarray((6.0, 2.0), dtype=jnp.float32)
    )
    config = EnvConfig(
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
    )
    state, *_ = reset(config, jax.random.key(1))
    return config, state


def _step_scenario(
    config: EnvConfig, state: EnvState
) -> tuple[EnvState, Observation, ActionMask]:
    """Advance one inert tick and return the public target-mask surfaces."""
    next_state, observation, _, _, action_mask, _ = step(
        config,
        state,
        _current_action_mask(config, state),
        _stay_action(),
        jax.random.key(2),
    )
    return next_state, observation, action_mask


def _assert_public_target_contract(
    observation: Observation,
    action_mask: ActionMask,
) -> None:
    """Assert perception and basic-legality surfaces remain separately owned."""
    assert "ally_targetability_mask" not in Observation._fields
    assert "enemy_targetability_mask" not in Observation._fields
    assert observation.ally_visibility_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert observation.enemy_visibility_mask.shape == (
        MAX_AGENT_SLOTS,
        MAX_AGENTS_PER_TEAM,
    )
    assert action_mask.select_target_mask.shape == (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)
    assert action_mask.select_target_mask.dtype == jnp.bool_
    assert action_mask.select_target_use_ultimate_joint_mask.shape == (
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
        2,
    )
    assert action_mask.select_target_use_ultimate_joint_mask.dtype == jnp.bool_
    assert bool(
        jnp.array_equal(
            action_mask.select_target_mask,
            jnp.any(action_mask.select_target_use_ultimate_joint_mask, axis=-1),
        )
    )
    assert bool(
        jnp.array_equal(
            action_mask.use_ultimate_mask,
            jnp.any(action_mask.select_target_use_ultimate_joint_mask, axis=1),
        )
    )


def _basic_relation_masks(action_mask: ActionMask) -> tuple[Array, Array]:
    """Return ally/enemy slices from the authoritative no-ultimate lane."""
    basic_lane = action_mask.select_target_use_ultimate_joint_mask[..., 0]
    return (
        basic_lane[:, _ALLY_TARGET_START : _ALLY_TARGET_START + MAX_AGENTS_PER_TEAM],
        basic_lane[:, _ENEMY_TARGET_START:],
    )


@pytest.mark.parametrize(
    "shielded_source_slot",
    (
        pytest.param(_ACTOR_SLOT, id="team-a"),
        pytest.param(_ENEMY_SLOT, id="team-b"),
    ),
)
def test_spawn_shield_source_allows_every_move_and_only_neutral_combat(
    shielded_source_slot: int,
) -> None:
    """A shielded source retains all movement and exactly one inert combat pair."""
    config, state = _target_scenario()
    shielded_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[
            shielded_source_slot
        ].set(3)
    )

    observation, action_mask = _build_observation_and_action_mask(
        shielded_state, config
    )
    source_joint_mask = action_mask.select_target_use_ultimate_joint_mask[
        shielded_source_slot
    ]

    _assert_public_target_contract(observation, action_mask)
    assert action_mask.move_mask[shielded_source_slot].shape == (NUM_MOVE_ACTIONS,)
    assert bool(jnp.all(action_mask.move_mask[shielded_source_slot]))
    assert bool(action_mask.select_target_mask[shielded_source_slot, 0])
    assert not bool(jnp.any(action_mask.select_target_mask[shielded_source_slot, 1:]))
    assert bool(action_mask.use_ultimate_mask[shielded_source_slot, 0])
    assert not bool(jnp.any(action_mask.use_ultimate_mask[shielded_source_slot, 1:]))
    assert source_joint_mask.shape == (NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS)
    assert bool(source_joint_mask[0, 0])
    assert int(jnp.sum(source_joint_mask)) == 1


@pytest.mark.parametrize(
    (
        "actor_slot",
        "actor_class_id",
        "shielded_candidate_slot",
        "target_action",
    ),
    (
        pytest.param(
            _ACTOR_SLOT,
            PRIEST_CLASS_ID,
            _ALLY_SLOT,
            2,
            id="team-a-support-to-shielded-ally",
        ),
        pytest.param(
            _ACTOR_SLOT,
            HUNTER_CLASS_ID,
            _ENEMY_SLOT,
            1 + MAX_AGENTS_PER_TEAM,
            id="team-a-offense-to-shielded-enemy",
        ),
        pytest.param(
            _ENEMY_SLOT,
            MAGE_CLASS_ID,
            _ACTOR_SLOT,
            1 + MAX_AGENTS_PER_TEAM,
            id="team-b-offense-to-shielded-enemy",
        ),
    ),
)
def test_spawn_shield_removes_candidates_from_authoritative_combat_masks(
    actor_slot: int,
    actor_class_id: int,
    shielded_candidate_slot: int,
    target_action: int,
) -> None:
    """Exclude shielded ally and enemy recipients from the shared joint mask."""
    config, state = _target_scenario(actor_class_id)
    _, control_mask = _build_observation_and_action_mask(state, config)
    shielded_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[
            shielded_candidate_slot
        ].set(3)
    )

    observation, shielded_mask = _build_observation_and_action_mask(
        shielded_state,
        config,
    )

    _assert_public_target_contract(observation, shielded_mask)
    assert bool(
        jnp.any(
            control_mask.select_target_use_ultimate_joint_mask[
                actor_slot,
                target_action,
            ]
        )
    )
    assert not bool(
        jnp.any(
            shielded_mask.select_target_use_ultimate_joint_mask[
                actor_slot,
                target_action,
            ]
        )
    )
    assert not bool(shielded_mask.select_target_mask[actor_slot, target_action])


@pytest.mark.parametrize(
    (
        "actor_class_id",
        "expects_ally_targets",
        "expects_enemy_targets",
    ),
    [
        pytest.param(MAGE_CLASS_ID, False, True, id="mage"),
        pytest.param(WARRIOR_CLASS_ID, False, True, id="warrior"),
        pytest.param(HUNTER_CLASS_ID, False, True, id="hunter"),
        pytest.param(ROGUE_CLASS_ID, False, True, id="rogue"),
        pytest.param(PRIEST_CLASS_ID, True, False, id="priest"),
        pytest.param(NEUTRAL_CLASS_ID, False, False, id="neutral"),
    ],
)
def test_basic_targetability_matches_actor_capability(
    actor_class_id: int,
    expects_ally_targets: bool,
    expects_enemy_targets: bool,
) -> None:
    """Prove the complete ally/enemy relation contract for each actor class."""
    config, state = _target_scenario(actor_class_id)
    _, observation, action_mask = _step_scenario(config, state)

    expected_ally = jnp.asarray(
        (expects_ally_targets, expects_ally_targets, False, False, False)
    )
    expected_enemy = jnp.asarray((expects_enemy_targets, False, False, False, False))

    _assert_public_target_contract(observation, action_mask)
    basic_ally, basic_enemy = _basic_relation_masks(action_mask)
    assert bool(jnp.array_equal(basic_ally[_ACTOR_SLOT], expected_ally))
    assert bool(jnp.array_equal(basic_enemy[_ACTOR_SLOT], expected_enemy))
    assert bool(action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, 0, 0])


@pytest.mark.parametrize(
    "active_stun_channels",
    [
        pytest.param(
            (STUN_CHANNEL_WARRIOR_CHARGE,),
            id="warrior-charge",
        ),
        pytest.param(
            (STUN_CHANNEL_HUNTER_TRAP,),
            id="hunter-trap",
        ),
        pytest.param(
            (STUN_CHANNEL_ROGUE_POISON,),
            id="rogue-poison",
        ),
        pytest.param(
            (
                STUN_CHANNEL_WARRIOR_CHARGE,
                STUN_CHANNEL_HUNTER_TRAP,
                STUN_CHANNEL_ROGUE_POISON,
            ),
            id="simultaneous-stuns",
        ),
    ],
)
def test_actor_stun_disables_nonempty_targets_but_preserves_none(
    active_stun_channels: tuple[int, ...],
) -> None:
    """Prove every stun source removes actor control but preserves target-none."""
    config, state = _target_scenario()
    stun_maxima = (
        combat.WARRIOR_CHARGE_STUN_DURATION_TICKS,
        combat.HUNTER_TRAP_STUN_DURATION_TICKS,
        combat.ROGUE_POISON_STUN_DURATION_TICKS,
    )
    for channel in active_stun_channels:
        state = state._replace(
            stun_durations=state.stun_durations.at[_ACTOR_SLOT, channel].set(
                stun_maxima[channel]
            )
        )

    observation, action_mask = _build_observation_and_action_mask(state, config)

    _assert_public_target_contract(observation, action_mask)
    basic_ally, basic_enemy = _basic_relation_masks(action_mask)
    assert not bool(jnp.any(basic_ally[_ACTOR_SLOT]))
    assert not bool(jnp.any(basic_enemy[_ACTOR_SLOT]))
    assert bool(action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, 0, 0])
    assert not bool(
        jnp.any(action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, 1:, 0])
    )


@pytest.mark.parametrize(
    "candidate_stun_channel",
    [
        pytest.param(
            STUN_CHANNEL_WARRIOR_CHARGE,
            id="warrior-charge",
        ),
        pytest.param(
            STUN_CHANNEL_HUNTER_TRAP,
            id="hunter-trap",
        ),
        pytest.param(
            STUN_CHANNEL_ROGUE_POISON,
            id="rogue-poison",
        ),
    ],
)
def test_candidate_stun_does_not_change_targetability(
    candidate_stun_channel: int,
) -> None:
    """Prove stun is an actor-control predicate, not a candidate predicate."""
    config, state = _target_scenario()
    _, control_action_mask = _build_observation_and_action_mask(state, config)
    stun_maxima = (
        combat.WARRIOR_CHARGE_STUN_DURATION_TICKS,
        combat.HUNTER_TRAP_STUN_DURATION_TICKS,
        combat.ROGUE_POISON_STUN_DURATION_TICKS,
    )
    stunned_state = state._replace(
        stun_durations=state.stun_durations.at[_ENEMY_SLOT, candidate_stun_channel].set(
            stun_maxima[candidate_stun_channel]
        )
    )
    _, stunned_action_mask = _build_observation_and_action_mask(
        stunned_state,
        config,
    )

    control_ally, control_enemy = _basic_relation_masks(control_action_mask)
    stunned_ally, stunned_enemy = _basic_relation_masks(stunned_action_mask)
    assert bool(control_enemy[_ACTOR_SLOT, 0])
    assert bool(stunned_enemy[_ACTOR_SLOT, 0])
    assert bool(
        jnp.array_equal(
            stunned_ally[_ACTOR_SLOT],
            control_ally[_ACTOR_SLOT],
        )
    )
    assert bool(
        jnp.array_equal(
            stunned_enemy[_ACTOR_SLOT],
            control_enemy[_ACTOR_SLOT],
        )
    )
    assert bool(
        jnp.array_equal(
            stunned_action_mask.select_target_mask[_ACTOR_SLOT],
            control_action_mask.select_target_mask[_ACTOR_SLOT],
        )
    )


@pytest.mark.parametrize(
    "actor_class_id",
    [
        pytest.param(MAGE_CLASS_ID, id="damage-actor"),
        pytest.param(PRIEST_CLASS_ID, id="healing-actor"),
    ],
)
def test_actor_stun_changes_only_control_and_exposed_stun_features(
    actor_class_id: int,
) -> None:
    """Prove stun preserves perception while its public status columns update."""
    config, state = _target_scenario(actor_class_id)
    _, control_observation, control_action_mask = _step_scenario(config, state)
    stunned_state = state._replace(
        stun_durations=state.stun_durations.at[
            _ACTOR_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ].set(2)
    )
    _, stunned_observation, stunned_action_mask = _step_scenario(config, stunned_state)

    assert bool(
        jnp.array_equal(
            stunned_observation.ally_visibility_mask,
            control_observation.ally_visibility_mask,
        )
    )
    assert bool(
        jnp.array_equal(
            stunned_observation.enemy_visibility_mask,
            control_observation.enemy_visibility_mask,
        )
    )

    stun_start = AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION
    stun_end = AGENT_FEATURE_STUN_ROGUE_POISON_DURATION + 1
    for stunned_features, control_features in (
        (stunned_observation.self_features, control_observation.self_features),
        (
            stunned_observation.ally_unit_features,
            control_observation.ally_unit_features,
        ),
        (
            stunned_observation.enemy_unit_features,
            control_observation.enemy_unit_features,
        ),
    ):
        assert bool(
            jnp.array_equal(
                stunned_features[..., :stun_start],
                control_features[..., :stun_start],
            )
        )
        assert bool(
            jnp.array_equal(
                stunned_features[..., stun_end:],
                control_features[..., stun_end:],
            )
        )

    assert bool(
        stunned_observation.self_features[
            _ACTOR_SLOT, AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION + 1
        ]
        == 1.0
    )
    assert bool(jnp.any(control_action_mask.select_target_mask[_ACTOR_SLOT, 1:]))
    assert not bool(jnp.any(stunned_action_mask.select_target_mask[_ACTOR_SLOT, 1:]))


@pytest.mark.parametrize(
    "candidate_health_fraction",
    [
        pytest.param(1.0, id="full-health"),
        pytest.param(0.01, id="low-positive-health"),
    ],
)
def test_priest_targetability_is_independent_of_ally_health(
    candidate_health_fraction: float,
) -> None:
    """Prove health magnitude does not determine basic target legality."""
    config, state = _target_scenario(PRIEST_CLASS_ID)
    candidate_health = (
        config.agent_profile.max_health[_ALLY_SLOT] * candidate_health_fraction
    )
    state = state._replace(
        current_health=state.current_health.at[_ALLY_SLOT].set(candidate_health)
    )

    _, _, action_mask = _step_scenario(config, state)

    basic_ally, _ = _basic_relation_masks(action_mask)
    assert bool(basic_ally[_ACTOR_SLOT, _ALLY_SLOT])
    assert bool(
        action_mask.select_target_use_ultimate_joint_mask[
            _ACTOR_SLOT, _ALLY_TARGET_START + _ALLY_SLOT, 0
        ]
    )


UnrelatedStateCase = Literal[
    "ultimate-cooldown",
    "slow-duration",
    "anti-heal-duration",
    "mage-burst-duration",
    "priest-slow-floor-duration",
]


@pytest.mark.parametrize(
    "unrelated_state_case",
    [
        pytest.param("ultimate-cooldown", id="ultimate-cooldown"),
        pytest.param("slow-duration", id="slow-duration"),
        pytest.param("anti-heal-duration", id="anti-heal-duration"),
        pytest.param("mage-burst-duration", id="mage-burst-duration"),
        pytest.param(
            "priest-slow-floor-duration",
            id="priest-slow-floor-duration",
        ),
    ],
)
def test_unrelated_combat_state_does_not_change_basic_targetability(
    unrelated_state_case: UnrelatedStateCase,
) -> None:
    """Prove unrelated cooldown and status fields do not enter basic legality."""
    config, state = _target_scenario()
    _, control_action_mask = _build_observation_and_action_mask(state, config)

    if unrelated_state_case == "ultimate-cooldown":
        changed_state = state._replace(
            ultimate_cooldowns=state.ultimate_cooldowns.at[_ACTOR_SLOT].set(7)
        )
    elif unrelated_state_case == "slow-duration":
        changed_state = state._replace(
            slow_durations=state.slow_durations.at[_ACTOR_SLOT, 0].set(3)
        )
    elif unrelated_state_case == "anti-heal-duration":
        changed_state = state._replace(
            rogue_poison_anti_heal_durations=(
                state.rogue_poison_anti_heal_durations.at[_ACTOR_SLOT].set(3)
            )
        )
    elif unrelated_state_case == "mage-burst-duration":
        changed_state = state._replace(
            mage_burst_damage_amplification_durations=(
                state.mage_burst_damage_amplification_durations.at[_ACTOR_SLOT].set(3)
            )
        )
    else:
        changed_state = state._replace(
            priest_blessing_of_freedom_slow_floor_durations=(
                state.priest_blessing_of_freedom_slow_floor_durations.at[
                    _ACTOR_SLOT
                ].set(combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS)
            )
        )

    _, changed_action_mask = _build_observation_and_action_mask(
        changed_state,
        config,
    )

    control_ally, control_enemy = _basic_relation_masks(control_action_mask)
    changed_ally, changed_enemy = _basic_relation_masks(changed_action_mask)
    assert bool(
        jnp.array_equal(
            changed_ally[_ACTOR_SLOT],
            control_ally[_ACTOR_SLOT],
        )
    )
    assert bool(
        jnp.array_equal(
            changed_enemy[_ACTOR_SLOT],
            control_enemy[_ACTOR_SLOT],
        )
    )
    assert bool(
        jnp.array_equal(
            changed_action_mask.select_target_use_ultimate_joint_mask[
                _ACTOR_SLOT, :, 0
            ],
            control_action_mask.select_target_use_ultimate_joint_mask[
                _ACTOR_SLOT, :, 0
            ],
        )
    )


@pytest.mark.parametrize(
    ("actor_active", "actor_alive"),
    [
        pytest.param(False, False, id="inactive"),
        pytest.param(True, False, id="dead"),
    ],
)
def test_inactive_or_dead_actor_exposes_only_target_none(
    actor_active: bool,
    actor_alive: bool,
) -> None:
    """Prove nonacting actors expose target-none without enabling unit targets."""
    config, state = _target_scenario()
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

    _, _, action_mask = _step_scenario(config, state)

    basic_ally, basic_enemy = _basic_relation_masks(action_mask)
    assert not bool(jnp.any(basic_ally[_ACTOR_SLOT]))
    assert not bool(jnp.any(basic_enemy[_ACTOR_SLOT]))
    assert bool(action_mask.select_target_mask[_ACTOR_SLOT, 0])
    assert not bool(jnp.any(action_mask.select_target_mask[_ACTOR_SLOT, 1:]))
    assert bool(action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, 0, 0])
    assert (
        int(jnp.sum(action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT]))
        == 1
    )


def test_no_team_slots_never_form_basic_target_relations() -> None:
    """Prove matching NO_TEAM_ID values do not constitute an ally relation."""
    config, state = _target_scenario(PRIEST_CLASS_ID)
    no_team_ids = config.agent_profile.team_ids.at[_ACTOR_SLOT].set(NO_TEAM_ID)
    no_team_ids = no_team_ids.at[_ALLY_SLOT].set(NO_TEAM_ID)
    config = config._replace(
        agent_profile=config.agent_profile._replace(team_ids=no_team_ids)
    )

    _, observation, action_mask = _step_scenario(config, state)

    assert bool(observation.ally_visibility_mask[_ACTOR_SLOT, _ACTOR_SLOT])
    assert bool(observation.ally_visibility_mask[_ACTOR_SLOT, _ALLY_SLOT])
    basic_ally, basic_enemy = _basic_relation_masks(action_mask)
    assert not bool(jnp.any(basic_ally[_ACTOR_SLOT]))
    assert not bool(jnp.any(basic_enemy[_ACTOR_SLOT]))
    assert bool(action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, 0, 0])
    assert not bool(
        jnp.any(action_mask.select_target_use_ultimate_joint_mask[_ACTOR_SLOT, 1:, 0])
    )


def test_jitted_step_matches_eager_class_and_stun_targetability() -> None:
    """Prove compiled target masks match eager class and actor-stun semantics."""
    config, state = _target_scenario()
    state = state._replace(
        stun_durations=state.stun_durations.at[
            _ALLY_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ].set(2)
    )

    eager = _step_scenario(config, state)
    compiled_step = jax.jit(step)
    compiled_state, compiled_observation, _, _, compiled_action_mask, _ = cast(
        tuple[EnvState, Observation, object, object, ActionMask, object],
        compiled_step(
            config,
            state,
            _current_action_mask(config, state),
            _stay_action(),
            jax.random.key(2),
        ),
    )

    assert jax.tree_util.tree_structure(compiled_state) == jax.tree_util.tree_structure(
        eager[0]
    )
    assert bool(
        jnp.array_equal(
            compiled_observation.ally_visibility_mask,
            eager[1].ally_visibility_mask,
        )
    )
    assert bool(
        jnp.array_equal(
            compiled_observation.enemy_visibility_mask,
            eager[1].enemy_visibility_mask,
        )
    )
    assert bool(
        jnp.array_equal(
            compiled_action_mask.select_target_mask, eager[2].select_target_mask
        )
    )
    assert bool(
        compiled_action_mask.select_target_use_ultimate_joint_mask[
            _ACTOR_SLOT, _ENEMY_TARGET_START, 0
        ]
    )
