"""Build TDM-GAMMA controller test inputs from a real Scenario 1 observation.

This support module owns test construction only; it holds no tests. It starts
from the compiled Scenario 1 observation (a real Core observation, canonical
5v5 on a 20 x 10 map) and edits it with ``_replace``: the recipient's own row,
visible ally and enemy rows (including Hunter Trap, Charge and Poison stun
columns), the opposing public roster, both relative spawn-pad banks, own
cooldown, stun and spawn shield, and wall rows. Masks are synthetic exact
masks in Core's ActionMask layout. Banks are empty or carry one permitted
teammate sighting. The callers ``act``, ``beta_act`` and ``alpha_act`` run the
jitted controllers with a fixed key. Team B observers use Scenario 1 slot 9 and
Team A observers slot 4; both sit in own-team row 4.
"""

from collections.abc import Callable, Sequence
from functools import cache
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from scripts.dev.visual_debugger.authoring_compiler import CompiledDevScenarioV1
from tests.scenario_controller_fixtures import load_scenario_1

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_WALL,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.reactive_tdm_alpha import reactive_tdm_alpha_policy
from marl_battlegrounds.policies.reactive_tdm_beta import reactive_tdm_beta_policy
from marl_battlegrounds.policies.reactive_tdm_gamma import reactive_tdm_gamma_policy
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    SharedObsSensorSourceBankV2,
    build_shared_obs_sensor_source_bank,
)

GAMMA = cast(SharedObsPolicy, jax.jit(reactive_tdm_gamma_policy))
BETA = cast(SharedObsPolicy, jax.jit(reactive_tdm_beta_policy))
ALPHA = cast(SharedObsPolicy, jax.jit(reactive_tdm_alpha_policy))
CANONICAL_CLASSES = (
    MAGE_CLASS_ID,
    WARRIOR_CLASS_ID,
    HUNTER_CLASS_ID,
    ROGUE_CLASS_ID,
    PRIEST_CLASS_ID,
)
BASIC_RADIUS = {
    MAGE_CLASS_ID: 3.0,
    WARRIOR_CLASS_ID: 1.5,
    HUNTER_CLASS_ID: 3.5,
    ROGUE_CLASS_ID: 1.5,
    PRIEST_CLASS_ID: 3.0,
}
OWN_PADS = tuple((19.5, 1.0 + 2.0 * row) for row in range(5))
OPPOSING_PADS = tuple((0.5, 1.0 + 2.0 * row) for row in range(5))


@cache
def scenario() -> CompiledDevScenarioV1:
    return load_scenario_1()


def _slot(tree: object, slot: int) -> object:
    def take(leaf: Array) -> Array:
        return leaf[slot]

    return jax.tree.map(take, tree)


def unit_row(
    template: Array,
    xy: tuple[float, float],
    *,
    class_id: int,
    hp: float = 40.0,
    max_hp: float = 100.0,
    radius: float = 0.5,
    speed: float = 1.0,
    trap_ticks: float = 0.0,
    charge_stun: float = 0.0,
    poison_stun: float = 0.0,
    cooldown: float = 0.0,
) -> Array:
    return (
        jnp.zeros_like(template)
        .at[AGENT_FEATURE_ACTIVE]
        .set(1)
        .at[AGENT_FEATURE_ALIVE]
        .set(1)
        .at[AGENT_FEATURE_X]
        .set(xy[0])
        .at[AGENT_FEATURE_Y]
        .set(xy[1])
        .at[AGENT_FEATURE_CLASS_ID]
        .set(class_id)
        .at[AGENT_FEATURE_CURRENT_HEALTH]
        .set(hp)
        .at[AGENT_FEATURE_MAX_HEALTH]
        .set(max_hp)
        .at[AGENT_FEATURE_RADIUS]
        .set(radius)
        .at[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        .set(speed)
        .at[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
        .set(BASIC_RADIUS.get(class_id, 0.0))
        .at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
        .set(trap_ticks)
        .at[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION]
        .set(charge_stun)
        .at[AGENT_FEATURE_STUN_ROGUE_POISON_DURATION]
        .set(poison_stun)
        .at[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING]
        .set(cooldown)
    )


def observation(
    class_id: int,
    *,
    team: str = "B",
    xy: tuple[float, float] = (10.0, 5.0),
    hp: float = 60.0,
    speed: float = 1.0,
    width: float = 20.0,
    height: float = 10.0,
) -> Observation:
    base = cast(Observation, _slot(scenario().observation, 9 if team == "B" else 4))
    own = unit_row(base.self_features, xy, class_id=class_id, hp=hp, speed=speed)
    return base._replace(
        self_features=own,
        ally_unit_features=jnp.zeros_like(base.ally_unit_features).at[4].set(own),
        enemy_unit_features=jnp.zeros_like(base.enemy_unit_features),
        ally_visibility_mask=jnp.array([False, False, False, False, True]),
        enemy_visibility_mask=jnp.zeros(5, dtype=jnp.bool_),
        map_obstacle_features=jnp.zeros_like(base.map_obstacle_features),
        context_features=base.context_features.at[CONTEXT_FEATURE_MAP_WIDTH]
        .set(width)
        .at[CONTEXT_FEATURE_MAP_HEIGHT]
        .set(height),
        spawn_lifecycle=base.spawn_lifecycle._replace(
            spawn_pad_positions_by_agent_by_team=jnp.array(
                (OWN_PADS, OPPOSING_PADS), dtype=jnp.float32
            ),
            active_mask_by_agent_by_team=jnp.ones((2, 5), dtype=jnp.bool_),
            class_ids_by_agent_by_team=jnp.array(
                (CANONICAL_CLASSES, CANONICAL_CLASSES), dtype=jnp.int32
            ),
            spawn_shield_actual_durations_by_agent_by_team=jnp.zeros(
                (2, 5), dtype=jnp.int32
            ),
        ),
    )


def enemy(
    obs: Observation,
    row: int,
    xy: tuple[float, float],
    *,
    class_id: int = WARRIOR_CLASS_ID,
    hp: float = 40.0,
    trap_ticks: float = 0.0,
    charge_stun: float = 0.0,
    poison_stun: float = 0.0,
    visible: bool = True,
    radius: float = 0.5,
) -> Observation:
    features = unit_row(
        obs.self_features,
        xy,
        class_id=class_id,
        hp=hp,
        radius=radius,
        trap_ticks=trap_ticks,
        charge_stun=charge_stun,
        poison_stun=poison_stun,
    )
    if not visible:
        features = jnp.zeros_like(features)
    return obs._replace(
        enemy_unit_features=obs.enemy_unit_features.at[row].set(features),
        enemy_visibility_mask=obs.enemy_visibility_mask.at[row].set(visible),
    )


def ally(
    obs: Observation,
    row: int,
    xy: tuple[float, float],
    *,
    class_id: int = WARRIOR_CLASS_ID,
    hp: float = 40.0,
    max_hp: float = 100.0,
    trap_ticks: float = 0.0,
    visible: bool = True,
) -> Observation:
    features = unit_row(
        obs.self_features,
        xy,
        class_id=class_id,
        hp=hp,
        max_hp=max_hp,
        trap_ticks=trap_ticks,
    )
    return obs._replace(
        ally_unit_features=obs.ally_unit_features.at[row].set(features),
        ally_visibility_mask=obs.ally_visibility_mask.at[row].set(visible),
    )


def own_state(
    obs: Observation,
    *,
    cooldown: float = 0.0,
    trap_ticks: float = 0.0,
    charge_stun: float = 0.0,
    poison_stun: float = 0.0,
    shield: int = 0,
    hp: float | None = None,
    active: bool = True,
    alive: bool = True,
) -> Observation:
    own = (
        obs.self_features.at[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING]
        .set(cooldown)
        .at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
        .set(trap_ticks)
        .at[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION]
        .set(charge_stun)
        .at[AGENT_FEATURE_STUN_ROGUE_POISON_DURATION]
        .set(poison_stun)
        .at[AGENT_FEATURE_ACTIVE]
        .set(float(active))
        .at[AGENT_FEATURE_ALIVE]
        .set(float(alive))
    )
    if hp is not None:
        own = own.at[AGENT_FEATURE_CURRENT_HEALTH].set(hp)
    lifecycle = obs.spawn_lifecycle
    shields = lifecycle.spawn_shield_actual_durations_by_agent_by_team.at[
        0, obs.self_ally_index
    ].set(shield)
    return obs._replace(
        self_features=own,
        ally_unit_features=obs.ally_unit_features.at[obs.self_ally_index].set(own),
        spawn_lifecycle=lifecycle._replace(
            spawn_shield_actual_durations_by_agent_by_team=shields
        ),
    )


def roster(obs: Observation, classes: Sequence[int | None]) -> Observation:
    # Classes for opposing roster rows 0..4; None marks an unused slot.
    active = jnp.array([value is not None for value in classes], dtype=jnp.bool_)
    ids = jnp.array([0 if value is None else value for value in classes], jnp.int32)
    lifecycle = obs.spawn_lifecycle
    return obs._replace(
        spawn_lifecycle=lifecycle._replace(
            active_mask_by_agent_by_team=lifecycle.active_mask_by_agent_by_team.at[
                1
            ].set(active),
            class_ids_by_agent_by_team=lifecycle.class_ids_by_agent_by_team.at[1].set(
                ids
            ),
        )
    )


def pads(
    obs: Observation,
    opposing: Sequence[tuple[float, float]],
    own: Sequence[tuple[float, float]] | None = None,
) -> Observation:
    lifecycle = obs.spawn_lifecycle
    banks = lifecycle.spawn_pad_positions_by_agent_by_team.at[1].set(
        jnp.array(opposing, dtype=jnp.float32)
    )
    if own is not None:
        banks = banks.at[0].set(jnp.array(own, dtype=jnp.float32))
    return obs._replace(
        spawn_lifecycle=lifecycle._replace(spawn_pad_positions_by_agent_by_team=banks)
    )


def wall(
    obs: Observation,
    index: int,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    theta: float = 0.0,
) -> Observation:
    row = (
        jnp.zeros_like(obs.map_obstacle_features[index])
        .at[OBSTACLE_FEATURE_ACTIVE]
        .set(1)
        .at[OBSTACLE_FEATURE_TYPE]
        .set(OBSTACLE_TYPE_WALL)
        .at[OBSTACLE_FEATURE_X]
        .set(xy[0])
        .at[OBSTACLE_FEATURE_Y]
        .set(xy[1])
        .at[OBSTACLE_FEATURE_WIDTH]
        .set(width)
        .at[OBSTACLE_FEATURE_HEIGHT]
        .set(height)
        .at[OBSTACLE_FEATURE_THETA]
        .set(theta)
        .at[OBSTACLE_FEATURE_RADIUS]
        .set(0)
    )
    return obs._replace(
        map_obstacle_features=obs.map_obstacle_features.at[index].set(row)
    )


def mask(
    *pairs: tuple[int, int],
    moves: Sequence[int] | None = None,
) -> ActionMask:
    # (Target None, no Ultimate) plus the given pairs; moves=None admits all nine
    # moves, otherwise Stay plus the listed moves.
    joint = jnp.zeros((11, 2), dtype=jnp.bool_).at[0, 0].set(True)
    for target, ultimate in pairs:
        joint = joint.at[target, ultimate].set(True)
    move_mask = jnp.ones(9, dtype=jnp.bool_)
    if moves is not None:
        move_mask = jnp.zeros(9, dtype=jnp.bool_).at[0].set(True)
        for move in moves:
            move_mask = move_mask.at[move].set(True)
    return ActionMask(move_mask, joint.any(axis=1), joint.any(axis=0), joint)


@cache
def empty_bank() -> SharedObsSensorSourceBankV2:
    def empty_team(leaf: Array) -> Array:
        return jnp.zeros_like(leaf[1])

    bank = build_shared_obs_sensor_source_bank(scenario().observation)
    return jax.tree.map(empty_team, bank)


def sighting_bank(
    source: int, candidate: int, features: Array
) -> SharedObsSensorSourceBankV2:
    # One teammate source sees one candidate (0..4 allies, 5..9 enemies).
    bank = empty_bank()
    return bank._replace(
        unit_features_by_source_and_candidate=bank.unit_features_by_source_and_candidate.at[
            source, candidate
        ].set(features),
        unit_visibility_by_source_and_candidate=bank.unit_visibility_by_source_and_candidate.at[
            source, candidate
        ].set(True),
    )


def act(
    obs: Observation,
    action_mask: ActionMask | None = None,
    *,
    bank: SharedObsSensorSourceBankV2 | None = None,
    availability: Array | None = None,
    key: int = 0,
    policy: Callable[..., ActorAction] = GAMMA,
) -> ActorAction:
    return policy(
        obs,
        action_mask if action_mask is not None else mask(),
        jax.random.key(key),
        bank if bank is not None else empty_bank(),
        availability if availability is not None else jnp.zeros(5, dtype=jnp.bool_),
    )


def beta_act(obs: Observation, action_mask: ActionMask | None = None) -> ActorAction:
    return act(obs, action_mask, policy=BETA)


def alpha_act(obs: Observation, action_mask: ActionMask | None = None) -> ActorAction:
    return act(obs, action_mask, policy=ALPHA)


def as_ints(action: ActorAction) -> tuple[int, int, int]:
    return int(action.move), int(action.select_target), int(action.use_ultimate)
