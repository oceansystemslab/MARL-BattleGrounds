"""Check TDM-GAMMA version 2 through real Core steps, routes and old identities.

Most scenes are small compiled dev scenarios run through Core's real step.
GAMMA plays Team B; Team A units follow fixed scripted actions. Every
submitted action is accepted unchanged, and every Trap reading also checks
that the enemy row is visible. The class-order check and the kept-rule checks
call the controllers directly on edited Scenario 1 observations from
tests.tdm_gamma_fixtures. Where a version 2 rule decides the answer, the test
also runs BETA on the same actor inputs and shows that BETA chooses
differently, so BETA's code would fail it.

- Trap order: a GAMMA Hunter Traps an enemy Priest and stays still, although
  a Warrior is nearer (inside ALPHA's 2-unit Trap distance), has lower HP and
  is also legal to Trap. BETA Traps that Warrior. Without a Priest, the Mage
  comes before a nearer, weaker Rogue that ALPHA and BETA Trap: ALPHA's
  "nearest enemy within 2 units" Trap rule is gone.
- Trap hold: the next five observations show Trap ticks 4, 3, 2, 1, 0 on the
  visible Priest row. At 4, 3 and 2 the GAMMA Hunter and Warrior could legally
  hit the Priest but choose no damage on it, where BETA's Warrior would hit
  it, and the Priest's HP stays still. At 1 the GAMMA Warrior hits it. Movement stays
  BETA's: the Hunter backs away from the trapped Priest when it is the nearest
  enemy. A fresh scene built from the state after the cast gives the same
  actions, so nothing is remembered.
- An ALPHA Warrior on GAMMA's team (a mixed team) hits an enemy showing 4
  Trap ticks, while both GAMMA Hunters shoot someone else and BETA's would hit
  it. The next observation shows 0 with the enemy visible; GAMMA hits it
  again and a second, now-ready Hunter Traps it again where BETA would not.
- A Trap and a Warrior Charge stun on one enemy: GAMMA reads only the Hunter
  Trap column. A second ready Hunter never Traps before natural expiry; it
  holds its band at 1 tick, walks in at 0 (where BETA holds still) and Traps
  on the next tick. Two Hunters trapping on one tick still show 4.
- Search: a GAMMA Mage and Rogue with no enemy in view walk toward the mean
  of the enemy spawn pads over several real steps, closer every step, where
  BETA heads for the map centre. Seeing and losing an enemy switches between
  BETA's chase and that search, with no remembered sighting.
- Real Core masks: a trapped GAMMA unit submits Stay and no combat. A unit
  with a spawn shield after a real respawn moves toward its goal at the
  configured shield speed, with no combat. Dead and unused slots submit the
  no-op.
- Eager, jit, team and two-game vmap execution agree. Hidden enemy rows and
  unavailable sources never change an action.
- Rules GAMMA keeps: the Priest map-centre move and the alive-and-active
  guard match BETA and ALPHA, and the Rogue nearest-enemy fallback move
  matches BETA. The ALPHA and BETA descriptor digests keep their 59c157c
  values.
"""

from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scripts.dev.visual_debugger.authoring_compiler import (
    CompiledDevScenarioV1,
    canonicalize_inactive_rows,
    compile_dev_scenario,
)
from scripts.dev.visual_debugger.authoring_models import (
    DevPointV1,
    DevWallV1,
    new_scenario_draft,
)
from tests.tdm_gamma_fixtures import (
    BETA,
    GAMMA,
    act,
    ally,
    alpha_act,
    as_ints,
    beta_act,
    enemy,
    mask,
    observation,
    own_state,
    scenario,
    unit_row,
)

from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
)
from marl_battlegrounds.core.env import step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MOVE_EAST,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.recording_identity import canonical_digest_sha256
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.reactive_common import (
    _body_aware_move,  # pyright: ignore[reportPrivateUsage]
    centers,
    refine_movement,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    HUNTER_CLOSE_DISTANCE,
    HUNTER_FAR_DISTANCE,
    HUNTER_TRAP_DISTANCE,
    reactive_tdm_alpha_controller_descriptor,
    reactive_tdm_alpha_policy,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
)
from marl_battlegrounds.policies.reactive_tdm_gamma import reactive_tdm_gamma_policy
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV2,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
    mask_source_bank_for_recipient,
)

_TEAM = cast(
    Callable[..., ActorAction],
    jax.jit(execute_shared_obs_team_policy, static_argnums=(5, 6)),
)
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)
_REFINE = cast(Callable[..., Array], jax.jit(refine_movement))
_BODY = cast(Callable[..., Array], jax.jit(_body_aware_move))
_FULL_HEALTH = {
    "mage": 80.0,
    "warrior": 200.0,
    "hunter": 100.0,
    "rogue": 100.0,
    "priest": 100.0,
}
_NO_OP = (MOVE_STAY, 0, 0)


class _Unit(NamedTuple):
    class_name: str
    xy: tuple[float, float]
    hp: float | None = None
    cooldown: int = 0
    trap_ticks: int = 0


class _Trajectory(NamedTuple):
    scenario: CompiledDevScenarioV1
    states: list[EnvState]
    observations: list[Observation]
    masks: list[ActionMask]
    joints: list[Action]


def _take[T](tree: T, index: int) -> T:
    def take(leaf: Array) -> Array:
        return leaf[index]

    return cast(T, jax.tree.map(take, tree))


def _compile(
    team_a: Sequence[_Unit],
    team_b: Sequence[_Unit],
    *,
    walls: Sequence[tuple[float, float, float, float]] = (),
    episode: Mapping[str, object] | None = None,
    global_state: Mapping[str, object] | None = None,
) -> CompiledDevScenarioV1:
    # Controller behaviour, not scoring: keep the original one-point rule.
    content = new_scenario_draft("gamma_trajectory", red_zone_depth=0.0).content
    rosters = list(content.roster)
    states = list(content.agent_states)
    for first_slot, units in ((0, team_a), (5, team_b)):
        for index, unit in enumerate(units):
            slot = first_slot + index
            rosters[slot] = rosters[slot].model_copy(
                update={"class_name": unit.class_name}
            )
            states[slot] = states[slot].model_copy(
                update={
                    "position": DevPointV1(x=unit.xy[0], y=unit.xy[1]),
                    "current_health": _FULL_HEALTH[unit.class_name]
                    if unit.hp is None
                    else unit.hp,
                    "ultimate_cooldown_remaining": unit.cooldown,
                    "hunter_trap_stun_duration": unit.trap_ticks,
                }
            )
    obstacles = tuple(
        DevWallV1(
            object_id=f"wall-{index}",
            center_x=x,
            center_y=y,
            width=width,
            height=height,
        )
        for index, (x, y, width, height) in enumerate(walls)
    )
    content = content.model_copy(
        update={
            "team_a_size": len(team_a),
            "team_b_size": len(team_b),
            "roster": tuple(rosters),
            "agent_states": tuple(states),
            "embedded_map": content.embedded_map.model_copy(
                update={"obstacles": obstacles}
            ),
            "episode": content.episode.model_copy(update=dict(episode or {})),
            "global_state": content.global_state.model_copy(
                update=dict(global_state or {})
            ),
        }
    )
    return compile_dev_scenario(canonicalize_inactive_rows(content))


def _availability(compiled: CompiledDevScenarioV1) -> Array:
    profile = compiled.config.agent_profile
    return build_default_shared_obs_information_availability(
        profile.active_mask, profile.team_ids
    )


def _team_b(
    compiled: CompiledDevScenarioV1,
    obs: Observation,
    action_mask: ActionMask,
    policy: Callable[..., ActorAction],
    key: int = 0,
) -> ActorAction:
    return _TEAM(
        obs,
        action_mask,
        jax.random.split(jax.random.key(key), 10),
        build_shared_obs_sensor_source_bank(obs),
        _availability(compiled),
        policy,
        TEAM_B_ID,
    )


def _run(
    compiled: CompiledDevScenarioV1,
    ticks: int,
    *,
    team_a_first_slot: Mapping[int, tuple[int, int, int]] | None = None,
    alpha_slots: Sequence[int] = (),
) -> _Trajectory:
    # Team A slot 0 follows the per-tick script; other Team A slots do nothing.
    script = team_a_first_slot or {}
    state, obs, action_mask = (
        compiled.initial_state,
        compiled.observation,
        compiled.action_mask,
    )
    trajectory = _Trajectory(compiled, [state], [obs], [action_mask], [])
    for tick in range(ticks):
        team_b = _team_b(compiled, obs, action_mask, reactive_tdm_gamma_policy, tick)
        if alpha_slots:
            alpha = _team_b(compiled, obs, action_mask, reactive_tdm_alpha_policy, tick)
            chosen = jnp.isin(jnp.arange(5), jnp.asarray(alpha_slots))
            team_b = ActorAction(
                *(jnp.where(chosen, x, y) for x, y in zip(alpha, team_b, strict=True))
            )
        move, target, ultimate = script.get(tick, _NO_OP)
        zeros = jnp.zeros(5, dtype=jnp.int32)
        team_a = ActorAction(
            zeros.at[0].set(move), zeros.at[0].set(target), zeros.at[0].set(ultimate)
        )
        joint = build_joint_action_from_actor_actions(team_a, team_b)
        state, obs, _, _, action_mask, info = _STEP(
            compiled.config,
            state,
            action_mask,
            joint,
            jax.random.key(tick),
        )
        accepted = info.transition_facts.action_acceptance_facts.accepted_joint_action
        for submitted, actual in zip(joint, accepted, strict=True):
            np.testing.assert_array_equal(actual, submitted)
        trajectory.states.append(state)
        trajectory.observations.append(obs)
        trajectory.masks.append(action_mask)
        trajectory.joints.append(joint)
    return trajectory


def _action(trajectory: _Trajectory, tick: int, slot: int) -> tuple[int, int, int]:
    joint = trajectory.joints[tick]
    return (
        int(joint.move[slot]),
        int(joint.select_target[slot]),
        int(joint.use_ultimate[slot]),
    )


def _enemy_status(obs: Observation, slot: int, row: int) -> tuple[bool, float]:
    # Visibility and observed Trap ticks of one enemy row, seen by one slot.
    features = obs.enemy_unit_features[slot, row]
    return (
        bool(obs.enemy_visibility_mask[slot, row]),
        float(features[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]),
    )


def _basic_legal(action_mask: ActionMask, slot: int, target: int) -> bool:
    return bool(action_mask.select_target_use_ultimate_joint_mask[slot, target, 0])


def _trap_legal(action_mask: ActionMask, slot: int, target: int) -> bool:
    return bool(action_mask.select_target_use_ultimate_joint_mask[slot, target, 1])


def _distance(obs: Observation, slot: int, row: int) -> float:
    own = centers(obs.self_features[slot])
    offset = centers(obs.enemy_unit_features[slot, row]) - own
    return float(jnp.sqrt(jnp.sum(jnp.square(offset))))


def _actor(
    compiled: CompiledDevScenarioV1,
    obs: Observation,
    action_mask: ActionMask,
    slot: int,
) -> tuple[Observation, ActionMask, SharedObsSensorSourceBankV2, Array]:
    # One Team B actor's inputs exactly as the team executor delivers them.
    availability = _availability(compiled)[slot, 5:]
    bank = mask_source_bank_for_recipient(
        _take(build_shared_obs_sensor_source_bank(obs), 1), availability
    )
    return _take(obs, slot), _take(action_mask, slot), bank, availability


def _actor_at(
    trajectory: _Trajectory, tick: int, slot: int
) -> tuple[Observation, ActionMask, SharedObsSensorSourceBankV2, Array]:
    return _actor(
        trajectory.scenario,
        trajectory.observations[tick],
        trajectory.masks[tick],
        slot,
    )


def _beta_at(trajectory: _Trajectory, tick: int, slot: int) -> tuple[int, int, int]:
    # BETA's choice from the exact inputs GAMMA received at this tick.
    obs, action_mask, bank, availability = _actor_at(trajectory, tick, slot)
    return as_ints(
        act(obs, action_mask, bank=bank, availability=availability, policy=BETA)
    )


def _pad_mean(obs: Observation) -> Array:
    return jnp.mean(obs.spawn_lifecycle.spawn_pad_positions_by_agent_by_team[1], axis=0)


def _search_move(obs: Observation, action_mask: ActionMask) -> int:
    intent = _pad_mean(obs) - centers(obs.self_features)
    return int(_REFINE(obs, action_mask, intent, approach=True))


@pytest.fixture(scope="module")
def priest_trap() -> _Trajectory:
    # Team A: a Warrior (row 0, target 6) and a Priest (row 1, target 7).
    # GAMMA: a Hunter (slot 5) and a Warrior (slot 6) east of the Priest.
    return _run(
        _compile(
            [_Unit("warrior", (8.2, 5.0), hp=90.0), _Unit("priest", (12.6, 5.0))],
            [_Unit("hunter", (10.0, 5.0)), _Unit("warrior", (15.0, 5.0))],
        ),
        5,
    )


def test_hunter_traps_the_priest_first_where_beta_traps_the_nearer_warrior(
    priest_trap: _Trajectory,
) -> None:
    first, first_mask = priest_trap.observations[0], priest_trap.masks[0]
    assert _trap_legal(first_mask, 5, 6) and _trap_legal(first_mask, 5, 7)
    # The Warrior is nearer (inside ALPHA's Trap distance) and has lower HP.
    assert _distance(first, 5, 0) <= HUNTER_TRAP_DISTANCE < _distance(first, 5, 1)
    health = first.enemy_unit_features[5, :2, AGENT_FEATURE_CURRENT_HEALTH]
    assert float(health[0]) < float(health[1])
    assert _action(priest_trap, 0, 5) == (MOVE_STAY, 7, 1)
    assert _beta_at(priest_trap, 0, 5)[1:] == (6, 1)
    after = priest_trap.observations[1]
    assert _enemy_status(after, 5, 1) == (True, 4.0)
    assert _enemy_status(after, 5, 0) == (True, 0.0)


def test_trap_counts_down_and_gamma_damage_waits_for_one_tick(
    priest_trap: _Trajectory,
) -> None:
    for slot in (5, 6):
        observed = [
            _enemy_status(priest_trap.observations[t], slot, 1) for t in range(1, 6)
        ]
        assert observed == [
            (True, 4.0),
            (True, 3.0),
            (True, 2.0),
            (True, 1.0),
            (True, 0.0),
        ]
    for tick in (1, 2, 3):
        for slot in (5, 6):
            # A legal Basic exists, so holding fire is GAMMA's choice, not Core's.
            assert _basic_legal(priest_trap.masks[tick], slot, 7)
            assert _action(priest_trap, tick, slot)[1] != 7
        assert _beta_at(priest_trap, tick, 6)[1] == 7
    assert _action(priest_trap, 4, 6)[1:] == (7, 0)
    health = [float(state.current_health[1]) for state in priest_trap.states[1:6]]
    assert health[0] == health[1] == health[2] == health[3] > health[4]


def test_movement_stays_betas_and_counts_the_trapped_priest(
    priest_trap: _Trajectory,
) -> None:
    for tick in range(1, 5):
        for slot in (5, 6):
            gamma_move = _action(priest_trap, tick, slot)[0]
            assert gamma_move == _beta_at(priest_trap, tick, slot)[0]
    # At tick 2 the Priest, trapped for 3 more ticks, is the Hunter's nearest
    # enemy. Backing away from it and backing away from the Warrior differ.
    tick_2 = priest_trap.observations[2]
    assert _enemy_status(tick_2, 5, 1) == (True, 3.0)
    assert _distance(tick_2, 5, 1) < _distance(tick_2, 5, 0) <= HUNTER_CLOSE_DISTANCE
    obs, action_mask, _, _ = _actor_at(priest_trap, 2, 5)
    own = centers(obs.self_features)
    away = [
        int(_REFINE(obs, action_mask, own - centers(row), approach=False))
        for row in obs.enemy_unit_features[:2]
    ]
    assert away[0] != away[1]
    assert _action(priest_trap, 2, 5)[0] == away[1]


def test_same_state_in_a_fresh_scene_gives_the_same_actions(
    priest_trap: _Trajectory,
) -> None:
    state, obs = priest_trap.states[1], priest_trap.observations[1]

    def xy(slot: int) -> tuple[float, float]:
        position = state.agent_positions[slot]
        return float(position[0]), float(position[1])

    def hp(slot: int) -> float:
        return float(state.current_health[slot])

    cooldown = int(obs.self_features[5, AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING])
    # A fresh scene authored with tick 1's state never saw the cast.
    twin = _compile(
        [
            _Unit("warrior", xy(0), hp=hp(0)),
            _Unit("priest", xy(1), hp=hp(1), trap_ticks=4),
        ],
        [_Unit("hunter", xy(5), cooldown=cooldown), _Unit("warrior", xy(6))],
    )
    assert _enemy_status(twin.observation, 5, 1) == (True, 4.0)
    fresh = _run(twin, 1)
    for slot in (5, 6):
        assert _action(fresh, 0, slot) == _action(priest_trap, 1, slot)


def test_ally_damage_breaks_the_trap_and_gamma_uses_the_enemy_again() -> None:
    trajectory = _run(
        _compile(
            [
                _Unit("warrior", (8.0, 5.0), hp=100.0),
                _Unit("warrior", (12.7, 7.6), hp=150.0),
            ],
            [
                _Unit("hunter", (10.9, 5.0)),
                _Unit("warrior", (7.0, 5.0)),
                _Unit("hunter", (9.6, 7.9), cooldown=2),
            ],
        ),
        3,
        alpha_slots=(1,),
    )
    assert _action(trajectory, 0, 5) == (MOVE_STAY, 6, 1)
    assert _enemy_status(trajectory.observations[1], 5, 0) == (True, 4.0)
    # The ALPHA Warrior hits the enemy at 4 ticks. Both GAMMA Hunters could hit
    # it too, and BETA's would, but GAMMA's shoot the other Warrior.
    assert _action(trajectory, 1, 6)[1:] == (6, 0)
    for slot in (5, 7):
        assert _basic_legal(trajectory.masks[1], slot, 6)
        assert _action(trajectory, 1, slot)[1] == 7
        assert _beta_at(trajectory, 1, slot)[1] == 6

    after = trajectory.observations[2]
    assert _enemy_status(after, 5, 0) == (True, 0.0)
    assert _enemy_status(after, 5, 1) == (True, 0.0)
    after_mask = trajectory.masks[2]
    # The other Warrior is also a legal Basic, so choosing row 0 shows that the
    # broken-Trap enemy is back in the damage pool.
    assert _basic_legal(after_mask, 5, 6) and _basic_legal(after_mask, 5, 7)
    assert _action(trajectory, 2, 5)[1:] == (6, 0)
    assert _action(trajectory, 2, 5) == _beta_at(trajectory, 2, 5)
    # New Trap: the second Hunter is ready now and traps the same enemy. It is
    # beyond ALPHA's 2-unit Trap distance, so BETA would not Trap.
    assert float(after.self_features[7, AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING]) == 0
    assert _distance(after, 7, 0) > HUNTER_TRAP_DISTANCE
    assert _action(trajectory, 2, 7) == (MOVE_STAY, 6, 1)
    assert _beta_at(trajectory, 2, 7)[2] == 0
    assert _enemy_status(trajectory.observations[3], 5, 0) == (True, 4.0)


@pytest.mark.parametrize(
    ("hunter_xy", "hunter_cooldown", "trap_ticks", "hunter_combat", "warrior_target"),
    [((10.5, 5.0), 0, 4.0, (0, 0), 0), ((11.2, 5.0), 1, 0.0, (6, 1), 6)],
    ids=["trap_with_charge_stun", "charge_stun_only"],
)
def test_charge_stun_is_never_read_as_a_trap(
    hunter_xy: tuple[float, float],
    hunter_cooldown: int,
    trap_ticks: float,
    hunter_combat: tuple[int, int],
    warrior_target: int,
) -> None:
    trajectory = _run(
        _compile(
            [_Unit("warrior", (8.0, 5.0), hp=39.0)],
            [
                _Unit("hunter", hunter_xy, cooldown=hunter_cooldown),
                _Unit("warrior", (6.0, 7.0)),
            ],
        ),
        2,
    )
    assert _action(trajectory, 0, 6)[1:] == (6, 1)
    after = trajectory.observations[1]
    charge = float(
        after.enemy_unit_features[5, 0, AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION]
    )
    assert charge == 1
    assert _enemy_status(after, 5, 0) == (True, trap_ticks)
    assert _basic_legal(trajectory.masks[1], 5, 6)
    assert _basic_legal(trajectory.masks[1], 6, 6)
    assert _action(trajectory, 1, 5)[1:] == hunter_combat
    assert _action(trajectory, 1, 6)[1:] == (warrior_target, 0)


def test_second_ready_hunter_traps_again_only_after_natural_expiry() -> None:
    trajectory = _run(
        _compile(
            [_Unit("warrior", (8.0, 5.0))],
            [_Unit("hunter", (10.5, 5.0)), _Unit("hunter", (8.0, 7.5), cooldown=1)],
        ),
        7,
    )
    assert _action(trajectory, 0, 5) == (MOVE_STAY, 6, 1)
    observed = [_enemy_status(trajectory.observations[t], 6, 0) for t in range(1, 6)]
    assert observed == [(True, 4.0), (True, 3.0), (True, 2.0), (True, 1.0), (True, 0.0)]
    for tick in range(1, 6):
        ready = trajectory.observations[tick].self_features[6]
        assert float(ready[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING]) == 0
    assert [_action(trajectory, t, 6)[2] for t in range(1, 6)] == [0, 0, 0, 0, 0]
    # At 1 tick the enemy is not in the new-Trap pool, so the Hunter holds its
    # band; at 0 the same band position turns into a Trap approach, where BETA
    # keeps holding.
    for tick in (4, 5):
        distance = _distance(trajectory.observations[tick], 6, 0)
        assert HUNTER_CLOSE_DISTANCE < distance <= HUNTER_FAR_DISTANCE
        assert _basic_legal(trajectory.masks[tick], 6, 6)
    assert _action(trajectory, 4, 6) == (MOVE_STAY, 6, 0)
    assert _action(trajectory, 5, 6)[0] == MOVE_SOUTH
    assert _beta_at(trajectory, 5, 6) == (MOVE_STAY, 6, 0)
    assert _action(trajectory, 6, 6) == (MOVE_STAY, 6, 1)
    assert _enemy_status(trajectory.observations[7], 6, 0) == (True, 4.0)


def test_two_hunters_trapping_on_one_tick_show_four() -> None:
    trajectory = _run(
        _compile(
            [_Unit("warrior", (8.0, 5.0))],
            [_Unit("hunter", (10.5, 5.0)), _Unit("hunter", (8.0, 7.5))],
        ),
        1,
    )
    assert _action(trajectory, 0, 5) == (MOVE_STAY, 6, 1)
    assert _action(trajectory, 0, 6) == (MOVE_STAY, 6, 1)
    assert _enemy_status(trajectory.observations[1], 5, 0) == (True, 4.0)


def test_alphas_nearest_two_unit_trap_is_gone() -> None:
    obs = observation(HUNTER_CLASS_ID)
    obs = enemy(obs, 0, (8.2, 5.0), class_id=MAGE_CLASS_ID, hp=70.0)
    obs = enemy(obs, 1, (10.0, 6.2), class_id=ROGUE_CLASS_ID, hp=50.0)
    legal = mask((6, 0), (6, 1), (7, 0), (7, 1))
    # The Rogue is nearer, inside ALPHA's Trap distance, and has lower HP.
    rows = obs.enemy_unit_features[:2]
    offsets = centers(rows) - centers(obs.self_features)
    distances = jnp.sqrt(jnp.sum(jnp.square(offsets), axis=-1))
    assert float(distances[1]) < float(distances[0]) <= HUNTER_TRAP_DISTANCE
    assert float(rows[1, AGENT_FEATURE_CURRENT_HEALTH]) < float(
        rows[0, AGENT_FEATURE_CURRENT_HEALTH]
    )
    assert as_ints(alpha_act(obs, legal))[1:] == (7, 1)
    beta = as_ints(beta_act(obs, legal))
    assert beta[1:] == (7, 1) and beta[0] != MOVE_STAY
    assert as_ints(act(obs, legal)) == (MOVE_STAY, 6, 1)


def test_search_walks_toward_the_enemy_pad_mean_over_several_steps() -> None:
    trajectory = _run(
        _compile(
            [_Unit("warrior", (19.0, 5.0))],
            [_Unit("mage", (12.0, 8.5)), _Unit("rogue", (12.0, 1.5))],
        ),
        4,
    )
    for slot in (5, 6):
        distances: list[float] = []
        for tick in range(5):
            obs, action_mask, _, _ = _actor_at(trajectory, tick, slot)
            assert not bool(jnp.any(obs.enemy_visibility_mask))
            offset = _pad_mean(obs) - centers(obs.self_features)
            distances.append(float(jnp.sqrt(jnp.sum(jnp.square(offset)))))
            if tick < 4:
                move = _action(trajectory, tick, slot)
                assert move == (_search_move(obs, action_mask), 0, 0)
                # BETA heads for the map centre, a different move at every tick.
                assert _beta_at(trajectory, tick, slot)[0] != move[0]
        assert all(later < earlier for earlier, later in pairwise(distances))


def test_seeing_and_losing_an_enemy_switches_chase_and_search() -> None:
    trajectory = _run(
        _compile(
            [_Unit("warrior", (11.5, 8.0))],
            [_Unit("warrior", (8.0, 4.0))],
            walls=((8.0, 7.0, 4.0, 0.4),),
        ),
        3,
        team_a_first_slot={0: (MOVE_WEST, 0, 0), 1: (MOVE_EAST, 0, 0)},
    )
    seen = [_enemy_status(trajectory.observations[t], 5, 0)[0] for t in range(3)]
    assert seen == [True, False, True]
    last_seen = centers(trajectory.observations[0].enemy_unit_features[5, 0])
    for tick in range(3):
        obs, action_mask, _, _ = _actor_at(trajectory, tick, 5)
        move = _action(trajectory, tick, 5)[0]
        search = _search_move(obs, action_mask)
        if seen[tick]:
            last_seen = centers(obs.enemy_unit_features[0])
            assert move == _beta_at(trajectory, tick, 5)[0] != search
            continue
        np.testing.assert_array_equal(obs.enemy_unit_features[0], 0.0)
        # While hidden, chasing the last sighting would mean remembering it,
        # and BETA would head for the map centre.
        intent = last_seen - centers(obs.self_features)
        remembered = int(_REFINE(obs, action_mask, intent, approach=True))
        assert move == search
        assert search != remembered
        assert search != _beta_at(trajectory, tick, 5)[0]


def test_trapped_gamma_unit_submits_stay_and_no_combat() -> None:
    trajectory = _run(
        _compile([_Unit("hunter", (8.0, 5.0))], [_Unit("warrior", (10.0, 5.0))]),
        2,
        team_a_first_slot={0: (MOVE_STAY, 6, 1)},
    )
    after = trajectory.observations[1]
    assert float(after.self_features[5, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]) == 4
    # The Hunter is visible and inside Basic range; only Core's mask stops a hit.
    assert bool(after.enemy_visibility_mask[5, 0])
    radius = float(after.self_features[5, AGENT_FEATURE_BASIC_INTERACTION_RADIUS])
    assert _distance(after, 5, 0) <= radius
    stunned = trajectory.masks[1]
    np.testing.assert_array_equal(stunned.move_mask[5], jnp.arange(9) == MOVE_STAY)
    assert np.argwhere(
        np.asarray(stunned.select_target_use_ultimate_joint_mask[5])
    ).tolist() == [[0, 0]]
    assert _action(trajectory, 1, 5) == _NO_OP


@pytest.fixture(scope="module")
def respawn() -> _Trajectory:
    return _run(
        _compile(
            [_Unit("mage", (2.0, 5.0))],
            [_Unit("warrior", (4.0, 5.0), hp=5.0, cooldown=20)],
            episode={
                "spawn_shield_duration_steps": 3,
                "spawn_shield_movement_speed": 1.5,
                "team_b_respawn_wave_period_steps": 2,
            },
            global_state={"team_b_respawn_countdown": 1},
        ),
        3,
        team_a_first_slot={0: (MOVE_STAY, 6, 0)},
    )


def test_shielded_respawned_unit_moves_at_shield_speed_without_combat(
    respawn: _Trajectory,
) -> None:
    assert not bool(respawn.states[1].alive_mask[5])
    assert bool(respawn.states[2].alive_mask[5])
    obs, action_mask, _, _ = _actor_at(respawn, 2, 5)
    shields = obs.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team
    assert int(shields[0, obs.self_ally_index]) == 3
    own = obs.self_features
    for column in (
        AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
        AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
        AGENT_FEATURE_STUN_ROGUE_POISON_DURATION,
    ):
        assert float(own[column]) == 0
    assert bool(jnp.all(action_mask.move_mask))
    legal = np.argwhere(np.asarray(action_mask.select_target_use_ultimate_joint_mask))
    assert legal.tolist() == [[0, 0]]
    move, target, ultimate = _action(respawn, 2, 5)
    assert move == _search_move(obs, action_mask) != MOVE_STAY
    assert (target, ultimate) == (0, 0)
    step_taken = (
        respawn.states[3].agent_positions[5] - respawn.states[2].agent_positions[5]
    )
    np.testing.assert_allclose(
        step_taken,
        1.5 * UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY[move],
        atol=1e-5,
    )


def test_dead_and_unused_slots_submit_the_no_op(respawn: _Trajectory) -> None:
    dead = respawn.masks[1]
    np.testing.assert_array_equal(dead.move_mask[5], jnp.arange(9) == MOVE_STAY)
    legal = np.argwhere(np.asarray(dead.select_target_use_ultimate_joint_mask[5]))
    assert legal.tolist() == [[0, 0]]
    assert _action(respawn, 1, 5) == _NO_OP
    for tick in range(3):
        for slot in range(6, 10):
            assert _action(respawn, tick, slot) == _NO_OP


def test_eager_jit_team_and_batched_execution_agree() -> None:
    games = [
        scenario(),
        _compile(
            [_Unit("warrior", (8.0, 5.0))],
            [_Unit("hunter", (10.5, 5.0)), _Unit("warrior", (10.0, 7.0))],
        ),
    ]
    keys = jax.random.split(jax.random.key(0), 10)
    teams: list[ActorAction] = []
    for game, compiled in enumerate(games):
        obs, action_mask = compiled.observation, compiled.action_mask
        team = _team_b(compiled, obs, action_mask, reactive_tdm_gamma_policy)
        teams.append(team)
        for slot in range(5, 10):
            inputs = _actor(compiled, obs, action_mask, slot)
            jitted = GAMMA(inputs[0], inputs[1], keys[slot], *inputs[2:])
            expected = tuple(int(field[slot - 5]) for field in team)
            assert as_ints(jitted) == expected
            # Eager calls are slow, so only the canonical 5v5 game runs them.
            if game == 0:
                eager = reactive_tdm_gamma_policy(
                    inputs[0], inputs[1], keys[slot], *inputs[2:]
                )
                assert as_ints(eager) == expected

    def stack(*leaves: Array) -> Array:
        return jnp.stack(leaves)

    def run_team(
        obs: Observation,
        action_mask: ActionMask,
        game_keys: Array,
        bank: SharedObsSensorSourceBankV2,
        availability: Array,
    ) -> ActorAction:
        return _TEAM(
            obs,
            action_mask,
            game_keys,
            bank,
            availability,
            reactive_tdm_gamma_policy,
            TEAM_B_ID,
        )

    batched_team = cast(Callable[..., ActorAction], jax.jit(jax.vmap(run_team)))
    batched = batched_team(
        jax.tree.map(stack, *(compiled.observation for compiled in games)),
        jax.tree.map(stack, *(compiled.action_mask for compiled in games)),
        jnp.stack([keys, keys]),
        jax.tree.map(
            stack,
            *(build_shared_obs_sensor_source_bank(c.observation) for c in games),
        ),
        jnp.stack([_availability(compiled) for compiled in games]),
    )
    for game, team in enumerate(teams):
        for batched_field, field in zip(batched, team, strict=True):
            np.testing.assert_array_equal(batched_field[game], field)


def test_hidden_rows_and_unavailable_sources_never_change_actions() -> None:
    compiled = scenario()
    obs, action_mask, _, _ = _actor(
        compiled, compiled.observation, compiled.action_mask, 5
    )
    raw_bank = _take(build_shared_obs_sensor_source_bank(compiled.observation), 1)
    hidden = [row for row in range(5) if not bool(obs.enemy_visibility_mask[row])]
    assert hidden
    # A lure: a 1-HP enemy Priest just north of the actor. Seen, it changes GAMMA.
    own_xy = centers(obs.self_features)
    lure = unit_row(
        obs.self_features,
        (float(own_xy[0]), float(own_xy[1]) + 0.8),
        class_id=PRIEST_CLASS_ID,
        hp=1.0,
    )
    only_source = jnp.zeros(5, dtype=jnp.bool_).at[3].set(True)

    def gamma(
        observed: Observation, bank: SharedObsSensorSourceBankV2
    ) -> tuple[int, int, int]:
        return as_ints(act(observed, action_mask, bank=bank, availability=only_source))

    base = gamma(obs, raw_bank)
    lured_rows = obs.enemy_unit_features.at[jnp.asarray(hidden)].set(lure)
    assert gamma(obs._replace(enemy_unit_features=lured_rows), raw_bank) == base
    seen_lure = obs._replace(
        enemy_unit_features=lured_rows,
        enemy_visibility_mask=obs.enemy_visibility_mask.at[hidden[0]].set(True),
    )
    assert gamma(seen_lure, raw_bank) != base

    def lure_sources(sources: list[int]) -> SharedObsSensorSourceBankV2:
        features = raw_bank.unit_features_by_source_and_candidate
        visibility = raw_bank.unit_visibility_by_source_and_candidate
        for source in sources:
            features = features.at[source, 5 + hidden[0]].set(lure)
            visibility = visibility.at[source, 5 + hidden[0]].set(True)
        return raw_bank._replace(
            unit_features_by_source_and_candidate=features,
            unit_visibility_by_source_and_candidate=visibility,
        )

    assert gamma(obs, lure_sources([0, 1, 2, 4])) == base
    assert gamma(obs, lure_sources([3])) != base


def test_priest_map_centre_move_matches_beta_and_alpha() -> None:
    obs = observation(PRIEST_CLASS_ID, xy=(3.0, 2.0))
    origin = centers(obs.self_features)
    to_centre = int(
        _REFINE(obs, mask(), jnp.array([10.0, 5.0]) - origin, approach=True)
    )
    to_pads = int(_REFINE(obs, mask(), _pad_mean(obs) - origin, approach=True))
    assert to_centre != to_pads
    assert as_ints(act(obs)) == as_ints(beta_act(obs)) == as_ints(alpha_act(obs))
    assert as_ints(act(obs))[0] == to_centre


def test_dead_or_inactive_guard_matches_beta_and_alpha() -> None:
    live = enemy(observation(WARRIOR_CLASS_ID), 0, (9.0, 5.0), class_id=MAGE_CLASS_ID)
    permissive = mask((6, 0), (6, 1))
    assert as_ints(act(live, permissive)) != _NO_OP
    for stopped in (own_state(live, alive=False), own_state(live, active=False)):
        assert as_ints(act(stopped, permissive)) == _NO_OP
        assert as_ints(beta_act(stopped, permissive)) == _NO_OP
        assert as_ints(alpha_act(stopped, permissive)) == _NO_OP


def test_rogue_nearest_fallback_move_matches_beta() -> None:
    obs = observation(ROGUE_CLASS_ID)
    obs = enemy(obs, 1, (7.0, 5.0), class_id=WARRIOR_CLASS_ID)
    obs = enemy(obs, 3, (13.0, 8.0), class_id=WARRIOR_CLASS_ID)
    obs = ally(obs, 0, (9.0, 5.0), class_id=WARRIOR_CLASS_ID)
    # The touching ally makes body-aware steering differ from simple steering.
    prey = centers(obs.enemy_unit_features[1])
    bodies = jnp.concatenate((obs.ally_unit_features, obs.enemy_unit_features))
    blockers = jnp.zeros(10, dtype=jnp.bool_).at[0].set(True)
    simple = int(_REFINE(obs, mask(), prey - centers(obs.self_features), approach=True))
    assert int(_BODY(obs, mask(), prey, bodies, blockers)) != simple
    assert int(act(obs).move) == int(beta_act(obs).move) == simple


def test_alpha_and_beta_descriptor_digests_are_unchanged() -> None:
    assert canonical_digest_sha256(reactive_tdm_alpha_controller_descriptor()) == (
        "1b22a5f008d628733e23ce78c6acacc7b490d6e3a58b2b4acef1d38bde20c5a5"
    )
    assert canonical_digest_sha256(reactive_tdm_beta_controller_descriptor()) == (
        "527cd69a91bc9a5eda926afa461fc1a8b8e351eb96adf6e431ebd33187576f7a"
    )
