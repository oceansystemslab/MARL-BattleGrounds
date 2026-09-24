"""Check the TDM-GAMMA Hunter's Trap order (GAMMA rule 3, descriptor version 2).

Every case builds one Team B Hunter at (10, 5) on open ground from the shared
GAMMA fixtures and runs the compiled controller. Enemy row r is target column
r + 6. Masks follow Core's ranges: Trap (the Hunter's Ultimate) up to 3 units,
Basic up to 3.5 units. Each case also shows that BETA, or a plausible wrong
rule, gives a different answer, so no case passes by chance.

Contracts checked:

- Cast order: when several enemies have a legal Trap, the Hunter Traps the
  first class present in the order Priest, Mage, Rogue, Warrior, Hunter. Each
  step of the order is checked with the lower classes nearer, weaker and in
  lower rows, so a nearest, lowest-HP or lowest-row rule fails.
- Within a class: lowest current HP, then lowest row. A legal Trap comes
  before HP: a weaker member with no legal Trap is skipped.
- New Trap only: enemies with 1 or 2 Trap ticks are not Trap targets, so the
  next class is chosen. With only such enemies in view the Hunter never Traps,
  even within 2 units (BETA's old rule), and plays BETA's Hunter.
- Casting: a cast chooses Stay, works at full mask range (2.9 units, where
  BETA never Traps), and beats walking toward a higher class out of range.
- Approach: with no legal cast and its Trap ready, the Hunter walks toward the
  first class in view (lowest HP within it, skipping trapped enemies) with
  simple steering, not body-aware steering, while its Basic still hits a
  nearer legal enemy. That Basic keeps the Trap hold: while walking in, it
  does not shoot an enemy with 2 Trap ticks (BETA would), so no Basic is
  chosen when that is the only legal target.
- Readiness: a cooldown or its own spawn shield means not ready, so the Hunter
  plays BETA's Hunter (hold at 3.3 units while its Basic is legal, else
  approach). Any own stun makes the private _trap_ready helper say not ready.
- A dead or inactive Hunter returns the no-op action even with a legal cast.
- The descriptor reports version 2 and hunter_trap.order, which matches
  TRAP_ORDER.
"""

from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from tests.tdm_gamma_fixtures import (
    act,
    ally,
    as_ints,
    beta_act,
    enemy,
    mask,
    observation,
    own_state,
)

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.reactive_common import (
    _body_aware_move,  # pyright: ignore[reportPrivateUsage]
    refine_movement,
)
from marl_battlegrounds.policies.reactive_tdm_gamma import (
    TRAP_ORDER,
    _trap_ready,  # pyright: ignore[reportPrivateUsage]
    reactive_tdm_gamma_controller_descriptor,
)

Action = tuple[int, int, int]
Case = tuple[Observation, ActionMask, Action]
ApproachCase = tuple[Observation, ActionMask, Action, tuple[float, float]]

SIMPLE = cast(Callable[..., Array], jax.jit(refine_movement))
BODY_AWARE = cast(Callable[..., Array], jax.jit(_body_aware_move))
TRAP_READY = cast(Callable[[Observation], Array], jax.jit(_trap_ready))

CLASS_NAMES = {
    PRIEST_CLASS_ID: "Priest",
    MAGE_CLASS_ID: "Mage",
    ROGUE_CLASS_ID: "Rogue",
    WARRIOR_CLASS_ID: "Warrior",
    HUNTER_CLASS_ID: "Hunter",
}

# (class, row, position, HP) in Trap order. Lower classes sit nearer (2.1 to
# 2.9 units), have less HP and use lower rows, so the nearest, lowest-HP and
# lowest-row enemy is always the last class listed.
LINEUP = (
    (PRIEST_CLASS_ID, 4, (12.05, 7.05), 50.0),
    (MAGE_CLASS_ID, 3, (10.0, 2.3), 40.0),
    (ROGUE_CLASS_ID, 2, (7.5, 5.0), 30.0),
    (WARRIOR_CLASS_ID, 1, (10.0, 7.3), 20.0),
    (HUNTER_CLASS_ID, 0, (12.1, 5.0), 10.0),
)


def trap_mask(*, basic: Sequence[int] = (), trap: Sequence[int] = ()) -> ActionMask:
    # Legal Basic and Trap bits for the given enemy rows; all moves legal.
    pairs = [(row + 6, 0) for row in basic] + [(row + 6, 1) for row in trap]
    return mask(*pairs)


def simple_move(
    obs: Observation,
    action_mask: ActionMask,
    intent: tuple[float, float],
    *,
    approach: bool,
) -> int:
    return int(
        SIMPLE(
            obs,
            action_mask,
            jnp.asarray(intent, dtype=jnp.float32),
            approach=jnp.asarray(approach),
        )
    )


def body_aware_move(obs: Observation, action_mask: ActionMask, prey_row: int) -> int:
    bodies = jnp.concatenate((obs.ally_unit_features, obs.enemy_unit_features))
    blockers = jnp.concatenate((obs.ally_visibility_mask, obs.enemy_visibility_mask))
    blockers = blockers.at[obs.self_ally_index].set(False).at[5 + prey_row].set(False)
    prey = obs.enemy_unit_features[
        prey_row, jnp.asarray([AGENT_FEATURE_X, AGENT_FEATURE_Y])
    ]
    return int(BODY_AWARE(obs, action_mask, prey, bodies, blockers))


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(index, id=f"{CLASS_NAMES[entry[0]].lower()}_first")
        for index, entry in enumerate(LINEUP)
    ],
)
def test_hunter_traps_the_first_class_in_the_order(first: int) -> None:
    # Classes above `first` are out of view; every enemy in view has a legal Trap.
    obs = observation(HUNTER_CLASS_ID)
    for class_id, row, xy, hp in LINEUP[first:]:
        obs = enemy(obs, row, xy, class_id=class_id, hp=hp)
    rows = [row for _, row, _, _ in LINEUP[first:]]
    action_mask = trap_mask(basic=rows, trap=rows)
    # BETA only Traps within 2 units, and every enemy here is farther away.
    assert as_ints(beta_act(obs, action_mask))[2] == 0
    assert as_ints(act(obs, action_mask)) == (MOVE_STAY, LINEUP[first][1] + 6, 1)


def weaker_priest_in_a_higher_row() -> Case:
    obs = enemy(
        observation(HUNTER_CLASS_ID), 1, (12.5, 5.0), class_id=PRIEST_CLASS_ID, hp=50.0
    )
    obs = enemy(obs, 3, (10.0, 7.5), class_id=PRIEST_CLASS_ID, hp=20.0)
    obs = enemy(obs, 0, (7.8, 5.0), class_id=MAGE_CLASS_ID, hp=5.0)
    return obs, trap_mask(basic=(0, 1, 3), trap=(0, 1, 3)), (MOVE_STAY, 9, 1)


def equal_health_priests_take_the_lower_row() -> Case:
    # The row 3 Priest is nearer, so a nearest rule would pick it.
    obs = enemy(
        observation(HUNTER_CLASS_ID), 1, (12.8, 5.0), class_id=PRIEST_CLASS_ID, hp=30.0
    )
    obs = enemy(obs, 3, (10.0, 7.1), class_id=PRIEST_CLASS_ID, hp=30.0)
    return obs, trap_mask(basic=(1, 3), trap=(1, 3)), (MOVE_STAY, 7, 1)


def legal_priest_before_a_weaker_priest_out_of_trap_range() -> Case:
    # The weaker Priest is 3.3 units away: Basic range, not Trap range.
    obs = enemy(
        observation(HUNTER_CLASS_ID), 1, (13.3, 5.0), class_id=PRIEST_CLASS_ID, hp=20.0
    )
    obs = enemy(obs, 3, (10.0, 7.5), class_id=PRIEST_CLASS_ID, hp=50.0)
    obs = enemy(obs, 0, (7.8, 5.0), class_id=MAGE_CLASS_ID, hp=5.0)
    return obs, trap_mask(basic=(0, 1, 3), trap=(0, 3)), (MOVE_STAY, 9, 1)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.__name__)
        for case in (
            weaker_priest_in_a_higher_row,
            equal_health_priests_take_the_lower_row,
            legal_priest_before_a_weaker_priest_out_of_trap_range,
        )
    ],
)
def test_hunter_traps_the_weakest_legal_member_of_the_class(
    case: Callable[[], Case],
) -> None:
    obs, action_mask, expected = case()
    assert as_ints(beta_act(obs, action_mask)) != expected
    assert as_ints(act(obs, action_mask)) == expected


@pytest.mark.parametrize(
    ("priest_ticks", "expected_target"),
    [
        pytest.param(0.0, 10, id="untrapped_priest"),
        pytest.param(1.0, 9, id="priest_at_1_tick"),
        pytest.param(2.0, 9, id="priest_at_2_ticks"),
    ],
)
def test_trap_skips_enemies_that_already_have_trap_ticks(
    priest_ticks: float, expected_target: int
) -> None:
    obs = enemy(
        observation(HUNTER_CLASS_ID),
        4,
        (11.5, 5.0),
        class_id=PRIEST_CLASS_ID,
        trap_ticks=priest_ticks,
    )
    obs = enemy(obs, 3, (10.0, 7.5), class_id=MAGE_CLASS_ID)
    action_mask = trap_mask(basic=(3, 4), trap=(3, 4))
    # BETA Traps the nearest legal enemy within 2 units, whatever its Trap ticks.
    assert as_ints(beta_act(obs, action_mask)) == (MOVE_WEST, 10, 1)
    assert as_ints(act(obs, action_mask)) == (MOVE_STAY, expected_target, 1)


@pytest.mark.parametrize(
    ("trap_ticks", "distance", "beta_expected", "expected"),
    [
        pytest.param(
            1.0, 1.5, (MOVE_WEST, 7, 1), (MOVE_WEST, 7, 0), id="no_retrap_within_2"
        ),
        pytest.param(
            1.0, 2.5, (MOVE_WEST, 7, 0), (MOVE_WEST, 7, 0), id="beta_action_at_1_tick"
        ),
        # Trap hold (rule 2) also drops the Basic on a 2-tick enemy.
        pytest.param(
            2.0, 2.5, (MOVE_WEST, 7, 0), (MOVE_WEST, 0, 0), id="beta_move_at_2_ticks"
        ),
    ],
)
def test_no_new_trap_enemy_keeps_betas_hunter_and_never_traps(
    trap_ticks: float, distance: float, beta_expected: Action, expected: Action
) -> None:
    obs = enemy(
        observation(HUNTER_CLASS_ID), 1, (10.0 + distance, 5.0), trap_ticks=trap_ticks
    )
    action_mask = trap_mask(basic=(1,), trap=(1,))
    # A cast would Stay and an approach would head East; BETA retreats West.
    assert simple_move(obs, action_mask, (distance, 0.0), approach=True) == MOVE_EAST
    assert as_ints(beta_act(obs, action_mask)) == beta_expected
    assert as_ints(act(obs, action_mask)) == expected


@pytest.mark.parametrize(
    ("distance", "beta_expected"),
    [
        pytest.param(1.5, (MOVE_WEST, 7, 1), id="at_1_5_beta_retreats"),
        pytest.param(2.9, (MOVE_WEST, 7, 0), id="at_2_9_beta_never_traps"),
    ],
)
def test_hunter_traps_at_full_mask_range_and_stays(
    distance: float, beta_expected: Action
) -> None:
    obs = enemy(observation(HUNTER_CLASS_ID), 1, (10.0 + distance, 5.0))
    action_mask = trap_mask(basic=(1,), trap=(1,))
    assert as_ints(beta_act(obs, action_mask)) == beta_expected
    assert as_ints(act(obs, action_mask)) == (MOVE_STAY, 7, 1)


def test_a_legal_lower_class_cast_beats_walking_to_a_far_priest() -> None:
    obs = enemy(observation(HUNTER_CLASS_ID), 4, (16.0, 5.0), class_id=PRIEST_CLASS_ID)
    obs = enemy(obs, 1, (10.0, 7.5))
    action_mask = trap_mask(basic=(1,), trap=(1,))
    # Walking to the Priest would head East; BETA retreats South from the Warrior.
    assert simple_move(obs, action_mask, (6.0, 0.0), approach=True) == MOVE_EAST
    assert as_ints(beta_act(obs, action_mask)) == (MOVE_SOUTH, 7, 0)
    assert as_ints(act(obs, action_mask)) == (MOVE_STAY, 7, 1)


def far_priest_and_mage(obs: Observation) -> Observation:
    # A strong Priest 6 units East, a weak Mage 6 units West, and a Warrior 3.3
    # units North: Basic range but not Trap range, so nothing has a legal Trap.
    obs = enemy(obs, 4, (16.0, 5.0), class_id=PRIEST_CLASS_ID, hp=80.0)
    obs = enemy(obs, 3, (4.0, 5.0), class_id=MAGE_CLASS_ID, hp=20.0)
    return enemy(obs, 1, (10.0, 8.3), hp=60.0)


def priest_before_a_weaker_mage() -> ApproachCase:
    obs = far_priest_and_mage(observation(HUNTER_CLASS_ID))
    return obs, trap_mask(basic=(1,)), (MOVE_EAST, 7, 0), (-6.0, 0.0)


def weaker_of_two_priests() -> ApproachCase:
    obs = enemy(
        observation(HUNTER_CLASS_ID), 1, (16.0, 5.0), class_id=PRIEST_CLASS_ID, hp=70.0
    )
    obs = enemy(obs, 4, (4.0, 5.0), class_id=PRIEST_CLASS_ID, hp=30.0)
    obs = enemy(obs, 0, (10.0, 8.3), hp=60.0)
    return obs, trap_mask(basic=(0,)), (MOVE_WEST, 6, 0), (6.0, 0.0)


def untrapped_mage_before_a_one_tick_priest() -> ApproachCase:
    obs = enemy(
        observation(HUNTER_CLASS_ID),
        4,
        (16.0, 5.0),
        class_id=PRIEST_CLASS_ID,
        trap_ticks=1.0,
    )
    obs = enemy(obs, 3, (4.0, 5.0), class_id=MAGE_CLASS_ID)
    obs = enemy(obs, 1, (10.0, 8.3), hp=60.0)
    return obs, trap_mask(basic=(1,)), (MOVE_WEST, 7, 0), (6.0, 0.0)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.__name__)
        for case in (
            priest_before_a_weaker_mage,
            weaker_of_two_priests,
            untrapped_mage_before_a_one_tick_priest,
        )
    ],
)
def test_ready_hunter_without_a_cast_walks_to_the_first_class(
    case: Callable[[], ApproachCase],
) -> None:
    obs, action_mask, expected, wrong_intent = case()
    # The wrong enemy lies the other way, so walking to it gives another move.
    assert simple_move(obs, action_mask, wrong_intent, approach=True) != expected[0]
    # BETA holds still: its nearest enemy is 3.3 units away with a legal Basic.
    assert as_ints(beta_act(obs, action_mask)) == (MOVE_STAY, expected[1], 0)
    assert as_ints(act(obs, action_mask)) == expected


def test_trap_approach_uses_simple_steering_and_keeps_the_basic() -> None:
    obs = ally(observation(HUNTER_CLASS_ID), 0, (11.2, 5.0))
    obs = enemy(obs, 4, (15.0, 5.0), class_id=PRIEST_CLASS_ID)
    obs = enemy(obs, 2, (10.0, 1.7), class_id=HUNTER_CLASS_ID)
    action_mask = trap_mask(basic=(2,))
    simple = simple_move(obs, action_mask, (5.0, 0.0), approach=True)
    # The ally stands head-on within one stride, so body-aware steering detours.
    assert body_aware_move(obs, action_mask, prey_row=4) != simple
    # BETA holds 3.3 units from the enemy Hunter, its nearest enemy.
    assert as_ints(beta_act(obs, action_mask)) == (MOVE_STAY, 8, 0)
    assert as_ints(act(obs, action_mask)) == (simple, 8, 0)


@pytest.mark.parametrize(
    ("cooldown", "shield", "expected"),
    [
        pytest.param(3.0, 0, (MOVE_STAY, 7, 0), id="cooldown_holds_with_basic"),
        pytest.param(0.0, 1, (MOVE_NORTH, 0, 0), id="spawn_shield_approaches"),
    ],
)
def test_unready_hunter_plays_betas_hunter(
    cooldown: float, shield: int, expected: Action
) -> None:
    obs = own_state(observation(HUNTER_CLASS_ID), cooldown=cooldown, shield=shield)
    obs = far_priest_and_mage(obs)
    # Core admits only (Target None, no Ultimate) under a spawn shield.
    action_mask = trap_mask() if shield else trap_mask(basic=(1,))
    # A ready Hunter would walk East to the Priest instead.
    assert simple_move(obs, action_mask, (6.0, 0.0), approach=True) != expected[0]
    assert as_ints(beta_act(obs, action_mask)) == expected
    assert as_ints(act(obs, action_mask)) == expected


@pytest.mark.parametrize(
    ("trap_ticks", "charge_stun", "poison_stun"),
    [
        pytest.param(2.0, 0.0, 0.0, id="hunter_trap"),
        pytest.param(0.0, 2.0, 0.0, id="warrior_charge"),
        pytest.param(0.0, 0.0, 2.0, id="rogue_poison"),
    ],
)
def test_any_own_stun_makes_the_trap_not_ready(
    trap_ticks: float, charge_stun: float, poison_stun: float
) -> None:
    # Core forces Stay on a stunned actor, so the helper is checked directly.
    obs = observation(HUNTER_CLASS_ID)
    assert bool(TRAP_READY(obs))
    stunned = own_state(
        obs, trap_ticks=trap_ticks, charge_stun=charge_stun, poison_stun=poison_stun
    )
    assert not bool(TRAP_READY(stunned))


@pytest.mark.parametrize("state", ["dead", "inactive"])
def test_dead_or_inactive_hunter_returns_no_op_despite_a_legal_cast(
    state: str,
) -> None:
    obs = enemy(observation(HUNTER_CLASS_ID), 1, (12.5, 5.0))
    action_mask = trap_mask(basic=(1,), trap=(1,))
    assert as_ints(act(obs, action_mask)) == (MOVE_STAY, 7, 1)
    gone = own_state(obs, alive=state != "dead", active=state != "inactive")
    assert as_ints(act(gone, action_mask)) == (0, 0, 0)


def test_descriptor_reports_version_2_and_the_trap_order() -> None:
    descriptor = reactive_tdm_gamma_controller_descriptor()
    hunter_trap = cast(dict[str, object], descriptor["hunter_trap"])
    assert descriptor["version"] == 2
    assert hunter_trap["order"] == ["Priest", "Mage", "Rogue", "Warrior", "Hunter"]
    assert [CLASS_NAMES[class_id] for class_id in TRAP_ORDER] == hunter_trap["order"]


def test_trap_approach_keeps_the_trap_hold_on_its_basic() -> None:
    obs = enemy(observation(HUNTER_CLASS_ID), 4, (16.0, 5.0), class_id=PRIEST_CLASS_ID)
    obs = enemy(obs, 1, (10.0, 8.3), class_id=WARRIOR_CLASS_ID, trap_ticks=2.0)
    action_mask = trap_mask(basic=(1,))
    walk = simple_move(obs, action_mask, (6.0, 0.0), approach=True)
    assert walk != MOVE_STAY
    # BETA holds 3.3 units from the trapped Warrior and shoots it.
    assert as_ints(beta_act(obs, action_mask)) == (MOVE_STAY, 7, 0)
    # GAMMA walks to the Priest and holds its fire on the trapped Warrior.
    assert as_ints(act(obs, action_mask)) == (walk, 0, 0)
