"""Check GAMMA version 2's search and Trap-hold rules, and that all else is BETA.

Every case edits the real Scenario 1 observation through tests.tdm_gamma_fixtures
and runs the jitted GAMMA and BETA controllers. Enemy row r is target r + 6.
Each behaviour case first checks that BETA, or the wrong rule, gives a
different answer on the same geometry, so BETA's code fails it and no case
passes by chance.

Contracts checked:

- Search. With no living enemy in view, Warriors, Mages, Hunters and Rogues
  walk toward the mean of all five opposing spawn pads (relative bank 1). They
  use the full vector with simple steering: not the map centre (BETA's move),
  not their own pads, not a unit vector and not movement without wall
  steering. Team A and Team B observers each read their own bank 1 from the
  real Core pads. Changing the own pads in bank 0 changes nothing. Pads of
  unused roster slots count. Within Core's GEOMETRY_TOLERANCE of the mean the
  move is Stay; twice that far away the actor moves. A Priest with no enemy in
  view acts exactly like BETA's Priest.
- Trap hold. Warrior, Mage, Hunter and Rogue Basics hit an enemy with 0 or 1
  Trap ticks and skip it at 2 or 3 ticks, taking another legal target or none.
  Warrior Charge and Rogue Poison are held at 2 ticks and allowed at 1. A Mage
  whose only Basic target has 2 ticks does not Burst (BETA does); at 1 tick it
  Bursts; with a second, untrapped Basic target it Bursts. Movement and Priest
  healing stay BETA's around units with 3 Trap ticks: Warriors walk to a
  trapped nearest enemy, Mages and Hunters back away from one, a Rogue chases
  a trapped Priest, and a Priest backs away from a trapped enemy and heals a
  trapped ally. Each answer differs from BETA's answer with those units out of
  view. A Hunter still holds its 3 to 3.5 unit band on a trapped enemy whose
  Basic bit is legal, while it does not shoot it.
- Everything else is BETA. With enemies in view, no Trap ticks anywhere, and a
  Hunter whose Trap is neither legal nor ready, GAMMA returns BETA's action for
  every class on 16 seeded random scenes. The scenes vary enough that many
  moves and targets appear, and a Rogue running ALPHA's rules would fail.
- A dead or inactive actor returns the no-op (0, 0, 0), both while searching
  and in a fight where a living actor would act.
- Only living enemies count. A visible enemy row that is dead, inactive or at
  zero health is never chosen, even when its mask bits are legal and a living
  copy of it would be chosen.
- The random key changes nothing: three different keys give the same action
  for every class on seeded scenes and while searching.
"""

import random
from collections.abc import Callable, Sequence
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from tests.tdm_gamma_fixtures import (
    CANONICAL_CLASSES,
    OPPOSING_PADS,
    OWN_PADS,
    act,
    ally,
    alpha_act,
    as_ints,
    beta_act,
    enemy,
    mask,
    observation,
    own_state,
    pads,
    roster,
    scenario,
    wall,
)

from marl_battlegrounds.core.geometry import GEOMETRY_TOLERANCE
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MOVE_STAY,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.reactive_common import centers, refine_movement

NON_PRIESTS = (WARRIOR_CLASS_ID, MAGE_CLASS_ID, HUNTER_CLASS_ID, ROGUE_CLASS_ID)
NON_PRIEST_IDS = ["warrior", "mage", "hunter", "rogue"]
SEED_COUNT = 16

SIMPLE = cast(Callable[..., Array], jax.jit(refine_movement))


def simple_move(
    obs: Observation, intent: Sequence[float] | Array, *, approach: bool = True
) -> int:
    # GAMMA's search steering: refine_movement with the full intent vector.
    return int(
        SIMPLE(
            obs,
            mask(),
            jnp.asarray(intent, dtype=jnp.float32),
            approach=jnp.asarray(approach),
        )
    )


def pad_mean(points: Sequence[tuple[float, float]] | Array) -> Array:
    return jnp.mean(jnp.asarray(points, dtype=jnp.float32), axis=0)


def origin_of(obs: Observation) -> Array:
    return centers(obs.self_features)


def hide_trapped(obs: Observation) -> Observation:
    # The wrong rule for movement: every unit (not self) with 2 or more Trap
    # ticks drops out of view.
    enemy_trapped = obs.enemy_unit_features[:, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
    ally_trapped = obs.ally_unit_features[:, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
    enemy_hidden = enemy_trapped >= 2
    ally_hidden = (ally_trapped >= 2).at[obs.self_ally_index].set(False)
    return obs._replace(
        enemy_unit_features=jnp.where(
            enemy_hidden[:, None], 0.0, obs.enemy_unit_features
        ),
        enemy_visibility_mask=obs.enemy_visibility_mask & ~enemy_hidden,
        ally_unit_features=jnp.where(ally_hidden[:, None], 0.0, obs.ally_unit_features),
        ally_visibility_mask=obs.ally_visibility_mask & ~ally_hidden,
    )


@pytest.mark.parametrize(
    ("class_id", "team", "xy", "walled"),
    [
        pytest.param(WARRIOR_CLASS_ID, "B", (10.0, 8.0), False, id="warrior_team_b"),
        pytest.param(
            MAGE_CLASS_ID, "A", (10.5, 4.0), True, id="mage_team_a_behind_a_wall"
        ),
        pytest.param(HUNTER_CLASS_ID, "B", (13.0, 2.0), False, id="hunter_team_b"),
        pytest.param(ROGUE_CLASS_ID, "A", (6.0, 8.5), False, id="rogue_team_a"),
    ],
)
def test_search_walks_to_the_opposing_pad_mean(
    class_id: int, team: str, xy: tuple[float, float], walled: bool
) -> None:
    obs = observation(class_id, team=team, xy=xy)
    if walled:
        obs = wall(obs, 0, (8.0, 5.0), 1.0, 6.0)
    full = pad_mean(OPPOSING_PADS) - origin_of(obs)
    expected = simple_move(obs, full)
    # BETA walks to the map centre; bank 0 holds the actor's own pads.
    assert as_ints(beta_act(obs))[0] != expected
    assert simple_move(obs, pad_mean(OWN_PADS) - origin_of(obs)) != expected
    if walled:
        # The wall stands between the actor and the far mean. A unit vector
        # ends before the wall, and no wall steering walks into it.
        unit = full / jnp.sqrt(jnp.sum(jnp.square(full)))
        assert simple_move(obs, unit) != expected
        assert simple_move(obs, full, approach=False) != expected
    assert as_ints(act(obs)) == (expected, 0, 0)


@pytest.mark.parametrize(("team", "slot"), [("A", 4), ("B", 9)])
def test_each_team_reads_its_own_relative_bank_one(team: str, slot: int) -> None:
    # Put back the real Core pads for this observer. Bank 0 holds its own
    # team's pads and bank 1 the other team's, so Team A's bank 1 is in the
    # east and Team B's is in the west.
    banks = scenario().observation.spawn_lifecycle.spawn_pad_positions_by_agent_by_team[
        slot
    ]
    obs = observation(WARRIOR_CLASS_ID, team=team, xy=(10.0, 8.0))
    obs = obs._replace(
        spawn_lifecycle=obs.spawn_lifecycle._replace(
            spawn_pad_positions_by_agent_by_team=banks
        )
    )
    expected = simple_move(obs, pad_mean(banks[1]) - origin_of(obs))
    assert simple_move(obs, pad_mean(banks[0]) - origin_of(obs)) != expected
    assert as_ints(beta_act(obs))[0] != expected
    assert as_ints(act(obs)) == (expected, 0, 0)


def test_own_pads_in_bank_zero_change_nothing() -> None:
    base = observation(ROGUE_CLASS_ID, xy=(12.0, 3.0))
    expected = simple_move(base, pad_mean(OPPOSING_PADS) - origin_of(base))
    for own in (
        OWN_PADS,
        tuple((4.0 + 2.0 * slot, 9.0) for slot in range(5)),
        ((16.0, 1.0),) * 5,
    ):
        obs = pads(base, OPPOSING_PADS, own)
        assert simple_move(obs, pad_mean(own) - origin_of(obs)) != expected
        assert as_ints(act(obs)) == (expected, 0, 0)


def test_all_five_opposing_pads_count_including_unused_slots() -> None:
    # Roster slots 3 and 4 are unused, but their pads in the north-east still
    # pull the mean away from the three used pads in the west.
    layout = ((2.0, 2.0), (2.0, 4.0), (2.0, 6.0), (18.0, 9.0), (18.0, 9.0))
    obs = pads(observation(HUNTER_CLASS_ID, xy=(10.0, 3.0)), layout)
    obs = roster(obs, [MAGE_CLASS_ID, WARRIOR_CLASS_ID, PRIEST_CLASS_ID, None, None])
    expected = simple_move(obs, pad_mean(layout) - origin_of(obs))
    assert simple_move(obs, pad_mean(layout[:3]) - origin_of(obs)) != expected
    assert as_ints(beta_act(obs))[0] != expected
    assert as_ints(act(obs)) == (expected, 0, 0)


def test_search_stays_within_geometry_tolerance_of_the_mean() -> None:
    # The pad mean is (4, 2), away from the map centre, so BETA moves there.
    layout = ((3.0, 2.0), (5.0, 2.0), (4.0, 1.0), (4.0, 3.0), (4.0, 2.0))
    offsets = [scale * GEOMETRY_TOLERANCE for scale in (0.0, 0.5, -0.5, 2.0, -2.0)]
    for class_id in NON_PRIESTS:
        for offset in offsets:
            obs = pads(observation(class_id, xy=(4.0 + offset, 2.0)), layout)
            if abs(offset) <= GEOMETRY_TOLERANCE:
                expected = MOVE_STAY
                assert as_ints(beta_act(obs))[0] != MOVE_STAY
            else:
                expected = simple_move(obs, (-offset, 0.0))
                assert expected != MOVE_STAY
            assert as_ints(act(obs)) == (expected, 0, 0), (class_id, offset)


@pytest.mark.parametrize(
    "with_ally", [False, True], ids=["alone_walks_to_map_centre", "follows_its_ally"]
)
def test_priest_without_enemies_matches_beta(with_ally: bool) -> None:
    obs = observation(PRIEST_CLASS_ID, xy=(13.0, 8.0))
    action_mask = mask()
    if with_ally:
        obs = ally(obs, 0, (13.0, 5.5), hp=50.0)
        action_mask = mask((1, 0))
    beta = as_ints(beta_act(obs, action_mask))
    # A Priest that searched like the others would head west instead.
    assert simple_move(obs, pad_mean(OPPOSING_PADS) - origin_of(obs)) != beta[0]
    assert as_ints(act(obs, action_mask)) == beta


@pytest.mark.parametrize("class_id", NON_PRIESTS, ids=NON_PRIEST_IDS)
def test_basic_skips_an_enemy_with_two_or_more_trap_ticks(class_id: int) -> None:
    # Row 0 is a 20-HP Priest 1 unit east: the lowest-HP legal Basic target
    # for every class, inside every Basic radius. Row 1, when present, is an
    # untrapped 60-HP Warrior 1 unit north with a legal Basic too.
    base = observation(class_id)
    if class_id == HUNTER_CLASS_ID:
        base = own_state(base, cooldown=5.0)  # Keeps the Trap order out.
    for trap_ticks in (0.0, 1.0, 2.0, 3.0):
        alone = enemy(
            base,
            0,
            (11.0, 5.0),
            class_id=PRIEST_CLASS_ID,
            hp=20.0,
            trap_ticks=trap_ticks,
        )
        paired = enemy(alone, 1, (10.0, 6.0), class_id=WARRIOR_CLASS_ID, hp=60.0)
        for obs, action_mask, held_target in (
            (alone, mask((6, 0)), 0),
            (paired, mask((6, 0), (7, 0)), 7),
        ):
            beta = as_ints(beta_act(obs, action_mask))
            assert beta[1:] == (6, 0), trap_ticks
            target = 6 if trap_ticks <= 1 else held_target
            assert as_ints(act(obs, action_mask)) == (beta[0], target, 0), trap_ticks


@pytest.mark.parametrize(
    ("class_id", "trap_ticks"),
    [
        pytest.param(WARRIOR_CLASS_ID, 1.0, id="warrior_charge_at_one_tick"),
        pytest.param(WARRIOR_CLASS_ID, 2.0, id="warrior_charge_held_at_two_ticks"),
        pytest.param(ROGUE_CLASS_ID, 1.0, id="rogue_poison_at_one_tick"),
        pytest.param(ROGUE_CLASS_ID, 2.0, id="rogue_poison_held_at_two_ticks"),
    ],
)
def test_charge_and_poison_wait_for_one_trap_tick(
    class_id: int, trap_ticks: float
) -> None:
    # Row 0 is a 30-HP Priest with legal Basic and Ultimate bits: under the
    # Warrior's 40-HP Charge limit and inside the Rogue's Basic radius. Row 1
    # is an untrapped 35-HP Warrior with a legal Basic only.
    obs = enemy(
        observation(class_id),
        0,
        (11.0, 5.0),
        class_id=PRIEST_CLASS_ID,
        hp=30.0,
        trap_ticks=trap_ticks,
    )
    obs = enemy(obs, 1, (10.0, 6.0), class_id=WARRIOR_CLASS_ID, hp=35.0)
    action_mask = mask((6, 0), (6, 1), (7, 0))
    move, target, ultimate = as_ints(beta_act(obs, action_mask))
    assert (target, ultimate) == (6, 1)
    expected = (move, 6, 1) if trap_ticks <= 1 else (move, 7, 0)
    assert as_ints(act(obs, action_mask)) == expected


@pytest.mark.parametrize(
    ("trap_ticks", "second_target", "burst"),
    [
        pytest.param(1.0, False, 1, id="one_tick_bursts"),
        pytest.param(2.0, False, 0, id="two_ticks_no_burst"),
        pytest.param(2.0, True, 1, id="two_ticks_with_untrapped_basic_target_bursts"),
    ],
)
def test_mage_burst_needs_a_basic_target_the_trap_hold_allows(
    trap_ticks: float, second_target: bool, burst: int
) -> None:
    # The Priest stands exactly 2 units east, so Mage spacing says Stay.
    obs = enemy(
        observation(MAGE_CLASS_ID),
        4,
        (12.0, 5.0),
        class_id=PRIEST_CLASS_ID,
        trap_ticks=trap_ticks,
    )
    pairs = [(10, 0), (0, 1)]
    if second_target:
        obs = enemy(obs, 1, (10.0, 7.5), class_id=WARRIOR_CLASS_ID)
        pairs.append((7, 0))
    action_mask = mask(*pairs)
    assert as_ints(beta_act(obs, action_mask)) == (MOVE_STAY, 0, 1)
    assert as_ints(act(obs, action_mask)) == (MOVE_STAY, 0, burst)


def warrior_facing_trapped_warrior() -> tuple[Observation, ActionMask]:
    obs = enemy(observation(WARRIOR_CLASS_ID), 0, (12.0, 5.0), trap_ticks=3.0)
    return enemy(obs, 1, (10.0, 1.0)), mask()


def mage_backing_off_trapped_warrior() -> tuple[Observation, ActionMask]:
    obs = enemy(observation(MAGE_CLASS_ID), 0, (11.5, 5.0), trap_ticks=3.0)
    return enemy(obs, 1, (10.0, 1.0)), mask()


def hunter_backing_off_trapped_warrior() -> tuple[Observation, ActionMask]:
    # A cooldown keeps the Trap approach out of this spacing case.
    obs = own_state(observation(HUNTER_CLASS_ID), cooldown=5.0)
    obs = enemy(obs, 0, (12.0, 5.0), trap_ticks=3.0)
    return enemy(obs, 1, (10.0, 1.0)), mask()


def rogue_chasing_trapped_priest() -> tuple[Observation, ActionMask]:
    obs = enemy(
        observation(ROGUE_CLASS_ID),
        0,
        (13.0, 5.0),
        class_id=PRIEST_CLASS_ID,
        trap_ticks=3.0,
    )
    return enemy(obs, 1, (10.0, 1.0), class_id=MAGE_CLASS_ID), mask()


def priest_with_trapped_enemy_and_ally() -> tuple[Observation, ActionMask]:
    # The Priest retreats from the trapped nearest enemy and heals the
    # trapped, hurt ally in row 0.
    obs = enemy(observation(PRIEST_CLASS_ID), 0, (11.5, 5.0), trap_ticks=3.0)
    obs = enemy(obs, 1, (10.0, 1.5), class_id=ROGUE_CLASS_ID)
    obs = ally(obs, 0, (10.0, 6.5), hp=20.0, trap_ticks=3.0)
    obs = ally(obs, 1, (7.0, 5.0), hp=50.0)
    return obs, mask((1, 0), (2, 0))


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.__name__)
        for case in (
            warrior_facing_trapped_warrior,
            mage_backing_off_trapped_warrior,
            hunter_backing_off_trapped_warrior,
            rogue_chasing_trapped_priest,
            priest_with_trapped_enemy_and_ally,
        )
    ],
)
def test_moves_around_trapped_units_match_beta(
    case: Callable[[], tuple[Observation, ActionMask]],
) -> None:
    # Row 0 is trapped for 3 ticks; row 1 is an untrapped enemy to the south.
    obs, action_mask = case()
    beta = as_ints(beta_act(obs, action_mask))
    assert as_ints(beta_act(hide_trapped(obs), action_mask)) != beta
    assert as_ints(act(obs, action_mask)) == beta


def test_hunter_holds_its_band_on_a_trapped_enemy_it_may_not_shoot() -> None:
    # The only enemy has 3 Trap ticks, so no Trap approach starts. Spacing
    # reads the mask's Basic bit, not the Trap hold: a rule that used the
    # held pool would approach east instead of holding.
    obs = enemy(observation(HUNTER_CLASS_ID), 0, (13.2, 5.0), trap_ticks=3.0)
    action_mask = mask((6, 0))
    assert simple_move(obs, (3.2, 0.0)) != MOVE_STAY
    assert as_ints(beta_act(obs, action_mask)) == (MOVE_STAY, 6, 0)
    assert as_ints(act(obs, action_mask)) == (MOVE_STAY, 0, 0)


def random_scene(class_id: int, seed: int) -> tuple[Observation, ActionMask]:
    # One to five visible living enemies with no Trap ticks, zero to three
    # allies, sometimes a wall, and mask bits that fit the actor's class.
    rng = random.Random(seed)
    xy = (rng.uniform(3.0, 17.0), rng.uniform(2.0, 8.0))
    obs = observation(class_id, team="A" if seed % 2 else "B", xy=xy)
    if class_id == HUNTER_CLASS_ID:
        obs = own_state(obs, cooldown=float(rng.randint(1, 19)))  # Not ready.
    pairs: list[tuple[int, int]] = []
    for row in rng.sample(range(5), rng.randint(1, 5)):
        spot = (
            min(max(xy[0] + rng.uniform(-4.5, 4.5), 0.5), 19.5),
            min(max(xy[1] + rng.uniform(-4.5, 4.5), 0.5), 9.5),
        )
        obs = enemy(
            obs,
            row,
            spot,
            class_id=rng.choice(CANONICAL_CLASSES),
            hp=rng.uniform(5.0, 100.0),
        )
        if class_id != PRIEST_CLASS_ID and rng.random() < 0.6:
            pairs.append((row + 6, 0))
        # The Hunter's Trap is never legal here.
        if class_id in (WARRIOR_CLASS_ID, ROGUE_CLASS_ID) and rng.random() < 0.5:
            pairs.append((row + 6, 1))
    for row in range(rng.randint(0, 3)):
        spot = (
            min(max(xy[0] + rng.uniform(-4.0, 4.0), 0.5), 19.5),
            min(max(xy[1] + rng.uniform(-4.0, 4.0), 0.5), 9.5),
        )
        obs = ally(obs, row, spot, hp=rng.uniform(5.0, 100.0))
    if class_id == PRIEST_CLASS_ID:
        for target in range(1, 6):
            if rng.random() < 0.6:
                pairs.append((target, 0))
            if rng.random() < 0.4:
                pairs.append((target, 1))
    if class_id == MAGE_CLASS_ID and rng.random() < 0.5:
        pairs.append((0, 1))
    if rng.random() < 0.3:
        centre = (rng.uniform(4.0, 16.0), rng.uniform(2.0, 8.0))
        obs = wall(obs, 0, centre, 0.4, 3.0)
    return obs, mask(*pairs)


@pytest.mark.parametrize(
    "class_id", [*NON_PRIESTS, PRIEST_CLASS_ID], ids=[*NON_PRIEST_IDS, "priest"]
)
def test_everything_else_matches_beta(class_id: int) -> None:
    actions: list[tuple[int, int, int]] = []
    alpha_differs = False
    for seed in range(SEED_COUNT):
        obs, action_mask = random_scene(class_id, seed)
        gamma = as_ints(act(obs, action_mask))
        assert gamma == as_ints(beta_act(obs, action_mask)), seed
        alpha_differs |= as_ints(alpha_act(obs, action_mask)) != gamma
        actions.append(gamma)
    # The scenes must exercise real choices, not one repeated answer.
    assert len({move for move, _, _ in actions}) >= 4
    assert sum(target > 0 for _, target, _ in actions) >= 3
    if class_id != HUNTER_CLASS_ID:
        assert any(ultimate for _, _, ultimate in actions)
    if class_id == ROGUE_CLASS_ID:
        assert alpha_differs


@pytest.mark.parametrize("state", ["dead", "inactive"])
def test_dead_or_inactive_actor_returns_the_no_op(state: str) -> None:
    # Every class has a legal choice here: Basic and Ultimate on a 30-HP
    # Priest, a Burst, and a heal on a 20-HP ally.
    action_mask = mask((1, 0), (1, 1), (6, 0), (6, 1), (0, 1))
    for class_id in (*NON_PRIESTS, PRIEST_CLASS_ID):
        searching = observation(class_id, xy=(13.0, 8.0))
        fighting = ally(observation(class_id), 0, (10.0, 6.0), hp=20.0)
        fighting = enemy(fighting, 0, (11.0, 5.0), class_id=PRIEST_CLASS_ID, hp=30.0)
        for obs in (searching, fighting):
            assert as_ints(act(obs, action_mask)) != (0, 0, 0), class_id
            gone = own_state(obs, alive=state != "dead", active=state != "inactive")
            assert as_ints(act(gone, action_mask)) == (0, 0, 0), class_id


@pytest.mark.parametrize("state", ["dead", "inactive", "zero_health"])
def test_visible_rows_that_are_not_living_are_never_chosen(state: str) -> None:
    # Row 0 is a weak Priest every class would pick while it lives (Basic,
    # Charge, Poison or Trap); row 1 is a living Warrior with only a Basic bit.
    action_mask = mask((6, 0), (6, 1), (7, 0))
    for class_id in NON_PRIESTS:
        living = enemy(
            observation(class_id), 0, (11.0, 5.0), class_id=PRIEST_CLASS_ID, hp=10.0
        )
        living = enemy(living, 1, (11.2, 5.5), class_id=WARRIOR_CLASS_ID, hp=90.0)
        assert as_ints(act(living, action_mask))[1] == 6, class_id
        if state == "zero_health":
            gone = enemy(living, 0, (11.0, 5.0), class_id=PRIEST_CLASS_ID, hp=0.0)
        else:
            feature = AGENT_FEATURE_ALIVE if state == "dead" else AGENT_FEATURE_ACTIVE
            gone = living._replace(
                enemy_unit_features=living.enemy_unit_features.at[0, feature].set(0)
            )
        assert bool(gone.enemy_visibility_mask[0])
        assert as_ints(act(gone, action_mask))[1:] == (7, 0), class_id


def test_the_random_key_changes_nothing() -> None:
    for class_id in (*NON_PRIESTS, PRIEST_CLASS_ID):
        scenes = [random_scene(class_id, seed) for seed in range(3)]
        scenes.append((observation(class_id, xy=(13.0, 8.0)), mask()))
        for obs, action_mask in scenes:
            actions = {
                as_ints(act(obs, action_mask, key=key)) for key in (0, 1, 2**31 - 1)
            }
            assert len(actions) == 1, class_id
