"""Check Team Deathmatch Red Zone scoring through public trajectories.

Each team's Red Zone is a full-height strip on its own spawn side,
team_deathmatch_red_zone_depth map units deep. A new death gives the other team
2 points when the victim's centre, at the start of the tick when combat
resolves, is inside the victim's own team's strip; every other new death gives
1 point. Both bounds are included, and depth 0.0 turns the rule off. The map is
12 x 12 with Team A's pads at x = 2 and Team B's at x = 10, so at depth 5 Team
A's strip is x <= 5 and Team B's is x >= 7; Hunters hit up to 3.5 away.

Every case starts from an authored state, reads the observation (column 19
holds the depth) and the action mask, chooses legal actions, steps, and
captures the transition as context V4 and frame V3. Capture re-derives the
points on the host from the recorded start positions and refuses a mismatch,
so each case also checks that the host event validator agrees with Core.

The cases: inside, on and just outside both boundaries at depths 0, 5 and 6;
an invader killed inside the enemy's strip scores 1; overlapping strips
(depth 7) and a full-width strip (depth 12) score 2; swapped banks swap the
strips; both teams score their own points on one tick; two contributors on
one victim give one death, one agent_died event and one score event of +2;
a corpse scores nothing, a respawn is not a death, and the respawned agent's
later death scores 2 again; points use the start position even when the
victim then walks or Charges across the boundary; a threshold overshoot picks
the higher score, or a draw on the final tick; five Red Zone deaths take
Team A from 16,777,206 to exactly 16,777,216 at the top threshold 16,777,207,
which the context shows exactly and validate_env_state accepts; configs that
differ only in depth differ at reset only in column 19; and eager, jit, an
external vmap over stacked depth-0 and depth-5 configs and a real lax.scan
agree.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import current_evaluation_context
from tests.test_team_deathmatch_semantics import (
    _assert_task_result,  # pyright: ignore[reportPrivateUsage]
    _joint_action,  # pyright: ignore[reportPrivateUsage]
    _scenario,  # pyright: ignore[reportPrivateUsage]
    _take_step,  # pyright: ignore[reportPrivateUsage]
    _task_config,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds.core.config import validate_env_config, validate_env_state
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURE_TDM_ALLY_SCORE,
    CONTEXT_FEATURE_TDM_ENEMY_SCORE,
    CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH,
    MAX_AGENTS_PER_TEAM,
    MOVE_EAST,
    MOVE_WEST,
    TASK_MODE_OUTCOME_DRAW,
    TASK_MODE_OUTCOME_ONGOING,
    TASK_MODE_OUTCOME_TEAM_A_WIN,
    TASK_MODE_OUTCOME_TEAM_B_WIN,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v3,
    capture_initial_evaluation_frame_v3,
)
from marl_battlegrounds.evaluation.models import (
    ActionRejectedEventV1,
    AgentDiedEventV1,
    EvaluationTransitionV1,
    LethalDamageContributionEventV1,
    TeamDeathmatchCompletedEventV1,
    TeamDeathmatchScoreChangedEventV1,
)

# Target action for the acting agent's enemy observation row 0; row i is
# _ENEMY_0 + i. For Team A row i is slot 5 + i, and for Team B it is slot i.
_ENEMY_0 = 1 + MAX_AGENTS_PER_TEAM
_TOP_THRESHOLD = 16_777_207

type _StepResult = tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]
type _Step = Callable[[EnvConfig, EnvState, ActionMask, Action, Array], _StepResult]
type _Reset = Callable[
    [EnvConfig, Array], tuple[EnvState, Observation, ActionMask, Info]
]
# One compiled step serves every case: the configs share shapes and dtypes, so
# a new depth, class, pad bank or position does not compile again.
_compiled_step = cast(_Step, jax.jit(step))


class _Tick(NamedTuple):
    state: EnvState
    observation: Observation
    reward: Reward
    done_flags: DoneFlags
    info: Info
    transition: EvaluationTransitionV1


class _Game:
    # One authored game and its captured record. Each tick checks that every
    # configured actor observes the depth, steps through the compiled Core
    # step and captures the transition, which re-derives the points on the host.
    def __init__(
        self,
        config: EnvConfig,
        *,
        low_health_slots: Iterable[int] = (),
        positions: Mapping[int, tuple[float, float]] | None = None,
        scores: tuple[int, int] = (0, 0),
        step_count: int = 0,
    ) -> None:
        self.config = config
        self.state, self.observation, self.action_mask, _ = _scenario(
            config,
            scores=scores,
            step_count=step_count,
            low_health_slots=low_health_slots,
            positions=positions,
        )
        self._context = current_evaluation_context(config)
        self._frame = capture_initial_evaluation_frame_v3(
            self._context, self.state, self.observation, self.action_mask
        )
        self._ticks = 0

    def choose(
        self,
        *target_rows: tuple[int, int],
        moves: Iterable[tuple[int, int]] = (),
        ultimates: Iterable[int] = (),
    ) -> Action:
        # The actors choose only actions their current mask allows.
        ultimate_slots = tuple(ultimates)
        for slot, target in target_rows:
            use_ultimate = int(slot in ultimate_slots)
            assert bool(
                self.action_mask.select_target_use_ultimate_joint_mask[
                    slot, target, use_ultimate
                ]
            )
        moves = tuple(moves)
        for slot, move in moves:
            assert bool(self.action_mask.move_mask[slot, move])
        return _joint_action(*target_rows, moves=moves, ultimates=ultimate_slots)

    def step(self, action: Action) -> _Tick:
        configured = np.asarray(self.config.agent_profile.active_mask)
        np.testing.assert_array_equal(
            np.asarray(
                self.observation.context_features[:, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH]
            ),
            np.where(
                configured, np.float32(self.config.team_deathmatch_red_zone_depth), 0
            ),
        )
        self._ticks += 1
        state, observation, reward, done_flags, action_mask, info = _compiled_step(
            self.config,
            self.state,
            self.action_mask,
            action,
            jax.random.key(self._ticks),
        )
        transition, self._frame = capture_evaluation_transition_unit_v3(
            self._context,
            self._frame,
            state,
            observation,
            action_mask,
            info.transition_facts,
            reward,
            done_flags,
        )
        self.state, self.observation, self.action_mask = state, observation, action_mask
        return _Tick(state, observation, reward, done_flags, info, transition)


def _scores(tick: _Tick) -> tuple[int, int]:
    team_a, team_b = np.asarray(tick.state.team_deathmatch_scores).tolist()
    return team_a, team_b


def _score_events(tick: _Tick) -> tuple[int, int]:
    # Captured score increments in Team A, Team B order; 0 without an event.
    increments = [0, 0]
    for event in tick.transition.events:
        if isinstance(event, TeamDeathmatchScoreChangedEventV1):
            increments[event.team_index] = event.score_increment
    return increments[0], increments[1]


def _newly_dead(tick: _Tick) -> tuple[int, ...]:
    return tuple(
        int(slot)
        for slot in np.flatnonzero(
            np.asarray(
                tick.info.transition_facts.death_facts.is_newly_dead_by_recipient
            )
        )
    )


def _assert_points(
    tick: _Tick, expected: tuple[int, int], *, start: tuple[int, int] = (0, 0)
) -> None:
    # Successor scores, captured score events and both teams' context columns.
    increments = (expected[0] - start[0], expected[1] - start[1])
    assert _scores(tick) == expected
    assert _score_events(tick) == increments
    # Slot 0 reads its own team's score first; slot 5 reads Team B's first.
    context = np.asarray(tick.observation.context_features)
    ally_then_enemy = slice(
        CONTEXT_FEATURE_TDM_ALLY_SCORE, CONTEXT_FEATURE_TDM_ENEMY_SCORE + 1
    )
    assert tuple(context[0, ally_then_enemy].tolist()) == expected
    assert (
        tuple(context[MAX_AGENTS_PER_TEAM, ally_then_enemy].tolist()) == expected[::-1]
    )


def _f32_after(value: float, direction: float) -> float:
    return float(np.nextafter(np.float32(value), np.float32(direction)))


_JUST_ABOVE_5 = _f32_after(5.0, np.inf)
_JUST_BELOW_7 = _f32_after(7.0, -np.inf)
_JUST_ABOVE_6 = _f32_after(6.0, np.inf)
_JUST_BELOW_6 = _f32_after(6.0, -np.inf)


@pytest.mark.parametrize(
    ("depth", "cases"),
    (
        pytest.param(
            0.0,
            (
                (0, 4.0, 1),
                (0, 5.0, 1),
                (0, _JUST_ABOVE_5, 1),
                (5, 8.0, 1),
                (5, 7.0, 1),
                (5, _JUST_BELOW_7, 1),
            ),
            id="depth-0-rule-off",
        ),
        pytest.param(
            5.0,
            (
                (0, 4.0, 2),
                (0, 5.0, 2),
                (0, _JUST_ABOVE_5, 1),
                (5, 8.0, 2),
                (5, 7.0, 2),
                (5, _JUST_BELOW_7, 1),
            ),
            id="depth-5",
        ),
        pytest.param(
            6.0,
            (
                (0, 5.0, 2),
                (0, 6.0, 2),
                (0, _JUST_ABOVE_6, 1),
                (5, 7.0, 2),
                (5, 6.0, 2),
                (5, _JUST_BELOW_6, 1),
            ),
            id="depth-6",
        ),
    ),
)
def test_deaths_inside_on_and_just_outside_each_boundary_score_by_depth(
    depth: float, cases: tuple[tuple[int, float, int], ...]
) -> None:
    # Each case: (victim slot, victim x, points the other team gains). The
    # shooter stands 3 map units east of a Team A victim or west of a Team B one.
    config = _task_config(red_zone_depth=depth)
    validate_env_config(config)
    for victim, victim_x, points in cases:
        shooter = MAX_AGENTS_PER_TEAM if victim == 0 else 0
        shooter_x = victim_x + 3.0 if victim == 0 else victim_x - 3.0
        game = _Game(
            config,
            low_health_slots=(victim,),
            positions={victim: (victim_x, 2.0), shooter: (shooter_x, 2.0)},
        )
        tick = game.step(game.choose((shooter, _ENEMY_0)))
        expected = (0, points) if victim == 0 else (points, 0)
        assert _newly_dead(tick) == (victim,)
        _assert_points(tick, expected)


def test_an_invader_killed_inside_the_enemy_strip_scores_one() -> None:
    # Team A's slot 0 stands in Team B's strip and Team B's slot 5 in Team A's.
    config = _task_config(team_sizes=(2, 2), red_zone_depth=5.0)
    game = _Game(
        config,
        low_health_slots=(0, 5),
        positions={0: (8.0, 2.0), 6: (11.0, 2.0), 5: (4.0, 6.0), 1: (1.0, 6.0)},
    )
    tick = game.step(game.choose((6, _ENEMY_0), (1, _ENEMY_0)))
    assert _newly_dead(tick) == (0, 5)
    _assert_points(tick, (1, 1))


def test_overlapping_and_full_width_strips_check_each_victim_against_its_own() -> None:
    # Team A's victim (slot 0, y = 2) and Team B's victim (slot 5, y = 6) die on
    # one tick, each shot by an enemy 3 map units away. Rows: (depth, Team A
    # victim x, its shooter's x, Team B victim x, its shooter's x, scores).
    for depth, team_a_x, team_b_shooter_x, team_b_x, team_a_shooter_x, expected in (
        (7.0, 6.0, 9.0, 6.0, 3.0, (2, 2)),
        (7.0, 7.5, 4.5, 4.5, 7.5, (1, 1)),
        (12.0, 11.0, 8.0, 1.0, 4.0, (2, 2)),
        (5.0, 11.0, 8.0, 1.0, 4.0, (1, 1)),
    ):
        config = _task_config(team_sizes=(2, 2), red_zone_depth=depth)
        validate_env_config(config)
        game = _Game(
            config,
            low_health_slots=(0, 5),
            positions={
                0: (team_a_x, 2.0),
                6: (team_b_shooter_x, 2.0),
                5: (team_b_x, 6.0),
                1: (team_a_shooter_x, 6.0),
            },
        )
        tick = game.step(game.choose((6, _ENEMY_0), (1, _ENEMY_0)))
        assert _newly_dead(tick) == (0, 5)
        _assert_points(tick, expected)


def test_swapped_banks_swap_the_strips() -> None:
    # (swap_banks, Team A victim x, Team B victim x, scores). Swapped, Team A's
    # strip is x >= 7 and Team B's is x <= 5.
    for swap_banks, team_a_x, team_b_x, expected in (
        (False, 4.0, 7.0, (2, 2)),
        (True, 4.0, 7.0, (1, 1)),
        (False, 8.0, 5.0, (1, 1)),
        (True, 8.0, 5.0, (2, 2)),
    ):
        config = _task_config(red_zone_depth=5.0, swap_banks=swap_banks)
        validate_env_config(config)
        game = _Game(
            config,
            low_health_slots=(0, 5),
            positions={0: (team_a_x, 2.0), 5: (team_b_x, 2.0)},
        )
        tick = game.step(game.choose((0, _ENEMY_0), (5, _ENEMY_0)))
        assert _newly_dead(tick) == (0, 5)
        _assert_points(tick, expected)


def test_both_teams_score_their_own_points_on_one_tick() -> None:
    # Team A's victim is inside its strip (2 points to Team B); Team B's victim
    # at x = 6.5 is outside its strip (1 point to Team A).
    config = _task_config(red_zone_depth=5.0)
    game = _Game(
        config, low_health_slots=(0, 5), positions={0: (4.0, 2.0), 5: (6.5, 2.0)}
    )
    tick = game.step(game.choose((0, _ENEMY_0), (5, _ENEMY_0)))
    assert _newly_dead(tick) == (0, 5)
    _assert_points(tick, (1, 2))
    _assert_task_result(
        tick.reward,
        tick.done_flags,
        tick.info,
        outcome=TASK_MODE_OUTCOME_ONGOING,
        terminated=False,
        truncated=False,
    )
    assert bool(jnp.all(tick.reward.rewards == 0.0))


def test_two_contributors_on_one_victim_give_one_red_zone_death() -> None:
    config = _task_config(team_sizes=(2, 1), red_zone_depth=5.0)
    game = _Game(config, low_health_slots=(5,), positions={5: (7.0, 3.0)})
    tick = game.step(game.choose((0, _ENEMY_0), (1, _ENEMY_0)))
    death_facts = tick.info.transition_facts.death_facts
    assert _newly_dead(tick) == (5,)
    assert np.flatnonzero(
        np.asarray(death_facts.contributed_to_new_death_by_source)
    ).tolist() == [0, 1]
    _assert_points(tick, (2, 0))
    events = tick.transition.events
    assert [
        event.recipient_global_slot
        for event in events
        if isinstance(event, AgentDiedEventV1)
    ] == [5]
    assert [
        (event.source_global_slot, event.recipient_global_slot)
        for event in events
        if isinstance(event, LethalDamageContributionEventV1)
    ] == [(0, 5), (1, 5)]
    assert [
        (
            event.team_id,
            event.score_increment,
            event.previous_score,
            event.successor_score,
        )
        for event in events
        if isinstance(event, TeamDeathmatchScoreChangedEventV1)
    ] == [(1, 2, 0, 2)]


def test_five_red_zone_deaths_reach_the_top_threshold_bound_exactly() -> None:
    # All five Team B Hunters stand on their strip's boundary x = 7.0.
    config = _task_config(
        team_sizes=(5, 5), score_threshold=_TOP_THRESHOLD, red_zone_depth=5.0
    )
    validate_env_config(config)
    with pytest.raises(ValueError, match="team_deathmatch_score_threshold must be in"):
        validate_env_config(
            config._replace(team_deathmatch_score_threshold=_TOP_THRESHOLD + 1)
        )
    game = _Game(
        config,
        low_health_slots=range(MAX_AGENTS_PER_TEAM, 2 * MAX_AGENTS_PER_TEAM),
        scores=(16_777_206, 0),
    )
    tick = game.step(
        game.choose(*((slot, _ENEMY_0 + slot) for slot in range(MAX_AGENTS_PER_TEAM)))
    )
    assert _newly_dead(tick) == (5, 6, 7, 8, 9)
    _assert_points(tick, (16_777_216, 0), start=(16_777_206, 0))
    context = np.asarray(tick.observation.context_features)
    assert context[0, CONTEXT_FEATURE_TDM_ALLY_SCORE] == np.float32(16_777_216.0)
    assert int(context[0, CONTEXT_FEATURE_TDM_ALLY_SCORE]) == 2**24
    assert validate_env_state(config, tick.state) is None
    _assert_task_result(
        tick.reward,
        tick.done_flags,
        tick.info,
        outcome=TASK_MODE_OUTCOME_TEAM_A_WIN,
        terminated=True,
        truncated=False,
    )


def test_a_corpse_scores_nothing_and_a_respawned_victim_scores_two_again() -> None:
    # Team A's slot 1 waits near Team B's pad at (10, 2), where slot 5 respawns.
    config = _task_config(team_sizes=(2, 1), max_steps=40, red_zone_depth=5.0)
    game = _Game(config, low_health_slots=(5,), positions={1: (7.5, 4.0)})
    first_death = game.step(game.choose((0, _ENEMY_0)))
    assert _newly_dead(first_death) == (5,)
    _assert_points(first_death, (2, 0))

    # Shooting the corpse is not allowed, so the action is rejected: no points.
    assert not bool(game.action_mask.select_target_mask[0, _ENEMY_0])
    corpse = game.step(_joint_action((0, _ENEMY_0)))
    assert _newly_dead(corpse) == ()
    _assert_points(corpse, (2, 0), start=(2, 0))
    assert [
        event.actor_global_slot
        for event in corpse.transition.events
        if isinstance(event, ActionRejectedEventV1)
    ] == [0]

    # Wait for Team B's respawn wave; a respawn is not a death.
    respawn: _Tick | None = None
    for _ in range(config.max_steps):
        tick = game.step(game.choose())
        _assert_points(tick, (2, 0), start=(2, 0))
        respawn_facts = tick.info.transition_facts.respawn_facts
        if bool(respawn_facts.was_respawned_this_transition_by_agent[5]):
            respawn = tick
            break
    assert respawn is not None
    assert _newly_dead(respawn) == ()
    assert np.asarray(respawn.state.agent_positions[5]).tolist() == [10.0, 2.0]

    # Slot 1 shoots the respawned victim, 6 health per hit, until it dies.
    second_death: _Tick | None = None
    for _ in range(config.max_steps):
        tick = game.step(game.choose((1, _ENEMY_0)))
        if _newly_dead(tick):
            second_death = tick
            break
        _assert_points(tick, (2, 0), start=(2, 0))
    assert second_death is not None
    assert _newly_dead(second_death) == (5,)
    _assert_points(second_death, (4, 0), start=(2, 0))


def test_points_use_the_victim_s_position_when_combat_resolves() -> None:
    config = _task_config(red_zone_depth=5.0)
    # A Team B victim on its boundary walks west out of its strip: 2 points.
    game = _Game(config, low_health_slots=(5,), positions={5: (7.0, 2.0)})
    tick = game.step(game.choose((0, _ENEMY_0), moves=((5, MOVE_WEST),)))
    assert np.asarray(tick.state.agent_positions[5]).tolist() == [6.0, 2.0]
    _assert_points(tick, (2, 0))
    # A Team B victim just outside walks east into its strip: 1 point.
    game = _Game(
        config, low_health_slots=(5,), positions={5: (6.5, 2.0), 0: (3.5, 2.0)}
    )
    tick = game.step(game.choose((0, _ENEMY_0), moves=((5, MOVE_EAST),)))
    assert np.asarray(tick.state.agent_positions[5]).tolist() == [7.5, 2.0]
    _assert_points(tick, (1, 0))

    # A dying Warrior inside Team A's strip Charges to x = 7.0: still 2 points.
    warrior_config = _task_config(
        red_zone_depth=5.0, class_rows=((0, WARRIOR_CLASS_ID),)
    )
    game = _Game(
        warrior_config,
        low_health_slots=(0,),
        positions={0: (4.5, 6.0), 5: (8.0, 6.0)},
    )
    tick = game.step(game.choose((0, _ENEMY_0), (5, _ENEMY_0), ultimates=(0,)))
    charge = (
        tick.info.transition_facts.physical_facts.charge_phase_displacement_by_agent
    )
    assert np.asarray(charge[0]).tolist() == [2.5, 0.0]
    assert np.asarray(tick.state.agent_positions[0]).tolist() == [7.0, 6.0]
    assert _newly_dead(tick) == (0,)
    _assert_points(tick, (0, 2))


def test_threshold_overshoot_with_red_zone_points_picks_the_higher_score() -> None:
    # K = 2 from (1, 1). Team A's victim at x = 4 is inside its strip (Team B
    # +2); Team B's victim at x = 6.5 is outside (+1) and at x = 7.0 inside (+2).
    config = _task_config(score_threshold=2, max_steps=10, red_zone_depth=5.0)
    for team_b_x, step_count, expected, outcome, truncated, basis in (
        (6.5, 0, (2, 3), TASK_MODE_OUTCOME_TEAM_B_WIN, False, "score_threshold"),
        (7.0, 9, (3, 3), TASK_MODE_OUTCOME_DRAW, True, "score_threshold_at_horizon"),
    ):
        game = _Game(
            config,
            low_health_slots=(0, 5),
            positions={0: (4.0, 2.0), 5: (team_b_x, 2.0)},
            scores=(1, 1),
            step_count=step_count,
        )
        tick = game.step(game.choose((0, _ENEMY_0), (5, _ENEMY_0)))
        _assert_points(tick, expected, start=(1, 1))
        _assert_task_result(
            tick.reward,
            tick.done_flags,
            tick.info,
            outcome=outcome,
            terminated=True,
            truncated=truncated,
        )
        team_b_reward = 1.0 if outcome == TASK_MODE_OUTCOME_TEAM_B_WIN else 0.0
        assert np.asarray(tick.reward.rewards)[[0, 5]].tolist() == [
            -team_b_reward,
            team_b_reward,
        ]
        assert [
            (event.outcome, event.completion_basis)
            for event in tick.transition.events
            if isinstance(event, TeamDeathmatchCompletedEventV1)
        ] == [("team_b_win" if team_b_reward else "draw", basis)]


def test_depth_is_public_at_reset_and_scoring_agrees_under_jit_vmap_and_scan() -> None:
    zero_config = _task_config(red_zone_depth=0.0)
    config = _task_config(red_zone_depth=5.0)

    # Before any score differs, the depth is the only public difference.
    compiled_reset = cast(_Reset, jax.jit(reset))
    zero_start = compiled_reset(zero_config, jax.random.key(3))
    start = compiled_reset(config, jax.random.key(3))
    for zero_leaf, leaf in zip(
        jax.tree.leaves(zero_start[1]._replace(context_features=jnp.zeros(()))),
        jax.tree.leaves(start[1]._replace(context_features=jnp.zeros(()))),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(zero_leaf), np.asarray(leaf))
    differs = np.asarray(start[1].context_features != zero_start[1].context_features)
    assert np.argwhere(differs).tolist() == [[0, 19], [5, 19]]
    assert np.asarray(start[0].team_deathmatch_scores).tolist() == [0, 0]

    # Team B's victim stands on its boundary x = 7.0; Team A's slot 0 shoots.
    zero_state, _, zero_mask, _ = _scenario(zero_config, low_health_slots=(5,))
    state, _, action_mask, _ = _scenario(config, low_health_slots=(5,))
    action = _joint_action((0, _ENEMY_0))
    eager = _take_step(config, state, action_mask, action)
    assert np.asarray(eager[0].team_deathmatch_scores).tolist() == [2, 0]
    compiled = _compiled_step(config, state, action_mask, action, jax.random.key(1))
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(eager_leaf), np.asarray(compiled_leaf))

    def stack(*leaves: object) -> Array:
        return jnp.stack(tuple(jnp.asarray(leaf) for leaf in leaves))

    batched_step = cast(_Step, jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None, 0))))
    batched = batched_step(
        cast(EnvConfig, jax.tree.map(stack, zero_config, config)),
        cast(EnvState, jax.tree.map(stack, zero_state, state)),
        cast(ActionMask, jax.tree.map(stack, zero_mask, action_mask)),
        action,
        jax.random.split(jax.random.key(4), 2),
    )
    assert np.asarray(batched[0].team_deathmatch_scores).tolist() == [[1, 0], [2, 0]]
    assert np.asarray(
        batched[1].context_features[:, 0, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH]
    ).tolist() == [0.0, 5.0]

    def scan_step(
        carry: tuple[EnvState, ActionMask], key: Array
    ) -> tuple[tuple[EnvState, ActionMask], tuple[EnvState, Observation, Info]]:
        current_state, current_mask = carry
        next_state, observation, _, _, next_mask, info = step(
            config, current_state, current_mask, action, key
        )
        return (next_state, next_mask), (next_state, observation, info)

    _, (states, observations, infos) = jax.lax.scan(
        scan_step, (state, action_mask), jax.random.split(jax.random.key(5), 3)
    )
    assert np.asarray(states.team_deathmatch_scores).tolist() == [[2, 0]] * 3
    assert np.asarray(
        infos.transition_facts.death_facts.is_newly_dead_by_recipient[:, 5]
    ).tolist() == [True, False, False]
    assert np.asarray(
        observations.context_features[:, 0, CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH]
    ).tolist() == [5.0, 5.0, 5.0]
