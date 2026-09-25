"""Check the 44 full Red Zone metric columns against real Core transitions.

A Red Zone death is a new death with the victim's centre inside its own team's
Red Zone when combat resolves. The full metrics count it once for the victim
and once for each helper that Core's kill-credit owner already names (damage
or useful Priest healing on the death tick). The enemy team gets 2 points, but
kills and deaths still count once. The contracts checked here:

- one victim and two helpers give 2 points, one team Red Zone kill, one victim
  death, two helper counts and shares of 1 and 1; excess-only healing earns no
  help;
- several victims on one tick are each checked against their own team's
  strip, both boundaries are inside, an invader killed in the enemy strip is
  not a Red Zone death, Team A Red Zone kills equal Team B Red Zone deaths and
  death shares add up to 1;
- a respawned agent that dies in its Red Zone again counts again;
- inactive slots and zero denominators are unavailable, and a no-transition
  update changes nothing;
- the neutral task marks all 44 columns unavailable; Team Deathmatch at depth
  0.0 gives valid zero counts and unavailable shares;
- a historical replay without a recorded Red Zone rule marks all 44 columns
  unavailable and keeps every older column equal to direct computation;
- live evaluation, the saved full_metrics.csv and replay analysis agree for the
  entire captured episode when the capture starts after simulator tick 0;
- up to each captured tick, replay analysis equals a recount of the recorded
  frames: victims through Core's red_zone_death_mask, helper counts from the
  reconstructed kill contributions to those victims, kill participation as an
  agent's helper count over its team's Red Zone kills, and death share as an
  agent's Red Zone deaths over its team's (each share unavailable while its
  team total is 0); the cursor CSV matches the cursor table;
- priority and none metric modes keep no full counters.
"""

import csv
import io
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import (
    captured_evaluation_trajectory,
    evaluation_env_config,
    neutral_action,
)
from tests.full_metric_fixtures import (
    MetricRun,
    actions,
    advance,
    assert_missing,
    start_run,
    update_metrics,
    value,
    values,
)
from tests.test_evaluation_replay import runtime_provenance as runtime_provenance

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.axis_mappings import global_slot_to_target_action
from marl_battlegrounds.core.env import (
    build_canonical_no_transition_info_object,
    red_zone_death_mask,
    reset,
)
from marl_battlegrounds.core.types import ActionMask, EnvState, Info, Reward
from marl_battlegrounds.evaluation.analysis import analyze_replay
from marl_battlegrounds.evaluation.capture import (
    reconstruct_env_state_v1,
    reconstruct_transition_facts_v1,
)
from marl_battlegrounds.evaluation.catalog import reconstruct_env_config_v1
from marl_battlegrounds.evaluation.combat_metrics import combat_quantities
from marl_battlegrounds.evaluation.episode_metrics import (
    initialize_priority,
    priority_values,
    update_priority,
)
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.full_metrics import (
    full_values,
    initialize_full,
    update_full,
)
from marl_battlegrounds.evaluation.metric_catalog import METRIC_COLUMNS
from marl_battlegrounds.evaluation.policy_execution import Policy, PolicyTree
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.evaluation.replay_v2 import build_replay_v2
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput

RED_ZONE_NAMES = tuple(
    column.name for column in METRIC_COLUMNS if column.family == "red_zone"
)


def _place(
    positions: dict[int, tuple[float, float]], health: dict[int, float]
) -> Callable[[EnvState], EnvState]:
    def arrange(state: EnvState) -> EnvState:
        placed = state.agent_positions
        for slot, xy in positions.items():
            placed = placed.at[slot].set(xy)
        current = state.current_health
        for slot, amount in health.items():
            current = current.at[slot].set(amount)
        return state._replace(agent_positions=placed, current_health=current)

    return arrange


@pytest.mark.parametrize("useful", (True, False), ids=("useful", "excess"))
def test_one_victim_with_two_helpers_gives_two_points_one_kill_and_full_shares(
    useful: bool,
) -> None:
    # Team A: Mage 0, Warrior 1, Priest 2. Team B: Hunter 5, Rogue 6.
    # The Hunter stands inside Team B's strip (x >= 15 at depth 5).
    run = start_run(
        red_zone_depth=5.0,
        arrange=_place(
            {0: (13.0, 1.5), 2: (13.0, 4.0), 5: (16.0, 1.5)},
            {0: 79 if useful else 80, 5: 1},
        ),
    )
    run, info = advance(run, actions((0, 5, False), (2, 0, False)))
    assert info.transition_facts.death_facts.is_newly_dead_by_recipient[5]
    result = values(run)
    assert value(result, "team_a_score") == 2
    assert value(result, "team_a_kills") == 1
    assert value(result, "team_a_red_zone_kills") == 1
    assert value(result, "team_b_red_zone_deaths") == 1
    assert value(result, "agent_5_red_zone_deaths") == 1
    assert value(result, "agent_5_red_zone_death_fraction") == 1
    assert value(result, "agent_0_red_zone_kill_contributions") == 1
    assert value(result, "agent_0_red_zone_kill_participation") == 1
    # Useful healing of the attacker on the death tick is help; excess is not.
    assert value(result, "agent_2_red_zone_kill_contributions") == int(useful)
    assert value(result, "agent_2_red_zone_kill_participation") == int(useful)
    assert value(result, "agent_1_red_zone_kill_participation") == 0
    shares = [
        value(result, f"agent_{slot}_red_zone_kill_participation") for slot in (0, 1, 2)
    ]
    assert all(0 <= share <= 1 for share in shares)
    assert sum(shares) == (2 if useful else 1)
    assert value(result, "team_b_red_zone_kills") == 0
    assert value(result, "team_a_red_zone_deaths") == 0
    assert value(result, "agent_6_red_zone_deaths") == 0
    # Zero denominators are unavailable, never zero shares.
    for slot in (0, 1, 2):
        assert_missing(result, f"agent_{slot}_red_zone_death_fraction")
    for slot in (5, 6):
        assert_missing(result, f"agent_{slot}_red_zone_kill_participation")
    # Inactive slots have no Red Zone measurements at all.
    for slot in (3, 4, 7, 8, 9):
        for stem in (
            "red_zone_kill_contributions",
            "red_zone_kill_participation",
            "red_zone_deaths",
            "red_zone_death_fraction",
        ):
            assert_missing(result, f"agent_{slot}_{stem}")


def test_several_victims_use_their_own_strips_and_inclusive_boundaries() -> None:
    # Five Mages a side. Victims 5 (x = 15.0, boundary) and 6 are inside Team B's
    # strip; 7 is outside it. Victim 3 is on Team A's boundary (x = 5.0); victim
    # 4 is a Team A invader killed inside Team B's strip, which is not its own.
    run = start_run(
        team_sizes=(5, 5),
        classes=tuple((slot, 1) for slot in range(10)),
        red_zone_depth=5.0,
        arrange=_place(
            {
                0: (12.0, 1.5),
                5: (15.0, 1.5),
                1: (13.5, 4.5),
                6: (16.5, 4.5),
                2: (11.5, 7.5),
                7: (14.5, 7.5),
                8: (8.0, 10.5),
                3: (5.0, 10.5),
                9: (18.0, 7.5),
                4: (18.0, 10.5),
            },
            dict.fromkeys((3, 4, 5, 6, 7), 1.0),
        ),
    )
    run, info = advance(
        run,
        actions(
            (0, 5, False), (1, 6, False), (2, 7, False), (8, 3, False), (9, 4, False)
        ),
    )
    np.testing.assert_array_equal(
        info.transition_facts.death_facts.is_newly_dead_by_recipient,
        [False, False, False, True, True, True, True, True, False, False],
    )
    np.testing.assert_array_equal(run.state.team_deathmatch_scores, [5, 3])
    result = values(run)
    assert value(result, "team_a_kills") == 3
    assert value(result, "team_b_kills") == 2
    assert value(result, "team_a_red_zone_kills") == 2
    assert value(result, "team_b_red_zone_deaths") == 2
    assert value(result, "team_b_red_zone_kills") == 1
    assert value(result, "team_a_red_zone_deaths") == 1
    expected_deaths = {3: 1, 4: 0, 5: 1, 6: 1, 7: 0}
    for slot, count in expected_deaths.items():
        assert value(result, f"agent_{slot}_red_zone_deaths") == count
    for team in (range(5), range(5, 10)):
        fractions = [
            value(result, f"agent_{slot}_red_zone_death_fraction") for slot in team
        ]
        assert sum(fractions) == pytest.approx(1, abs=1e-6)
    assert value(result, "agent_5_red_zone_death_fraction") == 0.5
    assert value(result, "agent_3_red_zone_death_fraction") == 1
    for slot, count, share in (
        (0, 1, 0.5),
        (1, 1, 0.5),
        (2, 0, 0),
        (8, 1, 1),
        (9, 0, 0),
    ):
        assert value(result, f"agent_{slot}_red_zone_kill_contributions") == count
        assert value(result, f"agent_{slot}_red_zone_kill_participation") == share
    padding = build_canonical_no_transition_info_object(run.state)
    padded = update_metrics(run.full, run.config, run.state, run.mask, padding)
    for before, after in zip(
        jax.tree.leaves(run.full), jax.tree.leaves(padded), strict=True
    ):
        np.testing.assert_array_equal(before, after)


def test_a_respawned_agent_that_dies_in_its_red_zone_again_counts_again() -> None:
    run = start_run(
        team_sizes=(1, 1),
        red_zone_depth=5.0,
        arrange=_place({0: (15.5, 1.5), 5: (17.5, 1.5)}, {5: 1}),
    )
    run = run._replace(config=run.config._replace(max_steps=64))
    attack = actions((0, 5, False))
    target = global_slot_to_target_action(0, 5)
    returned = False
    for _ in range(60):
        legal = bool(run.mask.select_target_use_ultimate_joint_mask[0, target, 0])
        run, info = advance(run, attack if legal else neutral_action())
        returned |= bool(
            info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
                5
            ]
        )
        if value(values(run), "agent_5_deaths") == 2:
            break
    assert returned
    result = values(run)
    assert value(result, "agent_5_deaths") == 2
    assert value(result, "agent_5_red_zone_deaths") == 2
    assert value(result, "team_b_red_zone_deaths") == 2
    assert value(result, "team_a_red_zone_kills") == 2
    assert value(result, "agent_0_red_zone_kill_contributions") == 2
    assert value(result, "agent_0_red_zone_kill_participation") == 1
    assert value(result, "team_a_score") == 4


def test_neutral_task_is_unavailable_and_depth_zero_gives_zero_counts() -> None:
    config = evaluation_env_config()
    state, _, mask, _ = reset(config, jax.random.key(0))
    neutral = MetricRun(
        config,
        state,
        mask,
        initialize_full(config, state),
        initialize_priority(),
        state.step_count,
        jnp.asarray(0, jnp.int32),
    )
    neutral, _ = advance(neutral, neutral_action())
    result = values(neutral)
    for name in RED_ZONE_NAMES:
        assert_missing(result, name)

    run = start_run(
        red_zone_depth=0.0,
        arrange=_place({0: (13.0, 1.5), 5: (16.0, 1.5)}, {5: 1}),
    )
    run, info = advance(run, actions((0, 5, False)))
    assert info.transition_facts.death_facts.is_newly_dead_by_recipient[5]
    result = values(run)
    assert value(result, "team_a_score") == 1
    for name in (
        "team_a_red_zone_kills",
        "team_b_red_zone_deaths",
        "agent_5_red_zone_deaths",
        "agent_0_red_zone_kill_contributions",
    ):
        assert value(result, name) == 0
    assert_missing(result, "agent_0_red_zone_kill_participation")
    assert_missing(result, "agent_5_red_zone_death_fraction")


def test_historical_replay_marks_red_zone_unavailable_and_keeps_older_columns(
    runtime_provenance: RuntimeProvenanceV1,
) -> None:
    # A Team Deathmatch capture with a historical V1 context: it has no Red Zone
    # rule, so reconstruction uses depth 0.0 and direct counts are valid zeros.
    trajectory = captured_evaluation_trajectory(
        config=evaluation_env_config(task_mode=1, team_deathmatch_score_threshold=20),
        transition_count=2,
        expected_horizon=100,
    )
    replay = build_replay_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=runtime_provenance,
    )
    analysis = analyze_replay(LoadedReplay(replay, None, "not_recorded"), full=True)
    config = reconstruct_env_config_v1(trajectory.context)
    initial = reconstruct_env_state_v1(trajectory.frames[0])
    full = initialize_full(config, initial)
    priority = initialize_priority()
    outcome = jnp.asarray(0, jnp.int32)
    for start, transition in zip(
        trajectory.frames, trajectory.transitions, strict=False
    ):
        info = Info(reconstruct_transition_facts_v1(transition.facts))
        mask = ActionMask(
            *(
                jnp.asarray(getattr(start.action_mask, name), bool)
                for name in ActionMask._fields
            )
        )
        full = update_full(full, config, reconstruct_env_state_v1(start), mask, info)
        priority = update_priority(
            priority,
            Reward(jnp.asarray(transition.canonical_reward_by_agent, jnp.float32)),
            info,
        )
        outcome = info.transition_facts.team_deathmatch_facts.outcome
    final = reconstruct_env_state_v1(trajectory.frames[-1])
    direct = full_values(
        full,
        config,
        priority_values(config, final, initial.step_count, priority, outcome),
    )
    statistics = cast(
        list[dict[str, object]],
        analysis.summary(0, scope="final")["statistics"],
    )
    for index, (column, scalar) in enumerate(
        zip(METRIC_COLUMNS, statistics, strict=True)
    ):
        if column.family == "red_zone":
            active = column.scope == "team" or bool(
                trajectory.context.roster[column.subjects[0]].configured_active
            )
            # Direct depth-0 counts would be valid zeros; analysis shows dashes.
            assert bool(direct.valid[index]) == (
                active and column.denominator is None
            ), column.name
            assert scalar["valid"] is False and scalar["value"] is None, column.name
            assert scalar["applicable"] == active, column.name
            continue
        assert scalar["valid"] == bool(direct.valid[index]), column.name
        if scalar["valid"]:
            assert scalar["value"] == pytest.approx(
                float(direct.values[index]), rel=1e-5, abs=1e-5
            ), column.name
    row = next(csv.DictReader(io.StringIO(analysis.csv(0, scope="final"))))
    assert all(row[name] == "" for name in RED_ZONE_NAMES)


def _attack_first_enemy(
    variables: PolicyTree,
    carry: PolicyTree,
    actor_input: ActorInput,
    action_mask: ActionMask,
    key: Array,
) -> tuple[ActorAction, PolicyTree]:
    # Stay still and use the Basic ability on the first legal enemy row, if any.
    enemy = action_mask.select_target_use_ultimate_joint_mask[6:, 0]
    target = jnp.where(enemy.any(), 6 + jnp.argmax(enemy), 0).astype(jnp.int32)
    return ActorAction(jnp.int32(0), target, jnp.int32(0)), carry


def test_live_saved_csv_and_replay_analysis_agree_after_a_late_capture_start(
    tmp_path: Path,
) -> None:
    config = evaluation_env_config(
        task_mode=1,
        team_deathmatch_score_threshold=20,
        team_deathmatch_red_zone_depth=5.0,
        max_steps=8,
    )
    state, _, _, _ = reset(config, jax.random.key(0))
    # Mage 0 kills Hunter 5 on Team B's boundary (x = 15.0); Warrior 1 kills
    # Rogue 6 outside Team B's strip. Each attacker has exactly one legal enemy.
    arrange = _place(
        {0: (12.0, 1.5), 5: (15.0, 1.5), 1: (12.0, 5.5), 6: (13.5, 5.5)},
        {5: 1, 6: 1},
    )
    state = arrange(state)._replace(step_count=jnp.int32(5))
    method = Policy("attack-first", _attack_first_enemy)
    episodes = (EpisodeSpec(1, config, initial_state=state),)
    live = evaluate_episodes(
        method, method, episodes, num_envs=1, metrics="full", replay_episodes=[1]
    )
    saved = evaluate_episodes(
        method,
        method,
        episodes,
        num_envs=1,
        metrics="full",
        output_dir=tmp_path / "run",
    )
    assert saved.paths is not None
    with saved.paths["full_metrics"].open(newline="") as stream:
        (saved_row,) = list(csv.DictReader(stream))
    table = saved.table("full_metrics")
    analysis = analyze_replay(
        LoadedReplay(live.replays[0], None, "not_recorded"), full=True
    )
    final = {
        str(row["name"]): row
        for row in cast(
            list[dict[str, object]], analysis.summary(0, scope="final")["statistics"]
        )
    }
    exported = next(csv.DictReader(io.StringIO(analysis.csv(0, scope="final"))))
    assert _value_or_none(final["team_a_red_zone_kills"]) == 1
    assert _value_or_none(final["agent_5_red_zone_death_fraction"]) == 1
    for name in RED_ZONE_NAMES:
        memory = float(live.full_metrics[name][0])
        stored = None if saved_row[name] == "" else float(saved_row[name])
        read = float(table[name][0])
        replayed = _value_or_none(final[name])
        if replayed is None:
            assert np.isnan(memory) and stored is None and np.isnan(read), name
            assert exported[name] == "", name
        else:
            assert memory == stored == read == replayed == float(exported[name]), name

    # Up to Current Tick: every captured prefix equals a recount of the recorded
    # frames. Victims come from Core's own classifier; helpers are the
    # reconstructed kill contributions to those victims.
    replay = live.replays[0]
    assert replay.frames[0].simulator_step_count == 5
    recorded = reconstruct_env_config_v1(replay.header.context)
    counts = np.zeros(10, np.int32)
    helpers = np.zeros(10, np.int32)
    for index in range(analysis.frame_count):
        if index:
            facts = reconstruct_transition_facts_v1(replay.transitions[index - 1].facts)
            start_frame = replay.frames[index - 1]
            start = reconstruct_env_state_v1(start_frame)
            victims = red_zone_death_mask(
                recorded,
                start.agent_positions,
                facts.death_facts.is_newly_dead_by_recipient,
            )
            counts += np.asarray(victims, np.int32)
            mask = ActionMask(
                *(
                    jnp.asarray(getattr(start_frame.action_mask, name), bool)
                    for name in ActionMask._fields
                )
            )
            helped = combat_quantities(recorded, start, mask, Info(facts))
            helpers += np.asarray(
                (helped.kill_contributions & victims[None, :]).sum(axis=1), np.int32
            )
        cursor = {
            str(row["name"]): row
            for row in cast(
                list[dict[str, object]], analysis.summary(index)["statistics"]
            )
        }
        cursor_csv = next(csv.DictReader(io.StringIO(analysis.csv(index))))
        for slot in (0, 1, 2, 5, 6):
            assert (
                _value_or_none(cursor[f"agent_{slot}_red_zone_deaths"]) == counts[slot]
            )
        assert _value_or_none(cursor["team_a_red_zone_kills"]) == counts[5:].sum()
        assert _value_or_none(cursor["team_b_red_zone_deaths"]) == counts[5:].sum()
        for slot in (0, 1, 2, 5, 6):
            own = slice(0, 5) if slot < 5 else slice(5, 10)
            enemy = slice(5, 10) if slot < 5 else slice(0, 5)
            team_kills, team_deaths = counts[enemy].sum(), counts[own].sum()
            assert (
                _value_or_none(cursor[f"agent_{slot}_red_zone_kill_contributions"])
                == helpers[slot]
            )
            for stem, total, share in (
                ("red_zone_kill_participation", team_kills, helpers[slot]),
                ("red_zone_death_fraction", team_deaths, counts[slot]),
            ):
                expected = None if total == 0 else pytest.approx(share / total)
                assert _value_or_none(cursor[f"agent_{slot}_{stem}"]) == expected
        for name in RED_ZONE_NAMES:
            number = _value_or_none(cursor[name])
            assert cursor_csv[name] == ("" if number is None else str(number)), name
    assert counts.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert helpers.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _value_or_none(row: dict[str, object]) -> float | None:
    return None if row["value"] is None else float(cast(float, row["value"]))


def test_priority_and_none_modes_keep_no_full_counters() -> None:
    for metrics in ("priority", "none", "full"):
        env = marl_bgs.make("tdm", map_id=42, num_envs=2, metrics=metrics)
        _, state = env.reset(jax.random.key(3))
        if metrics == "full":
            assert state.full is not None
            for name in ("red_zone_deaths", "red_zone_kill_contributions"):
                assert state.full[name].shape == (2, 10)
                assert state.full[name].dtype == jnp.int32
                assert int(state.full[name].sum()) == 0
        else:
            assert state.full is None
