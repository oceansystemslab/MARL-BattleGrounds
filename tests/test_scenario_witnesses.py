"""Check the packaged scenario solutions turn by turn and against the Red Zones.

The replay check compares saved per-turn results for every packaged solution
except Scenario 3, which uses its own solution and missed-move tests. Older
authoring revisions and their ablation cases remain separate.

A second check reads all nine witness files, Scenario 3 included, beside the
installed scenario records, without running the simulator. Each witness names
its installed scenario's revisions and digests, and the scenario declares Red
Zone depth 5.0. The check states the Red Zone rule again on its own: a team is
on the right when the exact sum of its five float32 pad x values is greater
than five times half the float32 map width; the left strip is x from 0 to d32
and the right strip is x from float32(w32 - d32) to w32, both ends included.
A counted position is where an agent alive at the start of a tick stands at
the start (the authored start, or the last tick's position) and after the
tick. Every counted position is outside both teams' strips, no newly dead
agent started inside its own team's strip, and every score change equals
those deaths' points. The only positions allowed inside a strip belong to
agents standing on their own spawn pad after a respawn; each line's respawns
are listed. The restated rule matches the host copy in evaluation.models.
"""

import json
from collections.abc import Callable
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.scenario_witness_assertions import assert_witness_tick

from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    TEAM_B_ID,
    ActionMask,
    DoneFlags,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.models import (
    red_zone_team_on_right,
    red_zone_x_range,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
    reactive_tdm_alpha_policy,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
    reactive_tdm_beta_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)
from marl_battlegrounds.tasks import load_tdm_scenario

_TEAM = cast(
    Callable[..., ActorAction],
    jax.jit(execute_shared_obs_team_policy, static_argnums=(5, 6)),
)
_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)
_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("scenario_id", "case_name"),
    [
        (1, "scenario_1_heal_hunter"),
        (1, "scenario_1_heal_rogue"),
        (2, "scenario_2"),
        (4, "scenario_4"),
        (5, "scenario_5"),
        (6, "scenario_6"),
        (7, "scenario_7"),
        (8, "scenario_8"),
    ],
)
def test_packaged_solution_preserves_every_turn(
    scenario_id: int, case_name: str
) -> None:
    expected = json.loads((_FIXTURES / f"{case_name}_witness.json").read_text())
    scenario = load_tdm_scenario(scenario_id)
    assert expected["scenario_id"] == scenario_id
    assert scenario.info.approved_source.revision == expected["approved_revision"]
    assert scenario.info.source.revision == expected["source_revision"]
    assert scenario.info.source.semantic_digest == expected["semantic_digest"]
    assert (
        scenario.info.resolved_configuration_digest == expected["configuration_digest"]
    )
    assert (
        scenario.info.resolved_initial_state_digest == expected["initial_state_digest"]
    )
    beta = scenario_id in (5, 8)
    descriptor = (
        reactive_tdm_beta_controller_descriptor()
        if beta
        else reactive_tdm_alpha_controller_descriptor()
    )
    assert expected["opponent"] == ("tdm-beta" if beta else "tdm-alpha")
    assert expected["opponent_version"] == descriptor["version"]
    constant_key = scenario_id in (1, 2, 4)
    assert expected["key_schedule"] == (
        "constant_zero" if constant_key else "turn_index"
    )
    opponent = reactive_tdm_beta_policy if beta else reactive_tdm_alpha_policy

    state, observation, mask, _ = initialize_scenario_state(
        scenario.initial_state, scenario.config
    )
    profile = scenario.config.agent_profile
    availability = build_default_shared_obs_information_availability(
        profile.active_mask, profile.team_ids
    )
    rows = expected["ticks"]
    assert rows, f"{case_name}: missing expected turns"
    slots = np.arange(10)
    for tick, row in enumerate(rows, start=1):
        assert row["tick"] == tick
        key = jax.random.key(0 if constant_key else tick - 1)
        # Expected opponent actions are assertions only, never policy inputs.
        team_b = _TEAM(
            observation,
            mask,
            jax.random.split(key, 10),
            build_shared_obs_sensor_source_bank(observation),
            availability,
            opponent,
            TEAM_B_ID,
        )
        team_a = ActorAction(
            *(
                jnp.asarray(row[head][:5], dtype=jnp.int32)
                for head in ActorAction._fields
            )
        )
        action = build_joint_action_from_actor_actions(team_a, team_b)
        assert np.asarray(mask.move_mask)[slots, np.asarray(action.move)].all()
        assert np.asarray(mask.select_target_use_ultimate_joint_mask)[
            slots, np.asarray(action.select_target), np.asarray(action.use_ultimate)
        ].all()
        before_step = int(state.step_count)
        state, observation, reward, done, mask, info = _STEP(
            scenario.config, state, mask, action, key
        )
        assert int(state.step_count) == before_step + 1
        assert_witness_tick(
            row,
            state=state,
            action=action,
            reward=reward,
            done=done,
            info=info,
            case_name=case_name,
        )
        # No padding or extra turns may hide an early or late end.
        assert bool(done.done) == (tick == len(rows))

    np.testing.assert_array_equal(state.team_deathmatch_scores, [20, 19])


# Pad respawns in each witness line, by tick: the only Red Zone exemption.
_PAD_RESPAWNS: dict[str, dict[int, tuple[int, ...]]] = {
    "scenario_3": {5: (0, 3, 4, 5, 6, 7, 9)},
    "scenario_5": {5: (2, 3, 6, 7, 9)},
    "scenario_6": {5: (1, 3, 5, 7, 8, 9)},
    "scenario_8": {5: (0, 2, 5, 6, 7, 9)},
}


def _float32(value: float) -> float:
    return float(np.float32(value))


@pytest.mark.parametrize(
    ("scenario_id", "case_name"),
    [
        (1, "scenario_1_heal_hunter"),
        (1, "scenario_1_heal_rogue"),
        (2, "scenario_2"),
        (3, "scenario_3"),
        (4, "scenario_4"),
        (5, "scenario_5"),
        (6, "scenario_6"),
        (7, "scenario_7"),
        (8, "scenario_8"),
    ],
)
def test_witness_positions_stay_outside_red_zones(
    scenario_id: int, case_name: str
) -> None:
    expected = json.loads((_FIXTURES / f"{case_name}_witness.json").read_text())
    package = files("marl_battlegrounds").joinpath("data", "tdm")
    info = json.loads(package.joinpath("manifest.json").read_bytes())["scenarios"][
        scenario_id - 1
    ]
    content = json.loads(
        package.joinpath("scenarios", f"{scenario_id}.json").read_bytes()
    )
    configuration, start = content["configuration"], content["initial_snapshot"]
    assert expected["scenario_id"] == info["scenario_id"] == scenario_id
    assert expected["source_revision"] == info["source"]["revision"]
    assert expected["approved_revision"] == info["approved_source"]["revision"]
    assert expected["semantic_digest"] == info["source"]["semantic_digest"]
    assert (
        expected["configuration_digest"]
        == info["resolved_configuration_digest"]
        == configuration["canonical_digest_sha256"]
    )
    assert expected["initial_state_digest"] == info["resolved_initial_state_digest"]
    assert configuration["schema_version"] == 2
    assert configuration["team_deathmatch_red_zone_depth"] == 5.0

    width = np.float32(configuration["map_width"])
    depth = np.float32(configuration["team_deathmatch_red_zone_depth"])
    pads = configuration["team_spawn_pad_positions"]
    on_right = tuple(
        sum((Fraction(_float32(x)) for x, _ in bank), Fraction(0))
        > 5 * Fraction(float(width)) / 2
        for bank in pads
    )
    left_strip = (0.0, float(depth))
    right_strip = (float(np.float32(width - depth)), float(width))
    strips = tuple(right_strip if right else left_strip for right in on_right)
    assert on_right[0] != on_right[1]
    for bank, right, strip in zip(pads, on_right, strips, strict=True):
        assert red_zone_team_on_right(float(width), [x for x, _ in bank]) == right
        assert red_zone_x_range(float(width), float(depth), right) == strip

    def inside(x: float, strip: tuple[float, float]) -> bool:
        return strip[0] <= _float32(x) <= strip[1]

    alive = list(start["alive_mask"])
    positions = [list(row) for row in start["agent_positions"]]
    scores = list(start["team_deathmatch_scores"])
    respawn_pads: dict[int, list[float]] = {}
    respawns: dict[int, tuple[int, ...]] = {}
    for row in expected["ticks"]:
        tick = int(row["tick"])
        points = [0, 0]
        for slot in range(10):
            team = slot // 5
            before, after = positions[slot], row["positions"][slot]
            if row["newly_dead"][slot]:
                assert alive[slot] and not row["alive"][slot]
                # A victim scores from its start position; no exemption applies.
                victim_points = 2 if inside(before[0], strips[team]) else 1
                assert victim_points == 1, (case_name, tick, slot)
                points[1 - team] += victim_points
            if not alive[slot]:
                if row["alive"][slot]:
                    # A respawn lands on one of the team's own pads.
                    assert [_float32(value) for value in after] in [
                        [_float32(value) for value in pad] for pad in pads[team]
                    ]
                    assert inside(after[0], strips[team])
                    respawn_pads[slot] = after
                    respawns[tick] = (*respawns.get(tick, ()), slot)
                continue
            for position in (before, after):
                if respawn_pads.get(slot) == position:
                    continue
                assert not any(inside(position[0], strip) for strip in strips), (
                    case_name,
                    tick,
                    slot,
                    position,
                )
        scores = [score + gain for score, gain in zip(scores, points, strict=True)]
        assert row["scores"] == scores, (case_name, tick)
        alive, positions = list(row["alive"]), [list(p) for p in row["positions"]]
    assert respawns == _PAD_RESPAWNS.get(case_name, {})
