"""Check controlled slot schedules and their paired mean-effect statistics.

Physical classes and complete respawn banks must stay fixed for a focal-side
comparison. Synthetic complete blocks prove score direction, weights, uncertainty,
coverage rejection and the declared power calculation without training a model.
Built schedules use the declared Red Zone depth (default 5.0, or 6.0 and 0.0 when
given); explicit configs keep their own rules, so a depth beside them is refused.
"""

import math
from typing import Any

import numpy as np
import pytest

from marl_battlegrounds.tasks import (
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)
from marl_battlegrounds.training.analysis import (
    slot_planning_blocks,
    summarize_slot_diagnostic,
)
from marl_battlegrounds.training.validation import make_slot_diagnostic_schedule


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for map_id in (42, 43):
        for block in range(4):
            for team in (0, 1):
                for end in (0, 1):
                    # Both physical sides have effects [0, 0, 1, 1].
                    focal = float(block >= 2) if team == 0 else 0.0
                    rows.append(
                        {
                            "map_id": map_id,
                            "seed_id": (map_id - 42) * 4 + block + 1,
                            "focal_team": team,
                            "spawn_locations": end,
                            "system_game_score": focal if team == 0 else 1 - focal,
                        }
                    )
    return rows


def test_controlled_schedule_preserves_class_positions_and_all_respawn_pads() -> None:
    a, b = canonical_tournament_rosters()
    config = make_standard_team_deathmatch_config(
        map_id=42, team_a_roster=a, team_b_roster=b, max_steps=1
    )
    schedules = make_slot_diagnostic_schedule(
        maps=(42,), seed_blocks=2, configs=(config,)
    )
    assert [len(rows) for rows in schedules] == [4, 4]
    all_rows = (*schedules[0], *schedules[1])
    assert {row.episode_id for row in all_rows} == set(range(1, 9))
    assert {row.seed_id for row in all_rows} == {1, 2}
    for first, second in zip(schedules[0], schedules[1], strict=True):
        assert first.seed_id == second.seed_id
        assert first.spawn_locations == second.spawn_locations
    for block in range(2):
        for side in (0, 1):
            on_a = schedules[0][2 * block + side]
            on_b = schedules[1][2 * block + 1 - side]
            np.testing.assert_array_equal(
                on_a.env_config.team_spawn_pad_positions[0],
                on_b.env_config.team_spawn_pad_positions[1],
            )
            for name, a_field, b_field in zip(
                on_a.env_config.agent_profile._fields,
                on_a.env_config.agent_profile,
                on_b.env_config.agent_profile,
                strict=True,
            ):
                if name != "team_ids":
                    np.testing.assert_array_equal(a_field[:5], b_field[5:])
            assert on_a.metadata is not None and on_b.metadata is not None
            assert (
                on_a.metadata["physical_side"] == on_b.metadata["physical_side"] == side
            )


def test_built_schedule_uses_the_declared_red_zone_depth() -> None:
    def depths(schedules: tuple[tuple[Any, ...], tuple[Any, ...]]) -> set[float]:
        return {
            float(row.env_config.team_deathmatch_red_zone_depth)
            for rows in schedules
            for row in rows
        }

    assert depths(make_slot_diagnostic_schedule(maps=(42,), seed_blocks=1)) == {5.0}
    assert depths(
        make_slot_diagnostic_schedule(maps=(42,), seed_blocks=1, red_zone_depth=6.0)
    ) == {6.0}
    assert depths(
        make_slot_diagnostic_schedule(maps=(42,), seed_blocks=1, red_zone_depth=0.0)
    ) == {0.0}
    a, b = canonical_tournament_rosters()
    config = make_standard_team_deathmatch_config(
        map_id=42, team_a_roster=a, team_b_roster=b, red_zone_depth=0.0
    )
    with pytest.raises(ValueError, match="owns its rules"):
        make_slot_diagnostic_schedule(
            maps=(42,), seed_blocks=1, configs=(config,), red_zone_depth=0.0
        )


def test_exact_fixed_population_schedule_and_planning_bound() -> None:
    schedules = make_slot_diagnostic_schedule()
    assert [len(rows) for rows in schedules] == [1600, 1600]
    assert len({row.episode_id for rows in schedules for row in rows}) == 3200
    assert len({row.seed_id for rows in schedules for row in rows}) == 800
    # An independent calculation of ceil((z_0.975 + z_0.8)^2 / 0.1^2).
    assert (
        slot_planning_blocks()
        == math.ceil((1.959963984540054 + 0.8416212335729143) ** 2 / 0.01)
        == 785
    )
    with pytest.raises(ValueError):
        slot_planning_blocks(effect=0)


def test_score_direction_stratified_variance_and_holm_match_known_blocks() -> None:
    result = summarize_slot_diagnostic(_rows(), maps=(42, 43), seed_blocks=4)
    expected_error = math.sqrt((1 / 3) / 4 / 2)
    expected_p = math.erfc(0.5 / expected_error / math.sqrt(2))
    assert result["games"] == 32
    assert result["independent_blocks"] == 8
    assert result["primary"]["effect"] == 0.5
    assert result["primary"]["standard_error"] == pytest.approx(expected_error)
    assert result["primary"]["p_value"] == pytest.approx(expected_p)
    for side in result["sides"]:
        assert side["effect"] == 0.5
        assert side["holm_p_value"] == pytest.approx(min(1.0, 2 * expected_p))
    assert (
        summarize_slot_diagnostic(list(reversed(_rows())), maps=(42, 43), seed_blocks=4)
        == result
    )


def test_opposite_side_effects_do_not_disappear_from_side_results() -> None:
    rows = _rows()
    for row in rows:
        side = row["focal_team"] ^ row["spawn_locations"]
        if side == 1:
            row["system_game_score"] = 1 - row["system_game_score"]
    result = summarize_slot_diagnostic(rows, maps=(42, 43), seed_blocks=4)
    assert result["primary"]["effect"] == 0
    assert result["primary"]["p_value"] is None
    assert [row["effect"] for row in result["sides"]] == [0.5, -0.5]


@pytest.mark.parametrize("change", ("missing", "duplicate", "seed", "team"))
def test_bad_slot_coverage_is_rejected(change: str) -> None:
    rows = _rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "seed":
        rows[-1]["seed_id"] = 1
    else:
        rows[-1]["focal_team"] = 2
    with pytest.raises(ValueError):
        summarize_slot_diagnostic(rows, maps=(42, 43), seed_blocks=4)


def test_zero_variance_never_claims_significance_or_equality() -> None:
    rows = [{**row, "system_game_score": 0.5} for row in _rows()]
    result = summarize_slot_diagnostic(rows, maps=(42, 43), seed_blocks=4)
    assert result["primary"]["p_value"] is None
    assert result["primary"]["ci_low"] is None
    assert not result["systematic_effect_detected"]
