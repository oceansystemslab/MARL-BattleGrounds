"""Check physical pair ownership and exact complete-population headline arithmetic.

These host fixtures contain final game measurements, including authored points,
large exact scores, zero deaths and malformed evidence. They do not run games or
claim an official tournament population has been released or qualified.
"""

# pyright: reportPrivateUsage=false

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from marl_battlegrounds.evaluation.evaluation_conditions import config_record
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.evaluation.tournament import _prepare_pair_evidence
from marl_battlegrounds.evaluation.tournament_headlines import (
    HEADLINE_COLUMNS,
    _distribution,
    content_digest,
    match_digest,
    summarize_headlines,
)
from marl_battlegrounds.evaluation.tournament_schedule import (
    TournamentMatch,
    build_tournament_schedule,
)
from marl_battlegrounds.evaluation.tournament_statistics import _population, _summarize
from marl_battlegrounds.tasks import (
    _swap_spawn_banks,
    make_standard_team_deathmatch_config,
)


def _fixture(count: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = tuple(f"entrant-{i:02}" for i in range(count))
    source = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("mage",), team_b_roster=("mage",)
    )
    source_id, source_content = config_record(source)
    exchanged_id, exchanged_content = config_record(_swap_spawn_banks(source))
    configs = {source_id: source_content, exchanged_id: exchanged_content}
    systems: dict[str, dict[str, object]] = {}
    identifiers: dict[str, str] = {}
    for name in names:
        identifier, registration = normalize_system_registration(
            name, phase="tournament"
        )
        systems[identifier] = registration
        identifiers[name] = identifier
    schedule = tuple(
        replace(
            match,
            source_config_id=source_id,
            resolved_config_id=source_id
            if match.spawn_locations == 0
            else exchanged_id,
        )
        for match in build_tournament_schedule(names, (0,), episodes_per_pair=4)
    )
    passes: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for match in schedule:
        pass_id = f"{match.team_a}-{match.team_b}"
        key = f'["tournament","{pass_id}"]'
        entry = passes.setdefault(
            key,
            {
                "phase": "tournament",
                "pass_id": pass_id,
                "episodes": {},
                "system_ids": {
                    "team_a": identifiers[match.team_a],
                    "team_b": identifiers[match.team_b],
                },
            },
        )
        entry["episodes"][str(match.episode_id)] = {
            "configuration_digest": match.resolved_config_id,
            "source_config_id": source_id,
            "spawn_locations": match.spawn_locations,
            "comparison_kind": "verified_spawn_pair",
            "seed_id": match.seed_id,
            "map_id": 0,
            "initial_state_digest": None,
        }
        length = (1, 4, 6, 9)[(match.episode_id - 1) % 4]
        rows.append(
            {
                "run_id": "fixture-run",
                "phase": "tournament",
                "pass_id": pass_id,
                "episode_id": match.episode_id,
                "block_id": match.block_id,
                "bootstrap_group": None,
                "seed_id": match.seed_id,
                "map_id": 0,
                "config_id": match.resolved_config_id,
                "team_a_policy": match.team_a,
                "team_b_policy": match.team_b,
                "outcome": 1,
                "episode_length": length,
                "team_a_score": 10 + length,
                "team_b_score": 2,
                "score_difference": float(8 + length),
                "team_a_return": float(length),
                "team_b_return": float(-length),
                "team_a_kills": 2.0,
                "team_b_kills": 0.0,
                "team_a_deaths": 0.0,
                "team_b_deaths": 2.0,
            }
        )
    evidence = _prepare_pair_evidence(
        schedule, rows, configurations=configs, systems=systems, passes=passes
    )
    return rows, evidence


def test_headlines_count_owners_once_and_keep_authored_scores_distinct() -> None:
    rows, evidence = _fixture()
    headline = summarize_headlines(rows, evidence)
    by_name = {row["system_name"]: row for row in headline}
    a, b = by_name["entrant-00"], by_name["entrant-01"]
    assert tuple(a) == HEADLINE_COLUMNS
    assert a["games_played"] == b["games_played"] == 4
    assert a["completed_pairs"] == b["completed_pairs"] == 2
    assert a["total_steps_played"] == b["total_steps_played"] == 20
    assert a["cumulative_policy_return"] == 20
    assert b["cumulative_policy_return"] == -20
    assert a["mean_policy_return"] == 5
    assert a["episode_length_median"] == 5
    assert a["episode_length_mean"] == 5
    assert a["episode_length_std"] == pytest.approx(np.std([1, 4, 6, 9]))
    assert a["total_score_difference"] == 52
    assert b["total_score_difference"] == -52
    assert a["total_kills"] == 8
    assert a["total_deaths"] == 0
    assert a["kd_ratio"] is None
    assert b["kd_ratio"] == 0
    assert sum(row["total_steps_played"] for row in headline) == 2 * sum(
        row["episode_length"] for row in rows
    )
    assert summarize_headlines(rows[::-1], evidence) == headline


@pytest.mark.parametrize("count", [12, 13])
def test_complete_population_fixtures_have_every_game_and_pair(count: int) -> None:
    rows, evidence = _fixture(count)
    headline = summarize_headlines(rows, evidence)
    assert len(headline) == count
    assert [row["system_id"] for row in headline] == sorted(
        evidence["population_system_ids"]
    )
    assert all(row["games_played"] == 4 * (count - 1) for row in headline)
    assert all(row["completed_pairs"] == 2 * (count - 1) for row in headline)


def test_large_exact_scores_use_the_float32_measurement_conversion() -> None:
    rows, evidence = _fixture()
    rows[0]["team_a_score"] = 2**24 + 3
    rows[0]["team_b_score"] = 0
    rows[0]["score_difference"] = float(np.float32(2**24 + 3))
    evidence["match_digest"] = match_digest(rows)
    result = {row["system_name"]: row for row in summarize_headlines(rows, evidence)}
    assert result["entrant-00"]["total_score_difference"] == 2**24 + 3 + 12 + 14 + 17
    assert _distribution([7], "x")["x_std"] == 0
    assert _distribution([1, 5, 8], "x")["x_median"] == 5
    assert _distribution([1, 6], "x")["x_median"] == 3.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("team_a_return", None),
        ("team_b_kills", float("nan")),
        ("team_a_deaths", -1),
        ("score_difference", 123),
    ],
)
def test_missing_or_invalid_measurements_never_shrink_the_sample(
    field: str, value: object
) -> None:
    rows, evidence = _fixture()
    rows[0][field] = value
    if not isinstance(value, float) or np.isfinite(value):
        evidence["match_digest"] = match_digest(rows)
    with pytest.raises(ValueError):
        summarize_headlines(rows, evidence)


@pytest.mark.parametrize(
    "fault",
    ["missing", "duplicate", "owner", "source", "config", "same_choice", "custom"],
)
def test_invalid_pair_evidence_is_rejected_before_headline(fault: str) -> None:
    rows, evidence = _fixture()
    schedule = tuple(TournamentMatch(**record) for record in evidence["schedule"])
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "owner":
        rows[0]["team_a_policy"] = "entrant-01"
    elif fault == "source":
        schedule = (replace(schedule[0], source_config_id="missing"), *schedule[1:])
    elif fault == "config":
        evidence["configurations"] = deepcopy(evidence["configurations"])
        evidence["configurations"][schedule[0].resolved_config_id]["max_steps"] += 1
    elif fault == "same_choice":
        schedule = (schedule[0], replace(schedule[1], spawn_locations=0), *schedule[2:])
    else:
        next(iter(evidence["passes"].values()))["episodes"]["1"]["comparison_kind"] = (
            "custom"
        )
    with pytest.raises(ValueError):
        _prepare_pair_evidence(
            schedule,
            rows,
            configurations=evidence["configurations"],
            systems=evidence["systems"],
            passes=evidence["passes"],
        )


def test_geometry_free_fixed_and_historical_joins_have_identical_statistics() -> None:
    schedule = build_tournament_schedule(("a", "b", "c"), (0, 1), episodes_per_pair=8)
    outcomes = {m.episode_id: (1, 3, 2)[m.episode_id % 3] for m in schedule}
    historical = tuple(
        replace(
            m,
            team_a=m.team_b if m.spawn_locations else m.team_a,
            team_b=m.team_a if m.spawn_locations else m.team_b,
            pairing_protocol=None,
            spawn_locations=None,
        )
        for m in schedule
    )
    old_outcomes = {
        m.episode_id: (
            outcomes[m.episode_id]
            if m.team_a < m.team_b or outcomes[m.episode_id] == 3
            else 3 - outcomes[m.episode_id]
        )
        for m in historical
    }
    current = _population(schedule, outcomes, None)
    old = _population(historical, old_outcomes, None)
    for field in (
        "blocks",
        "block_cells",
        "cell_counts",
        "opponent_weights",
        "rate_weights",
    ):
        np.testing.assert_array_equal(getattr(current, field), getattr(old, field))
    for new_group, old_group in zip(current.strata, old.strata, strict=True):
        np.testing.assert_array_equal(new_group, old_group)
    assert _summarize(
        schedule, outcomes, seed=4, opponent_weights=None, replicates=16
    ) == _summarize(
        historical, old_outcomes, seed=4, opponent_weights=None, replicates=16
    )
    empty_protocol = tuple(replace(match, pairing_protocol="") for match in historical)
    with pytest.raises(ValueError, match="protocol"):
        _population(empty_protocol, old_outcomes, None)
    mixed = (replace(schedule[0], pairing_protocol="team-swap-v1"), *schedule[1:])
    with pytest.raises(ValueError, match="protocol"):
        _population(mixed, outcomes, None)


def test_changed_rows_or_schedule_cannot_reuse_finished_evidence() -> None:
    rows, evidence = _fixture()
    rows[0]["team_a_return"] += 1
    with pytest.raises(ValueError, match="game rows"):
        summarize_headlines(rows, evidence)
    evidence["match_digest"] = match_digest(rows)
    evidence["schedule_digest"] = content_digest(
        [
            asdict(match)
            for match in build_tournament_schedule(
                ("x", "y"), (1,), episodes_per_pair=2
            )
        ]
    )
    with pytest.raises(ValueError, match="schedule"):
        summarize_headlines(rows, evidence)
