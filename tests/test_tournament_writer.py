"""Check tournament outcomes and summaries at the writer's durable boundary."""

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from marl_battlegrounds import make
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation import run_writer
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.run_writer import MATCH_COLUMNS, RunWriter
from marl_battlegrounds.evaluation.tournament_schedule import build_tournament_schedule
from marl_battlegrounds.evaluation.tournament_statistics import (
    TournamentStatistics,
    summarize_tournament,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


@pytest.fixture(scope="module")
def completed() -> tuple[EpisodeInfo, EpisodeInfo]:
    config = make_standard_team_deathmatch_config(
        map_id=0,
        team_a_roster=("mage",),
        team_b_roster=("priest",),
        max_steps=1,
    )
    env = make("tdm")
    key = jax.random.key(0)
    _, state = env.reset(key, config, episode_id=1)
    zero = jnp.zeros(10, jnp.int32)
    info = cast(EpisodeInfo, jax.jit(env.step)(key, state, Action(zero, zero, zero))[4])
    assert bool(info.completed)
    return info, info._replace(episode_id=jnp.int32(2), priority=None)


@pytest.fixture(scope="module")
def statistics() -> TournamentStatistics:
    return summarize_tournament(
        build_tournament_schedule(["A", "B"], [0], episodes_per_pair=2), {1: 3, 2: 3}
    )


def _register(writer: RunWriter) -> None:
    writer.register_episodes(
        [
            {
                "episode_id": 1,
                "seed_id": 1,
                "map_id": 0,
                "block_id": 1,
                "bootstrap_group": "shared-weather",
            },
            {
                "episode_id": 2,
                "seed_id": 1,
                "map_id": 0,
                "block_id": 1,
                "bootstrap_group": "shared-weather",
            },
        ]
    )


def test_match_table_keeps_outcomes_optional_priority_and_block_identity(
    tmp_path: Path,
    completed: tuple[EpisodeInfo, EpisodeInfo],
) -> None:
    with RunWriter(
        tmp_path, phase="tournament", policies={"team_a": "A", "team_b": "B"}
    ) as writer:
        _register(writer)
        for info in completed:
            writer.write(info)
        writer.flush()
        with writer.paths["match_results"].open(newline="") as stream:
            table = csv.DictReader(stream)
            assert tuple(table.fieldnames or ()) == MATCH_COLUMNS
            rows = list(table)
        assert writer.completed_episode_ids == {1, 2}
        assert "priority_metrics" not in writer.paths
        assert [row["outcome"] for row in rows] == ["3", "3"]
        assert [row["block_id"] for row in rows] == ["1", "1"]
        assert [row["bootstrap_group"] for row in rows] == ["shared-weather"] * 2
        assert float(rows[0]["episode_length"]) == 1
        assert "agent_1_return" not in rows[0]
        assert all(
            rows[1][name] == ""
            for name in PRIORITY_METRIC_NAMES
            if name not in ("episode_length", "team_a_score", "team_b_score")
        )


def test_summary_is_idempotent_across_explicit_resume_and_rejects_new_population(
    tmp_path: Path,
    completed: tuple[EpisodeInfo, EpisodeInfo],
    statistics: TournamentStatistics,
) -> None:
    with RunWriter(tmp_path, phase="tournament") as writer:
        _register(writer)
        for info in completed:
            writer.write(info)
        writer.write_tournament_results(statistics)
        before = {name: path.read_bytes() for name, path in writer.paths.items()}
        run_dir = writer.run_dir
        writer.write_tournament_results(statistics)
        assert before == {
            name: path.read_bytes() for name, path in writer.paths.items()
        }
    with RunWriter(resume_from=run_dir, phase="tournament") as writer:
        writer.write_tournament_results(statistics)
        for name in (
            "match_results",
            "tournament_results",
            "matchup_results",
            "map_results",
        ):
            assert writer.paths[name].read_bytes() == before[name]
        with pytest.raises(ValueError, match="differs"):
            writer.write_tournament_results(
                replace(statistics, metadata={"changed": True})
            )


def test_failed_summary_publication_recovers_without_duplicate_rows(
    tmp_path: Path,
    completed: tuple[EpisodeInfo, EpisodeInfo],
    statistics: TournamentStatistics,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = RunWriter(tmp_path, phase="tournament")
    _register(writer)
    for info in completed:
        writer.write(info)
    writer.flush()
    run_dir = writer.run_dir
    durable = writer.paths["match_results"].read_bytes()
    publish = run_writer._atomic_json  # pyright: ignore[reportPrivateUsage]

    def fail_summary(path: Path, value: object) -> None:
        if isinstance(value, dict) and "tournament_summary" in value:
            raise OSError("injected publication failure")
        publish(path, cast(object, value))

    monkeypatch.setattr(run_writer, "_atomic_json", fail_summary)
    with pytest.raises(OSError, match="publication failure"):
        writer.write_tournament_results(statistics)
    writer.close()
    assert "tournament_summary" not in json.loads(
        (run_dir / "run_details.json").read_bytes()
    )
    monkeypatch.setattr(run_writer, "_atomic_json", publish)
    with RunWriter(resume_from=run_dir, phase="tournament") as resumed:
        assert resumed.paths["match_results"].read_bytes() == durable
        resumed.write_tournament_results(statistics)
        for name, expected in (
            ("tournament_results", statistics.tournament_results),
            ("matchup_results", statistics.matchup_results),
            ("map_results", statistics.map_results),
        ):
            with resumed.paths[name].open(newline="") as stream:
                assert len(list(csv.DictReader(stream))) == len(expected)


def test_invalid_summary_does_not_queue_an_earlier_valid_table(
    tmp_path: Path,
    statistics: TournamentStatistics,
) -> None:
    with RunWriter(tmp_path, phase="tournament") as writer:
        with pytest.raises(ValueError, match="nonempty"):
            writer.write_tournament_results(replace(statistics, map_results=()))
        writer.flush()
        assert set(writer.paths) == {"run_details"}
