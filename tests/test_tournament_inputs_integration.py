"""Check complete shared tournament routes and saved inline-map recovery.

List, short-file and full-descriptor calls play the same tiny declared games.
A local challenger keeps Team A while spawn ends exchange. Interrupted v2
runs retain every map before the first game and resume without choosing current
maps. Factories run once in the caller's process and thread, and their returned
methods are reused without taking ownership of external providers. These CPU
checks establish workflow correctness, not GPU speed or learning.
"""

# Shared preparation and writer hooks provide fault injection, not another runner.
# pyright: reportPrivateUsage=false

import csv
import json
import os
import sys
import threading
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, Never, cast

import jax
import numpy as np
import pytest

from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    System,
    SystemInput,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_context import capture_recording_provenance
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.tournament import run_tournament
from marl_battlegrounds.evaluation.tournament_headlines import content_digest
from marl_battlegrounds.evaluation.tournament_inputs import prepare_tournament_inputs
from marl_battlegrounds.policies.actor import ActorAction


def _entrants() -> tuple[Policy, Policy]:
    return (
        replace(policy("random"), name="first", variables=np.asarray(1)),
        replace(policy("random"), name="second", variables=np.asarray(2)),
    )


def _stable_source(monkeypatch: pytest.MonkeyPatch) -> None:
    provenance = capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "capture_recording_provenance",
        same_source,
    )


def _game_values(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "run_id"} for row in rows
    ]


def test_list_short_file_and_full_config_use_the_same_complete_games(
    tmp_path: Path,
) -> None:
    entrants = ("random", "tdm-alpha")
    config, _, _ = prepare_tournament_inputs(
        entrants, maps=[12], games_per_opponent=2, seed=91, max_steps=1
    )
    short = tmp_path / "field.json"
    short.write_text(
        json.dumps(
            {
                "entrants": list(entrants),
                "maps": [12],
                "games_per_opponent": 2,
                "seed": 91,
                "max_steps": 1,
            }
        )
    )
    listed = run_tournament(
        entrants,
        maps=[12],
        games_per_opponent=2,
        seed=91,
        max_steps=1,
        num_envs=2,
        chunk_size=1,
    )
    from_short = run_tournament(config=short, num_envs=2, chunk_size=1)
    from_full = run_tournament(config=config, num_envs=2, chunk_size=1)
    assert (
        listed.metadata["snapshot_id"]
        == from_short.metadata["snapshot_id"]
        == from_full.metadata["snapshot_id"]
    )
    assert (
        _game_values(listed.matches)
        == _game_values(from_short.matches)
        == _game_values(from_full.matches)
    )
    assert (
        listed.tournament_results
        == from_short.tournament_results
        == from_full.tournament_results
    )
    assert (
        listed.metadata["canonical_plan"]
        == from_short.metadata["canonical_plan"]
        == from_full.metadata["canonical_plan"]
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["field.json"]


@pytest.mark.parametrize("whole_team", [False, True])
@pytest.mark.parametrize("route", ["list", "config"])
def test_challenger_routes_stay_team_a_and_keep_complete_spawn_pairs(
    whole_team: bool,
    route: str,
) -> None:
    actor = replace(policy("random"), name="challenger", variables=np.asarray(3))
    challenger = shared_policy(actor) if whole_team else actor
    config, bindings, _ = prepare_tournament_inputs(
        ("random", "tdm-alpha"), maps=[12], games_per_opponent=2, max_steps=1
    )
    options: dict[str, Any] = {"config": config}
    if route == "list":
        options = {
            "policies": ["random", "tdm-alpha"],
            "maps": [12],
            "games_per_opponent": 2,
            "max_steps": 1,
        }
    result = run_tournament(**options, challenger=challenger, num_envs=2, chunk_size=1)
    games = result.metadata["canonical_plan"]["games"]
    challenger_id = result.metadata["challenger_id"]
    assert challenger_id is not None
    selected = [row for row in games if row["team_a"] == challenger_id]
    assert len(games) == 6 and len(selected) == 4
    assert result.metadata["num_matches"] == len(result.metadata["schedule"]) == 6
    assert result.metadata["schedule_digest"] == content_digest(
        result.metadata["schedule"]
    )
    assert {row["name"] for row in result.metadata["policies"]} == {
        "random",
        "tdm-alpha",
        "challenger",
    }
    assert {row["execution"]["action_stream_version"] for row in selected} == {
        "evaluation-systems-v1" if whole_team else "episode-fold-in-v1"
    }
    assert {row["team_b"] for row in selected} == set(bindings)
    for first, second in zip(selected[::2], selected[1::2], strict=True):
        assert first["pair_id"] == second["pair_id"]
        assert first["execution"]["seed_id"] == second["execution"]["seed_id"]
        assert [first["spawn_locations"], second["spawn_locations"]] == [0, 1]
        assert first["resolved_config_id"] != second["resolved_config_id"]
    assert all(row["team_a_policy"] == "challenger" for row in result.matches[2:])
    assert len(config["participants"]) == 2


@pytest.mark.parametrize("completed_games", [0, 1])
def test_inline_sources_survive_interruption_before_later_map_is_played(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_games: int
) -> None:
    _stable_source(monkeypatch)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    original_execute = evaluator._evaluate_tournament_episodes
    original_write = RunWriter.write

    def stop_before(*args: object, **options: object) -> None:
        raise RuntimeError("injected inline-source interruption")

    def stop_after(writer: RunWriter, info: EpisodeInfo) -> None:
        original_write(writer, info)
        writer.flush()
        raise RuntimeError("injected inline-source interruption")

    if completed_games:
        monkeypatch.setattr(RunWriter, "write", stop_after)
    else:
        monkeypatch.setattr(evaluator, "_evaluate_tournament_episodes", stop_before)
    with pytest.raises(RuntimeError, match="inline-source interruption"):
        run_tournament(
            _entrants(),
            maps=[12, 13],
            games_per_opponent=4,
            max_steps=1,
            metrics="none",
            num_envs=1,
            chunk_size=1,
            output_dir=tmp_path,
        )
    directory = next(tmp_path.iterdir())
    manifest = json.loads((directory / "run_details.json").read_text())
    config = manifest["details"]["canonical_config"]
    expected = {
        row["map_id"]: row["source_config_id"]
        for row in config["conditions"]["map_sources"]
    }
    assert set(expected) == {12, 13}
    for source in config["conditions"]["map_sources"]:
        assert isinstance(
            config["assets"][source["source_config_asset"]]["inline"], Mapping
        )
    monkeypatch.setattr(evaluator, "_evaluate_tournament_episodes", original_execute)
    monkeypatch.setattr(RunWriter, "write", original_write)

    def unavailable(*args: object, **options: object) -> None:
        raise AssertionError("Resume chose current maps instead of saved contents")

    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.tournament_inputs"),
        "normalize_episode_specs",
        unavailable,
    )
    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.map_identity"),
        "registered_map_metadata",
        unavailable,
    )
    result = run_tournament(
        _entrants(), resume_from=directory, num_envs=1, chunk_size=1
    )
    assert [row["episode_id"] for row in result.matches] == [1, 2, 3, 4]
    assert result.metadata["configuration_ids_by_map"] == {
        str(key): value for key, value in expected.items()
    }
    assert result.metadata["executed_this_call"] == 4 - completed_games
    assert {
        fact["determinism"] for fact in result.metadata["method_sampling"].values()
    } == {"stochastic"}
    with (directory / "match_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and len({row["episode_id"] for row in rows}) == 4


def test_weighted_field_challenger_fails_before_actions_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Incomplete opponent weights must fail before any game")

    monkeypatch.setattr(evaluator, "_evaluate_tournament_episodes", unexpected)
    config, _, _ = prepare_tournament_inputs(
        ["random", "tdm-alpha"],
        maps=[12],
        games_per_opponent=2,
        max_steps=1,
        opponent_weights={"random": 1.0, "tdm-alpha": 2.0},
    )
    challenger = replace(policy("random"), name="third", variables=np.asarray(3))
    with pytest.raises(ValueError, match="opponent_weights must name every entrant"):
        run_tournament(config=config, challenger=challenger, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("form", ["list", "tuple", "generator", "cli"])
def test_short_config_accepts_equal_sequence_forms_without_losing_captures(
    form: str, tmp_path: Path
) -> None:
    from marl_battlegrounds._cli import build_parser
    from marl_battlegrounds.tasks import list_tdm_maps

    short = tmp_path / "field.json"
    short.write_text(
        json.dumps(
            {
                "entrants": ["random", "tdm-alpha"],
                "maps": [12],
                "games_per_opponent": 2,
                "max_steps": 1,
                "full_metrics_episodes": [2, 1, 1],
                "replay_episodes": [1, 2],
            }
        )
    )
    options: dict[str, Any] = {
        "maps": (list_tdm_maps()[12],),
        "full_metrics_episodes": [1, 2],
        "replay_episodes": [2, 1, 2],
    }
    if form == "tuple":
        options["full_metrics_episodes"] = (2, 1)
        options["replay_episodes"] = (1, 2)
    elif form == "generator":
        options["full_metrics_episodes"] = (value for value in [2, 1, 2])
        options["replay_episodes"] = (value for value in [1, 2])
    elif form == "cli":
        parsed = build_parser().parse_args(
            [
                "tournament",
                "--config",
                str(short),
                "--maps",
                "12",
                "--full-metrics-episodes",
                "1,2",
                "--replay-episodes",
                "2,1",
            ]
        )
        options = {name: getattr(parsed, name) for name in options}
    result = run_tournament(config=short, num_envs=2, chunk_size=1, **options)
    assert len(result.matches) == 2
    np.testing.assert_array_equal(result.full_metrics["episode_id"], [1, 2])
    assert len(result.replays) == 2
    assert result.metadata["full_metrics_episodes"] == [1, 2]
    assert result.metadata["replay_episodes"] == [1, 2]
    assert {row["map_id"] for row in result.matches} == {12}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["field.json"]


def test_full_descriptor_resume_keeps_saved_capture_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stable_source(monkeypatch)
    config, _, _ = prepare_tournament_inputs(
        ("random", "tdm-alpha"), maps=[12], games_per_opponent=2, max_steps=1
    )
    original = run_tournament(
        config=config,
        full_metrics_episodes=[2, 1],
        replay_episodes=[2, 1],
        num_envs=2,
        chunk_size=1,
        output_dir=tmp_path,
    )
    assert original.run_dir is not None
    before = {
        path: path.read_bytes()
        for path in original.run_dir.rglob("*")
        if path.is_file()
    }
    resumed = run_tournament(
        resume_from=original.run_dir,
        full_metrics_episodes=(value for value in [2, 1]),
        replay_episodes=(value for value in [1, 2]),
    )
    assert resumed.metadata["full_metrics_episodes"] == [2, 1]
    assert resumed.metadata["replay_episodes"] == original.metadata["replay_episodes"]
    assert resumed.matches == original.matches
    assert resumed.metadata["executed_this_call"] == 0
    assert {
        path: path.read_bytes()
        for path in original.run_dir.rglob("*")
        if path.is_file()
    } == before


def test_pending_dispatch_indexes_the_plan_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation import canonical, tournament_execution
    from marl_battlegrounds.evaluation.tournament_records import TournamentRecords

    indexes: list[TournamentRecords] = []

    def records(*args: object, **kwargs: object) -> TournamentRecords:
        result = cast(Callable[..., TournamentRecords], TournamentRecords)(
            *args, **kwargs
        )
        indexes.append(result)
        return result

    def dispatch(
        jobs: Iterable[tournament_execution.TournamentJob],
        *args: object,
        **kwargs: object,
    ) -> Never:
        before = len(indexes)
        pending = tuple(jobs)
        assert len(pending) == 3
        assert len(indexes) - before == 1
        assert len({game.episode_id for job in pending for game in job.episodes}) == 6
        raise RuntimeError("Inspected pending jobs without playing")

    monkeypatch.setattr(canonical, "TournamentRecords", records)
    monkeypatch.setattr(tournament_execution, "execute_tournament_jobs", dispatch)
    with pytest.raises(RuntimeError, match="Inspected pending jobs"):
        run_tournament(
            ["random", "tdm-alpha", replace(policy("random"), name="third")],
            maps=[12],
            games_per_opponent=2,
            max_steps=1,
            num_envs=2,
        )


def test_factory_stays_in_caller_context_and_is_reused_across_matchups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, int, int]] = []
    caller = os.getpid(), threading.get_ident()

    def event(phase: str) -> None:
        events.append((phase, os.getpid(), threading.get_ident()))

    class Provider:
        def __reduce__(self) -> Never:
            raise AssertionError("The tournament must not serialize the provider")

        def apply(
            self,
            variables: PolicyTree,
            memory: PolicyTree,
            inputs: SystemInput,
            keys: jax.Array,
        ) -> tuple[ActorAction, PolicyTree]:
            event("apply")
            assert variables["provider"] is self
            assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(inputs))
            zero = np.zeros(inputs.active_mask.shape, np.int32)
            action = ActorAction(*(cast(jax.Array, zero) for _ in range(3)))
            return action, memory

        def close(self) -> None:
            raise AssertionError("Researchers own the provider lifetime")

    provider = Provider()

    def make_host() -> System:
        event("construct")
        return System(
            "caller-host",
            provider.apply,
            variables={"provider": provider},
            execution="host",
        )

    module = types.ModuleType("test_tournament_caller_factory")
    module.make_host = make_host  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = run_tournament(
        [module.__name__ + ":make_host", "random", "tdm-alpha"],
        maps=[12],
        games_per_opponent=2,
        max_steps=2,
        num_envs=2,
        chunk_size=1,
        replay_episodes=(1, 2),
        output_dir=tmp_path,
    )
    assert [phase for phase, _, _ in events].count("construct") == 1
    assert sum(phase == "apply" for phase, _, _ in events) >= 4
    assert {(pid, thread) for _, pid, thread in events} == {caller}
    assert len(result.matches) == 6
    assert result.paths is not None
    manifest = json.loads(Path(result.paths["run_details"]).read_text())
    assert "execution_resources" not in manifest
    assert "execution_resources" not in result.metadata
    assert len({row["episode_id"] for row in result.matches}) == 6


def test_factory_allocation_error_propagates_once_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    error = MemoryError("The provider could not allocate its model")

    def make_host() -> Never:
        calls.append("construct")
        raise error

    module = types.ModuleType("test_tournament_failed_factory")
    module.make_host = make_host  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(MemoryError) as caught:
        run_tournament(
            [module.__name__ + ":make_host", "random"],
            maps=[12],
            games_per_opponent=2,
            max_steps=2,
            output_dir=tmp_path,
        )
    assert caught.value is error
    assert calls == ["construct"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("short_config", [False, True])
@pytest.mark.parametrize("option", ["num_envs", "chunk_size"])
@pytest.mark.parametrize("value", [0, False, None])
def test_invalid_execution_size_never_constructs_an_entrant(
    monkeypatch: pytest.MonkeyPatch, short_config: bool, option: str, value: object
) -> None:
    def make_host() -> Never:
        raise AssertionError("Invalid execution settings must fail before loading")

    module = types.ModuleType("test_tournament_invalid_size_factory")
    module.make_host = make_host  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    entrants = [module.__name__ + ":make_host", "random"]
    options: dict[str, Any] = {option: value}
    with pytest.raises((ValueError, TypeError), match=option):
        if short_config:
            run_tournament(
                config={"entrants": entrants, "maps": [12], "games_per_opponent": 2},
                **options,
            )
        else:
            run_tournament(entrants, maps=[12], games_per_opponent=2, **options)
