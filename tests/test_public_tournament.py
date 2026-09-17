"""Check small complete tournaments, selected outputs and resumed execution."""


# Historical protocol fixtures exercise the shared private executor directly.
# pyright: reportPrivateUsage=false

import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, EvaluationResult
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    independent_policies,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_context import capture_recording_provenance
from marl_battlegrounds.evaluation.run_writer import MATCH_COLUMNS
from marl_battlegrounds.evaluation.tournament import run_tournament
from marl_battlegrounds.tasks import CANONICAL_TDM_EVALUATION_MAP_IDS, list_tdm_maps


def _entrants(count: int = 2) -> tuple[Policy, ...]:
    return tuple(
        replace(policy("random"), name=f"model-{index}") for index in range(count)
    )


@pytest.mark.parametrize("name", ["aaa-bad", "zzz-bad"])
def test_adapter_roster_mismatch_rejects_before_new_tournament_files(
    name: str, tmp_path: Path
) -> None:
    bad = independent_policies([replace(policy("random"), name=name)])
    with pytest.raises(ValueError, match="roster"):
        run_tournament(
            [bad, "random"],
            maps=[12],
            episodes_per_pair=2,
            max_steps=1,
            output_dir=tmp_path / "bad",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ({"map_id": 0}, {"resource_sha256": "0" * 64}))
def test_saved_map_details_are_rejected_before_tournament_creates_files(
    change: dict[str, object], tmp_path: Path
) -> None:
    saved = list_tdm_maps()[12].model_copy(update=change)
    destination = tmp_path / "tournament"
    with pytest.raises(
        ValueError, match=r"choose the map again with list_tdm_maps\(\)"
    ):
        run_tournament(_entrants(), maps=[saved], output_dir=destination)
    assert not destination.exists()


def test_twenty_match_canonical_smoke_has_complete_balanced_evidence_without_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_tournament(
        _entrants(), episodes_per_pair=20, max_steps=1, chunk_size=2
    )
    assert [row["episode_id"] for row in result.matches] == list(range(1, 21))
    assert all(tuple(row) == MATCH_COLUMNS for row in result.matches)
    assert all(
        row["outcome"] == 3 and row["episode_length"] == 1 for row in result.matches
    )
    assert {row["run_id"] for row in result.matches} == {result.metadata["run_id"]}
    assert len(cast(dict[str, object], result.metadata["configurations"])) == 10
    for map_id in CANONICAL_TDM_EVALUATION_MAP_IDS:
        assert sum(row["map_id"] == map_id for row in result.matches) == 4
    for first, second in zip(result.matches[::2], result.matches[1::2], strict=True):
        assert first["block_id"] == second["block_id"]
        assert first["seed_id"] == second["seed_id"]
        assert first["team_a_policy"] == second["team_a_policy"]
        assert first["config_id"] != second["config_id"]
    assert len(result.tournament_results) == len(result.matchup_results) == 2
    assert len(result.map_results) == 10
    assert all(row["expected_score"] == 0.5 for row in result.tournament_results)
    assert result.full_metrics == {} and result.replays == () and result.paths is None
    assert list(tmp_path.iterdir()) == []


def test_three_entrants_preserve_global_ids_and_sparse_diagnostic_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("marl_battlegrounds.evaluation.tournament")

    def forbidden_headline(*args: object, **kwargs: object) -> None:
        raise AssertionError("none mode must skip headline aggregation")

    monkeypatch.setattr(module, "summarize_headlines", forbidden_headline)
    result = run_tournament(
        _entrants(3),
        maps=[list_tdm_maps()[12]],
        episodes_per_pair=2,
        metrics="none",
        full_metrics_episodes=[2],
        replay_episodes=[5],
        max_steps=1,
        chunk_size=1,
    )
    assert [row["episode_id"] for row in result.matches] == list(range(1, 7))
    assert {row["outcome"] for row in result.matches} == {3}
    assert [row["episode_length"] for row in result.matches] == [1] * 6
    assert [row["team_a_score"] for row in result.matches] == [0] * 6
    assert [row["team_b_score"] for row in result.matches] == [0] * 6
    assert all(name in result.full_metrics for name in FULL_METRIC_NAMES)
    np.testing.assert_array_equal(result.full_metrics["episode_id"], [2])
    np.testing.assert_array_equal(
        result.full_metrics["run_id"], [result.metadata["run_id"]]
    )
    assert len(result.replays) == 1
    assert result.replays[0].header.context.identity.episode_id.endswith("episode-5")
    assert result.replays[0].header.context.identity.run_id == result.metadata["run_id"]
    assert len(result.tournament_results) == 3
    assert len(result.matchup_results) == 6


def test_persisted_tournament_resumes_without_reexecution_or_repeated_summary_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "capture_recording_provenance",
        same_source,
    )
    entrants = _entrants()
    result = run_tournament(
        entrants,
        maps=[12],
        episodes_per_pair=4,
        max_steps=1,
        chunk_size=2,
        full_metrics_episodes=[1],
        output_dir=tmp_path,
    )
    assert result.paths is not None
    assert "priority_metrics" not in result.paths
    assert result.full_metrics == {}
    paths = result.paths
    durable = {
        name: path.read_bytes() for name, path in paths.items() if name != "run_details"
    }
    module = import_module("marl_battlegrounds.evaluation.tournament")

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("a completed saved tournament must not refit")

    monkeypatch.setattr(module, "summarize_tournament", forbidden_fit)
    resumed = run_tournament(
        tuple(reversed(entrants)),
        maps=[12],
        episodes_per_pair=4,
        max_steps=1,
        chunk_size=3,
        num_envs=1,
        full_metrics_episodes=[1],
        resume_from=paths["run_details"].parent,
    )
    assert resumed.headline_metrics == result.headline_metrics
    assert resumed.matches == result.matches
    assert resumed.tournament_results == result.tournament_results
    manifest = json.loads(paths["run_details"].read_bytes())
    for entry in manifest["passes"].values():
        if entry.get("pass_role") == "tournament_coordinator":
            entry["result_state"]["status"] = "failed"
            entry["result_state"]["reason"] = "injected late summary publication error"
    paths["run_details"].write_text(json.dumps(manifest))
    repaired = run_tournament(entrants, resume_from=paths["run_details"].parent)
    assert repaired.status == "complete"
    assert repaired.tournament_results == result.tournament_results
    original_maps = module.normalize_episode_specs

    def changed_maps(*args: object, **kwargs: object) -> tuple[EpisodeSpec, ...]:
        specs = cast(tuple[EpisodeSpec, ...], original_maps(*args, **kwargs))
        return tuple(
            replace(
                spec,
                env_config=spec.env_config._replace(
                    map_width=spec.env_config.map_width + 1
                ),
            )
            for spec in specs
        )

    monkeypatch.setattr(module, "normalize_episode_specs", changed_maps)
    with pytest.raises(ValueError, match="explicit map contents"):
        run_tournament(entrants, maps=[12], resume_from=paths["run_details"].parent)
    assert (
        run_tournament(entrants, resume_from=paths["run_details"].parent).matches
        == result.matches
    )
    for name, data in durable.items():
        assert paths[name].read_bytes() == data
    with paths["full_metrics"].open(newline="") as stream:
        assert [row["episode_id"] for row in csv.DictReader(stream)] == ["1"]
    with pytest.raises(ValueError, match="identity differs"):
        run_tournament(
            (replace(entrants[0], variables=np.asarray(2.0)), entrants[1]),
            maps=[12],
            episodes_per_pair=4,
            max_steps=1,
            full_metrics_episodes=[1],
            resume_from=paths["run_details"].parent,
        )


def test_interrupted_pair_pass_resumes_only_missing_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "capture_recording_provenance",
        same_source,
    )
    module = import_module("marl_battlegrounds.evaluation.tournament")
    original = module.evaluate_episodes
    executions: list[tuple[int, ...]] = []

    def stop_after_first_pass(
        first: Policy,
        second: Policy,
        episodes: Sequence[EpisodeSpec],
        **options: object,
    ) -> EvaluationResult:
        if executions:
            raise RuntimeError("injected interrupted tournament")
        result = cast(EvaluationResult, original(first, second, episodes, **options))
        executions.append(tuple(row.episode_id for row in result.episodes))
        return result

    monkeypatch.setattr(module, "evaluate_episodes", stop_after_first_pass)
    with pytest.raises(RuntimeError, match="interrupted tournament"):
        run_tournament(
            _entrants(3),
            maps=[12],
            episodes_per_pair=4,
            max_steps=1,
            chunk_size=2,
            output_dir=tmp_path,
        )
    assert executions == [(1, 2, 3, 4)]
    run_dir = next(tmp_path.iterdir())
    assert not (run_dir / "tournament_results.csv").exists()
    with (run_dir / "match_results.csv").open(newline="") as stream:
        assert [row["episode_id"] for row in csv.DictReader(stream)] == [
            "1",
            "2",
            "3",
            "4",
        ]

    def observe_resumption(
        first: Policy,
        second: Policy,
        episodes: Sequence[EpisodeSpec],
        **options: object,
    ) -> EvaluationResult:
        result = cast(EvaluationResult, original(first, second, episodes, **options))
        executions.append(tuple(row.episode_id for row in result.episodes))
        return result

    monkeypatch.setattr(module, "evaluate_episodes", observe_resumption)
    result = run_tournament(
        _entrants(3),
        maps=[12],
        episodes_per_pair=4,
        max_steps=1,
        chunk_size=2,
        resume_from=run_dir,
    )
    assert executions == [(1, 2, 3, 4), (), (5, 6, 7, 8), (9, 10, 11, 12)]
    assert [row["episode_id"] for row in result.matches] == list(range(1, 13))


@pytest.mark.parametrize("selection", ([5], [0], [True], [1.5]))
def test_invalid_global_selection_fails_before_creating_run_files(
    selection: list[int],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "new-output"
    with pytest.raises((ValueError, TypeError)):
        run_tournament(
            _entrants(),
            maps=[12],
            episodes_per_pair=4,
            replay_episodes=selection,
            output_dir=destination,
        )
    assert not destination.exists()


def test_system_and_policy_tournament_has_complete_uniform_views() -> None:
    first, second = _entrants()
    result = run_tournament(
        (first, shared_policy(second)),
        maps=[0],
        episodes_per_pair=2,
        max_steps=1,
        num_envs=2,
        chunk_size=1,
    )
    assert result.status == "complete"
    assert len(result.headline_metrics) == 2
    assert result.table("tournament_headline_metrics")["games_played"].tolist() == [
        2,
        2,
    ]
    assert result.table("matches")["episode_id"].tolist() == [1, 2]
    assert len(result.table("tournament_rankings")["rank"]) == 2
    assert result.matches[0]["team_a_policy"] == result.matches[1]["team_a_policy"]


def test_complete_games_finalize_once_without_replaying_and_inherit_saved_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("marl_battlegrounds.evaluation.tournament")
    result = run_tournament(
        _entrants(), maps=[0], episodes_per_pair=2, max_steps=1, output_dir=tmp_path
    )
    assert result.paths is not None
    directory = result.paths["run_details"].parent
    path = directory / "run_details.json"
    manifest = json.loads(path.read_bytes())
    manifest.pop("tournament_summary")
    for name in (
        "tournament_results.csv",
        "matchup_results.csv",
        "map_results.csv",
        "tournament_headline_metrics.csv",
    ):
        manifest["tables"].pop(name)
    path.write_text(json.dumps(manifest))
    original = module.summarize_tournament
    fits: list[int] = []

    def counted_fit(*args: object, **kwargs: object) -> object:
        fits.append(1)
        return original(*args, **kwargs)

    def forbidden_execution(*args: object, **kwargs: object) -> None:
        raise AssertionError("saved completed games must not run again")

    monkeypatch.setattr(module, "summarize_tournament", counted_fit)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    monkeypatch.setattr(evaluator, "_jax_chunk", forbidden_execution)
    resumed = run_tournament(_entrants(), resume_from=directory)
    assert len(fits) == 1
    assert resumed.matches == result.matches
    assert resumed.headline_metrics == result.headline_metrics
    again = run_tournament(_entrants(), resume_from=directory)
    assert len(fits) == 1
    assert again.headline_metrics == result.headline_metrics
    changed = json.loads(path.read_bytes())
    pair = next(
        entry for entry in changed["passes"].values() if entry["pass_id"] == "pair-1"
    )
    pair["details"]["seed"] = 999
    path.write_text(json.dumps(changed))
    raw = directory / "match_results.csv"
    raw.write_bytes(raw.read_bytes() + b"uncommitted suffix must remain untouched\n")
    snapshot = {
        file: file.read_bytes() for file in directory.rglob("*") if file.is_file()
    }
    with pytest.raises(ValueError, match="scientific conditions"):
        run_tournament(_entrants(), resume_from=directory)
    assert {file: file.read_bytes() for file in snapshot} == snapshot


def test_compatible_legacy_partial_run_keeps_reversed_teams_and_original_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation.evaluate import (
        _run_evaluation,
        normalize_episode_specs,
        policy_description,
    )
    from marl_battlegrounds.evaluation.evaluation_conditions import config_record
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.tasks import canonical_tournament_rosters

    first, second = _entrants()
    a, b = canonical_tournament_rosters()
    config = normalize_episode_specs([0], 1, a, b, 20, 1)[0].env_config
    source_id, _ = config_record(config)
    details: dict[str, object] = {
        "seed": 0,
        "rng_protocol": "episode-fold-in-v1",
        "policies": [
            policy_description(p, p.variables, p.initial_carry, include_digests=True)
            for p in (first, second)
        ],
        "map_ids": [0],
        "episodes_per_pair": 4,
        "num_matches": 4,
        "score_threshold": 20,
        "max_steps": 1,
        "metrics": "priority",
        "full_metrics_episodes": [],
        "replay_episodes": [],
        "opponent_weights": None,
        "configuration_ids_by_map": {"0": source_id},
        "schedule_digest": "legacy-fixture-schedule",
    }
    specs = [
        EpisodeSpec(
            i,
            config,
            0,
            (i + 1) // 2,
            metadata={
                "block_id": (i + 1) // 2,
                "bootstrap_group": None,
                "paired_comparison_key": f"block-{(i + 1) // 2}",
            },
        )
        for i in (1, 3)
    ]
    with RunWriter(
        tmp_path, phase="tournament", pass_id="schedule", details=details
    ) as writer:
        _run_evaluation(
            first,
            second,
            specs,
            writer=writer,
            phase="tournament",
            pass_id="pair-side-1",
            num_envs=2,
            chunk_size=1,
        )
        directory = writer.run_dir
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    original = evaluator._run_evaluation
    executions: list[tuple[int, ...]] = []

    def observed_legacy(*args: object, **kwargs: object) -> EvaluationResult:
        assert kwargs.get("contract") is None
        result = cast(EvaluationResult, original(*args, **kwargs))
        executions.append(tuple(row.episode_id for row in result.episodes))
        return result

    monkeypatch.setattr(evaluator, "_run_evaluation", observed_legacy)
    result = run_tournament(
        (first, second), resume_from=directory, num_envs=2, chunk_size=1
    )
    assert executions == [(), (2, 4)]
    assert result.status == "complete"
    assert result.headline_metrics == ()
    assert result.matches[0]["team_a_policy"] == result.matches[1]["team_b_policy"]
    assert result.matches[0]["config_id"] == result.matches[1]["config_id"]
