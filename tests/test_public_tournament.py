"""Small complete tournaments exercise paired execution, selection and resumption."""

import csv
from collections.abc import Sequence
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, EvaluationResult
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import Policy, policy
from marl_battlegrounds.evaluation.recording_context import capture_recording_provenance
from marl_battlegrounds.evaluation.run_writer import MATCH_COLUMNS
from marl_battlegrounds.evaluation.tournament import run_tournament
from marl_battlegrounds.tasks import CANONICAL_TDM_EVALUATION_MAP_IDS, list_tdm_maps


def _entrants(count: int = 2) -> tuple[Policy, ...]:
    return tuple(
        replace(policy("random"), name=f"model-{index}") for index in range(count)
    )


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
    assert len(cast(dict[str, object], result.metadata["configurations"])) == 5
    for map_id in CANONICAL_TDM_EVALUATION_MAP_IDS:
        assert sum(row["map_id"] == map_id for row in result.matches) == 4
    for first, second in zip(result.matches[::2], result.matches[1::2], strict=True):
        assert first["block_id"] == second["block_id"]
        assert first["seed_id"] == second["seed_id"]
        assert first["team_a_policy"] == second["team_b_policy"]
    assert len(result.tournament_results) == len(result.matchup_results) == 2
    assert len(result.map_results) == 10
    assert all(row["expected_score"] == 0.5 for row in result.tournament_results)
    assert result.full_metrics == {} and result.replays == () and result.paths is None
    assert list(tmp_path.iterdir()) == []


def test_three_entrants_preserve_global_ids_and_sparse_diagnostic_independence() -> (
    None
):
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
    assert [row["episode_length"] for row in result.matches] == [
        None,
        1,
        None,
        None,
        None,
        None,
    ]
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
    assert resumed.matches == result.matches
    assert resumed.tournament_results == result.tournament_results
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
            _entrants(),
            maps=[12],
            episodes_per_pair=4,
            max_steps=1,
            chunk_size=2,
            output_dir=tmp_path,
        )
    assert executions == [(1, 3)]
    run_dir = next(tmp_path.iterdir())
    assert not (run_dir / "tournament_results.csv").exists()
    with (run_dir / "match_results.csv").open(newline="") as stream:
        assert [row["episode_id"] for row in csv.DictReader(stream)] == ["1", "3"]

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
        _entrants(),
        maps=[12],
        episodes_per_pair=4,
        max_steps=1,
        chunk_size=2,
        resume_from=run_dir,
    )
    assert executions == [(1, 3), (), (2, 4)]
    assert [row["episode_id"] for row in result.matches] == [1, 2, 3, 4]


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
