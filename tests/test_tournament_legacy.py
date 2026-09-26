"""Check shared job dispatch and unchanged historical tournament recovery.

The shared iterator preserves job settings and releases each loaded pair before
returning its result. Errors stop submission of later work. Old list folders
are built with the existing writer, keep pinned maps and pass IDs on recovery,
and never gain canonical plan files or invented physical-pair claims.
"""

# Compatibility tests inspect the saved historical format and private dispatcher.
# pyright: reportPrivateUsage=false

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest
from tests.tournament_legacy_fixtures import create_legacy_tournament

from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, EvaluationResult
from marl_battlegrounds.evaluation.policy_execution import Policy, policy
from marl_battlegrounds.evaluation.recording_context import capture_recording_provenance
from marl_battlegrounds.evaluation.tournament import _resume_legacy_tournament
from marl_battlegrounds.evaluation.tournament_execution import (
    TournamentJob,
    execute_tournament_jobs,
)


def _entrants() -> tuple[Policy, Policy]:
    return (
        replace(policy("random"), name="old-a"),
        replace(policy("random"), name="old-b"),
    )


def test_job_dispatch_preserves_arguments_and_releases_before_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("marl_battlegrounds.evaluation.evaluate")
    events: list[object] = []
    first, second = _entrants()
    result = cast(EvaluationResult, object())
    writer = object()
    maps = {12: {"name": "Pinned"}}

    @contextmanager
    def load(job: TournamentJob) -> Generator[tuple[Policy, Policy]]:
        events.append(("load", job.job_id))
        try:
            yield first, second
        finally:
            events.append(("release", job.job_id))

    def execute(
        a: Policy, b: Policy, specs: object, **options: object
    ) -> EvaluationResult:
        assert (a, b) == (first, second)
        assert specs == ()
        events.append(options)
        return result

    monkeypatch.setattr(module, "_evaluate_tournament_episodes", execute)
    monkeypatch.setattr(module, "_run_evaluation", execute)
    jobs = (
        TournamentJob(7, "a", "b", (), 91, "tournament", "pair-2", (9,), (11,)),
        TournamentJob(
            8, "b", "a", (), 92, "tournament", "pair-side-3", legacy_contract=True
        ),
    )
    iterator = execute_tournament_jobs(
        jobs,
        load,
        writer=cast(Any, writer),
        run_id="saved-run",
        num_envs=3,
        metrics="none",
        chunk_size=5,
        registered_maps=maps,
    )
    assert next(iterator) == (jobs[0], result)
    assert events[-1] == ("release", 7)
    assert events[1] == {
        "seed": 91,
        "num_envs": 3,
        "metrics": "none",
        "full_metrics_episodes": (9,),
        "replay_episodes": (11,),
        "writer": writer,
        "phase": "tournament",
        "pass_id": "pair-2",
        "chunk_size": 5,
        "run_id": "saved-run",
        "registered_maps": maps,
    }
    assert next(iterator) == (jobs[1], result)
    assert events[-1] == ("release", 8)
    assert cast(dict[str, object], events[4])["registered_maps"] is None
    assert cast(dict[str, object], events[4])["pass_id"] == "pair-side-3"
    assert list(iterator) == []


def test_failed_job_releases_pair_without_starting_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("marl_battlegrounds.evaluation.evaluate")
    events: list[str] = []

    @contextmanager
    def load(job: TournamentJob) -> Generator[tuple[Policy, Policy]]:
        events.append(f"load-{job.job_id}")
        try:
            yield _entrants()
        finally:
            events.append(f"release-{job.job_id}")

    def fail(*args: object, **kwargs: object) -> EvaluationResult:
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(module, "_evaluate_tournament_episodes", fail)
    jobs = tuple(
        TournamentJob(i, "a", "b", (), 1, "tournament", str(i)) for i in (1, 2)
    )
    with pytest.raises(RuntimeError, match="controlled failure"):
        list(
            execute_tournament_jobs(
                jobs,
                load,
                writer=None,
                run_id="run",
                num_envs=1,
                metrics="none",
                chunk_size=1,
            )
        )
    assert events == ["load-1", "release-1"]


def test_historical_adapter_refuses_a_new_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires resume_from"):
        _resume_legacy_tournament(_entrants(), resume_from=None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("completed_passes", [0, 1])
def test_historical_maps_passes_and_saved_results_survive_shared_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_passes: int
) -> None:
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", same_source)
    methods = _entrants()
    directory = create_legacy_tournament(
        tmp_path,
        methods,
        maps=(12, 13),
        episodes_per_pair=4,
        completed_passes=completed_passes,
    )
    original = json.loads((directory / "run_details.json").read_bytes())
    maps = original["details"]["configuration_ids_by_map"]
    module = import_module("marl_battlegrounds.evaluation.tournament")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Historical resume must not select current maps")

    monkeypatch.setattr(module, "normalize_episode_specs", forbidden)
    result = _resume_legacy_tournament(
        methods, resume_from=directory, num_envs=2, chunk_size=1
    )
    assert [row["episode_id"] for row in result.matches] == [1, 2, 3, 4]
    assert result.metadata["configuration_ids_by_map"] == maps
    assert {row["pass_id"] for row in result.matches} == {"pair-1"}
    assert result.metadata["schedule"] == original["details"]["schedule"]
    manifest = json.loads((directory / "run_details.json").read_bytes())
    assert "tournament_reuse" not in manifest
    assert not (directory / "tournament_config.json").exists()
    assert not (directory / "tournament_games.jsonl").exists()
    before = {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(evaluator, "_evaluate_tournament_episodes", forbidden)
    again = _resume_legacy_tournament(methods, resume_from=directory)
    assert again.matches == result.matches
    assert again.tournament_results == result.tournament_results
    assert {path: path.read_bytes() for path in before} == before


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
    config = normalize_episode_specs([0], 1, a, b, 20, 1, red_zone_depth=0.0)[
        0
    ].env_config
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
    result = _resume_legacy_tournament(
        (first, second), resume_from=directory, num_envs=2, chunk_size=1
    )
    assert executions == [(), (2, 4)]
    assert result.status == "complete"
    assert result.headline_metrics == ()
    assert result.matches[0]["team_a_policy"] == result.matches[1]["team_b_policy"]
    assert result.matches[0]["config_id"] == result.matches[1]["config_id"]
