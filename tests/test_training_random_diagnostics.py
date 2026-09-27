"""Check opt-in Random captures, shared evidence and exact runner recovery.

Small CPU learners use real K20/H300 Random games on the five fixed validation
maps. The checks keep native M8 records, prove diagnostics do not change learner
state or keys, and reject altered shared evidence before recovery writes.
New Random records retain sampling facts and unavailable public bounds through
initialization reuse and resume; one paired draw per map is not an interval.
A PQN-VDN run binds initialization reuse to its greedy identity over both
parameters and BatchNorm statistics (a statistics-only change is a different
actor), and its results name the method and optimizer count.
These are engineering fixtures, not learning or configuration-screen trials.
The shared runs use red_zone_depth 20.0, as wide as every map, so each death is
a Red Zone death: every saved Random record (task schema 3) carries the depth
and each cell's points are exactly twice its recorded kills, with at least one
kill played. For a record with a depth the progress display reports recorded
kills (1 kill where the score is 2); an older record reads kills from scores.
"""

# Recovery injection and strict evidence checks inspect owned host boundaries.
# pyright: reportPrivateUsage=false

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.training import _content, checkpoints, runner, validation
from marl_battlegrounds.training._run_io import progress_text, validate_host_state
from marl_battlegrounds.training.runner import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
)


def _state_digest(state: Any) -> str:  # noqa: ANN401
    def plain(leaf: jax.Array) -> jax.Array:
        return (
            jax.random.key_data(leaf)
            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            else leaf
        )

    values = jax.tree.map(plain, state)
    return tree_digest(values)


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _no_content(**kwargs: object) -> SimpleNamespace:
    del kwargs
    return SimpleNamespace(source_configs=SimpleNamespace(max_steps=(300,)))


def _isolate_initialization_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from marl_battlegrounds.training import collection

    # These two tests stop at actor identity. Real workflow tests below own
    # content, frozen member binding and deployed-team validation preparation.
    def retain(value: object, references: object) -> object:
        del references
        return value

    def undeployed(*args: object) -> tuple[dict[str, object], None, dict[str, str]]:
        del args
        return {}, None, {}

    def no_teams(*args: object, **kwargs: object) -> None:
        del args, kwargs

    monkeypatch.setattr(_content, "prepare_training_content", _no_content)
    for name in ("_bind_opponent_references", "_bind_partner_references"):
        monkeypatch.setattr(collection, name, retain)
    monkeypatch.setattr(runner, "_validation_setup", undeployed)
    monkeypatch.setattr(validation, "prepare_validation_teams", no_teams)


def test_initialization_reuse_binds_the_selected_method_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import learner

    variables = {"weights": jnp.arange(4, dtype=jnp.float32)}
    state = SimpleNamespace(
        carry=SimpleNamespace(history=SimpleNamespace(current_variables=variables))
    )

    def initialize(**kwargs: object) -> tuple[SimpleNamespace, SimpleNamespace]:
        return SimpleNamespace(learner_actor=None, actor=None), state

    monkeypatch.setattr(learner, "init_learner", initialize)
    _isolate_initialization_identity(monkeypatch)
    digests: list[str] = []

    def read_initialization(
        path: object,
        *,
        actor_digest: str,
        seed_pairs: int,
        red_zone_depth: float,
        deployment: object,
    ) -> None:
        assert path == str(tmp_path / "initialization.json") and seed_pairs == 1
        # The shared result must have been recorded at the run's own depth.
        assert red_zone_depth == 5.0 and deployment is None
        digests.append(actor_digest)
        raise ValueError("Stop at the initialization identity check")

    monkeypatch.setattr(validation, "read_random_initialization", read_initialization)
    for method in ("mappo", "ippo", "ff_mappo", "ff_ippo"):
        config = config_from_dict(
            {
                "keep_past": 0,
                "method": method,
                "num_envs": 4,
                "total_env_steps": 8,
                "ppo": {"rollout_length": 2, "spawn_frame": "world"},
                "random_diagnostic_seed_pairs": 1,
                "random_initialization_result": str(tmp_path / "initialization.json"),
            }
        )
        with pytest.raises(ValueError, match="initialization identity check"):
            runner.train(config, output_dir=tmp_path / method)
        assert not (tmp_path / method).exists()
    assert digests[0] == tree_digest(variables)
    assert len(set(digests)) == 4


def test_pqn_initialization_reuse_binds_parameters_and_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.baselines import pqn
    from marl_battlegrounds.training import pqn_learner

    network = pqn.initialize_pqn(jax.random.key(3), planned_learning_blocks=1).network

    def shift(leaf: jax.Array) -> jax.Array:
        return leaf + 1

    shifted = network._replace(batch_stats=jax.tree.map(shift, network.batch_stats))
    digests: list[str] = []

    def read_initialization(
        path: object,
        *,
        actor_digest: str,
        seed_pairs: int,
        red_zone_depth: float,
        deployment: object,
    ) -> None:
        del path, seed_pairs
        assert red_zone_depth == 5.0 and deployment is None
        digests.append(actor_digest)
        raise ValueError("Stop at the initialization identity check")

    monkeypatch.setattr(validation, "read_random_initialization", read_initialization)
    _isolate_initialization_identity(monkeypatch)
    config = config_from_dict(
        {
            "keep_past": 0,
            "method": "pqn_vdn",
            "num_envs": 4,
            "total_env_steps": 64,
            "pqn": {
                "rollout_length": 4,
                "memory_window": 2,
                "epochs": 1,
                "num_minibatches": 2,
            },
            "random_diagnostic_seed_pairs": 1,
            "random_initialization_result": str(tmp_path / "initialization.json"),
        }
    )
    for index, variables in enumerate((network, shifted)):
        state = SimpleNamespace(
            carry=SimpleNamespace(
                history=SimpleNamespace(
                    current_variables=pqn.PQNActorVariables(variables, jnp.float32(1.0))
                )
            )
        )

        def initialize(
            *, chosen: SimpleNamespace = state, **kwargs: object
        ) -> tuple[SimpleNamespace, SimpleNamespace]:
            del kwargs
            return SimpleNamespace(learner_actor=None, actor=None), chosen

        monkeypatch.setattr(pqn_learner, "init_pqn_learner", initialize)
        with pytest.raises(ValueError, match="initialization identity check"):
            runner.train(config, output_dir=tmp_path / f"run-{index}")
        assert not (tmp_path / f"run-{index}").exists()
    expected = checkpoints._inference_digest(
        {
            "kind": "actor",
            "actor_digest": tree_digest(checkpoints._pqn_actor_item(network)),
            "input_scale": 1.0,
            "spawn_frame": "left",
            "epsilon": 0.0,
            "tie_rule": "first_legal_maximum",
            "schemas": checkpoints.checkpoint_schemas("pqn_vdn"),
        }
    )
    assert digests[0] == expected and digests[1] != digests[0]
    assert validation._method_fields({"method": "pqn_vdn", "optimizer_steps": 3}) == {
        "method": "pqn_vdn",
        "optimizer_steps": 3,
    }
    assert validation._method_fields({"optimizer_steps": 3}) == {}


@pytest.fixture(scope="module")
def random_runs(tmp_path_factory: pytest.TempPathFactory) -> Any:  # noqa: ANN401
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )

    directory = tmp_path_factory.mktemp("random-diagnostics")
    base = TrainConfig(
        keep_past=0,
        seed=1701,
        num_envs=4,
        total_env_steps=8,
        # Every map is 20 units wide, so each death is inside its victim's
        # own Red Zone: every kill gives 2 points.
        red_zone_depth=20.0,
        ppo=PPOConfig(rollout_length=1, epochs=1, input_scale=0.01),
        shaping=True,
        shaping_mode="score_delta",
        metrics="none",
        recording=True,
        checkpoint_env_steps=(4, 8),
        checkpoint_interval_updates=1,
        verbose=False,
    )
    identity = checkpoints.runtime_identity()
    provenance = capture_recording_provenance()
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    final_states: dict[str, str] = {}
    calls: list[tuple[str, int]] = []
    original_report = runner._Run.report
    original_random = runner._Run.validate_random

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    def report(self: runner._Run) -> tuple[Path, ...]:
        final_states[self.root.name] = _state_digest(self.state)
        return original_report(self)

    def random_check(self: runner._Run, actor: Path) -> None:
        before = _state_digest(self.state)
        calls.append((self.root.name, self.host["env_steps"]))
        original_random(self, actor)
        assert _state_digest(self.state) == before

    with pytest.MonkeyPatch.context() as patch:
        # Separate persistence tests own source-drift rejection. Pin identities
        # while other agents write unrelated approved source during this suite.
        patch.setattr(checkpoints, "runtime_identity", lambda: identity)
        patch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
        patch.setattr(runner._Run, "report", report)
        patch.setattr(runner._Run, "validate_random", random_check)
        baseline = runner.train(base, output_dir=directory / "baseline")
        config = replace(base, random_diagnostic_seed_pairs=1)
        original_chunk = evaluator._jax_system_chunk
        saw_completed = False

        def interrupt_games(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            nonlocal saw_completed
            if saw_completed:
                raise RuntimeError("Declared partial Random interruption")
            result = original_chunk(*args, **kwargs)
            saw_completed = bool(jax.device_get(result[0].completed.completed).any())
            return result

        patch.setattr(evaluator, "_jax_system_chunk", interrupt_games)
        with pytest.raises(RuntimeError, match="partial Random"):
            runner.train(config, output_dir=directory / "first")
        first_root = directory / "first"
        (details_path,) = (first_root / "validation").rglob("run_details.json")
        partial = next(iter(json.loads(details_path.read_text())["passes"].values()))
        assert 0 < len(partial["completed_episode_ids"]) < 10
        partial_games = {
            path: path.read_bytes() for path in details_path.parent.rglob("*.csv")
        }
        assert partial_games
        pointer = json.loads((first_root / "latest_checkpoint.json").read_text())
        interrupted = checkpoints.read_checkpoint_details(
            first_root / pointer["relative_path"]
        )
        assert interrupted["counters"] == {"env_steps": 0, "updates": 0}
        assert interrupted["metadata"]["host_state"]["pending"] == {
            "routine": False,
            "random": True,
        }
        patch.setattr(evaluator, "_jax_system_chunk", original_chunk)
        first = runner.train(resume_from=first_root / pointer["relative_path"])
        assert all(
            path.read_bytes().startswith(raw) for path, raw in partial_games.items()
        )
        records = json.loads((first.run_dir / "random_diagnostics.json").read_text())
        shared = directory / "initialization.json"
        shared.write_text(json.dumps(records[0]))
        requested: list[int] = []
        original_validate = validation.validate_random

        def validate(actor: str | Path, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
            requested.append(checkpoints.artifact_identity(actor)["env_steps"])
            return original_validate(actor, **kwargs)

        patch.setattr(validation, "validate_random", validate)
        reused_config = replace(config, random_initialization_result=str(shared))
        reused = runner.train(reused_config, output_dir=directory / "reused")
        assert requested == [4, 8]
        yield {
            "directory": directory,
            "config": config,
            "reused_config": reused_config,
            "baseline": baseline,
            "first": first,
            "reused": reused,
            "shared": shared,
            "records": records,
            "states": final_states,
            "calls": calls,
        }


@pytest.mark.parametrize("pairs", [0, -1, True, 1.5, "4"])
def test_invalid_random_pair_count_is_rejected(pairs: Any) -> None:  # noqa: ANN401
    with pytest.raises(ValueError, match="random_diagnostic_seed_pairs"):
        TrainConfig(random_diagnostic_seed_pairs=pairs)


@pytest.mark.parametrize("reference", ["", 3, True])
def test_invalid_initialization_reference_is_rejected(reference: Any) -> None:  # noqa: ANN401
    with pytest.raises(ValueError, match="random_initialization_result"):
        TrainConfig(
            random_diagnostic_seed_pairs=4,
            random_initialization_result=reference,
        )


def test_random_settings_are_optional_and_roundtrip() -> None:
    historical = config_from_dict({"num_envs": 4})
    assert historical.random_diagnostic_seed_pairs is None
    assert historical.random_initialization_result is None
    declared = replace(
        historical,
        random_diagnostic_seed_pairs=4,
        random_initialization_result="/frozen/initialization.json",
    )
    assert config_from_dict(config_to_dict(declared)) == declared
    with pytest.raises(ValueError, match="enabled Random"):
        TrainConfig(random_initialization_result="/frozen/initialization.json")


def _progress_record(steps: int, kills: float, deaths: float) -> dict[str, Any]:
    return {
        "task_id": f"random-{steps}",
        "env_steps": steps,
        "games": 40,
        "wall_seconds": steps / 100,
        "training_seconds": steps / 200,
        "score": 0.5,
        "reused_initialization": steps == 0,
        "cells": [
            {
                "map_id": index,
                "mean_team_a_score": kills + index,
                "mean_team_b_score": deaths,
                "wins": 0,
                "draws": 8,
                "losses": 0,
            }
            for index in range(5)
        ],
    }


def test_random_progress_compares_combat_with_shared_initialization() -> None:
    initial = _progress_record(0, 1, 4)
    latest = _progress_record(4096, 2, 2)
    records = [latest, initial]
    before = deepcopy(records)
    result = runner._random_progress(records)
    assert result is not None
    assert result == {
        "opponent": "Random",
        "task_id": "random-4096",
        "initial_task_id": "random-0",
        "env_steps": 4096,
        "games": 40,
        "wall_seconds": 40.96,
        "training_seconds": 20.48,
        "score": 0.5,
        "wins": 0,
        "draws": 40,
        "losses": 0,
        "mean_kills_for": 4.0,
        "mean_kills_against": 2.0,
        "kill_margin": 2.0,
        "initial_kill_margin": -1.0,
        "kill_margin_change": 3.0,
    }
    assert records == before


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), -1, True])
def test_random_progress_missing_scores_are_not_zero(missing: object) -> None:
    initial = _progress_record(0, 1, 4)
    latest = _progress_record(4096, 2, 2)
    latest["cells"][0]["mean_team_a_score"] = missing
    result = runner._random_progress([initial, latest])
    assert result is not None
    assert result["mean_kills_for"] is None
    assert result["mean_kills_against"] is None
    assert result["kill_margin"] is None
    assert result["kill_margin_change"] is None
    assert result["initial_kill_margin"] == -1
    assert result["draws"] == 40


def test_random_progress_reports_kills_not_points_for_red_zone_records() -> None:
    record = _progress_record(4096, 2, 0)
    for cell in record["cells"]:
        cell.update(mean_team_a_score=2.0, mean_team_a_kills=1.0, mean_team_b_kills=0)
    # Without a depth the record predates Red Zone: its scores were its kills.
    legacy = runner._random_progress([record])
    assert legacy is not None and legacy["mean_kills_for"] == 2.0
    result = runner._random_progress([{**record, "red_zone_depth": 5.0}])
    assert result is not None
    assert (result["mean_kills_for"], result["mean_kills_against"]) == (1.0, 0.0)
    assert result["kill_margin"] == 1.0


def test_random_progress_without_initialization_has_no_invented_change() -> None:
    assert runner._random_progress([]) is None
    result = runner._random_progress([_progress_record(4096, 2, 2)])
    assert result is not None
    assert result["kill_margin"] == 2
    assert result["initial_task_id"] is None
    assert result["initial_kill_margin"] is None
    assert result["kill_margin_change"] is None


def test_random_capture_publishes_combat_from_existing_host_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    run = object.__new__(runner._Run)
    run.root = tmp_path
    run.config = TrainConfig(random_diagnostic_seed_pairs=4)
    run.validation_options = {}
    run.host = {
        "env_steps": 4096,
        "random_results": [_progress_record(0, 1, 4)],
        "training_seconds": 20.0,
        "validation_games": 40,
        "validation_seconds": 1.0,
    }
    run.start = time.monotonic()
    run.prior_elapsed = 40.0
    run.created_at = time.time() - 40
    run.status = {}
    states: list[dict[str, Any]] = []
    calls: list[Path] = []
    description_reads: list[Path] = []
    checkpoint_id = "a" * 64
    actor = tmp_path / "mappo_actor_step_000000004096"
    actor.mkdir()

    def describe(path: str | Path) -> dict[str, Any]:
        assert Path(path) == actor
        description_reads.append(Path(path))
        return {
            "kind": "actor",
            "schemas": checkpoints.checkpoint_schemas("mappo"),
            "metadata": {"checkpoint_id": checkpoint_id, "env_steps": 4096},
        }

    def publish(phase: str, **facts: object) -> None:
        run.status.update(phase=phase, **facts)
        states.append(deepcopy(run.status))

    def evaluate(actor: Path, *, output_dir: Path, **_: object) -> dict[str, Any]:
        calls.append(actor)
        assert output_dir == (
            tmp_path
            / "validation"
            / f"random_mappo_actor_step_000000004096_{checkpoint_id}"
        )
        return {**_progress_record(4096, 2, 2), "checkpoint_id": checkpoint_id}

    def event(*_args: object, **_kwargs: object) -> None:
        pass

    monkeypatch.setattr(run, "set_status", publish)
    monkeypatch.setattr(run, "event", event)
    monkeypatch.setattr(validation, "validate_random", evaluate)
    monkeypatch.setattr(checkpoints, "read_checkpoint_description", describe)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    run.validate_random(actor)
    saved = json.loads((tmp_path / "random_diagnostics.json").read_text())
    assert calls == [actor]
    assert description_reads == [actor, actor]
    assert saved[-1]["checkpoint_id"] == checkpoint_id != actor.name
    assert states[-1]["random_validation"] == runner._random_progress(saved)
    assert states[-1]["random_validation"]["kill_margin_change"] == 3
    assert run.host["validation_games"] == 80
    run.validate_random(actor)
    assert calls == [actor]
    assert description_reads == [actor, actor]
    assert len(states) == 2


def test_native_random_captures_preserve_complete_learning_state(
    random_runs: dict[str, Any],
) -> None:
    data = random_runs
    assert len(set(data["states"].values())) == 1
    assert not (data["baseline"].run_dir / "random_diagnostics.json").exists()
    assert data["calls"] == [("first", 0)] + [
        (name, point) for name in ("first", "reused") for point in (0, 4, 8)
    ]
    kills = 0.0
    for result in (data["first"], data["reused"]):
        records = json.loads((result.run_dir / "random_diagnostics.json").read_text())
        assert [row["env_steps"] for row in records] == [0, 4, 8]
        assert all(row["games"] == 10 and row["complete"] for row in records)
        for row in records:
            assert row["sampling_evidence"]["determinism"] == "Stochastic"
            assert row["independent_blocks"] == row["declared_blocks"] == 5
            # One paired draw per map supports a mean, but no within-map interval.
            assert row["ci_low"] is row["ci_high"] is None
            assert "stratum" in row["interval_status"]
            assert row["conditional_ci_low"] <= row["conditional_ci_high"]
        # At depth 20 every kill is a Red Zone kill: points are twice the kills.
        for row in records:
            assert row["red_zone_depth"] == 20.0 and row["schema_version"] == 3
            for cell in row["cells"]:
                for team in ("a", "b"):
                    assert cell[f"mean_team_{team}_score"] == (
                        2 * cell[f"mean_team_{team}_kills"]
                    )
                    kills += cell[f"mean_team_{team}_kills"]
        assert all(row["root"] == 19_043_001 for row in records)
        assert all(row["wall_seconds"] >= 0 for row in records)
        assert records[0]["training_seconds"] == 0
        assert records[-1]["training_seconds"] > 0
        assert result.selected_actor is None
        status = json.loads((result.run_dir / "status.json").read_text())
        assert (
            sum(status[f"validation_{name}"] for name in ("wins", "draws", "losses"))
            == 10
        )
        assert status["random_validation"] == runner._random_progress(records)
        assert "Random" in progress_text(status)
        assert (
            len((result.run_dir / "training_updates.jsonl").read_text().splitlines())
            == 2
        )
    assert kills > 0
    reused = json.loads(
        (data["reused"].run_dir / "random_diagnostics.json").read_text()
    )[0]
    original = data["records"][0]
    assert reused["reused_initialization"] is True
    assert reused["reference_path"] == str(data["shared"])
    for name in (
        "task_id",
        "checkpoint_id",
        "actor_digest",
        "actor_path",
        "summary_path",
        "pass_paths",
        "cells",
    ):
        assert reused[name] == original[name]
    assert not checkpoints.validation_directory(
        data["reused"].run_dir,
        "random",
        reused["checkpoint_id"],
        actor=Path(reused["actor_path"]),
    ).exists()


def test_screen_analysis_reads_real_random_game_evidence(
    random_runs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import analysis

    package = tmp_path / "screen"
    run = package / "jobs" / "b4-t1-fixture" / "run"
    run.mkdir(parents=True)
    source = random_runs["first"].run_dir
    # Keep original absolute actor and M8 evidence paths. Only small host records
    # need copying; this one CPU case does not pretend to be the 12-case study.
    for name in (
        "run_details.json",
        "status.json",
        "run_events.jsonl",
        "exposure.json",
        "training_updates.jsonl",
        "random_diagnostics.json",
        "latest_checkpoint.json",
    ):
        shutil.copy2(source / name, run / name)
    for description in (source / "checkpoints").glob("*/checkpoint_details.json"):
        target = run / description.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(description, target)
    config = json.loads((run / "run_details.json").read_text())["config"]
    case = {"name": "b4-t1-fixture", "num_envs": 4, "rollout_length": 1}
    (package / "declaration.json").write_text(json.dumps({"cases": [case]}))
    (package / "budgets.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        **case,
                        "total_env_steps": 8,
                        "updates": 2,
                        "checkpoint_env_steps": [4, 8],
                        "median_update_seconds": 1.0,
                        "config": config,
                    }
                ]
            }
        )
    )
    (package / "study.json").write_text(json.dumps({"status": "running"}))

    def no_plot(*_: object) -> dict[str, str]:
        return {}

    monkeypatch.setattr(analysis, "_screen_plot", no_plot)
    report = analysis.analyze_screen(package)
    assert report["complete"]
    assert report["evidence_errors"] == []
    assert report["unique_validation_tasks"] == 3
    assert report["cases"][0]["final_score"] == random_runs["records"][-1]["score"]


def test_completed_random_games_resume_without_new_experience(
    random_runs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = random_runs["reused"].run_dir
    saved = json.loads((root / "latest_checkpoint.json").read_text())
    checkpoint = root / saved["relative_path"]
    description = checkpoints.read_checkpoint_details(checkpoint)
    assert description["metadata"]["host_state"]["pending"] == {
        "routine": False,
        "random": True,
    }
    before_rows = (root / "training_updates.jsonl").read_bytes()
    before_games = {
        path: path.read_bytes() for path in (root / "validation").rglob("*.csv")
    }
    actor_before = checkpoints.artifact_identity(random_runs["reused"].final_actor)
    collection = import_module("marl_battlegrounds.training.collection")
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")

    def forbidden(*args: object, **kwargs: object) -> Any:  # noqa: ANN401
        pytest.fail("Completed final recovery must not collect or evaluate new games")

    monkeypatch.setattr(collection, "collect_training_rollout", forbidden)
    monkeypatch.setattr(evaluator, "evaluate", forbidden)
    restored = runner.train(resume_from=checkpoint)
    assert restored.completed_env_steps == 8
    assert (root / "training_updates.jsonl").read_bytes() == before_rows
    assert all(path.read_bytes() == contents for path, contents in before_games.items())
    assert checkpoints.artifact_identity(restored.final_actor) == actor_before
    assert not (root / "checkpoint_recovery.json").exists()
    records = json.loads((root / "random_diagnostics.json").read_text())
    assert [row["env_steps"] for row in records] == [0, 4, 8]


def test_shared_inference_mismatch_rejects_before_output(
    random_runs: dict[str, Any], tmp_path: Path
) -> None:
    config = random_runs["reused_config"]
    destination = tmp_path / "wrong-scale"
    with pytest.raises(ValueError, match="expected actor"):
        runner.train(
            replace(config, ppo=replace(config.ppo, input_scale=1.0)),
            output_dir=destination,
        )
    assert not destination.exists()


def test_changed_shared_result_rejects_before_recovery_mutation(
    random_runs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, shared = random_runs["reused"].run_dir, random_runs["shared"]
    pointer = json.loads((root / "latest_checkpoint.json").read_text())
    before = _files(root)
    original = shared.read_bytes()
    corrupted = json.loads(original)
    corrupted["score"] = 0.123
    shared.write_text(json.dumps(corrupted))

    def forbidden(*args: object, **kwargs: object) -> Any:  # noqa: ANN401
        pytest.fail("Invalid shared evidence must reject before recovery writes")

    monkeypatch.setattr(checkpoints, "resume_recording", forbidden)
    monkeypatch.setattr(runner, "restore_log_cursor", forbidden)
    try:
        with pytest.raises(ValueError, match="summary or task"):
            runner.train(resume_from=root / pointer["relative_path"])
    finally:
        shared.write_bytes(original)
    assert _files(root) == before
    assert not (root / "checkpoint_recovery.json").exists()


@pytest.mark.parametrize(
    "change", ["pairs", "task", "summary", "missing", "run", "seed"]
)
def test_random_reference_checks_original_task_and_rows(
    random_runs: dict[str, Any], change: str
) -> None:
    record = deepcopy(random_runs["records"][0])
    pairs = 1
    expected: dict[str, Any] = {}
    target: Path | None = None
    original: bytes | None = None
    if change == "pairs":
        pairs = 2
    elif change == "run":
        expected["run_id"] = "another-run"
    elif change == "seed":
        expected["seed"] = random_runs["config"].seed + 1
    elif change == "task":
        target = Path(record["summary_path"]).with_name("task.json")
        original = target.read_bytes()
        value = json.loads(original)
        value["root"] += 1
        target.write_text(json.dumps(value))
    elif change == "summary":
        record["cells"][0]["wins"] += 1
    else:
        target = Path(record["summary_path"])
        original = target.read_bytes()
        target.unlink()
    try:
        with pytest.raises(ValueError):
            validation.verify_random_result(
                record,
                actor_digest=record["actor_digest"],
                seed_pairs=pairs,
                env_steps=0,
                **expected,
            )
    finally:
        if target is not None and original is not None:
            target.write_bytes(original)


@pytest.mark.parametrize("change", ["pending", "coverage", "duplicate", "timing"])
def test_saved_random_host_state_is_strict(
    random_runs: dict[str, Any], change: str
) -> None:
    root = random_runs["first"].run_dir
    pointer = json.loads((root / "latest_checkpoint.json").read_text())
    description = checkpoints.read_checkpoint_details(root / pointer["relative_path"])
    host = description["metadata"]["host_state"]
    if change == "pending":
        host["pending"]["random"] = False
    elif change == "coverage":
        host["random_results"] = []
    elif change == "duplicate":
        host["random_results"].append(host["random_results"][0])
    else:
        host["random_results"][0]["elapsed_seconds"] = host["elapsed_seconds"] + 100
    with pytest.raises(ValueError, match="Random"):
        validate_host_state(root, description, panel=None)
