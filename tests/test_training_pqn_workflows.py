"""Check recurrent PQN-VDN through complete native-horizon public workflows.

Every case uses four games, blocks of 4 rounds, a memory window of 4 (so W = 8
initial rounds, collected as two chunks of 4) and one epoch of two
minibatches. The base run has 80 real transitions: two initial chunks, then
three learning blocks of two optimizer steps each. Contracts checked here:
the plain run with an Alpha panel on the five validation maps validates at
initialization, after initial collection and after learning, captures every
declared boundary, never selects an actor saved before learning, confirms,
selects, exports, loads and evaluates the greedy actor with its real M8
registration and its saved parameters and statistics, plays a two-entrant
tournament whose match and ranking tables are saved, and writes PQN-VDN
reports (columns, figure and summary line); a recorded run interrupted after
a save resumes to the same learner numbers, used counts and exposure as an
uninterrupted run without a panel, keeping its earlier log rows and games;
Random diagnostics run fresh, are reused by a later run's initialization,
pass recovery verification, and refuse reuse when only the initial
statistics differ, while re-signed checkpoints with a wrong source-exposure
length or a wrong optimizer count are refused before recovery changes any
file; saves inside initial collection (16 and 32 transitions) and before a
one-round final block resume exactly, and progress shows the initial
collection label; and curriculum, reward shaping and both together reach the
collection and complete, with each block's logged task and shaping means
equal to its rollout's. Saved results carry method and optimizer counts.
These CPU cases establish workflow correctness, not learning or GPU speed.
"""

# pyright: reportPrivateUsage=false
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines.pqn import PQNConfig

type Tree = Any
_PQN = PQNConfig(rollout_length=4, memory_window=4, epochs=1, num_minibatches=2)


def _fixed_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.training import checkpoints

    identity = checkpoints.runtime_identity(method="pqn_vdn")
    monkeypatch.setattr(
        checkpoints, "runtime_identity", lambda method="mappo": identity
    )
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)


def _config(**changes: Any) -> Any:  # noqa: ANN401
    from marl_battlegrounds import training

    settings: dict[str, Any] = {
        "method": "pqn_vdn",
        "seed": 19049101,
        "num_envs": 4,
        "total_env_steps": 80,
        "pqn": _PQN,
        "validation_fractions": (0.4, 0.6, 0.8, 1.0),
        "routine_seed_pairs": 1,
        "confirmation_seed_pairs": 1,
        "purpose": "development",
        "metrics": "priority",
        "recording": False,
        "verbose": False,
        **changes,
    }
    return training.TrainConfig(**settings)


def _capture_saves(
    monkeypatch: pytest.MonkeyPatch,
    stop_root: Path | None,
    stop_steps: int,
    keep: set[int] | None = None,
    extract: Callable[[Tree], Tree] | None = None,
) -> dict[Path, dict[int, Tree]]:
    from marl_battlegrounds.training import runner

    saved: dict[Path, dict[int, Tree]] = {}
    # Always wrap the real save: monkeypatch keeps every earlier wrapper (and
    # what it captured) alive until the test ends, so wrappers must not chain.
    original = getattr(runner._Run.save, "__wrapped__", runner._Run.save)

    def save(execution: runner._Run) -> Path:
        checkpoint = original(execution)
        steps = execution.host["env_steps"]
        # Each learner holds a 369 MB opponent bank; keep only what is needed.
        state = execution.state if keep is None or steps in keep else None
        saved.setdefault(execution.root, {})[steps] = (
            state if extract is None or state is None else extract(state)
        )
        if execution.root == stop_root and steps == stop_steps:
            raise KeyboardInterrupt("Declared stop after a saved block")
        return checkpoint

    save.__wrapped__ = original  # pyright: ignore[reportFunctionMemberAccess]
    monkeypatch.setattr(runner._Run, "save", save)
    return saved


def _leaf_digests(tree: Tree) -> dict[str, str]:
    leaves = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(tree)[0],
    )
    digests: dict[str, str] = {}
    for path, leaf in leaves:
        if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            leaf = jax.random.key_data(leaf)
        value = np.asarray(leaf)
        hasher = hashlib.sha256(f"{value.dtype}{value.shape}".encode())
        hasher.update(value.tobytes())
        digests[jax.tree_util.keystr(path)] = hasher.hexdigest()
    return digests


def _resign_host_state(
    checkpoint: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    from marl_battlegrounds.training import checkpoints

    description = checkpoint / "checkpoint_details.json"
    details = json.loads(description.read_text())
    change(details["metadata"]["host_state"])
    unsigned = {key: value for key, value in details.items() if key != "checkpoint_id"}
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(unsigned)
    ).hexdigest()
    description.write_text(json.dumps(details))


def _run_files(root: Path) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checkpoint_details.json"
    }


def _latest(root: Path) -> Path:
    pointer = json.loads((root / "latest_checkpoint.json").read_text())
    return root / pointer["relative_path"]


def _rows(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "training_updates.jsonl").open()]


def test_plain_complete_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
        tree_digest,
    )
    from marl_battlegrounds.training import checkpoints

    _fixed_identity(monkeypatch)

    def network(state: Tree) -> Tree:
        return state.carry.history.current_variables.network

    saved = _capture_saves(monkeypatch, None, -1, extract=network)
    result = training.train(
        _config(
            validation_opponents=("tdm-alpha",), checkpoint_env_steps=(16, 32, 48, 64)
        ),
        output_dir=tmp_path / "run",
    )
    root = result.run_dir
    assert result.completed_env_steps == 80 and result.completed_updates == 6
    assert sorted(saved[root]) == [0, 16, 32, 48, 64, 80]
    rows = _rows(root)
    assert [row["env_steps"] for row in rows] == [16, 32, 48, 64, 80]
    assert [row["completed_blocks"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["learning_blocks"] for row in rows] == [0, 0, 1, 2, 3]
    assert [row["initial_rounds_completed"] for row in rows] == [4, 8, 8, 8, 8]
    assert [row["loss"] is None for row in rows] == [True, True, False, False, False]
    assert [row["learning_rate"] is None for row in rows] == [True, True] + [False] * 3
    assert rows[-1]["used_td_pairs"] == 1 * 4 * ((20 - 8) + 3 * 3)
    assert rows[-1]["used_prefix_td_pairs"] == 3 * 1 * 4 * 4
    assert "policy_loss" not in rows[-1] and "replay_rows_per_lane" not in rows[-1]
    validations = json.loads((root / "validation_results.json").read_text())
    routine = [row for row in validations if row["purpose"] == "routine"]
    assert [row["env_steps"] for row in routine] == [0, 32, 48, 64, 80]
    assert [row["optimizer_steps"] for row in routine] == [0, 0, 2, 4, 6]
    assert all(row["method"] == "pqn_vdn" for row in validations)
    assert all(row["complete"] and row["games"] == 10 for row in validations)
    confirmed = {
        row["env_steps"] for row in validations if row["purpose"] == "confirmation"
    }
    assert confirmed <= {48, 64, 80} and 32 not in confirmed
    actors = sorted(
        checkpoints.artifact_identity(path)["env_steps"]
        for path in (root / "actors").iterdir()
    )
    assert actors == [0, 16, 32, 48, 64, 80]
    assert result.selected_actor is not None
    selected = checkpoints.artifact_identity(result.selected_actor)
    assert selected["method"] == "pqn_vdn" and selected["optimizer_steps"] in (2, 4, 6)
    loaded = training.load_system(result.selected_actor)
    assert float(loaded.variables.epsilon) == 0.0
    ancestor = checkpoints.read_checkpoint_description(
        root / "checkpoints" / selected["metadata"]["checkpoint_id"]
    )
    item = checkpoints._pqn_actor_item(loaded.variables.network)
    assert tree_digest(item) == ancestor["actor_digest"]
    equal(loaded.variables.network, saved[root][selected["env_steps"]])
    evaluation = marl_bgs.evaluate(
        loaded,
        "tdm-alpha",
        num_episodes=2,
        num_envs=4,
        maps=[42],
        seed=19049103,
        metrics="priority",
        phase="validation",
        output_dir=tmp_path / "selected-evaluation",
    )
    assert len(evaluation.completed_episode_ids) == 2
    assert evaluation.paths is not None
    evaluated = json.loads(evaluation.paths["run_details"].read_text())
    system_id, registration = normalize_system_registration(
        loaded, phase="validation", frozen=True
    )
    assert evaluated["systems"][system_id] == registration
    tournament = marl_bgs.run_tournament(
        (loaded, "tdm-alpha"),
        maps=[42],
        episodes_per_pair=2,
        num_envs=4,
        seed=19049104,
        output_dir=tmp_path / "tournament",
    )
    assert tournament.status == "complete"
    assert len(tournament.table("tournament_rankings")["rank"]) == 2
    assert tournament.matches and tournament.paths is not None
    assert all(Path(path).is_file() for path in tournament.paths.values())
    exposure = json.loads((root / "exposure.json").read_text())
    assert exposure["used_exposure"]["used_td_pairs"] == 84
    assert sum(exposure["used_exposure"]["by_stage"]) == 84
    report = training.analyze([root], output_dir=tmp_path / "review")
    assert report["complete"] and report["runs"][0]["method"] == "pqn_vdn"
    assert Path(report["artifacts"]["pqn_png"]).stat().st_size > 0
    header = Path(report["artifacts"]["curve"]).read_text().splitlines()[0].split(",")
    assert {
        "loss",
        "used_td_pairs",
        "used_prefix_td_pairs",
        "epsilon",
        "learning_rate",
    } <= set(header)
    assert "used_exposure" not in header
    summary = Path(report["artifacts"]["summary"]).read_text()
    assert "Optimizer steps: 6." in summary
    assert "Initial random transitions generated but never learned: 16." in summary


def test_recorded_interruption_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import checkpoints

    _fixed_identity(monkeypatch)
    interrupted = tmp_path / "interrupted"
    saved = _capture_saves(
        monkeypatch, interrupted, 48, keep={48, 80}, extract=_leaf_digests
    )
    config = _config(
        recording=True,
        validation_opponents=("tdm-alpha",),
        checkpoint_env_steps=(48,),
    )
    # The reference differs only by its absent panel; validation never changes
    # learner numbers, so its full evaluation work is not repeated.
    reference = training.train(
        replace(config, validation_opponents=None), output_dir=tmp_path / "reference"
    )
    with pytest.raises(KeyboardInterrupt, match="saved block"):
        training.train(config, output_dir=interrupted)
    first_rows = (interrupted / "training_updates.jsonl").read_bytes()
    games = {
        path: path.read_bytes() for path in (interrupted / "validation").rglob("*.csv")
    }
    assert games
    checkpoint = _latest(interrupted)
    assert checkpoints.read_checkpoint_details(checkpoint)["counters"] == {
        "updates": 2,
        "env_steps": 48,
        "completed_blocks": 3,
        "learning_blocks": 1,
    }
    assert saved[reference.run_dir][48] == saved[interrupted][48]
    resumed = training.train(resume_from=checkpoint)
    assert saved[reference.run_dir][80] == saved[interrupted][80]
    assert (interrupted / "training_updates.jsonl").read_bytes().startswith(first_rows)
    assert all(path.read_bytes() == raw for path, raw in games.items())
    assert resumed.completed_updates == reference.completed_updates == 6
    exposures = [
        json.loads((root / "exposure.json").read_text())["used_exposure"]
        for root in (reference.run_dir, interrupted)
    ]
    assert exposures[0] == exposures[1]
    last = [_rows(root)[-1] for root in (reference.run_dir, interrupted)]
    for key in (
        "env_steps",
        "completed_blocks",
        "learning_blocks",
        "completed_updates",
        "used_sequences",
        "used_td_pairs",
        "used_agent_utilities",
        "used_prefix_td_pairs",
        "used_exposure",
    ):
        assert last[0][key] == last[1][key]
    assert (
        checkpoints.artifact_identity(resumed.final_actor)["actor_digest"]
        == checkpoints.artifact_identity(reference.final_actor)["actor_digest"]
    )
    assert resumed.selected_actor is not None
    report = training.analyze([interrupted], output_dir=tmp_path / "review")
    assert report["complete"]


def test_random_diagnostics_reuse_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines import pqn
    from marl_battlegrounds.training import pqn_learner

    _fixed_identity(monkeypatch)
    first = training.train(
        _config(random_diagnostic_seed_pairs=1), output_dir=tmp_path / "first"
    )
    records = json.loads((first.run_dir / "random_diagnostics.json").read_text())
    assert [row["env_steps"] for row in records] == [0, 80]
    assert all(row["method"] == "pqn_vdn" for row in records)
    assert [row["optimizer_steps"] for row in records] == [0, 6]
    initial = tmp_path / "initial-random.json"
    initial.write_text(json.dumps(records[0]))
    config = _config(
        random_diagnostic_seed_pairs=1,
        random_initialization_result=str(initial),
        checkpoint_env_steps=(48,),
    )
    # Equal weights with changed initial statistics are a different actor.
    original_initialize = pqn_learner.initialize_pqn

    def shifted(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        state = original_initialize(*args, **kwargs)
        stats = state.network.batch_stats
        moved = {
            **stats,
            "BatchNorm_0": {
                **stats["BatchNorm_0"],
                "var": stats["BatchNorm_0"]["var"] + 1,
            },
        }
        return state._replace(
            network=pqn.PQNInferenceVariables(state.network.params, moved)
        )

    with monkeypatch.context() as patch:
        patch.setattr(pqn_learner, "initialize_pqn", shifted)
        with pytest.raises(ValueError, match="differs from the expected actor"):
            training.train(config, output_dir=tmp_path / "changed-statistics")
    assert not (tmp_path / "changed-statistics").exists()
    second_root = tmp_path / "second"
    _capture_saves(monkeypatch, second_root, 48, keep=set())
    with pytest.raises(KeyboardInterrupt):
        training.train(config, output_dir=second_root)
    checkpoint = _latest(second_root)
    description = checkpoint / "checkpoint_details.json"
    original = description.read_bytes()

    def longer_source_exposure(host: dict[str, Any]) -> None:
        host["used_exposure"]["by_source"].append(0)

    def wrong_optimizer_count(host: dict[str, Any]) -> None:
        host["random_results"][-1]["optimizer_steps"] += 1

    def wrong_method(host: dict[str, Any]) -> None:
        host["random_results"][-1]["method"] = "qmix"

    for change, match in (
        (longer_source_exposure, "source bank"),
        (wrong_optimizer_count, "optimizer count"),
        (wrong_method, "names another method"),
    ):
        _resign_host_state(checkpoint, change)
        before = _run_files(second_root)
        with pytest.raises(ValueError, match=match):
            training.train(resume_from=checkpoint)
        assert _run_files(second_root) == before
        assert not (second_root / "checkpoint_recovery.json").exists()
        description.write_bytes(original)
    resumed = training.train(resume_from=checkpoint)
    reused = json.loads((second_root / "random_diagnostics.json").read_text())
    assert [row["env_steps"] for row in reused] == [0, 48, 80]
    assert reused[0]["reused_initialization"] is True
    assert resumed.completed_updates == 6


def test_initial_collection_and_partial_final_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import checkpoints

    _fixed_identity(monkeypatch)
    # 84 transitions: chunks at 16 and 32, blocks at 48, 64 and 80, then one
    # final round.
    config = _config(total_env_steps=84, checkpoint_env_steps=(16, 32, 80))
    first = _capture_saves(monkeypatch, None, -1, keep={84}, extract=_leaf_digests)
    reference = training.train(
        replace(config, verbose=True), output_dir=tmp_path / "reference"
    )
    output = capsys.readouterr().out
    assert "Initial Random Collection" in output
    assert "Optimizer Steps:" in output and "Learning Blocks:" in output
    assert [row["env_steps"] for row in _rows(reference.run_dir)] == [
        16,
        32,
        48,
        64,
        80,
        84,
    ]
    for stop, counters in (
        (16, (0, 1, 0)),
        (32, (0, 2, 0)),
        (80, (6, 5, 3)),
    ):
        root = tmp_path / f"stop-{stop}"
        saved = _capture_saves(
            monkeypatch, root, stop, keep={84}, extract=_leaf_digests
        )
        with pytest.raises(KeyboardInterrupt):
            training.train(config, output_dir=root)
        checkpoint = _latest(root)
        found = checkpoints.read_checkpoint_details(checkpoint)["counters"]
        assert (
            found["updates"],
            found["completed_blocks"],
            found["learning_blocks"],
        ) == counters
        resumed = training.train(resume_from=checkpoint)
        assert resumed.completed_updates == reference.completed_updates == 8
        assert resumed.completed_env_steps == 84
        assert saved[root][84] == first[reference.run_dir][84]
        assert (
            _rows(root)[-1]["used_td_pairs"]
            == _rows(reference.run_dir)[-1]["used_td_pairs"]
        )
        assert (
            checkpoints.artifact_identity(resumed.final_actor)["actor_digest"]
            == checkpoints.artifact_identity(reference.final_actor)["actor_digest"]
        )


def _treatment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, bool]
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import collection as collection_module
    from marl_battlegrounds.training.runner import config_from_dict, config_to_dict

    _fixed_identity(monkeypatch)
    original_collect = collection_module.collect_training_rollout
    means: list[tuple[float, float]] = []

    def collect(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        collection = args[0]
        assert collection.shaping is bool(changes.get("shaping"))
        assert (collection.schedule.arrays.stage_count > 1) == bool(
            changes.get("curriculum")
        )
        after, rollout = original_collect(*args, **kwargs)
        rows = rollout.transitions
        valid, active = np.asarray(rows.valid), np.asarray(rows.active)
        taken = valid[..., None] & active
        task = np.where(taken, np.asarray(rows.task_rewards), 0).sum()
        shaping = np.where(valid, np.asarray(rows.shaping_reward), 0).sum()
        means.append(
            (float(task) / int(taken.sum()), float(shaping) / int(valid.sum()))
        )
        return after, rollout

    monkeypatch.setattr(collection_module, "collect_training_rollout", collect)
    config = _config(seed=19049102, total_env_steps=256, **changes)
    assert config_from_dict(config_to_dict(config)) == config
    result = training.train(config, output_dir=tmp_path / "run")
    # 64 rounds: 8 initial rounds, then 14 learning blocks of two steps.
    assert result.completed_env_steps == 256
    assert result.completed_updates == 14 * 2
    rows = _rows(result.run_dir)
    # Tiny random-play runs may never score, so shaping can honestly be zero;
    # each logged mean must equal the rollout it came from instead.
    assert len(rows) == len(means) == 16
    for row, (task, shaping) in zip(rows, means, strict=True):
        assert row["task_reward_mean"] == pytest.approx(task, abs=1e-7)
        assert row["shaping_mean"] == pytest.approx(shaping, abs=1e-7)
        if not changes.get("shaping"):
            assert row["shaping_mean"] == 0
    exposure = json.loads((result.run_dir / "exposure.json").read_text())
    assert exposure["used_exposure"]["used_td_pairs"] == 1 * 4 * ((64 - 8) + 14 * 3)


def test_curriculum_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _treatment(tmp_path, monkeypatch, {"curriculum": True})


def test_shaping_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _treatment(tmp_path, monkeypatch, {"shaping": True})


def test_curriculum_shaping_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _treatment(tmp_path, monkeypatch, {"curriculum": True, "shaping": True})
