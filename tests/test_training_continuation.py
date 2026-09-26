"""Check explicit child training, parent immutability and exact child resume.

Real short learners exercise the public route. Each child and grandchild matches
direct collection and learning from the same saved parent, including every
numerical state value and key. Declarations and failures are checked before child
writes. These CPU checks make no learned-behavior or speed claim.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pytest
from tests.training_continuation_helpers import capture_runs, compare_direct_child

from marl_battlegrounds.baselines.methods import TrainingMethod

# The test checks the private fork boundary used by the public function.
# pyright: reportPrivateUsage=false


def _identities(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("method", ["mappo", "ippo", "ff_mappo", "ff_ippo"])
def test_full_parent_child_resume_and_second_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: TrainingMethod
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, extend_training, train
    from marl_battlegrounds.training.runner import TrainConfig

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    starts, ends = capture_runs(monkeypatch)
    settings = PPOConfig(rollout_length=2, epochs=1)
    parent = train(
        TrainConfig(
            method=method,
            num_envs=4,
            total_env_steps=220 if method == "ff_mappo" else 12,
            curriculum=method == "ff_mappo",
            seed=718,
            ppo=settings,
            checkpoint_interval_updates=1,
            metrics="none",
            recording=method == "mappo",
            verbose=False,
        ),
        output_dir=tmp_path / "parent",
    )
    before = _identities(parent.run_dir)
    latest = json.loads((parent.run_dir / "latest_checkpoint.json").read_text())
    checkpoint = parent.run_dir / latest["relative_path"]
    import jax
    import numpy as np

    from marl_battlegrounds.training import runner

    prepare = runner._prepare_extension

    def checked_prepare(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        before_state = args[2].state
        result = prepare(*args, **kwargs)
        after_state = result[2]
        before_paths = cast(
            list[tuple[tuple[object, ...], Any]],
            jax.tree_util.tree_flatten_with_path(before_state)[0],
        )
        after_paths = cast(
            list[tuple[tuple[object, ...], Any]],
            jax.tree_util.tree_flatten_with_path(after_state)[0],
        )
        before_leaves = {str(path): value for path, value in before_paths}
        exempt = (
            ".schedule.",
            ".stage_rounds",
            ".stage_counts",
            ".stage_ordinal",
            ".stage_round_budget",
            ".completed_stage_counts",
            ".stage_complete",
        )
        for path, value in after_paths:
            key = str(path)
            # Compare names below rather than serializing a private tree format.
            names = "." + ".".join(
                str(getattr(item, "name", getattr(item, "idx", ""))) for item in path
            )
            if not any(part in names for part in exempt):
                previous = before_leaves[key]
                if jax.numpy.issubdtype(value.dtype, jax.dtypes.prng_key):
                    value, previous = (
                        jax.random.key_data(value),
                        jax.random.key_data(previous),
                    )
                np.testing.assert_array_equal(value, previous, err_msg=names)
        return result

    monkeypatch.setattr(runner, "_prepare_extension", checked_prepare)
    child = extend_training(
        checkpoint, additional_env_steps=4, output_dir=tmp_path / "child"
    )
    assert child.completed_env_steps == parent.completed_env_steps + 4
    assert child.completed_updates == parent.completed_updates + 1
    assert _identities(parent.run_dir) == before
    compare_direct_child(parent.run_dir, child.run_dir, starts, ends, settings, method)
    child_details = json.loads((child.run_dir / "run_details.json").read_text())
    parent_details = json.loads((parent.run_dir / "run_details.json").read_text())
    assert child_details["training_lineage"]["root_run_id"] == parent_details["run_id"]
    latest_child = json.loads((child.run_dir / "latest_checkpoint.json").read_text())
    child_checkpoint = child.run_dir / latest_child["relative_path"]
    resumed = train(resume_from=child_checkpoint)
    assert resumed.completed_env_steps == child.completed_env_steps
    assert resumed.completed_updates == child.completed_updates
    grandchild = extend_training(
        child_checkpoint, additional_env_steps=4, output_dir=tmp_path / "grandchild"
    )
    assert grandchild.completed_env_steps == child.completed_env_steps + 4
    assert grandchild.completed_updates == child.completed_updates + 1
    compare_direct_child(
        child.run_dir, grandchild.run_dir, starts, ends, settings, method
    )
    assert _identities(parent.run_dir) == before
    if method == "mappo":
        saved = checkpoints.read_checkpoint_details(checkpoint)
        record = saved["metadata"]["recording"]
        token = saved["recording_token"]
        bundle = (
            parent.run_dir
            / record["relative_path"]
            / "recording_checkpoints"
            / token["checkpoint_id"]
            / "checkpoint.json"
        )
        bundle.write_bytes(bundle.read_bytes() + b" ")
        corrupt_boundary = _identities(parent.run_dir)
        child = tmp_path / "rejected-recording"
        with pytest.raises(ValueError, match="recording checkpoint metadata digest"):
            extend_training(checkpoint, additional_env_steps=4, output_dir=child)
        assert not child.exists()
        assert _identities(parent.run_dir) == corrupt_boundary


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_extension_rejects_bad_budget_without_files(
    tmp_path: Path, budget: object
) -> None:
    from marl_battlegrounds.training import extend_training

    with pytest.raises(ValueError, match="additional_env_steps"):
        extend_training(
            tmp_path / "missing",
            additional_env_steps=cast(int, budget),
            output_dir=tmp_path / "child",
        )
    assert not (tmp_path / "child").exists()


@pytest.mark.parametrize(
    "field", [None, "seed", "red_zone_depth", "recording", "total_env_steps"]
)
def test_saved_child_keeps_parent_scientific_settings(
    monkeypatch: pytest.MonkeyPatch, field: str | None
) -> None:
    from copy import deepcopy

    from marl_battlegrounds.training import checkpoints, runner

    parent_config = runner.config_to_dict(
        runner.TrainConfig(num_envs=4, total_env_steps=8)
    )
    child_config = {**parent_config, "total_env_steps": 12}
    parent = {
        "kind": "learner",
        "checkpoint_id": "parent-id",
        "counters": {"env_steps": 8},
        "metadata": {
            "config": parent_config,
            "run_id": "parent-run",
            "source": {"sha": "parent"},
        },
    }
    context = {
        "schema_version": 1,
        "parent_checkpoint": "parent/checkpoints/parent-id",
        "parent_checkpoint_id": "parent-id",
        "parent_run_id": "parent-run",
        "parent_source": {"sha": "parent"},
        "child_source": {"sha": "child"},
        "start_env_steps": 8,
        "additional_env_steps": 4,
        "resulting_total_env_steps": 12,
        "root_run_id": "parent-run",
        "segment": {"root_schedule": {"total_env_steps": 8}},
    }
    details = {
        "metadata": {
            "config": child_config,
            "source": {"sha": "child"},
            "continuation": context,
            "validation_declaration": {"panel_path": None},
        }
    }

    def read_parent(_: object) -> dict[str, Any]:
        return deepcopy(parent)

    def saved_config(value: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(value["metadata"]["config"])

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", read_parent)
    monkeypatch.setattr(checkpoints, "saved_training_config", saved_config)
    if field is None:
        config = runner._saved_config(details)
        assert config.total_env_steps == 12
        assert config.seed == parent_config["seed"]
    else:
        child_config[field] = (
            not child_config[field] if field == "recording" else child_config[field] + 1
        )
        with pytest.raises(ValueError, match="Continuation"):
            runner._saved_config(details)


def test_continuation_source_refuses_unknown_transition() -> None:
    from marl_battlegrounds.training.checkpoints import (
        continuation_source_compatibility,
    )

    source = {"schema_version": 1, "commit": "same"}
    assert (
        continuation_source_compatibility(source, source)["qualification"]
        == "Exact source"
    )
    with pytest.raises(ValueError, match="qualified continuation"):
        continuation_source_compatibility(source, {**source, "commit": "different"})


def test_completed_game_totals_survive_child_and_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from copy import deepcopy

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, extend_training, runner, train

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    names = (
        "stage_completed",
        "stage_score_sums",
        "stage_length_sum",
        "stage_k20_count",
    )
    starts: dict[Path, dict[str, Any]] = {}
    ends: dict[Path, dict[str, Any]] = {}
    execute = runner._Run.execute

    def observe(self: Any) -> Any:  # noqa: ANN401
        starts[self.root] = deepcopy(self.host)
        result = execute(self)
        ends[self.root] = deepcopy(self.host)
        return result

    monkeypatch.setattr(runner._Run, "execute", observe)
    parent = train(
        runner.TrainConfig(
            method="ff_ippo",
            num_envs=4,
            total_env_steps=1200,
            seed=9035,
            ppo=PPOConfig(rollout_length=2, epochs=1, minibatches=1),
            checkpoint_interval_updates=1000,
            metrics="none",
            verbose=False,
        ),
        output_dir=tmp_path / "parent",
    )
    assert sum(map(sum, ends[parent.run_dir]["stage_completed"])) >= 4
    parent_bytes = _identities(parent.run_dir)
    for name in ("child", "grandchild"):
        previous = parent
        pointer = json.loads((previous.run_dir / "latest_checkpoint.json").read_text())
        parent = extend_training(
            previous.run_dir / pointer["relative_path"],
            additional_env_steps=4,
            output_dir=tmp_path / name,
        )
        exposure = json.loads((parent.run_dir / "exposure.json").read_text())
        for field in names:
            assert starts[parent.run_dir][field] == ends[previous.run_dir][field]
            assert exposure[field] == ends[parent.run_dir][field]
        assert sum(map(sum, exposure["stage_completed"])) >= sum(
            map(sum, ends[previous.run_dir]["stage_completed"])
        )
    assert _identities(tmp_path / "parent") == parent_bytes


@pytest.mark.parametrize("empty_folder", [False, True])
def test_child_resume_names_its_missing_parent_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_folder: bool
) -> None:
    from marl_battlegrounds.training import checkpoints, runner

    parent = tmp_path / "original-parent/checkpoints/selected"
    if empty_folder:
        parent.mkdir(parents=True)

    def saved_training_config(_: dict[str, Any]) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(checkpoints, "saved_training_config", saved_training_config)
    details = {
        "metadata": {
            "continuation": {"schema_version": 1, "parent_checkpoint": str(parent)}
        }
    }
    with pytest.raises(ValueError, match="continuation parent") as caught:
        runner._saved_config(details)
    assert str(parent) in str(caught.value)
    assert "description at that path" in str(caught.value)
    assert not parent.exists() or not list(parent.iterdir())
