"""Check explicit child training, parent immutability and exact child resume.

Real short learners exercise the public route. Each child and grandchild matches
direct collection and learning from the same saved parent, including every
numerical state value and key. Declarations and failures are checked before child
writes. Seed branches reach real action and learner key consumers, and an
interrupted child resumes exactly without applying its branch twice. These CPU
checks make no learned-behavior or speed claim.
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
            keep_past=0,
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
        runner.TrainConfig(keep_past=0, num_envs=4, total_env_steps=8)
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
        "changes": {},
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


def test_continuation_source_records_changed_identity_for_contract_checks() -> None:
    from marl_battlegrounds.training.checkpoints import (
        continuation_source_compatibility,
    )

    source = {"schema_version": 1, "commit": "same"}
    assert (
        continuation_source_compatibility(source, source)["qualification"]
        == "Exact source"
    )
    changed = {**source, "commit": "different"}
    record = continuation_source_compatibility(source, changed)
    assert record["parent_source"] == source
    assert record["child_source"] == changed
    assert record["qualification"] == "Checkpoint contract checks"
    with pytest.raises(ValueError, match="complete source identities"):
        continuation_source_compatibility({}, changed)


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
            keep_past=0,
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


@pytest.mark.parametrize("method", ["ff_ippo", "qmix", "pqn_vdn"])
def test_seed_branch_changes_real_streams_and_resume_does_not_branch_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: TrainingMethod
) -> None:
    from copy import deepcopy
    from functools import partial

    import jax
    import numpy as np
    from tests.training_continuation_helpers import latest
    from tests.training_learner_helpers import equal

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.baselines.pqn import PQNConfig
    from marl_battlegrounds.baselines.qmix import QMIXConfig
    from marl_battlegrounds.training import (
        checkpoints,
        extend_training,
        learner,
        pqn_learner,
        qmix_learner,
        runner,
        train,
    )
    from marl_battlegrounds.training import collection as collection_module
    from marl_battlegrounds.training.distributions import training_keys
    from marl_battlegrounds.training.runner import TrainConfig

    identity = checkpoints.runtime_identity(method=method)

    def current(**_kwargs: object) -> dict[str, object]:
        return identity

    monkeypatch.setattr(checkpoints, "runtime_identity", current)
    starts, ends = capture_runs(monkeypatch)
    config = TrainConfig(
        keep_past=0,
        method=method,
        num_envs=4,
        total_env_steps=24 if method == "pqn_vdn" else 16,
        ppo=PPOConfig(rollout_length=2, epochs=1)
        if method == "ff_ippo"
        else PPOConfig(),
        qmix=QMIXConfig(
            rollout_length=2,
            buffer_size=4,
            min_buffer_size=2,
            sample_sequence_length=2,
            sample_batch_size=2,
            epochs=1,
            eps_min=1.0,
        )
        if method == "qmix"
        else None,
        pqn=PQNConfig(
            rollout_length=2,
            memory_window=1,
            epochs=1,
            num_minibatches=1,
            lr_linear_decay=False,
            eps_start=1.0,
            eps_finish=1.0,
        )
        if method == "pqn_vdn"
        else None,
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
    )
    parent = train(config, output_dir=tmp_path / "parent")
    before = _identities(parent.run_dir)
    parent_source = deepcopy(cast(dict[str, object], identity["source"]))
    identity = {**identity, "source": {**parent_source, "package_version": "99.0"}}
    # Observe the real consumer called inside the compiled learner update.
    module, name, key_index = (
        (qmix_learner, "_sample_rows", 1)
        if method == "qmix"
        else (pqn_learner, "epoch_permutations", 0)
        if method == "pqn_vdn"
        else (learner, "update_ppo", 2)
    )
    consumer = getattr(module, name)
    consumed: list[np.ndarray[Any, np.dtype[np.uint32]]] = []

    def remember(key: jax.Array) -> None:
        consumed.append(np.asarray(key).copy())

    def observe(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        jax.debug.callback(
            remember,
            jax.random.key_data(args[key_index]),
            ordered=True,
        )
        return consumer(*args, **kwargs)

    monkeypatch.setattr(module, name, observe)
    changes = {"seed": 91}
    child = extend_training(
        parent.final_checkpoint,
        additional_env_steps=16,
        output_dir=tmp_path / "first",
        changes=changes,
    )
    assert consumed
    old = ends[parent.run_dir]
    collection, branched = starts[child.run_dir]
    root_name = "sampling_root" if method == "qmix" else "shuffle_root"
    old_key, new_key = getattr(old, root_name), getattr(branched, root_name)
    if method != "pqn_vdn":
        index = int(old.completed_updates) + (method == "ff_ippo")
        old_key, new_key = (
            jax.random.fold_in(key, index) for key in (old_key, new_key)
        )
    np.testing.assert_array_equal(consumed[0], jax.random.key_data(new_key))
    assert not np.array_equal(consumed[0], jax.random.key_data(old_key))
    apply = jax.jit(
        partial(
            collection_module._apply,
            actor=collection.actor,
            opponent=collection.opponent,
        )
    )
    old_actions, new_actions = (
        cast(Any, apply(carry))[0] for carry in (old.carry, branched.carry)
    )
    assert any(
        not np.array_equal(a, b) for a, b in zip(old_actions, new_actions, strict=True)
    )
    for stream in ("reset", "opponent", "initialization"):
        original = training_keys(
            old.carry.root_key, old.carry.state.reset_generation, stream=stream
        )
        changed = training_keys(
            branched.carry.root_key, old.carry.state.reset_generation, stream=stream
        )
        assert not np.array_equal(
            jax.random.key_data(original), jax.random.key_data(changed)
        )
    assert not np.array_equal(
        jax.random.key_data(old.carry.memory.init_key),
        jax.random.key_data(branched.carry.memory.init_key),
    )
    details = checkpoints.read_checkpoint_description(child.final_checkpoint)
    assert details["metadata"]["continuation"]["parent_source"] == parent_source
    assert details["metadata"]["source"] == identity["source"]
    expected = ends[child.run_dir]
    save = runner._Run.save
    interrupted = tmp_path / "interrupted"

    def interrupt_after_save(execution: runner._Run) -> Path:
        result = save(execution)
        if (
            execution.root == interrupted
            and parent.completed_env_steps
            < execution.host["env_steps"]
            < child.completed_env_steps
        ):
            raise KeyboardInterrupt("Stop after a durable child boundary")
        return result

    monkeypatch.setattr(runner._Run, "save", interrupt_after_save)
    with pytest.raises(KeyboardInterrupt, match="durable child boundary"):
        extend_training(
            parent.final_checkpoint,
            additional_env_steps=16,
            output_dir=interrupted,
            changes=changes,
        )
    stopped = ends[interrupted]
    assert (
        parent.completed_env_steps
        < int(stopped.carry.progress.rounds) * 4
        < child.completed_env_steps
    )
    monkeypatch.setattr(runner._Run, "save", save)
    resumed = train(resume_from=latest(interrupted))
    equal(starts[interrupted][1], stopped)
    equal(ends[resumed.run_dir], expected)
    assert _identities(parent.run_dir) == before


@pytest.mark.parametrize("method", ["ff_ippo", "qmix", "pqn_vdn"])
def test_changed_reward_child_refill_resume_and_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: TrainingMethod
) -> None:
    import numpy as np
    from tests.training_continuation_helpers import fixed_source
    from tests.training_learner_helpers import equal

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.baselines.pqn import PQNConfig
    from marl_battlegrounds.baselines.qmix import QMIXConfig
    from marl_battlegrounds.training import checkpoints, extend_training, train
    from marl_battlegrounds.training.runner import TrainConfig

    fixed_source(monkeypatch, method)
    starts, ends = capture_runs(monkeypatch)
    config = TrainConfig(
        method=method,
        num_envs=4,
        total_env_steps=24,
        keep_past=0,
        shaping=False,
        reward="tests.training_continuation_helpers:reward_one",
        ppo=PPOConfig(rollout_length=2, epochs=1)
        if method == "ff_ippo"
        else PPOConfig(),
        qmix=QMIXConfig(
            rollout_length=2,
            buffer_size=6,
            min_buffer_size=3,
            sample_sequence_length=2,
            sample_batch_size=2,
            epochs=1,
        )
        if method == "qmix"
        else None,
        pqn=PQNConfig(
            rollout_length=2,
            memory_window=2,
            epochs=1,
            num_minibatches=1,
            lr_linear_decay=False,
        )
        if method == "pqn_vdn"
        else None,
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
    )
    parent = train(config, output_dir=tmp_path / "parent")
    previous = ends[parent.run_dir]
    child = extend_training(
        parent.final_checkpoint,
        additional_env_steps=4,
        output_dir=tmp_path / "child",
        changes={"reward": "tests.training_continuation_helpers:reward_seven"},
    )
    collection, boundary = starts[child.run_dir]
    equal(boundary.carry.state, previous.carry.state)
    equal(boundary.carry.memory, previous.carry.memory)
    equal(
        boundary.carry.history.current_variables,
        previous.carry.history.current_variables,
    )
    equal(boundary.carry.root_key, previous.carry.root_key)
    assert collection.reward_identity["reference"].endswith(":reward_seven")
    if method == "ff_ippo":
        equal(boundary.actor_opt_state, previous.actor_opt_state)
        equal(boundary.critic_opt_state, previous.critic_opt_state)
        equal(boundary.value_norm, previous.value_norm)
        assert child.completed_updates == parent.completed_updates + 1
    else:
        equal(boundary.opt_state, previous.opt_state)
        assert child.completed_updates == parent.completed_updates
        if method == "qmix":
            assert not bool(boundary.replay.is_full)
            assert int(boundary.replay.current_index) == 0
            equal(boundary.target_q_params, previous.target_q_params)
            clean = ends[child.run_dir].replay.experience
        else:
            assert int(boundary.recent.size) == 0
            clean = ends[child.run_dir].recent.rows
        rewards = np.asarray(clean.task_reward)[np.asarray(clean.valid)]
        assert rewards.size > 0 and np.all(rewards > 5.0)
    details = checkpoints.read_checkpoint_details(child.final_checkpoint)
    change = details["metadata"]["continuation"]["reward_change"]
    assert change["changed"]
    assert change["cleared_stored_experience"] == (method != "ff_ippo")
    assert change["optimizer_and_statistics"] == "carried_over"
    rows = [
        json.loads(line)
        for line in (child.run_dir / "training_updates.jsonl").read_text().splitlines()
    ]
    assert rows and all(row["custom_reward_mean"] == 7.0 for row in rows)
    assert all(row["shaping_mean"] == 0.0 for row in rows)
    before_resume = ends[child.run_dir]
    resumed = train(resume_from=child.final_checkpoint)
    assert resumed.completed_updates == child.completed_updates
    equal(ends[child.run_dir], before_resume)
    grandchild = extend_training(
        child.final_checkpoint,
        additional_env_steps=12,
        output_dir=tmp_path / "grandchild",
    )
    _, next_boundary = starts[grandchild.run_dir]
    if method == "qmix":
        equal(next_boundary.replay, before_resume.replay)
    elif method == "pqn_vdn":
        equal(next_boundary.recent, before_resume.recent)
    assert grandchild.completed_updates > child.completed_updates
    grandchild_end = ends[grandchild.run_dir]
    train(resume_from=grandchild.final_checkpoint)
    equal(ends[grandchild.run_dir], grandchild_end)
