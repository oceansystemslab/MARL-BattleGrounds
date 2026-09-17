"""Check complete games through the public evaluation and policy interfaces."""

import csv
import io
import json
from collections.abc import Callable
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import TASK_MODE_OUTCOME_DRAW, ActionMask, EnvConfig
from marl_battlegrounds.evaluation.analysis import analyze_replay
from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    episode_keys,
    evaluate,
    evaluate_episodes,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    PRIORITY_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.policy_execution import Policy, PolicyTree, policy
from marl_battlegrounds.evaluation.recording_context import capture_recording_provenance
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS, RunWriter
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    TDMMapInfo,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
)


def _config(max_steps: int = 2) -> EnvConfig:
    return make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("priest",),
        team_b_roster=("mage", "mage"),
        max_steps=max_steps,
    )


def test_public_evaluate_cycles_one_total_budget_and_creates_no_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = evaluate(
        "random",
        "random",
        num_episodes=3,
        maps=[
            TDMMapInfo.model_validate_json(list_tdm_maps()[12].model_dump_json()),
            13,
        ],
        num_envs=2,
        max_steps=2,
        chunk_size=4,
        system_roster=("priest",),
        opponent_roster=("mage", "mage"),
        phase="validation",
        pass_id="2500",
        spawn_mode="default",
    )
    assert [
        (row.episode_id, row.map_id, row.episode_length) for row in result.episodes
    ] == [
        (1, 12, 2),
        (2, 13, 2),
        (3, 12, 2),
    ]
    assert all(row.outcome == TASK_MODE_OUTCOME_DRAW for row in result.episodes)
    assert set(PRIORITY_METRIC_NAMES) <= result.priority_metrics.keys()
    assert tuple(result.priority_metrics) == (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES)
    assert set(result.priority_metrics["run_id"]) == {result.metadata["run_id"]}
    assert tuple(result.priority_metrics["config_id"]) == tuple(
        episode.config_id for episode in result.episodes
    )
    assert set(result.priority_metrics["config_id"]) == set(
        cast(dict[str, object], result.metadata["configurations"])
    )
    np.testing.assert_array_equal(result.priority_metrics["team_a_draw"], [1, 1, 1])
    np.testing.assert_array_equal(result.priority_metrics["phase"], ["validation"] * 3)
    assert not any(
        name.endswith("_return") and name.startswith("agent_")
        for name in result.priority_metrics
    )
    np.testing.assert_array_equal(result.priority_metrics["team_a_return"], [0, 0, 0])
    assert result.full_metrics == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ({"map_id": 0}, {"resource_sha256": "0" * 64}))
def test_saved_map_details_are_rejected_before_evaluation_creates_files(
    change: dict[str, object], tmp_path: Path
) -> None:
    saved = list_tdm_maps()[12].model_copy(update=change)
    destination = tmp_path / "evaluation"
    with pytest.raises(
        ValueError, match=r"choose the map again with list_tdm_maps\(\)"
    ):
        evaluate(
            "random",
            "random",
            num_episodes=1,
            maps=[saved],
            output_dir=destination,
            spawn_mode="default",
        )
    assert not destination.exists()


def test_default_maps_are_canonical_and_metrics_can_be_completely_disabled() -> None:
    result = evaluate(
        "random",
        "random",
        num_episodes=5,
        max_steps=1,
        num_envs=3,
        metrics="none",
        chunk_size=2,
        spawn_mode="default",
    )
    assert (
        tuple(row.map_id for row in result.episodes) == CANONICAL_TDM_EVALUATION_MAP_IDS
    )
    assert result.priority_metrics == result.full_metrics == {}
    assert len(result.episodes) == 5


def test_selected_full_metrics_are_self_contained_with_uncomputed_rows_absent() -> None:
    result = evaluate(
        "random",
        "random",
        num_episodes=3,
        maps=[_config()],
        num_envs=2,
        metrics="none",
        full_metrics_episodes=[2, 2],
        chunk_size=4,
        spawn_mode="default",
    )
    np.testing.assert_array_equal(result.priority_metrics["episode_id"], [2])
    np.testing.assert_array_equal(result.full_metrics["episode_id"], [2])
    assert set(FULL_METRIC_NAMES) <= result.full_metrics.keys()
    for name in PRIORITY_METRIC_NAMES:
        np.testing.assert_array_equal(
            result.full_metrics[name], result.priority_metrics[name]
        )
    assert result.full_metrics["episode_length"][0] == 2
    assert result.full_metrics["agent_0_damage_done"][0] == 0


@pytest.mark.parametrize("execution", ("jax", "host"))
def test_repeated_evaluation_reuses_kernels_when_capture_selections_change(
    execution: Literal["jax", "host"],
) -> None:
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    controller = Policy("random", policy("random").apply, execution=execution)
    specs = tuple(
        EpisodeSpec(index + 1, _config(length), seed_id=100 + index)
        for index, length in enumerate((1, 2, 2))
    )
    kernel = evaluator._jax_chunk if execution == "jax" else evaluator._step_environment
    first = evaluate_episodes(
        controller,
        controller,
        specs,
        seed=28,
        num_envs=2,
        metrics="none",
        full_metrics_episodes=(1,),
        replay_episodes=(1, 2),
        chunk_size=1,
        run_id="selection-reuse-proof",
    )
    # Count the actual shared kernels, not only an outer caller's traces.
    cache_sizes = (evaluator._reset._cache_size(), kernel._cache_size())
    assert all(size > 0 for size in cache_sizes)
    second = evaluate_episodes(
        controller,
        controller,
        specs,
        seed=28,
        num_envs=2,
        metrics="none",
        full_metrics_episodes=(3,),
        replay_episodes=(2, 3),
        chunk_size=1,
        run_id="selection-reuse-proof",
    )
    assert (evaluator._reset._cache_size(), kernel._cache_size()) == cache_sizes
    assert first.episodes == second.episodes
    assert tuple(row.episode_length for row in second.episodes) == (1, 2, 2)
    np.testing.assert_array_equal(first.full_metrics["episode_id"], (1,))
    np.testing.assert_array_equal(first.full_metrics["episode_length"], (1,))
    np.testing.assert_array_equal(second.full_metrics["episode_id"], (3,))
    np.testing.assert_array_equal(second.full_metrics["episode_length"], (2,))
    np.testing.assert_array_equal(second.priority_metrics["episode_id"], (3,))

    first_replays = {
        replay.header.context.seed_protocol.episode_seed: replay
        for replay in first.replays
    }
    second_replays = {
        replay.header.context.seed_protocol.episode_seed: replay
        for replay in second.replays
    }
    assert set(first_replays) == {100, 101}
    assert set(second_replays) == {101, 102}
    assert first_replays[101].frames == second_replays[101].frames
    assert first_replays[101].transitions == second_replays[101].transitions


def test_host_random_actions_and_recurrent_memory_survive_batch_order_and_refills() -> (
    None
):
    configs = [_config(value) for value in (1, 3, 2, 1, 2)]
    specs = tuple(
        EpisodeSpec(index + 1, config, seed_id=100 + index)
        for index, config in enumerate(configs)
    )
    observed: dict[tuple[int, ...], tuple[int, ...]] = {}

    def record(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del variables
        action = random_policy(actor.observation, mask, key)
        identity = tuple(int(value) for value in jax.random.key_data(key))
        assert identity not in observed
        observed[identity] = (int(memory), *(int(value) for value in action))
        return action, memory + 1

    controller = Policy("record", record, initial_carry=jnp.int32(0), execution="host")
    first = evaluate_episodes(
        controller, controller, specs, num_envs=1, chunk_size=2, metrics="none", seed=19
    )
    baseline = observed.copy()
    observed.clear()
    second = evaluate_episodes(
        controller,
        controller,
        tuple(reversed(specs)),
        num_envs=3,
        chunk_size=2,
        metrics="none",
        seed=19,
    )
    assert first.episodes == second.episodes
    assert observed == baseline
    assert len(observed) == 10 * sum((1, 3, 2, 1, 2))
    assert sum(value[0] == 0 for value in observed.values()) == 50
    assert sum(value[0] == 2 for value in observed.values()) == 10


def test_jax_and_host_execution_use_identical_per_actor_random_streams() -> None:
    result_a = evaluate(
        "random",
        "random",
        num_episodes=3,
        maps=[_config(3)],
        num_envs=2,
        chunk_size=2,
        metrics="full",
        seed=31,
        spawn_mode="default",
    )
    original = policy("random")
    host = Policy("random-host", original.apply, execution="host")
    result_b = evaluate(
        host,
        host,
        num_episodes=3,
        maps=[_config(3)],
        num_envs=2,
        chunk_size=2,
        metrics="full",
        seed=31,
        spawn_mode="default",
    )
    assert result_a.episodes == result_b.episodes
    for name in FULL_METRIC_NAMES:
        np.testing.assert_allclose(
            np.asarray(result_a.full_metrics[name], dtype=np.float32),
            np.asarray(result_b.full_metrics[name], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
            err_msg=name,
        )


def test_numpy_variables_are_frozen_for_whole_pass() -> None:
    variable = np.asarray(7, dtype=np.int32)
    values: list[int] = []

    def inspect(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del actor, mask, key
        values.append(int(variables))
        variable[...] = 99
        zero = jnp.int32(0)
        return ActorAction(zero, zero, zero), memory

    controller = Policy("frozen", inspect, variables=variable, execution="host")
    evaluate(
        controller,
        controller,
        num_episodes=2,
        maps=[_config(1)],
        num_envs=1,
        chunk_size=2,
        metrics="none",
        spawn_mode="default",
    )
    assert values == [7] * 20


@pytest.mark.parametrize("error", [RuntimeError, TimeoutError, KeyboardInterrupt])
@pytest.mark.parametrize("destination", ["none", "output", "shared"])
def test_provider_failures_timeouts_and_cancellation_are_propagated(
    error: type[BaseException],
    destination: str,
    tmp_path: Path,
) -> None:
    failure = error("provider failed")

    def fail(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del variables, memory, actor, mask, key
        raise failure

    controller = Policy("failure", fail, execution="host")
    with (
        pytest.raises(error, match="provider failed") as caught,
        ExitStack() as stack,
    ):
        writer = (
            stack.enter_context(RunWriter(tmp_path))
            if destination == "shared"
            else None
        )
        evaluate(
            controller,
            "random",
            num_episodes=1,
            maps=[_config(1)],
            metrics="none",
            writer=writer,
            output_dir=tmp_path if destination == "output" else None,
            pass_id="provider-failure",
            spawn_mode="default",
        )
    assert caught.value is failure
    assert "failure" in caught.value.__notes__[0]
    assert "episode IDs [1]" in caught.value.__notes__[0]
    if destination == "none":
        assert list(tmp_path.iterdir()) == []
    else:
        run_dir = next(tmp_path.iterdir())
        assert caught.value.__notes__.count(f"Run directory: {run_dir}") == 1
        failures = [
            json.loads(line)
            for line in (run_dir / "failures.jsonl").read_text().splitlines()
        ]
        assert len(failures) == 1
        assert failures[0]["type"] == error.__name__
        assert failures[0]["message"] == "provider failed"
        assert failures[0]["notes"] == caught.value.__notes__
        manifest = json.loads((run_dir / "run_details.json").read_text())
        assert next(iter(manifest["passes"].values()))["completed_episode_ids"] == []


@pytest.mark.parametrize("failed_reset", (1, 2), ids=("initial", "refill"))
def test_shared_writer_records_reset_failure_before_caller_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_reset: int
) -> None:
    execution = import_module("marl_battlegrounds.evaluation.evaluate")
    reset = cast(Callable[..., object], execution._reset)  # pyright: ignore[reportPrivateUsage]
    calls = 0
    failure = RuntimeError("reset failed")

    def fail_reset(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == failed_reset:
            raise failure
        return reset(*args, **kwargs)

    monkeypatch.setattr(execution, "_reset", fail_reset)
    with RunWriter(tmp_path) as writer:
        with pytest.raises(RuntimeError, match="reset failed") as caught:
            evaluate(
                "random",
                "random",
                num_episodes=2,
                maps=[_config(1)],
                num_envs=1,
                chunk_size=1,
                metrics="none",
                writer=writer,
                pass_id="reset-failure",
                spawn_mode="default",
            )
        assert caught.value is failure
        failures = (writer.run_dir / "failures.jsonl").read_text().splitlines()
        assert len(failures) == 1
        assert json.loads(failures[0])["message"] == "reset failed"
        assert writer.completed_episode_ids == (
            frozenset() if failed_reset == 1 else frozenset((1,))
        )


def test_authored_nonzero_initial_epoch_preserves_scores_and_local_length() -> None:
    config = _config(7)
    state, _, _, _ = core.reset(config, jax.random.key(0))
    state = state._replace(
        step_count=jnp.int32(5), team_deathmatch_scores=jnp.asarray([2, 3], jnp.int32)
    )
    episodes = (EpisodeSpec(7, config, initial_state=state), EpisodeSpec(9, _config(1)))
    result = evaluate_episodes(
        policy("random"),
        policy("random"),
        episodes,
        num_envs=2,
        chunk_size=4,
        replay_episodes=[7],
    )
    assert [
        (row.episode_id, row.episode_length, row.team_a_score, row.team_b_score)
        for row in result.episodes
    ] == [(7, 2, 2, 3), (9, 1, 0, 0)]
    np.testing.assert_array_equal(result.priority_metrics["episode_length"], [2, 1])
    assert len(result.replays) == 1
    analysis = analyze_replay(LoadedReplay(result.replays[0], None, "not_recorded"))
    for index, tick in enumerate((5, 6, 7)):
        summary = analysis.summary(index)
        row = next(csv.DictReader(io.StringIO(analysis.csv(index))))
        assert summary["frame_index"] == index
        assert summary["simulator_step_count"] == tick
        assert row["frame_index"] == str(index)
        assert row["simulator_step_count"] == str(tick)
        assert float(row["episode_length"]) == index


def test_invalid_selection_and_schedule_fail_before_policy_execution() -> None:
    with pytest.raises(ValueError, match="outside the schedule"):
        evaluate(
            "random",
            "random",
            num_episodes=1,
            full_metrics_episodes=[2],
            spawn_mode="default",
        )
    with pytest.raises(ValueError, match="positive int32"):
        evaluate(
            "random",
            "random",
            num_episodes=1,
            full_metrics_episodes=[True],
            spawn_mode="default",
        )
    with pytest.raises(ValueError, match="unique"):
        evaluate_episodes(
            policy("random"),
            policy("random"),
            [EpisodeSpec(1, _config()), EpisodeSpec(1, _config())],
        )


def test_selected_replays_in_memory_need_no_metrics_or_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = evaluate(
        "random",
        "random",
        num_episodes=3,
        maps=[_config(3)],
        num_envs=2,
        metrics="none",
        replay_episodes=[2],
        chunk_size=2,
        spawn_mode="default",
    )
    assert result.priority_metrics == result.full_metrics == {}
    assert result.completed_episode_ids == (1, 2, 3)
    assert result.paths is None
    assert len(result.replays) == 1
    replay = result.replays[0]
    assert replay.header.context.identity.episode_id.endswith("episode-2")
    assert replay.header.context.identity.run_id == result.metadata["run_id"]
    assert len(replay.frames) == 4
    assert len(replay.transitions) == 3
    assert replay.header.context.seed_protocol.episode_seed == 2
    assert replay.header.context.policy_assignments[0].assignment_status == "assigned"
    assert list(tmp_path.iterdir()) == []


def test_persistence_keeps_scalar_tables_and_replays_independently(
    tmp_path: Path,
) -> None:
    result = evaluate(
        "random",
        "random",
        num_episodes=3,
        maps=[_config(2)],
        num_envs=2,
        metrics="none",
        full_metrics_episodes=[1],
        replay_episodes=[2],
        chunk_size=2,
        output_dir=tmp_path,
        spawn_mode="default",
    )
    assert result.priority_metrics == result.full_metrics == {}
    assert result.replays == ()
    assert result.completed_episode_ids == (1, 2, 3)
    assert result.paths is not None
    for name in ("priority_metrics", "full_metrics"):
        with result.paths[name].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["episode_id"] for row in rows] == ["1"]
        assert "agent_1_return" not in rows[0]
        assert float(rows[0]["team_a_return"]) == 0
    replays = list(result.paths["replays"].glob("*.json"))
    assert len(replays) == 1
    assert replays[0].name.startswith("tdm_custom_map_")
    assert "__episode-2__seed-0__stream-2__a-random__b-random__" in replays[0].name
    details = json.loads(result.paths["run_details"].read_text())
    record = next(iter(details["passes"].values()))["replays"]["2"]
    assert record["path"] == str(
        replays[0].relative_to(result.paths["run_details"].parent)
    )
    assert record["canonical_digest_sha256"] in replays[0].name


def test_resume_skips_durable_episodes_and_rejects_changed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Freeze source identity for this test: independent workspace edits must not
    # impersonate a deliberate code change between our two evaluation calls.
    provenance = capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "capture_recording_provenance",
        same_source,
    )
    failures_enabled = True
    failing_key = jax.random.fold_in(
        episode_keys(
            jax.random.key(0),
            jnp.asarray([3], jnp.uint32),
            jnp.asarray([0], jnp.uint32),
            2,
        )[0],
        0,
    )
    variable = np.asarray(0, np.int32)

    def controlled(
        variables: PolicyTree,
        memory: PolicyTree,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, PolicyTree]:
        del variables
        if failures_enabled and np.array_equal(
            jax.random.key_data(key), jax.random.key_data(failing_key)
        ):
            raise RuntimeError("interrupted after durable chunk")
        return random_policy(actor.observation, mask, key), memory

    controller = Policy("resumable", controlled, variables=variable, execution="host")
    with pytest.raises(RuntimeError, match="interrupted after durable chunk"):
        evaluate(
            controller,
            "random",
            num_episodes=4,
            maps=[_config(1)],
            num_envs=2,
            chunk_size=1,
            output_dir=tmp_path,
            spawn_mode="default",
        )
    run_dir = next(tmp_path.iterdir())
    manifest = json.loads((run_dir / "run_details.json").read_text())
    assert next(iter(manifest["passes"].values()))["completed_episode_ids"] == [1, 2]
    failures = (run_dir / "failures.jsonl").read_text().splitlines()
    assert len(failures) == 1
    assert json.loads(failures[0])["message"] == "interrupted after durable chunk"
    failures_enabled = False
    resumed = evaluate(
        controller,
        "random",
        num_episodes=4,
        maps=[_config(1)],
        num_envs=3,
        chunk_size=2,
        resume_from=run_dir,
        spawn_mode="default",
    )
    assert [row.episode_id for row in resumed.episodes] == [3, 4]
    assert resumed.completed_episode_ids == (1, 2, 3, 4)
    assert resumed.paths is not None
    with resumed.paths["priority_metrics"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["episode_id"] for row in rows] == ["1", "2", "3", "4"]
    repeated = evaluate(
        controller,
        "random",
        num_episodes=4,
        maps=[_config(1)],
        num_envs=1,
        resume_from=run_dir,
        spawn_mode="default",
    )
    assert repeated.episodes == ()
    with pytest.raises(ValueError, match="seed differs"):
        evaluate(
            controller,
            "random",
            num_episodes=4,
            maps=[_config(1)],
            resume_from=run_dir,
            seed=1,
            spawn_mode="default",
        )
    with pytest.raises(
        ValueError, match="explicit maps differ from the saved source choices"
    ):
        evaluate(
            controller,
            "random",
            num_episodes=4,
            maps=[_config(2)],
            resume_from=run_dir,
            spawn_mode="default",
        )
    variable[...] = 1
    with pytest.raises(ValueError, match="identity differs"):
        evaluate(
            controller,
            "random",
            num_episodes=4,
            maps=[_config(1)],
            resume_from=run_dir,
            spawn_mode="default",
        )


def test_shared_writer_appends_distinct_validation_passes(tmp_path: Path) -> None:
    with RunWriter(tmp_path) as writer:
        first = evaluate(
            "random",
            "random",
            num_episodes=1,
            maps=[_config(1)],
            writer=writer,
            phase="validation",
            pass_id="1000",
            chunk_size=1,
            spawn_mode="default",
        )
        second = evaluate(
            "random",
            "random",
            num_episodes=1,
            maps=[_config(1)],
            writer=writer,
            phase="validation",
            pass_id="2000",
            chunk_size=1,
            spawn_mode="default",
        )
        assert first.paths == second.paths
        assert first.completed_episode_ids == second.completed_episode_ids == (1,)
        with writer.paths["priority_metrics"].open(newline="") as stream:
            assert [row["pass_id"] for row in csv.DictReader(stream)] == [
                "1000",
                "2000",
            ]
