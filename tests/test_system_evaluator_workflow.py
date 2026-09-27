"""Trace complete System evaluations through refill, records and scoped results.

CPU cases compare actual replay trajectories, fresh recurrent memory and action
randomness. They cover scalar/odd worker counts, mixed host/JAX execution, one
provider call per decision, provider failure before advancement, and compilation
reuse for same-shaped changed parameters. No learner or Core rule changes here.
Fixed batches also keep padding inactive before custom initialization, preserve
real game keys through refill and resume, and never publish padded games. A
default evaluation saves replay V4 files whose resolved config V2 and context
column 19 record the default Red Zone depth 5.0. All scheduled rosters are
checked through nested teams and possible pool members before any method,
resource scope or output file is opened.
"""

# Public workflow proofs also inspect their shared compiled execution boundary.
# pyright: reportPrivateUsage=false

from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    evaluate,
    evaluate_episodes,
)
from marl_battlegrounds.evaluation.models import ResolvedEnvConfigV2
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    SystemOutput,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4
from marl_battlegrounds.evaluation.results import EvaluationResult
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.tasks import (
    DEFAULT_TDM_RED_ZONE_DEPTH,
    AgentClassName,
    make_standard_team_deathmatch_config,
)

runner = import_module("marl_battlegrounds.evaluation.evaluate")


def _specs() -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(
            i + 1,
            make_standard_team_deathmatch_config(
                map_id=12,
                team_a_roster=("priest",),
                team_b_roster=("mage", "mage"),
                max_steps=horizon,
            ),
            seed_id=100 + i,
        )
        for i, horizon in enumerate((1, 3, 2, 1))
    )


def _init(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, jnp.int32)


def _act(
    variables: PolicyTree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    def actors(key: Array) -> Array:
        return jax.random.split(key, 5)

    action = jax.vmap(jax.vmap(random_policy))(
        inputs.actors.observation,
        inputs.action_mask,
        jax.vmap(actors)(keys),
    )
    prefer_idle = (memory[:, None] % 2 == 0) & (variables > 0)
    action = action._replace(
        move=jnp.where(
            prefer_idle & inputs.action_mask.move_mask[..., 0], 0, action.move
        )
    )
    return SystemOutput(
        action,
        memory + inputs.valid.astype(jnp.int32),
        learning_outputs=jnp.zeros((*inputs.active_mask.shape, 64)),
    )


def test_explicit_shared_adapter_retains_bare_policy_trajectories() -> None:
    method = policy("random")
    options: dict[str, Any] = dict(
        num_envs=2,
        chunk_size=2,
        metrics="none",
        replay_episodes=(1, 2, 3, 4),
        run_id="parity",
    )
    bare = evaluate_episodes(method, method, _specs(), **options)
    adapted = evaluate_episodes(
        shared_policy(method), shared_policy(method), _specs(), **options
    )
    first = {r.header.context.identity.episode_id: r for r in bare.replays}
    second = {r.header.context.identity.episode_id: r for r in adapted.replays}
    # Run identities differ; per-game replay content has the same ID suffix.
    for a, b in zip(first.values(), second.values(), strict=True):
        assert a.frames == b.frames
        assert a.transitions == b.transitions
    assert bare.episodes == adapted.episodes


def test_recurrent_system_refill_and_chunking_keep_games_identical() -> None:
    method = System("count", _act, variables=jnp.int32(1), init=_init)
    first = evaluate_episodes(
        method,
        method,
        _specs(),
        num_envs=1,
        chunk_size=1,
        replay_episodes=(1, 2, 3, 4),
        metrics="none",
        run_id="refill",
    )
    second = evaluate_episodes(
        method,
        method,
        tuple(reversed(_specs())),
        num_envs=3,
        chunk_size=2,
        replay_episodes=(1, 2, 3, 4),
        metrics="none",
        run_id="refill",
    )

    def indexed(result: EvaluationResult) -> dict[int, ReplayArtifactV4]:
        return {
            int(r.header.context.identity.episode_id.split("-")[-1]): r
            for r in result.replays
        }

    left, right = indexed(first), indexed(second)
    for i in left:
        assert left[i].frames == right[i].frames
        assert left[i].transitions == right[i].transitions
    assert first.episodes == second.episodes


@pytest.mark.parametrize("host_team", [0, 1])
def test_host_system_opaque_memory_and_batch_calls_survive_refill(
    host_team: int,
) -> None:
    observed: dict[tuple[int, ...], int] = {}
    batch_shapes: list[int] = []
    opaque = object()

    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: Array
    ) -> list[dict[str, object]]:
        del variables, keys
        return [{"count": 0, "session": opaque} for _ in inputs.valid]

    def host(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables
        batch_shapes.append(len(inputs.valid))
        result = list(memory)
        for lane in np.flatnonzero(inputs.valid):
            assert memory[lane]["session"] is opaque
            identity = tuple(int(x) for x in jax.random.key_data(keys[lane]))
            assert identity not in observed
            observed[identity] = memory[lane]["count"]
            result[lane] = {"count": memory[lane]["count"] + 1, "session": opaque}
        zero = cast(Array, np.zeros(inputs.active_mask.shape, np.int32))
        return SystemOutput(
            ActorAction(zero, zero, zero), result, learning_outputs=opaque
        )

    provider = System("provider", host, init=initialize, execution="host")
    numerical = System("count", _act, variables=jnp.int32(0), init=_init)
    methods = (provider, numerical) if host_team == 0 else (numerical, provider)
    evaluate_episodes(*methods, _specs(), num_envs=1, chunk_size=2, metrics="none")
    reference = dict(observed)
    assert set(batch_shapes) == {1}
    observed.clear()
    batch_shapes.clear()
    evaluate_episodes(
        *methods, tuple(reversed(_specs())), num_envs=3, chunk_size=2, metrics="none"
    )
    assert observed == reference
    assert set(batch_shapes) == {3}
    assert len(batch_shapes) == 3


def test_host_failure_precedes_environment_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fail(
        variables: PolicyTree, memory: PolicyTree, inputs: SystemInput, keys: Array
    ) -> SystemOutput:
        del variables, memory, inputs, keys
        raise RuntimeError("provider stopped")

    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append(1)
        raise AssertionError("environment advanced after provider failure")

    monkeypatch.setattr(runner, "_step_environment", forbidden)
    with pytest.raises(RuntimeError, match="provider stopped"):
        evaluate_episodes(
            System("failing", fail, execution="host"),
            "random",
            _specs(),
            num_envs=2,
            metrics="none",
        )
    assert not calls


def test_same_shape_checkpoint_values_reuse_system_chunk_program() -> None:
    method = System("counter", _act, variables=jnp.int32(0), init=_init)
    evaluate_episodes(
        method, method, _specs(), num_envs=2, metrics="none", chunk_size=1
    )
    count = runner._jax_system_chunk._cache_size()
    changed = replace(method, variables=jnp.int32(1))
    result = evaluate_episodes(
        changed, changed, _specs(), num_envs=2, metrics="none", chunk_size=1
    )
    assert runner._jax_system_chunk._cache_size() == count
    assert result.status == "complete"


def test_saved_system_pair_routes_and_replays_join_actual_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = runner.capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(runner, "capture_recording_provenance", same_source)
    method = System("counter", _act, variables=jnp.int32(0), init=_init)
    result = evaluate(
        method,
        method,
        num_episodes=2,
        maps=[12],
        max_steps=2,
        num_envs=2,
        chunk_size=3,
        metrics="none",
        full_metrics_episodes=[1],
        save_replays=2,
        output_dir=tmp_path,
    )
    assert result.status == "complete"
    assert len(result.replay_paths) == 2
    # A default evaluation records the default Red Zone depth in each replay.
    for path in result.replay_paths:
        replay = load_replay(path).replay
        assert type(replay) is ReplayArtifactV4
        resolved = replay.header.context.resolved_env_config
        assert type(resolved) is ResolvedEnvConfigV2
        assert resolved.team_deathmatch_red_zone_depth == DEFAULT_TDM_RED_ZONE_DEPTH
        assert DEFAULT_TDM_RED_ZONE_DEPTH == 5.0
        assert {
            row[19]
            for frame in replay.frames
            for row in frame.base_observation.context_features
        } == {5.0}
    assert len(result.table("episodes")["episode_id"]) == 2
    assert list(result.table("full_metrics")["episode_id"]) == [1]
    assert result.metadata["spawn_balance"]["paired_complete"] is True
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "Read-only verification must not open a writer or initialize a method"
        )

    monkeypatch.setattr(runner.RunWriter, "__init__", forbidden)
    monkeypatch.setattr(runner, "_init_system_pair", forbidden)
    options: dict[str, Any] = dict(
        num_episodes=2,
        maps=[12],
        max_steps=2,
        metrics="none",
        full_metrics_episodes=[1],
        save_replays=2,
        resume_from=result.run_dir,
    )
    runner._verify_evaluation(method, method, **options)
    with pytest.raises(ValueError, match="identity"):
        runner._verify_evaluation(
            replace(method, variables=jnp.int32(1)), method, **options
        )
    with pytest.raises(ValueError, match="saved"):
        runner._verify_evaluation(method, method, seed=123, **options)
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == before


def test_fixed_batch_tails_keep_exact_games_and_reuse_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = System("count", _act, variables=jnp.int32(1), init=_init)
    config = _specs()[0].env_config
    original_init = runner._init_system_pair
    original_chunk = runner._jax_system_chunk
    starts: list[tuple[np.ndarray, np.ndarray]] = []
    transitions: list[tuple[np.ndarray, np.ndarray]] = []

    def initialize(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - Observe the shared runner without changing its call contract.
        state = args[6]
        starts.append((np.asarray(state.episode_id), np.asarray(~state.done.done)))
        return original_init(*args, **kwargs)

    def chunk(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - Observe the shared runner without changing its call contract.
        result = original_chunk(*args, **kwargs)
        state = result[0].state
        transitions.append(
            (
                np.asarray(state.episode_id),
                np.asarray(state.cumulative_transition_count),
            )
        )
        return result

    monkeypatch.setattr(runner, "_init_system_pair", initialize)
    monkeypatch.setattr(runner, "_jax_system_chunk", chunk)
    compiled = None
    for count in (1, 31, 32, 33):
        ids = (np.iinfo(np.int32).max, *range(1, count))
        specs = tuple(EpisodeSpec(i, config, seed_id=i) for i in ids)
        result = evaluate_episodes(
            method,
            method,
            specs,
            num_envs=32,
            keep_batch_size=True,
            chunk_size=1,
            metrics="none",
        )
        assert result.metadata["num_envs"] == 32
        assert set(result.completed_episode_ids) == set(ids)
        assert set(result.metadata["schedule"]) == set(ids)
        start_ids, valid = starts[-1]
        assert start_ids.shape == (32,)
        assert valid.sum() == min(count, 32)
        assert set(start_ids[~valid]).isdisjoint(ids)
        assert len(set(start_ids)) == 32
        assert np.all(start_ids > 0)
        for lane_ids, steps in transitions:
            assert np.all(steps[~np.isin(lane_ids, ids)] == 0)
        transitions.clear()
        if compiled is None:
            compiled = original_chunk._cache_size()
        assert original_chunk._cache_size() == compiled


def test_fixed_batch_setting_rejects_non_boolean_before_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_batch_size must be a bool"):
        evaluate(
            "random",
            "random",
            num_episodes=2,
            maps=[12],
            keep_batch_size=cast(bool, 1),
            output_dir=tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_fixed_batch_matches_real_replays_and_default_stays_small() -> None:
    method = System("count", _act, variables=jnp.int32(1), init=_init)
    specs = _specs()[:1]
    options: dict[str, Any] = dict(
        num_envs=32,
        chunk_size=1,
        metrics="none",
        replay_episodes=(1,),
        run_id="fixed-batch-parity",
    )
    small = evaluate_episodes(method, method, specs, **options)
    padded = evaluate_episodes(method, method, specs, keep_batch_size=True, **options)
    assert small.metadata["num_envs"] == 1
    assert padded.metadata["num_envs"] == 32
    assert small.episodes == padded.episodes
    assert len(padded.replays) == 1
    assert small.replays[0].frames == padded.replays[0].frames
    assert small.replays[0].transitions == padded.replays[0].transitions


def test_fixed_batch_host_resume_initializes_only_real_pending_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = runner.capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(runner, "capture_recording_provenance", same_source)
    opaque = object()
    initial_valid: list[np.ndarray] = []
    chosen_keys: list[tuple[int, ...]] = []
    should_fail = True
    calls = 0

    def initialize(
        variables: PolicyTree, inputs: SystemInput, keys: Array
    ) -> dict[str, list[object | None]]:
        del variables, keys
        initial_valid.append(np.asarray(inputs.valid).copy())
        return {"lanes": [opaque if valid else None for valid in inputs.valid]}

    def reset_memory(
        memory: dict[str, list[object | None]],
        fresh: dict[str, list[object | None]],
        mask: Array,
    ) -> dict[str, list[object | None]]:
        return {
            "lanes": [
                new if selected else old
                for old, new, selected in zip(
                    memory["lanes"], fresh["lanes"], mask, strict=True
                )
            ]
        }

    def host(
        variables: PolicyTree,
        memory: dict[str, list[object | None]],
        inputs: SystemInput,
        keys: Array,
    ) -> SystemOutput:
        nonlocal calls
        del variables
        calls += 1
        if should_fail and calls == 2:
            raise RuntimeError("stop after saved first batch")
        assert inputs.valid.shape == (32,)
        for lane in np.flatnonzero(inputs.valid):
            assert memory["lanes"][lane] is opaque
            chosen_keys.append(tuple(int(x) for x in jax.random.key_data(keys[lane])))
        zero = cast(Array, np.zeros(inputs.active_mask.shape, np.int32))
        return SystemOutput(ActorAction(zero, zero, zero), memory)

    provider = System(
        "opaque", host, init=initialize, reset_memory=reset_memory, execution="host"
    )
    config = _specs()[0].env_config
    specs = tuple(EpisodeSpec(i, config) for i in range(1, 34))
    options: dict[str, Any] = dict(
        num_envs=32, keep_batch_size=True, chunk_size=1, metrics="none"
    )
    with pytest.raises(RuntimeError, match="stop after saved first batch"):
        evaluate_episodes(provider, "random", specs, output_dir=tmp_path, **options)
    run_dir = next(tmp_path.iterdir())
    should_fail = False
    initial_valid.clear()
    resumed = evaluate_episodes(
        provider, "random", specs, resume_from=run_dir, **options
    )
    assert resumed.completed_episode_ids == tuple(range(1, 34))
    assert [episode.episode_id for episode in resumed.episodes] == [33]
    assert resumed.metadata["num_envs"] == 32
    assert len(resumed.table("episodes")["episode_id"]) == 33
    assert initial_valid[0].shape == (32,)
    assert np.array_equal(initial_valid[0], np.arange(32) == 0)
    assert len(chosen_keys) == 33
    assert len(set(chosen_keys)) == 33
    assert calls == 3


@pytest.mark.parametrize("kind", ["missing", "overlap", "pool", "short_adapter"])
@pytest.mark.parametrize("side", [0, 1])
def test_later_composed_roster_fails_before_methods_resources_or_output(
    kind: str, side: int, tmp_path: Path
) -> None:
    from marl_battlegrounds.evaluation.system_evaluation import (
        prepare_evaluation_system,
        validate_evaluation_rosters,
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        del args, kwargs
        pytest.fail("roster checks must finish before methods or resources run")

    provider = System(
        "Unused provider",
        forbidden,
        init=forbidden,
        execution="host",
        resource_scope=forbidden,
    )
    first_roster: tuple[AgentClassName, ...]
    second_roster: tuple[AgentClassName, ...]
    if kind == "overlap":
        first_roster, second_roster = ("priest", "mage"), ("mage", "mage")
        method = marl_bgs.team(provider, provider, slots=[0, "mage"])
    elif kind == "short_adapter":
        first_roster, second_roster = ("mage", "priest"), ("priest", "mage")
        short = marl_bgs.independent_policies(
            (marl_bgs.Policy("One slot", forbidden, execution="host"),)
        )
        method = marl_bgs.team(short, provider, slots=["mage", "priest"])
    else:
        first_roster, second_roster = ("mage", "warrior"), ("mage", "priest")
        method = marl_bgs.team(provider, provider, slots=["mage", "warrior"])
        if kind == "pool":
            method = marl_bgs.pool({provider: 1, method: 0.000001})
    specs = tuple(
        EpisodeSpec(
            index + 1,
            make_standard_team_deathmatch_config(
                map_id=0,
                team_a_roster=roster if side == 0 else ("mage",),
                team_b_roster=roster if side == 1 else ("mage",),
                max_steps=1,
            ),
            seed_id=index,
        )
        for index, roster in enumerate((first_roster, second_roster))
    )
    a, b = (method, provider) if side == 0 else (provider, method)
    execution_a, variables_a, _ = prepare_evaluation_system(a)
    execution_b, variables_b, _ = prepare_evaluation_system(b)
    # Only the later refill is incompatible.
    validate_evaluation_rosters(
        execution_a,
        execution_b,
        specs[0].env_config,
        variables_a=variables_a,
        variables_b=variables_b,
    )
    output = tmp_path / "results"
    error = "active prefix" if kind == "short_adapter" else "cover each"
    with pytest.raises(ValueError, match=error):
        evaluate_episodes(
            a,
            b,
            specs,
            num_envs=1,
            chunk_size=1,
            metrics="none",
            output_dir=output,
        )
    assert not output.exists()
    if kind == "pool":
        disabled = marl_bgs.pool({provider: 1, method: 0})
        execution, variables, _ = prepare_evaluation_system(disabled)
        other, other_variables, _ = prepare_evaluation_system(provider)
        # A zero-share member cannot own slots in these frozen games.
        validate_evaluation_rosters(
            execution if side == 0 else other,
            other if side == 0 else execution,
            specs[1].env_config,
            variables_a=variables if side == 0 else other_variables,
            variables_b=other_variables if side == 0 else variables,
        )
