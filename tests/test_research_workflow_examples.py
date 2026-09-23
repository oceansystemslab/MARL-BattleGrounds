"""Check complete example ownership, fixed-shape curricula and numerical saves.

These CPU integration checks execute public environment/System/tracker calls.
They compare changing curriculum stages with one compiled shape and independent
seed execution with batched execution. Updates are illustrative counters, not
an optimizer or a learning claim. Existing restart tests own writer rollback.
The PPO example passes each method through the shared public workflow and
rejects attempts to override the method owned by a config or saved checkpoint.
"""

import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples import episode_tracking as curriculum
from examples import evaluation as evaluation_example
from examples import mappo_training as training_example
from examples import recorded_rollout as recorded
from examples import research_methods

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.tasks import list_tdm_maps

type Tree = Any
_rollout = cast(Callable[..., tuple[Tree, Tree]], curriculum.rollout)
_direct_scan = cast(Callable[..., tuple[recorded.Carry, Tree]], recorded.direct_scan)


def _equal(actual: Tree, expected: Tree) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(left, right)


def test_factory_returns_ordered_independent_policies_without_actions() -> None:
    custom = research_methods.load_custom()
    selected = research_methods.load_selected()
    assert custom.components is not None
    assert [entry["name"] for entry in custom.components] == ["tdm-alpha", "tdm-beta"]
    assert selected.name == "tdm-alpha"
    assert research_methods.load_custom() is not custom


def test_curriculum_pools_use_training_catalog_once() -> None:
    maps = [item for item in list_tdm_maps() if item.split == "training"]
    assert [item.map_id for item in maps] == list(range(42))
    assert [item.map_id for item in maps if item.curriculum] == list(range(12))
    for size in (*range(1, 13), 42):
        probabilities = np.asarray(curriculum.curriculum_probabilities(size))
        assert probabilities.shape == (42,)
        assert probabilities.dtype == np.float32
        np.testing.assert_array_equal(np.flatnonzero(probabilities), np.arange(size))
        np.testing.assert_allclose(probabilities[:size], 1 / size)
        np.testing.assert_allclose(probabilities.sum(), 1)
    for invalid in (0, 13, 41, 43, True):
        with pytest.raises(ValueError):
            curriculum.curriculum_probabilities(invalid)


def test_growing_curriculum_reuses_compilation_and_latest_stage_counts() -> None:
    env, carry = curriculum.make_context(42, automatic=False, num_envs=2)
    traces = 0

    @jax.jit
    def advance(current: Tree, probabilities: jax.Array, weight: jax.Array) -> Tree:
        nonlocal traces
        traces += 1
        return _rollout(env, current, probabilities, weight, automatic=False)

    bank = carry[4].source_configs
    for stage, size in enumerate((1, 2, 12, 42)):
        rng, obs, state, memory, tracking = carry
        carry = (
            rng,
            obs,
            state,
            memory,
            tracking.begin_stage(state, total_env_steps=32),
        )
        carry, transitions = cast(
            tuple[Tree, Tree],
            advance(
                carry,
                curriculum.curriculum_probabilities(size),
                jnp.float32(stage / 10),
            ),
        )
        jax.block_until_ready(transitions)
        latest_state, latest_tracking = carry[2], carry[4]
        summary = latest_tracking.stage_summary(latest_state)
        assert summary["status"] == "complete"
        assert summary["default_spawn_steps"] == summary["swapped_spawn_steps"] == 16
        assert np.all(np.asarray(latest_tracking.source_index) < size)
        _equal(latest_tracking.source_configs, bank)
    assert traces == 1


def test_new_pool_resets_finished_lanes_without_replacing_continuing_games() -> None:
    env, carry = curriculum.make_context(19, automatic=False, num_envs=2)
    rng, _, state, memory, tracking = carry
    indices = jnp.array([0, 1], jnp.int32)
    bank = tracking.source_configs

    def select(values: jax.Array) -> jax.Array:
        return values[indices]

    selected = jax.tree.map(select, bank)
    obs, state = env.reset(
        jax.random.key(20),
        marl_bgs.balanced_spawn_configs(selected, num_envs=2),
        state=state,
    )
    tracking = marl_bgs.init_episode_tracking(
        env, state, source_configs=bank, source_indices=indices
    )
    tracking = tracking.begin_stage(state, total_env_steps=4)
    carry, _ = _rollout(
        env,
        (rng, obs, state, memory, tracking),
        curriculum.curriculum_probabilities(2),
        jnp.float32(0.5),
        automatic=False,
        num_steps=2,
    )
    assert carry[4].stage_summary(carry[2])["status"] == "complete"
    before_ids = np.asarray(carry[2].episode_id).copy()
    rng, obs, state, memory, tracking = carry
    carry = (rng, obs, state, memory, tracking.begin_stage(state, total_env_steps=4))
    forced_pool = jnp.zeros(42, jnp.float32).at[41].set(1)
    carry, _ = _rollout(
        env, carry, forced_pool, jnp.float32(0.6), automatic=False, num_steps=2
    )
    np.testing.assert_array_equal(carry[4].source_index, [41, 1])
    assert int(carry[2].episode_id[0]) != before_ids[0]
    assert int(carry[2].episode_id[1]) == before_ids[1]
    assert carry[4].stage_summary(carry[2])["status"] == "complete"
    assert bool(carry[2].done.done[1])


def test_independent_updates_and_saved_carries_match_batched_execution(
    tmp_path: Path,
) -> None:
    contexts = [
        recorded.context(
            seed, automatic=True, recording=False, metrics="none", num_envs=2
        )
        for seed in (42, 43)
    ]
    contexts[1] = contexts[1]._replace(
        weight=jnp.float32(0.75), update_count=jnp.int32(7)
    )
    tokens = []
    for seed, current in zip((42, 43), contexts, strict=True):
        with marl_bgs.RunWriter(
            tmp_path / f"recording-{seed}", phase="training", pass_id=f"seed-{seed}"
        ) as writer:
            token = writer.checkpoint_recording()
        recorded.save_numerical_checkpoint(
            tmp_path / f"seed-{seed}.npz", current, token
        )
        tokens.append(token)
    assert tokens[0]["run_id"] != tokens[1]["run_id"]

    def run_one(current: recorded.Carry) -> recorded.Carry:
        latest, transitions = _direct_scan(current)
        return recorded.illustrative_update(latest, transitions)

    batched = jax.tree.map(lambda *values: jnp.stack(values), *contexts)
    actual = cast(recorded.Carry, jax.jit(jax.vmap(run_one))(batched))
    compiled_one = jax.jit(run_one)
    expected = [cast(recorded.Carry, compiled_one(current)) for current in contexts]
    for index in range(2):

        def select_seed(values: jax.Array, index: int = index) -> jax.Array:
            return values[index]

        one = jax.tree.map(select_seed, actual)
        _equal(one, expected[index])
        assert int(one.update_count) == (1 if index == 0 else 8)
        assert one.tracking.stage_summary(one.state)["status"] == "complete"
    other_before = jax.tree.map(jnp.copy, expected[1])
    restored, token = recorded.load_numerical_checkpoint(
        tmp_path / "seed-42.npz", contexts[0]
    )
    assert token == tokens[0]
    _equal(restored, contexts[0])
    _equal(compiled_one(restored), expected[0])
    _equal(expected[1], other_before)


@pytest.mark.parametrize(
    "workflow, explicit, expected",
    (
        ("evaluate", None, 32),
        ("validation", None, 32),
        ("tournament", None, 100),
        ("evaluate", 6, 6),
        ("tournament", 20, 20),
    ),
)
def test_evaluation_example_resolves_valid_workflow_budgets(
    workflow: str,
    explicit: int | None,
    expected: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def receive(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            tournament_results=(), completed_episode_ids=(), run_dir=None
        )

    def writer(_path: Path) -> nullcontext[SimpleNamespace]:
        return nullcontext(SimpleNamespace(paths={}, flush=lambda: None))

    args = ["evaluation.py", workflow]
    if workflow == "validation":
        args.extend(("--output-dir", str(tmp_path / "validation")))
    if explicit is not None:
        args.extend(("--episodes", str(explicit)))
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(marl_bgs, "evaluate", receive)
    monkeypatch.setattr(marl_bgs, "run_tournament", receive)
    monkeypatch.setattr(marl_bgs, "RunWriter", writer)
    evaluation_example.main()
    assert len(calls) == (2 if workflow == "validation" else 1)
    name = "episodes_per_pair" if workflow == "tournament" else "num_episodes"
    assert all(call[name] == expected for call in calls)
    assert all(call["num_envs"] == 32 for call in calls)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", (None, "mappo", "ippo", "ff_mappo", "ff_ippo"))
def test_training_example_passes_method_to_the_shared_public_workflow(
    method: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds import training

    calls: list[tuple[training.TrainConfig | None, dict[str, object]]] = []
    loaded: list[Path] = []
    reviews: list[list[Path]] = []
    actor = tmp_path / "selected-actor"

    def train(
        config: training.TrainConfig | None, **kwargs: object
    ) -> training.TrainResult:
        calls.append((config, kwargs))
        return training.TrainResult(
            tmp_path, tmp_path / "final-actor", actor, 32, 2, "complete", ()
        )

    def load(path: Path) -> object:
        loaded.append(path)
        return object()

    def analyze(paths: list[Path], **_kwargs: object) -> dict[str, object]:
        reviews.append(paths)
        return {"artifacts": {"summary": str(tmp_path / "summary.md")}}

    args = ["mappo_training.py", "--output-dir", str(tmp_path)]
    if method is not None:
        args.extend(("--method", method))
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(training, "load_system", load)
    monkeypatch.setattr(training, "analyze", analyze)
    training_example.main()
    assert len(calls) == 1
    config, kwargs = calls[0]
    assert config is not None and config.method == (method or "mappo")
    assert (config.num_envs, config.ppo.rollout_length, config.total_env_steps) == (
        4,
        4,
        32,
    )
    assert kwargs == {"output_dir": tmp_path, "resume_from": None}
    assert loaded == [actor] and reviews == [[tmp_path]]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("route", ("config", "resume"))
def test_training_example_rejects_method_override_of_saved_configuration(
    route: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ["mappo_training.py", "--method", "ippo"]
    if route == "config":
        args.extend(("--output-dir", str(tmp_path), "--config", "saved.json"))
    else:
        args.extend(("--resume-from", str(tmp_path)))
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        training_example.main()
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())
