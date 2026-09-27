"""Check complete example ownership, fixed-shape curricula and numerical saves.

These CPU integration checks execute public environment/System/tracker calls.
They compare changing curriculum stages with one compiled shape and independent
seed execution with batched execution. Updates are illustrative counters, not
an optimizer or a learning claim. Existing restart tests own writer rollback.
The PPO example passes each method through the shared public workflow and
rejects attempts to override the method owned by a config or saved checkpoint.
The QMIX example passes its tiny QMIX settings through the same workflow, and
the PQN-VDN example its tiny pqn settings, with the curriculum and shaping
switches reaching the config and --evaluate playing "tdm-alpha" in one
evaluation and one two-entrant tournament; the PQN-VDN example refuses those
switches with --config or --resume-from before any training call.
Cross-play keeps fixed focal/partner/opponent cells, declared familiarity,
actual snapshot labels and both spawn ends in recoverable saved game rows.
The local host example uses no provider. The own-learner example performs real
actor/critic updates, preserves actor information rights and legal choices,
loads exact own actor snapshots, then validates, selects and evaluates them.
The declared-stage curriculum entry point trains its exact tiny budget, loads
the actor, reports actual stage exposure and saves held-out paired evaluation.
These are workflow proofs, not evidence of learned competence.
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
from examples import own_learner, research_methods
from examples import pqn_training as pqn_example
from examples import qmix_training as qmix_example
from examples import recorded_rollout as recorded

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
            tmp_path,
            tmp_path / "final-actor",
            actor,
            32,
            2,
            "complete",
            (),
            tmp_path / "checkpoints" / "final",
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


def test_qmix_example_runs_the_shared_public_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training

    calls: list[tuple[training.TrainConfig | None, dict[str, object]]] = []
    loaded: list[Path] = []
    actor = tmp_path / "final-actor"

    def train(
        config: training.TrainConfig | None, **kwargs: object
    ) -> training.TrainResult:
        calls.append((config, kwargs))
        return training.TrainResult(
            tmp_path,
            actor,
            None,
            192,
            3,
            "complete",
            (),
            tmp_path / "checkpoints" / "final",
        )

    def load(path: Path) -> object:
        loaded.append(path)
        return object()

    def analyze(paths: list[Path], **_kwargs: object) -> dict[str, object]:
        return {"artifacts": {"summary": str(tmp_path / "summary.md")}}

    monkeypatch.setattr(
        sys, "argv", ["qmix_training.py", "--output-dir", str(tmp_path)]
    )
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(training, "load_system", load)
    monkeypatch.setattr(training, "analyze", analyze)
    qmix_example.main()
    config, kwargs = calls[0]
    assert config is not None and config.method == "qmix" and config.qmix is not None
    assert (config.num_envs, config.total_env_steps) == (4, 192)
    assert (config.qmix.rollout_length, config.qmix.min_buffer_size) == (8, 32)
    assert config.checkpoint_interval_updates == 1600
    assert kwargs == {"output_dir": tmp_path, "resume_from": None}
    assert loaded == [actor]


def test_pqn_example_runs_the_shared_public_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training

    calls: list[tuple[training.TrainConfig | None, dict[str, object]]] = []
    played: list[tuple[str, object]] = []
    loaded: list[Path] = []
    actor = tmp_path / "final-actor"

    def train(
        config: training.TrainConfig | None, **kwargs: object
    ) -> training.TrainResult:
        calls.append((config, kwargs))
        return training.TrainResult(
            tmp_path,
            actor,
            None,
            176,
            20,
            "complete",
            (),
            tmp_path / "checkpoints" / "final",
        )

    def load(path: Path) -> object:
        loaded.append(path)
        return "loaded"

    def evaluate(system: object, opponent: object, **_kwargs: object) -> None:
        played.append(("evaluate", opponent))

    def tournament(entrants: tuple[object, ...], **_kwargs: object) -> None:
        played.append(("tournament", entrants))

    def analyze(paths: list[Path], **_kwargs: object) -> dict[str, object]:
        return {"artifacts": {"summary": str(tmp_path / "summary.md")}}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pqn_training.py",
            "--output-dir",
            str(tmp_path),
            "--curriculum",
            "--shaping",
            "--evaluate",
        ],
    )
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(training, "load_system", load)
    monkeypatch.setattr(training, "analyze", analyze)
    monkeypatch.setattr(marl_bgs, "evaluate", evaluate)
    monkeypatch.setattr(marl_bgs, "run_tournament", tournament)
    pqn_example.main()
    config, kwargs = calls[0]
    assert config is not None and config.method == "pqn_vdn" and config.pqn is not None
    assert (config.num_envs, config.total_env_steps) == (4, 176)
    assert (config.pqn.rollout_length, config.pqn.memory_window) == (4, 2)
    assert config.curriculum and config.shaping
    assert config.checkpoint_interval_updates == 1600
    assert config.validation_opponents is None and config.validation_panel is None
    assert kwargs == {"output_dir": tmp_path, "resume_from": None}
    assert loaded == [actor]
    assert played == [
        ("evaluate", "tdm-alpha"),
        ("tournament", ("loaded", "tdm-alpha")),
    ]
    # Curriculum and shaping flags cannot silently vanish into a saved config.
    for extra in (
        (
            "--output-dir",
            str(tmp_path / "other"),
            "--config",
            "saved.json",
            "--shaping",
        ),
        ("--resume-from", str(tmp_path), "--curriculum"),
    ):
        monkeypatch.setattr(sys, "argv", ["pqn_training.py", *extra])
        with pytest.raises(SystemExit) as error:
            pqn_example.main()
        assert error.value.code == 2
    assert len(calls) == 1


def test_cross_play_saves_fixed_cells_and_resumes_with_live_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import csv
    import json
    from importlib import import_module

    from examples import cross_play_and_zsc as example

    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = evaluator.capture_recording_provenance(num_envs=2)

    def fixed_provenance(**kwargs: object) -> dict[str, object]:
        del kwargs
        return provenance

    # Concurrent repository edits must not change this test's saved revision.
    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    calls: list[np.ndarray] = []
    source_values = np.array([0.25], dtype=np.float32)

    def host(variables: Tree, memory: Tree, inputs: Tree, keys: Tree) -> Tree:
        np.testing.assert_array_equal(variables, [0.25])
        source_values[0] = 0.75
        calls.append(np.array(inputs.controlled_mask, copy=True))
        return example.host_idle(variables, memory, inputs, keys)

    host_member = marl_bgs.System(
        "Local Idle Host",
        host,
        variables=source_values,
        execution="host",
        checkpoint="local-idle-code-v1",
    )
    options: dict[str, Tree] = {
        "focals": {"first": "random", "second": "tdm-alpha"},
        "partners": {"jax": "random", "host": host_member},
        "opponents": {"first": "random", "second": "tdm-beta"},
        "familiarity": {"first": {"jax": "familiar", "host": "held_out"}},
        "max_steps": 1,
    }
    rows = example.run(tmp_path, **options)
    assert len(rows) == 16
    assert capsys.readouterr().out.count("Mean Point Margin:") == 8
    assert source_values[0] == 0.75
    source_values[0] = 0.25
    assert all("-" not in Path(row["run_path"]).parts[0] for row in rows)
    assert calls and all(np.all(value[:, :4] == 0) for value in calls)
    assert all(np.all(value[:, 4]) for value in calls)
    declaration = json.loads((tmp_path / "cross_play_settings.json").read_text())
    for focal in ("first", "second"):
        for partner in ("jax", "host"):
            for opponent in ("first", "second"):
                cell = [
                    row
                    for row in rows
                    if (row["focal"], row["partner"], row["opponent"])
                    == (focal, partner, opponent)
                ]
                assert {row["spawn_end"] for row in cell} == {"team_a", "team_b"}
                assert len({row["seed_id"] for row in cell}) == 1
                expected = (
                    ("familiar" if partner == "jax" else "held_out")
                    if focal == "first"
                    else "unknown"
                )
                assert {row["familiarity"] for row in cell} == {expected}
                saved = marl_bgs.load_results(tmp_path / cell[0]["run_path"])
                entry = next(iter(saved.metadata["passes"].values()))
                assert len(saved.table("episodes")["episode_id"]) == 2
                for row in cell:
                    assert row["team_a_system_id"] == entry["system_ids"]["team_a"]
                    assert row["team_b_system_id"] == entry["system_ids"]["team_b"]
                    assert row["partner_checkpoint"] == (
                        "local-idle-code-v1" if partner == "host" else None
                    )
                    assert row["focal_checkpoint"] is None
                    assert (
                        row["partner_id"]
                        == declaration["members"]["partners"][partner]["id"]
                    )
                    assert len(row["focal_id"]) == len(row["partner_id"]) == 64
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    count = len(calls)
    assert example.run(tmp_path, **options) == rows
    assert len(calls) == count
    assert all(path.read_bytes() == content for path, content in before.items())
    with (tmp_path / "games.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 16
    options["familiarity"] = {"first": {"host": "familiar"}}
    with pytest.raises(ValueError, match="declaration changed"):
        example.run(tmp_path, **options)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_cross_play_cli_forwards_explicit_references_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples import cross_play_and_zsc as example

    observed: dict[str, Tree] = {}

    def run(output_dir: Path, **kwargs: Tree) -> list[dict[str, Tree]]:
        observed.update(output_dir=output_dir, **kwargs)
        return []

    monkeypatch.setattr(example, "run", run)
    assert (
        example.main(
            [
                "--output-dir",
                str(tmp_path),
                "--focal",
                "saved/actor",
                "--partner",
                "tdm-alpha",
                "--opponent",
                "tdm-beta",
                "--max-steps",
                "300",
                "--seed-pairs",
                "8",
                "--maps",
                "47",
                "48",
                "--num-envs",
                "32",
            ]
        )
        == 0
    )
    assert observed["focals"] == {"saved/actor": "saved/actor"}
    assert observed["partners"] == {"tdm-alpha": "tdm-alpha"}
    assert observed["opponents"] == {"tdm-beta": "tdm-beta"}
    assert observed["max_steps"] == 300 and observed["seed_pairs"] == 8
    assert observed["maps"] == [47, 48] and observed["num_envs"] == 32


@pytest.fixture(scope="module")
def own_ctde_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return own_learner.train(
        tmp_path_factory.mktemp("own_ctde") / "run",
        updates=2,
        rollout_length=2,
        num_envs=2,
        max_steps=3,
        custom_reward=True,
        fade_rounds=4,
    )


def test_own_ctde_updates_actor_and_critic_and_reloads_exact_inference(
    own_ctde_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert own_ctde_run["env_steps"] == 8
    for initial, final in zip(
        own_ctde_run["initial"], own_ctde_run["parameters"], strict=True
    ):
        assert any(
            not np.array_equal(left, right)
            for left, right in zip(
                jax.tree.leaves(initial), jax.tree.leaves(final), strict=True
            )
        )
        assert all(np.all(np.isfinite(value)) for value in jax.tree.leaves(final))
    assert all(
        row["same_call_log_probability_error"] < 2e-6 for row in own_ctde_run["metrics"]
    )
    path = own_ctde_run["snapshots"][-1]
    monkeypatch.setenv("MARL_OWN_ACTOR", str(path))
    saved = own_learner.load_selected()
    current = own_learner.actor(own_ctde_run["parameters"][0])
    env = marl_bgs.make("tdm", map_id=0, num_envs=2, max_steps=2, metrics="none")
    observations, state = env.reset(jax.random.key(801))
    opponent = marl_bgs.shared_policy(marl_bgs.policy("random"))
    outputs = []
    for method in (current, saved):
        memory = marl_bgs.init_systems(
            method, opponent, observations, state, jax.random.key(802)
        )
        outputs.append(
            marl_bgs.apply_systems(
                method, opponent, memory, observations, state, jax.random.key(803)
            )
        )
    _equal(outputs[0], outputs[1])
    inputs = env.policy_inputs(observations, state)
    result = own_learner.act(
        current.variables, (), inputs, jax.random.split(jax.random.key(804), 2)
    )
    paired = 2 * result.actions.select_target + result.actions.use_ultimate
    assert np.all(
        jnp.take_along_axis(
            inputs.action_mask.move_mask, result.actions.move[..., None], -1
        )
    )
    assert np.all(
        jnp.take_along_axis(
            inputs.action_mask.select_target_use_ultimate_joint_mask.reshape(
                (2, 5, 22)
            ),
            paired[..., None],
            -1,
        )
    )

    def clear_other_recipients(value: jax.Array) -> jax.Array:
        return value.at[:, 1:].set(jnp.zeros_like(value[:, 1:]))

    changed = inputs._replace(
        actors=jax.tree.map(clear_other_recipients, inputs.actors)
    )
    other = own_learner.act(
        current.variables, (), changed, jax.random.split(jax.random.key(804), 2)
    )
    for first, second in zip(
        jax.tree.leaves(result.actions), jax.tree.leaves(other.actions), strict=True
    ):
        np.testing.assert_array_equal(first[:, 0], second[:, 0])
    np.testing.assert_array_equal(
        result.learning_outputs[:, 0], other.learning_outputs[:, 0]
    )
    before = (path / "actor_weights.npz").read_bytes()
    with pytest.raises(FileExistsError):
        own_learner.save_actor(path, current.variables, env_steps=8)
    assert (path / "actor_weights.npz").read_bytes() == before


def test_own_ctde_bootstraps_cutoffs_but_not_native_horizon_draws() -> None:
    env = marl_bgs.make("tdm", map_id=0, num_envs=2, max_steps=2, metrics="none")
    _, state = env.reset(jax.random.key(811))
    outcomes: list[jax.Array] = []
    for index in range(2):
        actions = env.sample_actions(jax.random.key(812 + index), state)
        _, state, _, done, _ = env.step(jax.random.key(814 + index), state, actions)
        outcomes.append(
            own_learner.td_targets(jnp.ones(2), jnp.full(2, 10.0), done.done)
        )
    np.testing.assert_allclose(outcomes[0], 10.9)
    np.testing.assert_array_equal(outcomes[1], 1)


def test_own_ctde_finishes_public_validation_selection_and_evaluation(
    own_ctde_run: dict[str, Any], tmp_path: Path
) -> None:
    selected = own_learner.validate_and_evaluate(
        own_ctde_run["snapshots"], tmp_path / "validation", num_envs=2
    )
    assert selected["selection_rule"] == "point_margin"
    assert Path(selected["selected_actor"]) in own_ctde_run["snapshots"]
    result = marl_bgs.load_results(selected["evaluation_run"])
    assert result.status == "complete"
    assert set(result.table("episodes")["map_id"]) == {47}
    assert len(result.table("episodes")["episode_id"]) == 2


def test_own_ctde_cli_keeps_training_batch_separate_from_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, int] = {}

    def train(output_dir: Path, **options: object) -> dict[str, Any]:
        observed["training"] = cast(int, options["num_envs"])
        return {"snapshots": [output_dir / "actor"], "env_steps": 8192}

    def evaluate(
        snapshots: list[Path], output_dir: Path, *, num_envs: int
    ) -> dict[str, str]:
        del snapshots, output_dir
        observed["validation"] = num_envs
        return {"selected_actor": "actor"}

    monkeypatch.setattr(own_learner, "train", train)
    monkeypatch.setattr(own_learner, "validate_and_evaluate", evaluate)
    monkeypatch.setattr(
        sys, "argv", ["own_learner", "--output-dir", str(tmp_path), "--num-envs", "128"]
    )
    own_learner.main()
    assert observed == {"training": 128, "validation": 32}


def test_declared_curriculum_example_trains_loads_and_evaluates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from examples import curriculum as example

    from marl_battlegrounds.training import checkpoints

    run_dir = tmp_path / "curriculum"
    monkeypatch.setattr(sys, "argv", ["curriculum.py", "--output-dir", str(run_dir)])
    example.main()
    printed = capsys.readouterr().out
    details = json.loads((run_dir / "run_details.json").read_text())
    assert details["config"]["total_env_steps"] == 32
    assert details["config"]["num_envs"] == 4
    assert details["config"]["keep_past"] == 0
    stages = details["config"]["curriculum"]
    assert stages[0]["rosters"] == {"system": ["mage", "mage"], "opponent": ["warrior"]}
    assert stages[1]["rosters"] == {
        "system": ["mage", "mage", "priest"],
        "opponent": ["warrior", "hunter"],
    }
    assert details["schedule"]["stage_maps"] == [[0], [1, 2, 3]]
    assert details["schedule"]["score_thresholds"] == [5, 20]
    assert details["schedule"]["assigned_round_counts"] == [4, 4]
    exposure = json.loads((run_dir / "exposure.json").read_text())
    steps, starts = (
        exposure["steps_by_episode_stage"],
        exposure["starts_by_episode_stage"],
    )
    assert sum(steps) == exposure["env_steps"] == 32
    assert sum(starts) >= 4
    assert f"Actual Transitions By Stage: {steps}" in printed
    assert f"Actual Game Starts By Stage: {starts}" in printed
    actor = Path(
        next(line[7:] for line in printed.splitlines() if line.startswith("Actor: "))
    )
    saved = checkpoints.read_checkpoint_description(actor)
    assert saved["metadata"]["env_steps"] == 32
    assert actor.parent == run_dir / "actors"
    evaluation_dir = Path(
        next(
            line[12:]
            for line in printed.splitlines()
            if line.startswith("Evaluation: ")
        )
    )
    assert evaluation_dir.parent == run_dir / "curriculum_evaluation"
    evaluated = marl_bgs.load_results(evaluation_dir)
    assert evaluated.status == "complete"
    episodes = evaluated.table("episodes")
    assert len(episodes["episode_id"]) == 4
    assert set(episodes["map_id"]) == {42}
    saved_pass = next(iter(evaluated.metadata["passes"].values()))
    assert sorted(
        episode["spawn_locations"] for episode in saved_pass["episodes"].values()
    ) == [0, 0, 1, 1]
