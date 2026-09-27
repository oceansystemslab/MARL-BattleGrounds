"""Check learner ownership and custom rewards through real training blocks.

Frozen partners act in physical slots while all six built-in learners update
only their assigned slots. Native reward, shaping and custom adjustments keep
their own meanings. Compact Q rows retain ownership without storing duplicate
custom rewards. PPO ignores partner values and action probabilities in every
learning statistic. The public partner example trains repeated-class rosters,
selects an actor through deployed-team validation, reloads it and evaluates
each frozen partner as a separate composed team. Its software fixture narrows
validation to one map while keeping real 300-step games; the runnable example
keeps all five development maps. CPU checks prove integration, not learned
behavior or speed.
"""

# pyright: reportPrivateUsage=false
import json
from collections.abc import Callable
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.baselines import ppo, pqn, qmix
from marl_battlegrounds.core.types import EnvState
from marl_battlegrounds.environment import TrainingFacts
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    learner,
    make_training_schedule,
    pqn_learner,
    prepare_training_content,
    qmix_learner,
    scan_training_rollout,
)
from marl_battlegrounds.training.collection import learner_memory

type Tree = Any


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


def _reward(
    before: EnvState, facts: TrainingFacts, after: EnvState, progress: Array
) -> Array:
    del before, facts, after, progress
    return jnp.arange(1, 11, dtype=jnp.float32)


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "method", ("mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn")
)
def test_partner_slots_keep_physical_inputs_but_only_owned_slots_learn(
    prepared: PreparedTrainingContent, method: str
) -> None:
    schedule = make_training_schedule(
        total_env_steps=20,
        num_envs=2,
        curriculum=[
            {
                "share": 1.0,
                "maps": [0],
                "rosters": {
                    "system": ["mage", "mage", "priest"],
                    "opponent": ["warrior", "hunter", "priest"],
                },
            }
        ],
    )
    options: dict[str, Any] = dict(
        schedule=schedule,
        prepared=prepared,
        seed=42,
        metrics="none",
        keep_past=0,
        history_capture_capacity=0,
        learner_slots=(0, 2),
        partner_population={"fixed": "random"},
        partner_selection={"fixed": 1.0},
        reward=_reward,
        reward_identity={"test": "slot rewards"},
    )
    settings: Tree
    state: Tree
    updater: Callable[..., Tree]
    validate: Callable[..., None]
    if method == "qmix":
        settings = qmix.QMIXConfig(
            rollout_length=2,
            buffer_size=4,
            min_buffer_size=2,
            sample_sequence_length=2,
            sample_batch_size=2,
            epochs=1,
            parameter_sharing="none",
        )
        collection, state = qmix_learner.init_qmix_learner(**options, qmix=settings)
        updater = partial(qmix_learner.update_qmix_learner, qmix=settings)
        validate = partial(qmix_learner.validate_qmix_learner, qmix=settings)
    elif method == "pqn_vdn":
        settings = pqn.PQNConfig(
            rollout_length=2,
            memory_window=1,
            epochs=1,
            num_minibatches=1,
            parameter_sharing="none",
        )
        collection, state = pqn_learner.init_pqn_learner(**options, pqn=settings)
        updater = partial(
            pqn_learner.update_pqn_learner,
            pqn=settings,
            planned_learning_blocks=pqn.pqn_planned_learning_blocks(10, settings),
        )
        validate = partial(pqn_learner.validate_pqn_learner, pqn=settings)
    else:
        settings = ppo.PPOConfig(
            rollout_length=2,
            epochs=1,
            minibatches=1,
            groups=1,
            parameter_sharing="none",
        )
        collection, state = learner.init_learner(**options, ppo=settings, method=method)
        updater = partial(learner.update_learner, ppo=settings, method=method)
        validate = partial(learner.validate_learner, ppo=settings, method=method)
    validate(collection, state)
    partners = state.carry.partner_values
    update = cast(Callable[..., Tree], jax.jit(updater))
    for length in (2, 1, 2) if method == "pqn_vdn" else (2,):
        collect = cast(
            Callable[..., Tree],
            jax.jit(partial(scan_training_rollout, collection, length=length)),
        )
        carry, rollout = collect(state.carry)
        rows = rollout.transitions
        expected_owned = np.broadcast_to(
            np.asarray([True, False, True, False, False]), rows.active.shape
        )
        np.testing.assert_array_equal(rows.learner_active, expected_owned)
        assert np.asarray(rows.active)[..., 1].all()
        _equal(rollout.initial_memory, learner_memory(state.carry))
        assert rows.custom_rewards is not None
        summary = learner._summary(rollout, carry.progress.opponent_steps.shape[0], 0)
        assert int(summary.active_samples) == length * 2 * 2
        assert summary.custom_reward_sum is not None
        assert float(summary.custom_reward_sum) == length * 2 * 4
        assert float(summary.task_reward_sum) == float(
            jnp.sum(jnp.where(rows.learner_active, rows.task_rewards, 0.0))
        )
        if method in {"qmix", "pqn_vdn"}:
            adjusted = rows._replace(
                task_rewards=jnp.where(rows.active, jnp.float32(0.25), 0.0),
                shaping_reward=jnp.full(rows.valid.shape, 0.75, jnp.float32),
            )
            if method == "qmix":
                compact = qmix_learner._replay_rows(adjusted)

                def swap(value: Array) -> Array:
                    return value.swapaxes(0, 1)

                sample = jax.tree.map(swap, compact)
                batch = qmix_learner._expand_sample(sample, settings)
            else:
                compact, memory = pqn_learner._pqn_rows(adjusted)
                batch = pqn_learner._expand_minibatch(compact, memory[0], settings)
            np.testing.assert_array_equal(compact.learner_active, rows.learner_active)
            np.testing.assert_allclose(compact.task_reward, 2.25)
            np.testing.assert_allclose(batch.rewards, 3.0)
            assert batch.learner_active is not None
        else:
            batch, _ = learner.build_ppo_batch(
                state, rollout, ppo=settings, method=method
            )
            assert np.asarray(batch.rewards)[~expected_owned].sum() == 0
            assert np.asarray(batch.final_values)[~expected_owned[-1]].sum() == 0
            assert bool(learner._behavior_valid(rollout))
            if method == "ff_ippo":
                _check_ppo_partner_independence(state, batch, settings, method)
                outputs = rows.learning_outputs._replace(
                    log_prob=jnp.where(
                        rows.active & ~rows.learner_active,
                        jnp.nan,
                        rows.learning_outputs.log_prob,
                    ),
                    action_indices=jnp.where(
                        rows.active & ~rows.learner_active,
                        2**20,
                        rows.learning_outputs.action_indices,
                    ),
                )
                rollout = rollout._replace(
                    transitions=rows._replace(learning_outputs=outputs)
                )
                assert bool(learner._behavior_valid(rollout))
        state, result = update(state, carry, rollout)
        assert not bool(result.failed), int(result.failure_reason)
        if method not in {"qmix", "pqn_vdn"}:
            assert int(result.metrics.actor_samples[0, 0]) == 8
            assert int(result.metrics.critic_samples[0, 0]) == 8
        assert float(result.summary.custom_reward_sum) == length * 2 * 4
    assert int(state.completed_updates) == 1
    validate(collection, state)
    _equal(state.carry.partner_values, partners)
    opt = (
        state.opt_state[0]
        if method == "qmix"
        else state.opt_state
        if method == "pqn_vdn"
        else state.actor_opt_state
    )
    counts = [
        value
        for value in jax.tree.leaves(opt)
        if jnp.issubdtype(value.dtype, jnp.integer)
    ]
    assert counts
    for value in counts:
        np.testing.assert_array_equal(value, [1, 0, 1, 0, 0])


def _check_ppo_partner_independence(
    state: learner.LearnerState,
    batch: ppo.PPOBatch,
    settings: ppo.PPOConfig,
    method: str,
) -> None:
    assert batch.learner_active is not None
    assert batch.old_normalized_values is not None
    numerical = learner._ppo_state(state)
    update = cast(
        Callable[..., Tree],
        jax.jit(partial(ppo.update_ppo, config=settings, method=method)),
    )
    key = jax.random.key(914)
    wanted = update(numerical, batch, key)
    partner = batch.active & ~batch.learner_active
    changed = batch._replace(
        actions=jnp.where(partner, 2**20, batch.actions),
        rewards=jnp.where(partner, 100.0, batch.rewards),
        old_log_prob=jnp.where(partner, -1e6, batch.old_log_prob),
        old_values=jnp.where(partner, 1e30, batch.old_values),
        old_normalized_values=jnp.where(partner, 30.0, batch.old_normalized_values),
        final_values=jnp.where(partner[-1], 60.0, batch.final_values),
    )
    _equal(update(numerical, changed, key), wanted)
    poisoned = changed._replace(
        old_log_prob=jnp.where(partner, jnp.nan, batch.old_log_prob),
        rewards=jnp.where(partner, jnp.inf, batch.rewards),
        old_values=jnp.where(partner, jnp.nan, batch.old_values),
        old_normalized_values=jnp.where(partner, jnp.inf, batch.old_normalized_values),
        final_values=jnp.where(partner[-1], jnp.nan, batch.final_values),
    )
    _equal(update(numerical, poisoned, key), wanted)
    assert bool(learner._finite(ppo._owned_ppo_batch(poisoned)))
    empty = batch._replace(learner_active=jnp.zeros_like(batch.learner_active))
    unchanged, metrics = update(numerical, empty, key)
    _equal(unchanged, numerical)
    assert not np.asarray(metrics.actor_samples).any()
    assert not np.asarray(metrics.critic_samples).any()


@pytest.mark.parametrize(
    ("method", "sharing"),
    [
        ("mappo", "all"),
        ("ippo", "class"),
        ("ff_mappo", "none"),
        ("ff_ippo", "none"),
        ("qmix", "class"),
        ("pqn_vdn", "all"),
    ],
)
def test_public_partner_example_reassembles_saved_actor_with_every_partner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, sharing: str
) -> None:
    from examples import partner_training
    from tests.training_continuation_helpers import capture_runs, fixed_source
    from tests.training_learner_helpers import equal

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.training import checkpoints, load_system, train, validation

    fixed_source(monkeypatch, method)
    provenance = capture_recording_provenance(num_envs=2)

    def fixed_provenance(**kwargs: object) -> dict[str, object]:
        del kwargs
        return provenance

    monkeypatch.setattr(
        import_module("marl_battlegrounds.evaluation.evaluate"),
        "capture_recording_provenance",
        fixed_provenance,
    )
    monkeypatch.setattr(validation, "VALIDATION_MAPS", (42,))
    starts, ends = capture_runs(monkeypatch)
    root = tmp_path / "partners"
    report = partner_training.run(root, method=method, parameter_sharing=sharing)
    assert report["completed_env_steps"] == (10 if method == "pqn_vdn" else 4)
    assert report["completed_updates"] == 1
    assert report["learner_slots"] == [0, 2]
    saved = checkpoints.read_checkpoint_details(report["full_checkpoint"])
    assert saved["collection"]["learner_slots"] == [0, 2]
    assert [row["name"] for row in saved["collection"]["partner_members"]] == [
        "scripted",
        "custom",
    ]
    assert saved["collection"]["partner_members"][1].get("reference") is None
    config = saved["metadata"]["config"]
    family = "qmix" if method == "qmix" else "pqn" if method == "pqn_vdn" else "ppo"
    assert config[family]["parameter_sharing"] == sharing
    assert config["curriculum"][0]["rosters"] == report["rosters"]
    deployment = saved["metadata"]["validation_deployment"]
    assert deployment["system_roster"] == report["rosters"]["system"]
    assert deployment["opponent_roster"] == report["rosters"]["opponent"]
    assert deployment["learner_slots"] == [0, 2]
    initial = starts[root / "training"][1]
    trained = ends[root / "training"]
    _equal(initial.carry.partner_values, trained.carry.partner_values)
    opt = (
        trained.opt_state[0]
        if method == "qmix"
        else trained.opt_state
        if method in {"qmix", "pqn_vdn"}
        else trained.actor_opt_state
    )
    expected = (
        1
        if sharing == "all"
        else [1, 0, 0, 0, 0]
        if sharing == "class"
        else [1, 0, 1, 0, 0]
    )
    counts = [
        leaf for leaf in jax.tree.leaves(opt) if jnp.issubdtype(leaf.dtype, jnp.integer)
    ]
    assert counts
    for count in counts:
        np.testing.assert_array_equal(count, expected)
    selected = json.loads(Path(report["selection"]).read_text())
    identity = checkpoints.artifact_identity(report["actor"])
    assert selected["checkpoint_id"] == identity["metadata"]["checkpoint_id"]
    assert selected["actor_digest"] == identity["actor_digest"]
    assert selected["env_steps"] == report["completed_env_steps"]
    assert selected["purpose"] == "confirmation" and selected["complete"]
    assert [row["name"] for row in selected["partner_results"]] == [
        "scripted",
        "custom",
    ]
    assert all(row["label"] == "familiar" for row in selected["partner_results"])
    actor = load_system(report["actor"])
    assert [row["training_partner"] for row in report["evaluations"]] == [
        "scripted",
        "custom",
    ]
    identities: set[str] = set()
    for row in report["evaluations"]:
        games = marl_bgs.load_results(row["run_dir"])
        assert games.status == "complete"
        coverage = games.metadata["spawn_balance"]
        assert coverage["paired_complete"]
        identities.add(coverage["system_id"])
        assert len(games.table("episodes")["episode_id"]) == 2
        assert not games.table("priority_metrics")
        details = json.loads((Path(row["run_dir"]) / "run_details.json").read_text())
        components = details["systems"][coverage["system_id"]]["components"]
        assert components[0]["checkpoint"] == actor.checkpoint
        for configured in details["configurations"].values():
            assert configured["agent_profile"]["class_ids"] == [
                1,
                2,
                1,
                5,
                3,
                2,
                1,
                2,
                3,
                5,
            ]
    assert len(identities) == 2
    resumed = train(
        resume_from=report["full_checkpoint"],
        partners={"scripted": "tdm-alpha", "custom": partner_training.load_idle()},
    )
    assert resumed.selected_actor == Path(report["actor"])
    assert resumed.completed_updates == report["completed_updates"]
    equal(ends[root / "training"], trained)
