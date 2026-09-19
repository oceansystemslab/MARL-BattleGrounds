"""Check sampled public decisions, reset ownership and recording continuation.

Synthetic banks change only horizons and are labelled separately from the verified
canonical binding. Tests join the real public choose/step/track/reset path to its
preselected control, bounded collection and writer recovery. Compact declarations
must describe the producing episode; provenance never becomes actor information.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples import training_distributions as example

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    prepare_training_content,
)

type Tree = Any


def _jit(function: Callable[..., Tree]) -> Tree:
    return cast(Tree, jax.jit(function))


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def context(prepared: PreparedTrainingContent) -> example.Context:
    bank = prepared.source_configs._replace(
        max_steps=jnp.where(jnp.arange(42) % 2 == 0, 2, 5).astype(jnp.int32)
    )
    return example.make_context(
        prepared=prepared,
        source_configs=bank,
        num_envs=32,
        recording=True,
        eligible_maps=jnp.arange(42) < 2,
    )


@pytest.fixture(scope="module")
def step(context: example.Context) -> Tree:
    return example.make_step(context.actor, context.opponent)


@pytest.fixture(scope="module")
def trajectory(context: example.Context, step: Tree) -> Tree:
    def scan(carry: example.Carry) -> Tree:
        return jax.lax.scan(step, carry, None, length=6)

    return _jit(scan)(context.carry)


def _equal(actual: Tree, expected: Tree) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_b32_public_trajectory_keeps_exact_first_transition_declarations(
    context: example.Context, trajectory: Tree
) -> None:
    carry, (transitions, info, trace) = trajectory
    assert context.source_bank_is_synthetic
    assert np.all(np.asarray(transitions.advanced))
    assert np.any(np.asarray(transitions.completed))
    assert np.any(np.asarray(carry.state.reset_generation) > 0)
    np.testing.assert_array_equal(
        carry.state.cumulative_transition_count, np.full(32, 6)
    )
    np.testing.assert_array_equal(
        carry.tracking.stage_counts[:, :2].sum(axis=0), (96, 96)
    )
    assert not np.any(np.asarray(carry.tracking.error_flags))
    starts = info.episode_start_records
    np.testing.assert_array_equal(starts.valid, info.decision_step == 0)
    np.testing.assert_array_equal(
        starts.source_class_ids[starts.valid],
        info.config.agent_profile.class_ids[starts.valid],
    )
    np.testing.assert_array_equal(trace.episode_id, info.episode_id)
    np.testing.assert_array_equal(trace.decision_step, info.decision_step)
    assert np.any(np.asarray(transitions.actions.move) != 0)
    assert np.any(
        np.asarray(carry.source_class_ids) != np.asarray(context.carry.source_class_ids)
    )
    _equal(carry.tracking.source_configs, context.carry.tracking.source_configs)


def test_preselected_configs_and_keys_match_the_same_public_loop(
    context: example.Context, step: Tree, trajectory: Tree
) -> None:
    def record(current: example.Carry, unused: None) -> Tree:
        del unused
        inputs = example.decision_inputs(current)
        latest, _ = step(current, inputs)
        return latest, inputs

    def prepare(carry: example.Carry) -> Tree:
        return jax.lax.scan(record, carry, None, length=6)

    def replay(carry: example.Carry, inputs: example.DecisionInputs) -> Tree:
        return jax.lax.scan(step, carry, inputs)

    _, tape = _jit(prepare)(context.carry)
    actual = _jit(replay)(context.carry, tape)
    _equal(actual, trajectory)


def test_partial_reset_keeps_continuing_state_and_current_declarations(
    context: example.Context, step: Tree
) -> None:
    compiled = _jit(step)
    before, _ = compiled(context.carry, None)
    after, (transition, info, _) = compiled(before, None)
    reset = np.asarray(info.completed)
    assert reset.any() and not reset.all()
    keys = example.decision_inputs(before)
    _, unreset_state, _, _, _ = _jit(before.env.step)(
        keys.step_keys, before.state, transition.actions
    )
    for actual, expected in zip(
        jax.tree.leaves(after.state), jax.tree.leaves(unreset_state), strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(actual)[~reset], np.asarray(expected)[~reset]
        )
    np.testing.assert_array_equal(
        after.source_indices[~reset], before.source_indices[~reset]
    )
    np.testing.assert_array_equal(
        after.source_class_ids[~reset], before.source_class_ids[~reset]
    )
    np.testing.assert_array_equal(after.state.reset_generation[reset], 1)
    np.testing.assert_array_equal(
        after.memory.reset_generation, before.state.reset_generation
    )


def test_provenance_changes_do_not_change_actor_choices(
    context: example.Context, step: Tree
) -> None:
    changed = context.carry._replace(
        tracking=replace(
            context.carry.tracking, source_table_id=jnp.arange(8, dtype=jnp.uint32)
        )
    )
    compiled = _jit(step)
    first, (first_output, _, _) = compiled(context.carry, None)
    second, (second_output, _, _) = compiled(changed, None)
    _equal(first_output.actions, second_output.actions)
    _equal(first.memory, second.memory)


def test_direct_scan_and_bounded_collection_match_all_records(
    context: example.Context, step: Tree, trajectory: Tree, tmp_path: Path
) -> None:
    expected, (transitions, _, _) = trajectory
    systems: dict[str, object] = {"team_a": context.actor, "team_b": context.opponent}
    with marl_bgs.RunWriter(tmp_path, phase="training", policies=systems) as writer:
        actual, retained = marl_bgs.collect_rollout(
            step,
            context.carry,
            num_steps=6,
            writer=writer,
            record_capacity=32,
            source_configs=context.carry.tracking.source_configs,
        )
        run_dir = writer.run_dir
    _equal(actual, expected)
    _equal(retained, transitions)
    details = json.loads((run_dir / "run_details.json").read_text())
    starts = next(iter(details["passes"].values()))["episode_starts"]
    assert starts
    for row in starts.values():
        assert len(row["source_class_ids"]) == 10
        assert row["resolved_config_id"] in details["configurations"]


def test_recording_rewind_keeps_live_replay_and_exact_numerical_suffix(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    bank = prepared.source_configs._replace(max_steps=jnp.full(42, 3, jnp.int32))
    context = example.make_context(
        prepared=prepared,
        source_configs=bank,
        num_envs=2,
        recording=True,
        replay_episodes=(1,),
        eligible_maps=jnp.arange(42) < 2,
    )
    step = example.make_step(context.actor, context.opponent)
    systems: dict[str, object] = {"team_a": context.actor, "team_b": context.opponent}
    with marl_bgs.RunWriter(tmp_path, phase="training", policies=systems) as writer:
        midpoint, _ = marl_bgs.collect_rollout(
            step,
            context.carry,
            num_steps=1,
            writer=writer,
            source_configs=bank,
        )
        saved = example.save_continuation(
            context, midpoint, writer.checkpoint_recording()
        )
        run_dir = writer.run_dir
        expected, outputs = marl_bgs.collect_rollout(
            step,
            midpoint,
            num_steps=4,
            writer=writer,
            source_configs=bank,
        )
    recorded = {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".jsonl"}
    }
    restored = example.restore_continuation(saved)
    _equal(restored.carry, midpoint)
    with marl_bgs.RunWriter(
        resume_from=run_dir,
        recording_checkpoint=saved.recording_token,
        phase="training",
        policies=systems,
    ) as writer:
        actual, retained = marl_bgs.collect_rollout(
            step,
            restored.carry,
            num_steps=4,
            writer=writer,
            source_configs=bank,
        )
    _equal((actual, retained), (expected, outputs))
    for name, contents in recorded.items():
        assert (run_dir / name).read_bytes() == contents


def test_failed_saved_binding_stops_before_writer_recovery(
    context: example.Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = example.save_continuation(context, context.carry)
    bad_binding = context.binding.model_copy(update={"canonical_digest": "0" * 64})
    saved = replace(saved, context=replace(context, binding=bad_binding))
    recording = tmp_path / "existing.csv"
    recording.write_bytes(b"Preserved durable data\n")
    calls = []

    def writer(**_kwargs: object) -> None:
        calls.append("writer")
        recording.write_bytes(b"Wrongly rewound")

    monkeypatch.setattr(marl_bgs, "RunWriter", writer)
    with pytest.raises(ValueError, match="digest"):
        restored = example.restore_continuation(saved)
        marl_bgs.RunWriter(resume_from=tmp_path, details={"restored": restored})
    assert calls == []
    assert recording.read_bytes() == b"Preserved durable data\n"


def test_disabled_recording_and_untrained_mappo_use_the_public_path(
    prepared: PreparedTrainingContent,
) -> None:
    context = example.make_context(
        prepared=prepared, num_envs=2, mappo=True, metrics="none"
    )
    step = example.make_step(context.actor, context.opponent)
    traces: list[int] = []

    def rollout(carry: example.Carry) -> Tree:
        traces.append(1)
        return jax.lax.scan(step, carry, None, length=2)

    compiled = _jit(rollout)
    actual, (transition, info, _) = compiled(context.carry)
    assert info.episode_start_records is None
    assert info.replay is None
    assert context.carry.tracking.source_table_id is None
    assert np.shape(actual.memory.team_a) == (2, 5, 128)
    assert transition.learning.action_indices.shape == (2, 2, 5)
    assert np.all(np.isfinite(np.asarray(transition.learning.log_prob)))
    assert np.any(np.asarray(actual.memory.team_a) != 0)

    def change_parameter(value: jax.Array) -> jax.Array:
        return value + jnp.asarray(0.0002, value.dtype)

    changed = actual._replace(
        variables_a=jax.tree.map(change_parameter, actual.variables_a),
        memory=actual.memory._replace(team_a=actual.memory.team_a + jnp.float32(0.125)),
        root_key=jax.random.key(43),
        eligible_maps=jnp.arange(42) < 3,
        team_size=jnp.asarray(2, jnp.int32),
    )
    example.validate_training_distribution(
        eligible_maps=changed.eligible_maps, team_size=changed.team_size
    )
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree.leaves(context.carry.observations),
            jax.tree.leaves(changed.observations),
            strict=True,
        )
    )
    continuing, _ = compiled(changed)
    bank = prepared.source_configs._replace(max_steps=jnp.full(42, 2, jnp.int32))
    synthetic = example.make_context(
        prepared=prepared,
        source_configs=bank,
        seed=44,
        num_envs=2,
        team_size=2,
        eligible_maps=jnp.arange(42) < 3,
        mappo=True,
        metrics="none",
    )
    reset, _ = compiled(synthetic.carry)
    jax.block_until_ready((continuing, reset))
    assert len(traces) == 1
    np.testing.assert_array_equal(continuing.state.cumulative_transition_count, 4)
    np.testing.assert_array_equal(reset.state.reset_generation, 1)
    assert not np.any(np.asarray(continuing.tracking.error_flags))
    assert not np.any(np.asarray(reset.tracking.error_flags))


def test_example_rejects_unadmitted_source_rules(
    prepared: PreparedTrainingContent,
) -> None:
    bank = prepared.source_configs._replace(
        team_deathmatch_score_threshold=prepared.source_configs.team_deathmatch_score_threshold
        + 1
    )
    with pytest.raises(ValueError, match="only episode horizons"):
        example.make_context(prepared=prepared, source_configs=bank, num_envs=2)
