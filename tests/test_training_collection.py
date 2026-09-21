"""Check compact training collection through public environment and System calls.

CPU cases cover exact prefixes, deferred resets, stage and update boundaries,
same-call MAPPO data, finite learner padding, optional physical/priority output,
recording recovery and failure guards. Short-horizon banks are explicit test
fixtures created after canonical admission; they are never presented as verified
training content. No case trains a learner or establishes GPU throughput.
Both shaping modes use the producing game's scores. Disabled shaping traces
neither reward helper, and invalid modes fail during setup. A zero pinned
opponent share traces the same rollout program as today; a positive share needs
early history capture, pins the first-update actor in slot 0 and keeps the
21-row exposure layout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.test_baseline_ppo import (
    _actor_forward,  # pyright: ignore[reportPrivateUsage]
)

import marl_battlegrounds as marl_bgs
import marl_battlegrounds.training.collection as collection_module
from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    action_entropy,
    action_log_prob,
    categorical_action_mask,
    encode_actions,
)
from marl_battlegrounds.baselines.inputs import encode_training_state
from marl_battlegrounds.baselines.ppo import initialize_ppo, make_recurrent_mappo_system
from marl_battlegrounds.core import env as core
from marl_battlegrounds.environment import Environment, EnvironmentState, EpisodeInfo
from marl_battlegrounds.episode_tracking import init_episode_tracking
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTrace,
    System,
    SystemInput,
    SystemOutput,
    init_systems,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    TrainingTransition,
    advance_training_step,
    collect_training_rollout,
    init_training_collection,
    make_training_schedule,
    prepare_training_content,
    refresh_opponents,
    sample_training_configs,
    scan_training_rollout,
    training_keys,
    training_summary,
)

type Tree = Any
type Context = tuple[TrainingCollection, TrainingCarry]
type StepResult = tuple[
    TrainingCarry, tuple[TrainingTransition, EpisodeInfo, PolicyTrace]
]


def _jit[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    return cast(Callable[P, R], jax.jit(function))


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def _initial(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del keys
    return jnp.full(inputs.active_mask.shape, variables["value"], jnp.float32)


def _apply(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    del keys
    zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    updated = memory + inputs.valid[:, None]
    return SystemOutput(
        ActorAction(zero, zero, zero),
        updated,
        learning_outputs={
            "calls": updated,
            "weight": jnp.full(inputs.active_mask.shape, variables["value"]),
        },
    )


def _actor() -> System:
    return System(
        "Collection Counter", _apply, init=_initial, variables={"value": jnp.float32(3)}
    )


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def plain(prepared: PreparedTrainingContent) -> Context:
    actor = _actor()
    return init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(total_env_steps=8, num_envs=2),
        prepared=prepared,
        metrics="none",
    )


def _synthetic(context: Context, *, horizon: int = 2, partial: bool = False) -> Context:
    collection, carry = context
    assert carry.tracking.source_configs is not None
    # Admission above used the verified installed bank. This explicit fixture
    # changes horizons only; it is not passed back as PreparedTrainingContent.
    limits = jnp.full(42, horizon, jnp.int32)
    if partial:
        assert int(carry.source_indices[0]) != int(carry.source_indices[1])
        limits = limits.at[carry.source_indices[0]].set(1)
    bank = carry.tracking.source_configs._replace(max_steps=limits)
    generations = jnp.zeros(2, jnp.int32)
    sampled = sample_training_configs(
        bank,
        carry.root_key,
        generations,
        eligible_maps=carry.schedule.eligible_maps[0],
        team_size=carry.schedule.team_sizes[0],
    )
    env = marl_bgs.make(
        "tdm",
        env_config=sampled.config,
        num_envs=2,
        metrics=cast(Any, collection.metrics),
    )
    observations, state = env.reset(
        training_keys(carry.root_key, generations, stream="reset")
    )
    memory = init_systems(
        collection.actor,
        collection.opponent,
        observations,
        state,
        training_keys(carry.root_key, generations, stream="initialization"),
        variables_a=carry.history.current_variables,
        variables_b=carry.history,
    )
    tracking = init_episode_tracking(
        env,
        state,
        source_configs=bank,
        source_indices=sampled.source_indices,
        source_class_ids=sampled.source_class_ids,
        record_starts=collection.recording,
    ).begin_stage(state, total_env_steps=int(carry.schedule.round_budgets[0]) * 2)
    return collection, carry._replace(
        env=env,
        observations=observations,
        state=state,
        memory=memory,
        tracking=tracking,
        source_indices=sampled.source_indices,
        source_class_ids=sampled.source_class_ids,
    )


@pytest.fixture(scope="module")
def rich(prepared: PreparedTrainingContent) -> Context:
    actor = _actor()
    context = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(total_env_steps=12, num_envs=2),
        prepared=prepared,
        recording=True,
        collect_training_state=True,
        shaping=True,
    )
    return _synthetic(context)


def _scan(context: Context, *, length: int) -> tuple[TrainingCarry, TrainingRollout]:
    collection, carry = context
    return _scan_function(collection, length)(carry)


@lru_cache
def _scan_function(
    collection: TrainingCollection, length: int
) -> Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]]:
    def scan(values: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
        return scan_training_rollout(collection, values, length=length)

    return _jit(scan)


@lru_cache
def _stepper(collection: TrainingCollection) -> Callable[[TrainingCarry], StepResult]:
    def step(values: TrainingCarry) -> StepResult:
        return advance_training_step(collection, values)

    return _jit(step)


def _join(a: Array, b: Array) -> Array:
    return jnp.concatenate((a, b))


def _team_a(leaf: Array) -> Array:
    return leaf[:, :5]


def _writer_policies(collection: TrainingCollection) -> dict[str, object]:
    return {"team_a": collection.actor, "team_b": collection.opponent}


def test_canonical_setup_rejects_changed_bank_and_frozen_actor_before_execution(
    prepared: PreparedTrainingContent, plain: Context
) -> None:
    collection, carry = plain
    np.testing.assert_array_equal(carry.state.config.max_steps, 300)
    np.testing.assert_array_equal(carry.progress.rounds, 0)
    np.testing.assert_array_equal(carry.memory.team_a, 3)
    changed = replace(
        prepared,
        source_configs=prepared.source_configs._replace(
            max_steps=jnp.ones(42, jnp.int32)
        ),
    )
    actor = _actor()
    with pytest.raises(ValueError, match="identity"):
        init_training_collection(
            actor, actor.variables, schedule=collection.schedule, prepared=changed
        )
    with pytest.raises(ValueError, match="checkpoint"):
        init_training_collection(
            replace(actor, checkpoint="Frozen"),
            actor.variables,
            schedule=collection.schedule,
            prepared=prepared,
        )


@pytest.mark.parametrize("mode", ["dense", "", None, True])
def test_unknown_shaping_mode_fails_before_content_preparation(
    plain: Context, monkeypatch: pytest.MonkeyPatch, mode: object
) -> None:
    def forbidden(*_args: Tree, **_kwargs: Tree) -> None:
        pytest.fail("Invalid shaping mode reached content preparation")

    monkeypatch.setattr(collection_module, "prepare_training_content", forbidden)
    collection, _ = plain
    actor = _actor()
    with pytest.raises(ValueError, match="mode"):
        init_training_collection(
            actor,
            actor.variables,
            schedule=collection.schedule,
            shaping=False,
            shaping_mode=cast(str, mode),
        )


def test_split_direct_blocks_keep_identical_decisions_memory_and_successor(
    rich: Context,
) -> None:
    collection, carry = rich
    whole, output = _scan(rich, length=8)
    middle, first = _scan(rich, length=3)
    split, second = _scan((collection, middle), length=5)
    joined = jax.tree.map(_join, first.transitions, second.transitions)
    _equal(joined, output.transitions)
    _equal(split, whole)
    _equal(output.final_observations, whole.observations)
    _equal(output.initial_memory, carry.memory.team_a)
    _equal(second.initial_memory, middle.memory.team_a)
    np.testing.assert_array_equal(
        output.transitions.ended[:6], [[False, False], [True, True]] * 3
    )
    np.testing.assert_array_equal(whole.state.reset_generation, 2)
    np.testing.assert_array_equal(whole.memory.team_a, 5)
    assert int(output.real_steps) == 6
    assert output.transitions.training_state is not None
    assert output.transitions.training_state.shape == (8, 2, 919)
    summary = training_summary(collection, whole)
    assert summary["env_steps"] == 12
    assert sum(cast(list[int], summary["steps_by_map"])) == 12
    assert sum(cast(list[int], summary["steps_by_episode_stage"])) == 12
    assert sum(cast(list[int], summary["steps_by_opponent"])) == 12
    assert sum(cast(list[int], summary["starts_by_opponent"])) == 6
    assert summary["incomplete_games"] == 0
    assert summary["learner_samples"] is None


def test_each_stored_epoch_matches_public_reset_step_and_producing_info(
    rich: Context,
) -> None:
    collection, current = rich
    step = _stepper(collection)
    for _ in range(4):
        before = current
        if bool(jnp.any(before.state.done.done)):
            assert before.tracking.source_configs is not None
            generations = before.state.reset_generation + 1
            sampled = sample_training_configs(
                before.tracking.source_configs,
                before.root_key,
                generations,
                eligible_maps=before.schedule.eligible_maps[
                    before.tracking.stage_ordinal
                ],
                team_size=before.schedule.team_sizes[before.tracking.stage_ordinal],
            )
            observations, state = _jit(before.env.reset_done)(
                training_keys(before.root_key, generations, stream="reset"),
                before.state,
                sampled.config,
            )
        else:
            observations, state = before.observations, before.state
        current, (row, info, _) = step(before)
        _equal(row.observations, observations)
        _equal(
            row.training_state, encode_training_state(state.core_state, state.config)
        )
        _equal(row.action_mask, jax.tree.map(_team_a, state.action_mask))
        np.testing.assert_array_equal(
            row.active, state.config.agent_profile.active_mask[:, :5]
        )
        np.testing.assert_array_equal(row.alive, state.core_state.alive_mask[:, :5])
        keys = training_keys(
            before.root_key,
            state.reset_generation,
            stream="step",
            decision_step=state.core_state.step_count - state.initial_step_count,
        )
        expected = _jit(before.env.step)(keys, state, row.actions)
        _equal(current.state, expected[1])
        np.testing.assert_array_equal(row.task_rewards, expected[2].rewards[:, :5])
        np.testing.assert_array_equal(row.episode_id, info.episode_id)
        np.testing.assert_array_equal(row.decision_step, info.decision_step)
        np.testing.assert_array_equal(row.ended, info.completed)
        np.testing.assert_array_equal(
            row.final_scores, jnp.where(info.completed[:, None], info.team_scores, 0)
        )
        assert info.priority is not None and row.priority is not None
        availability = info.priority.valid & info.completed[:, None]
        np.testing.assert_array_equal(row.priority.valid, availability)
        np.testing.assert_array_equal(
            row.priority.values, jnp.where(availability, info.priority.values, 0)
        )


def test_recorded_output_and_restored_writer_suffix_match_direct(
    rich: Context, tmp_path: Path
) -> None:
    collection, carry = rich
    direct, expected = _scan(rich, length=8)
    with marl_bgs.RunWriter(
        tmp_path, phase="training", policies=_writer_policies(collection)
    ) as writer:
        actual, recorded = collect_training_rollout(
            collection, carry, length=8, writer=writer
        )
        run_dir = writer.run_dir
    _equal(actual, direct)
    _equal(recorded, expected)
    details = json.loads((run_dir / "run_details.json").read_text())
    starts = next(iter(details["passes"].values()))["episode_starts"]
    assert len(starts) == 6

    with marl_bgs.RunWriter(
        tmp_path / "resume", phase="training", policies=_writer_policies(collection)
    ) as writer:
        midpoint, _ = collect_training_rollout(
            collection, carry, length=2, writer=writer
        )
        token = writer.checkpoint_recording()
        saved = jax.device_put(jax.device_get(midpoint))
        expected_carry, expected_suffix = collect_training_rollout(
            collection, midpoint, length=6, writer=writer
        )
        resume_dir = writer.run_dir
    files = {
        str(path.relative_to(resume_dir)): path.read_bytes()
        for path in resume_dir.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".jsonl"}
    }
    with marl_bgs.RunWriter(
        resume_from=resume_dir,
        recording_checkpoint=token,
        phase="training",
        policies=_writer_policies(collection),
    ) as writer:
        recovered, suffix = collect_training_rollout(
            collection, saved, length=6, writer=writer
        )
        before = (resume_dir / "run_details.json").read_bytes()
        unchanged, empty = collect_training_rollout(
            collection, recovered, length=0, writer=writer
        )
        _equal(unchanged, recovered)
        assert int(empty.real_steps) == 0
        assert (resume_dir / "run_details.json").read_bytes() == before
    _equal(recovered, expected_carry)
    _equal(suffix, expected_suffix)
    for name, contents in files.items():
        assert (resume_dir / name).read_bytes() == contents


def test_prefixes_and_zero_length_keep_padding_finite_and_leave_state_unchanged(
    plain: Context,
) -> None:
    collection, initial = plain
    step = _stepper(collection)
    fixed_scan = _scan_function(collection, 4)
    boundaries = [initial]
    for _ in range(4):
        boundaries.append(step(boundaries[-1])[0])
    for prefix in (0, 1, 3, 4):
        before = boundaries[4 - prefix]
        after, rollout = fixed_scan(before)
        _equal(after, boundaries[4])
        assert int(rollout.real_steps) == prefix
        row = rollout.transitions
        assert row.priority is None and row.training_state is None
        assert rollout.final_training_state is None
        np.testing.assert_array_equal(
            row.valid, np.broadcast_to((np.arange(4) < prefix)[:, None], (4, 2))
        )
        np.testing.assert_array_equal(row.episode_id[prefix:], -1)
        np.testing.assert_array_equal(row.actions.move[prefix:], 0)
        np.testing.assert_array_equal(row.learning_outputs["calls"][prefix:], 0)
        masks = categorical_action_mask(row.action_mask)
        np.testing.assert_array_equal(masks[prefix:, ..., 0], True)
        np.testing.assert_array_equal(masks[prefix:, ..., 1:], False)
        logits = jnp.zeros((*masks.shape[:-1], NUM_ACTIONS), jnp.float32)
        indices = jnp.zeros(masks.shape[:-1], jnp.int32)
        logp = action_log_prob(logits, masks, indices)
        entropy = action_entropy(logits, masks)
        assert np.isfinite(np.asarray(logp)).all()
        assert np.isfinite(np.asarray(entropy)).all()

        def loss(values: Array, valid: Array, masks: Array, indices: Array) -> Array:
            return jnp.sum(
                jnp.where(
                    valid[..., None],
                    action_log_prob(values, masks, indices)
                    + action_entropy(values, masks),
                    0,
                )
            )

        value, gradients = jax.value_and_grad(loss)(logits, row.valid, masks, indices)
        assert np.isfinite(np.asarray(value)).all()
        assert np.isfinite(np.asarray(gradients)).all()
        np.testing.assert_array_equal(gradients[prefix:], 0)
    unchanged, empty = _scan(plain, length=0)
    _equal(unchanged, initial)
    assert empty.transitions.valid.shape == (0, 2)
    _equal(empty.final_observations, initial.observations)


def test_stage_terminal_and_update_boundary_reset_uses_new_request_and_weights(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _actor()
    context = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(
            total_env_steps=80, num_envs=2, curriculum=True
        ),
        prepared=prepared,
        metrics="none",
    )
    collection, initial = _synthetic(context, horizon=1)
    first, first_rollout = _scan((collection, initial), length=1)
    np.testing.assert_array_equal(first.state.done.done, True)
    np.testing.assert_array_equal(first.state.reset_generation, 0)
    assert int(first.tracking.stage_ordinal) == 1
    np.testing.assert_array_equal(first_rollout.transitions.requested_stage, 0)
    history, event = refresh_opponents(
        first.history,
        {"value": jnp.float32(7)},
        completed_rounds=first.progress.rounds,
        update_index=jnp.int32(1),
        schedule=first.schedule,
    )
    assert not bool(event.created)
    refreshed = first._replace(history=history)
    second, row = _scan((collection, refreshed), length=1)
    np.testing.assert_array_equal(row.transitions.episode_start, True)
    np.testing.assert_array_equal(row.transitions.episode_stage, 1)
    np.testing.assert_array_equal(row.transitions.requested_stage, 1)
    np.testing.assert_array_equal(row.transitions.active.sum(axis=-1), 2)
    np.testing.assert_array_equal(row.transitions.learner_update, 1)
    np.testing.assert_array_equal(row.transitions.learning_outputs["weight"], 7)
    np.testing.assert_array_equal(second.memory.team_a, 8)
    _equal(row.initial_memory, first.memory.team_a)
    final, remaining = _scan((collection, second), length=128)
    assert int(remaining.real_steps) == 38
    np.testing.assert_array_equal(final.progress.stage_complete, True)
    np.testing.assert_array_equal(remaining.transitions.learner_update[:38], 1)
    assert int(final.history.current_update) == 1
    np.testing.assert_array_equal(final.state.reset_generation, 39)
    # The admitted H300 games stay live through the same short schedule. Their
    # first distribution owns all experience despite seventeen stage requests.
    ongoing, live = _scan(context, length=128)
    np.testing.assert_array_equal(ongoing.state.reset_generation, 0)
    np.testing.assert_array_equal(live.transitions.episode_stage[:40], 0)
    np.testing.assert_array_equal(ongoing.progress.exposure[0], 40)
    np.testing.assert_array_equal(ongoing.progress.exposure[1:], 0)
    np.testing.assert_array_equal(ongoing.progress.stage_complete, True)


def test_partial_reset_preserves_the_continuing_game_and_memory(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _actor()
    context = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(total_env_steps=8, num_envs=2),
        prepared=prepared,
        metrics="none",
    )
    collection, initial = _synthetic(context, horizon=100, partial=True)
    step = _stepper(collection)
    first, _ = step(initial)
    np.testing.assert_array_equal(first.state.done.done, [True, False])
    second, (row, _, _) = step(first)
    np.testing.assert_array_equal(second.state.reset_generation, [1, 0])
    np.testing.assert_array_equal(row.episode_start, [True, False])
    np.testing.assert_array_equal(second.memory.team_a[:, 0], [4, 5])
    for a, b in zip(
        jax.tree.leaves(first.state.config),
        jax.tree.leaves(second.state.config),
        strict=True,
    ):
        np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(second.source_class_ids[1], first.source_class_ids[1])


@pytest.mark.parametrize("fault", ["generation", "episode_id", "history"])
def test_post_reset_failure_stops_before_memory_actions_and_writer_drain(
    rich: Context, fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, _ = rich
    terminal, _ = _scan(rich, length=2)
    if fault == "generation":
        terminal = terminal._replace(
            state=terminal.state._replace(
                reset_generation=jnp.full(2, 2**31 - 1, jnp.int32)
            )
        )
    elif fault == "episode_id":
        terminal = terminal._replace(
            state=terminal.state._replace(
                last_reserved_episode_id=jnp.full(2, 2**31 - 1, jnp.int32)
            )
        )
    else:
        original = collection_module.assign_opponents

        def fail_assignment(*args: Tree, **kwargs: Tree) -> Tree:
            return original(*args, **kwargs)._replace(error=jnp.asarray(True))

        monkeypatch.setattr(collection_module, "assign_opponents", fail_assignment)
        reset_pending = collection_module._reset_pending  # pyright: ignore[reportPrivateUsage]

        def traced_reset(
            settings: TrainingCollection, values: TrainingCarry
        ) -> TrainingCarry:
            return reset_pending(settings, values)

        # Use a fresh branch identity so JAX traces the injected assignment,
        # rather than reusing the earlier successful reset branch program.
        monkeypatch.setattr(collection_module, "_reset_pending", traced_reset)
    collection = replace(collection)
    failed, (row, info, trace) = _stepper(collection)(terminal)
    _equal(failed.memory, terminal.memory)
    _equal(failed.progress.rounds, terminal.progress.rounds)
    _equal(failed.progress.exposure, terminal.progress.exposure)
    _equal(
        failed.state.cumulative_transition_count,
        terminal.state.cumulative_transition_count,
    )
    np.testing.assert_array_equal(row.valid, False)
    np.testing.assert_array_equal(trace.valid, False)
    np.testing.assert_array_equal(info.decision_step, -1)
    np.testing.assert_array_equal(info.completed, False)
    assert info.episode_start_records is not None
    np.testing.assert_array_equal(info.episode_start_records.valid, False)
    assert np.any(np.asarray(info.lifecycle_error)) or np.any(
        np.asarray(info.episode_tracking_error)
    )
    frozen, padding = _scan((collection, failed), length=3)
    _equal(frozen, failed)
    np.testing.assert_array_equal(padding.transitions.valid, False)
    with pytest.raises(ValueError, match="error"):
        training_summary(collection, failed)
    with marl_bgs.RunWriter(
        tmp_path, phase="training", policies=_writer_policies(collection)
    ) as writer:
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises(ValueError):
            collect_training_rollout(collection, terminal, length=3, writer=writer)
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        assert not list(writer.run_dir.glob("*.csv"))


def test_untrained_mappo_retains_same_call_native_choices_and_probabilities(
    prepared: PreparedTrainingContent,
) -> None:
    weights = initialize_ppo(jax.random.key(912)).actor_params
    actor = make_recurrent_mappo_system(weights)
    collection, carry = init_training_collection(
        actor,
        weights,
        schedule=make_training_schedule(total_env_steps=2, num_envs=2),
        prepared=prepared,
        metrics="none",
        collect_training_state=True,
    )
    final, rollout = _scan((collection, carry), length=3)
    row = rollout.transitions
    native = ActorAction(*(leaf[0, :, :5] for leaf in row.actions))
    np.testing.assert_array_equal(
        row.learning_outputs.action_indices[0], encode_actions(native)
    )
    inputs = carry.env.policy_inputs(carry.observations, carry.state)
    expected_memory, logits = _jit(_actor_forward)(weights, carry.memory.team_a, inputs)
    mask = categorical_action_mask(inputs.action_mask)
    expected = action_log_prob(logits, mask, row.learning_outputs.action_indices[0])
    np.testing.assert_allclose(
        row.learning_outputs.log_prob[0], expected, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(
        final.memory.team_a, expected_memory, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_array_equal(row.learning_outputs.action_indices[1:], 0)
    np.testing.assert_array_equal(row.learning_outputs.log_prob[1:], 0)
    assert row.training_state is not None
    np.testing.assert_array_equal(
        row.training_state[0],
        encode_training_state(carry.state.core_state, carry.state.config),
    )
    assert rollout.final_training_state is not None
    np.testing.assert_array_equal(
        rollout.final_training_state,
        encode_training_state(final.state.core_state, final.state.config),
    )


def test_writer_choice_and_lengths_fail_without_advancing(
    plain: Context, rich: Context, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="writer"):
        collect_training_rollout(*rich, length=1)
    with (
        marl_bgs.RunWriter(tmp_path) as writer,
        pytest.raises(ValueError, match="writer"),
    ):
        collect_training_rollout(*plain, length=1, writer=writer)
    for length in (-1, True, 1.5, 2**31):
        with pytest.raises(ValueError, match="length"):
            scan_training_rollout(*plain, length=cast(int, length))


def test_same_shape_weights_history_keys_and_shorter_prefix_reuse_compilation(
    plain: Context,
) -> None:
    collection, initial = plain
    traces: list[None] = []

    def scan(carry: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
        traces.append(None)
        return scan_training_rollout(collection, carry, length=4)

    compiled = _jit(scan)
    first, _ = compiled(initial)
    changed = initial._replace(
        root_key=jax.random.key(999),
        history=initial.history._replace(current_variables={"value": jnp.float32(9)}),
    )
    second, _ = compiled(changed)
    middle, _ = _scan((collection, initial), length=1)
    history, _ = refresh_opponents(
        middle.history,
        {"value": jnp.float32(5)},
        completed_rounds=middle.progress.rounds,
        update_index=jnp.int32(1),
        schedule=middle.schedule,
    )
    last, suffix = compiled(middle._replace(history=history))
    empty, _ = compiled(first)
    jax.block_until_ready((first, second, last, empty))
    assert len(traces) == 1
    assert int(suffix.real_steps) == 3
    _equal(empty, first)
    assert int(second.memory.team_a[0, 0]) == 7
    assert int(last.history.count) == 1


@pytest.mark.parametrize("fault", ["lifecycle", "history"])
def test_failed_initial_setup_stops_before_either_memory_initializer(
    prepared: PreparedTrainingContent, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_init(_variables: Tree, _inputs: SystemInput, _keys: Array) -> Array:
        pytest.fail("Failed setup reached System memory initialization")

    actor = replace(_actor(), init=forbidden_init)
    if fault == "lifecycle":
        reset = Environment.reset

        def failed_reset(
            self: Environment, *args: Tree, **kwargs: Tree
        ) -> tuple[Observations, EnvironmentState]:
            observations, state = reset(self, *args, **kwargs)
            return observations, state._replace(
                lifecycle_error=jnp.ones_like(state.lifecycle_error)
            )

        monkeypatch.setattr(Environment, "reset", failed_reset)
    else:
        initialize = collection_module.init_opponent_history

        def failed_history(*args: Tree, **kwargs: Tree) -> Tree:
            return initialize(*args, **kwargs)._replace(error=jnp.asarray(True))

        monkeypatch.setattr(collection_module, "init_opponent_history", failed_history)
    with pytest.raises(ValueError, match="initialization failed"):
        init_training_collection(
            actor,
            actor.variables,
            schedule=make_training_schedule(total_env_steps=2, num_envs=2),
            prepared=prepared,
        )


def test_exhausted_one_step_and_future_update_fail_without_applying_actions(
    plain: Context,
) -> None:
    collection, carry = plain
    complete, _ = _scan(plain, length=4)
    for invalid in (
        complete,
        carry._replace(
            history=carry.history._replace(last_refresh_rounds=jnp.int32(1))
        ),
    ):
        failed, (row, info, _) = _stepper(collection)(invalid)
        _equal(failed.memory, invalid.memory)
        _equal(failed.state, invalid.state)
        _equal(failed.progress, invalid.progress)
        np.testing.assert_array_equal(row.valid, False)
        assert np.any(np.asarray(info.episode_tracking_error) & 2)
        with pytest.raises(ValueError, match="error"):
            collect_training_rollout(collection, failed, length=1)


@pytest.mark.parametrize("shaping_mode", ["potential", "score_delta"])
def test_disabled_projection_and_shaping_are_absent_from_traced_collection(
    plain: Context, monkeypatch: pytest.MonkeyPatch, shaping_mode: str
) -> None:
    def forbidden(*_args: Tree, **_kwargs: Tree) -> Array:
        pytest.fail("Disabled optional work reached its helper")

    monkeypatch.setattr(collection_module, "encode_training_state", forbidden)
    monkeypatch.setattr(collection_module, "team_potential_shaping", forbidden)
    monkeypatch.setattr(collection_module, "team_score_delta_shaping", forbidden)
    collection, carry = plain
    collection = replace(collection, shaping_mode=shaping_mode)
    _, output = _scan((collection, carry), length=1)
    assert output.transitions.training_state is None
    assert output.final_training_state is None
    np.testing.assert_array_equal(output.transitions.shaping_reward, 0)


def test_generic_collection_imports_with_optional_training_packages_blocked() -> None:
    program = """
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'flax', 'optax', 'orbax'}:
            raise ImportError('Optional training package blocked: ' + fullname)
        return None
sys.meta_path.insert(0, BlockTraining())
import marl_battlegrounds.training
assert 'jax' not in sys.modules
from marl_battlegrounds.training import init_training_collection, make_training_schedule
assert callable(init_training_collection)
assert make_training_schedule(total_env_steps=2, num_envs=2).num_envs == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shaping_mode", ["potential", "score_delta"])
def test_distinct_score_fixture_keeps_both_completions_across_a_reset(
    rich: Context,
    shaping_mode: str,
) -> None:
    collection, initial = rich
    if shaping_mode != "potential":
        collection = replace(collection, shaping_mode=shaping_mode)
    current, _ = _scan((collection, initial), length=1)
    scores = jnp.array([[2, 0], [0, 3]], jnp.int32)
    changed = current.state.core_state._replace(team_deathmatch_scores=scores)
    snapshots = []
    for lane in range(2):

        def lane_value(value: Array, lane: int = lane) -> Array:
            return value[lane]

        def host_config_value(value: Array) -> Tree:
            return value.item() if value.ndim == 0 else value

        # A declared test snapshot varies score ownership without needing a
        # learned combat policy. Core rebuilds the matching observation/mask.
        snapshots.append(
            core.initialize_scenario_state(
                jax.tree.map(lane_value, changed),
                jax.tree.map(
                    host_config_value, jax.tree.map(lane_value, current.state.config)
                ),
            )
        )

    def stack(a: Array, b: Array) -> Array:
        return jnp.stack((a, b))

    state, observation, mask, _ = jax.tree.map(stack, *snapshots)
    wrapper = current.state._replace(
        core_state=state, observation=observation, action_mask=mask
    )
    seeded = current._replace(
        state=wrapper, observations=current.env.get_observations(wrapper)
    )
    step = _stepper(collection)
    next_carry, (row, info, _) = step(seeded)
    np.testing.assert_array_equal(info.team_scores, scores)
    np.testing.assert_array_equal(row.final_scores, scores)
    np.testing.assert_array_equal(row.ended, True)
    np.testing.assert_allclose(
        row.shaping_reward,
        (-0.02, 0.03) if shaping_mode == "potential" else (0.0, 0.0),
        atol=2e-9,
    )
    assert row.priority is not None and info.priority is not None
    expected_valid = info.priority.valid & info.completed[:, None]
    np.testing.assert_array_equal(
        row.priority.values, jnp.where(expected_valid, info.priority.values, 0)
    )
    following, output = _scan((collection, next_carry), length=3)
    np.testing.assert_array_equal(
        output.transitions.ended[:2], [[False, False], [True, True]]
    )
    np.testing.assert_array_equal(output.transitions.final_scores[:2], 0)
    np.testing.assert_array_equal(output.transitions.shaping_reward, 0)
    assert np.all(
        np.asarray(output.transitions.episode_id[:2]) != np.asarray(row.episode_id)
    )
    assert int(following.progress.rounds) == 5
    _equal(
        initial.state.core_state.team_deathmatch_scores, jnp.zeros((2, 2), jnp.int32)
    )


def test_reset_selects_a_real_frozen_opponent_and_joins_version_exposure(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _actor()
    context = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(total_env_steps=12, num_envs=2),
        seed=3,
        prepared=prepared,
        metrics="none",
    )
    collection, current = _synthetic(context)
    for update, value in ((1, 7), (2, 9)):
        current, _ = _scan((collection, current), length=1)
        history, event = refresh_opponents(
            current.history,
            {"value": jnp.float32(value)},
            completed_rounds=current.progress.rounds,
            update_index=jnp.int32(update),
            schedule=current.schedule,
        )
        assert bool(event.created)
        current = current._replace(history=history)
    saved = current.history.historical_variables
    after, (row, _, _) = _stepper(collection)(current)
    np.testing.assert_array_equal(row.opponent_snapshot, [-1, 0])
    np.testing.assert_array_equal(row.opponent_update, [2, 1])
    np.testing.assert_array_equal(row.learner_update, 2)
    np.testing.assert_array_equal(row.learning_outputs["weight"], 9)
    np.testing.assert_array_equal(after.memory.team_a, 10)
    np.testing.assert_array_equal(after.memory.team_b[:, 0], [10, 8])
    _equal(after.history.historical_variables, saved)
    summary = training_summary(collection, after)
    assert summary["steps_by_opponent"] == [5, 1] + [0] * 19
    assert summary["starts_by_opponent"] == [3, 1] + [0] * 19


def test_pinned_share_and_early_capture_must_be_declared_together(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _actor()
    plain = make_training_schedule(total_env_steps=12, num_envs=2)
    early = make_training_schedule(
        total_env_steps=12, num_envs=2, early_history_capture=True
    )
    with pytest.raises(ValueError, match="early_history_capture"):
        init_training_collection(
            actor,
            actor.variables,
            schedule=plain,
            prepared=prepared,
            metrics="none",
            pinned_opponent_share=0.1,
        )
    with pytest.raises(ValueError, match="positive share"):
        init_training_collection(
            actor, actor.variables, schedule=early, prepared=prepared, metrics="none"
        )
    with pytest.raises(ValueError, match=r"0\.8"):
        init_training_collection(
            actor,
            actor.variables,
            schedule=early,
            prepared=prepared,
            metrics="none",
            pinned_opponent_share=0.9,
        )
    with pytest.raises(TypeError, match="bool"):
        init_training_collection(
            actor,
            actor.variables,
            schedule=early,
            prepared=prepared,
            metrics="none",
            pinned_opponent_share=cast(Any, True),
        )


def test_pinned_share_reaches_the_rollout_program_only_when_positive(
    plain: Context,
) -> None:
    # Default and explicit 0.0 trace one program by construction; identity with
    # the pre-change rollout is proven by the diff and the GPU equivalence job.
    collection, carry = plain
    default = str(
        jax.make_jaxpr(partial(scan_training_rollout, collection, length=2))(carry)
    )
    explicit = replace(collection, pinned_opponent_share=0.0)
    assert default == str(
        jax.make_jaxpr(partial(scan_training_rollout, explicit, length=2))(carry)
    )
    pinned = replace(collection, pinned_opponent_share=0.1)
    assert default != str(
        jax.make_jaxpr(partial(scan_training_rollout, pinned, length=2))(carry)
    )


def test_early_capture_pins_the_first_update_actor_in_slot_zero(
    prepared: PreparedTrainingContent,
) -> None:
    actor = _actor()
    context = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(
            total_env_steps=40, num_envs=2, early_history_capture=True
        ),
        seed=3,
        prepared=prepared,
        metrics="none",
        pinned_opponent_share=0.5,
    )
    collection, real = context
    assert collection.pinned_opponent_share == 0.5
    assert int(real.schedule.history_threshold_rounds[0]) == 1
    # Continuation checks need the verified bank, so prove the early capture on
    # the real content first; no game ends within one real round here.
    validate = collection_module._validate_training_continuation  # pyright: ignore[reportPrivateUsage]
    validate(collection, real, expected_root_bits=collection.root_bits)
    first, _ = _scan((collection, real), length=1)
    history, event = refresh_opponents(
        first.history,
        {"value": jnp.float32(7)},
        completed_rounds=first.progress.rounds,
        update_index=jnp.int32(1),
        schedule=first.schedule,
    )
    assert bool(event.created)
    assert int(event.slot) == 0
    assert int(history.captured_rounds[0]) == 1
    assert int(history.captured_updates[0]) == 1
    assert int(history.count) == 1
    validate(collection, first._replace(history=history))
    # Short synthetic games then show the pinned actor actually being drawn.
    collection, current = _synthetic(context)
    current, _ = _scan((collection, current), length=1)
    history, event = refresh_opponents(
        current.history,
        {"value": jnp.float32(7)},
        completed_rounds=current.progress.rounds,
        update_index=jnp.int32(1),
        schedule=current.schedule,
    )
    assert bool(event.created)
    current = current._replace(history=history)
    seen: set[int] = set()
    for _ in range(10):
        current, (row, _, _) = _stepper(collection)(current)
        seen.update(np.asarray(row.opponent_snapshot).tolist())
    assert seen == {-1, 0}
    summary = training_summary(collection, current)
    steps = cast(list[int], summary["steps_by_opponent"])
    starts = cast(list[int], summary["starts_by_opponent"])
    assert len(steps) == 21
    assert len(starts) == 21
    assert starts[1] > 0
    assert steps[1] > 0
    # Trivial here: the bank holds one snapshot throughout, so no other slot can
    # be drawn; the count-above-one split is proven at the assign_opponents level.
    assert starts[2:] == [0] * 19
    assert sum(steps) == int(current.progress.rounds) * 2


def test_canonical_continuation_checks_zero_midpoint_and_final_before_writer_recovery(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    actor = _actor()
    collection, initial = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(total_env_steps=8, num_envs=2),
        prepared=prepared,
        recording=True,
        metrics="none",
    )
    validate = collection_module._validate_training_continuation  # pyright: ignore[reportPrivateUsage]
    validate(collection, initial, expected_root_bits=collection.root_bits)
    with marl_bgs.RunWriter(
        tmp_path, phase="training", policies=_writer_policies(collection)
    ) as writer:
        middle, _ = collect_training_rollout(
            collection, initial, length=1, writer=writer
        )
        history, event = refresh_opponents(
            middle.history,
            {"value": jnp.float32(5)},
            completed_rounds=middle.progress.rounds,
            update_index=jnp.int32(1),
            schedule=middle.schedule,
        )
        assert bool(event.created)
        middle = middle._replace(history=history)
        validate(collection, middle)
        token = writer.checkpoint_recording()
        saved = jax.device_put(jax.device_get(middle))
        final, expected = collect_training_rollout(
            collection, middle, length=4, writer=writer
        )
        validate(collection, final)
        run_dir = writer.run_dir
    validate(collection, saved, expected_root_bits=collection.root_bits)
    with marl_bgs.RunWriter(
        resume_from=run_dir,
        recording_checkpoint=token,
        phase="training",
        policies=_writer_policies(collection),
    ) as writer:
        restored, actual = collect_training_rollout(
            collection, saved, length=4, writer=writer
        )
    _equal(restored, final)
    _equal(actual, expected)


@pytest.mark.parametrize(
    "fault",
    [
        "root",
        "schema",
        "shape",
        "schedule",
        "reward",
        "exposure",
        "map",
        "opponent",
        "proof",
        "memory",
        "source",
        "lanes",
        "bank",
        "mapping",
        "captured",
    ],
)
def test_canonical_continuation_rejects_mutated_owned_state(
    plain: Context, fault: str
) -> None:
    collection, carry = plain
    if fault == "root":
        carry = carry._replace(root_key=jax.random.key(1))
    elif fault == "schema":
        collection = replace(collection, key_schema=999)
    elif fault == "shape":
        carry = carry._replace(
            memory=carry.memory._replace(team_a=carry.memory.team_a[:1])
        )
    elif fault == "schedule":
        carry = carry._replace(
            schedule=carry.schedule._replace(total_rounds=jnp.int32(3))
        )
    elif fault == "reward":
        carry = carry._replace(discount=jnp.float32(0.5))
    elif fault == "exposure":
        carry = carry._replace(
            progress=carry.progress._replace(
                exposure=carry.progress.exposure.at[0, 0].set(1)
            )
        )
    elif fault == "map":
        carry = carry._replace(
            progress=carry.progress._replace(
                map_steps=carry.progress.map_steps.at[0, 0].set(1)
            )
        )
    elif fault == "opponent":
        carry = carry._replace(
            progress=carry.progress._replace(
                opponent_steps=carry.progress.opponent_steps.at[0, 0].set(1)
            )
        )
    elif fault == "proof":
        carry = carry._replace(
            progress=carry.progress._replace(
                stage_complete=carry.progress.stage_complete.at[0].set(True)
            )
        )
    elif fault == "memory":
        carry = carry._replace(
            memory=carry.memory._replace(episode_id=carry.memory.episode_id + 1)
        )
    elif fault == "source":
        carry = carry._replace(source_indices=carry.source_indices + 1)
    elif fault == "lanes":
        carry = carry._replace(
            tracking=replace(
                carry.tracking, spawn_locations=carry.tracking.spawn_locations[::-1]
            )
        )
    elif fault == "mapping":
        carry = carry._replace(
            history=carry.history._replace(
                threshold_to_snapshot=carry.history.threshold_to_snapshot.at[0].set(0)
            )
        )
    elif fault == "captured":
        carry = carry._replace(
            history=carry.history._replace(
                captured_rounds=carry.history.captured_rounds.at[0].set(1)
            )
        )
    else:
        assert carry.tracking.source_configs is not None
        carry = carry._replace(
            tracking=replace(
                carry.tracking,
                source_configs=carry.tracking.source_configs._replace(
                    max_steps=jnp.ones(42, jnp.int32)
                ),
            )
        )
    with pytest.raises(ValueError):
        collection_module._validate_training_continuation(collection, carry)  # pyright: ignore[reportPrivateUsage]


def test_changed_content_is_rejected_before_writer_can_rewind(
    plain: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, carry = plain
    binding = collection.binding.model_copy(update={"canonical_digest": "0" * 64})
    collection = replace(collection, binding=binding)
    durable = tmp_path / "durable.csv"
    durable.write_bytes(b"Existing complete rows\n")
    calls: list[None] = []

    def rewind(**_kwargs: Tree) -> None:
        calls.append(None)
        durable.write_bytes(b"Rewound unexpectedly\n")

    monkeypatch.setattr(marl_bgs, "RunWriter", rewind)
    with pytest.raises(ValueError):
        collection_module._validate_training_continuation(collection, carry)  # pyright: ignore[reportPrivateUsage]
        marl_bgs.RunWriter(resume_from=tmp_path)
    assert calls == []
    assert durable.read_bytes() == b"Existing complete rows\n"


def test_malformed_schedule_is_rejected_before_system_initialization(
    prepared: PreparedTrainingContent,
) -> None:
    def forbidden(_variables: Tree, _inputs: SystemInput, _keys: Array) -> Array:
        pytest.fail("Malformed schedule reached memory initialization")

    actor = replace(_actor(), init=forbidden)
    schedule = make_training_schedule(total_env_steps=8, num_envs=2)
    changed = replace(
        schedule, arrays=schedule.arrays._replace(total_rounds=jnp.int32(1))
    )
    with pytest.raises(ValueError, match="schedule"):
        init_training_collection(
            actor, actor.variables, schedule=changed, prepared=prepared
        )
