"""Check bounded child segments without changing games or cumulative experience.

CPU collection with a counter System proves the complete 17-stage parent,
repeated child budgets, preserved unfinished games, original distribution IDs,
exact parent/local proofs and unchanged random/history/memory state at each
fork. Host checks cover early stops, schedule reconstruction, corruption and
history capacity. A tiny PPO child also checks that the first and second native
collection/update calls reuse their compiled code after device-committed restore.
These checks make no learning or GPU-speed claim.
"""

from dataclasses import replace
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_training_collection import _actor, _equal

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.training._compilation import training_compiler_options
from marl_battlegrounds.training.collection import (
    _begin_training_segment,
    _compiled_rollout,
    _continuation_history_thresholds,
    _validate_training_continuation,
    collect_training_rollout,
    init_training_collection,
    training_summary,
)
from marl_battlegrounds.training.curriculum import (
    TrainingSchedule,
    _check_training_schedule,
    _continuation_details,
    _make_continuation_schedule,
    _restore_continuation_schedule,
    make_training_schedule,
)
from marl_battlegrounds.training.learner import init_learner, update_learner
from marl_battlegrounds.training.opponents import refresh_opponents

# pyright: reportPrivateUsage=false

type Tree = Any


@pytest.fixture(scope="module")
def completed() -> Tree:
    actor = _actor()
    collection, initial = init_training_collection(
        actor,
        actor.variables,
        schedule=make_training_schedule(
            total_env_steps=110, num_envs=2, curriculum=True
        ),
        metrics="none",
    )
    final, _ = collect_training_rollout(collection, initial, length=55)
    _validate_training_continuation(collection, final)
    return collection, initial, final


def _details(schedule: TrainingSchedule) -> dict[str, Any]:
    value = _continuation_details(schedule)
    assert value is not None
    return value


def _unchanged_boundary(parent: Tree, child: Tree) -> None:
    _equal(
        parent,
        child._replace(
            schedule=parent.schedule,
            progress=child.progress._replace(
                completed_stage_counts=parent.progress.completed_stage_counts,
                stage_complete=parent.progress.stage_complete,
            ),
            tracking=replace(
                child.tracking,
                stage_ordinal=parent.tracking.stage_ordinal,
                stage_round_budget=parent.tracking.stage_round_budget,
                stage_rounds=parent.tracking.stage_rounds,
                stage_counts=parent.tracking.stage_counts,
                full_batch_rounds_valid=parent.tracking.full_batch_rounds_valid,
            ),
        ),
    )


def test_complete_seventeen_stage_parent_keeps_its_proof_through_two_children(
    completed: Tree,
) -> None:
    collection, _, parent = completed
    original_proof = np.asarray(parent.progress.completed_stage_counts).copy()
    schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=55, additional_env_steps=10
    )
    child_collection, child = _begin_training_segment(
        collection, parent, schedule=schedule
    )
    _unchanged_boundary(parent, child)
    assert child.progress.exposure.shape == (17, 2)
    assert child.progress.completed_stage_counts.shape == (17, 2, 3)
    assert int(child.schedule.stage_count) == 1
    assert child.schedule.distribution_offset is not None
    assert int(child.schedule.distribution_offset) == 16
    np.testing.assert_array_equal(child.progress.stage_complete, False)
    assert int(child.progress.rounds) == 55
    details = _continuation_details(child_collection.schedule)
    assert details is not None
    np.testing.assert_array_equal(
        details["parent_stage_proof"]["completed_stage_counts"], original_proof
    )
    np.testing.assert_array_equal(details["parent_stage_proof"]["stage_complete"], True)
    with pytest.raises(TypeError):
        cast(dict[str, object], child_collection.schedule.continuation)[
            "round_offset"
        ] = 0
    final, rollout = collect_training_rollout(child_collection, child, length=7)
    assert int(rollout.real_steps) == 5
    np.testing.assert_array_equal(rollout.transitions.requested_stage[:5], 16)
    np.testing.assert_array_equal(
        rollout.transitions.episode_stage[:5],
        np.broadcast_to(parent.progress.episode_stage, (5, 2)),
    )
    _validate_training_continuation(child_collection, final)
    assert int(final.progress.rounds) == 60
    summary = cast(dict[str, Any], training_summary(child_collection, final))
    assert summary["segment_env_steps"] == 10
    assert summary["env_steps"] == 120
    assert sum(summary["steps_by_episode_stage"]) == 120
    assert len(summary["score_thresholds_by_episode_stage"]) == 17
    np.testing.assert_array_equal(
        parent.progress.completed_stage_counts, original_proof
    )
    restored = _restore_continuation_schedule(details)
    _equal(restored.arrays, child_collection.schedule.arrays)
    next_schedule = _make_continuation_schedule(
        restored, completed_rounds=60, additional_env_steps=492
    )
    next_collection, following = _begin_training_segment(
        child_collection, final, schedule=next_schedule
    )
    _unchanged_boundary(final, following)
    assert int(following.schedule.stage_count) == 1
    assert following.schedule.round_offset is not None
    assert int(following.schedule.round_offset) == 60
    assert _details(next_collection.schedule)["segment_index"] == 2
    final_again, _ = collect_training_rollout(next_collection, following, length=246)
    _validate_training_continuation(next_collection, final_again)
    assert training_summary(next_collection, final_again)["segment_env_steps"] == 492
    assert int(final_again.progress.rounds) == 306
    np.testing.assert_array_equal(final_again.progress.episode_stage, 16)
    np.testing.assert_array_equal(final_again.progress.exposure[0], 300)
    np.testing.assert_array_equal(final_again.progress.exposure[16], 6)
    np.testing.assert_array_equal(
        final_again.state.config.agent_profile.active_mask, True
    )
    direct_schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=55, additional_env_steps=502
    )
    direct_collection, direct = _begin_training_segment(
        collection, parent, schedule=direct_schedule
    )
    reference, _ = collect_training_rollout(direct_collection, direct, length=251)
    _unchanged_boundary(reference, final_again)


def test_mid_stage_child_retains_future_root_boundaries(completed: Tree) -> None:
    collection, initial, _ = completed
    parent, _ = collect_training_rollout(collection, initial, length=1)
    child_schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=1, additional_env_steps=2
    )
    child_collection, child = _begin_training_segment(
        collection, parent, schedule=child_schedule
    )
    _unchanged_boundary(parent, child)
    assert (
        _details(child_collection.schedule)["parent_stage_proof"]["stage_complete"]
        == [False] * 17
    )
    final, _ = collect_training_rollout(child_collection, child, length=1)
    schedule = _make_continuation_schedule(
        child_collection.schedule, completed_rounds=2, additional_env_steps=116
    )
    new_collection, _ = _begin_training_segment(
        child_collection, final, schedule=schedule
    )
    arrays = new_collection.schedule.arrays
    expected_ends = [
        int(value)
        for value in collection.schedule.arrays.round_ends[:-1]
        if int(value) > 2
    ]
    expected_ends.append(60)
    np.testing.assert_array_equal(
        arrays.round_ends[: int(arrays.stage_count)], expected_ends
    )
    assert int(arrays.stage_count) <= 17
    assert arrays.distribution_count is not None
    assert int(arrays.distribution_count) == 17
    assert _details(new_collection.schedule)["root_schedule"]["total_env_steps"] == 110
    np.testing.assert_array_equal(
        arrays.team_sizes, collection.schedule.arrays.team_sizes
    )
    np.testing.assert_array_equal(
        arrays.eligible_maps, collection.schedule.arrays.eligible_maps
    )


def test_fresh_array_layout_and_invalid_child_declarations(completed: Tree) -> None:
    collection, _, parent = completed
    assert len(jax.tree.leaves(collection.schedule.arrays)) == 8
    for rounds, added in ((56, 2), (55, 0), (55, 1), (True, 2), (55, 2**32)):
        with pytest.raises(ValueError):
            _make_continuation_schedule(
                collection.schedule, completed_rounds=rounds, additional_env_steps=added
            )
    schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=55, additional_env_steps=2
    )
    with pytest.raises(ValueError, match="parent stage proof"):
        _restore_continuation_schedule(_details(schedule))
    child_collection, child = _begin_training_segment(
        collection, parent, schedule=schedule
    )
    details = _continuation_details(child_collection.schedule)
    assert details is not None
    details["parent_stage_proof"]["completed_stage_counts"][0][0][0] += 1
    with pytest.raises(ValueError, match="parent stage proof"):
        _restore_continuation_schedule(details)
    for mutated in (
        child._replace(schedule=child.schedule._replace(round_offset=jnp.int32(0))),
        child._replace(
            progress=child.progress._replace(episode_stage=jnp.full(2, 17, jnp.int32))
        ),
        child._replace(
            progress=child.progress._replace(stage_complete=jnp.ones(17, jnp.bool_))
        ),
    ):
        with pytest.raises(ValueError):
            _validate_training_continuation(child_collection, mutated)
    with pytest.raises(ValueError, match="schedule"):
        _check_training_schedule(
            replace(
                child_collection.schedule,
                arrays=child.schedule._replace(distribution_offset=jnp.int32(0)),
            )
        )


def test_history_threshold_changes_keep_frozen_slots_and_old_request_proof(
    completed: Tree,
) -> None:
    collection, _, parent = completed
    history, event = refresh_opponents(
        parent.history,
        parent.history.current_variables,
        completed_rounds=parent.progress.rounds,
        update_index=jnp.int32(1),
        schedule=parent.schedule,
    )
    assert bool(event.created)
    parent = parent._replace(history=history)
    old_mapping = np.asarray(history.threshold_to_snapshot).copy()
    schedule = _make_continuation_schedule(
        collection.schedule,
        completed_rounds=55,
        additional_env_steps=4,
        history_threshold_rounds=(55, 57),
    )
    child_collection, child = _begin_training_segment(
        collection, parent, schedule=schedule
    )
    assert child.schedule.history_threshold_count is not None
    assert int(child.schedule.history_threshold_count) == 2
    assert int(child.history.count) == 1
    np.testing.assert_array_equal(child.history.threshold_to_snapshot, [0] + [-1] * 19)
    _equal(child.history.historical_variables, history.historical_variables)
    _equal(child.history.captured_rounds, history.captured_rounds)
    details = _details(child_collection.schedule)
    np.testing.assert_array_equal(
        details["parent_stage_proof"]["history_threshold_to_snapshot"], old_mapping
    )
    bad = _make_continuation_schedule(
        collection.schedule,
        completed_rounds=55,
        additional_env_steps=4,
        history_threshold_rounds=(57,),
    )
    with pytest.raises(ValueError, match="retain every saved snapshot"):
        _begin_training_segment(collection, parent, schedule=bad)


def test_full_history_preserves_all_slots_and_refuses_an_extra_capture(
    completed: Tree,
) -> None:
    collection, _, parent = completed
    history = parent.history
    for index, rounds in enumerate(
        np.asarray(parent.schedule.history_threshold_rounds, dtype=np.int32), 1
    ):
        history, event = refresh_opponents(
            history,
            history.current_variables,
            completed_rounds=jnp.int32(rounds),
            update_index=jnp.int32(index),
            schedule=parent.schedule,
        )
        assert bool(event.created)
    parent = parent._replace(history=history)
    assert int(history.count) == 20
    assert len(_continuation_history_thresholds(parent, future_rounds=())) == 20
    with pytest.raises(ValueError, match="20 history"):
        _continuation_history_thresholds(parent, future_rounds=(57,))
    schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=55, additional_env_steps=4
    )
    _, child = _begin_training_segment(collection, parent, schedule=schedule)
    _equal(child.history, parent.history)
    _unchanged_boundary(parent, child)


def test_saved_parent_proof_types_and_original_boundaries_are_checked(
    completed: Tree,
) -> None:
    collection, _, parent = completed
    schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=55, additional_env_steps=4
    )
    child_collection, _ = _begin_training_segment(collection, parent, schedule=schedule)
    for fault in (
        "root_end",
        "proof_float",
        "proof_bool",
        "old_threshold",
        "schema_bool",
    ):
        details = _details(child_collection.schedule)
        if fault == "root_end":
            details["root_rounding_report"]["cumulative_round_ends"][-1] += 1
        elif fault == "proof_float":
            details["parent_stage_proof"]["stage_counts"][0][0] = float(
                details["parent_stage_proof"]["stage_counts"][0][0]
            )
        elif fault == "proof_bool":
            details["parent_stage_proof"]["stage_ordinal"] = True
        elif fault == "old_threshold":
            details["parent_schedule"]["history_threshold_rounds"][0] = True
        else:
            details["schema_version"] = True
        with pytest.raises(ValueError):
            _restore_continuation_schedule(details)
    _, initial, _ = completed
    old_pending = set(
        int(value) for value in np.asarray(initial.schedule.history_threshold_rounds)
    )
    assert _continuation_history_thresholds(initial, future_rounds=()) == tuple(
        sorted(old_pending)
    )
    with pytest.raises(ValueError, match="checkpoint"):
        _continuation_history_thresholds(parent, future_rounds=(55,))


def _placement_signature(tree: Tree) -> tuple[tuple[object, ...], ...]:
    paths = cast(
        list[tuple[tuple[Any, ...], Any]],
        jax.tree_util.tree_flatten_with_path(tree)[0],
    )
    return tuple(
        (
            jax.tree_util.keystr(path),
            leaf.shape,
            leaf.dtype,
            getattr(leaf, "weak_type", None),
            leaf.committed,
            leaf.sharding,
        )
        for path, leaf in paths
    )


@pytest.mark.parametrize("committed", [False, True])
def test_child_keeps_parent_placement_and_exact_values(
    completed: Tree, committed: bool
) -> None:
    collection, _, parent = completed
    schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=55, additional_env_steps=4
    )
    _, reference = _begin_training_segment(collection, parent, schedule=schedule)
    if committed:
        parent = jax.device_put(parent, cast(Any, jax.devices("cpu")[0]))
    _, child = _begin_training_segment(collection, parent, schedule=schedule)
    _equal(child, reference)
    _unchanged_boundary(parent, child)
    for leaf in jax.tree.leaves(child):
        assert leaf.committed == committed
        assert leaf.sharding == parent.root_key.sharding
    for old, new in zip(
        jax.tree.leaves(parent.state), jax.tree.leaves(child.state), strict=True
    ):
        assert new is old


def test_committed_child_reuses_native_collection_and_ppo_update() -> None:
    settings = PPOConfig(rollout_length=2, epochs=1)
    collection, state = init_learner(
        schedule=make_training_schedule(total_env_steps=8, num_envs=4),
        ppo=settings,
        method="ff_ippo",
        metrics="none",
    )
    parent = jax.device_put(state, cast(Any, jax.devices("cpu")[0]))
    schedule = _make_continuation_schedule(
        collection.schedule, completed_rounds=0, additional_env_steps=24
    )
    child_collection, carry = _begin_training_segment(
        collection, parent.carry, schedule=schedule
    )
    state = parent._replace(carry=carry)
    scan = _compiled_rollout(child_collection, 2)
    update = cast(
        Any,
        jax.jit(
            partial(update_learner, ppo=settings, method="ff_ippo"),
            compiler_options=training_compiler_options(),
        ),
    )
    initial_signature = _placement_signature(state)
    first_inputs = None
    for _ in range(2):
        collected, rollout = scan(state.carry)
        inputs = _placement_signature((state, collected, rollout))
        if first_inputs is None:
            first_inputs = inputs
        else:
            assert inputs == first_inputs
        state, result = update(state, collected, rollout)
        jax.block_until_ready((state, result))
        assert bool(result.performed) and not bool(result.failed)
        assert _placement_signature(state) == initial_signature
        assert scan._cache_size() == 1
        assert update._cache_size() == 1
