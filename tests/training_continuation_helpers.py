"""Compare child learners with direct continuation from their saved parent.

This CPU support keeps the declared partial blocks and future schedules equal
on both paths. It checks every saved numerical leaf, including replay or recent
rows, frozen actors, memory, optimizer state, statistics, counters and keys.
"""

# pyright: reportPrivateUsage=false
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import jax
import pytest
from tests.training_learner_helpers import equal

from marl_battlegrounds.training import checkpoints, runner
from marl_battlegrounds.training._compilation import training_compiler_options
from marl_battlegrounds.training._continuation_schedules import (
    continuation_collection,
    continuation_state,
    schedule_continuation,
)
from marl_battlegrounds.training.collection import (
    _begin_training_segment,
    collect_training_rollout,
    scan_training_rollout,
)


def capture_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[Path, Any], dict[Path, Any]]:
    starts: dict[Path, Any] = {}
    ends: dict[Path, Any] = {}
    initialize = runner._Run.__init__
    save = runner._Run.save

    def remember_start(execution: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        initialize(execution, *args, **kwargs)
        starts[execution.root] = (execution.collection, execution.state)

    def remember_end(execution: Any) -> Path:  # noqa: ANN401
        result = save(execution)
        ends[execution.root] = execution.state
        return result

    monkeypatch.setattr(runner._Run, "__init__", remember_start)
    monkeypatch.setattr(runner._Run, "save", remember_end)
    return starts, ends


def fixed_source(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    identity = checkpoints.runtime_identity(method=method)

    def current(*_args: object, **_kwargs: object) -> dict[str, object]:
        return identity

    monkeypatch.setattr(checkpoints, "runtime_identity", current)


def latest(root: Path) -> Path:
    pointer = json.loads((root / "latest_checkpoint.json").read_text())
    return root / pointer["relative_path"]


def compare_direct_child(
    parent_root: Path,
    child_root: Path,
    starts: dict[Path, Any],
    ends: dict[Path, Any],
    settings: Any,  # noqa: ANN401
    method: str,
) -> None:
    parent_collection, _ = starts[parent_root]
    child_collection, child_start = starts[child_root]
    parent = ends[parent_root]
    expected_collection, carry = _begin_training_segment(
        parent_collection, parent.carry, schedule=child_collection.schedule
    )
    context = schedule_continuation(child_collection.schedule)
    assert context is not None
    expected_collection = continuation_collection(expected_collection, context)
    initial = settings.initial_rounds if method == "pqn_vdn" else 0
    expected = parent._replace(carry=carry)
    details = json.loads((child_root / "run_details.json").read_text())
    if "exploration" in details["continuation"]["changes"]:
        expected = continuation_state(
            expected,
            context,
            num_envs=child_collection.schedule.num_envs,
            initial_rounds=initial,
        )
    equal(child_start, expected)
    if method == "pqn_vdn":
        from marl_battlegrounds.training.pqn_learner import update_pqn_learner

        update = partial(
            update_pqn_learner,
            pqn=settings,
            planned_learning_blocks=cast(int, context.pqn_planned_learning_blocks),
            continuation=context,
        )
    elif method == "qmix":
        from marl_battlegrounds.training.qmix_learner import update_qmix_learner

        update = partial(update_qmix_learner, qmix=settings, continuation=context)
    else:
        from marl_battlegrounds.training.learner import update_learner

        assert method in {"mappo", "ippo", "ff_mappo", "ff_ippo"}
        update = partial(
            update_learner, ppo=settings, method=method, continuation=context
        )
    compiled = cast(
        Callable[[Any, Any, Any], tuple[Any, Any]],
        jax.jit(update, compiler_options=training_compiler_options()),
    )
    real_rounds = 0
    scans: dict[int, Any] = {}
    assert expected_collection.host_opponent is None
    while int(expected.carry.progress.rounds) < int(
        expected.carry.schedule.total_rounds
    ):
        rounds = int(expected.carry.progress.rounds)
        length = (
            min(settings.rollout_length, initial - rounds)
            if rounds < initial
            else settings.rollout_length
        )
        if expected_collection.recording:
            # Keep recording counters, but do not write the same games twice.
            if length not in scans:
                scans[length] = jax.jit(
                    partial(scan_training_rollout, expected_collection, length=length),
                    compiler_options=training_compiler_options(),
                )
            carried, rollout = scans[length](expected.carry)
        else:
            carried, rollout = collect_training_rollout(
                expected_collection, expected.carry, length=length
            )
        expected, result = compiled(expected, carried, rollout)
        accepted = (
            result.accepted if method in {"qmix", "pqn_vdn"} else result.performed
        )
        assert bool(accepted) and not bool(result.failed)
        real_rounds += int(rollout.real_steps)
    assert real_rounds == int(expected.carry.progress.rounds) - context.start_rounds
    equal(ends[child_root], expected)


def reward_one(*_args: object) -> jax.Array:
    return jax.numpy.ones((10,), jax.numpy.float32)


def reward_seven(*_args: object) -> jax.Array:
    return jax.numpy.full((10,), 7.0, jax.numpy.float32)
