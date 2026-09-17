"""Pack refill-free evaluation chunks into the existing private writer transport.

The evaluator retains one completion per lane and only compact selected routing.
This host boundary filters occupied device rows before transferring them. It does
not collect training rollouts, change schemas or own any writer transaction.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.evaluate import (
        _Completed,  # pyright: ignore[reportPrivateUsage]
    )


import jax
import jax.numpy as jnp
import numpy as np

from marl_battlegrounds.environment import EnvironmentState
from marl_battlegrounds.evaluation.collection_types import (
    CollectedAssignments,
    CollectedBatch,
    CollectedBuffers,
    CollectedCompletions,
    CollectedCounts,
    CollectionErrors,
    ConfigEvidence,
)
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets


@partial(jax.jit, static_argnames=("leading",))
def _rows[T](tree: T, indices: np.ndarray, leading: int = 1) -> T:
    """Select occupied rows of a numerical tree before transferring its payload.

    indices names flattened leading axes. Remaining axes keep their exact shape,
    dtype and values. None stays None. The caller owns index bounds and validity.
    Only leading is static; index values remain dynamic. A changed row count
    changes the output shape and may compile separately. No host transfer occurs.
    """

    def select(value: jax.Array) -> jax.Array:
        """Flatten recording axes and gather only occupied rows."""
        return value.reshape((-1, *value.shape[leading:]))[indices]

    return cast(T, jax.tree.map(select, tree))


def evaluation_record_batch(
    before: EnvironmentState,
    completed: _Completed,
    replay: ReplayPackets | None,
    assignments: CollectedAssignments | None,
) -> CollectedBatch:
    """Prepare one evaluation chunk for shared all-family writer validation.

    Parameters
    ----------
    before : EnvironmentState
        Native state at the start of a refill-free chunk. Its config is actual
        evidence for each lane, including lanes with no records this chunk.
    completed : evaluator _Completed
        Existing first-completion accumulator with sticky lifecycle failures.
    replay : ReplayPackets | None
        Optional selected packets with time/capacity leading axes.
    assignments : CollectedAssignments | None
        Compact selected decisions with time/capacity axes. Each config_index is
        its original lane, independent of the supplied trace's ownership claims.

    Returns
    -------
    CollectedBatch
        Device payloads sliced to occupied prefixes, small counts and errors.
        No whole method carry, learner outputs or final training inputs are kept.
        Starts are absent because evaluation registers its exact host schedule.

    Notes
    -----
    Host only. Small validity masks are read to choose bounded device rows. The
    writer validates all errors and joins before publication, then transfers the
    occupied payload once. No file, registration or replay state is changed here.
    """
    selected = np.flatnonzero(np.asarray(jax.device_get(completed.completed)))
    priority_mask = (
        np.zeros(len(selected), dtype=bool)
        if completed.priority is None
        else np.asarray(
            jax.device_get(jnp.any(completed.priority.valid[selected], axis=-1))
        )
    )
    full_mask = (
        np.zeros(len(selected), dtype=bool)
        if completed.full is None
        else np.asarray(
            jax.device_get(jnp.any(completed.full.valid[selected], axis=-1))
        )
    )
    priority_mask = priority_mask & ~full_mask
    p_index = np.full(len(selected), -1, np.int32)
    f_index = np.full(len(selected), -1, np.int32)
    p_index[priority_mask] = np.arange(priority_mask.sum(), dtype=np.int32)
    f_index[full_mask] = np.arange(full_mask.sum(), dtype=np.int32)
    headers = CollectedCompletions(
        completed.episode_id[selected],
        completed.outcome[selected],
        completed.decision_step[selected],
        completed.length[selected],
        completed.scores[selected],
        jnp.asarray(selected, jnp.int32),
        jnp.asarray(p_index),
        jnp.asarray(f_index),
    )
    traces = None
    if assignments is not None:
        valid = np.asarray(jax.device_get(assignments.trace.valid)).reshape(-1)
        traces = cast(
            CollectedAssignments,
            _rows(assignments, np.flatnonzero(valid), leading=2),
        )
    packets = None
    if replay is not None:
        valid = np.asarray(jax.device_get(replay.valid)).reshape(-1)
        packets = cast(ReplayPackets, _rows(replay, np.flatnonzero(valid), leading=2))
    local = before.core_state.step_count - before.initial_step_count
    evidence = ConfigEvidence(
        before.episode_id,
        jnp.where(before.done.done, -1, local),
        before.config,
    )
    count = before.episode_id.shape[0]
    empty = jnp.full((), -1, jnp.int32)
    errors = CollectionErrors(
        completed.lifecycle_error,
        jnp.zeros(count, jnp.int32),
        jnp.zeros(count, jnp.int32),
        empty,
        empty,
        empty,
        empty,
        empty,
    )
    buffers = CollectedBuffers(
        evidence,
        None,
        headers,
        _rows(completed.priority, selected[priority_mask]),
        _rows(completed.full, selected[full_mask]),
        traces,
        packets,
    )
    sizes = (
        count,
        0,
        len(selected),
        int(priority_mask.sum()),
        int(full_mask.sum()),
        0 if traces is None else traces.trace.valid.shape[0],
        0 if packets is None else packets.valid.shape[0],
    )
    return CollectedBatch(
        buffers,
        CollectedCounts(*(jnp.asarray(n, jnp.int32) for n in sizes)),
        errors,
    )
