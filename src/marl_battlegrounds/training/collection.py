"""Collect exact training decisions with curriculum, shaping and self-play.

Host setup verifies content and freezes callables. Numerical carry keeps changing
weights, games, memories and counters. Pure scan and bounded recording share one
step. A named pinned opponent plays the lanes assigned to history slot 0: a JAX
method inside the compiled step, a host method through a host loop in
collect_training_rollout. An optional ActorVariablesAtStep hook, such as QMIX
exploration, sets changing actor values from the round count before every
decision; without one the step is unchanged. No critic, optimizer, replay,
learning run or durable learner checkpoint lives here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds._method_loading import load_method

# Shared private helpers retain one owner for execution and stage accounting.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.baselines.inputs import (
    TRAINING_STATE_FEATURE_SIZE,
    encode_training_state,
)
from marl_battlegrounds.collection import _check_collection_writer, collect_rollout
from marl_battlegrounds.core.types import Action, ActionMask
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    EpisodeInfo,
    make,
)
from marl_battlegrounds.episode_tracking import (
    EpisodeTrackingState,
    _boundary_snapshot,
    init_episode_tracking,
)
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.models import canonical_digest_sha256
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTrace,
    System,
    SystemState,
    _action_keys,
    _initialization_keys,
    init_systems,
    shared_policy,
    system_inputs,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
    ordered_source_bank_identity,
    policy_description,
    tree_digest,
)
from marl_battlegrounds.evaluation.system_evaluation import (
    freeze_evaluation_method,
    prepare_evaluation_system,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training._compilation import training_compiler_options
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    TrainingContentBinding,
    pinned_opponent_evidence,
    prepare_training_content,
)
from marl_battlegrounds.training._execution import _apply_and_track, _reset_finished
from marl_battlegrounds.training.curriculum import (
    ScheduleArrays,
    TrainingProgress,
    TrainingSchedule,
    _advance_training_schedule,
    _check_training_schedule,
    _continuation_details,
    _init_training_progress,
    _make_continuation_schedule,
    _restore_continuation_schedule,
)
from marl_battlegrounds.training.distributions import (
    TRAINING_KEY_SCHEMA_VERSION,
    sample_training_configs,
    training_keys,
)
from marl_battlegrounds.training.opponents import (
    HOST_ACTIONS_SYSTEM,
    HostOpponent,
    OpponentHistory,
    _history_invalid,
    _history_shapes,
    _pinned_share,
    assign_opponents,
    init_opponent_history,
    make_opponent_system,
)
from marl_battlegrounds.training.shaping import (
    team_potential_shaping,
    team_score_delta_shaping,
    validate_shaping,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from marl_battlegrounds.evaluation.run_writer import RunWriter

type Tree = Any


class ActorVariablesAtStep(Protocol):
    """Set changing actor values, such as an exploration rate, before a decision.

    A learner that needs a per-decision value supplies one of these to
    init_training_collection. Collection calls it once per real round, inside
    the compiled step, on the shared current actor variables only. Team A and
    every current Team B lane then act with the returned tree. Historical
    snapshots and a pinned opponent keep their own values, and the carry keeps
    the stored variables. Implementations must be pure JAX: same tree
    structure, shapes and dtypes in and out, no random keys, no host work.
    Keep one instance for the run; it is part of the static collection.
    """

    def __call__(self, variables: Tree, completed_rounds: Array) -> Tree:
        """Return the variables to use for the next decision.

        Parameters
        ----------
        variables : PyTree
            history.current_variables as stored in the carry.
        completed_rounds : Array
            Int32 0-d real rounds completed before this decision (not
            multiplied by the number of games).

        Returns
        -------
        PyTree
            A tree with the same structure, shapes and dtypes.
        """
        ...

    @property
    def identity(self) -> dict[str, object]:
        """JSON record saved with checkpoints and compared on restore."""
        ...


class TrainingCarry(NamedTuple):
    """Carry one complete numerical collection boundary without a learner.

    env/observations/state match the true last successor, before a pending reset.
    root_key is Threefry. memory holds separate teams; tracking owns the only
    source bank. source_indices (B,) and source_class_ids (B,10) describe live
    games. schedule and progress count rounds; history owns all actor variables.
    discount/coefficient are validated scalar float32 values. pinned_opponent
    is () unless a named System is pinned: then (variables, memory template)
    for a JAX method, or (this step's host actions, ()) for a host method,
    whose own memory stays on the collection's HostOpponent. No host evidence,
    critic, optimizer, writer or expanded actor features enters this tree.
    """

    env: Environment
    observations: Observations
    state: EnvironmentState
    root_key: Array
    memory: SystemState
    tracking: EpisodeTrackingState
    source_indices: Array
    source_class_ids: Array
    schedule: ScheduleArrays
    progress: TrainingProgress
    history: OpponentHistory
    discount: Array
    coefficient: Array
    pinned_opponent: Tree = ()


class TrainingTransition(NamedTuple):
    """Keep compact facts from one native B-lane decision; scan adds axis T.

    observations and Team A action_mask describe the pre-action epoch. actions
    retain native (B,10) heads; learning_outputs are Team A's same-call tree.
    task_rewards is float32 (B,5); shaping_reward is one float32 value per game.
    active/alive are bool (B,5); episode_start/ended/valid are bool (B,).
    Completion outcome/length (B,) and final_scores (B,2) are zero unless ended.
    priority is existing MetricValues, masked to real endings, or None.
    Int32 identity/version fields are (B,); requested_stage and episode_stage
    name original distribution slots, including in child segments. The former
    is the latest reset request; the latter names the producing game's reset.
    opponent_snapshot=-1 means current.
    opponent_update is the learner version the opponent's weights come from;
    for games against a named pinned System (opponent_snapshot 0 when the
    collection has one) it is -2, because that System is not a learner version.
    training_state is float32 (B,920) or None and never enters an actor.
    Invalid padding has neutral-only masks, zero payloads and -1 identities.
    """

    observations: Observations
    action_mask: ActionMask
    actions: Action
    learning_outputs: Tree
    task_rewards: Array
    shaping_reward: Array
    active: Array
    alive: Array
    episode_start: Array
    ended: Array
    valid: Array
    outcome: Array
    episode_length: Array
    final_scores: Array
    priority: MetricValues | None
    episode_id: Array
    decision_step: Array
    requested_stage: Array
    episode_stage: Array
    source_index: Array
    learner_update: Array
    opponent_update: Array
    opponent_snapshot: Array
    training_state: Array | None


class TrainingRollout(NamedTuple):
    """Return fixed-length transitions and the true final bootstrap boundary.

    transitions has leading (T,B). initial_memory is Team A's block-entry tree;
    the first real row's episode_start handles a pending episode reset. Final
    observations/masks and active/alive/ended belong to the returned carry before
    reset, never to padding. final_training_state is optional (B,920).
    real_steps is scalar int32 rounds, not B-multiplied experience. Zero means
    unchanged carry and no learning/bootstrap work. Critic memory is external.
    """

    transitions: TrainingTransition
    initial_memory: Tree
    final_observations: Observations
    final_action_mask: ActionMask
    final_active: Array
    final_alive: Array
    final_ended: Array
    final_training_state: Array | None
    real_steps: Array


@dataclass(frozen=True, eq=False)
class TrainingCollection:
    """Keep stable host configuration beside numerical TrainingCarry.

    actor/opponent contain callables, not changing weights. binding is verified
    content evidence; schedule is the host rounding report. shaping, projection,
    shaping_mode, metrics and recording are static choices. shaping_mode is
    "potential" by default or the explicit "score_delta" combat objective.
    learning_spec/info_spec contain
    shape-only output trees used for finite padding and failure diagnostics.
    schedule also holds the immutable parent proof for a child segment.
    carry_spec records shapes and static settings without retaining arrays;
    root_bits, key_schema and reward_settings bind in-memory continuation.
    pinned_opponent_share is the static probability that a reset lane meets the
    pinned first-update actor in history slot 0; zero keeps the 80/20 recipe.
    pinned_opponent is None, or the JSON record of a named System that plays
    slot-0 lanes instead (reference, name, execution, registration and its ID,
    variables digest, evidence, memory rule). host_opponent is the mutable
    HostOpponent of a pinned host method, else None; a collection that has one
    serves each training round once and must not be replayed.
    actor_variables_at_step is None (the default, used by PPO and PQN-VDN,
    whose rate changes only between blocks), or the learner's
    ActorVariablesAtStep hook, which sets values such as QMIX's exploration
    rate on the shared current variables before each decision.
    Reuse this descriptor across blocks to reuse compiled functions. It is not
    a PyTree or a durable checkpoint and must stay outside numerical carry.
    """

    actor: System
    opponent: System
    binding: TrainingContentBinding
    schedule: TrainingSchedule
    shaping: bool
    collect_training_state: bool
    metrics: str
    recording: bool
    learning_spec: Tree
    info_spec: EpisodeInfo
    carry_spec: Tree
    root_bits: tuple[int, ...]
    key_schema: int
    reward_settings: tuple[float, float]
    shaping_mode: str = "potential"
    pinned_opponent_share: float = 0.0
    pinned_opponent: dict[str, Any] | None = None
    host_opponent: HostOpponent | None = None
    actor_variables_at_step: ActorVariablesAtStep | None = None


def _zeros(tree: Tree) -> Tree:
    """Allocate zero leaves from a numerical tree or shape-only specification."""

    def leaf(value: Tree) -> Array:
        """Create one array from its shape/dtype without reading any values."""
        return jnp.zeros(value.shape, value.dtype)

    return jax.tree.map(leaf, tree)


def _length(value: int) -> int:
    """Validate a static nonnegative Python length within the int32 loop limit."""
    if type(value) is not int or not 0 <= value <= np.iinfo(np.int32).max:
        raise ValueError("length must be a nonnegative Python integer within int32")
    return value


def _team_mask(state: EnvironmentState) -> ActionMask:
    """Select Team A's native mask rows without expanding private actor inputs."""

    def rows(value: Array) -> Array:
        """Select the five Team A slots on the native actor axis."""
        return value[:, :5]

    return cast(ActionMask, jax.tree.map(rows, state.action_mask))


def _neutral_mask(mask: ActionMask) -> ActionMask:
    """Build computational padding admitting only Stay/None/no Ultimate."""
    empty = cast(ActionMask, _zeros(mask))
    return ActionMask(
        empty.move_mask.at[..., 0].set(True),
        empty.select_target_mask.at[..., 0].set(True),
        empty.use_ultimate_mask.at[..., 0].set(True),
        empty.select_target_use_ultimate_joint_mask.at[..., 0, 0].set(True),
    )


def _failed(carry: TrainingCarry) -> Array:
    """Test sticky environment, tracker and history errors across the batch."""
    return (
        jnp.any(carry.state.lifecycle_error)
        | jnp.any(carry.tracking.error_flags != 0)
        | _history_invalid(carry.history)
        | (carry.history.last_refresh_rounds > carry.progress.rounds)
    )


def _uncommitted(leaf: Tree) -> Tree:
    """Return a device-committed array as an uncommitted copy; keep other leaves.

    Parameters
    ----------
    leaf : Any
        One leaf of a pinned System's variables or memory template.

    Returns
    -------
    Any
        For a jax.Array committed to a device, an equal array (same dtype,
        shape and values; typed keys stay typed) on the default device with no
        commitment, like every other array of a fresh TrainingCarry. Any other
        leaf, including NumPy arrays and uncommitted arrays, unchanged.

    Notes
    -----
    Host-only, called once at setup. Copies the leaf through the host.
    """
    if not isinstance(leaf, jax.Array) or not leaf.committed:
        return leaf
    host = jax.device_get(leaf)
    return host if isinstance(host, jax.Array) else jnp.asarray(host)


def _check_carry(carry: TrainingCarry) -> None:
    """Reject a failed numerical boundary on the host before reporting success."""
    if bool(np.asarray(_failed(carry))):
        raise ValueError(
            "Training collection has a lifecycle, tracking or opponent error"
        )


def _retain(carry: TrainingCarry) -> TrainingCarry:
    """Keep all numerical fields unchanged in a skipped reset branch."""
    return carry


def _validate_training_continuation(  # pyright: ignore[reportUnusedFunction]
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    expected_root_bits: tuple[int, ...] | None = None,
    recheck_installed_content: bool = True,
) -> None:
    """Check an immutable in-memory continuation before reopening its writer.

    By default, reverify installed content against the descriptor. Periodic
    learner saves may set recheck_installed_content=False to reuse the descriptor
    verified at setup or restore. Both routes check the actual carried bank,
    fixed settings, key schema/root, numerical structure and paired counters.
    expected_root_bits optionally adds the caller's saved root identity. This
    host check reads files and small arrays but changes no state or recording.
    It raises ValueError on incompatible evidence. It does not authenticate
    arbitrary objects, detect a discarded matching branch or implement durable
    learner resume. Preserve descriptor and carry together, with lane order
    unchanged. Packet 4 owns serializing and validating a full learner checkpoint.
    """
    _check_training_schedule(collection.schedule)
    if recheck_installed_content:
        prepare_training_content(expected=collection.binding)
    if carry.tracking.source_configs is None or (
        ordered_source_bank_identity(carry.tracking.source_configs)[0]
        != collection.binding.source_bank.canonical_digest
    ):
        raise ValueError("Continuation source bank does not match verified content")
    if jax.tree.structure(carry) != jax.tree.structure(collection.carry_spec):
        raise ValueError("Continuation structure or static settings changed")
    for leaf, spec in zip(
        jax.tree.leaves(carry), jax.tree.leaves(collection.carry_spec), strict=True
    ):
        if leaf.shape != spec.shape or leaf.dtype != spec.dtype:
            raise ValueError("Continuation array shapes or dtypes changed")
    if (
        collection.key_schema != TRAINING_KEY_SCHEMA_VERSION
        or str(jax.random.key_impl(carry.root_key)) != "threefry2x32"
        or tuple(int(x) for x in np.asarray(jax.random.key_data(carry.root_key)))
        != collection.root_bits
        or (
            expected_root_bits is not None
            and expected_root_bits != collection.root_bits
        )
    ):
        raise ValueError("Continuation random root or key schema changed")
    for actual, expected in zip(
        carry.schedule, collection.schedule.arrays, strict=True
    ):
        if actual is None or expected is None:
            matches = actual is expected
        else:
            matches = np.array_equal(actual, expected)
        if not matches:
            raise ValueError("Continuation schedule changed")
    if (float(carry.discount), float(carry.coefficient)) != collection.reward_settings:
        raise ValueError("Continuation reward settings changed")
    if (
        carry.tracking.num_envs != collection.schedule.num_envs
        or carry.tracking.record_starts != collection.recording
        or _history_shapes(carry.history) != collection.schedule.num_envs
    ):
        raise ValueError("Continuation batch or recording settings changed")
    _check_carry(carry)
    snapshot = _boundary_snapshot(carry.tracking, carry.state)
    rounds = int(carry.progress.rounds)
    offset = (
        0 if carry.schedule.round_offset is None else int(carry.schedule.round_offset)
    )
    distribution_offset = (
        0
        if carry.schedule.distribution_offset is None
        else int(carry.schedule.distribution_offset)
    )
    distributions = int(
        carry.schedule.stage_count
        if carry.schedule.distribution_count is None
        else carry.schedule.distribution_count
    )
    if not offset <= rounds <= int(carry.schedule.total_rounds) or not np.all(
        snapshot["accounted"] == rounds
    ):
        raise ValueError("Continuation progress and tracked experience disagree")
    for owner in (carry.tracking, carry.memory):
        if not np.array_equal(
            owner.episode_id, carry.state.episode_id
        ) or not np.array_equal(owner.reset_generation, carry.state.reset_generation):
            raise ValueError("Continuation memory or tracking episode bindings changed")
    if not np.array_equal(
        carry.source_indices, carry.tracking.source_index
    ) or not np.array_equal(carry.source_class_ids, carry.tracking.source_class_ids):
        raise ValueError("Continuation source declarations changed")
    indices = np.asarray(carry.source_indices)
    bank_thresholds = np.asarray(
        carry.tracking.source_configs.team_deathmatch_score_threshold
    )
    if np.any(indices < 0) or np.any(indices >= len(bank_thresholds)):
        raise ValueError("Continuation source indices are outside the verified bank")
    actual_thresholds = np.asarray(carry.state.config.team_deathmatch_score_threshold)
    episode_stages = np.asarray(carry.progress.episode_stage)
    if np.any(episode_stages < 0) or np.any(episode_stages >= distributions):
        raise ValueError("Continuation episode stages are outside the schedule")
    if not np.array_equal(
        actual_thresholds, bank_thresholds[indices]
    ) or not np.array_equal(
        actual_thresholds, np.asarray(carry.schedule.score_thresholds)[episode_stages]
    ):
        raise ValueError(
            "Continuation score thresholds disagree with their episode sources"
        )
    count = collection.schedule.num_envs
    expected_spawns = np.concatenate((np.zeros(count // 2), np.ones(count // 2)))
    if not np.array_equal(carry.tracking.spawn_locations, expected_spawns):
        raise ValueError("Continuation fixed spawn lane order changed")
    progress = jax.device_get(carry.progress)
    stages = int(carry.schedule.stage_count)
    budgets = np.asarray(carry.schedule.round_budgets)
    ends = np.asarray(carry.schedule.round_ends)
    stage = min(int(np.searchsorted(ends[:stages], rounds, side="right")), stages - 1)
    stage_rounds = rounds - (int(ends[stage - 1]) if stage else offset)
    proof = np.zeros((17, count, 3), np.int64)
    complete = (np.arange(17) < stages) & (ends <= rounds)
    for index in range(stages):
        if complete[index]:
            proof[index, : count // 2, 0] = int(budgets[index])
            proof[index, count // 2 :, 1] = int(budgets[index])
    current = np.zeros((count, 3), np.int64)
    current[: count // 2, 0] = stage_rounds
    current[count // 2 :, 1] = stage_rounds
    if (
        int(snapshot["ordinal"]) != stage
        or int(snapshot["budget"]) != int(budgets[stage])
        or int(snapshot["rounds"]) != stage_rounds
        or not bool(snapshot["full"])
        or not np.array_equal(snapshot["counts"], current)
        or not np.array_equal(progress.stage_complete, complete)
        or not np.array_equal(progress.completed_stage_counts, proof)
    ):
        raise ValueError("Continuation stage proof disagrees with its exact budget")
    for counters in (progress.exposure, progress.map_steps, progress.opponent_steps):
        if np.any(counters < 0) or any(
            sum(int(x) for x in counters[:, lane]) != rounds for lane in range(count)
        ):
            raise ValueError("Continuation exposure counters disagree with experience")
    if (
        np.any(progress.starts < 0)
        or np.any(progress.starts > progress.exposure)
        or np.any(progress.actor_decisions < 0)
        or np.any(progress.actor_decisions > progress.exposure[..., None])
        or np.any(progress.opponent_starts < 0)
        or np.any(progress.opponent_starts > progress.opponent_steps)
        or np.any(progress.episode_stage < 0)
        or np.any(progress.episode_stage > distribution_offset + stage)
        or any(
            sum(int(x) for x in progress.starts[:, lane])
            != sum(int(x) for x in progress.opponent_starts[:, lane])
            for lane in range(count)
        )
    ):
        raise ValueError("Continuation episode exposure or decision counts are invalid")
    history_count = int(carry.history.count)
    update = int(carry.history.current_update)
    refreshed = int(carry.history.last_refresh_rounds)
    captured_rounds, captured_updates, mapping = jax.device_get(
        (
            carry.history.captured_rounds,
            carry.history.captured_updates,
            carry.history.threshold_to_snapshot,
        )
    )
    used_rounds = captured_rounds[:history_count]
    used_updates = captured_updates[:history_count]
    threshold_count = (
        20
        if carry.schedule.history_threshold_count is None
        else int(carry.schedule.history_threshold_count)
    )
    active_thresholds = np.arange(20) < threshold_count
    expected_mapping = np.searchsorted(
        used_rounds, np.asarray(carry.schedule.history_threshold_rounds), side="left"
    )
    expected_mapping = np.where(
        active_thresholds & (expected_mapping < history_count), expected_mapping, -1
    )
    if (
        not 0 <= history_count <= update <= refreshed <= rounds
        or ((update == 0) != (refreshed == 0))
        or np.any(used_rounds <= 0)
        or np.any(used_rounds > refreshed)
        or np.any(np.diff(used_rounds) <= 0)
        or np.any(used_updates <= 0)
        or np.any(used_updates > update)
        or np.any(np.diff(used_updates) <= 0)
        or np.any(captured_rounds[history_count:] != -1)
        or np.any(captured_updates[history_count:] != -1)
        or not np.array_equal(mapping, expected_mapping)
        or set(int(x) for x in mapping if x >= 0) != set(range(history_count))
        or np.any(
            (
                active_thresholds
                & (np.asarray(carry.schedule.history_threshold_rounds) <= refreshed)
            )
            != (mapping >= 0)
        )
    ):
        raise ValueError("Continuation opponent snapshot metadata is inconsistent")


def _continuation_history_thresholds(  # pyright: ignore[reportUnusedFunction]
    carry: TrainingCarry, *, future_rounds: tuple[int, ...]
) -> tuple[int, ...]:
    """Keep saved snapshots and pending captures while adding explicit requests.

    carry must already pass its parent's collection check. future_rounds contains
    strictly increasing Python integers after the actual checkpoint round; the
    runner checks them against the child's declared end and learner boundaries.
    Return an ordered active threshold tuple with one marker per existing slot,
    all distinct pending old thresholds and the added points. Passed duplicate
    requests are retained in the archived parent proof, not copied into new
    active slots. Reject invalid points or more than 20 required table entries.
    Host-only: reads small history arrays and changes no numerical state.
    """
    rounds = int(carry.progress.rounds)
    if (
        not isinstance(cast(object, future_rounds), tuple)
        or any(
            type(value) is not int or not rounds < value <= np.iinfo(np.int32).max
            for value in future_rounds
        )
        or tuple(sorted(set(future_rounds))) != future_rounds
    ):
        raise ValueError(
            "Future captures must be ordered distinct rounds after the checkpoint"
        )
    count = int(carry.history.count)
    active = (
        20
        if carry.schedule.history_threshold_count is None
        else int(carry.schedule.history_threshold_count)
    )
    captured = tuple(
        int(value) for value in np.asarray(carry.history.captured_rounds)[:count]
    )
    thresholds = np.asarray(carry.schedule.history_threshold_rounds)[:active]
    mapping = np.asarray(carry.history.threshold_to_snapshot)[:active]
    pending = {int(value) for value in thresholds[mapping == -1]}
    combined = (*captured, *sorted(pending | set(future_rounds)))
    if len(combined) > 20:
        raise ValueError(
            "Added captures cannot fit the 20 history slots and pending requests"
        )
    return combined


def _begin_training_segment(  # pyright: ignore[reportUnusedFunction]
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    schedule: TrainingSchedule,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Start a declared child budget without taking a decision or resetting a game.

    Parameters
    ----------
    collection, carry : TrainingCollection, TrainingCarry
        Exact parent descriptor and latest complete collection boundary. The
        caller must already have restored and validated the full learner, installed
        content and checkpoint identity, and kept that parent unchanged.
    schedule : TrainingSchedule
        Pending child schedule from curriculum._make_continuation_schedule at
        this carry's cumulative rounds. An optional learner declaration is kept.
        Explicit history thresholds may regroup passed duplicate requests, but
        every existing snapshot must keep a marker and no slot may be removed.

    Returns
    -------
    tuple[TrainingCollection, TrainingCarry]
        Child descriptor with an immutable parent stage proof, and its checked
        carry. Only schedule, local stage proof, current tracker accounting and
        an explicitly changed threshold mapping differ. Cumulative experience,
        exposure, games, observations, memories, random keys and frozen history
        variables/capture identities are unchanged. Carry both returned values.

    Raises
    ------
    ValueError
        Parent validation fails, the declaration describes another boundary,
        a history change drops old snapshots, or the child accounting is invalid.

    Notes
    -----
    Host-only. Reuses the installed-content check from restore and checks the
    carried source bank, structure, keys and counters again. Opens
    no writer, consumes no key, initializes no System and performs no transition.
    A device-committed parent keeps that placement for every replacement leaf;
    existing leaves on that device are reused. An uncommitted parent stays
    uncommitted. The runner binds source/run identity and recording ancestry.
    """
    _validate_training_continuation(collection, carry, recheck_installed_content=False)
    declaration = _continuation_details(schedule)
    if declaration is None:
        raise ValueError("A new training segment requires a continuation schedule")
    rounds = int(carry.progress.rounds)
    active = (
        20
        if schedule.arrays.history_threshold_count is None
        else int(schedule.arrays.history_threshold_count)
    )
    thresholds = tuple(
        int(value) for value in np.asarray(schedule.arrays.history_threshold_rounds)
    )
    same_history = np.array_equal(
        schedule.arrays.history_threshold_rounds,
        carry.schedule.history_threshold_rounds,
    ) and (
        (
            None
            if schedule.arrays.history_threshold_count is None
            else int(schedule.arrays.history_threshold_count)
        )
        == (
            None
            if carry.schedule.history_threshold_count is None
            else int(carry.schedule.history_threshold_count)
        )
    )
    expected = _make_continuation_schedule(
        collection.schedule,
        completed_rounds=rounds,
        additional_env_steps=declaration["additional_env_steps"],
        history_threshold_rounds=None if same_history else thresholds[:active],
    )
    expected_details = _continuation_details(expected)
    assert expected_details is not None
    if any(
        declaration.get(key) != value
        for key, value in expected_details.items()
        if key != "parent_stage_proof"
    ):
        raise ValueError("Child schedule differs from the actual parent boundary")
    if (
        jax.tree.structure(schedule.arrays) != jax.tree.structure(expected.arrays)
        or any(
            not np.array_equal(left, right)
            for left, right in zip(
                jax.tree.leaves(schedule.arrays),
                jax.tree.leaves(expected.arrays),
                strict=True,
            )
        )
        or dict(schedule.rounding_report) != dict(expected.rounding_report)
    ):
        raise ValueError("Child schedule arrays differ from its declaration")
    progress = jax.device_get(carry.progress)
    snapshot = _boundary_snapshot(carry.tracking, carry.state)
    proof = {
        "rounds": rounds,
        "stage_ordinal": int(snapshot["ordinal"]),
        "stage_round_budget": int(snapshot["budget"]),
        "stage_rounds": int(snapshot["rounds"]),
        "stage_counts": snapshot["counts"].tolist(),
        "full_batch_rounds_valid": bool(snapshot["full"]),
        "completed_stage_counts": np.asarray(progress.completed_stage_counts).tolist(),
        "stage_complete": np.asarray(progress.stage_complete).tolist(),
        "history_threshold_to_snapshot": np.asarray(
            carry.history.threshold_to_snapshot
        ).tolist(),
    }
    if (
        declaration["parent_stage_proof"] is not None
        and declaration["parent_stage_proof"] != proof
    ):
        raise ValueError("Child parent stage proof differs from the actual carry")
    declaration["parent_stage_proof"] = proof
    schedule = _restore_continuation_schedule(declaration)
    history = carry.history
    if not same_history:
        count = int(history.count)
        captures = np.asarray(history.captured_rounds)[:count]
        mapping = np.searchsorted(
            captures, np.asarray(thresholds[:active]), side="left"
        )
        mapping = np.where(mapping < count, mapping, -1).astype(np.int32)
        if set(int(value) for value in mapping if value >= 0) != set(range(count)):
            raise ValueError("Child history requests must retain every saved snapshot")
        history = history._replace(
            threshold_to_snapshot=jnp.asarray(
                np.pad(mapping, (0, 20 - active), constant_values=-1)
            )
        )
    changed = carry._replace(
        schedule=schedule.arrays,
        progress=carry.progress._replace(
            completed_stage_counts=jnp.zeros_like(
                carry.progress.completed_stage_counts
            ),
            stage_complete=jnp.zeros_like(carry.progress.stage_complete),
        ),
        tracking=replace(
            carry.tracking,
            stage_ordinal=jnp.int32(0),
            stage_round_budget=schedule.arrays.round_budgets[0],
            stage_rounds=jnp.int32(0),
            stage_counts=jnp.zeros_like(carry.tracking.stage_counts),
            full_batch_rounds_valid=jnp.bool_(True),
        ),
        history=history,
    )
    if carry.root_key.committed:
        # A mixed placement would compile again after the first block commits it.
        changed = jax.device_put(changed, carry.root_key.sharding)
    child = replace(
        collection, schedule=schedule, carry_spec=jax.eval_shape(_retain, changed)
    )
    _validate_training_continuation(child, changed, recheck_installed_content=False)
    return child, changed


def init_training_collection(
    actor: System,
    actor_variables: Tree,
    *,
    schedule: TrainingSchedule,
    seed: int = 42,
    prepared: PreparedTrainingContent | None = None,
    shaping: bool = False,
    shaping_mode: str = "potential",
    discount: float = 0.99,
    coefficient: float = 0.01,
    collect_training_state: bool = False,
    metrics: str = "priority",
    recording: bool = False,
    pinned_opponent_share: float = 0.0,
    pinned_opponent: System | Policy | str | None = None,
    actor_variables_at_step: ActorVariablesAtStep | None = None,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Verify and initialize one fixed-batch experiment without choosing actions.

    Parameters
    ----------
    actor, actor_variables
        Native lane-independent JAX System and matching actor-only variable
        tree. Explicit arrays override descriptor values. Frozen checkpoint
        labels, Policy adapters, host methods and custom reset hooks fail.
    schedule : TrainingSchedule
        Host result of make_training_schedule. Its positive even batch and
        exact whole-batch budget remain fixed for this collection.
    seed : int, default 42
        Python integer for the one Threefry root; bool is rejected.
    prepared : PreparedTrainingContent or None
        Existing verified preparation, or None to verify installed content now.
        Supplied bank bytes must match its immutable binding. No external bank
        or synthetic horizon is admitted by this public setup.
    shaping : bool, default False
        Enable team feedback. Disabled work is absent from execution.
    shaping_mode : {"potential", "score_delta"}, default="potential"
        Potential feedback preserves the discounted task objective. Score-delta
        feedback adds coefficient times the team's new points minus the enemy's
        new points (a Red Zone death gives 2 points, any other death 1),
        including terminal actions. Both leave native task rewards unchanged. The mode
        is checked even when shaping is disabled and stays fixed for the run.
    discount, coefficient : float, defaults 0.99, 0.01
        Learner discount in [0,1] and nonnegative finite shaping scale.
    collect_training_state : bool, default False
        Retain the separate physical feature view once per game/transition.
    metrics : {"priority", "none"}, default "priority"
        Completed priority results or no metric work/storage.
    recording : bool, default False
        Enable start declarations for a writer supplied at every host collection.
    pinned_opponent_share : float, default 0.0
        Static probability, within [0, 0.8], that a reset lane meets the pinned
        first-update actor kept in history slot 0. Zero keeps the 80% current
        and 20% uniform-history recipe and traces today's program. A positive
        share requires schedule.early_history_capture=True, and that flag
        requires a positive share; the pair is checked here before any reset.
    pinned_opponent : System, Policy, str or None, default None
        None keeps slot 0 as the network's first-update snapshot, exactly as
        before. Otherwise the method that plays every lane assigned to slot 0,
        instead of that snapshot: a System or Policy object, or a reference
        string resolved by ``load_method`` (a built-in name, an absolute
        actor-export or full learner checkpoint directory, or ``module:function``).
        Requires a positive
        pinned_opponent_share. It is frozen once with M8's
        ``freeze_evaluation_method`` and registered the way evaluation and
        validation register it. A JAX method runs inside the compiled step,
        with its variables and memory template in the carry as uncommitted
        arrays like the rest of the fresh carry, so the first blocks compile
        once; a host method runs through the host loop of
        collect_training_rollout and its memory is never saved. An
        independent_policies System must match every Team B roster size in the
        schedule. Its training history is recorded by
        ``pinned_opponent_evidence``; nothing is refused for missing history.
    actor_variables_at_step : ActorVariablesAtStep or None, default None
        None keeps today's step exactly. Otherwise a learner hook, such as
        ``baselines.qmix.QMIXExploration``, that collection calls before every
        real decision with the rounds completed so far, on the shared current
        variables only (see ActorVariablesAtStep). Setup checks with
        ``jax.eval_shape`` that it keeps the variable tree, shapes and dtypes,
        and that its identity is a JSON object; it runs no real decision.

    Returns
    -------
    tuple[TrainingCollection, TrainingCarry]
        Stable host descriptor and complete numerical state. Setup reads files
        when preparing content and, for a string pinned_opponent, when loading
        an actor export/checkpoint (and an export's sibling learner description),
        or importing and
        calling a factory once. It places arrays, initializes memory (including
        the pinned method's placeholder memory, with every game marked invalid)
        and traces output shapes. It creates no writer and executes no actor
        decision.

    Raises
    ------
    TypeError, ValueError
        Unsupported settings, content/bank mismatch, variable structure mismatch,
        a pinned share outside its contract or disagreeing with the schedule, a
        pinned opponent without a positive share, an unresolvable reference, an
        independent roster that the schedule would break, a variables hook that
        changes the tree or has no JSON identity, or failed initialization.
        Errors from a factory propagate with their own type.
        Failures precede any real transition.
    """
    if not isinstance(cast(object, actor), System):
        raise TypeError("actor must be a native JAX System")
    if not isinstance(cast(object, schedule), TrainingSchedule):
        raise TypeError("schedule must come from make_training_schedule")
    if schedule.continuation is not None:
        raise ValueError(
            "Initialize an original schedule, then restore or start its child segment"
        )
    _check_training_schedule(schedule)
    if type(seed) is not int or any(
        type(x) is not bool for x in (shaping, collect_training_state, recording)
    ):
        raise TypeError("seed must be an integer and collection switches must be bool")
    if metrics not in ("none", "priority"):
        raise ValueError("training metrics must be priority or none")
    validate_shaping(discount=discount, coefficient=coefficient, mode=shaping_mode)
    _pinned_share(pinned_opponent_share)
    if (pinned_opponent_share > 0) != schedule.early_history_capture:
        raise ValueError(
            "A positive pinned_opponent_share requires early_history_capture in the "
            "schedule, and early capture requires a positive share"
        )
    if actor.checkpoint is not None:
        raise ValueError(
            "An evolving training actor cannot keep a frozen checkpoint label"
        )
    if jax.tree.structure(actor.variables) != jax.tree.structure(actor_variables):
        raise ValueError("actor_variables must match the System variable structure")
    for a, b in zip(
        jax.tree.leaves(actor.variables), jax.tree.leaves(actor_variables), strict=True
    ):
        if getattr(a, "shape", None) != getattr(b, "shape", None) or getattr(
            a, "dtype", None
        ) != getattr(b, "dtype", None):
            raise ValueError(
                "actor_variables must match the System leaf shapes and dtypes"
            )
    if actor_variables_at_step is not None:
        _check_variables_hook(actor_variables_at_step, actor_variables)
    history = init_opponent_history(actor_variables, num_envs=schedule.num_envs)
    if prepared is None:
        prepared = prepare_training_content(score_thresholds=schedule.score_thresholds)
    elif not isinstance(cast(object, prepared), PreparedTrainingContent):
        raise TypeError("prepared must be verified PreparedTrainingContent")
    elif (
        ordered_source_bank_identity(prepared.source_configs)[0]
        != prepared.binding.source_bank.canonical_digest
    ):
        raise ValueError(
            "Prepared source bank does not match its verified content identity"
        )
    if prepared.binding.score_thresholds != schedule.score_thresholds:
        raise ValueError("Prepared score thresholds differ from the training schedule")
    pinned_record: dict[str, Any] | None = None
    host: HostOpponent | None = None
    pinned_values: Tree = ()
    pinned_system: System | None = None
    if pinned_opponent is not None:
        if not pinned_opponent_share > 0:
            raise ValueError("A pinned opponent needs a positive pinned_opponent_share")
        reference = pinned_opponent if isinstance(pinned_opponent, str) else None
        method = (
            load_method(pinned_opponent)
            if isinstance(pinned_opponent, str)
            else pinned_opponent
        )
        frozen = freeze_evaluation_method(method)
        system = shared_policy(frozen) if isinstance(frozen, Policy) else frozen
        if system._policies and not system._shared:
            stages = int(schedule.arrays.stage_count)
            sizes = {
                int(size) for size in np.asarray(schedule.arrays.team_sizes)[:stages]
            }
            if sizes != {len(system._policies)}:
                raise ValueError(
                    "An independent_policies pinned opponent needs every Team B "
                    f"roster size to equal its {len(system._policies)} Policies; "
                    f"the schedule uses {sorted(sizes)}"
                )
        registration = (
            policy_description(
                frozen, frozen.variables, frozen.initial_carry, include_digests=True
            )
            if isinstance(frozen, Policy)
            else normalize_system_registration(frozen, phase="validation", frozen=True)[
                1
            ]
        )
        export = (
            Path(reference)
            if reference is not None and Path(reference).is_dir()
            else None
        )
        if system.execution == "host":
            host = HostOpponent(system, schedule.num_envs)
            pinned_system = HOST_ACTIONS_SYSTEM
            zero = jnp.zeros((schedule.num_envs, 5), jnp.int32)
            pinned_values = (ActorAction(zero, zero, zero), ())
            memory_rule = "host only, not saved" if host.stateful else "none"
            variables_digest = None
        else:
            _, variables, template = prepare_evaluation_system(frozen)
            pinned_system = system
            # A loaded export's weights arrive committed to one device while
            # the rest of the fresh carry is not; that mix makes the rollout
            # and the update compile twice. Give them the carry's placement.
            pinned_values = jax.tree.map(_uncommitted, (variables, template))
            memory_rule = "saved in the carry"
            variables_digest = tree_digest(variables)
        # JSON round trip: checkpoints compare this record with its saved JSON.
        pinned_record = json.loads(
            json.dumps(
                {
                    "reference": reference,
                    "name": system.name,
                    "execution": system.execution,
                    "registration_id": canonical_digest_sha256(registration),
                    "registration": registration,
                    "variables_digest": variables_digest,
                    "evidence": pinned_opponent_evidence(
                        prepared.binding, frozen, export=export
                    ),
                    "memory_rule": memory_rule,
                }
            )
        )
    opponent = make_opponent_system(actor, pinned=pinned_system)
    root = jax.random.key(seed, impl="threefry2x32")
    generation = jnp.zeros(schedule.num_envs, jnp.int32)
    sampled = sample_training_configs(
        prepared.source_configs,
        root,
        generation,
        eligible_maps=schedule.arrays.eligible_maps[0],
        team_size=schedule.arrays.team_sizes[0],
        score_threshold=schedule.arrays.score_thresholds[0],
    )
    env = make(
        "tdm",
        env_config=sampled.config,
        num_envs=schedule.num_envs,
        metrics=cast(Any, metrics),
    )
    observations, state = env.reset(training_keys(root, generation, stream="reset"))
    if bool(np.asarray(jnp.any(state.lifecycle_error) | _history_invalid(history))):
        raise ValueError("Training reset or opponent initialization failed")
    actor = replace(actor, variables=())
    memory = init_systems(
        actor,
        opponent,
        observations,
        state,
        training_keys(root, generation, stream="initialization"),
        variables_a=actor_variables,
        variables_b=history if pinned_system is None else (history, *pinned_values),
    )
    if host is not None:
        host.start(
            jax.device_get(system_inputs(observations, state, team=1)),
            _initialization_keys(memory.init_key, state.episode_id, 1),
        )
    tracking = init_episode_tracking(
        env,
        state,
        source_configs=prepared.source_configs,
        source_indices=sampled.source_indices,
        source_class_ids=sampled.source_class_ids,
        record_starts=recording,
    ).begin_stage(
        state, total_env_steps=int(schedule.arrays.round_budgets[0]) * schedule.num_envs
    )
    carry = TrainingCarry(
        env,
        observations,
        state,
        root,
        memory,
        tracking,
        sampled.source_indices,
        sampled.source_class_ids,
        schedule.arrays,
        _init_training_progress(num_envs=schedule.num_envs),
        history,
        jnp.asarray(discount, jnp.float32),
        jnp.asarray(coefficient, jnp.float32),
        pinned_values,
    )
    _check_carry(carry)
    shape = jax.eval_shape(partial(_apply, actor=actor, opponent=opponent), carry)
    collection = TrainingCollection(
        actor,
        opponent,
        prepared.binding,
        schedule,
        shaping,
        collect_training_state,
        metrics,
        recording,
        shape[2][0],
        shape[4][4],
        jax.eval_shape(_retain, carry),
        tuple(int(x) for x in np.asarray(jax.random.key_data(root))),
        TRAINING_KEY_SCHEMA_VERSION,
        (float(carry.discount), float(carry.coefficient)),
        shaping_mode=shaping_mode,
        pinned_opponent_share=float(pinned_opponent_share),
        pinned_opponent=pinned_record,
        host_opponent=host,
        actor_variables_at_step=actor_variables_at_step,
    )
    return collection, carry


def _check_variables_hook(hook: ActorVariablesAtStep, variables: Tree) -> None:
    """Check a variables hook without running it on real values.

    hook must be callable with a JSON-object identity; variables is the actor
    tree it will receive. ``jax.eval_shape`` traces one call with an int32
    round count and requires the same tree structure, shapes and dtypes back.
    Raises TypeError or ValueError; creates no arrays.
    """
    if not callable(hook):
        raise TypeError("actor_variables_at_step must be callable")
    identity = getattr(hook, "identity", None)
    if not isinstance(identity, dict):
        raise TypeError("actor_variables_at_step needs a JSON object identity")
    json.dumps(identity, allow_nan=False, sort_keys=True)
    result = jax.eval_shape(hook, variables, jax.ShapeDtypeStruct((), jnp.int32))
    before = jax.tree.leaves(variables)
    after = jax.tree.leaves(result)
    if jax.tree.structure(result) != jax.tree.structure(variables) or any(
        getattr(a, "shape", None) != b.shape or getattr(a, "dtype", None) != b.dtype
        for a, b in zip(before, after, strict=True)
    ):
        raise ValueError(
            "actor_variables_at_step must keep the variable tree, shapes and dtypes"
        )


def _apply(carry: TrainingCarry, actor: System, opponent: System) -> Tree:
    """Supply exact action/step keys and dynamic variables to the shared step."""
    state = carry.state
    local_step = state.core_state.step_count - state.initial_step_count
    return _apply_and_track(
        carry.env,
        carry.observations,
        state,
        carry.memory,
        carry.tracking,
        actor=actor,
        opponent=opponent,
        variables_a=carry.history.current_variables,
        variables_b=carry.history
        if len(carry.pinned_opponent) == 0
        else (carry.history, *carry.pinned_opponent),
        action_keys=training_keys(
            carry.root_key,
            state.reset_generation,
            stream="action",
            decision_step=local_step,
        ),
        step_keys=training_keys(
            carry.root_key,
            state.reset_generation,
            stream="step",
            decision_step=local_step,
        ),
        source_indices=carry.source_indices,
        source_class_ids=carry.source_class_ids,
    )


def _padding(
    collection: TrainingCollection, carry: TrainingCarry
) -> TrainingTransition:
    """Create a finite invalid row without policy, environment or counter work."""
    b = carry.state.episode_id.shape[0]
    zero = jnp.zeros(b, jnp.int32)
    false = jnp.zeros(b, jnp.bool_)
    absent = jnp.full(b, -1, jnp.int32)
    actions = Action(*(jnp.zeros((b, 10), jnp.int32) for _ in range(3)))
    return TrainingTransition(
        _zeros(carry.observations),
        _neutral_mask(_team_mask(carry.state)),
        actions,
        _zeros(collection.learning_spec),
        jnp.zeros((b, 5), jnp.float32),
        jnp.zeros(b, jnp.float32),
        jnp.zeros((b, 5), jnp.bool_),
        jnp.zeros((b, 5), jnp.bool_),
        false,
        false,
        false,
        zero,
        zero,
        jnp.zeros((b, 2), jnp.int32),
        _zeros(collection.info_spec.priority),
        absent,
        absent,
        absent,
        absent,
        absent,
        absent,
        absent,
        absent,
        jnp.zeros((b, TRAINING_STATE_FEATURE_SIZE), jnp.float32)
        if collection.collect_training_state
        else None,
    )


def _failure_result(
    collection: TrainingCollection, carry: TrainingCarry
) -> tuple[TrainingCarry, tuple[TrainingTransition, EpisodeInfo, PolicyTrace]]:
    """Return sticky error-only diagnostics; never apply a compensating action.

    The bounded collector consumes this invalid diagnostic solely to raise
    before its next writer drain. Preserve failed reset state and old memory/
    counts. Ordinary budget padding uses a different route and is never recorded.
    """
    flags = carry.tracking.error_flags | jnp.where(
        _history_invalid(carry.history)
        | (carry.history.last_refresh_rounds > carry.progress.rounds),
        2,
        0,
    ).astype(jnp.int32)
    carry = carry._replace(tracking=replace(carry.tracking, error_flags=flags))
    info = cast(EpisodeInfo, _zeros(collection.info_spec))._replace(
        episode_id=carry.state.episode_id,
        config=carry.state.config,
        decision_step=jnp.full_like(carry.state.episode_id, -1),
        lifecycle_error=carry.state.lifecycle_error,
        episode_tracking_error=flags,
    )
    trace = PolicyTrace(
        carry.state.episode_id,
        jnp.full_like(carry.state.episode_id, -1),
        jnp.zeros_like(carry.state.episode_id, dtype=jnp.bool_),
        jnp.full_like(carry.memory.policy_trace.policy_ids, -1),
    )
    return carry, (_padding(collection, carry), info, trace)


def _reset_pending(
    collection: TrainingCollection, carry: TrainingCarry
) -> TrainingCarry:
    """Adopt the requested stage/history only for lanes starting a new game.

    collection supplies the static settings, here the pinned opponent share
    passed to assign_opponents; carry is the dynamic state whose finished lanes
    are reset. Returns the new carry; continuing lanes keep every value.
    """
    mask = carry.state.done.done
    stage = carry.tracking.stage_ordinal
    if carry.schedule.distribution_offset is not None:
        stage = stage + carry.schedule.distribution_offset
    assert carry.tracking.source_configs is not None
    observations, state, indices, classes = _reset_finished(
        carry.env,
        carry.observations,
        carry.state,
        carry.source_indices,
        carry.source_class_ids,
        source_configs=carry.tracking.source_configs,
        root_key=carry.root_key,
        eligible_maps=carry.schedule.eligible_maps[stage],
        team_size=carry.schedule.team_sizes[stage],
        score_threshold=carry.schedule.score_thresholds[stage],
    )
    history = assign_opponents(
        carry.history,
        mask,
        training_keys(carry.root_key, state.reset_generation, stream="opponent"),
        pinned_share=collection.pinned_opponent_share,
    )
    progress = carry.progress._replace(
        episode_stage=jnp.where(mask, stage, carry.progress.episode_stage)
    )
    return carry._replace(
        observations=observations,
        state=state,
        source_indices=indices,
        source_class_ids=classes,
        history=history,
        progress=progress,
    )


def _real_step(
    collection: TrainingCollection, carry: TrainingCarry
) -> tuple[TrainingCarry, tuple[TrainingTransition, EpisodeInfo, PolicyTrace]]:
    """Collect one already-checked decision and advance accounting afterward.

    A variables hook, when present, sets the shared current variables used for
    this one decision only; the returned carry keeps the stored variables.
    """
    before = carry.state
    acting = carry
    if collection.actor_variables_at_step is not None:
        history = carry.history
        acting = carry._replace(
            history=history._replace(
                current_variables=collection.actor_variables_at_step(
                    history.current_variables, carry.progress.rounds
                )
            )
        )
    actions, memory, learning, tracking, result = _apply(
        acting, collection.actor, collection.opponent
    )
    observations, state, rewards, _, info = result
    progress, tracking, info = _advance_training_schedule(
        carry.progress, tracking, before, state, info, schedule=carry.schedule
    )
    valid = info.decision_step >= 0
    ended = info.completed & valid
    lanes = jnp.arange(before.episode_id.shape[0])
    slot = carry.history.lane_snapshot + 1
    progress = progress._replace(
        map_steps=progress.map_steps.at[carry.source_indices % 42, lanes].add(
            valid.astype(jnp.int32)
        ),
        opponent_starts=progress.opponent_starts.at[slot, lanes].add(
            (valid & before.episode_start).astype(jnp.int32)
        ),
        opponent_steps=progress.opponent_steps.at[slot, lanes].add(
            valid.astype(jnp.int32)
        ),
    )
    priority = info.priority
    if priority is not None:
        availability = priority.valid & ended[:, None]
        priority = MetricValues(
            jnp.where(availability, priority.values, 0), availability
        )
    opponent_update = jnp.where(
        carry.history.lane_snapshot < 0,
        carry.history.current_update,
        carry.history.captured_updates[jnp.maximum(carry.history.lane_snapshot, 0)],
    )
    if collection.pinned_opponent is not None:
        # Slot 0 marks the named pinned System, not a learner version.
        opponent_update = jnp.where(
            carry.history.lane_snapshot == 0, -2, opponent_update
        )
    if not collection.shaping:
        shaping = jnp.zeros_like(rewards.rewards[:, 0])
    elif collection.shaping_mode == "potential":
        shaping = team_potential_shaping(
            before.core_state.team_deathmatch_scores,
            info,
            discount=carry.discount,
            coefficient=carry.coefficient,
        )[:, 0]
    else:
        shaping = team_score_delta_shaping(
            before.core_state.team_deathmatch_scores,
            info,
            coefficient=carry.coefficient,
        )[:, 0]
    row = TrainingTransition(
        carry.observations,
        _team_mask(before),
        actions,
        learning[0],
        rewards.rewards[:, :5],
        shaping,
        before.config.agent_profile.active_mask[:, :5],
        before.core_state.alive_mask[:, :5],
        before.episode_start,
        ended,
        valid,
        jnp.where(ended, info.outcome, 0),
        jnp.where(ended, info.episode_length, 0),
        jnp.where(ended[:, None], info.team_scores, 0),
        priority,
        info.episode_id,
        info.decision_step,
        jnp.full_like(
            info.episode_id,
            carry.tracking.stage_ordinal
            + (
                0
                if carry.schedule.distribution_offset is None
                else carry.schedule.distribution_offset
            ),
        ),
        carry.progress.episode_stage,
        carry.source_indices,
        jnp.full_like(info.episode_id, carry.history.current_update),
        opponent_update,
        carry.history.lane_snapshot,
        encode_training_state(before.core_state, before.config)
        if collection.collect_training_state
        else None,
    )
    trace = memory.policy_trace._replace(
        policy_ids=jnp.full_like(memory.policy_trace.policy_ids, -1)
    )
    memory = memory._replace(policy_trace=trace)
    return carry._replace(
        observations=observations,
        state=state,
        memory=memory,
        tracking=tracking,
        progress=progress,
    ), (row, info, trace)


def _refuse_host(collection: TrainingCollection, name: str) -> None:
    """Refuse a pure helper for a collection whose pinned opponent runs on the host.

    Raises ValueError naming the helper when collection.host_opponent is set;
    does nothing otherwise. Called at trace time, so it adds nothing to programs.
    """
    if collection.host_opponent is not None:
        raise ValueError(
            f"{name} cannot call a pinned host method; use collect_training_rollout, "
            "which runs the host route"
        )


def advance_training_step(
    collection: TrainingCollection, carry: TrainingCarry
) -> tuple[TrainingCarry, tuple[TrainingTransition, EpisodeInfo, PolicyTrace]]:
    """Take one real round, resetting finished games immediately before actions.

    collection is a stable host descriptor; carry is dynamic and must have real
    budget remaining. Errors are sticky numerical evidence: a failed reset stops
    the whole batch before System memory initialization or actor application.
    Return next carry and (transition, producing info, trace). This pure helper
    reads/writes no files. Host callers must check the returned failure state
    before learning; collect_training_rollout does that automatically. Calling
    this single-step helper after exhaustion marks an accounting error without
    advancing. Use rollout helpers for ordinary safe budget padding. A
    collection whose pinned opponent is a host method raises ValueError here,
    because this compiled step cannot call the host; use
    collect_training_rollout.
    """
    _refuse_host(collection, "advance_training_step")
    return _advance_training_step(collection, carry)


def _advance_training_step(
    collection: TrainingCollection, carry: TrainingCarry
) -> tuple[TrainingCarry, tuple[TrainingTransition, EpisodeInfo, PolicyTrace]]:
    """Take one real round; the shared body of every collection route.

    Same contract as advance_training_step, without the host refusal: the host
    loop calls it after putting the host method's actions in the carry.
    """

    def proceed(values: TrainingCarry) -> Tree:
        """Reset pending lanes, then guard again before any policy work."""
        values = cast(
            TrainingCarry,
            jax.lax.cond(
                jnp.any(values.state.done.done),
                partial(_reset_pending, collection),
                _retain,
                values,
            ),
        )
        return cast(
            Tree,
            jax.lax.cond(
                _failed(values),
                partial(_failure_result, collection),
                partial(_real_step, collection),
                values,
            ),
        )

    budget_error = carry.progress.rounds >= carry.schedule.total_rounds
    carry = carry._replace(
        tracking=replace(
            carry.tracking,
            error_flags=carry.tracking.error_flags
            | jnp.where(budget_error, 2, 0).astype(jnp.int32),
        )
    )
    return cast(
        tuple[TrainingCarry, tuple[TrainingTransition, EpisodeInfo, PolicyTrace]],
        jax.lax.cond(
            _failed(carry), partial(_failure_result, collection), proceed, carry
        ),
    )


def _package(
    collection: TrainingCollection,
    initial: TrainingCarry,
    carry: TrainingCarry,
    rows: TrainingTransition,
) -> TrainingRollout:
    """Bind the original memory and actual successor to the returned time rows."""
    state = carry.state
    return TrainingRollout(
        rows,
        initial.memory.team_a,
        carry.observations,
        _team_mask(state),
        state.config.agent_profile.active_mask[:, :5],
        state.core_state.alive_mask[:, :5],
        state.done.done,
        encode_training_state(state.core_state, state.config)
        if collection.collect_training_state
        else None,
        carry.progress.rounds - initial.progress.rounds,
    )


def scan_training_rollout(
    collection: TrainingCollection, carry: TrainingCarry, *, length: int = 128
) -> tuple[TrainingCarry, TrainingRollout]:
    """Collect a fixed output length inside JAX, masking the exact budget suffix.

    length is a static nonnegative Python integer. Keep collection fixed and
    carry dynamic under jit. Each real round advances all B lanes once. Padding
    advances nothing and admits only the neutral computational action. Zero
    length returns unchanged carry and a zero-length tree. A sticky failure
    freezes later work; inspect it on the host before using any returned data.
    There is no writer, actor replay, optimizer, hidden synchronization or I/O.
    A collection whose pinned opponent is a host method raises ValueError at
    trace time; collect_training_rollout runs it through the host route.
    """
    _refuse_host(collection, "scan_training_rollout")
    length = _length(length)
    initial = carry

    def step(
        values: TrainingCarry, _: None
    ) -> tuple[TrainingCarry, TrainingTransition]:
        """Skip the whole numerical decision for exhausted or failed collection."""

        def real(x: TrainingCarry) -> tuple[TrainingCarry, TrainingTransition]:
            """Retain only compact learning rows from a real public decision."""
            after, (row, _, _) = _advance_training_step(collection, x)
            return after, row

        def pad(x: TrainingCarry) -> tuple[TrainingCarry, TrainingTransition]:
            """Preserve carry and emit a safe, explicitly invalid time row."""
            return x, _padding(collection, x)

        return cast(
            tuple[TrainingCarry, TrainingTransition],
            jax.lax.cond(
                (values.progress.rounds < values.schedule.total_rounds)
                & ~_failed(values),
                real,
                pad,
                values,
            ),
        )

    carry, rows = jax.lax.scan(step, carry, None, length=length)
    return carry, _package(collection, initial, carry, rows)


@lru_cache(maxsize=32)
def _compiled_rollout(collection: TrainingCollection, length: int) -> Tree:
    """Cache the host-owned outer JIT with the shared training compiler policy.

    The pure scan stays usable under callers' own jit/scan transformations.
    Only this host collector selects the built-in continuation compiler options.
    """
    return jax.jit(
        partial(scan_training_rollout, collection, length=length),
        compiler_options=training_compiler_options(),
    )


@lru_cache(maxsize=32)
def _recorded_step(
    collection: TrainingCollection,
) -> Callable[[TrainingCarry, None], Tree]:
    """Reuse callback identity so bounded chunks compile once per structure."""

    def step(carry: TrainingCarry, _unused: None) -> Tree:
        """Match the ordinary collector callback without capturing live arrays."""
        return _advance_training_step(collection, carry)

    return step


def collect_training_rollout(
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    length: int = 128,
    writer: RunWriter | None = None,
) -> tuple[TrainingCarry, TrainingRollout]:
    """Collect one block on the host, optionally through the existing writer.

    collection/carry come from setup or an exact checked continuation. length
    is nonnegative and defaults to 128. Supply writer exactly when recording is
    enabled; attach it before the first decision. Return next carry and the same
    compact TrainingRollout as scan. The real prefix stops at the exact budget;
    padding reaches neither callback nor writer. Host failure checks precede
    success. Writer exceptions propagate through its existing recovery route.
    Recorded and unrecorded paths use the same training compiler policy.
    This function synchronizes small counters and is not itself jittable.

    When the pinned opponent is a host method, each real round runs as: a
    compiled reset of finished games, one host call through M8's _host_apply
    for the pinned lanes (HostOpponent.act, which serves each round once), and
    the ordinary compiled step with those actions as data; with a writer, each
    round goes through the same bounded recorder. The rows, padding and
    failure rules equal the other routes'. A method's error propagates with its
    own type before the round's step, and the collection then refuses reuse.
    Each round costs two compiled calls and one host synchronization, plus the
    Team B inputs copied to the host (about 94 KB per game) and the method's
    own time.
    """
    length = _length(length)
    if collection.recording != (writer is not None):
        raise ValueError("Supply a writer exactly when collection recording is enabled")
    _check_carry(carry)
    if collection.host_opponent is not None:
        if writer is not None:
            _check_collection_writer(
                writer,
                has_steps=length > 0
                and int(carry.progress.rounds) < int(carry.schedule.total_rounds),
            )
        try:
            return _collect_host_rollout(
                collection, carry, length=length, writer=writer
            )
        except BaseException as error:
            if writer is not None:
                writer.record_failure(error)
            raise
    initial = carry
    if writer is None:
        carry, rollout = _compiled_rollout(collection, length)(carry)
    else:
        real = min(
            length, int(carry.schedule.total_rounds) - int(carry.progress.rounds)
        )
        carry, rows = collect_rollout(
            _recorded_step(collection),
            carry,
            num_steps=real,
            writer=writer,
            source_configs=carry.tracking.source_configs,
            output_steps=length,
            compiler_options=training_compiler_options(),
        )
        # The generic collector owns zero storage, not learner mask semantics.
        padding = _padding(collection, initial)
        positions = jnp.arange(length) < real

        def pad_row(actual: Array, empty: Array) -> Array:
            """Replace only the generic collector's unwritten output suffix."""
            return jnp.where(
                positions.reshape((length,) + (1,) * empty.ndim), actual, empty
            )

        rows = cast(
            TrainingTransition,
            jax.tree.map(pad_row, rows, padding),
        )
        rollout = _package(collection, initial, carry, rows)
    _check_carry(carry)
    return carry, rollout


def training_summary(
    collection: TrainingCollection, carry: TrainingCarry
) -> dict[str, object]:
    """Read small counters and distinguish requested stages from played exposure.

    Return JSON-ready counts/rounding without retaining or transferring a
    trajectory. Integers are aggregated with Python precision, not int32 device
    reductions. Failed carry raises ValueError. No learning/sample-use or skill
    claim is inferred; learner_samples is None until a later learner owns it.
    score_thresholds_by_episode_stage labels the original reset-time stages.
    In a child, experience/exposure stay cumulative, stage_complete and
    completed_stage_counts describe only its local segment, and segment_rounds,
    segment_env_steps and continuation name its new work and archived parent proof.
    exposure_by_score_threshold groups starts, env_steps and actor_decisions by
    the K actually used by each game, including games carried across stage ends.
    With a named pinned opponent the result also holds its record and
    opponent_rows, the meaning of each of the 21 opponent counts (row 1 is the
    pinned System); without one those keys are absent, as before.
    """
    _check_carry(carry)
    p = jax.device_get(carry.progress)

    def totals(values: Tree) -> list[int]:
        """Sum each leading category with arbitrary-precision host integers."""
        return [sum(int(x) for x in np.asarray(row).flat) for row in np.asarray(values)]

    rounds = int(p.rounds)
    segment_offset = (
        0 if carry.schedule.round_offset is None else int(carry.schedule.round_offset)
    )
    stages = int(
        carry.schedule.stage_count
        if carry.schedule.distribution_count is None
        else carry.schedule.distribution_count
    )
    thresholds = np.asarray(carry.schedule.score_thresholds)[:stages].tolist()
    starts, steps, decisions = (
        totals(p.starts),
        totals(p.exposure),
        totals(p.actor_decisions),
    )
    by_threshold = [
        {
            "score_threshold": threshold,
            "starts": sum(
                starts[i] for i, value in enumerate(thresholds) if value == threshold
            ),
            "env_steps": sum(
                steps[i] for i, value in enumerate(thresholds) if value == threshold
            ),
            "actor_decisions": sum(
                decisions[i] for i, value in enumerate(thresholds) if value == threshold
            ),
        }
        for threshold in dict.fromkeys(thresholds)
    ]
    return {
        "rounds": rounds,
        "env_steps": rounds * collection.schedule.num_envs,
        "requested_env_steps": collection.schedule.total_env_steps,
        "rounding": dict(collection.schedule.rounding_report),
        "stage_complete": np.asarray(p.stage_complete).tolist(),
        "completed_stage_counts": np.asarray(p.completed_stage_counts).tolist(),
        "score_thresholds_by_episode_stage": thresholds,
        "exposure_by_score_threshold": by_threshold,
        "starts_by_episode_stage": starts,
        "steps_by_episode_stage": steps,
        "actor_decisions_by_episode_stage": decisions,
        "steps_by_map": totals(p.map_steps),
        "starts_by_opponent": totals(p.opponent_starts),
        "steps_by_opponent": totals(p.opponent_steps),
        "incomplete_games": sum(
            bool(x)
            for x in np.asarray(~carry.state.done.done & ~carry.state.episode_start)
        ),
        "learner_samples": None,
        **(
            {}
            if collection.schedule.continuation is None
            else {
                "continuation": _continuation_details(collection.schedule),
                "segment_rounds": rounds - segment_offset,
                "segment_env_steps": (rounds - segment_offset)
                * collection.schedule.num_envs,
            }
        ),
        **(
            {}
            if collection.pinned_opponent is None
            else {
                "pinned_opponent": collection.pinned_opponent,
                "opponent_rows": [
                    "current weights",
                    f"pinned: {collection.pinned_opponent['name']}",
                    *(f"history slot {slot}" for slot in range(1, 20)),
                ],
            }
        ),
    }


def _host_decision(collection: TrainingCollection, carry: TrainingCarry) -> Tree:
    """Reset finished games and gather Team B's decision inputs for the host.

    Returns (carry after resets, Team B SystemInput, pinned lanes (B,) bool,
    Team B action keys, Team B initialization keys, reset generations (B,),
    completed rounds, failure flag). Resets happen exactly as
    _advance_training_step would do them, so the step that follows finds no
    pending reset. Keys are derived as M8 derives them for Team B. Pure.
    """

    def reset(values: TrainingCarry) -> TrainingCarry:
        """Reset lanes whose games finished, as the ordinary step does first."""
        return cast(
            TrainingCarry,
            jax.lax.cond(
                jnp.any(values.state.done.done),
                partial(_reset_pending, collection),
                _retain,
                values,
            ),
        )

    stop = _failed(carry) | (carry.progress.rounds >= carry.schedule.total_rounds)
    carry = cast(TrainingCarry, jax.lax.cond(stop, _retain, reset, carry))
    state = carry.state
    local_step = state.core_state.step_count - state.initial_step_count
    roots = training_keys(
        carry.root_key,
        state.reset_generation,
        stream="action",
        decision_step=local_step,
    )
    return (
        carry,
        system_inputs(carry.observations, state, team=1),
        carry.history.lane_snapshot == 0,
        _action_keys(roots, state.episode_id, 1),
        _initialization_keys(carry.memory.init_key, state.episode_id, 1),
        state.reset_generation,
        carry.progress.rounds,
        _failed(carry),
    )


@lru_cache(maxsize=32)
def _host_decision_compiled(collection: TrainingCollection) -> Tree:
    """Cache the compiled host-route reset and input gathering per collection."""
    return jax.jit(
        partial(_host_decision, collection),
        compiler_options=training_compiler_options(),
    )


@lru_cache(maxsize=32)
def _host_step_compiled(collection: TrainingCollection) -> Tree:
    """Cache the compiled single step the host route runs after each host call."""
    return jax.jit(
        partial(_advance_training_step, collection),
        compiler_options=training_compiler_options(),
    )


def _first_row(leaf: Array) -> Array:
    """Take the single row of a one-step recorded collection."""
    return leaf[0]


def _stack_rows(*leaves: Array) -> Array:
    """Stack per-round rows along a new leading time axis."""
    return jnp.stack(leaves)


def _empty_rows(leaf: Array) -> Array:
    """Make a zero-length time axis shaped like one padding row."""
    return jnp.zeros((0, *leaf.shape), leaf.dtype)


def _collect_host_rollout(
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    length: int,
    writer: RunWriter | None,
) -> tuple[TrainingCarry, TrainingRollout]:
    """Collect one block when the pinned opponent is a host method.

    Parameters and return value follow collect_training_rollout. Real rounds
    stop at the exact budget or at a sticky failure; the rest of the block is
    the recorded route's padding. The host method's errors propagate with
    their own type. A failure found at the start of a round is reported to the
    writer before it is raised, as the recorded route does. Team B inputs stay
    on the device; HostOpponent.act copies only what the method needs.
    Host-only and not jittable.
    """
    holder = cast(HostOpponent, collection.host_opponent)
    initial = carry
    real = min(length, int(carry.schedule.total_rounds) - int(carry.progress.rounds))
    decide = _host_decision_compiled(collection)
    step = _host_step_compiled(collection)
    rows: list[TrainingTransition] = []
    for _ in range(real):
        carry, inputs, pinned, keys, init_keys, generations, rounds, failed = cast(
            tuple[TrainingCarry, Any, Array, Array, Array, Array, Array, Array],
            decide(carry),
        )
        if bool(failed):
            if writer is not None:
                # The recorded route reports this failure to its writer inside
                # the recorded step; the host route reports it here.
                error = ValueError(
                    "Training collection has a lifecycle, tracking or opponent error"
                )
                writer.record_failure(error)
                raise error
            break
        actions = holder.act(
            int(rounds),
            inputs,
            np.asarray(pinned),
            keys,
            init_keys,
            np.asarray(generations),
        )
        carry = carry._replace(
            pinned_opponent=(ActorAction(*(jnp.asarray(head) for head in actions)), ())
        )
        if writer is None:
            carry, (row, _, _) = cast(
                tuple[TrainingCarry, tuple[TrainingTransition, Any, Any]], step(carry)
            )
        else:
            carry, one = collect_rollout(
                _recorded_step(collection),
                carry,
                num_steps=1,
                writer=writer,
                source_configs=carry.tracking.source_configs,
                output_steps=1,
                compiler_options=training_compiler_options(),
            )
            row = cast(TrainingTransition, jax.tree.map(_first_row, one))
        rows.append(row)
    padding = _padding(collection, initial)
    rows.extend([padding] * (length - len(rows)))
    stacked = (
        cast(TrainingTransition, jax.tree.map(_stack_rows, *rows))
        if rows
        else cast(
            TrainingTransition,
            jax.tree.map(_empty_rows, padding),
        )
    )
    rollout = _package(collection, initial, carry, stacked)
    _check_carry(carry)
    return carry, rollout
