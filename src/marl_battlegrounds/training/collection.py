"""Collect exact training decisions with curriculum, shaping and self-play.

Host setup verifies content and freezes callables. Numerical carry keeps changing
weights, games, memories and counters. Pure scan and bounded recording share one
step. No critic, optimizer, learning run or durable learner checkpoint lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

# Shared private helpers retain one owner for execution and stage accounting.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.baselines.inputs import (
    TRAINING_STATE_FEATURE_SIZE,
    encode_training_state,
)
from marl_battlegrounds.collection import collect_rollout
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
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTrace,
    System,
    SystemState,
    init_systems,
)
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training._compilation import training_compiler_options
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    TrainingContentBinding,
    prepare_training_content,
)
from marl_battlegrounds.training._execution import _apply_and_track, _reset_finished
from marl_battlegrounds.training.curriculum import (
    ScheduleArrays,
    TrainingProgress,
    TrainingSchedule,
    _advance_training_schedule,
    _init_training_progress,
    make_training_schedule,
)
from marl_battlegrounds.training.distributions import (
    TRAINING_KEY_SCHEMA_VERSION,
    sample_training_configs,
    training_keys,
)
from marl_battlegrounds.training.opponents import (
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


class TrainingCarry(NamedTuple):
    """Carry one complete numerical collection boundary without a learner.

    env/observations/state match the true last successor, before a pending reset.
    root_key is Threefry. memory holds separate teams; tracking owns the only
    source bank. source_indices (B,) and source_class_ids (B,10) describe live
    games. schedule and progress count rounds; history owns all actor variables.
    discount/coefficient are validated scalar float32 values. No host evidence,
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


class TrainingTransition(NamedTuple):
    """Keep compact facts from one native B-lane decision; scan adds axis T.

    observations and Team A action_mask describe the pre-action epoch. actions
    retain native (B,10) heads; learning_outputs are Team A's same-call tree.
    task_rewards is float32 (B,5); shaping_reward is one float32 value per game.
    active/alive are bool (B,5); episode_start/ended/valid are bool (B,).
    Completion outcome/length (B,) and final_scores (B,2) are zero unless ended.
    priority is existing MetricValues, masked to real endings, or None.
    Int32 identity/version fields are (B,); opponent_snapshot=-1 means current.
    training_state is float32 (B,919) or None and never enters an actor.
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
    reset, never to padding. final_training_state is optional (B,919).
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
    carry_spec records shapes and static settings without retaining arrays;
    root_bits, key_schema and reward_settings bind in-memory continuation.
    pinned_opponent_share is the static probability that a reset lane meets the
    pinned first-update actor in history slot 0; zero keeps the 80/20 recipe.
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
        if not np.array_equal(actual, expected):
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
    if not 0 <= rounds <= int(carry.schedule.total_rounds) or not np.all(
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
    if np.any(episode_stages < 0) or np.any(
        episode_stages >= int(carry.schedule.stage_count)
    ):
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
    stage_rounds = rounds - (int(ends[stage - 1]) if stage else 0)
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
        or np.any(progress.episode_stage > stage)
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
    expected_mapping = np.searchsorted(
        used_rounds, np.asarray(carry.schedule.history_threshold_rounds), side="left"
    )
    expected_mapping = np.where(expected_mapping < history_count, expected_mapping, -1)
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
            (np.asarray(carry.schedule.history_threshold_rounds) <= refreshed)
            != (mapping >= 0)
        )
    ):
        raise ValueError("Continuation opponent snapshot metadata is inconsistent")


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
        feedback adds coefficient times new kills minus new deaths, including
        terminal actions. Both leave native task rewards unchanged. The mode
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

    Returns
    -------
    tuple[TrainingCollection, TrainingCarry]
        Stable host descriptor and complete numerical state. Setup reads files
        only when preparing content, places arrays, initializes memory and traces
        output shapes. It creates no writer and executes no actor decision.

    Raises
    ------
    TypeError, ValueError
        Unsupported settings, content/bank mismatch, variable structure mismatch,
        a pinned share outside its contract or disagreeing with the schedule, or
        failed initialization. Failures precede any real transition.
    """
    if not isinstance(cast(object, actor), System):
        raise TypeError("actor must be a native JAX System")
    if not isinstance(cast(object, schedule), TrainingSchedule):
        raise TypeError("schedule must come from make_training_schedule")
    checked_schedule = make_training_schedule(
        total_env_steps=schedule.total_env_steps,
        num_envs=schedule.num_envs,
        curriculum=schedule.curriculum,
        score_threshold_curriculum=schedule.score_threshold_curriculum,
        early_history_capture=schedule.early_history_capture,
    )
    if (
        jax.tree.structure(schedule.arrays)
        != jax.tree.structure(checked_schedule.arrays)
        or any(
            left.dtype != right.dtype or not np.array_equal(left, right)
            for left, right in zip(
                schedule.arrays, checked_schedule.arrays, strict=True
            )
        )
        or dict(schedule.rounding_report) != dict(checked_schedule.rounding_report)
    ):
        raise ValueError("schedule arrays and report must match its declared settings")
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
    opponent = make_opponent_system(actor)
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
        variables_b=history,
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
    )
    return collection, carry


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
        variables_b=carry.history,
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
    """Collect one already-checked decision and advance accounting afterward."""
    before = carry.state
    actions, memory, learning, tracking, result = _apply(
        carry, collection.actor, collection.opponent
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
        jnp.full_like(info.episode_id, carry.tracking.stage_ordinal),
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
    advancing. Use rollout helpers for ordinary safe budget padding.
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
    """
    length = _length(length)
    initial = carry

    def step(
        values: TrainingCarry, _: None
    ) -> tuple[TrainingCarry, TrainingTransition]:
        """Skip the whole numerical decision for exhausted or failed collection."""

        def real(x: TrainingCarry) -> tuple[TrainingCarry, TrainingTransition]:
            """Retain only compact learning rows from a real public decision."""
            after, (row, _, _) = advance_training_step(collection, x)
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
        return advance_training_step(collection, carry)

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
    """
    length = _length(length)
    if collection.recording != (writer is not None):
        raise ValueError("Supply a writer exactly when collection recording is enabled")
    _check_carry(carry)
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
    score_thresholds_by_episode_stage labels active reset-time stages.
    exposure_by_score_threshold groups starts, env_steps and actor_decisions by
    the K actually used by each game, including games carried across stage ends.
    """
    _check_carry(carry)
    p = jax.device_get(carry.progress)

    def totals(values: Tree) -> list[int]:
        """Sum each leading category with arbitrary-precision host integers."""
        return [sum(int(x) for x in np.asarray(row).flat) for row in np.asarray(values)]

    rounds = int(p.rounds)
    stages = int(carry.schedule.stage_count)
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
    }
