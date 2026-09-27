"""Collect exact training decisions with curriculum, shaping and self-play.

Host setup verifies content and freezes callables. Numerical carry keeps changing
weights, games, memories and counters. Pure scan and bounded recording share one
step. A named pinned opponent plays lanes assigned to the separate pin (-2): a JAX
method inside the compiled step, a host method through a host loop in
collect_training_rollout. An optional ActorVariablesAtStep hook, such as QMIX
exploration, sets changing actor values from the round count before every
decision; without one the step is unchanged. No critic, optimizer, replay,
learning run or durable learner checkpoint lives here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
    _CompositionMemory,
    _execution,
    _initial_memory,
    _initialization_keys,
    _member_keys,
    _validate_adapter_roster,
    init_systems,
    pool,
    shared_policy,
    system_inputs,
    team,
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
    OpponentSelection,
    _history_invalid,
    _history_shapes,
    _pinned_share,
    _schema,
    assign_opponents,
    configure_opponent_selection,
    init_opponent_history,
    make_opponent_system,
    opponent_counter_row,
    opponent_identity,
    opponent_selection_names,
    select_opponents,
)
from marl_battlegrounds.training.shaping import (
    RewardFunction,
    reward_adjustments,
    team_potential_shaping,
    team_score_delta_shaping,
    validate_reward,
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
    opponent_selection is None for the legacy recipe, otherwise the saved
    numerical future-game rule, dense start counter and external member indices.
    partner_selection is None without fixed partners, otherwise the same selector
    for each team. partner_values stores a JAX partner pool once; both teams use
    separate choices and memory. Host partners instead supply partner_actions
    for this step and retain their opaque memory on the collection.
    frozen_host_values stores numeric frozen variables/templates for checkpoints,
    first the host opponent if present, then the shared host partner population.
    It does not save opaque host memory or drive live execution.
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
    opponent_selection: OpponentSelection | None = None
    partner_selection: tuple[OpponentSelection, OpponentSelection] | None = None
    partner_values: Tree = ()
    partner_actions: tuple[ActorAction, ActorAction] | None = None
    frozen_host_values: tuple[Tree, ...] = ()


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
    opponent_snapshot is a stable capture ID, -1 current, -2 legacy permanent
    pin, or -3-index for a named member in its fixed declaration order.
    Reusing a physical history slot never changes a stored transition identity.
    opponent_update is the learner version the opponent's weights come from;
    for games against a named pinned System it is -2, because that System
    is not a learner version.
    training_state is float32 (B,920) or None and never enters an actor.
    custom_rewards is float32 (B,5) for Team A when a reward callback is enabled,
    otherwise None. It stays separate from native rewards and built-in shaping.
    learner_active is optional bool (B,5) ownership; None means all active
    slots learn. It never replaces the physical active mask. Padding owns none.
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
    custom_rewards: Array | None = None
    learner_active: Array | None = None


class TrainingRollout(NamedTuple):
    """Return fixed-length transitions and the true final bootstrap boundary.

    transitions has leading (T,B). initial_memory is Team A's block-entry tree;
    the first real row's episode_start handles a pending episode reset. Final
    observations/masks and active/alive/ended belong to the returned carry before
    reset, never to padding. final_training_state is optional (B,920).
    final_learner_active matches successor ownership or None for all slots.
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
    final_learner_active: Array | None = None


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
    permanent first-update actor outside rotating history; zero keeps the 80/20 recipe.
    pinned_opponent is None, or the JSON record of a named System that plays
    permanent-pin lanes instead (reference, name, execution, registration and its ID,
    variables digest, evidence, memory rule). host_opponent is the mutable
    HostOpponent of a pinned host method, else None; a collection that has one
    serves each training round once and must not be replayed.
    actor_variables_at_step is None (the default, used by PPO and PQN-VDN,
    whose rate changes only between blocks), or the learner's
    ActorVariablesAtStep hook, which sets values such as QMIX's exploration
    rate on the shared current variables before each decision.
    opponent_names, opponent_records and opponent_systems hold the fixed named
    declaration order, frozen identity records and original member callbacks.
    Changing ordinary shares leaves these alone. Append new members only at a
    child boundary. Resource scopes belong to the calling runner.
    reward is the optional pure scalar callback; reward_identity is its copied
    JSON evidence from shaping.resolve_reward. Both are None when disabled.
    The callback receives privileged training state, never actor state.
    learner_slots is None for the unchanged full-learner path, otherwise fixed
    physical slots 0..4. learner_actor retains the native learner callbacks;
    actor is the deployed team. partner records and Systems follow the opponent
    binding contract. host_partners holds separate full-batch A/B memories.
    Named opponents control their whole team; only native self/past/permanent
    actor copies use the mirrored learner slots and a separate partner draw.
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
    opponent_names: tuple[str, ...] = ()
    opponent_records: tuple[dict[str, Any], ...] = ()
    opponent_systems: tuple[System, ...] = ()
    reward: RewardFunction | None = None
    reward_identity: dict[str, object] | None = None
    vectorize_opponent_lanes: bool = True
    opponent_trace_rows: tuple[int, ...] = ()
    learner_slots: tuple[int, ...] | None = None
    learner_actor: System | None = None
    partner_names: tuple[str, ...] = ()
    partner_records: tuple[dict[str, Any], ...] = ()
    partner_systems: tuple[System, ...] = ()
    host_partners: tuple[HostOpponent, HostOpponent] | None = None
    partner_trace_rows: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    pinned_system: System | None = None


def _frozen_host_values(
    opponent: HostOpponent | None, partners: tuple[HostOpponent, HostOpponent] | None
) -> tuple[Tree, ...]:
    """Save numeric frozen parameters once per holder population, never its memory."""
    result: list[Tree] = []
    if opponent is not None:
        result.append(opponent.checkpoint_values())
    if partners is not None:
        result.append(partners[0].checkpoint_values())
    return tuple(result)


def _check_learner_slots(
    slots: tuple[int, ...] | None, schedule: TrainingSchedule
) -> tuple[int, ...] | None:
    """Check fixed physical learner slots and at least one learner in every stage."""
    if slots is None:
        return None
    if (
        not isinstance(cast(object, slots), (tuple, list))
        or not slots
        or any(type(slot) is not int or not 0 <= slot < 5 for slot in slots)
        or len(set(slots)) != len(slots)
    ):
        raise ValueError(
            "learner_slots must contain distinct physical slots 0 through 4"
        )
    if len(slots) == 5:
        return None
    for stage, teams in enumerate(_stage_active_masks(schedule)):
        if not np.all(np.any(teams[:, list(slots)], axis=-1)):
            raise ValueError(
                f"Training stage {stage} has no active learner slot on a mirrored team"
            )
    return tuple(slots)


def _stage_active_masks(schedule: TrainingSchedule) -> np.ndarray:
    """Read both declared roster masks for every preserved distribution."""
    stages = int(
        schedule.arrays.stage_count
        if schedule.arrays.distribution_count is None
        else schedule.arrays.distribution_count
    )
    roster = schedule.arrays.roster_class_ids
    return np.stack(
        [
            (np.asarray(roster[stage]).reshape(2, 5) != 0)
            if roster is not None and np.any(np.asarray(roster[stage]))
            else np.tile(np.arange(5) < int(schedule.arrays.team_sizes[stage]), (2, 1))
            for stage in range(stages)
        ]
    )


def _check_partner_rosters(
    system: System, schedule: TrainingSchedule, slots: tuple[int, ...]
) -> None:
    """Use the shared adapter check on every controlled future-stage slot."""
    active = jnp.asarray(_stage_active_masks(schedule).reshape(-1, 5))
    controlled = active & jnp.asarray(
        [slot not in slots for slot in range(5)], jnp.bool_
    )
    _validate_adapter_roster(
        _execution(system),
        active,
        controlled=controlled,
        valid=jnp.any(controlled, axis=1),
    )


def _partner_rule(names: tuple[str, ...], selection: object) -> object:
    """Require only declared frozen names; omitted shares use every member equally."""
    if any(
        not isinstance(cast(object, name), str)
        or not name.strip()
        or name in ("self", "past")
        for name in names
    ):
        raise ValueError("Partner names must be nonempty and exclude self/past")
    if selection is None:
        return {name: 1.0 for name in names} if names else None
    selected = opponent_selection_names(selection)
    if set(selected) - set(names):
        raise ValueError("Partner selection must name declared frozen members")
    return selection


def _select_partners(
    history: OpponentHistory,
    settings: tuple[OpponentSelection, OpponentSelection],
    root: Array,
    state: EnvironmentState,
    *,
    mask: Array | None = None,
) -> tuple[OpponentSelection, OpponentSelection]:
    """Share dense game order but separate team keys for weighted partner choices."""
    if mask is None:
        mask = jnp.ones_like(state.episode_start)
    keys = training_keys(root, state.reset_generation, stream="opponent")

    def tagged(key: Array, tag: int) -> Array:
        """Keep partner sampling separate from opponent and other-team draws."""
        return jax.random.fold_in(key, tag)

    return cast(
        tuple[OpponentSelection, OpponentSelection],
        tuple(
            select_opponents(
                history,
                side,
                mask,
                jax.vmap(partial(tagged, tag=0x50415254 + team_index))(keys),
            )[1]
            for team_index, side in enumerate(settings)
        ),
    )


def _training_teams(
    learner: System,
    partners: System,
    slots: tuple[int, ...],
    pinned: System | None,
    *,
    vectorize_lanes: bool,
) -> tuple[System, System]:
    """Compose native learner slots, retaining whole-team external opponents."""
    complement = tuple(slot for slot in range(5) if slot not in slots)
    actor = team(learner, partners, slots=(slots, complement))
    native = team(
        make_opponent_system(learner, vectorize_lanes=vectorize_lanes),
        partners,
        slots=(slots, complement),
    )

    # Templates keep selectors, not frozen weights. Carry owns the values once.
    def layout(system: System) -> System:
        """Keep the fixed composition layout without captured changing parameters."""
        return replace(
            system,
            variables=system.variables._replace(members=((), ()), templates=((), ())),
        )

    actor, native = layout(actor), layout(native)
    if pinned is None:
        return actor, native
    opponent = pool({native: 1.0, pinned: 1.0})
    return actor, replace(
        opponent,
        variables=opponent.variables._replace(
            members=(native.variables, ()),
            templates=((), ()),
        ),
    )


def _team_variables(
    actor: System,
    opponent: System,
    history: OpponentHistory,
    pinned: Tree,
    selection: tuple[OpponentSelection, OpponentSelection] | None,
    partners: Tree,
    host_actions: tuple[ActorAction, ActorAction] | None,
) -> tuple[Tree, Tree]:
    """Supply current learner weights and one shared frozen pool as dynamic values."""
    if selection is None:
        return history.current_variables, history if not len(pinned) else (
            history,
            *pinned,
        )

    def fixed(side: int) -> tuple[Tree, Tree]:
        """Use that side's saved member choice or already produced host actions."""
        if host_actions is not None:
            return host_actions[side], ()
        return partners[0]._replace(choices=selection[side].choices), partners[1]

    a_values, a_template = fixed(0)
    a = actor.variables._replace(
        members=(history.current_variables, a_values), templates=((), a_template)
    )
    b_values, b_template = fixed(1)
    native_history = (
        history
        if not len(pinned)
        else history._replace(
            lane_snapshot=jnp.where(
                history.lane_snapshot == -2, -1, history.lane_snapshot
            )
        )
    )
    b_layout = opponent.variables if not len(pinned) else opponent.variables.members[0]
    native = b_layout._replace(
        members=(native_history, b_values), templates=((), b_template)
    )
    if not len(pinned):
        return a, native
    return a, opponent.variables._replace(
        members=(native, pinned[0]),
        templates=((), pinned[1]),
        choices=(history.lane_snapshot == -2).astype(jnp.int32),
    )


def _learner_active(
    collection: TrainingCollection, state: EnvironmentState
) -> Array | None:
    """Keep physical activity separate from slots whose decisions train."""
    if collection.learner_slots is None:
        return None
    return state.config.agent_profile.active_mask[:, :5] & jnp.asarray(
        [slot in collection.learner_slots for slot in range(5)], jnp.bool_
    )


def learner_memory(carry: TrainingCarry) -> Tree:
    """Return only the learner's memory; fixed partners keep their own full trees."""
    return (
        carry.memory.team_a
        if carry.partner_selection is None
        else carry.memory.team_a.members[0]
    )


def _partner_inputs(
    observations: Observations,
    state: EnvironmentState,
    slots: tuple[int, ...] | None,
    side: int,
) -> Tree:
    """Give a fixed partner full permitted inputs and only its complement slots."""
    assert slots is not None
    inputs = system_inputs(observations, state, team=side)
    owned = inputs.active_mask & jnp.asarray(
        [slot not in slots for slot in range(5)], jnp.bool_
    )
    return inputs._replace(
        controlled_mask=owned, valid=inputs.valid & jnp.any(owned, axis=1)
    )


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
    selection = carry.opponent_selection
    if selection is not None:
        categories = 2 + len(collection.opponent_names)
        shares = np.asarray(selection.shares)
        order = np.asarray(selection.order)
        choices = np.asarray(selection.choices)
        starts = (
            sum(int(x) for x in np.asarray(carry.state.reset_generation))
            + collection.schedule.num_envs
        )
        if (
            shares.shape != (categories,)
            or order.ndim != 1
            or not order.size
            or choices.shape != carry.state.episode_id.shape
            or np.any(~np.isfinite(shares))
            or np.any(shares < 0)
            or (not bool(selection.ordered) and not np.isclose(shares.sum(), 1.0))
            or np.any(order < 0)
            or np.any(order >= categories)
            or np.any(choices < 0)
            or (
                len(collection.opponent_names)
                and np.any(choices >= len(collection.opponent_names))
            )
            or int(selection.game_starts) != starts
            or not 0 <= int(selection.order_start) <= starts
            or selection.past_ids.shape != carry.history.captured_ids.shape
            or selection.past_weights.shape != carry.history.captured_ids.shape
            or np.any(~np.isfinite(selection.past_weights))
            or np.any(np.asarray(selection.past_weights) < 0)
        ):
            raise ValueError(
                "Continuation opponent selection or game-start count is invalid"
            )
    _check_learner_slots(collection.learner_slots, collection.schedule)
    if (carry.partner_selection is None) != (collection.learner_slots is None):
        raise ValueError("Continuation learner slots and partner selection disagree")
    if carry.partner_selection is not None:
        if (
            carry.progress.partner_used is None
            or carry.progress.partner_used.shape != (len(collection.partner_names),)
            or carry.progress.partner_used.dtype != jnp.bool_
        ):
            raise ValueError(
                "Continuation partner exposure flags differ from declared members"
            )
        starts = (
            sum(int(x) for x in np.asarray(carry.state.reset_generation))
            + collection.schedule.num_envs
        )
        for settings in carry.partner_selection:
            shares, order, choices = map(
                np.asarray, (settings.shares, settings.order, settings.choices)
            )
            if (
                shares.shape != (2 + len(collection.partner_names),)
                or np.any(shares[:2] != 0)
                or np.any(~np.isfinite(shares))
                or np.any(shares < 0)
                or (not bool(settings.ordered) and not np.isclose(shares.sum(), 1.0))
                or order.ndim != 1
                or not order.size
                or (
                    bool(settings.ordered)
                    and np.any((order < 2) | (order >= len(shares)))
                )
                or choices.shape != carry.state.episode_id.shape
                or np.any((choices < 0) | (choices >= len(collection.partner_names)))
                or int(settings.game_starts) != starts
                or not 0 <= int(settings.order_start) <= starts
            ):
                raise ValueError(
                    "Continuation partner choices or game-start count are invalid"
                )
    if carry.progress.opponent_steps.shape[0] != int(
        carry.history.capture_capacity
    ) + 2 + len(collection.opponent_names):
        raise ValueError(
            "Continuation opponent counter rows differ from declared members"
        )
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
    if carry.schedule.roster_class_ids is not None:
        declared = np.asarray(carry.schedule.roster_class_ids)[episode_stages]
        fixed = np.any(declared != 0, axis=-1)
        if not np.array_equal(
            np.asarray(carry.source_class_ids)[fixed], declared[fixed]
        ):
            raise ValueError(
                "Continuation rosters disagree with their producing stages"
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
    history = carry.history
    update = int(history.current_update)
    refreshed = int(history.last_refresh_rounds)
    ids, captured_rounds, captured_updates, mapping = map(
        np.asarray,
        jax.device_get(
            (
                history.captured_ids,
                history.captured_rounds,
                history.captured_updates,
                history.threshold_to_snapshot,
            )
        ),
    )
    occupied = ids >= 0
    order = np.argsort(ids[occupied])
    used_rounds, used_updates = (
        captured_rounds[occupied][order],
        captured_updates[occupied][order],
    )
    threshold_count = (
        20
        if carry.schedule.history_threshold_count is None
        else int(carry.schedule.history_threshold_count)
    )
    active = np.arange(20) < threshold_count
    captures = int(history.next_capture_id)
    if (
        not 0 <= int(history.count) <= captures <= update <= refreshed <= rounds
        or captures > int(history.capture_capacity)
        or ((update == 0) != (refreshed == 0))
        or len(set(ids[occupied].tolist())) != int(history.count)
        or np.any(used_rounds <= 0)
        or np.any(used_rounds > refreshed)
        or np.any(np.diff(used_rounds) < int(history.minimum_capture_rounds))
        or np.any(used_updates <= 0)
        or np.any(used_updates > update)
        or np.any(np.diff(used_updates) <= 0)
        or np.any(captured_rounds[~occupied] != -1)
        or np.any(captured_updates[~occupied] != -1)
        or np.any(mapping < -1)
        or np.any(mapping >= captures)
        or np.any(mapping[~active] != -1)
        or np.any(
            (mapping >= 0)
            & (np.asarray(carry.schedule.history_threshold_rounds) > refreshed)
        )
        or (captures == 0 and int(history.last_capture_rounds) != -1)
        or (captures > 0 and not 0 < int(history.last_capture_rounds) <= refreshed)
    ):
        raise ValueError("Continuation opponent snapshot metadata is inconsistent")


def _continuation_history_thresholds(  # pyright: ignore[reportUnusedFunction]
    carry: TrainingCarry, *, future_rounds: tuple[int, ...]
) -> tuple[int, ...]:
    """Keep saved snapshots and pending captures while adding explicit requests.

    carry must already pass its parent's collection check. future_rounds contains
    strictly increasing Python integers after the actual checkpoint round; the
    runner checks them against the child's declared end and learner boundaries.
    Return pending parent thresholds and added points in order. Completed
    requests stay in the archived parent proof; stable capture IDs stay with
    history and saved records. They need no entry in the new request table.
    Reject invalid points or more than 20 required table entries.
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
    active = (
        20
        if carry.schedule.history_threshold_count is None
        else int(carry.schedule.history_threshold_count)
    )
    thresholds = np.asarray(carry.schedule.history_threshold_rounds)[:active]
    mapping = np.asarray(carry.history.threshold_to_snapshot)[:active]
    pending = {int(value) for value in thresholds[mapping == -1]}
    combined = tuple(sorted(pending | set(future_rounds)))
    if len(combined) > 20:
        raise ValueError("Added captures exceed the 20 explicit pending requests")
    return combined


def _expanded_stage_sources(
    collection: TrainingCollection, carry: TrainingCarry, schedule: TrainingSchedule
) -> tuple[TrainingContentBinding, EpisodeTrackingState]:
    """Append score-source blocks without changing any live game's source row.

    The new schedule retains every old distribution ID and winning score. Only
    new score blocks may extend the verified source bank. Installed scientific
    content and all old bank rows must match exactly. Return the new binding and
    tracker with its extended bank/digest; game bindings, counters and memories
    are unchanged. This host boundary reads content, never writes files or resets.
    """
    old_scores = collection.binding.score_thresholds
    scores = schedule.score_thresholds
    if scores == old_scores:
        return collection.binding, carry.tracking
    if scores[: len(old_scores)] != old_scores:
        raise ValueError("Child source scores must preserve the original bank prefix")
    prepared = prepare_training_content(
        score_thresholds=scores, red_zone_depth=collection.binding.red_zone_depth
    )
    omitted = {"source_bank", "source_configurations", "score_thresholds"}
    old_content = {
        k: v
        for k, v in collection.binding.scientific_projection().items()
        if k not in omitted
    }
    new_content = {
        k: v
        for k, v in prepared.binding.scientific_projection().items()
        if k not in omitted
    }
    if canonical_digest_sha256(old_content) != canonical_digest_sha256(new_content):
        raise ValueError("Changed stages cannot replace installed scientific content")
    assert carry.tracking.source_configs is not None
    old_rows = 42 * len(old_scores)
    if any(
        not np.array_equal(old, new[:old_rows])
        for old, new in zip(
            jax.tree.leaves(carry.tracking.source_configs),
            jax.tree.leaves(prepared.source_configs),
            strict=True,
        )
    ):
        raise ValueError(
            "Changed stages must retain every original source configuration"
        )
    table_id = carry.tracking.source_table_id
    if table_id is not None:
        table_id = jnp.asarray(
            np.frombuffer(
                bytes.fromhex(prepared.binding.source_bank.canonical_digest),
                dtype=">u4",
            ).astype(np.uint32)
        )
    return prepared.binding, replace(
        carry.tracking, source_configs=prepared.source_configs, source_table_id=table_id
    )


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
        Explicit history thresholds may drop completed requests, which remain
        in the parent proof. Every pending request must remain.

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
        a history change drops a pending request, or child accounting is invalid.

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
    replacement_stages = next(
        (
            change["curriculum"]
            for change in declaration.get("curriculum_changes", ())
            if change["segment_index"] == declaration["segment_index"]
        ),
        None,
    )
    expected = _make_continuation_schedule(
        collection.schedule,
        completed_rounds=rounds,
        additional_env_steps=declaration["additional_env_steps"],
        history_threshold_rounds=None if same_history else thresholds[:active],
        curriculum=replacement_stages,
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
    _check_learner_slots(collection.learner_slots, schedule)
    if collection.learner_slots is not None:
        for member in collection.partner_systems:
            _check_partner_rosters(member, schedule, collection.learner_slots)
    history = carry.history
    if not same_history:
        old_active = (
            20
            if carry.schedule.history_threshold_count is None
            else int(carry.schedule.history_threshold_count)
        )
        old_thresholds = np.asarray(carry.schedule.history_threshold_rounds)[
            :old_active
        ]
        old_mapping = np.asarray(history.threshold_to_snapshot)[:old_active]
        saved = dict(zip(old_thresholds.tolist(), old_mapping.tolist(), strict=True))
        pending = {int(point) for point, identity in saved.items() if identity < 0}
        if not pending <= set(thresholds[:active]):
            raise ValueError(
                "Child history requests must retain pending parent captures"
            )
        mapping = [saved.get(point, -1) for point in thresholds[:active]]
        history = history._replace(
            threshold_to_snapshot=jnp.asarray(mapping + [-1] * (20 - active), jnp.int32)
        )
    binding, tracking = _expanded_stage_sources(collection, carry, schedule)
    changed = carry._replace(
        schedule=schedule.arrays,
        progress=carry.progress._replace(
            completed_stage_counts=jnp.zeros_like(
                carry.progress.completed_stage_counts
            ),
            stage_complete=jnp.zeros_like(carry.progress.stage_complete),
        ),
        tracking=replace(
            tracking,
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
        collection,
        binding=binding,
        schedule=schedule,
        carry_spec=jax.eval_shape(_retain, changed),
    )
    _validate_training_continuation(child, changed, recheck_installed_content=False)
    return child, changed


def _resize_training_history(  # pyright: ignore[reportUnusedFunction]
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    keep_past: int,
    history_capture_capacity: int,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Resize history once at a checked child boundary without changing live games.

    collection/carry describe the existing verified boundary. keep_past is the
    new nonnegative rolling window, and history_capture_capacity is the total
    stable capture bound, including old IDs. Old counter rows are never removed.
    Keep every unfinished game's copy plus the newest eligible copies. A shrink
    that cannot fit those copies in K+1 slots raises ValueError before mutation.
    K=0 needs no live historical games. Finished lanes can return to self before
    their pending reset. Stable IDs, clocks, current/pinned values, live memory,
    replay/recent rows and cumulative counters stay unchanged.

    Return the descriptor with refreshed shape evidence and the resized carry.
    This host-only setup reads small metadata and gathers each bank leaf once;
    changed shapes may compile a new learner program. No game or action runs.
    """
    if type(keep_past) is not int or keep_past < 0:
        raise ValueError("keep_past must be a nonnegative integer")
    old_capacity = int(carry.history.capture_capacity)
    if (
        type(history_capture_capacity) is not int
        or history_capture_capacity < int(carry.history.next_capture_id)
        or history_capture_capacity < old_capacity
    ):
        raise ValueError("History counter capacity must retain every saved row and ID")
    history = carry.history
    capacity = keep_past + 1 if keep_past else 0
    ids, eligible, slots, ended = map(
        np.asarray,
        jax.device_get(
            (
                history.captured_ids,
                history.eligible,
                history.lane_snapshot,
                carry.state.done.done,
            )
        ),
    )
    live = set(int(slot) for slot in slots[~ended] if slot >= 0)
    newest = sorted(
        np.flatnonzero(eligible), key=lambda slot: int(ids[slot]), reverse=True
    )[:keep_past]
    retained = sorted(live | set(newest), key=lambda slot: int(ids[slot]))
    if len(retained) > capacity:
        raise ValueError(
            f"keep_past={keep_past} cannot retain {len(retained)} required copies "
            f"in {capacity} slots; unfinished games still use "
            f"capture IDs {[int(ids[slot]) for slot in sorted(live)]}"
        )
    # Preserve the old layout when its size is unchanged; no ordinary copy is needed.
    if capacity == ids.size:
        next_history = history
    else:
        selected = jnp.asarray(retained, jnp.int32)

        def resize(bank: Array) -> Array:
            """Gather retained resident leaves, then add empty physical slots."""
            kept = jnp.take(bank, selected, axis=0)
            padding = [(0, capacity - len(retained)), *[(0, 0)] * (bank.ndim - 1)]
            return jnp.pad(kept, padding)

        remap = {old: new for new, old in enumerate(retained)}
        lane_slots = np.asarray(
            [remap.get(int(slot), -1) if slot >= 0 else slot for slot in slots],
            np.int32,
        )
        metadata = {}
        for name in ("captured_ids", "captured_rounds", "captured_updates"):
            values = np.take(
                np.asarray(getattr(history, name)),
                np.asarray(retained, np.intp),
                axis=0,
            )
            metadata[name] = jnp.asarray(
                np.pad(values, (0, capacity - len(retained)), constant_values=-1)
            )
        next_history = history._replace(
            historical_variables=jax.tree.map(resize, history.historical_variables),
            count=jnp.int32(len(retained)),
            lane_snapshot=jnp.asarray(lane_slots),
            eligible=jnp.asarray(
                [old in newest for old in retained]
                + [False] * (capacity - len(retained))
            ),
            **metadata,
        )
    padding = history_capture_capacity - old_capacity

    def pad_counts(values: Array) -> Array:
        """Insert fresh capture rows before stable named-member rows."""
        return jnp.concatenate(
            (
                values[: old_capacity + 2],
                jnp.zeros((padding, values.shape[1]), values.dtype),
                values[old_capacity + 2 :],
            )
        )

    progress = carry.progress._replace(
        opponent_starts=pad_counts(carry.progress.opponent_starts),
        opponent_steps=pad_counts(carry.progress.opponent_steps),
    )
    next_history = next_history._replace(
        capture_capacity=jnp.int32(history_capture_capacity)
    )
    selection = carry.opponent_selection
    if selection is not None:
        selection = configure_opponent_selection(
            collection.opponent_names,
            next_history,
            _selection_value(collection, selection),
            previous=selection,
        )
    partner_selection = carry.partner_selection
    if partner_selection is not None:
        rule = partner_selection_value(collection, carry)
        partner_selection = cast(
            tuple[OpponentSelection, OpponentSelection],
            tuple(
                configure_opponent_selection(
                    collection.partner_names, next_history, rule, previous=side
                )
                for side in partner_selection
            ),
        )
    changed = carry._replace(
        history=next_history,
        progress=progress,
        opponent_selection=selection,
        partner_selection=partner_selection,
    )
    if carry.root_key.committed:
        changed = jax.device_put(changed, carry.root_key.sharding)
    collection = replace(collection, carry_spec=jax.eval_shape(_retain, changed))
    return _opponent_component_table(collection, changed), changed


def change_training_reward(
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    shaping: bool,
    shaping_mode: str,
    discount: float,
    coefficient: float,
    reward: RewardFunction | None = None,
    reward_identity: dict[str, object] | None = None,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Change reward settings at a checked child boundary without playing a step.

    Parameters match init_training_collection. The caller first restores the
    parent, then clears stored shaped experience when its objective changed.
    This helper changes only reward settings and optional facts in the static
    environment handle. Games, observations, memory, keys and progress remain
    unchanged. Shape tracing checks callback output and refreshes padding data;
    it executes no numerical reset or step. Unsupported callback shape/dtype or
    non-JSON identity fails before a changed collection is returned.
    """
    if type(shaping) is not bool:
        raise TypeError("shaping must be bool")
    validate_shaping(discount=discount, coefficient=coefficient, mode=shaping_mode)
    if (reward is None) != (reward_identity is None):
        raise ValueError("reward and reward_identity must be supplied together")
    if reward is not None:
        if not callable(reward):
            raise TypeError("reward must be callable")
        if not isinstance(reward_identity, dict) or any(
            not isinstance(key, str)
            for key in cast(dict[object, object], reward_identity)
        ):
            raise TypeError("reward_identity must be a JSON object with string keys")
        reward_identity = json.loads(json.dumps(reward_identity, allow_nan=False))
    changed = carry._replace(
        env=replace(
            carry.env,
            _execution=replace(carry.env._execution, training_facts=reward is not None),
        ),
        discount=jnp.asarray(discount, jnp.float32),
        coefficient=jnp.asarray(coefficient, jnp.float32),
    )
    if carry.root_key.committed:
        changed = jax.device_put(changed, carry.root_key.sharding)
    shape = jax.eval_shape(
        partial(_apply, actor=collection.actor, opponent=collection.opponent), changed
    )
    if reward is not None:
        assert shape[4][4].training_facts is not None

        def scalar_spec(value: Array | jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
            """Remove the environment axis for the scalar callback contract."""
            return jax.ShapeDtypeStruct(value.shape[1:], value.dtype)

        validate_reward(
            reward,
            jax.tree.map(scalar_spec, changed.state.core_state),
            jax.tree.map(scalar_spec, shape[4][4].training_facts),
            jax.tree.map(scalar_spec, shape[4][1].core_state),
            changed.progress.rounds,
        )
    return replace(
        collection,
        shaping=shaping,
        shaping_mode=shaping_mode,
        reward=reward,
        reward_identity=reward_identity,
        reward_settings=(float(changed.discount), float(changed.coefficient)),
        info_spec=shape[4][4],
        carry_spec=jax.eval_shape(_retain, changed),
    ), changed


def _check_opponent_rosters(system: System, schedule: TrainingSchedule) -> None:
    """Require independent Policy lists to cover every declared Team B roster."""
    if system._policies and not system._shared:
        stages = int(
            schedule.arrays.stage_count
            if schedule.arrays.distribution_count is None
            else schedule.arrays.distribution_count
        )
        sizes: set[int] = set()
        explicit = schedule.arrays.roster_class_ids
        for stage in range(stages):
            row = None if explicit is None else np.asarray(explicit[stage])
            sizes.add(
                int(schedule.arrays.team_sizes[stage])
                if row is None or not np.any(row)
                else int(np.count_nonzero(row[5:]))
            )
        if sizes != {len(system._policies)}:
            raise ValueError(
                "An independent_policies pinned opponent needs every Team B "
                f"roster size to equal its {len(system._policies)} Policies; "
                f"the schedule uses {sorted(sizes)}"
            )


def _prepare_opponent_member(
    name: str,
    value: System | Policy | str,
    schedule: TrainingSchedule,
    binding: TrainingContentBinding,
    *,
    team_index: int | None = 1,
) -> tuple[System, dict[str, Any]]:
    """Freeze and register one declared member using evaluation's existing owners."""
    if (
        not isinstance(cast(object, name), str)
        or not name.strip()
        or name in ("self", "past")
    ):
        raise ValueError("Opponent names must be nonempty and exclude self/past")
    reference = value if isinstance(value, str) else None
    method = freeze_evaluation_method(
        load_method(value) if isinstance(value, str) else value
    )
    system = shared_policy(method) if isinstance(method, Policy) else method
    if team_index is not None:
        _check_opponent_rosters(system, schedule)
    registration = normalize_system_registration(
        system, phase="validation", frozen=True
    )[1]
    record = {
        "name": name,
        "reference": reference,
        "system_name": system.name,
        "registration_id": canonical_digest_sha256(registration),
        "registration": registration,
        "execution": system.execution,
        "variables_digest": tree_digest(system.variables)
        if system.execution == "jax"
        else None,
    }
    export = (
        Path(reference) if reference is not None and Path(reference).is_dir() else None
    )
    record["evidence"] = pinned_opponent_evidence(binding, method, export=export)
    return system, json.loads(json.dumps(record))


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
    keep_past: int = 20,
    history_capture_capacity: int = 20,
    minimum_capture_rounds: int = 1,
    capture_interval_rounds: int = 0,
    opponent_population: Mapping[str, System | Policy | str] | None = None,
    opponent_selection: Mapping[str, float] | Sequence[str] | None = None,
    vectorize_opponent_lanes: bool = True,
    reward: RewardFunction | None = None,
    reward_identity: dict[str, object] | None = None,
    learner_slots: tuple[int, ...] | None = None,
    partner_population: Mapping[str, System | Policy | str] | None = None,
    partner_selection: Mapping[str, float] | Sequence[str] | None = None,
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
        first-update actor kept outside rotating history. Zero keeps the 80% current
        and 20% uniform-history recipe and traces today's program. A positive
        share requires schedule.early_history_capture=True, and that flag
        requires a positive share; the pair is checked here before any reset.
    pinned_opponent : System, Policy, str or None, default None
        None uses the network's separate first-update pin. Otherwise this
        method plays every permanent-pin lane (assignment -2): a System or Policy
        object, or a reference
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

    keep_past, history_capture_capacity : int, defaults=20, 20
        Resident rolling window size and maximum stable capture count. The bank
        has keep_past+1 slots, or zero when disabled; counters have capacity+2
        rows. The public runner computes capacity from its capture schedule.
    minimum_capture_rounds, capture_interval_rounds : int, defaults=1, 0
        Minimum actual capture gap and optional recurring gap, in full rounds.
        Public train checks the maximum game horizon before setup and passes it
        here. The low-level default keeps small direct fixtures usable. A zero
        interval uses only the schedule's explicit thresholds.

    vectorize_opponent_lanes : bool, default=True
        Internal built-in actor execution choice. False uses lax.map for mixed
        opponent lanes when grouped models cannot use an outer vmap. The shared
        default and full-current batch call keep their existing compiled routes.

    reward : callable or None, default=None
        Pure JAX callback from shaping.resolve_reward. It receives one game's
        public training state before/after the action, selected training facts,
        and completed rounds before the step. Return float32 (10,) adjustments.
        Team A feedback is stored separately; learner masks own its reduction.
        None skips facts and callback work. Setup checks shape/dtype, while the
        learner rejects nonfinite real-step values.
    reward_identity : dict[str, object] or None, default=None
        Matching JSON evidence from shaping.resolve_reward, copied at setup.
        Supply both callback and identity, or neither. The checked continuation
        owner may use the saved identity with a shape-only placeholder while
        restoring a parent; replace it before collecting the changed child.

    opponent_population : mapping[str, System | Policy | str] or None
        Frozen named members in stable declaration order. Values use the same
        loader, registration, roster and information rules as evaluation. Extra
        members may start with zero share and become available to a later rule.
        Open member resource scopes around setup and use; the public runner
        handles this. Host members keep full-batch opaque memory outside carry.
    opponent_selection : mapping[str, float], sequence[str] or None
        Relative shares or an exact repeating order across actual game starts,
        including initial games, in ascending lane order. Names are self, past
        and declared aliases. Past uses uniform eligible copies unless changed
        through update_opponent_selection; no eligible copy falls back to self.
        None plus no population keeps the existing recipe and its random draws.
        Cannot mix this API with legacy pinned settings. Padding consumes no
        choice; unfinished games keep their chosen member. Game starts obey the
        environment's existing int32 allocation limit.

    learner_slots : tuple[int, ...] or None, default=None
        Physical Team A slots whose decisions train. None or all five slots
        keeps the full-learner route. Every stage needs an active learner slot.
        All other active slots use fixed partners; physical roster masks stay
        unchanged. Self/past/native pin opponents mirror this slot split.
    partner_population, partner_selection : mapping or sequence or None
        Frozen named methods and their relative shares or repeating game order,
        using the same name and binding rules as opponents. Omitted shares use
        all declared partners equally. Partners may not be self/past. Both teams
        draw at actual game starts, using separate team keys for weighted draws
        and the same dense game index for lists. A living game keeps its member.
        Named external opponents remain whole-team replacements. Open resource
        scopes before setup and retain them through collection, as the runner does.

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
        type(x) is not bool
        for x in (shaping, collect_training_state, recording, vectorize_opponent_lanes)
    ):
        raise TypeError("seed must be an integer and collection switches must be bool")
    if metrics not in ("none", "priority"):
        raise ValueError("training metrics must be priority or none")
    validate_shaping(discount=discount, coefficient=coefficient, mode=shaping_mode)
    if (reward is None) != (reward_identity is None):
        raise ValueError("reward and reward_identity must be supplied together")
    if reward is not None:
        if not callable(reward):
            raise TypeError("reward must be callable")
        if not isinstance(reward_identity, dict) or any(
            not isinstance(key, str)
            for key in cast(dict[object, object], reward_identity)
        ):
            raise TypeError("reward_identity must be a JSON object with string keys")
        reward_identity = json.loads(json.dumps(reward_identity, allow_nan=False))
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
    generic = opponent_selection is not None or opponent_population is not None
    frozen_members: list[System] = []
    names: tuple[str, ...] = ()
    members: tuple[dict[str, Any], ...] = ()
    if generic:
        opponent_selection_names(opponent_selection)
        if pinned_opponent is not None or pinned_opponent_share:
            raise ValueError(
                "Generic opponent selection cannot mix with legacy pinned settings"
            )
        population = {} if opponent_population is None else dict(opponent_population)
        names = tuple(population)
        if any(
            not isinstance(cast(object, n), str)
            or not n.strip()
            or n in ("self", "past")
            for n in names
        ):
            raise ValueError(
                "Opponent binding names must be nonempty and exclude self/past"
            )
        unknown = set(opponent_selection_names(opponent_selection)) - {
            "self",
            "past",
            *names,
        }
        if unknown:
            raise ValueError(f"Opponent members need bindings: {sorted(unknown)}")
        frozen_members = []
        records: list[dict[str, Any]] = []
        for name, value in population.items():
            system, record = _prepare_opponent_member(
                name, value, schedule, prepared.binding
            )
            records.append(record)
            frozen_members.append(system)
        members = tuple(json.loads(json.dumps(records)))
        if frozen_members:
            pinned_opponent = pool({member: 1.0 for member in frozen_members})
    slots = _check_learner_slots(learner_slots, schedule)
    partner_names = () if partner_population is None else tuple(partner_population)
    partner_systems: list[System] = []
    partner_records: list[dict[str, Any]] = []
    partner_rule = _partner_rule(partner_names, partner_selection)
    if slots is None:
        if partner_names or partner_selection is not None:
            raise ValueError(
                "Partners require learner_slots with at least one free slot"
            )
    else:
        if not partner_names:
            raise ValueError(
                "Fixed partners must cover the slots outside learner_slots"
            )
        assert partner_population is not None
        for name, value in partner_population.items():
            system, record = _prepare_opponent_member(
                name, value, schedule, prepared.binding, team_index=None
            )
            _check_partner_rosters(system, schedule, slots)
            partner_systems.append(system)
            partner_records.append(record)
    history = init_opponent_history(
        actor_variables,
        num_envs=schedule.num_envs,
        keep_past=keep_past,
        capture_capacity=history_capture_capacity,
        minimum_capture_rounds=minimum_capture_rounds,
        capture_interval_rounds=capture_interval_rounds,
        pin_first_update=pinned_opponent_share > 0 and pinned_opponent is None,
        external_pin=pinned_opponent is not None,
    )
    pinned_record: dict[str, Any] | None = None
    host: HostOpponent | None = None
    pinned_values: Tree = ()
    pinned_system: System | None = None
    if pinned_opponent is not None:
        if not generic and not pinned_opponent_share > 0:
            raise ValueError("A pinned opponent needs a positive pinned_opponent_share")
        reference = pinned_opponent if isinstance(pinned_opponent, str) else None
        method = (
            load_method(pinned_opponent)
            if isinstance(pinned_opponent, str)
            else pinned_opponent
        )
        frozen = method if generic else freeze_evaluation_method(method)
        system = shared_policy(frozen) if isinstance(frozen, Policy) else frozen
        _check_opponent_rosters(system, schedule)
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
            host = HostOpponent(system, schedule.num_envs, external_selection=generic)
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
    opponent = make_opponent_system(
        actor, pinned=pinned_system, vectorize_lanes=vectorize_opponent_lanes
    )
    root = jax.random.key(seed, impl="threefry2x32")
    generation = jnp.zeros(schedule.num_envs, jnp.int32)
    sampled = sample_training_configs(
        prepared.source_configs,
        root,
        generation,
        eligible_maps=schedule.arrays.eligible_maps[0],
        team_size=schedule.arrays.team_sizes[0],
        score_threshold=schedule.arrays.score_thresholds[0],
        roster_class_ids=None
        if schedule.arrays.roster_class_ids is None
        else schedule.arrays.roster_class_ids[0],
    )
    env = make(
        "tdm",
        env_config=sampled.config,
        num_envs=schedule.num_envs,
        metrics=cast(Any, metrics),
        training_facts=reward is not None,
    )
    observations, state = env.reset(training_keys(root, generation, stream="reset"))
    selection_state = None
    if generic:
        selection_state = configure_opponent_selection(
            names, history, opponent_selection
        )
        history, selection_state = select_opponents(
            history,
            selection_state,
            jnp.ones(schedule.num_envs, jnp.bool_),
            training_keys(root, generation, stream="opponent"),
        )
        if pinned_system is not None:
            if host is not None:
                host.variables = host.variables._replace(
                    choices=selection_state.choices
                )
            else:
                pinned_values = (
                    pinned_values[0]._replace(choices=selection_state.choices),
                    pinned_values[1],
                )
    else:
        history = assign_opponents(
            history,
            jnp.ones(schedule.num_envs, jnp.bool_),
            training_keys(root, generation, stream="opponent"),
            pinned_share=pinned_opponent_share,
        )
    if bool(np.asarray(jnp.any(state.lifecycle_error) | _history_invalid(history))):
        raise ValueError("Training reset or opponent initialization failed")
    native_actor = actor
    partner_settings: tuple[OpponentSelection, OpponentSelection] | None = None
    partner_values: Tree = ()
    partner_actions: tuple[ActorAction, ActorAction] | None = None
    host_partners: tuple[HostOpponent, HostOpponent] | None = None
    if slots is not None:
        partner_pool = pool({member: 1.0 for member in partner_systems})
        initial_selection = configure_opponent_selection(
            partner_names, history, partner_rule
        )
        partner_settings = (initial_selection, initial_selection)
        partner_settings = _select_partners(history, partner_settings, root, state)
        if partner_pool.execution == "host":
            host_partners = (
                HostOpponent(partner_pool, schedule.num_envs, external_selection=True),
                HostOpponent(partner_pool, schedule.num_envs, external_selection=True),
            )
            zero = jnp.zeros((schedule.num_envs, 5), jnp.int32)
            idle_actions = ActorAction(zero, zero, zero)
            partner_actions = (idle_actions, idle_actions)
            partner_system = HOST_ACTIONS_SYSTEM
        else:
            _, values, template = prepare_evaluation_system(partner_pool)
            partner_values = jax.tree.map(_uncommitted, (values, template))
            partner_system = partner_pool
        actor, opponent = _training_teams(
            native_actor,
            partner_system,
            slots,
            pinned_system,
            vectorize_lanes=vectorize_opponent_lanes,
        )
    else:
        actor = replace(actor, variables=())
    variables_a, variables_b = _team_variables(
        actor,
        opponent,
        history,
        pinned_values,
        partner_settings,
        partner_values,
        partner_actions,
    )
    memory = init_systems(
        actor,
        opponent,
        observations,
        state,
        training_keys(root, generation, stream="initialization"),
        variables_a=variables_a,
        variables_b=variables_b,
    )
    if host is not None:
        host.start(
            jax.device_get(system_inputs(observations, state, team=1)),
            _initialization_keys(memory.init_key, state.episode_id, 1),
        )
    if host_partners is not None:
        for side, holder in enumerate(host_partners):
            view = _partner_inputs(observations, state, slots, side)
            holder.start(
                jax.device_get(view),
                _member_keys(
                    _initialization_keys(memory.init_key, state.episode_id, side), 1
                ),
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
        _init_training_progress(
            num_envs=schedule.num_envs,
            history_capture_capacity=history_capture_capacity,
            opponent_members=len(names),
            partner_members=len(partner_names),
        ),
        history,
        jnp.asarray(discount, jnp.float32),
        jnp.asarray(coefficient, jnp.float32),
        pinned_values,
        selection_state,
        partner_settings,
        partner_values,
        partner_actions,
        _frozen_host_values(host, host_partners),
    )
    _check_carry(carry)
    shape = jax.eval_shape(partial(_apply, actor=actor, opponent=opponent), carry)
    if reward is not None:
        assert shape[4][4].training_facts is not None

        def scalar_spec(value: Array | jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
            """Drop only the environment axis for a scalar callback shape check."""
            return jax.ShapeDtypeStruct(value.shape[1:], value.dtype)

        validate_reward(
            reward,
            jax.tree.map(scalar_spec, state.core_state),
            jax.tree.map(scalar_spec, shape[4][4].training_facts),
            jax.tree.map(scalar_spec, shape[4][1].core_state),
            jnp.int32(0),
        )
    collection = TrainingCollection(
        actor,
        opponent,
        prepared.binding,
        schedule,
        shaping,
        collect_training_state,
        metrics,
        recording,
        shape[2][0] if slots is None else shape[2][0][0],
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
        opponent_names=names,
        opponent_records=members,
        opponent_systems=tuple(frozen_members),
        reward=reward,
        reward_identity=reward_identity,
        vectorize_opponent_lanes=vectorize_opponent_lanes,
        learner_slots=slots,
        learner_actor=None if slots is None else replace(native_actor, variables=()),
        partner_names=partner_names,
        partner_records=tuple(partner_records),
        partner_systems=tuple(partner_systems),
        host_partners=host_partners,
        pinned_system=pinned_system,
    )
    return _opponent_component_table(collection, carry), carry


def _opponent_component_table(
    collection: TrainingCollection, carry: TrainingCarry
) -> TrainingCollection:
    """Label the writer's fixed component table with stable training identities."""
    if not collection.recording:
        return collection
    components = list(collection.opponent.components or ())
    lookup = {
        str(component.get("version")): index
        for index, component in enumerate(components)
    }
    rows: list[int] = []
    for record in opponent_member_records(collection, carry):
        version = f"opponent_snapshot_{record['snapshot']}"
        if version not in lookup:
            component = {"name": record["label"], "version": version}
            if record["kind"] == "past":
                component["checkpoint"] = f"capture_{record['capture_id']}"
            elif record["kind"] == "named":
                component["checkpoint"] = record["registration_id"]
                if record.get("variables_digest") is not None:
                    component["parameters_digest"] = record["variables_digest"]
            lookup[version] = len(components)
            components.append(component)
        rows.append(lookup[version])
    actor = collection.actor
    partner_rows: list[tuple[int, ...]] = []
    for team_components in (list(actor.components or ()), components):
        partner_lookup = {
            str(item.get("version")): index
            for index, item in enumerate(team_components)
        }
        indices: list[int] = []
        for index, member in enumerate(collection.partner_records):
            version = f"partner_member_{index}"
            if version not in partner_lookup:
                entry = {
                    "name": member["name"],
                    "version": version,
                    "checkpoint": member["registration_id"],
                }
                if member.get("variables_digest") is not None:
                    entry["parameters_digest"] = member["variables_digest"]
                partner_lookup[version] = len(team_components)
                team_components.append(entry)
            indices.append(partner_lookup[version])
        partner_rows.append(tuple(indices))
        if len(partner_rows) == 1:
            actor = replace(actor, components=tuple(team_components))
    return replace(
        collection,
        actor=actor,
        opponent=replace(collection.opponent, components=tuple(components)),
        opponent_trace_rows=tuple(rows),
        partner_trace_rows=cast(
            tuple[tuple[int, ...], tuple[int, ...]], tuple(partner_rows)
        ),
    )


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
    variables_a, variables_b = _team_variables(
        actor,
        opponent,
        carry.history,
        carry.pinned_opponent,
        carry.partner_selection,
        carry.partner_values,
        carry.partner_actions,
    )
    return _apply_and_track(
        carry.env,
        carry.observations,
        state,
        carry.memory,
        carry.tracking,
        actor=actor,
        opponent=opponent,
        variables_a=variables_a,
        variables_b=variables_b,
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
        custom_rewards=jnp.zeros((b, 5), jnp.float32)
        if collection.reward is not None
        else None,
        learner_active=None
        if collection.learner_slots is None
        else jnp.zeros((b, 5), jnp.bool_),
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
        roster_class_ids=None
        if carry.schedule.roster_class_ids is None
        else carry.schedule.roster_class_ids[stage],
    )
    selection = carry.opponent_selection
    pinned = carry.pinned_opponent
    keys = training_keys(carry.root_key, state.reset_generation, stream="opponent")
    if selection is None:
        history = assign_opponents(
            carry.history, mask, keys, pinned_share=collection.pinned_opponent_share
        )
    else:
        history, selection = select_opponents(carry.history, selection, mask, keys)
        if collection.opponent_names and collection.host_opponent is None:
            pinned = (pinned[0]._replace(choices=selection.choices), pinned[1])
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
        pinned_opponent=pinned,
        opponent_selection=selection,
        partner_selection=None
        if carry.partner_selection is None
        else _select_partners(
            history, carry.partner_selection, carry.root_key, state, mask=mask
        ),
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
    opponent_snapshot, opponent_update = opponent_identity(carry.history)
    if carry.opponent_selection is not None:
        opponent_snapshot = jnp.where(
            opponent_snapshot == -2,
            -3 - carry.opponent_selection.choices,
            opponent_snapshot,
        )
    slot = opponent_counter_row(opponent_snapshot, carry.history.capture_capacity)
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
    if carry.partner_selection is not None:
        assert (
            progress.partner_used is not None and collection.learner_slots is not None
        )
        complement = jnp.asarray(
            [slot not in collection.learner_slots for slot in range(5)], jnp.bool_
        )
        used = progress.partner_used
        for side, settings in enumerate(carry.partner_selection):
            alive = before.core_state.alive_mask[:, side * 5 : (side + 1) * 5]
            active = before.config.agent_profile.active_mask[
                :, side * 5 : (side + 1) * 5
            ]
            played = valid & jnp.any(active & alive & complement, axis=1)
            if side == 1 and len(carry.pinned_opponent):
                played &= carry.history.lane_snapshot != -2
            used = used.at[settings.choices].max(played)
        progress = progress._replace(partner_used=used)
    priority = info.priority
    if priority is not None:
        availability = priority.valid & ended[:, None]
        priority = MetricValues(
            jnp.where(availability, priority.values, 0), availability
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
    custom = None
    if collection.reward is not None:
        assert info.training_facts is not None
        custom = jax.vmap(
            partial(reward_adjustments, collection.reward),
            in_axes=(0, 0, 0, None),
        )(
            before.core_state,
            info.training_facts,
            state.core_state,
            carry.progress.rounds,
        )[:, :5]
    row = TrainingTransition(
        carry.observations,
        _team_mask(before),
        actions,
        learning[0] if collection.learner_slots is None else learning[0][0],
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
        opponent_snapshot,
        encode_training_state(before.core_state, before.config)
        if collection.collect_training_state
        else None,
        custom_rewards=custom,
        learner_active=_learner_active(collection, before),
    )
    policy_ids = jnp.full_like(memory.policy_trace.policy_ids, -1)
    if collection.recording:
        policy_ids = memory.policy_trace.policy_ids.at[:, 5:].set(
            jnp.where(
                before.config.agent_profile.active_mask[:, 5:],
                jnp.asarray(collection.opponent_trace_rows, jnp.int32)[slot, None],
                -1,
            )
        )
    if collection.recording and carry.partner_selection is not None:
        assert collection.learner_slots is not None
        complement = jnp.asarray(
            [slot not in collection.learner_slots for slot in range(5)], jnp.bool_
        )
        for side, settings in enumerate(carry.partner_selection):
            active = before.config.agent_profile.active_mask[
                :, side * 5 : (side + 1) * 5
            ]
            owned = active & complement
            if side == 1 and len(carry.pinned_opponent):
                owned &= (carry.history.lane_snapshot != -2)[:, None]
            member_ids = jnp.asarray(collection.partner_trace_rows[side], jnp.int32)[
                settings.choices
            ]
            prior = policy_ids[:, side * 5 : (side + 1) * 5]
            policy_ids = policy_ids.at[:, side * 5 : (side + 1) * 5].set(
                jnp.where(owned, member_ids[:, None], prior)
            )
    trace = memory.policy_trace._replace(policy_ids=policy_ids)
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
    if collection.host_opponent is not None or collection.host_partners is not None:
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
        learner_memory(initial),
        carry.observations,
        _team_mask(state),
        state.config.agent_profile.active_mask[:, :5],
        state.core_state.alive_mask[:, :5],
        state.done.done,
        encode_training_state(state.core_state, state.config)
        if collection.collect_training_state
        else None,
        carry.progress.rounds - initial.progress.rounds,
        final_learner_active=_learner_active(collection, state),
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
    if collection.host_opponent is not None or collection.host_partners is not None:
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


def _selection_value(
    collection: TrainingCollection, settings: OpponentSelection
) -> Mapping[str, float] | Sequence[str]:
    """Read the small saved selection table as its original public value."""
    return _selection_declaration(collection.opponent_names, settings)


def _selection_declaration(
    member_names: tuple[str, ...], settings: OpponentSelection
) -> Mapping[str, float] | Sequence[str]:
    """Read one selector without changing its choice cursor or live bindings."""
    names = ("self", "past", *member_names)
    if bool(settings.ordered):
        return [names[int(index)] for index in np.asarray(settings.order)]
    return dict(zip(names, np.asarray(settings.shares).tolist(), strict=True))


def _bind_opponent_references(  # pyright: ignore[reportUnusedFunction]
    collection: TrainingCollection, references: Mapping[str, str]
) -> TrainingCollection:
    """Attach original references after the runner resolves and opens members.

    references must come from the runner's trusted loader results for this same
    collection. It preserves source paths lost when resolved Systems are passed
    into setup. Recompute exposure evidence with the frozen member and original
    export path. This changes no registration, weights, memory or execution.
    Unknown names or invalid reference strings raise before returning a change.
    """
    if set(references) - set(collection.opponent_names):
        raise ValueError("Opponent references must name declared members")
    records: list[dict[str, Any]] = []
    for member, record in zip(
        collection.opponent_systems, collection.opponent_records, strict=True
    ):
        reference = references.get(record["name"])
        if reference is None:
            records.append(record)
            continue
        if not isinstance(cast(object, reference), str) or not reference.strip():
            raise ValueError("Opponent references must be nonempty strings")
        export = Path(reference) if Path(reference).is_dir() else None
        records.append(
            {
                **record,
                "reference": reference,
                "evidence": pinned_opponent_evidence(
                    collection.binding, member, export=export
                ),
            }
        )
    return replace(collection, opponent_records=tuple(records))


def _bind_partner_references(  # pyright: ignore[reportUnusedFunction]
    collection: TrainingCollection, references: Mapping[str, str]
) -> TrainingCollection:
    """Attach trusted original partner references using the shared binding owner."""
    rebound = _bind_opponent_references(
        replace(
            collection,
            opponent_names=collection.partner_names,
            opponent_records=collection.partner_records,
            opponent_systems=collection.partner_systems,
        ),
        references,
    )
    return replace(collection, partner_records=rebound.opponent_records)


def partner_selection_value(
    collection: TrainingCollection, carry: TrainingCarry
) -> object:
    """Read the saved partner rule for checkpoint reconstruction; None disables it."""
    if carry.partner_selection is None:
        return None
    value = _selection_declaration(collection.partner_names, carry.partner_selection[0])
    if isinstance(value, Mapping):
        return {
            name: share for name, share in value.items() if name not in ("self", "past")
        }
    return value


def update_partner_selection(
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    selection: Mapping[str, float] | Sequence[str] | None = None,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Change future partner shares/order on both teams, preserving living games.

    Names must already be declared frozen members. None keeps the saved rule.
    Both teams retain their own choices and memory. A changed list restarts at
    its first entry on the next real game; unchanged lists keep their position.
    Same-shaped changes keep the descriptor and compiled program reusable.
    """
    if carry.partner_selection is None:
        if selection is not None:
            raise ValueError("Partner selection requires a declared partner population")
        return collection, carry
    rule = (
        partner_selection_value(collection, carry) if selection is None else selection
    )
    _partner_rule(collection.partner_names, rule)
    settings = cast(
        tuple[OpponentSelection, OpponentSelection],
        tuple(
            configure_opponent_selection(
                collection.partner_names, carry.history, rule, previous=side
            )
            for side in carry.partner_selection
        ),
    )
    changed = carry._replace(partner_selection=settings)
    if carry.root_key.committed:
        changed = jax.device_put(changed, carry.root_key.sharding)
    if _schema(changed) == _schema(carry):
        return collection, changed
    return replace(collection, carry_spec=jax.eval_shape(_retain, changed)), changed


def _partner_pool_memory(carry: TrainingCarry, side: int) -> Tree:
    """Read a JAX partner pool's whole memory from its ordinary team composition."""
    if side == 0:
        return carry.memory.team_a.members[1]
    native = (
        carry.memory.team_b
        if not len(carry.pinned_opponent)
        else carry.memory.team_b.members[0]
    )
    return native.members[1]


def _replace_partner_memories(
    carry: TrainingCarry, memories: tuple[Tree, Tree]
) -> TrainingCarry:
    """Replace only partner pool trees; learner/external opponent memory stays put."""
    a = carry.memory.team_a
    a = a._replace(members=(a.members[0], memories[0]))
    b = carry.memory.team_b
    native = b if not len(carry.pinned_opponent) else b.members[0]
    native = native._replace(members=(native.members[0], memories[1]))
    b = (
        native
        if not len(carry.pinned_opponent)
        else b._replace(members=(native, b.members[1]))
    )
    return carry._replace(memory=carry.memory._replace(team_a=a, team_b=b))


def append_training_partners(
    collection: TrainingCollection,
    carry: TrainingCarry,
    additions: Mapping[str, System | Policy | str],
    *,
    selection: Mapping[str, float] | Sequence[str] | None = None,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Append frozen partner names at a child boundary without replacing live members.

    Resource scopes must already be open. Existing weights, selected members and
    full-batch memories stay exact. Only new members initialize with no valid
    lanes. Host state keeps its existing holder clocks and reset hooks. New
    structures may compile anew. Enable partner training when creating the run;
    this helper does not change learner_slots or convert a full-learner run.
    """
    if carry.partner_selection is None or collection.learner_slots is None:
        raise ValueError("Append partners to an existing partner-trained run")
    if not additions:
        return update_partner_selection(collection, carry, selection=selection)
    if set(additions) & set(collection.partner_names):
        raise ValueError(
            "Added partner names must be new; old members cannot be rebound"
        )
    new_systems: list[System] = []
    new_records: list[dict[str, Any]] = []
    for name, member in additions.items():
        system, record = _prepare_opponent_member(
            name, member, collection.schedule, collection.binding, team_index=None
        )
        _check_partner_rosters(system, collection.schedule, collection.learner_slots)
        new_systems.append(system)
        new_records.append(record)
    systems = (*collection.partner_systems, *new_systems)
    names = (*collection.partner_names, *additions)
    rule = (
        partner_selection_value(collection, carry) if selection is None else selection
    )
    _partner_rule(names, rule)
    settings = cast(
        tuple[OpponentSelection, OpponentSelection],
        tuple(
            configure_opponent_selection(names, carry.history, rule, previous=side)
            for side in carry.partner_selection
        ),
    )
    combined = pool({member: 1.0 for member in systems})
    _, fresh_values, template = prepare_evaluation_system(combined)
    count = len(collection.partner_names)
    old_holders = collection.host_partners
    old_values = (
        carry.partner_values[0] if old_holders is None else old_holders[0].variables
    )
    values = fresh_values._replace(
        members=(*old_values.members, *fresh_values.members[count:]),
        templates=(*old_values.templates, *fresh_values.templates[count:]),
    )
    memories: list[Tree] = []
    holders: list[HostOpponent] = []
    for side in range(2):
        old_memory = (
            _partner_pool_memory(carry, side)
            if old_holders is None
            else old_holders[side].memory
        )
        inputs = _partner_inputs(
            carry.observations, carry.state, collection.learner_slots, side
        )
        idle = inputs._replace(
            valid=jnp.zeros_like(inputs.valid),
            controlled_mask=jnp.zeros_like(inputs.active_mask),
        )
        keys = _member_keys(
            _initialization_keys(carry.memory.init_key, carry.state.episode_id, side), 1
        )
        extra = []
        for index in range(count, len(systems)):
            execution = _execution(systems[index])
            extra.append(
                _initial_memory(
                    execution,
                    values.members[index],
                    values.templates[index],
                    jax.device_get(idle) if execution.execution == "host" else idle,
                    _member_keys(keys, index),
                )
            )
        memory = _CompositionMemory(old_memory.choice, (*old_memory.members, *extra))
        if combined.execution == "host":
            holder = HostOpponent(
                combined, collection.schedule.num_envs, external_selection=True
            )
            holder.variables = values._replace(choices=settings[side].choices)
            holder.template, holder.memory = template, memory
            if old_holders is not None:
                old = old_holders[side]
                holder.generations = old.generations.copy()
                holder.next_round, holder.failure, holder.failed_round = (
                    old.next_round,
                    old.failure,
                    old.failed_round,
                )
            else:
                playing = np.ones(collection.schedule.num_envs, np.bool_)
                if side == 1 and len(carry.pinned_opponent):
                    playing &= np.asarray(carry.history.lane_snapshot) != -2
                holder.generations = np.where(
                    playing, np.asarray(carry.state.reset_generation), -1
                ).astype(np.int64)
            holders.append(holder)
            memories.append(())
        else:
            memories.append(memory)
    assert collection.learner_actor is not None
    pinned = collection.pinned_system
    actor, opponent = _training_teams(
        replace(collection.learner_actor, variables=carry.history.current_variables),
        HOST_ACTIONS_SYSTEM if holders else combined,
        collection.learner_slots,
        pinned,
        vectorize_lanes=collection.vectorize_opponent_lanes,
    )
    changed = _replace_partner_memories(carry, cast(tuple[Tree, Tree], tuple(memories)))
    zero = jnp.zeros((collection.schedule.num_envs, 5), jnp.int32)
    idle_action = ActorAction(zero, zero, zero)
    changed = changed._replace(
        partner_values=() if holders else (values, template),
        partner_actions=(idle_action, idle_action) if holders else None,
        partner_selection=settings,
        progress=carry.progress._replace(
            partner_used=jnp.pad(
                cast(Array, carry.progress.partner_used), ((0, len(additions)),)
            )
        ),
        frozen_host_values=_frozen_host_values(
            collection.host_opponent,
            cast(tuple[HostOpponent, HostOpponent], tuple(holders))
            if holders
            else None,
        ),
    )
    if carry.root_key.committed:
        changed = jax.device_put(changed, carry.root_key.sharding)
    updated = replace(
        collection,
        actor=replace(actor, components=collection.actor.components),
        opponent=replace(opponent, components=collection.opponent.components),
        partner_names=names,
        partner_records=(*collection.partner_records, *new_records),
        partner_systems=systems,
        host_partners=cast(tuple[HostOpponent, HostOpponent], tuple(holders))
        if holders
        else None,
        carry_spec=jax.eval_shape(_retain, changed),
    )
    return _opponent_component_table(updated, changed), changed


def append_training_opponents(
    collection: TrainingCollection,
    carry: TrainingCarry,
    additions: Mapping[str, System | Policy | str],
    *,
    selection: Mapping[str, float] | Sequence[str] | None = None,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Append frozen named members at a child boundary without restarting games.

    additions contains new aliases only. The caller opens their resource scopes
    before this call and keeps them open through training. Existing member
    values, full-batch memories, choices and host clocks remain unchanged. Only
    new members receive invalid-lane placeholder initialization. New games use
    selection, or retain the previous rule when None. A larger pool changes
    static structure and may compile a new program. No action or game reset runs.
    Legacy permanent-pin collections cannot use this route. Invalid names or
    member contracts raise before a new collection is returned.
    """
    if collection.pinned_opponent_share:
        raise ValueError("Append named opponents through the generic selection API")
    if not additions:
        return update_opponent_selection(collection, carry, selection=selection)
    if set(additions) & set(collection.opponent_names):
        raise ValueError(
            "Added opponent names must be new; old members cannot be rebound"
        )
    new_systems: list[System] = []
    new_records: list[dict[str, Any]] = []
    for name, member in additions.items():
        system, record = _prepare_opponent_member(
            name, member, collection.schedule, collection.binding
        )
        new_systems.append(system)
        new_records.append(record)
    names = (*collection.opponent_names, *additions)
    systems = (*collection.opponent_systems, *new_systems)
    combined = pool({member: 1.0 for member in systems})
    _, variables, template = prepare_evaluation_system(combined)
    old_count = len(collection.opponent_names)
    old_holder = collection.host_opponent
    old_values = (
        old_holder.variables
        if old_holder is not None
        else carry.pinned_opponent[0]
        if old_count
        else None
    )
    if carry.partner_selection is not None:
        network_memory = (
            carry.memory.team_b.members[0] if old_count else carry.memory.team_b
        )
        external_memory = carry.memory.team_b.members[1] if old_count else None
    else:
        network_memory = carry.memory.team_b[0] if old_count else carry.memory.team_b
        external_memory = carry.memory.team_b[1] if old_count else None
    old_pool_memory = old_holder.memory if old_holder is not None else external_memory
    if old_values is not None:
        variables = variables._replace(
            members=(*old_values.members, *variables.members[old_count:]),
            templates=(*old_values.templates, *variables.templates[old_count:]),
        )
    inputs = system_inputs(carry.observations, carry.state, team=1)
    idle = inputs._replace(
        valid=jnp.zeros_like(inputs.valid),
        controlled_mask=jnp.zeros_like(inputs.active_mask),
    )
    keys = _initialization_keys(carry.memory.init_key, carry.state.episode_id, 1)
    new_memories = []
    for index in range(old_count, len(systems)):
        execution = _execution(systems[index])
        new_memories.append(
            _initial_memory(
                execution,
                variables.members[index],
                variables.templates[index],
                jax.device_get(idle) if execution.execution == "host" else idle,
                _member_keys(keys, index),
            )
        )
    previous = carry.opponent_selection
    if selection is None and previous is not None:
        selection = _selection_value(collection, previous)
    history = carry.history._replace(external_pin=jnp.bool_(True))
    settings = configure_opponent_selection(
        names, history, selection, previous=previous
    )
    if previous is None:
        starts = (
            jnp.sum(carry.state.reset_generation, dtype=jnp.int32)
            + carry.state.episode_id.shape[0]
        )
        settings = settings._replace(game_starts=starts, order_start=starts)
    variables = variables._replace(choices=settings.choices)
    memory = _CompositionMemory(
        settings.choices if old_pool_memory is None else old_pool_memory.choice,
        (*(() if old_pool_memory is None else old_pool_memory.members), *new_memories),
    )
    holder = None
    if combined.execution == "host":
        holder = HostOpponent(
            combined, collection.schedule.num_envs, external_selection=True
        )
        holder.variables, holder.template, holder.memory = variables, template, memory
        if old_holder is not None:
            holder.generations = old_holder.generations.copy()
            holder.next_round = old_holder.next_round
            holder.failure, holder.failed_round = (
                old_holder.failure,
                old_holder.failed_round,
            )
        elif old_count:
            holder.generations = np.where(
                np.asarray(history.lane_snapshot) == -2,
                np.asarray(carry.state.reset_generation),
                -1,
            ).astype(np.int64)
        zero = jnp.zeros((collection.schedule.num_envs, 5), jnp.int32)
        pinned_values = (ActorAction(zero, zero, zero), ())
        team_b = (network_memory, ())
        pinned_system = HOST_ACTIONS_SYSTEM
    else:
        pinned_values = (variables, template)
        team_b = (network_memory, memory)
        pinned_system = combined
    registration = normalize_system_registration(
        combined, phase="validation", frozen=True
    )[1]
    pinned_record = {
        "reference": None,
        "name": combined.name,
        "execution": combined.execution,
        "registration_id": canonical_digest_sha256(registration),
        "registration": registration,
        "variables_digest": tree_digest(variables._replace(choices=None))
        if combined.execution == "jax"
        else None,
        "evidence": pinned_opponent_evidence(collection.binding, combined),
        "memory_rule": "saved in the carry"
        if holder is None
        else "host only, not saved",
    }
    progress = carry.progress._replace(
        opponent_starts=jnp.pad(
            carry.progress.opponent_starts, ((0, len(additions)), (0, 0))
        ),
        opponent_steps=jnp.pad(
            carry.progress.opponent_steps, ((0, len(additions)), (0, 0))
        ),
    )
    if carry.partner_selection is not None:
        team_b = _CompositionMemory(
            (history.lane_snapshot == -2).astype(jnp.int32), tuple(team_b)
        )
        assert (
            collection.learner_actor is not None
            and collection.learner_slots is not None
        )
        _, opponent = _training_teams(
            replace(
                collection.learner_actor, variables=carry.history.current_variables
            ),
            HOST_ACTIONS_SYSTEM
            if collection.host_partners is not None
            else pool({member: 1.0 for member in collection.partner_systems}),
            collection.learner_slots,
            pinned_system,
            vectorize_lanes=collection.vectorize_opponent_lanes,
        )
    else:
        opponent = make_opponent_system(
            replace(collection.actor, variables=carry.history.current_variables),
            pinned=pinned_system,
            vectorize_lanes=collection.vectorize_opponent_lanes,
        )
    changed = carry._replace(
        history=history,
        progress=progress,
        opponent_selection=settings,
        pinned_opponent=pinned_values,
        memory=carry.memory._replace(team_b=team_b),
        frozen_host_values=_frozen_host_values(holder, collection.host_partners),
    )
    if carry.root_key.committed:
        changed = jax.device_put(changed, carry.root_key.sharding)
    updated = replace(
        collection,
        opponent=opponent,
        pinned_system=pinned_system,
        opponent_names=names,
        opponent_records=(*collection.opponent_records, *new_records),
        opponent_systems=systems,
        pinned_opponent=pinned_record,
        host_opponent=holder,
        carry_spec=jax.eval_shape(_retain, changed),
    )
    updated = replace(
        updated,
        opponent=replace(updated.opponent, components=collection.opponent.components),
    )
    return _opponent_component_table(updated, changed), changed


def update_opponent_selection(
    collection: TrainingCollection,
    carry: TrainingCarry,
    *,
    selection: Mapping[str, float] | Sequence[str] | None = None,
    past: Mapping[int, float] | None = None,
) -> tuple[TrainingCollection, TrainingCarry]:
    """Change future opponent choices at a host update boundary.

    selection uses the declared named members plus self/past. None retains the
    current rule; on the legacy default path it starts the 80/20 rule. past gives
    eligible capture IDs and relative weights; omitted IDs get zero, while
    future new captures start with uniform weight. No live game, member values,
    memory, replay row or random stream changes. Return a refreshed descriptor
    only if shapes change (for example, a different order length).
    Legacy pinned settings cannot be combined with this API. Invalid settings
    raise before returning changed state. This helper starts no game.
    """
    if collection.pinned_opponent_share:
        raise ValueError("Generic selection cannot change legacy pinned settings")
    if selection is None and carry.opponent_selection is not None:
        selection = _selection_value(collection, carry.opponent_selection)
    settings = configure_opponent_selection(
        collection.opponent_names,
        carry.history,
        selection,
        previous=carry.opponent_selection,
        past=past,
    )
    if carry.opponent_selection is None:
        settings = settings._replace(
            game_starts=jnp.sum(carry.state.reset_generation, dtype=jnp.int32)
            + carry.state.episode_id.shape[0],
            order_start=jnp.sum(carry.state.reset_generation, dtype=jnp.int32)
            + carry.state.episode_id.shape[0],
        )
    changed = carry._replace(opponent_selection=settings)
    if carry.root_key.committed:
        changed = jax.device_put(changed, carry.root_key.sharding)
    if _schema(changed) == _schema(carry):
        return collection, changed
    return replace(collection, carry_spec=jax.eval_shape(_retain, changed)), changed


def opponent_member_records(
    collection: TrainingCollection, carry: TrainingCarry
) -> list[dict[str, Any]]:
    """Return stable counter-row labels and exact named member registrations.

    Rows are self, legacy pin, reserved stable capture IDs, then named members.
    Past rows state whether the copy is resident and eligible for new games.
    Resident captures include their actual step/update. Evicted copies retain
    their ID; the run's append-only capture exports supply their original time
    and actor identity. Empty reserved rows have no capture timing. This reads
    only small host metadata, never weights or trajectories.
    """
    history = jax.device_get(
        carry.history._replace(
            current_variables=(),
            historical_variables=(),
            pinned_variables=None,
        )
    )
    captures = {
        int(i): (int(r), int(u), bool(e))
        for i, r, u, e in zip(
            np.asarray(history.captured_ids).tolist(),
            np.asarray(history.captured_rounds).tolist(),
            np.asarray(history.captured_updates).tolist(),
            np.asarray(history.eligible).tolist(),
            strict=True,
        )
        if i >= 0
    }
    result: list[dict[str, Any]] = [
        {"snapshot": -1, "label": "current weights", "name": "self", "kind": "self"},
        {
            "snapshot": -2,
            "name": "pin",
            "label": "permanent first-update pin"
            if collection.pinned_opponent is None
            else f"pinned: {collection.pinned_opponent['name']}",
            "kind": "pin",
        },
    ]
    for identifier in range(int(history.capture_capacity)):
        record: dict[str, Any] = {
            "snapshot": identifier,
            "capture_id": identifier,
            "name": "past",
            "label": f"capture {identifier}",
            "kind": "past",
            "resident": identifier in captures,
            "eligible": captures.get(identifier, (0, 0, False))[2],
        }
        if identifier in captures:
            rounds, update, _ = captures[identifier]
            record.update(
                env_steps=rounds * collection.schedule.num_envs, update=update
            )
        result.append(record)
    for index, member in enumerate(collection.opponent_records):
        result.append(
            {"snapshot": -3 - index, "label": member["name"], "kind": "named", **member}
        )
    return result


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
    opponent_rows labels stable counts: current, permanent pin, then capture IDs.
    A named pinned opponent also contributes its saved registration record.
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
        **(
            {}
            if carry.schedule.roster_class_ids is None
            else {
                "roster_class_ids_by_episode_stage": np.asarray(
                    carry.schedule.roster_class_ids
                )[:stages].tolist(),
                "eligible_maps_by_episode_stage": np.asarray(
                    carry.schedule.eligible_maps
                )[:stages].tolist(),
            }
        ),
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
        "opponent_rows": [
            record["label"] for record in opponent_member_records(collection, carry)
        ],
        "opponent_members": opponent_member_records(collection, carry),
        **(
            {}
            if collection.pinned_opponent is None
            else {"pinned_opponent": collection.pinned_opponent}
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
    result = (
        carry,
        system_inputs(carry.observations, state, team=1),
        carry.history.lane_snapshot == -2,
        _action_keys(roots, state.episode_id, 1),
        _initialization_keys(carry.memory.init_key, state.episode_id, 1),
        state.reset_generation,
        carry.progress.rounds,
        _failed(carry),
    )

    if collection.host_partners is None:
        return result
    return (
        *result,
        (
            _partner_inputs(carry.observations, state, collection.learner_slots, 0),
            _partner_inputs(carry.observations, state, collection.learner_slots, 1),
            _action_keys(roots, state.episode_id, 0),
            _initialization_keys(carry.memory.init_key, state.episode_id, 0),
        ),
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
    holder = collection.host_opponent
    initial = carry
    real = min(length, int(carry.schedule.total_rounds) - int(carry.progress.rounds))
    decide = _host_decision_compiled(collection)
    step = _host_step_compiled(collection)
    rows: list[TrainingTransition] = []
    for _ in range(real):
        decision = decide(carry)
        carry, inputs, pinned, keys, init_keys, generations, rounds, failed = cast(
            tuple[TrainingCarry, Any, Array, Array, Array, Array, Array, Array],
            decision[:8],
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
        if holder is not None:
            if carry.opponent_selection is not None:
                holder.variables = holder.variables._replace(
                    choices=np.asarray(carry.opponent_selection.choices)
                )
            actions = holder.act(
                int(rounds),
                inputs,
                np.asarray(pinned),
                keys,
                init_keys,
                np.asarray(generations),
            )
            carry = carry._replace(
                pinned_opponent=(
                    ActorAction(*(jnp.asarray(head) for head in actions)),
                    (),
                )
            )
        if collection.host_partners is not None:
            assert carry.partner_selection is not None
            input_a, input_b, a_keys, a_init_keys = decision[8]
            chosen: list[ActorAction] = []
            for side, partner_holder in enumerate(collection.host_partners):
                partner_holder.variables = partner_holder.variables._replace(
                    choices=np.asarray(carry.partner_selection[side].choices)
                )
                playing = np.ones(collection.schedule.num_envs, np.bool_)
                if side == 1 and len(carry.pinned_opponent):
                    playing &= ~np.asarray(pinned)
                actions = partner_holder.act(
                    int(rounds),
                    input_a if side == 0 else input_b,
                    playing,
                    _member_keys(a_keys if side == 0 else keys, 1),
                    _member_keys(a_init_keys if side == 0 else init_keys, 1),
                    np.asarray(generations),
                )
                chosen.append(ActorAction(*(jnp.asarray(head) for head in actions)))
            carry = carry._replace(
                partner_actions=cast(tuple[ActorAction, ActorAction], tuple(chosen))
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
