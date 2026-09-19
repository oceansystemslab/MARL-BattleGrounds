"""Track episode sources and real experience without owning a training method.

Initialize on the host, carry the immutable tracker beside EnvironmentState, and
call track_episode_step once for each submitted step result. Numerical updates
work in JAX loops. Public stage methods inspect only small counters on the host;
training uses the shared numerical reset after its own compiled boundary checks.
Recording is optional; disabled recording performs no hashing or disk work.
"""

from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from marl_battlegrounds.core.types import DoneFlags, EnvConfig, Reward
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    EpisodeInfo,
    _checked_increment,  # pyright: ignore[reportPrivateUsage]
    episode_advanced,
)
from marl_battlegrounds.evaluation.recording_types import (
    TRACKING_ERROR_ACCOUNTING,
    TRACKING_ERROR_DECLARATION,
    TRACKING_ERROR_OVERFLOW,
    EpisodeStartRecords,
)
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import (
    _source_config_with_class_ids,  # pyright: ignore[reportPrivateUsage]
    prepare_exact_env_config,
    spawn_locations_for_source,
)

if TYPE_CHECKING:
    from marl_battlegrounds.autoreset import AutoReset

type StepResult = tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]

_MAX_COUNT = int(np.iinfo(np.int32).max)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeTrackingState:
    """Keep one source bank, live episode bindings and checked experience counts.

    Attributes
    ----------
    num_envs : int | None
        Static native batch size, or None for scalar execution. All lane fields
        have shape L: () for scalar calls or (B,) for a native batch.
    record_starts : bool
        Static recording choice. False omits start records and source hashing.
    source_configs : EnvConfig | None
        One immutable dynamic bank, with leading source axis N on every leaf.
        None means no known bank. Do not mutate, reorder or donate this bank.
    source_table_id : Array | None
        Uint32 (8,) full bank digest when recording; zero for no bank. None when
        recording is disabled. A digest is a claim, not verified provenance.
    episode_id, reset_generation, config_origin_generation : Array
        Int32 L identifying each bound episode and its configuration origin.
    source_index, spawn_locations : Array
        Int32 L source row and spawn choice (0 default, 1 swapped, -1 unknown).
    source_class_ids : Array
        Int32 L+(10,) source roster before spawn exchange. Ten -1 values retain
        the original profile. Explicit rows use compact classes 1..5 with zero
        padding per team. Unknown sources always carry the -1 sentinel.
    source_known : Array
        Bool L; a checked source relationship exists. Ambiguous banks can have
        source_known=True and spawn_locations=-1.
    first_transition_seen : Array
        Bool L; this episode has already advanced. This also marks permitted
        unrecorded attachment midway through an episode.
    episode_start_stage : Array
        Int32 L original start-stage ordinal, or -1 when undeclared or unknown.
    last_decision_step, accounted_transition_count : Array
        Int32 L last accounted local decision and cumulative real-step count.
        The local index is -1 before the episode's first real transition.
    error_flags : Array
        Sticky int32 L flags: declaration=1, accounting=2, counter overflow=4.
    stage_ordinal, stage_round_budget, stage_rounds : Array
        Int32 scalars. Ordinal starts at -1; the other fields start at zero.
    stage_counts : Array
        Int32 L+(3,) counts for default, swapped and unreported real steps.
        Before the first stage these count experience since initialization.
    full_batch_rounds_valid : Array
        Bool scalar. False when any declared round includes padding.

    Notes
    -----
    All non-static fields are dynamic JAX data. Counters stop at 2,147,483,647
    and retain error evidence instead of wrapping. Save the tracker together
    with its exact environment, RNG and method memory to continue later. This
    record does not coordinate a learner checkpoint with a recording writer.
    """

    num_envs: int | None = field(metadata={"static": True})
    record_starts: bool = field(metadata={"static": True})
    source_configs: EnvConfig | None
    source_table_id: Array | None
    episode_id: Array
    reset_generation: Array
    config_origin_generation: Array
    source_index: Array
    source_class_ids: Array
    spawn_locations: Array
    source_known: Array
    first_transition_seen: Array
    episode_start_stage: Array
    last_decision_step: Array
    accounted_transition_count: Array
    error_flags: Array
    stage_ordinal: Array
    stage_round_budget: Array
    stage_rounds: Array
    stage_counts: Array
    full_batch_rounds_valid: Array

    def begin_stage(
        self, state: EnvironmentState, *, total_env_steps: int
    ) -> EpisodeTrackingState:
        """Begin a balanced stage while preserving all live episode bindings.

        Parameters
        ----------
        state : EnvironmentState
            Latest environment state from the same execution context. Reset-only
            changes are allowed; the next tracked step handles new bindings.
        total_env_steps : int
            Positive real-step budget, excluding bool. Must be divisible by the
            positive even native batch size. Each lane must have enough remaining
            int32 count capacity for its share.

        Returns
        -------
        EpisodeTrackingState
            New tracker with cleared stage counters and the next stage ordinal.
            The input tracker and live games are unchanged.

        Raises
        ------
        ValueError
            Accounting or lifecycle errors, an unfinished prior stage, scalar or
            odd execution, invalid budget, or exhausted counters.

        Notes
        -----
        Host-only. Reads small counters, never the full state or source bank.
        It cannot detect an older matching state/tracker pair or discarded branch.
        """
        snapshot = _boundary_snapshot(self, state)
        count = self.num_envs
        if count is None or count <= 0 or count % 2:
            raise ValueError("balanced stages require a positive even native batch")
        if (
            isinstance(total_env_steps, (bool, np.bool_))
            or not isinstance(total_env_steps, Integral)
            or total_env_steps <= 0
            or total_env_steps % count
        ):
            raise ValueError(
                "total_env_steps must be positive and divisible by num_envs"
            )
        rounds = int(total_env_steps) // count
        if rounds > _MAX_COUNT or any(
            rounds > _MAX_COUNT - int(value) for value in snapshot["accounted"].flat
        ):
            raise ValueError("stage budget exceeds a lane's remaining int32 capacity")
        ordinal = int(snapshot["ordinal"])
        if ordinal >= 0 and _summary(self, snapshot)["status"] != "complete":
            raise ValueError(
                "the prior stage must be complete before beginning another"
            )
        if ordinal == _MAX_COUNT:
            raise ValueError("stage ordinal has exhausted its int32 capacity")
        return _begin_stage_numerical(self, round_budget=jnp.asarray(rounds, jnp.int32))

    def stage_summary(self, state: EnvironmentState) -> dict[str, int | str | None]:
        """Read exact stage totals after checking the latest environment state.

        Parameters
        ----------
        state : EnvironmentState
            Latest state paired with this tracker. Legal reset-only changes need
            no accounting call. Pass both values returned by the latest scan.

        Returns
        -------
        dict
            requested_env_steps (None before a stage), env_steps, default_spawn_steps,
            swapped_spawn_steps, unreported_spawn_steps, status and reason. Counts
            are Python integers. Status is not_started, incomplete or complete.
            A complete stage has reason=None. Incomplete reasons, in priority order,
            are invalid_full_batch_round, budget_exceeded, budget_unmet,
            unreported_spawn_steps and unequal_spawn_steps.

        Raises
        ------
        ValueError
            Current/accounted counts differ, errors are pending or stored stage
            counters disagree. Errors are not ordinary incomplete coverage.

        Notes
        -----
        Host-only and read-only. Transfers small counters, never full state or
        the source bank. It consumes no recording records and cannot detect an
        older matching state/tracker pair or discarded execution branch.
        """
        return _summary(self, _boundary_snapshot(self, state))


def _begin_stage_numerical(
    tracking: EpisodeTrackingState, *, round_budget: Array
) -> EpisodeTrackingState:
    """Clear stage counters after the caller has checked a stage boundary.

    tracking is the latest numerical tracker. round_budget is a positive int32
    scalar giving real rounds in the next stage. The caller must first check
    prior completion, matching environment counts, pending failures and enough
    room in all counters and the stage ordinal. No checks or host reads occur
    here, so this helper can run inside JAX loops. Return a new tracker with the
    next ordinal and cleared stage fields; preserve all episode bindings and
    cumulative counters. Inputs are unchanged.
    """
    return replace(
        tracking,
        stage_ordinal=tracking.stage_ordinal + jnp.int32(1),
        stage_round_budget=round_budget,
        stage_rounds=jnp.asarray(0, jnp.int32),
        stage_counts=jnp.zeros_like(tracking.stage_counts),
        full_batch_rounds_valid=jnp.asarray(True),
    )


def _lane_shape(num_envs: int | None) -> tuple[int, ...]:
    """Return the scalar or native lane prefix for a fixed execution structure."""
    return () if num_envs is None else (num_envs,)


def _check_lane_array(
    value: Array, shape: tuple[int, ...], name: str, *, boolean: bool = False
) -> None:
    """Check lane shape and int32/bool dtype without reading device values.

    boolean selects bool fields; other fields require int32. Reject wrong static
    metadata before broadcasting or arithmetic can conceal malformed input.
    """
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    expected = jnp.dtype(jnp.bool_ if boolean else jnp.int32)
    if value.dtype != expected:
        raise TypeError(f"{name} must have dtype {expected.name}")


def _source_indices(
    values: object, shape: tuple[int, ...], size: int, consumed: Array
) -> tuple[Array, Array]:
    """Normalize integer source indices without accepting narrowing overflow.

    Scalars broadcast to shape. Only consumed lanes have range checks. Concrete
    invalid values raise; traced invalid values return an error mask and -1 safe
    placeholder. Boolean/float indices and other shapes always fail.
    """
    if not isinstance(values, Tracer):
        raw = np.asarray(values)
        if raw.dtype.kind not in "iu":
            raise TypeError("source_indices must contain integers, not bool or floats")
        if raw.shape not in ((), shape):
            raise ValueError(f"source_indices must be scalar or have shape {shape}")
        raw = np.broadcast_to(raw, shape)
        invalid = (raw >= size) | ((raw < -1) if raw.dtype.kind == "i" else False)
        if not isinstance(consumed, Tracer) and np.any(invalid & np.asarray(consumed)):
            raise ValueError(
                "consumed source_indices must be -1 or an existing source row"
            )
        narrowed = np.where(invalid, -1, raw).astype(np.int32)
        return jnp.asarray(narrowed), jnp.asarray(invalid)
    if jnp.issubdtype(values.dtype, jnp.bool_) or not jnp.issubdtype(
        values.dtype, jnp.integer
    ):
        raise TypeError("source_indices must contain integers, not bool or floats")
    if values.shape not in ((), shape):
        raise ValueError(f"source_indices must be scalar or have shape {shape}")
    values = jnp.broadcast_to(cast(Array, values), shape)
    invalid = values >= size
    if jnp.issubdtype(values.dtype, jnp.signedinteger):
        invalid |= values < -1
    return jnp.where(invalid, -1, values).astype(jnp.int32), invalid


def _source_classes(values: object, shape: tuple[int, ...], consumed: Array) -> Array:
    """Normalize ten-slot integer rows before narrowing, ignoring unused lanes.

    A single (10,) row broadcasts to shape+(10,). None means ten -1 values.
    Shape and integer dtype are always checked. Consumed concrete out-of-range
    values raise before narrowing. Traced bad rows keep an invalid -2 marker;
    the shared roster resolver checks compactness and retains a failure flag.
    """
    wanted = (*shape, 10)
    if values is None:
        return jnp.full(wanted, -1, jnp.int32)
    if not isinstance(values, Tracer):
        raw = np.asarray(values)
        if raw.dtype.kind not in "iu":
            raise TypeError(
                "source_class_ids must contain integers, not bool or floats"
            )
        if raw.shape not in ((10,), wanted):
            raise ValueError(f"source_class_ids must have shape (10,) or {wanted}")
        raw = np.broadcast_to(raw, wanted)
        invalid = (raw > 5) | ((raw < -1) if raw.dtype.kind == "i" else False)
        if not isinstance(consumed, Tracer) and np.any(
            invalid & np.asarray(consumed)[..., None]
        ):
            raise ValueError(
                "source_class_ids must contain classes 0..5 or ten -1 values"
            )
        values = jnp.asarray(np.where(invalid, -2, raw).astype(np.int32))
    else:
        if not jnp.issubdtype(values.dtype, jnp.integer):
            raise TypeError(
                "source_class_ids must contain integers, not bool or floats"
            )
        if values.shape not in ((10,), wanted):
            raise ValueError(f"source_class_ids must have shape (10,) or {wanted}")
        invalid = values > 5
        if jnp.issubdtype(values.dtype, jnp.signedinteger):
            invalid |= values < -1
        values = jnp.where(invalid, -2, cast(Array, values)).astype(jnp.int32)
    return jnp.where(consumed[..., None], jnp.broadcast_to(values, wanted), -1)


def _bind(
    bank: EnvConfig | None, config: EnvConfig, indices: Array, classes: Array
) -> tuple[Array, Array, Array]:
    """Check declared source rows against exact resolved configs numerically.

    Safe gather indices never create ownership: a negative declaration remains
    unknown. Rebuild only the declared profile through the shared catalog helper,
    then compare the entire configuration and spawn banks. Return known flags,
    default/swapped/unknown choices and roster validity. Invalid rows never bind.
    """
    if bank is None:
        _, valid = _source_config_with_class_ids(config, classes)
        return (
            jnp.zeros(indices.shape, bool),
            jnp.full(indices.shape, -1, jnp.int32),
            valid,
        )

    def select(value: Array) -> Array:
        """Gather declared rows with a safe placeholder for unknown sources."""
        return value[jnp.maximum(indices, 0)]

    selected = jax.tree.map(select, bank)
    selected, valid = _source_config_with_class_ids(selected, classes)
    matches, choice = spawn_locations_for_source(config, selected)
    known = (indices >= 0) & matches & valid
    return known, jnp.where(known, choice, -1).astype(jnp.int32), valid


def init_episode_tracking(
    env: Environment | AutoReset,
    state: EnvironmentState,
    *,
    source_configs: EnvConfig | None = None,
    source_indices: object = None,
    source_class_ids: object = None,
    record_starts: bool = False,
) -> EpisodeTrackingState:
    """Attach a tracker to exact current games without taking a simulator step.

    Parameters
    ----------
    env : Environment | AutoReset
        Configured execution handle. The wrapper's explicit base env is used.
    state : EnvironmentState
        Current matching state. Existing lifecycle failures are rejected.
    source_configs : EnvConfig | None, default=None
        Immutable scalar source or ordered bank. None uses the environment's
        known constructor source if available. Every requested resolved config
        is checked; an unused exchanged spawn choice need not be valid.
    source_indices : integer scalar or array, optional
        Source row per lane; -1 means unknown. A scalar broadcasts. Explicit banks
        with omitted indices remain unknown. Implicit constructor sources bind
        row zero only while the lane still has untouched constructor provenance.
    source_class_ids : integer array, optional
        (10,) or L+(10,) compact class rows, before spawn exchange. None or ten
        -1 values retain the original source profile. Explicit rows require a
        known source index and may change only its catalog-derived profile.
    record_starts : bool, default=False
        Emit compact starts on first real transitions. True requires every lane
        to be before its first transition and computes one ordered-bank digest.

    Returns
    -------
    EpisodeTrackingState
        Immutable tracker with one dynamic bank. Carry the returned tracker beside
        state; researchers choose sources while the library owns reset bindings.

    Raises
    ------
    TypeError, ValueError
        Invalid structure, dtype, source declaration, config, lifecycle state or
        recording attachment after a first transition.

    Notes
    -----
    Host setup only; do not call inside jit or scan. With recording disabled,
    attachment midway through a game counts only subsequent experience and keeps
    its original start stage unknown. Reinitializing cannot repair lost tracking.
    """
    base = env if isinstance(env, Environment) else getattr(env, "env", None)
    if not isinstance(base, Environment):
        raise TypeError("env must be an Environment or AutoReset")
    if type(record_starts) is not bool:
        raise TypeError("record_starts must be bool")
    shape = _lane_shape(base.num_envs)
    _check_lane_array(state.episode_id, shape, "state.episode_id")
    if any(isinstance(value, Tracer) for value in jax.tree.leaves(state)):
        raise TypeError("init_episode_tracking is host setup, outside jit or scan")
    if np.any(np.asarray(state.lifecycle_error)):
        raise ValueError("cannot initialize tracking with a lifecycle error")
    prepare_exact_env_config(state.config, num_envs=base.num_envs)
    explicit_bank = source_configs is not None
    source = source_configs if explicit_bank else base._source_config  # pyright: ignore[reportPrivateUsage]
    bank = None
    if source is not None:
        if type(source) is not EnvConfig:
            raise TypeError("source_configs must be an EnvConfig or None")
        active_shape = np.shape(source.agent_profile.active_mask)
        size = active_shape[0] if len(active_shape) == 2 else 1
        bank = prepare_exact_env_config(source, num_envs=size)
    size = 0 if bank is None else int(bank.agent_profile.active_mask.shape[0])
    if source_indices is None:
        indices = jnp.where(
            (state.config_origin_generation == -1)
            & (not explicit_bank)
            & (bank is not None),
            0,
            -1,
        ).astype(jnp.int32)
    else:
        indices, _ = _source_indices(source_indices, shape, size, jnp.ones(shape, bool))
    classes = _source_classes(source_class_ids, shape, jnp.ones(shape, bool))
    if np.any(np.asarray((indices < 0) & jnp.any(classes != -1, axis=-1))):
        raise ValueError("source_class_ids require a declared source index")
    known, choices, _ = _bind(bank, state.config, indices, classes)
    if np.any(np.asarray((indices >= 0) & ~known)):
        raise ValueError("source declaration does not match the resolved configuration")
    seen = ~state.episode_start
    if record_starts and np.any(np.asarray(seen)):
        raise ValueError(
            "recording must begin before the episode's first real transition"
        )
    table_id = None
    if record_starts:
        words = np.zeros(8, np.uint32)
        if bank is not None:
            from marl_battlegrounds.evaluation.recording_identity import (
                ordered_source_bank_identity,
            )

            digest, _, _ = ordered_source_bank_identity(bank)
            words = np.frombuffer(bytes.fromhex(digest), dtype=">u4").astype(np.uint32)
        table_id = jnp.asarray(words)
    zeros = jnp.zeros(shape, jnp.int32)
    minus_one = jnp.full(shape, -1, jnp.int32)
    return EpisodeTrackingState(
        base.num_envs,
        record_starts,
        bank,
        table_id,
        state.episode_id,
        state.reset_generation,
        state.config_origin_generation,
        indices,
        classes,
        choices,
        known,
        seen,
        minus_one,
        state.core_state.step_count - state.initial_step_count - 1,
        state.cumulative_transition_count,
        zeros,
        jnp.asarray(-1, jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.zeros((*shape, 3), jnp.int32),
        jnp.asarray(True),
    )


def track_episode_step(
    tracking: EpisodeTrackingState,
    before_state: EnvironmentState,
    step_result: StepResult,
    *,
    source_indices: object = None,
    source_class_ids: object = None,
) -> tuple[EpisodeTrackingState, StepResult]:
    """Account once for an exact step result and attach optional recording claims.

    Parameters
    ----------
    tracking : EpisodeTrackingState
        Latest tracker from this execution context, carried beside before_state.
    before_state : EnvironmentState
        Exact pre-step state. Reset-only changes since the previous update are
        allowed. Do not rewrite config, counters or epoch fields outside reset.
    step_result : tuple
        Exact five-part result (observations, state, reward, done, info). AutoReset
        may return fresh state; info still owns the old producing transition.
    source_indices : integer scalar or array, optional
        Declaration used only for lanes whose reset generation changed. A scalar
        broadcasts. Without a declaration, reuse preserves a matching origin and
        an override clears it. Continuing-lane values are ignored after shape and
        dtype checks. Invalid concrete consumed indices raise; traced failures
        remain sticky numerical errors.
    source_class_ids : integer array, optional
        (10,) or L+(10,) source roster rows used only on changed generations.
        Continuing lanes retain their rows. When both declarations are omitted,
        same-origin resets retain the previous roster; changed origins clear it.
        An explicit source index without classes selects the original source.
        An explicit roster cannot establish ownership without a source index.

    Returns
    -------
    tuple[EpisodeTrackingState, tuple]
        New tracker and the same five-part result with enriched info. Every call
        supplies sticky episode_tracking_error. Starts are None when disabled.
        No input, action, reward, policy memory or simulator state is changed.

    Raises
    ------
    TypeError, ValueError
        Invalid static structures, dtypes or concrete consumed source indices.
        Dynamic declaration/accounting failures set flags for host boundaries.

    Notes
    -----
    Pure numerical JAX work. Terminal transitions count; padding and resets do
    not. Existing failure evidence cannot be repaired by skipping or repeating a
    call. Relationship checks run only when a new episode needs binding. Source
    banks are dynamic, not copied per transition. Authored or ambiguous starts
    count as unreported experience and cannot earn spawn-balance credit.
    """
    if len(step_result) != 5:
        raise ValueError("step_result must be the environment's five-part result")
    observations, state, reward, done, info = step_result
    shape = _lane_shape(tracking.num_envs)
    for name, value in (
        ("before_state.episode_id", before_state.episode_id),
        ("state.episode_id", state.episode_id),
        ("info.episode_id", info.episode_id),
        ("info.decision_step", info.decision_step),
        ("info.episode_length", info.episode_length),
        (
            "before_state.cumulative_transition_count",
            before_state.cumulative_transition_count,
        ),
        ("state.cumulative_transition_count", state.cumulative_transition_count),
        ("tracking.accounted_transition_count", tracking.accounted_transition_count),
    ):
        _check_lane_array(value, shape, name)
    for name, value in (
        ("info.completed", info.completed),
        ("info.lifecycle_error", info.lifecycle_error),
    ):
        _check_lane_array(value, shape, name, boolean=True)
    changed = before_state.reset_generation != tracking.reset_generation
    indices = tracking.source_index
    choices = tracking.spawn_locations
    known = tracking.source_known
    classes = tracking.source_class_ids
    declaration_error = jnp.zeros(shape, bool)
    size = (
        0
        if tracking.source_configs is None
        else int(tracking.source_configs.agent_profile.active_mask.shape[0])
    )
    keep = before_state.config_origin_generation == tracking.config_origin_generation
    if source_indices is None:
        candidate_indices = jnp.where(keep, indices, -1)
    else:
        candidate_indices, invalid_indices = _source_indices(
            source_indices, shape, size, changed
        )
        declaration_error |= changed & invalid_indices
    if source_class_ids is None and source_indices is None:
        candidate_classes = jnp.where(keep[..., None], classes, -1)
    else:
        candidate_classes = _source_classes(source_class_ids, shape, changed)
    declaration_error |= (
        changed & (candidate_indices < 0) & jnp.any(candidate_classes != -1, axis=-1)
    )
    rebound_known, rebound_choices, roster_valid = cast(
        tuple[Array, Array, Array],
        jax.lax.cond(
            jnp.any(changed),
            lambda: _bind(
                tracking.source_configs,
                before_state.config,
                candidate_indices,
                candidate_classes,
            ),
            lambda: (known, choices, jnp.ones(shape, bool)),
        ),
    )
    if not isinstance(roster_valid, Tracer) and np.any(
        np.asarray(changed & ~roster_valid)
    ):
        raise ValueError(
            "source_class_ids require compact classes 0..5 or ten -1 values"
        )
    declaration_error |= changed & ~roster_valid
    declaration_error |= changed & (candidate_indices >= 0) & ~rebound_known
    indices = jnp.where(changed, candidate_indices, indices)
    choices = jnp.where(changed, rebound_choices, choices)
    known = jnp.where(changed, rebound_known, known)
    classes = jnp.where(changed[..., None], candidate_classes, classes)
    classes = jnp.where(known[..., None], classes, -1)
    seen = jnp.where(changed, False, tracking.first_transition_seen)
    last_decision = jnp.where(changed, -1, tracking.last_decision_step)
    start_stage = jnp.where(changed, -1, tracking.episode_start_stage)
    advanced = episode_advanced(info)
    count, exhausted = _checked_increment(
        before_state.cumulative_transition_count, advanced.astype(jnp.int32)
    )
    local_before = before_state.core_state.step_count - before_state.initial_step_count
    reset_after = state.reset_generation != before_state.reset_generation
    regular_epoch = (
        (state.reset_generation == before_state.reset_generation)
        & (state.episode_id == before_state.episode_id)
        & (
            state.core_state.step_count - state.initial_step_count
            == info.episode_length
        )
    )
    reset_epoch = (
        reset_after
        & (before_state.reset_generation < _MAX_COUNT)
        & (state.reset_generation - before_state.reset_generation == 1)
        & (info.completed | before_state.done.done)
        & state.episode_start
        & (state.core_state.step_count == state.initial_step_count)
    )
    accounting_error = (
        (
            before_state.cumulative_transition_count
            != tracking.accounted_transition_count
        )
        | (state.cumulative_transition_count != count)
        | (info.episode_id != before_state.episode_id)
        | (before_state.reset_generation < tracking.reset_generation)
        | (~changed & (before_state.episode_id != tracking.episode_id))
        | (
            ~changed
            & (
                before_state.config_origin_generation
                != tracking.config_origin_generation
            )
        )
        | (local_before != last_decision + 1)
        | (info.episode_length != local_before + advanced.astype(jnp.int32))
        | (advanced & (info.decision_step != local_before))
        | (~advanced & (info.decision_step != -1))
        | ~(regular_epoch | reset_epoch)
        | before_state.lifecycle_error
        | state.lifecycle_error
        | info.lifecycle_error
    )
    flags = (
        tracking.error_flags
        | jnp.where(declaration_error, TRACKING_ERROR_DECLARATION, 0)
        | jnp.where(accounting_error, TRACKING_ERROR_ACCOUNTING, 0)
        | jnp.where(exhausted, TRACKING_ERROR_OVERFLOW, 0)
    ).astype(jnp.int32)
    valid = flags == 0
    usable_advance = advanced & valid
    first = usable_advance & ~seen
    start_stage = jnp.where(first, tracking.stage_ordinal, start_stage)
    category = jnp.where(
        known & (choices >= 0) & ~before_state.authored_start, choices, 2
    )
    increment = jax.nn.one_hot(category, 3, dtype=jnp.int32) * usable_advance[..., None]
    counts, count_overflow = _checked_increment(tracking.stage_counts, increment)
    declared = tracking.stage_ordinal >= 0
    rounds, round_overflow = _checked_increment(
        tracking.stage_rounds, declared.astype(jnp.int32)
    )
    flags |= jnp.where(
        jnp.any(count_overflow, axis=-1) | round_overflow,
        TRACKING_ERROR_OVERFLOW,
        0,
    ).astype(jnp.int32)
    valid &= flags == 0
    first &= valid
    updated = replace(
        tracking,
        episode_id=jnp.where(valid, before_state.episode_id, tracking.episode_id),
        reset_generation=jnp.where(
            valid, before_state.reset_generation, tracking.reset_generation
        ),
        config_origin_generation=jnp.where(
            valid,
            before_state.config_origin_generation,
            tracking.config_origin_generation,
        ),
        source_index=jnp.where(valid, indices, tracking.source_index),
        source_class_ids=jnp.where(
            valid[..., None], classes, tracking.source_class_ids
        ),
        spawn_locations=jnp.where(valid, choices, tracking.spawn_locations),
        source_known=jnp.where(valid, known, tracking.source_known),
        first_transition_seen=jnp.where(
            valid, seen | advanced, tracking.first_transition_seen
        ),
        episode_start_stage=jnp.where(valid, start_stage, tracking.episode_start_stage),
        last_decision_step=jnp.where(
            valid & advanced,
            info.decision_step,
            jnp.where(valid, last_decision, tracking.last_decision_step),
        ),
        accounted_transition_count=jnp.where(
            valid, count, tracking.accounted_transition_count
        ),
        error_flags=flags,
        stage_counts=jnp.where(valid[..., None], counts, tracking.stage_counts),
        stage_rounds=rounds,
        full_batch_rounds_valid=tracking.full_batch_rounds_valid
        & (~declared | jnp.all(advanced & valid)),
    )
    starts = None
    if tracking.record_starts:
        assert tracking.source_table_id is not None
        record_known = first & known
        starts = EpisodeStartRecords(
            jnp.where(first, info.episode_id, -1),
            jnp.where(first, before_state.reset_generation, -1),
            jnp.where(record_known[..., None], tracking.source_table_id, jnp.uint32(0)),
            jnp.where(record_known, indices, -1),
            jnp.where(record_known, choices, -1),
            jnp.where(first, start_stage, -1),
            record_known,
            first & before_state.authored_start,
            first,
            jnp.where(record_known[..., None], classes, -1),
        )
    info = info._replace(episode_start_records=starts, episode_tracking_error=flags)
    return updated, (observations, state, reward, done, info)


def _boundary_snapshot(
    tracking: EpisodeTrackingState, state: EnvironmentState
) -> dict[str, np.ndarray]:
    """Transfer only small boundary fields and reject pending accounting errors.

    The returned host arrays are read-only inputs to summary calculation. Reset
    generations and configs are deliberately absent: reset-only changes are legal.
    """
    shape = _lane_shape(tracking.num_envs)
    _check_lane_array(state.cumulative_transition_count, shape, "state counts")
    values = jax.device_get(
        {
            "current": state.cumulative_transition_count,
            "accounted": tracking.accounted_transition_count,
            "lifecycle": state.lifecycle_error,
            "errors": tracking.error_flags,
            "counts": tracking.stage_counts,
            "ordinal": tracking.stage_ordinal,
            "budget": tracking.stage_round_budget,
            "rounds": tracking.stage_rounds,
            "full": tracking.full_batch_rounds_valid,
        }
    )
    snapshot = {name: np.asarray(value) for name, value in values.items()}
    mismatch = snapshot["current"] != snapshot["accounted"]
    failed = mismatch | snapshot["lifecycle"] | (snapshot["errors"] != 0)
    if np.any(failed):
        lanes = [tuple(int(i) for i in row) for row in np.argwhere(failed)]
        raise ValueError(
            f"episode accounting failed at lanes {lanes}: expected "
            f"{snapshot['accounted'][failed].tolist()}, observed "
            f"{snapshot['current'][failed].tolist()}, tracking errors "
            f"{snapshot['errors'][failed].tolist()}, lifecycle errors "
            f"{snapshot['lifecycle'][failed].tolist()}"
        )
    if np.any(snapshot["counts"] < 0) or int(snapshot["rounds"]) < 0:
        raise ValueError("stage counters must be nonnegative")
    return snapshot


def _summary(
    tracking: EpisodeTrackingState, snapshot: dict[str, np.ndarray]
) -> dict[str, int | str | None]:
    """Compute read-only totals with Python integers and fixed status precedence."""
    columns = snapshot["counts"].reshape(-1, 3)
    totals = [sum(int(value) for value in columns[:, index]) for index in range(3)]
    actual = sum(totals)
    declared = int(snapshot["ordinal"]) >= 0
    requested = (tracking.num_envs or 1) * int(snapshot["budget"]) if declared else None
    status, reason = "not_started", "stage_not_declared"
    if declared:
        assert requested is not None
        rounds = int(snapshot["rounds"])
        budget = int(snapshot["budget"])
        if bool(snapshot["full"]) and actual != rounds * (tracking.num_envs or 1):
            raise ValueError("stage rounds and real-transition totals disagree")
        status = "incomplete"
        if not bool(snapshot["full"]):
            reason = "invalid_full_batch_round"
        elif rounds > budget or actual > requested:
            reason = "budget_exceeded"
        elif rounds < budget or actual < requested:
            reason = "budget_unmet"
        elif totals[2]:
            reason = "unreported_spawn_steps"
        elif totals[0] != totals[1]:
            reason = "unequal_spawn_steps"
        else:
            status, reason = "complete", None
    return {
        "requested_env_steps": requested,
        "env_steps": actual,
        "default_spawn_steps": totals[0],
        "swapped_spawn_steps": totals[1],
        "unreported_spawn_steps": totals[2],
        "status": status,
        "reason": reason,
    }
