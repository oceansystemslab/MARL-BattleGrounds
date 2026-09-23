"""Run frozen methods with exact schedules, compiled chunks and lane refill.

The schedule owns exact configs, episode IDs and random-stream identities.
JAX policies run in batches; host methods use synchronous Python calls with
the same decision inputs. Optional metrics/replays go to memory or RunWriter.
Map defaults and saved-first conditions resolve before execution. This module
does not train methods or use training trackers, automatic resets or collectors.
"""

# Numerical execution and recording use shared private authorities.
# pyright: reportPrivateUsage=false

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import Any, NamedTuple, cast
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from marl_battlegrounds._tdm_assets import current_map_id
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.config import (
    validate_env_config,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.types import Action, DoneFlags, EnvConfig, EnvState, Reward
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    EpisodeInfo,
    InitialSnapshot,
    MetricMode,
    make,
)
from marl_battlegrounds.evaluation.collection_types import CollectedAssignments
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.evaluation_capture import evaluation_record_batch
from marl_battlegrounds.evaluation.evaluation_conditions import (
    OMITTED,
    Omitted,
    capture_ids,
    config_record,
    default_maps,
    option,
    prepare_schedule,
    prepare_source_choices,
    read_saved_pass,
    restore_config,
    roster_default,
    saved_specs,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    METRIC_SCHEMA_ID,
    METRIC_SCHEMA_VERSION,
    PRIORITY_METRIC_NAMES,
)
from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV3
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyApply,
    PolicyTrace,
    PolicyTree,
    System,
    SystemState,
    _apply_system_pair,
    _init_system_pair,
    _SystemExecution,
    apply_policy_batch,
    initial_policy_carry,
    select_policy_carry,
)
from marl_battlegrounds.evaluation.recording_context import (
    build_recording_context,
    capture_recording_provenance,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
    policy_description,
)
from marl_battlegrounds.evaluation.recording_identity import (
    tree_digest as _tree_digest,
)
from marl_battlegrounds.evaluation.recording_types import validate_recording_errors
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
from marl_battlegrounds.evaluation.replay_recording import ReplayCollector
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.evaluation.results import EpisodeResult, EvaluationResult
from marl_battlegrounds.evaluation.run_writer import (
    IDENTITY_COLUMNS,
    RunWriter,
    _json_bytes,  # pyright: ignore[reportPrivateUsage]
    _json_value,  # pyright: ignore[reportPrivateUsage]
    _prepare_pass_identity,
    configuration_identity,
)
from marl_battlegrounds.evaluation.system_evaluation import (
    freeze_evaluation_method,
    prepare_evaluation_system,
    replace_initialization_roots,
    validate_evaluation_rosters,
)
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    AgentClassName,
    TDMMapInfo,
    _roster_ids,
    _swap_spawn_banks,
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)

type Columns = dict[str, NDArray[np.generic]]
type MapInput = int | TDMMapInfo | EnvConfig
_ROSTER_A, _ROSTER_B = canonical_tournament_rosters()


@dataclass(frozen=True)
class EpisodeSpec:
    """One planned episode; RNG identity is independent of its unique row ID.

    Parameters
    ----------
    episode_id : int
        Positive int32-compatible ID, unique within the scheduled pass.
    env_config : EnvConfig
        Exact scalar EnvConfig. Evaluation does not move agents or
        exchange its banks implicitly. This public field replaces config;
        positional construction and historical serialized keys are unchanged.
    map_id : int | None
        Optional public map identity for recorded metadata. The config
        supplies actual conditions; a label does not change its values.
    seed_id : int | None
        Optional uint32 seed identity. If omitted, episode_id supplies
        it. A paired experiment may deliberately reuse seed_id in two games.
    initial_state : EnvState | None
        Optional exact Core EnvState for an authored start. It
        must be valid for env_config; the executor prepares its paired inputs.
    metadata : Mapping[str, object] | None
        Optional descriptive fields retained with the scheduled episode.

    source_config : EnvConfig | None
        Optional exact source for a declared complete spawn-bank relationship.
    spawn_locations : int | None
        Optional 0 for source banks or 1 for exchanged banks. This describes the
        supplied config; it never changes it. Requires matching source evidence.
    paired_comparison_key : str | None
        Optional nonempty key joining exactly two supplied conditions. Both need
        equal explicit seeds/maps. An authored comparison stays custom; a single
        authored episode without this key needs no counterpart.

    Construction only stores this immutable description. evaluate_episodes owns
    schedule/config/start validation before execution. Reordering specifications
    does not change their assigned random streams. Matching seeds do not promise
    matching sampled actions when observations or other conditions differ.
    """

    episode_id: int
    env_config: EnvConfig
    map_id: int | None = None
    seed_id: int | None = None
    initial_state: EnvState | None = None
    metadata: Mapping[str, object] | None = None
    source_config: EnvConfig | None = None
    spawn_locations: int | None = None
    paired_comparison_key: str | None = None

    @property
    def random_seed_id(self) -> int:
        """Return explicit seed_id, or episode_id when no seed was supplied."""
        return self.episode_id if self.seed_id is None else self.seed_id


class _Completed(NamedTuple):
    """Keep each lane's first terminal payload until the host handles this chunk."""

    completed: Array
    episode_id: Array
    outcome: Array
    length: Array
    scores: Array
    config: EnvConfig
    priority: MetricValues | None
    full: MetricValues | None
    decision_step: Array
    lifecycle_error: Array

    @property
    def info(self) -> EpisodeInfo:
        """Package retained old-episode completion data for metric persistence."""
        return EpisodeInfo(
            self.episode_id,
            self.completed,
            self.outcome,
            self.config,
            self.priority,
            self.full,
            None,
            self.decision_step,
            self.length,
            self.scores,
            self.lifecycle_error,
        )


class _Carry(NamedTuple):
    """JAX chunk state: compact inputs, game state, two memory trees and completions."""

    observations: Observations
    state: EnvironmentState
    policy_a: PolicyTree
    policy_b: PolicyTree
    completed: _Completed


def episode_keys(
    root: Array, seed_ids: Array, local_steps: Array, stream: int
) -> Array:
    """Derive one random key per planned episode and local transition.

    Parameters
    ----------
    root : Array
        One typed or legacy JAX root key.
    seed_ids : Array
        uint32-compatible vector (B,) of planned episode seed identities.
    local_steps : Array
        Integer vector (B,) of steps since each episode's start.
    stream : int
        Integer stream tag; callers separate reset, game and actor draws.

    Returns
    -------
    Array
        B typed keys, or legacy keys shaped (B, 2). Each key folds stream, then
        its seed identity, then its local step into root.

    Notes
    -----
        Pure numerical and jittable. Equal coordinates reproduce equal keys;
        changing scheduling order or chunk size does not change those coordinates.
        The caller validates values and chooses distinct stream tags.
    """
    roots = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        jax.random.fold_in(root, stream), seed_ids
    )
    return jax.vmap(jax.random.fold_in)(roots, local_steps)


def _actor_keys(root: Array, seed_ids: Array, local_steps: Array) -> Array:
    """Fold ten global actor slots into each lane's dedicated policy stream."""
    keys = episode_keys(root, seed_ids, local_steps, 2)
    return jax.vmap(jax.vmap(jax.random.fold_in, in_axes=(None, 0)), in_axes=(0, None))(
        keys, jnp.arange(10, dtype=jnp.uint32)
    )


def _empty_completed(state: EnvironmentState) -> _Completed:
    """Allocate empty per-lane completion buffers matching enabled metric structures."""
    count = state.episode_id.shape[0]

    def empty(size: int) -> MetricValues:
        """Create unavailable metric rows with one environment axis and fixed column
        count.
        """
        return MetricValues(
            jnp.zeros((count, size), jnp.float32), jnp.zeros((count, size), jnp.bool_)
        )

    return _Completed(
        jnp.zeros(count, jnp.bool_),
        state.episode_id,
        jnp.zeros(count, jnp.int32),
        jnp.zeros(count, jnp.int32),
        jnp.zeros((count, 2), jnp.int32),
        state.config,
        empty(len(PRIORITY_METRIC_NAMES)) if state.priority is not None else None,
        empty(len(FULL_METRIC_NAMES)) if state.full is not None else None,
        jnp.full(count, -1, jnp.int32),
        state.lifecycle_error,
    )


def _retain_completion(
    old: _Completed, state: EnvironmentState, info: EpisodeInfo
) -> _Completed:
    """Retain terminal info and every lane's sticky failure across chunk rounds.

    state is retained for the existing chunk-call contract. Episode summaries
    come only from info; unfinished or padded rounds cannot replace them.
    Lifecycle failures are combined on every round, including unfinished games.
    """
    del state
    current = _Completed(
        info.completed,
        info.episode_id,
        info.outcome,
        info.episode_length,
        info.team_scores,
        info.config,
        info.priority,
        info.full,
        info.decision_step,
        info.lifecycle_error,
    )
    retained = cast(_Completed, select_policy_carry(info.completed, current, old))
    return retained._replace(lifecycle_error=old.lifecycle_error | info.lifecycle_error)


@jax.jit
def _reset(
    env: Environment,
    keys: Array,
    config: EnvConfig,
    ids: Array,
    state: EnvironmentState | None = None,
    reset_mask: Array | None = None,
    initial: InitialSnapshot | None = None,
) -> tuple[Observations, EnvironmentState]:
    """Call the shared compiled reset boundary with scheduled IDs and exact configs."""
    return env.reset(
        keys,
        config,
        episode_id=ids,
        state=state,
        reset_mask=reset_mask,
        initial=initial,
    )


def _initial_snapshots(
    specs: Sequence[EpisodeSpec],
    keys: Array,
) -> InitialSnapshot | None:
    """Prepare authored starts only when this batch contains one.

    Build matching Core state/observation/masks for every lane, using ordinary
    reset for the remaining lanes. Inputs have already passed host validation.
    Return None for an entirely ordinary batch; do not retain extra start history.
    """
    if all(spec.initial_state is None for spec in specs):
        return None
    rows: list[InitialSnapshot] = []
    for index, spec in enumerate(specs):
        state, observation, mask, _ = (
            core.reset(spec.env_config, keys[index])
            if spec.initial_state is None
            else core.initialize_scenario_state(spec.initial_state, spec.env_config)
        )
        rows.append((state, observation, mask))

    def stack(*values: Array) -> Array:
        """Stack scalar authored/ordinary starts into the active native batch."""
        return jnp.stack(values)

    return cast(InitialSnapshot, jax.tree.map(stack, *rows))


@jax.jit(static_argnames=("apply_a", "apply_b", "chunk_size"))
def _jax_chunk(
    env: Environment,
    apply_a: PolicyApply,
    apply_b: PolicyApply,
    variables_a: PolicyTree,
    variables_b: PolicyTree,
    carry: _Carry,
    root_key: Array,
    seed_ids: Array,
    chunk_size: int,
) -> tuple[_Carry, ReplayPackets | None]:
    """Advance a fixed number of numerical rounds without host scheduling.

    Parameters and actor memory remain dynamic; policy callables and chunk length
    fix the program structure. Retain terminal payloads and freeze finished-lane
    memory until the host replaces lanes. Return final carry and optional selected
    replay packets, not a full observation history.
    """

    def advance(current: _Carry, unused: None) -> tuple[_Carry, ReplayPackets | None]:
        """Choose both teams' actions at one epoch, step once and retain first
        completions.
        """
        del unused
        local_steps = (
            current.state.core_state.step_count - current.state.initial_step_count
        )
        actor_keys = _actor_keys(root_key, seed_ids, local_steps)
        actions, next_a, next_b = apply_policy_batch(
            apply_a,
            apply_b,
            variables_a,
            variables_b,
            current.policy_a,
            current.policy_b,
            current.observations,
            current.state.action_mask,
            actor_keys,
            ~current.state.done.done,
        )
        observations, state, _, _, info = env.step(
            episode_keys(root_key, seed_ids, local_steps, 1), current.state, actions
        )
        return _Carry(
            observations,
            state,
            next_a,
            next_b,
            _retain_completion(current.completed, state, info),
        ), info.replay

    return jax.lax.scan(advance, carry, None, length=chunk_size)


@jax.jit
def _step_environment(
    env: Environment,
    key: Array,
    state: EnvironmentState,
    action: Action,
) -> tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]:
    """Reuse the wrapper's compiled step for batches driven by host policies."""
    return env.step(key, state, action)


def _host_chunk(
    env: Environment,
    team_a: Policy,
    team_b: Policy,
    variables_a: PolicyTree,
    variables_b: PolicyTree,
    carry: _Carry,
    root_key: Array,
    seed_ids: Array,
    chunk_size: int,
) -> tuple[_Carry, ReplayPackets | None]:
    """Advance synchronous host policies at the same pre-step decision boundary.

    Skip method calls for finished lanes, retain their memory and submit inert
    padding actions. Each round may transfer inputs; stack returned actions before
    one batched game step. Return final carry and optional per-round capture.
    """
    packets: list[ReplayPackets] = []
    for _ in range(chunk_size):
        local_steps = carry.state.core_state.step_count - carry.state.initial_step_count
        keys = _actor_keys(root_key, seed_ids, local_steps)
        action, next_a, next_b = apply_policy_batch(
            team_a.apply,
            team_b.apply,
            variables_a,
            variables_b,
            carry.policy_a,
            carry.policy_b,
            carry.observations,
            carry.state.action_mask,
            keys,
            ~carry.state.done.done,
            execution_a=team_a.execution,
            execution_b=team_b.execution,
        )
        observations, state, _, _, info = cast(
            tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo],
            _step_environment(
                env,
                episode_keys(root_key, seed_ids, local_steps, 1),
                carry.state,
                action,
            ),
        )
        carry = _Carry(
            observations,
            state,
            next_a,
            next_b,
            _retain_completion(carry.completed, state, info),
        )
        if info.replay is not None:
            packets.append(info.replay)

    def stack_packets(*values: Array) -> Array:
        """Stack selected replay packets over the fixed host chunk length."""
        return jnp.stack(values)

    return carry, (
        cast(ReplayPackets, jax.tree.map(stack_packets, *packets)) if packets else None
    )


class _SystemCarry(NamedTuple):
    """Keep numerical or opaque System memory beside the existing game carry."""

    observations: Observations
    state: EnvironmentState
    memory: SystemState
    completed: _Completed


def _selected_assignments(
    env: Environment,
    info: EpisodeInfo,
    trace: PolicyTrace,
) -> CollectedAssignments | None:
    """Gather compact replay-selected decisions without metric or config history.

    Return capacity-shaped rows for this round, or None when capture is disabled.
    Independent info IDs/epochs and lane indices remain beside supplied trace
    claims. trace.valid marks occupied rows; padding cannot become a decision.
    """
    capacity = min(env.num_envs or 1, env._execution.replay_capacity)
    if not capacity:
        return None
    selected = jnp.any(info.episode_id[:, None] == env._replay_ids[None, :], axis=1)
    selected &= info.decision_step >= 0
    indices = jnp.nonzero(selected, size=capacity, fill_value=0)[0]
    occupied = jnp.arange(capacity) < jnp.sum(selected)

    def take(value: Array) -> Array:
        """Select fixed-capacity rows from one trace field."""
        return value[indices]

    picked = cast(PolicyTrace, jax.tree.map(take, trace))
    picked = picked._replace(valid=picked.valid & occupied)
    return CollectedAssignments(
        picked,
        info.episode_id[indices],
        info.decision_step[indices],
        info.episode_length[indices],
        indices.astype(jnp.int32),
    )


def _system_advance(
    env: Environment,
    execution_a: _SystemExecution,
    execution_b: _SystemExecution,
    variables_a: PolicyTree,
    variables_b: PolicyTree,
    current: _SystemCarry,
    root_key: Array,
    seed_ids: Array,
    capture_routes: bool,
) -> tuple[_SystemCarry, tuple[ReplayPackets | None, CollectedAssignments | None]]:
    """Apply both methods once, then advance the same pre-step environment.

    Stable descriptors supply callables. Parameters, memories and scheduled roots
    remain dynamic. Learner outputs are discarded inside application, before any
    host transfer. Refilling is the host scheduler's separate responsibility.
    """
    local = current.state.core_state.step_count - current.state.initial_step_count
    actions, memory, _ = _apply_system_pair(
        execution_a,
        execution_b,
        variables_a,
        variables_b,
        current.memory,
        current.observations,
        current.state,
        episode_keys(root_key, seed_ids, local, 2),
        actor_keys=_actor_keys(root_key, seed_ids, local),
        keep_learning_outputs=False,
    )
    observations, state, _, _, info = cast(
        tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo],
        _step_environment(
            env, episode_keys(root_key, seed_ids, local, 1), current.state, actions
        ),
    )
    return _SystemCarry(
        observations,
        state,
        memory,
        _retain_completion(current.completed, state, info),
    ), (
        info.replay,
        _selected_assignments(env, info, memory.policy_trace)
        if capture_routes
        else None,
    )


@jax.jit(static_argnames=("execution_a", "execution_b", "chunk_size", "capture_routes"))
def _jax_system_chunk(
    env: Environment,
    execution_a: _SystemExecution,
    execution_b: _SystemExecution,
    variables_a: PolicyTree,
    variables_b: PolicyTree,
    carry: _SystemCarry,
    root_key: Array,
    seed_ids: Array,
    chunk_size: int,
    capture_routes: bool = False,
) -> tuple[_SystemCarry, tuple[ReplayPackets | None, CollectedAssignments | None]]:
    """Run a fixed numerical chunk with dynamic System values and compact capture."""

    def advance(
        current: _SystemCarry, unused: None
    ) -> tuple[_SystemCarry, tuple[ReplayPackets | None, CollectedAssignments | None]]:
        """Run one decision while retaining only requested recording outputs."""
        del unused
        return _system_advance(
            env,
            execution_a,
            execution_b,
            variables_a,
            variables_b,
            current,
            root_key,
            seed_ids,
            capture_routes,
        )

    return jax.lax.scan(advance, carry, None, length=chunk_size)


def _host_system_chunk(
    env: Environment,
    execution_a: _SystemExecution,
    execution_b: _SystemExecution,
    variables_a: PolicyTree,
    variables_b: PolicyTree,
    carry: _SystemCarry,
    root_key: Array,
    seed_ids: Array,
    chunk_size: int,
    capture_routes: bool = False,
) -> tuple[_SystemCarry, tuple[ReplayPackets | None, CollectedAssignments | None]]:
    """Keep a host method outside jit while its numerical opponent stays batched.

    A provider failure propagates before the environment step. Opaque memory is
    retained without array conversion. No provider rollback or retry is attempted.
    """
    captures = []
    for _ in range(chunk_size):
        carry, capture = _system_advance(
            env,
            execution_a,
            execution_b,
            variables_a,
            variables_b,
            carry,
            root_key,
            seed_ids,
            capture_routes,
        )
        captures.append(capture)
    return carry, jax.tree.map(lambda *values: jnp.stack(values), *captures)


def positive_int(value: object, name: str) -> int:
    """Validate a positive integer setting at a host boundary.

    Parameters
    ----------
    value : object
        Python or NumPy Integral value; booleans are rejected.
    name : str
        Setting name used in the error message.

    Returns
    -------
    int
        Equivalent Python int greater than zero.

    Raises
    ------
    ValueError
        value is boolean, non-integral or less than one.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _stack_configs(specs: Sequence[EpisodeSpec]) -> EnvConfig:
    """Prepare one active batch of validated scalar episode configurations.

    Parameters
    ----------
    specs : Sequence[EpisodeSpec]
        Nonempty active or refill batch. Each configuration has already passed
        setup checks and has matching field shapes. Repeated immutable source
        objects are transferred to the host once within this call.

    Returns
    -------
    EnvConfig
        One device array per field, with the episode axis first. JAX's dtype
        promotion rules apply before host stacking. Values and source objects
        remain unchanged. This host setup helper does not retain a numerical
        copy of the complete schedule or run inside a compiled simulation step.
    """
    host_by_identity: dict[int, EnvConfig] = {}
    configs = []
    for spec in specs:
        identity = id(spec.env_config)
        if identity not in host_by_identity:
            host_by_identity[identity] = jax.device_get(spec.env_config)
        configs.append(host_by_identity[identity])

    def stack(*values: object) -> Array:
        """Stack matching host leaves and upload this whole field once."""
        dtype = jnp.result_type(*values)
        return jnp.asarray(
            np.stack([np.asarray(value, dtype=dtype) for value in values])
        )

    return cast(EnvConfig, jax.tree.map(stack, *configs))


def normalize_episode_specs(
    maps: Iterable[MapInput] | None,
    num_episodes: int,
    team_a_roster: Sequence[AgentClassName],
    team_b_roster: Sequence[AgentClassName],
    score_threshold: int,
    max_steps: int,
) -> tuple[EpisodeSpec, ...]:
    """Resolve map choices into an exact cyclic episode schedule.

    Parameters
    ----------
    maps : Iterable[MapInput] | None
        Integer IDs, TDMMapInfo or scalar EnvConfig entries. None uses the
        canonical evaluation map IDs; an explicit sequence must not be empty.
    num_episodes : int
        Positive total count, already validated by the caller.
    team_a_roster : Sequence[AgentClassName]
        Ordered classes for map-built Team A configs.
    team_b_roster : Sequence[AgentClassName]
        Ordered classes for map-built Team B configs.
    score_threshold : int
        Score target for map-built configs.
    max_steps : int
        Horizon in ticks for map-built configs.

    Returns
    -------
    tuple[EpisodeSpec, ...]
        Tuple of num_episodes EpisodeSpec rows with IDs 1..N. Choices cycle in
        supplied order. Repeated IDs reuse one factory config; explicit configs
        are kept exactly and receive no inferred map label.

    Raises
    ------
    TypeError
        A choice is neither EnvConfig, TDMMapInfo nor an integer ID.
    ValueError
        Choices are empty or a map-built config is invalid.

    Notes
    -----
        Host setup only. Explicit configs ignore the roster/rule arguments and are
        validated later by evaluate_episodes. No bank exchange, phase-based map
        selection or random episode sampling occurs.
    """
    choices = tuple(CANONICAL_TDM_EVALUATION_MAP_IDS if maps is None else maps)
    if not choices:
        raise ValueError("maps must contain at least one map or configuration")
    resolved: list[tuple[int | None, EnvConfig]] = []
    cache: dict[int, EnvConfig] = {}
    for item in choices:
        if isinstance(item, EnvConfig):
            resolved.append((None, item))
            continue
        map_id = current_map_id(item) if isinstance(item, TDMMapInfo) else item
        if isinstance(map_id, bool) or not isinstance(map_id, Integral):
            raise TypeError(
                "maps must contain integer map IDs, TDMMapInfo or EnvConfig"
            )
        map_id = int(map_id)
        if map_id not in cache:
            cache[map_id] = make_standard_team_deathmatch_config(
                map_id=map_id,
                team_a_roster=team_a_roster,
                team_b_roster=team_b_roster,
                score_threshold=score_threshold,
                max_steps=max_steps,
            )
        resolved.append((map_id, cache[map_id]))
    return tuple(
        EpisodeSpec(index + 1, config, map_id)
        for index in range(num_episodes)
        for map_id, config in (resolved[index % len(resolved)],)
    )


@dataclass
class _MetricTable:
    """Preallocate selected host metric rows, using NaN for unavailable values."""

    names: tuple[str, ...]
    ids: tuple[int, ...]
    rows: dict[int, int]
    values: NDArray[np.float32]

    @classmethod
    def create(cls, names: tuple[str, ...], ids: Iterable[int]) -> _MetricTable:
        """Allocate rows in sorted ID order; an empty selection allocates no value
        rows.
        """
        ordered = tuple(sorted(ids))
        return cls(
            names,
            ordered,
            {value: index for index, value in enumerate(ordered)},
            np.full((len(ordered), len(names)), np.nan, dtype=np.float32),
        )

    def append(self, episode_id: int, values: MetricValues | None, lane: int) -> None:
        """Store one selected completion's valid values without retaining device
        history.
        """
        if episode_id not in self.rows:
            return
        if values is None:
            raise RuntimeError("requested episode metrics were not produced")
        self.values[self.rows[episode_id]] = np.where(
            values.valid[lane], values.values[lane], np.nan
        )

    def columns(
        self,
        specs: dict[int, EpisodeSpec],
        identity: dict[str, str | None],
        config_ids: dict[int, str],
    ) -> Columns:
        """Build aligned identity and value columns in the fixed output schema order."""
        if not self.ids:
            return {}
        columns: Columns = {
            **{
                name: np.asarray([value] * len(self.ids))
                for name, value in identity.items()
            },
            "episode_id": np.asarray(self.ids, dtype=np.int32),
            "seed_id": np.asarray(
                [specs[value].random_seed_id for value in self.ids], dtype=np.uint32
            ),
            "map_id": np.asarray([specs[value].map_id for value in self.ids]),
            "config_id": np.asarray(
                [config_ids[id(specs[value].env_config)] for value in self.ids]
            ),
        }
        configs = {
            id(specs[value].env_config): specs[value].env_config for value in self.ids
        }
        profiles = {
            identity: jax.device_get(config.agent_profile)
            for identity, config in configs.items()
        }
        classes = np.asarray(
            [profiles[id(specs[value].env_config)].class_ids for value in self.ids]
        )
        active = np.asarray(
            [profiles[id(specs[value].env_config)].active_mask for value in self.ids]
        )
        for slot in range(10):
            columns[f"agent_{slot}_class_id"] = classes[:, slot]
            columns[f"agent_{slot}_active"] = active[:, slot]
        columns.update(
            {name: self.values[:, index] for index, name in enumerate(self.names)}
        )
        return {name: columns[name] for name in (*IDENTITY_COLUMNS, *self.names)}


def _run_evaluation(
    team_a: System | Policy | str,
    team_b: System | Policy | str,
    episodes: Sequence[EpisodeSpec],
    *,
    seed: int = 0,
    num_envs: int = 128,
    keep_batch_size: bool = False,
    metrics: MetricMode = "priority",
    full_metrics_episodes: Iterable[int] = (),
    replay_episodes: Iterable[int] = (),
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    writer: RunWriter | None = None,
    phase: str = "evaluation",
    pass_id: str = "1",
    chunk_size: int = 16,
    run_id: str | None = None,
    contract: dict[str, object] | None = None,
    saved: tuple[dict[str, object], dict[str, object]] | None = None,
    source_choices: Sequence[EpisodeSpec] | None = None,
    registered_maps: Mapping[int, Mapping[str, Any]] | None = None,
    _verify_only: bool = False,
) -> EvaluationResult:
    """Execute already resolved frozen methods through the chunk/refill authority.

    Parameters
    ----------
    team_a : System | Policy | str
        Frozen Team A method or built-in name. Numerical variables and memory
        templates are snapshotted once. No learning update is performed.
    team_b : System | Policy | str
        Frozen Team B method with separate episode memory. Host sessions remain
        caller-owned; this function cannot freeze or undo their external effects.
    episodes : Sequence[EpisodeSpec]
        EpisodeSpec sequence in scheduling order. Each item supplies
        an exact scalar config, a unique positive int32 episode ID, an
        optional seed ID and an optional authored start. Banks are not
        exchanged. Configs must pass Core's host validator: rule scalars
        are Python int/float values, while array fields are JAX arrays.
    seed : int
        Root uint32 integer, default 0. Episode seed IDs and local step
        counts determine streams independently of scheduling order.
    num_envs : int
        Positive maximum batch size, default 128. The actual batch is
        capped by pending episodes unless keep_batch_size is True.
    keep_batch_size : bool, default=False
        Keep num_envs lanes even when fewer games remain, including on resume.
        Unused lanes are inactive before method initialization and produce no
        actions, transitions or records. Their inputs have valid=False. This
        preserves compiled batch shapes across short validation schedules.
    metrics : MetricMode
        "priority" by default; "full" adds all full metrics for every
        episode, and "none" skips default collection. Explicit full metric
        selections still collect their full and priority values.
    full_metrics_episodes : Iterable[int]
        Episode IDs selected for full metrics, default
        empty. Every selected ID must appear in episodes.
    replay_episodes : Iterable[int]
        Episode IDs selected for replay capture, default empty.
        Every selected ID must appear in episodes. Without a writer, replay
        objects are returned in memory; capture does not require a folder.
    output_dir : str | Path | None
        Optional parent folder for a new run. RunWriter creates a
        run folder below it. This call closes the writer it creates.
    resume_from : str | Path | None
        Optional existing run folder. Resume the same recorded
        pass and schedule, skipping its durably completed episode IDs.
        Partial episodes restart; policy memory is not a saved checkpoint.
        Batch size and chunk size may change. Supply this or output_dir,
        not both.
    writer : RunWriter | None
        Optional caller-owned open RunWriter. This call starts the named
        pass, writes and flushes its results, and records failures. It leaves
        the writer open. Do not also supply output_dir or resume_from.
    phase : str
        Nonempty saved label, default "evaluation". It does not choose
        maps, enforce map enrollment or change the supplied game conditions.
        With a writer, "tournament" selects match tables and requires pairing
        metadata; run_tournament supplies that metadata.
    pass_id : str
        Nonempty pass label, default "1". Together with phase it names
        the writer pass; resume checks that its saved details still match.
    chunk_size : int
        Positive number of steps per scheduling chunk, default 16.
        Finished lanes keep their terminal state until the chunk ends;
        pending episodes then replace those lanes and reset actor memory.
    run_id : str | None
        Optional in-memory run identity. The default is a fresh generated
        identity. With a writer, any supplied value must equal writer.run_id.
    contract : dict | None
        Private resolved evaluation contract, default None for the historical
        schedule route. Public resolvers supply versioned conditions and source
        evidence. This executor does not apply map defaults or expand schedules.
    saved : tuple[dict, dict] | None
        Private saved manifest/pass snapshot read before output mutation. None
        means no earlier snapshot was supplied. Scientific identity is checked
        again by the writer before recovery.
    source_choices : Sequence[EpisodeSpec] | None
        Complete ordered sources for a newly resolved generated schedule. Include
        unplayed choices. None preserves a resumed contract's saved references.
    registered_maps : mapping or None
        Private exact source map identities from an immutable tournament snapshot.
        None uses the current catalog. Supplied entries still require an approved
        current or historical identity and exact source geometry.
    _verify_only : bool, default=False
        Private read-only check of an existing saved pass. True returns empty
        result rows after the writer's shared identity/start checks, before any
        writer creation, method initialization or environment execution.

    Returns
    -------
    EvaluationResult
        EvaluationResult. Without a writer, priority_metrics and full_metrics
        map column names to one-dimensional NumPy arrays ordered by episode ID;
        metric values are float32 and unavailable values are NaN. Unselected
        tables are empty dicts. These columns can be passed to pandas.DataFrame.
        With a writer, metric tables and replays stay in its files, and their
        in-memory result fields are empty. paths names those files, or is None.
        episodes contains compact results from this call. completed_episode_ids
        includes both earlier durable completions and this call's completions.
        metadata records the actual schedule, policies, settings and run identity.

    Raises
    ------
    TypeError
        A config, authored start or numerical policy input has an
        unsupported type or dtype.
    ValueError
        A setting, schedule, config, selection or output combination
        is invalid, or saved run details do not match the resumed pass.
    RuntimeError
        The writer is closed/failed, or execution violates an
        internal completion or requested-metric contract.
    OSError
        Run files cannot be opened, locked, read or written.

    This is a Python scheduling boundary, not a function to wrap in jax.jit.
    JAX policies run in compiled batches with dynamic variables and actor memory.
    If either policy uses host execution, policy calls run in Python and add
    per-step transfers. Both routes copy completion data to the host per chunk.
    Mutable NumPy policy values are copied at entry; input descriptions are not
    edited. Policy/provider errors propagate, including provider timeouts.
    """
    team_a = freeze_evaluation_method(team_a)
    team_b = freeze_evaluation_method(team_b)
    legacy = isinstance(team_a, Policy) and isinstance(team_b, Policy)
    specs = tuple(episodes)
    if not specs:
        raise ValueError("episodes must contain at least one specification")
    if type(keep_batch_size) is not bool:
        raise ValueError("keep_batch_size must be a bool")
    requested_batch_size = positive_int(num_envs, "num_envs")
    batch_size = (
        requested_batch_size
        if keep_batch_size
        else min(requested_batch_size, len(specs))
    )
    chunk_size = positive_int(chunk_size, "chunk_size")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("seed must be a uint32 integer")
    spec_by_id = {spec.episode_id: spec for spec in specs}
    if len(spec_by_id) != len(specs):
        raise ValueError("episode IDs must be unique")
    validated: set[int] = set()
    validated_starts: set[tuple[int, int]] = set()
    for spec in specs:
        episode_id = positive_int(spec.episode_id, "episode_id")
        if episode_id > np.iinfo(np.int32).max:
            raise ValueError("episode IDs must fit int32")
        if (
            isinstance(spec.random_seed_id, bool)
            or not isinstance(spec.random_seed_id, Integral)
            or not 0 <= spec.random_seed_id <= 0xFFFFFFFF
        ):
            raise ValueError("episode seed IDs must be uint32 integers")
        if spec.initial_state is not None:
            start_identity = (id(spec.env_config), id(spec.initial_state))
            if start_identity not in validated_starts:
                validate_scenario_initial_state(spec.env_config, spec.initial_state)
                validated_starts.add(start_identity)
        if id(spec.env_config) not in validated:
            validate_env_config(spec.env_config)
            if int(spec.env_config.task_mode) != 1:
                raise ValueError("evaluate currently supports TDM configurations")
            validated.add(id(spec.env_config))
    if writer is not None and (output_dir is not None or resume_from is not None):
        raise ValueError("writer cannot be combined with output_dir or resume_from")
    if output_dir is not None and run_id is not None:
        raise ValueError(
            "run_id is for in-memory results or an existing writer; "
            "new runs allocate their own ID"
        )
    if not phase or not pass_id:
        raise ValueError("phase and pass_id must be nonempty")
    env = make(
        "tdm",
        num_envs=batch_size,
        metrics=metrics,
        full_metrics_episodes=full_metrics_episodes,
        replay_episodes=replay_episodes,
    )
    full_selection = env.full_metrics_episodes
    replay_selection = env.replay_episodes
    for name, selected in (
        ("full_metrics_episodes", full_selection),
        ("replay_episodes", replay_selection),
    ):
        if not set(selected) <= spec_by_id.keys():
            raise ValueError(f"{name} contains an episode outside the schedule")
    full_ids = set(spec_by_id) if metrics == "full" else set(full_selection)
    priority_ids = set(spec_by_id) if metrics != "none" else full_ids
    execution_a, variables_a, base_carry_a = prepare_evaluation_system(team_a)
    execution_b, variables_b, base_carry_b = prepare_evaluation_system(team_b)
    for cfg in {id(spec.env_config): spec.env_config for spec in specs}.values():
        validate_evaluation_rosters(execution_a, execution_b, cfg)
    recording = bool(
        writer is not None
        or output_dir is not None
        or resume_from is not None
        or replay_selection
    )
    metadata: dict[str, object] = {
        "metric_schema_id": METRIC_SCHEMA_ID,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "seed": int(seed),
        "rng_protocol": "episode-fold-in-v1",
        "phase": phase,
        "pass_id": pass_id,
        "num_episodes": len(specs),
        "episode_ids": sorted(spec_by_id),
        "num_envs": batch_size,
        "chunk_size": chunk_size,
        "metrics": metrics,
        "full_metrics_episodes": sorted(full_ids),
        "replay_episodes": list(replay_selection),
        "policies": [
            policy_description(
                team,
                team.variables,
                team.initial_carry,
                include_digests=recording or contract is not None,
            )
            if isinstance(team, Policy)
            else normalize_system_registration(team, phase=phase, frozen=True)[1]
            for team in (team_a, team_b)
        ],
    }
    if not legacy:
        metadata["rng_protocol"] = "evaluation-systems-v1"
    if recording:
        metadata.update(capture_recording_provenance(num_envs=batch_size))
    schedule: dict[int, dict[str, object]] = {}
    if contract is None:
        configs = {id(spec.env_config): spec.env_config for spec in specs}

        def normalize(value: object) -> Array:
            """Convert a resolved config leaf to the numerical form used by content
            hashing.
            """
            return jnp.asarray(value)

        configuration_records = {
            key: configuration_identity(jax.tree.map(normalize, config))
            for key, config in configs.items()
        }
        config_ids = {key: value[0] for key, value in configuration_records.items()}
        metadata["configurations"] = {
            key: value for key, value in configuration_records.values()
        }
        schedule: dict[int, dict[str, object]] = {}
        if recording:
            starts = {
                id(spec.initial_state): spec.initial_state
                for spec in specs
                if spec.initial_state is not None
            }
            start_digests = {key: _tree_digest(value) for key, value in starts.items()}
            schedule = {
                spec.episode_id: {
                    **({} if spec.metadata is None else spec.metadata),
                    "episode_id": spec.episode_id,
                    "seed_id": spec.random_seed_id,
                    "map_id": spec.map_id,
                    "configuration_digest": config_ids[id(spec.env_config)],
                    "initial_state_digest": (
                        start_digests[id(spec.initial_state)]
                        if spec.initial_state is not None
                        else None
                    ),
                    "expected_horizon": int(spec.env_config.max_steps)
                    - (
                        int(spec.initial_state.step_count)
                        if spec.initial_state is not None
                        else 0
                    ),
                }
                for spec in specs
            }
        if recording:
            metadata["schedule_digest"] = sha256(
                _json_bytes(_json_value(list(schedule.values())))
            ).hexdigest()
    else:
        schedule, configuration_contents, config_ids = prepare_schedule(
            specs,
            registered_maps=registered_maps,
            saved_declarations=None
            if saved is None
            else cast(Mapping[str, object], saved[1].get("episodes")),
        )
        saved_details = (
            {} if saved is None else cast(dict[str, object], saved[1]["details"])
        )
        old_contract = cast(
            dict[str, object], saved_details.get("evaluation_contract", {})
        )
        if source_choices is not None:
            choices = prepare_source_choices(
                source_choices, schedule, configuration_contents, config_ids
            )
            if old_contract and choices != old_contract.get("source_choices"):
                raise ValueError("explicit maps differ from the saved source choices")
            contract["source_choices"] = choices
        elif contract.get("source_choices"):
            old_contents = cast(
                dict[str, dict[str, object]], saved_details.get("configurations", {})
            )
            for choice in cast(list[dict[str, object]], contract["source_choices"]):
                identifier = cast(str, choice["source_config_id"])
                if identifier not in configuration_contents:
                    content = old_contents.get(identifier)
                    if (
                        content is None
                        or config_record(restore_config(content, validate=False))[0]
                        != identifier
                    ):
                        raise ValueError("saved source choice lacks matching content")
                    configuration_contents[identifier] = content
        metadata["configurations"] = configuration_contents
        metadata["schedule_digest"] = sha256(
            _json_bytes(_json_value(list(schedule.values())))
        ).hexdigest()
        metadata["evaluation_contract"] = contract
    # Raw in-memory access uses the same exact declarations without triggering IO.
    descriptions = cast(list[dict[str, object]], metadata["policies"])
    policies: dict[str, object] = dict(
        zip(("team_a", "team_b"), descriptions, strict=True)
    )
    pass_details = dict(metadata)
    metadata["schedule"] = schedule
    registrations = {
        name: normalize_system_registration(value, phase=phase)
        for name, value in policies.items()
    }
    metadata["systems"] = {key: value for key, value in registrations.values()}
    metadata["system_ids"] = {name: value[0] for name, value in registrations.items()}
    if _verify_only:
        if saved is None:
            raise ValueError("Read-only verification requires a saved pass")
        pass_key, _, _ = _prepare_pass_identity(
            saved[0], phase, pass_id, policies, None, pass_details
        )
        RunWriter._validate_saved_starts(
            saved[0], allow_pending=True, pass_key=pass_key
        )
        return EvaluationResult({}, {}, (), metadata, (), None)
    with ExitStack() as cleanup:
        if writer is None and (output_dir is not None or resume_from is not None):
            writer = cleanup.enter_context(
                RunWriter(
                    output_dir=output_dir,
                    resume_from=resume_from,
                    phase=phase,
                    pass_id=pass_id,
                    policies=policies,
                    details=pass_details,
                    _expected_run_id=run_id if resume_from is not None else None,
                )
            )
        elif writer is not None:
            if run_id is not None and run_id != writer.run_id:
                raise ValueError("run_id differs from the writer identity")
            writer.start_pass(
                phase=phase, pass_id=pass_id, policies=policies, details=pass_details
            )

            def record_failure(
                exception_type: object,
                error: BaseException | None,
                traceback: object,
                recorder: RunWriter = writer,
            ) -> None:
                """Record an escaping failure without taking ownership of the caller's
                writer.
                """
                if error is not None:
                    recorder.record_failure(error)

            cleanup.push(record_failure)
        if writer is not None:
            if run_id is not None and run_id != writer.run_id:
                raise ValueError("run_id differs from the writer identity")
            run_id = writer.run_id
        elif run_id is None:
            run_id = "in-memory-" + uuid4().hex
        metadata["run_id"] = run_id
        identity = {
            "run_id": run_id,
            "phase": phase,
            "pass_id": pass_id,
            "team_a_policy": team_a.name,
            "team_b_policy": team_b.name,
            "checkpoint_id": None,
        }
        previous: frozenset[int] = frozenset()
        if writer is not None:
            writer.register_episodes(schedule.values())
            previous = writer.completed_episode_ids
            if contract is not None:
                writer.mark_pass_result(
                    "incomplete", schedule_digest=str(metadata["schedule_digest"])
                )
        pending = tuple(spec for spec in specs if spec.episode_id not in previous)
        if not pending:
            metadata["completion_order"] = []
            if writer is not None and contract is not None:
                writer.mark_pass_result(
                    "complete", schedule_digest=str(metadata["schedule_digest"])
                )
            return EvaluationResult(
                {},
                {},
                (),
                metadata,
                tuple(sorted(previous)),
                writer.paths if writer is not None else None,
            )
        effective_batch_size = (
            batch_size if keep_batch_size else min(batch_size, len(pending))
        )
        if effective_batch_size != batch_size:
            metadata["num_envs"] = effective_batch_size
            runtime = cast(dict[str, object], metadata["runtime_provenance"])
            metadata["runtime_provenance"] = {
                **runtime,
                "environment_count": effective_batch_size,
                "batch_shape": [effective_batch_size],
            }
            pass_details.update(
                num_envs=effective_batch_size,
                runtime_provenance=metadata["runtime_provenance"],
            )
            if writer is not None:
                writer.start_pass(
                    phase=phase,
                    pass_id=pass_id,
                    policies=policies,
                    details=pass_details,
                )
        batch_size = effective_batch_size
        env = make(
            "tdm",
            num_envs=batch_size,
            metrics=metrics,
            full_metrics_episodes=full_selection,
            replay_episodes=replay_selection,
        )
        priority = _MetricTable.create(
            PRIORITY_METRIC_NAMES, priority_ids if writer is None else ()
        )
        full = _MetricTable.create(
            FULL_METRIC_NAMES, full_ids if writer is None else ()
        )
        replays: list[ReplayArtifactV3] = []
        collector = None
        if replay_selection and writer is None:

            def context(
                packet: ReplayPackets,
            ) -> tuple[EvaluationEpisodeContextV3, RuntimeProvenanceV1]:
                """Build known context for the first packet of one selected in-memory
                replay.
                """
                return build_recording_context(
                    packet.config,
                    run_id=run_id,
                    phase=phase,
                    pass_id=pass_id,
                    episode=schedule[int(packet.episode_id)],
                    policies=policies,
                    details=metadata,
                )

            collector = ReplayCollector(context)
            cleanup.callback(collector.close)
        initial_a = initial_policy_carry(base_carry_a, batch_size) if legacy else ()
        initial_b = initial_policy_carry(base_carry_b, batch_size) if legacy else ()
        root = jax.random.key(int(seed))
        lanes = list(pending[:batch_size])
        real_lane_count = len(lanes)
        # Padding IDs cannot alias any real game, including completed resume rows.
        padding_count = batch_size - real_lane_count
        if padding_count > np.iinfo(np.int32).max - len(spec_by_id):
            raise ValueError("not enough unused positive int32 IDs for padding")
        candidate = 1
        while len(lanes) < batch_size:
            if candidate not in spec_by_id:
                lanes.append(EpisodeSpec(candidate, lanes[0].env_config))
            candidate += 1

        def seeds() -> Array:
            """Read the current lane schedule's independent uint32 random-stream IDs."""
            return jnp.asarray(
                [spec.random_seed_id for spec in lanes], dtype=jnp.uint32
            )

        def ids() -> Array:
            """Read current lane episode identities as positive int32 values."""
            return jnp.asarray([spec.episode_id for spec in lanes], dtype=jnp.int32)

        reset_keys = episode_keys(root, seeds(), jnp.zeros(batch_size, jnp.int32), 0)
        observations, state = cast(
            tuple[Observations, EnvironmentState],
            _reset(
                env,
                reset_keys,
                _stack_configs(lanes),
                ids(),
                initial=_initial_snapshots(lanes, reset_keys),
            ),
        )
        if padding_count:
            padding = jnp.arange(batch_size) >= real_lane_count
            # Finished-lane handling already skips actions, transitions and records.
            # Set this before init so custom memory sees the correct valid mask.
            state = state._replace(
                done=state.done._replace(truncated=state.done.truncated | padding),
                collect_metrics=state.collect_metrics & ~padding,
                collect_full_metrics=state.collect_full_metrics & ~padding,
                collect_replay=state.collect_replay & ~padding,
            )
        carry: _Carry | _SystemCarry
        if legacy:
            carry = _Carry(
                observations, state, initial_a, initial_b, _empty_completed(state)
            )
        else:
            memory = _init_system_pair(
                execution_a,
                execution_b,
                variables_a,
                variables_b,
                (base_carry_a, base_carry_b),
                observations,
                state,
                episode_keys(root, seeds(), jnp.zeros(batch_size, jnp.int32), 3),
            )
            carry = _SystemCarry(observations, state, memory, _empty_completed(state))
        next_episode = batch_size
        results: dict[int, EpisodeResult] = {}
        while len(results) < len(pending):
            carry = carry._replace(
                completed=carry.completed._replace(
                    completed=jnp.zeros(batch_size, jnp.bool_),
                )
            )
            before_chunk = carry.state
            assignments = None
            try:
                if isinstance(carry, _SystemCarry):
                    runner = (
                        _jax_system_chunk
                        if team_a.execution == team_b.execution == "jax"
                        else _host_system_chunk
                    )
                    carry, (packets, assignments) = cast(
                        tuple[
                            _SystemCarry,
                            tuple[ReplayPackets | None, CollectedAssignments | None],
                        ],
                        runner(
                            env,
                            execution_a,
                            execution_b,
                            variables_a,
                            variables_b,
                            carry,
                            root,
                            seeds(),
                            chunk_size,
                            capture_routes=writer is not None,
                        ),
                    )
                elif team_a.execution == team_b.execution == "jax":
                    carry, packets = cast(
                        tuple[_Carry, ReplayPackets | None],
                        _jax_chunk(
                            env,
                            team_a.apply,
                            team_b.apply,
                            variables_a,
                            variables_b,
                            carry,
                            root,
                            seeds(),
                            chunk_size,
                        ),
                    )
                else:
                    carry, packets = _host_chunk(
                        env,
                        cast(Policy, team_a),
                        cast(Policy, team_b),
                        variables_a,
                        variables_b,
                        carry,
                        root,
                        seeds(),
                        chunk_size,
                    )
                if writer is not None and assignments is not None:
                    writer._write_collected(
                        evaluation_record_batch(
                            before_chunk,
                            carry.completed,
                            packets,
                            assignments,
                        )
                    )
                    summary_fields = (
                        "completed",
                        "episode_id",
                        "outcome",
                        "length",
                        "scores",
                        "decision_step",
                        "lifecycle_error",
                    )
                    summary_values = jax.device_get(
                        tuple(getattr(carry.completed, name) for name in summary_fields)
                    )
                    completed = carry.completed._replace(
                        **dict(zip(summary_fields, summary_values, strict=True)),
                        priority=None,
                        full=None,
                    )
                    packets = None
                else:
                    completed, packets = jax.device_get((carry.completed, packets))
                publication = completed.info._replace(replay=packets)
                if writer is not None:
                    if assignments is None:
                        writer.write(publication)
                else:
                    validate_recording_errors(publication)
                    if packets is not None and collector is not None:
                        replays.extend(collector.write(packets))
            except BaseException as error:
                error.add_note(
                    f"Evaluation policies {team_a.name!r}/{team_b.name!r}; "
                    f"active episode IDs {[spec.episode_id for spec in lanes]}"
                )
                raise
            replace = np.zeros(batch_size, dtype=bool)
            for lane in np.flatnonzero(completed.completed):
                episode_id = int(completed.episode_id[lane])
                if episode_id in results:
                    raise RuntimeError("duplicate completed episode emitted")
                spec = spec_by_id[episode_id]
                results[episode_id] = EpisodeResult(
                    episode_id,
                    spec.random_seed_id,
                    spec.map_id,
                    int(completed.outcome[lane]),
                    int(completed.length[lane]),
                    int(completed.scores[lane, 0]),
                    int(completed.scores[lane, 1]),
                    config_ids[id(spec.env_config)],
                )
                priority.append(episode_id, completed.priority, int(lane))
                full.append(episode_id, completed.full, int(lane))
                if next_episode < len(pending):
                    lanes[lane] = pending[next_episode]
                    next_episode += 1
                    replace[lane] = True
            if np.any(replace):
                mask = jnp.asarray(replace)
                reset_keys = episode_keys(
                    root, seeds(), jnp.zeros(batch_size, jnp.int32), 0
                )
                observations, state = cast(
                    tuple[Observations, EnvironmentState],
                    _reset(
                        env,
                        reset_keys,
                        _stack_configs(lanes),
                        ids(),
                        carry.state,
                        mask,
                        _initial_snapshots(lanes, reset_keys),
                    ),
                )
                if isinstance(carry, _SystemCarry):
                    memory = replace_initialization_roots(
                        carry.memory,
                        episode_keys(
                            root, seeds(), jnp.zeros(batch_size, jnp.int32), 3
                        ),
                        mask,
                    )
                    carry = _SystemCarry(observations, state, memory, carry.completed)
                else:
                    carry = _Carry(
                        observations,
                        state,
                        select_policy_carry(mask, initial_a, carry.policy_a),
                        select_policy_carry(mask, initial_b, carry.policy_b),
                        carry.completed,
                    )
        if writer is not None:
            writer.flush()
            if contract is not None:
                writer.mark_pass_result(
                    "complete", schedule_digest=str(metadata["schedule_digest"])
                )
        metadata["completion_order"] = list(results)
        return EvaluationResult(
            priority.columns(spec_by_id, identity, config_ids),
            full.columns(spec_by_id, identity, config_ids),
            tuple(results[value] for value in sorted(results)),
            metadata,
            tuple(sorted(previous | results.keys())),
            writer.paths if writer is not None else None,
            tuple(replays),
        )


def evaluate_episodes(
    system: System | Policy | str,
    opponent: System | Policy | str,
    episodes: Sequence[EpisodeSpec],
    *,
    seed: int | Omitted = OMITTED,
    num_envs: int = 128,
    keep_batch_size: bool = False,
    metrics: MetricMode | Omitted = OMITTED,
    full_metrics_episodes: Iterable[int] | Omitted = OMITTED,
    replay_episodes: Iterable[int] | Omitted = OMITTED,
    save_replays: int | Omitted = OMITTED,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    writer: RunWriter | None = None,
    phase: str = "evaluation",
    pass_id: str = "1",
    chunk_size: int = 16,
    run_id: str | None = None,
) -> EvaluationResult:
    """Run an exact schedule once, with fixed Team A/B method ownership.

    Parameters
    ----------
    system, opponent : System | Policy | str
        Team A and Team B methods, or built-in names. Numerical values are frozen
        once for this pass. Memories are fresh for each team and episode. Host
        sessions remain caller-owned; keep their external behavior fixed.
    episodes : sequence of EpisodeSpec
        Nonempty exact scalar configurations and unique positive int32 IDs.
        No automatic bank exchange, extra game or counterpart is added. A single
        authored episode is valid. Declared comparisons need two exact conditions.
    seed : int, default=0
        uint32 root for scheduled episode/decision keys. Omission on resume uses
        the saved value; explicitly passing zero checks that value instead.
    num_envs : int, default=128
        Positive maximum worker count, reduced to the number of pending games.
        It does not change schedule identities or random coordinates.
    keep_batch_size : bool, default=False
        Keep num_envs lanes on initial and resumed short schedules. Extra lanes
        have valid=False before System initialization; they produce no games,
        transitions or records. Host methods must respect that valid mask.
        This avoids compiling a new batch shape for a short validation pass.
    metrics : {'priority', 'full', 'none'}, default='priority'
        Default scalar collection. none still retains required game outcomes.
    full_metrics_episodes, replay_episodes : iterable of int, default=()
        Scheduled IDs for optional full measurements or replays. Selected full
        games retain priority values too, even under none. No selection is implicit.
    save_replays : int, default=0
        Select the first N scheduled IDs. Must not exceed the schedule length.
        Nonempty shorthand and explicit replay selections must identify the same
        set; conflicting selections fail instead of being combined.
    output_dir : str | Path | None, default=None
        Parent directory for a newly created run. No path means no files unless
        writer or resume_from is supplied. In-memory replays remain available.
    resume_from : str | Path | None, default=None
        Exact saved run directory. Omitted scientific settings inherit saved
        values; explicit settings assert equality. Supply exact authored arrays
        again when only their digest was saved. Completed games are not replayed.
    writer : RunWriter | None, default=None
        Optional caller-owned writer. Invalid conditions fail before pass changes.
        This call flushes its records and leaves the supplied writer open.
        Supply at most one of writer, output_dir and resume_from.
    phase, pass_id : str, default='evaluation' / '1'
        Nonempty identity of this pass. Exact schedules do not infer maps from
        phase. Resumption selects this saved identity, never the newest pass.
    chunk_size : int, default=16
        Positive decisions per compiled chunk. Finished lanes pad until refill.
        Worker count and chunk size may change on compatible resume.
    run_id : str | None, default=None
        Optional in-memory run identity; otherwise generated. A supplied value
        must equal the writer's identity when saving.

    Returns
    -------
    EvaluationResult
        Newly completed compact episodes and legacy metric/replay fields, plus
        uniform status, table and bounded iter_table access. Saved/resumed table
        views include all durable records in this pass. Disabled optional tables
        are empty; unavailable historical evidence raises with its reason.

    Raises
    ------
    ValueError, TypeError
        A method layout, configuration, selection, schedule, version or explicit
        resume assertion is invalid. Rejection precedes new output or recovery.
    RuntimeError
        Execution or a recording invariant fails. Provider errors propagate;
        no action is fabricated and external state is not rolled back.
    OSError
        Run files cannot be read, locked, written or synchronized.

    Notes
    -----
    This is a host scheduling call, not an outer jit/vmap/gradient interface.
    Numerical methods stay compiled and batched. Host methods receive permitted
    batched inputs; their JAX opponent remains batched. Each method chooses each
    valid action once. Learning outputs are discarded without host transfer.
    Validation uses fresh state/RNG/memory and does not edit training carry.
    """
    return _evaluate_tournament_episodes(
        system,
        opponent,
        episodes,
        seed=seed,
        num_envs=num_envs,
        keep_batch_size=keep_batch_size,
        metrics=metrics,
        full_metrics_episodes=full_metrics_episodes,
        replay_episodes=replay_episodes,
        save_replays=save_replays,
        output_dir=output_dir,
        resume_from=resume_from,
        writer=writer,
        phase=phase,
        pass_id=pass_id,
        chunk_size=chunk_size,
        run_id=run_id,
    )


def _evaluate_tournament_episodes(
    system: System | Policy | str,
    opponent: System | Policy | str,
    episodes: Sequence[EpisodeSpec],
    *,
    seed: int | Omitted = OMITTED,
    num_envs: int = 128,
    keep_batch_size: bool = False,
    metrics: MetricMode | Omitted = OMITTED,
    full_metrics_episodes: Iterable[int] | Omitted = OMITTED,
    replay_episodes: Iterable[int] | Omitted = OMITTED,
    save_replays: int | Omitted = OMITTED,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    writer: RunWriter | None = None,
    phase: str = "evaluation",
    pass_id: str = "1",
    chunk_size: int = 16,
    run_id: str | None = None,
    registered_maps: Mapping[int, Mapping[str, Any]] | None = None,
) -> EvaluationResult:
    """Share exact-schedule resolution with immutable tournament source maps.

    Arguments, defaults, outputs and errors match evaluate_episodes. The private
    registered_maps argument maps source map IDs to exact serialized TDMMapInfo
    identities; None uses current catalog metadata. Supplied identities undergo
    approved-history and source-geometry checks before writer mutation. No public
    setting or alternate executor is introduced.
    """
    saved = read_saved_pass(resume_from, writer, phase, pass_id)
    details = None if saved is None else saved[1]["details"]
    previous_contract = None if details is None else details.get("evaluation_contract")
    saved_options = (
        None
        if details is None
        else (previous_contract["options"] if previous_contract else details)
    )
    options = {
        "seed": option(seed, saved_options, "seed", 0),
        "metrics": option(metrics, saved_options, "metrics", "priority"),
        "full_metrics_episodes": option(
            tuple(full_metrics_episodes)
            if not isinstance(full_metrics_episodes, Omitted)
            else OMITTED,
            saved_options,
            "full_metrics_episodes",
            (),
        ),
        "replay_episodes": option(
            tuple(replay_episodes)
            if not isinstance(replay_episodes, Omitted)
            else OMITTED,
            saved_options,
            "replay_episodes",
            (),
        ),
        "save_replays": option(save_replays, saved_options, "save_replays", 0),
    }
    specs = tuple(episodes)
    selected = capture_ids(
        [s.episode_id for s in specs],
        options["replay_episodes"],
        options["save_replays"],
    )
    contract: dict[str, object] | None = (
        None
        if saved is not None and previous_contract is None
        else {
            "version": 1,
            "schedule_kind": "explicit",
            "options": options,
            "episode_ids": [s.episode_id for s in specs],
            "map_selection": "explicit",
            "spawn_mode": "explicit",
        }
    )
    return _run_evaluation(
        system,
        opponent,
        specs,
        seed=options["seed"],
        num_envs=num_envs,
        keep_batch_size=keep_batch_size,
        metrics=options["metrics"],
        full_metrics_episodes=options["full_metrics_episodes"],
        replay_episodes=selected,
        output_dir=output_dir,
        resume_from=resume_from,
        writer=writer,
        phase=phase,
        pass_id=pass_id,
        chunk_size=chunk_size,
        run_id=run_id,
        contract=contract,
        saved=saved,
        registered_maps=registered_maps,
    )


def evaluate(
    system: System | Policy | str,
    opponent: System | Policy | str,
    *,
    num_episodes: int,
    maps: Iterable[MapInput] | None = None,
    spawn_mode: str | Omitted = OMITTED,
    system_roster: Sequence[AgentClassName] | None | Omitted = OMITTED,
    opponent_roster: Sequence[AgentClassName] | None | Omitted = OMITTED,
    seed: int | Omitted = OMITTED,
    num_envs: int = 128,
    keep_batch_size: bool = False,
    metrics: MetricMode | Omitted = OMITTED,
    full_metrics_episodes: Iterable[int] | Omitted = OMITTED,
    replay_episodes: Iterable[int] | Omitted = OMITTED,
    save_replays: int | Omitted = OMITTED,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    writer: RunWriter | None = None,
    score_threshold: int | Omitted = OMITTED,
    max_steps: int | Omitted = OMITTED,
    phase: str = "evaluation",
    pass_id: str = "1",
    chunk_size: int = 16,
) -> EvaluationResult:
    """Evaluate a frozen researcher method against an opponent with ready defaults.

    Parameters
    ----------
    system, opponent : System | Policy | str
        Team A and Team B methods, kept on those teams throughout the pass.
        Accept shared/independent adapters, recurrent JAX methods, host methods
        and built-in names. Numerical values are snapshotted once; each game gets
        fresh memory. Opaque provider sessions cannot be numerically frozen.
    num_episodes : int
        Positive total game count, excluding bool, within int32 IDs. Paired mode
        needs an even count. This is not a per-map or per-opponent budget.
    maps : iterable of int | TDMMapInfo | EnvConfig | None, default=None
        Ordered map choices, including repeats or exact custom sources. Omission
        selects validation maps only for phase='validation' and test maps only
        for phase='evaluation'. Other new phases require explicit maps. Empty
        selections fail. Resume first uses saved map/config content, not today's
        catalog. Explicit maps must match the saved scientific conditions.
    spawn_mode : {'paired', 'default', 'swapped'}, default='paired'
        Paired games use one source, equal declared seeds and opposite complete
        spawn banks. Maps cycle by pair. Fixed modes cycle by game and permit odd
        budgets. Teams, slots, rosters, directions and action meanings never swap.
        Only requested configurations are validated. Game lengths can differ.
    system_roster, opponent_roster : sequence of class names | None, default=None
        Ordered active classes for map construction; None uses canonical 5v5.
        Exact EnvConfig sources retain their layouts. Explicit conflicting roster
        or nondefault rule overrides fail. Repeated classes are separate agents.
    seed : int, default=0
        uint32 root; scheduled seed IDs and local decisions determine each stream.
    num_envs : int, default=128
        Positive maximum execution batch, bounded by pending games. Placement
        and chunk size do not change schedule identity or episode random streams.
    keep_batch_size : bool, default=False
        Keep num_envs lanes when fewer games remain, including on resume. Extra
        lanes have valid=False before System initialization and produce no
        games, transitions or records. Host methods must respect that mask.
        Useful for reusing a compiled batch shape across validation checkpoints.
    metrics : {'priority', 'full', 'none'}, default='priority'
        Default measurement level. Required lengths, scores and outcomes remain
        available under none without implying that priority measurements exist.
    full_metrics_episodes, replay_episodes : iterable of int, default=()
        Optional scheduled IDs. Selected full records include priority values
        even under none. Replays are independent of metric mode and file output.
    save_replays : int, default=0
        First N scheduled games to capture. Nonempty explicit replay IDs and this
        shorthand must agree. Zero leaves only the explicit selection active.
    output_dir, resume_from : str | Path | None, default=None
        New-run parent or exact existing run, respectively. Use at most one.
        No output options create no files. Resume skips durable completed games.
    writer : RunWriter | None, default=None
        Caller-owned writer, mutually exclusive with both path options. Conditions
        are checked before pass changes. The writer is flushed but left open.
    score_threshold, max_steps : int, default=20 / 300
        Rules for map-built configurations. Explicit custom configurations retain
        exact values; conflicting nondefault overrides are errors.
    phase, pass_id : str, default='evaluation' / '1'
        Nonempty saved pass identity. Exact phase names govern new automatic map
        selection. Use distinct pass IDs for repeated validation checkpoints.
    chunk_size : int, default=16
        Positive steps per execution chunk. Finished lanes remain terminal until
        refill, and padding is excluded from real transition counts.

    Returns
    -------
    EvaluationResult
        Legacy fields and uniform scoped table access, completion status and replay
        paths. Saved/resumed views include prior durable games. table allocates
        the requested table; iter_table reads bounded row chunks.

    Raises
    ------
    ValueError, TypeError
        Conditions, methods, captures or saved assertions conflict. Scientific
        validation occurs before new files, writer recovery or pass changes.
    RuntimeError, OSError
        Execution, provider or recording fails. Earlier durable chunks remain
        valid; no fake actions, retry or provider rollback is performed.

    Notes
    -----
    Only omitted scientific arguments inherit saved values. An explicit default
    such as seed=0 or metrics='priority' is an equality assertion on resume.
    The signature uses a private omission marker to preserve this distinction.
    Numerical parameters stay dynamic under compiled chunk execution. Host methods
    cross their documented input boundary. This host helper never trains or edits
    the caller's training state, RNG or memory. Use evaluate_episodes for exact
    authored starts, including a single condition without any comparison claim.
    """
    request = _generated_evaluation_request(
        system=system,
        opponent=opponent,
        num_episodes=num_episodes,
        maps=maps,
        spawn_mode=spawn_mode,
        system_roster=system_roster,
        opponent_roster=opponent_roster,
        seed=seed,
        num_envs=num_envs,
        keep_batch_size=keep_batch_size,
        metrics=metrics,
        full_metrics_episodes=full_metrics_episodes,
        replay_episodes=replay_episodes,
        save_replays=save_replays,
        output_dir=output_dir,
        resume_from=resume_from,
        writer=writer,
        score_threshold=score_threshold,
        max_steps=max_steps,
        phase=phase,
        pass_id=pass_id,
        chunk_size=chunk_size,
    )
    return _run_evaluation(
        request.pop("team_a"),
        request.pop("team_b"),
        request.pop("episodes"),
        **request,
    )


def _generated_evaluation_request(
    system: System | Policy | str,
    opponent: System | Policy | str,
    *,
    num_episodes: int,
    maps: Iterable[MapInput] | None = None,
    spawn_mode: str | Omitted = OMITTED,
    system_roster: Sequence[AgentClassName] | None | Omitted = OMITTED,
    opponent_roster: Sequence[AgentClassName] | None | Omitted = OMITTED,
    seed: int | Omitted = OMITTED,
    num_envs: int = 128,
    keep_batch_size: bool = False,
    metrics: MetricMode | Omitted = OMITTED,
    full_metrics_episodes: Iterable[int] | Omitted = OMITTED,
    replay_episodes: Iterable[int] | Omitted = OMITTED,
    save_replays: int | Omitted = OMITTED,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    writer: RunWriter | None = None,
    score_threshold: int | Omitted = OMITTED,
    max_steps: int | Omitted = OMITTED,
    phase: str = "evaluation",
    pass_id: str = "1",
    chunk_size: int = 16,
) -> dict[str, Any]:
    """Resolve evaluate's arguments without starting games or changing files.

    Arguments, defaults and validation errors match evaluate. Return keyword
    arguments for the shared executor, including the exact saved pass when
    resuming. The same resolver serves normal execution and read-only checks.
    """
    count = positive_int(num_episodes, "num_episodes")
    if count > np.iinfo(np.int32).max:
        raise ValueError("num_episodes must fit positive int32 IDs")
    saved = read_saved_pass(resume_from, writer, phase, pass_id)
    details = None if saved is None else saved[1]["details"]
    old_contract = None if details is None else details.get("evaluation_contract")
    old = (
        None
        if details is None
        else (old_contract["options"] if old_contract else details)
    )
    supplied = {
        "spawn_mode": spawn_mode,
        "system_roster": system_roster,
        "opponent_roster": opponent_roster,
        "seed": seed,
        "metrics": metrics,
        "full_metrics_episodes": tuple(full_metrics_episodes)
        if not isinstance(full_metrics_episodes, Omitted)
        else OMITTED,
        "replay_episodes": tuple(replay_episodes)
        if not isinstance(replay_episodes, Omitted)
        else OMITTED,
        "save_replays": save_replays,
        "score_threshold": score_threshold,
        "max_steps": max_steps,
    }
    defaults = dict(
        spawn_mode="paired",
        system_roster=None,
        opponent_roster=None,
        seed=0,
        metrics="priority",
        full_metrics_episodes=(),
        replay_episodes=(),
        save_replays=0,
        score_threshold=20,
        max_steps=300,
    )
    options = {
        name: option(value, old, name, defaults[name])
        for name, value in supplied.items()
    }
    legacy = saved is not None and old_contract is None
    mode = (
        "default"
        if legacy and isinstance(spawn_mode, Omitted)
        else options["spawn_mode"]
    )
    if mode not in ("paired", "default", "swapped"):
        raise ValueError("spawn_mode must be paired, default or swapped")
    if mode == "paired" and count % 2:
        raise ValueError("paired spawn locations require an even total game count")
    if saved is not None and count != (details or {}).get("num_episodes"):
        raise ValueError("num_episodes differs from the saved schedule")
    sources = None
    if saved is not None and maps is None:
        specs = saved_specs(saved)
    else:
        choices = tuple(default_maps(phase) if maps is None else maps)
        if not choices:
            raise ValueError("maps must contain at least one map or configuration")
        sources = normalize_episode_specs(
            choices,
            len(choices),
            roster_default(options["system_roster"], 0),
            roster_default(options["opponent_roster"], 1),
            options["score_threshold"],
            options["max_steps"],
        )
        for source in sources:
            if source.map_id is None:
                _check_source_overrides(source.env_config, options)
        exchanged: dict[int, EnvConfig] = {}
        schedule_rows: list[EpisodeSpec] = []
        for index in range(count):
            source = sources[(index // 2 if mode == "paired" else index) % len(sources)]
            choice = index % 2 if mode == "paired" else int(mode == "swapped")
            resolved = source.env_config
            if choice:
                key = id(source.env_config)
                if key not in exchanged:
                    exchanged[key] = _swap_spawn_banks(source.env_config)
                resolved = exchanged[key]
            pads = np.asarray(source.env_config.team_spawn_pad_positions)
            ambiguous = np.array_equal(pads[0], pads[1])
            schedule_rows.append(
                EpisodeSpec(
                    index + 1,
                    resolved,
                    source.map_id,
                    index // 2 + 1 if mode == "paired" else index + 1,
                    source_config=None if legacy else source.env_config,
                    spawn_locations=None if legacy or ambiguous else choice,
                    paired_comparison_key=f"pair-{index // 2 + 1}"
                    if mode == "paired"
                    else None,
                )
            )
        specs = tuple(schedule_rows)
    selected = capture_ids(
        [s.episode_id for s in specs],
        options["replay_episodes"],
        options["save_replays"],
    )
    contract: dict[str, object] | None = (
        None
        if legacy
        else {
            "version": 1,
            "schedule_kind": "generated",
            "options": options,
            "episode_ids": [s.episode_id for s in specs],
            "spawn_mode": mode,
            "map_selection": old_contract["map_selection"]
            if old_contract
            else "automatic"
            if maps is None
            else "explicit",
            **(
                {"source_choices": old_contract["source_choices"]}
                if old_contract and "source_choices" in old_contract
                else {}
            ),
        }
    )
    return dict(
        team_a=system,
        team_b=opponent,
        episodes=specs,
        seed=options["seed"],
        num_envs=num_envs,
        keep_batch_size=keep_batch_size,
        metrics=options["metrics"],
        full_metrics_episodes=options["full_metrics_episodes"],
        replay_episodes=selected,
        output_dir=output_dir,
        resume_from=resume_from,
        writer=writer,
        phase=phase,
        pass_id=pass_id,
        chunk_size=chunk_size,
        contract=contract,
        saved=saved,
        source_choices=None if legacy else sources,
    )


def _verify_evaluation(  # pyright: ignore[reportUnusedFunction] - Shared private training preflight.
    system: System | Policy | str,
    opponent: System | Policy | str,
    **options: Any,  # noqa: ANN401 - Reuse evaluate's complete keyword contract.
) -> None:
    """Check an existing generated evaluation pass without recovering its files.

    system, opponent and options follow evaluate, including required num_episodes.
    Supply resume_from and the exact phase/pass ID, plus scientific settings to
    assert. Missing passes or changed methods, roots, conditions or output choices
    raise ValueError. This reads saved files and freezes numerical method values,
    but creates no writer, calls no initializer/action and changes no file.
    """
    request = _generated_evaluation_request(system, opponent, **options)
    if request["saved"] is None or request["resume_from"] is None:
        raise ValueError(
            "Read-only evaluation verification needs an existing saved pass"
        )
    _run_evaluation(**request, _verify_only=True)


def _check_source_overrides(config: EnvConfig, options: Mapping[str, object]) -> None:
    """Reject explicit roster/nondefault rule conflicts with an exact source."""
    for team, name in enumerate(("system_roster", "opponent_roster")):
        roster = options[name]
        if roster is not None:
            requested = list(
                _roster_ids(cast(Sequence[AgentClassName], roster), name=name)
            )
            actual = np.asarray(config.agent_profile.class_ids)[
                team * 5 : (team + 1) * 5
            ]
            if requested != list(actual):
                raise ValueError(
                    f"{name} conflicts with the exact source configuration"
                )
    for option_name, field, default in (
        ("score_threshold", "team_deathmatch_score_threshold", 20),
        ("max_steps", "max_steps", 300),
    ):
        value = options[option_name]
        if value != default and value != getattr(config, field):
            raise ValueError(
                f"{option_name} conflicts with the exact source configuration"
            )
