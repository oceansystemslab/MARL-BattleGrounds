"""Run fixed-policy evaluation with a host scheduler and compiled game chunks.

The schedule owns exact configs, episode IDs and random-stream identities.
JAX policies run in batches; host methods use synchronous Python calls with
the same decision inputs. Optional metrics/replays go to memory or RunWriter.
This module does not train policies or apply future phase-specific map rules.
"""

from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import NamedTuple, cast
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
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
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
    PolicyTree,
    apply_policy_batch,
    controller_identity,
    freeze_variables,
    initial_policy_carry,
    policy,
    select_policy_carry,
)
from marl_battlegrounds.evaluation.recording_context import (
    build_recording_context,
    capture_recording_provenance,
)
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets
from marl_battlegrounds.evaluation.replay_recording import ReplayCollector
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.evaluation.run_writer import (
    IDENTITY_COLUMNS,
    RunWriter,
    configuration_identity,
)
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    AgentClassName,
    TDMMapInfo,
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

    @property
    def random_seed_id(self) -> int:
        """Return explicit seed_id, or episode_id when no seed was supplied."""
        return self.episode_id if self.seed_id is None else self.seed_id


@dataclass(frozen=True)
class EpisodeResult:
    """Compact host result for one completed episode.

    Attributes
    ----------
    episode_id : int
        Positive schedule ID identifying the completed row.
    seed_id : int
        uint32-compatible random-stream identity assigned by the schedule.
    map_id : int | None
        Declared map ID, or None for an unlabeled explicit config.
    outcome : int
        Core terminal code: 1 Team A win, 2 Team B win, 3 draw.
    episode_length : int
        Number of real transitions since this episode's start.
    team_a_score : int
        Final Team A score, including any authored starting score.
    team_b_score : int
        Final Team B score, including any authored starting score.
    config_id : str
        Content digest linking to metadata's resolved configuration.

    The frozen record contains no trajectory, model weights or open file handle.
    """

    episode_id: int
    seed_id: int
    map_id: int | None
    outcome: int
    episode_length: int
    team_a_score: int
    team_b_score: int
    config_id: str


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation outputs and the identities needed to interpret them.

    Attributes
    ----------
    priority_metrics : Columns
        Selected priority table as NumPy columns, or {} when
        unselected or retained by a writer.
    full_metrics : Columns
        Selected full table with the same ownership rule.
    episodes : tuple[EpisodeResult, ...]
        Compact results newly completed by this call, in episode-ID order.
    metadata : dict[str, object]
        Run, policy, config, schedule and runtime descriptions.
    completed_episode_ids : tuple[int, ...]
        Sorted union of prior durable and new completions;
        default () when constructing a result directly.
    paths : dict[str, Path] | None
        Produced run-file paths when a writer was used; otherwise None.
    replays : tuple[ReplayArtifactV3, ...]
        Completed in-memory replay artifacts; default (), and empty when
        a writer owns the replay files.

    Columns are one-dimensional arrays. Valid scalar metrics are float32;
    unavailable values are NaN. Identity columns keep their own string/integer
    types. The descriptor is frozen but its contained dicts/arrays are not deeply
    immutable. Results never own an open RunWriter.
    """

    priority_metrics: Columns
    full_metrics: Columns
    episodes: tuple[EpisodeResult, ...]
    metadata: dict[str, object]
    completed_episode_ids: tuple[int, ...] = ()
    paths: dict[str, Path] | None = None
    replays: tuple[ReplayArtifactV3, ...] = ()


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
    )


def _retain_completion(
    old: _Completed, state: EnvironmentState, info: EpisodeInfo
) -> _Completed:
    """Replace only newly completed lanes, preserving prior terminal data in the
    chunk.
    """
    current = _Completed(
        info.completed,
        info.episode_id,
        info.outcome,
        state.core_state.step_count - state.initial_step_count,
        state.core_state.team_deathmatch_scores,
        info.config,
        info.priority,
        info.full,
    )
    return cast(_Completed, select_policy_carry(info.completed, current, old))


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


def _tree_digest(tree: PolicyTree) -> str:
    """Hash frozen tree structure, leaf shapes/dtypes and host bytes once for
    recording.
    """
    leaves = jax.tree.leaves(tree)
    structure = cast(object, jax.tree.structure(tree))
    digest = sha256(str(structure).encode())
    for value in jax.device_get(leaves):
        array = np.asarray(value)
        header = f"{array.dtype.str}:{array.shape}".encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(np.ascontiguousarray(array).data)
    return digest.hexdigest()


def _callable_name(apply: PolicyApply) -> str:
    """Describe a callable by module/name, using its type when those fields are
    absent.
    """
    owner = type(apply)
    module = getattr(apply, "__module__", owner.__module__)
    name = getattr(apply, "__qualname__", owner.__qualname__)
    return f"{module}.{name}"


def policy_description(
    team: Policy,
    variables: PolicyTree,
    initial_carry: PolicyTree,
    *,
    include_digests: bool,
) -> dict[str, object]:
    """Describe the fixed policy values used by an evaluation pass.

    Parameters
    ----------
    team : Policy
        Policy descriptor supplying label, callable, execution and checkpoint.
    variables : PolicyTree
        Numerical variable tree already snapshotted for this pass.
    initial_carry : PolicyTree
        Numerical actor-memory template already snapshotted.
    include_digests : bool
        Whether recording needs content/controller hashes.

    Returns
    -------
    dict[str, object]
        Fresh metadata dict. The callable name and declared checkpoint are kept;
        variables_frozen is True. Digest/controller fields are None when disabled.

    Notes
    -----
        Host-only. Call after freezing values; this function does not freeze them
        itself or verify a caller's checkpoint label. Enabled hashes read numerical
        leaves to the host. Disabled hashes skip that content work.
    """
    return {
        "name": team.name,
        "checkpoint": team.checkpoint,
        "execution": team.execution,
        "variables_frozen": True,
        "callable_name": _callable_name(team.apply),
        "controller_identity": controller_identity(team) if include_digests else None,
        "variables_digest": _tree_digest(variables) if include_digests else None,
        "initial_carry_digest": _tree_digest(initial_carry)
        if include_digests
        else None,
    }


def _stack_configs(specs: Sequence[EpisodeSpec]) -> EnvConfig:
    """Stack validated scalar episode configs into a native numerical batch."""

    def stack(*values: object) -> Array:
        """Convert matching scalar-config leaves to arrays and stack their episode
        axis.
        """
        return jnp.stack(tuple(jnp.asarray(value) for value in values))

    return cast(EnvConfig, jax.tree.map(stack, *(spec.env_config for spec in specs)))


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


def evaluate_episodes(
    team_a: Policy,
    team_b: Policy,
    episodes: Sequence[EpisodeSpec],
    *,
    seed: int = 0,
    num_envs: int = 128,
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
) -> EvaluationResult:
    """Run two fixed policies over an explicit, nonempty TDM episode schedule.

    Parameters
    ----------
    team_a : Policy
        Team A Policy, including its callable, variables, initial actor
        memory and execution mode. This call does not train the policy.
    team_b : Policy
        Team B Policy with the same contract. Use policy(name) to adapt
        a built-in controller; this function requires Policy objects.
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
        capped by the number of pending episodes. One lane runs one game.
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
    specs = tuple(episodes)
    if not specs:
        raise ValueError("episodes must contain at least one specification")
    batch_size = min(positive_int(num_envs, "num_envs"), len(specs))
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
    variables_a, variables_b = (
        freeze_variables(team_a.variables),
        freeze_variables(team_b.variables),
    )
    base_carry_a = freeze_variables(team_a.initial_carry)
    base_carry_b = freeze_variables(team_b.initial_carry)
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
                team, variables, initial_carry, include_digests=recording
            )
            for team, variables, initial_carry in (
                (team_a, variables_a, base_carry_a),
                (team_b, variables_b, base_carry_b),
            )
        ],
    }
    if recording:
        metadata.update(capture_recording_provenance(num_envs=batch_size))
    policies: dict[str, object] = {"team_a": team_a.name, "team_b": team_b.name}
    pass_details = dict(metadata)
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
                )
            )
        elif writer is not None:

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
            writer.start_pass(
                phase=phase, pass_id=pass_id, policies=policies, details=pass_details
            )
        if writer is not None:
            if run_id is not None and run_id != writer.run_id:
                raise ValueError("run_id differs from the writer identity")
            run_id = writer.run_id
        elif run_id is None:
            run_id = "in-memory-" + uuid4().hex
        metadata["run_id"] = run_id
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
        identity = {
            "run_id": run_id,
            "phase": phase,
            "pass_id": pass_id,
            "team_a_policy": team_a.name,
            "team_b_policy": team_b.name,
            "checkpoint_id": None,
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
        previous: frozenset[int] = frozenset()
        if writer is not None:
            writer.register_episodes(schedule.values())
            previous = writer.completed_episode_ids
        pending = tuple(spec for spec in specs if spec.episode_id not in previous)
        if not pending:
            return EvaluationResult(
                {},
                {},
                (),
                metadata,
                tuple(sorted(previous)),
                writer.paths if writer is not None else None,
            )
        effective_batch_size = min(batch_size, len(pending))
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
        initial_a = initial_policy_carry(base_carry_a, batch_size)
        initial_b = initial_policy_carry(base_carry_b, batch_size)
        root = jax.random.key(int(seed))
        lanes = list(pending[:batch_size])

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
        carry = _Carry(
            observations, state, initial_a, initial_b, _empty_completed(state)
        )
        next_episode = batch_size
        results: dict[int, EpisodeResult] = {}
        while len(results) < len(pending):
            carry = carry._replace(
                completed=carry.completed._replace(
                    completed=jnp.zeros(batch_size, jnp.bool_),
                )
            )
            try:
                if team_a.execution == team_b.execution == "jax":
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
                        team_a,
                        team_b,
                        variables_a,
                        variables_b,
                        carry,
                        root,
                        seeds(),
                        chunk_size,
                    )
                completed, packets = jax.device_get((carry.completed, packets))
                if packets is not None:
                    if writer is not None:
                        writer.write_replay(packets)
                    elif collector is not None:
                        replays.extend(collector.write(packets))
                if writer is not None:
                    writer.write(completed.info)
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
                carry = _Carry(
                    observations,
                    state,
                    select_policy_carry(mask, initial_a, carry.policy_a),
                    select_policy_carry(mask, initial_b, carry.policy_b),
                    carry.completed,
                )
        if writer is not None:
            writer.flush()
        return EvaluationResult(
            priority.columns(spec_by_id, identity, config_ids),
            full.columns(spec_by_id, identity, config_ids),
            tuple(results[value] for value in sorted(results)),
            metadata,
            tuple(sorted(previous | results.keys())),
            writer.paths if writer is not None else None,
            tuple(replays),
        )


def evaluate(
    team_a: Policy | str,
    team_b: Policy | str,
    *,
    num_episodes: int,
    maps: Iterable[MapInput] | None = None,
    seed: int = 0,
    num_envs: int = 128,
    metrics: MetricMode = "priority",
    full_metrics_episodes: Iterable[int] = (),
    replay_episodes: Iterable[int] = (),
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    writer: RunWriter | None = None,
    team_a_roster: Sequence[AgentClassName] = _ROSTER_A,
    team_b_roster: Sequence[AgentClassName] = _ROSTER_B,
    score_threshold: int = 20,
    max_steps: int = 300,
    phase: str = "evaluation",
    pass_id: str = "1",
    chunk_size: int = 16,
) -> EvaluationResult:
    """Run two fixed policies with one total episode budget across chosen maps.

    Parameters
    ----------
    team_a : Policy | str
        Team A Policy, or built-in name "random", "tdm-alpha" or
        "tdm-beta". Variables stay fixed throughout this evaluation.
    team_b : Policy | str
        Team B Policy or built-in name, with the same contract.
    num_episodes : int
        Required positive total episode count across all maps.
        Episodes receive IDs 1 through num_episodes in scheduling order.
    maps : Iterable[MapInput] | None
        Optional nonempty iterable of integer map IDs, TDMMapInfo objects
        or exact scalar TDM EnvConfig objects. Entries cycle in supplied
        order, including repeats. None selects canonical evaluation maps
        47, 48, 49, 50 and 51. Explicit configs keep their own rules, roster
        and banks; use host configs accepted by Core's validator.
    seed : int
        Root uint32 integer, default 0. Episode IDs supply seed identities;
        streams also use local step counts, not batch placement.
    num_envs : int
        Positive maximum simultaneous games, default 128. The actual
        batch is capped by the number of pending episodes.
    metrics : MetricMode
        "priority" by default. "full" collects priority and full values
        for every episode; "none" skips default collection. Explicit full
        metric selections still collect both kinds of values.
    full_metrics_episodes : Iterable[int]
        IDs in 1..num_episodes selected for full metrics,
        default empty. No replay or file output is required.
    replay_episodes : Iterable[int]
        IDs in 1..num_episodes selected for replay capture,
        default empty. Replays are returned in memory without a writer.
    output_dir : str | Path | None
        Optional parent folder for a new run. This call creates and
        closes a RunWriter, which creates a run folder below this parent.
    resume_from : str | Path | None
        Optional existing run folder for the same pass and settings.
        Skip durable completions and restart unfinished episodes. Supply
        this or output_dir, not both; this is not a policy-memory checkpoint.
        Batch size and chunk size may change on resume.
    writer : RunWriter | None
        Optional caller-owned open RunWriter. This call starts the pass,
        writes and flushes results, and leaves it open. Do not combine it
        with output_dir or resume_from.
    team_a_roster : Sequence[AgentClassName]
        Ordered classes used when building configs from map IDs
        or TDMMapInfo. Defaults to the canonical tournament roster: mage,
        warrior, hunter, rogue, priest. Explicit EnvConfig entries ignore it.
    team_b_roster : Sequence[AgentClassName]
        Same rule and default class order for Team B.
    score_threshold : int
        Positive score target for constructed map configs,
        default 20. Explicit EnvConfig entries keep their own target.
    max_steps : int
        Positive episode horizon for constructed map configs,
        default 300. Explicit EnvConfig entries keep their own horizon.
    phase : str
        Nonempty saved label, default "evaluation". Changing it does not
        select maps or apply phase-specific map enrollment rules. With a writer,
        "tournament" selects match tables and requires pairing metadata;
        run_tournament supplies that metadata.
    pass_id : str
        Nonempty saved pass label, default "1". A resumed pass must
        match its recorded policies, schedule, seed and output selections.
    chunk_size : int
        Positive steps per scheduling chunk, default 16. Completed
        lanes are replaced by pending episodes at chunk boundaries.

    Returns
    -------
    EvaluationResult
        EvaluationResult from evaluate_episodes. Without a writer, selected
        metric tables are dicts of one-dimensional NumPy columns ordered by
        episode ID, ready for pandas.DataFrame. Metric values are float32; NaN
        means unavailable. Unselected tables are empty. Selected replay objects
        are also returned in memory. With a writer, tables/replays stay in its
        files and those in-memory fields are empty; paths names the files.
        episodes contains results completed by this call; completed_episode_ids
        also includes earlier durable completions. metadata records the run.

    Raises
    ------
    TypeError
        A map entry, config, roster or policy input has an unsupported
        type or dtype.
    ValueError
        A controller name, setting, map, config, selection or output
        combination is invalid, or resumed pass details do not match.
    RuntimeError
        The writer is closed/failed or execution breaks a required
        completion or metric contract.
    OSError
        Run files cannot be opened, locked, read or written.

    This host function prepares the schedule and calls evaluate_episodes; do not
    wrap it in jax.jit. JAX policies use compiled batches with dynamic variables
    and actor memory. Host policies use Python calls and per-step transfers.
    Both routes copy completion data to the host per chunk. Input configs and
    policy descriptions are not edited; mutable NumPy policy values are copied
    at entry. Errors from policy callables propagate. No bank exchange, training,
    checkpoint selection or tournament ranking is performed here.
    """
    count = positive_int(num_episodes, "num_episodes")
    first = policy(team_a) if isinstance(team_a, str) else team_a
    second = policy(team_b) if isinstance(team_b, str) else team_b
    return evaluate_episodes(
        first,
        second,
        normalize_episode_specs(
            maps, count, team_a_roster, team_b_roster, score_threshold, max_steps
        ),
        seed=seed,
        num_envs=num_envs,
        metrics=metrics,
        full_metrics_episodes=full_metrics_episodes,
        replay_episodes=replay_episodes,
        output_dir=output_dir,
        resume_from=resume_from,
        writer=writer,
        phase=phase,
        pass_id=pass_id,
        chunk_size=chunk_size,
    )
