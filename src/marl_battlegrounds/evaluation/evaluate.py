"""Eager scheduling over bounded compiled evaluation chunks, without a trainer."""

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
from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV2
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyApply,
    PolicyTree,
    apply_policies,
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
from marl_battlegrounds.evaluation.replay_v2 import ReplayArtifactV2
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
    make_standard_team_deathmatch_config,
)

type Columns = dict[str, NDArray[np.generic]]
type MapInput = int | TDMMapInfo | EnvConfig
_ROSTER: tuple[AgentClassName, ...] = ("mage", "warrior", "hunter", "rogue", "priest")


@dataclass(frozen=True)
class EpisodeSpec:
    """One planned episode; RNG identity is independent of its unique row ID.

    A tournament may deliberately reuse ``seed_id`` across paired side swaps.
    Reordering these specifications never changes their random streams.
    """

    episode_id: int
    config: EnvConfig
    map_id: int | None = None
    seed_id: int | None = None
    initial_state: EnvState | None = None
    metadata: Mapping[str, object] | None = None

    @property
    def random_seed_id(self) -> int:
        return self.episode_id if self.seed_id is None else self.seed_id


@dataclass(frozen=True)
class EpisodeResult:
    """Compact execution evidence, also usable by the tournament result writer."""

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
    """Completed execution evidence and directly pandas-compatible metric tables."""

    priority_metrics: Columns
    full_metrics: Columns
    episodes: tuple[EpisodeResult, ...]
    metadata: dict[str, object]
    completed_episode_ids: tuple[int, ...] = ()
    paths: dict[str, Path] | None = None
    replays: tuple[ReplayArtifactV2, ...] = ()


class _Completed(NamedTuple):
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
        """The completed old-episode payload consumed by persistence."""
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
    observations: Observations
    state: EnvironmentState
    policy_a: PolicyTree
    policy_b: PolicyTree
    completed: _Completed


def episode_keys(
    root: Array, seed_ids: Array, local_steps: Array, stream: int
) -> Array:
    """Derive keys by stream, planned seed identity and local episode transition."""
    roots = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        jax.random.fold_in(root, stream), seed_ids
    )
    return jax.vmap(jax.random.fold_in)(roots, local_steps)


def _actor_keys(root: Array, seed_ids: Array, local_steps: Array) -> Array:
    keys = episode_keys(root, seed_ids, local_steps, 2)
    return jax.vmap(jax.vmap(jax.random.fold_in, in_axes=(None, 0)), in_axes=(0, None))(
        keys, jnp.arange(10, dtype=jnp.uint32)
    )


def _empty_completed(state: EnvironmentState) -> _Completed:
    count = state.episode_id.shape[0]

    def empty(size: int) -> MetricValues:
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


@jax.jit(static_argnames=("env",))
def _reset(
    env: Environment,
    keys: Array,
    config: EnvConfig,
    ids: Array,
    state: EnvironmentState | None = None,
    reset_mask: Array | None = None,
    initial: InitialSnapshot | None = None,
) -> tuple[Observations, EnvironmentState]:
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
    """Build only the active batch's optional authored starts through Core."""
    if all(spec.initial_state is None for spec in specs):
        return None
    rows: list[InitialSnapshot] = []
    for index, spec in enumerate(specs):
        state, observation, mask, _ = (
            core.reset(spec.config, keys[index])
            if spec.initial_state is None
            else core.initialize_scenario_state(spec.initial_state, spec.config)
        )
        rows.append((state, observation, mask))

    def stack(*values: Array) -> Array:
        return jnp.stack(values)

    return cast(InitialSnapshot, jax.tree.map(stack, *rows))


@jax.jit(static_argnames=("env", "apply_a", "apply_b", "chunk_size"))
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
    def advance(current: _Carry, unused: None) -> tuple[_Carry, ReplayPackets | None]:
        del unused
        local_steps = (
            current.state.core_state.step_count - current.state.initial_step_count
        )
        actor_keys = _actor_keys(root_key, seed_ids, local_steps)
        actions, next_a, next_b = jax.vmap(
            apply_policies, in_axes=(None, None, None, None, 0, 0, 0, 0, 0)
        )(
            apply_a,
            apply_b,
            variables_a,
            variables_b,
            current.policy_a,
            current.policy_b,
            current.observations,
            current.state.action_mask,
            actor_keys,
        )
        active = ~current.state.done.done
        next_a = select_policy_carry(active, next_a, current.policy_a)
        next_b = select_policy_carry(active, next_b, current.policy_b)
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


@jax.jit(static_argnames=("env",))
def _step_environment(
    env: Environment,
    key: Array,
    state: EnvironmentState,
    action: Action,
) -> tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]:
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
    """Keep the same actor barrier for synchronous non-JAX providers."""
    packets: list[ReplayPackets] = []
    for _ in range(chunk_size):
        local_steps = carry.state.core_state.step_count - carry.state.initial_step_count
        keys = _actor_keys(root_key, seed_ids, local_steps)
        rows: list[tuple[Action, PolicyTree, PolicyTree]] = []
        done = np.asarray(carry.state.done.done)
        for index in range(done.size):

            def row(value: Array, index: int = index) -> Array:
                return value[index]

            memory_a = jax.tree.map(row, carry.policy_a)
            memory_b = jax.tree.map(row, carry.policy_b)
            if done[index]:
                zero = jnp.zeros(10, jnp.int32)
                rows.append((Action(zero, zero, zero), memory_a, memory_b))
            else:
                rows.append(
                    apply_policies(
                        team_a.apply,
                        team_b.apply,
                        variables_a,
                        variables_b,
                        memory_a,
                        memory_b,
                        jax.tree.map(row, carry.observations),
                        jax.tree.map(row, carry.state.action_mask),
                        keys[index],
                        execution_a=team_a.execution,
                        execution_b=team_b.execution,
                    )
                )

        def stack(*values: Array) -> Array:
            return jnp.stack(values)

        action, next_a, next_b = jax.tree.map(stack, *rows)
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
        return jnp.stack(values)

    return carry, (
        cast(ReplayPackets, jax.tree.map(stack_packets, *packets)) if packets else None
    )


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _tree_digest(tree: PolicyTree) -> str:
    """Fingerprint frozen numerical content once at the recording boundary."""
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
    """Describe a frozen policy with only identities the executor actually knows."""
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
    def stack(*values: object) -> Array:
        return jnp.stack(tuple(jnp.asarray(value) for value in values))

    return cast(EnvConfig, jax.tree.map(stack, *(spec.config for spec in specs)))


def normalize_episode_specs(
    maps: Iterable[MapInput] | None,
    num_episodes: int,
    team_a_roster: Sequence[AgentClassName],
    team_b_roster: Sequence[AgentClassName],
    score_threshold: int,
    max_steps: int,
) -> tuple[EpisodeSpec, ...]:
    choices = tuple(CANONICAL_TDM_EVALUATION_MAP_IDS if maps is None else maps)
    if not choices:
        raise ValueError("maps must contain at least one map or configuration")
    resolved: list[tuple[int | None, EnvConfig]] = []
    cache: dict[int, EnvConfig] = {}
    for item in choices:
        if isinstance(item, EnvConfig):
            resolved.append((None, item))
            continue
        map_id = item.map_id if isinstance(item, TDMMapInfo) else item
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
    names: tuple[str, ...]
    ids: tuple[int, ...]
    rows: dict[int, int]
    values: NDArray[np.float32]

    @classmethod
    def create(cls, names: tuple[str, ...], ids: Iterable[int]) -> _MetricTable:
        ordered = tuple(sorted(ids))
        return cls(
            names,
            ordered,
            {value: index for index, value in enumerate(ordered)},
            np.full((len(ordered), len(names)), np.nan, dtype=np.float32),
        )

    def append(self, episode_id: int, values: MetricValues | None, lane: int) -> None:
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
                [config_ids[id(specs[value].config)] for value in self.ids]
            ),
        }
        configs = {id(specs[value].config): specs[value].config for value in self.ids}
        profiles = {
            identity: jax.device_get(config.agent_profile)
            for identity, config in configs.items()
        }
        classes = np.asarray(
            [profiles[id(specs[value].config)].class_ids for value in self.ids]
        )
        active = np.asarray(
            [profiles[id(specs[value].config)].active_mask for value in self.ids]
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
    """Execute a frozen schedule; validation and tournaments reuse this boundary."""
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
            start_identity = (id(spec.config), id(spec.initial_state))
            if start_identity not in validated_starts:
                validate_scenario_initial_state(spec.config, spec.initial_state)
                validated_starts.add(start_identity)
        if id(spec.config) not in validated:
            validate_env_config(spec.config)
            if int(spec.config.task_mode) != 1:
                raise ValueError("evaluate currently supports TDM configurations")
            validated.add(id(spec.config))
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
    for name, selected in (
        ("full_metrics_episodes", env.full_metrics_episodes),
        ("replay_episodes", env.replay_episodes),
    ):
        if not set(selected) <= spec_by_id.keys():
            raise ValueError(f"{name} contains an episode outside the schedule")
    full_ids = set(spec_by_id) if metrics == "full" else set(env.full_metrics_episodes)
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
        or env.replay_episodes
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
        "replay_episodes": list(env.replay_episodes),
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
        configs = {id(spec.config): spec.config for spec in specs}

        def normalize(value: object) -> Array:
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
                    "configuration_digest": config_ids[id(spec.config)],
                    "initial_state_digest": (
                        start_digests[id(spec.initial_state)]
                        if spec.initial_state is not None
                        else None
                    ),
                    "expected_horizon": int(spec.config.max_steps)
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
            full_metrics_episodes=env.full_metrics_episodes,
            replay_episodes=env.replay_episodes,
        )
        priority = _MetricTable.create(
            PRIORITY_METRIC_NAMES, priority_ids if writer is None else ()
        )
        full = _MetricTable.create(
            FULL_METRIC_NAMES, full_ids if writer is None else ()
        )
        replays: list[ReplayArtifactV2] = []
        collector = None
        if env.replay_episodes and writer is None:

            def context(
                packet: ReplayPackets,
            ) -> tuple[EvaluationEpisodeContextV2, RuntimeProvenanceV1]:
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
            return jnp.asarray(
                [spec.random_seed_id for spec in lanes], dtype=jnp.uint32
            )

        def ids() -> Array:
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
                    config_ids[id(spec.config)],
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
    team_a_roster: Sequence[AgentClassName] = _ROSTER,
    team_b_roster: Sequence[AgentClassName] = _ROSTER,
    score_threshold: int = 20,
    max_steps: int = 300,
    phase: str = "evaluation",
    pass_id: str = "1",
    chunk_size: int = 16,
) -> EvaluationResult:
    """Evaluate two frozen policies with one total budget across the supplied maps.

    Maps cycle in supplied order; defaults are the five canonical evaluation maps.
    Parameters and recurrent memory remain dynamic JAX inputs. In-memory results
    contain numeric columns, ready for ``pandas.DataFrame(result.priority_metrics)``.
    No file or replay is needed to compute either metric selection.
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
