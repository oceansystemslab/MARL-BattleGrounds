"""Researcher-facing JAX environment with explicit, immutable episode state."""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.episode_metrics import (
    MetricValues,
    PriorityTotals,
    initialize_priority,
    priority_values,
    update_priority,
)
from marl_battlegrounds.evaluation.full_metrics import (
    FullTotals,
    full_values,
    initialize_full,
    update_full,
)
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.replay_capture import ReplayPackets, capture_packets
from marl_battlegrounds.policies.input import Observations, build_observations

type MetricMode = Literal["none", "priority", "full"]
type CoreStepResult = tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]
type InitialSnapshot = tuple[EnvState, Observation, ActionMask]


class EnvironmentState(NamedTuple):
    """Core's paired snapshot, its configuration and optional episode accounting.

    All leaves are dynamic JAX data. Configuration belongs to each episode, not
    mutable Python state on the environment object. This is not Core's EnvState.
    """

    config: EnvConfig
    core_state: EnvState
    observation: Observation
    action_mask: ActionMask
    done: DoneFlags
    episode_id: Array
    initial_step_count: Array
    collect_metrics: Array
    collect_full_metrics: Array
    collect_replay: Array
    priority: PriorityTotals | None
    full: FullTotals | None


class EpisodeInfo(NamedTuple):
    """Numeric step results; completed selects exactly one record per episode.

    These diagnostics are privileged and must not be passed to an actor. A
    collected rollout chunk retains all completion masks, not just its last tick.
    """

    episode_id: Array
    completed: Array
    outcome: Array
    config: EnvConfig
    priority: MetricValues | None
    full: MetricValues | None
    replay: ReplayPackets | None

    @property
    def class_ids(self) -> Array:
        """Completed episode's class assignment, including after a lane reset."""
        return self.config.agent_profile.class_ids

    @property
    def active_mask(self) -> Array:
        """Completed episode's roster mask without another stored copy."""
        return self.config.agent_profile.active_mask


def _episode_selection(values: Iterable[int], name: str) -> tuple[int, ...]:
    selected: set[int] = set()
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 < value <= np.iinfo(np.int32).max
        ):
            raise ValueError(f"{name} must contain positive int32 episode IDs")
        selected.add(int(value))
    return tuple(sorted(selected))


def _selected(episode_id: Array, selection: tuple[int, ...]) -> Array:
    if not selection:
        return jnp.zeros_like(episode_id, dtype=bool)
    ids = jnp.asarray(selection, jnp.int32)
    index = jnp.searchsorted(ids, episode_id)
    return (index < ids.size) & (ids[jnp.minimum(index, ids.size - 1)] == episode_id)


def _raise_invalid_reset() -> None:
    raise ValueError("episode IDs must be positive int32 and unique in active lanes")


def _check_episode_ids(ids: Array) -> None:
    # Like resolved Core configuration, dynamic IDs are a compiled-call
    # precondition. Check concrete inputs here; evaluate/writers also own their
    # schedule boundary. A failing callback inside cond is not vmap-safe and
    # would add host synchronization to researcher-owned reset loops.
    if isinstance(ids, Tracer):
        return
    ordered = np.sort(np.atleast_1d(np.asarray(ids)))
    if np.any(ordered <= 0) or np.any(ordered[1:] == ordered[:-1]):
        _raise_invalid_reset()


def _config_array(value: object) -> Array:
    array = jnp.asarray(value)
    return jnp.asarray(array, dtype=array.dtype)


def _episode_keys(key: Array, episode_ids: Array) -> Array:
    key_shape = jax.random.key_data(key).shape
    if key_shape == (episode_ids.shape[0], 2):
        return key
    if key_shape != (2,):
        raise ValueError("key must be one root key or one key per environment")
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, episode_ids)


@dataclass(frozen=True)
class Environment:
    """A static execution choice over Core; episode settings enter through reset.

    JIT ``reset`` and ``step`` or use them inside the researcher's compiled scan.
    An omitted batch size is scalar. A supplied size adds one leading batch axis.
    Native calls accept one root key (folded by episode ID), or one key per lane.
    """

    num_envs: int | None
    metrics: MetricMode
    full_metrics_episodes: tuple[int, ...]
    replay_episodes: tuple[int, ...]

    def get_action_mask(self, state: EnvironmentState) -> ActionMask:
        """Return the exact mask that Core paired with this state."""
        return state.action_mask

    def _reset_one(
        self, key: Array, config: EnvConfig, episode_id: Array
    ) -> EnvironmentState:
        core_state, observation, action_mask, _ = core.reset(config, key)
        return self._initialize_one(
            config, episode_id, core_state, observation, action_mask
        )

    def _initialize_one(
        self,
        config: EnvConfig,
        episode_id: Array,
        core_state: EnvState,
        observation: Observation,
        action_mask: ActionMask,
    ) -> EnvironmentState:
        """Package a paired Core snapshot through one metric/reset authority."""
        full = jnp.asarray(self.metrics == "full") | _selected(
            episode_id, self.full_metrics_episodes
        )
        collect = jnp.asarray(self.metrics != "none") | full
        return EnvironmentState(
            config,
            core_state,
            observation,
            action_mask,
            DoneFlags(jnp.asarray(False), jnp.asarray(False)),
            episode_id,
            core_state.step_count,
            collect,
            full,
            _selected(episode_id, self.replay_episodes),
            initialize_priority()
            if self.metrics != "none" or self.full_metrics_episodes
            else None,
            initialize_full(config, core_state)
            if self.metrics == "full" or self.full_metrics_episodes
            else None,
        )

    def reset(
        self,
        key: Array,
        config: EnvConfig,
        *,
        episode_id: int | Array,
        state: EnvironmentState | None = None,
        reset_mask: Array | None = None,
        initial: InitialSnapshot | None = None,
    ) -> tuple[Observations, EnvironmentState]:
        """Start episodes, or replace selected lanes atomically without autoreset.

        Config must be resolved by the existing task factories before tracing.
        Dynamic episode IDs must be positive and unique across active lanes;
        concrete inputs are checked here, before compiled execution.
        Native batches accept one config broadcast to all lanes or a config
        whose leaves already have a leading ``num_envs`` dimension.
        ``initial`` optionally supplies an already validated Core scenario
        state/observation/mask triple; native triples have a leading batch axis.
        """
        if (state is None) != (reset_mask is None):
            raise ValueError("partial reset requires both state and reset_mask")
        if not isinstance(episode_id, Tracer):
            host_ids = np.asarray(episode_id)
            if not np.issubdtype(host_ids.dtype, np.integer) or np.any(
                (host_ids <= 0) | (host_ids > np.iinfo(np.int32).max)
            ):
                _raise_invalid_reset()
        ids = jnp.asarray(episode_id)
        expected = () if self.num_envs is None else (self.num_envs,)
        if ids.shape != expected or not jnp.issubdtype(ids.dtype, jnp.integer):
            raise ValueError(f"episode_id must have integer dtype and shape {expected}")
        if ids.dtype.itemsize > 4:
            raise ValueError("episode_id must fit int32")
        ids = ids.astype(jnp.int32)
        config = jax.tree.map(_config_array, config)
        if self.num_envs is None:
            fresh = (
                self._reset_one(key, config, ids)
                if initial is None
                else self._initialize_one(config, ids, *initial)
            )
        else:
            if config.agent_profile.active_mask.ndim == 1:
                batch_size = self.num_envs

                def broadcast(value: Array) -> Array:
                    return jnp.broadcast_to(value, (batch_size, *value.shape))

                config = jax.tree.map(broadcast, config)
            elif config.agent_profile.active_mask.shape != (self.num_envs, 10):
                raise ValueError("batched configuration must match num_envs")
            keys = _episode_keys(key, ids)
            fresh = (
                jax.vmap(self._reset_one)(keys, config, ids)
                if initial is None
                else jax.vmap(self._initialize_one)(config, ids, *initial)
            )
        if state is not None and reset_mask is not None:
            mask = jnp.asarray(reset_mask)
            if mask.shape != expected or mask.dtype != jnp.bool_:
                raise ValueError(
                    f"reset_mask must have boolean dtype and shape {expected}"
                )

            def replace(new: Array, old: Array) -> Array:
                return jnp.where(
                    mask.reshape((*mask.shape, *((1,) * (new.ndim - mask.ndim)))),
                    new,
                    old,
                )

            fresh = jax.tree.map(replace, fresh, state)
        _check_episode_ids(fresh.episode_id)
        return self._observations(fresh), fresh

    def _observations(self, state: EnvironmentState) -> Observations:
        if self.num_envs is None:
            return build_observations(state.observation, state.config)
        return jax.vmap(build_observations)(state.observation, state.config)

    def _step_one(
        self, key: Array, state: EnvironmentState, actions: Action
    ) -> tuple[EnvironmentState, Reward, EpisodeInfo, Info]:
        def advance() -> CoreStepResult:
            return core.step(
                state.config, state.core_state, state.action_mask, actions, key
            )

        def pad() -> CoreStepResult:
            return (
                state.core_state,
                state.observation,
                Reward(jnp.zeros(10, jnp.float32)),
                state.done,
                state.action_mask,
                core.build_canonical_no_transition_info_object(state.core_state),
            )

        core_state, observation, reward, done, mask, info = cast(
            CoreStepResult, jax.lax.cond(state.done.done, pad, advance)
        )
        completed = ~state.done.done & done.done
        totals = state.priority
        values = None
        if totals is not None:
            totals = update_priority(totals, reward, info)
            values = priority_values(
                state.config,
                core_state,
                state.initial_step_count,
                totals,
                info.transition_facts.team_deathmatch_facts.outcome,
            )
            values = values._replace(valid=values.valid & state.collect_metrics)
        successor = EnvironmentState(
            state.config,
            core_state,
            observation,
            mask,
            done,
            state.episode_id,
            state.initial_step_count,
            state.collect_metrics,
            state.collect_full_metrics,
            state.collect_replay,
            totals,
            state.full,
        )
        result = EpisodeInfo(
            state.episode_id,
            completed,
            info.transition_facts.team_deathmatch_facts.outcome,
            state.config,
            values,
            None,
            None,
        )
        return successor, reward, result, info

    def _full_one(
        self,
        state: EnvironmentState,
        result: EpisodeInfo,
        facts: Info,
    ) -> tuple[FullTotals, MetricValues]:
        priority = result.priority
        assert state.full is not None and priority is not None
        totals = update_full(
            state.full, state.config, state.core_state, state.action_mask, facts
        )
        empty = MetricValues(
            jnp.zeros(len(FULL_METRIC_NAMES), jnp.float32),
            jnp.zeros(len(FULL_METRIC_NAMES), bool),
        )
        values = cast(
            MetricValues,
            jax.lax.cond(
                result.completed,
                lambda: full_values(totals, state.config, priority),
                lambda: empty,
            ),
        )
        return totals, values

    def _collect_full(
        self,
        state: EnvironmentState,
        successor: EnvironmentState,
        result: EpisodeInfo,
        facts: Info,
    ) -> tuple[EnvironmentState, EpisodeInfo]:
        if state.full is None:
            return successor, result
        shape = (*state.episode_id.shape, len(FULL_METRIC_NAMES))
        empty = MetricValues(jnp.zeros(shape, jnp.float32), jnp.zeros(shape, bool))
        selected = state.collect_full_metrics & ~state.done.done
        if self.num_envs is None:

            def collect() -> tuple[FullTotals, MetricValues]:
                return self._full_one(state, result, facts)

            totals, values = cast(
                tuple[FullTotals, MetricValues],
                jax.lax.cond(selected, collect, lambda: (state.full, empty)),
            )
        elif self.metrics == "full":
            totals = jax.vmap(update_full)(
                state.full, state.config, state.core_state, state.action_mask, facts
            )
            priority = result.priority
            assert priority is not None

            def finalize() -> MetricValues:
                values = jax.vmap(full_values)(totals, state.config, priority)
                return values._replace(valid=values.valid & result.completed[:, None])

            values = cast(
                MetricValues,
                jax.lax.cond(jnp.any(result.completed), finalize, lambda: empty),
            )
        else:
            # A scalar conditional inside a loop really skips unselected work;
            # vmap(cond) can instead evaluate both branches. Sparse schedules
            # favor this simple path; all-full collection stays vectorized.
            capacity = min(self.num_envs, len(self.full_metrics_episodes))
            indices = jnp.nonzero(selected, size=capacity, fill_value=0)[0]

            def collect_lane(
                offset: Array, carry: tuple[FullTotals, MetricValues]
            ) -> tuple[FullTotals, MetricValues]:
                lane = indices[offset]

                def read(value: Array) -> Array:
                    return value[lane]

                lane_totals, lane_values = self._full_one(
                    jax.tree.map(read, state),
                    jax.tree.map(read, result),
                    jax.tree.map(read, facts),
                )

                def write(batch: Array, value: Array) -> Array:
                    return batch.at[lane].set(value)

                return jax.tree.map(write, carry, (lane_totals, lane_values))

            totals, values = cast(
                tuple[FullTotals, MetricValues],
                jax.lax.fori_loop(
                    0, selected.sum(dtype=jnp.int32), collect_lane, (state.full, empty)
                ),
            )
        return successor._replace(full=totals), result._replace(full=values)

    def step(
        self, key: Array, state: EnvironmentState, actions: Action
    ) -> tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]:
        """Advance using the config and authoritative mask held in wrapper state."""
        if self.num_envs is None:
            successor, reward, info, facts = self._step_one(key, state, actions)
        else:
            keys = _episode_keys(key, state.episode_id)
            successor, reward, info, facts = jax.vmap(self._step_one)(
                keys, state, actions
            )
        successor, info = self._collect_full(state, successor, info, facts)
        if self.replay_episodes:
            info = info._replace(
                replay=capture_packets(
                    state,
                    successor,
                    reward,
                    facts,
                    capacity=min(self.num_envs or 1, len(self.replay_episodes)),
                )
            )
        return self._observations(successor), successor, reward, successor.done, info


def make(
    task: str,
    *,
    num_envs: int | None = None,
    metrics: MetricMode = "priority",
    full_metrics_episodes: Iterable[int] = (),
    replay_episodes: Iterable[int] = (),
) -> Environment:
    """Construct a thin environment; all actual episode config enters at reset."""
    if task != "tdm":
        raise ValueError("the supported task is 'tdm'")
    if num_envs is not None and (
        isinstance(num_envs, bool) or type(num_envs) is not int or num_envs < 1
    ):
        raise ValueError("num_envs must be a positive integer or omitted")
    if metrics not in ("none", "priority", "full"):
        raise ValueError("metrics must be 'none', 'priority', or 'full'")
    return Environment(
        num_envs,
        metrics,
        _episode_selection(full_metrics_episodes, "full_metrics_episodes"),
        _episode_selection(replay_episodes, "replay_episodes"),
    )
