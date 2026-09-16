"""Ready JAX environments with explicit, immutable episode state.

Use ``make`` to prepare an environment, ``reset`` to start games, and ``step``
to advance them. Finished games stay finished until an explicit reset. Core owns
observations, masks, game rules and rewards; this module owns batching, episode
IDs, lifecycle counters and optional metric/replay capture. It writes no files.

Scalar calls have no leading environment axis. Native batches add an axis of
length ``num_envs`` to each numerical state leaf. Returned state is a JAX PyTree:
carry it through ``jit``, ``vmap`` or ``lax.scan`` without hidden mutable state.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from numpy.typing import NDArray

from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    TEAM_A_ID,
    TEAM_B_ID,
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
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.policies.no_shared_obs import execute_no_shared_obs_team_policy
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
)
from marl_battlegrounds.spaces import (
    StructuredSpace,
)
from marl_battlegrounds.spaces import (
    action_space as describe_action_space,
)
from marl_battlegrounds.spaces import (
    observation_space as describe_observation_space,
)
from marl_battlegrounds.tasks import (
    AgentClassName,
    TDMScenario,
    balanced_spawn_configs,
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
    prepare_exact_env_config,
)

type MetricMode = Literal["none", "priority", "full"]
type CoreStepResult = tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]
type InitialSnapshot = tuple[EnvState, Observation, ActionMask]
type Selection = Array | NDArray[np.int32]

_MAX_LIFECYCLE_COUNT = np.iinfo(np.int32).max
RESET_DEFAULT = 0
RESET_REUSE = 1
RESET_OVERRIDE = 2
RESET_AUTHORED = 3
_AGENTS = tuple(f"agent_{index}" for index in range(MAX_AGENT_SLOTS))


class EnvironmentState(NamedTuple):
    """Core's paired snapshot, its configuration and optional episode accounting.

    All leaves are dynamic JAX data. Configuration belongs to each episode, not
    mutable Python state on the environment object. This is not Core's EnvState
    and must not be passed to actors as an observation. Carry the latest returned
    value through reset/step; fields describing a snapshot must stay together.

    Attributes
    ----------
    config : EnvConfig
        Exact resolved EnvConfig, including both complete spawn banks.
    core_state : EnvState
        The paired simulator state, owned by Core.
    observation : Observation
        Core's observations for all ten capacity slots.
    action_mask : ActionMask
        Core's legal choices for that same decision.
    done : DoneFlags
        Core termination/truncation flags. Reset clears them.
    episode_id : Array
        Positive int32 ID, scalar or ``(B,)`` for a native batch.
    initial_step_count : Array
        Core decision index at this episode's authored start.
    collect_metrics : Array
        Boolean per-game flag for priority collection.
    collect_full_metrics : Array
        Boolean per-game flag for full collection.
    collect_replay : Array
        Boolean per-game flag for selected replay capture.
    priority : PriorityTotals | None
        Priority counters, or None when that structure is disabled.
    full : FullTotals | None
        Full counters, or None when that structure is disabled.
    source_availability : Array
        Boolean recipient/source matrix ``(..., 10, 10)``.
        Only the permitted same-team, non-self source subset may be enabled.
    episode_start : Array
        True until the new episode's first real transition.
    reset_generation : Array
        Int32 reset count, initially zero for each lane.
    reset_origin : Array
        Int32 marker: default=0, reuse=1, override=2, authored=3.
    authored_start : Array
        Whether the current episode began from an exact snapshot.
    config_origin_generation : Array
        Generation of the latest explicit config or
        authored replacement; -1 means untouched constructor defaults.
    cumulative_transition_count : Array
        Int32 real advances in this execution
        context. Includes terminal advances; excludes resets and padding.
    lifecycle_error : Array
        Sticky Boolean evidence of invalid numerical IDs or
        exhausted lifecycle counters. Reset cannot clear an existing error.
    last_reserved_episode_id : Array
        Int32 allocation high-water mark per lane.
        Native allocation reads their maximum; only reset lanes are changed.

    Notes
    -----
    Lifecycle scalars gain shape ``(B,)`` in native batches. Checked counters have
    maximum 2,147,483,647. On exhaustion, preserve the last valid value and inspect
    ``lifecycle_error``; do not treat such a context as valid recorded training.
    Separate raw contexts have separate ID spaces. Recording and later training
    tracking have their own validation boundaries.
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
    source_availability: Array  # Boolean recipient/source matrix; 10 by 10 per game.
    episode_start: Array
    reset_generation: Array
    reset_origin: Array
    authored_start: Array
    config_origin_generation: Array
    cumulative_transition_count: Array
    lifecycle_error: Array
    last_reserved_episode_id: Array


class EpisodeInfo(NamedTuple):
    """Numeric step results; completed selects exactly one record per episode.

    These diagnostics are privileged and must not be passed to an actor. A
    collected rollout chunk retains all completion masks, not just its last tick.

    Attributes
    ----------
    episode_id : Array
        Int32 ID of the game that produced this transition.
    completed : Array
        Boolean scalar or ``(B,)``; true only on its final real step.
    outcome : Array
        Core's outcome code, meaningful for completed games.
    config : EnvConfig
        The producing game's exact EnvConfig, for participant ownership.
    priority : MetricValues | None
        Named metric values and validity flags, or None if disabled.
    full : MetricValues | None
        Named full values and validity flags, or None if disabled.
    replay : ReplayPackets | None
        Selected numerical replay packets, or None if capture is disabled.

    Notes
    -----
    Metric arrays have their existing schema and one leading lane axis when
    batched. Respect each value's validity flag and ``completed``; padding must
    not become another episode result. Capture does not write or publish a file.
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
        """Return producing-game int32 class IDs, shaped (10,) or (B, 10).

        This reads info.config directly. A later environment reset does not
        change which game these IDs describe. They are privileged metadata.
        """
        return self.config.agent_profile.class_ids

    @property
    def active_mask(self) -> Array:
        """Return the producing game's Boolean (10,) or (B, 10) roster mask.

        The array is read from info.config without another stored copy. It marks
        roster participation, not current life or visibility, and is privileged.
        """
        return self.config.agent_profile.active_mask


def _episode_selection(values: Iterable[int], name: str) -> tuple[int, ...]:
    """Validate recording selections and return sorted unique episode IDs.

    values is consumed once and must contain positive integers no larger than
    2,147,483,647; booleans are rejected. name identifies the offending selection
    in a ValueError. The host result is an immutable tuple; an empty input gives
    an empty tuple. No device execution or record writing occurs here.
    """
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


def _selected(episode_id: Array, ids: Selection) -> Array:
    """Test IDs against a sorted selection using dynamic numerical membership."""
    if ids.size == 0:
        return jnp.zeros_like(episode_id, dtype=bool)
    ids = jnp.asarray(ids)
    index = jnp.searchsorted(ids, episode_id)
    return (index < ids.size) & (ids[jnp.minimum(index, ids.size - 1)] == episode_id)


def _raise_invalid_reset() -> None:
    """Raise the shared ValueError for invalid or colliding reset episode IDs."""
    raise ValueError("episode IDs must be positive int32 and unique in active lanes")


def _check_episode_ids(ids: Array) -> None:
    # Like resolved Core configuration, dynamic IDs are a compiled-call
    # precondition. Check concrete inputs here; evaluate/writers also own their
    # schedule boundary. A failing callback inside cond is not vmap-safe and
    # would add host synchronization to researcher-owned reset loops.
    """Check concrete episode IDs for positivity and uniqueness on the host.

    ids has already passed the reset shape and integer-range checks. Invalid
    concrete values raise ValueError. JAX tracers are left to the compiled caller's
    documented precondition; this helper does not add a callback to the device
    program. Inspecting a concrete device array can transfer it to the host.
    """
    if isinstance(ids, Tracer):
        return
    ordered = np.sort(np.atleast_1d(np.asarray(ids)))
    if np.any(ordered <= 0) or np.any(ordered[1:] == ordered[:-1]):
        _raise_invalid_reset()


def _config_array(value: object) -> Array:
    """Convert a configuration leaf to a JAX array with an explicit dtype.

    value must be accepted by jnp.asarray. The second conversion preserves the
    inferred dtype while removing weak scalar typing. It does not validate the
    leaf's physical meaning or change a numerical value.
    """
    array = jnp.asarray(value)
    return jnp.asarray(array, dtype=array.dtype)


def _checked_increment(value: Array, amount: int | Array) -> tuple[Array, Array]:
    """Add a nonnegative amount without wrapping an int32 lifecycle counter.

    value is an int32 scalar or array within the valid counter range. amount is
    a nonnegative scalar or same-shaped increment within that range. Return the
    new values and Boolean overflow flags. An overflowing element keeps its old
    value. The caller combines these flags with any earlier failure evidence.
    """
    overflow = value > _MAX_LIFECYCLE_COUNT - amount
    return value + jnp.where(overflow, 0, amount), overflow


def _episode_keys(key: Array, episode_ids: Array) -> Array:
    """Keep explicit lane keys, or fold a root with native episode identities.

    Both typed and legacy JAX keys are accepted. Episode IDs have shape (B,).
    Invalid key batch shapes fail while preparing/tracing the numerical call.
    """
    key_shape = jax.random.key_data(key).shape
    if key_shape == (episode_ids.shape[0], 2):
        return key
    if key_shape != (2,):
        raise ValueError("key must be one root key or one key per environment")
    return jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, episode_ids)


@dataclass(frozen=True)
class _Execution:
    """Store the small set of choices that change a compiled program's structure.

    Attributes
    ----------
    num_envs : int or None
        Native batch size, or None for scalar execution.
    metrics : MetricMode
        Default mode: priority, full or none.
    full_capacity : int
        Number of distinct explicitly selected full-metric episode IDs.
    replay_capacity : int
        Number of distinct explicitly selected replay episode IDs.

    Notes
    -----
    Configuration arrays and the selected IDs themselves belong to dynamic
    Environment fields. This private frozen record is the static JAX metadata;
    it must not capture ordinary changing map or state values.
    """

    num_envs: int | None
    metrics: MetricMode
    full_capacity: int
    replay_capacity: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Environment:
    """A ready execution handle whose numerical inputs remain dynamic.

    Create instances through ``make``; the underscored constructor fields are
    internal execution data, not a second public configuration interface.
    JIT ``reset`` and ``step`` or use them inside the researcher's compiled scan.
    An omitted batch size is scalar. A supplied size adds one leading batch axis.
    Native calls accept one root key (folded by episode ID), or one key per lane.
    Reusing a handle does not mutate its defaults or advance any hidden RNG.
    Pass each returned state onward and supply distinct random keys as needed.
    Prepared config values are dynamic; batch shape, metric mode and capture
    capacities determine the compiled structure.
    """

    _execution: _Execution = field(metadata={"static": True})
    _full_ids: Selection
    _replay_ids: Selection
    _default_config: EnvConfig | None = None
    _source_config: EnvConfig | None = None
    _source_map_id: Array | None = None

    @property
    def num_envs(self) -> int | None:
        """Native batch size B, or None for calls without an environment axis."""
        return self._execution.num_envs

    @property
    def metrics(self) -> MetricMode:
        """Default metric mode: priority, full or none; explicit selections add full."""
        return self._execution.metrics

    @property
    def full_metrics_episodes(self) -> tuple[int, ...]:
        """Read the host selection for schedule validation and saved metadata."""
        return tuple(map(int, np.asarray(self._full_ids)))

    @property
    def replay_episodes(self) -> tuple[int, ...]:
        """Read the host selection; compiled capture uses lane flags instead."""
        return tuple(map(int, np.asarray(self._replay_ids)))

    @property
    def agents(self) -> tuple[str, ...]:
        """Fixed capacity slots; the configuration separately selects active agents."""
        return _AGENTS

    @property
    def num_agents(self) -> int:
        """Return the fixed capacity of ten actor slots, including inactive slots.

        Read agent_info(state)["active_mask"] for roster participation; capacity does
        not shrink when a team uses fewer than five agents.
        """
        return MAX_AGENT_SLOTS

    def action_space(self, agent: str) -> StructuredSpace:
        """Describe one capacity slot's action fields, dtypes and bounds.

        Parameters
        ----------
        agent : str
            A name in ``env.agents``, from ``agent_0`` to ``agent_9``.

        Returns
        -------
        StructuredSpace
            A cached host descriptor for the three scalar int32 action fields.
            ``contains`` checks structure, not current legal combinations. Use
            ``get_action_mask`` or ``sample_actions`` for state-dependent legality.

        Raises
        ------
        ValueError
            The agent name is not a capacity slot. Inactive slots
            still have a structural space; activity comes from the state.
        """
        if agent not in self.agents:
            raise ValueError(f"unknown agent {agent!r}")
        return describe_action_space()

    def observation_space(self, agent: str) -> StructuredSpace:
        """Describe one slot's structured Core observation without allocating it.

        Parameters
        ----------
        agent : str
            A name in ``env.agents``, from ``agent_0`` to ``agent_9``.

        Returns
        -------
        StructuredSpace
            A cached host descriptor of field shapes, dtypes and bounds for one
            actor. It has no environment or actor batch axis and does not include
            the surrounding ``Observations.source_availability`` matrix.

        Raises
        ------
        ValueError
            The name is not a capacity slot.

        Notes
        -----
        This describes Core's existing data. It does not flatten features or add
        identities, and structural membership does not authorize sharing inputs.
        """
        if agent not in self.agents:
            raise ValueError(f"unknown agent {agent!r}")
        return describe_observation_space()

    def agent_info(self, state: EnvironmentState) -> dict[str, Array]:
        """Read exact roster metadata for setup, routing and analysis.

        Parameters
        ----------
        state : EnvironmentState
            The current scalar or native EnvironmentState.

        Returns
        -------
        dict[str, Array]
            Existing ``class_ids`` and ``team_ids`` int32 arrays and Boolean
            ``active_mask``, each shaped ``(10,)`` or ``(B, 10)``. Values remain
            on their current device; no copies or host conversion are requested.

        Notes
        -----
        This is privileged runner metadata, not an actor input. Returning it
        does not expand any actor's information rights.
        """
        profile = state.config.agent_profile
        return {
            "class_ids": profile.class_ids,
            "team_ids": profile.team_ids,
            "active_mask": profile.active_mask,
        }

    def sample_actions(self, key: Array, state: EnvironmentState) -> Action:
        """Sample legal actions using the existing random-valid policy.

        Parameters
        ----------
        key : Array
            A JAX typed or legacy key. Native batches also accept one key
            per lane; a single native root is folded with each episode ID.
            Native keys must use the standard two-word key-data form, with
            raw key-data shape (2,) for a root or (B, 2) for B lane keys.
        state : EnvironmentState
            The current state, with Core's paired observations and masks.

        Returns
        -------
        Action
            Action with int32 ``move``, ``select_target`` and ``use_ultimate``
            arrays shaped ``(10,)`` or ``(B, 10)``. The sampler respects coupled
            target/Ultimate legality and the masks of inactive or dead actors.

        Notes
        -----
        Invalid native key-data shapes raise ValueError during tracing. Other
        malformed key inputs may raise JAX type or shape errors. This method does
        not validate a user-replaced mask; keep Core's current paired mask.
        Slot keys are derived in simulator slot order; they are not actor inputs.
        The call does not change state, advance a hidden RNG, construct SharedObs
        inputs or write records. It works inside jit/vmap/scan. Use a new key for
        another draw; equal keys and inputs reproduce the same draw.
        """
        return cast(Action, _sample_actions(self, key, state))

    def get_action_mask(self, state: EnvironmentState) -> ActionMask:
        """Return the exact ActionMask paired with the supplied state.

        The returned Boolean leaves retain their Core shapes, with a leading
        ``B`` axis for native batches. Use the joint target/Ultimate mask as well
        as the individual head masks. This getter neither copies nor recomputes
        the mask. Do not combine it with observations from another decision.
        """
        return state.action_mask

    def _reset_one(
        self, key: Array, config: EnvConfig, episode_id: Array
    ) -> EnvironmentState:
        """Reset one scalar game in Core and package its initial wrapper state.

        config is already resolved and validated; episode_id is the selected int32
        identity. key is one JAX reset key. Core owns the initial observation and mask.
        The returned EnvironmentState adds fresh metric and lifecycle fields without
        writing a record or changing constructor defaults.
        """
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
        """Wrap a matching Core state, observation and mask for a new scalar episode.

        config, core_state, observation and action_mask must describe the same start.
        episode_id is a valid scalar int32 ID. Initialize optional metrics from this
        start, record selection flags and restore default source permissions. Done
        flags and per-episode counters start clear. The outer reset path later merges
        retained lifetime counters and selected lanes.
        """
        full = jnp.asarray(self.metrics == "full") | _selected(
            episode_id, self._full_ids
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
            _selected(episode_id, self._replay_ids),
            initialize_priority()
            if self.metrics != "none" or self._execution.full_capacity
            else None,
            initialize_full(config, core_state)
            if self.metrics == "full" or self._execution.full_capacity
            else None,
            build_default_shared_obs_information_availability(
                config.agent_profile.active_mask, config.agent_profile.team_ids
            ),
            jnp.asarray(True),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(RESET_DEFAULT, jnp.int32),
            jnp.asarray(False),
            jnp.asarray(-1, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
            episode_id,
        )

    def reset(
        self,
        key: Array,
        env_config: EnvConfig | None = None,
        *,
        episode_id: int | Array | None = None,
        state: EnvironmentState | None = None,
        reset_mask: Array | None = None,
        initial: InitialSnapshot | None = None,
        scenario: TDMScenario | None = None,
    ) -> tuple[Observations, EnvironmentState]:
        """Start or replace episodes while retaining continuing lanes exactly.

        Parameters
        ----------
        key : Array
            Standard JAX typed key with shape (), or legacy uint32 key with
            shape (2,). A native batch also accepts typed keys with shape (B,)
            or legacy keys with shape (B, 2). Native key data must use the
            standard two-word form. A root is folded with the new episode IDs;
            explicit per-lane keys are used unchanged.
        env_config : EnvConfig | None, default=None
            Exact scalar config, broadcast across a native batch,
            or an exact batch whose every leaf has leading size B. Overrides
            retained state.config, which overrides constructor defaults.
            Reset never exchanges spawn banks. Omitted on the ready path.
        episode_id : int | Array | None, default=None
            Positive int32-compatible ID. Use a scalar for scalar execution
            or shape ``(B,)`` for a native batch; IDs do not broadcast.
            Resulting native IDs must be unique. Omission allocates IDs
            1..B initially, then reserves B fresh IDs whenever any lane resets.
            Explicit IDs may reserve gaps and never lower allocation history.
        state : EnvironmentState | None, default=None
            Latest returned state for a continuing execution context.
            Omission starts a new context with fresh counters and ID space.
        reset_mask : Array | None, default=None
            Boolean scalar or ``(B,)`` selecting lanes to replace.
            Requires state; omission with state selects every lane. False
            lanes retain every state leaf, including counters and configs.
        initial : InitialSnapshot | None, default=None
            Prepared ``(EnvState, Observation, ActionMask)`` snapshot
            for an exact authored start. Its shapes must match scalar/native
            execution and its values must belong to the resolved config.
            Prepare and validate it through Core's scenario initializer.
        scenario : TDMScenario | None, default=None
            Loaded TDMScenario with its own exact config and start.
            Cannot be combined with env_config or initial. Host setup creates
            the paired snapshot; native batches repeat it in all reset lanes.

        Returns
        -------
        tuple[Observations, EnvironmentState]
            ``(observations, state)`` for the resulting context. Observations
            contains Core's compact observations and source availability. Each
            reset lane has fresh metric counters and done=False. Existing
            cumulative real-transition counts and failure flags survive resets.

        Raises
        ------
        ValueError
            No configuration is available; mask/ID shapes or values
            are invalid; native IDs conflict; scenario inputs conflict; or
            scenario= is requested while tracing a compiled call; or a native
            random-key array has the wrong shape.
        TypeError
            A supplied config has an unsupported structure or dtype.
            Core's existing validators also reject invalid concrete configs.

        Notes
        -----
        This call consumes no transition and writes no records. Exact overrides
        are retained by later resets that reuse state. Config values may change
        inside jit/scan when selected from a source pool validated on the host;
        there are no per-step Python callbacks. Compiled callers also own valid
        paired initial snapshots and nonconflicting explicit IDs. Out-of-range
        numerical IDs or exhausted int32 counters set sticky lifecycle_error.
        Resetting zero lanes reserves no IDs. Separate native batches or external
        vmap contexts do not share an allocator; use explicit schedule IDs when
        reproducing the same experiment at a different executor batch size.
        """
        expected = () if self.num_envs is None else (self.num_envs,)
        if state is None and reset_mask is not None:
            raise ValueError("reset_mask requires a continuing state")
        if reset_mask is not None:
            reset_mask = jnp.asarray(reset_mask)
            if reset_mask.shape != expected or reset_mask.dtype != jnp.bool_:
                raise ValueError(
                    f"reset_mask must have boolean dtype and shape {expected}"
                )
        elif state is not None:
            reset_mask = jnp.ones(expected, dtype=jnp.bool_)
        origin = RESET_OVERRIDE if env_config is not None else RESET_DEFAULT
        if scenario is not None:
            if isinstance(key, Tracer):
                raise ValueError(
                    "prepare scenario before jit; pass its initial snapshot"
                )
            if env_config is not None or initial is not None:
                raise ValueError(
                    "scenario cannot be combined with env_config or initial"
                )
            env_config = scenario.config
            initial_state, observation, mask, _ = core.initialize_scenario_state(
                scenario.initial_state, scenario.config
            )
            initial = initial_state, observation, mask
            if self.num_envs is not None:
                count = self.num_envs

                def broadcast_initial(value: Array) -> Array:
                    """Repeat an authored snapshot across the native batch.

                    value is one validated snapshot leaf. Add a leading axis
                    of size count. Every selected lane uses the same start.
                    """
                    return jnp.broadcast_to(value, (count, *value.shape))

                initial = jax.tree.map(broadcast_initial, initial)
        if initial is not None:
            origin = RESET_AUTHORED
        if env_config is not None:
            config = prepare_exact_env_config(env_config, num_envs=self.num_envs)
        elif state is not None:
            config = state.config
            if initial is None:
                origin = RESET_REUSE
        elif self._default_config is not None:
            config = self._default_config
        else:
            raise ValueError(
                "reset needs env_config, a scenario, or configured defaults"
            )
        ids = None
        if episode_id is not None:
            if not isinstance(episode_id, Tracer):
                host_ids = np.asarray(episode_id)
                if not np.issubdtype(host_ids.dtype, np.integer) or np.any(
                    (host_ids <= 0) | (host_ids > _MAX_LIFECYCLE_COUNT)
                ):
                    _raise_invalid_reset()
            ids = jnp.asarray(episode_id)
            if ids.shape != expected or not jnp.issubdtype(ids.dtype, jnp.integer):
                raise ValueError(
                    f"episode_id must have integer dtype and shape {expected}"
                )
        result = cast(
            tuple[Observations, EnvironmentState],
            _reset_environment(
                self,
                key,
                config,
                ids,
                state,
                reset_mask,
                initial,
                jnp.asarray(origin, jnp.int32),
            ),
        )
        if episode_id is not None:
            _check_episode_ids(result[1].episode_id)
        return result

    def reset_done(
        self, key: Array, state: EnvironmentState, env_config: EnvConfig | None = None
    ) -> tuple[Observations, EnvironmentState]:
        """Start new games wherever the previous games have finished.

        Games that are still running keep their state and configuration.
        This call does not advance the simulation.

        Parameters
        ----------
        key : Array
            JAX random key for the new games. For a batch, provide one shared key
            or one key per game. A shared key is combined with each new episode ID.
            Typed keys are scalar or shape (B,); legacy uint32 keys have shape (2,)
            or (B, 2), where B is this environment's batch size.
        state : EnvironmentState
            Latest state returned by this environment. Replace a game when its
            state.done.done value is True.
        env_config : EnvConfig or None, default=None
            Exact configuration for replacement games. None keeps each game's
            current configuration, including any earlier override.

            For B games, supply one configuration to repeat across the batch,
            or a configuration whose every array has a leading axis of size B.
            Continuing games retain their configuration. Spawn banks are used
            as supplied; reset does not exchange them.

        Returns
        -------
        observations : Observations
            Observations for the resulting games. Replacement games have starting
            observations; continuing games keep their observations.
        state : EnvironmentState
            Updated state to pass to the next call. Replacement games have new
            episode IDs and cleared episode metrics. Cumulative transition counts
            and existing error flags remain.

        Raises
        ------
        TypeError
            The supplied configuration has an unsupported type or dtype.
        ValueError
            A supplied configuration has invalid shapes or values, or a batched
            random-key array has the wrong shape.

        Notes
        -----
        Save the terminal observation, reward and info from step before resetting
        if your learning method needs them. Inputs are not changed in place. If no
        games have finished, their state is unchanged and no IDs are allocated.
        Supplied configurations still pass through the reset validation boundary.

        This method supports compiled JAX loops. Configuration values selected inside
        a compiled loop must come from configurations already validated at host setup.
        Episode IDs and lifecycle counters have an int32 limit of 2,147,483,647.
        Exhaustion sets state.lifecycle_error instead of silently wrapping the count.

        See Also
        --------
        Environment.reset : Reset selected games or supply an authored start.
        """
        return self.reset(key, env_config, state=state, reset_mask=state.done.done)

    def _reset_numerical(
        self,
        key: Array,
        config: EnvConfig,
        ids: Array | None,
        state: EnvironmentState | None,
        reset_mask: Array | None,
        initial: InitialSnapshot | None,
        origin: Array,
    ) -> tuple[Observations, EnvironmentState]:
        """Allocate checked IDs and merge fresh state into selected lanes only.

        Host reset has resolved the exact configuration and structural inputs.
        The outer compiled reset skips this function when no lane is selected.
        Every state leaf retains the native prefix; high-water marks are updated
        only in selected lanes so continuing games remain byte-for-byte equal.
        """
        count = self.num_envs or 1
        shape = () if self.num_envs is None else (count,)
        offsets = (
            jnp.asarray(1, jnp.int32)
            if self.num_envs is None
            else jnp.arange(count, dtype=jnp.int32) + 1
        )
        error = jnp.zeros(shape, jnp.bool_)
        high_water = (
            jnp.asarray(0, jnp.int32)
            if state is None
            else jnp.max(state.last_reserved_episode_id)
        )
        if ids is None:
            reserved, overflow = _checked_increment(high_water, count)
            ids = high_water + jnp.where(overflow, 0, offsets)
            if state is not None:
                ids = jnp.where(overflow, state.episode_id, ids)
            error = error | overflow
            high_water = reserved
        else:
            valid = (ids > 0) & (ids <= _MAX_LIFECYCLE_COUNT)
            error = error | ~valid
            fallback = offsets if state is None else state.episode_id
            ids = jnp.where(valid, ids, fallback).astype(jnp.int32)
            high_water = jnp.maximum(high_water, jnp.max(ids))
        if self.num_envs is None:
            fresh = (
                self._reset_one(key, config, ids)
                if initial is None
                else self._initialize_one(config, ids, *initial)
            )
        else:
            keys = _episode_keys(key, ids)
            fresh = (
                jax.vmap(self._reset_one)(keys, config, ids)
                if initial is None
                else jax.vmap(self._initialize_one)(config, ids, *initial)
            )
        generation = fresh.reset_generation
        cumulative = fresh.cumulative_transition_count
        if state is not None:
            generation, exhausted = _checked_increment(state.reset_generation, 1)
            error = error | state.lifecycle_error | exhausted
            cumulative = state.cumulative_transition_count
        config_origin = jnp.where(
            (origin == RESET_OVERRIDE) | (origin == RESET_AUTHORED),
            generation,
            fresh.config_origin_generation
            if state is None
            else state.config_origin_generation,
        )
        fresh = fresh._replace(
            reset_generation=generation,
            reset_origin=jnp.broadcast_to(origin, shape),
            authored_start=jnp.broadcast_to(origin == RESET_AUTHORED, shape),
            config_origin_generation=config_origin,
            cumulative_transition_count=cumulative,
            lifecycle_error=error,
            last_reserved_episode_id=jnp.broadcast_to(high_water, shape),
        )
        if state is not None and reset_mask is not None:

            def replace(new: Array, old: Array) -> Array:
                """Choose fresh values in reset lanes and retain all other lane values.

                new and old have identical shapes. Broadcast the reset mask over later
                dimensions so every leaf uses the same lane choice.
                """
                return jnp.where(
                    reset_mask.reshape(
                        (*reset_mask.shape, *((1,) * (new.ndim - reset_mask.ndim)))
                    ),
                    new,
                    old,
                )

            fresh = jax.tree.map(replace, fresh, state)
        return self.get_observations(fresh), fresh

    def get_observations(self, state: EnvironmentState) -> Observations:
        """Read the same compact inputs that automatic replay capture records.

        Parameters
        ----------
        state : EnvironmentState
            The paired scalar/native state for the current decision.

        Returns
        -------
        Observations
            Observations holding the existing ``observation`` tree and Boolean
            ``source_availability`` matrix; no expanded SharedObs copies are made.

        Notes
        -----
        For custom source subsets, replace ``state.source_availability`` before
        calling this getter and choosing actions. Use a Boolean matrix shaped
        ``(..., 10, 10)``. Only active same-team sources other than self may be true.
        Step preserves that choice; reset restores the default in reset lanes.
        """
        return Observations(state.observation, state.source_availability)

    def _step_one(
        self, key: Array, state: EnvironmentState, actions: Action
    ) -> tuple[EnvironmentState, Reward, EpisodeInfo, Info]:
        """Delegate one scalar game to Core, then update wrapper-owned counters.

        Already-finished games use Core's canonical no-transition facts. Count
        advancement only from the returned has_transition flag. Return facts for
        full metrics/replay without rebuilding simulator decisions outside Core.
        """

        def advance() -> CoreStepResult:
            """Pass one game's exact state, mask, submitted actions and key to Core."""
            return core.step(
                state.config, state.core_state, state.action_mask, actions, key
            )

        def pad() -> CoreStepResult:
            """Keep an already-finished game's snapshot and emit no-transition facts.

            Return zero reward, retained done flags and the unchanged observation/mask.
            Another call cannot count or reward the completed transition again.
            """
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
        advances, overflow = _checked_increment(
            state.cumulative_transition_count,
            info.transition_facts.has_transition.astype(jnp.int32),
        )
        successor = state._replace(
            core_state=core_state,
            observation=observation,
            action_mask=mask,
            done=done,
            priority=totals,
            episode_start=state.episode_start & ~info.transition_facts.has_transition,
            cumulative_transition_count=advances,
            lifecycle_error=state.lifecycle_error | overflow,
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
        """Update selected full-metric counters and finalize a completed scalar game.

        state is the pre-transition wrapper state. result and facts describe that
        same transition. Full and priority collection must be present. Return the
        updated FullTotals and MetricValues; unfinished games get zero values with
        all validity flags False. Core facts remain the measurement authority.
        """
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
        """Collect full metrics through the route selected at environment setup.

        state is the pre-transition snapshot; successor, result and facts come from
        that same step. Return successor and result with selected full counters/values
        attached. If full collection is absent, return them unchanged.

        The full native route batches numerical updates. Sparse selection iterates
        only the selected lane count, bounded by the configured capture capacity.
        Only completed games expose valid final values. No file or report is written.
        """
        if state.full is None:
            return successor, result
        shape = (*state.episode_id.shape, len(FULL_METRIC_NAMES))
        empty = MetricValues(jnp.zeros(shape, jnp.float32), jnp.zeros(shape, bool))
        selected = state.collect_full_metrics & ~state.done.done
        if self.num_envs is None:

            def collect() -> tuple[FullTotals, MetricValues]:
                """Update full metrics for the selected scalar game."""
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
                """Finalize native full values and clear validity for unfinished games.

                The enclosing call supplies updated totals and matched priority values.
                No unfinished row may be treated as a completed episode report.
                """
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
            capacity = min(self.num_envs, self._execution.full_capacity)
            indices = jnp.nonzero(selected, size=capacity, fill_value=0)[0]

            def collect_lane(
                offset: Array, carry: tuple[FullTotals, MetricValues]
            ) -> tuple[FullTotals, MetricValues]:
                """Update one selected lane in the sparse full-metric accumulator.

                offset selects a lane from the bounded index array. carry holds
                batched totals and output values. Read that lane's paired inputs,
                then replace the same lane in the returned carry.
                """
                lane = indices[offset]

                def read(value: Array) -> Array:
                    """Read the selected lane from one numerical leaf."""
                    return value[lane]

                lane_totals, lane_values = self._full_one(
                    jax.tree.map(read, state),
                    jax.tree.map(read, result),
                    jax.tree.map(read, facts),
                )

                def write(batch: Array, value: Array) -> Array:
                    """Replace one lane while retaining all other lanes."""
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
        """Take one game step, retaining terminal games until an explicit reset.

        Parameters
        ----------
        key : Array
            A JAX typed/legacy key or native per-lane keys. A native root is
            folded with each episode ID; explicit lane keys pass unchanged.
        state : EnvironmentState
            The latest EnvironmentState, including its exact config and
            the mask paired with the observations used to choose actions.
        actions : Action
            Core Action with three int32 arrays, each ``(10,)`` or
            ``(B, 10)``. Use sample_actions or the current joint masks for
            legal choices. Core owns validation and rejected-action behavior.

        Returns
        -------
        tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]
            ``(observations, next_state, reward, done, info)``. Observations is
            the next compact Observations tree. Reward.rewards is float32 with
            shape ``(10,)`` or ``(B, 10)``. DoneFlags has scalar or ``(B,)``
            termination/truncation leaves. EpisodeInfo holds completion flags
            and selected metric/replay data; its fields are privileged.

        Notes
        -----
        A real terminal transition counts once and keeps its terminal output.
        Further steps on that finished lane return zero reward and no new
        transition/completion until reset. The cumulative counter follows Core's
        has_transition fact, not the number of calls. No game resets or files
        are written automatically. This pure numerical route supports jit/vmap/
        scan; use valid fixed-shape inputs and do not mix decision epochs.
        """
        return cast(
            tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo],
            _step_environment(self, key, state, actions),
        )

    def _step_numerical(
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
        if self._execution.replay_capacity:
            info = info._replace(
                replay=capture_packets(
                    state,
                    successor,
                    reward,
                    facts,
                    capacity=min(self.num_envs or 1, self._execution.replay_capacity),
                )
            )
        return self.get_observations(successor), successor, reward, successor.done, info


def _sample_one(key: Array, observation: Observation, mask: ActionMask) -> Action:
    """Apply existing actor sampling once per slot, then assemble the two teams."""
    keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        key, jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.uint32)
    )
    first = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation, mask, keys, random_policy, TEAM_A_ID
        ),
    )
    second = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation, mask, keys, random_policy, TEAM_B_ID
        ),
    )
    return build_joint_action_from_actor_actions(first, second)


@jax.jit
def _sample_actions(env: Environment, key: Array, state: EnvironmentState) -> Action:
    """Share one compiled sampler across ready handles with equal structure."""
    if env.num_envs is None:
        return _sample_one(key, state.observation, state.action_mask)
    keys = _episode_keys(key, state.episode_id)
    return jax.vmap(_sample_one)(keys, state.observation, state.action_mask)


@jax.jit
def _reset_environment(
    env: Environment,
    key: Array,
    config: EnvConfig,
    ids: Array | None,
    state: EnvironmentState | None,
    reset_mask: Array | None,
    initial: InitialSnapshot | None,
    origin: Array,
) -> tuple[Observations, EnvironmentState]:
    """Compile exact reset data and skip initialization when no lane is selected."""

    def initialize() -> tuple[Observations, EnvironmentState]:
        """Apply the already resolved reset inputs through the shared numerical path."""
        return env._reset_numerical(  # pyright: ignore[reportPrivateUsage]
            key, config, ids, state, reset_mask, initial, origin
        )

    if state is None:
        return initialize()
    assert reset_mask is not None
    # Finished-lane checks are common. Skip all initialization when none finished.
    return cast(
        tuple[Observations, EnvironmentState],
        jax.lax.cond(
            jnp.any(reset_mask),
            initialize,
            lambda: (env.get_observations(state), state),
        ),
    )


@jax.jit
def _step_environment(
    env: Environment, key: Array, state: EnvironmentState, actions: Action
) -> tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]:
    """Keep ready configuration values out of the step program's static key."""
    return env._step_numerical(key, state, actions)  # pyright: ignore[reportPrivateUsage]


def make(
    task: str,
    *,
    map_id: int | None = None,
    env_config: EnvConfig | None = None,
    team_a_roster: Sequence[AgentClassName] | None = None,
    team_b_roster: Sequence[AgentClassName] | None = None,
    score_threshold: int = 20,
    max_steps: int = 300,
    num_envs: int | None = None,
    balance_spawn_locations: bool = True,
    metrics: MetricMode = "priority",
    full_metrics_episodes: Iterable[int] = (),
    replay_episodes: Iterable[int] = (),
) -> Environment:
    """Prepare a map-based environment, an exact config, or an unconfigured handle.

    Parameters
    ----------
    task : str
        Supported task name, currently ``"tdm"``.
    map_id : int | None, default=None
        Public map ID from list_tdm_maps. Omission with no env_config
        makes an unconfigured handle that needs an exact config at reset.
    env_config : EnvConfig | None, default=None
        Exact EnvConfig or ready batch. Scalar values broadcast to
        num_envs; batched leaves must all match it. Bypasses automatic spawn
        balance. Cannot accompany map_id, rosters or nondefault game rules.
    team_a_roster : Sequence[AgentClassName] | None, default=None
        Ordered class names for Team A in map-based setup. Defaults
        to the first canonical_tournament_rosters entry. One to five agents;
        repeated classes are distinct agents. Supplied order fills first slots.
    team_b_roster : Sequence[AgentClassName] | None, default=None
        Same rule for Team B; defaults to the second canonical entry.
    score_threshold : int, default=20
        Map-based TDM winning score. Exact Python int in 1..16,777,212;
        default 20.
    max_steps : int, default=300
        Map-based episode limit in transitions. Exact Python int in
        1..16,777,216; default 300.
    num_envs : int | None, default=None
        Exact Python int B in 1..2,147,483,647, or None for scalar execution.
        This integer bound is not a memory-capacity promise. Batch size is
        structural and changes the compiled program shape.
    balance_spawn_locations : bool, default=True
        Default True. Map-based preparation requires a
        positive even B: first half keeps the source banks, second half
        exchanges both complete banks. Set False for scalar/odd map batches.
        Exact configs and unconfigured handles do not require an opt-out.
    metrics : MetricMode, default="priority"
        ``"priority"`` by default, ``"full"`` for full collection, or
        ``"none"`` to skip unselected scalar metric work. Schemas are unchanged.
    full_metrics_episodes : Iterable[int], default=()
        Optional positive int32 episode IDs selected for
        full metrics, even under priority/none. Duplicates are removed.
    replay_episodes : Iterable[int], default=()
        Optional positive int32 IDs selected for replay capture.
        Capture is independent of metric mode and writes no files itself.

    Returns
    -------
    Environment
        An immutable Environment. Map/exact-config handles are ready for
        ``env.reset(key)``. No game has started and no hidden RNG is allocated.

    Raises
    ------
    ValueError
        Unsupported task/mode, invalid batch or selection IDs,
        conflicting setup inputs, or invalid map/roster/config values.
    TypeError
        Unsupported config/roster structure or a non-Boolean balance
        setting. Physical validation uses the existing Core validators.

    Notes
    -----
    This is host setup. Compiled reset/step consume numerical config data; do not
    recreate factories inside a rollout. Spawn exchange preserves every other
    config field, including teams, slots, world directions and action meanings.
    It prepares training conditions; equal real training steps still depend on
    advancing both batch halves equally. Resets retain exact resolved banks,
    including later respawn locations, rather than exchanging them a second time.

    Examples
    --------
    >>> import jax
    >>> import marl_battlegrounds as marl_bgs
    >>> env = marl_bgs.make("tdm", map_id=0, num_envs=128)
    >>> obs, state = env.reset(jax.random.key(0))
    >>> actions = env.sample_actions(jax.random.key(1), state)
    >>> obs, state, reward, done, info = env.step(jax.random.key(2), state, actions)
    """
    if task != "tdm":
        raise ValueError("the supported task is 'tdm'")
    if num_envs is not None and (
        type(num_envs) is not int or not 1 <= num_envs <= np.iinfo(np.int32).max
    ):
        raise ValueError("num_envs must be a positive integer or omitted")
    if metrics not in ("none", "priority", "full"):
        raise ValueError("metrics must be 'none', 'priority', or 'full'")
    if type(balance_spawn_locations) is not bool:
        raise TypeError("balance_spawn_locations must be a boolean")
    has_rules = (
        team_a_roster is not None
        or team_b_roster is not None
        or score_threshold != 20
        or max_steps != 300
    )
    if env_config is not None and (map_id is not None or has_rules):
        raise ValueError(
            "env_config cannot be combined with map, roster or rule inputs"
        )
    if map_id is None and env_config is None and has_rules:
        raise ValueError("roster and rule inputs require map_id")
    source = None
    defaults = None
    if map_id is not None:
        rosters = canonical_tournament_rosters()
        source = make_standard_team_deathmatch_config(
            map_id=map_id,
            team_a_roster=rosters[0] if team_a_roster is None else team_a_roster,
            team_b_roster=rosters[1] if team_b_roster is None else team_b_roster,
            score_threshold=score_threshold,
            max_steps=max_steps,
        )
        if balance_spawn_locations:
            if num_envs is None:
                raise ValueError(
                    "automatic spawn balance requires a positive even batch"
                )
            defaults = balanced_spawn_configs(source, num_envs=num_envs)
        else:
            defaults = prepare_exact_env_config(source, num_envs=num_envs)
        source = jax.tree.map(_config_array, source)
    elif env_config is not None:
        defaults = prepare_exact_env_config(env_config, num_envs=num_envs)
    full_ids = _episode_selection(full_metrics_episodes, "full_metrics_episodes")
    replay_ids = _episode_selection(replay_episodes, "replay_episodes")
    # Keep setup selections on the host for metadata. JAX transfers these
    # dynamic leaves only when a compiled reset needs them.
    full_array = np.asarray(full_ids, np.int32)
    replay_array = np.asarray(replay_ids, np.int32)
    full_array.flags.writeable = False
    replay_array.flags.writeable = False
    return Environment(
        _Execution(num_envs, metrics, len(full_ids), len(replay_ids)),
        full_array,
        replay_array,
        defaults,
        source,
        None if map_id is None else jnp.asarray(map_id, jnp.int32),
    )
