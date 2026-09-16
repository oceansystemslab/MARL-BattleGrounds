"""Reset finished games while retaining their exact final learning inputs.

``AutoReset`` is an optional, immutable handle around the ordinary Environment.
Its step uses the existing simulator and reset paths. Returned observations belong
to the next decision; ``info.final`` keeps the old episode's compact final view.
No policy is called and no files are written. Use explicit resets when a curriculum
must choose the next configuration before starting a replacement game.
"""

from dataclasses import dataclass, field
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Reward,
)
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    EpisodeInfo,
    InitialSnapshot,
    MetricMode,
)
from marl_battlegrounds.evaluation.policy_execution import (
    SystemInput,
    _batched,  # pyright: ignore[reportPrivateUsage]
    _prepare_system_inputs,  # pyright: ignore[reportPrivateUsage]
    _team_index,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.spaces import StructuredSpace
from marl_battlegrounds.tasks import TDMScenario

_AUTO_RESET_DOMAIN = 0x4155544F


class FinalEpisodeData(NamedTuple):
    """Keep one post-transition, pre-reset snapshot for completed lanes.

    All leaves are dynamic JAX data. ``L`` is empty for scalar games or ``(B,)``
    for native batches. Only lanes where ``valid`` is true are final learning
    inputs. Other rows may hold the real pre-reset snapshot and must be ignored.
    This record contains no metric history, expanded actor bank or System memory.

    Attributes
    ----------
    observations : Observations
        Exact compact observations and source permissions after the old step.
    action_mask : ActionMask
        Core's matching Boolean legal-choice masks, with prefix L.
    active_mask : Array
        Boolean L + (10,) roster participation, including dead active agents.
    episode_id : Array
        Int32 L IDs of the old episodes that produced these snapshots.
    done : DoneFlags
        Old termination and truncation flags; no reset flags are substituted.
    valid : Array
        Boolean L mask, true only for an episode completed by this real step.
    training_state : EnvState or None
        Privileged old Core state when explicitly requested; None by default.
        It is for authorized training use only and never enters actor inputs.

    Notes
    -----
    Keep this data on the device when computing learning values. The writer
    excludes it from host recording transfers and serialization. Consumers choose
    their own termination/truncation learning rule; this record defines no target.
    """

    observations: Observations
    action_mask: ActionMask
    active_mask: Array
    episode_id: Array
    done: DoneFlags
    valid: Array
    training_state: EnvState | None


def _reset_key(key: Array) -> Array:
    """Tag one root key or each supplied lane key for automatic resets.

    Preserve typed/legacy key representation. Base step has already checked the
    supported root/per-lane shape. The fixed AUTO tag leaves action/step draws
    unchanged and makes reset draws independent of the number of finished lanes.
    """
    if jax.random.key_data(key).ndim == 1:
        return jax.random.fold_in(key, _AUTO_RESET_DOMAIN)
    return jax.vmap(jax.random.fold_in, in_axes=(0, None))(key, _AUTO_RESET_DOMAIN)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class AutoReset:
    """Return new games after terminal steps, with old final data in info.

    Parameters
    ----------
    env : Environment
        One base environment. Nested AutoReset handles are rejected. Its prepared
        configuration and selections remain dynamic JAX data.
    include_training_state : bool, default=False
        Also retain final privileged Core state for authorized training use.
        This choice is static compiled structure; only a Python bool is accepted.

    Raises
    ------
    TypeError
        env is not a base Environment or include_training_state is not a bool.

    Notes
    -----
    Carry the returned EnvironmentState into the next decision. No hidden mutable
    state or policy call is added. ``step`` alone changes reset timing; explicit
    reset calls keep the base signatures and exact-config behavior. A completion
    changes the reset generation, so apply_systems refreshes memory on the next
    decision. Agent death or respawn alone does not reset episode memory.
    The wrapper works inside jit/vmap/scan. No per-step host validation, callback,
    file write or expanded actor-input history is added.
    """

    env: Environment
    include_training_state: bool = field(default=False, metadata={"static": True})

    def __post_init__(self) -> None:
        """Reject invalid wrapper structure without reading numerical state."""
        if not isinstance(cast(object, self.env), Environment):
            raise TypeError(
                "AutoReset needs one base Environment; nesting is unsupported"
            )
        if type(self.include_training_state) is not bool:
            raise TypeError("include_training_state must be a bool")

    @property
    def num_envs(self) -> int | None:
        """Return the native batch size, or None for scalar calls."""
        return self.env.num_envs

    @property
    def metrics(self) -> MetricMode:
        """Return the base default metric mode: priority, full or none."""
        return self.env.metrics

    @property
    def full_metrics_episodes(self) -> tuple[int, ...]:
        """Read selected full-report IDs on the host; not a compiled accessor."""
        return self.env.full_metrics_episodes

    @property
    def replay_episodes(self) -> tuple[int, ...]:
        """Read selected replay IDs on the host; not a compiled accessor."""
        return self.env.replay_episodes

    @property
    def agents(self) -> tuple[str, ...]:
        """Return all ten capacity names; activity is supplied by agent_info."""
        return self.env.agents

    @property
    def num_agents(self) -> int:
        """Return ten capacity slots, including inactive and dead agents."""
        return self.env.num_agents

    def action_space(self, agent: str) -> StructuredSpace:
        """Describe one named slot's structural actions, not current legality.

        agent must be one of agents or the base method raises ValueError. Return
        its cached host StructuredSpace; get_action_mask owns current legality.
        """
        return self.env.action_space(agent)

    def observation_space(self, agent: str) -> StructuredSpace:
        """Describe one named slot's compact observation shapes and dtypes.

        agent must be one of agents or the base method raises ValueError. Return
        the cached host StructuredSpace without environment or actor batch axes.
        """
        return self.env.observation_space(agent)

    def agent_info(self, state: EnvironmentState) -> dict[str, Array]:
        """Read privileged class IDs, team IDs and activity from current state.

        Return arrays shaped (10,) or (B, 10), preserving device placement. This
        routing/analysis data does not grant an actor extra information rights.
        """
        return self.env.agent_info(state)

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
        """Forward an exact explicit reset without changing its key or arguments.

        Parameters
        ----------
        key : Array
            Typed/legacy root key; native batches also accept one key per lane.
            Unlike automatic resets, this explicit call adds no AUTO domain tag.
        env_config : EnvConfig or None, default=None
            Exact scalar/native config. It overrides retained state.config, then
            constructor defaults. Spawn banks are never exchanged during reset.
        episode_id : int, Array or None, default=None
            Explicit positive int32 IDs, scalar or (B,), unique within a batch.
            Omission allocates new IDs through the base reset authority.
        state : EnvironmentState or None, default=None
            Latest state; None starts an independent context. Existing cumulative
            counters, allocation history and errors survive selected resets.
        reset_mask : Array or None, default=None
            Boolean scalar/(B,) selecting replacement lanes. Requires state.
            With state and no mask, all lanes reset. Others remain unchanged.
        initial : InitialSnapshot or None, default=None
            Exact paired Core state, observation and mask. Must match the config
            and scalar/native shapes; None uses Core's ordinary initialization.
        scenario : TDMScenario or None, default=None
            Host-prepared authored start; cannot accompany env_config or initial.
            Scenario preparation must happen outside compiled execution.

        Returns
        -------
        tuple[Observations, EnvironmentState]
            Resulting observations and latest state. No transition is counted.

        Raises
        ------
        TypeError, ValueError
            Base reset rejects invalid configurations, shapes, IDs or conflicting
            start arguments. Compiled numerical failures set lifecycle_error.

        Notes
        -----
        This is the base reset implementation. See Environment.reset for the full
        exact-start and checked-int32 contract. No input changes in place.
        """
        return self.env.reset(
            key,
            env_config,
            episode_id=episode_id,
            state=state,
            reset_mask=reset_mask,
            initial=initial,
            scenario=scenario,
        )

    def reset_done(
        self, key: Array, state: EnvironmentState, env_config: EnvConfig | None = None
    ) -> tuple[Observations, EnvironmentState]:
        """Explicitly reset only finished lanes through the base reset authority.

        key is a typed/legacy root or native per-lane key and is forwarded without
        an AUTO tag. state is the latest context. Optional exact env_config
        replaces finished lanes; None keeps state.config. Return new observations
        and state, preserving continuing rows. Base validation errors propagate.
        No real transition occurs and no file is written.
        """
        return self.env.reset_done(key, state, env_config)

    def policy_inputs(
        self, observations: Observations, state: EnvironmentState, team: int = 0
    ) -> SystemInput:
        """Prepare the chosen team's permitted inputs for the current decision.

        observations/state must describe one epoch. team is integer 0 or 1,
        default 0; bool or other values raise ValueError. Return SystemInput with
        leading (B,5) actor/mask axes; scalar games become B=1. Source permissions
        remain restricted and actors' private data stays separate. Terminal
        learning views instead use final_policy_inputs. No host copy is made.
        """
        return self.env.policy_inputs(observations, state, team)

    def join_actions(self, a: ActorAction, b: ActorAction) -> Action:
        """Join Team A/B int32 heads in fixed slot order without changing choices.

        a/b have five actor slots, shaped (5,) or (1,5) for scalar execution and
        (B,5) for native batches. Return (10,) or (B,10) heads. Base assembly
        raises TypeError for wrong records, shapes or dtypes. Core accepts values.
        """
        return self.env.join_actions(a, b)

    def training_state(self, state: EnvironmentState) -> EnvState:
        """Return current privileged Core state for separately authorized learning.

        state is the latest scalar/native context. Return its exact core_state
        object with no copy. It must not influence actor actions through an
        unauthorized route. Completed old state is in info.final only if enabled.
        """
        return self.env.training_state(state)

    def sample_actions(self, key: Array, state: EnvironmentState) -> Action:
        """Sample legal coupled choices from the current state's exact masks.

        key is a fresh typed/legacy root or native per-lane key. Return int32
        Action heads shaped (10,) or (B,10), including inert inactive/dead slots.
        Preserve the base RNG schedule and reject its invalid key shapes. This
        explicit call samples; AutoReset.step itself never calls a policy.
        """
        return self.env.sample_actions(key, state)

    def get_observations(self, state: EnvironmentState) -> Observations:
        """Return current compact observations and exact source permissions.

        state must be the paired scalar/native context. Return its existing
        arrays without expanding actor banks or moving data to the host.
        """
        return self.env.get_observations(state)

    def get_action_mask(self, state: EnvironmentState) -> ActionMask:
        """Return the exact current scalar/native action masks without copying.

        state must match the observations used to choose actions. The joint
        target/Ultimate mask defines coupled legality; shapes stay unchanged.
        """
        return self.env.get_action_mask(state)

    def step(
        self, key: Array, state: EnvironmentState, actions: Action
    ) -> tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]:
        """Advance once, then replace finished games without losing final inputs.

        Parameters
        ----------
        key : Array
            Fresh typed/legacy key, or native per-lane keys. Base step receives it
            unchanged. Automatic reset receives fold_in(key, 0x4155544F), applied
            per supplied lane; base reset still folds root keys by new episode ID.
        state : EnvironmentState
            Latest paired scalar/native state. Already-terminal lanes produce
            padding, may reset, and have final.valid=False.
        actions : Action
            Submitted int32 (10,) or (B,10) heads from this pre-step epoch. Core
            owns acceptance; AutoReset never changes or resamples the action.

        Returns
        -------
        tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]
            Fresh next-decision observations/state, followed by old-step reward,
            done and info. info.final preserves pre-reset inputs, with valid equal
            to old info.completed. Only training_state retention is optional.
            Reset failures join info.lifecycle_error as well as the returned state.

        Notes
        -----
        No extra transition or policy call occurs. The outer done can be true
        while returned state.done is false because they describe different games.
        Use final_policy_inputs for completed old games and returned observations
        for continuing/new games. System memory refreshes on the next decision.
        Base action/key errors propagate. Calls support jit/vmap/scan; numerical
        ID exhaustion sets a sticky lifecycle error. No files are written.
        """
        observations, successor, reward, done, info = self.env.step(key, state, actions)
        final = FinalEpisodeData(
            observations,
            successor.action_mask,
            successor.config.agent_profile.active_mask,
            info.episode_id,
            done,
            info.completed,
            successor.core_state if self.include_training_state else None,
        )
        observations, successor = self.env.reset_done(_reset_key(key), successor)
        return (
            observations,
            successor,
            reward,
            done,
            info._replace(
                final=final,
                lifecycle_error=info.lifecycle_error | successor.lifecycle_error,
            ),
        )

    def final_policy_inputs(self, info: EpisodeInfo, team: int = 0) -> SystemInput:
        """Prepare permitted final inputs for completed old games, without a call.

        Parameters
        ----------
        info : EpisodeInfo
            Info returned by this wrapper step. Its compact final observations,
            masks and roster must stay paired. No new policy action is chosen.
        team : int, default=0
            Team A=0 or Team B=1; bool and other values are rejected.

        Returns
        -------
        SystemInput
            Separate actor inputs/masks shaped (B,5,...), scalar normalized to B=1.
            valid is final.valid, even though old done is true. episode_start is
            false. Private inputs and source restrictions remain intact; optional
            privileged training state is never included.

        Raises
        ------
        ValueError
            team is invalid or info contains no final snapshot.

        Notes
        -----
        Invalid rows are padding for this final view. This helper defines neither
        a bootstrap target nor a termination/truncation rule. It builds only the
        requested team's temporary actor bank through the shared input authority.
        """
        team = _team_index(team)
        final = info.final
        if final is None:
            raise ValueError("final_policy_inputs needs info from an AutoReset step")
        native = final.episode_id.ndim != 0
        return cast(
            SystemInput,
            _prepare_system_inputs(
                _batched(final.observations, native),
                _batched(final.action_mask, native),
                _batched(final.active_mask, native),
                _batched(jnp.zeros_like(final.valid), native),
                _batched(final.valid, native),
                team=team,
            ),
        )
