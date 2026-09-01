"""Fixed-shape M7 reference rollout for one homogeneous information regime.

Each epoch key splits into disjoint environment and ten globally ordered actor
keys. A real row stores ``s_t / m_t / a_t`` beside the complete successor
transition. After ``done_t``, canonical rows repeat the terminal successor with
``Info.transition_facts.has_transition == False``; that leaf is the public
validity vector and its sum is the number of real transitions.
"""

from annotationlib import Format
from inspect import Parameter, signature
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.env import build_canonical_no_transition_info_object, step
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    NUM_TEAMS,
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
from marl_battlegrounds.evaluation.models import ExecutionInformationMode
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.no_shared_obs import (
    NoSharedObsPolicy,
    execute_no_shared_obs_team_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsPolicy,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)

type ReferenceTeamPolicy = NoSharedObsPolicy | SharedObsPolicy
type ReferenceSuccessorHistory = tuple[
    EnvState,
    Observation,
    Reward,
    DoneFlags,
    ActionMask,
    Info,
]
type ReferenceCurrentHistory = tuple[EnvState, ActionMask, Action]


class ReferenceRolloutResult(NamedTuple):
    """Fixed histories plus the exact information topology consumed at runtime."""

    successors: ReferenceSuccessorHistory
    currents: ReferenceCurrentHistory
    information_availability: Array | None


def _require_scalar_policy_abi(
    policy: ReferenceTeamPolicy,
    execution_information_mode: ExecutionInformationMode,
    *,
    name: str,
) -> None:
    """Reject the opposite scalar ABI before entering JAX compilation."""
    expected_positional_count = 6 if execution_information_mode == "shared_obs" else 3
    try:
        parameters = tuple(
            signature(policy, annotation_format=Format.STRING).parameters.values()
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must expose an inspectable scalar policy ABI"
        ) from error
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_variadic_positionals = any(
        parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters
    )
    required_keyword_only = tuple(
        parameter
        for parameter in parameters
        if parameter.kind == Parameter.KEYWORD_ONLY
        and parameter.default is Parameter.empty
    )
    if (
        has_variadic_positionals
        or required_keyword_only
        or len(positional) != expected_positional_count
    ):
        raise TypeError(
            f"{name} does not implement the {execution_information_mode} scalar "
            f"policy ABI ({expected_positional_count} positional arguments)"
        )


def build_rollout_information_availability(
    config: EnvConfig,
    execution_information_mode: ExecutionInformationMode,
) -> Array | None:
    """Return the exact episode-wide availability consumed by SharedObs.

    NoSharedObs returns ``None`` and therefore has no source-bank or matrix
    materialization contract. Callers pass this same SharedObs matrix to frame
    capture rather than materializing or storing a second actor-input tensor.
    """
    if execution_information_mode == "shared_obs":
        return build_default_shared_obs_information_availability(
            config.agent_profile.active_mask,
            config.agent_profile.team_ids,
        )
    if execution_information_mode == "no_shared_obs":
        return None
    raise ValueError(
        "execution_information_mode must be 'shared_obs' or 'no_shared_obs'"
    )


def rollout(
    config: EnvConfig,
    initial_state: EnvState,
    initial_observation: Observation,
    initial_action_mask: ActionMask,
    environment_root_key: Array,
    team_a_policy: ReferenceTeamPolicy,
    team_b_policy: ReferenceTeamPolicy,
    *,
    execution_information_mode: ExecutionInformationMode,
) -> ReferenceRolloutResult:
    """Run one bounded episode and return fixed-length transition history.

    ``initial_observation`` and ``initial_action_mask`` must describe
    ``initial_state`` at the same decision epoch. The two policies receive
    separate fixed five-slot batches and choose all ten actions before the
    environment advances. The returned first tuple contains successor data;
    the second contains the corresponding current state, mask, and submitted
    action. The information mode is one required static episode contract for
    both teams; high-level launch surfaces choose ``shared_obs`` by default.
    """
    if execution_information_mode not in ("shared_obs", "no_shared_obs"):
        raise ValueError(
            "execution_information_mode must be 'shared_obs' or 'no_shared_obs'"
        )
    _require_scalar_policy_abi(
        team_a_policy,
        execution_information_mode,
        name="team_a_policy",
    )
    _require_scalar_policy_abi(
        team_b_policy,
        execution_information_mode,
        name="team_b_policy",
    )
    return cast(
        ReferenceRolloutResult,
        _rollout_jit(
            config,
            initial_state,
            initial_observation,
            initial_action_mask,
            environment_root_key,
            team_a_policy,
            team_b_policy,
            execution_information_mode,
            config.max_steps,
        ),
    )


@jax.jit(static_argnums=(5, 6, 7, 8))
def _rollout_jit(
    config: EnvConfig,
    initial_state: EnvState,
    initial_observation: Observation,
    initial_action_mask: ActionMask,
    environment_root_key: Array,
    team_a_policy: ReferenceTeamPolicy,
    team_b_policy: ReferenceTeamPolicy,
    execution_information_mode: ExecutionInformationMode,
    max_steps: int,
) -> ReferenceRolloutResult:
    """Compile one static-horizon scan while keeping configuration data dynamic."""
    shared_information_availability: Array | None
    if execution_information_mode == "shared_obs":
        shared_information_availability = (
            build_default_shared_obs_information_availability(
                config.agent_profile.active_mask,
                config.agent_profile.team_ids,
            )
        )
    else:
        shared_information_availability = None

    def _scanned_rollout(
        carry: tuple[EnvState, Observation, ActionMask, Array], episode_key: Array
    ) -> tuple[
        tuple[EnvState, Observation, ActionMask, Array],
        tuple[
            tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
            tuple[EnvState, ActionMask, Action],
        ],
    ]:
        """Advance one real transition or emit one canonical padding row."""

        def _perform_scanned_rollout(
            current_state: EnvState,
            current_observation: Observation,
            current_action_mask: ActionMask,
            team_keys: Array,
            step_key: Array,
        ) -> tuple[
            tuple[EnvState, Observation, ActionMask, Array],
            tuple[
                tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
                tuple[EnvState, ActionMask, Action],
            ],
        ]:
            """Choose all actor actions, assemble them, and advance the core once."""

            if execution_information_mode == "shared_obs":
                source_bank = build_shared_obs_sensor_source_bank(current_observation)
                assert shared_information_availability is not None
                team_a_action = cast(
                    ActorAction,
                    execute_shared_obs_team_policy(
                        current_observation,
                        current_action_mask,
                        team_keys,
                        source_bank,
                        shared_information_availability,
                        policy=cast(SharedObsPolicy, team_a_policy),
                        team_identity=TEAM_A_ID,
                    ),
                )
                team_b_action = cast(
                    ActorAction,
                    execute_shared_obs_team_policy(
                        current_observation,
                        current_action_mask,
                        team_keys,
                        source_bank,
                        shared_information_availability,
                        policy=cast(SharedObsPolicy, team_b_policy),
                        team_identity=TEAM_B_ID,
                    ),
                )
            else:
                team_a_action = cast(
                    ActorAction,
                    execute_no_shared_obs_team_policy(
                        current_observation,
                        current_action_mask,
                        team_keys,
                        policy=cast(NoSharedObsPolicy, team_a_policy),
                        team_identity=TEAM_A_ID,
                    ),
                )
                team_b_action = cast(
                    ActorAction,
                    execute_no_shared_obs_team_policy(
                        current_observation,
                        current_action_mask,
                        team_keys,
                        policy=cast(NoSharedObsPolicy, team_b_policy),
                        team_identity=TEAM_B_ID,
                    ),
                )

            current_joint_action = build_joint_action_from_actor_actions(
                team_a_action, team_b_action
            )

            (
                next_state,
                next_observation,
                rewards,
                done_flags,
                next_action_mask,
                info,
            ) = step(
                config,
                current_state,
                current_action_mask,
                current_joint_action,
                step_key,
            )

            return (
                (next_state, next_observation, next_action_mask, done_flags.done),
                (
                    (
                        next_state,
                        next_observation,
                        rewards,
                        done_flags,
                        next_action_mask,
                        info,
                    ),
                    (current_state, current_action_mask, current_joint_action),
                ),
            )

        def _episode_is_done(
            current_state: EnvState,
            current_observation: Observation,
            current_action_mask: ActionMask,
            team_keys: Array,
            step_key: Array,
        ) -> tuple[
            tuple[EnvState, Observation, ActionMask, Array],
            tuple[
                tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
                tuple[EnvState, ActionMask, Action],
            ],
        ]:
            """Emit an invalid padding row without invoking policy or core step.

            Padding done flags describe the retained terminal successor only;
            ``Info.transition_facts.has_transition`` remains the row-validity
            authority.
            """
            del team_keys, step_key

            episode_is_done = jnp.asarray(True, dtype=jnp.bool_)

            padding_action_values = jnp.zeros(
                (MAX_AGENT_SLOTS,),
                dtype=jnp.int32,
            )
            padding_action = Action(
                move=padding_action_values,
                select_target=padding_action_values,
                use_ultimate=padding_action_values,
            )

            padding_reward = Reward(
                rewards=jnp.zeros(
                    (MAX_AGENT_SLOTS,),
                    dtype=jnp.float32,
                )
            )
            padding_done_flags = DoneFlags(
                terminated=episode_is_done,
                truncated=(current_state.step_count >= max_steps),
            )

            return (
                (
                    current_state,
                    current_observation,
                    current_action_mask,
                    episode_is_done,
                ),
                (
                    (
                        current_state,
                        current_observation,
                        padding_reward,
                        padding_done_flags,
                        current_action_mask,
                        build_canonical_no_transition_info_object(current_state),
                    ),
                    (
                        current_state,
                        current_action_mask,
                        padding_action,
                    ),
                ),
            )

        (
            current_state,
            current_observation,
            current_action_mask,
            episode_is_done,
        ) = carry

        # Splitting remains unconditional so every scan position has one stable key.
        step_key, team_key = jax.random.split(episode_key)
        team_keys = jax.random.split(team_key, num=NUM_TEAMS * MAX_AGENTS_PER_TEAM)

        return cast(
            tuple[
                tuple[EnvState, Observation, ActionMask, Array],
                tuple[
                    tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
                    tuple[EnvState, ActionMask, Action],
                ],
            ],
            jax.lax.cond(
                ~episode_is_done,
                _perform_scanned_rollout,
                _episode_is_done,
                current_state,
                current_observation,
                current_action_mask,
                team_keys,
                step_key,
            ),
        )

    episode_keys = jax.random.split(environment_root_key, num=max_steps)

    _, episode_history = jax.lax.scan(
        _scanned_rollout,
        (
            initial_state,
            initial_observation,
            initial_action_mask,
            jnp.asarray(False, dtype=jnp.bool_),
        ),
        episode_keys,
    )

    successors, currents = episode_history
    return ReferenceRolloutResult(
        successors=successors,
        currents=currents,
        information_availability=shared_information_availability,
    )


__all__ = (
    "ReferenceRolloutResult",
    "ReferenceTeamPolicy",
    "build_rollout_information_availability",
    "rollout",
)
