"""Fixed-shape M7 reference rollout for two NoSharedObs team policies.

Each epoch key splits into disjoint environment and ten globally ordered actor
keys. A real row stores ``s_t / m_t / a_t`` beside the complete successor
transition. After ``done_t``, canonical rows repeat the terminal successor with
``Info.transition_facts.has_transition == False``; that leaf is the public
validity vector and its sum is the number of real transitions.
"""

from typing import cast

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
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.no_shared_obs import (
    NoSharedObsPolicy,
    execute_no_shared_obs_team_policy,
)


def rollout(
    config: EnvConfig,
    initial_state: EnvState,
    initial_observation: Observation,
    initial_action_mask: ActionMask,
    environment_root_key: Array,
    team_a_policy: NoSharedObsPolicy,
    team_b_policy: NoSharedObsPolicy,
) -> tuple[
    tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
    tuple[EnvState, ActionMask, Action],
]:
    """Run one bounded episode and return fixed-length transition history.

    ``initial_observation`` and ``initial_action_mask`` must describe
    ``initial_state`` at the same decision epoch. The two policies receive
    separate fixed five-slot batches and choose all ten actions before the
    environment advances. The returned first tuple contains successor data;
    the second contains the corresponding current state, mask, and submitted
    action.
    """
    return cast(
        tuple[
            tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
            tuple[EnvState, ActionMask, Action],
        ],
        _rollout_jit(
            config,
            initial_state,
            initial_observation,
            initial_action_mask,
            environment_root_key,
            team_a_policy,
            team_b_policy,
            config.max_steps,
        ),
    )


@jax.jit(static_argnums=(5, 6, 7))
def _rollout_jit(
    config: EnvConfig,
    initial_state: EnvState,
    initial_observation: Observation,
    initial_action_mask: ActionMask,
    environment_root_key: Array,
    team_a_policy: NoSharedObsPolicy,
    team_b_policy: NoSharedObsPolicy,
    max_steps: int,
) -> tuple[
    tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
    tuple[EnvState, ActionMask, Action],
]:
    """Compile one static-horizon scan while keeping configuration data dynamic."""

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

            team_a_action = cast(
                ActorAction,
                execute_no_shared_obs_team_policy(
                    current_observation,
                    current_action_mask,
                    team_keys,
                    policy=team_a_policy,
                    team_identity=TEAM_A_ID,
                ),
            )
            team_b_action = cast(
                ActorAction,
                execute_no_shared_obs_team_policy(
                    current_observation,
                    current_action_mask,
                    team_keys,
                    policy=team_b_policy,
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

    return episode_history
