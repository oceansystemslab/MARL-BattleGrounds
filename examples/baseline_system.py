"""Run an untrained recurrent MAPPO actor through the public environment API.

Install the training extra, then run ``python examples/baseline_system.py``.
Use JAX_PLATFORMS=cpu for correctness or JAX_PLATFORMS=cuda on the selected
GPU. The example uses 32 games, explicit episode resets and dynamic weights.
It writes no files and performs no learning or checkpoint selection.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines import ppo

type Tree = Any


def main() -> None:
    """Run sixteen decisions per game against the existing random opponent.

    Both teams keep their latest System memory. Finished games reset through
    the existing environment, so the next decision starts fresh episode memory.
    Reward totals are a wiring check; untrained actions establish no competence.
    """
    key, network_key, reset_key, memory_key = jax.random.split(jax.random.key(42), 4)
    actor_params = ppo.initialize_ppo(network_key).actor_params
    actor = ppo.make_recurrent_mappo_system(
        actor_params, name="Untrained Recurrent MAPPO"
    )
    opponent = marl_bgs.shared_policy(marl_bgs.policy("random"))
    env = marl_bgs.make("tdm", map_id=0, num_envs=32, max_steps=8, metrics="none")
    observations, state = env.reset(reset_key)
    memory = marl_bgs.init_systems(actor, opponent, observations, state, memory_key)

    @jax.jit
    def rollout(parameters: Tree, carry: Tree) -> tuple[Tree, Array]:
        """Carry games, keys and memory through sixteen compiled public steps.

        parameters contains only actor weights and remains a dynamic argument.
        carry holds the next key, observations, environment state and System
        memory. Return the latest carry and one total reward per decision.
        """

        def step(current: Tree, unused: None) -> tuple[Tree, Array]:
            """Choose once, apply one step and reset only finished games."""
            del unused
            rng, obs, before, methods = current
            rng, action_key, step_key, next_reset_key = jax.random.split(rng, 4)
            actions, methods, _ = marl_bgs.apply_systems(
                actor,
                opponent,
                methods,
                obs,
                before,
                action_key,
                variables_a=parameters,
            )
            _, after, reward, _, _ = env.step(step_key, before, actions)
            obs, after = env.reset_done(next_reset_key, after)
            return (rng, obs, after, methods), jnp.sum(reward.rewards)

        return jax.lax.scan(step, carry, None, length=16)

    _, rewards = cast(
        tuple[Tree, Array],
        rollout(actor_params, (key, observations, state, memory)),
    )
    jax.block_until_ready(rewards)
    print("Completed 512 environment transitions with an untrained MAPPO actor.")
    print("Reward totals are a wiring check, not evidence of learned behavior.")


if __name__ == "__main__":
    main()
