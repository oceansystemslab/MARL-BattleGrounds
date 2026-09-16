"""Run JAX, independent-policy and host methods through the public System API.

Run ``python examples/systems.py`` from an installed checkout. Set
JAX_PLATFORMS=cpu for CPU correctness or JAX_PLATFORMS=cuda for the 32-game GPU
example. No training algorithm, network service or output file is used. The
example retains learning values from the same call that chooses each action.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.types import ActorAction, SystemInput, SystemOutput

type Tree = Any


def initialize(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    """Create zero decision counts shaped (B,5); no action is sampled.

    variables and keys are unused. Fixed JAX storage includes neutral padding;
    inputs.valid selects the episodes for which the runner uses these rows.
    """
    del variables, keys
    return jnp.zeros(inputs.active_mask.shape, jnp.int32)


def act(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Sample legal actions and return their already-known log probabilities.

    variables is unused. memory counts valid decisions per actor. inputs contains
    separate permitted actors and exact masks; keys has one JAX key per lane.
    Return (B,5) action heads, selected updated counters and (B,5) joint-action
    log probabilities. Uniform move and coupled-combat draws are independent.
    The values describe submitted actions and are invalid on padding lanes.
    """
    del variables
    actor_keys = jax.vmap(lambda key: jax.random.split(key, 5))(keys)
    actions = jax.vmap(jax.vmap(random_policy))(
        inputs.actors.observation, inputs.action_mask, actor_keys
    )
    moves = jnp.sum(inputs.action_mask.move_mask, axis=-1)
    combat_pairs = jnp.sum(
        inputs.action_mask.select_target_use_ultimate_joint_mask, axis=(-2, -1)
    )
    log_probability = -jnp.log(moves.astype(jnp.float32)) - jnp.log(
        combat_pairs.astype(jnp.float32)
    )
    return SystemOutput(
        actions,
        memory + inputs.valid[:, None].astype(jnp.int32),
        learning_outputs={"log_probability": log_probability},
    )


def host_act(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Tree]:
    """Return legal idle actions from a local fake Python provider.

    variables and JAX keys are unused; inputs are NumPy arrays with stable B.
    This stateless provider returns zero action heads shaped (B,5) and unchanged
    empty memory. A real provider must honor valid and each actor's information
    rights, gather its own requests and restore the same response order.
    """
    del variables, keys
    zero = np.zeros(inputs.active_mask.shape, np.int32)
    return ActorAction(zero, zero, zero), memory


def main() -> None:
    """Run all three supported raw workflows without automatic recording.

    Compile a 16-round, 32-game recurrent rollout with explicit terminal resets.
    Retain both rewards and method learning values before resetting. Then show
    ordered independent Policies and one host/JAX decision on the same 2v3
    roster. Wait for numerical work before printing the output shapes.
    """
    env = marl_bgs.make(
        "tdm",
        map_id=0,
        num_envs=32,
        team_a_roster=("mage", "priest"),
        team_b_roster=("rogue", "rogue", "priest"),
    )
    a = marl_bgs.System("Recurrent Random", act, init=initialize)
    b = marl_bgs.shared_policy(marl_bgs.policy("random"))
    key, reset_key, init_key = jax.random.split(jax.random.key(42), 3)
    observations, state = env.reset(reset_key)
    memory = marl_bgs.init_systems(a, b, observations, state, init_key)

    @jax.jit
    def rollout(carry: Tree) -> tuple[Tree, Tree]:
        """Run fixed-shape numerical work; return the latest carry and learning data."""

        def step(current: Tree, unused: None) -> tuple[Tree, Tree]:
            """Choose once, retain transition data, then reset finished lanes."""
            del unused
            rng, obs, before, methods = current
            rng, action_key, step_key, next_reset_key = jax.random.split(rng, 4)
            actions, methods, learning = marl_bgs.apply_systems(
                a,
                b,
                methods,
                obs,
                before,
                action_key,
                variables_a=(),
                variables_b=(),
            )
            result = env.step(step_key, before, actions)
            data = marl_bgs.system_step_data(before, actions, result)
            next_observations, _, _, done, _ = result
            # Keep terminal observations and flags for the learner before reset.
            obs, after = env.reset_done(next_reset_key, result[1])
            return (rng, obs, after, methods), (
                data,
                learning[0],
                next_observations,
                done,
            )

        return jax.lax.scan(step, carry, None, length=16)

    (key, observations, state, memory), history = rollout(
        (key, observations, state, memory)
    )
    # Each supplied entry follows its roster slot, including repeated classes.
    independent = marl_bgs.independent_policies(
        (marl_bgs.policy("random"), marl_bgs.policy("random"))
    )
    key, init_key, action_key = jax.random.split(key, 3)
    separate = marl_bgs.init_systems(independent, b, observations, state, init_key)
    actions, separate, _ = marl_bgs.apply_systems(
        independent, b, separate, observations, state, action_key
    )
    state_result = env.step(jax.random.fold_in(key, 1), state, actions)
    observations, state = env.reset_done(jax.random.fold_in(key, 2), state_result[1])

    host = marl_bgs.System("Local Host", host_act, execution="host")
    key, init_key, action_key, step_key = jax.random.split(key, 4)
    mixed = marl_bgs.init_systems(host, b, observations, state, init_key)
    actions, mixed, _ = marl_bgs.apply_systems(
        host, b, mixed, observations, state, action_key
    )
    result = env.step(step_key, state, actions)
    jax.block_until_ready((history, result, memory, separate))
    print("Recurrent rollout learning shape:", history[1]["log_probability"].shape)
    print("Independent and host/JAX decisions completed for 32 games.")


if __name__ == "__main__":
    main()
