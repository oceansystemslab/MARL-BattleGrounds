"""Run a ready batch with legal random actions and explicit resets.

Also shows the Red Zone depth setting: 5.0 map units by default, 6.0 for a
deeper Red Zone and 0.0 for one point per death.

Run with: python examples/environment.py
"""

import jax
import jax.numpy as jnp

import marl_battlegrounds as marl_bgs


def main() -> None:
    """Run 16 batched steps, reset finished lanes, then show two Red Zone depths.

    The example uses 128 environments at the default Red Zone depth (5.0),
    fixed seed 42, and public legal sampling. It waits for all device work
    before printing. It then calls red_zone_choices. It writes no artifacts.
    """
    env = marl_bgs.make("tdm", map_id=0, num_envs=128)
    key = jax.random.key(42)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    for _ in range(16):
        key, action_key, step_key, reset_key = jax.random.split(key, 4)
        actions = env.sample_actions(action_key, state)
        obs, state, reward, done, info = env.step(step_key, state, actions)
        # Keep terminal transition data before resetting finished lanes.
        del reward, done, info
        obs, state = env.reset_done(reset_key, state)
    jax.block_until_ready((obs, state))
    print("Completed 16 steps in each of 128 environments.")
    red_zone_choices()


def red_zone_choices() -> None:
    """Build two small ready batches with explicit Red Zone depths.

    Each team's Red Zone is the full-height strip at its own spawn edge. When
    an agent dies with its centre inside its own team's Red Zone, the enemy
    team gets 2 points instead of 1; it is still one kill and one death. make
    uses 5.0 map units when red_zone_depth is omitted. Here 6.0 makes the
    strip deeper and 0.0 turns the rule off, so every death gives 1 point.
    Each batch has 2 environments and one reset with seed 0. The depth is an
    ordinary setting: changing it needs no code change. Prints each lane's
    recorded depth; writes no artifacts.
    """
    for depth in (6.0, 0.0):
        env = marl_bgs.make("tdm", map_id=0, num_envs=2, red_zone_depth=depth)
        _, state = env.reset(jax.random.key(0))
        depths = jnp.asarray(state.config.team_deathmatch_red_zone_depth)
        lanes = [float(value) for value in depths]
        print(f"Red Zone depth {depth}: each lane records {lanes}.")


if __name__ == "__main__":
    main()
