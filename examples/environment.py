"""Run a ready batch with legal random actions and explicit resets.

Run with: python examples/environment.py
"""

import jax

import marl_battlegrounds as marl_bgs


def main() -> None:
    """Run 16 batched steps and reset each finished lane explicitly.

    The example uses 128 environments, fixed seed 42, and public legal sampling.
    It waits for all device work before printing. It writes no artifacts.
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


if __name__ == "__main__":
    main()
