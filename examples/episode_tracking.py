"""Run explicit curricula, automatic resets and optional immediate recording.

Run ``python examples/episode_tracking.py`` with an installed MARL-BGs package.
Use JAX_PLATFORMS=cpu for correctness or cuda for the 32-game GPU workflow.
The default writes no files. --output-dir PATH saves a separate short AutoReset
run. These examples demonstrate recurrent methods and numerical continuation;
they do not implement a learner or coordinated learner/writer restart.
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.tasks import list_tdm_maps, make_standard_team_deathmatch_config
from marl_battlegrounds.types import SystemInput, SystemOutput

type Tree = Any


def initialize(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    """Return float32 (B,5) zero memory without producing an action.

    variables and keys are unused. Each actor keeps its own decision count.
    The System runner replaces only newly reset lanes before their next decision.
    """
    del variables, keys
    return jnp.zeros(inputs.active_mask.shape, jnp.float32)


def act(
    variables: Array, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Sample legal actions and return a same-call illustrative learning value.

    variables is a dynamic scalar weight. memory is float32 (B,5), with one
    count per actor. inputs keeps each actor's information separate. keys has
    one key per lane. Return sampled actions, next counts and weighted old
    counts as learning outputs; this value is not a trained critic or reward.
    """
    actor_keys = jax.vmap(lambda key: jax.random.split(key, 5))(keys)
    actions = jax.vmap(jax.vmap(random_policy))(
        inputs.actors.observation, inputs.action_mask, actor_keys
    )
    return SystemOutput(
        actions,
        memory + inputs.valid[:, None],
        learning_outputs=memory * variables,
    )


_METHOD = marl_bgs.System("Recurrent Random", act, init=initialize)
_OPPONENT = marl_bgs.shared_policy(marl_bgs.policy("random"))


def decision(
    env: Tree, carry: Tree, probabilities: Array, weight: Array, *, automatic: bool
) -> tuple[Any, Any]:
    """Return latest carry and exact transition data from one action call.

    env is a dynamic environment handle. carry holds RNG, observations, current
    environment state, System memory and tracker. probabilities is float32 (42,)
    over the training bank for explicit curricula; weight is a dynamic scalar.
    automatic is a static route choice. Return latest carry plus action/reward
    data, method learning outputs, illustrative terminal values and exact info.
    Fixed AutoReset retains its config. Learning outputs never enter recording.
    """
    key, observations, before, methods, tracker = carry
    key, action_key, step_key, next_reset_key, sample_key = jax.random.split(key, 5)
    indices = None
    if not automatic:
        indices = jax.random.choice(
            sample_key, probabilities.shape[0], shape=(env.num_envs,), p=probabilities
        )
        selected = jax.tree.map(lambda values: values[indices], tracker.source_configs)
        prepared = marl_bgs.balanced_spawn_configs(selected, num_envs=env.num_envs)
        observations, before = env.reset_done(next_reset_key, before, prepared)
    actions, methods, learning = marl_bgs.apply_systems(
        _METHOD,
        _OPPONENT,
        methods,
        observations,
        before,
        action_key,
        variables_a=weight,
    )
    result = env.step(step_key, before, actions)
    tracker, result = marl_bgs.track_episode_step(
        tracker,
        before,
        result,
        source_indices=indices,
    )
    observations, after, _, _, info = result
    data = marl_bgs.system_step_data(before, actions, result)
    # The terminal view preserves the old episode's permitted observations.
    final_inputs = env.final_policy_inputs(info) if automatic else None
    terminal_value = (
        (
            final_inputs.valid,
            jnp.sum(final_inputs.actors.observation.self_features, axis=-1),
        )
        if final_inputs is not None
        else ()
    )
    return (key, observations, after, methods, tracker), (
        data,
        learning,
        terminal_value,
        info,
    )


@partial(jax.jit, static_argnames=("automatic", "num_steps"))
def rollout(
    env: Tree,
    carry: Tree,
    probabilities: Array,
    weight: Array,
    *,
    automatic: bool,
    num_steps: int = 16,
) -> tuple[Any, Any]:
    """Run a fixed number of decisions with reusable compiled work.

    env, carry, float32 source probabilities and scalar weight remain dynamic.
    automatic selects the explicit/automatic reset structure. num_steps defaults
    to 16; a different length selects a separate compiled program. Return
    latest carry and per-decision learning data; discard unrecorded info history.
    """

    def scan_step(current: Tree, unused: None) -> tuple[Any, Any]:
        """Keep exact action/learning/final data; discard unrecorded diagnostics."""
        del unused
        current, data = decision(
            env, current, probabilities, weight, automatic=automatic
        )
        return current, data[:3]

    return jax.lax.scan(scan_step, carry, None, length=num_steps)


def curriculum_probabilities(pool_size: int) -> Array:
    """Return uniform float32 (42,) probabilities for one accepted training pool.

    pool_size is 1 through 12, or 42 for the final pool, in catalog order.
    This host helper rejects other sizes. Passing its values to rollout keeps
    one compiled shape; no validation or test map enters the bank.
    """
    if type(pool_size) is not int or pool_size not in (*range(1, 13), 42):
        raise ValueError("Choose a curriculum pool from 1 through 12, or 42")
    return (jnp.arange(42) < pool_size).astype(jnp.float32) / pool_size


def make_context(
    seed: int, *, automatic: bool, record_starts: bool = False, num_envs: int = 32
) -> tuple[Tree, Tree]:
    """Prepare one independent context without taking actions or writing files.

    seed owns RNG and memory. automatic uses a fixed initial source with AutoReset;
    otherwise the tracker owns all 42 training configurations in catalog order.
    record_starts=False skips recording identity preparation. num_envs defaults
    to 32 and must be positive and even for balanced stages. Smaller CPU batches
    are useful for correctness. Return env and (RNG, obs, state, memory, tracker).
    Source validation and optional hashing happen once outside compiled execution.
    """
    if type(num_envs) is not int or num_envs <= 0 or num_envs % 2:
        raise ValueError("Balanced stages need a positive even batch")
    roster_a, roster_b = marl_bgs.canonical_tournament_rosters()
    maps = tuple(item for item in list_tdm_maps() if item.split == "training")
    if len(maps) != 42 or [item.map_id for item in maps[:12]] != list(range(12)):
        raise ValueError("The example needs the declared 42-map training catalog")
    sources = [
        make_standard_team_deathmatch_config(
            map_id=item.map_id,
            max_steps=3 + index % 3,
            team_a_roster=roster_a,
            team_b_roster=roster_b,
        )
        for index, item in enumerate(maps[:1] if automatic else maps)
    ]
    for source in sources:
        marl_bgs.balanced_spawn_configs(source, num_envs=num_envs)
    bank = jax.tree.map(lambda *rows: jnp.stack(rows), *sources)
    base = marl_bgs.make(
        "tdm",
        num_envs=num_envs,
        env_config=marl_bgs.balanced_spawn_configs(sources[0], num_envs=num_envs),
        metrics="priority",
    )
    env = marl_bgs.AutoReset(base) if automatic else base
    rng, reset_key, init_key = jax.random.split(jax.random.key(seed), 3)
    obs, state = env.reset(reset_key)
    memory = marl_bgs.init_systems(_METHOD, _OPPONENT, obs, state, init_key)
    tracking = marl_bgs.init_episode_tracking(
        env, state, source_configs=bank, source_indices=0, record_starts=record_starts
    )
    return env, (rng, obs, state, memory, tracking)


def run(
    seed: int, *, automatic: bool, output_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Run the complete growing curriculum or two fixed AutoReset stages.

    seed owns this context's RNG, episodes, memory and tracker. automatic chooses
    fixed-configuration AutoReset; False uses explicit reset_done and grows
    training pools from 1 through 12, then all 42 maps. output_dir optionally
    creates one new recording run on the automatic route. Recorded calls use a
    host loop; otherwise a compiled scan returns the complete numerical carry.
    Return one summary per stage. Errors propagate; writers close on exit.
    """
    from contextlib import nullcontext

    env, carry = make_context(
        seed, automatic=automatic, record_starts=output_dir is not None
    )
    rng, obs, state, memory, tracking = carry
    writer_context: Any = nullcontext(None)
    if output_dir is not None:
        if not automatic:
            raise ValueError("The optional recording example uses AutoReset")
        writer_context = marl_bgs.RunWriter(
            output_dir,
            policies={"team_a": _METHOD, "team_b": _OPPONENT},
            phase="training",
            pass_id=f"seed-{seed}",
            details={"metrics": "priority"},
        )
    summaries = []
    with writer_context as writer:
        for stage_number, pool_size in enumerate(
            (1, 1) if automatic else (*range(1, 13), 42)
        ):
            tracking = tracking.begin_stage(state, total_env_steps=512)
            carry = (rng, obs, state, memory, tracking)
            probabilities = curriculum_probabilities(pool_size)
            weight = jnp.asarray(0.5 + 0.1 * stage_number, jnp.float32)
            if writer is None:
                carry, learning_data = rollout(
                    env, carry, probabilities, weight, automatic=automatic
                )
                jax.block_until_ready(learning_data)
            else:
                for _ in range(16):
                    carry, data = decision(
                        env, carry, probabilities, weight, automatic=automatic
                    )
                    info = data[3]
                    writer.register_episodes(
                        info.episode_start_records,
                        source_configs=carry[4].source_configs,
                    )
                    writer.write(info, policy_trace=carry[3].policy_trace)
            # Always use both latest returned states, including after a scan.
            rng, obs, state, memory, tracking = carry
            summaries.append(tracking.stage_summary(state))
        if writer is not None:
            writer.flush()
    return summaries


@jax.jit
def batched_rollout(
    env: Tree, carries: Tree, probabilities: Array, weight: Array
) -> Tree:
    """Run independent learner contexts along a separate leading seed axis.

    env is the common execution handle; carries has shape (S,B,...) on lane
    leaves and (S,...) on per-context values. Each seed retains its own tracker,
    memory and RNG. probabilities and weight are shared dynamic values here.
    Return latest carries and learning data. Stage checks remain host calls.
    """
    return jax.vmap(partial(rollout, automatic=True), in_axes=(None, 0, None, None))(
        env,
        carries,
        probabilities,
        weight,
    )


def run_batched() -> list[dict[str, Any]]:
    """Show two independent seeds sharing compiled work, without pooling budgets.

    Each seed has 32 environments. This is 64 simultaneous environments when
    run on a GPU. Only numerical carries are batched; no host writer/provider
    or stage-check method is mapped. This demonstration writes no files.
    """
    contexts = [make_context(seed, automatic=True) for seed in (42, 43)]
    ready = []
    for _, (rng, obs, state, memory, tracking) in contexts:
        ready.append(
            (rng, obs, state, memory, tracking.begin_stage(state, total_env_steps=512))
        )
    carries = jax.tree.map(lambda *values: jnp.stack(values), *ready)
    latest, _ = batched_rollout(
        contexts[0][0],
        carries,
        curriculum_probabilities(1),
        jnp.asarray(0.5, jnp.float32),
    )
    summaries = []
    for index in range(2):
        _, _, state, _, tracker = jax.tree.map(
            lambda values, index=index: values[index], latest
        )
        summaries.append(tracker.stage_summary(state))
    return summaries


def main() -> None:
    """Run independent contexts; optionally save an additional isolated host run.

    Each seed receives its own environment, RNG, memory, tracker and budget.
    No pooled count is used to certify balance. --output-dir is absent by default.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for seed in (42, 43):
        print("Explicit curriculum", seed, run(seed, automatic=False))
        print("AutoReset", seed, run(seed, automatic=True))
    print("Batched independent seeds", run_batched())
    if args.output_dir is not None:
        print("Recorded AutoReset", run(44, automatic=True, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
