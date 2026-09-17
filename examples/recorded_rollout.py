"""Collect optional training records and restore them with a numerical checkpoint.

Run ``python examples/recorded_rollout.py`` after installing MARL-BGs. The default
runs a compiled scan and writes no files. ``--output-dir PATH`` also demonstrates
bounded recording and a deliberate rewind to an earlier saved learner boundary.
Use ``--explicit-reset`` for a source-changing curriculum, or automatic resets by
default. Each seed owns a separate run, state, budget and checkpoint. The small
update below only demonstrates saving progress; it is not a learning algorithm.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config
from marl_battlegrounds.types import SystemInput, SystemOutput

type Tree = Any


class Carry(NamedTuple):
    """Hold every numerical value needed to continue this illustrative experiment.

    env is the dynamic execution handle. rng, observations, state, memory and
    tracking describe the next decision. source_index selects an explicit-reset
    curriculum row. weight, update_count and update_average represent caller-owned
    update progress; they are not a trained model or optimizer implementation.
    """

    env: Tree
    rng: Array
    observations: Tree
    state: Tree
    memory: Tree
    tracking: Tree
    source_index: Array
    weight: Array
    update_count: Array
    update_average: Array


def initialize(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    """Return zero float32 (B,5) actor memory without choosing an action.

    variables and keys are unused. The System runner resets only lanes whose
    episode generation changes. Each actor's memory remains separate.
    """
    del variables, keys
    return jnp.zeros(inputs.active_mask.shape, jnp.float32)


def act(
    variables: Array, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Choose legal actions once and return the same decision's learning values.

    variables is a dynamic scalar. memory is float32 (B,5). inputs contains only
    each actor's allowed data; keys supplies one independent root per lane.
    Return actions, incremented valid-lane memory and weighted old memory. These
    learning values stay outside the recording path.
    """
    actor_keys = jax.vmap(lambda key: jax.random.split(key, 5))(keys)
    actions = jax.vmap(jax.vmap(random_policy))(
        inputs.actors.observation, inputs.action_mask, actor_keys
    )
    return SystemOutput(
        actions, memory + inputs.valid[:, None], learning_outputs=variables * memory
    )


_METHOD = marl_bgs.System("Recorded Recurrent Random", act, init=initialize)
_OPPONENT = marl_bgs.shared_policy(marl_bgs.policy("random"))
_SYSTEMS: dict[str, object] = {"team_a": _METHOD, "team_b": _OPPONENT}


def step(current: Carry, unused: None) -> tuple[Carry, tuple[Tree, Tree, Tree]]:
    """Take one decision for direct scan or optional bounded collection.

    current holds all changing values, including the environment and method
    weight. unused is the empty scan input. Return the latest carry followed by
    (learner transition, old-episode info, same-decision policy trace). Explicit
    resets choose the current curriculum source. Automatic resets retain their
    resolved configuration. Neither route makes an extra policy call.
    """
    del unused
    (
        env,
        rng,
        observations,
        before,
        memory,
        tracking,
        source_index,
        weight,
        count,
        average,
    ) = current
    rng, action_key, step_key, reset_key = jax.random.split(rng, 4)
    automatic = isinstance(env, marl_bgs.AutoReset)
    if not automatic:
        source = jax.tree.map(
            lambda values: values[source_index], tracking.source_configs
        )
        prepared = marl_bgs.balanced_spawn_configs(source, num_envs=env.num_envs)
        observations, before = env.reset_done(reset_key, before, prepared)
    actions, memory, learning = marl_bgs.apply_systems(
        _METHOD, _OPPONENT, memory, observations, before, action_key, variables_a=weight
    )
    result = env.step(step_key, before, actions)
    tracking, result = marl_bgs.track_episode_step(
        tracking, before, result, source_indices=None if automatic else source_index
    )
    observations, after, _, _, info = result
    data = marl_bgs.system_step_data(before, actions, result)
    final = env.final_policy_inputs(info) if automatic else None
    terminal = (
        (final.valid, jnp.sum(final.actors.observation.self_features, axis=-1))
        if final is not None
        else ()
    )
    latest = Carry(
        env,
        rng,
        observations,
        after,
        memory,
        tracking,
        source_index,
        weight,
        count,
        average,
    )
    return latest, ((data, learning, terminal), info, memory.policy_trace)


@jax.jit
def direct_scan(carry: Carry) -> tuple[Carry, Tree]:
    """Run sixteen decisions with no recording storage, transfers or file output.

    carry is dynamic. Return its latest values and only the requested learner
    history. The direct scan remains available for ordinary numerical workflows.
    """

    def advance(current: Carry, unused: None) -> tuple[Carry, Tree]:
        """Discard recording outputs while retaining the exact learner transition."""
        latest, (transition, _, _) = step(current, unused)
        return latest, transition

    return jax.lax.scan(advance, carry, None, length=16)


def context(
    seed: int, *, automatic: bool, recording: bool, metrics: str, num_envs: int = 32
) -> Carry:
    """Prepare one independent batched experiment and its initial complete carry.

    seed owns RNG, memory and episode IDs. automatic selects AutoReset; otherwise
    resets use the chosen source index. recording enables starts and selects the
    initial and one continuing episode for replay, and the first for full metrics.
    metrics is priority/full/none. num_envs defaults to 32; use a positive even
    count for balanced stages, including smaller CPU correctness checks.
    No files or actions are produced here. The two immutable source rows have
    three- and five-transition horizons. All supplied configurations are exact.
    """
    rosters = marl_bgs.canonical_tournament_rosters()
    sources = [
        make_standard_team_deathmatch_config(
            map_id=map_id,
            max_steps=length,
            team_a_roster=rosters[0],
            team_b_roster=rosters[1],
        )
        for map_id, length in ((0, 3), (1, 5))
    ]
    bank = jax.tree.map(lambda *values: jnp.stack(values), *sources)
    indices = jnp.arange(num_envs, dtype=jnp.int32) % 2
    selected = jax.tree.map(lambda values: values[indices], bank)
    base = marl_bgs.make(
        "tdm",
        num_envs=num_envs,
        env_config=marl_bgs.balanced_spawn_configs(selected, num_envs=num_envs),
        metrics=metrics,
        replay_episodes=(1, 3 * num_envs + 1) if recording else (),
        full_metrics_episodes=(1,) if recording and metrics == "none" else (),
    )
    env = marl_bgs.AutoReset(base) if automatic else base
    rng, reset_key, init_key = jax.random.split(jax.random.key(seed), 3)
    observations, state = env.reset(reset_key)
    memory = marl_bgs.init_systems(_METHOD, _OPPONENT, observations, state, init_key)
    tracking = marl_bgs.init_episode_tracking(
        env, state, source_configs=bank, source_indices=indices, record_starts=recording
    ).begin_stage(state, total_env_steps=16 * num_envs)
    return Carry(
        env,
        rng,
        observations,
        state,
        memory,
        tracking,
        jnp.int32(0),
        jnp.float32(0.5),
        jnp.int32(0),
        jnp.float32(0),
    )


def illustrative_update(carry: Carry, transitions: Tree) -> Carry:
    """Advance small caller-owned update state once for a completed rollout.

    transitions comes from this rollout only. Update a scalar running average
    using same-call learning values and increment the saved update count. This
    demonstrates checkpoint ownership, not an optimizer or learning result.
    """
    value = jnp.mean(transitions[1][0])
    average = 0.9 * carry.update_average + 0.1 * value
    return carry._replace(
        weight=carry.weight + 0.001 * average,
        update_count=carry.update_count + 1,
        update_average=average,
    )


def save_numerical_checkpoint(
    path: Path, carry: Carry, token: dict[str, object]
) -> None:
    """Atomically save this example's numerical carry and recording token together.

    path is the caller's local checkpoint file. carry supplies all array leaves;
    its tree and static callable structure are rebuilt by this example on load.
    token must come from checkpoint_recording after successful collection. Store
    typed keys as key data plus their implementation name. No Python objects or
    provider sessions are pickled. Temporary files are replaced only after fsync;
    the parent directory is then synchronized. Errors propagate to the caller.
    """
    leaves = jax.tree.leaves(carry)
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    metadata: dict[str, Any] = {"version": 1, "token": token, "leaves": []}
    for index, leaf in enumerate(leaves):
        key = jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        data = jax.random.key_data(leaf) if key else leaf
        arrays[f"leaf_{index}"] = np.asarray(jax.device_get(data))
        metadata["leaves"].append(
            {"key_impl": str(jax.random.key_impl(leaf)) if key else None}
        )
    arrays["metadata"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode(), np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, allow_pickle=False, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_numerical_checkpoint(
    path: Path, template: Carry
) -> tuple[Carry, dict[str, object]]:
    """Load this example's arrays into a matching caller-supplied tree template.

    path is a trusted locally written example checkpoint; template comes from
    the same context/settings. Validate leaf count, numerical shape, dtype and
    typed-key implementation before rebuilding. Return the complete carry and
    its recording token. This is not a general checkpoint framework or a loader
    for arbitrary Python objects. Missing or incompatible files raise errors.
    """
    leaves, structure = jax.tree.flatten(template)
    restored = []
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
        if metadata["version"] != 1 or len(metadata["leaves"]) != len(leaves):
            raise ValueError("The saved numerical tree does not match this example")
        for index, (expected, description) in enumerate(
            zip(leaves, metadata["leaves"], strict=True)
        ):
            value = archive[f"leaf_{index}"]
            is_key = jnp.issubdtype(expected.dtype, jax.dtypes.prng_key)
            expected_data = jax.random.key_data(expected) if is_key else expected
            if value.shape != expected_data.shape or value.dtype != expected_data.dtype:
                raise ValueError(
                    f"Checkpoint leaf {index} has a different shape or dtype"
                )
            impl = description["key_impl"]
            if (str(jax.random.key_impl(expected)) if is_key else None) != impl:
                raise ValueError(
                    f"Checkpoint leaf {index} uses a different random key format"
                )
            data = jnp.asarray(value)
            restored.append(
                jax.random.wrap_key_data(data, impl=impl) if is_key else data
            )
    return jax.tree.unflatten(structure, restored), metadata["token"]


def run(
    seed: int, *, automatic: bool, metrics: str, output_dir: Path | None
) -> dict[str, object]:
    """Run one balanced stage, optionally demonstrating recorded rewind.

    seed and output_dir identify a separate learner/run. automatic chooses reset
    mode. metrics selects optional measurements; required outcomes remain when
    recording none-mode games. With no output directory, use a direct scan.
    With recording, save after eight decisions and one illustrative update, run
    ahead, then restore and repeat the unsaved second half. Compare final carry
    and learner outputs exactly. Return the latest read-only stage summary.
    """
    carry = context(
        seed, automatic=automatic, recording=output_dir is not None, metrics=metrics
    )
    if output_dir is None:
        carry, transitions = direct_scan(carry)
        carry = illustrative_update(carry, transitions)
    else:
        destination = output_dir / f"seed-{seed}"
        checkpoint = destination / "learner.npz"
        with marl_bgs.RunWriter(
            destination, phase="training", pass_id="training", policies=_SYSTEMS
        ) as writer:
            carry, transitions = marl_bgs.collect_rollout(
                step,
                carry,
                num_steps=8,
                writer=writer,
                source_configs=carry.tracking.source_configs,
            )
            carry = illustrative_update(carry, transitions)
            token = writer.checkpoint_recording()
            save_numerical_checkpoint(checkpoint, carry, token)
            template = carry
            run_dir = writer.run_dir
            # The next half is deliberately not published as a learner checkpoint.
            expected, outputs = marl_bgs.collect_rollout(
                step,
                carry,
                num_steps=8,
                writer=writer,
                source_configs=carry.tracking.source_configs,
            )
            expected = illustrative_update(expected, outputs)
        carry, token = load_numerical_checkpoint(checkpoint, template)
        with marl_bgs.RunWriter(
            resume_from=run_dir,
            recording_checkpoint=token,
            phase="training",
            pass_id="training",
            policies=_SYSTEMS,
        ) as writer:
            carry, transitions = marl_bgs.collect_rollout(
                step,
                carry,
                num_steps=8,
                writer=writer,
                source_configs=carry.tracking.source_configs,
            )
            carry = illustrative_update(carry, transitions)
        for actual, reference in zip(
            jax.tree.leaves((carry, transitions)),
            jax.tree.leaves((expected, outputs)),
            strict=True,
        ):
            if jnp.issubdtype(actual.dtype, jax.dtypes.prng_key):
                actual, reference = (
                    jax.random.key_data(actual),
                    jax.random.key_data(reference),
                )
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
    # Both values are the latest returned state, not the original scan input.
    state, tracking = carry.state, carry.tracking
    return tracking.stage_summary(state)


def main() -> None:
    """Parse a small complete workflow; each requested seed remains independent.

    --output-dir is absent by default, so ordinary invocation writes no files.
    --metrics defaults to priority. --explicit-reset selects curriculum setup.
    All commands use 32 environments per learner; CPU supports correctness runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--explicit-reset", action="store_true")
    parser.add_argument(
        "--metrics", choices=("priority", "full", "none"), default="priority"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    for seed in args.seeds:
        print(
            seed,
            run(
                seed,
                automatic=not args.explicit_reset,
                metrics=args.metrics,
                output_dir=args.output_dir,
            ),
        )


if __name__ == "__main__":
    main()
