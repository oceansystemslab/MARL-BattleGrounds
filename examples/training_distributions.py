"""Run sampled episodes with exact provenance and optional recording restart.

Run ``python examples/training_distributions.py`` after installing MARL-BGs.
The default uses 32 games, Random on both teams and 320 decisions, so the
canonical 300-transition horizon reaches a reset. No files or learner updates
are produced. ``--output-dir PATH`` records through the existing collector and
demonstrates a rewind using an in-memory continuation. ``--mappo`` requires the
training extra and uses explicitly untrained MAPPO actor weights for Team A.
Use JAX_PLATFORMS=cpu for correctness or cuda on the selected internal GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    EpisodeInfo,
    MetricMode,
    _checked_increment,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.episode_tracking import EpisodeTrackingState
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace, SystemState
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.training import (
    TRAINING_KEY_SCHEMA_VERSION,
    PreparedTrainingContent,
    SampledTrainingConfigs,
    TrainingContentBinding,
    prepare_training_content,
    sample_training_configs,
    training_keys,
    validate_training_distribution,
)
from marl_battlegrounds.types import System

type Tree = Any


class Carry(NamedTuple):
    """Keep the complete numerical continuation for this fixed batch.

    env owns the execution settings. root_key is one Threefry root; observations
    and state describe the next decision. memory and variables_a belong only to
    the Systems. tracking owns one shared source bank. source_indices (B,) and
    source_class_ids (B,10) describe the current games, including resets not yet
    seen by the tracker. eligible_maps is bool (42,); team_size is scalar int32.
    No host binding, writer, expanded actor inputs or critic enters this tree.
    """

    env: Environment
    root_key: Array
    observations: Observations
    state: EnvironmentState
    memory: SystemState
    tracking: EpisodeTrackingState
    source_indices: Array
    source_class_ids: Array
    eligible_maps: Array
    team_size: Array
    variables_a: Tree


class Transition(NamedTuple):
    """Retain small decision records without copying state or the source bank.

    actions has int32 (B,10) heads; rewards is float32 (B,10). completed and
    advanced are bool (B,) for the producing transition, including terminal
    advances. episode_id and decision_step are int32 (B,). learning contains
    only same-call actor outputs, or an empty tuple for Random.
    """

    actions: Action
    rewards: Array
    completed: Array
    advanced: Array
    episode_id: Array
    decision_step: Array
    learning: Tree


class DecisionInputs(NamedTuple):
    """Hold preselected inputs used only by the equivalent benchmark control.

    action_keys, step_keys and reset_keys have one typed key per lane. sampled
    contains the exact next-generation configurations and declarations. A scan
    may stack these values into a reference tape outside timing; the ordinary
    example never stores such a tape or samples when no game resets.
    """

    action_keys: Array
    step_keys: Array
    reset_keys: Array
    sampled: SampledTrainingConfigs


@dataclass(frozen=True)
class Context:
    """Keep host setup evidence and static Systems beside the numerical carry.

    binding describes the verified installed content and canonical source bank.
    source_bank_is_synthetic marks an explicitly supplied bank with changed test
    horizons; that bank has its own recorded identity and is not described by
    binding's canonical bank identity. actor/opponent keep their public System
    roles. These host values do not become actor inputs.
    """

    carry: Carry
    actor: System
    opponent: System
    binding: TrainingContentBinding
    source_bank_is_synthetic: bool


@dataclass(frozen=True)
class Continuation:
    """Save an in-memory example boundary, not a learner checkpoint format.

    context carries the exact arrays, controls, root, fixed batch and lane order.
    key_schema_version, root_bits and key_implementation record key ownership.
    num_envs and lane_order record the fixed batch and canonical lane positions.
    recording_token is None without recording, otherwise the writer's own token.
    Arrays are immutable; do not donate or replace saved leaves before reuse.
    No disk serialization or arbitrary Python-object loading is provided.
    """

    context: Context
    key_schema_version: int
    root_bits: tuple[int, ...]
    key_implementation: str
    num_envs: int
    lane_order: tuple[int, ...]
    recording_token: dict[str, object] | None


def _check_source_bank(prepared: PreparedTrainingContent, bank: EnvConfig) -> bool:
    """Allow only an explicitly synthetic horizon change to the verified bank.

    Both trees must have identical shapes, dtypes and values except max_steps.
    Return whether horizons differ. Core validation later checks their range.
    Reject other differences with ValueError; this host check admits no external
    map or changed class/rule source and performs no content reads or hashing.
    """
    expected = prepared.source_configs
    if jax.tree.structure(bank) != jax.tree.structure(expected):
        raise ValueError("The example source bank must keep the verified structure")
    for actual, reference in zip(
        jax.tree.leaves(bank), jax.tree.leaves(expected), strict=True
    ):
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise ValueError(
                "The example source bank must keep verified shapes and dtypes"
            )
    restored = bank._replace(max_steps=expected.max_steps)
    if any(
        not np.array_equal(np.asarray(actual), np.asarray(reference))
        for actual, reference in zip(
            jax.tree.leaves(restored), jax.tree.leaves(expected), strict=True
        )
    ):
        raise ValueError("Synthetic source banks may change only episode horizons")
    return not np.array_equal(
        np.asarray(bank.max_steps), np.asarray(expected.max_steps)
    )


def make_context(
    *,
    prepared: PreparedTrainingContent | None = None,
    seed: int = 42,
    num_envs: int = 32,
    team_size: int = 3,
    eligible_maps: Array | None = None,
    recording: bool = False,
    replay_episodes: tuple[int, ...] = (),
    metrics: MetricMode = "priority",
    mappo: bool = False,
    source_configs: EnvConfig | None = None,
) -> Context:
    """Verify setup, then initialize sampled games without choosing an action.

    prepared optionally reuses a prior successful content preflight. seed owns
    one Threefry root. num_envs is a fixed positive even batch; team_size is a
    plain integer in 1..5. eligible_maps defaults to all 42 training maps and
    otherwise must be bool (42,) with at least one True. recording enables start
    records; replay_episodes optionally selects episode IDs and defaults to none.
    metrics defaults to priority. mappo selects
    an untrained Team A actor and requires the existing training extra. Random
    remains Team B. source_configs optionally supplies an explicitly synthetic
    test bank differing only in horizons; it does not replace the base binding.

    Return Context with host evidence and a complete numerical Carry. Inputs are
    unchanged. Host preparation reads/validates content, allocates device arrays
    and optionally hashes tracking identity. No writer or learner is created.
    Invalid controls, bank changes or existing setup errors raise before actions.
    """
    if type(num_envs) is not int or num_envs <= 0 or num_envs % 2:
        raise ValueError("num_envs must be a positive even integer")
    if type(team_size) is not int or not 1 <= team_size <= 5:
        raise ValueError("team_size must be a plain integer in 1..5")
    prepared = prepare_training_content() if prepared is None else prepared
    bank = prepared.source_configs if source_configs is None else source_configs
    synthetic = False if source_configs is None else _check_source_bank(prepared, bank)
    eligible = jnp.ones(42, jnp.bool_) if eligible_maps is None else eligible_maps
    size = jnp.asarray(team_size, jnp.int32)
    validate_training_distribution(eligible_maps=eligible, team_size=size)
    root = jax.random.key(seed)
    generation = jnp.zeros(num_envs, jnp.int32)
    sampled = sample_training_configs(
        bank, root, generation, eligible_maps=eligible, team_size=size
    )
    opponent = marl_bgs.shared_policy(marl_bgs.policy("random"))
    actor = opponent
    variables: Tree = ()
    if mappo:
        from marl_battlegrounds.baselines import ppo

        network_key = training_keys(root, generation, stream="initialization")[0]
        variables = ppo.initialize_ppo(network_key).actor_params
        actor = ppo.make_recurrent_mappo_system(
            variables, name="Untrained Recurrent MAPPO"
        )
    env = marl_bgs.make(
        "tdm",
        env_config=sampled.config,
        num_envs=num_envs,
        metrics=metrics,
        replay_episodes=replay_episodes,
    )
    observations, state = env.reset(training_keys(root, generation, stream="reset"))
    memory = marl_bgs.init_systems(
        actor,
        opponent,
        observations,
        state,
        training_keys(root, generation, stream="initialization"),
    )
    tracking = marl_bgs.init_episode_tracking(
        env,
        state,
        source_configs=bank,
        source_indices=sampled.source_indices,
        source_class_ids=sampled.source_class_ids,
        record_starts=recording,
    )
    return Context(
        Carry(
            env,
            root,
            observations,
            state,
            memory,
            tracking,
            sampled.source_indices,
            sampled.source_class_ids,
            eligible,
            size,
            variables,
        ),
        actor,
        opponent,
        prepared.binding,
        synthetic,
    )


def decision_inputs(carry: Carry) -> DecisionInputs:
    """Prepare exact reference choices without changing the supplied carry.

    This benchmark helper samples all lanes even if they will not reset. Keep it
    outside measured execution when preparing a tape. The ordinary step derives
    these same keys but draws configurations only inside its reset condition.
    Counter overflow uses the existing checked increment; the environment owns
    the sticky failure when a reset is attempted at that limit.
    """
    generation = carry.state.reset_generation
    local_step = carry.state.core_state.step_count - carry.state.initial_step_count
    following, _ = _checked_increment(generation, 1)
    assert carry.tracking.source_configs is not None
    return DecisionInputs(
        training_keys(
            carry.root_key, generation, stream="action", decision_step=local_step
        ),
        training_keys(
            carry.root_key, generation, stream="step", decision_step=local_step
        ),
        training_keys(carry.root_key, following, stream="reset"),
        sample_training_configs(
            carry.tracking.source_configs,
            carry.root_key,
            following,
            eligible_maps=carry.eligible_maps,
            team_size=carry.team_size,
        ),
    )


def _advance(
    carry: Carry,
    actor: System,
    opponent: System,
    preselected: DecisionInputs | None,
) -> tuple[Carry, tuple[Transition, EpisodeInfo, PolicyTrace]]:
    """Share one public transition/reset body between sampled and tape execution.

    carry and optional preselected values are dynamic numerical inputs. actor
    and opponent are stable System descriptors. Track the producing episode
    before reset. Return its compact transition, exact info and policy trace;
    the next carry holds fresh observations only for completed lanes. No host
    content work, file writing or additional policy call occurs here.
    """
    before = carry.state
    local_step = before.core_state.step_count - before.initial_step_count
    action_keys = (
        training_keys(
            carry.root_key,
            before.reset_generation,
            stream="action",
            decision_step=local_step,
        )
        if preselected is None
        else preselected.action_keys
    )
    step_keys = (
        training_keys(
            carry.root_key,
            before.reset_generation,
            stream="step",
            decision_step=local_step,
        )
        if preselected is None
        else preselected.step_keys
    )
    actions, memory, learning = marl_bgs.apply_systems(
        actor,
        opponent,
        carry.memory,
        carry.observations,
        before,
        action_keys,
        variables_a=carry.variables_a,
    )
    result = carry.env.step(step_keys, before, actions)
    tracking, result = marl_bgs.track_episode_step(
        carry.tracking,
        before,
        result,
        source_indices=carry.source_indices,
        source_class_ids=carry.source_class_ids,
    )
    observations, after, rewards, _, info = result
    advanced = info.decision_step >= 0
    transition = Transition(
        actions,
        rewards.rewards,
        info.completed,
        advanced,
        info.episode_id,
        info.decision_step,
        learning[0],
    )
    next_carry = carry._replace(
        observations=observations, state=after, memory=memory, tracking=tracking
    )

    def reset_finished(current: Carry) -> Carry:
        """Draw once, reset finished games and retain all continuing declarations."""
        generations, _ = _checked_increment(current.state.reset_generation, 1)
        assert current.tracking.source_configs is not None
        sampled = (
            sample_training_configs(
                current.tracking.source_configs,
                current.root_key,
                generations,
                eligible_maps=current.eligible_maps,
                team_size=current.team_size,
            )
            if preselected is None
            else preselected.sampled
        )
        keys = (
            training_keys(current.root_key, generations, stream="reset")
            if preselected is None
            else preselected.reset_keys
        )
        reset_mask = current.state.done.done
        obs, state = current.env.reset_done(keys, current.state, sampled.config)
        return current._replace(
            observations=obs,
            state=state,
            source_indices=jnp.where(
                reset_mask, sampled.source_indices, current.source_indices
            ),
            source_class_ids=jnp.where(
                reset_mask[:, None], sampled.source_class_ids, current.source_class_ids
            ),
        )

    def retain(current: Carry) -> Carry:
        """Return continuing games without sampling or resetting."""
        return current

    next_carry = cast(
        Carry,
        jax.lax.cond(jnp.any(after.done.done), reset_finished, retain, next_carry),
    )
    return next_carry, (transition, info, memory.policy_trace)


def make_step(
    actor: System, opponent: System
) -> Callable[
    [Carry, DecisionInputs | None],
    tuple[Carry, tuple[Transition, EpisodeInfo, PolicyTrace]],
]:
    """Bind stable System descriptors once for scan and bounded collection.

    The returned step accepts (carry, None) for ordinary sampling, or one
    DecisionInputs row for the benchmark control. Weights, bank values, keys,
    memories and controls remain in dynamic arguments. Reuse this callable;
    constructing a new callable for every rollout would lose compiler reuse.
    """

    def step(
        carry: Carry, preselected: DecisionInputs | None
    ) -> tuple[Carry, tuple[Transition, EpisodeInfo, PolicyTrace]]:
        """Advance one public decision through the shared sampled/reset body."""
        return _advance(carry, actor, opponent, preselected)

    return step


def direct_scan(
    step: Callable[..., Tree], carry: Carry, *, num_steps: int = 320
) -> tuple[Carry, Transition]:
    """Scan the reusable step and retain compact transitions only.

    step comes from make_step; carry contains every changing value. num_steps
    fixes the compiled time axis and defaults to 320. This function composes
    inside jit. It neither writes files nor retains a wide EpisodeInfo history.
    Return latest carry and time-major compact Transition fields.
    """

    def advance(current: Carry, unused: None) -> tuple[Carry, Transition]:
        """Drop recording-only outputs while keeping each exact action result."""
        latest, (transition, _, _) = step(current, unused)
        return latest, transition

    return jax.lax.scan(advance, carry, None, length=num_steps)


def save_continuation(
    context: Context, carry: Carry, recording_token: dict[str, object] | None = None
) -> Continuation:
    """Keep a completed numerical boundary with its optional writer token.

    context supplies host binding and Systems; carry must be its latest complete
    return. recording_token comes from checkpoint_recording after collection.
    No disk format is created. Preserve this record without donating its arrays.
    """
    latest = Context(
        carry,
        context.actor,
        context.opponent,
        context.binding,
        context.source_bank_is_synthetic,
    )
    return Continuation(
        latest,
        TRAINING_KEY_SCHEMA_VERSION,
        tuple(int(value) for value in np.asarray(jax.random.key_data(carry.root_key))),
        str(jax.random.key_impl(carry.root_key)),
        cast(int, carry.env.num_envs),
        tuple(range(cast(int, carry.env.num_envs))),
        recording_token,
    )


def restore_continuation(saved: Continuation) -> Context:
    """Recheck content and key ownership before constructing any resumed writer.

    saved is a trusted in-memory boundary from save_continuation. Return its
    exact numerical carry and Systems after checking the installed scientific
    binding, Threefry schema/root and distribution controls. Changed batch or
    lane ordering is outside this exact continuation. Failure precedes writer
    recovery; this function opens no writer and mutates no durable output.
    """
    context = saved.context
    prepare_training_content(expected=context.binding)
    if (
        saved.key_schema_version != TRAINING_KEY_SCHEMA_VERSION
        or saved.key_implementation != "threefry2x32"
        or saved.key_implementation != str(jax.random.key_impl(context.carry.root_key))
        or saved.num_envs != context.carry.env.num_envs
        or saved.lane_order != tuple(range(saved.num_envs))
        or saved.root_bits
        != tuple(
            int(value)
            for value in np.asarray(jax.random.key_data(context.carry.root_key))
        )
    ):
        raise ValueError("Saved random-key ownership does not match this continuation")
    validate_training_distribution(
        eligible_maps=context.carry.eligible_maps, team_size=context.carry.team_size
    )
    return context


def _equal_trees(actual: Tree, expected: Tree) -> None:
    """Check exact continuation equality, including typed-key bits, on the host."""
    if jax.tree.structure(actual) != jax.tree.structure(expected):
        raise AssertionError("Continuation tree structures differ")
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def run(
    *, steps: int = 320, output_dir: Path | None = None, mappo: bool = False
) -> dict[str, object]:
    """Run one complete example, optionally recording and replaying its suffix.

    steps must be at least two. output_dir is absent by default; otherwise create
    a new recording there, save the midpoint in memory, run ahead, restore and
    repeat the suffix exactly. mappo selects an explicitly untrained actor.
    Return real-step and reset counts. Writers close on success or failure.
    """
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must be an integer of at least two")
    context = make_context(recording=output_dir is not None, mappo=mappo)
    step = make_step(context.actor, context.opponent)
    carry = context.carry
    if output_dir is None:

        def rollout(values: Carry) -> tuple[Carry, Transition]:
            """Run one compiled example with all changing values passed in."""
            return direct_scan(step, values, num_steps=steps)

        carry, transitions = cast(tuple[Carry, Transition], jax.jit(rollout)(carry))
        jax.block_until_ready(transitions)
    else:
        systems: dict[str, object] = {
            "team_a": context.actor,
            "team_b": context.opponent,
        }
        first = steps // 2
        with marl_bgs.RunWriter(
            output_dir, phase="training", pass_id="sampled", policies=systems
        ) as writer:
            carry, _ = marl_bgs.collect_rollout(
                step,
                carry,
                num_steps=first,
                writer=writer,
                source_configs=carry.tracking.source_configs,
            )
            saved = save_continuation(context, carry, writer.checkpoint_recording())
            run_dir = writer.run_dir
            expected, outputs = marl_bgs.collect_rollout(
                step,
                carry,
                num_steps=steps - first,
                writer=writer,
                source_configs=carry.tracking.source_configs,
            )
        restored = restore_continuation(saved)
        carry = restored.carry
        with marl_bgs.RunWriter(
            resume_from=run_dir,
            recording_checkpoint=saved.recording_token,
            phase="training",
            pass_id="sampled",
            policies=systems,
        ) as writer:
            carry, transitions = marl_bgs.collect_rollout(
                step,
                carry,
                num_steps=steps - first,
                writer=writer,
                source_configs=carry.tracking.source_configs,
            )
        _equal_trees((carry, transitions), (expected, outputs))
    return {
        "Real Transitions": int(
            np.asarray(jnp.sum(carry.state.cumulative_transition_count))
        ),
        "Episode Resets": int(np.asarray(jnp.sum(carry.state.reset_generation))),
        "Actor": "Untrained MAPPO" if mappo else "Random",
        "Recording": "Disabled" if output_dir is None else "Restored And Reproduced",
    }


def main() -> None:
    """Read example options and print wiring counts without making learning claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mappo", action="store_true")
    options = parser.parse_args()
    print(run(steps=options.steps, output_dir=options.output_dir, mappo=options.mappo))
    print("These are integration checks. No policy was trained.")


if __name__ == "__main__":
    main()
