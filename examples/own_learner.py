"""Train a small researcher-owned CTDE actor, then validate and evaluate it.

Run ``python examples/own_learner.py --output-dir artifacts/own_learner`` from
an installed checkout. The destination must be new. Defaults are a small wiring
example, not a trained baseline. For a larger declared budget, change --updates,
--rollout-length and --num-envs; their product counts environment transitions.
Use JAX_PLATFORMS=cpu for small CPU checks. GPU batches must follow the project's
allowed sizes; the default is 32. Add --custom-reward for a fading death signal.

CTDE means centralized training and decentralized execution. Only the critic
reads privileged physical state. The shared feed-forward actor processes each
permitted actor row separately. This file owns its network, Adam updates and
numerical actor snapshots; it does not call a built-in trainer. Snapshots are
immutable actor exports for this example, not resumable learner checkpoints.
"""

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_SIZE,
    TRAINING_STATE_FEATURE_SIZE,
    encode_actor_inputs,
    encode_training_state,
)
from marl_battlegrounds.core.types import EnvState
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    build_team_actor_input,
)
from marl_battlegrounds.training.analysis import select_checkpoint
from marl_battlegrounds.training.shaping import reward_adjustments
from marl_battlegrounds.training.validation import create_panel, validate_checkpoint
from marl_battlegrounds.types import ActionMask, ActorAction, SystemInput, SystemOutput

type Parameters = dict[str, Array]
type Tree = Any


class Carry(NamedTuple):
    """Keep the raw loop's key, current game inputs, System memory and rounds."""

    key: Array
    observations: Observations
    state: marl_bgs.EnvironmentState
    memory: marl_bgs.SystemState
    rounds: Array


class Rows(NamedTuple):
    """Keep compact actor observations and one privileged critic vector per game.

    Leading axes are (T,B); actor fields add five slots. targets and advantages
    use the producing successor before reset. Old log probabilities come from
    the same call that sampled the submitted actions. mask selects living,
    active actors on real decisions; advanced selects real game transitions.
    """

    observations: Observations
    masks: ActionMask
    actions: ActorAction
    old_log_probability: Array
    critic_features: Array
    targets: Array
    advantages: Array
    mask: Array
    advanced: Array


def initialize(key: Array, inputs: int, outputs: int) -> Parameters:
    """Create one small tanh network with 32 hidden units and explicit JAX keys.

    inputs/outputs are positive feature widths. Return float32 weights/biases;
    all applications preserve leading batch/actor axes without mixing rows.
    """
    hidden, head = jax.random.split(key)
    return {
        "hidden": jax.random.normal(hidden, (inputs, 32)) / jnp.sqrt(inputs),
        "hidden_bias": jnp.zeros(32, jnp.float32),
        "head": jax.random.normal(head, (32, outputs)) * jnp.float32(0.01),
        "head_bias": jnp.zeros(outputs, jnp.float32),
    }


def network(parameters: Parameters, features: Array) -> Array:
    """Apply independent rows after a fixed 0.01 input scale; preserve all axes."""
    hidden = jnp.tanh(
        features * 0.01 @ parameters["hidden"] + parameters["hidden_bias"]
    )
    return hidden @ parameters["head"] + parameters["head_bias"]


def distributions(
    parameters: Parameters, actors: ActorInput, masks: ActionMask
) -> tuple[Array, Array]:
    """Return masked log probabilities for nine moves and 22 combat pairs.

    actors contains permitted rows only. Coupled target/Ultimate legality comes
    from the supplied same-decision joint mask. No private rows are combined.
    """
    logits = network(parameters, encode_actor_inputs(actors))
    combat = masks.select_target_use_ultimate_joint_mask.reshape(
        (*logits.shape[:-1], 22)
    )
    return (
        jax.nn.log_softmax(jnp.where(masks.move_mask, logits[..., :9], -1e9)),
        jax.nn.log_softmax(jnp.where(combat, logits[..., 9:], -1e9)),
    )


def selected_log_probability(
    probabilities: tuple[Array, Array], actions: ActorAction
) -> Array:
    """Read each submitted move and combat pair's joint log probability."""
    moves, combat = probabilities
    return (
        jnp.take_along_axis(moves, actions.move[..., None], -1)[..., 0]
        + jnp.take_along_axis(
            combat, (2 * actions.select_target + actions.use_ultimate)[..., None], -1
        )[..., 0]
    )


def act(
    variables: Parameters, memory: Tree, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Choose legal independent actor actions and expose their same-call likelihood.

    variables holds actor weights only; memory is empty. inputs is the public
    permitted System view. keys has one key per game; split it by actor and head.
    The critic, raw state, rewards and other actors' private rows are absent.
    """
    probabilities = distributions(variables, inputs.actors, inputs.action_mask)

    def actor_keys(key: Array) -> Array:
        """Split one lane key into five actors and two action heads."""
        return jax.random.split(key, (5, 2))

    draws = jax.vmap(actor_keys)(keys)
    sample = jax.vmap(jax.vmap(jax.random.categorical))
    move = sample(draws[:, :, 0], probabilities[0]).astype(jnp.int32)
    combat = sample(draws[:, :, 1], probabilities[1]).astype(jnp.int32)
    actions = ActorAction(move, combat // 2, combat % 2)
    return SystemOutput(
        actions,
        memory,
        learning_outputs=selected_log_probability(probabilities, actions),
    )


def actor(parameters: Parameters, name: str = "Own CTDE Actor") -> marl_bgs.System:
    """Describe the decentralized actor with explicit weights and empty memory."""
    return marl_bgs.System(name, act, variables=parameters)


def fading_reward(
    before: EnvState,
    facts: marl_bgs.TrainingFacts,
    after: EnvState,
    progress: Array,
    *,
    fade_rounds: int,
) -> Array:
    """Fade a 0.01 death/contribution adjustment over declared training rounds.

    before/after are unused. Contributors are not exclusive killers. progress
    counts completed rounds; a fade budget of E transitions uses E/num_envs
    rounds. This changes the training objective while active, never task scores.
    """
    del before, after
    fade = jnp.maximum(0.0, 1.0 - progress.astype(jnp.float32) / fade_rounds)
    return (
        0.01
        * fade
        * (
            facts.contributed_to_new_death_by_source.astype(jnp.float32)
            - facts.is_newly_dead_by_recipient.astype(jnp.float32)
        )
    )


def td_targets(rewards: Array, next_values: Array, completed: Array) -> Array:
    """Add discounted successor value only while the native game continues.

    All inputs share one game shape. rewards/next_values are float32; completed
    is the native Boolean done flag, including horizon draws. A computational
    rollout cutoff leaves that flag false and keeps its producing successor.
    """
    return rewards + 0.99 * jnp.where(completed, 0, next_values)


def objective(
    parameters: tuple[Parameters, Parameters], rows: Rows
) -> tuple[Array, dict[str, Array]]:
    """Compute one on-policy actor-critic update from fixed collected decisions.

    The actor uses compact permitted observations and submitted-action masks.
    The critic alone uses physical-state features. TD targets and advantages
    are constants for this update; dead/inactive actors contribute no actor loss.
    """
    actor_weights, critic_weights = parameters
    actors = jax.vmap(jax.vmap(partial(build_team_actor_input, team=0)))(
        rows.observations
    )
    probabilities = distributions(actor_weights, actors, rows.masks)
    log_probability = selected_log_probability(probabilities, rows.actions)
    entropy = -sum(jnp.sum(jnp.exp(value) * value, axis=-1) for value in probabilities)
    count = jnp.maximum(jnp.sum(rows.mask), 1)
    policy_loss = (
        -jnp.sum(
            rows.mask * (log_probability * rows.advantages[..., None] + 0.01 * entropy)
        )
        / count
    )
    values = network(critic_weights, rows.critic_features)[..., 0]
    value_loss = jnp.sum(
        rows.advanced * jnp.square(rows.targets - values)
    ) / jnp.maximum(jnp.sum(rows.advanced), 1)
    return policy_loss + 0.5 * value_loss, {
        "actor_loss": policy_loss,
        "critic_loss": value_loss,
        "same_call_log_probability_error": jnp.max(
            jnp.where(rows.mask, jnp.abs(log_probability - rows.old_log_probability), 0)
        ),
        "env_steps": jnp.sum(rows.advanced),
    }


def save_actor(directory: Path, parameters: Parameters, *, env_steps: int) -> Path:
    """Create an immutable numerical actor snapshot and its own small metadata.

    directory must not exist. No built-in checkpoint format is claimed. The
    checkpoint ID hashes the exact saved bytes; env_steps is measured collection
    work. This actor-only snapshot cannot resume its optimizer or games.
    """
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "actor_weights.npz"
    np.savez(
        path,
        allow_pickle=False,
        **{key: np.asarray(value) for key, value in parameters.items()},
    )
    details = {
        "checkpoint_id": hashlib.sha256(path.read_bytes()).hexdigest(),
        "env_steps": env_steps,
        "name": f"Own CTDE Actor At {env_steps:,} Steps",
    }
    (directory / "actor.json").write_text(json.dumps(details, indent=2) + "\n")
    return directory


def load_actor(directory: Path) -> marl_bgs.System:
    """Load this example's numerical actor and verify its saved-byte identity."""
    details = json.loads((directory / "actor.json").read_text())
    path = directory / "actor_weights.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != details["checkpoint_id"]:
        raise ValueError("Actor weights differ from their saved identity")
    with np.load(path, allow_pickle=False) as saved:
        parameters = {name: jnp.asarray(saved[name]) for name in saved.files}
    return actor(parameters, details["name"])


def load_selected() -> marl_bgs.System:
    """Factory for CLI evaluation: set MARL_OWN_ACTOR to a saved actor folder.

    This zero-argument factory reads numerical weights once and performs no
    action. Use reference examples.own_learner:load_selected with evaluators.
    """
    return load_actor(Path(os.environ["MARL_OWN_ACTOR"]))


def train(
    output_dir: Path,
    *,
    updates: int = 4,
    rollout_length: int = 16,
    num_envs: int = 32,
    max_steps: int = 300,
    seed: int = 7,
    custom_reward: bool = False,
    fade_rounds: int = 1024,
) -> dict[str, Any]:
    """Run a small raw CTDE loop and save midpoint/final actor snapshots.

    output_dir must be new. Positive updates>=2, rollout_length and even
    num_envs define the budget without rounding. max_steps is the native game
    horizon; seed owns all keys. custom_reward enables the optional fact view
    and a declared fade_rounds horizon. Return snapshots, finite update metrics
    and initial/final parameters for the example's correctness checks.
    """
    if (
        updates < 2
        or rollout_length < 1
        or num_envs < 2
        or num_envs % 2
        or fade_rounds < 1
    ):
        raise ValueError("Use at least two updates, positive lengths and an even batch")
    output_dir.mkdir(parents=True, exist_ok=False)
    env = marl_bgs.make(
        "tdm",
        map_id=0,
        num_envs=num_envs,
        max_steps=max_steps,
        metrics="none",
        training_facts=custom_reward,
    )
    details = dict(
        seed=seed,
        training_maps=[0],
        updates=updates,
        rollout_length=rollout_length,
        num_envs=num_envs,
        max_steps=max_steps,
        custom_reward=custom_reward,
        fade_rounds=fade_rounds,
        discount=0.99,
        learning_rate=3e-4,
        algorithm="One-step actor-critic",
        example_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output_dir / "run_details.json").write_text(json.dumps(details, indent=2) + "\n")
    keys = jax.random.split(jax.random.key(seed), 5)
    initial = (
        initialize(keys[0], ACTOR_FEATURE_SIZE, 31),
        initialize(keys[1], TRAINING_STATE_FEATURE_SIZE, 1),
    )
    parameters = initial
    method = actor(initial[0])
    opponents = marl_bgs.pool({"random": 0.5, "tdm-alpha": 0.5})
    observations, state = env.reset(keys[2])
    memory = marl_bgs.init_systems(method, opponents, observations, state, keys[3])
    carry = Carry(keys[4], observations, state, memory, jnp.int32(0))
    optimizer = optax.adam(3e-4)
    optimizer_state = optimizer.init(parameters)

    def collect(
        weights: tuple[Parameters, Parameters], current: Carry, unused: None
    ) -> tuple[Carry, Rows]:
        """Act once, build targets from the producing successor, then reset."""
        del unused
        key, action_key, step_key, reset_key = jax.random.split(current.key, 4)
        before = current.state
        actions, memory, learning = marl_bgs.apply_systems(
            method,
            opponents,
            current.memory,
            current.observations,
            before,
            action_key,
            variables_a=weights[0],
        )
        result = env.step(step_key, before, actions)
        data = marl_bgs.system_step_data(before, actions, result)
        _, successor, _, done, info = result
        feedback = data.rewards
        if custom_reward:
            assert info.training_facts is not None
            extra = jax.vmap(
                partial(
                    reward_adjustments, partial(fading_reward, fade_rounds=fade_rounds)
                ),
                in_axes=(0, 0, 0, None),
            )(
                env.training_state(before),
                info.training_facts,
                env.training_state(successor),
                current.rounds,
            )
            feedback = feedback + extra[:, :5]
        active = data.active_mask
        team_reward = jnp.sum(jnp.where(active, feedback, 0), axis=-1) / jnp.maximum(
            jnp.sum(active, axis=-1), 1
        )
        features = encode_training_state(env.training_state(before), before.config)
        following = encode_training_state(
            env.training_state(successor), successor.config
        )
        values = network(weights[1], features)[..., 0]
        # Native horizon draws end this task. A rollout cutoff alone does not.
        targets = td_targets(
            team_reward, network(weights[1], following)[..., 0], done.done
        )

        def team_a(value: Array) -> Array:
            """Select Team A masks without changing their decision epoch."""
            return value[:, :5]

        rows = Rows(
            current.observations,
            jax.tree.map(team_a, before.action_mask),
            data.actions,
            learning[0],
            features,
            jax.lax.stop_gradient(targets),
            jax.lax.stop_gradient(targets - values),
            active & before.core_state.alive_mask[:, :5] & data.advanced[:, None],
            data.advanced,
        )
        observations, state = env.reset_done(reset_key, successor)
        return Carry(key, observations, state, memory, current.rounds + 1), rows

    @jax.jit
    def update(
        weights: Tree, opt_state: Tree, current: Carry
    ) -> tuple[Tree, Tree, Carry, Tree]:
        """Collect with dynamic weights and take one Adam step; keep outputs small."""
        current, rows = jax.lax.scan(
            partial(collect, weights), current, None, length=rollout_length
        )
        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
            weights, rows
        )
        changes, opt_state = optimizer.update(gradients, opt_state, weights)
        weights = optax.apply_updates(weights, changes)
        metrics["finite"] = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(value))
                    for value in jax.tree.leaves(
                        (weights, opt_state, gradients, metrics)
                    )
                ]
            )
        )
        return weights, opt_state, current, metrics

    metrics: list[dict[str, float]] = []
    snapshots: list[Path] = []
    env_steps = 0
    for index in range(updates):
        parameters, optimizer_state, carry, measured = cast(
            tuple[tuple[Parameters, Parameters], Tree, Carry, dict[str, Array]],
            update(parameters, optimizer_state, carry),
        )
        measured = jax.device_get(measured)
        if not bool(measured.pop("finite")):
            raise ValueError("Custom learner produced a nonfinite update")
        env_steps += int(measured["env_steps"])
        metrics.append({name: float(value) for name, value in measured.items()})
        if index + 1 in (updates // 2, updates):
            snapshots.append(
                save_actor(
                    output_dir / f"actor_step_{env_steps:012d}",
                    parameters[0],
                    env_steps=env_steps,
                )
            )
    (output_dir / "learning.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    return dict(
        snapshots=snapshots,
        metrics=metrics,
        initial=initial,
        parameters=parameters,
        env_steps=env_steps,
    )


def validate_and_evaluate(
    snapshots: list[Path], output_dir: Path, *, num_envs: int = 32
) -> dict[str, Any]:
    """Select real own snapshots on map 42, then evaluate on separate map 47.

    Every candidate receives the same frozen Random panel and paired spawn ends.
    A shared declared confirmation seed panel uses the public mean-point-margin
    rule. Extra checkpoint metadata comes from this learner's actual snapshots;
    it does not turn them into built-in learner artifacts. No folder is replaced.
    num_envs controls validation only; use 32 on GPU. The CLI caps this separate
    evaluation batch at 32 without changing the training experience budget.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    panel = create_panel(opponents=["random"], output_dir=output_dir / "panel")
    confirmations: list[dict[str, Any]] = []
    for path in snapshots:
        details = json.loads((path / "actor.json").read_text())
        result = validate_checkpoint(
            load_actor(path),
            panel,
            output_dir=output_dir / path.name,
            purpose="confirmation",
            seed_pairs=1,
            num_envs=num_envs,
            maps=[42],
        )
        confirmations.append(
            {
                **result,
                "checkpoint_id": details["checkpoint_id"],
                "env_steps": details["env_steps"],
                "actor_path": str(path),
            }
        )
    selected = select_checkpoint(confirmations)
    result = marl_bgs.evaluate(
        load_actor(Path(selected["actor_path"])),
        "random",
        maps=[47],
        num_episodes=2,
        num_envs=num_envs,
        seed=918,
        output_dir=output_dir / "final_evaluation",
    )
    summary = {
        "selected_actor": selected["actor_path"],
        "selection_rule": "point_margin",
        "evaluation_run": str(result.run_dir),
    }
    (output_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Parse a declared budget and run training, saved selection and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--custom-reward", action="store_true")
    parser.add_argument("--fade-rounds", type=int, default=1024)
    args = parser.parse_args()
    result = train(
        args.output_dir,
        updates=args.updates,
        rollout_length=args.rollout_length,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        seed=args.seed,
        custom_reward=args.custom_reward,
        fade_rounds=args.fade_rounds,
    )
    selected = validate_and_evaluate(
        result["snapshots"],
        args.output_dir / "validation",
        num_envs=min(args.num_envs, 32),
    )
    print(json.dumps({"env_steps": result["env_steps"], **selected}, indent=2))


if __name__ == "__main__":
    main()
