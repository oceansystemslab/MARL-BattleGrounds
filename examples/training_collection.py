"""Collect an exact experience budget with an untrained recurrent MAPPO actor.

Install the training extra. For a small CPU check, run::

    JAX_PLATFORMS=cpu python examples/training_collection.py \
        --mode Plain --num-envs 2 --total-env-steps 4 --length 4

The ordinary defaults use 32 games and blocks of 128 rounds. Select Plain, C,
RS or C-RS to enable neither option, curriculum, shaping or both. Printed counts
separate requested stages from the distributions actually played. No optimizer
update, historical snapshot, learner checkpoint or learning claim is made.
Only --output-dir enables files, through the existing training RunWriter.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import cast

_MODES = ("Plain", "C", "RS", "C-RS")


def run(
    *,
    mode: str = "Plain",
    num_envs: int = 32,
    total_env_steps: int = 4096,
    length: int = 128,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Collect untrained self-play and return the public exposure summary.

    Parameters
    ----------
    mode : {"Plain", "C", "RS", "C-RS"}, default "Plain"
        Plain uses neither curriculum nor shaping; C uses curriculum; RS uses
        score shaping; C-RS uses both. Task rules keep K20 and H300 in every mode.
    num_envs : int, default 32
        Positive even number of simultaneous games. Keep this fixed throughout
        the run. Small CPU checks may use two; GPU workloads use approved sizes.
    total_env_steps : int, default 4096
        Positive total real transitions across all games, divisible by num_envs.
        Curriculum needs enough rounds to give all 17 requested stages one round.
    length : int, default 128
        Positive output length per block. The final shorter real prefix is
        padded without choosing extra actions or counting extra experience.
    seed : int, default 42
        Python integer used for separate model initialization and collection
        keys. Boolean values are rejected.
    output_dir : Path or None, default None
        Optional parent for a new training run. None creates no files. A saved
        run records source declarations and completed outcomes/priority metrics;
        it is not a model checkpoint or a resumable learner.

    Returns
    -------
    dict[str, object]
        training_summary counts and exact rounding report, plus example mode,
        zero actor_updates and the run_dir string when recording was enabled.
        No trajectory is copied to the host or retained between blocks.

    Raises
    ------
    TypeError, ValueError
        Invalid example settings, schedule, content or collection state.
    OSError
        Requested recording cannot be written. Existing writer errors propagate.

    Notes
    -----
    Host entry point. It verifies installed content, initializes untrained
    weights and compiles collection. The shared PPO initializer also initializes
    critic/Adam arrays, which this example immediately discards. Only actor
    weights are retained, and they never change. With no completed learner
    updates, no opponent snapshots are captured; every game uses current
    self-play. Task rewards and shaping stay separate in each returned block.
    """
    if mode not in _MODES:
        raise ValueError("mode must be Plain, C, RS or C-RS")
    if type(length) is not int or length <= 0:
        raise ValueError("length must be a positive Python integer")
    if type(seed) is not int:
        raise TypeError("seed must be a Python integer, not bool")

    import jax

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines import ppo

    schedule = training.make_training_schedule(
        total_env_steps=total_env_steps,
        num_envs=num_envs,
        curriculum=mode in ("C", "C-RS"),
    )
    model_key = jax.random.fold_in(jax.random.key(seed), 0x4D415050)
    variables = ppo.initialize_ppo(model_key).actor_params
    actor = ppo.make_recurrent_mappo_system(variables, name="Untrained Recurrent MAPPO")
    collection, carry = training.init_training_collection(
        actor,
        variables,
        schedule=schedule,
        seed=seed,
        shaping=mode in ("RS", "C-RS"),
        recording=output_dir is not None,
    )
    destination = (
        marl_bgs.RunWriter(
            output_dir,
            phase="training",
            policies={"team_a": collection.actor, "team_b": collection.opponent},
            details={
                "training_collection": {
                    "mode": mode,
                    "seed": seed,
                    "actor_status": "Untrained; no updates",
                    "schedule": dict(schedule.rounding_report),
                    "content_binding": collection.binding.model_dump(mode="json"),
                }
            },
        )
        if output_dir is not None
        else nullcontext()
    )
    with destination as writer:
        total_rounds = total_env_steps // num_envs
        for _ in range((total_rounds + length - 1) // length):
            carry = training.collect_training_rollout(
                collection, carry, length=length, writer=writer
            )[0]
        summary = training.training_summary(collection, carry)
        summary["mode"] = mode
        summary["actor_updates"] = 0
        if writer is not None:
            summary["run_dir"] = str(writer.run_dir)
    return summary


def _print_summary(summary: dict[str, object]) -> None:
    """Print requested and played exposure from run's small host summary.

    summary is the dictionary returned by run, with one row for every active
    schedule stage and per-map/per-opponent counts. Write only terminal text;
    do not read trajectories or infer learner samples from actor decisions.
    """
    report = cast(dict[str, object], summary["rounding"])
    budgets = cast(tuple[int, ...], report["assigned_round_counts"])
    num_envs = cast(int, report["num_envs"])
    starts = cast(list[int], summary["starts_by_episode_stage"])
    steps = cast(list[int], summary["steps_by_episode_stage"])
    decisions = cast(list[int], summary["actor_decisions_by_episode_stage"])
    print(f"Mode: {summary['mode']}")
    print(
        f"Experience: {summary['env_steps']} of {summary['requested_env_steps']} "
        f"requested transitions across {num_envs} games"
    )
    for stage, rounds in enumerate(budgets):
        print(
            f"Stage {stage}: Requested {rounds * num_envs} steps; "
            f"played {steps[stage]} steps; started {starts[stage]} games; "
            f"made {decisions[stage]} active, living Team A decisions"
        )
    map_steps = cast(list[int], summary["steps_by_map"])
    print("Map Steps: " + ", ".join(f"{i}={n}" for i, n in enumerate(map_steps) if n))
    opponents = cast(list[int], summary["steps_by_opponent"])
    print(f"Opponent Steps: Current={opponents[0]}; historical={sum(opponents[1:])}")
    print(f"Incomplete Games: {summary['incomplete_games']}")
    print("Actor Status: Untrained; no weight updates or historical snapshots")
    print("These counts do not measure learner samples, learning gains or skill.")
    if "run_dir" in summary:
        print(f"Saved Run: {summary['run_dir']}")


def main() -> None:
    """Parse collection options, call run and print its exposure summary.

    The CLI uses the same function as Python callers. Invalid arguments fail
    before collection where possible. Files are opt-in through --output-dir.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=_MODES,
        default="Plain",
        help="Select curriculum (C), score shaping (RS), both, or neither.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=32, help="Positive even number of games."
    )
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=4096,
        help="Real transitions across all games; must divide by --num-envs.",
    )
    parser.add_argument(
        "--length", type=int, default=128, help="Positive output capacity in rounds."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for this run."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Save a training run; omit to create no files."
    )
    args = parser.parse_args()
    _print_summary(run(**vars(args)))


if __name__ == "__main__":
    main()
