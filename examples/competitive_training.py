"""Run small league, population-based training and response-population examples.

From an installed checkout, run ``JAX_PLATFORMS=cpu python
examples/competitive_training.py --output-dir artifacts/competitive_demo``.
The default performs real short FF-IPPO updates, common-panel checkpoint
selection, full-checkpoint continuation and measured response-population games.
Two-step evaluation games are an explicit software-demo condition; confirmation
uses normal 300-step games. None of these tiny budgets proves learned skill.

Python callers may supply named Systems, Policies or ordinary method references
to run(). The loops below belong to the researcher. Training, persistence,
validation and games use their existing library owners; no league manager or
new learner framework is introduced. Output folders must be new.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from marl_battlegrounds import Policy, System


def history_recipe(checkpoints: Sequence[str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return the requested 60/20/20 example settings and named method bindings.

    checkpoints must contain exactly six saved actor/checkpoint paths or trusted
    factory references. The eight frozen opponents (ALPHA, BETA and these six)
    each get 2.5%; current self gets 60%; rolling past copies share 20%. Save a
    past copy after each 5,000,000 real environment transitions and keep the latest
    twenty eligible copies. Capture uses the first completed update at or after
    the requested gap; the saved copy records its actual step. Before a copy
    exists, its share uses current self. Return TrainConfig keyword settings
    and the train(opponents=...) map.
    This recipe is an example, not a required population size or mixture.
    """
    if len(checkpoints) != 6:
        raise ValueError("This example recipe needs six named checkpoints")
    bindings = {"ALPHA": "tdm-alpha", "BETA": "tdm-beta"}
    bindings.update(
        {f"checkpoint_{i + 1}": value for i, value in enumerate(checkpoints)}
    )
    return {
        "opponents": {"self": 0.6, "past": 0.2, **dict.fromkeys(bindings, 0.025)},
        "keep_past": 20,
        "past_capture_interval": 5_000_000,
    }, bindings


def pfsp(stats: dict[str, Any], env_steps: int) -> dict[str, object]:
    """Give harder named opponents more future games, keeping 20% current self.

    stats is the public matchmaking callback record. Each named member has
    cumulative games/wins/draws/losses. A draw counts as half a win; an unseen
    member starts at an explicitly chosen 50% prior, not an observed result.
    Named weight is 0.01 + (1 - win fraction)**2, normalized into the remaining
    80%. With no named members, return all self-play. env_steps is the completed
    transition count and is unused by this simple rule. Return only the next
    opponent rule; running games and the member pool stay unchanged.
    """
    del env_steps
    weights: dict[str, float] = {}
    for member in stats["members"]:
        if member["kind"] != "named":
            continue
        result = member["cumulative"]
        games = result["games"]
        win_fraction = (
            (result["wins"] + 0.5 * result["draws"]) / games if games else 0.5
        )
        weights[member["name"]] = 0.01 + (1.0 - win_fraction) ** 2
    total = sum(weights.values())
    return {
        "opponents": {
            "self": 0.2,
            **{name: 0.8 * value / total for name, value in weights.items()},
        }
        if total
        else {"self": 1.0}
    }


def empirical_mixture(payoffs: np.ndarray, *, iterations: int = 200) -> np.ndarray:
    """Return a small finite fictitious-play opponent mixture on measured games.

    payoffs is a finite float (N,N) table: row method's mean native point margin
    against column method, including actually played diagonal cells. Missing
    cells are rejected. Start both players with one count per member, then add
    one row best response and one column best response to their current empirical
    frequencies per iteration. Exact ties choose the first listed member.
    iterations is a positive host-loop count. Return N normalized column counts.
    This finite calculation is neither an equilibrium certificate nor a claim
    that a neural response is a best response. Inputs are not changed.
    """
    values = np.asarray(payoffs, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[0] != values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("Payoffs must be a complete finite square table")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    row_counts = np.ones(values.shape[0], np.float64)
    column_counts = np.ones_like(row_counts)
    for _ in range(iterations):
        row = int(np.argmax(values @ (column_counts / column_counts.sum())))
        column = int(np.argmin((row_counts / row_counts.sum()) @ values))
        row_counts[row] += 1
        column_counts[column] += 1
    return column_counts / column_counts.sum()


def _cell_name(row: str, column: str, i: int, j: int) -> str:
    """Label an ordered pair with readable aliases and stable population indices.

    row and column are display aliases; i and j are their zero-based positions.
    Keep at most sixteen safe snake_case characters per alias. Exact names and
    System identities remain in payoffs.json and the evaluator records.
    """
    names = [
        re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:16].rstrip("_") or "method"
        for name in (row, column)
    ]
    return f"cell_{i:02d}_{names[0]}__{j:02d}_{names[1]}"


def measured_payoffs(
    population: Mapping[str, System | Policy | str],
    output_dir: Path,
    *,
    num_envs: int = 2,
    seed_pairs: int = 1,
    max_steps: int = 2,
    seed: int = 917,
) -> np.ndarray:
    """Play every ordered population pair and save its raw-result evidence.

    population is an ordered nonempty name-to-method map accepted by evaluate.
    Keep supplied methods fixed throughout this call; identity changes fail.
    output_dir is a new folder. num_envs bounds parallel games; seed_pairs gives
    pairs per cell. Each cell uses map 42 and both spawn ends with common seed.
    max_steps is the declared game horizon (2 is only a software demonstration).
    Return float64 (N,N) native Team A minus Team B mean points. Diagonal cells
    are measured, not assigned zero. All games, exact system IDs and actual
    spawn records remain in ordinary evaluator folders; payoffs.json links them.
    """
    import marl_battlegrounds as marl_bgs

    if not population:
        raise ValueError("A population must contain at least one method")
    output_dir.mkdir(parents=True, exist_ok=False)
    names = list(population)
    matrix = np.empty((len(names), len(names)), np.float64)
    identities: dict[str, str] = {}
    cells = []
    for i, row in enumerate(names):
        for j, column in enumerate(names):
            result = marl_bgs.evaluate(
                population[row],
                population[column],
                maps=[42],
                num_episodes=2 * seed_pairs,
                num_envs=num_envs,
                spawn_mode="paired",
                seed=seed,
                max_steps=max_steps,
                metrics="none",
                output_dir=output_dir / _cell_name(row, column, i, j),
            )
            games = result.table("episodes")
            if (
                result.status != "complete"
                or len(games.get("team_a_score", ())) != 2 * seed_pairs
            ):
                raise ValueError(
                    "Every population cell needs its declared completed games"
                )
            margin = np.asarray(games["team_a_score"], np.float64) - np.asarray(
                games["team_b_score"], np.float64
            )
            matrix[i, j] = float(margin.mean())
            coverage = result.metadata["spawn_balance"]
            for name, field in ((row, "system_id"), (column, "opponent_id")):
                identity = coverage[field]
                if (
                    not isinstance(identity, str)
                    or identities.get(name, identity) != identity
                ):
                    raise ValueError(
                        "A population method changed identity between cells"
                    )
                identities[name] = identity
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "mean_point_margin": matrix[i, j],
                    "run_dir": str(result.run_dir),
                    "spawn_balance": coverage,
                }
            )
    (output_dir / "payoffs.json").write_text(
        json.dumps(
            {
                "names": names,
                "system_ids": identities,
                "matrix": matrix.tolist(),
                "cells": cells,
                "map": 42,
                "seed": seed,
                "seed_pairs": seed_pairs,
                "max_steps": max_steps,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return matrix


def run(
    output_dir: Path,
    *,
    opponents: Mapping[str, System | Policy | str] | None = None,
    num_envs: int = 2,
    updates: int = 2,
    rollout_length: int = 2,
    seed: int = 42,
    evaluation_max_steps: int = 2,
) -> dict[str, Any]:
    """Run one small, complete league/PBT/response-population cycle.

    Parameters
    ----------
    output_dir : Path
        New root for all training, confirmation, game and decision records.
    opponents : mapping or None, default=None
        Named frozen Systems, Policies or method references accepted by train.
        None uses Random and ALPHA. Use at least one nonreserved name. Researcher
        host methods retain their ordinary lifetime and reproducibility limits.
    num_envs, updates, rollout_length : int, defaults=2, 2, 2
        Positive even parallel batch, real updates per fresh learner, and rounds
        per update. One training budget is their product in environment steps.
        CPU defaults are tiny. Confirmation uses a separate batch capped at 32.
        Use an allowed batch such as 32 on GPU.
    seed : int, default=42
        First learner seed; the second candidate uses seed+1. Evaluation seeds
        and confirmation roots are separate, fixed values in this example.
    evaluation_max_steps : int, default=2
        Explicit horizon for empirical population and final demonstration games.
        Confirmation always uses the evaluator's normal 300-step games.

    Returns
    -------
    dict
        Also saved in decisions.json: full winner/child checkpoint paths, the
        selected actor's final evaluation, and measured mixtures before/after
        adding one finitely trained response. Counts never imply competence.

    Notes
    -----
    PBT selects two candidates on one declared confirmation panel, extends the
    winner's full checkpoint, and changes only future rates/entropy. It preserves
    optimizer state, current games and absolute clocks. PSRO here means a small
    response-population loop: measure every cell, compute a finite empirical
    mixture, train a fresh response, add its frozen actor, and measure again.
    It does not claim an exact best response or equilibrium. Training starts
    fresh for that response so its first games use the new mixture. A subsequent
    full-checkpoint extension demonstrates appending the new opponent; only new
    games use the updated mixture. The script is not an automatic search service.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training.analysis import select_checkpoint
    from marl_battlegrounds.training.validation import create_panel, validate_checkpoint

    members = (
        dict(opponents)
        if opponents is not None
        else {"random": "random", "alpha": "tdm-alpha"}
    )
    if not members or set(members) & {"self", "past", "response"}:
        raise ValueError(
            "Use named opponents; self, past and response are reserved here"
        )
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer")
    output_dir.mkdir(parents=True, exist_ok=False)
    steps = num_envs * updates * rollout_length
    config = training.TrainConfig(
        method="ff_ippo",
        num_envs=num_envs,
        total_env_steps=steps,
        seed=seed,
        opponents={"self": 0.2, **dict.fromkeys(members, 0.8 / len(members))},
        keep_past=0,
        metrics="none",
        verbose=False,
        ppo=PPOConfig(rollout_length=rollout_length, epochs=1, minibatches=1, groups=1),
    )
    # Declare common selection conditions before either candidate is trained.
    panel = create_panel(
        opponents=["random"],
        output_dir=output_dir / "panel",
        roots={"routine": 918, "confirmation": 919},
    )
    candidates = [
        training.train(
            replace(config, seed=seed + i),
            output_dir=output_dir / f"candidate_{i}",
            opponents=members,
            matchmaking=pfsp,
        )
        for i in range(2)
    ]
    confirmed = [
        {
            **validate_checkpoint(
                candidate.final_checkpoint,
                panel,
                output_dir=output_dir / f"confirmation_{i}",
                purpose="confirmation",
                seed_pairs=1,
                num_envs=min(num_envs, 32),
                maps=[42],
            ),
            "full_checkpoint": str(candidate.final_checkpoint),
            "actor_path": str(candidate.final_actor),
        }
        for i, candidate in enumerate(candidates)
    ]
    winner = select_checkpoint(confirmed)
    child = training.extend_training(
        winner["full_checkpoint"],
        additional_env_steps=steps,
        output_dir=output_dir / "pbt_child",
        opponents=members,
        matchmaking=pfsp,
        changes={
            "learning_rate": {
                "kind": "constant",
                "actor_lr": 0.0001,
                "critic_lr": 0.0001,
            },
            "ppo": {"entropy_coefficient": 0.02},
        },
    )
    selected_actor = training.load_system(winner["actor_path"])
    final = marl_bgs.evaluate(
        selected_actor,
        "tdm-alpha",
        maps=[47],
        num_episodes=2,
        num_envs=num_envs,
        spawn_mode="paired",
        seed=920,
        max_steps=evaluation_max_steps,
        metrics="none",
        output_dir=output_dir / "selected_evaluation",
    )
    before = measured_payoffs(
        members,
        output_dir / "population_before",
        num_envs=num_envs,
        max_steps=evaluation_max_steps,
    )
    mixture = dict(zip(members, empirical_mixture(before).tolist(), strict=True))
    response = training.train(
        replace(config, seed=seed + 2, opponents=mixture),
        output_dir=output_dir / "response",
        opponents=members,
    )
    expanded = {**members, "response": training.load_system(response.final_actor)}
    after = measured_payoffs(
        expanded,
        output_dir / "population_after",
        num_envs=num_envs,
        max_steps=evaluation_max_steps,
    )
    next_mixture = dict(zip(expanded, empirical_mixture(after).tolist(), strict=True))
    next_round = training.extend_training(
        response.final_checkpoint,
        additional_env_steps=steps,
        output_dir=output_dir / "response_child",
        opponents=expanded,
        changes={"opponents": next_mixture},
    )
    report = {
        "winner_checkpoint_id": winner["checkpoint_id"],
        "winner_checkpoint": winner["full_checkpoint"],
        "winner_actor": winner["actor_path"],
        "pbt_child": str(child.final_checkpoint),
        "selected_evaluation": str(final.run_dir),
        "mixture_before": mixture,
        "response_actor": str(response.final_actor),
        "mixture_after": next_mixture,
        "response_child": str(next_round.final_checkpoint),
        "env_steps_per_training_budget": steps,
        "evaluation_max_steps": evaluation_max_steps,
        "confirmation": {"maps": [42], "seed_pairs": 1, "root": 919, "max_steps": 300},
        "fictitious_play_iterations": 200,
    }
    (output_dir / "decisions.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    """Parse the small example's command line and print its saved decision paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--opponent", action="append", default=[], metavar="NAME=REFERENCE"
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-max-steps", type=int, default=2)
    args = parser.parse_args()
    opponents: dict[str, str] = {}
    for item in args.opponent:
        if "=" not in item:
            parser.error("Each --opponent needs NAME=REFERENCE")
        name, reference = item.split("=", 1)
        if not name or not reference or name in opponents:
            parser.error(
                "Opponent names must be nonempty and distinct, with a reference"
            )
        opponents[name] = reference
    print(
        json.dumps(
            run(
                args.output_dir,
                opponents=opponents or None,
                num_envs=args.num_envs,
                updates=args.updates,
                rollout_length=args.rollout_length,
                seed=args.seed,
                evaluation_max_steps=args.evaluation_max_steps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
