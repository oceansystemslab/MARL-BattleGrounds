"""Check the small competitive loops against real saved training and games.

The example must select on common confirmation conditions, fork a winner's
complete learner, preserve its optimizer/live game state, train and add a real
response, and derive mixtures only from a complete measured point table. Helper
checks cover the declared PFSP prior, the 60/20/20 recipe and missing-cell
rejection. CPU budgets are software proofs, not learning or speed evidence.
"""

import json
import sys
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples import competitive_training as example

# pyright: reportPrivateUsage=false
from tests.training_continuation_helpers import capture_runs, fixed_source

import marl_battlegrounds as marl_bgs

type Tree = Any


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def test_history_recipe_and_pfsp_have_explicit_shares_and_unseen_prior() -> None:
    settings, bindings = example.history_recipe([f"actor_{i}" for i in range(6)])
    assert len(bindings) == 8
    assert bindings["ALPHA"] == "tdm-alpha"
    assert bindings["BETA"] == "tdm-beta"
    assert settings["opponents"]["self"] == 0.6
    assert settings["opponents"]["past"] == 0.2
    assert sum(settings["opponents"][name] for name in bindings) == pytest.approx(0.2)
    assert settings["keep_past"] == 20
    assert settings["past_capture_interval"] == 5_000_000
    with pytest.raises(ValueError, match="six"):
        example.history_recipe(["only_one"])
    records = [
        {"kind": "named", "name": name, "cumulative": result}
        for name, result in (
            ("unseen", {"games": 0, "wins": 0, "draws": 0, "losses": 0}),
            ("draw", {"games": 4, "wins": 0, "draws": 4, "losses": 0}),
            ("hard", {"games": 4, "wins": 0, "draws": 0, "losses": 4}),
            ("easy", {"games": 4, "wins": 4, "draws": 0, "losses": 0}),
        )
    ]
    weights = example.pfsp({"members": records}, 100)["opponents"]
    assert isinstance(weights, dict)
    assert sum(cast(dict[str, float], weights).values()) == pytest.approx(1.0)
    assert weights["self"] == 0.2
    assert weights["unseen"] == weights["draw"]
    assert weights["hard"] > weights["unseen"] > weights["easy"]
    assert example.pfsp({"members": []}, 0) == {"opponents": {"self": 1.0}}


def test_empirical_mixture_rejects_missing_cells_and_responds_to_measured_margin() -> (
    None
):
    matrix = np.asarray([[1.0, -2.0], [1.0, -2.0]])
    before = matrix.copy()
    weights = example.empirical_mixture(matrix, iterations=10)
    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_allclose(weights, [1 / 12, 11 / 12])
    for bad in (np.empty((0, 0)), np.ones((2, 3)), np.asarray([[np.nan]])):
        with pytest.raises(ValueError, match="complete finite"):
            example.empirical_mixture(bad)
    with pytest.raises(ValueError, match="positive"):
        example.empirical_mixture(matrix, iterations=0)


def test_complete_competitive_example_preserves_full_state_and_measures_every_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "ff_ippo")
    starts, ends = capture_runs(monkeypatch)
    root = tmp_path / "competitive"
    report = example.run(root, updates=1)
    assert report == json.loads((root / "decisions.json").read_text())
    assert report["env_steps_per_training_budget"] == 4
    parent_root = Path(report["winner_checkpoint"]).parent.parent
    parent = ends[parent_root]
    child = starts[root / "pbt_child"][1]
    assert int(parent.completed_updates) == 1
    for field in (
        "actor_opt_state",
        "critic_opt_state",
        "critic_params",
        "critic_memory",
        "value_norm",
        "shuffle_root",
    ):
        _equal(getattr(parent, field), getattr(child, field))
    _equal(parent.carry.state, child.carry.state)
    _equal(parent.carry.memory, child.carry.memory)
    _equal(
        parent.carry.history.current_variables, child.carry.history.current_variables
    )
    assert int(child.completed_updates) == int(parent.completed_updates)
    assert int(ends[root / "pbt_child"].completed_updates) == 2
    declaration = json.loads((root / "pbt_child" / "run_details.json").read_text())
    changes = declaration["continuation"]["changes"]
    assert changes["learning_rate"]["actor_lr"] == 0.0001
    assert changes["ppo"]["entropy_coefficient"] == 0.02
    for name, size, mixture_key in (
        ("population_before", 2, "mixture_before"),
        ("population_after", 3, "mixture_after"),
    ):
        record = json.loads((root / name / "payoffs.json").read_text())
        assert len(record["cells"]) == size * size
        assert len(record["system_ids"]) == size
        assert sum(cell["row"] == cell["column"] for cell in record["cells"]) == size
        np.testing.assert_allclose(
            list(report[mixture_key].values()),
            example.empirical_mixture(np.asarray(record["matrix"])),
        )
        for cell in record["cells"]:
            saved = marl_bgs.load_results(cell["run_dir"])
            assert saved.status == "complete"
            assert saved.metadata["spawn_balance"]["paired_complete"]
            assert len(saved.table("episodes")["episode_id"]) == 2
            assert not saved.table("priority_metrics")
    response = ends[root / "response"]
    continued = starts[root / "response_child"][1]
    _equal(response.actor_opt_state, continued.actor_opt_state)
    _equal(response.carry.state, continued.carry.state)
    assert set(report["mixture_after"]) == {"random", "alpha", "response"}
    assert int(ends[root / "response_child"].completed_updates) == 2
    selected = marl_bgs.load_results(report["selected_evaluation"])
    assert selected.status == "complete"
    assert selected.metadata["spawn_balance"]["paired_complete"]
    with pytest.raises(FileExistsError):
        example.run(root, updates=1)


def test_cli_rejects_duplicate_alias_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "duplicate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "competitive_training",
            "--output-dir",
            str(root),
            "--opponent",
            "same=random",
            "--opponent",
            "same=tdm-alpha",
        ],
    )
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert not root.exists()
    assert (
        example._cell_name("My Actor!", "Other/Team", 2, 3)
        == "cell_02_my_actor__03_other_team"
    )
