"""Scientific estimator, pairing, weighting and unavailable-uncertainty proofs."""

# Private reduced-replicate plumbing keeps semantic tests bounded; the public
# 5,000-replicate entry point receives its own genuine execution proof below.
# pyright: reportPrivateUsage=false

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from marl_battlegrounds.evaluation import tournament_statistics as stats
from marl_battlegrounds.evaluation.tournament_schedule import (
    TournamentMatch,
    build_tournament_schedule,
)

type Schedule = tuple[TournamentMatch, ...]
type FloatArray = NDArray[np.float64]


def _outcomes(
    schedule: Schedule, block_patterns: tuple[tuple[int, int], ...]
) -> dict[int, int]:
    """Patterns describe canonical first-policy W/D/L as 1/3/2 on both sides."""
    result: dict[int, int] = {}
    for match in schedule:
        first_side = match.team_a < match.team_b
        outcome = block_patterns[(match.block_id - 1) % len(block_patterns)][
            int(not first_side)
        ]
        result[match.episode_id] = (
            outcome if first_side or outcome == 3 else 3 - outcome
        )
    return result


def _summary(
    schedule: Schedule,
    outcomes: Mapping[int, int],
    *,
    replicates: int = 80,
    seed: int = 42,
    weights: Mapping[str, float] | None = None,
) -> stats.TournamentStatistics:
    return stats._summarize(
        schedule, outcomes, seed=seed, opponent_weights=weights, replicates=replicates
    )


def _rows(result: stats.TournamentStatistics) -> dict[str, stats.ResultRow]:
    return {str(row["policy"]): row for row in result.tournament_results}


def test_davidson_gradient_and_prior_match_explicit_probability_model() -> None:
    contrasts = cast(FloatArray, stats.linalg.helmert(3, full=False)).T
    pairs = np.asarray(((0, 1), (0, 2), (1, 2)), np.int64)
    counts = np.asarray(((10, 3, 1), (2, 8, 3), (4, 0, 6)), np.float64)
    parameters = np.asarray((0.4, -0.7, 0.9))
    loss, gradient = stats._objective(parameters, counts, contrasts, pairs)
    strengths = contrasts @ parameters[:-1]
    probabilities: list[FloatArray] = []
    for a, b in pairs:
        raw = np.asarray(
            (
                np.exp(strengths[a]),
                np.exp(parameters[-1] + (strengths[a] + strengths[b]) / 2),
                np.exp(strengths[b]),
            )
        )
        probabilities.append(raw / raw.sum())
    expected = -np.sum(counts * np.log(probabilities)) + (
        np.sum(strengths**2) + parameters[-1] ** 2
    ) / (2 * np.log(10.0) ** 2)
    assert loss == pytest.approx(expected, abs=1e-12)
    step = 1e-5
    numeric = np.asarray(
        [
            (
                stats._objective(parameters + delta, counts, contrasts, pairs)[0]
                - stats._objective(parameters - delta, counts, contrasts, pairs)[0]
            )
            / (2 * step)
            for delta in step * np.eye(3)
        ]
    )
    np.testing.assert_allclose(gradient, numeric, atol=1e-8, rtol=1e-8)
    assert stats._objective(np.zeros(3), counts, contrasts, pairs)[0] == pytest.approx(
        counts.sum() * np.log(3)
    )


@pytest.mark.parametrize(
    "patterns",
    [
        ((1, 1), (1, 3), (1, 2), (3, 2), (2, 2)),
        ((3, 3),) * 8 + ((1, 3), (2, 3)),
        ((1, 1),) * 19 + ((1, 2),),
    ],
    ids=("ordinary", "draw-heavy", "near-boundary"),
)
def test_finite_centered_ordinary_drawheavy_and_boundary_estimates(
    patterns: tuple[tuple[int, int], ...],
) -> None:
    schedule = build_tournament_schedule(("a", "b", "c"), (1, 9), episodes_per_pair=80)
    result = _summary(schedule, _outcomes(schedule, patterns))
    assert sum(
        cast(float, row["elo"]) for row in result.tournament_results
    ) == pytest.approx(3600)
    for row in result.tournament_results:
        assert np.isfinite(cast(float, row["elo"]))
        assert sum(
            cast(float, row[key]) for key in ("win_rate", "draw_rate", "loss_rate")
        ) == pytest.approx(1)
        assert row["expected_score"] == pytest.approx(
            cast(float, row["win_rate"]) + cast(float, row["draw_rate"]) / 2
        )
        assert row["elo_interval_status"] == "Available"
        assert row["expected_score_interval_status"] == "Available"
        assert (
            0
            <= cast(float, row["expected_score_ci_low"])
            <= cast(float, row["expected_score_ci_high"])
            <= 1
        )


def test_input_order_and_global_side_swap_leave_every_report_field_unchanged() -> None:
    schedule = build_tournament_schedule(("c", "a", "b"), (7, 3), episodes_per_pair=40)
    outcomes = _outcomes(schedule, ((1, 1), (3, 2), (2, 2), (1, 3), (1, 2)))
    baseline = _summary(schedule, outcomes)
    assert baseline == _summary(schedule[::-1], dict(reversed(tuple(outcomes.items()))))
    swapped = tuple(replace(m, team_a=m.team_b, team_b=m.team_a) for m in schedule)
    opposite = {
        key: value if value == 3 else 3 - value for key, value in outcomes.items()
    }
    assert baseline == _summary(swapped, opposite)


@pytest.mark.parametrize("pattern", (((3, 3),), ((1, 2),), ((1, 1),)))
def test_constant_paired_data_has_no_invented_confidence(
    pattern: tuple[tuple[int, int], ...],
) -> None:
    schedule = build_tournament_schedule(("a", "b"), (1,), episodes_per_pair=40)
    result = _summary(schedule, _outcomes(schedule, pattern))
    for row in result.tournament_results:
        for metric in ("elo", "expected_score"):
            assert row[f"{metric}_interval_status"] == "Insufficient Variation"
            assert row[f"{metric}_ci_low"] is row[f"{metric}_ci_high"] is None
    if pattern == ((1, 2),):
        # Every episode varies, but each genuinely independent pair is exactly
        # .5. An episode bootstrap would manufacture unsupported uncertainty.
        assert result.tournament_results[0]["expected_score"] == 0.5


def test_single_block_reports_insufficiency_and_does_not_hide_point_results() -> None:
    schedule = build_tournament_schedule(("a", "b"), (1,), episodes_per_pair=2)
    result = _summary(schedule, _outcomes(schedule, ((1, 3),)))
    first = result.tournament_results[0]
    assert first["expected_score"] == 0.75
    assert cast(float, first["elo"]) > 1200
    assert (
        first["elo_interval_status"]
        == first["expected_score_interval_status"]
        == "Insufficient Blocks"
    )


def test_elo_can_vary_when_expected_score_is_constant() -> None:
    schedule = build_tournament_schedule(("a", "b"), (1,), episodes_per_pair=40)
    schedule = tuple(
        replace(m, bootstrap_group=f"unit-{(m.block_id - 1) // 2}") for m in schedule
    )
    outcomes = _outcomes(schedule, ((1, 1), (1, 2), (1, 1), (3, 3)))
    # Each four-episode unit scores .75, but 3W/1L and 2W/2D imply different
    # decisive odds. Suppressing both intervals together would lose evidence.
    first = _summary(schedule, outcomes).tournament_results[0]
    assert first["expected_score"] == 0.75
    assert first["expected_score_interval_status"] == "Insufficient Variation"
    assert first["elo_interval_status"] == "Available"


@pytest.mark.parametrize(
    "probabilities",
    ((0.55, 0.3, 0.15), (0.1, 0.8, 0.1), (0.94, 0.05, 0.01)),
    ids=("ordinary", "draw-heavy", "near-boundary"),
)
@pytest.mark.parametrize("data_seed", (67, 703), ids=("diagnostic", "fresh"))
def test_small_synthetic_coverage_check_with_perfect_within_pair_dependence(
    probabilities: tuple[float, float, float],
    data_seed: int,
    record_property: Callable[[str, object], None],
) -> None:
    schedule = build_tournament_schedule(("a", "b"), (1,), episodes_per_pair=100)
    win, draw, loss = probabilities
    truth_score = win + draw / 2
    truth_elo = 1200 + 200 * np.log(win / loss) / np.log(10)
    rng = np.random.Generator(np.random.PCG64(data_seed))
    available = {"elo": 0, "expected_score": 0}
    covered = {"elo": 0, "expected_score": 0}
    for trial in range(20):
        # One categorical outcome per independent block, duplicated on both
        # sides. Individual episodes are therefore not independent samples.
        block_outcomes = rng.choice((1, 3, 2), size=50, p=probabilities)
        outcomes = {
            m.episode_id: int(value if m.team_a == "a" or value == 3 else 3 - value)
            for m in schedule
            for value in (block_outcomes[m.block_id - 1],)
        }
        row = _summary(
            schedule, outcomes, replicates=120, seed=trial
        ).tournament_results[0]
        if not np.any(block_outcomes == 1) or not np.any(block_outcomes == 2):
            assert row["elo_interval_status"] == "Insufficient Variation"
        for metric, truth in (("elo", truth_elo), ("expected_score", truth_score)):
            if row[f"{metric}_interval_status"] == "Available":
                available[metric] += 1
                covered[metric] += int(
                    cast(float, row[f"{metric}_ci_low"])
                    <= truth
                    <= cast(float, row[f"{metric}_ci_high"])
                )
    # A bounded regression sanity check, not a claim of proven 95% coverage.
    # The intentionally broad floor tolerates Monte Carlo error and discreteness
    # but catches grossly narrow or displaced intervals in all three regimes.
    record_property("available", available)
    record_property("covered", covered)
    for metric in available:
        if metric == "expected_score" or probabilities[2] >= 0.1:
            assert available[metric] >= 16, (probabilities, available, covered)
        else:
            assert 0 < available[metric] < 16, (probabilities, available, covered)
        assert covered[metric] >= 0.7 * available[metric], (
            probabilities,
            available,
            covered,
        )


def test_separated_groups_block_joint_elo_intervals_even_with_individual_losses() -> (
    None
):
    schedule = build_tournament_schedule(
        ("a", "b", "c", "d"), (1,), episodes_per_pair=20
    )
    outcomes: dict[int, int] = {}
    for match in schedule:
        pair = tuple(sorted((match.team_a, match.team_b)))
        winner = (
            pair[match.block_id % 2] if pair in (("a", "b"), ("c", "d")) else pair[0]
        )
        outcomes[match.episode_id] = 1 if match.team_a == winner else 2
    result = _summary(schedule, outcomes)
    assert result.metadata["decisive_win_graph_strongly_connected"] is False
    for row in result.tournament_results:
        assert cast(int, row["wins"]) > 0 and cast(int, row["losses"]) > 0
        assert row["elo_interval_status"] == "Insufficient Variation"
        assert row["elo_ci_low"] is row["elo_ci_high"] is None
        assert row["expected_score_interval_status"] == "Available"


def test_declared_larger_coupling_is_retained_and_preserves_cell_budgets() -> None:
    schedule = build_tournament_schedule(("a", "b"), (1, 2), episodes_per_pair=40)
    # Same ten independent coordinates across two maps. Opposite map outcomes
    # cancel exactly inside each coordinate, although each map alone varies.
    schedule = tuple(
        replace(m, bootstrap_group=f"coordinate-{(m.block_id - 1) % 10}")
        for m in schedule
    )
    outcomes = {
        m.episode_id: (1 if m.team_a == "a" else 2)
        if (m.block_id % 2 == (m.map_id % 2))
        else (2 if m.team_a == "a" else 1)
        for m in schedule
    }
    population = stats._population(schedule, outcomes, None)
    assert [group.shape for group in population.strata] == [(10, 2)]
    rng = np.random.Generator(np.random.PCG64(82))
    for _ in range(20):
        counts = stats._sample_counts(population, rng)
        np.testing.assert_array_equal(counts.sum(axis=1), (20, 20))
        assert counts[:, 0].sum() == counts[:, 2].sum() == 20
    result = _summary(schedule, outcomes)
    assert (
        result.tournament_results[0]["expected_score_interval_status"]
        == "Insufficient Variation"
    )
    independent = tuple(replace(m, bootstrap_group=None) for m in schedule)
    assert result.tournament_results[0]["independent_blocks"] == 10
    assert result.matchup_results[0]["independent_blocks"] == 10
    assert all(row["independent_blocks"] == 10 for row in result.map_results)
    assert (
        _summary(independent, outcomes).tournament_results[0][
            "expected_score_interval_status"
        ]
        == "Available"
    )


def test_weighted_population_and_fractional_worst_twenty_percent() -> None:
    schedule = build_tournament_schedule(("a", "b", "c"), (1, 2), episodes_per_pair=20)
    # For a, b-cell scores are (0,1), c-cell scores are (1,.5).
    outcomes: dict[int, int] = {}
    for match in schedule:
        pair = tuple(sorted((match.team_a, match.team_b)))
        if pair == ("a", "c") and match.map_id == 2:
            value = 3
        else:
            winner = "b" if pair == ("a", "b") and match.map_id == 1 else pair[0]
            value = 1 if match.team_a == winner else 2
        outcomes[match.episode_id] = value
    result = _summary(schedule, outcomes, weights={"a": 1, "b": 1, "c": 3})
    first = _rows(result)["a"]
    assert first["expected_score"] == pytest.approx(0.25 * 0.5 + 0.75 * 0.75)
    assert first["win_rate"] == pytest.approx(0.5)
    assert first["draw_rate"] == pytest.approx(0.375)
    assert first["loss_rate"] == pytest.approx(0.125)
    # Worst cell mass .125 at score0, then only .075 of the .375 cell at .5.
    assert first["worst_20_percent_expected_score"] == pytest.approx(0.1875)
    by_map = {row["map_id"]: row for row in result.map_results if row["policy"] == "a"}
    assert by_map[1]["expected_score"] == 0.75
    assert by_map[2]["expected_score"] == 0.625
    assert stats._weighted_tail(
        np.arange(55, dtype=np.float64) / 54, np.full(55, 1 / 55)
    ) == pytest.approx(np.mean(np.arange(11) / 54))


def test_matchup_intervals_cannot_be_replaced_by_a_favorable_pooled_score() -> None:
    schedule = build_tournament_schedule(("a", "b", "c"), (1,), episodes_per_pair=100)
    outcomes: dict[int, int] = {}
    for match in schedule:
        pair = tuple(sorted((match.team_a, match.team_b)))
        first_wins = match.block_id % 10 < (8 if pair == ("a", "b") else 2)
        winner = pair[int(not first_wins)]
        outcomes[match.episode_id] = 1 if match.team_a == winner else 2
    result = _summary(schedule, outcomes, weights={"a": 1, "b": 99, "c": 1})
    first = _rows(result)["a"]
    assert cast(float, first["expected_score_ci_low"]) > 0.5
    by_opponent = {
        str(row["opponent"]): row
        for row in result.matchup_results
        if row["policy"] == "a"
    }
    assert cast(float, by_opponent["b"]["expected_score_ci_low"]) > 0.5
    assert cast(float, by_opponent["c"]["expected_score_ci_high"]) < 0.5
    assert first["independent_blocks"] == 100
    assert all(row["independent_blocks"] == 50 for row in by_opponent.values())
    assert result.matchup_results == _summary(schedule, outcomes).matchup_results
    reverse = next(
        row
        for row in result.matchup_results
        if row["policy"] == "b" and row["opponent"] == "a"
    )
    assert reverse["expected_score_ci_low"] == pytest.approx(
        1 - cast(float, by_opponent["b"]["expected_score_ci_high"])
    )
    assert reverse["expected_score_ci_high"] == pytest.approx(
        1 - cast(float, by_opponent["b"]["expected_score_ci_low"])
    )


def test_weight_validation_is_reusable_before_execution_and_scale_safe() -> None:
    assert stats.validate_opponent_weights(("a", "b"), None) == {"a": 1.0, "b": 1.0}
    with pytest.raises(ValueError, match="weight"):
        stats.validate_opponent_weights(("a", "b"), {"a": 1, "b": np.inf})
    schedule = build_tournament_schedule(("a", "b"), (1,), episodes_per_pair=4)
    outcomes = _outcomes(schedule, ((1, 3),))
    result = _summary(schedule, outcomes, weights={"a": 1e308, "b": 1e-308})
    assert [row["expected_score"] for row in result.tournament_results] == [0.75, 0.25]


@pytest.mark.parametrize(
    "kind",
    (
        "missing",
        "extra",
        "ongoing",
        "duplicate",
        "side",
        "seed",
        "budget",
        "coupling",
        "hidden-dependence",
    ),
)
def test_malformed_or_partial_tournament_cannot_be_qualified(kind: str) -> None:
    schedule = build_tournament_schedule(("a", "b", "c"), (1, 2), episodes_per_pair=8)
    outcomes = _outcomes(schedule, ((1, 3), (2, 1)))
    if kind == "missing":
        outcomes.pop(1)
    elif kind == "extra":
        outcomes[999] = 1
    elif kind == "ongoing":
        outcomes[1] = 0
    elif kind == "duplicate":
        schedule += (schedule[0],)
    elif kind == "side":
        schedule = (replace(schedule[0], team_b="a"), *schedule[1:])
    elif kind == "seed":
        schedule = (replace(schedule[0], seed_id=999), *schedule[1:])
    elif kind == "budget":
        schedule = schedule[2:]
        outcomes = {key: value for key, value in outcomes.items() if key > 2}
    elif kind == "coupling":
        schedule = (replace(schedule[0], bootstrap_group="coupled"), *schedule[1:])
    else:
        schedule = tuple(
            replace(m, seed_id=1) if m.block_id == 2 else m for m in schedule
        )
    with pytest.raises(ValueError):
        _summary(schedule, outcomes)


@pytest.mark.parametrize("weights", ({"a": 1}, {"a": 0, "b": 1}, {"a": np.nan, "b": 1}))
def test_invalid_opponent_weights_are_not_silently_normalized(
    weights: dict[str, float],
) -> None:
    schedule = build_tournament_schedule(("a", "b"), (1,), episodes_per_pair=4)
    with pytest.raises(ValueError, match="weight"):
        _summary(schedule, _outcomes(schedule, ((1, 3),)), weights=weights)


@pytest.mark.parametrize("fail_at", (1, 3), ids=("point", "bootstrap"))
def test_failed_fit_is_loud_and_no_bootstrap_replicate_is_discarded(
    monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    original = cast(Callable[..., object], stats.optimize.minimize)
    calls = 0

    def fit(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            return SimpleNamespace(
                x=np.zeros(2), success=False, message="injected failure"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(stats.optimize, "minimize", fit)
    schedule = build_tournament_schedule(("a", "b"), (1,), episodes_per_pair=8)
    with pytest.raises(RuntimeError, match="injected failure"):
        _summary(schedule, _outcomes(schedule, ((1, 1), (2, 2), (1, 3), (3, 2))))
    assert calls == fail_at


def test_public_five_thousand_replicates_are_real_and_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = build_tournament_schedule(
        tuple(f"method-{i:02}" for i in range(12)), (17, 20, 25, 35, 39)
    )
    rng = np.random.Generator(np.random.PCG64(112))
    outcomes = {
        match.episode_id: int(rng.choice((1, 2, 3), p=(0.4, 0.3, 0.3)))
        for match in schedule
    }
    original = stats._fit
    calls = 0

    def counted(
        counts: FloatArray,
        contrasts: FloatArray,
        pairs: NDArray[np.int64],
        initial: FloatArray,
        *,
        context: str,
    ) -> FloatArray:
        nonlocal calls
        calls += 1
        return original(counts, contrasts, pairs, initial, context=context)

    monkeypatch.setattr(stats, "_fit", counted)
    result = stats.summarize_tournament(schedule, outcomes, seed=113)
    assert calls == 5001
    assert result.metadata["bootstrap_replicates"] == 5000
    assert result.tournament_results[0]["elo_interval_status"] == "Available"
    # A second reduced request repeats the same first resampling stream exactly.
    population = stats._population(schedule, outcomes, None)
    first = stats._sample_counts(population, np.random.Generator(np.random.PCG64(113)))
    np.testing.assert_array_equal(
        first,
        stats._sample_counts(population, np.random.Generator(np.random.PCG64(113))),
    )
