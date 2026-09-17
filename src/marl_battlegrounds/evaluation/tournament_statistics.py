"""Qualify complete paired tournaments and estimate ratings with draw outcomes.

summarize_tournament validates equal pair/map budgets and resamples whole seed
blocks 5,000 times. Its Davidson rating model includes draws. Population rates
use equal map weighting and declared opponent weights. Intervals describe
repeated evaluation of these fixed systems/maps, not independent training runs.
All fitting and resampling use host NumPy/SciPy; no simulator runs here.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from itertools import combinations
from numbers import Integral
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, special  # pyright: ignore[reportMissingTypeStubs]

from marl_battlegrounds.evaluation.tournament_schedule import (
    TournamentMatch,
    valid_policy_name,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type ResultRow = dict[str, str | int | float | None]

_BOOTSTRAP_REPLICATES = 5000
_GRADIENT_TOLERANCE = 1e-4
_MAX_ITERATIONS = 1000
_LOG_TEN = float(np.log(10.0))
ELO_CENTER = 1200.0


@dataclass(frozen=True)
class TournamentStatistics:
    """Host summary tables with the method details needed to interpret them.

    Attributes
    ----------
    tournament_results : tuple[ResultRow, ...]
        One row per entrant: centered rating, rates, interval
        availability and weighted worst-20-percent expected score.
    matchup_results : tuple[ResultRow, ...]
        Directed policy/opponent rows with rates and score intervals.
    map_results : tuple[ResultRow, ...]
        Policy/map rows with opponent-weighted rates and match counts.
    metadata : dict[str, object]
        Estimator, solver, versions, convergence settings, bootstrap seed,
        independence checks and declared weighting.

    Notes
    -----
    Rows use Python scalars; unavailable bounds are None with an explicit status.
    The record is frozen but row dicts are not deeply immutable. No confidence
    interval here measures variation across independently trained policies.
    """

    tournament_results: tuple[ResultRow, ...]
    matchup_results: tuple[ResultRow, ...]
    map_results: tuple[ResultRow, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _Population:
    """Validated complete match population and compact resampling arrays.

    Rows preserve whole paired blocks and any declared larger dependence groups.
    Pair/map cells have equal exposure. Opponent and rate weights are precomputed
    once for reuse by every bootstrap replicate; all numeric arrays are float64
    counts/weights or int64 indexes on the host.
    """

    names: tuple[str, ...]
    maps: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    blocks: FloatArray
    block_cells: IntArray
    strata: tuple[IntArray, ...]
    resampling_pools: tuple[IntArray, ...]
    cell_counts: FloatArray
    opponent_weights: FloatArray
    rate_weights: FloatArray


def validate_opponent_weights(
    names: Sequence[str],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate the declared weight for every opponent identity.

    Parameters
    ----------
    names : Sequence[str]
        Complete entrant-name sequence, validated for distinctness by caller.
    weights : Mapping[str, float] | None
        Mapping with exactly those keys, or None for equal weights.

    Returns
    -------
    dict[str, float]
        Fresh dict of finite positive Python floats in names order. Defaults are
        1.0; weights are not normalized here.

    Raises
    ------
    ValueError
        Keys differ from entrants or any value is nonfinite/nonpositive.
    TypeError
        A supplied value cannot be converted to float.

    Notes
    -----
    Host-only and read-only. Population construction normalizes these weights
    per focal entrant after removing that entrant from its opponent set.
    """
    if weights is not None and set(weights) != set(names):
        raise ValueError("opponent_weights must name every entrant exactly once")
    result = {name: 1.0 if weights is None else float(weights[name]) for name in names}
    values = np.asarray(tuple(result.values()), np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Opponent weights must be finite and strictly positive")
    return result


def _population(
    schedule: Sequence[TournamentMatch],
    outcomes: Mapping[int, int],
    opponent_weights: Mapping[str, float] | None,
) -> _Population:
    """Validate exact schedule coverage and build paired resampling units.

    Require unique legal IDs, terminal outcomes, versioned pairs with equal map/seed,
    equal pair/map budgets and coherent declared dependence. Group units by coverage
    so resampling preserves exposure. Invalid populations raise ValueError before
    any fit; the returned arrays are host-only and inputs are not changed.
    """
    if not schedule:
        raise ValueError("The tournament schedule is empty")
    episodes: set[int] = set()
    blocks: dict[int, list[TournamentMatch]] = defaultdict(list)
    for match in schedule:
        if not isinstance(match, TournamentMatch):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Expected TournamentMatch schedule records")
        for name, value, lower, upper in (
            ("episode_id", match.episode_id, 1, 2**31 - 1),
            ("block_id", match.block_id, 1, 2**32 - 1),
            ("seed_id", match.seed_id, 0, 2**32 - 1),
            ("map_id", match.map_id, 0, 2**31 - 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not lower <= value <= upper
            ):
                raise ValueError(f"Invalid tournament {name}")
        if (
            not valid_policy_name(match.team_a)
            or not valid_policy_name(match.team_b)
            or match.team_a == match.team_b
        ):
            raise ValueError("A match needs two distinct nonempty policy identities")
        if match.bootstrap_group is not None and not valid_policy_name(
            match.bootstrap_group
        ):
            raise ValueError("bootstrap_group must be a nonempty string or None")
        if match.episode_id in episodes:
            raise ValueError("Duplicate tournament episode identity")
        episodes.add(match.episode_id)
        blocks[match.block_id].append(match)
    protocols = {
        "team-swap-v1" if match.pairing_protocol is None else match.pairing_protocol
        for match in schedule
    }
    if len(protocols) != 1 or not protocols <= {"team-swap-v1", "fixed-team-spawn-v1"}:
        raise ValueError("Tournament pairing protocols must be known and consistent")
    protocol = next(iter(protocols))
    if set(outcomes) != episodes:
        raise ValueError("Outcomes must cover the complete exact tournament schedule")
    if any(
        isinstance(key, bool)
        or not isinstance(key, Integral)
        or isinstance(value, bool)
        or not isinstance(value, Integral)
        or value not in (1, 2, 3)
        for key, value in outcomes.items()
    ):
        raise ValueError("Tournament outcomes must be terminal codes 1, 2 or 3")

    names = tuple(sorted({name for m in schedule for name in (m.team_a, m.team_b)}))
    maps = tuple(sorted({m.map_id for m in schedule}))
    pairs = tuple(combinations(range(len(names)), 2))
    pair_ids = {(names[a], names[b]): i for i, (a, b) in enumerate(pairs)}
    cells = np.empty(len(blocks), dtype=np.int64)
    values = np.zeros((len(blocks), 3), dtype=np.float64)
    units: dict[tuple[str, str | int], list[int]] = defaultdict(list)
    seed_groups: dict[int, tuple[str, str | int]] = {}
    for index, (_, matches) in enumerate(sorted(blocks.items())):
        if len(matches) != 2:
            raise ValueError(
                "Every seed block must contain exactly two side assignments"
            )
        first, second = sorted(matches, key=lambda m: m.team_a)
        sides_match = (first.team_a, first.team_b) == (second.team_b, second.team_a)
        if protocol == "fixed-team-spawn-v1":
            sides_match = (
                (first.team_a, first.team_b) == (second.team_a, second.team_b)
                and all(
                    not isinstance(m.spawn_locations, bool)
                    and isinstance(m.spawn_locations, Integral)
                    for m in matches
                )
                and {first.spawn_locations, second.spawn_locations} == {0, 1}
                and first.source_config_id == second.source_config_id
            )
        if (
            not sides_match
            or first.map_id != second.map_id
            or first.seed_id != second.seed_id
            or first.bootstrap_group != second.bootstrap_group
        ):
            raise ValueError("Broken side, map, seed or coupling join within a block")
        canonical_pair = (
            min(first.team_a, first.team_b),
            max(first.team_a, first.team_b),
        )
        cells[index] = pair_ids[canonical_pair] * len(maps) + maps.index(first.map_id)
        group = (
            ("block", first.block_id)
            if first.bootstrap_group is None
            else ("declared", first.bootstrap_group)
        )
        old_group = seed_groups.setdefault(first.seed_id, group)
        if old_group != group:
            raise ValueError(
                "Reused seed coordinates need one declared bootstrap_group"
            )
        units[group].append(index)
        for match in matches:
            outcome = outcomes[match.episode_id]
            category = (
                1
                if outcome == 3
                else 0
                if (outcome == 1) == (match.team_a == canonical_pair[0])
                else 2
            )
            values[index, category] += 1

    counts = np.zeros((len(pairs) * len(maps), 3), dtype=np.float64)
    np.add.at(counts, cells, values)
    if np.any(counts.sum(axis=1) == 0) or np.ptp(counts.sum(axis=1)):
        raise ValueError("Every unordered pair/map cell must have the same full budget")
    # A declared larger block is indivisible. Equal coverage signatures preserve
    # every pair/map budget when those units are resampled together.
    groups: dict[tuple[tuple[int, int], ...], list[list[int]]] = defaultdict(list)
    for _, indices in sorted(units.items()):
        ordered = sorted(indices, key=lambda i: (int(cells[i]), i))
        signature = tuple(sorted(Counter(int(cells[i]) for i in ordered).items()))
        groups[signature].append(ordered)
    strata = tuple(np.asarray(group, np.int64) for _, group in sorted(groups.items()))
    # Vectorize equal-shaped strata together: Big12 makes one RNG call per
    # replicate, rather than repeating 330 tiny Python/RNG calls per replicate.
    pools: dict[tuple[int, ...], list[IntArray]] = defaultdict(list)
    for stratum in strata:
        pools[stratum.shape].append(stratum)
    resampling_pools = tuple(np.stack(group) for _, group in sorted(pools.items()))

    declared_weights = validate_opponent_weights(names, opponent_weights)
    weights = np.asarray(tuple(declared_weights.values()), dtype=np.float64)
    weights = np.broadcast_to(weights, (len(names), len(names))).copy()
    np.fill_diagonal(weights, 0)
    weights /= weights.max(axis=1, keepdims=True)
    weights /= weights.sum(axis=1, keepdims=True)
    rate_weights = np.zeros((len(names), 3, len(counts), 3), dtype=np.float64)
    for pair_index, (a, b) in enumerate(pairs):
        for map_index in range(len(maps)):
            cell = pair_index * len(maps) + map_index
            exposure = len(maps) * counts[cell].sum()
            rate_weights[a, :, cell] = np.eye(3) * weights[a, b] / exposure
            rate_weights[b, :, cell] = np.eye(3)[:, ::-1] * weights[b, a] / exposure
    return _Population(
        names,
        maps,
        pairs,
        values,
        cells,
        strata,
        resampling_pools,
        counts,
        weights,
        rate_weights,
    )


def _objective(
    parameters: FloatArray,
    counts: FloatArray,
    contrasts: FloatArray,
    pair_indices: IntArray,
    log_reference: FloatArray | None = None,
) -> tuple[float, FloatArray]:
    """Return the penalized Davidson loss and analytic gradient for one parameter
    vector.

    Counts are win/draw/loss rows by unordered pair. Contrasts enforce centered
    strengths; the last parameter models draws. An optional saturated-model log
    reference changes only the objective's numerical offset, not its gradient.
    Use float64 arrays with compatible validated dimensions.
    """
    strengths = contrasts @ parameters[:-1]
    difference = (strengths[pair_indices[:, 0]] - strengths[pair_indices[:, 1]]) / 2
    logits = np.column_stack(
        (difference, np.full_like(difference, parameters[-1]), -difference)
    )
    log_normalizer = cast(FloatArray, special.logsumexp(logits, axis=1))
    total = counts.sum(axis=1)
    log_probability = logits - log_normalizer[:, None]
    reference = 0.0 if log_reference is None else log_reference
    loss = float(
        np.sum(counts * (reference - log_probability))
        + parameters @ parameters / (2 * _LOG_TEN**2)
    )
    residual = total[:, None] * np.exp(logits - log_normalizer[:, None]) - counts
    difference_gradient = (residual[:, 0] - residual[:, 2]) / 2
    gradient = np.zeros(contrasts.shape[0], dtype=np.float64)
    np.add.at(gradient, pair_indices[:, 0], difference_gradient)
    np.add.at(gradient, pair_indices[:, 1], -difference_gradient)
    result = np.concatenate((contrasts.T @ gradient, [residual[:, 1].sum()]))
    result += parameters / _LOG_TEN**2
    return loss, result


def _fit(
    counts: FloatArray,
    contrasts: FloatArray,
    pairs: IntArray,
    initial: FloatArray,
    *,
    context: str,
) -> FloatArray:
    # Subtract the saturated model's NLL inside each term, before summation.
    # This parameter-independent constant preserves the approved estimator and
    # gradient while avoiding subtraction of almost equal large NLL totals.
    """Fit the fixed regularized rating model and require numerical convergence.

    Use BFGS with the shared tolerance/iteration limit and a stable saturated-model
    offset. Return a finite parameter vector only when the solver and gradient check
    agree; otherwise raise RuntimeError with the point/bootstrap context.
    """
    log_reference = np.log(
        np.divide(
            counts,
            counts.sum(axis=1, keepdims=True),
            out=np.ones_like(counts),
            where=counts > 0,
        )
    )
    result = cast(
        Any,
        optimize.minimize(
            _objective,
            initial,
            args=(counts, contrasts, pairs, log_reference),
            jac=True,
            method="BFGS",
            options={"gtol": _GRADIENT_TOLERANCE, "maxiter": _MAX_ITERATIONS},
        ),
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    value, gradient = _objective(parameters, counts, contrasts, pairs, log_reference)
    if (
        not result.success
        or not np.isfinite(value)
        or not np.all(np.isfinite(parameters))
        or not np.all(np.isfinite(gradient))
        or np.max(np.abs(gradient)) > _GRADIENT_TOLERANCE
    ):
        raise RuntimeError(f"Davidson fit failed ({context}): {result.message}")
    return parameters


def _rates(population: _Population, counts: FloatArray) -> FloatArray:
    """Reduce pair/map outcome counts to weighted win/draw/loss rates for each
    entrant.
    """
    return np.einsum("prck,ck->pr", population.rate_weights, counts)


def _sample_counts(population: _Population, rng: np.random.Generator) -> FloatArray:
    """Resample complete dependence units and return matched pair/map outcome counts.

    Draw with replacement inside equal-coverage pools using the supplied NumPy
    generator. Each call advances that explicit RNG; it does not split paired games
    or invent outcomes absent from their recorded blocks.
    """
    selected = np.concatenate(
        [
            np.take_along_axis(
                pool,
                rng.integers(pool.shape[1], size=pool.shape[:2])[..., None],
                axis=1,
            ).ravel()
            for pool in population.resampling_pools
        ]
    )
    counts = np.zeros_like(population.cell_counts)
    np.add.at(counts, population.block_cells[selected], population.blocks[selected])
    return counts


def _interval(
    samples: FloatArray, *, enough_blocks: bool, variation_supported: bool = True
) -> tuple[float | None, float | None, str]:
    """Return percentile bounds only when blocks and variation support them.

    samples is a finite float64 sample vector. Insufficient independent blocks or
    effective variation returns (None, None, reason); otherwise use linear 2.5/97.5
    percentiles and return status Available. No inference about training seeds occurs.
    """
    if not enough_blocks:
        return None, None, "Insufficient Blocks"
    if not variation_supported:
        return None, None, "Insufficient Variation"
    tolerance = 64 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(samples))))
    if float(np.ptp(samples)) <= tolerance:
        return None, None, "Insufficient Variation"
    low, high = np.quantile(samples, (0.025, 0.975), method="linear")
    if high - low <= tolerance:
        return None, None, "Insufficient Variation"
    return float(low), float(high), "Available"


def _decisive_graph_connected(counts: FloatArray, pairs: IntArray, size: int) -> bool:
    """Check whether observed wins connect every entrant in both directions.

    Inputs are win/draw/loss counts by pair and their entrant indexes. Draws add no
    decisive edge. A missing reversal cannot be invented by resampling: regularized
    point ratings may remain finite while their intervals stay unavailable.
    """
    reachable = np.eye(size, dtype=np.bool_)
    reachable[pairs[:, 0], pairs[:, 1]] |= counts[:, 0] > 0
    reachable[pairs[:, 1], pairs[:, 0]] |= counts[:, 2] > 0
    for intermediate in range(size):
        reachable |= reachable[:, intermediate, None] & reachable[None, intermediate, :]
    return bool(reachable.all())


def _weighted_tail(scores: FloatArray, weights: FloatArray) -> float:
    """Average the lowest score mass totaling 20 percent, splitting boundary weight if
    needed.
    """
    order = np.argsort(scores, kind="stable")
    ordered_weights = weights[order]
    before = np.cumsum(ordered_weights) - ordered_weights
    mass = np.minimum(ordered_weights, np.maximum(0.0, 0.2 - before))
    return float(scores[order] @ mass / 0.2)


def _rate_fields(rates: FloatArray) -> ResultRow:
    """Package win/draw/loss rates and expected score, giving each draw half a point."""
    win, draw, loss = (float(value) for value in rates)
    return {
        "expected_score": win + 0.5 * draw,
        "win_rate": win,
        "draw_rate": draw,
        "loss_rate": loss,
    }


def _summarize(
    schedule: Sequence[TournamentMatch],
    outcomes: Mapping[int, int],
    *,
    seed: int,
    opponent_weights: Mapping[str, float] | None,
    replicates: int,
) -> TournamentStatistics:
    """Fit one complete population and its paired-block resampling distribution.

    The public wrapper fixes replicates=5000; tests may supply a smaller count.
    Validate seed and population first, reuse the point fit as bootstrap start,
    and retain unsupported intervals as None with a reason. Return host summary
    rows and exact method metadata; failed fits stop qualification.
    """
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("The bootstrap seed must be a uint32 integer")
    population = _population(schedule, outcomes, opponent_weights)
    names, maps, pairs = population.names, population.maps, population.pairs
    contrasts = cast(FloatArray, linalg.helmert(len(names), full=False)).T
    pair_indices = np.asarray(pairs, np.int64)
    observed_counts = population.cell_counts.reshape(len(pairs), len(maps), 3).sum(
        axis=1
    )
    point = _fit(
        observed_counts,
        contrasts,
        pair_indices,
        np.zeros(len(names), np.float64),
        context="point estimate",
    )
    elo = ELO_CENTER + (400 / _LOG_TEN) * (contrasts @ point[:-1])
    rates = _rates(population, population.cell_counts)
    scores = rates[:, 0] + 0.5 * rates[:, 1]
    rating_samples = np.broadcast_to(elo, (replicates, len(names))).copy()
    score_samples = np.broadcast_to(scores, (replicates, len(names))).copy()
    pair_scores = (
        observed_counts[:, 0] + 0.5 * observed_counts[:, 1]
    ) / observed_counts.sum(axis=1)
    pair_score_samples = np.broadcast_to(pair_scores, (replicates, len(pairs))).copy()
    varying = any(
        not np.array_equal(population.blocks[unit], population.blocks[units[0]])
        for units in population.strata
        for unit in units[1:]
    )
    if varying:
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        for replicate in range(replicates):
            counts = _sample_counts(population, rng)
            pair_counts = counts.reshape(len(pairs), len(maps), 3).sum(axis=1)
            fitted = _fit(
                pair_counts,
                contrasts,
                pair_indices,
                point,
                context=f"bootstrap replicate {replicate + 1}",
            )
            rating_samples[replicate] = ELO_CENTER + (400 / _LOG_TEN) * (
                contrasts @ fitted[:-1]
            )
            sampled_rates = _rates(population, counts)
            score_samples[replicate] = sampled_rates[:, 0] + 0.5 * sampled_rates[:, 1]
            pair_score_samples[replicate] = (
                pair_counts[:, 0] + 0.5 * pair_counts[:, 1]
            ) / pair_counts.sum(axis=1)

    enough_elo = all(len(units) >= 2 for units in population.strata)
    elo_supported = _decisive_graph_connected(
        observed_counts,
        pair_indices,
        len(names),
    )
    enough_score = np.ones(len(names), dtype=np.bool_)
    enough_matchup = np.ones(len(pairs), dtype=np.bool_)
    matchup_blocks = np.zeros(len(pairs), np.int64)
    map_blocks = np.zeros((len(names), len(maps)), np.int64)
    entrant_blocks = np.zeros(len(names), np.int64)
    for units in population.strata:
        for unit in units:
            cell_ids = set(int(cell) for cell in population.block_cells[unit])
            pair_ids = {cell // len(maps) for cell in cell_ids}
            entrants = {i for pair_id in pair_ids for i in pairs[pair_id]}
            map_slots = {
                (i, cell % len(maps))
                for cell in cell_ids
                for i in pairs[cell // len(maps)]
            }
            matchup_blocks[list(pair_ids)] += 1
            entrant_blocks[list(entrants)] += 1
            for i, map_index in map_slots:
                map_blocks[i, map_index] += 1
        if len(units) < 2:
            for cell in population.block_cells[units[0]]:
                pair_id = int(cell) // len(maps)
                a, b = pairs[pair_id]
                enough_score[[a, b]] = False
                enough_matchup[pair_id] = False

    cube = np.zeros((len(names), len(names), len(maps), 3), dtype=np.float64)
    for index, (a, b) in enumerate(pairs):
        counts = population.cell_counts[index * len(maps) : (index + 1) * len(maps)]
        cube[a, b], cube[b, a] = counts, counts[:, ::-1]
    matchup_rows: list[ResultRow] = []
    map_rows: list[ResultRow] = []
    tournament_rows: list[ResultRow] = []
    for i, name in enumerate(names):
        opponents = [j for j in range(len(names)) if i != j]
        cells = cube[i, opponents]
        cell_rates = cells / cells.sum(axis=-1, keepdims=True)
        for j, values in zip(opponents, cells, strict=True):
            pair_id = pairs.index((min(i, j), max(i, j)))
            directed_samples = pair_score_samples[:, pair_id]
            if i > j:
                directed_samples = 1 - directed_samples
            low, high, status = _interval(
                directed_samples, enough_blocks=bool(enough_matchup[pair_id])
            )
            matchup_rows.append(
                {
                    "policy": name,
                    "opponent": names[j],
                    "matches": int(values.sum()),
                    "independent_blocks": int(matchup_blocks[pair_id]),
                    "expected_score_ci_low": low,
                    "expected_score_ci_high": high,
                    "expected_score_interval_status": status,
                    "wins": int(values[:, 0].sum()),
                    "draws": int(values[:, 1].sum()),
                    "losses": int(values[:, 2].sum()),
                    **_rate_fields(
                        (values / values.sum(axis=1, keepdims=True)).mean(axis=0)
                    ),
                }
            )
        for map_index, map_id in enumerate(maps):
            values = cells[:, map_index]
            map_rows.append(
                {
                    "policy": name,
                    "map_id": map_id,
                    "matches": int(values.sum()),
                    "independent_blocks": int(map_blocks[i, map_index]),
                    "wins": int(values[:, 0].sum()),
                    "draws": int(values[:, 1].sum()),
                    "losses": int(values[:, 2].sum()),
                    **_rate_fields(
                        population.opponent_weights[i, opponents]
                        @ cell_rates[:, map_index]
                    ),
                }
            )
        elo_low, elo_high, elo_status = _interval(
            rating_samples[:, i],
            enough_blocks=enough_elo,
            variation_supported=elo_supported,
        )
        score_low, score_high, score_status = _interval(
            score_samples[:, i], enough_blocks=bool(enough_score[i])
        )
        tail_weights = np.broadcast_to(
            population.opponent_weights[i, opponents, None] / len(maps),
            cell_rates.shape[:-1],
        )
        tournament_rows.append(
            {
                "policy": name,
                "elo": float(elo[i]),
                "elo_ci_low": elo_low,
                "elo_ci_high": elo_high,
                "elo_interval_status": elo_status,
                "expected_score_ci_low": score_low,
                "expected_score_ci_high": score_high,
                "expected_score_interval_status": score_status,
                "matches": int(cells.sum()),
                "independent_blocks": int(entrant_blocks[i]),
                "wins": int(cells[..., 0].sum()),
                "draws": int(cells[..., 1].sum()),
                "losses": int(cells[..., 2].sum()),
                **_rate_fields(rates[i]),
                "worst_20_percent_expected_score": _weighted_tail(
                    (cell_rates[..., 0] + 0.5 * cell_rates[..., 1]).ravel(),
                    tail_weights.ravel(),
                ),
            }
        )
    return TournamentStatistics(
        tuple(tournament_rows),
        tuple(matchup_rows),
        tuple(map_rows),
        {
            "estimator": "centered-davidson-v1",
            "solver": "scipy.optimize.minimize:BFGS",
            "scipy_version": version("scipy"),
            "numpy_version": np.__version__,
            "dtype": "float64",
            "gradient_tolerance": _GRADIENT_TOLERANCE,
            "max_iterations": _MAX_ITERATIONS,
            "objective_value": _objective(
                point,
                observed_counts,
                contrasts,
                pair_indices,
            )[0],
            "optimization_offset": "parameter-independent saturated-model NLL",
            "draw_log_parameter": float(point[-1]),
            "logits": ["(s_i-s_j)/2", "z", "-(s_i-s_j)/2"],
            "penalty": "(sum(s*s)+z*z)/(2*ln(10)**2)",
            "elo_scale": "1200+400*s/ln(10)",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": int(seed),
            "bootstrap_rng": "numpy.random.PCG64",
            "interval": "paired-block percentile 95%; linear quantiles",
            "uncertainty_population": (
                "fixed entrants and maps; not training-run variation"
            ),
            "independent_unit_counts": [len(units) for units in population.strata],
            "decisive_win_graph_strongly_connected": elo_supported,
            "elo_interval_support": (
                "observed decisive reversals across every policy partition"
            ),
            "opponent_weights": {
                name: 1.0 if opponent_weights is None else float(opponent_weights[name])
                for name in names
            },
            "map_weights": "equal",
            "constant_resampling_distribution": not varying,
        },
    )


def summarize_tournament(
    schedule: Sequence[TournamentMatch],
    outcomes: Mapping[int, int],
    *,
    seed: int = 0,
    opponent_weights: Mapping[str, float] | None = None,
) -> TournamentStatistics:
    """Summarize a complete paired tournament using the fixed 5,000-resample method.

    Parameters
    ----------
    schedule : Sequence[TournamentMatch]
        Complete TournamentMatch sequence. Every pair/map cell has equal
        exposure; each block has opposite policy sides sharing map and seed.
        Reused seed coordinates across blocks need one declared bootstrap_group.
    outcomes : Mapping[int, int]
        Exact mapping from every scheduled episode ID to Core terminal
        code 1 (Team A win), 2 (Team B win), or 3 (draw). No extra/missing IDs.
    seed : int
        uint32 integer for NumPy PCG64 resampling, default 0.
    opponent_weights : Mapping[str, float] | None
        Finite positive weights naming every entrant, or None for
        equal opponent weights. Maps always receive equal rate weighting.

    Returns
    -------
    TournamentStatistics
        TournamentStatistics with sorted entrant, directed matchup and map rows.
        Ratings are centered at 1200. Expected score gives a draw half a point.
        Paired-block 95 percent intervals use 5,000 replicates when variation exists;
        unavailable bounds are None with an explicit reason.

    Raises
    ------
    ValueError
        Schedule, outcomes, block joins, weights or seed are invalid.
    RuntimeError
        The point fit or any needed bootstrap fit fails convergence.

    Notes
    -----
    Host-only NumPy/SciPy work; no games, file writes or JAX compilation.
    Resampling keeps declared dependence groups whole and preserves pair/map
    exposure. Constant samples skip redundant fitting and yield unavailable
    variation-based intervals. Opponent weights affect aggregate rates and
    tail scores, not the count-based rating fit. These intervals concern fixed
    entrants and maps; they do not measure training-run variation.
    """
    return _summarize(
        schedule,
        outcomes,
        seed=seed,
        opponent_weights=opponent_weights,
        replicates=_BOOTSTRAP_REPLICATES,
    )
