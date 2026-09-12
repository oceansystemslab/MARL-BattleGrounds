"""Compare stock tournament estimators on identical synthetic paired outcomes.

This developer benchmark never runs simulator episodes or qualifies a manuscript
campaign. Public reporting uses the accepted SciPy CPU estimator. JAX timings
retain convergence failures and never publish confidence intervals from failed
fits. The default comparison contains 5,000 paired bootstrap replicates.
"""

# Benchmark the existing numerical authority without expanding the public API.
# pyright: reportPrivateUsage=false

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from marl_battlegrounds.evaluation import tournament_statistics as stats
from marl_battlegrounds.evaluation.tournament_schedule import build_tournament_schedule
from marl_battlegrounds.tasks import CANONICAL_TDM_EVALUATION_MAP_IDS

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BenchmarkResult = tuple[FloatArray, FloatArray, dict[str, Any]]


def timed[T](function: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    value = function()
    return value, perf_counter() - start


def prepare(
    replicates: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray, float]:
    schedule = build_tournament_schedule(
        tuple(f"method-{i:02}" for i in range(12)), CANONICAL_TDM_EVALUATION_MAP_IDS
    )
    rng = np.random.Generator(np.random.PCG64(112))
    outcomes = {
        match.episode_id: int(rng.choice((1, 2, 3), p=(0.4, 0.3, 0.3)))
        for match in schedule
    }
    start = perf_counter()
    population = stats._population(schedule, outcomes, None)
    contrasts = stats.linalg.helmert(12, full=False).T
    pairs = np.asarray(population.pairs, np.int64)
    counts = population.cell_counts.reshape(66, 5, 3).sum(axis=1)
    rng = np.random.Generator(np.random.PCG64(113))
    samples = np.empty((replicates, 66, 3), np.float64)
    scores = np.empty((replicates, 12), np.float64)
    for index in range(replicates):
        cells = stats._sample_counts(population, rng)
        samples[index] = cells.reshape(66, 5, 3).sum(axis=1)
        rates = stats._rates(population, cells)
        scores[index] = rates[:, 0] + 0.5 * rates[:, 1]
    return counts, samples, scores, contrasts, pairs, perf_counter() - start


def scipy_run(
    counts: FloatArray, samples: FloatArray, contrasts: FloatArray, pairs: IntArray
) -> BenchmarkResult:
    point, point_time = timed(
        lambda: stats._fit(
            counts, contrasts, pairs, np.zeros(12), context="comparison point"
        )
    )
    parameters, bootstrap_time = timed(
        lambda: np.stack(
            [
                stats._fit(
                    value,
                    contrasts,
                    pairs,
                    point,
                    context=f"comparison replicate {i + 1}",
                )
                for i, value in enumerate(samples)
            ]
        )
    )
    reference = np.log(
        np.divide(
            counts,
            counts.sum(axis=1, keepdims=True),
            out=np.ones_like(counts),
            where=counts > 0,
        )
    )
    tighter = cast(
        Any,
        stats.optimize.minimize(
            stats._objective,
            point,
            args=(counts, contrasts, pairs, reference),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-6, "maxiter": 1000},
        ),
    )
    sensitivity = {
        "tighter_gtol": 1e-6,
        "success": bool(tighter.success),
        "gradient_inf": float(np.max(np.abs(tighter.jac))),
        "max_elo_change": float(
            np.max(
                np.abs(400 / np.log(10) * (contrasts @ (tighter.x[:-1] - point[:-1])))
            )
        ),
    }
    return (
        point,
        parameters,
        {
            "point_fit_seconds": point_time,
            "bootstrap_fit_seconds": bootstrap_time,
            "compile_seconds": 0.0,
            "upload_seconds": 0.0,
            "download_seconds": 0.0,
            "qualified": True,
            "failed_fits": 0,
            "sensitivity": sensitivity,
        },
    )


def jax_run(
    counts: FloatArray,
    samples: FloatArray,
    contrasts: FloatArray,
    pairs: IntArray,
    backend: str,
) -> BenchmarkResult:
    import jax
    import jax.numpy as jnp
    from jax import Array, enable_x64
    from jax.scipy import optimize
    from jax.scipy.special import logsumexp

    platform = "gpu" if backend == "jax-gpu" else "cpu"
    device = cast(Any, jax.devices(platform)[0])

    def objective(
        parameters: Array, observations: Array, contrasts: Array, pairs: Array
    ) -> Array:
        strengths = contrasts @ parameters[:-1]
        difference = (strengths[pairs[:, 0]] - strengths[pairs[:, 1]]) / 2
        logits = jnp.stack(
            (difference, jnp.full_like(difference, parameters[-1]), -difference), axis=1
        )
        log_probability = logits - logsumexp(logits, axis=1, keepdims=True)
        reference = jnp.log(
            jnp.where(
                observations > 0,
                observations / observations.sum(axis=1, keepdims=True),
                1.0,
            )
        )
        return jnp.sum(observations * (reference - log_probability)) + jnp.sum(
            parameters**2
        ) / (2 * jnp.log(10.0) ** 2)

    def fit(
        observations: Array, initial: Array, contrasts: Array, pairs: Array
    ) -> tuple[Array, bool | Array, int | Array, Array, Array, int | Array]:
        result = optimize.minimize(
            objective,
            initial,
            args=(observations, contrasts, pairs),
            method="BFGS",
            options={"gtol": 1e-4, "maxiter": 1000, "norm": jnp.inf},
        )
        return (
            result.x,
            result.success,
            result.status,
            result.fun,
            result.jac,
            result.nit,
        )

    with (
        enable_x64(),
        jax.default_device(device),
        jax.default_matmul_precision("highest"),
    ):
        inputs, upload_time = timed(
            lambda: cast(
                tuple[Array, ...],
                jax.block_until_ready(
                    jax.device_put(
                        (counts, samples, contrasts, pairs, np.zeros(12)), device
                    )
                ),
            )
        )
        d_counts, d_samples, d_contrasts, d_pairs, d_zero = inputs
        point_args = (d_counts, d_zero, d_contrasts, d_pairs)
        point_function = jax.jit(fit)
        point_executable, point_compile = timed(
            lambda: point_function.lower(*point_args).compile()
        )
        point_result, point_time = timed(
            lambda: cast(
                tuple[Array, ...], jax.block_until_ready(point_executable(*point_args))
            )
        )
        point = point_result[0]
        batch_args = (d_samples, point, d_contrasts, d_pairs)
        batch_function = jax.jit(jax.vmap(fit, in_axes=(0, None, None, None)))
        batch_executable, batch_compile = timed(
            lambda: batch_function.lower(*batch_args).compile()
        )
        batch_result, first_batch = timed(
            lambda: cast(
                tuple[Array, ...], jax.block_until_ready(batch_executable(*batch_args))
            )
        )
        warm_times: list[float] = []
        for _ in range(3):
            del batch_result
            batch_result, duration = timed(
                lambda: cast(
                    tuple[Array, ...],
                    jax.block_until_ready(batch_executable(*batch_args)),
                )
            )
            warm_times.append(duration)
        host, download_time = timed(
            lambda: jax.device_get((point_result, batch_result))
        )
        point_host, batch_host = host
        status = np.concatenate((np.atleast_1d(point_host[2]), batch_host[2]))
        success = np.concatenate((np.atleast_1d(point_host[1]), batch_host[1]))
        gradients = np.vstack((point_host[4], batch_host[4]))
        parameters = np.vstack((point_host[0], batch_host[0]))
        objectives = np.concatenate((np.atleast_1d(point_host[3]), batch_host[3]))
        qualified = (
            success
            & np.all(np.isfinite(parameters), axis=1)
            & np.all(np.isfinite(gradients), axis=1)
            & np.isfinite(objectives)
            & (np.max(np.abs(gradients), axis=1) <= 1e-4)
        )
        # Different stock line searches are permitted; objective/gradient truth
        # is independently checked against the production numerical authority.
        gradient_errors: list[float] = []
        objective_errors: list[float] = []
        for observations, fitted, device_gradient, device_objective in zip(
            (counts, *samples[:5]),
            parameters[:6],
            gradients[:6],
            objectives[:6],
            strict=True,
        ):
            reference = np.log(
                np.divide(
                    observations,
                    observations.sum(axis=1, keepdims=True),
                    out=np.ones_like(observations),
                    where=observations > 0,
                )
            )
            value, gradient = stats._objective(
                fitted, observations, contrasts, pairs, reference
            )
            gradient_errors.append(float(np.max(np.abs(gradient - device_gradient))))
            objective_errors.append(float(abs(value - device_objective)))
        return (
            point_host[0],
            batch_host[0],
            {
                "point_fit_seconds": point_time,
                "bootstrap_fit_seconds": float(np.median(warm_times)),
                "compile_seconds": point_compile + batch_compile,
                "point_compile_seconds": point_compile,
                "bootstrap_compile_seconds": batch_compile,
                "first_bootstrap_seconds": first_batch,
                "warm_bootstrap_seconds": warm_times,
                "upload_seconds": upload_time,
                "download_seconds": download_time,
                "qualified": bool(qualified.all()),
                "failed_fits": int((~qualified).sum()),
                "failed_fit_indices_first20": np.flatnonzero(~qualified)[:20].tolist(),
                "status_counts": {
                    str(code): int(np.sum(status == code)) for code in np.unique(status)
                },
                "max_gradient_inf": float(np.max(np.abs(gradients))),
                "objective_parity_max_abs": max(objective_errors),
                "gradient_parity_max_abs": max(gradient_errors),
                "device": str(device),
                "device_kind": device.device_kind,
                "jax_version": jax.__version__,
                "_fit_arrays": {
                    "status": status,
                    "success": success,
                    "gradient_inf": np.max(np.abs(gradients), axis=1),
                    "objective": objectives,
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("scipy", "jax-cpu", "jax-gpu"), required=True
    )
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    counts, samples, scores, contrasts, pairs, preparation = prepare(args.replicates)
    print(f"Prepared {args.replicates} exact paired resamples", flush=True)
    if args.backend == "scipy":
        point, parameters, result = scipy_run(counts, samples, contrasts, pairs)
    else:
        point, parameters, result = jax_run(
            counts, samples, contrasts, pairs, args.backend
        )
    result["all_fits_converged"] = result["qualified"]
    result["qualified"] = result["qualified"] and args.replicates == 5000
    start = perf_counter()
    ratings = stats.ELO_CENTER + 400 / np.log(10) * (parameters[:, :-1] @ contrasts.T)
    point_elo = stats.ELO_CENTER + 400 / np.log(10) * (contrasts @ point[:-1])
    if result["qualified"]:
        result["elo_intervals"] = np.quantile(
            ratings, (0.025, 0.975), axis=0, method="linear"
        ).T.tolist()
        result["expected_score_intervals"] = np.quantile(
            scores, (0.025, 0.975), axis=0, method="linear"
        ).T.tolist()
    result["summary_seconds"] = perf_counter() - start
    result.update(
        {
            "backend": args.backend,
            "replicates": args.replicates,
            "episode_count": 6600,
            "independent_blocks": 3300,
            "entrant_count": 12,
            "map_count": 5,
            "data_seed": 112,
            "bootstrap_seed": 113,
            "dtype": "float64",
            "gradient_inf_tolerance": 1e-4,
            "max_iterations": 1000,
            "elo_center": stats.ELO_CENTER,
            "sample_counts_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            "point_elo": point_elo.tolist(),
            "host_preparation_seconds": preparation,
            "source_sha256": hashlib.sha256(
                Path(stats.__file__).read_bytes()
            ).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": (
                "Synthetic completed tournament; no environment execution. Different "
                "existing SciPy/JAX line searches, same objective, resamples, float64 "
                "and convergence. Failure counts retained; no unqualified confidence "
                "intervals published."
            ),
        }
    )
    result["steady_total_seconds"] = sum(
        result[key]
        for key in (
            "host_preparation_seconds",
            "point_fit_seconds",
            "bootstrap_fit_seconds",
            "upload_seconds",
            "download_seconds",
            "summary_seconds",
        )
    )
    result["first_total_seconds"] = (
        result["steady_total_seconds"]
        + result["compile_seconds"]
        + result.get("first_bootstrap_seconds", result["bootstrap_fit_seconds"])
        - result["bootstrap_fit_seconds"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fit_arrays = result.pop("_fit_arrays", {})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(
        args.output.with_suffix(".npz"),
        point=point,
        parameters=parameters,
        **fit_arrays,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key.endswith("seconds")
                or key in ("qualified", "failed_fits", "status_counts")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
