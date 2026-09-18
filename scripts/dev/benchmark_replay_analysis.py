"""Measure complete prefix analysis and retain scalar prefixes for comparison.

Use identical replay files with each source revision; this records their digests
and the analysis source digest. Loading, first analysis, warmed complete analysis,
CSV formatting and summary construction are timed separately. No simulator runs.
"""

# Qualification reads retained scalar buffers directly, outside timed formatting.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from marl_battlegrounds.evaluation.analysis import analyze_replay
from marl_battlegrounds.evaluation.replay_io import load_replay


def _measure[T](call: Callable[[], T], repeats: int) -> tuple[T, dict[str, object]]:
    """Call a synchronous operation repeatedly and return its last result and timings.

    No warm-up is hidden: every requested call contributes a sample in milliseconds.
    The operation must finish its own asynchronous work before returning.
    Reject a repeat count below one.
    """
    if repeats < 1:
        raise ValueError("measurement requires a positive repetition count")
    started = time.perf_counter()
    result = call()
    samples = [(time.perf_counter() - started) * 1000]
    for _ in range(repeats - 1):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000)
    return result, {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> int:
    """Measure replay loading, full analysis, and final text formatting separately.

    Command-line arguments choose replay files, output directory, and repeats.
    Writes JSON timing records and compressed scalar-prefix arrays. Repeated
    analyses must produce equal values and validity masks.

    Returns
    -------
    int
        Zero after every requested replay is checked and recorded.

    Raises
    ------
    SystemExit
        If arguments are invalid or help was requested.
    AssertionError
        If repeated analysis changes the saved prefix data.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two warm repetitions are required")
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for path in args.replays:
        loaded, loading = _measure(partial(load_replay, path), 1)
        first, first_timing = _measure(partial(analyze_replay, loaded, full=True), 1)
        analysis, warm = _measure(
            partial(analyze_replay, loaded, full=True), args.repeats
        )
        # Ensure repeated preparation leaves every prefix and validity unchanged.
        np.testing.assert_array_equal(analysis._values, first._values)
        np.testing.assert_array_equal(analysis._valid, first._valid)
        _, csv = _measure(
            partial(analysis.csv, analysis.frame_count - 1, scope="final"), args.repeats
        )
        _, summary = _measure(
            partial(analysis.summary, analysis.frame_count - 1, scope="final"),
            args.repeats,
        )
        prefixes = args.output / f"{analysis.source_replay_digest}.npz"
        np.savez_compressed(
            prefixes,
            values=analysis._values,
            valid=analysis._valid,
            columns=np.asarray([column.name for column in analysis.columns]),
        )
        results.append(
            {
                "replay": str(path.resolve()),
                "replay_digest": analysis.source_replay_digest,
                "analysis_source_digest": analysis.analysis_source_digest,
                "frames": analysis.frame_count,
                "numeric_columns": len(analysis.columns),
                "load": loading,
                "first_analysis_including_compilation": first_timing,
                "warm_complete_prefix_analysis": warm,
                "final_csv_format": csv,
                "final_summary": summary,
                "scalar_prefix_bytes": analysis._values.nbytes + analysis._valid.nbytes,
                "process_peak_ram_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                "prefix_arrays": str(prefixes.resolve()),
            }
        )
        (args.output / "prefix-analysis.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(
            f"{path.name}: {analysis.frame_count} frames, {warm['median_ms']:.3f} ms",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
