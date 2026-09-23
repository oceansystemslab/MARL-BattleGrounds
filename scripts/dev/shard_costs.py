"""Refresh the measured test-shard cost table and predict shard times.

Run from the repository root after preparing the locked uv environment:

    scripts/dev/check.sh --timings /tmp/timings          # full Python gate + timings
    uv run --no-sync python scripts/dev/shard_costs.py update /tmp/timings
    uv run --no-sync python scripts/dev/shard_costs.py plan
    scripts/dev/check_frontend.sh --timings /tmp/timings  # full browser gate + timings
    uv run --no-sync python scripts/dev/shard_costs.py browser /tmp/timings

``update`` reads the ``python-shard-N.xml`` JUnit files that ``check.sh
--timings`` saves, sums each test file's seconds, splits files that are too
large for one shard at test-function boundaries where the scheduler allows it,
rewrites ``scripts/dev/pytest_shard_costs.json`` and prints the predicted
seconds of every shard. ``plan`` prints that prediction for the current table
without writing anything. ``browser`` reads the ``browser-profile-N.json``
reports from ``check_frontend.sh --timings`` and prints seconds per browser
profile, spec file and test, for editing
``web/visual_debugger/e2e/ci-shards.json`` by hand.

``update`` and ``plan`` collect (but do not run) the whole Python test suite on
the CPU to learn the current test families. The scheduler rules themselves
live in ``scripts/dev/pytest_shard.py``; this tool reuses them. See the
"Rebalancing the test shards" section of ``docs/dev/quality_gates.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.dev.pytest_shard import (  # noqa: E402
    MEASURED_COSTS_PATH,
    MEASURED_COSTS_SCHEMA,
    ModuleFixtureKey,
    ShardCostProfile,
    TestFamilyKey,
    assign_test_families,
    build_test_work_units,
    dynamic_fixture_request_sites_from_item,
    family_key_from_item,
    load_measured_cost_profile,
    logical_path_from_family,
    module_fixture_keys_from_item,
    shard_costs,
    validate_split_dynamic_fixture_requests,
    validate_split_fixture_affinity,
)

DEFAULT_SHARD_COUNT = 12
RESERVED_REASON = "Hosted CI runs Pyright before the tests of shard 12."
RESERVED_SECONDS = {"12": [0] * 11 + [50]}


@dataclass(frozen=True)
class CollectedTests:
    """Facts the scheduler needs about every collected test family.

    Attributes
    ----------
    item_counts : Mapping[TestFamilyKey, int]
        Collected test count for each unparameterized test family.
    module_fixtures_by_family : Mapping[TestFamilyKey, frozenset[ModuleFixtureKey]]
        Module-scoped fixtures each family uses; split files may not share them.
    request_sites_by_family : Mapping[TestFamilyKey, frozenset[str]]
        Dynamic fixture request sites each family uses; split files may not have any.
    item_nodeids : frozenset[str]
        Exact collected case IDs, including every parameter suffix. Used to
        reject incomplete or stale timing reports before replacing the table.
    """

    item_counts: Mapping[TestFamilyKey, int]
    module_fixtures_by_family: Mapping[TestFamilyKey, frozenset[ModuleFixtureKey]]
    request_sites_by_family: Mapping[TestFamilyKey, frozenset[str]]
    item_nodeids: frozenset[str]


@dataclass(frozen=True)
class ShardPrediction:
    """Predicted cost of one shard under a cost profile.

    Attributes
    ----------
    shard : int
        One-based shard number.
    seconds : int
        Sum of its work-unit costs plus reserved work, in table seconds.
    units : int
        Number of whole work units assigned to it.
    largest_unit : str
        Identifier of its most costly unit, such as ``file:tests/test_x.py``.
    """

    shard: int
    seconds: int
    units: int
    largest_unit: str


def family_nodeid_from_junit(classname: str, name: str) -> str:
    """Turn one JUnit test case name into the scheduler's family node ID.

    Parameters
    ----------
    classname : str
        Pytest's dotted JUnit class name, such as ``tests.test_x`` or
        ``tests.test_x.TestGroup``. The first two parts name the test file
        (all test files sit directly in ``tests/``); later parts are classes.
    name : str
        Test case name. A parameter suffix such as ``[case-1]`` is removed so
        every parameter case maps to its one family.

    Returns
    -------
    str
        Family node ID such as ``tests/test_x.py::TestGroup::test_y``.

    Raises
    ------
    ValueError
        The class name does not start with a test file inside ``tests``.
    """
    return _case_nodeid_from_junit(classname, name.split("[", maxsplit=1)[0])


def _case_nodeid_from_junit(classname: str, name: str) -> str:
    """Return a pytest case ID, keeping its parameter suffix.

    ``classname`` is pytest's dotted ``tests`` module and optional classes;
    ``name`` is its nonempty case name. Raises ValueError for another module or
    an empty name. This repository keeps Python test files directly in tests/.
    """
    parts = classname.split(".")
    if len(parts) < 2 or parts[0] != "tests" or not parts[1]:
        raise ValueError(f"JUnit class name is not a tests/ module: {classname!r}")
    path = f"{parts[0]}/{parts[1]}.py"
    if not name:
        raise ValueError("JUnit test case name must be nonempty")
    return "::".join((path, *parts[2:], name))


def read_junit_seconds(
    directory: Path,
    shard_count: int = DEFAULT_SHARD_COUNT,
    *,
    expected_cases: Collection[str] | None = None,
) -> dict[str, float]:
    """Sum test-family seconds from passing, nonduplicated shard reports.

    Parameters
    ----------
    directory : Path
        Folder holding ``python-shard-1.xml`` to ``python-shard-N.xml``, written
        by ``scripts/dev/check.sh --timings``.
    shard_count : int, optional
        Number of shard files that must all be present. Default 12.
    expected_cases : Collection[str] | None, optional
        Exact current pytest case IDs. When supplied, every case must appear
        exactly once across the reports. The update command always supplies
        these IDs. None checks report contents without claiming full coverage.

    Returns
    -------
    dict of str to float
        Seconds per family node ID. Each case's time includes its setup and
        teardown, so a module fixture's cost lands on the first test using it.

    Raises
    ------
    FileNotFoundError
        A shard's timing file is missing.
    ValueError
        A file is not valid JUnit XML, holds no test, reports a failure or
        error, records a duplicate case or an invalid duration, or differs from
        expected_cases. JUnit reports prove test coverage, not static checks.
    """
    seconds: defaultdict[str, float] = defaultdict(float)
    seen: set[str] = set()
    for shard in range(1, shard_count + 1):
        path = directory / f"python-shard-{shard}.xml"
        if not path.is_file():
            raise FileNotFoundError(f"missing timing file for shard {shard}: {path}")
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError as error:
            raise ValueError(f"not valid JUnit XML: {path}") from error
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        problems = sum(
            int(suite.get("failures", "0")) + int(suite.get("errors", "0"))
            for suite in suites
        )
        cases = list(root.iter("testcase"))
        if problems or not cases:
            raise ValueError(
                f"shard {shard} did not pass completely ({path}); use timings "
                "from a gate where every shard passed"
            )
        for case in cases:
            classname = case.get("classname", "")
            if not classname:
                raise ValueError(f"shard {shard} recorded a collection error: {path}")
            name = case.get("name", "")
            case_id = _case_nodeid_from_junit(classname, name)
            if case_id in seen:
                raise ValueError(f"duplicate JUnit case: {case_id}")
            seen.add(case_id)
            if case.find("failure") is not None or case.find("error") is not None:
                raise ValueError(f"JUnit case did not pass completely: {case_id}")
            duration = float(case.get("time", "0"))
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"invalid JUnit duration for {case_id}: {duration}")
            nodeid = family_nodeid_from_junit(classname, name)
            seconds[nodeid] += duration
    if expected_cases is not None:
        expected = set(expected_cases)
        missing, unexpected = expected - seen, seen - expected
        if missing or unexpected:
            raise ValueError(
                "JUnit cases differ from current collection: "
                f"{len(missing)} missing, {len(unexpected)} unexpected; "
                f"examples: missing={sorted(missing)[:3]}, "
                f"unexpected={sorted(unexpected)[:3]}"
            )
    return dict(seconds)


def collect_tests(repository: Path = _REPOSITORY_ROOT) -> CollectedTests:
    """Collect the Python test suite without running it.

    Parameters
    ----------
    repository : Path
        Repository root whose ``tests`` folder pytest collects.

    Returns
    -------
    CollectedTests
        Family counts and fixture facts, keyed exactly as the scheduler keys them.

    Raises
    ------
    RuntimeError
        Pytest collection fails.

    Notes
    -----
    Imports every test module, so it takes about 20 seconds and uses the CPU
    backend unless ``JAX_PLATFORMS`` is already set. Pytest's own output is
    captured and dropped.
    """
    import pytest

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    item_counts: Counter[TestFamilyKey] = Counter()
    fixtures: dict[TestFamilyKey, set[ModuleFixtureKey]] = {}
    requests: dict[TestFamilyKey, set[str]] = {}
    item_nodeids: set[str] = set()

    class _Collector:
        """Pytest plugin that records scheduler facts for every collected item."""

        def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
            """Record each item's family, module fixtures and request sites."""
            for item in items:
                item_nodeids.add(item.nodeid)
                family = family_key_from_item(item)
                item_counts[family] += 1
                fixtures.setdefault(family, set()).update(
                    module_fixture_keys_from_item(item)
                )
                requests.setdefault(family, set()).update(
                    dynamic_fixture_request_sites_from_item(item)
                )

    with contextlib.redirect_stdout(io.StringIO()):
        status = pytest.main(
            [
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                str(repository / "tests"),
            ],
            plugins=[_Collector()],
        )
    if status not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        raise RuntimeError(f"pytest collection failed with exit status {status}")
    return CollectedTests(
        item_counts=dict(item_counts),
        module_fixtures_by_family={
            family: frozenset(keys) for family, keys in fixtures.items()
        },
        request_sites_by_family={
            family: frozenset(sites) for family, sites in requests.items()
        },
        item_nodeids=frozenset(item_nodeids),
    )


def _can_split(path: str, families: Sequence[str], collected: CollectedTests) -> bool:
    """Say whether the scheduler would accept pulling families out of one file.

    ``path`` is the test file, ``families`` the family node IDs to give their
    own units, and ``collected`` the current collection. Returns False, instead
    of raising the scheduler's ValueError, when the split would separate
    families that share a module-scoped fixture or when the file requests
    fixtures dynamically.
    """
    candidate = ShardCostProfile(
        extracted_family_costs={nodeid: 1 for nodeid in families},
        residual_file_costs={path: 1},
    )
    try:
        validate_split_fixture_affinity(collected.module_fixtures_by_family, candidate)
        validate_split_dynamic_fixture_requests(
            collected.request_sites_by_family, candidate
        )
    except ValueError:
        return False
    return True


def build_cost_table(
    family_seconds: Mapping[str, float],
    collected: CollectedTests,
    *,
    shard_count: int = DEFAULT_SHARD_COUNT,
    max_unit_seconds: int | None = None,
    measured_on: str,
    measured_with: str,
) -> tuple[dict[str, object], list[str]]:
    """Turn measured family seconds into a cost table the scheduler can load.

    Parameters
    ----------
    family_seconds : Mapping[str, float]
        Measured seconds per family node ID, from ``read_junit_seconds``.
    collected : CollectedTests
        Current collection. Families that were not collected are ignored, so
        stale timings never enter the table.
    shard_count : int, optional
        Number of shards to balance for. Default 12.
    max_unit_seconds : int or None, optional
        Largest work unit wanted. Files above it are split at test-function
        boundaries, slowest functions first, until the rest fits. None uses
        half of the average shard time.
    measured_on : str
        Date of the measurement, such as ``2026-09-22``.
    measured_with : str
        One plain sentence saying how the timings were taken.

    Returns
    -------
    tuple
        The JSON-ready table and a list of plain-English warnings, for example
        a file the scheduler may not split or one test function that alone is
        larger than ``max_unit_seconds``.

    Notes
    -----
    Seconds are rounded up to whole seconds and are at least 1. Files that
    were collected but have no timing keep the scheduler's default of one
    unit per test.
    """
    collected_nodeids = {family[1] for family in collected.item_counts}
    families_by_path: dict[str, list[str]] = {}
    for family in collected.item_counts:
        families_by_path.setdefault(logical_path_from_family(family), []).append(
            family[1]
        )
    seconds_by_family = {
        nodeid: value
        for nodeid, value in family_seconds.items()
        if nodeid in collected_nodeids
    }
    file_seconds: defaultdict[str, float] = defaultdict(float)
    for nodeid, value in seconds_by_family.items():
        file_seconds[nodeid.split("::", maxsplit=1)[0]] += value
    total = sum(file_seconds.values())
    limit = (
        max_unit_seconds
        if max_unit_seconds is not None
        else max(1, math.ceil(total / shard_count / 2))
    )

    files: dict[str, int] = {}
    split_families: dict[str, int] = {}
    split_residuals: dict[str, int] = {}
    warnings: list[str] = []
    for path in sorted(file_seconds):
        seconds = file_seconds[path]
        families = sorted(
            families_by_path.get(path, ()),
            key=lambda nodeid: (-seconds_by_family.get(nodeid, 0.0), nodeid),
        )
        if seconds <= limit or len(families) < 2:
            files[path] = max(1, math.ceil(seconds))
            if seconds > limit:
                warnings.append(
                    f"{path} alone takes {seconds:.0f} s, over the {limit} s unit "
                    "target, and has one test function; split its parameter cases "
                    "into two functions or make it faster."
                )
            continue
        extracted: list[str] = []
        residual = seconds
        for nodeid in families[:-1]:
            if residual <= limit:
                break
            extracted.append(nodeid)
            residual -= seconds_by_family.get(nodeid, 0.0)
        if not _can_split(path, extracted, collected):
            files[path] = max(1, math.ceil(seconds))
            warnings.append(
                f"{path} takes {seconds:.0f} s but its test functions share "
                "module fixtures or request fixtures dynamically, so the scheduler "
                "cannot split it; move tests into a new file to split it."
            )
            continue
        for nodeid in extracted:
            family_cost = seconds_by_family.get(nodeid, 0.0)
            split_families[nodeid] = max(1, math.ceil(family_cost))
            if family_cost > limit:
                warnings.append(
                    f"{nodeid} alone takes {family_cost:.0f} s, over the {limit} s "
                    "unit target; split its parameter cases into two functions "
                    "or make it faster."
                )
        split_residuals[path] = max(1, math.ceil(residual))
    untimed = sorted(set(families_by_path) - set(file_seconds))
    if untimed:
        warnings.append(
            f"{len(untimed)} collected test files have no timing and cost one unit "
            f"per test, for example {', '.join(untimed[:3])}."
        )
    table: dict[str, object] = {
        "schema": MEASURED_COSTS_SCHEMA,
        "measured_on": measured_on,
        "measured_with": measured_with,
        "reserved_seconds": RESERVED_SECONDS,
        "reserved_reason": RESERVED_REASON,
        "files": files,
        "split_families": split_families,
        "split_residuals": split_residuals,
    }
    return table, warnings


def predict_shards(
    profile: ShardCostProfile,
    collected: CollectedTests,
    shard_count: int = DEFAULT_SHARD_COUNT,
) -> list[ShardPrediction]:
    """Predict each shard's cost with the scheduler's own packing.

    Parameters
    ----------
    profile : ShardCostProfile
        Cost profile to test, usually from ``load_measured_cost_profile``.
    collected : CollectedTests
        Current collection.
    shard_count : int, optional
        Number of shards. Default 12.

    Returns
    -------
    list of ShardPrediction
        One prediction per shard, in shard order.

    Raises
    ------
    ValueError
        The profile is stale for this collection or splits a file unsafely,
        exactly as the real ``--ci-shard`` run would reject it.
    """
    validate_split_fixture_affinity(collected.module_fixtures_by_family, profile)
    validate_split_dynamic_fixture_requests(collected.request_sites_by_family, profile)
    assignments = assign_test_families(
        collected.item_counts, shard_count, cost_profile=profile
    )
    costs = shard_costs(assignments, collected.item_counts, cost_profile=profile)
    units = build_test_work_units(collected.item_counts, profile)
    owner = {
        family: index
        for index, families in enumerate(assignments)
        for family in families
    }
    by_shard: dict[int, list[tuple[int, str]]] = {}
    for unit in units:
        by_shard.setdefault(owner[unit.families[0]], []).append(
            (unit.cost, unit.identifier)
        )
    predictions: list[ShardPrediction] = []
    for index, cost in enumerate(costs):
        shard_units = sorted(by_shard.get(index, []), reverse=True)
        predictions.append(
            ShardPrediction(
                shard=index + 1,
                seconds=cost,
                units=len(shard_units),
                largest_unit=shard_units[0][1] if shard_units else "",
            )
        )
    return predictions


def format_predictions(predictions: Sequence[ShardPrediction]) -> str:
    """Render shard predictions as a plain text table with a balance summary.

    Parameters
    ----------
    predictions : Sequence[ShardPrediction]
        Output of ``predict_shards``.

    Returns
    -------
    str
        One line per shard, then the slowest, average and fastest shard in
        seconds and the slowest divided by the average.
    """
    lines = ["Shard  Seconds  Units  Largest unit"]
    lines.extend(
        f"{row.shard:>5}  {row.seconds:>7}  {row.units:>5}  {row.largest_unit}"
        for row in predictions
    )
    seconds = [row.seconds for row in predictions]
    average = sum(seconds) / len(seconds)
    lines.append(
        f"Slowest {max(seconds)} s, average {average:.0f} s, fastest {min(seconds)} s;"
        f" slowest / average = {max(seconds) / average:.2f}"
    )
    return "\n".join(lines)


def read_browser_seconds(directory: Path) -> dict[int, list[tuple[str, str, float]]]:
    """Read per-test seconds from saved Playwright JSON reports.

    Parameters
    ----------
    directory : Path
        Folder holding ``browser-profile-N.json`` files written by
        ``scripts/dev/check_frontend.sh --timings``.

    Returns
    -------
    dict of int to list of tuple
        For each one-based profile, ``(spec file, test title, seconds)`` rows.
        A test's seconds sum all its recorded attempts.

    Raises
    ------
    FileNotFoundError
        No ``browser-profile-*.json`` file exists in the folder.
    ValueError
        A report is not a Playwright JSON report.
    """
    paths = sorted(directory.glob("browser-profile-*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no browser-profile-*.json timing files in {directory}"
        )
    profiles: dict[int, list[tuple[str, str, float]]] = {}
    for path in paths:
        profile = int(path.stem.rsplit("-", maxsplit=1)[1])
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"not a Playwright JSON report: {path}") from error
        rows: list[tuple[str, str, float]] = []
        if not isinstance(report, dict):
            raise ValueError(f"not a Playwright JSON report: {path}")
        for suite in _json_list(cast(dict[str, object], report).get("suites")):
            _walk_browser_suite(suite, "", rows)
        profiles[profile] = rows
    return profiles


def _json_list(value: object) -> list[dict[str, object]]:
    """Return the JSON objects in a list value, or an empty list for anything else."""
    if not isinstance(value, list):
        return []
    return [
        cast(dict[str, object], entry)
        for entry in cast(list[object], value)
        if isinstance(entry, dict)
    ]


def _walk_browser_suite(
    suite: Mapping[str, object], file: str, rows: list[tuple[str, str, float]]
) -> None:
    """Append one ``(spec file, test title, seconds)`` row per spec in a suite tree.

    ``file`` is the spec file inherited from the parent suite; a suite's own
    ``file`` wins. Seconds sum every result's ``duration`` (milliseconds).
    """
    suite_file = str(suite.get("file") or file)
    for spec in _json_list(suite.get("specs")):
        milliseconds = 0.0
        for test in _json_list(spec.get("tests")):
            for result in _json_list(test.get("results")):
                duration = result.get("duration", 0)
                if isinstance(duration, int | float):
                    milliseconds += float(duration)
        rows.append((suite_file, str(spec.get("title", "")), milliseconds / 1000.0))
    for child in _json_list(suite.get("suites")):
        _walk_browser_suite(child, suite_file, rows)


def format_browser_seconds(
    profiles: Mapping[int, Sequence[tuple[str, str, float]]],
) -> str:
    """Render browser timings per profile, spec file and test as plain text.

    Parameters
    ----------
    profiles : Mapping[int, Sequence[tuple[str, str, float]]]
        Output of ``read_browser_seconds``.

    Returns
    -------
    str
        Each profile's summed test seconds, then its spec files and tests,
        slowest first, followed by the slowest, average and fastest profile.

    Notes
    -----
    Playwright's per-test durations leave out ``beforeAll`` and ``afterAll``
    hooks and worker start-up, which cost some profiles one to two minutes.
    Compare the whole-profile seconds that the gate prints as well.
    """
    lines: list[str] = []
    totals: list[float] = []
    for profile in sorted(profiles):
        rows = profiles[profile]
        total = sum(row[2] for row in rows)
        totals.append(total)
        lines.append(
            f"Profile {profile}: {total:.0f} s of test time, {len(rows)} tests"
        )
        by_file: defaultdict[str, float] = defaultdict(float)
        for file, _title, seconds in rows:
            by_file[file] += seconds
        for file, seconds in sorted(by_file.items(), key=lambda row: -row[1]):
            lines.append(f"  {seconds:>6.0f} s  {file}")
        for file, title, seconds in sorted(rows, key=lambda row: -row[2]):
            lines.append(f"      {seconds:>6.0f} s  {file} > {title}")
    if totals:
        average = sum(totals) / len(totals)
        lines.append(
            f"Slowest {max(totals):.0f} s, average {average:.0f} s, "
            f"fastest {min(totals):.0f} s"
        )
    return "\n".join(lines)


def _current_commit() -> str:
    """Return the checkout's short commit hash, or ``unknown`` without Git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``update``, ``plan`` or ``browser`` command.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command-line arguments without the program name. None reads ``sys.argv``.

    Returns
    -------
    int
        0 on success. Argument errors exit through argparse with status 2.

    Notes
    -----
    ``update`` rewrites ``scripts/dev/pytest_shard_costs.json``; the other
    commands only print.
    """
    parser = argparse.ArgumentParser(
        description="Refresh the measured shard cost table and predict shard times."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser(
        "update", help="rewrite the cost table from check.sh --timings output"
    )
    update.add_argument("timings", type=Path, help="folder with python-shard-N.xml")
    update.add_argument(
        "--max-unit",
        type=int,
        default=None,
        help="largest wanted work unit in seconds (default: half an average shard)",
    )
    plan = commands.add_parser("plan", help="print predicted seconds for each shard")
    plan.add_argument("--shards", type=int, default=DEFAULT_SHARD_COUNT)
    browser = commands.add_parser(
        "browser", help="print browser profile timings from check_frontend.sh --timings"
    )
    browser.add_argument(
        "timings", type=Path, help="folder with browser-profile-N.json"
    )
    arguments = parser.parse_args(argv)

    if arguments.command == "browser":
        print(format_browser_seconds(read_browser_seconds(arguments.timings)))
        return 0
    collected = collect_tests()
    if arguments.command == "update":
        table, warnings = build_cost_table(
            read_junit_seconds(
                arguments.timings, expected_cases=collected.item_nodeids
            ),
            collected,
            max_unit_seconds=arguments.max_unit,
            measured_on=datetime.now(UTC).date().isoformat(),
            measured_with=(
                "Full gate (scripts/dev/check.sh --timings), all twelve Python "
                f"shards on a {os.cpu_count()}-CPU machine; per-test "
                "JUnit time including setup and teardown, summed per file and "
                "rounded up to whole seconds."
            ),
        )
        table["measured_commit"] = _current_commit()
        # Check the new table with the scheduler before replacing the old one.
        candidate = MEASURED_COSTS_PATH.with_suffix(".json.new")
        candidate.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        try:
            predictions = predict_shards(
                load_measured_cost_profile(candidate), collected
            )
        except ValueError:
            candidate.unlink()
            raise
        candidate.replace(MEASURED_COSTS_PATH)
        print(f"Wrote {MEASURED_COSTS_PATH.relative_to(_REPOSITORY_ROOT)}")
        for warning in warnings:
            print(f"Warning: {warning}")
        print(format_predictions(predictions))
        return 0
    print(
        format_predictions(
            predict_shards(load_measured_cost_profile(), collected, arguments.shards)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
