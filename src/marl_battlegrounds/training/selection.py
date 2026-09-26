"""Write separate checkpoint decisions from saved, verified validation games.

``reselect_checkpoint`` owns declaration files and decision history. The pure
ranking helpers remain in analysis; _selection_evidence verifies stored actors
and games without restoring models. This module never trains or plays games.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

Record = dict[str, Any]
_RULES = ("saved", "score_then_kills", "score_then_step")
_DEFAULTS: Record = {
    "schema_version": 1,
    "rule": "saved",
    "shortlist_size": 2,
    "across_runs": False,
    "seed_order": None,
    "allow_different_roots": False,
    "confirmation_root": None,
    "confirmation_seed_pairs": None,
    "previous_decision": None,
    "confirmation_results": [],
}


def _digest(value: object) -> str:
    """Hash a strict, sorted JSON value without changing its list order."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _read(path: Path) -> Record:
    """Read one JSON object; reject nonobjects and nonfinite values."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    result = cast(Record, value)
    _digest(result)
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Check an exact integer, excluding Booleans, at or above minimum."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _declaration(value: str | Path | Mapping[str, Any]) -> Record:
    """Resolve the small named rule and file-relative paths without opening games."""
    base = Path.cwd()
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        base = path.parent
        supplied = _read(path)
    elif isinstance(cast(object, value), Mapping):
        supplied = dict(value)
    else:
        raise TypeError("declaration must be a JSON path or mapping")
    if set(supplied) - ({"name"} | set(_DEFAULTS)):
        raise ValueError("Unknown checkpoint selection declaration fields")
    result = {**_DEFAULTS, **supplied}
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("Checkpoint selection needs declaration schema_version=1")
    if not isinstance(result.get("name"), str) or not result["name"].strip():
        raise ValueError("Give this separate selection decision a nonempty name")
    if result["rule"] not in _RULES:
        raise ValueError(f"Selection rule must be one of {_RULES}")
    _integer(result["shortlist_size"], "shortlist_size", minimum=1)
    for name in ("across_runs", "allow_different_roots"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be Boolean")
    for name in ("confirmation_root", "confirmation_seed_pairs"):
        if result[name] is not None:
            _integer(result[name], name, minimum=int(name.endswith("pairs")))
    if result["confirmation_root"] is not None and result["confirmation_root"] >= 2**32:
        raise ValueError("confirmation_root must fit an unsigned 32-bit integer")
    order = result["seed_order"]
    if order is not None:
        if not isinstance(order, list):
            raise ValueError("seed_order must be a list of distinct training seeds")
        order = cast(list[Any], order)
        for seed in order:
            _integer(seed, "seed_order entry")
        if len(set(order)) != len(order):
            raise ValueError("seed_order must not repeat a seed")
    if result["across_runs"] and not order:
        raise ValueError("Across-run selection requires an explicit seed_order")
    if not result["across_runs"] and order is not None:
        raise ValueError("seed_order applies only to across_runs=True")
    paths = result["confirmation_results"]
    if not isinstance(paths, list) or any(
        not isinstance(p, (str, Path)) for p in cast(list[object], paths)
    ):
        raise ValueError("confirmation_results must list saved summary paths")
    paths = cast(list[str | Path], paths)
    result["confirmation_results"] = [str((base / p).resolve()) for p in paths]
    if len(set(result["confirmation_results"])) != len(paths):
        raise ValueError("confirmation_results must not repeat a path")
    prior = result["previous_decision"]
    if prior is not None:
        if not isinstance(prior, (str, Path)):
            raise ValueError("previous_decision must be a decision JSON path")
        result["previous_decision"] = str((base / prior).resolve())
    _digest(result)
    return result


def _key(row: Mapping[str, Any], rule: str) -> tuple[float, float, int, str]:
    """Apply a declared score, kill-difference, earlier-step, then identity order."""
    from marl_battlegrounds.training.analysis import (
        _selection_key,  # pyright: ignore[reportPrivateUsage]
    )

    if rule == "saved":
        return _selection_key(row)
    difference = 0.0
    if rule == "score_then_kills":
        raw = row.get("mean_kill_difference")
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(raw)
        ):
            raise ValueError("This rule needs verified mean kill difference")
        difference = float(raw)
    return (
        -float(row["score"]),
        -difference,
        int(row["env_steps"]),
        str(row["checkpoint_id"]),
    )


def _roots(rows: Sequence[Mapping[str, Any]], declaration: Mapping[str, Any]) -> str:
    """Require common effective roots unless their comparison was declared unpaired."""
    from marl_battlegrounds.training.validation import validation_root_comparison

    return validation_root_comparison(
        rows, allow_different_roots=declaration["allow_different_roots"]
    )


def _conditions(rows: Sequence[Mapping[str, Any]]) -> None:
    """Keep candidate panels, scoring layouts, maps and depth interpretations equal."""
    if (
        len(
            {
                (
                    row["panel_digest"],
                    row.get("selection_schema_version", 1),
                    tuple(row["maps"]),
                    row.get("red_zone_depth", 0.0),
                )
                for row in rows
            }
        )
        > 1
    ):
        raise ValueError("Selection cannot mix panels, maps or scoring conditions")


def _request(
    evidence: Record, row: Record, declaration: Record, destination: Path
) -> Record:
    """Describe the exact public call needed to obtain a missing confirmation."""
    from marl_battlegrounds.training.validation import (
        check_confirmation_roots,
        panel_task_description,
    )

    panel = evidence["panel"]
    saved: Record = evidence.get("validation_declaration") or {}
    root = declaration["confirmation_root"]
    if root is None and saved and panel.schema_version == 2:
        root = saved["roots"]["confirmation"]
    pairs = declaration["confirmation_seed_pairs"]
    if pairs is None:
        pairs = saved.get(
            "confirmation_seed_pairs",
            evidence["config"].get("confirmation_seed_pairs", 50),
        )
    _integer(pairs, "confirmation_seed_pairs", minimum=1)
    # A verified old task has pre-Red-Zone rules. New games must name zero,
    # because the current validation API otherwise defaults to five map units.
    depth = row.get("red_zone_depth", 0.0)
    task = panel_task_description(
        checkpoint_id=row["checkpoint_id"],
        actor_digest=row["actor_digest"],
        env_steps=row["env_steps"],
        panel=panel,
        purpose="confirmation",
        seed_pairs=pairs,
        root_seed=root,
        red_zone_depth=depth,
    )
    used_roots: set[int] = set(evidence.get("used_roots", ()))
    used_roots.update(evidence.get("comparison_used_roots", ()))
    used_roots.update(
        panel.roots[p] for p in ("routine", "initialization") if p in panel.roots
    )
    check_confirmation_roots(
        task["root"], evidence["records"], used_roots=sorted(used_roots)
    )
    actor = evidence["actors"][row["checkpoint_id"]]
    arguments: Record = {
        "checkpoint": actor["actor_path"],
        "panel": str(evidence["panel_path"]),
        "output_dir": str(destination / "confirmation" / task["task_id"]),
        "purpose": "confirmation",
        "seed_pairs": pairs,
        "red_zone_depth": depth,
    }
    if root is not None:
        arguments["root_seed"] = root
    call = (
        "validate_checkpoint("
        + ", ".join(f"{name}={value!r}" for name, value in arguments.items())
        + ")"
    )
    return {
        "checkpoint_id": row["checkpoint_id"],
        "artifact_id": actor["artifact_id"],
        "actor_digest": row["actor_digest"],
        "expected_task": task,
        "function": "marl_battlegrounds.training.validation.validate_checkpoint",
        "arguments": arguments,
        "python": (
            "from marl_battlegrounds.training.validation "
            "import validate_checkpoint\n" + call
        ),
    }


def _run_decision(evidence: Record, declaration: Record, destination: Path) -> Record:
    """Rebuild this run's shortlist and withhold its winner until every task exists."""
    from marl_battlegrounds.training.analysis import (
        _candidates,  # pyright: ignore[reportPrivateUsage]
    )

    result: Record = {
        "run_id": evidence["run_id"],
        "run_dir": evidence["run_dir"],
        "seed": evidence["seed"],
        "status": "incomplete",
        "shortlist": [],
        "winner": None,
        "needs_confirmation": [],
    }
    if evidence.get("failures"):
        result["reason"] = evidence["failures"]
        return result
    routine = [row for row in evidence["records"] if row["purpose"] == "routine"]
    final = evidence["final_checkpoint_id"]
    if not routine or final is None:
        result["reason"] = (
            "The run needs its final actor and complete routine validation"
        )
        return result
    try:
        candidates = _candidates(routine)
    except ValueError as error:
        if evidence.get("validation_declaration") and error.args == (
            "No eligible trained checkpoint is available",
        ):
            result["reason"] = "No eligible trained checkpoint"
            return result
        raise
    if final not in candidates:
        result["reason"] = "The final actor has no eligible routine validation"
        return result
    _conditions(list(candidates.values()))
    result["routine_comparison"] = _roots(list(candidates.values()), declaration)
    ordered = sorted(
        candidates, key=lambda key: _key(candidates[key], declaration["rule"])
    )[: declaration["shortlist_size"]]
    if final not in ordered:
        ordered.append(final)
    confirmations = [
        row for row in evidence["records"] if row["purpose"] == "confirmation"
    ]
    compared: list[Record] = []
    for identifier in ordered:
        row = candidates[identifier]
        actor = evidence["actors"][identifier]
        request = _request(evidence, row, declaration, destination)
        expected = request["expected_task"]
        # Old tasks omit depth; accept only verified pre-Red-Zone records here.
        matches = [
            item
            for item in confirmations
            if all(
                item.get(name, 0.0 if name == "red_zone_depth" else None)
                == expected[name]
                for name in (
                    "checkpoint_id",
                    "actor_digest",
                    "env_steps",
                    "panel_digest",
                    "root",
                    "maps",
                    "seed_pairs",
                    "red_zone_depth",
                )
            )
        ]
        if len(matches) > 1:
            raise ValueError(
                "More than one confirmation task matches a shortlisted checkpoint"
            )
        result["shortlist"].append(
            {
                "checkpoint_id": identifier,
                "artifact_id": actor["artifact_id"],
                "actor_digest": actor["actor_digest"],
                "actor_path": actor["actor_path"],
                "routine_task_id": row["task_id"],
                "confirmation_task_id": matches[0]["task_id"] if matches else None,
            }
        )
        if matches:
            compared.append(matches[0])
        else:
            result["needs_confirmation"].append(request)
    if result["needs_confirmation"]:
        result.update(
            status="needs_confirmation",
            reason=(
                "Play the listed missing confirmation tasks, "
                "then write a linked decision"
            ),
        )
        return result
    _conditions(compared)
    result["confirmation_comparison"] = _roots(compared, declaration)
    eligible = _candidates(
        [{**row, "red_zone_depth": row.get("red_zone_depth", 0.0)} for row in compared]
    )
    winner = dict(
        min(
            (row for row in compared if row["checkpoint_id"] in eligible),
            key=lambda row: _key(row, declaration["rule"]),
        )
    )
    actor = evidence["actors"][winner["checkpoint_id"]]
    result.update(
        status="complete",
        winner={
            **winner,
            "artifact_id": actor["artifact_id"],
            "actor_path": actor["actor_path"],
        },
    )
    return result


def read_selection_decision(
    path: str | Path,
    *,
    verify_evidence: bool = False,
) -> Record:
    """Read a decision, optionally rechecking its actors, games and ranking.

    path names selection_decision.json. With verify_evidence=False, read the
    content hash only; incomplete decisions remain readable for progress reports.
    True requires a completed decision, rechecks recorded source files and
    rebuilds the decision from its games without writing files or loading models.
    A mismatch raises ValueError. Hashes are integrity checks, not signatures.
    """
    source = Path(path).resolve()
    result = _read(source)
    identity = result.pop("decision_id", None)
    if result.get("schema_version") != 1 or identity != _digest(result):
        raise ValueError("Selection decision content differs from its saved identity")
    checked = {**result, "decision_id": identity}
    if verify_evidence:
        if result.get("status") != "complete":
            raise ValueError(
                "Incomplete selection cannot supply completed selection evidence; "
                "omit selection to report ordinary progress"
            )
        declared = _declaration(result["declaration"])
        previous = (
            read_selection_decision(declared["previous_decision"])
            if declared["previous_decision"]
            else None
        )
        rebuilt = _build_selection(
            [Path(root) for root in result["run_dirs"]],
            declared,
            source.parent,
            previous,
        )
        if rebuilt != checked:
            raise ValueError(
                "Selection decision differs from its verified source evidence"
            )
    return checked


def _build_selection(
    roots: Sequence[Path],
    declared: Record,
    destination: Path,
    previous: Record | None,
) -> Record:
    """Verify sources and resolve a decision without creating output files.

    roots and declared are normalized inputs. destination names the decision
    folder solely for missing-task instructions. previous is the checked linked
    record, or None. Both writing and later report verification use this owner.
    """
    from marl_battlegrounds.training._selection_evidence import read_run_evidence

    if previous is not None:
        stable = set(declared) - {"previous_decision", "confirmation_results"}
        if any(declared[name] != previous["declaration"][name] for name in stable):
            raise ValueError(
                "A linked decision must keep the original declared selection rule"
            )
        if [str(root) for root in roots] != previous["run_dirs"]:
            raise ValueError(
                "A linked decision must keep the original ordered run list"
            )
    extra = [Path(path) for path in declared["confirmation_results"]]
    evidence = [
        read_run_evidence(
            root,
            confirmation_results=extra,
            confirmation_root=declared["confirmation_root"],
            confirmation_seed_pairs=declared["confirmation_seed_pairs"],
        )
        for root in roots
    ]
    for root, item in zip(roots, evidence, strict=True):
        item["run_dir"] = str(root)
    consumed = {
        str(Path(row["result_source"]).resolve())
        for item in evidence
        for row in item["records"]
        if "result_source" in row
    }
    if set(declared["confirmation_results"]) - consumed:
        raise ValueError("A supplied confirmation does not belong to these runs")
    if previous is not None:
        for source in previous["source_files"]:
            with Path(source["path"]).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != source["sha256"]:
                raise ValueError("Source evidence changed since the linked decision")
    if declared["across_runs"]:
        seeds = [item["seed"] for item in evidence if item["seed"] is not None]
        order = declared["seed_order"]
        if (
            len(order) != len(evidence)
            or len(set(seeds)) != len(seeds)
            or not set(seeds).issubset(order)
        ):
            raise ValueError(
                "Across-run seed_order must name every distinct run seed exactly once"
            )
        used_roots = {
            row["root"]
            for item in evidence
            for row in item["records"]
            if row["purpose"] in ("routine", "initialization")
        }
        used_roots.update(
            root for item in evidence for root in item.get("used_roots", ())
        )
        for item in evidence:
            item["comparison_used_roots"] = used_roots
        # A missing run has no verified seed yet. Keep it visible and incomplete;
        # its unverified place in the declared order cannot produce a winner.
    runs = [_run_decision(item, declared, destination) for item in evidence]
    routine_comparison = None
    if declared["across_runs"]:
        shortlisted = {
            (run["run_id"], row["routine_task_id"])
            for run in runs
            for row in run["shortlist"]
        }
        routine_comparison = _roots(
            [
                row
                for item in evidence
                for row in item["records"]
                if (item["run_id"], row["task_id"]) in shortlisted
            ],
            declared,
        )
    if previous is not None:
        for before, after in zip(previous["runs"], runs, strict=True):
            fields = ("checkpoint_id", "artifact_id", "actor_digest", "routine_task_id")
            if [{key: row[key] for key in fields} for row in before["shortlist"]] != [
                {key: row[key] for key in fields} for row in after["shortlist"]
            ]:
                raise ValueError("The linked decision's routine shortlist changed")
    status = (
        "complete"
        if all(row["status"] == "complete" for row in runs)
        else (
            "incomplete"
            if any(row["status"] == "incomplete" for row in runs)
            else "needs_confirmation"
        )
    )
    across = None
    if declared["across_runs"] and status == "complete":
        winners = [row["winner"] for row in runs]
        _conditions(winners)
        confirmation_comparison = _roots(winners, declared)
        order = {seed: index for index, seed in enumerate(declared["seed_order"])}
        best = min(
            runs,
            key=lambda run: (
                *_key(run["winner"], declared["rule"])[:2],
                order[run["seed"]],
                *_key(run["winner"], declared["rule"])[2:],
            ),
        )
        across = {
            "routine_comparison": routine_comparison,
            "confirmation_comparison": confirmation_comparison,
            "run_id": best["run_id"],
            "run_dir": best["run_dir"],
            "seed": best["seed"],
            "winner": best["winner"],
        }
    result: Record = {
        "schema_version": 1,
        "name": declared["name"],
        "status": status,
        "declaration": declared,
        "declaration_digest": _digest(declared),
        "run_dirs": [str(root) for root in roots],
        "runs": runs,
        "across_run_selection": across,
        "previous_decision_id": previous["decision_id"] if previous else None,
        "source_files": [file for item in evidence for file in item["source_files"]],
        "meaning": (
            "Separate declared re-selection; original run selections are unchanged"
        ),
    }
    result["decision_id"] = _digest(result)
    for source in result["source_files"]:
        with Path(source["path"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != source["sha256"]:
            raise ValueError("Source evidence changed while making the decision")
    return result


def reselect_checkpoint(
    run_dirs: Sequence[str | Path],
    *,
    declaration: str | Path | Mapping[str, Any],
    output_dir: str | Path,
) -> Record:
    """Write a new named checkpoint decision from verified saved validation.

    Parameters
    ----------
    run_dirs : sequence of str or Path
        Distinct saved training runs. Each normally receives its own decision.
        Missing, failed or unfinished runs remain visible and prevent an
        across-run winner. A missing seed leaves its declared order unverified.
    declaration : str, Path or mapping
        Schema 1 JSON with a required name. rule is saved (default),
        score_then_kills or score_then_step. shortlist_size defaults to two;
        the final checkpoint is always included. across_runs defaults to False;
        True requires the exact seed_order for ties between run finalists.
        allow_different_roots defaults to False. confirmation_root and
        confirmation_seed_pairs optionally declare fresh confirmation tasks.
        A child run defaults to its saved effective confirmation root and pair
        count. Fresh roots must differ from all routine and initialization roots
        used by that run and its inherited evidence; across-run comparisons
        also check the other runs' used roots.
        previous_decision links a pending decision. confirmation_results lists
        completed validation_summary.json files for that linked decision.
        Paths in a declaration file resolve beside that file; mapping paths
        resolve from the working directory. Unknown fields are rejected.
    output_dir : str or Path
        Exact new or empty directory outside every input run. Receives one
        immutable selection_decision.json. Missing-game calls write to separate
        task folders there only when the caller later runs them.

    Returns
    -------
    dict
        The saved decision, including its content ID, rule, run decisions and
        status: complete, needs_confirmation or incomplete. Missing confirmations
        include exact validate_checkpoint calls. No winner is invented.

    Raises
    ------
    ValueError
        A declaration, actor, task or saved game is corrupt or incompatible;
        output is occupied; roots are mixed without permission; or a linked
        decision changes its original rule or routine evidence.
    OSError
        Required files cannot be read or the new decision cannot be written.

    Notes
    -----
    Hashes actor payloads and reads saved game tables. Existing artifact imports
    may initialize JAX. Use JAX_PLATFORMS=cpu before Python starts for a separate
    saved-analysis process; an existing caller's backend is not changed here.
    Restores no actor or learner, calls no provider and runs no game. Original
    selections and run folders stay unchanged. This is a new analysis rule, not
    retrospective proof that it was the original study rule. Final actors remain
    explicitly named.
    """
    if isinstance(run_dirs, (str, Path)) or not run_dirs:
        raise ValueError("run_dirs must be a nonempty sequence of run directories")
    roots = [Path(path).resolve() for path in run_dirs]
    if len(set(roots)) != len(roots):
        raise ValueError("run_dirs must not repeat a directory")
    declared = _declaration(declaration)
    destination = Path(output_dir).resolve()
    if any(destination == root or destination.is_relative_to(root) for root in roots):
        raise ValueError("Write the separate decision outside the original run folders")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(
            "output_dir must be new or empty; use a new folder for each decision"
        )
    previous = None
    if declared["previous_decision"]:
        previous = read_selection_decision(declared["previous_decision"])
    elif declared["confirmation_results"]:
        raise ValueError("Additional confirmations require previous_decision")
    result = _build_selection(roots, declared, destination, previous)
    destination.mkdir(parents=True, exist_ok=True)
    from marl_battlegrounds.training._run_io import atomic_json

    # Reserve this new decision before using the shared durable JSON writer.
    # A second caller can never replace a decision that appeared during reading.
    reservation = destination / ".selection.lock"
    with reservation.open("x"):
        pass
    try:
        if any(path != reservation for path in destination.iterdir()):
            raise ValueError("output_dir became occupied while making the decision")
        atomic_json(destination / "selection_decision.json", result)
    finally:
        reservation.unlink()
    return result
