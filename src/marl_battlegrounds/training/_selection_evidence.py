"""Verify saved training selection evidence without loading a model or playing games.

The selection owner calls ``read_run_evidence`` before creating a new decision.
Checkpoint helpers verify actor files and retained learner descriptions; validation
owns task descriptions and score reduction; M8 owns saved episode tables. This
module joins those records, checks their declared conditions and records the bytes
read. Continuation references bind original learner, actor and task owners without
copying parent logs or rewriting their identities. It does not restore arrays,
call factories, initialize a learner or write files.
"""

from __future__ import annotations

# These shared private helpers keep scientific rules with their existing owners.
# pyright: reportPrivateUsage=false
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.training import validation
from marl_battlegrounds.training.analysis import summarize_validation

Record = dict[str, Any]


def _object(value: object, label: str) -> Record:
    """Require a JSON object for label; never replace malformed evidence by defaults."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(Record, value)


def _list(value: object, label: str) -> list[Any]:
    """Require a JSON list while retaining its checked heterogeneous row values."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[Any], value)


def _hash(path: Path) -> str:
    """Hash one regular evidence file in bounded chunks, without following links."""
    if path.resolve() != path or not path.is_file():
        raise ValueError(f"Selection evidence must be a regular file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _Snapshot:
    """Retain file hashes and reject evidence changed during one reader call."""

    def __init__(self) -> None:
        """Start an empty, call-local list of source files."""
        self.files: dict[Path, str] = {}

    def track(self, path: Path, digest: str | None = None) -> None:
        """Bind path to its current hash or an already verified payload digest."""
        value = _hash(path) if digest is None else digest
        if path in self.files and self.files[path] != value:
            raise ValueError(f"Selection evidence changed while reading: {path}")
        self.files[path] = value

    def read(self, path: Path) -> Any:  # noqa: ANN401 - JSON has mixed shapes.
        """Read finite JSON, rejecting duplicate keys and tracking its exact bytes."""
        self.track(path)

        def pairs(items: list[tuple[str, Any]]) -> Record:
            """Reject ambiguous duplicate JSON keys before making a dictionary."""
            result: Record = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"Repeated JSON key in {path}: {key}")
                result[key] = value
            return result

        def constant(value: str) -> None:
            """Reject nonfinite JSON numbers instead of accepting Python extensions."""
            raise ValueError(f"Nonfinite JSON value in {path}: {value}")

        return json.loads(
            path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant
        )

    def finish(self) -> list[Record]:
        """Recheck every bound file and return stable, sorted path/hash records."""
        for path, expected in self.files.items():
            if _hash(path) != expected:
                raise ValueError(f"Selection evidence changed while reading: {path}")
        return [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(self.files.items())
        ]


def _panel(path: Path, snapshot: _Snapshot) -> validation.FrozenPanel:
    """Read a frozen panel's saved identities without loading its methods.

    path names panel.json; snapshot records the bytes used. Return a FrozenPanel
    with no executable methods. Invalid schemas, hashes or registrations raise
    ValueError. No opponent artifact is loaded.

    Hashing and root rules use the validation owner. Serialized method identities
    use M8's normal registration owner. Historical opponent payloads need not
    remain available just to inspect games already played against them.
    """
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )

    content = _object(snapshot.read(path), "Frozen panel")
    entries = content.get("members")
    schema = content.get("schema_version")
    if (
        type(schema) is not int
        or schema not in (1, 2)
        or content.get("panel_digest") != validation._panel_digest(content)
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("Invalid frozen panel schema or digest")
    members: list[validation.PanelMember] = []
    entries = _list(content.get("members"), "Panel members")
    roots: Record = {}
    if schema == 2:
        if content.get("selection_schema_version") != 2:
            raise ValueError("Invalid frozen panel selection schema")
        validation._ranked_depth(content.get("ranking_evidence"))
        roots = validation._panel_roots(content.get("roots"))
        if roots != content.get("roots"):
            raise ValueError("Frozen panel lacks its complete purpose roots")
        for value in entries:
            row = _object(value, "Panel member")
            registration = _object(row.get("registration"), "Panel registration")
            identifier, normalized = normalize_system_registration(
                registration, phase="validation"
            )
            if (
                identifier != row.get("registration_id")
                or normalized != registration
                or registration.get("name") != row.get("name")
            ):
                raise ValueError("Frozen panel member registration differs")
            members.append(
                validation.PanelMember(
                    name=row["name"],
                    registration_id=identifier,
                    registration=registration,
                    reference=row.get("reference"),
                )
            )
    else:
        if (
            content.get("provisional") is not True
            or type(content.get("qualified")) is not bool
            or content.get("inference") != "Sampled masked categorical"
        ):
            raise ValueError("Invalid historical frozen panel")
        for value in entries:
            row = _object(value, "Historical panel member")
            member_path = Path(row["path"])
            if not member_path.is_absolute():
                member_path = path.parent / member_path
            members.append(
                validation.PanelMember(
                    row["name"],
                    member_path,
                    row["actor_digest"],
                    row["checkpoint_id"],
                    row["run_id"],
                    row["seed"],
                    row["env_steps"],
                )
            )
        if (
            [member.name for member in members] != ["Halfway", "Final"]
            or members[0].run_id != members[1].run_id
            or members[0].seed != members[1].seed
            or not 0 < members[0].env_steps < members[1].env_steps
        ):
            raise ValueError("Historical panel lineage or order differs")
        if content["qualified"]:
            evidence = _list(content.get("qualification_evidence"), "Qualification")
            if len(evidence) != 2:
                raise ValueError("Historical panel qualification evidence is missing")
            for member, result in zip(entries, evidence, strict=True):
                validation._qualify_random(member, _object(result, "Qualification"))
    if len({member.name for member in members}) != len(members):
        raise ValueError("Frozen panel repeats an opponent")
    return validation.FrozenPanel(
        path,
        content["panel_digest"],
        tuple(members),
        content.get("qualified", True),
        schema_version=schema,
        roots=roots,
    )


def _actor(root: Path, identifier: str, run: Record, snapshot: _Snapshot) -> Record:
    """Bind a fully verified actor export to its retained learner description.

    root is the original training run and identifier is its learner checkpoint ID. run
    carries the saved run config and provenance; snapshot tracks every read file.
    Return the checkpoint owner's actor identity with actor_path, artifact_id
    (the export ID) and checkpoint_id (the learner boundary). Conflicting files,
    settings or provenance raise ValueError.

    A pruned learner is valid ancestry; its missing numerical files are not read.
    The standalone actor must still be complete. Inference identity is distinct
    from its export directory's identity and its originating learner boundary.
    """
    from marl_battlegrounds.training import checkpoints

    checkpoints._digest(identifier, "Selection checkpoint ID")
    actor_path = root / "actors" / identifier
    parent_path = root / "checkpoints" / identifier
    snapshot.track(parent_path / "checkpoint_details.json")
    parent = checkpoints.read_checkpoint_description(parent_path)
    snapshot.track(actor_path / "actor_details.json")
    description = checkpoints.read_checkpoint_description(actor_path)
    identity = checkpoints.artifact_identity(actor_path)
    for relative, entry in description["files"].items():
        snapshot.track(actor_path / relative, entry["sha256"])
    config = run["config"]
    method = config.get("method", "mappo")
    settings = checkpoints._settings_block(config)
    metadata = parent["metadata"]
    if (
        parent["kind"] != "learner"
        or parent["checkpoint_id"] != identifier
        or description["kind"] != "actor"
        or identity["metadata"].get("checkpoint_id") != identifier
        or identity["run_id"] != run["run_id"]
        or identity["seed"] != config.get("seed")
        or type(identity["seed"]) is not int
        or identity["env_steps"] != parent["counters"]["env_steps"]
        or identity["weight_digest"] != parent["actor_digest"]
        or identity["schemas"] != parent["schemas"]
        or identity["schemas"] != run.get("schemas")
        or identity["input_scale"] != settings.get("input_scale", 1.0)
        or identity["spawn_frame"] != settings.get("spawn_frame", "world")
        or any(
            metadata.get(key) != run.get(key)
            for key in (
                "run_id",
                "config",
                "source",
                "dependencies",
                "execution",
                "continuation",
                "validation_declaration",
            )
        )
        or (
            method in ("qmix", "pqn_vdn")
            and identity.get("optimizer_steps") != parent["counters"]["updates"]
        )
    ):
        raise ValueError("Saved actor differs from its learner boundary or run")
    return {
        **identity,
        "actor_path": str(actor_path),
        "artifact_id": identity["checkpoint_id"],
        "checkpoint_id": identifier,
    }


def _pass_rows(
    path: Path,
    task: Record,
    actor: Record,
    panel: validation.FrozenPanel,
    index: int,
    snapshot: _Snapshot,
) -> list[Record]:
    """Check one saved M8 pass's methods, rules and roots, then read its rows.

    path names the recorded pass directory. task is the checked validation task,
    actor is its verified frozen actor, and index selects the opponent in panel.
    snapshot tracks the manifest and raw tables. Return completed game rows from
    validation's reader, with actual kills when the task records Red Zone rules.
    Reject changed methods, conditions or generated map/seed/spawn assignments.
    Read saved JSON and tables only; do not rebuild a model or run a game.
    """
    from marl_battlegrounds.evaluation.evaluation_conditions import same_float32
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )
    from marl_battlegrounds.evaluation.results import load_results
    from marl_battlegrounds.evaluation.run_writer import _json_bytes

    manifest = _object(snapshot.read(path / "run_details.json"), "M8 run")
    for name in manifest.get("tables", {}):
        snapshot.track(path / name)
    member = panel.members[index]
    pass_id = validation.validation_pass_id(task["task_id"], member.name)
    saved = load_results(path, phase="validation", pass_id=pass_id)
    if saved.status != "complete" or len(saved.metadata["passes"]) != 1:
        raise ValueError("Selection validation pass is incomplete or ambiguous")
    entry = next(iter(saved.metadata["passes"].values()))
    details = entry["details"]
    contract = details.get("evaluation_contract", {})
    policies = entry["policies"]
    focal, opponent = policies["team_a"], policies["team_b"]
    if (
        focal.get("checkpoint") != actor["actor_digest"]
        or focal.get("variables_frozen") is not True
    ):
        raise ValueError("Validation Team A is not the declared frozen actor")
    for team, description in policies.items():
        identifier, _ = normalize_system_registration(description, phase="validation")
        if identifier != entry.get("system_ids", {}).get(team):
            raise ValueError("Saved validation method registration differs")
    if panel.schema_version == 2:
        if entry["system_ids"]["team_b"] != member.registration_id:
            raise ValueError("Validation opponent differs from its frozen panel")
        root = task["members"][index]["root"]
    else:
        if opponent.get("checkpoint") != member.actor_digest:
            raise ValueError("Historical validation opponent differs from its panel")
        root = task["root"]
    games = len(task["maps"]) * task["seed_pairs"] * 2
    options: Record = {
        "seed": root,
        "spawn_mode": "paired",
        "score_threshold": 20,
        "max_steps": 300,
        "metrics": "priority",
        "save_replays": 0,
        "system_roster": None,
        "opponent_roster": None,
        "full_metrics_episodes": [],
        "replay_episodes": [],
    }
    if "red_zone_depth" in task:
        options["red_zone_depth"] = task["red_zone_depth"]
    if (
        contract.get("version") != 1
        or contract.get("schedule_kind") != "generated"
        or contract.get("map_selection") != "explicit"
        or contract.get("spawn_mode") != "paired"
        or contract.get("options") != options
        or details.get("seed") != root
        or details.get("num_episodes") != games
        or details.get("phase") != "validation"
        or details.get("pass_id") != pass_id
        or details.get("metrics") != "priority"
        or [choice["map_id"] for choice in contract.get("source_choices", [])]
        != task["maps"]
    ):
        raise ValueError("Saved validation rules, maps or root differ from its task")
    configs = manifest["configurations"]
    for key, config in configs.items():
        if hashlib.sha256(_json_bytes(config)).hexdigest() != key:
            raise ValueError("Saved validation configuration digest differs")
    sources = {
        choice["map_id"]: choice["source_config_id"]
        for choice in contract["source_choices"]
    }
    episodes = _object(entry.get("episodes"), "Validation schedule")
    if set(episodes) != {str(value) for value in range(1, games + 1)}:
        raise ValueError("Saved validation schedule has missing or extra games")
    for identifier, episode in episodes.items():
        # Generated M8 passes cycle through maps once per pair of spawn ends.
        # Read this integer schedule without restoring configs or actor arrays.
        game = int(identifier)
        pair = (game - 1) // 2
        if (
            episode.get("episode_id") != game
            or episode.get("seed_id") != pair + 1
            or episode.get("map_id") != task["maps"][pair % len(task["maps"])]
            or episode.get("spawn_locations") != (game - 1) % 2
            or episode.get("expected_horizon") != 300
            or episode.get("initial_state_digest") is not None
        ):
            raise ValueError(
                "Saved validation game differs from its generated schedule"
            )
        source = configs[sources[episode["map_id"]]]
        current = configs[episode["configuration_digest"]]
        expected = dict(source)
        if episode["spawn_locations"] == 1:
            expected["team_spawn_pad_positions"] = source["team_spawn_pad_positions"][
                ::-1
            ]
        if (
            episode.get("source_config_id") != sources[episode["map_id"]]
            or current != expected
            or not same_float32(
                source.get("team_deathmatch_red_zone_depth", 0.0),
                task.get("red_zone_depth", 0.0),
            )
            or source.get("team_deathmatch_score_threshold") != 20
            or source.get("max_steps") != 300
            or source.get("task_mode") != 1
            or source["agent_profile"]["class_ids"] != [1, 2, 3, 4, 5] * 2
            or source["agent_profile"]["active_mask"] != [True] * 10
        ):
            raise ValueError("Saved validation configuration or spawn pair differs")
    rows = validation._rows(
        path, pass_id=pass_id, opponent=member.name, kills="red_zone_depth" in task
    )
    if len(rows) != games or len({row["episode_id"] for row in rows}) != games:
        raise ValueError("Saved validation rows have missing or extra games")
    for row in rows:
        declared = episodes.get(str(row["episode_id"]))
        if declared is None or any(
            row.get(name) != declared.get(name)
            for name in ("map_id", "seed_id", "spawn_locations")
        ):
            raise ValueError("Saved validation row differs from its generated schedule")
    return rows


def _record(
    path: Path,
    original: Record | None,
    run: Record,
    panel: validation.FrozenPanel,
    actors: dict[str, Record],
    root: Path,
    snapshot: _Snapshot,
    confirmation_root: int | None,
    confirmation_seed_pairs: int | None = None,
) -> Record:
    """Verify one original or supplied task and recompute its saved score summary.

    path names validation_summary.json. original is the run's matching result,
    or None for a supplied confirmation. run and panel hold checked metadata;
    root locates local actors. actors may also contain checked inherited actors
    from read_inherited_candidates; their original ownership stays unchanged.
    Cache verified actors by learner ID in actors
    and track every source in snapshot. confirmation_root applies only to a
    supplied confirmation; None uses the panel default. confirmation_seed_pairs
    is the separately declared positive count for supplied confirmations; None
    uses the original run config. Original records always keep their saved
    counts. Return the original fields plus result_source and, for verified
    pre-Red-Zone games, their kill
    difference. Reject tasks, games, sampling facts or summaries that disagree.
    No saved file changes and no model or game executes.
    """
    saved = _object(snapshot.read(path), "Validation summary")
    if original is not None and saved != {
        key: value for key, value in original.items() if key != "elapsed_seconds"
    }:
        raise ValueError("Run validation record differs from its saved summary")
    purpose = saved.get("purpose")
    if purpose not in ("routine", "initialization", "confirmation") or (
        original is None and purpose != "confirmation"
    ):
        raise ValueError(
            "Selection needs routine, initialization or confirmation evidence"
        )
    identifier = saved["checkpoint_id"]
    if identifier not in actors:
        actors[identifier] = _actor(root, identifier, run, snapshot)
    actor = actors[identifier]
    config = run["config"]
    depth = config.get("red_zone_depth")
    # Explicit new confirmation games for a pre-rule run use its historical zero.
    if original is None and depth is None and "red_zone_depth" in saved:
        depth = 0.0
    pairs = (
        confirmation_seed_pairs
        if original is None and confirmation_seed_pairs is not None
        else config[
            "confirmation_seed_pairs"
            if purpose == "confirmation"
            else "routine_seed_pairs"
        ]
    )
    declaration = validation.saved_validation_declaration(run, panel)
    if declaration is not None:
        if original is not None or confirmation_seed_pairs is None:
            pairs = declaration[
                "confirmation_seed_pairs"
                if purpose == "confirmation"
                else "routine_seed_pairs"
            ]
        expected_declaration = dict(declaration)
        if original is None:
            expected_declaration["confirmation_seed_pairs"] = pairs
            if confirmation_root is not None:
                expected_declaration["roots"] = {
                    **declaration["roots"],
                    "confirmation": confirmation_root,
                }
        expected = validation.declared_panel_task(
            expected_declaration,
            panel,
            checkpoint_id=identifier,
            actor_digest=actor["actor_digest"],
            env_steps=actor["env_steps"],
            purpose=purpose,
        )
    else:
        expected = validation.panel_task_description(
            checkpoint_id=identifier,
            actor_digest=actor["actor_digest"],
            env_steps=actor["env_steps"],
            panel=panel,
            purpose=purpose,
            seed_pairs=pairs,
            red_zone_depth=depth,
            root_seed=confirmation_root if original is None else None,
        )
    task = _object(snapshot.read(path.parent / "task.json"), "Validation task")
    if task != expected or any(
        saved.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Saved validation task differs from its declared identity")
    paths = _list(saved.get("pass_paths"), "Validation pass paths")
    if len(paths) != len(panel.members):
        raise ValueError("Validation needs one saved pass per opponent")
    rows: list[Record] = []
    facts: dict[str, Record] = {}
    for index, value in enumerate(paths):
        pass_path = Path(value)
        if pass_path.parent != path.parent / f"opponent-{index}":
            raise ValueError("Saved validation pass is outside its task")
        rows.extend(_pass_rows(pass_path, task, actor, panel, index, snapshot))
        sampling_path = pass_path.parent / "sampling_facts.json"
        if sampling_path.exists():
            snapshot.read(sampling_path)
        member = panel.members[index]
        facts[member.name] = validation._read_pass_sampling(
            pass_path,
            task_id=task["task_id"],
            pass_id=validation.validation_pass_id(task["task_id"], member.name),
        )
    sampling = None
    if "sampling_evidence" in saved:
        sampling = validation.validation_sampling_evidence(
            rows,
            facts=facts,
            scheduled_games=len(task["maps"]) * pairs * 2 * len(panel.members),
            independent_opponents=panel.schema_version == 2,
        )
        if sampling != saved["sampling_evidence"]:
            raise ValueError("Saved sampling evidence differs from its pass facts")
    reduced = summarize_validation(
        rows,
        maps=task["maps"],
        opponents=[member.name for member in panel.members],
        seed_pairs=pairs,
        independent_opponents=panel.schema_version == 2,
        actual_kills="red_zone_depth" in task,
        bootstrap_draws=saved["bootstrap_draws"],
        bootstrap_seed=saved["bootstrap_seed"],
        sampling_evidence=sampling,
    )
    expected_summary = {
        **task,
        **reduced,
        "pass_paths": paths,
        **validation._method_fields(actor),
    }
    if expected_summary != saved:
        raise ValueError("Validation summary differs from its saved games")
    result = {**(saved if original is None else original), "result_source": str(path)}
    if "mean_kill_difference" not in result and "red_zone_depth" not in task:
        # Only verified pre-rule task schemas make points equal to kills.
        differences = [row["team_a_score"] - row["team_b_score"] for row in rows]
        result["mean_kill_difference"] = sum(differences) / len(differences)
    return result


def _panel_path(root: Path, run: Mapping[str, Any]) -> Path:
    """Resolve a saved panel with its existing path rule.

    Child declarations save absolute paths; relative declaration paths use the
    run file's directory. Old config strings retain train's working-directory
    path rule. A missing explicit path keeps the original local panel layout.
    """
    declaration = run.get("validation_declaration")
    supplied = (
        cast(Mapping[str, Any], declaration).get("panel_path")
        if isinstance(declaration, Mapping)
        else run["config"].get("validation_panel")
    )
    path = (
        root / "validation_panel" / "panel.json" if supplied is None else Path(supplied)
    )
    if path.is_absolute():
        return path
    return root / path if isinstance(declaration, Mapping) else path.absolute()


def _reference_context(
    checkpoint: Path, snapshot: _Snapshot
) -> tuple[Path, Record, Record, dict[str, Record], validation.FrozenPanel | None]:
    """Bind a parent boundary to its original run, ancestry and frozen panel.

    checkpoint is a full or retained learner folder; numerical payloads are not
    restored. snapshot records immutable descriptions. Reject a different run,
    config, source, dependency, execution or validation declaration. Return the
    run path, run metadata, boundary description, active ancestry and saved panel.
    """
    from marl_battlegrounds.training import checkpoints
    from marl_battlegrounds.training._run_io import checkpoint_ancestry

    if checkpoint.parent.name != "checkpoints" or checkpoint.resolve() != checkpoint:
        raise ValueError(
            "Inherited boundary must be inside its original checkpoint folder"
        )
    root = checkpoint.parent.parent
    run = _object(snapshot.read(root / "run_details.json"), "Parent training run")
    snapshot.track(checkpoint / "checkpoint_details.json")
    boundary = checkpoints.read_checkpoint_description(checkpoint)
    if boundary["checkpoint_id"] != checkpoint.name or boundary["kind"] != "learner":
        raise ValueError("Inherited boundary identity differs")
    for field in (
        "run_id",
        "config",
        "source",
        "dependencies",
        "execution",
        "continuation",
        "validation_declaration",
    ):
        if boundary["metadata"].get(field) != run.get(field):
            raise ValueError("Inherited boundary differs from its original run")
    ancestry = checkpoint_ancestry(root, boundary)
    for identifier in ancestry:
        snapshot.track(root / "checkpoints" / identifier / "checkpoint_details.json")
    path = _panel_path(root, run)
    panel = _panel(path, snapshot) if path.exists() else None
    if run.get("panel_digest") != (None if panel is None else panel.digest):
        raise ValueError("Parent run and frozen panel identities differ")
    validation.saved_validation_declaration(run, panel)
    return root, run, boundary, ancestry, panel


def freeze_inherited_candidates(
    checkpoint: Path, *, declaration: Mapping[str, Any]
) -> list[Record]:
    """Pin eligible parent task sources at one checked learner boundary.

    checkpoint is the original chosen parent learner. declaration is the child's
    frozen validation declaration. Return small immutable ancestry references;
    no parent logs, replays or payloads are copied. The caller has already fully
    restored the parent before starting any child output. Task discovery reads
    the parent's current result index, but only actor IDs in the chosen ancestry
    can enter. A completed task for that boundary is allowed even when its saved
    host state still marked validation pending. Descendants and abandoned branches
    cannot enter. All original task and actor checks remain with their owners.
    """
    snapshot = _Snapshot()
    checkpoint = checkpoint.absolute()
    root, run, boundary, ancestry, panel = _reference_context(checkpoint, snapshot)
    inherited = run.get("continuation", {}).get("inherited_candidates", [])
    references = [dict(item) for item in _list(inherited, "Inherited candidates")]
    prior: Record = {"actors": {}, "source_files": []}
    if references:
        prior = read_inherited_candidates(
            references, declaration=run["validation_declaration"]
        )
        for item in prior["source_files"]:
            snapshot.track(Path(item["path"]), item["sha256"])
    index = root / "validation_results.json"
    # Discovery is mutable; only verified immutable task files are retained.
    before = _hash(index) if index.exists() else None
    values: object = json.loads(index.read_bytes()) if before is not None else []
    if isinstance(values, dict):
        values = cast(Record, values).get("results")
    rows = _list(values, "Parent validation results")
    tasks: list[Record] = []
    actors: dict[str, Record] = dict(prior["actors"])
    for raw in rows:
        row = _object(raw, "Parent validation result")
        identifier = row.get("checkpoint_id")
        if identifier not in ancestry and not (
            row.get("purpose") == "confirmation" and identifier in prior["actors"]
        ):
            continue
        if panel is None:
            raise ValueError("Parent validation has no declared frozen panel")
        if (
            ancestry[identifier]["counters"]["env_steps"]
            if identifier in ancestry
            else prior["actors"][identifier]["env_steps"]
        ) > boundary["counters"]["env_steps"]:
            raise ValueError("Inherited actor follows the chosen parent boundary")
        purpose = row.get("purpose")
        if (
            purpose == "confirmation"
            and boundary["counters"]["env_steps"] != run["config"]["total_env_steps"]
        ):
            continue
        path = (
            root / "validation" / f"{purpose}-{identifier}" / "validation_summary.json"
        )
        if not path.exists():
            # An unfinished original task is not candidate evidence.
            continue
        checked = _record(path, row, run, panel, actors, root, snapshot, None)
        tasks.append(
            {
                "summary_path": str(path),
                "checkpoint_id": identifier,
                "task_id": checked["task_id"],
                "purpose": checked["purpose"],
            }
        )
    if before is not None and _hash(index) != before:
        raise ValueError("Parent validation index changed during candidate discovery")
    if tasks:
        references.append(
            {
                "schema_version": 1,
                "parent_checkpoint": str(checkpoint),
                "parent_checkpoint_id": boundary["checkpoint_id"],
                "run_dir": str(root),
                "run_id": run["run_id"],
                "tasks": tasks,
                "source_files": snapshot.finish(),
            }
        )
    read_inherited_candidates(references, declaration=declaration)
    return references


def read_inherited_candidates(
    references: Sequence[Mapping[str, Any]],
    *,
    declaration: Mapping[str, Any],
    _active: frozenset[str] = frozenset(),
    _verified: dict[str, tuple[Record, frozenset[str]]] | None = None,
) -> Record:
    """Verify frozen parent sources and return compatible original candidates.

    references come from freeze_inherited_candidates and retain only immutable
    source paths/hashes. declaration supplies the child's panel, maps/rules and
    root comparison permission. Return records, actors keyed by original learner
    ID, source_files and used_roots. Incompatible scores remain excluded, while
    their actual routine/initialization roots remain freshness facts. No actor or
    task is relabelled, no arrays are restored, and no file changes.

    _active is the internal set of parent paths being checked; callers leave it
    empty. It rejects cycles while following earlier child confirmations.
    _verified holds completed recursive results within this one top-level read.
    Each cache hit retains the original source hashes, and the enclosing final
    snapshot still hashes every consumed file. No result is cached across calls.
    Missing or changed pinned files, ambiguous IDs and an unpermitted comparison
    of different routine roots raise ValueError or the original I/O error.
    """
    verified = {} if _verified is None else _verified
    cache_key = validation._digest([references, declaration])
    cached = verified.get(cache_key)
    if cached is not None:
        if _active & cached[1]:
            raise ValueError("Inherited candidate ancestry is cyclic")
        return cached[0]
    parent_paths: set[str] = set()
    result: Record = {"records": [], "actors": {}, "source_files": [], "used_roots": []}
    snapshot = _Snapshot()
    seen_tasks: set[str] = set()
    used_roots: set[int] = set()
    for value in references:
        reference = _object(value, "Inherited candidate reference")
        if (
            set(reference)
            != {
                "schema_version",
                "parent_checkpoint",
                "parent_checkpoint_id",
                "run_dir",
                "run_id",
                "tasks",
                "source_files",
            }
            or type(reference["schema_version"]) is not int
            or reference["schema_version"] != 1
        ):
            raise ValueError("Invalid inherited candidate reference")
        pinned = _list(reference["source_files"], "Inherited source files")
        expected_files: dict[Path, str] = {}
        for source in pinned:
            source = _object(source, "Inherited source")
            path = Path(source["path"])
            digest = _hash(path)
            if digest != source["sha256"] or path in expected_files:
                raise ValueError("Inherited source changed or repeats a path")
            expected_files[path] = digest
        if reference["parent_checkpoint"] in _active:
            raise ValueError("Inherited candidate ancestry is cyclic")
        parent_paths.add(reference["parent_checkpoint"])
        local = _Snapshot()
        root, run, boundary, ancestry, panel = _reference_context(
            Path(reference["parent_checkpoint"]), local
        )
        if (
            reference["run_dir"] != str(root)
            or reference["run_id"] != run["run_id"]
            or reference["parent_checkpoint_id"] != boundary["checkpoint_id"]
            or panel is None
        ):
            raise ValueError("Inherited reference differs from its original boundary")
        prior: Record = {"actors": {}, "source_files": [], "used_roots": []}
        previous = run.get("continuation", {}).get("inherited_candidates", [])
        if previous:
            prior = read_inherited_candidates(
                previous,
                declaration=run["validation_declaration"],
                _active=_active | {reference["parent_checkpoint"]},
                _verified=verified,
            )
            previous_key = validation._digest([previous, run["validation_declaration"]])
            parent_paths.update(verified[previous_key][1])
            for item in prior["source_files"]:
                local.track(Path(item["path"]), item["sha256"])
            used_roots.update(prior["used_roots"])
        actors: dict[str, Record] = dict(prior["actors"])
        for raw in _list(reference["tasks"], "Inherited tasks"):
            task = _object(raw, "Inherited task")
            identifier = task.get("checkpoint_id")
            if not isinstance(identifier, str):
                raise ValueError("Inherited checkpoint identity must be text")
            if identifier not in ancestry and not (
                task.get("purpose") == "confirmation" and identifier in prior["actors"]
            ):
                raise ValueError(
                    "Inherited candidate is outside the chosen active ancestry"
                )
            if (
                task.get("purpose") == "confirmation"
                and boundary["counters"]["env_steps"]
                != run["config"]["total_env_steps"]
            ):
                raise ValueError(
                    "Inherited confirmation follows the chosen training boundary"
                )
            path = Path(task["summary_path"])
            expected = (
                root
                / "validation"
                / f"{task.get('purpose')}-{identifier}"
                / "validation_summary.json"
            )
            if path != expected:
                raise ValueError("Inherited task path differs from its original owner")
            original = _object(local.read(path), "Inherited summary")
            row = _record(path, original, run, panel, actors, root, local, None)
            if row["checkpoint_id"] != identifier or row["task_id"] != task.get(
                "task_id"
            ):
                raise ValueError("Inherited task identity differs")
            if row["purpose"] in ("routine", "initialization"):
                used_roots.add(row["root"])
            compatible = (
                declaration.get("inherit_parent_candidates", True)
                and row["panel_digest"] == declaration["panel_digest"]
                and row.get("red_zone_depth") == declaration.get("red_zone_depth")
                and row.get("selection_schema_version", 1)
                == (2 if declaration.get("panel_schema_version") == 2 else 1)
                and row["maps"] == list(validation.VALIDATION_MAPS)
            )
            if not compatible:
                continue
            if (
                row["purpose"] == "routine"
                and row["root"] != declaration["roots"]["routine"]
                and not declaration["allow_different_roots"]
            ):
                raise ValueError(
                    "Different inherited roots require allow_different_roots=True"
                )
            if row["task_id"] in seen_tasks:
                continue
            seen_tasks.add(row["task_id"])
            actor = actors[identifier]
            previous_actor = result["actors"].get(identifier)
            if previous_actor is not None and previous_actor != actor:
                raise ValueError(
                    "Inherited learner identity has conflicting actor owners"
                )
            result["actors"][identifier] = actor
            result["records"].append(row)
        actual_files = {Path(item["path"]): item["sha256"] for item in local.finish()}
        if actual_files != expected_files:
            raise ValueError(
                "Inherited source list differs from the verified task sources"
            )
        for path, digest in actual_files.items():
            snapshot.track(path, digest)
    result["used_roots"] = sorted(used_roots)
    roots = declaration.get("roots", {})
    if roots:
        validation.check_confirmation_roots(
            roots["confirmation"], result["records"], used_roots=result["used_roots"]
        )
    result["source_files"] = snapshot.finish()
    verified[cache_key] = (result, frozenset(parent_paths))
    return result


def read_run_evidence(
    root: Path,
    *,
    confirmation_results: Sequence[Path] = (),
    confirmation_root: int | None = None,
    confirmation_seed_pairs: int | None = None,
) -> Record:
    """Read checked selection inputs from one saved training run, without writes.

    Parameters
    ----------
    root : Path
        Exact run directory. Missing or unfinished runs return their status and
        reasons; malformed existing records raise instead of becoming missing data.
        A child also reads its frozen compatible inherited task references under
        their original owners; its final actor still belongs to the child.
    confirmation_results : sequence of Path, default=()
        Additional validation_summary.json files from explicit confirmation calls
        for this same run and frozen panel. Their tasks and games are checked too.
    confirmation_root : int or None, default=None
        Separately declared root for supplied confirmations only. None uses the
        child's saved effective root, or the panel default for an ordinary run.
        Original run records always use their saved declaration.
        Historical panels do not allow an override. The selection owner checks
        freshness against every relevant routine and initialization root.
    confirmation_seed_pairs : int or None, default=None
        Separately declared number of seed pairs per map and opponent for
        supplied confirmations only. Must be a positive plain integer; booleans
        are refused. None uses the original run's confirmation count. Original
        routine, initialization and confirmation records keep their run's counts.

    Returns
    -------
    dict
        Run ID, seed, config, status, reason/failures, metadata-only panel and path,
        final checkpoint ID (None while unfinished), original selection, checked
        records, actors indexed by learner checkpoint ID, and source_files with
        path/SHA256. Records add result_source only in the returned copy. Actors
        include actor_path, artifact_id, checkpoint_id and actor_digest. Panel
        methods is empty: this panel describes evidence and cannot execute games.
        Child results also include validation_declaration and used_roots. Their
        actors and records may name explicitly verified original parent paths.

    Raises
    ------
    ValueError, OSError
        Existing evidence is corrupt, conflicts with its declaration, names another
        run or changes while reading. No original file or historical meaning changes.

    Notes
    -----
    Existing shared imports may initialize JAX; this is not a backend-free API.
    No model arrays are restored. Actor payloads are hashed; pruned learner
    payloads are not needed.
    This verifies saved evidence, not learning quality or independent training seeds.
    """
    if confirmation_seed_pairs is not None and (
        type(confirmation_seed_pairs) is not int or confirmation_seed_pairs < 1
    ):
        raise ValueError("confirmation_seed_pairs must be a positive plain integer")
    root = root.absolute()
    snapshot = _Snapshot()
    result: Record = {
        "run_id": root.name,
        "seed": None,
        "config": {},
        "status": "missing",
        "reason": None,
        "failures": [],
        "panel_path": None,
        "panel": None,
        "final_checkpoint_id": None,
        "original_selection": None,
        "records": [],
        "actors": {},
        "source_files": [],
    }
    if not (root / "run_details.json").exists():
        result["reason"] = "Run details are missing"
        return result
    run = _object(snapshot.read(root / "run_details.json"), "Training run")
    config = _object(run.get("config"), "Training config")
    state = (
        _object(snapshot.read(root / "status.json"), "Training status")
        if (root / "status.json").exists()
        else {}
    )
    result.update(
        run_id=run["run_id"],
        seed=config.get("seed"),
        config=config,
        status=state.get("status", "incomplete"),
    )
    if state.get("run_id", run["run_id"]) != run["run_id"]:
        raise ValueError("Training status names another run")
    if (root / "checkpoint_recovery.json").exists():
        snapshot.read(root / "checkpoint_recovery.json")
        result.update(status="recovering", reason="Checkpoint recovery is unfinished")
        result["source_files"] = snapshot.finish()
        return result
    panel_path = _panel_path(root, run)
    if panel_path.exists():
        panel = _panel(panel_path, snapshot)
        if panel.digest != run.get("panel_digest"):
            raise ValueError("Training run and frozen panel identities differ")
        result.update(panel_path=panel_path, panel=panel)
    else:
        panel = None
        result["reason"] = "Frozen validation panel is missing"
    declaration = validation.saved_validation_declaration(run, panel)
    inherited: Record = {
        "records": [],
        "actors": {},
        "source_files": [],
        "used_roots": [],
    }
    if declaration is not None:
        inherited = read_inherited_candidates(
            run.get("continuation", {}).get("inherited_candidates", []),
            declaration=declaration,
        )
        result["records"].extend(inherited["records"])
        result["actors"].update(inherited["actors"])
        result["validation_declaration"] = declaration
        result["used_roots"] = inherited["used_roots"]
        for item in inherited["source_files"]:
            snapshot.track(Path(item["path"]), item["sha256"])
    raw_values: object = (
        snapshot.read(root / "validation_results.json")
        if (root / "validation_results.json").exists()
        else []
    )
    if isinstance(raw_values, dict):
        raw_values = cast(Record, raw_values).get("results")
    values = _list(raw_values, "Validation results")
    if (values or confirmation_results) and panel is None:
        raise ValueError("Validation results have no frozen panel")
    seen: set[str] = {row["task_id"] for row in result["records"]}
    for original in [*values, *confirmation_results]:
        external = isinstance(original, Path)
        if isinstance(original, Path):
            path = original.absolute()
            supplied = _object(snapshot.read(path), "Supplied confirmation")
            task = _object(snapshot.read(path.parent / "task.json"), "Supplied task")
            unsigned = {key: value for key, value in task.items() if key != "task_id"}
            if (
                task.get("task_id") != validation._digest(unsigned)
                or any(supplied.get(key) != value for key, value in task.items())
                or supplied.get("purpose") != "confirmation"
            ):
                raise ValueError("Supplied confirmation has an invalid task")
            if (
                supplied.get("checkpoint_id") not in inherited["actors"]
                and not (root / "actors" / str(supplied.get("checkpoint_id"))).exists()
            ):
                # The caller checks that another supplied run consumes this path.
                continue
            row = None
        else:
            row = _object(original, "Validation record")
            path = (
                root
                / "validation"
                / f"{row['purpose']}-{row['checkpoint_id']}"
                / "validation_summary.json"
            )
        if not path.exists() and not external and result["status"] != "complete":
            result["failures"].append(f"Validation evidence is missing: {path}")
            continue
        assert panel is not None
        checked = _record(
            path,
            row,
            run,
            panel,
            result["actors"],
            root,
            snapshot,
            confirmation_root,
            confirmation_seed_pairs,
        )
        if checked["task_id"] in seen:
            raise ValueError("Selection evidence repeats a validation task")
        seen.add(checked["task_id"])
        result["records"].append(checked)
    selection_path = root / "selection.json"
    if selection_path.exists():
        result["original_selection"] = _object(
            snapshot.read(selection_path), "Original selection"
        )
    if result["status"] == "complete":
        final = state.get("final_actor")
        if not isinstance(final, str) or Path(final).parent != root / "actors":
            raise ValueError("Completed training has no exact final actor")
        identifier = Path(final).name
        if identifier not in result["actors"]:
            result["actors"][identifier] = _actor(root, identifier, run, snapshot)
        if result["actors"][identifier]["env_steps"] != config["total_env_steps"]:
            raise ValueError("Final actor differs from the completed training budget")
        result["final_checkpoint_id"] = identifier
    else:
        result["reason"] = (
            result["reason"] or state.get("reason") or "Training run is unfinished"
        )
    result["source_files"] = snapshot.finish()
    return result
