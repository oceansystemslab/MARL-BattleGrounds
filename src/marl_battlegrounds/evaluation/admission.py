"""Keep maintainer admission decisions separate from researcher tournament runs.

These private host helpers manage one explicitly selected local store. They keep
immutable snapshots and approval evidence, a qualified submission queue, and one
monthly edition. Membership changes only after complete game evidence and the
retained field's refit are durable. Nothing here downloads assets, edits the
installed catalog, chooses scientific rules, or publishes to a remote service.
"""

from __future__ import annotations

import copy
import fcntl
import os
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

# Shared private record/writer helpers remain the owners of their contracts.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.evaluation.tournament_config import (
    canonical_json,
    load_tournament_config,
    read_config_json,
    snapshot_identity,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.results import CanonicalTournamentResult
    from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier
    from marl_battlegrounds.evaluation.tournament_records import TournamentRecords
    from marl_battlegrounds.evaluation.tournament_reuse import ReusePlan

_STATE_FORMAT = "marlbg-admission-state"
_RULE_FIELDS = (
    "games_per_opponent",
    "resource_limits",
    "timing_definitions",
    "scientific_eligibility",
    "reproduction",
    "public_test_feedback",
    "episode_local_adaptation",
    "revised_method_policy",
    "tied_incumbent_eviction",
)
_GAME_GATES = _RULE_FIELDS[:7]
_LONDON = ZoneInfo("Europe/London")


def _utc_now() -> datetime:
    """Return the store commit clock in UTC; tests may replace this clock."""
    return datetime.now(UTC)


def _instant(value: str | datetime, label: str) -> datetime:
    """Parse an offset-bearing ISO instant and return UTC, rejecting naive times."""
    try:
        result = datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO timestamp with an offset") from error
    if not isinstance(cast(object, result), datetime) or result.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone offset")
    return result.astimezone(UTC)


def _text(value: object, label: str) -> str:
    """Return nonempty text or explain the missing host record field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _digest(value: object, label: str) -> str:
    """Require a lowercase SHA-256 identity before using it as a local filename."""
    result = _text(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _identity(value: object) -> str:
    """Hash a finite JSON value through the shared tournament encoding."""
    return sha256(canonical_json(value)).hexdigest()


def _sync_directory(path: Path) -> None:
    """Make a local directory's preceding entry changes durable."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_directory(path: Path) -> None:
    """Create and synchronize missing parents without changing existing entries."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        _sync_directory(directory.parent)


def _publish_json(path: Path, value: object) -> None:
    """Reuse the writer's atomic JSON publication, then synchronize its directory."""
    from marl_battlegrounds.evaluation.run_writer import _atomic_json

    _atomic_json(path, value)
    _sync_directory(path.parent)


def _immutable_json(root: Path, group: str, identity: str, value: object) -> Path:
    """Write one content-addressed record; reject a conflicting existing file."""
    directory = root / group
    _make_directory(directory)
    path = directory / f"{_digest(identity, group)}.json"
    if path.exists():
        if canonical_json(read_config_json(path)) != canonical_json(value):
            raise ValueError(f"Immutable {group} record has conflicting content")
        _sync_directory(directory)
    else:
        _publish_json(path, value)
    return path


@contextmanager
def _locked_store(store: str | Path) -> Generator[tuple[Path, dict[str, Any]]]:
    """Lock an existing store directory and read its last atomic state snapshot.

    A concurrent owner raises ValueError rather than waiting indefinitely. The
    lock creates no file. Callers must publish changed state explicitly; an
    exception before that publication leaves the previous membership selected.
    """
    root = Path(store).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Admission store does not exist; initialize it explicitly")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another maintainer operation owns this store") from error
        state = read_config_json(root / "admission_state.json")
        if state.get("format") != _STATE_FORMAT or state.get("version") != 1:
            raise ValueError("Unsupported admission store format")
        yield root, state
    finally:
        os.close(descriptor)


def _commit(root: Path, state: dict[str, Any]) -> None:
    """Publish membership and applied attempts in one state replacement."""
    state["revision"] += 1
    _publish_json(root / "admission_state.json", state)


def _admission_population_size(config: Mapping[str, Any]) -> int:
    """Read the frozen field size, preserving version 1's twelve-entry admission.

    The shared descriptor validator checks participant uniqueness and the minimum
    field size. Version 2 uses that declared field throughout admission; a
    version-1 custom tournament does not gain historical admission rights.
    """
    size = len(config["participants"])
    if config["version"] == 1 and size != 12:
        raise ValueError("Version-1 admission requires exactly twelve entrants")
    return size


def _snapshot(root: Path, identifier: str) -> dict[str, Any]:
    """Read a pinned field without consulting today's installed release alias."""
    config = load_tournament_config(
        root / "snapshots" / f"{_digest(identifier, 'snapshot_id')}.json",
        official=False,
    )
    if config["snapshot_id"] != identifier:
        raise ValueError("Admission needs its exact pinned snapshot")
    _admission_population_size(config)
    return config


def _rules(root: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Read immutable approval evidence and check its content identity."""
    identifier = _digest(state["rules_approval_id"], "rules_approval_id")
    result = read_config_json(root / "rules" / f"{identifier}.json")
    if _identity(result) != identifier:
        raise ValueError("Admission approval evidence has changed")
    return result


def initialize_admission_store(
    store: str | Path,
    config: str | Path | Mapping[str, object],
    *,
    rules: Mapping[str, object],
) -> dict[str, Any]:
    """Create a maintainer store from a pinned field and approved rules.

    Parameters
    ----------
    store : path-like
        New or empty directory. This never targets the installed catalog.
    config : path-like or mapping
        Existing immutable field description. Version 1 requires twelve entrants;
        version 2 uses the frozen participant list. A real store requires an
        unchanged snapshot pinned by the installed release catalog. An
        unpublished fixture is allowed only when fixture_only is true.
        Initialization does not itself qualify a new official population.
    rules : mapping
        Immutable approval record with approval_id, fixture_only and every
        gate: games_per_opponent, resource_limits, timing_definitions,
        scientific_eligibility, reproduction, public_test_feedback,
        episode_local_adaptation, revised_method_policy and
        tied_incumbent_eviction. Use null for unresolved rules. Non-null
        values are maintainer-supplied decisions, never library defaults.
        The game budget must be a positive multiple of ten for version 1,
        or twice the configured map count for version 2.
        A supplied tied-incumbent rule uses an explicit entrant_order list;
        the library does not choose an ordering. Fixture-only stores stay
        separate from installed official releases.

    Returns
    -------
    dict
        A copied initial state. Later operations reread the durable state.

    Raises
    ------
    ValueError
        Fields, population or an existing store are incompatible. Missing rule
        approval blocks later games/publication rather than creating defaults.
    """
    approvals = copy.deepcopy(dict(rules))
    if set(approvals) != {"approval_id", "fixture_only", *_RULE_FIELDS}:
        raise ValueError(
            "Approval record must name every rule, using null when unresolved"
        )
    _text(approvals["approval_id"], "approval_id")
    if type(approvals["fixture_only"]) is not bool:
        raise ValueError("fixture_only must be a boolean")
    resolved = load_tournament_config(config, official=not approvals["fixture_only"])
    _admission_population_size(resolved)
    budget = approvals["games_per_opponent"]
    budget_unit = (
        10
        if resolved["version"] == 1
        else 2 * len(resolved["conditions"]["map_sources"])
    )
    if budget is not None and (
        type(budget) is not int or budget <= 0 or budget % budget_unit
    ):
        if resolved["version"] == 1:
            raise ValueError(
                "Approved official budget must be a positive multiple of ten"
            )
        raise ValueError(
            "Approved official budget must be a positive multiple of "
            f"{budget_unit}, twice the configured map count"
        )
    canonical_json(approvals)
    root = Path(store).expanduser().resolve()
    _make_directory(root)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "admission_state.json").exists():
            raise ValueError("Admission store is already initialized")
        rules_id = _identity(approvals)
        _immutable_json(root, "rules", rules_id, approvals)
        _immutable_json(root, "snapshots", resolved["snapshot_id"], resolved)
        state: dict[str, Any] = {
            "format": _STATE_FORMAT,
            "version": 1,
            "revision": 0,
            "published_snapshot_id": resolved["snapshot_id"],
            "provisional_snapshot_id": resolved["snapshot_id"],
            "active_edition": None,
            "rules_approval_id": rules_id,
            "submissions": {},
            "attempts": {},
            "applied_attempt_ids": [],
            "prepared_releases": {},
            "published_releases": {},
            "closed_editions": {},
        }
        _publish_json(root / "admission_state.json", state)
        return copy.deepcopy(state)
    finally:
        os.close(descriptor)


def release_window(release_at: str | datetime) -> dict[str, str]:
    """Return monthly London/UTC release and 72-elapsed-hour cutoff instants.

    ``release_at`` is an offset-bearing timestamp or aware datetime. It must
    denote midnight on the first day in Europe/London. Subtraction happens in
    UTC so a clock change cannot turn the cutoff into three local calendar days.
    This pure host helper changes no files.
    """
    release = _instant(release_at, "release_at")
    local = release.astimezone(_LONDON)
    if (local.day, local.hour, local.minute, local.second, local.microsecond) != (
        1,
        0,
        0,
        0,
        0,
    ):
        raise ValueError("Release must be 00:00 Europe/London on the first of a month")
    cutoff = release - timedelta(hours=72)
    return {
        "edition_id": local.strftime("%Y-%m"),
        "release_at_utc": release.isoformat(),
        "release_at_local": local.isoformat(),
        "cutoff_at_utc": cutoff.isoformat(),
        "cutoff_at_local": cutoff.astimezone(_LONDON).isoformat(),
    }


def _edition(state: dict[str, Any], release_at: str | datetime) -> dict[str, Any]:
    """Select one open monthly edition, without replacing a different edition."""
    window = release_window(release_at)
    current = cast(dict[str, Any] | None, state["active_edition"])
    if current is not None:
        if current["edition_id"] != window["edition_id"]:
            raise ValueError("Publish or explicitly defer the current edition first")
        if current["state"] != "open":
            raise ValueError("This edition is frozen; it cannot accept new attempts")
        return current
    if window["edition_id"] in state["closed_editions"]:
        raise ValueError("A closed edition cannot be reopened")
    created: dict[str, Any] = {
        **window,
        "base_snapshot_id": state["provisional_snapshot_id"],
        "provisional_snapshot_id": state["provisional_snapshot_id"],
        "applied_attempt_ids": [],
        "state": "open",
    }
    state["active_edition"] = created
    return created


def _read_asset(descriptor: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    """Read checked local submission evidence through the shared asset authority."""
    from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier

    if descriptor.get("role") != role:
        raise ValueError(f"Admission evidence must have role {role!r}")
    result = AssetVerifier({"assets": {"record": descriptor}}).read_json("record")
    if not isinstance(result, dict):
        raise ValueError("Admission evidence must contain a JSON object")
    return cast(dict[str, Any], result)


def _merge_assets(target: dict[str, Any], additional: Mapping[str, Any]) -> None:
    """Merge asset references, allowing relocated hints but no content collision."""
    for identifier, asset in additional.items():
        if identifier in target:
            if any(
                target[identifier].get(key) != asset.get(key)
                for key in ("sha256", "size_bytes", "role")
            ):
                raise ValueError(
                    "Challenger asset alias conflicts with the pinned snapshot"
                )
            continue
        target[identifier] = copy.deepcopy(asset)


def _qualified(
    root: Path, state: Mapping[str, Any], submission: Mapping[str, Any]
) -> bool:
    """Check qualification evidence and explicit rules without inventing eligibility."""
    if (
        submission["complete_at_utc"] is None
        or submission["qualification_asset"] is None
    ):
        return False
    evidence = _read_asset(submission["qualification_asset"], role="qualification")
    if (
        evidence.get("controller_id") != submission["controller_id"]
        or evidence.get("rules_approval_id") != state["rules_approval_id"]
        or type(evidence.get("approved")) is not bool
    ):
        raise ValueError(
            "Qualification evidence belongs to another controller or rules"
        )
    rules = _rules(root, state)
    if not evidence["approved"] or any(rules[field] is None for field in _GAME_GATES):
        return False
    return not (
        evidence.get("revises_controller_id") is not None
        and rules["revised_method_policy"] is None
    )


def register_submission(
    store: str | Path, submission: Mapping[str, Any]
) -> dict[str, Any]:
    """Register or complete one immutable controller submission in durable FIFO order.

    ``submission`` supplies submission_id, controller_id, descriptor_asset,
    received_at_utc, complete_at_utc and qualification_asset. Asset values are
    shared hash/size/path descriptors; null completion/qualification is allowed.
    A descriptor asset contains a participant descriptor and its asset mapping.
    Re-registration may fill missing completion or qualification evidence, but
    cannot rewrite identity, receive time or a completed timestamp. Once-only
    completion order survives retries. This verifies local evidence and writes
    only this explicit store; it executes no controller or admission game.
    """
    value = copy.deepcopy(dict(submission))
    expected = {
        "submission_id",
        "controller_id",
        "descriptor_asset",
        "received_at_utc",
        "complete_at_utc",
        "qualification_asset",
    }
    if set(value) != expected:
        raise ValueError(
            "Submission fields must match the admission submission contract"
        )
    _text(value["submission_id"], "submission_id")
    _digest(value["controller_id"], "controller_id")
    received = _instant(value["received_at_utc"], "received_at_utc")
    value["received_at_utc"] = received.isoformat()
    if value["complete_at_utc"] is not None:
        complete = _instant(value["complete_at_utc"], "complete_at_utc")
        if complete < received or complete > _utc_now():
            raise ValueError(
                "Submission completion must follow receipt and not be in the future"
            )
        value["complete_at_utc"] = complete.isoformat()
    descriptor = _read_asset(value["descriptor_asset"], role="registration")
    participant = descriptor.get("participant")
    if (
        not isinstance(participant, dict)
        or participant.get("controller_id") != value["controller_id"]
    ):
        raise ValueError("Submission descriptor does not identify this controller")
    with _locked_store(store) as (root, state):
        previous = state["submissions"].get(value["submission_id"])
        if previous is not None:
            for field in expected:
                if previous[field] is not None and previous[field] != value[field]:
                    raise ValueError(f"Submission cannot replace its recorded {field}")
            if all(previous[field] == value[field] for field in expected):
                return copy.deepcopy(previous)
            if previous["state"] in {"running", "promoted", "not_promoted"}:
                raise ValueError("An active or applied submission cannot be edited")
        value["completion_sequence"] = (
            None if previous is None else previous["completion_sequence"]
        )
        if (
            value["complete_at_utc"] is not None
            and value["completion_sequence"] is None
        ):
            value["completion_sequence"] = 1 + max(
                (
                    item["completion_sequence"] or 0
                    for item in state["submissions"].values()
                ),
                default=0,
            )
        value["state"] = (
            "incomplete"
            if value["complete_at_utc"] is None
            else "qualified"
            if _qualified(root, state, value)
            else "awaiting_approval"
        )
        state["submissions"][value["submission_id"]] = value
        _commit(root, state)
        return copy.deepcopy(value)


def _snapshot_inputs(
    config: Mapping[str, Any],
) -> tuple[AssetVerifier, dict[str, Path]]:
    """Verify declared host evidence assets without loading any model weights."""
    from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier

    verifier = AssetVerifier(config)
    paths = _verify_assets(
        verifier,
        tuple(
            identifier
            for identifier, asset in config["assets"].items()
            if asset["role"] not in {"model", "replay"}
        ),
    )
    return verifier, paths


def _verify_assets(
    verifier: AssetVerifier, identifiers: Sequence[str]
) -> dict[str, Path]:
    """Verify host files and inline metadata through the shared asset owner.

    Inline version-2 records have no filesystem path. Return paths only for
    file-backed assets; callers read either form through verifier.read_json.
    """
    paths = verifier.require(
        tuple(key for key in identifiers if "inline" not in verifier.assets[key])
    )
    for key in identifiers:
        if "inline" in verifier.assets[key]:
            verifier.read_json(key)
    return paths


def _population_evidence(
    config: Mapping[str, Any],
    *,
    _inputs: tuple[AssetVerifier, dict[str, Path]] | None = None,
) -> dict[str, Any]:
    """Check the frozen field's outcomes, full rows and actual physical pairs.

    This delegates schedule, stored-row and physical checks to their existing
    authorities. It runs no games or fit. The returned compact digest records
    the checked population; ordinary researcher execution never calls this
    stricter full-coverage admission boundary.
    """
    from marl_battlegrounds.evaluation.tournament_evidence import prepare_reuse_evidence
    from marl_battlegrounds.evaluation.tournament_records import TournamentRecords
    from marl_battlegrounds.evaluation.tournament_reuse import (
        analysis_schedule,
        resolve_reuse_plan,
    )

    verifier, paths = _snapshot_inputs(config) if _inputs is None else _inputs
    size = _admission_population_size(config)
    plan = resolve_reuse_plan(
        config, paths, asset_reader=verifier.read_json, require_reuse=True
    )
    expected_ids = {item["entrant_id"] for item in config["participants"]}
    if plan.jobs or set(plan.participant_ids) != expected_ids:
        raise ValueError("Admission needs the complete recorded participant field")
    records = TournamentRecords(
        config, plan.games, (), verifier, manifest={"run_id": "admission-check"}
    )
    records.require_coverage(metrics="full")
    matches = tuple(
        row for batch in records.iter_rows("match_results.csv") for row in batch
    )
    full_count = sum(
        len(batch)
        for batch in records.iter_rows(
            "full_metrics.csv", columns=("run_id", "phase", "pass_id", "episode_id")
        )
    )
    if full_count != len(plan.games):
        raise ValueError("Admission population is missing full report rows")
    participants = {item["entrant_id"]: item for item in config["participants"]}
    registrations: dict[str, Mapping[str, Any]] = {}
    for identifier, item in participants.items():
        registration = verifier.read_json(item["registration_asset"])
        if not isinstance(registration, Mapping):
            raise ValueError("Participant registration must be a JSON object")
        registrations[identifier] = cast(Mapping[str, Any], registration)
    evidence = prepare_reuse_evidence(
        records,
        analysis_schedule(
            plan,
            {identifier: item["name"] for identifier, item in participants.items()},
        ),
        matches,
        participants=participants,
        registrations=registrations,
    )
    return {
        "snapshot_id": config["snapshot_id"],
        "games": len(plan.games),
        "matchups": size * (size - 1) // 2,
        "full_rows": full_count,
        "physical_evidence_id": _identity(evidence),
        "asset_ids": {
            identifier: config["assets"][identifier]["sha256"]
            for identifier, asset in config["assets"].items()
            if asset["role"] not in {"model", "replay"}
        },
    }


def prepare_admission(
    store: str | Path, submission_id: str, *, release_at: str | datetime
) -> dict[str, Any]:
    """Pin the next qualified submission to an exact full-metric admission schedule.

    The caller names a registered submission and a monthly release instant.
    Complete eligible submissions run in their saved completion order. This
    verifies the provisional field's existing full reports and freezes the
    challenger schedule, but makes no action or model call. The executor must
    use the returned snapshot/schedule and metrics="full" from the first game.
    Repeating an active attempt returns the same schedule and random streams.
    Missing rules, late submissions and another active attempt raise ValueError
    before a new attempt is published.
    """
    from dataclasses import asdict

    from marl_battlegrounds.evaluation.tournament_reuse import resolve_reuse_plan

    with _locked_store(store) as (root, state):
        edition = _edition(state, release_at)
        if _utc_now() >= _instant(edition["release_at_utc"], "release_at"):
            raise ValueError(
                "Release boundary has passed; freeze or defer this edition first"
            )
        submission = state["submissions"].get(submission_id)
        if submission is None:
            raise ValueError("Submission is not registered")
        if submission["state"] in {"promoted", "not_promoted"}:
            raise ValueError("Submission already has an applied admission decision")
        if not _qualified(root, state, submission):
            raise ValueError(
                "Submission qualification or required rule approval is missing"
            )
        cutoff = _instant(edition["cutoff_at_utc"], "cutoff")
        if _instant(submission["complete_at_utc"], "completion") > cutoff:
            raise ValueError("Submission completed after this edition's 72-hour cutoff")
        eligible = sorted(
            (
                item
                for item in state["submissions"].values()
                if item["state"] not in {"promoted", "not_promoted"}
                and item["complete_at_utc"] is not None
                and _instant(item["complete_at_utc"], "completion") <= cutoff
                and _qualified(root, state, item)
            ),
            key=lambda item: item["completion_sequence"],
        )
        if not eligible or eligible[0]["submission_id"] != submission_id:
            raise ValueError(
                "An earlier qualified complete submission must be processed first"
            )
        for attempt in state["attempts"].values():
            if attempt["applied_at_utc"] is not None:
                continue
            if (
                attempt["edition_id"] == edition["edition_id"]
                and attempt["base_snapshot_id"] == state["provisional_snapshot_id"]
            ):
                if attempt["submission_id"] != submission_id:
                    raise ValueError(
                        "Another admission attempt owns this provisional revision"
                    )
                if attempt["rules_approval_id"] != state["rules_approval_id"]:
                    raise ValueError("Resume the attempt with its pinned rules")
                if submission["state"] != "running":
                    submission["state"] = "running"
                    _commit(root, state)
                return copy.deepcopy(attempt)
        config = _snapshot(root, state["provisional_snapshot_id"])
        rules = _rules(root, state)
        if config["conditions"]["games_per_opponent"] != rules["games_per_opponent"]:
            raise ValueError(
                "Admission budget must equal the explicitly approved protocol"
            )
        inputs = _snapshot_inputs(config)
        full_coverage = _population_evidence(config, _inputs=inputs)
        controller = _read_asset(submission["descriptor_asset"], role="registration")
        participant = controller["participant"]
        if any(
            item["controller_id"] == submission["controller_id"]
            for item in config["participants"]
        ):
            raise ValueError(
                "Exact incumbent duplicate; use the configured-field researcher route"
            )
        if any(
            item["entrant_id"] == participant["entrant_id"]
            or item["name"] == participant["name"]
            for item in config["participants"]
        ):
            raise ValueError("Challenger identity and display name must be distinct")
        candidate = copy.deepcopy(config)
        _merge_assets(candidate["assets"], controller["assets"])
        candidate["participants"].append(participant)
        candidate["release"] = None
        candidate["snapshot_id"] = snapshot_identity(candidate)
        load_tournament_config(candidate, official=False)
        if not rules["fixture_only"]:
            from marl_battlegrounds.evaluation.tournament_assets import (
                AssetVerifier,
                controller_content_identity,
            )

            verifier = AssetVerifier(candidate)
            identifier, known = controller_content_identity(
                participant["controller"], verifier
            )
            if not known or identifier != participant["controller_id"]:
                raise ValueError(
                    "Official admission needs a complete controller identity"
                )
        verifier, paths = inputs
        from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier

        participant_kinds: dict[str, str] | None = None
        if config["version"] == 2:
            manifest = verifier.read_json(config["conditions"]["schedule_asset"])
            if isinstance(manifest, Mapping) and "companion_stream_rule" in manifest:
                candidate_verifier = AssetVerifier(candidate)
                participant_kinds = {}
                for item in candidate["participants"]:
                    owner = (
                        candidate_verifier
                        if item["entrant_id"] == participant["entrant_id"]
                        else verifier
                    )
                    registration = owner.read_json(item["registration_asset"])
                    if not isinstance(registration, Mapping):
                        raise ValueError(
                            "Participant registration must be a JSON object"
                        )
                    participant_kinds[item["entrant_id"]] = _text(
                        cast(Mapping[str, object], registration).get("kind", "policy"),
                        "Participant kind",
                    )
        plan = resolve_reuse_plan(
            config,
            paths,
            asset_reader=verifier.read_json,
            challenger_id=participant["entrant_id"],
            participant_kinds=participant_kinds,
            require_reuse=True,
        )
        schedule = {
            "format": "marlbg-admission-schedule",
            "version": 1,
            "snapshot_id": config["snapshot_id"],
            "rules_approval_id": state["rules_approval_id"],
            "controller_id": submission["controller_id"],
            "metrics": "full",
            "plan": asdict(plan),
        }
        schedule_id = _identity(schedule)
        schedule_path = _immutable_json(root, "schedules", schedule_id, schedule)
        attempt_id = _identity(
            {
                "submission_id": submission_id,
                "edition_id": edition["edition_id"],
                "schedule_id": schedule_id,
            }
        )
        attempt = {
            "attempt_id": attempt_id,
            "submission_id": submission_id,
            "controller_id": submission["controller_id"],
            "base_snapshot_id": config["snapshot_id"],
            "rules_approval_id": state["rules_approval_id"],
            "edition_id": edition["edition_id"],
            "completed_at_utc": None,
            "schedule_asset": {
                "sha256": sha256(schedule_path.read_bytes()).hexdigest(),
                "size_bytes": schedule_path.stat().st_size,
                "role": "schedule",
                "path": str(schedule_path),
                "url": None,
            },
            "run_ref": None,
            "full_coverage": full_coverage,
            "joint_fit_id": None,
            "decision": "pending",
            "reason": None,
            "applied_at_utc": None,
            "resulting_snapshot_id": None,
            "committed_provisional_revision": None,
            "result_digest": None,
        }
        state["attempts"][attempt_id] = attempt
        submission["state"] = "running"
        _commit(root, state)
        return copy.deepcopy(attempt)


def record_admission_failure(store: str | Path, attempt_id: str, reason: str) -> None:
    """Record an incomplete attempt without inventing a game result or changing FIFO.

    This explicit maintainer failure note preserves the pinned schedule for a
    retry. Applied attempts cannot be changed. It changes only local store state.
    """
    _text(reason, "failure reason")
    with _locked_store(store) as (root, state):
        attempt = state["attempts"].get(attempt_id)
        if attempt is None or attempt["applied_at_utc"] is not None:
            raise ValueError("Failure must name an unapplied admission attempt")
        attempt["reason"] = reason
        state["submissions"][attempt["submission_id"]]["state"] = "failed"
        _commit(root, state)


def execute_admission(
    store: str | Path,
    attempt_id: str,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    num_envs: int = 128,
    chunk_size: int = 16,
) -> CanonicalTournamentResult:
    """Run one prepared admission through the shared full-capture executor.

    Parameters
    ----------
    store : path-like
        Explicit local maintainer store with an eligible prepared attempt.
    attempt_id : str
        Identity returned by prepare_admission. Its snapshot, rules and exact
        schedule stay pinned. A blocked or applied attempt cannot execute again.
    output_dir, resume_from : path-like or None
        Supply exactly one: a parent for a new durable run, or its existing run
        directory for a compatible resume. No-file admission is unsupported.
    num_envs : int, default 128
        Requested environment batch; the shared executor owns validation.
    chunk_size : int, default 16
        Numerical steps per evaluation chunk, checked by the executor.

    Returns
    -------
    CanonicalTournamentResult
        Full metrics captured from each game's first transition. Membership is
        unchanged until a separate apply_admission call verifies the result.

    Notes
    -----
    The existing tournament executor loads trusted controllers only for active
    games. Resuming a complete run loads no controller. This never downloads,
    retries a failed action or publishes membership. The store lock is released before
    games run; the selected run writer owns its own execution lock. Exceptions
    propagate and leave the prepared schedule available for an explicit retry.
    """
    from marl_battlegrounds.evaluation.canonical import _run_resolved_tournament
    from marl_battlegrounds.evaluation.tournament_reuse import ReusePlan

    if (output_dir is None) == (resume_from is None):
        raise ValueError(
            "Admission requires exactly one saved output or resume directory"
        )
    with _locked_store(store) as (root, state):
        attempt = state["attempts"].get(attempt_id)
        edition = state["active_edition"]
        if (
            attempt is None
            or attempt["decision"] != "pending"
            or attempt["applied_at_utc"] is not None
            or edition is None
            or edition["state"] != "open"
            or attempt["edition_id"] != edition["edition_id"]
            or attempt["base_snapshot_id"] != state["provisional_snapshot_id"]
            or attempt["rules_approval_id"] != state["rules_approval_id"]
            or _utc_now() >= _instant(edition["release_at_utc"], "release_at")
        ):
            raise ValueError(
                "Admission attempt is blocked, applied or no longer current"
            )
        submission = state["submissions"][attempt["submission_id"]]
        if not _qualified(root, state, submission):
            raise ValueError("Admission qualification no longer verifies")
        config = _snapshot(root, attempt["base_snapshot_id"])
        description = _read_asset(submission["descriptor_asset"], role="registration")
        frozen = _read_asset(attempt["schedule_asset"], role="schedule")
        raw = frozen["plan"]
        plan = ReusePlan(
            tuple(raw["games"]),
            tuple(raw["jobs"]),
            tuple(raw["source_descriptors"]),
            dict(raw["budget"]),
            dict(raw["schedule_manifest"]),
            tuple(raw["participant_ids"]),
        )
    return _run_resolved_tournament(
        config,
        challenger_descriptor=description["participant"],
        challenger_assets=description["assets"],
        plan=plan,
        require_reuse=True,
        metrics="full",
        output_dir=output_dir,
        resume_from=resume_from,
        num_envs=num_envs,
        chunk_size=chunk_size,
    )


def apply_admission(
    store: str | Path, attempt_id: str, result: object
) -> dict[str, Any]:
    """Apply one complete field-plus-challenger result after shared evidence checks.

    ``result`` must come from the full-capture canonical executor for this saved
    attempt. It is checked before any membership change. Exact cutoff ties do
    not promote. An unresolved tied-incumbent eviction stores a blocked decision.
    A promotion prepares and refits the retained field first, then commits its
    pointer and applied-attempt entry atomically. The release deadline is checked
    again after preparation, just before that commit. Repeating identical applied
    evidence is a no-op; different evidence raises ValueError.
    """
    with _locked_store(store) as (root, state):
        attempt = state["attempts"].get(attempt_id)
        if attempt is None:
            raise ValueError("Admission attempt is not registered")
        base = _snapshot(root, attempt["base_snapshot_id"])
        evidence = _admission_evidence(result, config=base, attempt=attempt)
        result_id = _identity(
            {key: value for key, value in evidence.items() if key != "completed_at_utc"}
        )
        if attempt["applied_at_utc"] is not None:
            if attempt["result_digest"] != result_id:
                raise ValueError("Applied admission has different result evidence")
            _sync_directory(root)
            return copy.deepcopy(attempt)
        edition = state["active_edition"]
        if (
            edition is None
            or edition["state"] != "open"
            or edition["edition_id"] != attempt["edition_id"]
            or state["provisional_snapshot_id"] != attempt["base_snapshot_id"]
            or state["rules_approval_id"] != attempt["rules_approval_id"]
        ):
            raise ValueError(
                "Admission result has a stale edition, population or rules"
            )
        now = _utc_now()
        if now >= _instant(edition["release_at_utc"], "release_at"):
            raise ValueError(
                "Late admission cannot change this release; roll it forward"
            )
        ratings = evidence["ratings"]
        incumbents = [item["entrant_id"] for item in base["participants"]]
        lowest = min(ratings[identifier] for identifier in incumbents)
        cutoff = [
            identifier for identifier in incumbents if ratings[identifier] == lowest
        ]
        challenger = evidence["challenger_id"]
        promoted = ratings[challenger] > lowest
        rules = _rules(root, state)
        removed: str | None = None
        if promoted:
            if len(cutoff) == 1:
                removed = cutoff[0]
            elif rules["tied_incumbent_eviction"] is None:
                attempt.update(
                    decision="blocked",
                    reason="Tied incumbent eviction needs explicit approval",
                    result_digest=result_id,
                )
                _commit(root, state)
                return copy.deepcopy(attempt)
            else:
                selection = cast(dict[str, Any], rules["tied_incumbent_eviction"])
                if (
                    not isinstance(cast(object, selection), dict)
                    or set(selection) != {"entrant_order"}
                    or set(selection["entrant_order"]) != set(incumbents)
                ):
                    raise ValueError(
                        "Approved tie rule must explicitly order "
                        "the current incumbent IDs"
                    )
                removed = next(
                    identifier
                    for identifier in selection["entrant_order"]
                    if identifier in cutoff
                )
        next_id = base["snapshot_id"]
        if promoted:
            retained = (
                *(identifier for identifier in incumbents if identifier != removed),
                challenger,
            )
            replacement = _retained_admission_snapshot(
                root, base, result, retained_ids=retained
            )
            next_id = replacement["snapshot_id"]
            _immutable_json(root, "snapshots", next_id, replacement)
        _immutable_json(root, "results", result_id, evidence)
        # Immutable result/snapshot assets precede this one authoritative commit.
        now = _utc_now()
        if now >= _instant(edition["release_at_utc"], "release_at"):
            raise ValueError(
                "Late admission cannot change this release; roll it forward"
            )
        attempt.update(
            completed_at_utc=evidence.get("completed_at_utc", now.isoformat()),
            run_ref=evidence["run_ref"],
            full_coverage=evidence["full_coverage"],
            joint_fit_id=evidence["joint_fit_id"],
            decision="promoted" if promoted else "not_promoted",
            reason=None,
            applied_at_utc=now.isoformat(),
            resulting_snapshot_id=next_id,
            committed_provisional_revision=state["revision"] + 1,
            result_digest=result_id,
        )
        state["provisional_snapshot_id"] = next_id
        state["applied_attempt_ids"].append(attempt_id)
        edition["provisional_snapshot_id"] = next_id
        edition["applied_attempt_ids"].append(attempt_id)
        state["submissions"][attempt["submission_id"]]["state"] = attempt["decision"]
        _commit(root, state)
        return copy.deepcopy(attempt)


def _freeze_edition(
    state: dict[str, Any], release_at: str | datetime
) -> dict[str, Any]:
    """Select the last eligible committed population at the monthly boundary."""
    window = release_window(release_at)
    current = state["active_edition"]
    if current is None:
        current = _edition(state, release_at)
    if current["edition_id"] != window["edition_id"]:
        raise ValueError("Release does not name the active edition")
    boundary = _instant(window["release_at_utc"], "release_at")
    if _utc_now() < boundary:
        raise ValueError("The monthly release boundary has not arrived")
    if current["state"] == "published":
        raise ValueError("Published editions cannot be prepared again")
    selected = current["base_snapshot_id"]
    applied: list[str] = []
    revision = 0
    for identifier in current["applied_attempt_ids"]:
        attempt = state["attempts"][identifier]
        if (
            attempt["applied_at_utc"] is not None
            and _instant(attempt["applied_at_utc"], "applied_at") < boundary
        ):
            selected = attempt["resulting_snapshot_id"]
            revision = attempt["committed_provisional_revision"]
            applied.append(identifier)
    current.update(
        state="frozen", provisional_snapshot_id=selected, applied_attempt_ids=applied
    )
    # This is a derived local return value; the edition layout stays unchanged.
    return {**current, "provisional_revision": revision}


def _require_release_rules(rules: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Reject unresolved release gates without deciding any scientific values."""
    missing = [field for field in _GAME_GATES if rules[field] is None]
    if missing:
        raise ValueError("Release needs explicit approval for: " + ", ".join(missing))
    if rules["games_per_opponent"] != config["conditions"]["games_per_opponent"]:
        raise ValueError("Release budget differs from its approved protocol")


def _check_release_population(
    config: dict[str, Any], rules: Mapping[str, Any], verifier: AssetVerifier
) -> None:
    """Keep isolated fixture publication distinct from real official qualification."""
    if rules["fixture_only"]:
        return
    from marl_battlegrounds.evaluation.tournament_assets import (
        controller_content_identity,
    )
    from marl_battlegrounds.evaluation.tournament_config import _validate
    from marl_battlegrounds.tasks import canonical_tournament_rosters

    _validate(config, official=True)
    rosters = config["conditions"]["rosters"]
    if (
        config["version"] == 1
        and (tuple(rosters["team_a"]), tuple(rosters["team_b"]))
        != canonical_tournament_rosters()
    ):
        raise ValueError("Official release needs the canonical ordered rosters")
    for participant in config["participants"]:
        identifier, known = controller_content_identity(
            participant["controller"], verifier
        )
        if not known or identifier != participant["controller_id"]:
            raise ValueError(
                "Official release needs complete verified controller identities"
            )


def prepare_release(store: str | Path, release_at: str | datetime) -> dict[str, Any]:
    """Freeze and verify one monthly field snapshot without publishing it.

    The release instant must have arrived. Complete outcomes, full reports,
    physical pairs, referenced assets and explicit rule approvals are checked
    before the frozen selection is committed. This creates immutable local
    records only; it does not move a catalog default. Retries return the same
    prepared content. Missing approval or bytes raise ValueError.
    """
    with _locked_store(store) as (root, state):
        frozen = _freeze_edition(state, release_at)
        config = _snapshot(root, frozen["provisional_snapshot_id"])
        rules = _rules(root, state)
        _require_release_rules(rules, config)
        inputs = _snapshot_inputs(config)
        evidence = _population_evidence(config, _inputs=inputs)
        _verify_assets(inputs[0], tuple(config["assets"]))
        released = copy.deepcopy(config)
        released["release"] = {
            key: frozen[key]
            for key in (
                "release_at_utc",
                "release_at_local",
                "cutoff_at_utc",
                "cutoff_at_local",
            )
        }
        released["release"].update(
            rules_id=state["rules_approval_id"], approval_id=rules["approval_id"]
        )
        released["snapshot_id"] = snapshot_identity(released)
        released = load_tournament_config(released, official=False)
        _check_release_population(released, rules, inputs[0])
        _immutable_json(root, "snapshots", released["snapshot_id"], released)
        prepared = {
            "edition_id": frozen["edition_id"],
            "provisional_revision": frozen["provisional_revision"],
            "base_snapshot_id": frozen["provisional_snapshot_id"],
            "snapshot_id": released["snapshot_id"],
            "rules_approval_id": state["rules_approval_id"],
            **{
                key: frozen[key]
                for key in (
                    "release_at_utc",
                    "release_at_local",
                    "cutoff_at_utc",
                    "cutoff_at_local",
                )
            },
            "asset_evidence": evidence,
            "fixture_only": rules["fixture_only"],
        }
        prepared["release_id"] = _identity(prepared)
        existing = state["prepared_releases"].get(prepared["release_id"])
        if existing is not None:
            if existing != prepared:
                raise ValueError("Prepared release has conflicting content")
            return copy.deepcopy(existing)
        _immutable_json(root, "releases", prepared["release_id"], prepared)
        state["prepared_releases"][prepared["release_id"]] = prepared
        _commit(root, state)
        return copy.deepcopy(prepared)


def _next_edition(
    state: dict[str, Any], current: Mapping[str, Any], *, base: str
) -> None:
    """Open the next calendar edition while retaining unfinished FIFO positions."""
    local = _instant(current["release_at_utc"], "release_at").astimezone(_LONDON)
    next_local = datetime(
        local.year + (local.month == 12), local.month % 12 + 1, 1, tzinfo=_LONDON
    )
    state["active_edition"] = None
    state["provisional_snapshot_id"] = base
    _edition(state, next_local)
    for item in state["submissions"].values():
        if item["state"] in {"running", "failed"}:
            item["state"] = "deferred"


def publish_release(
    store: str | Path, prepared_release: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish an already prepared snapshot to this explicit local store's catalog.

    This separate maintainer operation rechecks approvals and assets, then moves
    the catalog entry/default atomically before recording completion in state.
    A retry after interruption reconciles an identical published catalog entry.
    Conflicting prepared content fails. It never edits the installed package,
    uploads files or makes fixture evidence scientifically qualified.
    """
    prepared = copy.deepcopy(dict(prepared_release))
    release_id = prepared.get("release_id")
    with _locked_store(store) as (root, state):
        if state["prepared_releases"].get(release_id) != prepared:
            raise ValueError(
                "Release does not match this store's immutable preparation"
            )
        completed = state["published_releases"].get(release_id)
        if completed is not None:
            return copy.deepcopy(completed)
        edition = state["active_edition"]
        if (
            edition is None
            or edition["edition_id"] != prepared["edition_id"]
            or edition["state"] != "frozen"
        ):
            raise ValueError("Release edition is not the currently frozen edition")
        if _utc_now() < _instant(prepared["release_at_utc"], "release_at"):
            raise ValueError("Publication cannot precede the monthly release instant")
        config = _snapshot(root, prepared["snapshot_id"])
        rules = _rules(root, state)
        if prepared["rules_approval_id"] != state["rules_approval_id"]:
            raise ValueError("Prepared release belongs to different approved rules")
        _require_release_rules(rules, config)
        base = _snapshot(root, prepared["base_snapshot_id"])
        inputs = _snapshot_inputs(base)
        if _population_evidence(base, _inputs=inputs) != prepared["asset_evidence"]:
            raise ValueError("Prepared release evidence has changed")
        _verify_assets(inputs[0], tuple(config["assets"]))
        _check_release_population(config, rules, inputs[0])
        catalog_path = root / "catalog.json"
        catalog: dict[str, Any] = (
            read_config_json(catalog_path)
            if catalog_path.exists()
            else {"default": None, "snapshots": {}}
        )
        catalog_entry = {
            "config": str(root / "snapshots" / f"{config['snapshot_id']}.json"),
            "release": config["release"],
        }
        existing = catalog["snapshots"].get(config["snapshot_id"])
        if existing is not None and existing != catalog_entry:
            raise ValueError("Local catalog already has conflicting snapshot content")
        if existing != catalog_entry or catalog["default"] != config["snapshot_id"]:
            catalog["snapshots"][config["snapshot_id"]] = catalog_entry
            catalog["default"] = config["snapshot_id"]
            _publish_json(catalog_path, catalog)
        # Catalog durability precedes state reconciliation. A retry recognizes it.
        edition["state"] = "published"
        state["closed_editions"][edition["edition_id"]] = copy.deepcopy(edition)
        state["published_snapshot_id"] = config["snapshot_id"]
        state["published_releases"][release_id] = prepared
        _next_edition(state, edition, base=config["snapshot_id"])
        _commit(root, state)
        return copy.deepcopy(prepared)


def defer_release(store: str | Path, edition_id: str, reason: str) -> dict[str, Any]:
    """Close an unpublished edition and carry its completed field into next month.

    The release instant must have arrived. The reason is retained in a separate
    immutable closure record. This explicit action does not move the catalog
    default or change submission FIFO order. Repeating the same reason is a
    no-op; different closure content raises ValueError.
    """
    _text(reason, "deferral reason")
    with _locked_store(store) as (root, state):
        previous = state["closed_editions"].get(edition_id)
        if previous is not None:
            if previous.get("defer_reason") != reason:
                raise ValueError("Edition already closed with a different decision")
            return copy.deepcopy(previous)
        edition = state["active_edition"]
        if edition is None or edition["edition_id"] != edition_id:
            raise ValueError("Deferral must name the active edition")
        frozen = _freeze_edition(state, edition["release_at_utc"])
        closed = {
            key: value for key, value in frozen.items() if key != "provisional_revision"
        }
        closed["defer_reason"] = reason
        state["closed_editions"][edition_id] = closed
        _next_edition(state, closed, base=frozen["provisional_snapshot_id"])
        _commit(root, state)
        return copy.deepcopy(closed)


def _result_context(
    result: object,
) -> tuple[CanonicalTournamentResult, dict[str, Any], TournamentRecords, ReusePlan]:
    """Read a complete saved canonical result and its original shared accessor.

    Admission requires durable full reports. A no-file researcher result remains
    useful for analysis but cannot be applied to the maintainer population.
    """
    from marl_battlegrounds.evaluation.results import CanonicalTournamentResult
    from marl_battlegrounds.evaluation.tournament_reuse import ReusePlan

    if not isinstance(result, CanonicalTournamentResult):
        raise ValueError("Admission requires a canonical tournament result")
    if (
        result.run_dir is None
        or result.status != "complete"
        or result._record_access is None
    ):
        raise ValueError("Admission requires a complete saved canonical result")
    metadata = result.metadata
    if metadata.get("metrics") != "full":
        raise ValueError("Admission requires full capture")
    raw = metadata.get("canonical_plan")
    if not isinstance(raw, dict):
        raise ValueError("Admission result lacks its exact canonical plan")
    raw = cast(dict[str, Any], raw)
    plan = ReusePlan(
        tuple(raw["games"]),
        tuple(raw["jobs"]),
        tuple(raw["source_descriptors"]),
        dict(raw["budget"]),
        dict(raw["schedule_manifest"]),
        tuple(raw["participant_ids"]),
    )
    return result, metadata, result._record_access, plan


def _admission_evidence(
    result: object, *, config: Mapping[str, Any], attempt: Mapping[str, Any]
) -> dict[str, Any]:
    """Check a saved field-plus-challenger fit against its attempt and raw records.

    This reuses physical and report authorities; it does not fit a second rating
    model. The existing summary transaction binds the stored unrounded ratings
    to the complete game evidence. Only narrow summaries are retained here.
    """
    from marl_battlegrounds.evaluation.run_writer import _json_bytes, _json_value
    from marl_battlegrounds.evaluation.tournament_evidence import prepare_reuse_evidence
    from marl_battlegrounds.evaluation.tournament_reuse import analysis_schedule

    tournament, metadata, records, plan = _result_context(result)
    if (
        tournament.snapshot_id != config["snapshot_id"]
        or tournament.challenger_id is None
    ):
        raise ValueError("Admission result used another pinned population")
    if tournament.games_per_opponent != config["conditions"]["games_per_opponent"]:
        raise ValueError("A research budget override cannot qualify promotion")
    # A provisional population is not yet a released official snapshot. Its
    # public compliance flag can be false while this pinned admission is valid.
    # The exact approved budget, schedule and store rules are checked directly.
    schedule = _read_asset(attempt["schedule_asset"], role="schedule")
    expected_plan = schedule["plan"]
    if canonical_json(metadata["canonical_plan"]) != canonical_json(expected_plan):
        raise ValueError(
            "Admission result changed its frozen schedule or random streams"
        )
    participants = {
        item["entrant_id"]: item for item in metadata["participant_descriptors"]
    }
    challenger = tournament.challenger_id
    expected_ids = {item["entrant_id"] for item in config["participants"]} | {
        challenger
    }
    if (
        len(participants) != _admission_population_size(config) + 1
        or set(participants) != expected_ids
        or participants[challenger]["controller_id"] != attempt["controller_id"]
    ):
        raise ValueError("Admission result used another challenger version")
    records.require_coverage(metrics="full")
    if any(
        not records.completed(game) or records.coverage(game) != "full"
        for game in plan.games
    ):
        raise ValueError("Admission must capture full measurements for every game")
    full_rows = sum(
        len(batch)
        for batch in records.iter_rows(
            "full_metrics.csv", columns=("run_id", "phase", "pass_id", "episode_id")
        )
    )
    if full_rows != len(plan.games):
        raise ValueError("Admission is missing full per-game reports")
    matches = tuple(
        row for batch in records.iter_rows("match_results.csv") for row in batch
    )
    evidence = prepare_reuse_evidence(
        records,
        analysis_schedule(
            plan,
            {identifier: item["name"] for identifier, item in participants.items()},
        ),
        matches,
        participants=participants,
        registrations=metadata["participant_registrations"],
    )
    if canonical_json(evidence) != canonical_json(metadata["physical_evidence"]):
        raise ValueError(
            "Admission physical evidence differs from its completed result"
        )
    manifest = read_config_json(cast(Path, tournament.run_dir) / "run_details.json")
    summary = manifest.get("tournament_summary", {})
    payload: dict[str, Any] = {
        "tournament_results": tournament.tournament_results,
        "matchup_results": tournament.matchup_results,
        "map_results": tournament.map_results,
        "metadata": metadata["statistics"],
        "qualification": {
            key: value
            for key, value in summary.get("qualification", {}).items()
            if key != "summary_digest"
        },
        "tournament_headline_metrics": tournament.headline_metrics,
    }
    fit_id = sha256(_json_bytes(_json_value(payload))).hexdigest()
    if summary.get("digest") != fit_id:
        raise ValueError("Admission ratings differ from the durable complete fit")
    names = {item["name"]: identifier for identifier, item in participants.items()}
    ratings = {
        names[row["policy"]]: row["elo"] for row in tournament.tournament_results
    }
    if set(ratings) != set(participants):
        raise ValueError("Admission ratings do not cover the field and challenger")
    return {
        "snapshot_id": config["snapshot_id"],
        "challenger_id": challenger,
        "controller_id": attempt["controller_id"],
        "ratings": ratings,
        "run_ref": {"run_id": manifest["run_id"], "run_dir": str(tournament.run_dir)},
        "full_coverage": {
            "games": len(plan.games),
            "full_rows": full_rows,
            "physical_evidence_id": _identity(evidence),
        },
        "joint_fit_id": fit_id,
    }


def _asset_bytes(root: Path, data: bytes, role: str) -> tuple[str, dict[str, Any]]:
    """Durably store small immutable metadata bytes under their content digest."""
    identifier = sha256(data).hexdigest()
    directory = root / "assets"
    _make_directory(directory)
    path = directory / identifier
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Immutable admission asset has changed")
        _sync_directory(directory)
    else:
        temporary = directory / f".{identifier}.tmp"
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(directory)
    return identifier, {
        "sha256": identifier,
        "size_bytes": len(data),
        "role": role,
        "path": str(path),
        "url": None,
    }


def _copy_prefix_asset(
    root: Path, path: Path, length: int, role: str
) -> tuple[str, dict[str, Any]]:
    """Copy a committed table prefix using a bounded one-MiB buffer and one hash."""
    directory = root / "assets"
    _make_directory(directory)
    temporary = directory / ".copy-prefix.tmp"
    digest = sha256()
    remaining = length
    with path.open("rb") as source, temporary.open("wb") as target:
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(
                    "Admission source is shorter than its committed prefix"
                )
            target.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        target.flush()
        os.fsync(target.fileno())
    identifier = digest.hexdigest()
    destination = directory / identifier
    if destination.exists():
        # Check existing content through the shared asset authority before reuse.
        from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier

        AssetVerifier(
            {
                "assets": {
                    identifier: {
                        "sha256": identifier,
                        "size_bytes": length,
                        "role": role,
                        "path": str(destination),
                    }
                }
            }
        ).require((identifier,))
        temporary.unlink()
        _sync_directory(directory)
    else:
        os.replace(temporary, destination)
        _sync_directory(directory)
    return identifier, {
        "sha256": identifier,
        "size_bytes": length,
        "role": role,
        "path": str(destination),
        "url": None,
    }


def _source_from_run(
    root: Path, run_dir: Path, assets: dict[str, Any]
) -> dict[str, Any]:
    """Freeze report prefixes and keep each row and selected replay's original identity.

    CSV prefixes are copied with bounded buffers. Finalized replay files stay
    referenced in their original run, so those selected files must remain
    available. Their checked size/hash never grants availability to an unrecorded
    replay. This helper creates no replay and changes no original run bytes.
    """
    from marl_battlegrounds.evaluation.tournament_assets import _hash_file
    from marl_battlegrounds.evaluation.tournament_reuse import record_source_identity

    manifest = read_config_json(run_dir / "run_details.json")
    manifest_id, asset = _asset_bytes(root, canonical_json(manifest), "run_manifest")
    assets[manifest_id] = asset
    source: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "manifest_asset": manifest_id,
        "tables": {},
        "replays": [],
    }
    for filename in (
        "match_results.csv",
        "full_metrics.csv",
        "tournament_results.csv",
        "matchup_results.csv",
        "map_results.csv",
    ):
        boundary = manifest.get("tables", {}).get(filename)
        if boundary is None or not boundary.get("durable_bytes"):
            continue
        path = run_dir / filename
        role = "full_report" if filename == "full_metrics.csv" else "outcomes_priority"
        identifier, asset = _copy_prefix_asset(
            root, path, boundary["durable_bytes"], role
        )
        assets[identifier] = asset
        with path.open("rb") as stream:
            header = stream.readline(boundary["durable_bytes"])
        source["tables"][filename.removesuffix(".csv")] = {
            "asset_id": identifier,
            "committed_bytes": boundary["durable_bytes"],
            "rows": boundary["rows"],
            "header_sha256": sha256(header).hexdigest(),
        }
    for entry in manifest.get("passes", {}).values():
        for episode_id, replay in entry.get("replays", {}).items():
            relative = Path(replay["path"])
            if len(relative.parts) != 2 or relative.parts[0] != "replays":
                raise ValueError("Admission replay path leaves its original run")
            path = run_dir / relative
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("Admission replay must be a regular original file")
            identifier, size = _hash_file(path)
            if size != replay["bytes"]:
                raise ValueError("Admission replay differs from its durable byte count")
            assets[identifier] = {
                "sha256": identifier,
                "size_bytes": size,
                "role": "replay",
                "path": str(path),
                "url": None,
            }
            source["replays"].append(
                {
                    "run_id": manifest["run_id"],
                    "phase": entry["phase"],
                    "pass_id": entry["pass_id"],
                    "episode_id": int(episode_id),
                    "asset_id": identifier,
                }
            )
    source["source_id"] = record_source_identity(source, assets)
    return source


def _retained_admission_snapshot(
    root: Path,
    config: Mapping[str, Any],
    result: object,
    *,
    retained_ids: Sequence[str],
) -> dict[str, Any]:
    """Retain the selected field's original games, refit, and prepare its companion.

    Raw games and full reports remain immutable source references. New local
    rows are frozen through bounded prefix copies; the existing writer publishes
    the retained-fit tables. A completed preparation is cached by source result
    and retained IDs so membership-commit retries do not refit or copy again.
    """
    from dataclasses import replace

    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.evaluation.tournament_records import origin_key
    from marl_battlegrounds.evaluation.tournament_reuse import (
        analysis_schedule,
        rebuild_companion,
    )
    from marl_battlegrounds.evaluation.tournament_statistics import summarize_tournament

    tournament, metadata, records, plan = _result_context(result)
    retention_id = _identity(
        {
            "run_id": records.manifest["run_id"],
            "summary": records.manifest["tournament_summary"]["digest"],
            "retained": list(retained_ids),
        }
    )
    cached = root / "retentions" / f"{retention_id}.json"
    if cached.exists():
        snapshot = _snapshot(root, read_config_json(cached)["snapshot_id"])
        _population_evidence(snapshot)
        return snapshot
    size = _admission_population_size(config)
    chosen = set(retained_ids)
    if (
        len(chosen) != size
        or len(retained_ids) != size
        or not chosen <= set(plan.participant_ids)
    ):
        raise ValueError(
            f"Promotion must retain exactly {size} distinct current participants"
        )
    selected = [
        copy.deepcopy(game)
        for game in plan.games
        if game["team_a"] in chosen and game["team_b"] in chosen
    ]
    matchups = size * (size - 1) // 2
    if len(selected) != matchups * tournament.games_per_opponent:
        raise ValueError(
            f"Promotion is missing games from its {matchups} retained matchups"
        )
    next_config = copy.deepcopy(metadata["canonical_config"])
    assets = next_config["assets"]
    _merge_assets(assets, metadata.get("participant_assets", {}))
    # Resume may have found identical assets at new prepared locations. Keep
    # those useful hints in the next snapshot without changing content identity.
    for identifier, verified in records.verifier.assets.items():
        if identifier in assets:
            for hint in ("path", "url"):
                assets[identifier][hint] = verified.get(hint)
    source = _source_from_run(root, cast(Path, tournament.run_dir), assets)
    if not any(
        item["source_id"] == source["source_id"]
        for item in next_config["record_sources"]
    ):
        next_config["record_sources"].append(source)
    for game in selected:
        if game["origin"] is None:
            game["origin"] = {"source_id": source["source_id"], **records.origin(game)}
    retained_plan = replace(
        plan, games=tuple(selected), participant_ids=tuple(retained_ids)
    )
    matches = {
        origin_key(row): row
        for batch in records.iter_rows("match_results.csv")
        for row in batch
    }
    outcomes = {
        game["logical_game_id"]: matches[origin_key(records.origin(game))]["outcome"]
        for game in selected
    }
    participants = {
        item["entrant_id"]: copy.deepcopy(item)
        for item in metadata["participant_descriptors"]
        if item["entrant_id"] in chosen
    }
    names = {identifier: item["name"] for identifier, item in participants.items()}
    weights = config["analysis"]["opponent_weights"]
    if weights is not None:
        weights = {
            names[identifier]: weights[identifier] for identifier in retained_ids
        }
    statistics = summarize_tournament(
        analysis_schedule(retained_plan, names),
        outcomes,
        seed=config["analysis"]["bootstrap_seed"],
        opponent_weights=weights,
        method_sampling=(
            {
                name: metadata["statistics"]["method_sampling"][name]
                for name in names.values()
            }
            if metadata.get("statistics", {}).get("method_sampling") is not None
            else None
        ),
    )
    with RunWriter(
        root / "refits",
        phase="tournament",
        pass_id=retention_id,
        details={"admission_retention": retention_id},
    ) as writer:
        writer.write_tournament_results(statistics)
        fit_dir = writer.run_dir
    fit_source = _source_from_run(root, fit_dir, assets)
    next_config["record_sources"].append(fit_source)
    ratings = {row["policy"]: row["elo"] for row in statistics.tournament_results}
    for item in participants.values():
        item["elo"] = ratings[item["name"]]
        item["result_ref"] = {
            "source_id": fit_source["source_id"],
            "table": "tournament_results",
            "policy": item["name"],
        }
    next_config["participants"] = sorted(
        participants.values(), key=lambda item: -item["elo"]
    )
    games_alias, companion_alias = (
        f"retained-{retention_id}",
        f"companion-{retention_id}",
    )
    manifest, games, companion = rebuild_companion(
        plan,
        selected,
        retained_ids,
        games_asset=games_alias,
        challenger_games_asset=companion_alias,
        map_ids=tuple(item["map_id"] for item in config["conditions"]["map_sources"]),
    )
    for alias, rows in ((games_alias, games), (companion_alias, companion)):
        _, asset = _asset_bytes(
            root, b"".join(canonical_json(row) + b"\n" for row in rows), "schedule"
        )
        assets[alias] = asset
    identifier, asset = _asset_bytes(root, canonical_json(manifest), "schedule")
    assets[identifier] = asset
    next_config["conditions"]["schedule_asset"] = identifier
    next_config["release"] = None
    _prune_snapshot_assets(next_config, games)
    next_config["snapshot_id"] = snapshot_identity(next_config)
    next_config = load_tournament_config(next_config, official=False)
    _population_evidence(next_config)
    _immutable_json(root, "snapshots", next_config["snapshot_id"], next_config)
    _immutable_json(
        root, "retentions", retention_id, {"snapshot_id": next_config["snapshot_id"]}
    )
    return next_config


def _prune_snapshot_assets(
    config: dict[str, Any], games: Sequence[Mapping[str, Any]]
) -> None:
    """Keep active snapshot references bounded by its field and retained origins.

    The caller owns this newly prepared mapping. Historical files and snapshots
    are never changed. Original run manifests are complete immutable evidence,
    not a request to import every historical config embedded in their metadata;
    their required external tables/replays are explicit record-source references.
    Controller and dependency records still retain their declared asset closure.
    """
    from marl_battlegrounds.evaluation.tournament_assets import (
        AssetVerifier,
        _is_game_list,
        _references,
    )

    used_sources = {game["origin"]["source_id"] for game in games}
    used_sources.update(
        item["result_ref"]["source_id"]
        for item in config["participants"]
        if item["result_ref"] is not None
    )
    config["record_sources"] = [
        source
        for source in config["record_sources"]
        if source["source_id"] in used_sources
    ]
    if {source["source_id"] for source in config["record_sources"]} != used_sources:
        raise ValueError("Retained population lost one of its original record sources")
    verifier = AssetVerifier(config)
    pending = set(_references(config))
    used: set[str] = set()
    while pending:
        identifier = pending.pop()
        if identifier in used:
            continue
        if identifier not in config["assets"]:
            raise ValueError("Retained snapshot references an undeclared asset")
        used.add(identifier)
        role = config["assets"][identifier]["role"]
        if role in {
            "model",
            "outcomes_priority",
            "full_report",
            "replay",
            "run_manifest",
        }:
            continue
        if "inline" not in config["assets"][identifier]:
            path = verifier.require((identifier,))[identifier]
            if role == "schedule" and _is_game_list(path):
                continue
        pending.update(_references(verifier.read_json(identifier)))
    config["assets"] = {
        identifier: asset
        for identifier, asset in config["assets"].items()
        if identifier in used
    }
