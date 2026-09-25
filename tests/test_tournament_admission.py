"""Check private admission ordering, explicit gates and monthly durable decisions.

Store-focused cases replace game evidence preparation with controlled records;
separate integration cases exercise the real schedule, report and fit authorities.
No test treats a fixture store as a qualified official release.

A release saved before the Red Zone rule (pins 14, 2, 3 and 12-key depth-0.0
configurations) is admitted under its original snapshot identity, with its
real population evidence. Edited source configuration content, even with its
asset hash and snapshot identity recomputed, is rejected. A Red Zone rule
mismatch is rejected before any write: the public config route refuses a
supplied red_zone_depth, and a challenger's admission against the old release,
which needs new games under the current rule, fails with the reuse-only
message before any run directory or store change.
"""

# The store tests intentionally exercise private failure boundaries.
# pyright: reportPrivateUsage=false

import copy
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from tests.canonical_fixtures import config_descriptor
from tests.canonical_record_fixtures import build_record_bundle

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.evaluation import admission
from marl_battlegrounds.evaluation.results import CanonicalTournamentResult
from marl_battlegrounds.evaluation.tournament_assets import AssetVerifier
from marl_battlegrounds.evaluation.tournament_config import (
    canonical_json,
    read_config_json,
    snapshot_identity,
)
from marl_battlegrounds.evaluation.tournament_reuse import ReusePlan


def _asset(path: Path, value: object, role: str) -> dict[str, Any]:
    data = canonical_json(value)
    path.write_bytes(data)
    return {
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
        "role": role,
        "path": str(path),
        "url": None,
    }


def _rules(**changes: object) -> dict[str, Any]:
    result: dict[str, Any] = {
        "approval_id": "fixture-only-approval",
        "fixture_only": True,
        **{field: {"fixture_approved": True} for field in admission._RULE_FIELDS},
        "games_per_opponent": 10,
        "tied_incumbent_eviction": None,
        "revised_method_policy": None,
    }
    result.update(changes)
    return result


def _population(value: dict[str, Any], **_: object) -> dict[str, Any]:
    return {"snapshot_id": value["snapshot_id"], "matchups": 66, "full_rows": 660}


def _inputs(value: dict[str, Any]) -> tuple[AssetVerifier, dict[str, Path]]:
    return AssetVerifier(value), {}


def _identity_evidence(result: dict[str, Any], **_: object) -> dict[str, Any]:
    return result


def _empty_paths(*_: object) -> dict[str, Path]:
    return {}


def _fail_commit(*_: object) -> None:
    raise OSError("State failure")


def _store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **rules: object
) -> tuple[Path, dict[str, Any]]:
    clock = datetime(2026, 10, 29, tzinfo=UTC)
    monkeypatch.setattr(admission, "_utc_now", lambda: clock)
    config = config_descriptor(entrants=12, root=tmp_path)
    store = tmp_path / "admission"
    state = admission.initialize_admission_store(store, config, rules=_rules(**rules))
    monkeypatch.setattr(admission, "_population_evidence", _population)
    monkeypatch.setattr(admission, "_snapshot_inputs", _inputs)

    def plan(config: dict[str, Any], _: object, **kwargs: object) -> ReusePlan:
        names = tuple(item["entrant_id"] for item in config["participants"])
        challenger = kwargs.get("challenger_id")
        assert isinstance(challenger, str)
        return ReusePlan((), (), (), {"resolved": 10}, {}, (*names, challenger))

    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.tournament_reuse.resolve_reuse_plan", plan
    )
    return store, state


def _submission(
    tmp_path: Path,
    state: dict[str, Any],
    name: str,
    *,
    complete: str | None = "2026-10-28T00:00:00+00:00",
    approved: bool = True,
) -> dict[str, Any]:
    participant = config_descriptor()["participants"][0]
    participant.update(
        entrant_id=name, name=name, controller_id=sha256(name.encode()).hexdigest()
    )
    return {
        "submission_id": name,
        "controller_id": participant["controller_id"],
        "descriptor_asset": _asset(
            tmp_path / f"{name}.json",
            {"participant": participant, "assets": {}},
            "registration",
        ),
        "received_at_utc": "2026-10-01T00:00:00+00:00",
        "complete_at_utc": complete,
        "qualification_asset": _asset(
            tmp_path / f"{name}-qualified.json",
            {
                "controller_id": participant["controller_id"],
                "rules_approval_id": state["rules_approval_id"],
                "approved": approved,
                "revises_controller_id": None,
            },
            "qualification",
        ),
    }


def _state(store: Path) -> dict[str, Any]:
    return read_config_json(store / "admission_state.json")


def _result(
    challenger: str, *, score: float = 1200.0, lowest_tie: bool = False
) -> dict[str, Any]:
    ratings = {f"fixture-{index}": 1210.0 + index for index in range(12)}
    ratings["fixture-0"] = 1200.0
    if lowest_tie:
        ratings["fixture-1"] = 1200.0
    ratings[challenger] = score
    return {
        "ratings": ratings,
        "challenger_id": challenger,
        "completed_at_utc": "2026-10-29T00:00:00+00:00",
        "run_ref": "fixture-run",
        "full_coverage": {"games": 780},
        "joint_fit_id": sha256(b"fit").hexdigest(),
    }


@pytest.mark.parametrize(
    ("release", "utc", "cutoff_local"),
    [
        (
            "2026-04-01T00:00:00+01:00",
            "2026-03-31T23:00:00+00:00",
            "2026-03-28T23:00:00+00:00",
        ),
        (
            "2026-11-01T00:00:00+00:00",
            "2026-11-01T00:00:00+00:00",
            "2026-10-29T00:00:00+00:00",
        ),
        (
            "2027-11-01T00:00:00+00:00",
            "2027-11-01T00:00:00+00:00",
            "2027-10-29T01:00:00+01:00",
        ),
    ],
)
def test_release_cutoff_uses_elapsed_hours(
    release: str, utc: str, cutoff_local: str
) -> None:
    window = admission.release_window(release)
    assert window["release_at_utc"] == utc
    assert window["cutoff_at_local"] == cutoff_local
    assert (
        datetime.fromisoformat(utc) - datetime.fromisoformat(window["cutoff_at_utc"])
    ).total_seconds() == 72 * 3600


@pytest.mark.parametrize(
    "value", ["2026-11-01", "2026-11-02T00:00:00+00:00", "2026-04-01T00:00:00+00:00"]
)
def test_release_rejects_non_london_boundary(value: str) -> None:
    with pytest.raises(ValueError):
        admission.release_window(value)


def test_completed_fifo_survives_later_qualification_and_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    first = _submission(tmp_path, state, "first", complete=None)
    later = _submission(tmp_path, state, "later")
    assert admission.register_submission(store, first)["completion_sequence"] is None
    assert admission.register_submission(store, later)["completion_sequence"] == 1
    first["complete_at_utc"] = "2026-10-28T00:00:00+00:00"
    assert admission.register_submission(store, first)["completion_sequence"] == 2
    before = (store / "admission_state.json").read_bytes()
    with pytest.raises(ValueError, match="earlier"):
        admission.prepare_admission(
            store, "first", release_at="2026-11-01T00:00:00+00:00"
        )
    assert (store / "admission_state.json").read_bytes() == before
    attempt = admission.prepare_admission(
        store, "later", release_at="2026-11-01T00:00:00+00:00"
    )
    admission.record_admission_failure(store, attempt["attempt_id"], "Interrupted")
    retry = admission.prepare_admission(
        store, "later", release_at="2026-11-01T00:00:00+00:00"
    )
    assert retry["schedule_asset"] == attempt["schedule_asset"]
    assert retry["attempt_id"] == attempt["attempt_id"]
    assert _state(store)["submissions"]["later"]["completion_sequence"] == 1


@pytest.mark.parametrize("self_declared_official", [False, True])
def test_real_store_rejects_unapproved_initial_population_before_creation(
    tmp_path: Path, self_declared_official: bool
) -> None:
    config = config_descriptor(
        entrants=12, root=tmp_path, official=self_declared_official
    )
    store = tmp_path / "real-store"
    with pytest.raises(ValueError, match=r"(?i)official|released|snapshot"):
        admission.initialize_admission_store(
            store, config, rules=_rules(fixture_only=False)
        )
    assert not store.exists()


@pytest.mark.parametrize(
    "roster", [("priest", "rogue", "hunter", "warrior", "mage"), ("mage",) * 5]
)
def test_real_release_rejects_noncanonical_ordered_rosters(
    tmp_path: Path, roster: tuple[str, ...]
) -> None:
    config = config_descriptor(entrants=12, root=tmp_path, official=True)
    config["conditions"]["rosters"] = {"team_a": list(roster), "team_b": list(roster)}
    config["snapshot_id"] = snapshot_identity(config)
    with pytest.raises(ValueError, match="canonical ordered rosters"):
        admission._check_release_population(
            config, _rules(fixture_only=False), AssetVerifier(config)
        )


@pytest.mark.parametrize("field", admission._GAME_GATES)
def test_missing_approval_never_starts_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    store, state = _store(tmp_path, monkeypatch, **{field: None})
    assert (
        admission.register_submission(
            store, _submission(tmp_path, state, "challenger")
        )["state"]
        == "awaiting_approval"
    )
    before = (store / "admission_state.json").read_bytes()
    with pytest.raises(ValueError, match="approval"):
        admission.prepare_admission(
            store, "challenger", release_at="2026-11-01T00:00:00+00:00"
        )
    assert (store / "admission_state.json").read_bytes() == before


def test_cutoff_equality_allowed_and_postcutoff_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        admission, "_utc_now", lambda: datetime(2026, 10, 30, tzinfo=UTC)
    )
    admission.register_submission(
        store,
        _submission(
            tmp_path, state, "late", complete="2026-10-29T00:00:00.000001+00:00"
        ),
    )
    admission.register_submission(
        store,
        _submission(tmp_path, state, "on-time", complete="2026-10-29T00:00:00+00:00"),
    )
    with pytest.raises(ValueError, match="cutoff"):
        admission.prepare_admission(
            store, "late", release_at="2026-11-01T00:00:00+00:00"
        )
    assert (
        admission.prepare_admission(
            store, "on-time", release_at="2026-11-01T00:00:00+00:00"
        )["submission_id"]
        == "on-time"
    )


def test_nonpromotion_tie_applies_once_and_keeps_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    admission.register_submission(store, _submission(tmp_path, state, "challenger"))
    attempt = admission.prepare_admission(
        store, "challenger", release_at="2026-11-01T00:00:00+00:00"
    )
    monkeypatch.setattr(
        admission, "_admission_evidence", _identity_evidence, raising=False
    )
    result = _result("challenger")
    applied = admission.apply_admission(store, attempt["attempt_id"], result)
    assert applied["decision"] == "not_promoted"
    assert _state(store)["provisional_snapshot_id"] == state["provisional_snapshot_id"]
    before = (store / "admission_state.json").read_bytes()
    assert admission.apply_admission(store, attempt["attempt_id"], result) == applied
    assert (store / "admission_state.json").read_bytes() == before
    with pytest.raises(ValueError, match="different result"):
        admission.apply_admission(
            store, attempt["attempt_id"], _result("challenger", score=1199.0)
        )


def test_tied_lowest_incumbents_block_without_invented_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    admission.register_submission(store, _submission(tmp_path, state, "challenger"))
    attempt = admission.prepare_admission(
        store, "challenger", release_at="2026-11-01T00:00:00+00:00"
    )
    monkeypatch.setattr(
        admission, "_admission_evidence", _identity_evidence, raising=False
    )
    blocked = admission.apply_admission(
        store, attempt["attempt_id"], _result("challenger", score=1250, lowest_tie=True)
    )
    assert blocked["decision"] == "blocked"
    current = _state(store)
    assert current["provisional_snapshot_id"] == state["provisional_snapshot_id"]
    assert current["applied_attempt_ids"] == []


def test_promotion_commit_failure_preserves_membership_then_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    submission = _submission(tmp_path, state, "challenger")
    admission.register_submission(store, submission)
    attempt = admission.prepare_admission(
        store, "challenger", release_at="2026-11-01T00:00:00+00:00"
    )
    monkeypatch.setattr(
        admission, "_admission_evidence", _identity_evidence, raising=False
    )

    def retained(
        root: Path, config: dict[str, Any], result: object, **kwargs: object
    ) -> dict[str, Any]:
        assert kwargs["retained_ids"] == (
            *[f"fixture-{i}" for i in range(1, 12)],
            "challenger",
        )
        changed = copy.deepcopy(config)
        changed["participants"][0] = read_config_json(
            submission["descriptor_asset"]["path"]
        )["participant"]
        changed["snapshot_id"] = snapshot_identity(changed)
        return changed

    monkeypatch.setattr(
        admission, "_retained_admission_snapshot", retained, raising=False
    )
    original = admission._publish_json

    def fail(path: Path, value: object) -> None:
        if path.name == "admission_state.json":
            raise OSError("Injected membership failure")
        original(path, value)

    monkeypatch.setattr(admission, "_publish_json", fail)
    with pytest.raises(OSError, match="Injected"):
        admission.apply_admission(
            store, attempt["attempt_id"], _result("challenger", score=1250)
        )
    assert _state(store)["provisional_snapshot_id"] == state["provisional_snapshot_id"]
    monkeypatch.setattr(admission, "_publish_json", original)
    applied = admission.apply_admission(
        store, attempt["attempt_id"], _result("challenger", score=1250)
    )
    assert applied["decision"] == "promoted"
    assert _state(store)["provisional_snapshot_id"] != state["provisional_snapshot_id"]
    assert _state(store)["applied_attempt_ids"] == [attempt["attempt_id"]]


def test_late_attempt_cannot_mutate_frozen_edition_and_deferral_preserves_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    admission.register_submission(store, _submission(tmp_path, state, "challenger"))
    attempt = admission.prepare_admission(
        store, "challenger", release_at="2026-11-01T00:00:00+00:00"
    )
    monkeypatch.setattr(
        admission, "_admission_evidence", _identity_evidence, raising=False
    )
    monkeypatch.setattr(
        admission, "_utc_now", lambda: datetime(2026, 11, 1, tzinfo=UTC)
    )
    before = (store / "admission_state.json").read_bytes()
    with pytest.raises(ValueError, match="Late"):
        admission.apply_admission(store, attempt["attempt_id"], _result("challenger"))
    assert (store / "admission_state.json").read_bytes() == before
    closed = admission.defer_release(store, "2026-11", "Still incomplete")
    current = _state(store)
    assert closed["state"] == "frozen"
    assert current["active_edition"]["edition_id"] == "2026-12"
    assert current["submissions"]["challenger"]["completion_sequence"] == 1
    assert current["published_snapshot_id"] == state["published_snapshot_id"]
    assert not (store / "catalog.json").exists()
    assert admission.defer_release(store, "2026-11", "Still incomplete") == closed


def test_retained_preparation_cannot_cross_release_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    submission = _submission(tmp_path, state, "challenger")
    admission.register_submission(store, submission)
    attempt = admission.prepare_admission(
        store, "challenger", release_at="2026-11-01T00:00:00+00:00"
    )
    monkeypatch.setattr(admission, "_admission_evidence", _identity_evidence)
    clock = [datetime(2026, 10, 31, 23, 59, 59, tzinfo=UTC)]
    monkeypatch.setattr(admission, "_utc_now", lambda: clock[0])

    def retained(
        _: Path, config: dict[str, Any], __: object, **___: object
    ) -> dict[str, Any]:
        changed = copy.deepcopy(config)
        changed["participants"][0] = read_config_json(
            submission["descriptor_asset"]["path"]
        )["participant"]
        changed["snapshot_id"] = snapshot_identity(changed)
        clock[0] = datetime(2026, 11, 1, tzinfo=UTC)
        return changed

    monkeypatch.setattr(admission, "_retained_admission_snapshot", retained)
    before = (store / "admission_state.json").read_bytes()
    with pytest.raises(ValueError, match="Late"):
        admission.apply_admission(
            store, attempt["attempt_id"], _result("challenger", score=1250)
        )
    assert (store / "admission_state.json").read_bytes() == before
    assert _state(store)["provisional_snapshot_id"] == state["provisional_snapshot_id"]
    assert _state(store)["applied_attempt_ids"] == []


def test_publish_catalog_interruption_is_reconciled_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        admission, "_utc_now", lambda: datetime(2026, 11, 1, tzinfo=UTC)
    )
    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.tournament_assets.AssetVerifier.require",
        _empty_paths,
    )
    prepared = admission.prepare_release(store, "2026-11-01T00:00:00+00:00")
    original = admission._commit
    monkeypatch.setattr(admission, "_commit", _fail_commit)
    with pytest.raises(OSError, match="State failure"):
        admission.publish_release(store, prepared)
    assert (
        read_config_json(store / "catalog.json")["default"] == prepared["snapshot_id"]
    )
    assert _state(store)["published_snapshot_id"] == state["published_snapshot_id"]
    monkeypatch.setattr(admission, "_commit", original)
    assert admission.publish_release(store, prepared) == prepared
    current = _state(store)
    assert current["published_snapshot_id"] == prepared["snapshot_id"]
    assert current["active_edition"]["edition_id"] == "2026-12"
    before = (store / "admission_state.json").read_bytes()
    assert admission.publish_release(store, prepared) == prepared
    assert (store / "admission_state.json").read_bytes() == before


def test_incomplete_qualification_does_not_block_qualified_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    waiting = _submission(tmp_path, state, "waiting")
    waiting["qualification_asset"] = None
    assert admission.register_submission(store, waiting)["state"] == "awaiting_approval"
    admission.register_submission(store, _submission(tmp_path, state, "qualified"))
    assert (
        admission.prepare_admission(
            store, "qualified", release_at="2026-11-01T00:00:00+00:00"
        )["submission_id"]
        == "qualified"
    )


def test_submission_identity_and_completion_cannot_be_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    submission = _submission(tmp_path, state, "challenger")
    admission.register_submission(store, submission)
    for field, value in (
        ("complete_at_utc", "2026-10-28T01:00:00+00:00"),
        ("received_at_utc", "2026-10-02T00:00:00+00:00"),
        ("qualification_asset", None),
    ):
        changed = {**submission, field: value}
        before = (store / "admission_state.json").read_bytes()
        with pytest.raises(ValueError, match="cannot replace"):
            admission.register_submission(store, changed)
        assert (store / "admission_state.json").read_bytes() == before


def test_corrupt_qualification_asset_fails_before_attempt_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    submission = _submission(tmp_path, state, "challenger")
    admission.register_submission(store, submission)
    Path(submission["qualification_asset"]["path"]).write_text("{}")
    before = (store / "admission_state.json").read_bytes()
    with pytest.raises(ValueError, match="changed or is corrupt"):
        admission.prepare_admission(
            store, "challenger", release_at="2026-11-01T00:00:00+00:00"
        )
    assert (store / "admission_state.json").read_bytes() == before


def test_snapshot_failure_never_selects_a_promoted_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, state = _store(tmp_path, monkeypatch)
    admission.register_submission(store, _submission(tmp_path, state, "challenger"))
    attempt = admission.prepare_admission(
        store, "challenger", release_at="2026-11-01T00:00:00+00:00"
    )
    monkeypatch.setattr(admission, "_admission_evidence", _identity_evidence)

    def fail_retention(*_: object, **__: object) -> dict[str, Any]:
        raise OSError("Retained snapshot failure")

    monkeypatch.setattr(admission, "_retained_admission_snapshot", fail_retention)
    before = (store / "admission_state.json").read_bytes()
    with pytest.raises(OSError, match="Retained"):
        admission.apply_admission(
            store, attempt["attempt_id"], _result("challenger", score=1250)
        )
    assert (store / "admission_state.json").read_bytes() == before
    assert _state(store)["provisional_snapshot_id"] == state["provisional_snapshot_id"]


def test_unapproved_release_and_early_release_do_not_freeze_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = _store(tmp_path, monkeypatch, resource_limits=None)
    before = (store / "admission_state.json").read_bytes()
    with pytest.raises(ValueError, match="not arrived"):
        admission.prepare_release(store, "2026-11-01T00:00:00+00:00")
    monkeypatch.setattr(
        admission, "_utc_now", lambda: datetime(2026, 11, 1, tzinfo=UTC)
    )
    with pytest.raises(ValueError, match="resource_limits"):
        admission.prepare_release(store, "2026-11-01T00:00:00+00:00")
    assert (store / "admission_state.json").read_bytes() == before


def test_conflicting_prepared_release_never_changes_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = _store(tmp_path, monkeypatch)
    monkeypatch.setattr(
        admission, "_utc_now", lambda: datetime(2026, 11, 1, tzinfo=UTC)
    )
    monkeypatch.setattr(AssetVerifier, "require", _empty_paths)
    prepared = admission.prepare_release(store, "2026-11-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="immutable preparation"):
        admission.publish_release(store, {**prepared, "provisional_revision": 99})
    assert not (store / "catalog.json").exists()


def test_prefix_copy_is_bounded_and_reuses_identical_bytes(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    source = tmp_path / "source.csv"
    data = b"a,b\n" + b"1,2\n" * 300_000
    source.write_bytes(data + b"uncommitted suffix")
    identifier, first = admission._copy_prefix_asset(
        root, source, len(data), "full_report"
    )
    assert identifier == sha256(data).hexdigest()
    assert Path(first["path"]).read_bytes() == data
    assert admission._copy_prefix_asset(root, source, len(data), "full_report") == (
        identifier,
        first,
    )
    assert not (root / "assets" / ".copy-prefix.tmp").exists()


def test_commit_directory_sync_failure_reconciles_applied_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = _store(tmp_path, monkeypatch)
    admission.register_submission(
        store, _submission(tmp_path, _state(store), "challenger")
    )
    attempt = admission.prepare_admission(
        store, "challenger", release_at="2026-11-01T00:00:00+00:00"
    )
    monkeypatch.setattr(admission, "_admission_evidence", _identity_evidence)
    (store / "results").mkdir()
    original = admission._sync_directory

    def fail_after_replace(path: Path) -> None:
        if path == store:
            raise OSError("Directory sync failed")
        original(path)

    monkeypatch.setattr(admission, "_sync_directory", fail_after_replace)
    with pytest.raises(OSError, match="Directory sync failed"):
        admission.apply_admission(store, attempt["attempt_id"], _result("challenger"))
    assert _state(store)["applied_attempt_ids"] == [attempt["attempt_id"]]
    monkeypatch.setattr(admission, "_sync_directory", original)
    result = admission.apply_admission(
        store, attempt["attempt_id"], _result("challenger")
    )
    assert result["decision"] == "not_promoted"
    assert _state(store)["applied_attempt_ids"] == [attempt["attempt_id"]]


def test_real_population_checks_all_66_matchups_and_full_rows(tmp_path: Path) -> None:
    bundle = build_record_bundle(tmp_path / "complete", full=True)
    evidence = admission._population_evidence(bundle["config"])
    assert evidence["matchups"] == 66
    assert evidence["games"] == evidence["full_rows"] == 660
    assert evidence["snapshot_id"] == bundle["config"]["snapshot_id"]
    assert len(evidence["physical_evidence_id"]) == 64


def test_real_priority_only_population_cannot_start_full_admission(
    tmp_path: Path,
) -> None:
    bundle = build_record_bundle(tmp_path / "priority", full=False)
    with pytest.raises(ValueError, match="full measurements"):
        admission._population_evidence(bundle["config"])


def test_real_admission_retains_refits_and_prepares_next_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = datetime(2026, 10, 29, tzinfo=UTC)
    monkeypatch.setattr(admission, "_utc_now", lambda: clock)
    bundle = build_record_bundle(tmp_path / "incumbents", full=True, max_steps=1)
    extra = build_record_bundle(
        tmp_path / "challenger", entrants=13, maps=1, max_steps=1
    )
    participant = extra["config"]["participants"][-1]
    asset_ids = {participant["registration_asset"]}
    asset_ids.update(
        value["asset_id"]
        for key, value in participant["controller"]["content"].items()
        if key != "adapter_bindings" and value is not None
    )
    descriptor = {
        "participant": participant,
        "assets": {key: extra["config"]["assets"][key] for key in asset_ids},
    }
    store = tmp_path / "store"
    state = admission.initialize_admission_store(
        store,
        bundle["config"],
        rules=_rules(
            tied_incumbent_eviction={
                "entrant_order": [
                    item["entrant_id"] for item in bundle["config"]["participants"]
                ]
            }
        ),
    )
    submission: dict[str, Any] = {
        "submission_id": "actual-fixture-admission",
        "controller_id": participant["controller_id"],
        "descriptor_asset": _asset(
            tmp_path / "submitted-controller.json", descriptor, "registration"
        ),
        "received_at_utc": "2026-10-01T00:00:00+00:00",
        "complete_at_utc": "2026-10-28T00:00:00+00:00",
        "qualification_asset": _asset(
            tmp_path / "submitted-qualification.json",
            {
                "controller_id": participant["controller_id"],
                "rules_approval_id": state["rules_approval_id"],
                "approved": True,
                "revises_controller_id": None,
            },
            "qualification",
        ),
    }
    admission.register_submission(store, submission)
    attempt = admission.prepare_admission(
        store, submission["submission_id"], release_at="2026-11-01T00:00:00+00:00"
    )
    result = admission.execute_admission(
        store,
        attempt["attempt_id"],
        output_dir=tmp_path / "runs",
        num_envs=32,
        chunk_size=1,
    )
    assert result.planned_games == 780
    assert result.reused_games == 660
    assert result.executed_games == 120
    assert result.metadata["metrics"] == "full"
    loader_calls: list[object] = []

    def unexpected_loader(*args: object, **_: object) -> None:
        loader_calls.append(args)
        raise AssertionError("Completed admission resume must not load models")

    with monkeypatch.context() as context:
        context.setattr(
            "marl_battlegrounds.evaluation.tournament_assets.load_tournament_controller",
            unexpected_loader,
        )
        context.setattr(
            "marl_battlegrounds.evaluation.canonical.load_tournament_controller",
            unexpected_loader,
        )
        resumed = admission.execute_admission(
            store,
            attempt["attempt_id"],
            resume_from=result.run_dir,
            num_envs=32,
            chunk_size=1,
        )
    assert loader_calls == []
    assert resumed.metadata["executed_this_call"] == 0
    assert [(row["policy"], row["elo"]) for row in resumed.tournament_results] == [
        (row["policy"], row["elo"]) for row in result.tournament_results
    ]
    applied = admission.apply_admission(store, attempt["attempt_id"], result)
    assert applied["decision"] == "promoted"
    replacement = admission._snapshot(store, applied["resulting_snapshot_id"])
    assert len(replacement["participants"]) == 12
    assert participant["entrant_id"] in {
        item["entrant_id"] for item in replacement["participants"]
    }
    assert all(
        item["elo"] is not None and item["result_ref"] is not None
        for item in replacement["participants"]
    )
    assert admission._population_evidence(replacement)["games"] == 660
    assert admission.apply_admission(store, attempt["attempt_id"], result) == applied
    removed = {item["entrant_id"] for item in bundle["config"]["participants"]} - {
        item["entrant_id"] for item in replacement["participants"]
    }
    assert len(removed) == 1
    removed_item = next(
        item
        for item in bundle["config"]["participants"]
        if item["entrant_id"] in removed
    )
    assert removed_item["registration_asset"] not in replacement["assets"]
    assert len(replacement["record_sources"]) == 3
    from marl_battlegrounds.evaluation.tournament_reuse import resolve_reuse_plan

    verifier, paths = admission._snapshot_inputs(replacement)
    following = resolve_reuse_plan(
        replacement, paths, challenger_id="next-unexecuted-fixture", require_reuse=True
    )
    assert len(following.games) == 780
    assert len([game for game in following.games if game["origin"] is not None]) == 660
    assert verifier is not None
    clock = datetime(2026, 11, 1, tzinfo=UTC)
    prepared = admission.prepare_release(store, "2026-11-01T00:00:00+00:00")
    assert prepared["asset_evidence"]["full_rows"] == 660
    released = admission._snapshot(store, prepared["snapshot_id"])
    assert len(released["participants"]) == 12
    assert all(
        item["elo"] is not None and item["result_ref"] is not None
        for item in released["participants"]
    )
    assert not (store / "catalog.json").exists()


def _challenger_submission(
    tmp_path: Path, state: dict[str, Any], challenger: dict[str, Any]
) -> dict[str, Any]:
    participant = challenger["config"]["participants"][-1]
    asset_ids = {participant["registration_asset"]}
    asset_ids.update(
        value["asset_id"]
        for key, value in participant["controller"]["content"].items()
        if key != "adapter_bindings" and value is not None
    )
    descriptor = {
        "participant": participant,
        "assets": {key: challenger["config"]["assets"][key] for key in asset_ids},
    }
    return {
        "submission_id": "pre-red-zone-challenger",
        "controller_id": participant["controller_id"],
        "descriptor_asset": _asset(
            tmp_path / "challenger-controller.json", descriptor, "registration"
        ),
        "received_at_utc": "2026-10-01T00:00:00+00:00",
        "complete_at_utc": "2026-10-28T00:00:00+00:00",
        "qualification_asset": _asset(
            tmp_path / "challenger-qualification.json",
            {
                "controller_id": participant["controller_id"],
                "rules_approval_id": state["rules_approval_id"],
                "approved": True,
                "revises_controller_id": None,
            },
            "qualification",
        ),
    }


def _tree(directory: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def _no_new_games(*args: object, **kwargs: object) -> None:
    raise AssertionError("an old release must be reused, not played again")


def test_release_saved_before_red_zone_is_admitted_under_its_original_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = build_record_bundle(tmp_path / "old", maps=1, max_steps=1, historical=True)
    config = bundle["config"]
    assert config["snapshot_id"] == snapshot_identity(config)
    store = tmp_path / "store"
    state = admission.initialize_admission_store(store, config, rules=_rules())
    assert state["published_snapshot_id"] == config["snapshot_id"]
    stored = admission._snapshot(store, config["snapshot_id"])
    assert stored["snapshot_id"] == snapshot_identity(stored) == config["snapshot_id"]
    # The public config route reuses every recorded game under the same identity.
    from marl_battlegrounds.evaluation import canonical

    monkeypatch.setattr(canonical, "_active_pair", _no_new_games)
    reused = marl_bgs.run_tournament(config=config, output_dir=tmp_path / "reuse")
    assert isinstance(reused, CanonicalTournamentResult)
    assert reused.snapshot_id == config["snapshot_id"]
    assert reused.planned_games == reused.reused_games == 132
    assert reused.executed_games == 0
    # A Red Zone rule never reaches a write: the config owns its rules.
    for depth in (0.0, 5.0):
        with pytest.raises(
            ValueError,
            match=r"^Config owns scientific settings; omit red_zone_depth$",
        ):
            marl_bgs.run_tournament(
                config=config, red_zone_depth=depth, output_dir=tmp_path / "runs"
            )
    assert not (tmp_path / "runs").exists()


def test_edited_source_configuration_is_rejected_even_when_rehashed(
    tmp_path: Path,
) -> None:
    bundle = build_record_bundle(
        tmp_path / "old", entrants=2, maps=1, max_steps=1, historical=True
    )
    config = copy.deepcopy(bundle["config"])
    source = config["conditions"]["map_sources"][0]
    asset = config["assets"][source["source_config_asset"]]
    path = Path(asset["path"])
    content = json.loads(path.read_bytes())
    assert len(content) == 12
    # The same one-point rule written as current content is still an edit.
    edited = canonical_json({**content, "team_deathmatch_red_zone_depth": 0.0})
    path.write_bytes(edited)
    asset.update(sha256=sha256(edited).hexdigest(), size_bytes=len(edited))
    config["snapshot_id"] = snapshot_identity(config)
    with pytest.raises(ValueError, match="differs from its declared identity"):
        marl_bgs.run_tournament(config=config, output_dir=tmp_path / "edited")
    assert not (tmp_path / "edited").exists()


def test_challenger_needing_new_games_under_red_zone_stops_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = datetime(2026, 10, 29, tzinfo=UTC)
    monkeypatch.setattr(admission, "_utc_now", lambda: clock)
    bundle = build_record_bundle(
        tmp_path / "old", full=True, max_steps=1, historical=True
    )
    evidence = admission._population_evidence(bundle["config"])
    assert evidence["snapshot_id"] == bundle["config"]["snapshot_id"]
    assert evidence["games"] == evidence["full_rows"] == 660
    store = tmp_path / "store"
    state = admission.initialize_admission_store(
        store, bundle["config"], rules=_rules()
    )
    challenger = build_record_bundle(
        tmp_path / "challenger", entrants=13, maps=1, max_steps=1
    )
    submission = _challenger_submission(tmp_path, state, challenger)
    admission.register_submission(store, submission)
    attempt = admission.prepare_admission(
        store, submission["submission_id"], release_at="2026-11-01T00:00:00+00:00"
    )
    before = _tree(store)
    # The challenger's games would be new games under the current rule, which
    # a release saved before Red Zone cannot hold.
    with pytest.raises(
        ValueError, match=r"^Snapshot configurations were saved before the Red Zone"
    ):
        admission.execute_admission(
            store,
            attempt["attempt_id"],
            output_dir=tmp_path / "runs",
            num_envs=32,
            chunk_size=1,
        )
    assert not (tmp_path / "runs").exists()
    assert _tree(store) == before
