"""Check complete artificial Big 12/13 bundles through real recording authorities.

The fixtures invent measurements but use actual configs, registrations, spawn
banks, versioned schedules and CSV schemas. These checks call no action method,
play no games and perform no rating fit. They establish useful host fixtures,
not scientific qualification or a released official tournament. Source-config
preparation returns frozen configuration IDs that name exactly the configs in
its verified cache, and that cache keeps signed zero exact.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

# Shared private preparation is inspected to prove cache and identity contracts.
# pyright: reportPrivateUsage=false
from canonical_record_fixtures import build_record_bundle
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.tournament_assets import (
    AssetVerifier,
    load_tournament_controller,
    prepare_tournament_assets,
)
from marl_battlegrounds.evaluation.tournament_evidence import prepare_reuse_evidence
from marl_battlegrounds.evaluation.tournament_headlines import validate_evidence
from marl_battlegrounds.evaluation.tournament_records import TournamentRecords
from marl_battlegrounds.evaluation.tournament_reuse import (
    analysis_schedule,
    resolve_reuse_plan,
)


@pytest.mark.parametrize("entrants", [12, 13])
def test_complete_record_population_passes_real_physical_checks(
    tmp_path: Path, entrants: int
) -> None:
    bundle = build_record_bundle(tmp_path, entrants=entrants)
    config = bundle["config"]
    verifier = AssetVerifier(config)
    plan = resolve_reuse_plan(config, bundle["paths"])
    records = TournamentRecords(
        config, plan.games, plan.jobs, verifier, manifest={"run_id": "fixture-analysis"}
    )
    records.require_coverage(metrics="priority")
    rows = [row for batch in records.iter_rows("match_results.csv") for row in batch]
    assert len(rows) == entrants * (entrants - 1) // 2 * 10
    assert len({row["episode_id"] for row in rows}) == len(rows)
    evidence = prepare_reuse_evidence(
        records,
        analysis_schedule(plan),
        rows,
        participants={row["entrant_id"]: row for row in config["participants"]},
        registrations=bundle["registrations"],
    )
    validate_evidence(rows, evidence)
    assert len(evidence["population_system_ids"]) == entrants
    assert config["release"] is None
    assert len({row["controller_id"] for row in config["participants"]}) == entrants


def test_fixture_full_report_and_factory_are_real_contract_inputs(
    tmp_path: Path,
) -> None:
    bundle = build_record_bundle(tmp_path, entrants=2, maps=1, full=True)
    config = bundle["config"]
    prepared = prepare_tournament_assets(config, roles=("full_report",))
    assert prepared["missing"] == []
    verifier = AssetVerifier(prepared["config"])
    plan = resolve_reuse_plan(config, bundle["paths"])
    records = TournamentRecords(
        config, plan.games, plan.jobs, verifier, manifest={"run_id": "fixture-analysis"}
    )
    records.require_coverage(metrics="full")
    reports = [row for batch in records.iter_rows("full_metrics.csv") for row in batch]
    assert len(reports) == 2
    assert all(set(FULL_METRIC_NAMES) <= row.keys() for row in reports)
    method = load_tournament_controller(config["participants"][0], verifier)
    assert method.name == "fixture-00"


def test_stored_rating_claims_use_the_exact_source_summary(tmp_path: Path) -> None:
    bundle = build_record_bundle(tmp_path, entrants=2, maps=1)
    config = bundle["config"]
    plan = resolve_reuse_plan(config, bundle["paths"])
    records = TournamentRecords(
        config, plan.games, plan.jobs, AssetVerifier(config), manifest={"run_id": "new"}
    )
    claims: list[dict[str, Any]] = [
        {
            **row,
            "elo": float(1212 - index),
            "result_ref": {
                "source_id": config["record_sources"][0]["source_id"],
                "table": "tournament_results",
                "policy": row["name"],
            },
        }
        for index, row in enumerate(config["participants"])
    ]
    records.verify_stored_ratings(claims)
    claims[0]["elo"] += 1
    with pytest.raises(ValueError, match="snapshot Elo"):
        records.verify_stored_ratings(claims)


def test_verified_configuration_cache_preserves_signed_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import evaluation_conditions
    from marl_battlegrounds.evaluation.canonical import _source_configs

    bundle = build_record_bundle(tmp_path, entrants=2, maps=1)
    config = bundle["config"]
    verifier = AssetVerifier(config)
    _, _, _, prepared, frozen_ids = _source_configs(config, verifier)
    assert set(frozen_ids.values()) == set(prepared)
    plan = resolve_reuse_plan(config, bundle["paths"])
    records = TournamentRecords(
        config, plan.games, plan.jobs, verifier, manifest={"run_id": "new"}
    )
    rows = [row for batch in records.iter_rows("match_results.csv") for row in batch]

    def duplicate_work(*args: object, **kwargs: object) -> object:
        raise AssertionError("verified configuration must not be restored twice")

    monkeypatch.setattr(evaluation_conditions, "restore_config", duplicate_work)
    kwargs: dict[str, Any] = {
        "participants": {row["entrant_id"]: row for row in config["participants"]},
        "registrations": bundle["registrations"],
        "verified_configurations": prepared,
    }
    prepare_reuse_evidence(records, analysis_schedule(plan), rows, **kwargs)
    manifest = records.source_manifest(plan.games[0])
    content = manifest["configurations"][plan.games[0]["source_config_id"]]
    zero = next(
        index for index, value in enumerate(content["obstacles"][0]) if value == 0.0
    )
    content["obstacles"][0][zero] = -0.0
    with pytest.raises(ValueError, match="differs from verified content"):
        prepare_reuse_evidence(records, analysis_schedule(plan), rows, **kwargs)


def test_failed_read_scope_does_not_commit_full_validation_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation.scalar_reports import (
        IndexedScalarTable,
        OriginKey,
        ScalarCell,
    )

    bundle = build_record_bundle(tmp_path, entrants=2, maps=1, full=True)
    config = bundle["config"]
    plan = resolve_reuse_plan(config, bundle["paths"])
    records = TournamentRecords(
        config, plan.games, plan.jobs, AssetVerifier(config), manifest={"run_id": "new"}
    )
    manifest_path = Path(
        config["assets"][config["record_sources"][0]["manifest_asset"]]["path"]
    )
    original = IndexedScalarTable.iter_rows
    altered = False
    reads = 0

    def change_and_restore(
        self: IndexedScalarTable,
        origins: Sequence[OriginKey],
        *,
        batch_size: int = 128,
        columns: Sequence[str] | None = None,
    ) -> Iterator[tuple[dict[str, ScalarCell], ...]]:
        nonlocal altered, reads
        for batch in original(self, origins, batch_size=batch_size, columns=columns):
            if batch and FULL_METRIC_NAMES[-1] in batch[0]:
                reads += 1
                if not altered:
                    payload = manifest_path.read_bytes()
                    manifest_path.write_bytes(payload + b" ")
                    manifest_path.write_bytes(payload)
                    altered = True
            yield batch

    monkeypatch.setattr(IndexedScalarTable, "iter_rows", change_and_restore)
    with pytest.raises(ValueError, match="changed during verification"):
        records.require_coverage(metrics="full")
    assert records._checked_full == set()
    failed_reads = reads
    records.require_coverage(metrics="full")
    assert reads == failed_reads + len(plan.games)
    assert records._checked_full == {game["logical_game_id"] for game in plan.games}
