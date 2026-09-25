"""Check configured orchestration, immutable reuse and active model execution.

Artificial complete records test the public reuse/save/resume route without games.
Small one-tick CPU games check the fresh path through the real evaluator, writer
and physical evidence checks. Fixture release pins are isolated test data, not an
official release or evidence of scientific qualification.

A current bundle pins the current scalar schema, run schema 2 and replay schema
4 and holds configurations at Red Zone depth 5.0. A bundle saved before the Red
Zone rule pins (14, 2, 3) and holds 12-key depth-0.0 configurations under their
original IDs: it loads and reuses every game while running none, and any request
that needs a new game fails with the reuse-only message before writing anything.
"""

import copy
import json
import re

# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import Any

import pytest
from tests.canonical_fixtures import config_descriptor

import marl_battlegrounds as marl_bgs
from canonical_record_fixtures import build_record_bundle
from marl_battlegrounds.evaluation import canonical
from marl_battlegrounds.evaluation import tournament_config as configs


def _pin(monkeypatch: pytest.MonkeyPatch, value: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(value)
    value["release"] = config_descriptor(entrants=12, official=True)["release"]
    from marl_battlegrounds.tasks import list_tdm_maps

    maps = {row.map_id: row for row in list_tdm_maps()}
    for source in value["conditions"]["map_sources"]:
        source["registered_map"] = maps[source["map_id"]].model_dump(mode="json")
        source["split"] = maps[source["map_id"]].split
    for index, participant in enumerate(value["participants"]):
        participant["elo"] = 1212.0 - index
        participant["result_ref"] = {
            "source_id": value["record_sources"][0]["source_id"],
            "table": "tournament_results",
            "policy": participant["name"],
        }
    value["snapshot_id"] = configs.snapshot_identity(value)
    monkeypatch.setattr(
        configs,
        "_installed_catalog",
        lambda: {
            "default": value["snapshot_id"],
            "snapshots": {
                value["snapshot_id"]: {"config": value, "release": value["release"]}
            },
        },
    )
    return value


def _unexpected(*args: object, **kwargs: object) -> None:
    raise AssertionError("This complete reuse path must not load or execute a model")


def test_twelve_only_reuse_saves_and_resumes_without_models_or_refitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs")
    selected = _pin(monkeypatch, bundle["config"])
    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    result = marl_bgs.run_canonical_tournament(
        config=selected, output_dir=tmp_path / "out"
    )
    assert result.status == "complete"
    assert result.protocol_compliant
    assert result.metadata["official_snapshot_verified"]
    assert result.planned_games == result.reused_games == 660
    assert result.executed_games == result.metadata["executed_this_call"] == 0
    assert len(result.table("tournament_rankings")["rank"]) == 12
    from marl_battlegrounds.evaluation import tournament_statistics

    monkeypatch.setattr(tournament_statistics, "summarize_tournament", _unexpected)
    resumed = marl_bgs.run_canonical_tournament(resume_from=result.run_dir)
    assert resumed.matches == result.matches
    assert resumed.tournament_results == result.tournament_results
    assert resumed.metadata["executed_this_call"] == 0


def test_custom_config_owns_population_and_scientific_defaults(tmp_path: Path) -> None:
    bundle = build_record_bundle(tmp_path / "inputs", entrants=3, maps=1)
    result = marl_bgs.run_tournament(config=bundle["config"], metrics="none")
    assert len(result.matches) == 6
    assert result.metadata["metrics"] == "none"
    assert result.headline_metrics == ()
    assert result.table("priority_metrics") == {}
    with pytest.raises(ValueError, match="competing populations"):
        marl_bgs.run_tournament(["random", "tdm-alpha"], config=bundle["config"])
    with pytest.raises(ValueError, match="omit max_steps"):
        marl_bgs.run_tournament(config=bundle["config"], max_steps=16)


def test_default_missing_bundle_fails_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        configs,
        "_installed_catalog",
        lambda: dict[str, Any](default=None, snapshots={}),
    )
    with pytest.raises(ValueError, match="official"):
        marl_bgs.run_canonical_tournament(output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("saved", [False, True])
def test_fresh_execution_preserves_original_identity_and_active_pair(
    tmp_path: Path,
    saved: bool,
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1, max_steps=1)
    result = canonical._run_resolved_tournament(
        bundle["config"],
        rerun_existing=True,
        num_envs=2,
        chunk_size=1,
        output_dir=tmp_path / "out" if saved else None,
    )
    assert result.status == "complete"
    assert result.executed_games == result.metadata["executed_this_call"] == 2
    assert result.reused_games == 0
    assert {row["episode_id"] for row in result.matches} == {
        game["execution"]["episode_id"] for game in bundle["games"]
    }
    assert all(row["episode_length"] == 1 for row in result.matches)
    assert {row["seed_id"] for row in result.matches} == {
        game["execution"]["seed_id"] for game in bundle["games"]
    }
    assert len(result.metadata["physical_evidence"]["games"]) == 2


def test_bare_policy_challenger_uses_shared_authority_and_fixed_team_a(
    tmp_path: Path,
) -> None:
    from canonical_record_fixtures import fixture_policy

    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1, max_steps=1)
    result = canonical._run_resolved_tournament(
        bundle["config"],
        system=fixture_policy(7),
        num_envs=2,
        chunk_size=1,
    )
    assert len(result.matches) == 6
    assert result.reused_games == 2
    assert result.executed_games == 4
    new = [
        game
        for game in result.metadata["canonical_plan"]["games"]
        if game["origin"] is None
    ]
    assert all(game["team_a"] == result.challenger_id for game in new)
    assert all(
        game["execution"]["action_stream_version"] == "evaluation-systems-v1"
        for game in new
    )
    assert (
        result.metadata["participant_registrations"][result.challenger_id]["kind"]
        == "policy"
    )


def test_reused_physical_failure_prevents_models_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from canonical_record_fixtures import fixture_policy
    from marl_battlegrounds.evaluation import tournament_evidence

    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1, max_steps=1)

    def fail(*_: object, **__: object) -> None:
        raise ValueError("Declared physical evidence is wrong")

    monkeypatch.setattr(tournament_evidence, "prepare_reuse_evidence", fail)
    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    with pytest.raises(ValueError, match="physical evidence"):
        canonical._run_resolved_tournament(
            bundle["config"],
            system=fixture_policy(7),
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_saved_games_finish_summary_without_reexecuting_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation import tournament_statistics

    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1, max_steps=1)
    original = tournament_statistics.summarize_tournament

    def fail(*_: object, **__: object) -> None:
        raise RuntimeError("Injected summary failure")

    monkeypatch.setattr(tournament_statistics, "summarize_tournament", fail)
    with pytest.raises(RuntimeError, match="summary failure"):
        canonical._run_resolved_tournament(
            bundle["config"],
            rerun_existing=True,
            num_envs=2,
            chunk_size=1,
            output_dir=tmp_path / "out",
        )
    run_dir = next((tmp_path / "out").iterdir())
    before = {
        p.relative_to(run_dir): p.read_bytes()
        for p in run_dir.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="metrics"):
        canonical._run_resolved_tournament(
            bundle["config"],
            rerun_existing=True,
            metrics="none",
            resume_from=run_dir,
        )
    assert before == {
        p.relative_to(run_dir): p.read_bytes()
        for p in run_dir.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(tournament_statistics, "summarize_tournament", original)
    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    result = canonical._run_resolved_tournament(
        bundle["config"],
        rerun_existing=True,
        resume_from=run_dir,
    )
    assert result.status == "complete"
    assert result.metadata["executed_this_call"] == 0
    assert len(result.matches) == 2


def test_relocated_hints_resume_without_rewriting_immutable_plan(
    tmp_path: Path,
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1)
    result = marl_bgs.run_tournament(
        config=bundle["config"], output_dir=tmp_path / "out"
    )
    assert result.run_dir is not None
    plan_path = result.run_dir / "tournament_config.json"
    before = plan_path.read_bytes()
    relocated = tmp_path / "relocated"
    (tmp_path / "inputs").rename(relocated)
    moved = copy.deepcopy(bundle["config"])
    moved["source_location"] = str(relocated)
    for asset in moved["assets"].values():
        asset["path"] = str(relocated / Path(asset["path"]).name)
    resumed = marl_bgs.run_tournament(config=moved, resume_from=result.run_dir)
    assert resumed.status == "complete"
    assert resumed.matches == result.matches
    assert resumed.metadata["executed_this_call"] == 0
    assert plan_path.read_bytes() == before
    assert (
        len(marl_bgs.load_results(result.run_dir).table("matches")["episode_id"]) == 2
    )
    assert marl_bgs.run_tournament(resume_from=result.run_dir).status == "complete"


def test_reused_and_new_replays_keep_original_capture_ids_under_none(
    tmp_path: Path,
) -> None:
    from hashlib import sha256

    from canonical_record_fixtures import fixture_policy
    from marl_battlegrounds.evaluation.admission import _source_from_run

    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1, max_steps=1)
    original = canonical._run_resolved_tournament(
        bundle["config"],
        rerun_existing=True,
        metrics="none",
        full_metrics_episodes=(1,),
        replay_episodes=(1,),
        output_dir=tmp_path / "original",
        num_envs=2,
        chunk_size=1,
    )
    assert original.run_dir is not None and original._record_access is not None
    config = copy.deepcopy(bundle["config"])
    source = _source_from_run(tmp_path / "frozen", original.run_dir, config["assets"])
    config["record_sources"].append(source)
    games = copy.deepcopy(original.metadata["canonical_plan"]["games"])
    for game in games:
        game["origin"] = {
            "source_id": source["source_id"],
            **original._record_access.origin(game),
        }
    data = b"".join(configs.canonical_json(game) + b"\n" for game in games)
    path = tmp_path / "new-games.jsonl"
    path.write_bytes(data)
    config["assets"]["games.jsonl"].update(
        path=str(path), sha256=sha256(data).hexdigest(), size_bytes=len(data)
    )
    config["snapshot_id"] = configs.snapshot_identity(config)
    config = configs.load_tournament_config(config, official=False)
    result = canonical._run_resolved_tournament(
        config,
        system=fixture_policy(7),
        metrics="none",
        full_metrics_episodes=(1, 3),
        replay_episodes=(1, 3),
        output_dir=tmp_path / "challenger",
        num_envs=2,
        chunk_size=1,
    )
    assert result.reused_games == 2 and result.executed_games == 4
    assert len(result.replay_paths) == 2
    assert len(result.table("full_metrics")["episode_id"]) == 2
    assert result.headline_metrics == ()
    assert result.replay_paths[0] == original.replay_paths[0]
    assert result.replay_paths[1].parent != original.replay_paths[0].parent
    assert result.run_dir is not None
    for missing in (result.run_dir / "full_metrics.csv", result.replay_paths[1]):
        content = missing.read_bytes()
        missing.unlink()
        before = {
            str(path.relative_to(result.run_dir)): sha256(path.read_bytes()).hexdigest()
            for path in result.run_dir.rglob("*")
            if path.is_file()
        }
        try:
            with pytest.raises((ValueError, OSError)):
                canonical._run_resolved_tournament(
                    config,
                    system=fixture_policy(7),
                    metrics="none",
                    full_metrics_episodes=(1, 3),
                    replay_episodes=(1, 3),
                    resume_from=result.run_dir,
                    num_envs=2,
                    chunk_size=1,
                )
            after = {
                str(path.relative_to(result.run_dir)): sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in result.run_dir.rglob("*")
                if path.is_file()
            }
            assert after == before
        finally:
            missing.write_bytes(content)


def test_pinned_official_roster_requires_canonical_order_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs")
    changed = copy.deepcopy(bundle["config"])
    for roster in changed["conditions"]["rosters"].values():
        roster[0], roster[1] = roster[1], roster[0]
    selected = _pin(monkeypatch, changed)
    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    with pytest.raises(ValueError, match="canonical ordered"):
        marl_bgs.run_canonical_tournament(config=selected, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_custom_release_text_cannot_claim_official_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = build_record_bundle(tmp_path / "inputs")
    selected = _pin(monkeypatch, bundle["config"])
    selected["release"]["approval_id"] = "self-declared-custom-approval"
    selected["snapshot_id"] = configs.snapshot_identity(selected)
    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    result = marl_bgs.run_tournament(config=selected, output_dir=tmp_path / "out")
    from marl_battlegrounds.evaluation.results import CanonicalTournamentResult

    assert isinstance(result, CanonicalTournamentResult)
    assert not result.protocol_compliant
    assert result.metadata["official_snapshot_verified"] is False
    assert result.metadata["budget"]["protocol_compliant"] is False
    assert result.metadata["canonical_plan"]["budget"]["protocol_compliant"] is False
    resumed = marl_bgs.run_tournament(resume_from=result.run_dir)
    assert isinstance(resumed, CanonicalTournamentResult)
    assert not resumed.protocol_compliant
    with pytest.raises(ValueError, match="custom"):
        marl_bgs.run_canonical_tournament(resume_from=result.run_dir)


_PRE_RED_ZONE_MESSAGE = (
    "Snapshot configurations were saved before the Red Zone rule; recorded "
    "games can be reused, but new games need a snapshot prepared with "
    "current configurations."
)


def _source_contents(bundle: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    return [
        (
            json.loads(bundle["paths"][source["source_config_asset"]].read_bytes()),
            source["source_config_id"],
        )
        for source in bundle["config"]["conditions"]["map_sources"]
    ]


def test_current_bundle_pins_current_records_and_the_default_red_zone(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.evaluation_conditions import (
        restore_recorded_config,
    )
    from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION

    bundle = build_record_bundle(tmp_path / "inputs", entrants=2, maps=1)
    pins = bundle["config"]["compatibility"]
    assert (pins["scalar_schema"], pins["run_schema"], pins["replay_schema"]) == (
        METRIC_SCHEMA_VERSION,
        2,
        4,
    )
    for content, identifier in _source_contents(bundle):
        assert len(content) == 13
        assert content["team_deathmatch_red_zone_depth"] == 5.0
        config, historical = restore_recorded_config(content, identifier)
        assert not historical
        assert config.team_deathmatch_red_zone_depth == 5.0


def test_pre_red_zone_bundle_reuses_every_game_and_refuses_new_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from canonical_record_fixtures import fixture_policy
    from marl_battlegrounds.evaluation.evaluation_conditions import (
        restore_recorded_config,
    )

    bundle = build_record_bundle(
        tmp_path / "inputs", entrants=2, maps=1, max_steps=1, historical=True
    )
    config = bundle["config"]
    pins = config["compatibility"]
    assert (pins["scalar_schema"], pins["run_schema"], pins["replay_schema"]) == (
        14,
        2,
        3,
    )
    for content, identifier in _source_contents(bundle):
        assert len(content) == 12
        assert "team_deathmatch_red_zone_depth" not in content
        restored, historical = restore_recorded_config(content, identifier)
        assert historical
        assert restored.team_deathmatch_red_zone_depth == 0.0
    assert configs.load_tournament_config(config, official=False) == config
    inputs = {
        path: path.read_bytes()
        for path in (tmp_path / "inputs").rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(canonical, "_active_pair", _unexpected)
    reused = marl_bgs.run_tournament(config=config, output_dir=tmp_path / "out")
    from marl_battlegrounds.evaluation.results import CanonicalTournamentResult

    assert isinstance(reused, CanonicalTournamentResult)
    assert reused.status == "complete"
    assert reused.planned_games == reused.reused_games == 2
    assert reused.executed_games == reused.metadata["executed_this_call"] == 0
    # A challenger adds games and a rerun replays recorded ones: both need new games.
    message = f"^{re.escape(_PRE_RED_ZONE_MESSAGE)}$"
    with pytest.raises(ValueError, match=message):
        canonical._run_resolved_tournament(
            config,
            system=fixture_policy(7),
            output_dir=tmp_path / "challenger",
            num_envs=2,
            chunk_size=1,
        )
    with pytest.raises(ValueError, match=message):
        canonical._run_resolved_tournament(
            config,
            rerun_existing=True,
            output_dir=tmp_path / "rerun",
            num_envs=2,
            chunk_size=1,
        )
    assert not (tmp_path / "challenger").exists()
    assert not (tmp_path / "rerun").exists()
    assert {
        path: path.read_bytes()
        for path in (tmp_path / "inputs").rglob("*")
        if path.is_file()
    } == inputs
