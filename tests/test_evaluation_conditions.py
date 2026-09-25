"""Check paired evaluation resolution and exact saved scientific conditions.

These CPU tests exercise schedules without running games wherever possible. The
small saved cases check current state, exact outcomes, resumed configuration
identity and rejection before any run/pass mutation. Authored single episodes
remain valid and are never expanded into undeclared comparisons.

Saved configs from before the Red Zone rule: the four 12-key configs in
tests/fixtures/scalar_schema_2/run_details.json restore at depth 0.0, and
config_record(historical=True) and restore_recorded_config reproduce their
recorded IDs and keys. Content that differs from its ID fails before any
array is built. Only depth +0.0 has a historical identity, a positive depth
can never be written as resolved config V1, and a saved pass from before the
rule is readable but cannot be resumed.

The Red Zone option: option(..., missing=) reads a pass saved before an option
existed as its missing value (0.0 for red_zone_depth) and compares explicit
values with it; red_zone_depth_option defaults to 5.0, inherits a saved depth,
accepts a value with the same float32 value, refuses ints, and refuses -0.0
against a saved or missing 0.0 (the float32 bytes differ). evaluate records
the depth in its pass options; 0.0 and 6.0 give different configuration IDs
from 5.0; omitted, an exact source keeps its own depth; a supplied depth that
differs from an exact source, even 5.0, raises "conflicts" before any file is
written. Public evaluate refuses to resume a pass saved before the rule with
the one clear message, changing no file, whether maps and red_zone_depth are
omitted or supplied (0.0 or 5.0). A pass counts as saved before the rule when
its generated contract lacks the depth option or its content has 12 keys;
each sign alone is enough, and a pass with no contract is covered too.
"""

# Host setup tests intentionally inspect the private resolver boundary.
# pyright: reportPrivateUsage=false

import json
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

from marl_battlegrounds.core import env as core
from marl_battlegrounds.evaluation import evaluation_conditions
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    evaluate,
    evaluate_episodes,
)
from marl_battlegrounds.evaluation.evaluation_conditions import (
    OMITTED,
    Omitted,
    config_record,
    default_maps,
    option,
    prepare_schedule,
    red_zone_depth_option,
    restore_config,
    restore_recorded_config,
    same_float32,
    saved_specs,
)
from marl_battlegrounds.evaluation.policy_execution import independent_policies, policy
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    _swap_spawn_banks,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
)

runner = import_module("marl_battlegrounds.evaluation.evaluate")
# Four raw configs saved before the Red Zone rule, keyed by their recorded IDs.
_PRE_RED_ZONE_RUN_DETAILS = (
    Path(__file__).parent / "fixtures" / "scalar_schema_2" / "run_details.json"
)


def _resolved(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - Exercise public keyword combinations.
    captured: dict[str, Any] = {}

    def capture(a: object, b: object, specs: object, **options: Any) -> dict[str, Any]:  # noqa: ANN401 - Intercept the resolver's keyword contract.
        captured.update(options)
        captured["specs"] = specs
        return captured

    monkeypatch.setattr(runner, "_run_evaluation", capture)
    evaluate("random", "random", **kwargs)
    return captured


@pytest.mark.parametrize("budget", [2, 100, 102])
def test_paired_budget_cycles_maps_by_complete_pair(
    budget: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _resolved(monkeypatch, num_episodes=budget, max_steps=1)
    specs = result["specs"]
    assert len(specs) == budget
    assert [s.episode_id for s in specs] == list(range(1, budget + 1))
    for index in range(0, budget, 2):
        first, second = specs[index : index + 2]
        assert (
            first.map_id
            == second.map_id
            == CANONICAL_TDM_EVALUATION_MAP_IDS[(index // 2) % 5]
        )
        assert first.seed_id == second.seed_id == index // 2 + 1
        assert first.paired_comparison_key == second.paired_comparison_key
        assert first.spawn_locations == 0 and second.spawn_locations == 1
        for a, b in zip(
            jax.tree.leaves(_swap_spawn_banks(first.env_config)),
            jax.tree.leaves(second.env_config),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
    if budget == 102:
        assert list(Counter(s.map_id for s in specs).values()) == [22, 20, 20, 20, 20]


@pytest.mark.parametrize("phase", ["validation-1", "train", "Validation"])
def test_custom_phase_needs_explicit_maps_before_files(
    phase: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Supply maps explicitly"):
        evaluate(
            "random", "random", num_episodes=2, phase=phase, output_dir=tmp_path / "bad"
        )
    assert not list(tmp_path.iterdir())


def test_validation_and_evaluation_default_maps_have_single_owner() -> None:
    assert default_maps("validation") == tuple(
        m.map_id for m in list_tdm_maps() if m.split == "validation"
    )
    assert default_maps("evaluation") == CANONICAL_TDM_EVALUATION_MAP_IDS


@pytest.mark.parametrize("mode", ["default", "swapped"])
def test_fixed_single_game_and_ordered_rosters(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _resolved(
        monkeypatch,
        num_episodes=1,
        maps=[12],
        spawn_mode=mode,
        system_roster=("priest",),
        opponent_roster=("mage", "mage"),
    )
    spec = result["specs"][0]
    np.testing.assert_array_equal(
        spec.env_config.agent_profile.class_ids, [5, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    )
    assert spec.spawn_locations == int(mode == "swapped")
    assert spec.paired_comparison_key is None


@pytest.mark.parametrize(
    "options,match",
    [
        ({"num_episodes": 1}, "even"),
        ({"maps": []}, "at least one"),
        ({"save_replays": 1, "replay_episodes": [2]}, "conflicts"),
        ({"save_replays": True}, "integer"),
        ({"num_episodes": True}, "positive integer"),
    ],
)
def test_bad_conditions_reject_before_output(
    options: dict[str, Any], match: str, tmp_path: Path
) -> None:
    values: dict[str, Any] = {"num_episodes": 2, "maps": [12], **options}
    with pytest.raises(ValueError, match=match):
        evaluate("random", "random", output_dir=tmp_path / "bad", **values)
    assert not list(tmp_path.iterdir())


def test_single_authored_spec_is_exact_and_needs_no_counterpart() -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12, max_steps=1, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    initial = core.reset(config, jax.random.key(4))[0]
    spec = EpisodeSpec(7, config, 12, 8, initial)
    declarations, _, _ = prepare_schedule([spec])
    assert declarations[7]["comparison_kind"] == "unpaired"
    result = evaluate_episodes(
        "random", "random", [spec], metrics="none", num_envs=1, chunk_size=1
    )
    assert [r.episode_id for r in result.episodes] == [7]
    assert result.episodes[0].episode_length == 1


def test_authored_pair_is_custom_and_bad_reserved_claims_fail() -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12, max_steps=1, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    initial = core.reset(config, jax.random.key(4))[0]
    specs = [
        EpisodeSpec(i, config, 12, 8, initial, paired_comparison_key="custom")
        for i in (1, 2)
    ]
    declarations, _, _ = prepare_schedule(specs)
    assert {r["comparison_kind"] for r in declarations.values()} == {"custom"}
    with pytest.raises(ValueError, match="exactly two"):
        prepare_schedule(specs[:1])
    with pytest.raises(ValueError, match="conflicts"):
        prepare_schedule([EpisodeSpec(1, config, metadata={"spawn_locations": 1})])


def test_config_roundtrip_keeps_exact_content_identity() -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12, max_steps=1, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    identifier, content = config_record(config)
    assert config_record(restore_config(content))[0] == identifier


def test_unplayed_source_choices_remain_ordered_saved_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = runner.capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(runner, "capture_recording_provenance", same_source)
    result = evaluate(
        "random",
        "random",
        num_episodes=2,
        maps=[12, 13, 13],
        max_steps=1,
        metrics="none",
        num_envs=2,
        output_dir=tmp_path,
    )
    assert result.run_dir is not None
    choices = result.metadata["evaluation_contract"]["source_choices"]
    assert [choice["map_id"] for choice in choices] == [12, 13, 13]
    assert choices[1] == choices[2]
    assert all(
        choice["source_config_id"] in result.metadata["configurations"]
        for choice in choices
    )
    resumed = evaluate("random", "random", num_episodes=2, resume_from=result.run_dir)
    assert resumed.metadata["evaluation_contract"]["source_choices"] == choices
    before = {
        p.relative_to(result.run_dir): p.read_bytes()
        for p in result.run_dir.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="source choices"):
        evaluate(
            "random",
            "random",
            num_episodes=2,
            maps=[12, 14, 13],
            resume_from=result.run_dir,
        )
    assert before == {
        p.relative_to(result.run_dir): p.read_bytes()
        for p in result.run_dir.rglob("*")
        if p.is_file()
    }


def test_in_memory_run_id_cannot_create_a_new_saved_run(tmp_path: Path) -> None:
    configuration = make_standard_team_deathmatch_config(
        map_id=12, max_steps=1, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    with pytest.raises(ValueError, match="new runs allocate their own ID"):
        evaluate_episodes(
            "random",
            "random",
            [EpisodeSpec(1, configuration)],
            output_dir=tmp_path / "bad",
            run_id="custom",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("claim", ["verified_spawn_pair", "unpaired", "custom"])
def test_reserved_comparison_claim_matches_final_verified_classification(
    claim: str,
) -> None:
    source = make_standard_team_deathmatch_config(
        map_id=12, max_steps=1, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    specs = [
        EpisodeSpec(
            i + 1,
            resolved,
            12,
            5,
            source_config=source,
            spawn_locations=i,
            paired_comparison_key="pair",
            metadata={"comparison_kind": claim},
        )
        for i, resolved in enumerate((source, _swap_spawn_banks(source)))
    ]
    if claim == "verified_spawn_pair":
        declarations, _, _ = prepare_schedule(specs)
        assert declarations[1]["comparison_kind"] == claim
    else:
        with pytest.raises(ValueError, match=r"conflicts.*comparison_kind"):
            prepare_schedule(specs)


def test_saved_first_resume_ignores_catalog_and_rejects_explicit_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = runner.capture_recording_provenance()

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(runner, "capture_recording_provenance", same_source)
    result = evaluate(
        "random",
        "random",
        num_episodes=2,
        maps=[12],
        phase="validation-1",
        metrics="none",
        seed=9,
        max_steps=1,
        num_envs=2,
        chunk_size=1,
        output_dir=tmp_path,
    )
    assert result.run_dir is not None
    before = {
        p.relative_to(result.run_dir): p.read_bytes()
        for p in result.run_dir.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="seed differs"):
        evaluate(
            "random",
            "random",
            num_episodes=2,
            phase="validation-1",
            resume_from=result.run_dir,
            seed=0,
        )
    after = {
        p.relative_to(result.run_dir): p.read_bytes()
        for p in result.run_dir.rglob("*")
        if p.is_file()
    }
    assert before == after

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("saved resolution consulted current maps")

    monkeypatch.setattr(runner, "normalize_episode_specs", unavailable)
    resumed = evaluate(
        "random",
        "random",
        num_episodes=2,
        phase="validation-1",
        resume_from=result.run_dir,
        num_envs=1,
        chunk_size=3,
    )
    assert resumed.status == "complete" and resumed.episodes == ()
    np.testing.assert_array_equal(
        result.table("episodes")["episode_id"], resumed.table("episodes")["episode_id"]
    )


def test_adapter_roster_failure_does_not_open_writer(tmp_path: Path) -> None:
    system = independent_policies([policy("random")])
    with pytest.raises(ValueError, match="roster"):
        evaluate(
            system,
            "random",
            num_episodes=2,
            maps=[12],
            max_steps=1,
            output_dir=tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_pinned_source_map_keeps_approved_historical_geometry() -> None:
    import jax.numpy as jnp

    from marl_battlegrounds._tdm_assets import map_history
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec
    from marl_battlegrounds.evaluation.evaluation_conditions import prepare_schedule
    from marl_battlegrounds.tasks import (
        make_canonical_team_deathmatch_evaluation_config,
    )

    old = next(row for row in map_history() if row.info.map_id == 47)
    source = make_canonical_team_deathmatch_evaluation_config(map_id=old.info.map_id)
    source = source._replace(
        map_width=old.geometry.map_width,
        map_height=old.geometry.map_height,
        obstacles=jnp.asarray(old.geometry.obstacles, dtype=jnp.float32),
        team_spawn_pad_positions=jnp.asarray(
            old.geometry.team_spawn_pad_positions, dtype=jnp.float32
        ),
    )
    spec = EpisodeSpec(1, source, old.info.map_id, 1, source_config=source)
    declarations, _, _ = prepare_schedule(
        [spec], registered_maps={old.info.map_id: old.info.model_dump(mode="json")}
    )
    metadata = {
        row["name"]: row["value"]
        for row in cast(list[dict[str, str]], declarations[1]["map_metadata"])
    }
    assert metadata["map_name"] == old.info.name
    bad = old.info.model_dump(mode="json")
    bad["name"] = "forged"
    with pytest.raises(ValueError, match="snapshot map identity"):
        prepare_schedule([spec], registered_maps={old.info.map_id: bad})


def _pre_red_zone_configs() -> dict[str, dict[str, Any]]:
    details = json.loads(_PRE_RED_ZONE_RUN_DETAILS.read_bytes())
    return cast(dict[str, dict[str, Any]], details["configurations"])


def _saved_pass(
    identifier: str, content: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"configurations": {identifier: content}},
        {
            "details": {"episode_ids": [1]},
            "episodes": {"1": {"configuration_digest": identifier}},
        },
    )


def test_pre_red_zone_configs_restore_at_depth_zero_under_their_recorded_ids() -> None:
    legacy = _pre_red_zone_configs()
    assert len(legacy) == 4
    for identifier, content in legacy.items():
        assert len(content) == 12 and "team_deathmatch_red_zone_depth" not in content
        config, historical = restore_recorded_config(content, identifier)
        assert historical is True
        assert config.team_deathmatch_red_zone_depth == 0.0
        assert restore_config(content).team_deathmatch_red_zone_depth == 0.0
        assert config_record(config, historical=True) == (identifier, content)
        current_id, current_content = config_record(config)
        assert current_id != identifier
        assert current_content == {**content, "team_deathmatch_red_zone_depth": 0.0}
        assert restore_recorded_config(current_content, current_id)[1] is False
        (spec,) = saved_specs(_saved_pass(current_id, current_content))
        assert spec.env_config.team_deathmatch_red_zone_depth == 0.0
        with pytest.raises(ValueError, match="saved before the Red Zone rule"):
            saved_specs(_saved_pass(identifier, content))


def test_recorded_identity_is_checked_first_and_positive_depths_have_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier, content = next(iter(_pre_red_zone_configs().items()))

    def restored_too_early(*args: object, **kwargs: object) -> None:
        raise AssertionError("content was restored before its identity check")

    with monkeypatch.context() as patch:
        patch.setattr(evaluation_conditions, "restore_config", restored_too_early)
        for changed in (
            {**content, "max_steps": content["max_steps"] + 1},
            {**content, "team_deathmatch_red_zone_depth": 0.0},
            {**content, "team_deathmatch_red_zone_depth": 5.0},
        ):
            with pytest.raises(ValueError, match="differs from its recorded identity"):
                restore_recorded_config(changed, identifier)

    config = make_standard_team_deathmatch_config(
        map_id=12, max_steps=1, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    assert config.team_deathmatch_red_zone_depth == 5.0
    for depth in (5.0, -0.0):
        with pytest.raises(ValueError, match=r"needs Red Zone depth \+0\.0"):
            config_record(
                config._replace(team_deathmatch_red_zone_depth=depth), historical=True
            )
    with pytest.raises(ValueError, match="cannot record a Red Zone depth"):
        build_resolved_env_config_v1(config)
    identifier, content = config_record(config)
    assert content["team_deathmatch_red_zone_depth"] == 5.0
    assert restore_recorded_config(content, identifier)[1] is False


def test_option_reads_a_pass_saved_before_an_option_as_its_missing_value() -> None:
    # Without missing, a saved pass lacking the key reads as the default and
    # accepts any explicit value (the older rule, kept for other options).
    assert option(OMITTED, None, "depth", 5.0) == 5.0
    assert option(OMITTED, {}, "depth", 5.0) == 5.0
    assert option(7.0, {}, "depth", 5.0) == 7.0
    # With missing, the lacking key reads as missing and explicit values must
    # equal it.
    assert option(OMITTED, None, "depth", 5.0, missing=0.0) == 5.0
    assert option(OMITTED, {}, "depth", 5.0, missing=0.0) == 0.0
    assert option(0.0, {}, "depth", 5.0, missing=0.0) == 0.0
    with pytest.raises(
        ValueError, match=r"^depth differs from the saved evaluation conditions$"
    ):
        option(5.0, {}, "depth", 5.0, missing=0.0)
    assert option(OMITTED, {"depth": 6.0}, "depth", 5.0, missing=0.0) == 6.0
    assert option(6.0, {"depth": 6.0}, "depth", 5.0, missing=0.0) == 6.0
    with pytest.raises(ValueError, match="differs from the saved"):
        option(0.0, {"depth": 6.0}, "depth", 5.0, missing=0.0)


def test_red_zone_depth_option_defaults_inherits_and_compares_in_float32() -> None:
    assert red_zone_depth_option(OMITTED, None) == 5.0
    assert red_zone_depth_option(OMITTED, {"seed": 0}) == 0.0
    assert red_zone_depth_option(OMITTED, {"red_zone_depth": 6.0}) == 6.0
    assert red_zone_depth_option(6.0, None) == 6.0
    # The same float32 value is the same rule, so the saved value comes back.
    assert red_zone_depth_option(5.0000001, {"red_zone_depth": 5.0}) == 5.0
    assert red_zone_depth_option(0.0, {"seed": 0}) == 0.0
    # Python's == calls -0.0 and 0.0 equal, but their float32 bytes differ.
    for supplied, saved in (
        (5.0, {"seed": 0}),
        (6.0, {"red_zone_depth": 5.0}),
        (-0.0, {"red_zone_depth": 0.0}),
        (-0.0, {"seed": 0}),
    ):
        with pytest.raises(
            ValueError,
            match=r"^red_zone_depth differs from the saved evaluation conditions$",
        ):
            red_zone_depth_option(supplied, saved)
    for wrong in (6, True, np.float64(6.0)):
        with pytest.raises(TypeError, match="red_zone_depth must be a Python float"):
            red_zone_depth_option(cast(Any, wrong), None)
    assert same_float32(5.0, 5.0000001)
    assert not same_float32(0.0, -0.0)
    assert not same_float32(5.0, 5.001)


def test_evaluate_records_the_depth_and_never_ignores_a_supplied_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identities: dict[float, set[str]] = {}
    depths: tuple[float | Omitted, ...] = (OMITTED, 0.0, 5.0, 6.0)
    for depth in depths:
        resolved = _resolved(
            monkeypatch,
            num_episodes=2,
            maps=[12],
            max_steps=1,
            red_zone_depth=depth,
        )
        expected = 5.0 if isinstance(depth, Omitted) else depth
        assert resolved["contract"]["options"]["red_zone_depth"] == expected
        assert {
            spec.env_config.team_deathmatch_red_zone_depth for spec in resolved["specs"]
        } == {expected}
        found = {config_record(spec.env_config)[0] for spec in resolved["specs"]}
        assert identities.setdefault(expected, found) == found
    # Each depth is a different rule, so it has different configuration IDs.
    assert not identities[0.0] & identities[5.0]
    assert not identities[6.0] & identities[5.0]
    source = make_standard_team_deathmatch_config(
        map_id=12,
        max_steps=1,
        team_a_roster=("mage",),
        team_b_roster=("priest",),
        red_zone_depth=0.0,
    )
    # Omitted, the exact source keeps its own depth; an equal value is accepted.
    for depth in depths[:2]:
        specs = _resolved(
            monkeypatch, num_episodes=2, maps=[source], red_zone_depth=depth
        )["specs"]
        assert {s.env_config.team_deathmatch_red_zone_depth for s in specs} == {0.0}
    for depth in (5.0, 6.0):
        with pytest.raises(
            ValueError,
            match=r"^red_zone_depth conflicts with the exact source configuration$",
        ):
            evaluate(
                "random",
                "random",
                num_episodes=2,
                maps=[source],
                red_zone_depth=depth,
                output_dir=tmp_path / "conflict",
            )
    with pytest.raises(TypeError, match="red_zone_depth must be a Python float"):
        evaluate(
            "random",
            "random",
            num_episodes=2,
            maps=[12],
            red_zone_depth=cast(Any, 6),
            output_dir=tmp_path / "integer",
        )
    assert not list(tmp_path.iterdir())


def _pre_red_zone_manifest(
    identifier: str,
    content: dict[str, Any],
    contract_options: dict[str, Any] | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {"episode_ids": [1], "num_episodes": 1}
    if contract_options is not None:
        details["evaluation_contract"] = {
            "version": 1,
            "schedule_kind": "generated",
            "options": contract_options,
            "episode_ids": [1],
        }
    return {
        "schema_version": 2,
        "configurations": {identifier: content},
        "passes": {
            '["evaluation","1"]': {
                "phase": "evaluation",
                "pass_id": "1",
                "details": details,
                "episodes": {"1": {"configuration_digest": identifier}},
            }
        },
    }


def test_public_evaluate_refuses_to_resume_a_pass_saved_before_the_red_zone_rule(
    tmp_path: Path,
) -> None:
    identifier, content = next(iter(_pre_red_zone_configs().items()))
    current_id, current_content = config_record(restore_config(content))
    old_options: dict[str, Any] = {"seed": 0, "spawn_mode": "default"}
    saved_passes = {
        # The old evaluator saved no contract; its content has 12 keys.
        "legacy": _pre_red_zone_manifest(identifier, content, None),
        # A generated pass from before the rule: no depth option, 12 keys.
        "contract": _pre_red_zone_manifest(identifier, content, old_options),
        # Each sign alone marks a pass saved before the rule.
        "no-depth-option": _pre_red_zone_manifest(
            current_id, current_content, old_options
        ),
        "old-content": _pre_red_zone_manifest(
            identifier, content, {**old_options, "red_zone_depth": 0.0}
        ),
    }
    message = (
        r"^This pass was saved before the Red Zone rule\. Its results stay "
        r"readable; resuming it needs the source version that recorded it\.$"
    )
    depths: tuple[float | Omitted, ...] = (OMITTED, 0.0, 5.0)
    for name, manifest in saved_passes.items():
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "run_details.json").write_text(json.dumps(manifest))
        before = (run_dir / "run_details.json").read_bytes()
        # The same clear refusal with or without maps and red_zone_depth.
        for maps in (None, [12]):
            for depth in depths:
                with pytest.raises(ValueError, match=message):
                    evaluate(
                        "random",
                        "random",
                        num_episodes=1,
                        maps=maps,
                        resume_from=run_dir,
                        red_zone_depth=depth,
                    )
        assert [path.name for path in run_dir.iterdir()] == ["run_details.json"]
        assert (run_dir / "run_details.json").read_bytes() == before
