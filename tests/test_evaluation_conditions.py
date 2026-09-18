"""Check paired evaluation resolution and exact saved scientific conditions.

These CPU tests exercise schedules without running games wherever possible. The
small saved cases check current state, exact outcomes, resumed configuration
identity and rejection before any run/pass mutation. Authored single episodes
remain valid and are never expanded into undeclared comparisons.
"""

# Host setup tests intentionally inspect the private resolver boundary.
# pyright: reportPrivateUsage=false

from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

from marl_battlegrounds.core import env as core
from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    evaluate,
    evaluate_episodes,
)
from marl_battlegrounds.evaluation.evaluation_conditions import (
    config_record,
    default_maps,
    prepare_schedule,
    restore_config,
)
from marl_battlegrounds.evaluation.policy_execution import independent_policies, policy
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    _swap_spawn_banks,
    list_tdm_maps,
    make_standard_team_deathmatch_config,
)

runner = import_module("marl_battlegrounds.evaluation.evaluate")


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
