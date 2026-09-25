"""Check replay metric analysis and its read-only result boundaries.

Analysis must use the shared numerical metric definitions and preserve missing
values instead of inventing measurements. The roster and targeting checks use
historical V1 contexts, so their map configs set Red Zone depth 0.0 (the V1
record has no depth field). The roster checks also cover the Red Zone columns:
an agent's Red Zone kill contributions and kill participation show exactly when
its kill contributions do (its team has an agent who can deal damage), its Red
Zone deaths and death share always show while it is active, and a team's Red
Zone kills show exactly when the team has an agent who can deal damage while
its Red Zone deaths always show.
"""

import csv
import io
from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from tests.evaluation_fixtures import captured_evaluation_trajectory, evaluation_context
from tests.test_evaluation_replay import runtime_provenance as runtime_provenance

from marl_battlegrounds.core.types import ActionMask, Info, Reward
from marl_battlegrounds.evaluation import analysis as analysis_module
from marl_battlegrounds.evaluation.analysis import (
    REPLAY_IDENTITY_COLUMNS,
    ReplayAnalysis,
    analyze_replay,
)
from marl_battlegrounds.evaluation.capture import (
    reconstruct_env_state_v1,
    reconstruct_transition_facts_v1,
)
from marl_battlegrounds.evaluation.catalog import reconstruct_env_config_v1
from marl_battlegrounds.evaluation.episode_metrics import (
    initialize_priority,
    priority_values,
    update_priority,
)
from marl_battlegrounds.evaluation.full_metrics import (
    full_values,
    initialize_full,
    update_full,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    METRIC_COLUMNS,
    METRIC_COLUMNS_BY_NAME,
    METRIC_SCHEMA_VERSION,
    METRIC_TOPICS,
    METRIC_VIEW_LABELS,
    PRIORITY_METRIC_COLUMNS,
)
from marl_battlegrounds.evaluation.metrics import (
    EvaluationEpisodeCompletionV1,
    EvaluationEpisodeObserverV1,
)
from marl_battlegrounds.evaluation.replay import (
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplay, LoadedReplayBundleV1
from marl_battlegrounds.evaluation.replay_v2 import build_replay_v2
from marl_battlegrounds.tasks import (
    AgentClassName,
    make_standard_team_deathmatch_config,
)


def test_search_catalog_uses_recorded_rosters_without_numerical_analysis() -> None:
    rosters: tuple[tuple[AgentClassName, ...], ...] = (
        ("mage", "warrior", "hunter", "rogue", "priest"),
        ("priest", "rogue", "mage", "warrior", "hunter"),
        ("priest", "priest", "warrior"),
        ("mage",),
    )
    for team_a, team_b in zip(rosters, reversed(rosters), strict=True):
        context = evaluation_context(
            config=make_standard_team_deathmatch_config(
                map_id=48,
                team_a_roster=team_a,
                team_b_roster=team_b,
                red_zone_depth=0.0,
            )
        )
        completion = EvaluationEpisodeCompletionV1(
            episode_id=context.identity.episode_id,
            completion_state="partial",
            expected_transition_count=1,
            validated_transition_count=0,
            last_valid_frame_index=0,
            last_valid_frame_id=f"{context.identity.episode_id}:frame:0",
            terminated=False,
            truncated=False,
            completion_bases=(),
            end_or_failure_reason="Catalog-only test before any transition.",
        )
        analysis = ReplayAnalysis(
            "source",
            "analysis",
            "not_recorded",
            context,
            completion,
            METRIC_COLUMNS,
            (),
            np.empty((0, len(METRIC_COLUMNS)), np.float32),
            np.empty((0, len(METRIC_COLUMNS)), np.bool_),
        )
        catalog = analysis.catalog()
        assert analysis.frame_count == 0
        assert catalog["class_names"] == (
            context.static_mechanics_catalog.class_name_by_id
        )
        agents = cast(list[dict[str, object]], catalog["agents"])
        assert len(agents) == 10
        for slot, (row, agent) in enumerate(zip(agents, context.roster, strict=True)):
            assert row == {
                "slot": slot,
                "class_id": agent.class_id,
                "class_name": context.static_mechanics_catalog.class_name_by_id[
                    agent.class_id
                ],
                "team_id": 1 if slot < 5 else 2,
                "active": agent.configured_active,
            }
        rows = cast(list[dict[str, object]], catalog["measurements"])
        assert len(rows) == len(METRIC_COLUMNS)
        for row in rows:
            assert "value" not in row and "valid" not in row
            assert isinstance(row["search_facts"], dict)
        by_name = {str(row["name"]): row for row in rows}
        # Slot identity must survive an impossible source class or inactive slot.
        assert by_name["agent_0_to_agent_1_healing_done"]["subjects"] == (0, 1)
        for slot in range(10):
            assert (
                by_name[f"agent_{slot}_deaths"]["applicable"] == agents[slot]["active"]
            )


def test_replay_class_filters_keep_helpers_and_recipients_distinct() -> None:
    # These are roster checks, not new numerical episodes. The public replay
    # tests below also check the resulting tables, search data and CSV values.
    teams: tuple[tuple[AgentClassName, ...], ...] = (
        ("mage", "warrior", "hunter", "rogue", "priest"),
        ("priest", "rogue", "mage", "warrior", "hunter"),
        ("priest", "priest", "mage", "warrior", "warrior"),
        ("priest", "priest", "priest", "priest", "priest"),
        ("mage", "mage", "mage", "mage", "mage"),
        ("priest",),
        ("mage",),
        ("warrior", "priest"),
    )
    for team_a, team_b in zip(teams, reversed(teams), strict=True):
        context = evaluation_context(
            config=make_standard_team_deathmatch_config(
                map_id=48,
                team_a_roster=team_a,
                team_b_roster=team_b,
                red_zone_depth=0.0,
            )
        )
        analysis = object.__new__(ReplayAnalysis)
        object.__setattr__(analysis, "context", context)

        def shown(name: str, replay_analysis: ReplayAnalysis = analysis) -> bool:
            # Check only roster rules here; public table/CSV checks follow below.
            reason = replay_analysis._not_applicable_reason(  # pyright: ignore[reportPrivateUsage]
                METRIC_COLUMNS_BY_NAME[name]
            )
            return reason is None

        active = tuple(agent for agent in context.roster if agent.configured_active)
        for agent in active:
            slot, team = agent.global_slot, agent.configured_team_id
            allies = tuple(a for a in active if a.configured_team_id == team)
            enemies = tuple(a for a in active if a.configured_team_id != team)
            healer = any(a.class_id == 5 for a in allies)
            ally_damage = any(a.class_id != 5 for a in allies)
            enemy_damage = any(a.class_id != 5 for a in enemies)
            for ability in ("", "basic_", "ultimate_"):
                # A patient may be any class, but only a Priest supplies healing.
                assert shown(f"agent_{slot}_{ability}rescue_contributions") == (
                    agent.class_id == 5 and enemy_damage
                )
                for ally in allies:
                    assert shown(
                        f"agent_{slot}_to_agent_{ally.global_slot}_"
                        f"{ability}rescue_contributions"
                    ) == (agent.class_id == 5 and enemy_damage)
                # The healer's class controls output, not the patient's class.
                assert shown(f"agent_{slot}_{ability}damage_received") == any(
                    a.class_id
                    in ((2, 3, 4) if ability == "ultimate_" else (1, 2, 3, 4))
                    for a in enemies
                )
                assert shown(f"agent_{slot}_{ability}healing_done") == (
                    agent.class_id == 5
                )
                assert shown(f"agent_{slot}_{ability}effective_healing_done") == (
                    agent.class_id == 5
                )
                assert shown(f"agent_{slot}_{ability}excess_healing") == (
                    agent.class_id == 5
                )
                assert shown(f"agent_{slot}_{ability}damage_done") == (
                    agent.class_id
                    in ((2, 3, 4) if ability == "ultimate_" else (1, 2, 3, 4))
                )
                for portion in ("effective", "excess"):
                    assert shown(
                        f"agent_{slot}_{ability}{portion}_healing_fraction"
                    ) == (agent.class_id == 5)
            assert shown(f"agent_{slot}_rescues") == (healer and enemy_damage)
            assert shown(f"agent_{slot}_rescue_opportunities") == (
                healer and enemy_damage
            )
            # A Trap may be present in the initial frame without a Hunter.
            assert shown(f"agent_{slot}_trap_intervals")
            for stem in ("trap_break_rate", "trap_mean_remaining_steps_at_break"):
                assert shown(f"agent_{slot}_{stem}") == enemy_damage
            assert shown(f"agent_{slot}_solo_kills") == (agent.class_id != 5)
            assert shown(f"agent_{slot}_trap_break_contributions") == (
                agent.class_id != 5
            )
            assert shown(f"agent_{slot}_ultimate_kill_contributions") == (
                agent.class_id in (2, 3, 4) or (agent.class_id == 5 and ally_damage)
            )
            assert shown(f"agent_{slot}_kill_contributions") == ally_damage
            # Red Zone kill help follows ordinary kill help; Red Zone deaths
            # belong to their victims.
            for stem in ("red_zone_kill_contributions", "red_zone_kill_participation"):
                assert shown(f"agent_{slot}_{stem}") == ally_damage
            for stem in ("red_zone_deaths", "red_zone_death_fraction"):
                assert shown(f"agent_{slot}_{stem}")
            assert shown(f"agent_{slot}_burst_damage_received") == any(
                a.class_id == 1 for a in enemies
            )
            assert shown(f"agent_{slot}_healing_prevented_by_poison") == healer
            assert shown(f"agent_{slot}_priest_healing_received") == healer
            for measure in (
                "healing_received",
                "effective_healing_received",
                "regenerated_healing",
            ):
                assert shown(f"agent_{slot}_{measure}")
            assert shown(f"agent_{slot}_healing_received_while_hunter_trap") == any(
                a.class_id == 5 and a.global_slot != slot for a in allies
            )
            for aura, class_id in (("mage", 1), ("warrior", 2)):
                assert shown(
                    f"agent_{slot}_{aura}_aura_covered_recipient_steps"
                ) == any(a.class_id == class_id for a in allies)
            # These effects may already exist when recording starts. Do not
            # require the class that originally applied them to still be here.
            for effect in ("hunter_trap", "rogue_poison_stun", "priest_freedom"):
                assert shown(f"agent_{slot}_{effect}_active_steps")
            for effect in ("hunter_trap", "rogue_poison_stun", "hunter_basic_slow"):
                assert (
                    shown(f"agent_{slot}_damage_received_while_{effect}")
                    == enemy_damage
                )
            for enemy in enemies:
                for stem, expected in (
                    ("solo_kills", agent.class_id != 5),
                    ("trap_break_contributions", agent.class_id != 5),
                    ("kill_contributions", ally_damage),
                    ("kill_contributions_to_hunter_trap_recipient", ally_damage),
                ):
                    assert (
                        shown(f"agent_{slot}_to_agent_{enemy.global_slot}_{stem}")
                        == expected
                    )
        for team, members in (
            ("a", tuple(a for a in active if a.configured_team_id == 1)),
            ("b", tuple(a for a in active if a.configured_team_id == 2)),
        ):
            damagers = sum(a.class_id != 5 for a in members)
            priests = sum(a.class_id == 5 for a in members)
            assert shown(f"team_{team}_healing_done") == bool(priests)
            assert shown(f"team_{team}_damage_done") == bool(damagers)
            assert shown(f"team_{team}_focus_fire_steps") == (damagers >= 2)
            assert shown(f"team_{team}_multi_contributor_kills") == (
                damagers >= 2 or (damagers >= 1 and priests >= 1)
            )
            assert shown(f"team_{team}_ally_distance_mean") == (len(members) >= 2)
            assert shown(f"team_{team}_trap_intervals")
            assert shown(f"team_{team}_trap_breaks") == bool(damagers)
            assert shown(f"team_{team}_red_zone_kills") == bool(damagers)
            assert shown(f"team_{team}_red_zone_deaths")
            # Match results stay available even when an event cannot happen.
            for measure in ("kills", "deaths", "score", "return"):
                assert shown(f"team_{team}_{measure}")
        for agent in context.roster:
            if not agent.configured_active:
                assert not shown(f"agent_{agent.global_slot}_rescues")
                for stem in (
                    "trap_intervals",
                    "trap_break_rate",
                    "trap_mean_remaining_steps_at_break",
                    "rescue_opportunities",
                    "red_zone_kill_contributions",
                    "red_zone_deaths",
                ):
                    assert not shown(f"agent_{agent.global_slot}_{stem}")


@pytest.mark.parametrize(
    ("team_a", "team_b", "mage_slots", "slot_zero_targets"),
    (
        (("warrior", "mage", "mage"), ("priest", "mage"), (1, 2, 6), (5, 6)),
        (("priest",), ("mage",), (5,), (0,)),
    ),
)
def test_replay_ultimate_targets_follow_recorded_classes_not_slot_numbers(
    runtime_provenance: RuntimeProvenanceV1,
    team_a: tuple[AgentClassName, ...],
    team_b: tuple[AgentClassName, ...],
    mage_slots: tuple[int, ...],
    slot_zero_targets: tuple[int, ...],
) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=48,
        team_a_roster=team_a,
        team_b_roster=team_b,
        max_steps=1,
        red_zone_depth=0.0,
    )
    trajectory = captured_evaluation_trajectory(
        config=config, transition_count=1, expected_horizon=1
    )
    replay = build_replay_v2(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=runtime_provenance,
    )
    analysis = analyze_replay(LoadedReplay(replay, None, "not_recorded"), full=True)
    rows = cast(list[dict[str, object]], analysis.summary(0)["statistics"])
    by_name = {row["name"]: row for row in rows}
    search_rows = {
        row["name"]: row
        for row in cast(list[dict[str, object]], analysis.catalog()["measurements"])
    }
    for agent in trajectory.context.roster:
        if not agent.configured_active:
            continue
        for ability in ("", "basic_", "ultimate_"):
            name = f"agent_{agent.global_slot}_{ability}damage_received"
            row = by_name[name]
            assert row["direction"] == (
                "descriptive" if agent.class_id == 2 else "lower"
            )
            assert row["guidance"] == search_rows[name]["guidance"]
            assert ("Warrior" in str(row["guidance"])) == (agent.class_id == 2)
    targeting = {
        cast(tuple[int, int], row["subjects"]): row
        for row in rows
        if row["scope"] == "source_recipient" and row["stem"] == "ultimate_applications"
    }
    assert len(targeting) == 100  # Fixed exported pairs survive roster changes.
    for source in mage_slots:
        assert {
            recipient
            for (actor, recipient), row in targeting.items()
            if actor == source and row["applicable"]
        } == {source}
        assert f"Agent ID {source} · Mage" in str(targeting[source, source]["subject"])
    assert {
        recipient
        for (actor, recipient), row in targeting.items()
        if actor == 0 and row["applicable"]
    } == set(slot_zero_targets)
    assert (
        "Mage" not in str(targeting[0, slot_zero_targets[0]]["subject"]).split(" → ")[0]
    )
    exported = next(csv.DictReader(io.StringIO(analysis.csv(0))))
    assert exported["agent_0_class_id"] == str(trajectory.context.roster[0].class_id)
    numeric_names = tuple(column.name for column in METRIC_COLUMNS)
    numeric_set = set(numeric_names)
    assert tuple(name for name in exported if name in numeric_set) == numeric_names
    # Full navigation keeps its complete menu even when a class has no users.
    # Only the metadata context changes; these checks do not read scalar values.
    for active in (True, False):
        roster = tuple(
            agent.model_copy(
                update={
                    "class_id": 2 if agent.global_slot % 2 else 1,
                    "configured_active": active,
                    "configured_team_id": 1 + agent.global_slot // 5,
                }
            )
            for agent in analysis.context.roster
        )
        metadata_analysis = replace(
            analysis, context=analysis.context.model_copy(update={"roster": roster})
        )
        catalog = metadata_analysis.catalog()
        topics = cast(tuple[dict[str, object], ...], catalog["topics"])
        assert len(topics) == 28
        assert sum(len(cast(list[object], topic["views"])) for topic in topics) == 44
        assert any(topic["name"] == "ultimate_priest" for topic in topics)
        definitions = {
            row["name"]: row
            for row in cast(list[dict[str, object]], catalog["measurements"])
        }
        for slot in range(10):
            row = definitions[f"agent_{slot}_damage_received"]
            assert row["direction"] == ("descriptive" if slot % 2 else "lower")
            assert row["applicable"] == active


@pytest.mark.parametrize("version", [1, 2])
def test_replay_scalar_prefixes_match_direct_metrics_and_wide_csv(
    runtime_provenance: RuntimeProvenanceV1,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    trajectory = captured_evaluation_trajectory(transition_count=3, expected_horizon=3)
    if version == 1:
        capture = EvaluationEpisodeObserverV1(trajectory.context)
        capture.start(trajectory.frames[0])
        for transition, frame in zip(
            trajectory.transitions, trajectory.frames[1:], strict=True
        ):
            capture.append(transition, frame)
        bundle = build_replay_bundle_v1(
            capture,
            capture.finalize(completion_state="complete"),
            runtime_provenance=runtime_provenance,
        )
        loaded = LoadedReplayBundleV1(
            bundle.replay, bundle.metric_report_artifact, "complete"
        )
    else:
        replay = build_replay_v2(
            trajectory.context,
            trajectory.frames,
            trajectory.transitions,
            runtime_provenance=runtime_provenance,
        )
        loaded = LoadedReplay(replay, None, "not_recorded")
    before = loaded.replay.model_dump_json()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "analysis must use captured arrays, without observer or simulation"
        )

    monkeypatch.setattr(EvaluationEpisodeObserverV1, "start", forbidden)
    monkeypatch.setattr("marl_battlegrounds.core.env.step", forbidden)
    monkeypatch.setattr("marl_battlegrounds.core.env.reset", forbidden)
    # Two blocks plus padding exercise the carry boundary, without a long replay.
    monkeypatch.setattr(analysis_module, "_BLOCK_SIZE", 2)
    analysis = analyze_replay(loaded, full=True)
    basic = analyze_replay(loaded)
    assert analysis.source_replay_digest == loaded.replay.canonical_digest_sha256
    assert loaded.replay.model_dump_json() == before
    assert analysis.frame_count == 4
    assert analysis.columns == METRIC_COLUMNS
    assert basic.columns == PRIORITY_METRIC_COLUMNS
    assert analysis.summary(0)["completion"] is None
    assert analysis.summary(0, scope="final")["frame_index"] == 3
    config = reconstruct_env_config_v1(trajectory.context)
    initial = reconstruct_env_state_v1(trajectory.frames[0])
    priority = initialize_priority()
    full = initialize_full(config, initial)
    for index, frame in enumerate(trajectory.frames):
        state = reconstruct_env_state_v1(frame)
        outcome = jnp.asarray(0, jnp.int32)
        if index:
            transition = trajectory.transitions[index - 1]
            start = trajectory.frames[index - 1]
            info = Info(reconstruct_transition_facts_v1(transition.facts))
            reward = Reward(
                jnp.asarray(transition.canonical_reward_by_agent, jnp.float32)
            )
            mask = ActionMask(
                *(
                    jnp.asarray(getattr(start.action_mask, name), bool)
                    for name in ActionMask._fields
                )
            )
            priority = update_priority(priority, reward, info)
            full = update_full(
                full, config, reconstruct_env_state_v1(start), mask, info
            )
            outcome = info.transition_facts.team_deathmatch_facts.outcome
        expected = full_values(
            full,
            config,
            priority_values(config, state, initial.step_count, priority, outcome),
        )
        summary = analysis.summary(index)
        topics = tuple(
            {
                "name": topic.name,
                "label": topic.label,
                "section": topic.section,
                "description": topic.description,
                "views": [
                    {"name": view, "label": METRIC_VIEW_LABELS[view]}
                    for view in (
                        ("totals", "recipients") if topic.paired else ("single",)
                    )
                ],
            }
            for topic in METRIC_TOPICS
        )
        assert summary["topics"] == topics
        assert basic.summary(index)["topics"] == topics[:1]
        assert "families" not in summary
        statistics = cast(list[dict[str, object]], summary["statistics"])
        csv_rows = list(csv.DictReader(io.StringIO(analysis.csv(index))))
        assert len(csv_rows) == 1
        row = csv_rows[0]
        assert tuple(row) == (
            *REPLAY_IDENTITY_COLUMNS,
            *(column.name for column in METRIC_COLUMNS),
        )
        assert row["frame_index"] == str(index)
        assert row["metric_schema_version"] == str(METRIC_SCHEMA_VERSION)
        assert row["scope"] == "cursor"
        for slot in range(10):
            assert row[f"agent_{slot}_class_id"] == str(
                trajectory.context.roster[slot].class_id
            )
        for column_index, column in enumerate(METRIC_COLUMNS):
            scalar = statistics[column_index]
            assert scalar["name"] == column.name
            assert scalar["order"] == column_index
            assert scalar["numerator"] == column.numerator
            assert scalar["denominator"] == column.denominator
            assert (scalar["numerator"] is None) == (scalar["denominator"] is None)
            locations = cast(list[dict[str, str]], scalar["locations"])
            assert {
                "topic": scalar["primary_topic"],
                "view": scalar["primary_view"],
            } in locations
            assert len(locations) == len({location["topic"] for location in locations})
            assert set(cast(dict[str, object], scalar["topic_order"])) == {
                location["topic"] for location in locations
            }
            assert "groups" not in scalar
            assert scalar["valid"] == bool(expected.valid[column_index])
            if scalar["valid"]:
                assert scalar["value"] == pytest.approx(
                    float(expected.values[column_index]), rel=1e-5, abs=1e-5
                )
                assert float(row[column.name]) == scalar["value"]
            else:
                assert row[column.name] == ""
                assert scalar["value"] is None
        by_name = {str(scalar["name"]): scalar for scalar in statistics}
        if index == 0:
            for name, topic, ability in (
                ("team_a_mage_burst_applications", "ultimate_mage", "Burst"),
                ("agent_1_ultimate_activations", "ultimate_warrior", "Charge"),
                ("agent_5_ultimate_activations", "ultimate_hunter", "Hunter Trap"),
                ("agent_6_ultimate_activations", "ultimate_rogue", "Rogue Poison"),
                ("agent_2_ultimate_activations", "ultimate_priest", "Salvation"),
            ):
                text = cast(dict[str, dict[str, str]], by_name[name]["topic_text"])
                assert ability in text[topic]["description"]
                assert by_name[name]["name"] == name
            priest = cast(
                dict[str, dict[str, str]],
                by_name["agent_2_ultimate_kill_contributions"]["topic_text"],
            )["ultimate_priest"]["description"]
            assert "useful" in priest
            assert "an ally who damages that enemy on its death tick" in priest
            assert "Mage" not in priest
            for name in (
                "agent_0_trap_intervals",
                "agent_0_trap_break_rate",
                "agent_0_trap_mean_remaining_steps_at_break",
                "agent_0_rescue_opportunities",
                "agent_2_rescue_opportunities",
            ):
                assert by_name[name]["applicable"] is True
                assert by_name[name]["subject_role"] == "recipient"
            # Team B has no Priest, although its Hunter and Rogue are valid
            # Trap recipients. Do not treat an enemy Priest as their healer.
            assert by_name["agent_5_trap_intervals"]["applicable"] is True
            assert by_name["agent_5_rescue_opportunities"]["applicable"] is False
        for slot, team in ((2, "Team A"), (5, "Team B")):
            denominator = str(
                by_name[f"agent_{slot}_healing_done_fraction"]["denominator"]
            )
            assert team in denominator
            assert "this team" not in denominator
        allocation = by_name["agent_0_to_agent_5_damage_done_allocation_fraction"]
        contribution = by_name["agent_0_to_agent_5_damage_done_fraction"]
        for fraction in (allocation, contribution):
            assert "this agent" in str(fraction["numerator"])
            assert "this target" in str(fraction["numerator"])
        assert allocation["denominator"] != contribution["denominator"]
        assert "Team A" in str(contribution["denominator"])
        # This real fixture is Mage/Warrior/Priest versus Hunter/Rogue. Structural
        # targeting remains meaningful even before any activation or denominator.
        for name in (
            "agent_0_to_agent_0_ultimate_applications",
            "agent_1_to_agent_5_ultimate_applications",
            "agent_2_to_agent_0_ultimate_applications",
            "agent_2_to_agent_2_ultimate_applications",
            "agent_5_to_agent_0_ultimate_applications",
        ):
            assert by_name[name]["applicable"] is True
        for name in (
            "agent_0_to_agent_5_ultimate_applications",
            "agent_1_to_agent_2_ultimate_applications",
            "agent_2_to_agent_5_ultimate_applications",
            "agent_3_deaths",
            "agent_0_to_agent_0_healing_done",
            "agent_2_to_agent_5_damage_done",
            "agent_0_to_agent_5_ultimate_damage_done",
        ):
            assert by_name[name]["applicable"] is False
        assert by_name["agent_2_to_agent_0_healing_done"]["applicable"] is True
        assert by_name["agent_0_to_agent_5_damage_done"]["applicable"] is True
        assert by_name["agent_0_to_agent_0_healing_done"]["value"] == 0
        assert by_name["agent_0_healing_done"]["applicable"] is False
        assert by_name["agent_0_healing_done"]["value"] == 0
        for team, expected_related in (("a", True), ("b", False)):
            locations = cast(
                list[dict[str, str]],
                by_name[f"team_{team}_ultimate_healing_done"]["locations"],
            )
            assert (
                any(location["topic"] == "ultimate_priest" for location in locations)
                is expected_related
            )
        assert by_name["team_b_ultimate_healing_done"]["applicable"] is False
        assert by_name["team_b_ultimate_healing_done"]["value"] == 0
        assert by_name["agent_0_healing_done"]["subject"] == (
            "Agent ID 0 · Mage · Team A"
        )
        # Scope and subject identities, rather than a CSV-name parser, tell the
        # tooltip which agent or team acted and which agent received the effect.
        source_pair = by_name["agent_2_to_agent_0_basic_applications"]
        assert source_pair["scope"] == "source_recipient"
        assert source_pair["subjects"] == (2, 0)
        assert source_pair["subject"] == (
            "Agent ID 2 · Priest · Team A → Agent ID 0 · Mage · Team A"
        )
        team_pair = by_name["team_a_to_agent_0_basic_applications"]
        assert team_pair["scope"] == "team_recipient"
        assert team_pair["subjects"] == (1, 0)
        assert team_pair["subject"] == "Team A → Agent ID 0 · Mage · Team A"
        recipient = by_name["agent_0_basic_priest_healing_received"]
        assert recipient["scope"] == "agent"
        assert recipient["subject_role"] == "recipient"
        allies = by_name["agent_0_and_agent_1_ally_distance_mean"]
        assert allies["scope"] == "ally_pair"
        assert allies["subject"] == (
            "Agent ID 0 · Mage · Team A ↔ Agent ID 1 · Warrior · Team A"
        )
        if index == 0:
            fraction = by_name["agent_0_to_agent_0_ultimate_application_fraction"]
            assert fraction["applicable"] is True
            assert fraction["valid"] is False
        for priority_row, full_row in zip(
            cast(list[dict[str, object]], basic.summary(index)["statistics"]),
            statistics[: len(PRIORITY_METRIC_COLUMNS)],
            strict=True,
        ):
            assert priority_row == {
                **{
                    key: value for key, value in full_row.items() if key != "topic_text"
                },
                "locations": [{"topic": "priority", "view": "single"}],
                "topic_order": {
                    "priority": cast(dict[str, object], full_row["topic_order"])[
                        "priority"
                    ]
                },
            }
            assert "topic_text" not in priority_row
    # Cached reads cannot collect more facts or mutate an earlier snapshot.
    catalog = analysis.catalog()
    measurements = cast(list[dict[str, object]], catalog["measurements"])
    assert [row["name"] for row in measurements] == [
        column.name for column in METRIC_COLUMNS
    ]
    assert catalog["source_replay_digest"] == analysis.source_replay_digest
    assert catalog["analysis_source_digest"] == analysis.analysis_source_digest
    assert catalog["topics"] == analysis.summary(0)["topics"]
    catalog_by_name = {str(row["name"]): row for row in measurements}
    summary_by_name = {
        str(row["name"]): row
        for row in cast(list[dict[str, object]], analysis.summary(0)["statistics"])
    }
    for row in measurements:
        assert "value" not in row and "valid" not in row
        assert isinstance(row["search_terms"], tuple)
        assert isinstance(row["search_facts"], dict)
        assert "search_facts" not in summary_by_name[str(row["name"])]
        assert (row["not_applicable_reason"] is None) is row["applicable"]
        assert row["applicable"] == summary_by_name[str(row["name"])]["applicable"]
        text = cast(dict[str, dict[str, str]], row.get("topic_text", {}))
        assert text == summary_by_name[str(row["name"])].get("topic_text", {})
        assert set(text) <= {
            location["topic"]
            for location in cast(list[dict[str, str]], row["locations"])
        }
        for topic, variant in text.items():
            allowed = (
                {
                    "label",
                    "description",
                    "subtitle",
                    "numerator",
                    "denominator",
                    "guidance",
                    "missing_when",
                }
                if topic
                in {
                    "ultimate_mage",
                    "ultimate_warrior",
                    "ultimate_hunter",
                    "ultimate_rogue",
                    "ultimate_priest",
                }
                else {"description", "subtitle"}
            )
            assert variant and set(variant) <= allowed
            for field, value in variant.items():
                assert (field == "subtitle" and value == "") or (
                    value and value[0].isupper()
                )
    assert catalog_by_name["agent_3_deaths"]["not_applicable_reason"] == (
        "Agent ID 3 is inactive in this replay."
    )
    for name, phrase, included in (
        ("agent_0_regenerated_healing", "regeneration", True),
        ("agent_0_priest_healing_received", "regeneration", False),
        ("agent_0_hunter_trap_active_steps", "stun duration", True),
        ("agent_0_hunter_trap_applications", "stun applications", True),
        ("agent_0_priest_freedom_active_steps", "stun duration", False),
        ("agent_0_mage_burst_active_steps", "stun duration", False),
        ("team_a_damage_prevented_by_warrior_aura", "damage reduction", True),
        ("team_a_damage_from_mage_aura", "damage reduction", False),
    ):
        terms = cast(tuple[str, ...], catalog_by_name[name]["search_terms"])
        assert (phrase in terms) is included
    assert catalog_by_name["agent_0_to_agent_5_ultimate_applications"][
        "not_applicable_reason"
    ] == ("Agent ID 0's Ultimate cannot target Agent ID 5.")
    assert catalog_by_name["agent_0_to_agent_5_ultimate_damage_done"][
        "not_applicable_reason"
    ] == ("Agent ID 0 cannot deal this damage with Ultimate abilities.")
    assert (
        catalog_by_name["agent_0_to_agent_0_ultimate_application_fraction"][
            "not_applicable_reason"
        ]
        is None
    )
    first = analysis.summary(0)
    monkeypatch.setattr(analysis_module, "_scan_block", forbidden)
    monkeypatch.setattr(analysis_module, "metric_locations", forbidden)
    monkeypatch.setattr(
        analysis_module.ReplayAnalysis, "_not_applicable_reason", forbidden
    )
    for index in (3, 1, 0, 3, 0):
        assert analysis.summary(index)["frame_index"] == index
    assert analysis.summary(0) == first
    assert analysis.catalog() == catalog
    assert all(
        "not_applicable_reason" not in row and "search_terms" not in row
        for row in cast(list[dict[str, object]], first["statistics"])
    )
    with pytest.raises(IndexError):
        analysis.summary(4)
    with pytest.raises(ValueError):
        analysis.summary(0, scope="invalid")  # pyright: ignore[reportArgumentType]
    np.testing.assert_equal(len(first["statistics"]), len(METRIC_COLUMNS))  # pyright: ignore[reportArgumentType]
