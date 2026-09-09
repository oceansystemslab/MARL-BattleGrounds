"""Scalar replay analysis shares numerical authority and immutable boundaries."""

import csv
import io
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from tests.evaluation_fixtures import captured_evaluation_trajectory
from tests.test_evaluation_replay import runtime_provenance as runtime_provenance

from marl_battlegrounds.core.types import ActionMask, Info, Reward
from marl_battlegrounds.evaluation import analysis as analysis_module
from marl_battlegrounds.evaluation.analysis import analyze_replay
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
    METRIC_FAMILIES,
    PRIORITY_METRIC_COLUMNS,
)
from marl_battlegrounds.evaluation.metrics import EvaluationEpisodeObserverV1
from marl_battlegrounds.evaluation.replay import (
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplay, LoadedReplayBundleV1
from marl_battlegrounds.evaluation.replay_v2 import build_replay_v2


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
        families = [
            {"name": name, "label": label, "description": description}
            for name, (label, description) in METRIC_FAMILIES.items()
        ]
        assert summary["families"] == families
        assert basic.summary(index)["families"] == families[:2]
        statistics = cast(list[dict[str, object]], summary["statistics"])
        csv_rows = list(csv.DictReader(io.StringIO(analysis.csv(index))))
        assert len(csv_rows) == 1
        row = csv_rows[0]
        assert row["frame_index"] == str(index)
        assert row["scope"] == "cursor"
        for slot in range(10):
            assert row[f"agent_{slot}_class_id"] == str(
                trajectory.context.roster[slot].class_id
            )
        for column_index, column in enumerate(METRIC_COLUMNS):
            scalar = statistics[column_index]
            assert scalar["name"] == column.name
            assert scalar["valid"] == bool(expected.valid[column_index])
            if scalar["valid"]:
                assert scalar["value"] == pytest.approx(
                    float(expected.values[column_index]), rel=1e-5, abs=1e-5
                )
                assert float(row[column.name]) == scalar["value"]
            else:
                assert row[column.name] == ""
                assert scalar["value"] is None
        assert (
            cast(list[dict[str, object]], basic.summary(index)["statistics"])
            == statistics[: len(PRIORITY_METRIC_COLUMNS)]
        )
    # Cached reads cannot collect more facts or mutate an earlier snapshot.
    first = analysis.summary(0)
    monkeypatch.setattr(analysis_module, "_scan_block", forbidden)
    for index in (3, 1, 0, 3, 0):
        assert analysis.summary(index)["frame_index"] == index
    assert analysis.summary(0) == first
    with pytest.raises(IndexError):
        analysis.summary(4)
    with pytest.raises(ValueError):
        analysis.summary(0, scope="invalid")  # pyright: ignore[reportArgumentType]
    np.testing.assert_equal(len(first["statistics"]), len(METRIC_COLUMNS))  # pyright: ignore[reportArgumentType]
