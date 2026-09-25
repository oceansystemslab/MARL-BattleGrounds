"""Check replay V4, the current record of the Team Deathmatch Red Zone rule.

Replay V4 holds context V4 (resolved config V2 with the depth) and frame V3 (20 context
columns; column 19 is the depth). It saves and loads as the exact canonical bytes its
reference describes, and the package's replay_from_packets builds it. load_replay
refuses an unsupported version 5 and V4 content relabelled as V3, and the V4 builder
refuses pre-Red-Zone contexts and frames. Records written before the rule (the fixtures
in tests/fixtures/historical_scenario_v4, captured at commit 59c157c: two replay V3
files, a scenario record V4 and an Agent POV V2 export of slot 0) still load byte for
byte and validate; the POV V2 file joins its source replay V3. The scenario record V5
loader refuses V4 bytes and the V4 loader refuses V5-labelled bytes. The POV V3 loader
refuses the POV V2 file as an unsupported schema version. reconstruct_env_config_v1
returns the recorded depth from a V2 record and 0.0 from a V1 record.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from tests.evaluation_fixtures import (
    current_captured_evaluation_trajectory,
    evaluation_env_config,
    pre_red_zone_captured_evaluation_trajectory,
)

import marl_battlegrounds.evaluation as evaluation_api
from marl_battlegrounds.evaluation import replay_v4
from marl_battlegrounds.evaluation.catalog import (
    build_resolved_env_config_v2,
    reconstruct_env_config_v1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV4,
    EvaluationFrameV3,
    ResolvedEnvConfigV1,
    ResolvedEnvConfigV2,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import ActorPovReplayArtifactV2
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    PreparedReplay,
    ReplayLoadError,
    load_actor_pov_replay_artifact_v2,
    load_actor_pov_replay_artifact_v3,
    load_replay,
    load_scenario_evaluation_record_v4,
    load_scenario_evaluation_record_v5,
    save_replay,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.evaluation.replay_v4 import (
    ReplayArtifactV4,
    build_replay_v4,
    replay_reference_v4,
    validate_replay_artifact_v4,
)
from marl_battlegrounds.evaluation.scenario import ScenarioEvaluationRecordV4

_PRE_RED_ZONE = Path(__file__).parent / "fixtures" / "historical_scenario_v4"
_DEPTH = 5.0


@pytest.fixture(scope="module")
def current_replay() -> ReplayArtifactV4:
    trajectory = current_captured_evaluation_trajectory(
        config=evaluation_env_config(
            task_mode=1,
            team_deathmatch_score_threshold=5,
            team_deathmatch_red_zone_depth=_DEPTH,
            max_steps=1,
        )
    )
    return build_replay_v4(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=RuntimeProvenanceV1(
            python_version="3.13.0",
            package_version="0.0.0",
            jax_version="0.7.0",
            jaxlib_version="0.7.0",
            numpy_version="2.3.0",
            pydantic_version="2.11.0",
            platform="linux",
            machine="x86_64",
            backend="cpu",
            device="generic-cpu",
            precision="float32",
            environment_count=1,
            batch_shape=(1,),
            policy_execution_included=False,
        ),
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.write_bytes(canonical_json_bytes(payload))
    return path


def test_replay_v4_saves_and_loads_its_exact_canonical_bytes(
    tmp_path: Path, current_replay: ReplayArtifactV4
) -> None:
    context = current_replay.header.context
    assert type(context) is EvaluationEpisodeContextV4
    assert type(context.resolved_env_config) is ResolvedEnvConfigV2
    assert context.resolved_env_config.team_deathmatch_red_zone_depth == _DEPTH
    for frame in current_replay.frames:
        assert type(frame) is EvaluationFrameV3
        assert [row[19] for row in frame.base_observation.context_features] == [
            _DEPTH if row.configured_active else 0.0 for row in context.roster
        ]
    validate_replay_artifact_v4(current_replay)

    path = tmp_path / "current.marlbg-replay.json"
    saved = save_replay(current_replay, path)
    loaded = load_replay(path)
    assert type(loaded.replay) is ReplayArtifactV4
    assert loaded.replay == current_replay
    assert (loaded.status, loaded.metric_report_artifact) == ("not_recorded", None)
    assert path.read_bytes() == canonical_json_bytes(current_replay)
    assert PreparedReplay(current_replay).replay_json_bytes == path.read_bytes()
    reference = replay_reference_v4(current_replay)
    assert (reference.schema_version, reference.replay_schema_version) == (4, 4)
    assert reference.canonical_byte_length == saved.replay_byte_length
    assert saved.replay_byte_length == path.stat().st_size
    assert evaluation_api.replay_from_packets is replay_v4.replay_from_packets


def test_replay_readers_and_builder_refuse_other_versions(
    tmp_path: Path, current_replay: ReplayArtifactV4
) -> None:
    payload = json.loads(canonical_json_bytes(current_replay))
    for version, code in (
        (5, "unsupported_schema_version"),
        (3, "model_validation_failed"),
    ):
        path = _write_json(
            tmp_path / f"version-{version}.marlbg-replay.json",
            {**payload, "schema_version": version},
        )
        with pytest.raises(ReplayLoadError) as failure:
            load_replay(path)
        assert failure.value.code == code

    # Frame zero is enough: the builder checks record types before any join.
    config = evaluation_env_config()
    historical = pre_red_zone_captured_evaluation_trajectory(
        transition_count=0, config=config
    )
    current = current_captured_evaluation_trajectory(transition_count=0, config=config)
    with pytest.raises(TypeError, match="exact context V4"):
        build_replay_v4(
            historical.context,  # pyright: ignore[reportArgumentType]
            current.frames,
            current.transitions,
            runtime_provenance=current_replay.header.runtime_provenance,
        )
    with pytest.raises(TypeError, match="exact frame V3"):
        build_replay_v4(
            current.context,
            historical.frames,  # pyright: ignore[reportArgumentType]
            historical.transitions,
            runtime_provenance=current_replay.header.runtime_provenance,
        )


def test_pre_red_zone_records_load_byte_for_byte_and_validate(
    tmp_path: Path, current_replay: ReplayArtifactV4
) -> None:
    replays: dict[str, ReplayArtifactV3] = {}
    for name in ("scenario_1", "no_shared_obs"):
        path = _PRE_RED_ZONE / f"{name}.marlbg-replay.json"
        loaded = load_replay(path)
        assert type(loaded.replay) is ReplayArtifactV3
        assert canonical_json_bytes(loaded.replay) == path.read_bytes()
        assert type(loaded.replay.header.context.resolved_env_config) is (
            ResolvedEnvConfigV1
        )
        replays[name] = loaded.replay

    record_path = _PRE_RED_ZONE / "scenario_1.marlbg-scenario.json"
    record = load_scenario_evaluation_record_v4(
        record_path, source_replay=replays["scenario_1"]
    )
    assert type(record) is ScenarioEvaluationRecordV4
    assert canonical_json_bytes(record) == record_path.read_bytes()

    pov_path = _PRE_RED_ZONE / "no_shared_obs_slot_0.marlbg-pov.json"
    pov = load_actor_pov_replay_artifact_v2(
        pov_path, source_replay=replays["no_shared_obs"]
    )
    assert type(pov) is ActorPovReplayArtifactV2
    assert canonical_json_bytes(pov) == pov_path.read_bytes()
    with pytest.raises(ReplayLoadError) as newer_pov_loader:
        load_actor_pov_replay_artifact_v3(pov_path)
    assert newer_pov_loader.value.code == "unsupported_schema_version"

    with pytest.raises(ReplayLoadError) as newer_loader:
        load_scenario_evaluation_record_v5(record_path, source_replay=current_replay)
    assert newer_loader.value.code == "unsupported_schema_version"
    relabelled = _write_json(
        tmp_path / "relabelled.marlbg-scenario.json",
        {**json.loads(record_path.read_bytes()), "schema_version": 5},
    )
    with pytest.raises(ReplayLoadError) as older_loader:
        load_scenario_evaluation_record_v4(
            relabelled, source_replay=replays["scenario_1"]
        )
    assert older_loader.value.code == "unsupported_schema_version"


def test_reconstructed_config_keeps_the_recorded_depth_or_zero_for_v1(
    current_replay: ReplayArtifactV4,
) -> None:
    context = current_replay.header.context
    config = reconstruct_env_config_v1(context)
    assert config.team_deathmatch_red_zone_depth == _DEPTH
    assert build_resolved_env_config_v2(config) == context.resolved_env_config

    historical = load_replay(_PRE_RED_ZONE / "no_shared_obs.marlbg-replay.json")
    historical_context = historical.replay.header.context
    assert type(historical_context.resolved_env_config) is ResolvedEnvConfigV1
    assert (
        reconstruct_env_config_v1(historical_context).team_deathmatch_red_zone_depth
        == 0.0
    )
