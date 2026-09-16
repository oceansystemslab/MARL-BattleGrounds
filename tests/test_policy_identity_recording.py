"""Check recorded actor inputs and rejection of incompatible input versions."""

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from pydantic import ValidationError
from tests.evaluation_fixtures import (
    captured_evaluation_trajectory,
    current_captured_evaluation_trajectory,
    evaluation_env_config,
    valid_shared_availability,
)

from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
    reconstruct_class_ids_by_agent_by_team_v3,
    reconstruct_shared_obs_sensor_source_bank_v2,
)
from marl_battlegrounds.evaluation.models import (
    BaseObservationV2,
    EvaluationEpisodeContextV3,
    EvaluationFrameV2,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay_io import load_replay, save_replay
from marl_battlegrounds.evaluation.replay_v2 import build_replay_v2, context_v2
from marl_battlegrounds.evaluation.replay_v3 import (
    ReplayArtifactV3,
    build_replay_v3,
    validate_replay_artifact_v3,
)
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.policies.shared_obs import (
    build_shared_obs_sensor_source_bank,
    mask_source_bank_for_recipient,
)


@pytest.mark.parametrize("mode", ["no_shared_obs", "shared_obs"])
@pytest.mark.parametrize("team_sizes", [(1, 1), (3, 2), (5, 5)])
def test_current_replay_preserves_exact_public_input_leaves(
    tmp_path: Path,
    mode: Literal["no_shared_obs", "shared_obs"],
    team_sizes: tuple[int, int],
) -> None:
    trajectory = current_captured_evaluation_trajectory(
        config=evaluation_env_config(team_sizes=team_sizes, max_steps=1),
        execution_information_mode=mode,
    )
    replay = build_replay_v3(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=capture_runtime_provenance("0.0.0"),
    )
    path = tmp_path / "current.marlbg-replay.json"
    save_replay(replay, path)
    loaded = load_replay(path).replay
    assert type(loaded) is ReplayArtifactV3
    assert loaded == replay
    assert path.read_bytes() == canonical_json_bytes(replay)
    assert tuple(tmp_path.iterdir()) == (path,)
    for observation, frame in zip(trajectory.observations, loaded.frames, strict=True):
        assert type(frame) is EvaluationFrameV2
        base = frame.base_observation
        for name in observation._fields:
            live = getattr(observation, name)
            recorded = getattr(base, name)
            if name == "spawn_lifecycle":
                for leaf in live._fields:
                    value = (
                        reconstruct_class_ids_by_agent_by_team_v3(trajectory.context)
                        if leaf == "class_ids_by_agent_by_team"
                        else getattr(recorded, leaf)
                    )
                    np.testing.assert_array_equal(value, getattr(live, leaf))
            elif name == "previous_timestep_actions":
                for leaf in live._fields:
                    np.testing.assert_array_equal(
                        getattr(recorded, leaf), getattr(live, leaf)
                    )
            else:
                np.testing.assert_array_equal(recorded, live)
        if mode == "shared_obs":
            bank = build_shared_obs_sensor_source_bank(observation)
            availability = valid_shared_availability(trajectory.context)
            for actor in range(10):
                team = actor // 5
                live_team = type(bank)(*(value[team] for value in bank))
                expected = mask_source_bank_for_recipient(
                    live_team, availability[actor, team * 5 : (team + 1) * 5]
                )
                actual = reconstruct_shared_obs_sensor_source_bank_v2(
                    trajectory.context, frame, actor
                )
                for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_current_recording_rejects_team_ids_wrong_self_rows_and_old_projection() -> (
    None
):
    trajectory = current_captured_evaluation_trajectory()
    base = trajectory.frames[0].base_observation
    for field, value in (
        ("self_ally_index", (4, *base.self_ally_index[1:])),
        (
            "self_features",
            tuple(
                tuple(1.0 if column == 3 else value for column, value in enumerate(row))
                for row in base.self_features
            ),
        ),
    ):
        with pytest.raises(ValidationError):
            BaseObservationV2.model_validate({**base.model_dump(), field: value})
    with pytest.raises(ValidationError, match="matching relative-input projection"):
        EvaluationEpisodeContextV3.model_validate(
            {
                **trajectory.context.model_dump(),
                "actor_projection": NO_SHARED_OBS_ACTOR_PROJECTION_V2,
            }
        )
    with pytest.raises(ValueError, match="cannot change"):
        context_v2(trajectory.context)


def test_old_and_current_roots_keep_their_original_bytes_and_fail_closed(
    tmp_path: Path,
) -> None:
    legacy = captured_evaluation_trajectory(transition_count=1)
    old = build_replay_v2(
        legacy.context,
        legacy.frames,
        legacy.transitions,
        runtime_provenance=capture_runtime_provenance("0.0.0"),
    )
    path = tmp_path / "old.marlbg-replay.json"
    original = canonical_json_bytes(old)
    save_replay(old, path)
    assert canonical_json_bytes(load_replay(path).replay) == original
    with pytest.raises(ValueError, match="exact declared root"):
        validate_replay_artifact_v3(old)  # pyright: ignore[reportArgumentType]
    current = current_captured_evaluation_trajectory()
    with pytest.raises(TypeError, match="exact frame V2"):
        build_replay_v3(
            current.context,
            legacy.frames,  # pyright: ignore[reportArgumentType]
            current.transitions,
            runtime_provenance=capture_runtime_provenance("0.0.0"),
        )


def test_current_actor_pov_persistence_keeps_exact_version_and_bytes(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.pov import export_actor_pov_replay_v2
    from marl_battlegrounds.evaluation.replay_io import (
        ReplayLoadError,
        load_actor_pov_replay_artifact_v1,
        load_actor_pov_replay_artifact_v2,
        save_actor_pov_replay_artifact_v2,
    )

    trajectory = current_captured_evaluation_trajectory()
    replay = build_replay_v3(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=capture_runtime_provenance("0.0.0"),
    )
    artifact = export_actor_pov_replay_v2(replay, global_slot=6)
    path = tmp_path / "actor.marlbg-pov.json"
    save_actor_pov_replay_artifact_v2(artifact, replay, path)
    assert load_actor_pov_replay_artifact_v2(path, source_replay=replay) == artifact
    assert load_actor_pov_replay_artifact_v2(path) == artifact
    assert path.read_bytes() == canonical_json_bytes(artifact)
    assert artifact.content.frames[0].self_ally_index == 1
    with pytest.raises(ReplayLoadError) as error:
        load_actor_pov_replay_artifact_v1(path)
    assert error.value.code == "unsupported_schema_version"


def test_custom_source_subsets_survive_separate_writer_chunks(tmp_path: Path) -> None:
    import jax
    import jax.numpy as jnp
    from jax import Array

    from marl_battlegrounds import EnvironmentState, make
    from marl_battlegrounds.core.types import ActionMask
    from marl_battlegrounds.environment import EpisodeInfo
    from marl_battlegrounds.evaluation.policy_execution import apply_policies
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.policies.actor import ActorAction
    from marl_battlegrounds.policies.input import ActorInput
    from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

    env = make("tdm", metrics="none", replay_episodes=(1,))
    roster = ("mage", "warrior", "hunter", "rogue", "priest")
    config = make_standard_team_deathmatch_config(
        map_id=12, max_steps=3, team_a_roster=roster, team_b_roster=roster
    )
    _, state = env.reset(jax.random.key(0), config, episode_id=1)
    full = state.source_availability
    delivered: list[tuple[ActorInput, ActorInput]] = []
    chosen: list[Array] = []
    traces: list[bool] = []

    def probe(
        variables: object,
        memory: object,
        actor: ActorInput,
        mask: ActionMask,
        key: Array,
    ) -> tuple[ActorAction, ActorInput]:
        del variables, memory, mask, key
        zero = jnp.asarray(0, jnp.int32)
        return ActorAction(zero, zero, zero), actor

    @jax.jit
    def advance(
        current: EnvironmentState, availability: Array
    ) -> tuple[EnvironmentState, EpisodeInfo, ActorInput, ActorInput]:
        traces.append(True)
        current = current._replace(source_availability=availability)
        observations = env.get_observations(current)
        action, received_a, received_b = apply_policies(
            probe,
            probe,
            (),
            (),
            (),
            (),
            observations,
            current.action_mask,
            jax.random.split(jax.random.key(1), 10),
        )
        _, successor, _, _, info = env.step(jax.random.key(2), current, action)
        return successor, info, received_a, received_b

    with RunWriter(
        tmp_path, policies={"team_a": "probe-a", "team_b": "probe-b"}
    ) as writer:
        for availability in (
            jnp.zeros_like(full),
            full & (jnp.arange(10) % 2 == 0),
            full,
        ):
            state, info, received_a, received_b = cast(
                tuple[EnvironmentState, EpisodeInfo, ActorInput, ActorInput],
                advance(state, availability),
            )
            chosen.append(availability)
            delivered.append((received_a, received_b))
            assert info.priority is None and info.full is None
            writer.write(info)
        run_dir = writer.run_dir
    assert len(traces) == 1
    paths = tuple(run_dir.rglob("*.marlbg-replay.json"))
    assert len(paths) == 1
    loaded = load_replay(paths[0]).replay
    assert isinstance(loaded, ReplayArtifactV3)
    assert len(loaded.frames) == 4
    for index, (frame, receivers) in enumerate(
        zip(loaded.frames[:-1], delivered, strict=True)
    ):
        np.testing.assert_array_equal(
            frame.shared_obs_information_availability_by_recipient_and_sensor_source,
            chosen[index],
        )
        for slot in range(10):

            def actor_row(leaf: Array, slot: int = slot) -> Array:
                return leaf[slot % 5]

            received = jax.tree.map(actor_row, receivers[slot // 5])
            reconstructed = reconstruct_shared_obs_sensor_source_bank_v2(
                loaded.header.context, frame, slot
            )
            for actual, expected in zip(
                reconstructed, received.source_bank, strict=True
            ):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                received.source_availability,
                chosen[index][slot, (slot // 5) * 5 : (slot // 5 + 1) * 5],
            )
    np.testing.assert_array_equal(
        loaded.frames[
            -1
        ].shared_obs_information_availability_by_recipient_and_sensor_source,
        chosen[-1],
    )
