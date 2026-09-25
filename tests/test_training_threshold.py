"""Check reset-only score curricula, exact sources, recording and disk recovery.

CPU checks cover the 13 declared shares, content binding version 3 for both
the K20-only bank and the 13-score bank (both state score_thresholds and the
default Red Zone depth), the version 1 binding saved before Red Zone scoring
(tests/fixtures/training_content_binding_v1.json), which still omits
score_thresholds, reads back as K20 and serializes to the same canonical bytes
and digest, threshold-major sources, unchanged map/roster keys, dynamic sampler
reuse, actor-visible reset timing, actual exposure and map counts, and recorded
native H300 endings. A real MAPPO save/restore checks the extended source bank and
rejects changed schedules before numerical restore. These checks establish no
GPU cost or learned advantage from easier training thresholds.
"""

from __future__ import annotations

import json

# pyright: reportPrivateUsage=false
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.core.types import CONTEXT_FEATURE_TDM_SCORE_THRESHOLD
from marl_battlegrounds.evaluation.models import (
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.training import checkpoints
from marl_battlegrounds.training._compilation import execution_identity
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    TrainingContentBinding,
    prepare_training_content,
)
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    _reset_pending,
    _validate_training_continuation,
    collect_training_rollout,
    init_training_collection,
    training_summary,
)
from marl_battlegrounds.training.curriculum import make_training_schedule
from marl_battlegrounds.training.distributions import (
    SampledTrainingConfigs,
    sample_training_configs,
)
from marl_battlegrounds.training.learner import init_learner
from marl_battlegrounds.training.runner import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
)

type Tree = Any

THRESHOLDS = (*range(1, 11), 12, 15, 20)
_V1_FIXTURE = Path(__file__).parent / "fixtures" / "training_content_binding_v1.json"


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content(score_thresholds=THRESHOLDS)


@pytest.fixture(scope="module")
def k20() -> PreparedTrainingContent:
    return prepare_training_content()


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def _apply(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Tree
) -> SystemOutput:
    del variables, keys
    zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return SystemOutput(
        ActorAction(zero, zero, zero),
        memory,
        learning_outputs=inputs.actors.observation.context_features[
            ..., CONTEXT_FEATURE_TDM_SCORE_THRESHOLD
        ],
    )


def test_exact_schedule_and_config_roundtrip() -> None:
    schedule = make_training_schedule(
        total_env_steps=1200, num_envs=2, score_threshold_curriculum=True
    )
    assert schedule.score_thresholds == THRESHOLDS
    assert int(schedule.arrays.stage_count) == 13
    np.testing.assert_array_equal(
        schedule.arrays.round_budgets[:13], (60, *([20] * 9), 30, 30, 300)
    )
    np.testing.assert_array_equal(schedule.arrays.score_thresholds[:13], THRESHOLDS)
    np.testing.assert_array_equal(schedule.arrays.team_sizes, 5)
    assert np.all(schedule.arrays.eligible_maps)
    config = TrainConfig(score_threshold_curriculum=True)
    assert config_from_dict(config_to_dict(config)) == config
    assert config_from_dict({}).score_threshold_curriculum is False
    for constructor in (
        lambda: TrainConfig(curriculum=True, score_threshold_curriculum=True),
        lambda: make_training_schedule(
            total_env_steps=1200,
            num_envs=2,
            curriculum=True,
            score_threshold_curriculum=True,
        ),
    ):
        with pytest.raises(ValueError, match="Choose"):
            constructor()
    with pytest.raises(TypeError, match="bool"):
        TrainConfig(score_threshold_curriculum=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="every active stage"):
        make_training_schedule(
            total_env_steps=4, num_envs=2, score_threshold_curriculum=True
        )


def test_content_preserves_old_serialization_and_binds_threshold_order(
    prepared: PreparedTrainingContent, k20: PreparedTrainingContent
) -> None:
    raw = _V1_FIXTURE.read_bytes()
    old = json.loads(raw)
    assert old["schema_version"] == 1 and "score_thresholds" not in old
    assert (
        canonical_digest_sha256(old, exclude={"canonical_digest"})
        == old["canonical_digest"]
    )
    loaded = TrainingContentBinding.model_validate_json(raw)
    assert loaded.score_thresholds == (20,)
    assert canonical_json_bytes(loaded) == canonical_json_bytes(old)
    single = k20.binding.model_dump(mode="json")
    assert single["schema_version"] == 3 and single["score_thresholds"] == [20]
    assert single["red_zone_depth"] == 5.0
    assert (
        canonical_digest_sha256(single, exclude={"canonical_digest"})
        == single["canonical_digest"]
    )
    assert prepared.binding.schema_version == 3
    assert prepared.binding.red_zone_depth == 5.0
    assert prepared.binding.score_thresholds == THRESHOLDS
    assert len(prepared.binding.source_configurations) == 546
    assert (
        prepared.binding.source_configurations[-42:]
        == k20.binding.source_configurations
    )
    for index, threshold in enumerate(THRESHOLDS):

        def select(value: Array, i: int = index) -> Array:
            return value[i * 42 : (i + 1) * 42]

        block = jax.tree.map(select, prepared.source_configs)
        np.testing.assert_array_equal(block.team_deathmatch_score_threshold, threshold)
        _equal(
            block._replace(
                team_deathmatch_score_threshold=k20.source_configs.team_deathmatch_score_threshold
            ),
            k20.source_configs,
        )
    changed = prepared.binding.model_dump(mode="json")
    changed["score_thresholds"] = list(reversed(THRESHOLDS))
    with pytest.raises(ValueError, match="digest"):
        TrainingContentBinding.model_validate_json(canonical_json_bytes(changed))
    reread = prepare_training_content(expected=prepared.binding)
    assert reread.binding == prepared.binding
    with pytest.raises(ValueError, match="incompatible"):
        prepare_training_content(expected=prepared.binding, score_thresholds=(20,))


@pytest.mark.parametrize("values", [(), (True,), (0,), (1, 1), (2**24,), (1.0,)])
def test_invalid_threshold_declarations_fail(values: Tree) -> None:
    with pytest.raises((TypeError, ValueError)):
        prepare_training_content(score_thresholds=values)


def test_sampler_keeps_keys_and_reuses_one_trace(
    prepared: PreparedTrainingContent, k20: PreparedTrainingContent
) -> None:
    calls = []
    root, generation = jax.random.key(7), jnp.zeros(4, jnp.int32)
    maps, size = jnp.ones(42, bool), jnp.int32(5)

    def sample(threshold: Array) -> SampledTrainingConfigs:
        calls.append("trace")
        return sample_training_configs(
            prepared.source_configs,
            root,
            generation,
            eligible_maps=maps,
            team_size=size,
            score_threshold=threshold,
        )

    compiled = cast(Callable[[Array], SampledTrainingConfigs], jax.jit(sample))
    base = sample_training_configs(
        k20.source_configs, root, generation, eligible_maps=maps, team_size=size
    )
    for index, threshold in enumerate(THRESHOLDS):
        actual = compiled(jnp.int32(threshold))
        np.testing.assert_array_equal(
            actual.source_indices, base.source_indices + index * 42
        )
        np.testing.assert_array_equal(actual.source_class_ids, base.source_class_ids)
        _equal(
            actual.config._replace(
                team_deathmatch_score_threshold=base.config.team_deathmatch_score_threshold
            ),
            base.config,
        )
    assert calls == ["trace"]
    with pytest.raises(ValueError, match="not present"):
        sample(jnp.int32(11))


def test_native_reset_visibility_recording_and_actual_exposure(
    prepared: PreparedTrainingContent, tmp_path: Path
) -> None:
    actor = System("Threshold Observer", _apply)
    collection, initial = init_training_collection(
        actor,
        (),
        schedule=make_training_schedule(
            total_env_steps=1200, num_envs=2, score_threshold_curriculum=True
        ),
        prepared=prepared,
        recording=True,
        metrics="none",
    )
    with marl_bgs.RunWriter(
        tmp_path,
        phase="training",
        policies={"team_a": collection.actor, "team_b": collection.opponent},
    ) as writer:
        ended, first = collect_training_rollout(
            collection, initial, length=300, writer=writer
        )
        token = writer.checkpoint_recording()
        final, second = collect_training_rollout(
            collection, ended, length=1, writer=writer
        )
        run_dir = writer.run_dir
    np.testing.assert_array_equal(first.transitions.learning_outputs, 1)
    np.testing.assert_array_equal(first.transitions.ended[-1], True)
    np.testing.assert_array_equal(first.transitions.episode_length[-1], 300)
    np.testing.assert_array_equal(ended.state.config.team_deathmatch_score_threshold, 1)
    assert int(ended.tracking.stage_ordinal) == 12
    np.testing.assert_array_equal(second.transitions.learning_outputs, 20)
    np.testing.assert_array_equal(second.transitions.episode_start, True)
    np.testing.assert_array_equal(final.source_indices // 42, 12)
    summary = training_summary(collection, final)
    by_k = {
        row["score_threshold"]: row
        for row in cast(list[dict[str, int]], summary["exposure_by_score_threshold"])
    }
    assert by_k[1]["env_steps"] == 600 and by_k[20]["env_steps"] == 2
    assert sum(cast(list[int], summary["steps_by_map"])) == 602
    assert all(row["env_steps"] == 0 for key, row in by_k.items() if key not in (1, 20))
    with marl_bgs.RunWriter(
        resume_from=run_dir,
        recording_checkpoint=token,
        phase="training",
        policies={"team_a": collection.actor, "team_b": collection.opponent},
    ) as writer:
        replayed, replay = collect_training_rollout(
            collection, ended, length=1, writer=writer
        )
    _equal(replayed, final)
    _equal(replay, second)
    partial = ended._replace(
        state=ended.state._replace(
            done=ended.state.done._replace(truncated=jnp.asarray([True, False]))
        )
    )
    reset = cast(
        Callable[..., TrainingCarry], jax.jit(_reset_pending, static_argnums=0)
    )(collection, partial)
    np.testing.assert_array_equal(
        reset.state.config.team_deathmatch_score_threshold, (20, 1)
    )
    np.testing.assert_array_equal(reset.progress.episode_stage, (12, 0))


def test_threshold_checkpoint_restores_bank_and_rejects_wrong_schedule(
    prepared: PreparedTrainingContent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ppo = PPOConfig(rollout_length=2, epochs=1, input_scale=0.01)
    config = TrainConfig(
        num_envs=4,
        total_env_steps=240,
        score_threshold_curriculum=True,
        ppo=ppo,
        metrics="none",
    )
    schedule = make_training_schedule(
        total_env_steps=240, num_envs=4, score_threshold_curriculum=True
    )
    collection, state = init_learner(
        schedule=schedule, ppo=ppo, prepared=prepared, metrics="none"
    )
    metadata: dict[str, object] = dict(
        run_id="threshold-test",
        attempt_id="first",
        parent_checkpoint=None,
        config=config_to_dict(config),
        source={"scope": "test"},
        dependencies={"scope": "test"},
        execution=execution_identity(),
        host_state={},
        log_cursors={},
    )
    path = checkpoints.save_checkpoint(
        tmp_path, collection, state, metadata=metadata, ppo=ppo
    )
    expected = {
        name: metadata[name]
        for name in ("config", "source", "dependencies", "execution")
    }
    restored = checkpoints.restore_checkpoint(
        path, collection, state, expected_metadata=expected, ppo=ppo
    )
    _equal(restored.state, state)
    _validate_training_continuation(
        collection, restored.state.carry, recheck_installed_content=False
    )
    a, ar = collect_training_rollout(collection, state.carry, length=2)
    b, br = collect_training_rollout(collection, restored.state.carry, length=2)
    _equal(a, b)
    _equal(ar, br)
    changed = replace(
        collection, schedule=make_training_schedule(total_env_steps=240, num_envs=4)
    )

    def forbidden(*args: Tree, **kwargs: Tree) -> Tree:
        pytest.fail("Wrong schedule reached numerical restore")

    monkeypatch.setattr(checkpoints, "_restore_arrays", forbidden)
    with pytest.raises(ValueError, match="collection settings"):
        checkpoints.restore_checkpoint(
            path, changed, state, expected_metadata=expected, ppo=ppo
        )
    wrong = state.carry._replace(
        state=state.carry.state._replace(
            config=state.carry.state.config._replace(
                team_deathmatch_score_threshold=jnp.full(4, 20, jnp.int32)
            )
        )
    )
    with pytest.raises(ValueError, match="score thresholds"):
        _validate_training_continuation(
            collection, wrong, recheck_installed_content=False
        )
