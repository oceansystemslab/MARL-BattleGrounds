"""Check map/roster sampling, random-key ownership and exact source profiles.

Independent scalar key and roster calculations, exhaustive permutation counts,
fixed-seed frequency checks and public partial reset cover the approved training
distribution. Synthetic source banks keep unit tests small; content preparation
has separate full-catalog tests. These CPU checks do not claim learning quality
or GPU throughput. No simulator or class-catalog rule is copied here.
"""

# pyright: reportPrivateUsage=false
from itertools import permutations
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import _tdm_assets, tasks
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import EnvironmentState
from marl_battlegrounds.episode_tracking import StepResult
from marl_battlegrounds.tasks import (
    _source_config_with_class_ids,
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
    prepare_exact_env_config,
)
from marl_battlegrounds.training import (
    TRAINING_KEY_SCHEMA_VERSION,
    SampledTrainingConfigs,
    sample_training_configs,
    training_keys,
    validate_training_distribution,
)
from marl_battlegrounds.training import distributions as sampling


def _row[T](tree: T, index: int) -> T:
    def take(value: Array) -> Array:
        return value[index]

    return jax.tree.map(take, tree)


def _assert_tree(actual: object, expected: object) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.fixture(scope="module")
def bank() -> EnvConfig:
    team_a, team_b = canonical_tournament_rosters()
    sources = [
        make_standard_team_deathmatch_config(
            map_id=i, max_steps=3, team_a_roster=team_a, team_b_roster=team_b
        )
        for i in (0, 20, 41)
    ]

    def stack(*values: ArrayLike) -> Array:
        return jnp.stack([jnp.asarray(values[i % 3]) for i in range(42)])

    return jax.tree.map(stack, *sources)


_sample = jax.jit(sample_training_configs)
_ALL_MAPS = np.ones(42, dtype=np.bool_)


def _draw(
    bank: EnvConfig,
    size: int,
    *,
    batch: int = 8,
    seed: int = 7,
    mask: Array | None = None,
) -> SampledTrainingConfigs:
    return cast(
        SampledTrainingConfigs,
        _sample(
            bank,
            jax.random.key(seed),
            jnp.arange(batch, dtype=jnp.int32),
            eligible_maps=jnp.asarray(_ALL_MAPS) if mask is None else mask,
            team_size=jnp.asarray(size, jnp.int32),
        ),
    )


@pytest.mark.parametrize("size", range(1, 6))
def test_sampled_profiles_order_padding_and_catalog(bank: EnvConfig, size: int) -> None:
    result = _draw(bank, size)
    assert (
        result.source_indices.shape == (8,) and result.source_indices.dtype == jnp.int32
    )
    assert (
        result.source_class_ids.shape == (8, 10)
        and result.source_class_ids.dtype == jnp.int32
    )
    assert all(leaf.shape[0] == 8 for leaf in jax.tree.leaves(result.config))
    canonical = np.asarray(
        [tasks._roster_ids(r, name="reference") for r in canonical_tournament_rosters()]
    )
    for lane, classes in enumerate(
        np.asarray(result.source_class_ids).reshape(8, 2, 5)
    ):
        for team, row in enumerate(classes):
            chosen = row[:size].tolist()
            assert len(set(chosen)) == size
            assert chosen == [int(c) for c in canonical[team] if c in chosen]
            assert np.all(row[size:] == 0)
            if size == 1:
                assert 5 not in chosen
            if size == 5:
                np.testing.assert_array_equal(row, canonical[team])
        actual = _row(result.config, lane)
        expected_profile = resolve_agent_profile(
            jnp.asarray(classes.reshape(10)), jnp.asarray([size, size], jnp.int32)
        )
        _assert_tree(actual.agent_profile, expected_profile)
        prepare_exact_env_config(actual, num_envs=None)
        selected = _row(bank, int(result.source_indices[lane]))
        expected = selected._replace(agent_profile=expected_profile)
        if lane >= 4:
            expected = expected._replace(
                team_spawn_pad_positions=selected.team_spawn_pad_positions[::-1]
            )
        _assert_tree(actual, expected)


@pytest.mark.parametrize("legacy", [False, True])
def test_keys_match_independent_scalar_coordinates_and_streams(legacy: bool) -> None:
    root = jax.random.PRNGKey(771) if legacy else jax.random.key(771)
    generation = jnp.asarray([0, 9, 22, 2_147_483_647], jnp.int32)
    steps = jnp.asarray([3, 10, 6, 0], jnp.int32)
    streams = ("map", "roster", "opponent", "action", "reset", "step", "initialization")
    assert TRAINING_KEY_SCHEMA_VERSION == 1
    results: list[bytes] = []
    for tag, name in enumerate(streams):
        kwargs = {"decision_step": steps} if name in ("action", "step") else {}
        got = training_keys(root, generation, stream=name, **kwargs)
        expected: list[Array] = []
        for lane, epoch in enumerate(np.asarray(generation)):
            key = root
            for coordinate in (tag, lane, int(epoch)):
                key = jax.random.fold_in(key, coordinate)
            if name in ("action", "step"):
                key = jax.random.fold_in(key, int(steps[lane]))
            expected.append(jax.random.key_data(key))
        np.testing.assert_array_equal(jax.random.key_data(got), np.stack(expected))
        results.append(np.asarray(jax.random.key_data(got)).tobytes())
    assert len(set(results)) == 7


def test_scalar_reference_sampling_and_nested_mapping(bank: EnvConfig) -> None:
    size = 2
    result = _draw(bank, size, batch=4)
    canonical = np.asarray(
        [tasks._roster_ids(r, name="reference") for r in canonical_tournament_rosters()]
    )
    for lane in range(4):
        keys: list[Array] = []
        for tag in (0, 1):
            key = jax.random.key(7)
            for coordinate in (tag, lane, lane):
                key = jax.random.fold_in(key, coordinate)
            keys.append(key)
        assert int(result.source_indices[lane]) == int(
            jax.random.categorical(keys[0], jnp.zeros(42))
        )
        expected: list[int] = []
        for team in range(2):
            order = np.asarray(
                jax.random.permutation(jax.random.fold_in(keys[1], team), 5)
            ).tolist()
            chosen = sorted(order[:size])
            expected.extend(
                [int(canonical[team, i]) for i in chosen] + [0] * (5 - size)
            )
        np.testing.assert_array_equal(result.source_class_ids[lane], expected)

    roots = jax.random.split(jax.random.key(88), 3)
    generations = jnp.zeros(4, jnp.int32)

    def draw(root: Array) -> SampledTrainingConfigs:
        return sample_training_configs(
            bank,
            root,
            generations,
            eligible_maps=jnp.asarray(_ALL_MAPS),
            team_size=jnp.asarray(2, jnp.int32),
        )

    mapped = cast(SampledTrainingConfigs, jax.jit(jax.vmap(draw))(roots))
    for index, root in enumerate(roots):
        _assert_tree(
            _row(mapped, index), cast(SampledTrainingConfigs, jax.jit(draw)(root))
        )


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_every_permutation_gives_equal_subset_counts(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    all_orders = jnp.asarray(list(permutations(range(5))), jnp.int32)
    canonical = jnp.asarray(
        tasks._roster_ids(canonical_tournament_rosters()[0], name="reference"),
        jnp.int32,
    )

    def permutation(key: Array, _count: int) -> Array:
        return key

    monkeypatch.setattr(jax.random, "permutation", permutation)

    def draw(order: Array) -> Array:
        return sampling._team_classes(order, canonical, jnp.asarray(size, jnp.int32))

    rows = np.asarray(jax.vmap(draw)(all_orders))
    unique, counts = np.unique(rows, axis=0, return_counts=True)
    assert len(set(counts.tolist())) == 1
    assert len(unique) == {1: 4, 2: 10, 3: 10, 4: 5}[size]
    assert np.all(unique[:, size:] == 0)


def test_fixed_seed_frequency_and_independent_teams(bank: EnvConfig) -> None:
    mask = jnp.zeros(42, bool).at[jnp.asarray([0, 13, 41])].set(True)
    result = _draw(bank, 1, batch=4096, seed=712, mask=mask)
    indices = np.asarray(result.source_indices)
    assert set(indices.tolist()) == {0, 13, 41}
    # Bounds declared before observation: 12% per map and 32% per class pair.
    counts = np.asarray([(indices == index).sum() for index in (0, 13, 41)])
    assert np.all(np.abs(counts - 4096 / 3) < 0.12 * (4096 / 3))
    pairs = np.asarray(result.source_class_ids)[:, [0, 5]]
    pair_counts = np.asarray(
        [
            np.all(pairs == (a, b), axis=1).sum()
            for a in range(1, 5)
            for b in range(1, 5)
        ]
    )
    assert np.all(np.abs(pair_counts - 256) < 0.32 * 256)
    assert np.any(pairs[:, 0] == pairs[:, 1])
    assert np.any(pairs[:, 0] != pairs[:, 1])


def test_changes_are_lane_local_and_bank_is_immutable(bank: EnvConfig) -> None:
    before = jax.device_get(bank)
    generation = jnp.zeros(8, jnp.int32)
    arguments = dict(
        eligible_maps=jnp.asarray(_ALL_MAPS), team_size=jnp.asarray(3, jnp.int32)
    )
    first = cast(
        SampledTrainingConfigs,
        _sample(bank, jax.random.key(7), generation, **arguments),
    )
    changed = cast(
        SampledTrainingConfigs,
        _sample(bank, jax.random.key(7), generation.at[2].set(99), **arguments),
    )
    for index in (0, 1, 3, 4, 5, 6, 7):
        _assert_tree(_row(first, index), _row(changed, index))
    action_before = training_keys(
        jax.random.key(7), generation, stream="action", decision_step=generation
    )
    _draw(bank, 1, batch=8, seed=7, mask=jnp.arange(42) < 3)
    _assert_tree(
        jax.random.key_data(action_before),
        jax.random.key_data(
            training_keys(
                jax.random.key(7), generation, stream="action", decision_step=generation
            )
        ),
    )
    _assert_tree(bank, before)
    _assert_tree(
        first,
        cast(
            SampledTrainingConfigs,
            _sample(bank, jax.random.key(7), generation, **arguments),
        ),
    )


@pytest.mark.parametrize(
    "mask,size,error",
    [
        (np.zeros(42, bool), np.int32(2), ValueError),
        (np.ones(41, bool), np.int32(2), ValueError),
        (np.ones(42, np.int32), np.int32(2), TypeError),
        (_ALL_MAPS, np.int32(0), ValueError),
        (_ALL_MAPS, np.int32(6), ValueError),
        (_ALL_MAPS, np.float32(2), TypeError),
        (_ALL_MAPS, True, TypeError),
        (_ALL_MAPS, np.asarray([2], np.int32), ValueError),
    ],
)
def test_invalid_controls_fail_on_host(
    bank: EnvConfig, mask: Array, size: Array, error: type[Exception]
) -> None:
    with pytest.raises(error):
        validate_training_distribution(eligible_maps=mask, team_size=size)
    with pytest.raises(error):
        sample_training_configs(
            bank,
            jax.random.key(1),
            jnp.zeros(2, jnp.int32),
            eligible_maps=mask,
            team_size=size,
        )


@pytest.mark.parametrize(
    "mode",
    ["odd", "empty", "negative", "float", "root_batch", "rbg", "unsafe_rbg", "bank"],
)
def test_invalid_sampling_shapes_keys_and_generations(
    bank: EnvConfig, mode: str
) -> None:
    root = jax.random.key(
        1, impl=mode if mode in ("rbg", "unsafe_rbg") else "threefry2x32"
    )
    generations = jnp.zeros(2, jnp.int32)
    if mode == "odd":
        generations = jnp.zeros(3, jnp.int32)
    elif mode == "empty":
        generations = jnp.zeros(0, jnp.int32)
    elif mode == "negative":
        generations = jnp.asarray([0, -1], jnp.int32)
    elif mode == "float":
        generations = jnp.zeros(2, jnp.float32)
    elif mode == "root_batch":
        root = jax.random.split(root, 2)
    elif mode == "bank":
        bank = _row(bank, 0)
    with pytest.raises((TypeError, ValueError)):
        sample_training_configs(
            bank,
            root,
            generations,
            eligible_maps=jnp.asarray(_ALL_MAPS),
            team_size=jnp.asarray(2, jnp.int32),
        )


def test_key_step_requirements_and_host_validation_boundary() -> None:
    root, generations = jax.random.key(7), jnp.zeros(2, jnp.int32)
    for stream, steps in (
        ("action", None),
        ("step", None),
        ("map", generations),
        ("missing", None),
    ):
        with pytest.raises(ValueError):
            training_keys(
                root,
                generations,
                stream=cast(sampling.TrainingStream, stream),
                decision_step=steps,
            )
    with pytest.raises(ValueError):
        training_keys(
            root, generations, stream="action", decision_step=jnp.zeros(3, jnp.int32)
        )
    with pytest.raises(ValueError):
        training_keys(
            root,
            generations,
            stream="step",
            decision_step=jnp.asarray([-1, 0], jnp.int32),
        )
    with pytest.raises(TypeError, match="host"):
        jax.jit(validate_training_distribution)(
            eligible_maps=jnp.asarray(_ALL_MAPS), team_size=jnp.asarray(2, jnp.int32)
        )


def test_shared_roster_helper_preserves_sentinel_and_generic_repeats(
    bank: EnvConfig,
) -> None:
    source = _row(bank, 0)
    sentinel = jnp.full(10, -1, jnp.int32)
    retained, valid = _source_config_with_class_ids(source, sentinel)
    assert bool(valid)
    _assert_tree(retained, source)
    classes = jnp.asarray([2, 2, 0, 0, 0, 5, 5, 5, 0, 0], jnp.int32)
    changed, valid = _source_config_with_class_ids(source, classes)
    assert bool(valid)
    _assert_tree(
        changed.agent_profile,
        resolve_agent_profile(classes, jnp.asarray([2, 3], jnp.int32)),
    )
    empty, valid = _source_config_with_class_ids(source, jnp.zeros(10, jnp.int32))
    assert bool(valid) and not bool(jnp.any(empty.agent_profile.active_mask))


def test_roster_helper_mixed_rows_and_external_mapping(bank: EnvConfig) -> None:
    def first_two(value: Array) -> Array:
        return value[:2]

    sources = jax.tree.map(first_two, bank)
    classes = jnp.asarray([[-1] * 10, [1, 3, 0, 0, 0, 2, 2, 0, 0, 0]], jnp.int32)
    resolved, valid = _source_config_with_class_ids(sources, classes)
    np.testing.assert_array_equal(valid, [True, True])
    _assert_tree(_row(resolved, 0), _row(bank, 0))
    mapped = cast(
        tuple[EnvConfig, Array],
        jax.jit(jax.vmap(_source_config_with_class_ids))(sources, classes),
    )
    _assert_tree(mapped, (resolved, valid))
    with pytest.raises(ValueError, match="shape"):
        _source_config_with_class_ids(sources, classes[0])


def test_canonical_order_is_read_from_its_authority(
    bank: EnvConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_a, original_b = canonical_tournament_rosters()

    def reversed_rosters() -> tuple[
        tuple[tasks.AgentClassName, ...], tuple[tasks.AgentClassName, ...]
    ]:
        return original_a[::-1], original_b[::-1]

    monkeypatch.setattr(sampling, "canonical_tournament_rosters", reversed_rosters)

    def draw(size: Array) -> SampledTrainingConfigs:
        return sample_training_configs(
            bank,
            jax.random.key(18),
            jnp.zeros(8, jnp.int32),
            eligible_maps=jnp.asarray(_ALL_MAPS),
            team_size=size,
        )

    call = jax.jit(draw)
    for size in (2, 5):
        result = cast(SampledTrainingConfigs, call(jnp.asarray(size, jnp.int32)))
        for row in np.asarray(result.source_class_ids).reshape(-1, 5):
            active = row[:size].tolist()
            assert active == sorted(active, reverse=True)
            np.testing.assert_array_equal(row[size:], 0)


@pytest.mark.parametrize(
    "classes",
    [
        np.asarray([1, 0, 2, 0, 0, 3, 0, 0, 0, 0], np.int32),
        np.asarray([1, -1, 0, 0, 0, 3, 0, 0, 0, 0], np.int32),
        np.full(10, 6, np.int32),
        np.full(10, 2**32 + 1, np.int64),
        np.full(10, 2**32 - 1, np.uint32),
        np.ones(10, bool),
        np.ones(10, np.float32),
    ],
)
def test_bad_roster_declarations_fail_before_catalog_or_narrowing(
    bank: EnvConfig, classes: Array
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _source_config_with_class_ids(_row(bank, 0), classes)
    if classes.dtype in (np.dtype(np.int32), np.dtype(np.uint32)):
        result, valid = cast(
            tuple[EnvConfig, Array],
            jax.jit(_source_config_with_class_ids)(_row(bank, 0), jnp.asarray(classes)),
        )
        assert not bool(valid)
        _assert_tree(result, _row(bank, 0))


def test_compilation_reuse_and_no_assets_or_hashes(
    bank: EnvConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    traces: list[None] = []

    def traced(
        source: EnvConfig, key: Array, generations: Array, mask: Array, size: Array
    ) -> SampledTrainingConfigs:
        traces.append(None)
        return sample_training_configs(
            source, key, generations, eligible_maps=mask, team_size=size
        )

    def forbid(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sampling performed host content work")

    monkeypatch.setattr(tasks, "map_geometry", forbid)
    monkeypatch.setattr(tasks, "asset_manifest", forbid)
    monkeypatch.setattr(tasks, "canonical_digest_sha256", forbid)
    monkeypatch.setattr(_tdm_assets, "map_geometry", forbid)
    call = jax.jit(traced)
    for size in range(1, 6):
        updated = bank._replace(max_steps=bank.max_steps + size)
        jax.block_until_ready(
            cast(
                SampledTrainingConfigs,
                call(
                    updated,
                    jax.random.key(size),
                    jnp.full(4, size, jnp.int32),
                    jnp.arange(42) < size,
                    jnp.asarray(size, jnp.int32),
                ),
            )
        )
    assert len(traces) == 1
    lowered = call.lower(
        bank,
        jax.random.key(0),
        jnp.zeros(4, jnp.int32),
        jnp.asarray(_ALL_MAPS),
        jnp.asarray(2, jnp.int32),
    ).as_text()
    assert "callback" not in lowered.lower()


def test_public_partial_reset_adopts_only_finished_lanes(bank: EnvConfig) -> None:
    first = _draw(bank, 1, batch=2)
    initial_config = first.config._replace(max_steps=jnp.asarray([1, 3], jnp.int32))
    env = marl_bgs.make("tdm", env_config=_row(bank, 0), num_envs=2, metrics="none")
    observations, state = env.reset(
        training_keys(jax.random.key(91), jnp.zeros(2, jnp.int32), stream="reset"),
        initial_config,
    )

    def maybe_reset(
        before: EnvironmentState, indices: Array, classes: Array
    ) -> tuple[EnvironmentState, Array, Array, Array]:
        def reset() -> tuple[EnvironmentState, Array, Array, Array]:
            following = before.reset_generation + before.done.done.astype(jnp.int32)
            sampled = sample_training_configs(
                bank,
                jax.random.key(91),
                following,
                eligible_maps=jnp.arange(42) == 41,
                team_size=jnp.asarray(4, jnp.int32),
            )
            _, after = env.reset_done(
                training_keys(jax.random.key(91), following, stream="reset"),
                before,
                sampled.config,
            )
            mask = before.done.done
            return (
                after,
                jnp.where(mask, sampled.source_indices, indices),
                jnp.where(mask[:, None], sampled.source_class_ids, classes),
                jnp.asarray(1),
            )

        return cast(
            tuple[EnvironmentState, Array, Array, Array],
            jax.lax.cond(
                jnp.any(before.done.done),
                reset,
                lambda: (before, indices, classes, jnp.asarray(0)),
            ),
        )

    reset = jax.jit(maybe_reset)
    retained = cast(
        tuple[EnvironmentState, Array, Array, Array],
        reset(state, first.source_indices, first.source_class_ids),
    )
    _assert_tree(retained[:3], (state, first.source_indices, first.source_class_ids))
    assert int(retained[3]) == 0
    zeros = jnp.zeros((2, 10), jnp.int32)
    _, ended, _, _, _ = cast(
        StepResult,
        jax.jit(env.step)(jax.random.key(4), state, Action(zeros, zeros, zeros)),
    )
    np.testing.assert_array_equal(ended.done.done, [True, False])
    after, indices, classes, calls = cast(
        tuple[EnvironmentState, Array, Array, Array],
        reset(ended, first.source_indices, first.source_class_ids),
    )
    assert int(calls) == 1 and int(indices[0]) == 41
    _assert_tree(_row(after.core_state, 1), _row(ended.core_state, 1))
    _assert_tree(_row(after.config, 1), _row(ended.config, 1))
    np.testing.assert_array_equal(classes[1], first.source_class_ids[1])
    assert int(indices[1]) == int(first.source_indices[1])
    np.testing.assert_array_equal(after.reset_generation, [1, 0])
    assert np.asarray(observations.observation.self_features).shape[0] == 2
