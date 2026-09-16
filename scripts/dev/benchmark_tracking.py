"""Measure Packet 4 tracking and reset costs through the shared benchmark CLI.

Use ``python -m scripts.dev.benchmark_evaluation --tracking --output PATH``.
Each selected case uses batch 32, sixteen decisions and five or more warm runs.
Short unequal 4/7-step games stress reset handling; these are not learning-speed
measurements. The manual case also runs against Packet 3's committed package.
Inputs and common retained outputs match; optional starts/final training data
are reported separately. This module never changes assets or production files.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

type Tree = Any

CASES = ("manual", "tracking", "autoreset", "starts", "training-state")


def run(args: argparse.Namespace) -> int:
    """Run one or all bounded tracking cases and write raw JSON evidence.

    args comes from benchmark_evaluation. package_root/assets_root freeze imports
    and maps; backend defaults to GPU. tracking_case defaults to all cases.
    Existing output files are rejected. Returns zero only after numerical checks,
    warm transfer guards and unchanged source/asset identities pass. Exceptions
    propagate with the partial case saved as failed. CPU is a diagnostic route.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.tasks import make_standard_team_deathmatch_config
    from scripts.dev.benchmark_evaluation import (
        _assert_equal,
        _bytes,
        _foundation_assets,
        _foundation_digest,
        _foundation_identity,
        _foundation_measure,
    )

    if (
        args.repeats < 5
        or args.sizes not in (None, [32])
        or args.lengths not in (None, [16])
    ):
        raise ValueError(
            "Tracking uses batch 32, rollout 16 and at least five warm samples"
        )
    if any(
        (
            args.foundations,
            args.systems,
            args.recording_contracts,
            args.metrics_only,
            args.include_scalar,
        )
    ):
        raise ValueError("Tracking cannot be combined with another benchmark mode")
    package = (args.package_root or Path(__file__).resolve().parents[2]).resolve()
    assets = (args.assets_root or package / "src/marl_battlegrounds/data/tdm").resolve()
    identity = _foundation_identity(package, assets)
    identity["tracking_harness_sha256"] = sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    _foundation_assets(assets)
    backend = args.backend or "gpu"
    device = jax.devices(backend)[0]
    args.output.mkdir(parents=True, exist_ok=True)
    common_reference: Any = None
    cases = (args.tracking_case,) if args.tracking_case else CASES
    for case in cases:
        path = args.output / f"tracking-{case}-{backend}.json"
        if path.exists():
            raise FileExistsError(
                f"Preserve prior evidence; choose a new output folder: {path}"
            )
        record: dict[str, Any] = {
            "status": "running",
            "case": case,
            "identity": identity,
            "batch": 32,
            "rollout_length": 16,
            "backend": backend,
            "device": str(device),
            "jax": jax.__version__,
            "scope": "Unequal short-game reset stress; fixed legal random methods",
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            with jax.default_device(device):
                started = time.perf_counter()
                roster_a, roster_b = marl_bgs.canonical_tournament_rosters()
                configs = [
                    make_standard_team_deathmatch_config(
                        map_id=12 if args.map_id is None else args.map_id,
                        max_steps=n,
                        team_a_roster=roster_a,
                        team_b_roster=roster_b,
                    )
                    for n in (4, 7)
                ]
                bank = jax.tree.map(lambda *values: jnp.stack(values), *configs)
                indices = jnp.arange(32, dtype=jnp.int32) % 2
                selected = jax.tree.map(
                    lambda values, indices=indices: values[indices], bank
                )
                prepared = marl_bgs.balanced_spawn_configs(selected, num_envs=32)
                base = marl_bgs.make(
                    "tdm", num_envs=32, env_config=prepared, metrics="priority"
                )
                automatic = case in ("autoreset", "starts", "training-state")
                tracked = case != "manual"
                env = (
                    marl_bgs.AutoReset(
                        base, include_training_state=case == "training-state"
                    )
                    if automatic
                    else base
                )
                _, initial = env.reset(jax.random.key(42))
                tracker = None
                if tracked:
                    tracker = marl_bgs.init_episode_tracking(
                        env,
                        initial,
                        source_configs=bank,
                        source_indices=indices,
                        record_starts=case == "starts",
                    ).begin_stage(initial, total_env_steps=512)
                method = marl_bgs.shared_policy(marl_bgs.policy("random"))
                memory = marl_bgs.init_systems(
                    method,
                    method,
                    env.get_observations(initial),
                    initial,
                    jax.random.key(44),
                )
                jax.block_until_ready((initial, tracker, memory))
                record["setup_including_validation_reset_tracking_ms"] = (
                    time.perf_counter() - started
                ) * 1000

                def rollout(
                    handle: Tree,
                    state: Tree,
                    tracking: Tree,
                    methods: Tree,
                    root: Tree,
                    *,
                    automatic: bool = automatic,
                    tracked: bool = tracked,
                    method: Tree = method,
                ) -> Tree:
                    """Keep state, checked counts and compact transition evidence.

                    The handle, source bank inside tracking, state, method memory
                    and root are dynamic. case selects only fixed program structure.
                    No host work or writer call occurs inside this scan.
                    """

                    def advance(carry: Tree, tick: Tree) -> tuple[Any, Any]:
                        """Choose once, retain old-step evidence, then reset."""
                        before, counts, binding, system_memory, rounds_valid = carry
                        action_key = jax.random.fold_in(root, tick * 2)
                        step_key = jax.random.fold_in(root, tick * 2 + 1)
                        actions, system_memory, _ = marl_bgs.apply_systems(
                            method,
                            method,
                            system_memory,
                            handle.get_observations(before),
                            before,
                            action_key,
                        )
                        result = handle.step(step_key, before, actions)
                        if tracked:
                            binding, result = marl_bgs.track_episode_step(
                                binding, before, result
                            )
                        obs, after, reward, done, info = result
                        valid = info.decision_step >= 0
                        if tracked:
                            counts = binding.stage_counts
                            rounds_valid = binding.full_batch_rounds_valid
                        else:
                            previous = counts.sum(axis=-1)
                            rounds_valid &= jnp.all(valid)
                            rounds_valid &= jnp.all(
                                before.cumulative_transition_count == previous
                            )
                            rounds_valid &= jnp.all(
                                after.cumulative_transition_count == previous + valid
                            )
                            category = (jnp.arange(32) >= 16).astype(jnp.int32)
                            counts = (
                                counts
                                + jax.nn.one_hot(category, 3, dtype=jnp.int32)
                                * valid[:, None]
                            )
                        if automatic:
                            final = info.final
                            old_features = final.observations.observation.self_features
                            old_masks = final.action_mask
                        else:
                            old_features = obs.observation.self_features
                            old_masks = after.action_mask
                            _, after = handle.reset_done(
                                jax.random.fold_in(step_key, 0x4155544F), after
                            )
                        common = (
                            actions,
                            reward,
                            done,
                            info.episode_id,
                            info.episode_length,
                            info.team_scores,
                            info.priority,
                            old_features,
                            old_masks,
                            info.completed,
                        )
                        extra = (
                            info.episode_start_records,
                            info.final.training_state if automatic else None,
                        )
                        return (after, counts, binding, system_memory, rounds_valid), (
                            common,
                            extra,
                        )

                    return jax.lax.scan(
                        advance,
                        (
                            state,
                            jnp.zeros((32, 3), jnp.int32),
                            tracking,
                            methods,
                            jnp.asarray(True),
                        ),
                        jnp.arange(16, dtype=jnp.int32),
                    )

                inputs = (env, initial, tracker, memory, jax.random.key(43))
                changed_bank = bank._replace(max_steps=bank.max_steps + 1)
                changed_selected = jax.tree.map(
                    lambda values, indices=indices: values[indices], changed_bank
                )
                changed_base = marl_bgs.make(
                    "tdm",
                    num_envs=32,
                    env_config=marl_bgs.balanced_spawn_configs(
                        changed_selected, num_envs=32
                    ),
                    metrics="priority",
                )
                changed_env = (
                    marl_bgs.AutoReset(
                        changed_base, include_training_state=case == "training-state"
                    )
                    if automatic
                    else changed_base
                )
                _, changed_state = changed_env.reset(jax.random.key(46))
                changed_tracker = (
                    marl_bgs.init_episode_tracking(
                        changed_env,
                        changed_state,
                        source_configs=changed_bank,
                        source_indices=indices,
                        record_starts=case == "starts",
                    ).begin_stage(changed_state, total_env_steps=512)
                    if tracked
                    else None
                )
                result, timing, _ = _foundation_measure(
                    rollout,
                    inputs,
                    args.repeats,
                    (
                        (env, initial, tracker, memory, jax.random.key(45)),
                        (
                            changed_env,
                            changed_state,
                            changed_tracker,
                            memory,
                            jax.random.key(47),
                        ),
                    ),
                )
                if timing["new_traces_for_changed_values"]:
                    raise AssertionError("Same-shaped values caused another trace")
                (
                    (state, counts, final_tracker, final_memory, rounds_valid),
                    (history, extras),
                ) = result
                expected = np.full(32, 16, np.int32)
                np.testing.assert_array_equal(np.asarray(counts).sum(axis=-1), expected)
                np.testing.assert_array_equal(
                    np.asarray(counts).sum(axis=0), [256, 256, 0]
                )
                if not bool(rounds_valid):
                    raise AssertionError(
                        "A padded or inconsistent round cannot complete"
                    )
                np.testing.assert_array_equal(
                    np.asarray(state.cumulative_transition_count), expected
                )
                common = (state, counts, final_memory, history)

                def key_words(value: Tree) -> Tree:
                    """Convert typed keys to their uint32 words for evidence only."""
                    return (
                        jax.random.key_data(value)
                        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
                        else value
                    )

                common = jax.tree.map(key_words, common)
                if common_reference is None:
                    common_reference = common
                else:
                    _assert_equal(common_reference, common)
                record["timing"] = timing
                record["real_transitions"] = 512
                record["transitions_per_second"] = 512000 / timing["warm_median_ms"]
                record["common_trajectory_sha256"] = _foundation_digest(common)
                record["retained_common_bytes"] = _bytes(common)
                record["optional_history_bytes"] = _bytes(extras)
                record["tracker_including_bank_bytes"] = _bytes(final_tracker)
                record["source_bank_bytes"] = _bytes(bank)
                if automatic:
                    from marl_battlegrounds.types import Action

                    zero = jnp.zeros((32, 10), jnp.int32)
                    info_shape = jax.eval_shape(
                        env.step, jax.random.key(91), initial, Action(zero, zero, zero)
                    )[4]
                    record["compact_final_record_bytes_per_step"] = _bytes(
                        info_shape.final
                    )
                else:
                    record["compact_final_record_bytes_per_step"] = 0
                record["manual_reference_scope"] = (
                    "Fixed setup-validated sources, retained configs across resets, "
                    "per-lane spawn counts, full-round and cumulative checks, exact "
                    "host budget/category check. General declaration/error handling "
                    "is an added tracker capability, covered by correctness tests."
                )
                if tracked:
                    started = time.perf_counter()
                    record["stage_summary"] = final_tracker.stage_summary(state)
                    record["stage_boundary_ms"] = (time.perf_counter() - started) * 1000
                    if record["stage_summary"]["status"] != "complete":
                        raise AssertionError(record["stage_summary"])
                record["device_memory_stats"] = device.memory_stats()
                record["process_peak_ram_bytes"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
                record["memory_scope"] = (
                    "Allocator/process peaks include setup and previous cases; "
                    "compiler memory is per executable. Logical bytes count aliases."
                )
            after = _foundation_identity(package, assets)
            after["tracking_harness_sha256"] = sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
            if identity != after:
                raise RuntimeError("Source or assets changed during measurement")
            record["status"] = "complete"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            path.write_text(json.dumps(record, indent=2) + "\n")
    return 0
