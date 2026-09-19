"""Check System assignment durability and first-transition source verification.

These host writer tests use one real environment transition as a shape/meaning
fixture. Synthetic action epochs then isolate trace bounds, gaps, rejected joins,
source claims and failure-before-publication without running long games.
"""

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action, EnvConfig
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.models import canonical_json_bytes
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace, System
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.run_writer import RunWriter, configuration_identity
from marl_battlegrounds.tasks import (
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)


def _never_call(*_args: object) -> object:
    raise AssertionError("writer executed the System")


def _systems() -> dict[str, object]:
    system = System(
        "router", _never_call, components=({"name": "one"}, {"name": "two"})
    )
    return {"team_a": system, "team_b": system}


@pytest.fixture(scope="module")
def first_info() -> EpisodeInfo:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("mage",), team_b_roster=("priest",), max_steps=8
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    _, state = env.reset(jax.random.key(0))
    zeros = jnp.zeros(10, dtype=jnp.int32)
    info = cast(
        EpisodeInfo,
        jax.jit(env.step)(jax.random.key(1), state, Action(zeros, zeros, zeros))[4],
    )
    assert not bool(info.completed) and int(info.decision_step) == 0
    return info


def _trace(info: EpisodeInfo, choice: int = 0) -> PolicyTrace:
    return PolicyTrace(
        info.episode_id,
        info.decision_step,
        jnp.ones_like(info.completed, dtype=jnp.bool_),
        jnp.where(info.active_mask, jnp.int32(choice), jnp.int32(-1)),
    )


def _epoch(info: EpisodeInfo, decision: int, episode: int = 1) -> EpisodeInfo:
    return info._replace(
        episode_id=jnp.asarray(episode, jnp.int32),
        decision_step=jnp.asarray(decision, jnp.int32),
        episode_length=jnp.asarray(decision + 1, jnp.int32),
    )


def _manifest(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((path / "run_details.json").read_text()))


def _pass(path: Path) -> dict[str, Any]:
    return next(iter(_manifest(path)["passes"].values()))


def _rows(path: Path) -> list[dict[str, str]]:
    with (path / "policy_assignments.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def _start(
    info: EpisodeInfo,
    source: EnvConfig | None,
    *,
    index: int = 0,
    spawn: int = 0,
    authored: bool = False,
) -> EpisodeStartRecords:
    if source is None:
        words = np.zeros(8, dtype=np.uint32)
    else:
        config_id = configuration_identity(source)[0]
        bank_id = sha256(
            json.dumps([config_id], separators=(",", ":")).encode() + b"\n"
        ).hexdigest()
        words = np.frombuffer(bytes.fromhex(bank_id), dtype=">u4").astype(np.uint32)
    return EpisodeStartRecords(
        info.episode_id,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(words),
        jnp.asarray(index if source is not None else -1, jnp.int32),
        jnp.asarray(spawn if source is not None else -1, jnp.int32),
        jnp.asarray(-1, jnp.int32),
        jnp.asarray(source is not None),
        jnp.asarray(authored),
        jnp.asarray(True),
    )


@pytest.mark.parametrize("spawn,authored", [(0, False), (1, False), (0, True)])
def test_sampled_roster_records_exact_bank_and_resolved_identity(
    tmp_path: Path, first_info: EpisodeInfo, spawn: int, authored: bool
) -> None:
    a, b = canonical_tournament_rosters()
    source = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=a, team_b_roster=b, max_steps=8
    )
    classes = first_info.config.agent_profile.class_ids
    starts = _start(first_info, source, spawn=spawn, authored=authored)._replace(
        source_class_ids=classes
    )
    actual = first_info.config
    if spawn:
        actual = actual._replace(
            team_spawn_pad_positions=actual.team_spawn_pad_positions[::-1]
        )
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(starts, source_configs=source)
        writer.write(first_info._replace(config=actual, episode_start_records=starts))
        writer.flush()
        path = writer.run_dir
        saved = _pass(path)["episode_starts"]["1"]
        assert saved["source_class_ids"] == np.asarray(classes).tolist()
        assert saved["verification"] == ("custom" if authored else "verified")
        assert saved["resolved_config_id"] == configuration_identity(actual)[0]
        source_id = configuration_identity(source)[0]
        assert _manifest(path)["source_banks"][saved["source_table_id"]] == [source_id]
    with RunWriter(resume_from=path) as resumed:
        assert _pass(resumed.run_dir)["episode_starts"]["1"] == saved


def test_relationship_cache_cannot_accept_a_changed_roster_or_spawn(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    a, b = canonical_tournament_rosters()
    source = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=a, team_b_roster=b, max_steps=8
    )
    classes = first_info.config.agent_profile.class_ids
    for change in ("roster", "spawn"):
        with RunWriter(tmp_path / change) as writer:
            start = _start(first_info, source)._replace(source_class_ids=classes)
            writer.register_episodes(start, source_configs=source)
            writer.write(first_info._replace(episode_start_records=start))
            second = _epoch(first_info, 0, episode=2)
            bad = _start(second, source, spawn=1 if change == "spawn" else 0)._replace(
                source_class_ids=classes.at[0].set(2) if change == "roster" else classes
            )
            writer.register_episodes(bad, source_configs=source)
            with pytest.raises(ValueError, match="declared source/spawn"):
                writer.write(second._replace(episode_start_records=bad))
            saved = _pass(writer.run_dir)["episode_starts"]
            assert saved["1"]["verification"] == "verified"
            assert saved["2"]["verification"] == "pending"


@pytest.mark.parametrize("initial_override", [False, True])
@pytest.mark.parametrize("durability", ["pending", "flushed", "resumed"])
def test_repeated_start_cannot_add_or_remove_explicit_roster(
    tmp_path: Path, first_info: EpisodeInfo, initial_override: bool, durability: str
) -> None:
    base = _start(first_info, first_info.config)
    explicit = base._replace(source_class_ids=first_info.config.agent_profile.class_ids)
    first, changed = (explicit, base) if initial_override else (base, explicit)
    writer = RunWriter(tmp_path)
    try:
        writer.register_episodes(first, source_configs=first_info.config)
        if durability == "flushed":
            writer.flush()
        elif durability == "resumed":
            path = writer.run_dir
            writer.close()
            writer = RunWriter(resume_from=path)
        before = (writer.run_dir / "run_details.json").read_bytes()
        with pytest.raises(ValueError, match="recorded start"):
            writer.register_episodes(changed, source_configs=first_info.config)
        assert (writer.run_dir / "run_details.json").read_bytes() == before
        with pytest.raises(ValueError, match="registered declaration"):
            writer.write(first_info._replace(episode_start_records=changed))
    finally:
        writer.close()


def test_saved_source_relationships_are_checked_once_per_distinct_content(
    tmp_path: Path, first_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import run_writer

    a, b = canonical_tournament_rosters()
    source = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=a, team_b_roster=b, max_steps=8
    )
    with RunWriter(tmp_path) as writer:
        for identifier in (1, 2):
            info = _epoch(first_info, 0, episode=identifier)
            start = _start(info, source)._replace(
                source_class_ids=info.config.agent_profile.class_ids
            )
            writer.register_episodes(start, source_configs=source)
            writer.write(info._replace(episode_start_records=start))
        path = writer.run_dir
    checks: list[None] = []
    compiled = run_writer._source_relationship_check()  # pyright: ignore[reportPrivateUsage]

    def counted(
        actual: EnvConfig, bank: EnvConfig, classes: jax.Array
    ) -> tuple[Any, Any]:
        checks.append(None)
        return compiled(actual, bank, classes)

    monkeypatch.setattr(run_writer, "_source_relationship_check", lambda: counted)
    with RunWriter(resume_from=path):
        pass
    assert len(checks) == 1


@pytest.mark.parametrize("early_resets", [False, True])
def test_assignment_capacity_flushes_without_completions(
    tmp_path: Path, first_info: EpisodeInfo, early_resets: bool
) -> None:
    with RunWriter(tmp_path, policies=_systems(), buffer_size=1) as writer:
        for decision in range(26):
            info = _epoch(
                first_info,
                0 if early_resets else decision,
                decision + 1 if early_resets else 1,
            )
            writer.write(info, policy_trace=_trace(info, decision % 2))
            assert (
                len(writer._assignment_open)  # pyright: ignore[reportPrivateUsage]
                + len(  # pyright: ignore[reportPrivateUsage]
                    writer._rows["policy_assignments.csv"]  # pyright: ignore[reportPrivateUsage]
                )
                <= 10
            )
        saved = _manifest(writer.run_dir)
        assert saved["tables"]["policy_assignments.csv"]["rows"] >= 50
        assert _pass(writer.run_dir)["completed_episode_ids"] == []
        path = writer.run_dir
    rows = _rows(path)
    assert len(rows) == 52
    assert not (path / "episodes.csv").exists()
    assert {row["global_slot"] for row in rows} == {"0", "5"}
    assert len({row["episode_id"] for row in rows}) == (26 if early_resets else 1)


def test_trace_resume_continuity_gaps_and_duplicate_rejection(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    with RunWriter(tmp_path, policies=_systems(), buffer_size=4) as writer:
        path = writer.run_dir
        for decision in (0, 1):
            info = _epoch(first_info, decision)
            writer.write(info, policy_trace=_trace(info))
    with RunWriter(resume_from=path, policies=_systems(), buffer_size=4) as writer:
        for decision in (2, 4):
            info = _epoch(first_info, decision)
            writer.write(info, policy_trace=_trace(info))
        writer.flush()
        before = (path / "policy_assignments.csv").read_bytes()
        with pytest.raises(ValueError, match="without duplicates"):
            writer.write(info, policy_trace=_trace(info))
        assert (path / "policy_assignments.csv").read_bytes() == before
    slot_rows = [row for row in _rows(path) if row["global_slot"] == "0"]
    assert [(row["decision_start"], row["decision_stop"]) for row in slot_rows] == [
        ("0", "2"),
        ("2", "3"),
        ("4", "5"),
    ]


def test_unchanged_assignments_at_capacity_extend_without_forced_flush(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    with RunWriter(tmp_path, policies=_systems(), buffer_size=1) as writer:
        path = writer.run_dir
        for decision in range(16):
            for episode in range(1, 6):
                info = _epoch(first_info, decision, episode)
                writer.write(info, policy_trace=_trace(info))
        assert "policy_assignments.csv" not in _manifest(path)["tables"]
    rows = _rows(path)
    assert len(rows) == 10
    assert {(row["decision_start"], row["decision_stop"]) for row in rows} == {
        ("0", "16")
    }


@pytest.mark.parametrize("failure", ["component", "epoch", "lifecycle", "tracking"])
def test_later_bad_lane_rejects_whole_call_before_replay_or_assignments(
    tmp_path: Path,
    first_info: EpisodeInfo,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def pair(value: jax.Array) -> jax.Array:
        return jnp.stack((value, value))

    records = jax.tree.map(pair, first_info)
    records = records._replace(episode_id=jnp.array([1, 2], dtype=jnp.int32))
    trace = _trace(records)
    if failure == "component":
        trace = trace._replace(policy_ids=trace.policy_ids.at[1, 5].set(20))
    elif failure == "epoch":
        trace = trace._replace(decision_step=trace.decision_step.at[1].set(1))
    elif failure == "lifecycle":
        records = records._replace(lifecycle_error=jnp.array([False, True]))
    else:
        records = records._replace(episode_tracking_error=jnp.array([0, 8], jnp.int32))
    replay_calls: list[object] = []
    with RunWriter(tmp_path, policies=_systems(), buffer_size=1) as writer:
        monkeypatch.setattr(writer, "write_replay", replay_calls.append)
        records = records._replace(replay=cast(Any, object()))
        with pytest.raises(ValueError):
            writer.write(records, policy_trace=trace)
        assert replay_calls == []
        assert "policy_assignments.csv" not in _manifest(writer.run_dir)["tables"]


@pytest.mark.parametrize("spawn, authored", [(0, False), (1, False), (0, True)])
def test_start_stays_pending_until_actual_first_transition(
    tmp_path: Path, first_info: EpisodeInfo, spawn: int, authored: bool
) -> None:
    source = first_info.config
    start = _start(first_info, source, spawn=spawn, authored=authored)
    resolved = (
        source._replace(team_spawn_pad_positions=source.team_spawn_pad_positions[::-1])
        if spawn
        else source
    )
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(start, source_configs=source)
        assert _pass(writer.run_dir)["episode_starts"]["1"]["verification"] == "pending"
        writer.write(first_info._replace(config=resolved, episode_start_records=start))
        writer.flush()
        saved = _pass(writer.run_dir)["episode_starts"]["1"]
        assert saved["verification"] == ("custom" if authored else "verified")
        assert saved["resolved_config_id"] == configuration_identity(resolved)[0]
        assert not (writer.run_dir / "episodes.csv").exists()


def test_unknown_custom_start_and_changed_reset_generation(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    start = _start(first_info, None)
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(start)
        writer.write(first_info._replace(episode_start_records=start))
        writer.flush()
        assert _pass(writer.run_dir)["episode_starts"]["1"]["verification"] == "custom"
        with pytest.raises(ValueError, match="differs from its recorded start"):
            writer.register_episodes(start._replace(reset_generation=jnp.int32(1)))


@pytest.mark.parametrize("failure", ["index", "bank", "nonpad", "epoch", "generation"])
def test_source_claims_fail_without_verified_publication(
    tmp_path: Path, first_info: EpisodeInfo, failure: str
) -> None:
    source = first_info.config
    start = _start(first_info, source)
    with RunWriter(tmp_path) as writer:
        if failure in {"index", "bank"}:
            bad = (
                start._replace(source_index=jnp.int32(1))
                if failure == "index"
                else start
            )
            with pytest.raises(ValueError, match="source bank or source index"):
                writer.register_episodes(
                    bad, source_configs=source if failure == "index" else None
                )
            assert _pass(writer.run_dir)["episode_starts"] == {}
            return
        writer.register_episodes(start, source_configs=source)
        evidence = first_info._replace(episode_start_records=start)
        if failure == "nonpad":
            evidence = evidence._replace(config=source._replace(max_steps=9))
        elif failure == "epoch":
            evidence = evidence._replace(decision_step=jnp.int32(1))
        else:
            evidence = evidence._replace(
                episode_start_records=start._replace(reset_generation=jnp.int32(1))
            )
        with pytest.raises(ValueError):
            writer.write(evidence)
        assert _pass(writer.run_dir)["episode_starts"]["1"]["verification"] == "pending"


def test_changed_mutable_source_bank_cannot_reuse_old_content_id(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    def mutable(leaf: jax.Array) -> np.ndarray[Any, Any]:
        return np.array(leaf, copy=True)

    source = jax.tree.map(mutable, first_info.config)
    start = _start(first_info, source)
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(start, source_configs=source)
        source.team_spawn_pad_positions[0, 0, 0] += 1
        before = canonical_json_bytes(_manifest(writer.run_dir))
        with pytest.raises(ValueError):
            writer.register_episodes(start, source_configs=source)
        assert canonical_json_bytes(_manifest(writer.run_dir)) == before


def test_scalar_trace_batch_one_and_terminal_padding_join(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    def batch(value: jax.Array) -> jax.Array:
        return value[None]

    with RunWriter(tmp_path, policies=_systems()) as writer:
        writer.write(first_info, policy_trace=jax.tree.map(batch, _trace(first_info)))
        padded = first_info._replace(decision_step=jnp.int32(-1))
        with pytest.raises(ValueError, match="action epoch"):
            writer.write(padded, policy_trace=_trace(first_info))


def test_immutable_source_bank_cache_skips_repeated_hashing(
    tmp_path: Path, first_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import run_writer

    def array(leaf: object) -> jax.Array:
        return jnp.asarray(leaf)

    source = jax.tree.map(array, first_info.config)
    start = _start(first_info, source)
    count = 0
    original = run_writer.configuration_identity

    def counted(config: EnvConfig) -> tuple[str, dict[str, object]]:
        nonlocal count
        count += 1
        return original(config)

    with RunWriter(tmp_path) as writer:
        monkeypatch.setattr(run_writer, "configuration_identity", counted)
        writer.register_episodes(start, source_configs=source)
        writer.register_episodes(start, source_configs=source)
        assert count == 1
        writer.register_episodes(start)
        assert count == 1


def test_valid_single_choice_does_not_validate_unused_exchange(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    from marl_battlegrounds.core.config import (
        resolve_agent_profile,
        validate_env_config,
    )

    source = first_info.config._replace(
        task_mode=0,
        team_deathmatch_score_threshold=0,
        agent_profile=resolve_agent_profile(
            jnp.array([5] + [0] * 9, jnp.int32), jnp.array([1, 0], jnp.int32)
        ),
        obstacles=jnp.zeros_like(first_info.config.obstacles),
        team_spawn_pad_positions=first_info.config.team_spawn_pad_positions.at[
            1, :, 0
        ].set(0),
    )
    from marl_battlegrounds.evaluation.recording_context import restore_recording_config

    source = restore_recording_config(source)
    validate_env_config(source)
    with pytest.raises(ValueError, match="radius-adjusted map bounds"):
        validate_env_config(
            source._replace(
                team_spawn_pad_positions=source.team_spawn_pad_positions[::-1]
            )
        )
    info = first_info._replace(config=source)
    start = _start(info, source)
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(start, source_configs=source)
        writer.write(info._replace(episode_start_records=start))
        writer.flush()
        assert (
            _pass(writer.run_dir)["episode_starts"]["1"]["verification"] == "verified"
        )


def test_assignment_suffix_recovers_with_its_last_durable_epoch(
    tmp_path: Path, first_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import run_writer

    with RunWriter(tmp_path, policies=_systems()) as writer:
        path = writer.run_dir
        writer.write(first_info, policy_trace=_trace(first_info))
        writer.flush()
        original = (path / "policy_assignments.csv").read_bytes()
        info = _epoch(first_info, 1)
        writer.write(info, policy_trace=_trace(info))

        def fail_manifest(_path: Path, _value: object) -> None:
            raise OSError("injected manifest failure")

        with monkeypatch.context() as patch:
            patch.setattr(run_writer, "_atomic_json", fail_manifest)
            with pytest.raises(OSError, match="injected"):
                writer.flush()
        assert (path / "policy_assignments.csv").stat().st_size > len(original)
        assert _pass(path)["trace_epochs"] == {"1": 0}
    with RunWriter(resume_from=path, policies=_systems()) as writer:
        assert (path / "policy_assignments.csv").read_bytes() == original
        writer.write(info, policy_trace=_trace(info))
    assert _pass(path)["trace_epochs"] == {"1": 1}


def test_manifest_replacement_is_recovery_authority_after_later_failure(
    tmp_path: Path, first_info: EpisodeInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation import run_writer

    with RunWriter(tmp_path, policies=_systems()) as writer:
        path = writer.run_dir
        writer.write(first_info, policy_trace=_trace(first_info))
        publish = run_writer._atomic_json  # pyright: ignore[reportPrivateUsage]

        def fail_after_publish(path: Path, value: object) -> None:
            publish(path, value)
            raise OSError("injected failure after metadata replacement")

        with monkeypatch.context() as patch:
            patch.setattr(run_writer, "_atomic_json", fail_after_publish)
            with pytest.raises(OSError, match="after metadata"):
                writer.flush()
        committed = (path / "policy_assignments.csv").read_bytes()
        assert _pass(path)["trace_epochs"] == {"1": 0}
    with RunWriter(resume_from=path, policies=_systems()) as writer:
        assert (path / "policy_assignments.csv").read_bytes() == committed
        following = _epoch(first_info, 1)
        writer.write(following, policy_trace=_trace(following))
    assert _pass(path)["trace_epochs"] == {"1": 1}


@pytest.mark.parametrize("mismatch", ["schema", "system", "schedule", "capture"])
def test_incompatible_resume_preserves_interrupted_assignment_suffix(
    tmp_path: Path, first_info: EpisodeInfo, mismatch: str
) -> None:
    details: dict[str, object] = {"schedule_digest": "a" * 64, "metrics": "none"}
    with RunWriter(tmp_path, policies=_systems(), details=details) as writer:
        writer.write(first_info, policy_trace=_trace(first_info))
        path = writer.run_dir
    with (path / "policy_assignments.csv").open("ab") as stream:
        stream.write(b"unfinished assignment")
    policies = _systems()
    if mismatch == "schema":
        manifest = _manifest(path)
        manifest["schema_version"] = 1
        (path / "run_details.json").write_text(json.dumps(manifest))
    elif mismatch == "system":
        policies["team_a"] = System("different", _never_call)
    elif mismatch == "schedule":
        details = {**details, "schedule_digest": "b" * 64}
    else:
        details = {**details, "metrics": "full"}
    before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    with pytest.raises(ValueError, match=r"schema|identity"):
        RunWriter(resume_from=path, policies=policies, details=details)
    after = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    assert after == before


def test_required_none_summary_does_not_claim_priority_coverage(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    complete = first_info._replace(completed=jnp.asarray(True), outcome=jnp.int32(3))
    with RunWriter(tmp_path, details={"metrics": "none"}) as writer:
        writer.write(complete)
        writer.flush()
        assert _pass(writer.run_dir)["recorded_metrics_by_episode"] == {"1": "none"}
        assert (writer.run_dir / "episodes.csv").is_file()
        assert not (writer.run_dir / "priority_metrics.csv").exists()


def test_reordered_source_bank_fails_before_declaration_changes(
    tmp_path: Path, first_info: EpisodeInfo
) -> None:
    source = first_info.config
    other = source._replace(max_steps=jnp.int32(9))

    def stack(left: object, right: object) -> jax.Array:
        return jnp.stack((jnp.asarray(left), jnp.asarray(right)))

    bank = jax.tree.map(stack, source, other)
    references = [configuration_identity(config)[0] for config in (source, other)]
    digest = sha256(
        json.dumps(references, separators=(",", ":")).encode() + b"\n"
    ).digest()
    words = jnp.asarray(np.frombuffer(digest, dtype=">u4").astype(np.uint32))
    start = _start(first_info, source)._replace(source_table_id=words)
    with RunWriter(tmp_path) as writer:
        writer.register_episodes(start, source_configs=bank)
        before = canonical_json_bytes(_manifest(writer.run_dir))

        def reverse(leaf: jax.Array) -> jax.Array:
            return leaf[::-1]

        with pytest.raises(ValueError, match="bank"):
            writer.register_episodes(start, source_configs=jax.tree.map(reverse, bank))
        assert canonical_json_bytes(_manifest(writer.run_dir)) == before


@pytest.mark.parametrize("resume", [False, True])
def test_completion_cannot_precede_a_recorded_trace(
    tmp_path: Path,
    first_info: EpisodeInfo,
    resume: bool,
) -> None:
    writer = RunWriter(tmp_path, policies=_systems())
    late = _epoch(first_info, 5)
    writer.write(late, policy_trace=_trace(late))
    writer.flush()
    path = writer.run_dir
    if resume:
        writer.close()
        writer = RunWriter(resume_from=path, policies=_systems())
    early = _epoch(first_info, 1)._replace(
        completed=jnp.asarray(True), outcome=jnp.int32(3)
    )
    with pytest.raises(ValueError, match="completion precedes"):
        writer.write(early)
    writer.close()
    assert not (path / "episodes.csv").exists()
    assert _pass(path)["completed_episode_ids"] == []


def test_same_call_trace_after_completion_rejects_before_publication(
    tmp_path: Path,
    first_info: EpisodeInfo,
) -> None:
    early = _epoch(first_info, 1)._replace(
        completed=jnp.asarray(True), outcome=jnp.int32(3)
    )
    late = _epoch(first_info, 5)

    def stack(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.stack((a, b))

    infos = jax.tree.map(stack, early, late)
    traces = _trace(infos)._replace(valid=jnp.asarray([False, True]))
    with RunWriter(tmp_path, policies=_systems()) as writer:
        with pytest.raises(ValueError, match="follows the supplied"):
            writer.write(infos, policy_trace=traces)
        path = writer.run_dir
    assert not list(path.glob("*.csv"))


def test_trace_only_call_must_match_declared_configuration(
    tmp_path: Path,
    first_info: EpisodeInfo,
) -> None:
    with RunWriter(tmp_path, policies=_systems()) as writer:
        writer.register_episodes([{"episode_id": 1, "configuration_digest": "0" * 64}])
        with pytest.raises(ValueError, match="recorded episode evidence"):
            writer.write(first_info, policy_trace=_trace(first_info))
        path = writer.run_dir
    assert not (path / "policy_assignments.csv").exists()


@pytest.mark.parametrize("field", ["episode_id", "decision_step", "policy_ids"])
def test_scalar_trace_rejects_extra_rows_in_any_field(
    tmp_path: Path,
    first_info: EpisodeInfo,
    field: str,
) -> None:
    def batch(value: jax.Array) -> jax.Array:
        return value[None]

    trace = jax.tree.map(batch, _trace(first_info))
    extra = jnp.repeat(getattr(trace, field), 2, axis=0)
    trace = trace._replace(**{field: extra})
    with RunWriter(tmp_path, policies=_systems()) as writer:
        with pytest.raises(ValueError, match="exactly one batch row"):
            writer.write(first_info, policy_trace=trace)
        assert not list(writer.run_dir.glob("*.csv"))
