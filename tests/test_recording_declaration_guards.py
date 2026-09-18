"""Keep recorded episode owners and source claims fixed across writer calls.

One real transition provides the recorded configuration and action epoch. Late
schedule/source declarations must fail before flushing earlier records or adding
metadata. Identical declarations remain safe before and after flush and resume.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.policy_execution import PolicyTrace, System
from marl_battlegrounds.evaluation.recording_types import EpisodeStartRecords
from marl_battlegrounds.evaluation.run_writer import (
    ASSIGNMENT_COLUMNS,
    RunWriter,
    configuration_identity,
)
from marl_battlegrounds.evaluation.scalar_reports import iter_policy_assignments
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

type Durability = Literal["pending", "flushed", "resumed"]


def _never_call(*_args: object) -> object:
    raise AssertionError("recording must not execute a method")


def _systems() -> dict[str, object]:
    method = System("Original", _never_call, components=({"name": "Original"},))
    return {"team_a": method, "team_b": method}


@pytest.fixture(scope="module")
def first_info() -> EpisodeInfo:
    config = make_standard_team_deathmatch_config(
        map_id=0, team_a_roster=("mage",), team_b_roster=("priest",), max_steps=8
    )
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    _, state = env.reset(jax.random.key(0))
    zeros = jnp.zeros(10, jnp.int32)
    info = cast(
        EpisodeInfo,
        jax.jit(env.step)(jax.random.key(1), state, Action(zeros, zeros, zeros))[4],
    )
    assert not bool(info.completed) and int(info.decision_step) == 0
    return info


def _trace(info: EpisodeInfo) -> PolicyTrace:
    return PolicyTrace(
        info.episode_id,
        info.decision_step,
        jnp.asarray(True),
        jnp.where(info.active_mask, jnp.int32(0), jnp.int32(-1)),
    )


def _source_start(info: EpisodeInfo) -> EpisodeStartRecords:
    config_id = configuration_identity(info.config)[0]
    bank_id = sha256(
        json.dumps([config_id], separators=(",", ":")).encode() + b"\n"
    ).digest()
    words = np.frombuffer(bank_id, dtype=">u4").astype(np.uint32)
    return EpisodeStartRecords(
        info.episode_id,
        jnp.int32(0),
        jnp.asarray(words),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(-1),
        jnp.asarray(True),
        jnp.asarray(False),
        jnp.asarray(True),
    )


def _set_durability(writer: RunWriter, durability: Durability) -> RunWriter:
    if durability == "flushed":
        writer.flush()
    elif durability == "resumed":
        path = writer.run_dir
        writer.close()
        return RunWriter(resume_from=path, policies=_systems())
    return writer


def _files(path: Path) -> dict[str, bytes]:
    return {
        entry.name: entry.read_bytes() for entry in path.iterdir() if entry.is_file()
    }


def _manifest(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((path / "run_details.json").read_bytes()))


def _assert_original_ownership(path: Path) -> None:
    manifest = _manifest(path)
    entry = next(iter(manifest["passes"].values()))
    rows = [
        row
        for batch in iter_policy_assignments(
            path / "policy_assignments.csv",
            manifest=manifest,
            expected_header=ASSIGNMENT_COLUMNS,
        )
        for row in batch
    ]
    assert len(rows) == 2
    assert {row["global_slot"] for row in rows} == {0, 5}
    for row in rows:
        assert row["system_id"] == entry["system_ids"][row["team"]]
        assert manifest["systems"][row["system_id"]]["name"] == "Original"


@pytest.mark.parametrize("durability", ["pending", "flushed", "resumed"])
@pytest.mark.parametrize("declaration", ["owner", "config", "seed", "map"])
def test_first_schedule_declaration_after_trace_cannot_relabel_records(
    tmp_path: Path,
    first_info: EpisodeInfo,
    durability: Durability,
    declaration: str,
) -> None:
    writer = RunWriter(tmp_path, policies=_systems())
    writer.write(first_info, policy_trace=_trace(first_info))
    writer = _set_durability(writer, durability)
    path = writer.run_dir
    record: dict[str, object] = {"episode_id": 1}
    record.update(
        {
            "owner": {"policies": {"team_a": "Changed", "team_b": "Changed"}},
            "config": {"configuration_digest": "0" * 64},
            "seed": {"seed_id": 100},
            "map": {"map_id": 1},
        }[declaration]
    )
    try:
        before = _files(path)
        with pytest.raises(ValueError, match="before recording"):
            writer.register_episodes([record])
        assert _files(path) == before
        writer.flush()
        entry = next(iter(_manifest(path)["passes"].values()))
        assert entry["episodes"] == {}
    finally:
        writer.close()
    _assert_original_ownership(path)


@pytest.mark.parametrize("durability", ["pending", "flushed", "resumed"])
def test_first_source_declaration_after_trace_cannot_relabel_records(
    tmp_path: Path, first_info: EpisodeInfo, durability: Durability
) -> None:
    writer = RunWriter(tmp_path, policies=_systems())
    writer.write(first_info, policy_trace=_trace(first_info))
    writer = _set_durability(writer, durability)
    path = writer.run_dir
    try:
        before = _files(path)
        with pytest.raises(ValueError, match="before recording"):
            writer.register_episodes(
                _source_start(first_info), source_configs=first_info.config
            )
        assert _files(path) == before
        writer.flush()
        manifest = _manifest(path)
        entry = next(iter(manifest["passes"].values()))
        assert entry["episode_starts"] == {}
        assert manifest["source_banks"] == {}
    finally:
        writer.close()
    _assert_original_ownership(path)


@pytest.mark.parametrize("durability", ["pending", "flushed", "resumed"])
def test_existing_schedule_and_source_repeat_but_cannot_change_after_trace(
    tmp_path: Path, first_info: EpisodeInfo, durability: Durability
) -> None:
    writer = RunWriter(tmp_path, policies=_systems())
    schedule = {"episode_id": 1, "seed_id": 17, "map_id": 0}
    start = _source_start(first_info)
    writer.register_episodes([schedule])
    writer.register_episodes(start, source_configs=first_info.config)
    writer.write(
        first_info._replace(episode_start_records=start),
        policy_trace=_trace(first_info),
    )
    writer = _set_durability(writer, durability)
    path = writer.run_dir
    try:
        writer.register_episodes([schedule])
        writer.register_episodes(start, source_configs=first_info.config)
        before = _files(path)
        for field, value in (("seed_id", 18), ("map_id", 1)):
            with pytest.raises(ValueError, match="recorded schedule"):
                writer.register_episodes([{**schedule, field: value}])
            assert _files(path) == before
        with pytest.raises(ValueError, match="recorded start"):
            writer.register_episodes(
                start._replace(reset_generation=jnp.int32(1)),
                source_configs=first_info.config,
            )
        assert _files(path) == before
        entry = next(iter(_manifest(path)["passes"].values()))
        assert entry["episodes"]["1"] == schedule
        assert entry["episode_starts"]["1"]["verification"] == "verified"
        assert entry["episode_starts"]["1"]["reset_generation"] == 0
    finally:
        writer.close()
    _assert_original_ownership(path)
