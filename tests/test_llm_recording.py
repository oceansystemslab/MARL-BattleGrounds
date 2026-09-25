"""Check actor call evidence against real game steps and durable writer boundaries.

Fake providers cover both teams, full/light/disabled files, malformed replies,
failed joint steps and resumed games. Calls from an abandoned attempt count as
cost; completed-game fallbacks come only from the attempt that finished it.
"""

import json
import os
import stat
from hashlib import sha256
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Literal

import pytest
from tests.llm_fixtures import FakeModel, send, server

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm


@pytest.mark.parametrize("mode", ["none", "light", "full"])
def test_saved_call_detail_and_totals_survive_completed_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["none", "light", "full"],
) -> None:
    import marl_battlegrounds.llm.recording as recording

    if mode == "none":

        def reject_hash(*args: object, **kwargs: object) -> None:
            raise AssertionError("Disabled call records hashed a prompt")

        monkeypatch.setattr(recording, "sha256", reject_hash)
    model = FakeModel()
    with server(model.serve) as url:
        method = llm.make_system("fake", url, records=mode)
        result = marl_bgs.evaluate(
            method,
            method,
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=2,
            output_dir=tmp_path,
        )
        assert result.run_dir is not None
        evidence: Any = result.metadata["host_evidence"]["llm"]
        assert len(evidence["attempts"]) == 1
        attempt, summary = next(iter(evidence["attempts"].items()))
        assert set(evidence["episode_attempts"].values()) == {attempt}
        for team in ("team_a", "team_b"):
            counts = summary["teams"][team]
            assert counts["model_calls"] == counts["tokenizer_calls"] == 20
            assert counts["played_actions"] == 20
            assert counts["fallback_actions"] == counts["abandoned_actions"] == 0
            assert counts["input_tokens"] > 0
            assert counts["missing_usage_replies"] == 20
        folder = result.run_dir / "model_calls"
        if mode == "none":
            assert not folder.exists()
        else:
            rows = list(llm.read_calls(folder))
            assert len(rows) == 40
            assert {row["team"] for row in rows} == {0, 1}
            assert (
                len(
                    {
                        (
                            row["episode_id"],
                            row["decision_step"],
                            row["team"],
                            row["actor"],
                        )
                        for row in rows
                    }
                )
                == 40
            )
            assert all(
                row["outcome"] == "played" and row["world_action"] == [0, 0, 0]
                for row in rows
            )
            assert all(row["attempt_id"] == attempt for row in rows)
            assert all(("request_json" in row) == (mode == "full") for row in rows)
            if mode == "full":
                expected = {
                    json.dumps(request, allow_nan=False, separators=(",", ":"))
                    for request in model.generations()
                }
                for row in rows:
                    assert row["request_json"] in expected
                    assert (
                        sha256(row["request_json"].encode()).hexdigest()
                        == row["request_sha256"]
                    )
        calls = len(model.calls)
        resumed = marl_bgs.evaluate(
            method,
            method,
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=2,
            resume_from=result.run_dir,
        )
        assert len(model.calls) == calls
        assert resumed.metadata["host_evidence"] == result.metadata["host_evidence"]
        loaded = marl_bgs.load_results(result.run_dir)
        assert loaded.metadata["host_evidence"] == result.metadata["host_evidence"]


def test_failed_joint_decision_abandons_first_team_and_keeps_new_resume_attempt(
    tmp_path: Path,
) -> None:
    good, bad = FakeModel(), FakeModel("not json")
    with server(good.serve) as good_url, server(bad.serve) as bad_url:
        first = llm.make_system("good", good_url, records="full")
        second = llm.make_system("bad", bad_url, records="full")
        with pytest.raises(llm.ReplyFormatError):
            marl_bgs.evaluate(
                first,
                second,
                num_episodes=2,
                maps=[0],
                num_envs=2,
                max_steps=1,
                output_dir=tmp_path,
            )
        run_dir = next(path.parent for path in tmp_path.rglob("run_details.json"))
        old = list(llm.read_calls(run_dir / "model_calls"))
        assert old and all(row["outcome"] == "abandoned" for row in old)
        assert (
            len([row for row in old if row["team"] == 0 and row["action"] is not None])
            == 10
        )
        old_attempt = old[0]["attempt_id"]
        bad.text = good.text
        result = marl_bgs.evaluate(
            first,
            second,
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=1,
            resume_from=run_dir,
        )
        evidence: Any = result.metadata["host_evidence"]["llm"]
        assert len(evidence["attempts"]) == 2
        assert old_attempt not in evidence["episode_attempts"].values()
        new = [
            row
            for row in llm.read_calls(run_dir / "model_calls")
            if row["outcome"] == "played"
        ]
        assert len(new) == 20
        assert all(row["history_entries"] == 0 for row in new)
        assert (
            evidence["attempts"][old_attempt]["teams"]["team_a"]["abandoned_actions"]
            == 10
        )
        assert (
            evidence["attempts"][old_attempt]["teams"]["team_b"]["reply_failures"] > 0
        )


def test_checked_fallback_counts_both_teams_in_completed_game_evidence(
    tmp_path: Path,
) -> None:
    model = FakeModel("invalid")
    with server(model.serve) as url:
        method = llm.make_system("fake", url, failure_policy="fallback", records="none")
        result = marl_bgs.evaluate(
            method,
            method,
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=1,
            output_dir=tmp_path,
        )
    evidence: Any = result.metadata["host_evidence"]["llm"]
    for episode, attempt in evidence["episode_attempts"].items():
        counts = evidence["attempts"][attempt]["episodes"][episode]
        for team in ("team_a", "team_b"):
            assert (
                counts[team]["fallback_actions"] == counts[team]["reply_failures"] == 5
            )
    assert result.run_dir is not None
    assert not (result.run_dir / "model_calls").exists()


def test_restart_counts_old_calls_but_only_completed_attempt_fallbacks(
    tmp_path: Path,
) -> None:
    fallback, ordinary = FakeModel("invalid"), FakeModel()
    failing = True

    def second_team(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        if failing and "current_timestep=1" in payload["messages"][0]["content"]:
            ordinary.text = "invalid"
        ordinary.serve(handler, payload)

    with server(fallback.serve) as left, server(second_team) as right:
        first = llm.make_system("first", left, failure_policy="fallback")
        second = llm.make_system("second", right)
        with pytest.raises(llm.ReplyFormatError):
            marl_bgs.evaluate(
                first,
                second,
                num_episodes=2,
                maps=[0],
                num_envs=2,
                max_steps=2,
                chunk_size=2,
                output_dir=tmp_path,
            )
        run_dir = next(path.parent for path in tmp_path.rglob("run_details.json"))
        loaded = marl_bgs.load_results(run_dir)
        old = llm.call_summary(loaded)
        assert old["all_attempts"]["team_a"]["fallback_actions"] == 10
        assert old["completed_games"]["team_a"]["fallback_actions"] == 0
        failing = False
        fallback.text = ordinary.text = '{"move":"stay","combat":"no_combat"}'
        resumed = marl_bgs.evaluate(
            first,
            second,
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=2,
            chunk_size=2,
            resume_from=run_dir,
        )
        summary = llm.call_summary(resumed)
        assert summary["all_attempts"]["team_a"]["fallback_actions"] == 10
        assert summary["completed_games"]["team_a"]["fallback_actions"] == 0
        assert summary["completed_games"]["team_a"]["played_actions"] == 20
        assert summary["all_attempts"]["team_a"]["model_calls"] == 40
        assert llm.call_summary(marl_bgs.load_results(run_dir)) == summary


def test_call_directory_sync_failure_stops_before_game_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    fsync = os.fsync
    failed = False

    def fail_call_directory(descriptor: int) -> None:
        nonlocal failed
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            target = os.readlink(f"/proc/self/fd/{descriptor}")
            if target.endswith("/model_calls"):
                failed = True
                raise OSError("injected call directory sync failure")
        fsync(descriptor)

    with server(model.serve) as url:
        monkeypatch.setattr(os, "fsync", fail_call_directory)
        method = llm.make_system("fake", url, failure_policy="fallback")
        with pytest.raises(OSError, match="call directory sync"):
            marl_bgs.evaluate(
                method,
                "random",
                num_episodes=2,
                maps=[0],
                num_envs=2,
                max_steps=1,
                output_dir=tmp_path,
            )
    assert failed
    manifest = json.loads(next(tmp_path.rglob("run_details.json")).read_text())
    assert all(
        not entry["completed_episode_ids"] for entry in manifest["passes"].values()
    )


def test_sealed_call_cannot_change_replacement_game_or_interrupted_outcome(
    tmp_path: Path,
) -> None:
    import marl_battlegrounds.llm.recording as recording
    from marl_battlegrounds.evaluation.host_evidence import (
        DecisionContext,
        HostRun,
        team_scope,
    )

    run = HostRun("run", "evaluation", "1", tmp_path, {})
    with run.scope():
        recording.attach('{"model":"fake"}')
        observer = run.observers["llm"]
        old = None
        with (
            pytest.raises(KeyboardInterrupt),
            observer.decision(DecisionContext((7,), (3,), (2,), (True,))),
            team_scope(1),
        ):
            old = recording.actor_channel(0, 2, "full", "method")
            assert old is not None
            old.request({"messages": [{"role": "user", "content": "old"}]})
            raise KeyboardInterrupt()
        assert old is not None
        old.update(world_action=[8, 0, 0], reply="late")
        old.count("model_calls", 99)
        with (
            observer.decision(DecisionContext((8,), (0,), (3,), (True,))),
            team_scope(1),
        ):
            new = recording.actor_channel(0, 2, "full", "method")
            assert new is not None
            new.request({"messages": [{"role": "user", "content": "new"}]})
    rows = list(llm.read_calls(tmp_path / "model_calls"))
    assert [
        (
            row["episode_id"],
            row["decision_step"],
            row["reset_generation"],
            row["outcome"],
        )
        for row in rows
    ] == [(7, 3, 2, "unknown"), (8, 0, 3, "played")]
    assert all(row["model_calls"] == 1 for row in rows)
    assert all("reply" not in row and "world_action" not in row for row in rows)


@pytest.mark.parametrize("mode", ["none", "light"])
@pytest.mark.parametrize("fallback", [False, True])
def test_nonfinite_reply_is_transport_failure_in_every_recording_mode(
    tmp_path: Path,
    mode: Literal["none", "light"],
    fallback: bool,
) -> None:
    model = FakeModel()

    def malformed(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        if handler.path == "/tokenize":
            model.serve(handler, payload)
            return
        send(
            handler,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": model.text},
                        "logprobs": float("nan"),
                    }
                ],
                "usage": {"prompt_tokens": len(payload["messages"][0]["content"])},
            },
        )

    with server(malformed) as url:
        method = llm.make_system(
            "fake", url, records=mode, failure_policy="fallback" if fallback else "stop"
        )
        if fallback:
            result = marl_bgs.evaluate(
                method,
                "random",
                num_episodes=2,
                maps=[0],
                num_envs=2,
                max_steps=1,
                output_dir=tmp_path,
            )
            counts = llm.call_summary(result)["completed_games"]["team_a"]
            assert counts["transport_failures"] == counts["fallback_actions"] == 10
        else:
            with pytest.raises(llm.TransportError):
                marl_bgs.evaluate(
                    method,
                    "random",
                    num_episodes=2,
                    maps=[0],
                    num_envs=2,
                    max_steps=1,
                    output_dir=tmp_path,
                )
