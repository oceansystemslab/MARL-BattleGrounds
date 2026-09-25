"""Check LLM default/custom examples through existing saved evaluation paths.

Loopback fake servers use development map 0 only. These checks cover ordinary
game output, replay selection, durable completed-game skipping and identifiable
method changes, runner-owned client lifetime and cleanup after failure. Separate
evidence checks cover model-call files and durable game attribution.
"""

import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
from examples.llm import main, parse_words, prompt_for_words
from tests.llm_fixtures import FakeModel, server

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm


@pytest.mark.parametrize("custom", [False, True])
def test_saved_default_and_custom_evaluation_skip_completed_games(
    tmp_path: Path,
    custom: bool,
) -> None:
    model = FakeModel(
        "stay no_combat" if custom else '{"move":"stay","combat":"no_combat"}'
    )
    with server(model.serve) as url, llm.Client(url, concurrency=4) as client:
        method = llm.make_system(
            "fake",
            client=client,
            history_turns=2,
            prompt_builder=prompt_for_words if custom else None,
            reply_parser=parse_words if custom else None,
            custom_version="test-v1" if custom else None,
        )
        result = marl_bgs.evaluate(
            method,
            "random",
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=2,
            output_dir=tmp_path,
            save_replays=1,
        )
        assert len(result.completed_episode_ids) == 2
        np.testing.assert_array_equal(
            result.table("episodes")["episode_length"], [2, 2]
        )
        assert len(model.generations()) == 20
        assert result.paths is not None
        run_dir = result.paths["run_details"].parent
        assert result.paths["run_details"].is_file()
        calls = len(model.calls)
        resumed = marl_bgs.evaluate(
            method,
            "random",
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=2,
            resume_from=run_dir,
            save_replays=1,
        )
        assert not resumed.episodes
        assert resumed.completed_episode_ids == result.completed_episode_ids
        assert len(model.calls) == calls


def test_changed_generation_settings_cannot_resume_a_saved_method(
    tmp_path: Path,
) -> None:
    model = FakeModel()
    with server(model.serve) as url, llm.Client(url) as client:
        first = llm.make_system("fake", client=client)
        result = marl_bgs.evaluate(
            first,
            "random",
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=1,
            output_dir=tmp_path,
        )
        assert result.paths is not None
        calls = len(model.calls)
        changed = llm.make_system("fake", client=client, sampling={"temperature": 1.0})
        with pytest.raises(ValueError):
            marl_bgs.evaluate(
                changed,
                "random",
                num_episodes=2,
                maps=[0],
                num_envs=2,
                max_steps=1,
                resume_from=result.paths["run_details"].parent,
            )
        assert len(model.calls) == calls


@pytest.mark.parametrize("format_name", ["default", "note", "words"])
def test_example_command_runs_and_closes_its_supplied_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    format_name: str,
) -> None:
    model = FakeModel(
        "stay no_combat"
        if format_name == "words"
        else '{"move":"stay","combat":"no_combat"}'
    )
    closed: list[llm.Client] = []
    original_close = llm.Client.close

    def close(client: llm.Client, *, grace: float = 2.0) -> None:
        original_close(client, grace=grace)
        closed.append(client)

    monkeypatch.setattr(llm.Client, "close", close)
    with server(model.serve) as url:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "examples/llm.py",
                "--model",
                "fake",
                "--server-url",
                url,
                "--format",
                format_name,
                "--history-turns",
                "2",
                "--games",
                "2",
                "--num-envs",
                "2",
                "--max-steps",
                "2",
                "--output-dir",
                str(tmp_path),
            ],
        )
        main()
    assert len(model.generations()) == 20
    assert len(closed) == 1
    with pytest.raises(RuntimeError, match="closed"):
        closed[0].request("chat/completions", {})
    output = capsys.readouterr().out
    assert "episode_length" in output
    assert "Game files:" in output


def test_managed_system_reopens_after_each_evaluation_and_shares_two_team_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[llm.Client] = []
    original_close = llm.Client.close

    def close(client: llm.Client, *, grace: float = 2.0) -> None:
        original_close(client, grace=grace)
        closed.append(client)

    monkeypatch.setattr(llm.Client, "close", close)
    model = FakeModel()
    with server(model.serve) as url:
        method = llm.make_system("fake", url)
        for _ in range(2):
            result = marl_bgs.evaluate(
                method, method, num_episodes=2, maps=[0], num_envs=2, max_steps=1
            )
            assert len(result.episodes) == 2
        assert len(model.generations()) == 40
        assert len(closed) == 2
        assert closed[0] is not closed[1]
        for client in closed:
            with pytest.raises(RuntimeError, match="closed"):
                client.request("chat/completions", {})


def test_managed_client_closes_after_model_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[llm.Client] = []
    original_close = llm.Client.close

    def close(client: llm.Client, *, grace: float = 2.0) -> None:
        original_close(client, grace=grace)
        closed.append(client)

    monkeypatch.setattr(llm.Client, "close", close)
    model = FakeModel("invalid reply")
    with server(model.serve) as url:
        method = llm.make_system("fake", url)
        with pytest.raises(llm.ReplyFormatError):
            marl_bgs.evaluate(
                method, "random", num_episodes=2, maps=[0], num_envs=2, max_steps=1
            )
    assert len(closed) == 1


def test_recording_custom_method_requires_identity_before_output_creation(
    tmp_path: Path,
) -> None:
    method = llm.make_system(
        "fake", prompt_builder=prompt_for_words, reply_parser=parse_words
    )
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="custom_version"):
        marl_bgs.evaluate(
            method,
            "random",
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=1,
            output_dir=target,
        )
    assert not target.exists()


def test_completed_managed_resume_does_not_acquire_a_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds.llm.system as module

    model = FakeModel()
    with server(model.serve) as url:
        method = llm.make_system("fake", url)
        result = marl_bgs.evaluate(
            method,
            "random",
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=1,
            output_dir=tmp_path,
        )
        assert result.paths is not None
        calls = len(model.calls)

        def forbidden(*_args: object, **_kwargs: object) -> None:
            pytest.fail("A completed resume must not acquire a new managed client")

        monkeypatch.setattr(module, "Client", forbidden)
        resumed = marl_bgs.evaluate(
            method,
            "random",
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=1,
            resume_from=result.paths["run_details"].parent,
        )
        assert resumed.completed_episode_ids == result.completed_episode_ids
        assert len(model.calls) == calls


def test_resource_scope_survives_freezing_and_fences_failed_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation.policy_execution import System
    from marl_battlegrounds.evaluation.system_evaluation import freeze_evaluation_method

    method = llm.make_system("fake")
    frozen = freeze_evaluation_method(method)
    assert isinstance(frozen, System)
    assert frozen.resource_scope is method.resource_scope
    assert method.resource_scope is not None
    failures: list[bool] = []

    def fail(_client: llm.Client, *, grace: float = 2.0) -> None:
        del grace
        failures.append(True)
        raise RuntimeError("Owned worker survived")

    monkeypatch.setattr(llm.Client, "close", fail)
    with (
        pytest.raises(RuntimeError, match="Owned worker survived"),
        ExitStack() as stack,
    ):
        stack.enter_context(method.resource_scope(False))
        stack.enter_context(method.resource_scope(False))
    assert len(failures) == 1
    with pytest.raises(RuntimeError, match="fenced"), method.resource_scope(False):
        pass


def test_borrowed_client_stays_open_and_cleanup_keeps_original_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes: list[llm.Client] = []
    client = llm.Client("http://127.0.0.1:1/v1")
    original = llm.Client.close

    def fail_close(self: llm.Client, *, grace: float = 2.0) -> None:
        original(self, grace=grace)
        closes.append(self)
        raise OSError("cleanup failure")

    monkeypatch.setattr(llm.Client, "close", fail_close)
    borrowed = llm.make_system("fake", client=client)
    assert borrowed.resource_scope is not None
    with borrowed.resource_scope(False):
        pass
    assert not closes
    managed = llm.make_system("fake")
    assert managed.resource_scope is not None
    body = ValueError("body failure")
    with pytest.raises(ValueError) as caught, managed.resource_scope(False):
        raise body
    assert caught.value is body
    assert any("cleanup failure" in note for note in body.__notes__)
    with pytest.raises(RuntimeError, match="fenced"), managed.resource_scope(False):
        pass
    assert len(closes) == 1
    original(client)
