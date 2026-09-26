"""Check LLM default/custom examples through existing saved evaluation paths.

Loopback fake servers use development map 0 only. These checks cover ordinary
game output, replay selection, durable completed-game skipping and identifiable
method changes, runner-owned client lifetime and cleanup after failure. Separate
evidence checks cover model-call files and durable game attribution. Factory
commands share this path. Authored starts preserve their game tick, and saved
world actions reproduce full trajectories from both spawn ends without requests.
"""

import json
import sys
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
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


@pytest.mark.parametrize("custom", [False, True])
def test_package_factory_command_and_resume_use_the_same_llm_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custom: bool,
) -> None:
    from marl_battlegrounds import _cli, _method_loading

    model = FakeModel(
        "east no_combat" if custom else '{"move":"east","combat":"no_combat"}'
    )
    loads: list[str] = []
    original = _method_loading.load_factory

    def load(reference: str) -> marl_bgs.System | marl_bgs.Policy:
        loads.append(reference)
        return original(reference)

    monkeypatch.setattr(_method_loading, "load_factory", load)
    reference = "examples.llm:" + (
        "make_custom_system" if custom else "make_default_system"
    )
    with server(model.serve) as url:
        monkeypatch.setenv("MARL_LLM_MODEL", "fake")
        monkeypatch.setenv("MARL_LLM_URL", url)
        monkeypatch.setenv("MARL_LLM_REVISION", "fixture-revision")
        monkeypatch.setenv("MARL_LLM_HISTORY", "2")
        arguments = [
            "evaluate",
            "--system",
            reference,
            "--opponent",
            "random",
            "--episodes",
            "2",
            "--maps",
            "0",
            "--max-steps",
            "2",
            "--num-envs",
            "2",
            "--chunk-size",
            "1",
            "--save-replays",
            "2",
        ]
        assert _cli.main([*arguments, "--output-dir", str(tmp_path)]) == 0
        assert loads == [reference]
        assert len(model.generations()) == 20
        path = next(tmp_path.rglob("run_details.json")).parent
        result = marl_bgs.load_results(path)
        calls = list(llm.read_calls(path))
        assert len(calls) == 20
        assert all(row["outcome"] == "played" for row in calls)
        assert {row["world_action"][0] for row in calls} == {3, 4}
        np.testing.assert_array_equal(
            result.table("episodes")["episode_length"], [2, 2]
        )
        requests = len(model.calls)
        assert _cli.main([*arguments, "--resume-from", str(path)]) == 0
        assert loads == [reference, reference]
        assert len(model.calls) == requests
        monkeypatch.setenv("MARL_LLM_REVISION", "changed-revision")
        assert _cli.main([*arguments, "--resume-from", str(path)]) == 1
        assert len(model.calls) == requests


@pytest.mark.parametrize("spawn", [0, 1])
def test_authored_epoch_and_saved_world_actions_reproduce_the_full_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawn: int
) -> None:
    from marl_battlegrounds.core import env as core
    from marl_battlegrounds.core.types import ActionMask
    from marl_battlegrounds.evaluation.policy_execution import (
        PolicyTree,
        SystemInput,
        SystemOutput,
    )
    from marl_battlegrounds.llm.system import History
    from marl_battlegrounds.policies.actor import ActorAction
    from marl_battlegrounds.policies.input import ActorInput
    from marl_battlegrounds.tasks import (
        balanced_spawn_configs,
        make_standard_team_deathmatch_config,
    )

    roster = ("warrior", "mage", "hunter", "rogue", "priest")
    config = make_standard_team_deathmatch_config(
        map_id=0, max_steps=7, team_a_roster=roster, team_b_roster=roster
    )
    banks = balanced_spawn_configs(config, num_envs=2)

    config = config._replace(
        team_spawn_pad_positions=banks.team_spawn_pad_positions[spawn]
    )
    initial, *_ = core.reset(config, jax.random.key(71))
    authored = initial._replace(
        step_count=jnp.int32(5),
        team_deathmatch_scores=jnp.asarray([2, 3], jnp.int32),
    )
    schedule = [marl_bgs.EpisodeSpec(7, config, initial_state=authored)]
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    original = evaluator._system_advance
    states: list[Any] = []

    def capture(*args: PolicyTree, **kwargs: PolicyTree) -> PolicyTree:
        result = original(*args, **kwargs)
        states.append(jax.device_get(result[0].state))
        return result

    def prompt(actor: ActorInput, masks: ActionMask, history: History) -> str:
        return (
            llm.format_actor_view(actor, masks)
            + "\nHistory turns: "
            + str([entry.timestep for entry in history])
        )

    monkeypatch.setattr(evaluator, "_system_advance", capture)
    model = FakeModel('{"move":"east","combat":"no_combat"}')
    with server(model.serve) as url:
        method = llm.make_system(
            "fake",
            url,
            history_turns=2,
            records="full",
            prompt_builder=prompt,
            custom_version="authored-proof-v1",
        )
        result = marl_bgs.evaluate_episodes(
            method,
            "random",
            schedule,
            num_envs=1,
            chunk_size=1,
            metrics="none",
            save_replays=1,
            output_dir=tmp_path,
        )
        assert result.run_dir is not None
        calls = list(llm.read_calls(result.run_dir))
        assert len(calls) == len(model.generations()) == 10
        assert {row["episode_id"] for row in calls} == {7}
        assert {row["decision_step"] for row in calls} == {0, 1}
        for row in calls:
            text = json.loads(row["request_json"])["messages"][0]["content"]
            tick = 5 + row["decision_step"]
            assert f"current_timestep={tick}" in text
            assert text.endswith(
                "History turns: []" if tick == 5 else "History turns: [5]"
            )
        assert result.episodes[0].episode_length == 2
        assert (result.episodes[0].team_a_score, result.episodes[0].team_b_score) == (
            2,
            3,
        )
        assert result.run_dir is not None
        replay = json.loads(
            next(result.run_dir.rglob("*.marlbg-replay.json")).read_text()
        )
        tape = np.asarray(
            [
                np.stack(
                    [
                        transition["facts"]["action_acceptance_facts"][
                            "submitted_joint_action"
                        ][head]
                        for head in ActorAction._fields
                    ],
                    axis=-1,
                )
                for transition in replay["transitions"]
            ],
            dtype=np.int32,
        )
        for row in calls:
            np.testing.assert_array_equal(
                row["world_action"], tape[row["decision_step"], row["actor"]]
            )
        request_count = len(model.calls)

        def init(
            variables: PolicyTree, inputs: SystemInput, key: jax.Array
        ) -> list[int]:
            return [0 for _ in inputs.valid]

        def play(
            variables: PolicyTree,
            memory: PolicyTree,
            inputs: SystemInput,
            key: jax.Array,
        ) -> SystemOutput:
            actions = np.zeros((*inputs.active_mask.shape, 3), dtype=np.int32)
            next_memory = list(memory)
            for lane, valid in enumerate(inputs.valid):
                if valid:
                    actions[lane] = variables[memory[lane]]
                    next_memory[lane] += 1
            return SystemOutput(
                ActorAction(*(jnp.asarray(actions[..., head]) for head in range(3))),
                next_memory,
            )

        methods = [
            marl_bgs.System(
                f"Saved Actions {team}",
                play,
                variables=tape[:, team * 5 : team * 5 + 5],
                init=init,
                execution="host",
            )
            for team in range(2)
        ]
        reproduced = marl_bgs.evaluate_episodes(
            methods[0],
            methods[1],
            schedule,
            num_envs=1,
            chunk_size=1,
            metrics="none",
            save_replays=1,
        )
        assert len(model.calls) == request_count
        assert reproduced.episodes[0].episode_length == 2
        assert len(states) == 4
        for actual, repeated in zip(states[:2], states[2:], strict=True):
            assert jax.tree.structure(actual) == jax.tree.structure(repeated)
            for first, second in zip(
                jax.tree.leaves(actual), jax.tree.leaves(repeated), strict=True
            ):
                np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("saved", [False, True])
def test_tournament_reuses_one_client_and_joins_counts_to_systems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom: bool, saved: bool
) -> None:
    from dataclasses import replace

    from marl_battlegrounds.evaluation.policy_execution import policy

    closed: list[llm.Client] = []
    original_close = llm.Client.close

    def close(client: llm.Client, *, grace: float = 2.0) -> None:
        original_close(client, grace=grace)
        closed.append(client)

    monkeypatch.setattr(llm.Client, "close", close)
    model = FakeModel(
        "stay no_combat" if custom else '{"move":"stay","combat":"no_combat"}'
    )
    with server(model.serve) as url:
        method = llm.make_system(
            "fake",
            url,
            name="m-llm",
            prompt_builder=prompt_for_words if custom else None,
            reply_parser=parse_words if custom else None,
            custom_version="tournament-test-v1" if custom else None,
        )
        entrants = [
            replace(policy("random"), name="a-random"),
            method,
            replace(policy("random"), name="z-random"),
        ]
        result = marl_bgs.run_tournament(
            entrants,
            maps=[0],
            episodes_per_pair=2,
            num_envs=2,
            max_steps=1,
            chunk_size=1,
            output_dir=tmp_path if saved else None,
        )
        assert len(result.matches) == 6
        assert len(model.generations()) == 20
        assert len(closed) == 1
        summary = llm.call_summary(result)
        assert summary["all_attempts"]["team_a"]["model_calls"] == 10
        assert summary["all_attempts"]["team_b"]["model_calls"] == 10
        owners = summary["by_system"]
        assert len(owners) == 3
        identifier, owner = next(
            (key, value) for key, value in owners.items() if value["name"] == "m-llm"
        )
        assert identifier in result.metadata["systems"]
        assert owner["all_attempts"]["model_calls"] == 20
        assert owner["completed_games"]["played_actions"] == 20
        assert owner["completed_games"]["fallback_actions"] == 0
        if saved:
            assert result.run_dir is not None
            assert result.paths is not None
            assert result.paths["model_calls"].is_dir()
            calls = list(llm.read_calls(result.run_dir))
            assert len(calls) == 20
            assert len({row["pass_id"] for row in calls}) == 2
            assert {row["team"] for row in calls} == {0, 1}
            before = {
                path: path.read_bytes()
                for path in result.run_dir.rglob("*")
                if path.is_file()
            }
            resumed = marl_bgs.run_tournament(
                entrants,
                maps=[0],
                episodes_per_pair=2,
                num_envs=2,
                max_steps=1,
                chunk_size=1,
                resume_from=result.run_dir,
            )
            assert len(model.generations()) == 20
            assert len(closed) == 1
            assert llm.call_summary(resumed) == summary
            assert llm.call_summary(marl_bgs.load_results(result.run_dir)) == summary
            assert resumed.paths is not None and resumed.paths["model_calls"].is_dir()
            assert before == {path: path.read_bytes() for path in before}


def test_partial_tournament_resume_acquires_no_finished_llm_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from typing import cast

    from marl_battlegrounds.evaluation.policy_execution import PolicyTree, policy
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )

    provenance = capture_recording_provenance()
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")

    def same_source(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", same_source)
    module = import_module("marl_battlegrounds.evaluation.tournament")
    original = module.evaluate_episodes
    finished = 0

    def interrupt(*args: PolicyTree, **kwargs: PolicyTree) -> marl_bgs.EvaluationResult:
        nonlocal finished
        if finished == 2:
            raise RuntimeError("Stop before the unrelated Random pair")
        result = original(*args, **kwargs)
        finished += 1
        return cast(marl_bgs.EvaluationResult, result)

    monkeypatch.setattr(module, "evaluate_episodes", interrupt)
    model = FakeModel()
    with server(model.serve) as url:
        method = llm.make_system("fake", url, name="a-llm")
        entrants = [
            method,
            replace(policy("random"), name="b-random"),
            replace(policy("random"), name="c-random"),
        ]
        with pytest.raises(RuntimeError, match="unrelated Random"):
            marl_bgs.run_tournament(
                entrants,
                maps=[0],
                episodes_per_pair=2,
                num_envs=2,
                max_steps=1,
                chunk_size=1,
                output_dir=tmp_path,
            )
        assert len(model.generations()) == 20
        run_dir = next(tmp_path.rglob("run_details.json")).parent
        before = {
            path: path.read_bytes()
            for path in (run_dir / "model_calls").rglob("*")
            if path.is_file()
        }

        def reject_client(*args: object, **kwargs: object) -> None:
            raise AssertionError("Completed LLM games acquired a new client")

        monkeypatch.setattr(llm.Client, "__init__", reject_client)
        monkeypatch.setattr(module, "evaluate_episodes", original)
        resumed = marl_bgs.run_tournament(
            entrants,
            maps=[0],
            episodes_per_pair=2,
            num_envs=2,
            max_steps=1,
            chunk_size=1,
            resume_from=run_dir,
        )
        assert len(resumed.matches) == 6
        assert len(model.generations()) == 20
        assert before == {path: path.read_bytes() for path in before}
        assert llm.call_summary(resumed)["all_attempts"]["team_a"]["model_calls"] == 20
