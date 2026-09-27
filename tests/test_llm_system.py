"""Prove LLM Systems use permitted views, exact actions and isolated history.

Loopback fake models exercise ordinary initialization, action selection and game
steps. Tests cover both teams, left-frame moves, immutable prospective memory,
custom formats, whole-entry context fitting and narrow failure fallback.
"""

# The tests inspect the adapter's private prospective-memory contract.
# pyright: reportPrivateUsage=false

import json
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from typing import Any, cast
from unittest.mock import Mock

import jax
import numpy as np
import pytest
from jax import Array
from tests.llm_fixtures import FakeModel as Model
from tests.llm_fixtures import send, server

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ALIVE,
    CONTEXT_FEATURE_CURRENT_TIMESTEP,
    ActionMask,
)
from marl_battlegrounds.environment import Environment, EnvironmentState
from marl_battlegrounds.evaluation.policy_execution import (
    SystemInput,
    SystemOutput,
    apply_systems,
    init_systems,
    policy,
    shared_policy,
)
from marl_battlegrounds.llm.actions import (
    MOVE_NAMES,
    InvalidActionError,
    ReplyFormatError,
    legal_action_names,
)
from marl_battlegrounds.llm.client import (
    Client,
    ContextLimitError,
    RequestCancelledError,
    TransportError,
)
from marl_battlegrounds.llm.system import History, make_system
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    mirror_move,
    team_on_right,
)
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


@pytest.fixture(scope="module")
def game() -> tuple[Environment, Observations, EnvironmentState]:
    roster = ("warrior", "mage", "hunter", "rogue", "priest")
    config = make_standard_team_deathmatch_config(
        map_id=0, max_steps=30, team_a_roster=roster, team_b_roster=roster
    )
    env = marl_bgs.make("tdm", env_config=config, num_envs=2, metrics="none")
    observations, state = env.reset(jax.random.key(2027))
    return env, observations, state


def inputs(game: tuple[Environment, Observations, EnvironmentState]) -> SystemInput:
    env, observations, state = game
    return cast(SystemInput, jax.device_get(env.policy_inputs(observations, state)))


def test_default_system_public_step_and_history_are_actor_local(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    model = Model()
    with server(model.serve) as url, Client(url, concurrency=4) as client:
        a = make_system("fake", client=client, history_turns=2)
        b = make_system("fake", client=client, history_turns=2)
        memory = init_systems(a, b, observations, state, jax.random.key(1))
        assert not model.calls
        for tick in range(3):
            previous = memory
            action, proposed, _ = apply_systems(
                a, b, memory, observations, state, jax.random.key(2 + tick)
            )
            assert all(
                len(history) == min(tick, 2)
                for lane in previous.team_a
                for history in lane
            )
            observations, state, _, _, _ = env.step(
                jax.random.key(10 + tick), state, action
            )
            memory = proposed
            for bank in (memory.team_a, memory.team_b):
                for lane in bank:
                    for slot, history in enumerate(lane):
                        assert len(history) == min(tick + 1, 2)
                        assert (
                            int(history[-1].actor.observation.self_ally_index) == slot
                        )
                        assert history[-1].timestep == tick
                        assert not np.asarray(
                            history[-1].actor.observation.self_features
                        ).flags.writeable
            assert memory.team_a is not memory.team_b
        assert len(model.generations()) == 60
        for request in model.generations():
            assert request["chat_template_kwargs"] == {"enable_thinking": False}
            assert request["max_tokens"] == 64
            assert request["response_format"]["type"] == "json_schema"


def test_left_frame_is_undone_once_through_public_action(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    model = Model('{"move":"east","combat":"no_combat"}')
    with server(model.serve) as url, Client(url) as client:
        a = make_system("fake", client=client)
        b = shared_policy(policy("random"))
        memory = init_systems(a, b, observations, state, jax.random.key(3))
        action, next_memory, _ = apply_systems(
            a, b, memory, observations, state, jax.random.key(4)
        )
        flag = team_on_right(env.policy_inputs(observations, state).actors)
        from marl_battlegrounds.llm.actions import MOVE_NAMES

        expected = mirror_move(
            cast(Array, np.full((2, 5), MOVE_NAMES.index("east"), np.int32)), flag
        )
        np.testing.assert_array_equal(action.move[:, :5], expected)
        assert all(not history for lane in next_memory.team_a for history in lane)
        env.step(jax.random.key(5), state, action)


def test_exact_custom_request_history_fitting_and_non_json_parser(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    model = Model("WAIT", limit=65)
    built: list[int] = []
    parsed: list[int] = []

    def builder(actor: ActorInput, masks: ActionMask, history: History) -> str:
        del actor, masks
        built.append(len(history))
        return "x" * (1 + len(history))

    def parser(
        reply: str, actor: ActorInput, masks: ActionMask, history: History
    ) -> ActorAction:
        del actor, masks
        assert reply == "WAIT"
        parsed.append(len(history))
        return ActorAction(*(cast(Array, np.int32(0)) for _ in range(3)))

    supplied = inputs(game)
    with server(model.serve) as url, Client(url, concurrency=1) as client:
        system = make_system(
            "fake",
            client=client,
            history_turns=2,
            prompt_builder=builder,
            reply_parser=parser,
        )
        assert system.init is not None
        memory = system.init(system.variables, supplied, jax.random.key(1))
        for _ in range(3):
            output = cast(
                SystemOutput,
                system.apply(system.variables, memory, supplied, jax.random.key(2)),
            )
            memory = output.next_memory
        assert set(built) == {0, 1, 2}
        assert set(parsed) == {0}
        assert len(model.generations()) == 30
        assert all("response_format" not in value for value in model.generations())
        assert all(len(history) == 2 for lane in memory for history in lane)


def test_context_overflow_stops_without_generation_or_fallback(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    model = Model(limit=64)
    supplied = inputs(game)
    with server(model.serve) as url, Client(url) as client:
        system = make_system("fake", client=client, failure_policy="fallback")
        assert system.init is not None
        memory = system.init(system.variables, supplied, jax.random.key(1))
        with pytest.raises(ContextLimitError):
            system.apply(system.variables, memory, supplied, jax.random.key(2))
        assert not model.generations()
        assert all(not history for lane in memory for history in lane)


@pytest.mark.parametrize(
    "reply", ["garbage", '{"move":"stay","combat":"enemy_2_ultimate"}']
)
def test_declared_reply_failure_can_fall_back(
    game: tuple[Environment, Observations, EnvironmentState],
    reply: str,
) -> None:
    model = Model(reply)
    supplied = inputs(game)
    with server(model.serve) as url, Client(url) as client:
        system = make_system(
            "fake", client=client, failure_policy="fallback", history_turns=2
        )
        assert system.init is not None
        memory = system.init(system.variables, supplied, jax.random.key(1))
        output = cast(
            SystemOutput,
            system.apply(system.variables, memory, supplied, jax.random.key(2)),
        )
        assert all(np.all(value == 0) for value in output.actions)
        assert all(len(history) == 1 for lane in output.next_memory for history in lane)
        assert all(not history for lane in memory for history in lane)


def test_custom_hook_bugs_do_not_become_fallback(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    def broken(_actor: ActorInput, _masks: ActionMask, _history: History) -> str:
        raise TransportError("This came from user code, not the HTTP client")

    supplied = inputs(game)
    with Client("http://127.0.0.1:1") as client:
        system = make_system(
            "fake", client=client, failure_policy="fallback", prompt_builder=broken
        )
        assert system.init is not None
        memory = system.init(system.variables, supplied, jax.random.key(1))
        with pytest.raises(RuntimeError, match="prompt builder"):
            system.apply(system.variables, memory, supplied, jax.random.key(2))


def test_custom_parser_cannot_mutate_masks_or_return_wrong_native_type(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    attempts: list[bool] = []

    def parser(
        _reply: str, _actor: ActorInput, masks: ActionMask, _history: History
    ) -> ActorAction:
        with pytest.raises(ValueError):
            np.asarray(masks.move_mask).flags.writeable = True
        attempts.append(True)
        return ActorAction(*(cast(Array, np.float32(0)) for _ in range(3)))

    model = Model("WAIT")
    supplied = inputs(game)
    with server(model.serve) as url, Client(url) as client:
        system = make_system(
            "fake", client=client, failure_policy="fallback", reply_parser=parser
        )
        assert system.init is not None
        memory = system.init(system.variables, supplied, jax.random.key(1))
        with pytest.raises(InvalidActionError):
            system.apply(system.variables, memory, supplied, jax.random.key(2))
        assert attempts


def test_dead_invalid_and_forced_actors_skip_model_work(
    game: tuple[Environment, Observations, EnvironmentState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.llm.system as module

    def forbidden(*_args: object, **_kwargs: object) -> str:
        pytest.fail("Forced actions must not build prompts or count tokens")

    monkeypatch.setattr(module, "format_actor_view", forbidden)
    monkeypatch.setattr(module, "_count", forbidden)
    supplied = inputs(game)
    move = np.zeros_like(supplied.action_mask.move_mask)
    move[..., 0] = True
    pair = np.zeros_like(supplied.action_mask.select_target_use_ultimate_joint_mask)
    pair[..., 0, 0] = True
    masks = ActionMask(
        *(
            cast(Array, value)
            for value in (move, pair.any(axis=-1), pair.any(axis=-2), pair)
        )
    )
    features = np.asarray(supplied.actors.observation.self_features).copy()
    features[0, 0, AGENT_FEATURE_ALIVE] = 0
    actors = supplied.actors._replace(
        observation=supplied.actors.observation._replace(
            self_features=cast(Array, features)
        )
    )
    supplied = supplied._replace(
        actors=actors, action_mask=masks, valid=cast(Array, np.asarray([True, False]))
    )
    with Client("http://127.0.0.1:1") as client:
        system = make_system("fake", client=client, history_turns=2)
        assert system.init is not None
        memory = system.init(system.variables, supplied, jax.random.key(1))
        output = cast(
            SystemOutput,
            system.apply(system.variables, memory, supplied, jax.random.key(2)),
        )
        assert output.next_memory[1] is memory[1]
        assert not output.next_memory[0][0]
        assert all(len(history) == 1 for history in output.next_memory[0][1:])


def test_team_b_failure_leaves_team_a_history_unchanged(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    _, observations, state = game
    good, bad = Model(), Model("invalid")
    with (
        server(good.serve) as first,
        server(bad.serve) as second,
        Client(first) as a_client,
        Client(second) as b_client,
    ):
        a = make_system("fake", client=a_client, history_turns=2)
        b = make_system("fake", client=b_client, history_turns=2)
        memory = init_systems(a, b, observations, state, jax.random.key(1))
        before = memory.team_a
        with pytest.raises(ReplyFormatError):
            apply_systems(a, b, memory, observations, state, jax.random.key(2))
        assert good.generations()
        assert memory.team_a is before
        assert all(not history for lane in before for history in lane)


def test_custom_counter_keeps_vllm_settings_and_needs_custom_identity() -> None:
    def counter(request: Mapping[str, object]) -> tuple[int, int]:
        return len(
            cast(list[dict[str, str]], request["messages"])[0]["content"]
        ), 100_000

    with Client("http://127.0.0.1:1") as client:
        first = make_system("fake", client=client)
        second = make_system("fake", client=client, token_counter=counter)
        assert first.variables.generation_json == second.variables.generation_json
        assert not second.variables.custom_identity_declared


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": None},
        {"top_p": 0},
        {"seed": True},
        {"ignore_eos": None},
        {"min_tokens": 65},
        {"top_k": 0},
        {"temperature": float("nan")},
    ],
)
def test_sampling_rejects_unresolved_values_before_network(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="sampling"):
        make_system("fake", sampling=settings)


def test_partial_game_reset_replaces_only_that_lanes_history(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    model = Model()
    with server(model.serve) as url, Client(url) as client:
        a = make_system("fake", client=client, history_turns=2)
        b = shared_policy(policy("random"))
        memory = init_systems(a, b, observations, state, jax.random.key(1))
        action, memory, _ = apply_systems(
            a, b, memory, observations, state, jax.random.key(2)
        )
        observations, state, *_ = env.step(jax.random.key(3), state, action)
        retained = memory.team_a[1][0][0]
        observations, state = env.reset(
            jax.random.key(4), state=state, reset_mask=jax.numpy.asarray([True, False])
        )
        _, memory, _ = apply_systems(
            a, b, memory, observations, state, jax.random.key(5)
        )
        assert all(len(history) == 1 for history in memory.team_a[0])
        assert all(len(history) == 2 for history in memory.team_a[1])
        assert memory.team_a[1][0][0] is retained
        assert memory.team_a[0][0][0].timestep == 0
        assert memory.team_a[1][0][-1].timestep == 1


def test_public_death_and_respawn_preserve_living_turn_history() -> None:
    import jax.numpy as jnp
    from tests.evaluation_fixtures import evaluation_env_config

    from marl_battlegrounds.core import env as core

    config = evaluation_env_config(team_sizes=(1, 1), max_steps=20)._replace(
        spawn_shield_duration_steps=0,
        team_respawn_wave_period_step_count=jnp.asarray((2, 2), jnp.int32),
    )
    initial, *_ = core.reset(config, jax.random.key(10))
    authored = initial._replace(
        step_count=jnp.int32(5),
        agent_positions=initial.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0)))
        .at[5]
        .set(jnp.asarray((6.5, 4.0))),
        current_health=initial.current_health.at[0].set(1.0).at[5].set(1.0),
        team_respawn_wave_countdowns=jnp.asarray((1, 1), jnp.int32),
    )
    prepared = core.initialize_scenario_state(authored, config)
    env = marl_bgs.make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(11), initial=prepared[:3])
    model = Model('{"move":"stay","combat":"enemy_0_basic"}')
    with server(model.serve) as url, Client(url) as client:
        a = make_system("fake", client=client, history_turns=2)
        b = make_system("fake", client=client, history_turns=2)
        memory = init_systems(a, b, observations, state, jax.random.key(12))
        first_entry = None
        for decision in range(3):
            action, memory, _ = apply_systems(
                a, b, memory, observations, state, jax.random.key(20 + decision)
            )
            if decision == 0:
                first_entry = memory.team_a[0][0][0]
                model.text = '{"move":"stay","combat":"no_combat"}'
            assert len(memory.team_a[0][0]) == (2 if decision == 2 else 1)
            assert memory.team_a[0][0][0] is first_entry
            assert len(model.generations()) == (4 if decision == 2 else 2)
            observations, state, *_ = env.step(
                jax.random.key(30 + decision), state, action
            )
            if decision < 2:
                np.testing.assert_array_equal(
                    np.asarray(state.core_state.alive_mask)[[0, 5]], [decision == 1] * 2
                )
        assert [entry.timestep for entry in memory.team_a[0][0]] == [5, 7]


def test_history_off_skips_snapshot_work(
    game: tuple[Environment, Observations, EnvironmentState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.llm.system as module

    def forbidden(_value: object) -> None:
        pytest.fail("Disabled history must not copy snapshots")

    supplied = inputs(game)
    model = Model()
    monkeypatch.setattr(module, "_snapshot", forbidden)
    with server(model.serve) as url, Client(url) as client:
        system = make_system("fake", client=client)
        assert system.init is not None
        memory = system.init(system.variables, supplied, jax.random.key(1))
        system.apply(system.variables, memory, supplied, jax.random.key(2))


def test_distinct_out_of_order_replies_keep_actor_lane_and_history(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    random = shared_policy(policy("random"))
    initial = init_systems(random, random, observations, state, jax.random.key(40))
    action, _, _ = apply_systems(
        random, random, initial, observations, state, jax.random.key(41)
    )
    next_observations, next_state, _, _, _ = env.step(jax.random.key(42), state, action)
    earlier = inputs(game)
    later = jax.device_get(env.policy_inputs(next_observations, next_state))

    # Each lane uses a real permitted public view from a different decision time.
    def different_times(a: Array, b: Array) -> Array:
        return cast(
            Array, np.concatenate((np.asarray(a)[:1], np.asarray(b)[1:]), axis=0)
        )

    supplied = jax.tree.map(different_times, earlier, later)
    condition = threading.Condition()
    next_reply = 9
    replies: list[int] = []
    expected: dict[int, int] = {}

    def builder(actor: ActorInput, masks: ActionMask, history: History) -> str:
        del history
        tick = int(actor.observation.context_features[CONTEXT_FEATURE_CURRENT_TIMESTEP])
        slot = int(actor.observation.self_ally_index)
        identity = tick * 5 + slot
        moves = legal_action_names(masks)["move"]
        move = moves[identity % len(moves)]
        with condition:
            expected[identity] = MOVE_NAMES.index(move)
        return json.dumps({"identity": identity, "move": move})

    def reply(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        nonlocal next_reply
        request = json.loads(payload["messages"][0]["content"])
        identity = request["identity"]
        with condition:
            assert condition.wait_for(lambda: identity == next_reply, timeout=3)
            send(
                handler,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {"move": request["move"], "combat": "no_combat"}
                                )
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1},
                },
            )
            replies.append(identity)
            next_reply -= 1
            condition.notify_all()

    with server(reply) as url, Client(url) as client:
        method = make_system(
            "fake",
            client=client,
            server_type="chat",
            frame="world",
            history_turns=2,
            prompt_builder=builder,
            token_counter=lambda _request: (1, 100),
        )
        assert method.init is not None
        memory = method.init(method.variables, supplied, jax.random.key(43))
        output = cast(
            SystemOutput,
            method.apply(method.variables, memory, supplied, jax.random.key(44)),
        )
    assert replies == list(reversed(range(10)))
    for lane in range(2):
        for slot in range(5):
            target = expected[lane * 5 + slot]
            assert int(output.actions.move[lane, slot]) == target
            entry = output.next_memory[lane][slot][-1]
            assert entry.timestep == lane
            assert int(entry.actor.observation.self_ally_index) == slot
            assert int(entry.action.move) == target
    assert all(not history for lane in memory for history in lane)


def test_chat_mode_uses_no_tokenizer_or_vllm_generation_fields(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    model = Model()
    supplied = inputs(game)
    with server(model.serve) as url, Client(url) as client:
        method = make_system("fake", client=client, server_type="chat")
        assert method.init is not None
        memory = method.init(method.variables, supplied, jax.random.key(1))
        method.apply(method.variables, memory, supplied, jax.random.key(2))
    assert len(model.calls) == 10
    assert all(path == "/v1/chat/completions" for path, _ in model.calls)
    for _, request in model.calls:
        assert not {
            "top_k",
            "min_p",
            "repetition_penalty",
            "chat_template_kwargs",
            "ignore_eos",
        }.intersection(request)
        assert request["temperature"] == 0
        assert request["max_tokens"] == 64


def test_cancellation_cannot_become_fallback_or_adopt_history(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    entered, release = threading.Event(), threading.Event()
    model = Model()

    def stalled(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        if handler.path.endswith("chat/completions"):
            entered.set()
            assert release.wait(3)
        model.serve(handler, payload)

    supplied = inputs(game)
    with server(stalled) as url, Client(url, concurrency=2) as client:
        method = make_system(
            "fake", client=client, history_turns=2, failure_policy="fallback"
        )
        assert method.init is not None
        memory = method.init(method.variables, supplied, jax.random.key(1))
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                method.apply, method.variables, memory, supplied, jax.random.key(2)
            )
            try:
                assert entered.wait(2)
                client.close()
                with pytest.raises(RequestCancelledError):
                    result.result(timeout=1)
            finally:
                release.set()
    assert all(not history for lane in memory for history in lane)


def test_default_fitting_renders_each_view_once_and_preserves_exact_text(
    game: tuple[Environment, Observations, EnvironmentState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.llm.system as module
    from marl_battlegrounds.llm.actions import COMBAT_NAMES

    real_format = module.format_actor_view
    tracked = Mock(wraps=real_format)
    counted_prompts: list[str] = []

    def counter(request: Mapping[str, object]) -> tuple[int, int]:
        prompt = cast(list[dict[str, str]], request["messages"])[0]["content"]
        counted_prompts.append(prompt)
        return (100_000 if "Earlier turn " in prompt else len(prompt), 30_000)

    model = Model()
    supplied = inputs(game)
    with server(model.serve) as url, Client(url, concurrency=1) as client:
        method = make_system(
            "fake", client=client, frame="world", history_turns=2, token_counter=counter
        )
        assert method.init is not None
        memory = method.init(method.variables, supplied, jax.random.key(1))
        for _ in range(2):
            memory = cast(
                SystemOutput,
                method.apply(method.variables, memory, supplied, jax.random.key(2)),
            ).next_memory
        counted_prompts.clear()
        monkeypatch.setattr(module, "format_actor_view", tracked)
        method.apply(method.variables, memory, supplied, jax.random.key(3))
    assert tracked.call_count == 30  # Ten actors: current plus two older views each.
    assert len(counted_prompts) == 30  # All three fitted candidates are counted.
    expected: list[str] = []
    for lane in range(2):
        for slot in range(5):

            def take(value: Array, lane: int = lane, slot: int = slot) -> Array:
                return value[lane, slot]

            actor = jax.tree.map(take, supplied.actors)
            masks = jax.tree.map(take, supplied.action_mask)
            current = real_format(actor, masks, frame="world")
            for count in (2, 1, 0):
                history = memory[lane][slot][-count:] if count else ()
                lines = [current]
                if history:
                    lines.append(
                        "Earlier turns follow. Their menus are historical; "
                        "use only the current menu above."
                    )
                    for entry in history:
                        lines.extend(
                            [
                                f"Earlier turn {entry.timestep}:",
                                real_format(
                                    entry.actor,
                                    entry.masks,
                                    frame="world",
                                    include_static=False,
                                    reply_instruction=None,
                                ),
                                f"Submitted: move={MOVE_NAMES[int(entry.action.move)]} "
                                f"combat={COMBAT_NAMES[int(entry.action.select_target)][int(entry.action.use_ultimate)]}",
                            ]
                        )
                    lines.append(
                        "End of history. Choose for the current view "
                        "and its current legal menu."
                    )
                expected.append("\n".join(lines))
    assert counted_prompts == expected


def test_custom_history_reuses_the_same_immutable_hook_inputs(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    seen: list[tuple[ActorInput, ActionMask]] = []

    def builder(actor: ActorInput, masks: ActionMask, _history: History) -> str:
        seen.append((actor, masks))
        return "Choose Stay and no combat"

    supplied = inputs(game)
    model = Model()
    with server(model.serve) as url, Client(url) as client:
        method = make_system(
            "fake", client=client, history_turns=2, prompt_builder=builder
        )
        assert method.init is not None
        memory = method.init(method.variables, supplied, jax.random.key(1))
        output = cast(
            SystemOutput,
            method.apply(method.variables, memory, supplied, jax.random.key(2)),
        )
    assert len(seen) == 10
    for lane in output.next_memory:
        for history in lane:
            entry = history[-1]
            assert any(
                entry.actor is actor and entry.masks is masks for actor, masks in seen
            )
            for value in jax.tree.leaves((entry.actor, entry.masks)):
                with pytest.raises(ValueError):
                    np.asarray(value).flags.writeable = True


def test_owned_nonprefix_actor_requests_preserve_full_views_and_slot_history(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = game
    model = Model()
    with server(model.serve) as url, Client(url) as client:
        member = make_system("fake", client=client, history_turns=2)
        method = marl_bgs.team(member, "random", slots=[[4], [0, 1, 2, 3]])
        opponent = shared_policy(policy("random"))
        memory = init_systems(method, opponent, observations, state, jax.random.key(7))
        actions, memory, _ = apply_systems(
            method, opponent, memory, observations, state, jax.random.key(8)
        )
        env.step(jax.random.key(9), state, actions)
        assert len(model.generations()) == 2
        for lane in memory.team_a.members[0]:
            assert [len(history) for history in lane] == [0, 0, 0, 0, 1]
            assert int(lane[4][0].actor.observation.self_ally_index) == 4
        model.calls.clear()
        inputs_now = inputs(game)._replace(controlled_mask=np.zeros((2, 5), np.bool_))
        result = member.apply(
            member.variables,
            memory.team_a.members[0],
            inputs_now,
            jax.random.split(jax.random.key(10), 2),
        )
        assert not model.calls
        assert result.next_memory == memory.team_a.members[0]
