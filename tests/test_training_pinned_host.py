"""Check a pinned host method through the collector's host route.

One counting rule written twice, as a JAX System and as a host System with
Python list memory, gives identical rollout rows, padding and budget suffix
through collect_training_rollout on fresh collections, and again when both
collections record through a writer. Setup opens no host memory for any lane.
The host holder equals M8's own full-batch host call round by round, for a
generic System and for a Policy adapter that uses its keys: it computes a
Policy on active slots in pinned games only, makes no call in a round with no
pinned game, returns zero actions for games it does not play, leaves inactive
slots' actions and memory at zero, and starts a lane's memory afresh when a
pinned lane begins a new game.
A host method's error keeps its own type, KeyboardInterrupt included, and
afterwards the collection refuses every call; a carry from an earlier round is
refused even after success, so host memory never meets an older game state. A
host method with no init, no reset hook and no template counts as memory-free,
and M8's own reset rule refuses any memory it returns. scan_training_rollout
and advance_training_step refuse a host-pinned collection because they cannot
call the host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines.ppo import initialize_ppo, make_recurrent_mappo_system
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.environment import make
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    SystemInput,
    _host_apply,  # pyright: ignore[reportPrivateUsage]
    _initial_memory,  # pyright: ignore[reportPrivateUsage]
    shared_policy,
    system_inputs,
)
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.system_evaluation import (
    freeze_evaluation_method,
    prepare_evaluation_system,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    prepare_training_content,
)
from marl_battlegrounds.training.collection import (
    TrainingCarry,
    TrainingCollection,
    advance_training_step,
    collect_training_rollout,
    init_training_collection,
    scan_training_rollout,
)
from marl_battlegrounds.training.curriculum import make_training_schedule
from marl_battlegrounds.training.opponents import HostOpponent

type Tree = Any


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def weights() -> Tree:
    return initialize_ppo(jax.random.key(61)).actor_params


def _collection(
    prepared: PreparedTrainingContent,
    weights: Tree,
    pinned: System,
    *,
    total: int = 24,
    recording: bool = False,
) -> tuple[TrainingCollection, TrainingCarry]:
    actor = make_recurrent_mappo_system(weights, spawn_frame="world")
    collection, carry = init_training_collection(
        actor,
        weights,
        schedule=make_training_schedule(
            total_env_steps=total, num_envs=4, early_history_capture=True
        ),
        seed=62,
        prepared=prepared,
        metrics="none",
        recording=recording,
        pinned_opponent_share=0.5,
        pinned_opponent=pinned,
    )
    history = carry.history._replace(
        lane_snapshot=jnp.asarray((-1, -2, -1, -2), jnp.int32),
    )
    return collection, carry._replace(history=history)


def _jax_init(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, jnp.int32)


def _jax_apply(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Array]:
    del variables, keys
    move = jnp.broadcast_to((memory % 9)[:, None], (memory.shape[0], 5))
    zero = jnp.zeros_like(move)
    return ActorAction(move, zero, zero), memory + inputs.valid.astype(jnp.int32)


class _Counter:
    def __init__(
        self, fail_at: int | None = None, error: type[BaseException] = RuntimeError
    ) -> None:
        self.opened = 0
        self.calls = 0
        self.fail_at = fail_at
        self.error = error

    def init(self, variables: Tree, inputs: SystemInput, keys: Array) -> list[int]:
        del variables, keys
        self.opened += int(np.sum(np.asarray(inputs.valid)))
        return [0] * len(np.asarray(inputs.valid))

    def apply(
        self, variables: Tree, memory: list[int], inputs: SystemInput, keys: Array
    ) -> tuple[ActorAction, list[int]]:
        del variables, keys
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise self.error("provider unavailable")
        valid = np.asarray(inputs.valid)
        move = np.broadcast_to(
            (np.asarray(memory, np.int32) % 9)[:, None], (len(memory), 5)
        ).astype(np.int32)
        zero = np.zeros_like(move)
        following = [
            count + int(flag) for count, flag in zip(memory, valid, strict=True)
        ]
        return ActorAction(move, zero, zero), following  # pyright: ignore[reportArgumentType]

    def system(self) -> System:
        return System("Host Counter", self.apply, init=self.init, execution="host")


def test_the_host_route_equals_its_jax_twin_and_opens_nothing_at_setup(
    prepared: PreparedTrainingContent, weights: Tree
) -> None:
    twin = System("Jax Counter", _jax_apply, init=_jax_init)
    host = _Counter()
    jax_collection, jax_carry = _collection(prepared, weights, twin)
    host_collection, host_carry = _collection(prepared, weights, host.system())
    assert host.opened == 0
    assert host_collection.pinned_opponent is not None
    assert host_collection.pinned_opponent["memory_rule"] == "host only, not saved"
    pinned_rows = 0
    # Six real rounds in three blocks of four: the last block is budget padding.
    for _ in range(3):
        jax_carry, jax_rollout = collect_training_rollout(
            jax_collection, jax_carry, length=4
        )
        host_carry, host_rollout = collect_training_rollout(
            host_collection, host_carry, length=4
        )
        for left, right in zip(
            jax.tree.leaves(jax_rollout), jax.tree.leaves(host_rollout), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        rows = host_rollout.transitions
        pinned = (np.asarray(rows.opponent_snapshot) == -2) & np.asarray(rows.valid)
        pinned_rows += int(pinned.sum())
        assert (np.asarray(rows.opponent_update)[pinned] == -2).all()
    assert pinned_rows > 0
    assert host.opened > 0 and host.calls > 0


HOST_CALLS: list[int] = []


def _steps_apply(
    variables: Tree, carry: Tree, actor: ActorInput, mask: ActionMask, key: Array
) -> tuple[ActorAction, Array]:
    del variables, actor, mask
    HOST_CALLS.append(1)
    count = jnp.asarray(carry, jnp.int32)[0]
    move = (count + jax.random.randint(key, (), 0, 9)) % 9
    zero = jnp.int32(0)
    return ActorAction(move, zero, zero), jnp.asarray(carry) + 1.0


# Pinned lanes, reset generations, and the lanes M8 must start afresh per round.
ROUNDS = (
    ((False, True, False, True), (0, 0, 0, 0), (False, True, False, True)),
    ((False, False, False, False), (0, 0, 0, 0), (False, False, False, False)),
    ((False, True, False, True), (0, 0, 0, 0), (False, False, False, False)),
    ((True, True, False, True), (1, 1, 0, 0), (True, True, False, False)),
)


@pytest.mark.parametrize("form", ["system", "policy"])
def test_the_host_holder_equals_m8_round_by_round(form: str) -> None:
    counter = _Counter()
    if form == "system":
        method = counter.system()
    else:
        source = Policy(
            "Steps",
            _steps_apply,
            initial_carry=np.zeros(1, np.float32),
            execution="host",
        )
        method = shared_policy(cast(Policy, freeze_evaluation_method(source)))
    env = make(
        "tdm",
        env_config=evaluation_env_config(team_sizes=(3, 3), max_steps=20),
        num_envs=4,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(70))
    inputs = system_inputs(observations, state, team=1)
    active = np.asarray(inputs.active_mask)
    keys = jax.random.split(jax.random.key(71), 4)
    init_keys = jax.random.split(jax.random.key(72), 4)
    holder = HostOpponent(method, 4)
    holder.start(jax.device_get(inputs), init_keys)
    execution, variables, template = prepare_evaluation_system(method)
    idle = jax.device_get(inputs)._replace(valid=np.zeros(4, np.bool_))
    memory = _initial_memory(execution, variables, template, idle, init_keys)
    for index, (pinned, generations, reset) in enumerate(ROUNDS):
        lanes = np.asarray(pinned)
        HOST_CALLS.clear()
        before = counter.calls
        actions = holder.act(
            index, inputs, lanes, keys, init_keys, np.asarray(generations)
        )
        calls = counter.calls
        if form == "policy":
            assert len(HOST_CALLS) == np.count_nonzero(active & lanes[:, None])
            for head in actions:
                np.testing.assert_array_equal(np.asarray(head)[~active], 0)
            np.testing.assert_array_equal(np.asarray(holder.memory)[~active], 0.0)
        assert holder.next_round == index + 1
        if not lanes.any():
            assert calls == before
            for head in actions:
                np.testing.assert_array_equal(np.asarray(head), 0)
            continue
        reference = _host_apply(
            execution,
            variables,
            template,
            memory,
            inputs._replace(valid=lanes),
            keys,
            init_keys,
            jnp.asarray(reset),
            keep_learning_outputs=False,
        )
        memory = reference.next_memory
        for head, expected in zip(actions, reference.actions, strict=True):
            np.testing.assert_array_equal(
                np.asarray(head)[lanes], np.asarray(expected)[lanes]
            )
            np.testing.assert_array_equal(np.asarray(head)[~lanes], 0)
        for mine, theirs in zip(
            jax.tree.leaves(holder.memory), jax.tree.leaves(memory), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


def _writer(root: Path, collection: TrainingCollection) -> RunWriter:
    policies: dict[str, object] = {
        team: normalize_system_registration(system, phase="training")[1]
        for team, system in (
            ("team_a", collection.actor),
            ("team_b", collection.opponent),
        )
    }
    return RunWriter(output_dir=root, phase="training", policies=policies)


def test_the_recorded_host_route_equals_its_recorded_jax_twin(
    prepared: PreparedTrainingContent, weights: Tree, tmp_path: Path
) -> None:
    twin = System("Jax Counter", _jax_apply, init=_jax_init)
    jax_collection, jax_carry = _collection(prepared, weights, twin, recording=True)
    host_collection, host_carry = _collection(
        prepared, weights, _Counter().system(), recording=True
    )
    jax_writer = _writer(tmp_path / "jax", jax_collection)
    host_writer = _writer(tmp_path / "host", host_collection)
    try:
        for _ in range(2):
            jax_carry, jax_rollout = collect_training_rollout(
                jax_collection, jax_carry, length=4, writer=jax_writer
            )
            host_carry, host_rollout = collect_training_rollout(
                host_collection, host_carry, length=4, writer=host_writer
            )
            for left, right in zip(
                jax.tree.leaves(jax_rollout),
                jax.tree.leaves(host_rollout),
                strict=True,
            ):
                np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    finally:
        jax_writer.close()
        host_writer.close()


def test_a_failed_host_method_keeps_its_error_and_the_collection_refuses_reuse(
    prepared: PreparedTrainingContent, weights: Tree
) -> None:
    host = _Counter(fail_at=6)
    collection, carry = _collection(prepared, weights, host.system())
    with pytest.raises(RuntimeError, match="provider unavailable"):
        collect_training_rollout(collection, carry, length=8)
    with pytest.raises(RuntimeError, match="failed at round"):
        collect_training_rollout(collection, carry, length=8)
    interrupted = _Counter(fail_at=1, error=KeyboardInterrupt)
    collection, carry = _collection(prepared, weights, interrupted.system())
    with pytest.raises(KeyboardInterrupt):
        collect_training_rollout(collection, carry, length=2)


def test_an_older_carry_is_refused_after_the_holder_moved_on(
    prepared: PreparedTrainingContent, weights: Tree
) -> None:
    collection, start = _collection(prepared, weights, _Counter().system())
    later, _ = collect_training_rollout(collection, start, length=2)
    with pytest.raises(ValueError, match="serves each round once"):
        collect_training_rollout(collection, start, length=2)
    collect_training_rollout(collection, later, length=2)


def _memoryless_apply(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, list[int]]:
    del variables, memory, keys
    count = len(np.asarray(inputs.valid))
    zero = np.zeros((count, 5), np.int32)
    return ActorAction(zero, zero, zero), [0] * count  # pyright: ignore[reportArgumentType]


def test_memory_from_a_method_classified_memory_free_is_refused(
    prepared: PreparedTrainingContent, weights: Tree
) -> None:
    system = System("Leaky", _memoryless_apply, execution="host")
    collection, carry = _collection(prepared, weights, system)
    assert collection.host_opponent is not None
    assert not collection.host_opponent.stateful
    with pytest.raises(TypeError, match="structure established by init"):
        collect_training_rollout(collection, carry, length=2)
    with pytest.raises(RuntimeError, match="failed at round"):
        collect_training_rollout(collection, carry, length=2)


def test_pure_helpers_refuse_a_host_pinned_collection(
    prepared: PreparedTrainingContent, weights: Tree
) -> None:
    collection, carry = _collection(prepared, weights, _Counter().system())
    with pytest.raises(ValueError, match="cannot call a pinned host method"):
        scan_training_rollout(collection, carry, length=2)
    with pytest.raises(ValueError, match="cannot call a pinned host method"):
        advance_training_step(collection, carry)


@pytest.mark.parametrize("state", ("closed", "failed", "busy", "pending"))
def test_bad_writer_is_refused_before_a_provider_call(
    prepared: PreparedTrainingContent, weights: Tree, tmp_path: Path, state: str
) -> None:
    host = _Counter()
    collection, carry = _collection(prepared, weights, host.system(), recording=True)
    writer = _writer(tmp_path, collection)
    if state == "closed":
        writer.close()
    elif state == "failed":
        writer.record_failure(RuntimeError("Earlier failure"))
    elif state == "pending":
        writer._pending_numerical_starts.add(1)  # pyright: ignore[reportPrivateUsage]
    else:
        writer._collection_active = True  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(RuntimeError):
            collect_training_rollout(collection, carry, length=2, writer=writer)
        assert host.opened == host.calls == 0
    finally:
        writer._collection_active = False  # pyright: ignore[reportPrivateUsage]
        writer.close()


@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_provider_failure_marks_the_writer_failed_with_the_original_error(
    prepared: PreparedTrainingContent,
    weights: Tree,
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    host = _Counter(fail_at=1, error=error_type)
    collection, carry = _collection(prepared, weights, host.system(), recording=True)
    writer = _writer(tmp_path, collection)
    try:
        with pytest.raises(error_type, match="provider unavailable") as caught:
            collect_training_rollout(collection, carry, length=2, writer=writer)
        assert writer._failed is caught.value  # pyright: ignore[reportPrivateUsage]
        assert any("Run directory:" in note for note in caught.value.__notes__)
        with pytest.raises(RuntimeError, match="writer failed"):
            writer.flush()
    finally:
        writer.close()
