"""Keep declared future learner clocks separate from a child's added budget.

The extension runner resolves these small host records before writing files.
Learners read the same records when updating or checking saved state. Original
optimizer, sample, replay and random-key clocks are never restarted here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

_PPO_METHODS = frozenset(("mappo", "ippo", "ff_mappo", "ff_ippo"))
_MAX_COUNT = 2**31 - 1
_PQN_FLOOR = 1e-10
CONTINUATION_CHANGE_KEYS = frozenset(
    {
        "validation",
        "learning_rate",
        "exploration",
        "history_capture_env_steps",
        "past_capture_interval",
        "keep_past",
        "opponents",
        "reward",
        "partners",
        "shaping",
        "shaping_coefficient",
        "shaping_mode",
        "seed",
        "curriculum",
        "ppo",
        "qmix",
        "pqn",
    }
)
_LOSS_SETTINGS: dict[str, frozenset[str]] = {
    "ppo": frozenset(
        {
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "entropy_coefficient",
            "value_coefficient",
            "max_grad_norm",
        }
    ),
    "qmix": frozenset({"gamma"}),
    "pqn": frozenset({"gamma", "td_lambda", "max_grad_norm"}),
}


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Read a plain count bounded by int32; reject bool and other types."""
    if type(value) is not int or not minimum <= value <= _MAX_COUNT:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_MAX_COUNT}]")
    return value


def _number(value: object, name: str, *, probability: bool = False) -> float:
    """Read a finite positive rate, or a probability including zero and one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (
        not 0 <= result <= 1 if probability else result <= 0
    ):
        raise ValueError(f"{name} is outside its allowed range")
    return result


@dataclass(frozen=True)
class LinearSchedule:
    """Describe one constant or linear future rule on an absolute clock.

    start is the absolute count at the fork, value is the first future value,
    end is the value after duration calls, and duration=0 means constant.
    These plain host values become constants inside the existing compiled loop.
    """

    start: int
    value: float
    end: float
    duration: int

    def at(self, count: int) -> float:
        """Return the host value at a count before an action or optimizer call."""
        if not self.duration:
            return self.value
        elapsed = min(max(count - self.start, 0), self.duration)
        if elapsed == self.duration:
            return self.end
        return (self.value - self.end) * (1.0 - elapsed / self.duration) + self.end

    def at_array(self, count: Any) -> Any:  # noqa: ANN401
        """Return the same float32 scalar inside JAX without changing any clock."""
        import jax.numpy as jnp

        if not self.duration:
            return jnp.float32(self.value)
        elapsed = jnp.clip(count - self.start, 0, self.duration)
        rate = (self.value - self.end) * (
            1.0 - elapsed.astype(jnp.float32) / self.duration
        ) + self.end
        return jnp.where(elapsed >= self.duration, jnp.float32(self.end), rate)


@dataclass(frozen=True)
class RewardReset:
    """Save the last reward change without restarting any cumulative clock.

    rounds, blocks and learning_blocks are the checked boundary where stored
    rows were cleared. last_refresh_rounds preserves the actual last actor
    publication while new rows arrive. pqn_learning_rounds counts real rows
    already used by PQN learning blocks, excluding all earlier refill windows.
    These are fixed offsets, not counters advanced during training.
    """

    rounds: int
    blocks: int
    learning_blocks: int
    last_refresh_rounds: int
    pqn_learning_rounds: int = 0


@dataclass(frozen=True)
class LearnerContinuation:
    """Hold the checked block origin and active future rules for one child.

    start_rounds, start_blocks and start_learning_blocks are the saved absolute
    boundary. pqn_planned_learning_blocks preserves the original horizon.
    actor_lr, critic_lr and q_lr replace only future constant rates when present.
    pqn_rate and exploration keep their own absolute origins across descendants.
    frozen_epsilon pairs with frozen_capture_ids, so reusing a physical bank
    slot cannot change an old copy's exploration rule. pinned_epsilon preserves
    the separate permanent pin. New copies follow the child's active rule.
    allow_terminal_rate records an explicit future-rate choice and passes to
    descendants, so an already accepted terminal rate needs no new permission.
    loss_settings keeps supported scalar overrides as immutable name/value pairs.
    reward_reset records the last explicit Q-reward change. Descendants retain
    it until another reward change clears their stored rows.
    """

    method: str
    start_rounds: int
    start_blocks: int
    start_learning_blocks: int
    pqn_planned_learning_blocks: int | None = None
    actor_lr: float | None = None
    critic_lr: float | None = None
    q_lr: float | None = None
    pqn_rate: LinearSchedule | None = None
    exploration: LinearSchedule | None = None
    frozen_epsilon: tuple[float, ...] = ()
    allow_terminal_rate: bool = False
    loss_settings: tuple[tuple[str, float], ...] = ()
    frozen_capture_ids: tuple[int, ...] = ()
    pinned_epsilon: float | None = None
    reward_reset: RewardReset | None = None


def learner_continuation(
    value: Mapping[str, object] | None,
) -> LearnerContinuation | None:
    """Validate a saved learner mapping and return its immutable host view.

    None means a fresh run. Unknown fields, methods, bad counts or rates fail
    before device work. This checks the declaration's structure; the extension
    owner must bind its origin and inherited rates to the verified parent.
    """
    if value is None:
        return None
    data = dict(value)
    version = data.pop("schema_version", None)
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported learner continuation schema")
    data.setdefault("loss_settings", ())
    data.setdefault(
        "frozen_capture_ids",
        tuple(range(len(cast(tuple[object, ...], data.get("frozen_epsilon", ()))))),
    )
    data.setdefault("pinned_epsilon", None)
    data.setdefault("reward_reset", None)
    fields = set(LearnerContinuation.__dataclass_fields__)
    if set(data) != fields:
        raise ValueError("Learner continuation fields differ")
    if type(data["allow_terminal_rate"]) is not bool:
        raise ValueError("Terminal-rate permission must be bool")
    method = data["method"]
    if not isinstance(method, str) or method not in _PPO_METHODS | {"qmix", "pqn_vdn"}:
        raise ValueError("Unknown continuation method")
    for name in ("start_rounds", "start_blocks", "start_learning_blocks"):
        data[name] = _integer(data[name], name)
    if cast(int, data["start_learning_blocks"]) > cast(int, data["start_blocks"]):
        raise ValueError("Learning blocks exceed completed blocks")
    reset = data["reward_reset"]
    if reset is not None:
        if method in _PPO_METHODS or not isinstance(reset, Mapping):
            raise ValueError("Only Q-learning can have a stored-reward reset")
        reset = dict(cast(Mapping[str, object], reset))
        if set(reset) != set(RewardReset.__dataclass_fields__):
            raise ValueError("Reward reset fields differ")
        checked = {name: _integer(value, name) for name, value in reset.items()}
        if (
            checked["rounds"] > cast(int, data["start_rounds"])
            or checked["blocks"] > cast(int, data["start_blocks"])
            or checked["learning_blocks"] > cast(int, data["start_learning_blocks"])
            or checked["learning_blocks"] > checked["blocks"]
            or checked["last_refresh_rounds"] > checked["rounds"]
            or checked["pqn_learning_rounds"] > checked["rounds"]
            or (method != "pqn_vdn" and checked["pqn_learning_rounds"] != 0)
        ):
            raise ValueError("Reward reset counts exceed the saved boundary")
        data["reward_reset"] = RewardReset(**checked)
    horizon = data["pqn_planned_learning_blocks"]
    if method == "pqn_vdn":
        data["pqn_planned_learning_blocks"] = _integer(
            horizon, "PQN horizon", minimum=1
        )
    elif horizon is not None:
        raise ValueError("Only PQN has a planned learning-block horizon")
    for name in ("actor_lr", "critic_lr", "q_lr"):
        if data[name] is not None:
            data[name] = _number(data[name], name)
    if method in _PPO_METHODS:
        if (
            data["q_lr"] is not None
            or data["pqn_rate"] is not None
            or data["exploration"] is not None
        ):
            raise ValueError("PPO continuation cannot change Q-learning schedules")
    elif data["actor_lr"] is not None or data["critic_lr"] is not None:
        raise ValueError("Q-learning continuation cannot set PPO rates")
    if method == "pqn_vdn" and data["q_lr"] is not None:
        raise ValueError("PQN rates belong to its explicit future optimizer schedule")
    if method != "pqn_vdn" and data["allow_terminal_rate"]:
        raise ValueError("Only PQN can permit its terminal rate")
    if method != "pqn_vdn" and data["pqn_rate"] is not None:
        raise ValueError("Only PQN has a future optimizer schedule")
    for name in ("pqn_rate", "exploration"):
        raw = data[name]
        if raw is None:
            continue
        if not isinstance(raw, Mapping) or set(cast(Mapping[str, object], raw)) != {
            "start",
            "value",
            "end",
            "duration",
        }:
            raise ValueError(f"Invalid {name} schedule")
        raw = cast(Mapping[str, object], raw)
        if name == "exploration" and method == "qmix":
            start = raw["start"]
            if type(start) is not int or start < 0:
                raise ValueError("Exploration start must be a nonnegative integer")
        else:
            start = _integer(raw["start"], f"{name} start")
        duration = _integer(raw["duration"], f"{name} duration")
        if (
            not (name == "exploration" and method == "qmix")
            and start + duration > _MAX_COUNT
        ):
            raise ValueError(f"{name} end exceeds int32")
        data[name] = LinearSchedule(
            start,
            _number(raw["value"], f"{name} value", probability=name == "exploration"),
            _number(raw["end"], f"{name} end", probability=name == "exploration"),
            duration,
        )
        rule = cast(LinearSchedule, data[name])
        if not duration and rule.value != rule.end:
            raise ValueError("A constant schedule must keep its initial value")
        if name == "pqn_rate" and duration and rule.end != _PQN_FLOOR:
            raise ValueError("PQN future decay must keep its terminal rate 1e-10")
    frozen = data["frozen_epsilon"]
    if not isinstance(frozen, (tuple, list)):
        raise ValueError("Frozen exploration proof must be a sequence of rates")
    data["frozen_epsilon"] = tuple(
        _number(item, "Frozen epsilon", probability=True)
        for item in cast(tuple[object, ...] | list[object], frozen)
    )
    capture_ids_raw = data["frozen_capture_ids"]
    if not isinstance(capture_ids_raw, (tuple, list)):
        raise ValueError("Frozen capture IDs must be a sequence")
    capture_ids = cast(tuple[object, ...] | list[object], capture_ids_raw)
    if (
        len(capture_ids) != len(data["frozen_epsilon"])
        or any(
            type(value) is not int or not 0 <= value <= _MAX_COUNT
            for value in capture_ids
        )
        or len(set(capture_ids)) != len(capture_ids)
    ):
        raise ValueError(
            "Frozen capture IDs must be distinct and match the saved rates"
        )
    data["frozen_capture_ids"] = tuple(capture_ids)
    if data["pinned_epsilon"] is not None:
        data["pinned_epsilon"] = _number(
            data["pinned_epsilon"], "Pinned epsilon", probability=True
        )
    if method in _PPO_METHODS and data["frozen_epsilon"]:
        raise ValueError("PPO has no frozen exploration rates")
    family = (
        "ppo" if method in _PPO_METHODS else "pqn" if method == "pqn_vdn" else "qmix"
    )
    settings = data["loss_settings"]
    if not isinstance(settings, (list, tuple)):
        raise ValueError("Saved loss settings must be name/value pairs")
    pairs: list[tuple[str, float]] = []
    for pair in cast(list[object], settings):
        if not isinstance(pair, (tuple, list)) or len(cast(list[object], pair)) != 2:
            raise ValueError("Saved loss settings must be name/value pairs")
        name, setting_value = cast(tuple[object, object], pair)
        if (
            not isinstance(name, str)
            or name not in _LOSS_SETTINGS[family]
            or name in dict(pairs)
        ):
            raise ValueError("Unsupported or repeated loss setting")
        if (
            isinstance(setting_value, bool)
            or not isinstance(setting_value, (int, float))
            or not math.isfinite(setting_value)
        ):
            raise ValueError("Loss settings must be finite numbers")
        pairs.append((name, float(setting_value)))
    data["loss_settings"] = tuple(pairs)
    return LearnerContinuation(**cast(dict[str, Any], data))


def schedule_continuation(schedule: Any) -> LearnerContinuation | None:  # noqa: ANN401
    """Read the learner record from a TrainingSchedule without importing JAX."""
    declaration = getattr(schedule, "continuation", None)
    if declaration is None:
        return None
    return learner_continuation(declaration.get("learner"))


def reward_refill_end(
    continuation: LearnerContinuation | None,
    *,
    initial_rounds: int,
    memory_window: int,
) -> int:
    """Return PQN's absolute end of initial collection or reward refill.

    A later reward change needs only memory_window clean rows. If the original
    initial phase is unfinished, its remaining boundary still applies.
    """
    reset = None if continuation is None else continuation.reward_reset
    return max(initial_rounds, 0 if reset is None else reset.rounds + memory_window)


def pqn_learning_rounds(
    rounds: int,
    *,
    continuation: LearnerContinuation | None,
    initial_rounds: int,
    memory_window: int,
) -> int:
    """Count real new rows used by PQN learning, omitting all refill windows.

    This derives sample totals from the existing round clock and the last saved
    reset offset. It does not count reused prefix rows or advance another clock.
    """
    reset = None if continuation is None else continuation.reward_reset
    before = 0 if reset is None else reset.pqn_learning_rounds
    end = reward_refill_end(
        continuation, initial_rounds=initial_rounds, memory_window=memory_window
    )
    return before + max(0, rounds - end)


def continuation_counts(
    rounds: int,
    *,
    continuation: LearnerContinuation,
    rollout_length: int,
    initial_rounds: int = 0,
    minimum: int = 0,
) -> tuple[int, int]:
    """Count cumulative accepted and learned blocks from the actual fork.

    rounds is absolute collected rounds; rollout_length is the unchanged block
    capacity. PQN passes its original W as initial_rounds. QMIX passes replay's
    minimum row count as minimum. The last child block may be partial. All
    arithmetic is host-only; this does not decide whether rounds is reachable.
    """
    start = continuation.start_rounds
    if rounds < start or rollout_length < 1:
        raise ValueError(
            "Continuation rounds precede its start or block size is invalid"
        )
    added = rounds - start
    if continuation.method == "pqn_vdn":
        refill_end = reward_refill_end(
            continuation,
            initial_rounds=initial_rounds,
            memory_window=initial_rounds - rollout_length,
        )
        warm = max(0, min(rounds, refill_end) - start)
        learned = max(0, rounds - max(start, refill_end))
        new_learning = -(-learned // rollout_length)
        new_blocks = -(-warm // rollout_length) + new_learning
    else:
        new_blocks = -(-added // rollout_length)
        new_learning = new_blocks
        if continuation.method == "qmix":
            if continuation.reward_reset is not None:
                minimum += continuation.reward_reset.rounds
            first = max(1, -(-(minimum - start) // rollout_length))
            new_learning = max(0, new_blocks - first + 1) if rounds >= minimum else 0
    return (
        continuation.start_blocks + new_blocks,
        continuation.start_learning_blocks + new_learning,
    )


def continuation_boundary(
    requested_rounds: int,
    *,
    total_rounds: int,
    continuation: LearnerContinuation,
    rollout_length: int,
    initial_rounds: int = 0,
    minimum: int = 0,
) -> tuple[int, int, int]:
    """Round an absolute request up to the child's next reachable boundary.

    Return absolute rounds, cumulative blocks and cumulative learning blocks.
    The fork itself is reachable. Remaining PQN warmup ends at original W;
    normal blocks begin at that end or at the actual saved fork. The child's
    end caps either phase, including a child with no new optimizer calls.
    """
    start = continuation.start_rounds
    if total_rounds < start or rollout_length < 1:
        raise ValueError(
            "Continuation total precedes its start or block size is invalid"
        )
    wanted = max(start, requested_rounds)
    anchor = start
    refill_end = (
        reward_refill_end(
            continuation,
            initial_rounds=initial_rounds,
            memory_window=initial_rounds - rollout_length,
        )
        if continuation.method == "pqn_vdn"
        else 0
    )
    if continuation.method == "pqn_vdn" and start < refill_end:
        if wanted <= refill_end:
            end = min(total_rounds, refill_end)
        else:
            anchor, end = refill_end, total_rounds
    else:
        end = total_rounds
    rounds = min(end, anchor + -(-(wanted - anchor) // rollout_length) * rollout_length)
    blocks, learning = continuation_counts(
        rounds,
        continuation=continuation,
        rollout_length=rollout_length,
        initial_rounds=initial_rounds,
        minimum=minimum,
    )
    return rounds, blocks, learning


def continuation_config[T](config: T, continuation: LearnerContinuation | None) -> T:
    """Apply declared rates and loss scalars without changing optimizer layout."""
    if continuation is None:
        return config
    values = {
        name: getattr(continuation, name)
        for name in ("actor_lr", "critic_lr", "q_lr")
        if getattr(continuation, name) is not None
    }
    values.update(dict(continuation.loss_settings))
    return cast(T, replace(cast(Any, config), **values)) if values else config


def resolve_learner_continuation(
    *,
    method: str,
    settings: Mapping[str, Any],
    counters: Mapping[str, int],
    num_envs: int,
    total_env_steps: int,
    changes: Mapping[str, Any],
    parent: Mapping[str, object] | None = None,
    frozen_epsilon: tuple[float, ...] = (),
    frozen_capture_ids: tuple[int, ...] | None = None,
    pinned_epsilon: float | None = None,
    original_total_env_steps: int | None = None,
    reward_reset: bool = False,
    last_refresh_rounds: int | None = None,
) -> dict[str, object]:
    """Freeze a child's learner rules and reject an unapproved PQN floor use.

    settings are the original method-config fields. counters are the verified
    parent's saved cumulative counters. total_env_steps is the child's absolute
    end. original_total_env_steps is required for a first PQN extension and is
    its original declared budget; a descendant inherits the saved horizon.
    changes accepts CONTINUATION_CHANGE_KEYS. This checks method loss scalars
    and seed bounds and resolves future rates. It performs no array or file work.
    reward_reset marks an explicit change to rewards stored by a Q learner;
    last_refresh_rounds must then be its verified history publication boundary.
    The returned JSON-ready record belongs in schedule.continuation['learner'].
    """
    if method not in _PPO_METHODS | {"qmix", "pqn_vdn"}:
        raise ValueError("Unknown continuation method")
    if type(reward_reset) is not bool:
        raise TypeError("reward_reset must be bool")
    if set(changes) - CONTINUATION_CHANGE_KEYS:
        raise ValueError("Unsupported continuation change")
    if "seed" in changes and (
        type(changes["seed"]) is not int or not 0 <= changes["seed"] < 2**32
    ):
        raise ValueError("Branch seed must be an integer in [0, 2**32)")
    family = (
        "ppo" if method in _PPO_METHODS else "pqn" if method == "pqn_vdn" else "qmix"
    )
    if (set(changes) & set(_LOSS_SETTINGS)) - {family}:
        raise ValueError("Continuation settings belong to another learner")
    overrides = changes.get(family, {})
    if not isinstance(overrides, Mapping):
        raise ValueError(f"Unsupported {family} continuation settings")
    overrides = cast(Mapping[str, Any], overrides)
    if set(overrides) - _LOSS_SETTINGS[family]:
        raise ValueError(f"Unsupported {family} continuation settings")
    previous = learner_continuation(parent)
    if previous is not None and previous.method != method:
        raise ValueError("A continuation cannot change its method")
    if num_envs < 1 or counters["env_steps"] % num_envs or total_env_steps % num_envs:
        raise ValueError(
            "Continuation experience must contain whole environment rounds"
        )
    rounds = counters["env_steps"] // num_envs
    total_rounds = total_env_steps // num_envs
    if total_rounds <= rounds:
        raise ValueError("A continuation needs positive added experience")
    _integer(total_rounds, "Continuation total rounds", minimum=1)
    blocks = counters.get("completed_blocks", counters["updates"])
    learning = counters.get("learning_blocks", counters["updates"])
    values = (
        asdict(previous)
        if previous is not None
        else asdict(LearnerContinuation(method, rounds, blocks, learning))
    )
    values.update(
        start_rounds=rounds,
        start_blocks=blocks,
        start_learning_blocks=learning,
        frozen_epsilon=list(frozen_epsilon),
        frozen_capture_ids=list(range(len(frozen_epsilon)))
        if frozen_capture_ids is None
        else list(frozen_capture_ids),
        pinned_epsilon=pinned_epsilon,
    )
    inherited = dict(values["loss_settings"])
    inherited.update(overrides)
    if inherited:
        from marl_battlegrounds.baselines.ppo import PPOConfig
        from marl_battlegrounds.baselines.pqn import PQNConfig
        from marl_battlegrounds.baselines.qmix import QMIXConfig

        # Reuse each method's existing scalar validation, not another range table.
        factory = {"ppo": PPOConfig, "qmix": QMIXConfig, "pqn": PQNConfig}[family]
        factory(**{**settings, **inherited})
    values["loss_settings"] = sorted(inherited.items())
    length = int(settings["rollout_length"])
    initial = int(settings["memory_window"]) + length if method == "pqn_vdn" else 0
    if reward_reset and method in {"qmix", "pqn_vdn"}:
        refreshed = _integer(last_refresh_rounds, "Last history publication")
        if refreshed > rounds:
            raise ValueError("History publication exceeds collected rounds")
        values["reward_reset"] = asdict(
            RewardReset(
                rounds,
                blocks,
                learning,
                refreshed,
                pqn_learning_rounds(
                    rounds,
                    continuation=previous,
                    initial_rounds=initial,
                    memory_window=int(settings["memory_window"]),
                )
                if method == "pqn_vdn"
                else 0,
            )
        )
    if method == "pqn_vdn" and previous is None:
        if original_total_env_steps is None or original_total_env_steps % num_envs:
            raise ValueError("The original PQN budget is required")
        original_rounds = original_total_env_steps // num_envs
        if original_rounds <= initial:
            raise ValueError("The original PQN budget must include learning")
        values["pqn_planned_learning_blocks"] = -(
            -(original_rounds - initial) // length
        )
    rate = changes.get("learning_rate")
    explicit_rate = rate is not None
    if explicit_rate:
        if not isinstance(rate, Mapping):
            raise ValueError("learning_rate must be a declaration")
        rate = dict(cast(Mapping[str, Any], rate))
        kind = rate.get("kind")
        if method == "pqn_vdn":
            values["allow_terminal_rate"] = True
        if method in _PPO_METHODS:
            if set(rate) != {"kind", "actor_lr", "critic_lr"} or kind != "constant":
                raise ValueError(
                    "PPO continuation needs constant actor_lr and critic_lr"
                )
            for name in ("actor_lr", "critic_lr"):
                values[name] = _number(rate[name], name)
        elif method == "qmix":
            if set(rate) != {"kind", "q_lr"} or kind != "constant":
                raise ValueError("QMIX continuation needs constant q_lr")
            values["q_lr"] = _number(rate["q_lr"], "q_lr")
        elif kind == "keep_terminal_rate" and set(rate) == {"kind"}:
            pass
        else:
            expected = {"kind", "q_lr"} | (
                {"optimizer_steps"} if kind == "linear" else set[str]()
            )
            if kind not in {"constant", "linear"} or set(rate) != expected:
                raise ValueError(
                    "PQN continuation needs constant, linear or keep_terminal_rate"
                )
            initial_rate = _number(rate["q_lr"], "q_lr")
            duration = (
                _integer(rate["optimizer_steps"], "optimizer_steps", minimum=1)
                if kind == "linear"
                else 0
            )
            values["pqn_rate"] = asdict(
                LinearSchedule(
                    counters["updates"],
                    initial_rate,
                    _PQN_FLOOR if duration else initial_rate,
                    duration,
                )
            )
    exploration = changes.get("exploration")
    if exploration is not None:
        if method in _PPO_METHODS or not isinstance(exploration, Mapping):
            raise ValueError("Only QMIX and PQN accept exploration declarations")
        exploration = dict(cast(Mapping[str, Any], exploration))
        kind = exploration.get("kind")
        unit = "env_steps" if method == "qmix" else "learning_blocks"
        expected = {"kind", "epsilon"} | (
            {"end_epsilon", unit} if kind == "linear" else set[str]()
        )
        if kind not in {"constant", "linear"} or set(exploration) != expected:
            raise ValueError(
                "Exploration needs a constant or method-specific linear declaration"
            )
        first = _number(exploration["epsilon"], "epsilon", probability=True)
        end = (
            _number(exploration["end_epsilon"], "end_epsilon", probability=True)
            if kind == "linear"
            else first
        )
        duration = (
            _integer(exploration[unit], unit, minimum=1) if kind == "linear" else 0
        )
        origin = counters["env_steps"] if method == "qmix" else learning
        values["exploration"] = asdict(LinearSchedule(origin, first, end, duration))
    values["schema_version"] = 1
    context = learner_continuation(values)
    assert context is not None
    final_blocks, final_learning = continuation_counts(
        total_rounds,
        continuation=context,
        rollout_length=length,
        initial_rounds=initial,
        minimum=int(settings.get("min_buffer_size", 0)),
    )
    _integer(final_blocks, "Final completed blocks")
    per_block = int(settings["epochs"]) * int(settings.get("num_minibatches", 1))
    if method == "pqn_vdn":
        final_updates = counters["updates"] + (final_learning - learning) * per_block
        _integer(final_updates, "Final optimizer steps")
        future = context.pqn_rate
        if future is None:
            reaches_floor = float(settings["q_lr"]) == _PQN_FLOOR or (
                bool(settings["lr_linear_decay"])
                and final_updates
                > (cast(int, context.pqn_planned_learning_blocks) * per_block)
            )
        else:
            reaches_floor = future.end == _PQN_FLOOR and (
                not future.duration or final_updates > future.start + future.duration
            )
        if (
            final_updates > counters["updates"]
            and reaches_floor
            and not context.allow_terminal_rate
        ):
            raise ValueError(
                "Added PQN optimizer calls would use the terminal learning rate 1e-10; "
                "declare a future learning_rate or {'kind': 'keep_terminal_rate'}"
            )
    elif method == "qmix":
        _integer(final_learning * int(settings["epochs"]), "Final optimizer steps")
    else:
        _integer(
            final_blocks * int(settings["epochs"]) * int(settings["minibatches"]),
            "Final optimizer bound",
        )
    return values


@dataclass(frozen=True)
class ContinuationExploration:
    """Apply QMIX's declared future rate at each absolute decision round.

    rule.start is in real transitions and num_envs is the unchanged lane count.
    Subtract the round origin before multiplying, so long cumulative runs do
    not overflow the int32 decision clock. No key or historical slot changes.
    """

    rule: LinearSchedule
    num_envs: int

    def __post_init__(self) -> None:
        """Require a whole-round start and a positive fixed batch on the host."""
        _integer(self.num_envs, "num_envs", minimum=1)
        if self.rule.start % self.num_envs:
            raise ValueError("QMIX exploration must start at a whole environment round")

    def __call__(self, variables: Any, completed_rounds: Any) -> Any:  # noqa: ANN401
        """Replace only current epsilon, keeping every parameter leaf intact."""
        return variables._replace(epsilon=self.rate(completed_rounds))

    def rate(self, completed_rounds: Any) -> Any:  # noqa: ANN401
        """Return the float32 rate with a bounded clock inside the compiled loop."""
        import jax.numpy as jnp

        if not self.rule.duration:
            return jnp.float32(self.rule.value)
        elapsed = jnp.clip(
            completed_rounds - self.rule.start // self.num_envs,
            0,
            -(-self.rule.duration // self.num_envs),
        )
        span = self.rule.duration / self.num_envs
        value = (self.rule.value - self.rule.end) * (
            1.0 - elapsed / span
        ) + self.rule.end
        return jnp.where(elapsed >= span, jnp.float32(self.rule.end), value).astype(
            jnp.float32
        )

    @property
    def identity(self) -> dict[str, object]:
        """Return the exact future rule bound into the collection checkpoint."""
        return {
            "kind": "qmix_continuation_epsilon",
            "version": 1,
            "rule": asdict(self.rule),
            "num_envs": self.num_envs,
        }


def continuation_collection[T](collection: T, context: LearnerContinuation | None) -> T:
    """Attach a future QMIX decision hook without resetting a game or array.

    Other methods and inherited original exploration keep their descriptor.
    This static replacement is also safe when preparing a child restore template.
    """
    if context is None or context.method != "qmix" or context.exploration is None:
        return collection
    games = cast(Any, collection).schedule.num_envs
    return cast(
        T,
        replace(
            cast(Any, collection),
            actor_variables_at_step=ContinuationExploration(context.exploration, games),
        ),
    )


def continuation_state[T](
    state: T,
    context: LearnerContinuation | None,
    *,
    num_envs: int,
    initial_rounds: int = 0,
) -> T:
    """Apply an explicitly declared current-epsilon amendment at a checked fork.

    Pass PQN's original W as initial_rounds to keep its remaining random phase.
    No actor parameter, frozen opponent, memory, counter, optimizer, statistic
    or key changes. Do not call this after restoring a child: its saved epsilon
    is already authoritative and its validator checks the declared rule.
    A changed epsilon keeps the saved scalar's device placement and commitment.
    """
    if context is None or context.exploration is None:
        return state
    import jax.numpy as jnp

    value = cast(Any, state)
    if context.method == "qmix":
        epsilon = ContinuationExploration(context.exploration, num_envs).rate(
            value.carry.progress.rounds
        )
    else:
        epsilon = jnp.where(
            value.carry.progress.rounds < initial_rounds,
            jnp.float32(1.0),
            context.exploration.at_array(value.learning_blocks),
        )
    history = value.carry.history
    previous = history.current_variables.epsilon
    if previous.committed:
        import jax

        epsilon = jax.device_put(epsilon, previous.sharding)
    return cast(
        T,
        value._replace(
            carry=value.carry._replace(
                history=history._replace(
                    current_variables=history.current_variables._replace(
                        epsilon=epsilon
                    )
                )
            )
        ),
    )


def branch_learner_keys(state: Any, seed: int, *, method: str) -> Any:  # noqa: ANN401
    """Branch library-owned randomness once, preserving learner and method memory.

    Fold a stable branch tag and the new uint32 seed into the restored root.
    Derive the method's shuffle/sample root using its existing domain tag, and
    fold the branch into every library-owned memory initialization key. Action,
    reset and opponent draws already derive from the collection root. Opaque
    System memory and remote randomness are untouched. The caller saves the
    resulting root bits with its continuation and must not call this on resume.
    """
    import jax

    from marl_battlegrounds.training.learner import SHUFFLE_ROOT_TAG
    from marl_battlegrounds.training.pqn_learner import PQN_SHUFFLE_ROOT_TAG
    from marl_battlegrounds.training.qmix_learner import QMIX_SAMPLING_ROOT_TAG

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Branch seed must be an integer in [0, 2**32)")

    def fold(key: Any) -> Any:  # noqa: ANN401
        """Keep branch randomness separate from every existing key domain."""
        return jax.random.fold_in(jax.random.fold_in(key, 0x42524E43), seed)

    carry = state.carry
    root = fold(carry.root_key)
    keys = carry.memory.init_key
    initial = jax.vmap(fold)(keys.reshape(-1)).reshape(keys.shape)
    carry = carry._replace(
        root_key=root, memory=carry.memory._replace(init_key=initial)
    )
    tag = (
        QMIX_SAMPLING_ROOT_TAG
        if method == "qmix"
        else (PQN_SHUFFLE_ROOT_TAG if method == "pqn_vdn" else SHUFFLE_ROOT_TAG)
    )
    field = "sampling_root" if method == "qmix" else "shuffle_root"
    return state._replace(carry=carry, **{field: jax.random.fold_in(root, tag)})
