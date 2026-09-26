# Copyright 2023 The JaxMARL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from JaxMARL for MARL-BGs.
# See docs/training/source_reuse.md for source identities and deliberate changes.
"""Provide recurrent PQN-VDN networks, its actor System and one minibatch update.

This module adapts JaxMARL's recurrent PQN-VDN (``pqn_vdn_rnn.py``) at
revision ``976aeb152cb184a5095021968bba94da96eb6394``. It owns the PQN
settings and budget guards, the shared local Q-network with input and hidden
BatchNorm, the epsilon-greedy actor System, the exploration and learning-rate
schedules, network initialization and one optimizer step on one expanded
minibatch of recent game sequences. It owns no collection loop, recent-data
storage, checkpoint, curriculum or training run; ``training.pqn_learner``
joins these pieces to the shared collection owner.

Actors read only their permitted SharedObs input and their own memory. The
team value is the plain sum of the local utilities (VDN): there is no mixer,
critic or physical-state input. BatchNorm statistics move only in training
minibatches; action selection always uses the saved statistics, so one lane's
action never depends on another lane's input. Networks trained before Red
Zone, on historical actor input schema 1, play through a separate cached
schema-1 hook that removes the Red Zone depth column and keeps the old
spawn-side formula. Requires the optional training extra. The network, action
rule, schedules and update are pure JAX that works inside jit, vmap and scan.
Host-only: the setting and batch checks, the float64 host twins of the
schedules, the System factory and the running-variance check used by saving,
exporting and loading. The default settings are the donor's SMAX values
(without its reward multiplier), not settings qualified for learning in
MARL-BGs.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array

try:
    import optax  # pyright: ignore[reportMissingTypeStubs]
    from flax import linen as nn
except ImportError as error:
    raise ImportError(
        "PQN-VDN needs the training extra. Install marl-battlegrounds[training]."
    ) from error

# The QMIX baseline already owns the checks, legal-maximum rule, epsilon
# mixture and team reward that PQN shares; importing them keeps one owner.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    _masked_logits,
    categorical_action_mask,
    decode_actions,
    mirror_action_indices,
)
from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_SIZE,
    ACTOR_INPUT_SCHEMA_VERSION,
    SPAWN_FRAMES,
    encode_actor_inputs,
    schema_1_actor_input,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.baselines.ppo import (
    _actor_input_schema,
    _input_scale,
    _spawn_frame,
)
from marl_battlegrounds.baselines.qmix import (
    QMIX_TIE_RULE,
    TEAM_SLOTS,
    _count,
    _explore,
    _finite,
    _positive,
    _unit_interval,
    greedy_actions,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.input import mirror_team_view

type Tree = Any

PQN_METHOD = "pqn_vdn"
"""Training method name that selects PQN-VDN in configurations and checkpoints."""
PQN_HIDDEN_SIZE = 512
"""Width of both hidden Dense layers and of each actor's recurrent memory."""
PQN_TIE_RULE = QMIX_TIE_RULE
"""Greedy rule: the lowest legal action index among equal best values wins."""
PQN_KEY_SCHEMA_VERSION = 1
"""Version of the learner's model-initialization and shuffle key tags."""
PQN_RECENT_WINDOW_SCHEMA_VERSION = 1
"""Version of the compact retained recent-row layout saved with the learner."""
PQN_NORMALIZATION_SCHEMA_VERSION = 1
"""Version of the BatchNorm statistics layout and its update rule."""
_BATCH_NORM_MOMENTUM = 0.99
_BATCH_NORM_EPSILON = 1e-5
_LEARNING_RATE_END = 1e-10
_INT32_MAX = 2**31 - 1


def _fraction(value: object, name: str) -> float:
    """Check one finite real number in (0, 1], excluding Booleans; return float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in (0, 1].")
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise ValueError(f"{name} must be a finite real number in (0, 1].")
    return number


@dataclass(frozen=True)
class PQNConfig:
    """Hold the static recurrent PQN-VDN settings.

    Parameters
    ----------
    rollout_length : int, default=128
        Real rounds in one normal collection block (T). One round is one
        decision in every game lane. Only the block that reaches the declared
        total budget may be shorter.
    memory_window : int, default=4
        Retained real rows per game lane (H), 1 <= H <= rollout_length. Each
        learning window is these H older rows followed by the block's new rows,
        and the H older rows take part in the loss (the donor's recent-data
        rule). Initial random collection runs H + T rounds before learning.
    epochs : int, default=4
        Passes over each learning window (E). Each pass reshuffles the games.
    num_minibatches : int, default=16
        Equal game groups per pass (M). It must divide num_envs; each optimizer
        step sees num_envs / M whole game sequences with all five teammates.
    q_lr : float, default=0.00025
        Starting RAdam learning rate. Positive finite real, not bool.
    lr_linear_decay : bool, default=True
        Python bool. True decays the rate linearly to 1e-10 over all planned
        optimizer steps of the declared run; False keeps q_lr constant.
    max_grad_norm : float, default=1.0
        Global gradient-norm limit applied before RAdam. Positive finite.
    gamma : float, default=0.99
        Discount per transition, in [0, 1].
    td_lambda : float, default=0.85
        Lambda-return weight, in [0, 1]. Zero gives one-step targets.
    eps_start : float, default=1.0
        Exploration rate used for the first learning block, in [0, 1]. Initial
        random collection always uses 1.0, whatever this value is.
    eps_finish : float, default=0.01
        Final exploration rate, 0 <= eps_finish <= eps_start.
    eps_decay_fraction : float, default=0.1
        Share of the planned learning blocks over which exploration falls
        linearly from eps_start to eps_finish, in (0, 1]. A fractional block
        span is kept, as in the donor.
    input_scale : float, default=1.0
        Positive finite multiplier applied to every actor feature before the
        input BatchNorm. That BatchNorm divides each feature by its own
        standard deviation, so the scale almost cancels: it changes only how
        large BatchNorm's epsilon (0.00001) is next to the feature variance,
        and the untrained network's first outputs. It is kept for identity
        and compatibility with the other methods; it is not a useful knob to
        tune.
    spawn_frame : str, default="left"
        Network frame for actor features, masks and actions: "left" reflects
        every game to look like a start from the left bank, "world" keeps raw
        coordinates. Stored recent rows stay in the world frame.

    Raises
    ------
    ValueError
        A count is not a plain integer in range, a float is nonfinite, Boolean
        or outside its range, lr_linear_decay is not a bool, the spawn frame is
        unknown, memory_window exceeds rollout_length, eps_finish exceeds
        eps_start, or H + T exceeds int32.

    Notes
    -----
    The architecture (input BatchNorm; two Dense 512, BatchNorm, ReLU layers;
    GRU 512; Dense 198), the BatchNorm constants (momentum 0.99, epsilon 1e-5),
    RAdam's own defaults and the 1e-10 schedule end are fixed contracts, not
    settings. There is no reward multiplier, target network, replay or value
    normalization. The record is immutable host setup; capture one instance in
    compiled code.
    """

    rollout_length: int = 128
    memory_window: int = 4
    epochs: int = 4
    num_minibatches: int = 16
    q_lr: float = 0.00025
    lr_linear_decay: bool = True
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    td_lambda: float = 0.85
    eps_start: float = 1.0
    eps_finish: float = 0.01
    eps_decay_fraction: float = 0.1
    input_scale: float = 1.0
    spawn_frame: str = "left"

    def __post_init__(self) -> None:
        """Reject invalid settings on the host before any array is created."""
        for name in ("rollout_length", "memory_window", "epochs", "num_minibatches"):
            _count(getattr(self, name), name)
        if self.memory_window > self.rollout_length:
            raise ValueError("PQN needs memory_window <= rollout_length")
        if self.memory_window + self.rollout_length > _INT32_MAX:
            raise ValueError("PQN's initial rounds would overflow int32")
        _positive(self.q_lr, "q_lr")
        _positive(self.max_grad_norm, "max_grad_norm")
        if type(self.lr_linear_decay) is not bool:
            raise ValueError("lr_linear_decay must be a Python bool")
        for name in ("gamma", "td_lambda", "eps_start", "eps_finish"):
            _unit_interval(getattr(self, name), name)
        if self.eps_finish > self.eps_start:
            raise ValueError("PQN needs eps_finish <= eps_start")
        _fraction(self.eps_decay_fraction, "eps_decay_fraction")
        _input_scale(self.input_scale)
        _spawn_frame(self.spawn_frame)

    @property
    def initial_rounds(self) -> int:
        """Rounds of initial random collection before learning: W = H + T."""
        return self.memory_window + self.rollout_length


DEFAULT_PQN_CONFIG = PQNConfig()
"""The donor's SMAX PQN-VDN settings without its reward multiplier, left frame."""


def validate_pqn_batch_size(num_envs: int, pqn: PQNConfig) -> None:
    """Check that num_envs fits PQN's minibatches and int32 block counters.

    Parameters
    ----------
    num_envs : int
        Plain positive number of game lanes (B).
    pqn : PQNConfig
        Settings whose rollout_length, memory_window, epochs and
        num_minibatches bound one block.

    Raises
    ------
    ValueError
        num_envs is not a plain positive int32 value; num_minibatches does not
        divide it; ``B * T * 5`` exceeds int32 (the shared update summary
        reduces one block's actor rows into int32); or
        ``E * B * (H + T - 1) * 5`` exceeds int32 (one learning block's used
        agent utilities, the largest device reduction).

    Notes
    -----
    Host-only; it creates no arrays. ``TrainConfig`` and ``init_pqn_learner``
    both call it before any file or array work. Whole-run totals are Python
    integers and are not limited by this check.
    """
    _count(num_envs, "num_envs")
    if num_envs % pqn.num_minibatches:
        raise ValueError("PQN needs num_minibatches to divide num_envs")
    if num_envs * pqn.rollout_length * TEAM_SLOTS > _INT32_MAX:
        raise ValueError("One PQN block's actor counts would overflow int32")
    pairs = pqn.memory_window + pqn.rollout_length - 1
    if pqn.epochs * num_envs * pairs * TEAM_SLOTS > _INT32_MAX:
        raise ValueError("One PQN block's used utilities would overflow int32")


def pqn_planned_learning_blocks(total_rounds: int, pqn: PQNConfig) -> int:
    """Return the number of learning blocks a declared run will perform.

    Parameters
    ----------
    total_rounds : int
        Plain positive declared budget in rounds (R = total_env_steps /
        num_envs), at most int32.
    pqn : PQNConfig
        Settings giving T, H, E and M.

    Returns
    -------
    int
        N = ceil((R - W) / T), where W = H + T initial random rounds. Each
        learning block performs E * M optimizer steps, so the learning-rate
        schedule spans N * E * M steps.

    Raises
    ------
    ValueError
        R is not a plain int in [1, 2**31 - 1]; R < W + 1 (there would be no
        learning and no positive schedule horizon); N * E * M or
        ceil(W / T) + N exceeds int32 (optimizer and block counters).

    Notes
    -----
    Host-only and allocation-free; the runner's config check, learner setup,
    validation and restore all call it. A saved boundary may still lie at or
    before W inside a longer valid run.
    """
    rounds = _count(total_rounds, "total_rounds")
    initial = pqn.initial_rounds
    if rounds < initial + 1:
        raise ValueError(
            "A PQN run needs total rounds of at least memory_window + "
            f"rollout_length + 1 ({initial + 1}); got {rounds}"
        )
    planned = -(-(rounds - initial) // pqn.rollout_length)
    if planned * pqn.epochs * pqn.num_minibatches > _INT32_MAX:
        raise ValueError("PQN's planned optimizer steps would overflow int32")
    if -(-initial // pqn.rollout_length) + planned > _INT32_MAX:
        raise ValueError("PQN's block counter would overflow int32")
    return planned


def epsilon_at(
    learning_blocks: Array, planned_learning_blocks: int, config: PQNConfig
) -> Array:
    """Return the exploration rate for the collection after some learned blocks.

    Parameters
    ----------
    learning_blocks : Array
        Int32 0-d count of learning blocks already completed (k). The first
        learning block collects with ``epsilon_at(0)``.
    planned_learning_blocks : int
        Plain positive N from :func:`pqn_planned_learning_blocks`.
    config : PQNConfig
        Settings giving eps_start, eps_finish and eps_decay_fraction.

    Returns
    -------
    Array
        Float32 0-d rate: Optax's ``linear_schedule(eps_start, eps_finish,
        eps_decay_fraction * N)`` at k, the donor's block clock. It equals
        eps_start up to float32 rounding at k = 0 and eps_finish exactly once k
        reaches the decay span (set directly, because compiled division by the
        span can round a few float32 steps below eps_finish).

    Raises
    ------
    ValueError
        planned_learning_blocks is not a plain positive int32 value.

    Notes
    -----
    Pure JAX; works under jit. Initial random collection does not use this
    clock; it always explores with rate 1. Compare saved values with
    :func:`pqn_epsilon_reference`, the float64 host twin, within 2e-6.
    """
    planned = _count(planned_learning_blocks, "planned_learning_blocks")
    # The donor passes a fractional span (EPS_DECAY * NUM_UPDATES); Optax's
    # linear schedule divides by it, so a float span keeps the donor's clock.
    span = cast(int, config.eps_decay_fraction * planned)
    schedule = optax.linear_schedule(config.eps_start, config.eps_finish, span)
    clock = jnp.asarray(learning_blocks, jnp.int32)
    # Compiled code may divide by the span through its reciprocal, which can
    # land a few float32 steps off eps_finish; the plateau is set exactly.
    return jnp.where(
        clock >= span,
        jnp.float32(config.eps_finish),
        jnp.asarray(schedule(clock), jnp.float32),
    )


def pqn_epsilon_reference(
    learning_blocks: int, planned_learning_blocks: int, config: PQNConfig
) -> float:
    """Return :func:`epsilon_at` as a float64 host value for logs and checks.

    Parameters
    ----------
    learning_blocks : int
        Nonnegative completed learning blocks (k); any Python integer.
    planned_learning_blocks : int
        Positive N.
    config : PQNConfig
        Settings giving eps_start, eps_finish and eps_decay_fraction.

    Returns
    -------
    float
        ``(eps_start - eps_finish) * (1 - c / span) + eps_finish`` with
        ``span = eps_decay_fraction * N`` and ``c = min(max(k, 0), span)``;
        exactly eps_finish once k reaches the span.

    Notes
    -----
    Pure Python with no checks and no device work, so writing a log row adds
    no device synchronization.
    """
    span = config.eps_decay_fraction * planned_learning_blocks
    clock = min(max(learning_blocks, 0), span)
    return (config.eps_start - config.eps_finish) * (
        1.0 - clock / span
    ) + config.eps_finish


def pqn_learning_rate(
    optimizer_count: int, planned_learning_blocks: int, config: PQNConfig
) -> float:
    """Return the scheduled RAdam rate at one optimizer count, on the host.

    Parameters
    ----------
    optimizer_count : int
        Nonnegative optimizer count before the step (0 for the first step).
    planned_learning_blocks : int
        Positive N; the schedule spans N * epochs * num_minibatches steps.
    config : PQNConfig
        Settings giving q_lr, lr_linear_decay, epochs and num_minibatches.

    Returns
    -------
    float
        q_lr when lr_linear_decay is False. Otherwise
        ``(q_lr - 1e-10) * (1 - c / steps) + 1e-10`` with
        ``c = min(max(count, 0), steps)``: q_lr at count 0 and 1e-10 at the
        count after the last planned step.

    Notes
    -----
    A float64 host copy of the Optax schedule used inside the optimizer; log
    rows use it without device work. Tests keep both equal within float32
    rounding.
    """
    if not config.lr_linear_decay:
        return float(config.q_lr)
    steps = planned_learning_blocks * config.epochs * config.num_minibatches
    clock = min(max(optimizer_count, 0), steps)
    return (config.q_lr - _LEARNING_RATE_END) * (
        1.0 - clock / steps
    ) + _LEARNING_RATE_END


class PQNInferenceVariables(NamedTuple):
    """Hold everything a PQN actor needs to compute action values.

    Attributes
    ----------
    params : PyTree
        Raw Flax ``params`` subtree of the Q-network, including the BatchNorm
        scale and bias leaves. These receive gradients.
    batch_stats : PyTree
        Raw Flax ``batch_stats`` subtree: running mean and variance of the
        three BatchNorm layers (input 5165, two hidden 512). They move only in
        training minibatches, never during action selection.

    Notes
    -----
    The field order is part of the saved identity: parameters first, then
    statistics. Changing statistics with the same parameters changes what the
    actor does and its identity. Checkpoints store this record as Flax's own
    ``{"params", "batch_stats"}`` dictionary, built and read by name.
    """

    params: Tree
    batch_stats: Tree


class PQNActorVariables(NamedTuple):
    """Carry everything that changes in a PQN actor between collection blocks.

    Attributes
    ----------
    network : PQNInferenceVariables
        Q-network parameters and BatchNorm statistics.
    epsilon : Array
        Float32 0-d exploration probability in [0, 1]. Zero means greedy.
        During training it is 1 in initial random collection and then the
        block clock's value; loaded and evaluated Systems use zero.

    Notes
    -----
    This immutable PyTree is the actor's whole variable tree, stored as the
    current actor and in each of the 20 historical opponent slots, so a
    captured opponent freezes its parameters, statistics and epsilon together.
    The class name and field order enter tree digests.
    """

    network: PQNInferenceVariables
    epsilon: Array


class PQNLearningOutputs(NamedTuple):
    """Report what one PQN decision must store for later learning.

    Attributes
    ----------
    pre_memory : Array
        Float32 (B,5,512) memory each actor held immediately before this
        decision, after M8 reset finished lanes. The learner stores it so a
        later unroll can start from the exact action-time memory without
        running the actor again.
    """

    pre_memory: Array


class PQNTrainState(NamedTuple):
    """Hold one temporary numerical view for initialization and updates.

    Attributes
    ----------
    network : PQNInferenceVariables
        Online Q-network parameters and BatchNorm statistics.
    opt_state : PyTree
        ``optax.chain(clip_by_global_norm, radam)`` state over ``network.params``
        only. With learning-rate decay it holds two integer counts (RAdam and
        schedule); without decay, one.
    optimizer_steps : Array
        Int32 0-d count of optimizer steps applied before this state.

    Notes
    -----
    There is one online network and no target network. The durable learner
    keeps the network in its actor history, not in a second long-lived copy;
    this record exists while one update is computed. Arrays stay dynamic.
    """

    network: PQNInferenceVariables
    opt_state: Tree
    optimizer_steps: Array


class PQNBatch(NamedTuple):
    """Hold one expanded minibatch: C time rows of D whole game sequences.

    Attributes
    ----------
    actor_features : Array
        Float32 (C,D,5,F) encoded permitted actor inputs in the network frame.
        F is ACTOR_FEATURE_SIZE (5165) for real inputs.
    action_mask : Array
        Bool (C,D,5,198) categorical legality in the network frame.
    actions : Array
        Int32 (C,D,5) recorded actions mapped into the network frame.
    rewards : Array
        Float32 (C,D) team training reward of each row: the mean task reward
        over configured Team A slots plus the team shaping reward.
    episode_start : Array
        Bool (C,D). True on an episode's first decision; memory resets first.
    ended : Array
        Bool (C,D). True when the row's transition ended its episode (win,
        loss or horizon draw); the next row then gives no bootstrap. An
        ordinary collection cutoff is False.
    valid : Array
        Bool (C,D). A chronological real prefix of L rows followed by padding.
        Padding is neutralized, keeps memory and never enters a TD pair,
        target or statistic.
    active : Array
        Bool (C,D,5). Configured Team A slots; temporarily dead actors stay
        active. Inactive slots add nothing to utilities, targets or statistics.
    initial_memory : Array
        Float32 (D,5,512) stored action-time memory of the first row.

    Notes
    -----
    Row t and t+1 form a TD pair only when both are valid, so the last real
    row supplies bootstrap values and statistics but is never a left row.
    """

    actor_features: Array
    action_mask: Array
    actions: Array
    rewards: Array
    episode_start: Array
    ended: Array
    valid: Array
    active: Array
    initial_memory: Array


class PQNMetrics(NamedTuple):
    """Report one minibatch step's small scalar results.

    Attributes
    ----------
    loss : Array
        Float32 mean squared team TD error over eligible pairs; 0 when none.
    mean_q : Array
        Float32 mean VDN value of the recorded actions over those pairs.
    mean_target : Array
        Float32 mean lambda-return target over those pairs.
    grad_norm : Array
        Float32 global gradient norm before clipping; 0 when no step ran.
    used_td_pairs : Array
        Int32 count of eligible adjacent pairs (valid left and right rows).
    used_agent_utilities : Array
        Int32 count of configured actors on the left rows of those pairs.
    finite : Array
        Bool. True when the loss, means, gradients and the whole candidate
        state (parameters, statistics and every optimizer leaf) are finite.
    performed : Array
        Bool. True when an optimizer step ran (at least one eligible pair).

    Notes
    -----
    Absent steps use finite zeros; host logs show their losses as null.
    """

    loss: Array
    mean_q: Array
    mean_target: Array
    grad_norm: Array
    used_td_pairs: Array
    used_agent_utilities: Array
    finite: Array
    performed: Array


class PQNScannedRNN(nn.Module):
    """Carry each actor's 512-wide memory through time with the donor's GRU.

    Parameters are broadcast across time. Each input row is (features, start,
    valid, active), shaped (N,512), (N,), (N,) and (N,) for N flattened actors.
    Memory resets to zero before the cell on a valid episode start; invalid
    rows keep their memory unchanged; on valid rows inactive slots get zero
    memory (and start from zero, so they stay zero). The parent names
    this module ``ScannedRNN_0`` and its cell ``GRUCell_0``, the donor's paths.
    """

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, carry: Array, inputs: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        """Apply one GRU step with resets, preserved padding and zero inactive memory.

        Parameters
        ----------
        carry : Array
            Float32 (N,512) memory immediately before this decision.
        inputs : tuple[Array, Array, Array, Array]
            Features float32 (N,512), episode-start flags bool (N,), valid
            flags bool (N,) and active flags bool (N,) for one time row.

        Returns
        -------
        tuple[Array, Array]
            Next memory and cell output, both float32 (N,512). Outputs of
            invalid or inactive rows are meaningless and must be ignored.
        """
        features, starts, valid, active = inputs
        before = jnp.where((starts & valid)[:, None], 0.0, carry)
        updated, output = cast(
            tuple[Array, Array],
            nn.GRUCell(features=PQN_HIDDEN_SIZE, name="GRUCell_0")(before, features),
        )
        moved = jnp.where(active[:, None], updated, 0.0)
        return jnp.where(valid[:, None], moved, carry), output


class PQNNetwork(nn.Module):
    """Map permitted actor features and memory to 198 action values per actor.

    The layers are the donor's SMAX configuration: input BatchNorm; Dense 512,
    BatchNorm, ReLU; Dense 512, BatchNorm, ReLU; a 512-wide GRU; Dense 198.
    Flax's default initializers are kept (LeCun-normal Dense and GRU input
    kernels, orthogonal recurrent kernels, zero biases, BatchNorm scale 1 and
    bias 0, running mean 0 and variance 1). Parameters and statistics are
    shared by all actors and teammates; memory stays separate per actor.

    input_scale is a fixed positive finite multiplier applied to every feature
    before the input BatchNorm; it changes no parameter shape.
    """

    input_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        memory: Array,
        features: Array,
        episode_start: Array,
        valid: Array,
        active: Array,
        *,
        train: bool = False,
    ) -> tuple[Array, Array]:
        """Read time-major feature rows with independent actor memory.

        Parameters
        ----------
        memory : Array
            Float32 (games,5,512) memory immediately before the first row.
        features : Array
            Float32 (rows,games,5,F) encoded permitted inputs. F must match the
            initialized width, normally ACTOR_FEATURE_SIZE.
        episode_start : Array
            Bool (rows,games), True when a new episode starts at that row.
        valid : Array
            Bool (rows,games) real decisions. Invalid rows keep prior memory
            and are excluded from normalization statistics.
        active : Array
            Bool (rows,games,5) configured slots. Inactive slots get zero
            values and zero memory and are excluded from statistics.
        train : bool, default=False
            Python bool. False uses the saved running statistics and changes
            nothing (action selection and initialization). True normalizes
            each feature over every valid active row of the call, time and
            flattened game/actor rows together, and must run with
            ``mutable=["batch_stats"]`` and at least one such row.

        Returns
        -------
        tuple[Array, Array]
            Final float32 (games,5,512) memory and float32 (rows,games,5,198)
            action values. Inactive slots' values are 0; values on invalid rows
            are meaningless and must be ignored.

        Raises
        ------
        ValueError
            input_scale is not a finite positive real number.

        Notes
        -----
        Excluded rows' features are replaced by zero before any arithmetic, so
        a nonfinite value in padding reaches no Dense layer, moment or
        gradient. In training mode a call with no valid active row divides by
        zero inside Flax's masked mean and gives NaN statistics. The PQN
        updater skips a minibatch with no eligible TD pair; a minibatch that
        has pairs but no active slot would give NaN statistics, which the
        block's finite check rejects, so such statistics are never
        published.
        """
        rows, games = valid.shape
        actors = games * TEAM_SLOTS
        scale = _input_scale(self.input_scale)
        if scale != 1.0:
            features = features * scale
        row = valid[..., None] & active
        x = jnp.where(row[..., None], features, 0.0).reshape(rows, actors, -1)
        mask = row.reshape(rows, actors, 1) if train else None

        def normalize(value: Array) -> Array:
            """Apply one donor BatchNorm with the call's row mask."""
            return nn.BatchNorm(
                use_running_average=not train,
                momentum=_BATCH_NORM_MOMENTUM,
                epsilon=_BATCH_NORM_EPSILON,
            )(value, mask=mask)

        x = normalize(x)
        for _ in range(2):
            x = nn.relu(normalize(nn.Dense(PQN_HIDDEN_SIZE)(x)))

        def per_actor(flags: Array) -> Array:
            """Broadcast (rows,games) flags to (rows,games*5) actor rows."""
            return jnp.broadcast_to(
                flags[..., None], (rows, games, TEAM_SLOTS)
            ).reshape(rows, actors)

        carry, x = PQNScannedRNN(name="ScannedRNN_0")(
            memory.reshape(actors, PQN_HIDDEN_SIZE),
            (
                x,
                per_actor(episode_start),
                per_actor(valid),
                active.reshape(rows, actors),
            ),
        )
        values = nn.Dense(NUM_ACTIONS)(x).reshape(rows, games, TEAM_SLOTS, NUM_ACTIONS)
        values = jnp.where(active[..., None], values, 0.0)
        return carry.reshape(games, TEAM_SLOTS, PQN_HIDDEN_SIZE), values


def _flax_variables(network: PQNInferenceVariables) -> dict[str, Tree]:
    """Return Flax's ``{"params", "batch_stats"}`` dictionary for apply."""
    return {"params": network.params, "batch_stats": network.batch_stats}


def _nonnegative_variances(  # pyright: ignore[reportUnusedFunction] - Shared with the learner and checkpoints.
    batch_stats: Tree,
) -> bool:
    """Say whether every BatchNorm running variance in a statistics tree is >= 0.

    Parameters
    ----------
    batch_stats : PyTree
        A network's BatchNorm statistics, such as
        ``PQNInferenceVariables.batch_stats``, or a stack of them with extra
        leading axes (historical slots). Leaves whose last path key is
        ``"var"`` are the running variances; ``"mean"`` leaves are ignored.

    Returns
    -------
    bool
        True when no running variance is below zero. A NaN is not below zero,
        so callers check finiteness separately. An empty tree gives True.

    Notes
    -----
    Host-only: it reads the values with ``bool``, so it syncs with the device
    and must not be called inside jit. Saving, restoring, exporting and
    loading use this one rule, because a negative variance would make the
    frozen normalization produce NaN.
    """
    leaves = cast(
        list[tuple[tuple[object, ...], Array]],
        jax.tree_util.tree_flatten_with_path(batch_stats)[0],
    )
    return all(
        not bool(jnp.any(leaf < 0))
        for path, leaf in leaves
        if getattr(path[-1], "key", None) == "var"
    )


def _check_typed_key(key: Array) -> None:
    """Require one typed scalar Threefry key.

    key is the initialization key. Raises ValueError when it is not a typed
    PRNG key of shape (), or when its implementation is not Threefry (legacy
    uint32 keys are refused). Host-only; reads metadata, no device work.
    """
    if not jnp.issubdtype(key.dtype, jax.dtypes.prng_key) or key.shape != ():
        raise ValueError("PQN initialization requires one typed PRNG key")
    if str(jax.random.key_impl(key)) != "threefry2x32":
        raise ValueError("PQN initialization requires a Threefry key")


def _network_variables(
    key: Array, input_scale: float, *, features: int = ACTOR_FEATURE_SIZE
) -> PQNInferenceVariables:
    """Initialize the Q-network from one key with train=False.

    key is a typed Threefry key and input_scale the network's finite positive
    feature multiplier (it changes no parameter shape or value). features is
    the input width, ACTOR_FEATURE_SIZE (5,165) by default; the donor
    reference test passes the donor's 5,164 so its recorded initialization
    stays comparable. Zero inputs shaped (rows=1, games=1, 5, features) fix
    parameter shapes; values depend only on the key, the width and the module
    paths. Returns the parameters and the initial statistics (mean 0,
    variance 1).
    """
    variables = PQNNetwork(input_scale=input_scale).init(
        key,
        jnp.zeros((1, TEAM_SLOTS, PQN_HIDDEN_SIZE), jnp.float32),
        jnp.zeros((1, 1, TEAM_SLOTS, features), jnp.float32),
        jnp.zeros((1, 1), jnp.bool_),
        jnp.ones((1, 1), jnp.bool_),
        jnp.ones((1, 1, TEAM_SLOTS), jnp.bool_),
        train=False,
    )
    return PQNInferenceVariables(variables["params"], variables["batch_stats"])


def pqn_optimizer(
    pqn: PQNConfig,
    planned_learning_blocks: int,
    *,
    learning_rate: Callable[[Array], Array] | None = None,
    optimizer_count: Array | None = None,
) -> Any:  # noqa: ANN401
    """Return the donor's optimizer for one declared run.

    Parameters
    ----------
    pqn : PQNConfig
        Settings giving q_lr, lr_linear_decay, max_grad_norm, E and M.
    planned_learning_blocks : int
        Plain positive N; the rate schedule spans N * E * M optimizer steps.

    learning_rate : callable or None, default=None
        Explicit future float32 rate at the absolute optimizer count. None
        keeps the original rule. The original lr_linear_decay flag still
        decides the optimizer tree, so saved counts and moments stay valid.
    optimizer_count : Array or None, default=None
        Int32 scalar before the next optimizer call. Required for an override
        when the original optimizer used a constant rate; otherwise unused.

    Returns
    -------
    optax.GradientTransformation
        ``optax.chain(optax.clip_by_global_norm(max_grad_norm),
        optax.radam(rate))`` with RAdam's own defaults. ``rate`` is
        ``optax.linear_schedule(q_lr, 1e-10, N * E * M)`` with decay, else
        q_lr. Stateless; initialize it on network parameters only.

    Raises
    ------
    ValueError
        planned_learning_blocks is not a plain positive int32 value, or the
        step count overflows int32.

    Notes
    -----
    Host-only construction; no arrays are created. The donor counts its
    learning blocks as floor(total / (T * B)) and excludes its initial random
    collection; BG uses N from :func:`pqn_planned_learning_blocks` so the
    schedule ends exactly at the declared budget.
    """
    planned = _count(planned_learning_blocks, "planned_learning_blocks")
    steps = planned * pqn.epochs * pqn.num_minibatches
    if steps > _INT32_MAX:
        raise ValueError("PQN's planned optimizer steps would overflow int32")
    rate: Any = (
        optax.linear_schedule(pqn.q_lr, _LEARNING_RATE_END, steps)
        if pqn.lr_linear_decay
        else pqn.q_lr
    )
    if learning_rate is not None:
        if pqn.lr_linear_decay:
            rate = learning_rate
        else:
            if optimizer_count is None:
                raise ValueError(
                    "A future constant-tree rate needs the optimizer count"
                )
            rate = learning_rate(optimizer_count)
    return optax.chain(
        optax.clip_by_global_norm(pqn.max_grad_norm), optax.radam(learning_rate=rate)
    )


def initialize_pqn(
    key: Array,
    *,
    pqn: PQNConfig = DEFAULT_PQN_CONFIG,
    planned_learning_blocks: int,
) -> PQNTrainState:
    """Create an untrained PQN network, its statistics and one RAdam state.

    Parameters
    ----------
    key : Array
        One typed Threefry key (``jax.random.key(seed)``). The learner derives
        it with ``fold_in(root, PQN_MODEL_INITIALIZATION_TAG)``.
    pqn : PQNConfig, default=DEFAULT_PQN_CONFIG
        Settings; input_scale, q_lr, lr_linear_decay, max_grad_norm, epochs
        and num_minibatches matter here.
    planned_learning_blocks : int
        Plain positive N for the learning-rate schedule.

    Returns
    -------
    PQNTrainState
        Float32 parameters, running mean 0 and variance 1, the optimizer state
        over the parameters only and ``optimizer_steps`` int32 0.

    Raises
    ------
    ValueError
        key is not one typed Threefry key, or planned_learning_blocks is invalid.

    Notes
    -----
    Allocates 4,596,512 float32 parameters (18,386,048 bytes), 12,378 float32
    statistics (49,512 bytes) and two moment trees (36,772,096 bytes) plus
    counters. Works under ``jax.eval_shape``.
    """
    _check_typed_key(key)
    network = _network_variables(key, pqn.input_scale)
    optimizer = pqn_optimizer(pqn, planned_learning_blocks)
    return PQNTrainState(network, optimizer.init(network.params), jnp.int32(0))


def pqn_actor_template() -> PQNInferenceVariables:
    """Return the Q-network's parameter and statistic shapes without allocating.

    Returns
    -------
    PQNInferenceVariables
        ``jax.ShapeDtypeStruct`` leaves of the parameters and statistics. No
        optimizer, window or history is traced. Use it to restore actor-only
        payloads; input_scale does not change any shape.
    """
    return cast(
        PQNInferenceVariables,
        jax.eval_shape(
            functools.partial(_network_variables, input_scale=1.0), jax.random.key(0)
        ),
    )


def _initial_pqn_memory(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    """Create float32 zero memory shaped (B,5,512) from inputs.active_mask.

    variables and keys satisfy the M8 initializer signature and are unused.
    M8 selects the lanes whose episode needs fresh memory.
    """
    del variables, keys
    return jnp.zeros((*inputs.active_mask.shape, PQN_HIDDEN_SIZE), jnp.float32)


def _apply_pqn_actor(
    variables: PQNActorVariables,
    memory: Array,
    inputs: SystemInput,
    keys: Array,
    *,
    input_scale: float = 1.0,
    spawn_frame_index: int = 0,
) -> SystemOutput:
    """Choose one epsilon-greedy action per actor from permitted inputs.

    Parameters
    ----------
    variables : PQNActorVariables
        Network parameters, saved statistics and the float32 0-d rate.
    memory : Array
        Float32 (B,5,512) actor memory before this decision (after M8 reset
        finished lanes).
    inputs : SystemInput
        Same-epoch permitted views, native masks and lifecycle flags. Each
        actor reads only its own view and memory.
    keys : Array
        B independent Threefry lane keys, typed (B,) or legacy uint32 (B,2).
        Five actor keys are split from each lane key.
    input_scale : float, default=1.0
        Fixed feature multiplier bound by the System factory.
    spawn_frame_index : int, default=0
        Index into SPAWN_FRAMES (0 world, 1 left), bound by the factory. A
        number, because M8 records numerical keyword defaults in identity.

    Returns
    -------
    SystemOutput
        Native int32 (B,5) world-frame actions, next memory, and learning
        outputs ``PQNLearningOutputs(memory)``: the memory received here.
        Invalid lanes keep memory; ignore their actions.

    Notes
    -----
    The network runs in inference mode over one time row with the saved
    statistics, so each lane depends only on its own input and memory. In
    the left frame, flagged lanes' views and masks are reflected before the
    network, ties break in that frame (first legal maximum after -inf
    masking), and the chosen index is mapped back to world directions. Dead
    and inactive actors have only the neutral action legal. Pure JAX.
    """
    actors, frame_mask = inputs.actors, inputs.action_mask
    flag: Array | None = None
    spawn_frame = SPAWN_FRAMES[spawn_frame_index]
    if spawn_frame != "world":
        flag = spawn_frame_flag(actors, spawn_frame)
        actors, frame_mask = mirror_team_view(
            actors,
            frame_mask,
            flag,
            obstacle_partners=team_obstacle_partners(actors),
        )
    features = encode_actor_inputs(actors)
    next_memory, values = cast(
        tuple[Array, Array],
        PQNNetwork(input_scale=input_scale).apply(
            _flax_variables(variables.network),
            memory,
            features[None],
            inputs.episode_start[None],
            inputs.valid[None],
            inputs.active_mask[None],
            train=False,
        ),
    )
    mask = categorical_action_mask(frame_mask)
    actor_keys = jax.vmap(functools.partial(jax.random.split, num=TEAM_SLOTS))(keys)
    indices = _explore(
        greedy_actions(values[0], mask), mask, actor_keys, variables.epsilon
    )
    if flag is not None:
        indices = mirror_action_indices(indices, flag)
    return SystemOutput(
        decode_actions(indices),
        next_memory,
        learning_outputs=PQNLearningOutputs(memory),
    )


@functools.lru_cache(maxsize=16)
def _pqn_actor_apply(
    scale: float, spawn_frame: str
) -> Callable[[PQNActorVariables, Array, SystemInput, Array], SystemOutput]:
    """Return a cached apply hook whose keyword defaults record scale and frame.

    scale and spawn_frame are already checked host values. The hook captures no
    weights or live inputs; M8 digests its numerical defaults into System
    identity and uses the hook as a compile key, so equal settings reuse one
    hook and one compiled program.
    """
    frame_index = SPAWN_FRAMES.index(spawn_frame)

    def apply(
        variables: PQNActorVariables,
        memory: Array,
        inputs: SystemInput,
        keys: Array,
        *,
        input_scale: float = scale,
        spawn_frame_index: int = frame_index,
    ) -> SystemOutput:
        """Apply the PQN actor with the factory-bound settings.

        Arguments, outputs and effects follow _apply_pqn_actor. Normal System
        execution supplies only the first four arguments.
        """
        return _apply_pqn_actor(
            variables,
            memory,
            inputs,
            keys,
            input_scale=input_scale,
            spawn_frame_index=spawn_frame_index,
        )

    return apply


@functools.lru_cache(maxsize=16)
def _schema_1_pqn_actor_apply(
    scale: float, spawn_frame: str
) -> Callable[[PQNActorVariables, Array, SystemInput, Array], SystemOutput]:
    """Cache a PQN-VDN hook for networks trained on actor input schema 1.

    Parameters
    ----------
    scale : float
        Already checked positive finite input scale.
    spawn_frame : str
        Already checked frame name, "world" or "left".

    Returns
    -------
    Callable
        An apply hook that turns each call's actor views into the historical
        19-column view with schema_1_actor_input, then runs the unchanged
        _apply_pqn_actor. The encoder uses the 5,164-feature schema-1 layout,
        and the "left" frame uses the old spawn-side formula for both the
        reflected view and the returned actions, so the actor plays exactly
        as before Red Zone. Keyword defaults record scale and frame index as
        in _pqn_actor_apply; the hook's own name and code give it a distinct
        registration ID.

    Notes
    -----
    Host-only construction; the hook captures no weights or live inputs.
    Caching keeps one callable, and so one compiled program, per setting.
    """
    frame_index = SPAWN_FRAMES.index(spawn_frame)

    def apply(
        variables: PQNActorVariables,
        memory: Array,
        inputs: SystemInput,
        keys: Array,
        *,
        input_scale: float = scale,
        spawn_frame_index: int = frame_index,
    ) -> SystemOutput:
        """Apply a schema-1 PQN-VDN actor to its historical 19-column view.

        Arguments, outputs and effects follow _apply_pqn_actor; only the
        context depth column is removed first. Normal System execution
        supplies only the first four arguments.
        """
        return _apply_pqn_actor(
            variables,
            memory,
            inputs._replace(actors=schema_1_actor_input(inputs.actors)),
            keys,
            input_scale=input_scale,
            spawn_frame_index=spawn_frame_index,
        )

    return apply


def make_pqn_system(
    network: PQNInferenceVariables,
    *,
    epsilon: float = 0.0,
    input_scale: float = 1.0,
    spawn_frame: str = "left",
    name: str = "PQN-VDN",
    checkpoint: str | None = None,
    actor_input_schema: int = ACTOR_INPUT_SCHEMA_VERSION,
) -> System:
    """Wrap PQN network variables as an M8 JAX System with epsilon-greedy actions.

    Parameters
    ----------
    network : PQNInferenceVariables
        Q-network parameters and BatchNorm statistics from initialize_pqn, a
        learner update or a loaded export. The factory does not inspect shapes;
        the network checks them when called.
    epsilon : float, default=0.0
        Exploration probability in [0, 1]. Zero gives greedy play (first legal
        maximum), which loaded and evaluated Systems use. Training stores 1 in
        initial random collection and then the block clock's value.
    input_scale : float, default=1.0
        Positive finite feature multiplier used to train these weights.
    spawn_frame : {"left", "world"}, default="left"
        Frame used to train these weights. It is part of System identity.
    name : str, default="PQN-VDN"
        Nonempty display name; it proves nothing about training.
    checkpoint : str or None, default=None
        Optional identity label. No file is opened.
    actor_input_schema : int, default=ACTOR_INPUT_SCHEMA_VERSION
        Actor input schema the network was trained on: 2, the current
        5,165-feature schema (the default), or 1 for networks trained before
        Red Zone (5,164 features). Schema 1 picks the cached
        _schema_1_pqn_actor_apply hook once here, so the actor sees and acts
        exactly as before. The factory does not compare the network's width
        with the schema.

    Returns
    -------
    System
        Stable apply/init hooks with ``PQNActorVariables(network, epsilon)`` as
        variables and float32 (B,5,512) per-actor memory. Memory resets only at
        episode starts; death and respawn keep it. Each decision also returns
        ``PQNLearningOutputs`` with the memory it received.

    Raises
    ------
    ValueError
        epsilon, input_scale, spawn_frame, actor_input_schema or name is
        invalid.

    Notes
    -----
    Pass changing variables through variables_a/b inside jit. Action
    selection never updates BatchNorm statistics. The System holds no
    optimizer, window or opponent history.
    """
    rate = _unit_interval(epsilon, "epsilon")
    scale = _input_scale(input_scale)
    frame = _spawn_frame(spawn_frame)
    hook = (
        _schema_1_pqn_actor_apply
        if _actor_input_schema(actor_input_schema) == 1
        else _pqn_actor_apply
    )
    apply = hook(scale, frame)
    return System(
        name,
        apply,
        variables=PQNActorVariables(network, jnp.asarray(rate, jnp.float32)),
        init=_initial_pqn_memory,
        checkpoint=checkpoint,
    )


def _check_batch(batch: PQNBatch) -> tuple[int, int]:
    """Check PQNBatch leaf shapes and dtypes statically; return (C, D).

    Raises ValueError for a mismatch. The feature width may differ from the
    real constant, so tests can check shapes cheaply.
    """
    if batch.valid.ndim != 2 or batch.valid.shape[0] < 2:
        raise ValueError("PQNBatch needs (C,D) rows with C >= 2")
    rows, games = batch.valid.shape
    expected = {
        "action_mask": ((rows, games, TEAM_SLOTS, NUM_ACTIONS), jnp.bool_),
        "actions": ((rows, games, TEAM_SLOTS), jnp.int32),
        "rewards": ((rows, games), jnp.float32),
        "episode_start": ((rows, games), jnp.bool_),
        "ended": ((rows, games), jnp.bool_),
        "valid": ((rows, games), jnp.bool_),
        "active": ((rows, games, TEAM_SLOTS), jnp.bool_),
        "initial_memory": ((games, TEAM_SLOTS, PQN_HIDDEN_SIZE), jnp.float32),
    }
    for field, (shape, dtype) in expected.items():
        value = getattr(batch, field)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"PQNBatch.{field} must have shape {shape} and {dtype}")
    features = batch.actor_features
    if (
        features.shape[:-1] != (rows, games, TEAM_SLOTS)
        or features.dtype != jnp.float32
    ):
        raise ValueError("PQNBatch.actor_features has the wrong shape or dtype")
    return rows, games


def lambda_returns(
    rewards: Array,
    ended: Array,
    next_value: Array,
    valid: Array,
    *,
    gamma: float,
    td_lambda: float,
) -> Array:
    """Return the donor's backward lambda-return targets for padded rows.

    Parameters
    ----------
    rewards : Array
        Float32 (C,D) team reward of each row's transition.
    ended : Array
        Bool (C,D). True when that transition ended its episode; the return is
        cut to the immediate reward there.
    next_value : Array
        Float32 (C,D) team bootstrap value of each row: the sum over active
        slots of the legal maximum action value. Row t+1's value bootstraps
        row t.
    valid : Array
        Bool (C,D) chronological real prefix of each column.
    gamma, td_lambda : float
        Discount and lambda weight in [0, 1].

    Returns
    -------
    Array
        Float32 (C-1,D) targets for left rows 0..C-2. Only rows t with valid
        t and t+1 are meaningful; the others must be masked. They are finite
        only when next_value is finite on every row.

    Notes
    -----
    For left row t with successor value V(t+1) and later return G(t+1):
    ``G(t) = r(t) + gamma * (1 - ended(t)) * V(t+1) + gamma * td_lambda *
    (G(t+1) - V(t+1))``, then ``G(t) = r(t)`` where ended(t). At the last left
    row of the real prefix the later return is taken as V(t+1), which gives
    exactly the donor's start ``r + gamma * (1 - ended) * V``. Pure JAX.
    """
    ended_float = ended.astype(jnp.float32)
    successor_valid = valid[1:]
    after = jnp.concatenate([valid[2:], jnp.zeros_like(valid[:1])], axis=0)
    starts_bootstrap = ~(successor_valid & after)

    def step(
        later: Array, row: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        """Compute one left row's return from the row after it."""
        reward, done, value, start = row
        later = jnp.where(start, value, later)
        target = (
            reward + gamma * (1.0 - done) * value + gamma * td_lambda * (later - value)
        )
        target = (1.0 - done) * target + done * reward
        return target, target

    _, targets = jax.lax.scan(
        step,
        next_value[-1],
        (rewards[:-1], ended_float[:-1], next_value[1:], starts_bootstrap),
        reverse=True,
    )
    return targets


def _minibatch_loss(
    params: Tree, batch_stats: Tree, batch: PQNBatch, pqn: PQNConfig
) -> tuple[Array, tuple[Tree, Array, Array]]:
    """Return the masked PQN-VDN loss and its training-forward by-products.

    params and batch_stats are the Flax subtrees before the step; batch is one
    checked PQNBatch with at least one eligible pair (the caller guarantees
    it); pqn supplies gamma, td_lambda and input_scale. Returns
    ``(loss, (new_batch_stats, mean_q, mean_target))``: the mean squared team
    TD error over eligible pairs, the statistics after this training forward,
    and the mean VDN value and mean target over those pairs. Padding is
    replaced by finite neutral values first; inactive slots add nothing.
    Differentiable in params only; targets carry no gradient. Pure JAX.
    """
    valid = batch.valid
    pair = valid[:-1] & valid[1:]
    denominator = jnp.maximum(jnp.sum(pair, dtype=jnp.int32), 1).astype(jnp.float32)
    neutral = jnp.arange(NUM_ACTIONS) == 0
    live = valid[..., None] & batch.active
    features = jnp.where(valid[..., None, None], batch.actor_features, 0.0)
    mask = jnp.where(live[..., None], batch.action_mask, neutral)
    actions = jnp.where(live, batch.actions, 0)
    rewards = jnp.where(valid, batch.rewards, 0.0)
    (_, values), updates = cast(
        tuple[tuple[Array, Array], dict[str, Tree]],
        PQNNetwork(input_scale=pqn.input_scale).apply(
            {"params": params, "batch_stats": batch_stats},
            batch.initial_memory,
            features,
            valid & batch.episode_start,
            valid,
            batch.active,
            train=True,
            mutable=["batch_stats"],
        ),
    )
    stopped = jax.lax.stop_gradient(values)
    best = jnp.max(_masked_logits(stopped, mask), axis=-1)
    team_best = jnp.sum(jnp.where(live, best, 0.0), axis=-1)
    targets = jax.lax.stop_gradient(
        lambda_returns(
            rewards,
            valid & batch.ended,
            team_best,
            valid,
            gamma=pqn.gamma,
            td_lambda=pqn.td_lambda,
        )
    )
    chosen = jnp.take_along_axis(values, actions[..., None], axis=-1)[..., 0]
    team = jnp.sum(jnp.where(live, chosen, 0.0), axis=-1)[:-1]
    error = jnp.where(pair, team - targets, 0.0)
    loss = jnp.sum(jnp.square(error)) / denominator
    mean_q = jnp.sum(jnp.where(pair, team, 0.0)) / denominator
    mean_target = jnp.sum(jnp.where(pair, targets, 0.0)) / denominator
    return loss, (updates["batch_stats"], mean_q, mean_target)


def update_pqn(
    state: PQNTrainState,
    batch: PQNBatch,
    *,
    pqn: PQNConfig,
    planned_learning_blocks: int,
    learning_rate: Callable[[Array], Array] | None = None,
) -> tuple[PQNTrainState, PQNMetrics]:
    """Compute one PQN-VDN optimizer step on one expanded minibatch.

    Parameters
    ----------
    state : PQNTrainState
        Network, statistics, optimizer and step count before this step.
    batch : PQNBatch
        C time rows of D whole game sequences in the network frame.
    pqn : PQNConfig
        Static settings: gamma, td_lambda, input_scale and the optimizer.
    planned_learning_blocks : int
        Static positive N that fixes the learning-rate schedule.

    learning_rate : callable or None, default=None
        Checked future rate at the absolute optimizer count. None preserves
        the original rule. It changes no optimizer state or counter layout.

    Returns
    -------
    tuple[PQNTrainState, PQNMetrics]
        Candidate state and this step's metrics. With no eligible pair the
        state is returned unchanged with absent metrics: no training forward,
        no statistics update and no optimizer step. The caller decides whether
        to accept a candidate; check ``metrics.finite`` and ``performed``.

    Raises
    ------
    ValueError
        Batch shapes or dtypes are wrong, or planned_learning_blocks is invalid.

    Notes
    -----
    One training-mode forward from ``initial_memory`` over all C rows gives
    the action values and new BatchNorm statistics (masked to valid active
    rows, the last real row included). The prediction is the VDN sum of the
    recorded actions' values over active slots. The target uses the same
    forward's stopped values: the sum over active slots of each slot's legal
    maximum (-inf masking), then :func:`lambda_returns`. There is no target
    network. The loss averages the squared team error over eligible pairs.
    Gradients are clipped by global norm, then RAdam steps. Pure JAX; wrap in
    jit with fixed settings.
    """
    _check_batch(batch)
    optimizer = pqn_optimizer(
        pqn,
        planned_learning_blocks,
        learning_rate=learning_rate,
        optimizer_count=state.optimizer_steps,
    )
    pair = batch.valid[:-1] & batch.valid[1:]
    count = jnp.sum(pair, dtype=jnp.int32)

    def step(_: None) -> tuple[PQNTrainState, PQNMetrics]:
        """Run the training forward, clip, and apply one RAdam step."""
        (loss, (stats, mean_q, mean_target)), grads = jax.value_and_grad(
            _minibatch_loss, has_aux=True
        )(state.network.params, state.network.batch_stats, batch, pqn)
        updates, opt_state = optimizer.update(
            grads, state.opt_state, state.network.params
        )
        params = optax.apply_updates(state.network.params, updates)
        candidate = PQNTrainState(
            PQNInferenceVariables(params, stats),
            opt_state,
            state.optimizer_steps + 1,
        )
        metrics = PQNMetrics(
            loss,
            mean_q,
            mean_target,
            optax.tree.norm(grads),
            count,
            jnp.sum(batch.active[:-1] & pair[..., None], dtype=jnp.int32),
            _finite((loss, mean_q, mean_target, grads, candidate)),
            jnp.bool_(True),
        )
        return candidate, metrics

    def skip(_: None) -> tuple[PQNTrainState, PQNMetrics]:
        """Return the unchanged state and absent metrics when no pair exists."""
        zero = jnp.float32(0.0)
        return state, PQNMetrics(
            zero,
            zero,
            zero,
            zero,
            jnp.int32(0),
            jnp.int32(0),
            jnp.bool_(True),
            jnp.bool_(False),
        )

    return cast(
        tuple[PQNTrainState, PQNMetrics], jax.lax.cond(count > 0, step, skip, None)
    )
