# Copyright 2022 InstaDeep Ltd. All rights reserved.
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
# Adapted from Mava for MARL-BGs.
# See docs/training/source_reuse.md for source identities and deliberate changes.
"""Provide recurrent QMIX networks, its actor System and one sampled update.

This module adapts Mava's recurrent QMIX (``rec_qmix``) at revision
``9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93``. It owns the QMIX settings, the
local Q-networks, the monotonic mixer, the epsilon-greedy actor System,
the exploration clock, network initialization and one Double-Q optimizer step
on an already expanded replay sample. It owns no replay buffer, collection
loop, checkpoint, curriculum or training run; ``training.qmix_learner`` joins
these pieces to the shared collection owner.

Actors read only their permitted SharedObs input and their own memory. The
mixer alone reads the separate 920-value physical training state; it never
enters action selection. Q-networks trained before Red Zone, on historical
actor input schema 1, play through a separate cached schema-1 hook that
removes the Red Zone depth column and keeps the old spawn-side formula.
Requires the optional training extra. Everything here is pure JAX that works
inside jit and scan, except the host-only setting checks. Shared models also
support vmap; grouped contractions do not support every extra mapped axis. The default
settings are the donor's starting values, not settings qualified for learning
in MARL-BGs.
"""

import dataclasses
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
        "QMIX needs the training extra. Install marl-battlegrounds[training]."
    ) from error

# The mask helper keeps one owner for categorical legality and its -inf rule.
# pyright: reportPrivateUsage=false
from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    _masked_logits,
    categorical_action_mask,
    decode_actions,
    mirror_action_indices,
    sample_actions,
)
from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_SIZE,
    ACTOR_INPUT_SCHEMA_VERSION,
    SPAWN_FRAMES,
    TRAINING_STATE_FEATURE_SIZE,
    encode_actor_inputs,
    schema_1_actor_input,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.baselines.ppo import (
    _PARAMETER_SHARING,
    MLPTorso,
    _actor_group_presence,
    _actor_input_schema,
    _batch_actor_groups,
    _group_dense,
    _group_gru,
    _group_order,
    _grouped_actor_inputs,
    _input_scale,
    _parameter_sharing,
    _restore_group_rows,
    _spawn_frame,
    _update_actor_groups,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.input import mirror_team_view

type Tree = Any

QMIX_METHOD = "qmix"
"""Training method name that selects QMIX in configurations and checkpoints."""
QMIX_HIDDEN_SIZE = 256
"""Width of the Q-network torsos and of each actor's recurrent memory."""
QMIX_EMBED_SIZE = 32
"""Width of the mixer's hidden layer (the donor's ``qmix_embed_dim``)."""
QMIX_HYPER_HIDDEN_SIZE = 64
"""Hidden width of the mixer's weight hypernetworks (``hyper_hidden_dim``)."""
TEAM_SLOTS = 5
"""Team A actor slots mixed into one team value; smaller rosters pad inactive."""
QMIX_TIE_RULE = "first_legal_maximum"
"""Greedy rule: the lowest legal action index among equal best scores wins."""
QMIX_KEY_SCHEMA_VERSION = 1
"""Version of the learner's model-initialization and sampling key tags."""
QMIX_REPLAY_SCHEMA_VERSION = 1
"""Version of the compact QMIX replay row layout."""
_INT32_MAX = 2**31 - 1


def _count(value: object, name: str, *, minimum: int = 1) -> int:
    """Check one plain integer count within [minimum, int32 max] and return it.

    Booleans, floats, arrays and out-of-range values raise ValueError. This
    host-only check performs no device work.
    """
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be a plain integer in [{minimum}, 2**31-1].")
    return value


def _unit_interval(value: object, name: str) -> float:
    """Check one finite real number in [0, 1], excluding Booleans; return float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [0, 1].")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a finite real number in [0, 1].")
    return number


def _positive(value: object, name: str) -> float:
    """Check one finite positive real number, excluding Booleans; return float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive real number.")
    return number


@dataclass(frozen=True)
class QMIXConfig:
    """Hold the static recurrent QMIX settings, using the donor's names.

    Parameters
    ----------
    rollout_length : int, default=8
        Most real rounds collected in one block before replay insertion and
        learning (T). One round is one decision in every game lane. Blocks may
        be shorter at the end of a budget or in low-level use.
    buffer_size : int, default=1000
        Replay capacity in time rows per game lane (C). The total number of rows
        is C times the number of lanes. It must be at least min_buffer_size and
        at least rollout_length.
    min_buffer_size : int, default=32
        Rows per lane that must be stored before any sample or optimizer step.
        It must be at least sample_sequence_length.
    sample_sequence_length : int, default=20
        Consecutive rows per sampled sequence (S). S rows give S-1 adjacent TD
        pairs. Must be at least 2.
    sample_batch_size : int, default=128
        Sequences drawn for each optimizer step (M), with replacement.
    epochs : int, default=4
        Fresh samples and optimizer steps per ready block.
    q_lr : float, default=0.00003
        Adam learning rate shared by the online Q-network and mixer. Adam keeps
        its defaults (b1 0.9, b2 0.999, eps 1e-8). There is no gradient clipping:
        the donor configuration names ``max_grad_norm`` but its optimizer never
        uses it, so it is not copied.
    hard_update : bool, default=True
        Python bool choosing the target-network rule. True copies the online
        networks when the optimizer count before a step is a multiple of
        update_period (counts 0, 200, 400 and so on). False blends the online
        networks into the targets with weight tau after every step.
    update_period : int, default=200
        Optimizer steps between hard target copies. Kept and checked even when
        hard_update is False.
    tau : float, default=0.01
        Soft-blend weight in [0, 1]. Kept and checked even when hard_update is
        True.
    gamma : float, default=0.99
        Discount per transition, in [0, 1].
    eps_min : float, default=0.05
        Exploration rate reached at the end of the decay, in [0, 1].
    eps_decay : int, default=100000
        Real environment transitions (all game lanes counted) over which the
        exploration rate falls linearly from 1 to eps_min. It is a number of
        transitions, not a rate.
    input_scale : float, default=1.0
        Positive finite multiplier applied to every actor feature before the
        first Dense layer. The mixer's physical state is never scaled.
    spawn_frame : str, default="left"
        Network frame for actor features, masks and actions: "left" reflects
        every game to look like a start from the left bank, "world" keeps raw
        coordinates. Replay rows and physical state stay in the world frame.

    parameter_sharing : {"all", "class", "none"}, default="all"
        Share actor weights across everyone, within each class, or not between
        physical slots. Grouped modes have five parameter groups. Memory stays
        separate per physical actor. Class changes at resets select new weights.

    Raises
    ------
    ValueError
        A count is not a plain integer in range, a float is nonfinite, Boolean
        or outside its range, hard_update is not a bool, the spawn frame is
        unknown, the buffer ordering ``buffer_size >= min_buffer_size >=
        sample_sequence_length >= 2`` or ``rollout_length <= buffer_size``
        fails, or a per-step or per-block sample count would overflow int32.

    Notes
    -----
    The network widths (Q 256, recurrent 256, mixer embedding 32, hypernetwork
    64) and the replay period of 1 are fixed contracts, not settings. The
    record is immutable host setup; capture one instance in compiled code.
    """

    rollout_length: int = 8
    buffer_size: int = 1000
    min_buffer_size: int = 32
    sample_sequence_length: int = 20
    sample_batch_size: int = 128
    epochs: int = 4
    q_lr: float = 0.00003
    hard_update: bool = True
    update_period: int = 200
    tau: float = 0.01
    gamma: float = 0.99
    eps_min: float = 0.05
    eps_decay: int = 100000
    input_scale: float = 1.0
    spawn_frame: str = "left"
    parameter_sharing: str = "all"

    def __post_init__(self) -> None:
        """Reject invalid settings on the host before any array is created."""
        for name in (
            "rollout_length",
            "buffer_size",
            "min_buffer_size",
            "sample_batch_size",
            "epochs",
            "update_period",
            "eps_decay",
        ):
            _count(getattr(self, name), name)
        _count(self.sample_sequence_length, "sample_sequence_length", minimum=2)
        if not (
            self.buffer_size >= self.min_buffer_size >= self.sample_sequence_length
        ):
            raise ValueError(
                "QMIX needs buffer_size >= min_buffer_size >= sample_sequence_length"
            )
        if self.rollout_length > self.buffer_size:
            raise ValueError("QMIX needs rollout_length <= buffer_size")
        _positive(self.q_lr, "q_lr")
        _input_scale(self.input_scale)
        _spawn_frame(self.spawn_frame)
        _parameter_sharing(self.parameter_sharing)
        for name in ("tau", "gamma", "eps_min"):
            _unit_interval(getattr(self, name), name)
        if type(self.hard_update) is not bool:
            raise ValueError("hard_update must be a Python bool")
        per_step = self.sample_batch_size * (self.sample_sequence_length - 1)
        if per_step * TEAM_SLOTS > _INT32_MAX:
            raise ValueError("One QMIX step's utility count would overflow int32")
        if self.epochs * per_step * TEAM_SLOTS > _INT32_MAX:
            raise ValueError("One QMIX block's sample counts would overflow int32")


DEFAULT_QMIX_CONFIG = QMIXConfig()
"""The donor's recurrent QMIX settings with the BG left spawn frame."""


def validate_qmix_batch_size(num_envs: int, config: QMIXConfig) -> None:
    """Check that QMIX's int32 counts for num_envs games cannot overflow.

    Parameters
    ----------
    num_envs : int
        Plain positive number of game lanes (B).
    config : QMIXConfig
        Settings whose rollout_length bounds one block and whose eps_decay
        sets the exploration clock.

    Raises
    ------
    ValueError
        ``num_envs * rollout_length * 5`` exceeds int32 (the shared update
        summary reduces one block's actor rows into int32 counters), or
        ``eps_decay + num_envs - 1`` exceeds int32 (the largest value of the
        saturated exploration clock in ``epsilon_at``).

    Notes
    -----
    Host-only; it creates no arrays. ``TrainConfig`` and
    ``init_qmix_learner`` both call it before any file or array work.
    Whole-run totals are Python integers and are not limited by this check.
    """
    _count(num_envs, "num_envs")
    if num_envs * config.rollout_length * TEAM_SLOTS > _INT32_MAX:
        raise ValueError("One QMIX block's actor counts would overflow int32")
    if config.eps_decay + num_envs - 1 > _INT32_MAX:
        raise ValueError("The QMIX exploration clock would overflow int32")


class QMIXActorVariables(NamedTuple):
    """Carry everything that changes in a QMIX actor between decisions.

    Attributes
    ----------
    params : PyTree
        Flax variables of the local Q-network (``{"params": ...}``). Grouped
        modes add a leading axis of five to every leaf.
        The mixer, target networks and optimizer never belong here.
    epsilon : Array
        Float32 0-d exploration probability in [0, 1]. Zero means greedy.
        During training the collection hook sets it from the exploration clock
        before every decision; loaded Systems use zero.

    Notes
    -----
    This immutable PyTree is the actor's whole variable tree, stored as the
    current actor and in each of the 20 historical opponent slots. The class
    name and field order enter tree digests, so they are part of the saved
    identity.
    """

    params: Tree
    epsilon: Array


class QMIXTrainState(NamedTuple):
    """Hold one temporary numerical view for initialization and one update step.

    Attributes
    ----------
    online_q, target_q : PyTree
        Flax variables of the online and target local Q-networks. Class and
        slot sharing add a leading axis of five to every Q-network leaf.
    online_mixer, target_mixer : PyTree
        Flax variables of the online and target mixers.
    opt_state : PyTree
        With shared parameters, one clip-then-Adam state over
        ``(online_q, online_mixer)``. Class and slot sharing use a tuple of five
        independent Q optimizer states and one mixer optimizer state. Each has
        its own clipping and complete Adam history, including counts.
    optimizer_steps : Array
        Int32 0-d count of successful optimizer steps before this state.

    Notes
    -----
    The durable learner keeps the online Q parameters in its actor history,
    not in a second long-lived copy; this record exists only while one update
    is computed. Arrays stay dynamic under jit.
    """

    online_q: Tree
    target_q: Tree
    online_mixer: Tree
    target_mixer: Tree
    opt_state: Tree
    optimizer_steps: Array


class QMIXBatch(NamedTuple):
    """Hold one expanded replay sample of M sequences, each S rows long.

    Attributes
    ----------
    actor_features : Array
        Float32 (M,S,5,F_A) encoded permitted actor inputs in the network frame.
        F_A is ACTOR_FEATURE_SIZE (5165) for real inputs; reference tests may
        use a smaller width.
    action_mask : Array
        Bool (M,S,5,198) categorical legality in the network frame.
    actions : Array
        Int32 (M,S,5) chosen categorical actions in the network frame.
    rewards : Array
        Float32 (M,S) team training reward of each row: the mean task reward
        over configured Team A slots plus the team shaping reward.
    episode_start : Array
        Bool (M,S). True on the first decision of an episode; recurrent memory
        resets before that row.
    ended : Array
        Bool (M,S). True when the row's transition ended its episode; the next
        row then gives no bootstrap. A cutoff that did not end the game is False.
    valid : Array
        Bool (M,S). False rows are padding; they are neutralized, keep carry
        unchanged and never enter a TD pair. Real replay samples are all True.
    active : Array
        Bool (M,S,5). Configured Team A slots. Inactive slots add nothing to the
        team value or the gradient, whatever their finite features and actions
        hold; the BG encoder pads them with finite values. Temporarily dead
        configured actors stay active.
    training_state : Array
        Float32 (M,S,F_S) world-frame physical state for the mixer only. F_S is
        TRAINING_STATE_FEATURE_SIZE (920) for real inputs.

    actor_group : Array or None, default=None
        Optional int32 with active.shape, group IDs 0..4 from raw self classes
        or physical slots. Class sharing requires it; all ignores it.
    learner_active : Array or None, default=None
        Optional bool with active.shape selecting owned actor slots. None uses
        physical active. Partner utilities do not enter current or successor
        team values; their actual actions already affect recorded transitions.

    Notes
    -----
    A TD pair uses rows t and t+1 and is eligible only when both are valid.
    With learner_active, a pair also needs a learner slot on its left row.
    Every sample starts from zero recurrent memory; there is no burn-in.
    """

    actor_features: Array
    action_mask: Array
    actions: Array
    rewards: Array
    episode_start: Array
    ended: Array
    valid: Array
    active: Array
    training_state: Array
    actor_group: Array | None = None
    learner_active: Array | None = None


class QMIXMetrics(NamedTuple):
    """Report one optimizer step's small scalar results.

    Attributes
    ----------
    loss : Array
        Float32 mean squared team TD error over eligible pairs; 0 when none.
    mean_q : Array
        Float32 mean online team value of the chosen actions over those pairs.
    mean_target : Array
        Float32 mean TD target over those pairs.
    sampled_sequences : Array
        Int32 count of sequences with at least one eligible pair.
    used_td_pairs : Array
        Int32 count of eligible adjacent pairs (valid left and right rows).
    used_agent_utilities : Array
        Int32 count of configured actors on the left rows of those pairs.
    finite : Array
        Bool. True when the loss, gradients and candidate state are finite.

    Notes
    -----
    Counts describe what this step used. Repeated draws of the same stored rows
    count again; they are not new experience.
    """

    loss: Array
    mean_q: Array
    mean_target: Array
    sampled_sequences: Array
    used_td_pairs: Array
    used_agent_utilities: Array
    finite: Array


class QMIXScannedGRU(nn.Module):
    """Carry each actor's 256-wide memory through time with the donor's GRU.

    Parameters are broadcast across time. Each input row is (embedding, reset,
    valid), shaped (E,5,256), (E,5) and (E,5). A reset happens before the cell;
    invalid rows keep their memory. The parent names this module
    ``ScannedRNN_0`` and its cell ``GRUCell_1``, the donor's parameter paths.
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
        self, carry: Array, inputs: tuple[Array, Array, Array]
    ) -> tuple[Array, Array]:
        """Apply one GRU step with resets before the cell and preserved padding.

        Parameters
        ----------
        carry : Array
            Float32 (E,5,256) memory immediately before this decision.
        inputs : tuple[Array, Array, Array]
            Embeddings float32 (E,5,256), episode-reset flags bool (E,5) and
            valid-decision flags bool (E,5). The scan supplies one time row.

        Returns
        -------
        tuple[Array, Array]
            Next memory and cell output, both float32 (E,5,256). Invalid rows
            keep their old memory and emit zero output.
        """
        features, resets, valid = inputs
        before = jnp.where((resets & valid)[..., None], 0.0, carry)
        updated, output = cast(
            tuple[Array, Array],
            nn.GRUCell(features=QMIX_HIDDEN_SIZE, name="GRUCell_1")(before, features),
        )
        return jnp.where(valid[..., None], updated, carry), jnp.where(
            valid[..., None], output, 0.0
        )


class RecurrentQNetwork(nn.Module):
    """Map permitted actor features and memory to 198 action values per actor.

    The layers are the donor's: Dense 256 with ReLU (``pre_torso``), a 256-wide
    GRU (``ScannedRNN_0``), Dense 256 with ReLU (``post_torso``) and a Dense
    198 head with orthogonal gain 0.01 (``Dense_0``). Parameters are shared by
    all actors and teammates; memory stays separate per actor.

    input_scale is a fixed positive finite multiplier applied to every feature
    before the first layer; it changes no parameter shape.
    """

    input_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        carry: Array,
        features: Array,
        resets: Array,
        valid: Array,
        *,
        actor_group: Array | None = None,
    ) -> tuple[Array, Array]:
        """Read feature sequences with independent actor memory.

        Parameters
        ----------
        carry : Array
            Float32 (E,5,256) memory immediately before the sequence.
        features : Array
            Float32 (T,E,5,F) encoded permitted inputs. F must match the
            initialized width, normally ACTOR_FEATURE_SIZE.
        resets : Array
            Bool (T,E,5), True when a new episode starts before that decision.
        valid : Array
            Bool (T,E,5) valid decisions. Invalid rows keep prior memory.

        actor_group : Array or None, default=None
            Optional int32 (T,E,5) group IDs 0..4. Variables then have a
            leading axis of five independent actor groups. None keeps the
            shared network. Initialize groups by mapping the shared init.

        Returns
        -------
        tuple[Array, Array]
            Final float32 (E,5,256) memory and float32 (T,E,5,198) raw action
            values. Values on invalid rows are meaningless and must be ignored.

        Raises
        ------
        ValueError
            input_scale is not a finite positive real number.
        """
        scale = _input_scale(self.input_scale)
        if scale != 1.0:
            features = features * scale
        if actor_group is not None:
            params = self.variables["params"]
            order, groups, sizes = _group_order(actor_group)
            x = features.reshape(-1, features.shape[-1])[order]
            x = nn.relu(_group_dense(x, params["pre_torso"]["Dense_0"], groups, sizes))
            embedding = _restore_group_rows(x, order, features.shape[:-1])
            carry, embedding = _group_gru(
                params["ScannedRNN_0"]["GRUCell_1"],
                carry,
                embedding,
                resets,
                valid,
                actor_group,
            )
            x = embedding.reshape(-1, embedding.shape[-1])[order]
            x = nn.relu(_group_dense(x, params["post_torso"]["Dense_0"], groups, sizes))
            x = _group_dense(x, params["Dense_0"], groups, sizes)
            return carry, _restore_group_rows(x, order, features.shape[:-1])
        embedding = MLPTorso((QMIX_HIDDEN_SIZE,), name="pre_torso")(features)
        carry, embedding = QMIXScannedGRU(name="ScannedRNN_0")(
            carry, (embedding, resets, valid)
        )
        embedding = MLPTorso((QMIX_HIDDEN_SIZE,), name="post_torso")(embedding)
        values = nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.orthogonal(0.01))(
            embedding
        )
        return carry, values


class _HyperNetwork(nn.Module):
    """Apply the donor's hypernetwork torso: Dense layers, ReLU between them.

    layer_sizes lists the Dense widths. Kernels use orthogonal gain sqrt(2) and
    biases start at zero. The last layer has no activation (the donor's
    ``activate_final=False``). Input (...,F) becomes (...,layer_sizes[-1]).
    """

    layer_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Map each physical-state row through the Dense layers.

        Parameters
        ----------
        inputs : Array
            Float32 (...,F) normalized physical state.

        Returns
        -------
        Array
            Float32 (...,layer_sizes[-1]) hypernetwork output.
        """
        values = inputs
        for index, size in enumerate(self.layer_sizes):
            values = nn.Dense(size, kernel_init=nn.initializers.orthogonal(2**0.5))(
                values
            )
            if index < len(self.layer_sizes) - 1:
                values = nn.relu(values)
        return values


class QMixingNetwork(nn.Module):
    """Combine five local utilities into one team value, monotonically.

    Hypernetworks read the layer-normalized physical state: ``hyper_w1``
    (920→64→160), ``hyper_b1`` (920→32), ``hyper_w2`` (920→64→32) and
    ``hyper_b2`` (920→32→ReLU→1). The first and second weights pass through
    absolute value, so raising any utility never lowers the team value. The
    hidden layer uses ELU. Parameter paths match the donor's QMixingNetwork.
    """

    def setup(self) -> None:
        """Create the four hypernetworks and the physical-state LayerNorm."""
        self.hyper_w1 = _HyperNetwork(
            (QMIX_HYPER_HIDDEN_SIZE, QMIX_EMBED_SIZE * TEAM_SLOTS)
        )
        self.hyper_b1 = _HyperNetwork((QMIX_EMBED_SIZE,))
        self.hyper_w2 = _HyperNetwork((QMIX_HYPER_HIDDEN_SIZE, QMIX_EMBED_SIZE))
        self.hyper_b2 = _HyperNetwork((QMIX_EMBED_SIZE, 1))
        self.layer_norm = nn.LayerNorm()

    def __call__(self, utilities: Array, state: Array) -> Array:
        """Return one team value for every row.

        Parameters
        ----------
        utilities : Array
            Float32 (...,5) chosen-action values of the five Team A slots.
            Callers set inactive slots to zero first; zero adds nothing.
        state : Array
            Float32 (...,F_S) world-frame physical state for the same rows.

        Returns
        -------
        Array
            Float32 (...) team values, one per leading position.

        Notes
        -----
        Slot i uses columns 32i to 32i+31 of the first weight output.
        """
        lead = utilities.shape[:-1]
        states = self.layer_norm(state)
        w1 = jnp.abs(self.hyper_w1(states)).reshape(*lead, TEAM_SLOTS, QMIX_EMBED_SIZE)
        b1 = self.hyper_b1(states).reshape(*lead, 1, QMIX_EMBED_SIZE)
        hidden = nn.elu(jnp.matmul(utilities[..., None, :], w1) + b1)
        w2 = jnp.abs(self.hyper_w2(states)).reshape(*lead, QMIX_EMBED_SIZE, 1)
        b2 = self.hyper_b2(states).reshape(*lead, 1, 1)
        return (jnp.matmul(hidden, w2) + b2).reshape(lead)


def _optimizer(config: QMIXConfig) -> Any:  # noqa: ANN401
    """Return the donor's optimizer: ``optax.chain(optax.adam(q_lr))``.

    Adam keeps its defaults and there is no clipping. The returned Optax
    transformation holds no state; initialize it on ``(online_q,
    online_mixer)``. Host-only construction; no arrays are created.
    """
    return optax.chain(optax.adam(learning_rate=config.q_lr))


def _check_key(key: Array) -> None:
    """Require one Threefry key, typed scalar or legacy uint32 (2,)."""
    if str(jax.random.key_impl(key)) != "threefry2x32":
        raise ValueError("QMIX initialization requires a Threefry key")
    if jax.random.key_data(key).shape != (2,):
        raise ValueError("QMIX initialization requires one key")


def _q_network_variables(key: Array, config: QMIXConfig) -> Tree:
    """Initialize the local Q-network's Flax variables from one key.

    Zero inputs shaped (T=1, E=1, 5, ACTOR_FEATURE_SIZE) fix parameter shapes;
    values depend only on the key and the module paths.
    """
    if config.parameter_sharing != "all":
        shared = dataclasses.replace(config, parameter_sharing="all")
        return jax.vmap(functools.partial(_q_network_variables, config=shared))(
            jax.random.split(key, 5)
        )
    return RecurrentQNetwork(input_scale=config.input_scale).init(
        key,
        jnp.zeros((1, TEAM_SLOTS, QMIX_HIDDEN_SIZE), jnp.float32),
        jnp.zeros((1, 1, TEAM_SLOTS, ACTOR_FEATURE_SIZE), jnp.float32),
        jnp.zeros((1, 1, TEAM_SLOTS), jnp.bool_),
        jnp.ones((1, 1, TEAM_SLOTS), jnp.bool_),
    )


def initialize_qmix(
    key: Array, config: QMIXConfig = DEFAULT_QMIX_CONFIG
) -> QMIXTrainState:
    """Create untrained QMIX networks, equal targets and complete Adam states.

    Parameters
    ----------
    key : Array
        One Threefry key, typed scalar or legacy uint32 (2,). It is used the way
        the donor uses its ``q_key``: the online Q-network, target Q-network,
        online mixer and target mixer all initialize from this same key, so
        online and target start equal. The learner derives it with
        ``fold_in(root, QMIX_MODEL_INITIALIZATION_TAG)``.
    config : QMIXConfig, default=DEFAULT_QMIX_CONFIG
        Settings for input_scale, parameter_sharing, q_lr, clipping and Adam.

    Returns
    -------
    QMIXTrainState
        Float32 network variables, the optimizer layout documented by
        QMIXTrainState and ``optimizer_steps`` int32 0. Grouped modes initialize
        five independently keyed Q-networks and keep one mixer.

    Raises
    ------
    ValueError
        key is not one Threefry key.

    Notes
    -----
    Allocates about 8.1 MB of float32 parameters for each online/target copy
    and twice that for Adam moments in shared mode. Class and slot modes keep
    five Q-network copies and their moments; the mixer is unchanged. Works
    under ``jax.eval_shape``.
    """
    _check_key(key)
    q_variables = _q_network_variables(key, config)
    mixer_variables = QMixingNetwork().init(
        key,
        jnp.zeros((1, 1, TEAM_SLOTS), jnp.float32),
        jnp.zeros((1, 1, TRAINING_STATE_FEATURE_SIZE), jnp.float32),
    )
    return QMIXTrainState(
        q_variables,
        q_variables,
        mixer_variables,
        mixer_variables,
        _optimizer(config).init((q_variables, mixer_variables))
        if config.parameter_sharing == "all"
        else (
            jax.vmap(_optimizer(config).init)(q_variables),
            _optimizer(config).init(mixer_variables),
        ),
        jnp.int32(0),
    )


def qmix_actor_template(config: QMIXConfig = DEFAULT_QMIX_CONFIG) -> Tree:
    """Return the Q-network's variable shapes and dtypes without allocating them.

    Parameters
    ----------
    config : QMIXConfig, default=DEFAULT_QMIX_CONFIG
        parameter_sharing selects the shared tree or a leading axis of five.
        Other settings do not affect this template.

    Returns
    -------
    PyTree
        ``jax.ShapeDtypeStruct`` leaves of the Q-network variables. No mixer,
        target, optimizer or replay is traced. Use it to restore actor-only
        payloads.
    """
    return jax.eval_shape(
        functools.partial(_q_network_variables, config=config), jax.random.key(0)
    )


def greedy_actions(q_values: Array, mask: Array) -> Array:
    """Pick the first legal maximum of each actor's 198 action values.

    Parameters
    ----------
    q_values : Array
        Finite float (...,198) action values in the network frame.
    mask : Array
        Bool (...,198) legality with at least one legal action per row.

    Returns
    -------
    Array
        Int32 (...) chosen indices. Illegal actions are set to negative infinity
        first, so every legal finite value beats them and the winner is always
        legal. Ties go to the lowest index.

    Raises
    ------
    ValueError
        The final widths are not 198 or the shapes differ.
    TypeError
        mask is not Boolean.

    Notes
    -----
    The donor masks with ``finfo(float32).min``, which could choose an illegal
    action if every legal value were exactly that minimum; negative infinity is
    a narrow safety adaptation. For ordinary finite values both give the same
    choice.
    """
    return jnp.argmax(_masked_logits(q_values, mask), axis=-1).astype(jnp.int32)


def action_probabilities(q_values: Array, mask: Array, epsilon: Array | float) -> Array:
    """Return the epsilon-greedy action distribution over 198 categories.

    Parameters
    ----------
    q_values : Array
        Finite float (...,198) action values.
    mask : Array
        Bool (...,198) legality with at least one legal action per row.
    epsilon : Array or float
        Exploration probability in [0, 1], a scalar or broadcastable to (...).

    Returns
    -------
    Array
        Float32 (...,198) probabilities: ``epsilon * uniform_legal + (1 -
        epsilon) * onehot(greedy)``. The greedy action also receives its share
        of the uniform part. Illegal actions get zero.

    Raises
    ------
    ValueError
        The final widths are not 198 or the shapes differ.
    TypeError
        mask is not Boolean.

    Notes
    -----
    This is the donor's MaskedEpsGreedyDistribution. The actor draws from the
    same distribution in two stages (see make_qmix_system).
    """
    legal = mask.astype(jnp.float32)
    uniform = legal / jnp.sum(legal, axis=-1, keepdims=True)
    greedy = jax.nn.one_hot(
        greedy_actions(q_values, mask), NUM_ACTIONS, dtype=jnp.float32
    )
    rate = jnp.asarray(epsilon, jnp.float32)[..., None]
    return rate * uniform + (1.0 - rate) * greedy


def epsilon_at(
    completed_rounds: Array, num_envs: int, eps_min: float, eps_decay: int
) -> Array:
    """Return the exploration rate before the next decision.

    Parameters
    ----------
    completed_rounds : Array
        Int32 0-d real rounds completed before the decision (one round is one
        decision in every lane). Not multiplied by num_envs.
    num_envs : int
        Plain positive number of game lanes (B).
    eps_min : float
        Final rate in [0, 1].
    eps_decay : int
        Plain positive number of real transitions over which the rate decays.

    Returns
    -------
    Array
        Float32 0-d rate. The clock is ``B * min(rounds, ceil(eps_decay / B))``
        real transitions, which never overflows int32. Once that clock reaches
        eps_decay the result is exactly eps_min. Before that it is the donor's
        ``max(eps_min, 1 - t / eps_decay * (1 - eps_min))`` in float32.

    Raises
    ------
    ValueError
        ``eps_decay + num_envs - 1`` exceeds int32, or a count is invalid.

    Notes
    -----
    Pure JAX; works under jit. Compiled and eager evaluations of the formula
    may differ by a few float32 units in the last place; compare saved values
    with :func:`epsilon_reference` within a tolerance.
    """
    _count(num_envs, "num_envs")
    _count(eps_decay, "eps_decay")
    if eps_decay + num_envs - 1 > _INT32_MAX:
        raise ValueError("The QMIX exploration clock would overflow int32")
    end_rounds = -(-eps_decay // num_envs)
    rounds = jnp.minimum(jnp.asarray(completed_rounds, jnp.int32), end_rounds)
    steps = rounds * jnp.int32(num_envs)
    decayed = 1.0 - (steps.astype(jnp.float32) / jnp.float32(eps_decay)) * (
        jnp.float32(1.0 - eps_min)
    )
    value = jnp.maximum(jnp.float32(eps_min), decayed)
    return jnp.where(steps >= eps_decay, jnp.float32(eps_min), value).astype(
        jnp.float32
    )


def epsilon_reference(
    completed_rounds: int, num_envs: int, eps_min: float, eps_decay: int
) -> float:
    """Return the exploration rate as a float64 host value for validation.

    Parameters
    ----------
    completed_rounds : int
        Nonnegative real rounds completed before the decision. Any Python
        integer is accepted; the clock saturates as in :func:`epsilon_at`.
    num_envs : int
        Positive number of game lanes (B).
    eps_min : float
        Final rate in [0, 1].
    eps_decay : int
        Positive number of real transitions over which the rate decays.

    Returns
    -------
    float
        ``eps_min`` exactly once ``B * min(rounds, ceil(eps_decay / B))``
        reaches eps_decay; otherwise ``max(eps_min, 1 - t / eps_decay *
        (1 - eps_min))`` in float64.

    Notes
    -----
    Pure Python with no checks and no device work. Checkpoint validation
    compares saved float32 rates with this value within a small tolerance,
    after checking the integer clock itself.
    """
    steps = num_envs * min(completed_rounds, -(-eps_decay // num_envs))
    if steps >= eps_decay:
        return float(eps_min)
    return max(float(eps_min), 1.0 - steps / eps_decay * (1.0 - eps_min))


@dataclass(frozen=True)
class QMIXExploration:
    """Set a QMIX actor's epsilon from the real-transition clock before actions.

    Parameters
    ----------
    eps_min : float
        Final exploration rate in [0, 1].
    eps_decay : int
        Real transitions over which the rate decays.
    num_envs : int
        Game lanes in the collection (B).

    Raises
    ------
    ValueError
        A setting is invalid or the clock bound would overflow int32.

    Notes
    -----
    The collection calls this once per real decision with the rounds completed
    before it, on the shared current variables only. Current self-play teams
    therefore share the rate; historical snapshots keep the rate they had when
    captured. It consumes no random key and keeps every shape and dtype.
    """

    eps_min: float
    eps_decay: int
    num_envs: int

    def __post_init__(self) -> None:
        """Check the settings and the int32 clock bound on the host."""
        _unit_interval(self.eps_min, "eps_min")
        _count(self.eps_decay, "eps_decay")
        _count(self.num_envs, "num_envs")
        if self.eps_decay + self.num_envs - 1 > _INT32_MAX:
            raise ValueError("The QMIX exploration clock would overflow int32")

    def __call__(
        self, variables: QMIXActorVariables, completed_rounds: Array
    ) -> QMIXActorVariables:
        """Return the variables with epsilon replaced for the next decision.

        Parameters
        ----------
        variables : QMIXActorVariables
            Current actor variables; the stored epsilon is ignored.
        completed_rounds : Array
            Int32 0-d rounds completed before this decision.

        Returns
        -------
        QMIXActorVariables
            Same params, epsilon from :func:`epsilon_at`.
        """
        return variables._replace(
            epsilon=epsilon_at(
                completed_rounds, self.num_envs, self.eps_min, self.eps_decay
            )
        )

    @property
    def identity(self) -> dict[str, object]:
        """JSON record saved with checkpoints and compared on restore."""
        return {
            "kind": "qmix_epsilon",
            "version": 1,
            "eps_min": self.eps_min,
            "eps_decay": self.eps_decay,
            "num_envs": self.num_envs,
        }


def team_task_reward(task_rewards: Array, active: Array) -> Array:
    """Average per-slot task rewards over configured Team A slots.

    Parameters
    ----------
    task_rewards : Array
        Float32 (...,5) native task reward of each Team A slot.
    active : Array
        Bool (...,5) configured slots.

    Returns
    -------
    Array
        Float32 (...) team task reward: the sum over active slots divided by
        ``max(active count, 1)``. A 1v1 win and a 5v5 win give the same value.
        Rows with no active slot give 0.
    """
    count = jnp.sum(active, axis=-1).astype(jnp.float32)
    total = jnp.sum(jnp.where(active, task_rewards, 0.0), axis=-1)
    return total / jnp.maximum(count, 1.0)


def _initial_q_memory(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    """Create float32 zero memory shaped (B,5,256) from inputs.active_mask.

    variables and keys satisfy the M8 initializer signature and are unused.
    M8 selects the lanes whose episode needs fresh memory.
    """
    del variables, keys
    return jnp.zeros((*inputs.active_mask.shape, QMIX_HIDDEN_SIZE), jnp.float32)


def _explore(greedy: Array, mask: Array, keys: Array, epsilon: Array) -> Array:
    """Draw epsilon-greedy categorical actions for (B,5) actors.

    greedy is int32 (B,5); mask bool (B,5,198); keys one Threefry key per actor,
    typed (B,5) or legacy uint32 (B,5,2); epsilon a float32 0-d rate. Each key
    splits into an exploration key and a uniform-choice key. With probability
    epsilon the actor takes a uniform legal action, otherwise the greedy one.
    This equals the donor's mixture exactly. Pure JAX.
    """
    pairs = jax.vmap(jax.vmap(functools.partial(jax.random.split, num=2)))(keys)
    explore_keys, uniform_keys = pairs[:, :, 0], pairs[:, :, 1]
    draws = jax.vmap(jax.vmap(jax.random.uniform))(explore_keys)
    uniform = sample_actions(jnp.zeros(mask.shape, jnp.float32), mask, uniform_keys)
    return jnp.where(draws < epsilon, uniform, greedy)


def _apply_q_actor(
    variables: QMIXActorVariables,
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
    variables : QMIXActorVariables
        Q-network variables and the float32 0-d exploration rate.
    memory : Array
        Float32 (B,5,256) actor memory before this decision.
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
        Native int32 (B,5) world-frame actions and next memory. There are no
        learning outputs. Invalid lanes keep memory; ignore their actions.

    Notes
    -----
    In the left frame, flagged lanes' views and masks are reflected before the
    network, ties break in that frame, and the chosen index is mapped back to
    world directions. Dead and inactive actors have only the neutral action
    legal, so both the greedy and the uniform choice return it. Pure JAX.
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
    valid = jnp.broadcast_to(inputs.valid[:, None], inputs.active_mask.shape)
    starts = jnp.broadcast_to(inputs.episode_start[:, None], inputs.active_mask.shape)
    memory, values = cast(
        tuple[Array, Array],
        RecurrentQNetwork(input_scale=input_scale).apply(
            variables.params, memory, features[None], starts[None], valid[None]
        ),
    )
    mask = categorical_action_mask(frame_mask)
    actor_keys = jax.vmap(functools.partial(jax.random.split, num=TEAM_SLOTS))(keys)
    indices = _explore(
        greedy_actions(values[0], mask), mask, actor_keys, variables.epsilon
    )
    if flag is not None:
        indices = mirror_action_indices(indices, flag)
    return SystemOutput(decode_actions(indices), memory)


@functools.lru_cache(maxsize=16)
def _q_actor_apply(
    scale: float, spawn_frame: str
) -> Callable[[QMIXActorVariables, Array, SystemInput, Array], SystemOutput]:
    """Return a cached apply hook whose keyword defaults record scale and frame.

    scale and spawn_frame are already checked host values. The hook captures no
    weights or live inputs; M8 digests its numerical defaults into System
    identity, so equal weights with different settings are different Systems.
    """
    frame_index = SPAWN_FRAMES.index(spawn_frame)

    def apply(
        variables: QMIXActorVariables,
        memory: Array,
        inputs: SystemInput,
        keys: Array,
        *,
        input_scale: float = scale,
        spawn_frame_index: int = frame_index,
    ) -> SystemOutput:
        """Apply the QMIX actor with the factory-bound settings.

        Arguments, outputs and effects follow _apply_q_actor. Normal System
        execution supplies only the first four arguments.
        """
        return _apply_q_actor(
            variables,
            memory,
            inputs,
            keys,
            input_scale=input_scale,
            spawn_frame_index=spawn_frame_index,
        )

    return apply


@functools.lru_cache(maxsize=16)
def _schema_1_q_actor_apply(
    scale: float, spawn_frame: str
) -> Callable[[QMIXActorVariables, Array, SystemInput, Array], SystemOutput]:
    """Cache a QMIX hook for Q-networks trained on actor input schema 1.

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
        _apply_q_actor. The encoder uses the 5,164-feature schema-1 layout,
        and the "left" frame uses the old spawn-side formula for both the
        reflected view and the returned actions, so the actor plays exactly
        as before Red Zone. Keyword defaults record scale and frame index as
        in _q_actor_apply; the hook's own name and code give it a distinct
        registration ID.

    Notes
    -----
    Host-only construction; the hook captures no weights or live inputs.
    Caching keeps one callable, and so one compiled program, per setting.
    """
    frame_index = SPAWN_FRAMES.index(spawn_frame)

    def apply(
        variables: QMIXActorVariables,
        memory: Array,
        inputs: SystemInput,
        keys: Array,
        *,
        input_scale: float = scale,
        spawn_frame_index: int = frame_index,
    ) -> SystemOutput:
        """Apply a schema-1 QMIX actor to its historical 19-column view.

        Arguments, outputs and effects follow _apply_q_actor; only the context
        depth column is removed first. Normal System execution supplies only
        the first four arguments.
        """
        return _apply_q_actor(
            variables,
            memory,
            inputs._replace(actors=schema_1_actor_input(inputs.actors)),
            keys,
            input_scale=input_scale,
            spawn_frame_index=spawn_frame_index,
        )

    return apply


@functools.lru_cache(maxsize=32)
def _grouped_qmix_actor_apply(
    scale: float, frame: str, sharing: str
) -> Callable[[QMIXActorVariables, Array, SystemInput, Array], SystemOutput]:
    """Bind the grouped actor's settings while preserving all shared hooks.

    Checked scale/frame/sharing become numeric keyword defaults in the hook's
    recorded identity. Parameters and per-actor memory remain dynamic inputs.
    """

    def apply(
        variables: QMIXActorVariables,
        memory: Array,
        inputs: SystemInput,
        keys: Array,
        *,
        input_scale: float = scale,
        spawn_frame_index: int = SPAWN_FRAMES.index(frame),
        parameter_sharing_index: int = _PARAMETER_SHARING.index(sharing),
    ) -> SystemOutput:
        """Choose grouped epsilon-greedy actions from one same-epoch actor call."""
        features, mask, groups, flag = _grouped_actor_inputs(
            inputs, _PARAMETER_SHARING[parameter_sharing_index], spawn_frame_index
        )
        valid = jnp.broadcast_to(inputs.valid[:, None], inputs.active_mask.shape)
        starts = jnp.broadcast_to(
            inputs.episode_start[:, None], inputs.active_mask.shape
        )
        next_memory, values = cast(
            tuple[Array, Array],
            RecurrentQNetwork(input_scale=input_scale).apply(
                variables.params,
                memory,
                features[None],
                starts[None],
                valid[None],
                actor_group=groups[None],
            ),
        )
        actor_keys = jax.vmap(functools.partial(jax.random.split, num=TEAM_SLOTS))(keys)
        indices = _explore(
            greedy_actions(values[0], mask), mask, actor_keys, variables.epsilon
        )
        if flag is not None:
            indices = mirror_action_indices(indices, flag)
        return SystemOutput(decode_actions(indices), next_memory)

    return apply


def make_qmix_system(
    params: Tree,
    *,
    epsilon: float = 0.0,
    input_scale: float = 1.0,
    spawn_frame: str = "left",
    name: str = "QMIX",
    checkpoint: str | None = None,
    actor_input_schema: int = ACTOR_INPUT_SCHEMA_VERSION,
    parameter_sharing: str = "all",
) -> System:
    """Wrap Q-network variables as an M8 JAX System with epsilon-greedy actions.

    Parameters
    ----------
    params : PyTree
        Q-network Flax variables from initialize_qmix or a later update. The
        factory does not inspect shapes; the network checks them when called.
    epsilon : float, default=0.0
        Exploration probability in [0, 1]. Zero gives greedy play (first legal
        maximum), which loaded and evaluated Systems use. Training replaces it
        before every decision through QMIXExploration.
    input_scale : float, default=1.0
        Positive finite feature multiplier used to train these weights.
    spawn_frame : {"left", "world"}, default="left"
        Frame used to train these weights. It is part of System identity.
    parameter_sharing : {"all", "class", "none"}, default="all"
        Actor weight ownership used during training. Grouped variables carry
        five groups; memory stays per physical actor. Historical schema 1
        supports all only. The mode is part of the System identity.
    name : str, default="QMIX"
        Nonempty display name; it proves nothing about training.
    checkpoint : str or None, default=None
        Optional identity label. No file is opened.
    actor_input_schema : int, default=ACTOR_INPUT_SCHEMA_VERSION
        Actor input schema the weights were trained on: 2, the current
        5,165-feature schema (the default), or 1 for Q-networks trained before
        Red Zone (5,164 features). Schema 1 picks the cached
        _schema_1_q_actor_apply hook once here, so the actor sees and acts
        exactly as before. The factory does not compare the weights' width
        with the schema.

    Returns
    -------
    System
        Stable apply/init hooks with ``QMIXActorVariables(params, epsilon)`` as
        variables and float32 (B,5,256) per-actor memory. Memory resets only at
        episode starts; death and respawn keep it.

    Raises
    ------
    ValueError
        epsilon, input_scale, spawn_frame, actor_input_schema or name is
        invalid.

    Notes
    -----
    Pass changing variables through variables_a/b inside jit. The System holds
    no mixer, target network, optimizer, replay or physical state.
    """
    rate = _unit_interval(epsilon, "epsilon")
    scale = _input_scale(input_scale)
    frame = _spawn_frame(spawn_frame)
    hook = (
        _schema_1_q_actor_apply
        if _actor_input_schema(actor_input_schema) == 1
        else _q_actor_apply
    )
    sharing = _parameter_sharing(parameter_sharing)
    if sharing != "all" and actor_input_schema != ACTOR_INPUT_SCHEMA_VERSION:
        raise ValueError("Grouped actors require the current actor input schema")
    apply = (
        hook(scale, frame)
        if sharing == "all"
        else _grouped_qmix_actor_apply(scale, frame, sharing)
    )
    return System(
        name,
        apply,
        variables=QMIXActorVariables(params, jnp.asarray(rate, jnp.float32)),
        init=_initial_q_memory,
        checkpoint=checkpoint,
    )


def _time_major(value: Array) -> Array:
    """Swap the first two axes: (M,S,...) to (S,M,...) and back."""
    return jnp.swapaxes(value, 0, 1)


def _chosen(values: Array, actions: Array) -> Array:
    """Gather (...,198) action values at int32 (...) indices; return (...)."""
    return jnp.take_along_axis(values, actions[..., None], axis=-1)[..., 0]


def _finite(tree: Tree) -> Array:
    """Return one bool: True when every inexact leaf of the tree is finite."""
    checks = [
        jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    return jnp.all(jnp.stack(checks)) if checks else jnp.bool_(True)


def _check_batch(batch: QMIXBatch) -> tuple[int, int]:
    """Check QMIXBatch leaf shapes and dtypes statically; return (M, S).

    Raises ValueError for a mismatch. Feature widths may differ from the real
    constants so reference tests can use small widths.
    """
    if batch.valid.ndim != 2 or batch.valid.shape[1] < 2:
        raise ValueError("QMIXBatch needs (M,S) rows with S >= 2")
    m, s = batch.valid.shape
    expected = {
        "action_mask": ((m, s, TEAM_SLOTS, NUM_ACTIONS), jnp.bool_),
        "actions": ((m, s, TEAM_SLOTS), jnp.int32),
        "rewards": ((m, s), jnp.float32),
        "episode_start": ((m, s), jnp.bool_),
        "ended": ((m, s), jnp.bool_),
        "valid": ((m, s), jnp.bool_),
        "active": ((m, s, TEAM_SLOTS), jnp.bool_),
    }
    for field, (shape, dtype) in expected.items():
        value = getattr(batch, field)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"QMIXBatch.{field} must have shape {shape} and {dtype}")
    for field in ("actor_features", "training_state"):
        value = getattr(batch, field)
        leading = (m, s, TEAM_SLOTS) if field == "actor_features" else (m, s)
        if value.shape[:-1] != leading or value.dtype != jnp.float32:
            raise ValueError(f"QMIXBatch.{field} has the wrong shape or dtype")
    for field, dtype in (("actor_group", jnp.int32), ("learner_active", jnp.bool_)):
        value = getattr(batch, field)
        if value is not None and (
            value.shape != batch.active.shape or value.dtype != dtype
        ):
            raise ValueError(f"QMIXBatch.{field} has the wrong shape or dtype")
    return m, s


def update_qmix(
    train_state: QMIXTrainState,
    batch: QMIXBatch,
    *,
    config: QMIXConfig = DEFAULT_QMIX_CONFIG,
) -> tuple[QMIXTrainState, QMIXMetrics]:
    """Compute one Double-Q QMIX optimizer step on one expanded sample.

    Parameters
    ----------
    train_state : QMIXTrainState
        Networks, optimizer and step count before this step.
    batch : QMIXBatch
        One sample of M sequences, S rows each, in the network frame.
    config : QMIXConfig, default=DEFAULT_QMIX_CONFIG
        Static settings: gamma, q_lr, the target rule and input_scale.

    Returns
    -------
    tuple[QMIXTrainState, QMIXMetrics]
        Candidate state and this step's metrics. With no eligible pair the
        state is returned unchanged (no optimizer or target movement). The
        caller decides whether to accept a candidate; check ``metrics.finite``.

    Raises
    ------
    ValueError
        Batch shapes or dtypes are wrong.

    Notes
    -----
    Invalid rows are replaced by finite neutral values before any network or
    arithmetic operation, and they do not move recurrent memory. Online and
    target Q-networks each unroll all S rows from zero memory, honoring episode
    starts inside the sample. The online greedy choice (first legal maximum)
    selects each next action, the target Q-network values it, and the target
    mixer gives the next team value. The target is
    ``reward + (1 - ended) * gamma * next_value`` and receives no gradient. The
    loss averages the squared team error over eligible pairs only. With
    learner_active, only owned slots supply current and successor utilities.
    Shared Adam steps once over online Q and mixer. Class and slot modes clip
    and update each actor group and the mixer separately, using the same joint
    loss. A group with no eligible left-row utility keeps its parameters,
    complete optimizer state and target exactly. A hard copy uses the run-wide
    step count before
    the step (copies at 0, 200, 400, ...); the soft rule blends with tau after
    every step. Pure JAX; wrap in jit with a fixed config.
    """
    m, _ = _check_batch(batch)
    valid = batch.valid
    owned = (
        batch.active
        if batch.learner_active is None
        else batch.active & batch.learner_active
    )
    pair = valid[:, :-1] & valid[:, 1:]
    if batch.learner_active is not None:
        pair = pair & jnp.any(owned[:, :-1], axis=-1)
    groups = _batch_actor_groups(
        config.parameter_sharing, batch.actor_group, batch.active.shape
    )
    neutral = jnp.arange(NUM_ACTIONS) == 0
    features = (
        jnp.where((valid[..., None] & owned)[..., None], batch.actor_features, 0.0)
        if batch.learner_active is not None
        else jnp.where(valid[..., None, None], batch.actor_features, 0.0)
    )
    mask = jnp.where(valid[..., None, None], batch.action_mask, neutral)
    actions = jnp.where(valid[..., None], batch.actions, 0)
    rewards = jnp.where(valid, batch.rewards, 0.0)
    ended = valid & batch.ended
    starts = valid & batch.episode_start
    active = valid[..., None] & owned
    state = jnp.where(valid[..., None], batch.training_state, 0.0)
    resets = jnp.broadcast_to(starts[..., None], active.shape)
    live = jnp.broadcast_to(valid[..., None], active.shape)
    carry = jnp.zeros((m, TEAM_SLOTS, QMIX_HIDDEN_SIZE), jnp.float32)
    network = RecurrentQNetwork(input_scale=config.input_scale)
    mixer = QMixingNetwork()

    def unroll(params: Tree) -> Array:
        """Return (M,S,5,198) values from zero memory over the sample rows."""
        _, values = cast(
            tuple[Array, Array],
            network.apply(
                params,
                carry,
                _time_major(features),
                _time_major(resets),
                _time_major(live),
                actor_group=None if groups is None else _time_major(groups),
            ),
        )
        return _time_major(values)

    target_values = unroll(train_state.target_q)
    count = jnp.sum(pair, dtype=jnp.int32)
    denominator = jnp.maximum(count, 1).astype(jnp.float32)

    def loss_fn(online: tuple[Tree, Tree]) -> tuple[Array, tuple[Array, Array]]:
        """Return the masked TD loss and (mean chosen team value, mean target)."""
        q_params, mixer_params = online
        values = unroll(q_params)
        next_actions = greedy_actions(jax.lax.stop_gradient(values[:, 1:]), mask[:, 1:])
        next_values = jnp.where(
            active[:, 1:], _chosen(target_values[:, 1:], next_actions), 0.0
        )
        next_team = cast(
            Array,
            mixer.apply(train_state.target_mixer, next_values, state[:, 1:]),
        )
        target = rewards[:, :-1] + (1.0 - ended[:, :-1]) * config.gamma * next_team
        target = jax.lax.stop_gradient(jnp.where(pair, target, 0.0))
        chosen = jnp.where(
            active[:, :-1], _chosen(values[:, :-1], actions[:, :-1]), 0.0
        )
        team = cast(Array, mixer.apply(mixer_params, chosen, state[:, :-1]))
        error = jnp.where(pair, team - target, 0.0)
        loss = jnp.sum(jnp.square(error)) / denominator
        mean_q = jnp.sum(jnp.where(pair, team, 0.0)) / denominator
        return loss, (mean_q, jnp.sum(target) / denominator)

    (loss, (mean_q, mean_target)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        (train_state.online_q, train_state.online_mixer)
    )
    optimizer = _optimizer(config)
    present = (
        None
        if groups is None
        else _actor_group_presence(groups[:, :-1], active[:, :-1] & pair[..., None])
    )

    def step(_: None) -> QMIXTrainState:
        """Apply Adam, then the target rule using the pre-step count."""
        if present is None:
            updates, opt_state = optimizer.update(grads, train_state.opt_state)
            online_q, online_mixer = cast(
                tuple[Tree, Tree],
                optax.apply_updates(
                    (train_state.online_q, train_state.online_mixer), updates
                ),
            )
        else:
            online_q, q_state = _update_actor_groups(
                train_state.online_q,
                train_state.opt_state[0],
                grads[0],
                present,
                optimizer,
            )
            mixer_delta, mixer_state = optimizer.update(
                grads[1], train_state.opt_state[1], train_state.online_mixer
            )
            online_mixer = optax.apply_updates(train_state.online_mixer, mixer_delta)
            opt_state = (q_state, mixer_state)
        before = train_state.optimizer_steps
        if config.hard_update:
            target_q = optax.periodic_update(
                online_q, train_state.target_q, before, config.update_period
            )
            target_mixer = optax.periodic_update(
                online_mixer, train_state.target_mixer, before, config.update_period
            )
        else:
            target_q = optax.incremental_update(
                online_q, train_state.target_q, config.tau
            )
            target_mixer = optax.incremental_update(
                online_mixer, train_state.target_mixer, config.tau
            )
        if present is not None:
            target_present = present

            def keep_absent(new: Array, old: Array) -> Array:
                """Keep an absent actor group's target on the shared target clock."""
                return jnp.where(
                    target_present.reshape((5,) + (1,) * (new.ndim - 1)), new, old
                )

            target_q = jax.tree.map(keep_absent, target_q, train_state.target_q)
        return QMIXTrainState(
            online_q, target_q, online_mixer, target_mixer, opt_state, before + 1
        )

    def keep(_: None) -> QMIXTrainState:
        """Return the unchanged state when no pair is eligible."""
        return train_state

    candidate = cast(QMIXTrainState, jax.lax.cond(count > 0, step, keep, None))
    metrics = QMIXMetrics(
        loss,
        mean_q,
        mean_target,
        jnp.sum(jnp.any(pair, axis=1), dtype=jnp.int32),
        count,
        jnp.sum(active[:, :-1] & pair[..., None], dtype=jnp.int32),
        _finite((loss, mean_q, mean_target, grads, candidate)),
    )
    return candidate, metrics
