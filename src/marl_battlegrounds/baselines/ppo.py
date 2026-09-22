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
"""Provide recurrent MAPPO numerical components and an actor-only M8 System.

This module adapts Mava's networks, GAE and grouped PPO updates. It owns no
environment loop, checkpoint writer, curriculum or training run. initialize_ppo
creates untrained actor/critic parameters and separate optimizer states.
make_recurrent_mappo_system exposes only actor parameters to M8. Learners call
critic_values and update_recurrent_ppo separately with privileged training data.

Requires the optional training extra. Inputs stay on device inside jit/scan.
Reference settings are starting values, not qualified learning settings for BG.
PPOConfig.spawn_frame is a BG adaptation: "left" or "right" reflects the actor's
permitted view so every game looks like a start from that bank, and maps the
chosen move back; "world" keeps the donor's raw coordinates bit for bit.
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
        "PPO baselines need the training extra. Install marl-battlegrounds[training]."
    ) from error

from marl_battlegrounds.baselines.actions import (
    NUM_ACTIONS,
    action_entropy,
    action_log_prob,
    categorical_action_mask,
    decode_actions,
    mirror_action_indices,
    sample_actions,
)
from marl_battlegrounds.baselines.inputs import (
    ACTOR_FEATURE_SIZE,
    SPAWN_FRAMES,
    TRAINING_STATE_FEATURE_SIZE,
    encode_actor_inputs,
    spawn_frame_flag,
    team_obstacle_partners,
)
from marl_battlegrounds.core.types import ActionMask
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    SystemInput,
    SystemOutput,
)
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    build_team_actor_input,
    mirror_team_view,
)

type Tree = Any
HIDDEN_SIZE = 128


def _input_scale(value: float) -> float:
    """Check a fixed positive finite real input scale and return a Python float.

    This host check accepts real scalar numbers, excluding Booleans. Arrays,
    traced values, strings and nonpositive/nonfinite values raise ValueError.
    It performs no device work and stores no normalization statistics.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("input_scale must be a finite positive real number.")
    return float(value)


def _spawn_frame(value: object) -> str:
    """Check a static spawn frame name and return it as a Python string.

    This host check accepts exactly the strings "world", "left" and "right".
    Any other value, including Booleans, None and traced arrays, raises
    ValueError. It performs no device work.
    """
    if not isinstance(value, str) or value not in SPAWN_FRAMES:
        raise ValueError('spawn_frame must be "world", "left" or "right".')
    return value


@dataclass(frozen=True)
class PPOConfig:
    """Hold static donor update settings, with separate actor/critic optimizers.

    Parameters
    ----------
    actor_lr, critic_lr : float, default=0.00025
        Positive constant learning rates for the separate Adam optimizers.
    rollout_length : int, default=128
        Decisions in a complete recurrent sequence. Shorter chunks are unsupported.
    epochs : int, default=4
        How many times each update uses the supplied rollout.
    minibatches : int, default=2
        Number of game-sequence subsets in each group and epoch.
    groups : int, default=2
        Number of groups whose gradients are averaged for one learner. These
        are not independent runs. Game count must be divisible by
        groups*minibatches.
    gamma : float, default=0.99
        Reward discount per transition, from zero through one.
    gae_lambda : float, default=0.95
        Advantage mixing weight, from zero through one.
    clip_epsilon : float, default=0.2
        Positive limit for policy-ratio changes and changes in predicted value.
    entropy_coefficient : float, default=0.01
        Nonnegative weight of policy entropy in the actor loss.
    value_coefficient : float, default=0.5
        Nonnegative weight of the donor value loss. That loss already includes
        its separate factor of 0.5 for squared errors.
    max_grad_norm : float, default=0.5
        Positive gradient-length limit, applied after group averaging and before
        each network's Adam update.
    adam_epsilon : float, default=0.00001
        Positive offset in Adam's denominator.
    input_scale : float, default=1.0
        Positive finite multiplier applied to every actor and critic feature
        before the first Dense layer. One preserves the donor's raw inputs.
        Use the same setting for collection, learning and loaded inference.
        No feature is removed and no running statistics are collected.
    spawn_frame : str, default="world"
        Which spawn bank the actor always seems to start from. "world" keeps
        raw coordinates and traces the donor's program bit for bit. "left"
        reflects the permitted view of any actor whose own team starts on the
        right bank about the map's vertical centerline, so every game looks
        like a left start, and maps the chosen move back before the game
        receives it; "right" does the same with the right bank as home. On
        the built-in Team Deathmatch maps, which are their own mirror images,
        a reflected right start shows the own pads near x = 0.5; a custom
        layout is reflected the same way with no such guarantee, and
        coordinates keep float32 rounding. Use the same setting for
        collection, learning and loaded inference; it is saved with exported
        actors and is part of their inference identity.

    Raises
    ------
    ValueError
        A count is not a positive Python int, or a numerical setting is nonfinite
        or outside its stated range. Boolean counts and non-real/Boolean input
        scales are rejected, as is a spawn frame other than "world", "left" or
        "right".

    Notes
    -----
    Values are immutable host setup settings. Capture one fixed instance or mark
    it static in jit. New settings may require a new compiled program. This
    record changes neither network width nor the environment's rules.
    """

    actor_lr: float = 0.00025
    critic_lr: float = 0.00025
    rollout_length: int = 128
    epochs: int = 4
    minibatches: int = 2
    groups: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    adam_epsilon: float = 0.00001
    input_scale: float = 1.0
    spawn_frame: str = "world"

    def __post_init__(self) -> None:
        """Reject invalid static update counts and numerical settings on the host."""
        _input_scale(self.input_scale)
        _spawn_frame(self.spawn_frame)
        for name in ("rollout_length", "epochs", "minibatches", "groups"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "actor_lr",
            "critic_lr",
            "clip_epsilon",
            "max_grad_norm",
            "adam_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in ("entropy_coefficient", "value_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")


DEFAULT_PPO_CONFIG = PPOConfig()


class PPOTrainState(NamedTuple):
    """Carry one learner's separate network parameters and Adam states.

    Attributes
    ----------
    actor_params, critic_params : PyTree
        Separate Flax variable trees, including their params collection.
    actor_opt_state, critic_opt_state : PyTree
        Separate Optax states for their matching network parameters.

    Notes
    -----
    Arrays are dynamic JAX leaves; groups never duplicate these states. Recurrent
    memory, random keys and environment state belong to the caller. This is a
    numerical update record, not a complete resumable learner checkpoint.
    """

    actor_params: Tree
    critic_params: Tree
    actor_opt_state: Tree
    critic_opt_state: Tree


class PPOLearningOutputs(NamedTuple):
    """Return actor action-time values without another network call.

    Attributes
    ----------
    action_indices : Array
        Int32 (B,5) submitted choices in the versioned 198-choice action space,
        always in world directions, whatever spawn frame the actor used.
    log_prob : Array
        Float32 (B,5) natural-log probability of each submitted choice under its
        action-time mask and actor weights, computed in the actor's frame.

    Notes
    -----
    B is the number of games. Ignore invalid lanes and exclude dead/inactive
    actors from policy learning. Neither field contains critic information or
    recurrent memory. No later policy call is used to reconstruct these values.
    """

    action_indices: Array
    log_prob: Array


class PPOBatch(NamedTuple):
    """Describe one compact Team A rollout for a recurrent PPO update.

    Attributes
    ----------
    observations : Observations
        Compact observations with leading (T,B), ten base rows and permissions.
        T is time and B is the number of games. Actor expansion stays temporary.
    training_state : Array
        Float32 (T,B,919) versioned physical-state features, stored once per game.
        This input is available to the critic only.
    action_mask : ActionMask
        Team A's action-time native masks, with leading (T,B,5).
    actions : Array
        Int32 (T,B,5) submitted categorical indices, from zero through 197.
    old_log_prob : Array
        Float32 (T,B,5) natural-log action probabilities from collection.
    old_values : Array
        Float32 (T,B,5) critic predictions made before the corresponding actions.
    rewards : Array
        Float32 (T,B,5) training rewards for the produced transitions.
    ended : Array
        Boolean (T,B) endings for the produced transitions, including horizon
        draws. An ordinary collection cutoff is not an ending.
    episode_start : Array
        Boolean (T,B) first-decision markers before each action. Death is not
        an episode start.
    valid : Array
        Boolean (T,B) real transitions, including terminal transitions and
        excluding padding.
    active, alive : Array
        Boolean (T,B,5) membership and life status before the action. Death keeps
        active True; inactive slots belong to neither loss.
    final_values : Array
        Float32 (B,5) values at the true last successor, before any automatic
        reset. GAE ignores these values when that transition ended the task.
    actor_memory, critic_memory : Array
        Separate float32 (B,5,128) carries just before the first decision.

    Notes
    -----
    All records must be paired to their actual decision epoch. Values, actions
    and masks must be finite/valid under their owning contracts. No numerical
    host validation is added to the compiled update. This record owns no RNG,
    output files, persistent expanded actor inputs or opponent behavior data.
    """

    observations: Observations
    training_state: Array
    action_mask: ActionMask
    actions: Array
    old_log_prob: Array
    old_values: Array
    rewards: Array
    ended: Array
    episode_start: Array
    valid: Array
    active: Array
    alive: Array
    final_values: Array
    actor_memory: Array
    critic_memory: Array


class PPOMinibatch(NamedTuple):
    """Hold temporary encoded groups for one shared optimizer application.

    Attributes
    ----------
    actor_features, critic_features : Array
        Separate float32 (G,T,E,5,F) inputs with their respective feature widths.
        G is gradient groups, T is time and E is selected games per group.
    action_mask : Array
        Boolean (G,T,E,5,198) exact combined action legality, in the actor's
        spawn frame.
    actions : Array
        Int32 (G,T,E,5) submitted categorical choices, in the actor's spawn
        frame (world directions unless the frame reflected them).
    old_log_prob, old_values : Array
        Float32 (G,T,E,5) action-time log probabilities and value predictions.
    advantages, targets : Array
        Float32 (G,T,E,5) fixed advantage and value targets for this update.
    episode_start, valid : Array
        Boolean (G,T,E,5) decision-epoch reset and real-transition masks.
    actor_samples, critic_samples : Array
        Boolean (G,T,E,5) loss eligibility. Each True entry must also be valid.
    actor_memory, critic_memory : Array
        Separate float32 (G,E,5,128) carries immediately before each sequence.

    Notes
    -----
    Expanded inputs are temporary and are not the stored rollout format.
    This lower-level numerical boundary also supports fixed-input comparisons.
    It performs no environment stepping or privilege routing: the caller must
    provide actor features produced solely from permitted actor inputs.
    """

    actor_features: Array
    critic_features: Array
    action_mask: Array
    actions: Array
    old_log_prob: Array
    old_values: Array
    advantages: Array
    targets: Array
    episode_start: Array
    valid: Array
    actor_samples: Array
    critic_samples: Array
    actor_memory: Array
    critic_memory: Array


class PPOMetrics(NamedTuple):
    """Report one update's losses and eligible sample counts.

    Attributes
    ----------
    actor_loss : Array
        Float32 actor loss, including the configured entropy term.
    entropy : Array
        Float32 masked policy entropy in natural-log units, before its weight.
    value_loss : Array
        Float32 donor half-squared-error loss, before value_coefficient.
    actor_samples, critic_samples : Array
        Integer counts of eligible samples across all groups.

    Notes
    -----
    update_minibatch returns scalar fields, averaging nonempty groups equally.
    update_recurrent_ppo stacks fields as (epochs,minibatches). Summing sample
    counts across this result counts repeated learner use, not new experience.
    Zero eligible samples yield zero corresponding metrics and no optimizer step.
    """

    actor_loss: Array
    entropy: Array
    value_loss: Array
    actor_samples: Array
    critic_samples: Array


class MLPTorso(nn.Module):
    """Apply the donor's 128-wide orthogonal Dense layer and ReLU.

    Input float32 (...,F) becomes (...,128). The torso gain is sqrt(2), biases
    start at zero and no layer normalization is added. Parameters are separate
    for each owning actor/critic torso.
    """

    @nn.compact
    def __call__(self, features: Array) -> Array:
        """Map each feature row to 128 values without mixing rows.

        Parameters
        ----------
        features : Array
            Float32 (...,F) inputs. F must match the initialized weight width.

        Returns
        -------
        Array
            Float32 (...,128) output after the dense layer and ReLU.
        """
        return nn.relu(
            nn.Dense(HIDDEN_SIZE, kernel_init=nn.initializers.orthogonal(2**0.5))(
                features
            )
        )


class ScannedRNN(nn.Module):
    """Carry independent actor rows through time using the donor GRU defaults.

    Parameters are broadcast across time. Each input row is (embedding, reset,
    valid), shaped (E,5,128), (E,5) and (E,5). Reset happens before the cell;
    invalid rows preserve memory. The scan adds leading T and splits no parameter
    RNG across time. Fresh carries are zero.
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
        """Apply one GRU step, with resets before the cell and preserved padding.

        Parameters
        ----------
        carry : Array
            Float32 (E,5,128) memory immediately before this decision.
        inputs : tuple[Array, Array, Array]
            Embeddings float32 (E,5,128), episode-reset flags bool (E,5), and
            valid-decision flags bool (E,5), in that order. The scan wrapper
            supplies these rows from time-first arrays.

        Returns
        -------
        tuple[Array, Array]
            Next memory and cell output, both float32 (E,5,128). Invalid rows
            retain their old memory and emit zero cell output. Valid reset
            rows start from zero memory before processing their embedding.
        """
        features, resets, valid = inputs
        before = jnp.where((resets & valid)[..., None], 0.0, carry)
        # Mava's carry helper creates another cell first. Keep its resulting
        # GRUCell_1 parameter name so matched keys preserve initialization.
        updated, output = cast(
            tuple[Array, Array],
            nn.GRUCell(features=HIDDEN_SIZE, name="GRUCell_1")(before, features),
        )
        return jnp.where(valid[..., None], updated, carry), jnp.where(
            valid[..., None], output, 0.0
        )


class DiscreteActionHead(nn.Module):
    """Map donor embeddings to 198 raw logits with orthogonal gain 0.01."""

    @nn.compact
    def __call__(self, features: Array) -> Array:
        """Produce one raw score per categorical action.

        Parameters
        ----------
        features : Array
            Float32 (...,128) post-GRU embeddings.

        Returns
        -------
        Array
            Float32 (...,198) logits. These are scores, not probabilities;
            the separate action helpers apply exact legality and sampling.
        """
        return nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.orthogonal(0.01))(
            features
        )


class RecurrentActor(nn.Module):
    """Apply the donor actor with separate per-actor memory and raw logits.

    Features are permitted float32 (T,E,5,F); carry is (E,5,128). Reset/valid
    masks are bool (T,E,5). The ordinary F is 5164; fixed-input reference tests
    can initialize the same network with another F. Parameters encode that F.
    No critic state or other actor's private row enters a row's calculation.

    input_scale is a fixed positive finite real multiplier, default 1.0, applied
    to every feature before the first Dense layer. It changes neither parameter
    shapes nor raw input schemas. Pass the training setting when loading weights.
    """

    input_scale: float = 1.0

    @nn.compact
    def __call__(
        self, carry: Array, features: Array, resets: Array, valid: Array
    ) -> tuple[Array, Array]:
        """Read permitted feature sequences with independent actor memory.

        Parameters
        ----------
        carry : Array
            Float32 (E,5,128) memory immediately before the sequence.
        features : Array
            Float32 (T,E,5,F) encoded actor inputs. F must match the initialized
            width, normally ACTOR_FEATURE_SIZE. Rows may not pool private inputs.
        resets : Array
            Boolean (T,E,5) true only when a new episode starts before a decision.
        valid : Array
            Boolean (T,E,5) valid decisions. Invalid rows retain prior memory.

        Returns
        -------
        tuple[Array, Array]
            Final float32 (E,5,128) memory and float32 (T,E,5,198) raw logits.
            Ignore logits for invalid rows. No action is sampled here.

        Notes
        -----
        Shapes, dtypes and information rights are caller preconditions. This
        Flax module works inside jit; its parameters are shared across time and
        actors. It reads no critic state and writes no external state.
        An invalid input_scale raises ValueError before numerical application.
        """
        scale = _input_scale(self.input_scale)
        if scale != 1.0:
            features = features * scale
        embedding = MLPTorso(name="pre_torso")(features)
        carry, embedding = ScannedRNN()(carry, (embedding, resets, valid))
        embedding = MLPTorso(name="post_torso")(embedding)
        return carry, DiscreteActionHead(name="action_head")(embedding)


class RecurrentValueNet(nn.Module):
    """Apply the separate donor critic to training-only features.

    Carry and masks follow RecurrentActor. Features use (T,E,5,F), normally a
    temporary broadcast of one 919-value physical view per game. Output values
    have shape (T,E,5). This module is never part of the actor-producing System.
    input_scale is a fixed positive finite real multiplier, default 1.0, applied
    before the first Dense layer. Use the same setting as the paired actor.
    """

    input_scale: float = 1.0

    @nn.compact
    def __call__(
        self, carry: Array, features: Array, resets: Array, valid: Array
    ) -> tuple[Array, Array]:
        """Read a training-only feature sequence with separate critic memory.

        Parameters
        ----------
        carry : Array
            Float32 (E,5,128) critic memory before the sequence.
        features : Array
            Float32 (T,E,5,F) physical-state features, normally F=919. The width
            must match the initialized critic. Actor code may not read this data.
        resets, valid : Array
            Boolean (T,E,5) episode-start and valid-decision flags. Resets occur
            before the GRU. Invalid rows retain their prior memory.

        Returns
        -------
        tuple[Array, Array]
            Final float32 (E,5,128) critic memory and float32 (T,E,5) predicted
            returns in the training reward's units. Ignore invalid-row values.

        Notes
        -----
        Shapes, dtypes and matching decision epochs are caller preconditions.
        The pure Flax calculation supports jit and performs no action selection.
        An invalid input_scale raises ValueError before numerical application.
        """
        scale = _input_scale(self.input_scale)
        if scale != 1.0:
            features = features * scale
        embedding = MLPTorso(name="pre_torso")(features)
        carry, embedding = ScannedRNN()(carry, (embedding, resets, valid))
        embedding = MLPTorso(name="post_torso")(embedding)
        values = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(embedding)
        return carry, values[..., 0]


def _optimizers(
    config: PPOConfig,
) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Create actor and critic clip-then-Adam transforms from static config.

    Return two Optax transformations in actor, critic order. Each uses its
    configured constant rate, gradient-norm limit and Adam epsilon. This creates
    no optimizer state and changes no parameters; callers initialize each state.
    """

    def make(rate: float) -> optax.GradientTransformation:
        """Use global norm clipping before constant-rate Adam for one network."""
        return optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(rate, eps=config.adam_epsilon),
        )

    return make(config.actor_lr), make(config.critic_lr)


def initialize_ppo(key: Array, config: PPOConfig = DEFAULT_PPO_CONFIG) -> PPOTrainState:
    """Initialize untrained BG-width actor/critic parameters and optimizers.

    Parameters
    ----------
    key : Array
        One fresh typed scalar Threefry key or legacy uint32 (2,) key. Legacy
        keys use JAX's configured default implementation, which must be Threefry.
    config : PPOConfig, default=DEFAULT_PPO_CONFIG
        Static optimizer and input-scale settings. The default uses the pinned
        donor settings. Scaling preserves the initialized parameter tree/bytes.

    Returns
    -------
    PPOTrainState
        Independent actor/critic variable trees and separate initialized Adam
        states. Widths match the versioned actor and physical-state encoders.

    Raises
    ------
    ValueError
        The key does not use Threefry.

    Notes
    -----
    This performs network initialization, including device work, but collects no
    experience, changes no supplied input and creates no files. The caller owns
    future random keys. Actor episode memory is initialized by the System; critic
    episode memory remains the learner's responsibility. Weights are untrained.
    """
    if str(jax.random.key_impl(key)) != "threefry2x32":
        raise ValueError("PPO initialization requires Threefry random keys.")
    actor_key, critic_key = jax.random.split(key)
    memory = jnp.zeros((1, 5, HIDDEN_SIZE), jnp.float32)
    starts = jnp.zeros((1, 1, 5), jnp.bool_)
    valid = jnp.ones_like(starts)
    actor = RecurrentActor(input_scale=config.input_scale).init(
        actor_key, memory, jnp.zeros((1, 1, 5, ACTOR_FEATURE_SIZE)), starts, valid
    )
    critic = RecurrentValueNet(input_scale=config.input_scale).init(
        critic_key,
        memory,
        jnp.zeros((1, 1, 5, TRAINING_STATE_FEATURE_SIZE)),
        starts,
        valid,
    )
    actor_opt, critic_opt = _optimizers(config)
    return PPOTrainState(actor, critic, actor_opt.init(actor), critic_opt.init(critic))


def _initial_actor_memory(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    """Create float32 zero memory with shape (B,5,128) from inputs.active_mask.

    variables and keys satisfy the M8 initializer signature and are unused.
    Every lane gets a fresh template; M8 selects the lanes requiring a reset.
    No actor call, action sample, host transfer or input change occurs.
    """
    del variables, keys
    return jnp.zeros((*inputs.active_mask.shape, HIDDEN_SIZE), jnp.float32)


def _apply_actor(
    variables: Tree,
    memory: Array,
    inputs: SystemInput,
    keys: Array,
    *,
    input_scale: float = 1.0,
    spawn_frame_index: int = 0,
) -> SystemOutput:
    """Apply the actor once using M8's paired current inputs and lane keys.

    Parameters
    ----------
    variables : PyTree
        Actor-only Flax variables; no critic parameters or optimizer state.
    memory : Array
        Float32 (B,5,128) actor memory before this decision.
    inputs : SystemInput
        Same-epoch permitted views, native action masks and lifecycle flags.
        Each actor keeps its separate information rights.
    keys : Array
        B independent Threefry lane keys, typed (B,) or legacy uint32 (B,2).
        Five actor streams are derived within each lane.
    input_scale : float, default=1.0
        Fixed positive finite feature multiplier from the actor's training
        settings. One keeps raw inputs. The System factory binds this setting.
    spawn_frame_index : int, default=0
        Position of the actor's training spawn frame in SPAWN_FRAMES: 0 is
        "world", 1 is "left", 2 is "right". A number rather than the name,
        because M8 records a System hook's numerical keyword defaults in its
        identity and skips text. "left" or "right" reflects flagged lanes'
        permitted view and move mask before the network, samples in that
        frame, and maps the chosen index back to world directions for both
        the returned actions and the learning outputs. "world" leaves the
        donor's program unchanged. The System factory binds this setting.

    Returns
    -------
    SystemOutput
        Native int32 (B,5) actions, next actor memory and PPOLearningOutputs
        from this same actor call. Invalid lanes retain memory; ignore their
        action and learning rows. No critic value or memory is returned.

    Raises
    ------
    TypeError, ValueError
        The input/action helper finds incompatible shapes or dtypes, or sampling
        keys do not use Threefry. Finite logits and nonempty legal masks remain
        caller preconditions; this function does not repair invalid inputs.

    Notes
    -----
    Pure device work supports jit/scan. Source rows and encoded features are
    temporary. M8 owns episode binding and selected memory resets. No input is
    changed, no extra actor call is made and no data is saved. In a reflected
    frame the obstacle partner decision is made once per lane from slot 0's
    table (team_obstacle_partners) and shared by the five actors; it is
    recomputed on every call, never cached.
    """
    actors, frame_mask = inputs.actors, inputs.action_mask
    flag: Array | None = None
    spawn_frame = SPAWN_FRAMES[spawn_frame_index]
    if spawn_frame != "world":
        flag = spawn_frame_flag(actors, spawn_frame)
        # The five actors of a lane share one map table: match obstacles once
        # per game rather than once per actor.
        actors, frame_mask = mirror_team_view(
            actors,
            frame_mask,
            flag,
            obstacle_partners=team_obstacle_partners(actors),
        )
    features = encode_actor_inputs(actors)
    valid = jnp.broadcast_to(inputs.valid[:, None], inputs.active_mask.shape)
    starts = jnp.broadcast_to(inputs.episode_start[:, None], inputs.active_mask.shape)
    memory, logits = cast(
        tuple[Array, Array],
        RecurrentActor(input_scale=input_scale).apply(
            variables, memory, features[None], starts[None], valid[None]
        ),
    )
    logits = logits[0]
    mask = categorical_action_mask(frame_mask)
    actor_keys = jax.vmap(functools.partial(jax.random.split, num=5))(keys)
    indices = sample_actions(logits, mask, actor_keys)
    if flag is None:
        return SystemOutput(
            decode_actions(indices),
            memory,
            learning_outputs=PPOLearningOutputs(
                indices, action_log_prob(logits, mask, indices)
            ),
        )
    # The probability belongs to the reflected frame the actor sampled in; the
    # index is mapped back so the game and the stored rows see world directions.
    log_prob = action_log_prob(logits, mask, indices)
    world = mirror_action_indices(indices, flag)
    return SystemOutput(
        decode_actions(world),
        memory,
        learning_outputs=PPOLearningOutputs(world, log_prob),
    )


@functools.lru_cache(maxsize=16)
def _scaled_actor_apply(
    scale: float, spawn_frame: str
) -> Callable[[Tree, Array, SystemInput, Array], SystemOutput]:
    """Reuse an apply hook whose keyword defaults identify its scale and frame.

    scale is an already checked Python float and spawn_frame an already checked
    frame name; both are required so one setting has one cache key. The
    returned hook uses the normal actor call and has no captured weights or
    mutable state. M8 digests a hook's numerical keyword defaults into System
    identity, so the frame is recorded as its index in SPAWN_FRAMES (0 world,
    1 left, 2 right), exactly as _apply_actor records it for the default
    System; equal weights used with different scales or frames therefore
    describe different Systems. Caching preserves callable identity during
    setup.
    """
    frame_index = SPAWN_FRAMES.index(spawn_frame)

    def apply(
        variables: Tree,
        memory: Array,
        inputs: SystemInput,
        keys: Array,
        *,
        input_scale: float = scale,
        spawn_frame_index: int = frame_index,
    ) -> SystemOutput:
        """Apply the actor using the factory-bound settings in the keyword defaults.

        Arguments, outputs and side effects follow _apply_actor. The optional
        input_scale and spawn_frame_index defaults record this hook's fixed
        inference settings; normal System execution supplies only variables,
        memory, inputs and keys.
        """
        return _apply_actor(
            variables,
            memory,
            inputs,
            keys,
            input_scale=input_scale,
            spawn_frame_index=spawn_frame_index,
        )

    return apply


def make_recurrent_mappo_system(
    actor_params: Tree,
    *,
    input_scale: float = 1.0,
    spawn_frame: str = "world",
    name: str = "Recurrent MAPPO",
    checkpoint: str | None = None,
) -> System:
    """Wrap actor parameters as an existing M8 JAX System, with sampled actions.

    Parameters
    ----------
    actor_params : PyTree
        Actor variable tree from initialize_ppo or a compatible update. This
        factory does not inspect array shapes; the actor checks them when called.
    input_scale : float, default=1.0
        Positive finite feature multiplier used to train these weights. One
        preserves historical raw-input behavior. The fixed apply hook records
        this setting in System identity; equal weights with different scales
        describe different inference. No weights or encoder fields are changed.
    spawn_frame : str, default="world"
        Spawn frame used to train these weights: "world", "left" or "right"
        (see PPOConfig). The apply hook records it in System identity; equal
        weights with different frames describe different inference.
    name : str, default="Recurrent MAPPO"
        Nonempty display name. It does not establish a trained model's identity.
    checkpoint : str or None, default=None
        Optional identity text attached to the System. No file is opened or
        loaded, and this text alone does not verify an artifact's contents.

    Returns
    -------
    System
        Existing M8 JAX System with stable actor/init callables and the supplied
        weights. Arrays are retained by reference. No action is chosen here.

    Raises
    ------
    ValueError
        The scale is not a positive finite real number, is Boolean, the spawn
        frame is not one of the three names, or the System rejects an empty or
        invalid display name.

    Notes
    -----
    Pass changing weights through variables_a/b to apply_systems inside jit.
    M8 owns episode reset generations, random keys and team routing. The System
    carries no critic weights, privileged inputs, critic memory or training run.
    The factory preserves the weights' training status and makes no competence
    claim. Execution accepts Threefry action keys only.
    """
    scale = _input_scale(input_scale)
    frame = _spawn_frame(spawn_frame)
    apply = (
        _apply_actor
        if scale == 1.0 and frame == "world"
        else _scaled_actor_apply(scale, frame)
    )
    return System(
        name,
        apply,
        variables=actor_params,
        init=_initial_actor_memory,
        checkpoint=checkpoint,
    )


def critic_values(
    params: Tree,
    memory: Array,
    features: Array,
    episode_start: Array,
    valid: Array,
    *,
    input_scale: float = 1.0,
) -> tuple[Array, Array]:
    """Apply the critic separately using one physical-state view per game.

    Parameters
    ----------
    params : PyTree
        Critic-only Flax variable tree with the matching feature width.
    memory : Array
        Float32 (B,5,128) critic memory before the sequence.
    features : Array
        Float32 (T,B,919) versioned physical-state features. Store one row per
        game; the function broadcasts to actor rows only during this call.
    episode_start : Array
        Boolean (T,B) resets before decisions. Death and respawn are not resets.
    valid : Array
        Boolean (T,B) valid decisions. Invalid rows retain previous memory.
    input_scale : float, default=1.0
        Fixed positive finite feature multiplier matching the learner's PPO
        config. One preserves historical raw-input values; this call applies
        the scale once inside the network and keeps stored features unchanged.

    Returns
    -------
    tuple[Array, Array]
        Final float32 (B,5,128) critic memory and float32 (T,B,5) value
        predictions in the training reward's units. Ignore invalid-row values.

    Raises
    ------
    ValueError
        input_scale is not a positive finite real number or is Boolean.

    Notes
    -----
    Matching epochs, shapes and dtypes are caller preconditions. This pure
    calculation supports jit/scan and changes no inputs. Privileged features
    and critic memory must never be routed into the actor-producing System.
    """
    shape = (*features.shape[:-1], 5)
    expanded = jnp.broadcast_to(features[..., None, :], (*shape, features.shape[-1]))
    return cast(
        tuple[Array, Array],
        RecurrentValueNet(input_scale=input_scale).apply(
            params,
            memory,
            expanded,
            jnp.broadcast_to(episode_start[..., None], shape),
            jnp.broadcast_to(valid[..., None], shape),
        ),
    )


def calculate_gae(
    rewards: Array,
    values: Array,
    ended: Array,
    final_values: Array,
    *,
    valid: Array | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Array, Array]:
    """Calculate advantages/targets with BG's produced-transition ending flags.

    Parameters
    ----------
    rewards : Array
        Float32 (T,B,5) rewards from each produced transition.
    values : Array
        Float32 (T,B,5) critic predictions before the corresponding actions.
    ended : Array
        Boolean (T,B) or (T,B,5) task endings after each produced transition.
        Wins, losses and horizon draws suppress later value estimates. Ordinary
        collection cutoffs do not; never substitute automatic-reset flags.
    final_values : Array
        Float32 (B,5) critic predictions at the true final successor, before reset.
    valid : Array or None, default=None
        Boolean (T,B) or (T,B,5) real-transition mask. None includes every row.
        False padding rows contribute no target and preserve the carried real
        successor while the reverse scan continues.
    gamma : float, default=0.99
        Reward discount per transition. Caller supplies a value from zero to one.
    gae_lambda : float, default=0.95
        Advantage mixing weight. Caller supplies a value from zero to one.

    Returns
    -------
    tuple[Array, Array]
        Advantages and value targets, both float32 (T,B,5). Invalid rows are zero.
        Targets use the supplied reward units; no reward normalization occurs.

    Raises
    ------
    ValueError
        Mask shapes cannot broadcast to the reward shape, or sequence inputs
        have incompatible leading lengths.

    Notes
    -----
    Inputs must be finite, have the stated dtypes and describe matching epochs.
    Numerical ranges and the transition flags' meaning are caller preconditions.
    This pure JAX reverse scan supports jit, changes no inputs and performs no
    environment reset, host transfer, random draw or optimizer update.
    """

    def expand(mask: Array) -> Array:
        """Broadcast lane flags across actors while preserving time/game axes."""
        return jnp.broadcast_to(
            mask[..., None] if mask.ndim == rewards.ndim - 1 else mask, rewards.shape
        )

    endings = expand(ended)
    live = jnp.ones_like(rewards, jnp.bool_) if valid is None else expand(valid)

    def backward(
        carry: tuple[Array, Array], row: tuple[Array, Array, Array, Array]
    ) -> tuple[tuple[Array, Array], Array]:
        """Use successor values only when this transition did not end its task."""
        advantage, next_value = carry
        reward, value, end, present = row
        continuation = (~end).astype(jnp.float32)
        delta = reward + gamma * next_value * continuation - value
        candidate = delta + gamma * gae_lambda * continuation * advantage
        return (
            jnp.where(present, candidate, advantage),
            jnp.where(present, value, next_value),
        ), jnp.where(present, candidate, 0.0)

    _, advantages = jax.lax.scan(
        backward,
        (jnp.zeros_like(final_values), final_values),
        (rewards, values, endings, live),
        reverse=True,
        unroll=16,
    )
    return advantages, jnp.where(live, advantages + values, 0.0)


def _mean(values: Array, mask: Array) -> Array:
    """Average finite values selected by a matching Boolean mask.

    values and mask have identical shapes. Return one scalar in values' units;
    an empty mask returns zero. Inputs stay unchanged and the work stays on device.
    """
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def _actor_loss(
    params: Tree, batch: PPOMinibatch, config: PPOConfig
) -> tuple[Array, Array]:
    """Return one group's masked actor loss and unscaled policy entropy.

    params is the shared actor variable tree. batch is one PPOMinibatch group,
    with time-first arrays and no leading G axis. config fixes clipping and the
    entropy weight. Standardize only that group's eligible advantages with the
    donor epsilon. Return two float32 scalars; the loss includes entropy's
    coefficient and entropy uses natural-log units. This performs a recurrent
    actor call but samples no actions and changes no parameters or input arrays.
    """
    _, logits = cast(
        tuple[Array, Array],
        RecurrentActor(input_scale=config.input_scale).apply(
            params,
            batch.actor_memory,
            batch.actor_features,
            batch.episode_start,
            batch.valid,
        ),
    )
    logp = action_log_prob(logits, batch.action_mask, batch.actions)
    mean = _mean(batch.advantages, batch.actor_samples)
    variance = _mean(jnp.square(batch.advantages - mean), batch.actor_samples)
    advantages = (batch.advantages - mean) / (jnp.sqrt(variance) + 1e-8)
    ratio = jnp.exp(logp - batch.old_log_prob)
    clipped = jnp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
    policy = -_mean(
        jnp.minimum(ratio * advantages, clipped * advantages), batch.actor_samples
    )
    entropy = _mean(action_entropy(logits, batch.action_mask), batch.actor_samples)
    return policy - config.entropy_coefficient * entropy, entropy


def _critic_loss(params: Tree, batch: PPOMinibatch, config: PPOConfig) -> Array:
    """Return one group's masked donor half-squared-error value loss.

    params is the shared critic variable tree. batch is one PPOMinibatch group
    without its leading G axis; config sets value-change clipping. Return one
    float32 scalar in squared reward units, before value_coefficient. Only
    critic_samples contribute. No actor inputs, optimizer updates or writes occur.
    """
    _, values = cast(
        tuple[Array, Array],
        RecurrentValueNet(input_scale=config.input_scale).apply(
            params,
            batch.critic_memory,
            batch.critic_features,
            batch.episode_start,
            batch.valid,
        ),
    )
    clipped = batch.old_values + jnp.clip(
        values - batch.old_values, -config.clip_epsilon, config.clip_epsilon
    )
    return 0.5 * _mean(
        jnp.maximum(
            jnp.square(values - batch.targets), jnp.square(clipped - batch.targets)
        ),
        batch.critic_samples,
    )


def update_minibatch(
    state: PPOTrainState, batch: PPOMinibatch, config: PPOConfig = DEFAULT_PPO_CONFIG
) -> tuple[PPOTrainState, PPOMetrics]:
    """Average eligible groups and apply one separate actor/critic optimizer step.

    Parameters
    ----------
    state : PPOTrainState
        One parameter/optimizer pair per network, shared across all groups.
    batch : PPOMinibatch
        Temporary grouped inputs. Its G axis must equal config.groups. Every
        field must follow the record's shape, epoch and sample-mask contracts.
    config : PPOConfig, default=DEFAULT_PPO_CONFIG
        Static clipping, optimizer and loss settings. Use the same optimizer
        structure that created state's actor and critic optimizer trees.

    Returns
    -------
    tuple[PPOTrainState, PPOMetrics]
        Successor network/optimizer state and scalar metrics. Actor and critic
        update independently. A network with no eligible samples retains all
        parameters and optimizer fields, including Adam momentum and counters,
        and reports zero associated losses/counts.

    Raises
    ------
    ValueError
        The batch's group count does not match config.groups.

    Notes
    -----
    Advantages, targets and old behavior values are fixed learner data; gradients
    do not flow into batch. Each group standardizes its own eligible advantages.
    Nonempty groups have equal gradient weight even when sample counts differ;
    empty groups do not dilute that average. The all-empty actor and critic
    branches separately skip their network/gradient work and optimizer update.

    Finite data, compatible parameter shapes and legal recorded actions are
    caller preconditions. Inputs remain unchanged. This supports jit with static
    config and makes no random draws, host callbacks, file writes or environment
    steps. An optimizer update is learner work, not newly collected experience.
    """
    if batch.actions.shape[0] != config.groups:
        raise ValueError("Minibatch group count must match PPOConfig.groups.")
    batch = jax.tree.map(jax.lax.stop_gradient, batch)
    actor_count = jnp.sum(batch.actor_samples, axis=(1, 2, 3))
    critic_count = jnp.sum(batch.critic_samples, axis=(1, 2, 3))

    def zero_group_gradients(params: Tree) -> Tree:
        """Create shape-matched zeros without running an empty network."""

        def zeros(value: Array) -> Array:
            """Add the group axis used by the gradient averaging boundary."""
            return jnp.zeros((config.groups, *value.shape), value.dtype)

        return jax.tree.map(zeros, params)

    def actor_gradients(_: None) -> Tree:
        """Run the actor only when at least one group has policy samples."""
        return jax.vmap(
            jax.value_and_grad(_actor_loss, has_aux=True), in_axes=(None, 0, None)
        )(state.actor_params, batch, config)

    def empty_actor(_: None) -> Tree:
        """Supply zero metrics and gradients for an entirely absent actor loss."""
        zero = jnp.zeros(config.groups, jnp.float32)
        return (zero, zero), zero_group_gradients(state.actor_params)

    (actor_loss, entropy), actor_grads = cast(
        tuple[tuple[Array, Array], Tree],
        jax.lax.cond(jnp.any(actor_count > 0), actor_gradients, empty_actor, None),
    )

    def critic_loss(params: Tree, group: PPOMinibatch) -> tuple[Array, Array]:
        """Keep the donor coefficient outside its already halved value loss."""
        value_loss = _critic_loss(params, group, config)
        return config.value_coefficient * value_loss, value_loss

    def critic_gradients(_: None) -> Tree:
        """Run the critic only when at least one group has value samples."""
        return jax.vmap(
            jax.value_and_grad(critic_loss, has_aux=True), in_axes=(None, 0)
        )(state.critic_params, batch)

    def empty_critic(_: None) -> Tree:
        """Supply zero metrics and gradients for an entirely absent value loss."""
        zero = jnp.zeros(config.groups, jnp.float32)
        return (zero, zero), zero_group_gradients(state.critic_params)

    (_, value_loss), critic_grads = cast(
        tuple[tuple[Array, Array], Tree],
        jax.lax.cond(jnp.any(critic_count > 0), critic_gradients, empty_critic, None),
    )

    def average(tree: Tree, counts: Array) -> Tree:
        """Weight nonempty groups equally before optimizer clipping."""
        present = counts > 0
        weights = present.astype(jnp.float32) / jnp.maximum(jnp.sum(present), 1)

        def average_leaf(value: Array) -> Array:
            """Reduce one leaf's leading group axis with the shared weights."""
            return jnp.sum(
                value * weights.reshape((weights.size,) + (1,) * (value.ndim - 1)),
                axis=0,
            )

        return jax.tree.map(average_leaf, tree)

    actor_grads = average(actor_grads, actor_count)
    critic_grads = average(critic_grads, critic_count)
    actor_opt, critic_opt = _optimizers(config)

    def update(
        params: Tree,
        opt_state: Tree,
        grads: Tree,
        counts: Array,
        optimizer: optax.GradientTransformation,
    ) -> tuple[Tree, Tree]:
        """Skip both parameters and all optimizer leaves when no samples exist."""

        def apply(_: None) -> tuple[Tree, Tree]:
            """Apply the already averaged gradient once to this network."""
            delta, next_opt = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, delta), next_opt

        def keep(_: None) -> tuple[Tree, Tree]:
            """Preserve the complete state when this network has no samples."""
            return params, opt_state

        return cast(
            tuple[Tree, Tree], jax.lax.cond(jnp.any(counts > 0), apply, keep, None)
        )

    actor, actor_state = update(
        state.actor_params, state.actor_opt_state, actor_grads, actor_count, actor_opt
    )
    critic, critic_state = update(
        state.critic_params,
        state.critic_opt_state,
        critic_grads,
        critic_count,
        critic_opt,
    )
    metrics = PPOMetrics(
        average(actor_loss, actor_count),
        average(entropy, actor_count),
        average(value_loss, critic_count),
        jnp.sum(actor_count),
        jnp.sum(critic_count),
    )
    return PPOTrainState(actor, critic, actor_state, critic_state), metrics


def _reflected_minibatch(
    actor_inputs: ActorInput, mask: ActionMask, actions: Array, spawn_frame: str
) -> tuple[ActorInput, Array, Array]:
    """Rebuild one minibatch's actor frame the way the actor saw it when acting.

    actor_inputs, mask and actions carry the leading (G,T,E,5) axes of one
    minibatch: the rebuilt permitted view, the stored world-frame Core masks and
    the stored world-frame categorical indices. spawn_frame is "left" or
    "right". Return the reflected view, the 198-way categorical mask in that
    frame and the indices mapped into it, so the recomputed log probability
    matches the action-time value for the same submitted move. The obstacle
    partner mask is computed once per recorded game and step from that row's
    own rebuilt table (team_obstacle_partners) and shared by its five actors;
    nothing is cached across rows or calls. Pure JAX.
    """
    flag = spawn_frame_flag(actor_inputs, spawn_frame)
    reflected, frame_mask = mirror_team_view(
        actor_inputs,
        mask,
        flag,
        obstacle_partners=team_obstacle_partners(actor_inputs),
    )
    return (
        reflected,
        categorical_action_mask(frame_mask),
        mirror_action_indices(actions, flag),
    )


def update_recurrent_ppo(
    state: PPOTrainState,
    batch: PPOBatch,
    key: Array,
    config: PPOConfig = DEFAULT_PPO_CONFIG,
) -> tuple[PPOTrainState, PPOMetrics]:
    """Update from a compact Team A rollout using whole recurrent sequences.

    Parameters
    ----------
    state : PPOTrainState
        One learner's actor/critic parameters and separate optimizer states.
    batch : PPOBatch
        Compact collected Team A data with actions shaped (T,B,5). T must equal
        config.rollout_length. B must be positive and divisible by
        config.groups*config.minibatches. Every field must describe its paired
        decision/transition epoch and satisfy PPOBatch's shape contract.
    key : Array
        One fresh typed scalar or legacy uint32 (2,) Threefry key for sequence
        shuffles. Legacy keys require the Threefry default implementation.
        The caller owns future keys; no next key is returned.
    config : PPOConfig, default=DEFAULT_PPO_CONFIG
        Static grouping, epoch, minibatch, GAE and optimizer settings, and the
        spawn frame the stored actions were chosen in. Capture one fixed
        configuration when compiling this function.

    Returns
    -------
    tuple[PPOTrainState, PPOMetrics]
        Successor parameters/optimizer states and metrics whose fields have
        shape (epochs,minibatches). Sample counts measure learner use, including
        repeated use across epochs. Caller-owned recurrent carries are unchanged.

    Raises
    ------
    ValueError
        The key is not Threefry, T differs from rollout_length, the actor axis
        is not five, or B cannot fill/divide the configured groups and minibatches.

    Notes
    -----
    GAE is computed once from the supplied old values. Each epoch shuffles whole
    game sequences within each group, preserving time and actor order. Only
    the selected minibatch is encoded. Groups share one optimizer per network.
    Death excludes policy samples while preserving value learning; inactive or
    padded rows are absent from both losses. Empty networks skip their update as
    documented by update_minibatch. When config.spawn_frame is "left" or
    "right", each minibatch's rebuilt view and mask are reflected and the
    stored world-frame indices are mapped into that frame before the log
    probability, exactly as at action time; "world" traces the donor's
    program unchanged.

    Use jit around this numerical call. Paired fields, finite values and legal
    recorded actions remain caller preconditions. Inputs are immutable. No
    environment advances, checkpoint saves, file writes or new experience occur.
    """
    if str(jax.random.key_impl(key)) != "threefry2x32":
        raise ValueError("PPO sequence shuffling requires Threefry random keys.")
    length, games, actors = batch.actions.shape
    if (
        length != config.rollout_length
        or actors != 5
        or games < config.groups * config.minibatches
        or games % (config.groups * config.minibatches)
    ):
        raise ValueError(
            "PPO needs T=rollout_length, five actor slots and B divisible "
            "by groups*minibatches."
        )
    samples = batch.valid[..., None] & batch.active
    advantages, targets = calculate_gae(
        batch.rewards,
        batch.old_values,
        batch.ended,
        batch.final_values,
        valid=samples,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    group_games = games // config.groups
    minibatch_games = group_games // config.minibatches
    keys = jax.random.split(key, config.groups)

    def epoch(
        carry: tuple[PPOTrainState, Array], unused: None
    ) -> tuple[tuple[PPOTrainState, Array], PPOMetrics]:
        """Shuffle game indices without changing their time/actor order."""
        del unused
        current, epoch_keys = carry
        split = jax.vmap(functools.partial(jax.random.split, num=3))(epoch_keys)
        next_keys, shuffle_keys = split[:, 0], split[:, 1]
        permutations = jax.vmap(
            functools.partial(jax.random.permutation, x=group_games)
        )(shuffle_keys)
        indices = permutations + jnp.arange(config.groups)[:, None] * group_games
        indices = indices.reshape(
            config.groups, config.minibatches, minibatch_games
        ).swapaxes(0, 1)

        def minibatch(
            current: PPOTrainState, selected: Array
        ) -> tuple[PPOTrainState, PPOMetrics]:
            """Expand only selected compact observations for this optimizer step."""

            def take(value: Array) -> Array:
                """Select (G,E) games from (T,B,...) and move G before time."""
                return jnp.swapaxes(jnp.take(value, selected, axis=1), 0, 1)

            compact = jax.tree.map(take, batch.observations)
            actor_inputs = jax.vmap(
                jax.vmap(
                    jax.vmap(build_team_actor_input, in_axes=(0, None)),
                    in_axes=(0, None),
                ),
                in_axes=(0, None),
            )(compact, 0)
            if config.spawn_frame == "world":
                canonical = None
                features = encode_actor_inputs(actor_inputs)
            else:
                canonical = _reflected_minibatch(
                    actor_inputs,
                    jax.tree.map(take, batch.action_mask),
                    take(batch.actions),
                    config.spawn_frame,
                )
                features = encode_actor_inputs(canonical[0])
            physical = take(batch.training_state)
            critic = jnp.broadcast_to(
                physical[..., None, :], (*features.shape[:-1], physical.shape[-1])
            )
            lane_shape = features.shape[:-1]
            valid = jnp.broadcast_to(take(batch.valid)[..., None], lane_shape)
            starts = jnp.broadcast_to(take(batch.episode_start)[..., None], lane_shape)
            member = take(batch.active)
            if canonical is None:
                mask = categorical_action_mask(jax.tree.map(take, batch.action_mask))
                actions = take(batch.actions)
            else:
                _, mask, actions = canonical
            encoded = PPOMinibatch(
                features,
                critic,
                mask,
                actions,
                take(batch.old_log_prob),
                take(batch.old_values),
                take(advantages),
                take(targets),
                starts,
                valid,
                valid & member & take(batch.alive),
                valid & member,
                batch.actor_memory[selected],
                batch.critic_memory[selected],
            )
            return update_minibatch(current, encoded, config)

        current, metrics = jax.lax.scan(minibatch, current, indices)
        return (current, next_keys), metrics

    (state, _), metrics = jax.lax.scan(epoch, (state, keys), None, length=config.epochs)
    return state, metrics
