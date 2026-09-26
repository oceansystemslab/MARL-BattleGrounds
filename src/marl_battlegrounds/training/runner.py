"""Run and resume one declared PPO, recurrent QMIX or recurrent PQN-VDN experiment.

TrainConfig owns validated host settings. train joins existing collection,
learner, checkpoint and evaluation authorities; numerical work stays outside
host logging and file handling. CLI commands use these same public functions.
All methods share the run layout, validation, selection and reports; the
saved method chooses the learner, its settings block and its host counts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from marl_battlegrounds._method_loading import validate_saved_method_reference
from marl_battlegrounds.baselines.methods import (
    TrainingMethod,
    is_ppo_method,
    method_settings_field,
    validate_training_method,
)
from marl_battlegrounds.baselines.ppo import (
    PPOConfig,
    validate_ppo_batch_size,
)
from marl_battlegrounds.baselines.pqn import (
    DEFAULT_PQN_CONFIG,
    PQNConfig,
    pqn_planned_learning_blocks,
    validate_pqn_batch_size,
)
from marl_battlegrounds.baselines.qmix import (
    DEFAULT_QMIX_CONFIG,
    QMIXConfig,
    validate_qmix_batch_size,
)
from marl_battlegrounds.tasks import DEFAULT_TDM_RED_ZONE_DEPTH
from marl_battlegrounds.training._run_io import (
    ProgressReporter,
    TrainingSpeedEstimate,
    append_jsonl,
    atomic_json,
    log_cursor,
    preserve_log_suffix,
    process_identity,
    qmix_fixed_block_counts,
    restore_log_cursor,
    run_lock,
    utc_now,
    validate_host_state,
    validate_log_cursor,
)
from marl_battlegrounds.training.opponents import (
    _pinned_share,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.training.shaping import validate_shaping

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.recording_checkpoint import PreparedRecordingFork
    from marl_battlegrounds.evaluation.run_writer import RunWriter
    from marl_battlegrounds.training.checkpoints import RestoredCheckpoint
    from marl_battlegrounds.training.collection import (
        TrainingCarry,
        TrainingCollection,
        TrainingRollout,
    )
    from marl_battlegrounds.training.learner import LearnerState, UpdateResult
    from marl_battlegrounds.training.validation import FrozenPanel

    type _Updater = Callable[
        [LearnerState, TrainingCarry, TrainingRollout],
        tuple[LearnerState, UpdateResult],
    ]


@dataclass(frozen=True)
class TrainConfig:
    """Declare one experiment without opening files or starting a backend.

    Parameters
    ----------
    seed : int, default=42
        Root seed for model and independent collection/shuffle/sampling streams.
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Learner saved with the run. MAPPO uses physical critic inputs; IPPO
        uses each actor's permitted inputs. The ff_ methods use two 128-wide
        layers without recurrent memory. "qmix" is recurrent QMIX with compact
        replay, configured by qmix instead of ppo. "pqn_vdn" is recurrent
        PQN-VDN (no replay or target network), configured by pqn.
    num_envs : int, default=32
        Fixed even environment batch. Recurrent PPO methods require
        divisibility by PPO groups times minibatches. Feedforward methods
        require divisibility by groups, and rollout_length*num_envs by groups
        times minibatches. QMIX has no divisibility rule, but
        num_envs*qmix.rollout_length*5 must fit a signed 32-bit integer.
        PQN-VDN needs num_envs divisible by pqn.num_minibatches and the
        int32 guards of ``validate_pqn_batch_size``.
    total_env_steps : int, default=10000000
        Exact real environment transitions, divisible by num_envs. Resets and
        padding do not count. Per-lane rounds must fit a signed 32-bit integer.
        Whole-run totals may exceed it. QMIX needs at least min_buffer_size
        rounds per game, and rounds times qmix.epochs must fit int32.
        PQN-VDN needs at least memory_window + rollout_length + 1 rounds per
        game (W initial random rounds, counted in the budget, then at least
        one learning round) and planned optimizer steps within int32.
    curriculum, shaping : bool, default=False
        Enable the existing 17-stage schedule and chosen team reward adjustment.
    score_threshold_curriculum : bool, default=False
        Use K1..10, K12, K15 and K20 at future resets, with 10%, nine shares
        of 1/30, 5%, 5% and 50% of requested experience. Keeps canonical 5v5
        and all training maps. Cannot be combined with curriculum=True.
        Evaluation stays K20/H300. Actual played shares can lag stage changes.
    red_zone_depth : float, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Team Deathmatch Red Zone depth in map units for every training game
        and every validation game of this run. When an agent dies inside its
        own team's Red Zone (the strip this deep at its own spawn side), the
        enemy team gets 2 points instead of 1; it is still one kill and one
        death. 0.0 keeps one point per death. It must be a Python float that
        passes Core's scalar rules: finite, not negative (-0.0 is refused) and,
        when positive, a normal float32 value; training setup also checks it
        against every map width before any file is written. It is fixed for a
        run: a resume that declares another depth is refused. A JSON config
        may write it as an integer such as 6, which reads as 6.0; a saved run
        config without it was saved before the rule and reads as 0.0.
    shaping_coefficient : float, default=0.01
        Finite nonnegative shaping weight. Task reward remains separately logged.
    shaping_mode : {"potential", "score_delta"}, default="potential"
        Potential preserves the discounted task objective. Score_delta adds
        reward for the team's new points minus the enemy's new points without
        terminal cancellation; it deliberately changes the training objective.
        Points follow the task's scoring, so with a positive red_zone_depth a
        Red Zone death moves it by 2 instead of 1. Used only when shaping is
        True. Evaluation always uses native task rewards and win rules.
    pinned_opponent_share : float, default=0.0
        Probability, within [0, 0.8], that each new training game meets the
        pinned first-update actor instead of drawing from the ordinary self-play
        recipe. Zero keeps 80% current weights and 20% uniform history with
        today's exact random draws. A positive share also moves the first
        history snapshot to the first completed update and drops the never-played
        100% snapshot, so slot 0 holds that near-untrained actor for the whole
        run; the remaining probability is 20% other history when any exists and
        current weights otherwise. Evaluation opponents are unaffected.
    pinned_opponent : str or None, default=None
        None keeps the pinned share playing that first-update actor. Otherwise
        a method reference that plays the pinned share instead: a built-in name
        ("random", "tdm-alpha", "tdm-beta", "tdm-gamma"), an absolute path to
        an actor export or complete learner checkpoint directory (relative paths fail),
        or a
        ``module:function`` factory, resolved by
        ``load_method`` when the run is set up and again on resume. Requires a
        positive pinned_opponent_share. JAX methods and host methods (for
        example LLM agents) are both accepted; a host method with memory cannot
        be resumed while one of its games is unfinished. The pinned opponent
        is a training opponent: results against it, and its evidence of
        protected-controller exposure, are recorded with the run.
    ppo : PPOConfig, default=PPOConfig()
        Immutable donor network update settings, including rollout length,
        input scale and spawn frame. A shared random initialization result can
        be reused only by a run with the same scale and frame. QMIX and
        PQN-VDN runs must leave it at the default; it is not saved for them.
    qmix : QMIXConfig or None, default=None
        QMIX settings (replay sizes, sampling, target rule, exploration, input
        scale and spawn frame). None means DEFAULT_QMIX_CONFIG for a QMIX run
        and is required for other runs. It is saved only for QMIX.
    pqn : PQNConfig or None, default=None
        PQN-VDN settings (T, H, epochs, minibatches, rates, schedules, input
        scale and spawn frame). None means DEFAULT_PQN_CONFIG for a PQN-VDN
        run and is required for other runs. It is saved only for PQN-VDN.
    metrics : {"priority", "none"}, default="priority"
        Existing episode metric level. No replay recording is implied.
    recording : bool, default=False
        Save episode tables through a separate training RunWriter. False skips
        recording drains; learner checkpoints and update summaries still save.
    validation_panel : str or None, default=None
        Existing frozen panel.json path. Use this or validation_opponents.
        Demonstration needs a panel by train setup; development may omit it.
    validation_opponents : tuple[str, ...] or None, default=None
        Built-in names, absolute actor export/checkpoint paths or
        module:function factories.
        Freeze these once before training. Live methods instead belong in
        train's validation_opponents keyword. Conflicting declarations fail.
    slot_diagnostic_actor : str or None, default=None
        Explicit absolute actor export (matching the run's method)
        for the optional slot comparison with a
        new System panel. Historical panels keep their final actor by default.
        Enabled diagnostics verify this artifact before learner setup.
    validation_fractions : tuple[float, ...], default=(0.1, ..., 1.0)
        Increasing experience fractions, rounded up to the next completed
        collection block. For PPO and QMIX blocks end at multiples of
        rollout_length rounds; for PQN-VDN they end at multiples of T rounds
        up to its W = memory_window + rollout_length initial rounds, then at
        W + j * T, and at the final round (for example 4,096, 4,224, 8,320
        transitions at 32 games with the defaults).
        Initialization is diagnostic only. The final fraction must be one.
    routine_seed_pairs, confirmation_seed_pairs : int, default=10, 50
        Independent seed pairs per validation map and opponent.
    checkpoint_interval_updates : int or None, default=None
        Recovery-save interval in completed updates: PPO updates for PPO,
        optimizer steps for QMIX and PQN-VDN. A save happens whenever
        completed_updates crosses a multiple of it (a QMIX block adds epochs
        steps, so 24 to 28 crosses 25; a PQN-VDN block adds epochs *
        num_minibatches steps, and initial chunks add none). None means 25
        for PPO and 1600 for QMIX and PQN-VDN (at the defaults: 102,400
        transitions at 32 games for QMIX, 25 learning blocks for PQN-VDN);
        QMIX checkpoints carry the replay (about 1 GB at 32 games);
        the resolved number is what is saved. Initialization, capture points
        and the final boundary always save. The number is resolved when the
        config is built, so ``dataclasses.replace(ppo_config, method="qmix")``
        or ``method="pqn_vdn"`` keeps PPO's 25; pass
        ``checkpoint_interval_updates=None`` in that replace call to get the
        1600 default.
    checkpoint_env_steps : tuple[int, ...], default=()
        Additional exact actor capture points, each exactly at a collection
        boundary in transitions (the same boundaries as validation_fractions:
        for PQN-VDN these are offset by its W initial rounds, so 8,320 is a
        boundary at 32 games with the defaults and 8,192 is refused).
    random_diagnostic_seed_pairs : int or None, default=None
        Enable fixed Random diagnostics at initialization, checkpoint_env_steps
        and the final budget. The positive count is paired seeds per each of
        five validation maps; four gives 40 games. No panel is required and no
        actor is selected from these results. None skips this optional work.
    random_initialization_result : str or None, default=None
        Original runner result copied to a JSON file for initialization reuse.
        Requires Random diagnostics. Its actor inference, task and complete M8
        evidence must match before output or recovery changes. Original evidence
        paths must remain available; later captures still run their own games.
    slot_diagnostic : bool, default=False
        Run the declared diagnostic after final training. Requires a frozen panel.
    purpose : {"development", "demonstration"}, default="development"
        Demonstration requires frozen validation and final selection.
    verbose : bool, default=True
        Print existing host summaries at most every ten seconds and phase changes.
        Adds no numerical calls or device transfers. Full-run use is qualified by
        the separate measured reporting-cost check. False skips display and extra
        ETA calculation while retaining required counters, timings and status.

    Raises
    ------
    ValueError, TypeError
        A count, option or numerical setting violates this contract. Booleans
        are not accepted as integer counts. No files are changed on rejection.
    """

    seed: int = 42
    method: TrainingMethod = "mappo"
    num_envs: int = 32
    total_env_steps: int = 10_000_000
    curriculum: bool = False
    score_threshold_curriculum: bool = False
    red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH
    shaping: bool = False
    shaping_coefficient: float = 0.01
    shaping_mode: Literal["potential", "score_delta"] = "potential"
    pinned_opponent_share: float = 0.0
    pinned_opponent: str | None = None
    ppo: PPOConfig = field(default_factory=PPOConfig)
    qmix: QMIXConfig | None = None
    pqn: PQNConfig | None = None
    metrics: Literal["priority", "none"] = "priority"
    recording: bool = False
    validation_panel: str | None = None
    validation_opponents: tuple[str, ...] | None = None
    slot_diagnostic_actor: str | None = None
    validation_fractions: tuple[float, ...] = tuple(i / 10 for i in range(1, 11))
    routine_seed_pairs: int = 10
    confirmation_seed_pairs: int = 50
    checkpoint_interval_updates: int | None = None
    checkpoint_env_steps: tuple[int, ...] = ()
    random_diagnostic_seed_pairs: int | None = None
    random_initialization_result: str | None = None
    slot_diagnostic: bool = False
    purpose: Literal["development", "demonstration"] = "development"
    verbose: bool = True

    def __post_init__(self) -> None:
        """Reject invalid host settings without allocating numerical arrays.

        Resolves qmix=None to DEFAULT_QMIX_CONFIG for QMIX runs, pqn=None to
        DEFAULT_PQN_CONFIG for PQN-VDN runs and checkpoint_interval_updates=None
        to 25 (PPO) or 1600 (QMIX and PQN-VDN). red_zone_depth is checked by
        the training content owner's copy of Core's scalar rules
        (``_content._validate_red_zone_depth``); map widths are checked later,
        when train prepares the content.
        """
        method = validate_training_method(self.method)
        if not isinstance(cast(object, self.ppo), PPOConfig):
            raise TypeError("ppo must be PPOConfig")
        if method == "qmix":
            if self.ppo != PPOConfig():
                raise ValueError("A QMIX run takes no PPO settings")
            if self.qmix is None:
                object.__setattr__(self, "qmix", DEFAULT_QMIX_CONFIG)
            if not isinstance(cast(object, self.qmix), QMIXConfig):
                raise TypeError("qmix must be QMIXConfig or None")
        elif self.qmix is not None:
            raise ValueError(
                "A PQN-VDN run takes no QMIX settings"
                if method == "pqn_vdn"
                else "A PPO run takes no QMIX settings"
            )
        if method == "pqn_vdn":
            if self.ppo != PPOConfig():
                raise ValueError("A PQN-VDN run takes no PPO settings")
            if self.pqn is None:
                object.__setattr__(self, "pqn", DEFAULT_PQN_CONFIG)
            if not isinstance(cast(object, self.pqn), PQNConfig):
                raise TypeError("pqn must be PQNConfig or None")
        elif self.pqn is not None:
            raise ValueError(
                "A QMIX run takes no PQN settings"
                if method == "qmix"
                else "A PPO run takes no PQN settings"
            )
        if self.checkpoint_interval_updates is None:
            object.__setattr__(
                self,
                "checkpoint_interval_updates",
                25 if is_ppo_method(method) else 1600,
            )
        if type(self.seed) is not int:
            raise TypeError("seed must be a Python integer, not bool")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for name in (
            "num_envs",
            "total_env_steps",
            "routine_seed_pairs",
            "confirmation_seed_pairs",
            "checkpoint_interval_updates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer")
        for name in (
            "curriculum",
            "score_threshold_curriculum",
            "shaping",
            "recording",
            "slot_diagnostic",
            "verbose",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.curriculum and self.score_threshold_curriculum:
            raise ValueError("Choose team/map curriculum or score-threshold curriculum")
        from marl_battlegrounds.training._content import (
            _validate_red_zone_depth,  # pyright: ignore[reportPrivateUsage]
        )

        _validate_red_zone_depth(self.red_zone_depth)
        if self.num_envs % 2:
            raise ValueError("num_envs must be even for paired spawn ends")
        if self.total_env_steps % self.num_envs:
            raise ValueError("total_env_steps must be divisible by num_envs")
        rounds = self.total_env_steps // self.num_envs
        if rounds > 2**31 - 1:
            raise ValueError("per-lane rounds must fit int32")
        if self.qmix is not None:
            validate_qmix_batch_size(self.num_envs, self.qmix)
            if rounds < self.qmix.min_buffer_size:
                raise ValueError(
                    "A QMIX budget needs at least min_buffer_size rounds per game"
                )
            if rounds * self.qmix.epochs > 2**31 - 1:
                raise ValueError("QMIX optimizer counts for this budget exceed int32")
            discount, rollout_length = self.qmix.gamma, self.qmix.rollout_length
        elif self.pqn is not None:
            validate_pqn_batch_size(self.num_envs, self.pqn)
            pqn_planned_learning_blocks(rounds, self.pqn)
            discount, rollout_length = self.pqn.gamma, self.pqn.rollout_length
        else:
            validate_ppo_batch_size(self.num_envs, self.ppo, method=method)
            discount, rollout_length = self.ppo.gamma, self.ppo.rollout_length
        if self.metrics not in ("priority", "none"):
            raise ValueError("training metrics must be priority or none")
        if (
            isinstance(self.shaping_coefficient, bool)
            or not math.isfinite(self.shaping_coefficient)
            or self.shaping_coefficient < 0
        ):
            raise ValueError("shaping_coefficient must be finite and nonnegative")
        validate_shaping(
            discount=discount,
            coefficient=self.shaping_coefficient,
            mode=self.shaping_mode,
        )
        _pinned_share(self.pinned_opponent_share)
        if self.pinned_opponent is not None:
            if (
                not isinstance(cast(object, self.pinned_opponent), str)
                or not self.pinned_opponent.strip()
            ):
                raise ValueError("pinned_opponent must be None or a nonempty string")
            if not self.pinned_opponent_share > 0:
                raise ValueError(
                    "pinned_opponent needs a positive pinned_opponent_share"
                )
            validate_saved_method_reference(self.pinned_opponent)
        if self.purpose not in ("development", "demonstration"):
            raise ValueError("purpose must be development or demonstration")
        if self.validation_panel is not None and (
            not isinstance(cast(object, self.validation_panel), str)
            or not self.validation_panel
        ):
            raise ValueError("validation_panel must be a nonempty path string or None")
        if self.random_diagnostic_seed_pairs is not None and (
            type(self.random_diagnostic_seed_pairs) is not int
            or self.random_diagnostic_seed_pairs <= 0
        ):
            raise ValueError(
                "random_diagnostic_seed_pairs must be a positive integer or None"
            )
        if self.random_initialization_result is not None and (
            not isinstance(cast(object, self.random_initialization_result), str)
            or not self.random_initialization_result
            or self.random_diagnostic_seed_pairs is None
        ):
            raise ValueError(
                "random_initialization_result needs a path and enabled "
                "Random diagnostics"
            )
        if self.validation_opponents is not None:
            if (
                not isinstance(cast(object, self.validation_opponents), tuple)
                or not self.validation_opponents
            ):
                raise ValueError(
                    "validation_opponents must be a nonempty tuple of references"
                )
            for reference in self.validation_opponents:
                validate_saved_method_reference(reference)
            if self.validation_panel is not None:
                raise ValueError(
                    "Declare validation_panel or validation_opponents, not both"
                )
        if self.slot_diagnostic_actor is not None and (
            not isinstance(cast(object, self.slot_diagnostic_actor), str)
            or not Path(self.slot_diagnostic_actor).is_absolute()
        ):
            raise ValueError("slot_diagnostic_actor must be an absolute actor path")
        if self.slot_diagnostic and not (
            self.validation_panel
            or self.validation_opponents
            or self.slot_diagnostic_actor
        ):
            raise ValueError("slot_diagnostic requires a frozen validation panel")
        fractions = self.validation_fractions
        if (
            not isinstance(cast(object, fractions), tuple)
            or not fractions
            or fractions[-1] != 1
        ):
            raise ValueError(
                "validation_fractions must be a nonempty tuple ending at 1"
            )
        previous = 0.0
        for fraction in fractions:
            if (
                isinstance(fraction, bool)
                or not math.isfinite(fraction)
                or not previous < fraction <= 1
            ):
                raise ValueError("validation_fractions must increase within (0, 1]")
            previous = fraction
        if not isinstance(cast(object, self.checkpoint_env_steps), tuple):
            raise TypeError("checkpoint_env_steps must be a tuple")
        from marl_battlegrounds.training.validation import (
            _collection_boundary,  # pyright: ignore[reportPrivateUsage]
        )

        initial = self.pqn.initial_rounds if self.pqn is not None else 0
        previous_step = 0
        for step in self.checkpoint_env_steps:
            if (
                type(step) is not int
                or not previous_step < step <= self.total_env_steps
                or _collection_boundary(
                    step,
                    total_env_steps=self.total_env_steps,
                    num_envs=self.num_envs,
                    rollout_length=rollout_length,
                    initial_rounds=initial,
                )[0]
                != step
            ):
                raise ValueError(
                    "checkpoint_env_steps must increase at complete update boundaries"
                )
            previous_step = step


@dataclass(frozen=True)
class TrainResult:
    """Return saved paths and exact counts after all declared work succeeds.

    run_dir identifies the experiment; final_actor and selected_actor identify
    frozen exports (selection is None without a panel or an eligible trained
    candidate). completed_env_steps and
    completed_updates exclude padding; for QMIX completed_updates counts
    optimizer steps (epochs per learning block) and for PQN-VDN optimizer
    steps (epochs times minibatches per learning block), not blocks. status is
    complete only after validation, export and reporting finish.
    evidence_paths names the shared report outputs.
    This record carries no rollout, live model state or learning qualification.
    """

    run_dir: Path
    final_actor: Path
    selected_actor: Path | None
    completed_env_steps: int
    completed_updates: int
    status: Literal["complete"]
    evidence_paths: tuple[Path, ...]


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """Return version-1 JSON-ready settings; preserve every resolved option.

    Parameters
    ----------
    config : TrainConfig
        A validated training config. Its defaults are already resolved, for
        example qmix=None has become DEFAULT_QMIX_CONFIG for a QMIX run.

    Returns
    -------
    dict[str, Any]
        A new dictionary: "schema_version": 1 first, then every TrainConfig
        field in field order after a JSON round trip, so tuples become lists
        and settings objects become dictionaries. Only the run's own settings
        block is kept (``methods.method_settings_field``): a PPO config omits
        qmix and pqn, so historical PPO bytes are unchanged; a QMIX config
        keeps only its resolved qmix settings and a PQN-VDN config only its
        resolved pqn settings. red_zone_depth is always written as a JSON
        number such as 5.0 (map units; 0.0 means the Red Zone rule is off), so
        every new saved config states its scoring rule. The schema version
        stays 1 (a config saved before the rule lacks the key).

    Notes
    -----
    Host-only; config is not changed. train saves this dictionary in the run's
    metadata, which checkpoints carry as metadata.config, and compares two of
    them to check a resume config. config_from_dict reads it back.
    """
    data = json.loads(json.dumps(asdict(config)))
    kept = method_settings_field(config.method)
    for name in ("ppo", "qmix", "pqn"):
        if name != kept:
            data.pop(name)
    return {"schema_version": 1, **data}


def config_from_dict(value: dict[str, Any]) -> TrainConfig:
    """Read version-1 settings and reject unknown fields before opening a run.

    Parameters
    ----------
    value : dict[str, Any]
        A JSON object with optional schema_version=1. Omitted settings use
        TrainConfig defaults; ppo, qmix and pqn are settings objects. A config
        may contain only its own method's block: a QMIX config no ppo or pqn,
        a PQN-VDN config no ppo or qmix, a PPO config no qmix or pqn (method
        defaults to "mappo"). red_zone_depth is the Red Zone depth in map
        units; 0.0 turns the rule off.

    Returns
    -------
    TrainConfig
        The validated config. Lists for validation fractions, extra save points
        (checkpoint_env_steps) and validation_opponents become tuples. A JSON
        integer red_zone_depth (not true or false) becomes the equal float, so
        6 reads as 6.0; an omitted depth takes the TrainConfig default
        (DEFAULT_TDM_RED_ZONE_DEPTH, 5.0).

    Raises
    ------
    ValueError
        schema_version is not the integer 1; a field is unknown; a settings
        block belongs to another method; an integer red_zone_depth is too
        large to become a float ("red_zone_depth must be finite"); or
        TrainConfig or a settings object rejects a value.
    TypeError
        ppo, qmix or pqn is not a JSON object; one of the three list settings
        is not a JSON array; a settings object gets an unknown key; or
        TrainConfig or a settings object rejects a value's type.

    Notes
    -----
    Host-only; the input dictionary is not changed. An omitted depth means
    5.0 here, so a config saved before the Red Zone rule must first pass
    through checkpoints.saved_training_config, which fills the missing depth
    with 0.0 (its original one-point scoring).
    """
    data = dict(value)
    version = data.pop("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported training config schema_version")
    unknown = set(data) - {item.name for item in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
    method = data.get("method", "mappo")
    label = {"qmix": "QMIX", "pqn_vdn": "PQN-VDN"}.get(str(method), "PPO")
    kept = method_settings_field(method)
    for name in ("ppo", "qmix", "pqn"):
        if name != kept and name in data:
            raise ValueError(f"A {label} training config takes no {name} settings")
    if "ppo" in data:
        if not isinstance(data["ppo"], dict):
            raise TypeError("ppo must be a JSON object")
        data["ppo"] = PPOConfig(**cast(dict[str, Any], data["ppo"]))
    if "qmix" in data and data["qmix"] is not None:
        if not isinstance(data["qmix"], dict):
            raise TypeError("qmix must be a JSON object")
        data["qmix"] = QMIXConfig(**cast(dict[str, Any], data["qmix"]))
    if "pqn" in data and data["pqn"] is not None:
        if not isinstance(data["pqn"], dict):
            raise TypeError("pqn must be a JSON object")
        data["pqn"] = PQNConfig(**cast(dict[str, Any], data["pqn"]))
    for name in (
        "validation_fractions",
        "checkpoint_env_steps",
        "validation_opponents",
    ):
        if name in data and data[name] is not None:
            if not isinstance(data[name], (list, tuple)):
                raise TypeError(f"{name} must be a JSON array")
            data[name] = tuple(data[name])
    # JSON has one number type: a hand-written 6 means the depth 6.0.
    if type(data.get("red_zone_depth")) is int:
        try:
            data["red_zone_depth"] = float(data["red_zone_depth"])
        except OverflowError as error:
            raise ValueError("red_zone_depth must be finite") from error
    return TrainConfig(**data)


def read_config(path: str | Path) -> TrainConfig:
    """Load a UTF-8 JSON training config through the shared strict validator.

    path must name a JSON object. Read/parse errors propagate; invalid settings
    raise ValueError or TypeError before training files or a backend are opened.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Training config must be a JSON object")
    return config_from_dict(cast(dict[str, Any], value))


def _preserve_reports(root: Path, attempt: str) -> None:
    """Archive existing derived reports before moving their active continuation.

    Preserve raw bytes, including malformed reports, under their content hashes
    in the old attempt folder. Fsync before the caller clears active summaries.
    Immutable evaluation tasks remain in their original directories. The caller
    holds the run lock and has validated the complete recovery checkpoint.
    """
    directory = root / "attempts" / attempt / "reports"
    for name in (
        "validation_results.json",
        "random_diagnostics.json",
        "selection.json",
        "exposure.json",
        "slot_diagnostic.json",
        "run_summary.md",
    ):
        source = root / name
        if not source.is_file():
            continue
        raw = source.read_bytes()
        directory.mkdir(parents=True, exist_ok=True)
        destination = (
            directory / f"{source.stem}-{sha256(raw).hexdigest()}{source.suffix}"
        )
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError("Archived report identity conflicts")
            continue
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def train(
    config: TrainConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    validation_opponents: Sequence[Any] | None = None,
) -> TrainResult:
    """Run or resume one complete declared training experiment on this device.

    Parameters
    ----------
    config : TrainConfig or None, default=None
        Required for a new run. Resume inherits saved settings when omitted;
        an explicit config must match all saved settings exactly.
    output_dir : str or Path or None, default=None
        Exact new or empty run directory. Required for new runs and mutually
        exclusive with resume_from. No generated parent run name is inserted.
    resume_from : str or Path or None, default=None
        Exact complete learner checkpoint directory. Discover the containing
        run, verify scientific/source/content facts, execution identity and
        numerical state before writer rewind. The saved compiler policy,
        backend, device kind, runtime and numerical settings must match.
        Older checkpoints without that identity remain readable but cannot
        resume. A saved config without ppo.spawn_frame means "world" whatever
        the current default is; a supplied config that disagrees is rejected.
        A checkpoint saved before the Red Zone rule (actor input schema 1) is
        refused before any file is touched; resuming it needs the source
        environment that created it. A supplied config whose red_zone_depth
        differs from the saved run's is refused with a message naming both
        depths, also before any file is touched.
        A QMIX or PQN-VDN resume restores into a shape-only template, so
        restoring never holds a second learner. Once training starts, train
        keeps no reference to the first learner state; a QMIX block then holds
        the previous replay and the new one plus working space (plan for up to
        about three replays). Failed recovery retains its explicit retry
        marker.

    validation_opponents : sequence of System, Policy or str, optional
        Live methods or reload references for a new frozen panel. A new run cannot
        also declare config.validation_opponents. With config.validation_panel or
        resume, these bindings must match every saved member in order; they never
        replace membership. A live-only client must be supplied again on resume.

    Returns
    -------
    TrainResult
        Saved actor/report paths and exact counts after all declared work ends.
        A development run without a panel has no selected actor. Completion
        records execution, not a claim of useful learned behavior.

    Raises
    ------
    ValueError, TypeError
        Settings, content, checkpoint or saved results are invalid/incompatible.
    RuntimeError
        Another writer owns the run or numerical execution fails.
    OSError
        Required output cannot be read or published. Failure state and the last
        complete checkpoint remain available when the filesystem permits it.

    Notes
    -----
    Runs synchronously and owns its files/process lock until completion. Device
    selection belongs to the caller's environment before importing JAX. No
    budget extension, replacement seed or silent retry occurs. Validation uses
    independent frozen Systems and keys, with its own saved M8 records, and
    plays under the run's red_zone_depth. Setup prepares the training content
    once (checking the depth against every map width) before a new
    validation panel is published, and passes it to the learner; a ranked
    panel whose recorded depth differs from the run's is refused.
    """
    return _train(
        config,
        output_dir=output_dir,
        resume_from=resume_from,
        validation_opponents=validation_opponents,
    )


def extend_training(
    checkpoint: str | Path,
    *,
    additional_env_steps: int,
    output_dir: str | Path,
    changes: Mapping[str, object] | None = None,
    validation_opponents: Sequence[Any] | None = None,
) -> TrainResult:
    """Continue a full learner checkpoint into a separate child run.

    Parameters
    ----------
    checkpoint : str or Path
        Complete post-Red-Zone learner folder, ``checkpoints/<id>``. Actor
        exports and pruned learner folders cannot continue training. All saved
        learner state, game memory and random streams are restored first.
    additional_env_steps : int
        Positive number of real environment transitions to add to the saved
        step, divisible by the unchanged environment batch. This does not
        restart the parent's curriculum or stretch its learning-rate decay.
    output_dir : str or Path
        Exact new or empty child folder outside the parent run. The parent is
        read only. The child can later use ordinary exact-config resume.
    changes : mapping or None, default=None
        Optional future declarations named validation, learning_rate,
        exploration and history_capture_env_steps. Other keys fail. Existing
        settings remain unchanged by default. PQN needs an explicit rate choice
        if any added optimizer update would use its terminal rate. See the
        continuation example for each supported declaration. A changed validation
        panel may have runtime-only ``bindings``; those clients are checked by
        their saved identities and are never serialized into the declaration.
    validation_opponents : sequence of System, Policy or str, optional
        Runtime bindings for the parent's frozen validation panel, as in train.
        Live-only members must be supplied again. Membership and saved identities
        must match. To change the future panel, use changes.validation.panel and
        its own optional bindings. Neither route changes the parent's panel.

    Returns
    -------
    TrainResult
        Child paths, cumulative counts and the best eligible saved actor. A
        run without a validation panel or eligible trained candidate has no
        selected actor. A child ending during warmup still exports its actor;
        status.json explains why no checkpoint was selected.

    Raises
    ------
    ValueError, TypeError
        The budget, declaration, full checkpoint or source transition is not
        supported. Actor loading remains independent of this training rule.
    RuntimeError, OSError
        A lock, numerical operation or file operation fails. Complete saved
        child boundaries remain available for explicit resume.

    Notes
    -----
    Runs synchronously on the caller's selected device. Saves the parent
    identity and future declaration before taking another training step.
    Parent and child are one training lineage, not independent training seeds.
    """
    if type(additional_env_steps) is not int or additional_env_steps <= 0:
        raise ValueError("additional_env_steps must be a positive integer")
    if changes is not None and not isinstance(cast(object, changes), Mapping):
        raise TypeError("changes must be a mapping or None")
    declared = {} if changes is None else dict(changes)
    allowed = {
        "validation",
        "learning_rate",
        "exploration",
        "history_capture_env_steps",
    }
    if unknown := set(declared) - allowed:
        raise ValueError(f"Unsupported continuation changes: {sorted(unknown)}")
    future_bindings = None
    validation_changes = declared.get("validation")
    if isinstance(validation_changes, Mapping):
        validation_changes = dict(cast(Mapping[str, object], validation_changes))
        future_bindings = validation_changes.pop("bindings", None)
        if future_bindings is not None:
            if "panel" not in validation_changes:
                raise ValueError(
                    "Use validation_opponents for the parent's panel; "
                    "changes.validation.bindings requires a declared future panel"
                )
            if isinstance(future_bindings, (str, bytes)) or not isinstance(
                future_bindings, Sequence
            ):
                raise TypeError("Validation bindings must be a sequence of methods")
        declared["validation"] = validation_changes
    # Runtime clients stay outside JSON. Scientific settings remain finite data.
    declared = json.loads(json.dumps(declared, allow_nan=False))
    return _train(
        output_dir=output_dir,
        resume_from=checkpoint,
        validation_opponents=validation_opponents,
        extension={
            "additional_env_steps": additional_env_steps,
            "changes": declared,
            "validation_bindings": future_bindings,
        },
    )


def _child_config(
    value: dict[str, Any], *, total_env_steps: int, root_total_env_steps: int
) -> TrainConfig:
    """Validate inherited settings while allowing a child to end during warmup.

    value contains ordinary saved settings. root_total_env_steps is the verified
    original fresh-run budget; total_env_steps is the child's declared absolute
    end. Fresh-run minimum warmup/curriculum lengths do not apply to a child
    segment. All method, batch, scientific and integer-limit checks still apply.
    This constructs a new frozen config and never mutates a caller's object.
    """
    if type(total_env_steps) is not int or total_env_steps <= 0:
        raise ValueError("Child total_env_steps must be a positive integer")
    if type(root_total_env_steps) is not int or root_total_env_steps <= 0:
        raise ValueError("Continuation requires its original valid training budget")
    checked = config_from_dict(
        {**value, "total_env_steps": max(total_env_steps, root_total_env_steps)}
    )
    if total_env_steps % checked.num_envs or any(
        step > total_env_steps for step in checked.checkpoint_env_steps
    ):
        raise ValueError("Child budget and save points must contain valid whole rounds")
    # Only the fresh-run minimum is different for a checked child. Construct its
    # declared total after the remaining fields passed the ordinary validator.
    object.__setattr__(checked, "total_env_steps", total_env_steps)
    return checked


def _saved_config(details: dict[str, Any]) -> TrainConfig:
    """Read ordinary settings or verify a child's parent before its config.

    Learner descriptions are content identified. A child keeps the parent's
    original budget and actual saved boundary; arbitrary changed settings do
    not gain the child-only warmup allowance. Parent descriptions must remain
    at their recorded paths, even when old payloads are pruned. An unreadable
    parent raises ValueError naming that path. No payload or run file is changed.
    """
    from marl_battlegrounds.training import checkpoints

    value = checkpoints.saved_training_config(details)
    context = details["metadata"].get("continuation")
    if context is None:
        return config_from_dict(value)
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        raise ValueError("Unsupported continuation declaration")
    context = cast(dict[str, Any], context)
    parent_path = Path(context["parent_checkpoint"])
    try:
        parent = checkpoints.read_checkpoint_description(parent_path)
    except ValueError as error:
        raise ValueError(
            f"Cannot read the continuation parent at {parent_path}: {error}. "
            "Keep its saved checkpoint folder and description at that path."
        ) from error
    previous = parent["metadata"].get("continuation")
    root = context["segment"]["root_schedule"]
    expected_root = (
        parent["metadata"]["config"]["total_env_steps"]
        if previous is None
        else previous["segment"]["root_schedule"]["total_env_steps"]
    )
    if (
        parent["kind"] != "learner"
        or parent["checkpoint_id"] != context["parent_checkpoint_id"]
        or parent["metadata"]["run_id"] != context["parent_run_id"]
        or parent["metadata"]["source"] != context["parent_source"]
        or context["child_source"] != details["metadata"]["source"]
        or parent["counters"]["env_steps"] != context["start_env_steps"]
        or context["start_env_steps"] + context["additional_env_steps"]
        != value["total_env_steps"]
        or context["resulting_total_env_steps"] != value["total_env_steps"]
        or root["total_env_steps"] != expected_root
        or context["root_run_id"]
        != (
            parent["metadata"]["run_id"]
            if previous is None
            else previous["root_run_id"]
        )
    ):
        raise ValueError("Continuation differs from its saved parent boundary")
    parent_config = checkpoints.saved_training_config(parent)
    inherited_fields = set(parent_config) - {
        "total_env_steps",
        "checkpoint_env_steps",
        "validation_panel",
        "validation_opponents",
    }
    expected_points = [
        point
        for point in parent_config["checkpoint_env_steps"]
        if context["start_env_steps"] < point <= value["total_env_steps"]
    ]
    if (
        set(value) != set(parent_config)
        or any(value[name] != parent_config[name] for name in inherited_fields)
        or value["checkpoint_env_steps"] != expected_points
        or value["validation_panel"]
        != details["metadata"]["validation_declaration"]["panel_path"]
        or value["validation_opponents"]
        != (
            None if value["validation_panel"] else parent_config["validation_opponents"]
        )
    ):
        raise ValueError("Continuation changed an inherited training setting")
    return _child_config(
        value,
        total_env_steps=value["total_env_steps"],
        root_total_env_steps=expected_root,
    )


def _train(
    config: TrainConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    validation_opponents: Sequence[Any] | None = None,
    extension: dict[str, Any] | None = None,
) -> TrainResult:
    """Share setup, checked restore and execution for training and a declared fork.

    Public callers use train or extend_training. extension is their normalized
    added budget and future-only changes; it never relaxes ordinary resume.
    """
    attempt_started = time.monotonic()
    started_at = utc_now()
    from marl_battlegrounds.training import checkpoints, learner, validation
    from marl_battlegrounds.training._compilation import execution_identity
    from marl_battlegrounds.training._content import prepare_training_content
    from marl_battlegrounds.training.curriculum import make_training_schedule

    if extension is None and (output_dir is None) == (resume_from is None):
        raise ValueError("Supply exactly one of output_dir or resume_from")
    if extension is not None and (output_dir is None or resume_from is None):
        raise ValueError("Continuation requires a parent checkpoint and child output")
    saved: dict[str, Any] | None = None
    checkpoint = None if resume_from is None else Path(resume_from).resolve()
    if checkpoint is not None:
        try:
            saved = checkpoints.read_checkpoint_details(checkpoint)
        except (ValueError, FileNotFoundError) as error:
            if extension is None:
                raise
            raise ValueError(
                "Continuation requires a complete learner checkpoint at "
                "checkpoints/<id>, including actor and learner payloads; "
                "actor exports and pruned folders cannot continue training"
            ) from error
        if saved["kind"] != "learner":
            raise ValueError(
                "Continuation requires a complete learner checkpoint at "
                "checkpoints/<id>; "
                "actors/<id> contains inference weights only"
                if extension is not None
                else "Resume requires a complete learner checkpoint"
            )
        # A checkpoint saved before Red Zone cannot resume here; refuse it
        # before any file (lock, panel or log) is touched.
        checkpoints._require_current_schemas(saved)  # pyright: ignore[reportPrivateUsage]
        inherited = _saved_config(saved)
        if config is not None and config.red_zone_depth != inherited.red_zone_depth:
            raise ValueError(
                f"Resume config declares red_zone_depth {config.red_zone_depth}, "
                f"but the saved run uses {inherited.red_zone_depth}. The depth is "
                "fixed for a run; start a new run to use another depth."
            )
        if config is not None and config_to_dict(config) != config_to_dict(inherited):
            raise ValueError("Resume config differs from the saved experiment")
        config = inherited
        root = checkpoint.parent.parent
        if (
            checkpoint.parent.name != "checkpoints"
            or not (root / "run_details.json").is_file()
        ):
            raise ValueError("Checkpoint is outside its original training run")
    else:
        if not isinstance(config, TrainConfig):
            raise TypeError("A new run requires TrainConfig")
        root = Path(str(output_dir)).resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError("output_dir must be new or empty")
    assert config is not None
    parent_root = root
    if extension is not None:
        root = Path(str(output_dir)).resolve()
        if root == parent_root or parent_root in root.parents:
            raise ValueError("Child output_dir must be outside the parent training run")
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError("output_dir must be new or empty")
        if extension["additional_env_steps"] % config.num_envs:
            raise ValueError("additional_env_steps must divide by num_envs exactly")
    if (
        saved is None
        and validation_opponents is not None
        and config.validation_opponents is not None
    ):
        raise ValueError("Declare validation_opponents once, in config or train")
    declared = (
        validation_opponents
        if validation_opponents is not None
        else config.validation_opponents
    )
    saved_continuation = (
        None if saved is None else saved["metadata"].get("continuation")
    )
    schedule = (
        make_training_schedule(
            total_env_steps=config.total_env_steps,
            num_envs=config.num_envs,
            curriculum=config.curriculum,
            score_threshold_curriculum=config.score_threshold_curriculum,
            early_history_capture=config.pinned_opponent_share > 0,
        )
        if saved_continuation is None
        else make_training_schedule(**saved_continuation["segment"]["root_schedule"])
    )
    # Prepare the content (every map checks the depth against its width)
    # before a new panel can be published, so a bad setting leaves no file.
    prepared = prepare_training_content(
        score_thresholds=schedule.score_thresholds,
        red_zone_depth=config.red_zone_depth,
    )
    panel_path = (
        Path(config.validation_panel)
        if config.validation_panel
        else parent_root / "validation_panel" / "panel.json"
    )
    if config.validation_panel or (saved is not None and panel_path.exists()):
        panel = validation.load_panel(
            panel_path, bindings=declared, red_zone_depth=config.red_zone_depth
        )
    elif declared is not None:
        if saved is not None:
            raise ValueError("Resume cannot introduce a new validation panel")
        panel = validation.create_panel(
            opponents=declared, output_dir=panel_path.parent
        )
    else:
        panel = None
    if config.slot_diagnostic and panel is None:
        raise ValueError("slot_diagnostic requires a frozen validation panel")
    if (
        config.slot_diagnostic
        and panel is not None
        and panel.schema_version == 2
        and not config.slot_diagnostic_actor
    ):
        raise ValueError(
            "A System panel needs an explicit slot_diagnostic_actor "
            "for this optional comparison"
        )
    if config.slot_diagnostic and config.slot_diagnostic_actor:
        checkpoints.artifact_identity(config.slot_diagnostic_actor)
    if config.purpose == "demonstration" and (panel is None or not panel.qualified):
        raise ValueError("Demonstration requires a qualified frozen validation panel")
    # Setup performs no real action. Resume replaces all template numerical values.
    state: Any
    if config.qmix is not None:
        from marl_battlegrounds.training import qmix_learner

        collection, state = qmix_learner.init_qmix_learner(
            schedule=schedule,
            seed=config.seed,
            prepared=prepared,
            shaping=config.shaping,
            shaping_coefficient=config.shaping_coefficient,
            shaping_mode=config.shaping_mode,
            metrics=config.metrics,
            recording=config.recording,
            pinned_opponent_share=config.pinned_opponent_share,
            pinned_opponent=config.pinned_opponent,
            qmix=config.qmix,
        )
    elif config.pqn is not None:
        from marl_battlegrounds.training import pqn_learner

        collection, state = pqn_learner.init_pqn_learner(
            schedule=schedule,
            seed=config.seed,
            prepared=prepared,
            shaping=config.shaping,
            shaping_coefficient=config.shaping_coefficient,
            shaping_mode=config.shaping_mode,
            metrics=config.metrics,
            recording=config.recording,
            pinned_opponent_share=config.pinned_opponent_share,
            pinned_opponent=config.pinned_opponent,
            pqn=config.pqn,
        )
    else:
        collection, state = learner.init_learner(
            schedule=schedule,
            seed=config.seed,
            prepared=prepared,
            ppo=config.ppo,
            method=config.method,
            shaping=config.shaping,
            shaping_coefficient=config.shaping_coefficient,
            shaping_mode=config.shaping_mode,
            metrics=config.metrics,
            recording=config.recording,
            pinned_opponent_share=config.pinned_opponent_share,
            pinned_opponent=config.pinned_opponent,
        )
    if saved_continuation is not None:
        import jax

        from marl_battlegrounds.training.curriculum import (
            _restore_continuation_schedule,  # pyright: ignore[reportPrivateUsage]
        )

        schedule = _restore_continuation_schedule(saved_continuation["segment"])
        carry = state.carry._replace(schedule=schedule.arrays)
        state = state._replace(carry=carry)
        collection = replace(
            collection, schedule=schedule, carry_spec=jax.eval_shape(_identity, carry)
        )
        from marl_battlegrounds.training._continuation_schedules import (
            continuation_collection,
            schedule_continuation,
        )

        collection = continuation_collection(
            collection, schedule_continuation(schedule)
        )
    if config.random_initialization_result is not None:
        from marl_battlegrounds.baselines.qmix import QMIX_TIE_RULE
        from marl_battlegrounds.evaluation.recording_identity import tree_digest

        current = state.carry.history.current_variables
        settings = config.qmix or config.pqn or config.ppo
        payload = (
            current.params
            if config.qmix is not None
            else checkpoints._pqn_actor_item(current.network)  # pyright: ignore[reportPrivateUsage]
            if config.pqn is not None
            else current
        )
        initial = {
            "kind": "actor",
            "actor_digest": tree_digest(payload),
            "input_scale": settings.input_scale,
            "spawn_frame": settings.spawn_frame,
            "schemas": checkpoints.checkpoint_schemas(config.method),
        }
        if not is_ppo_method(config.method):
            initial.update(epsilon=0.0, tie_rule=QMIX_TIE_RULE)
        initial_digest = checkpoints._inference_digest(initial)  # pyright: ignore[reportPrivateUsage]
        validation.read_random_initialization(
            config.random_initialization_result,
            actor_digest=initial_digest,
            seed_pairs=cast(int, config.random_diagnostic_seed_pairs),
            red_zone_depth=config.red_zone_depth,
        )
    # PPO keeps the argument-free call; QMIX also records Flashbax, PQN-VDN
    # records the same set as PPO.
    identity = (
        checkpoints.runtime_identity()
        if is_ppo_method(config.method)
        else checkpoints.runtime_identity(method=config.method)
    )
    source = cast(dict[str, Any], identity["source"])
    dependencies = identity["dependencies"]
    runtime = _runtime_details(str(source["package_version"]), config.num_envs)
    execution_identity_now = execution_identity(runtime=runtime["provenance"])
    extension_state = None
    if extension is not None:
        assert checkpoint is not None and saved is not None
        import jax

        compatibility = checkpoints.continuation_source_compatibility(
            saved["metadata"]["source"], source
        )
        template = jax.eval_shape(_identity, state)
        state = None
        parent_restore = checkpoints.restore_checkpoint(
            checkpoint,
            collection,
            template,
            expected_metadata={
                "config": config_to_dict(config),
                "source": source,
                "dependencies": dependencies,
                "execution": execution_identity_now,
            },
            ppo=config.ppo,
            method=config.method,
            qmix=config.qmix,
            pqn=config.pqn,
            _source_compatibility=compatibility,
        )
        validate_host_state(parent_root, parent_restore.details, panel=panel)
        config, collection, state, panel, declarations = _prepare_extension(
            config,
            collection,
            parent_restore,
            panel,
            extension,
            source,
        )
        declarations["continuation"]["source_compatibility"] = compatibility
        extension_state = parent_restore, declarations
    with ExitStack() as stack:
        prepared_recording = None
        if extension_state is not None and config.recording:
            prepared_recording = _prepare_extension_recording(extension_state[0], state)
            stack.callback(prepared_recording.close)
        # A child recording is fully checked before even its folder is created.
        root.mkdir(parents=True, exist_ok=True)
        stack.enter_context(run_lock(root))
        if extension_state is not None:
            parent_restore, declarations = extension_state
            owner = _start_extension(
                root=root,
                config=config,
                collection=collection,
                state=state,
                panel=panel,
                restored=parent_restore,
                declarations=declarations,
                source=source,
                dependencies=dependencies,
                execution=execution_identity_now,
                runtime=runtime,
                started_at=started_at,
                attempt_started=attempt_started,
                stack=stack,
                prepared_recording=prepared_recording,
            )
            state = parent_restore = extension_state = None
            return owner.execute()
        writer = None
        metadata: dict[str, Any]
        restored: checkpoints.RestoredCheckpoint | None = None
        if saved is not None:
            assert checkpoint is not None
            if not is_ppo_method(config.method):
                import jax

                # Release the fresh learner before restore reads the saved one.
                state = jax.eval_shape(_identity, state)
            restored = checkpoints.restore_checkpoint(
                checkpoint,
                collection,
                state,
                expected_metadata={
                    "config": config_to_dict(config),
                    "source": source,
                    "dependencies": dependencies,
                    "execution": execution_identity_now,
                },
                ppo=config.ppo,
                method=config.method,
                qmix=config.qmix,
                pqn=config.pqn,
            )
            metadata = dict(restored.details["metadata"])
            cursors = metadata.get("log_cursors", {})
            if set(cursors) != {"training_updates.jsonl"}:
                raise ValueError("Checkpoint has no valid training-log recovery cursor")
            validate_log_cursor(
                root / "training_updates.jsonl", cursors["training_updates.jsonl"]
            )
            run_details = json.loads((root / "run_details.json").read_text())
            if run_details["run_id"] != metadata["run_id"]:
                raise ValueError("Checkpoint belongs to a different training run")
            if run_details.get("schemas") != restored.details["schemas"]:
                raise ValueError("Run and checkpoint schema versions differ")
            if run_details.get("execution") != metadata["execution"]:
                raise ValueError("Run and checkpoint execution identities differ")
            if run_details["panel_digest"] != (None if panel is None else panel.digest):
                raise ValueError("Frozen validation panel differs from the saved run")
            state = restored.state
            host = metadata["host_state"]
            if host["env_steps"] != int(
                state.carry.progress.rounds
            ) * config.num_envs or host["completed_updates"] != int(
                state.completed_updates
            ):
                raise ValueError("Checkpoint host counters differ from learner state")
            if config.qmix is not None:
                from marl_battlegrounds.training.qmix_learner import (
                    _bank_size,  # pyright: ignore[reportPrivateUsage]
                )

                exposure = host.get("sampled_exposure")
                if not isinstance(exposure, dict) or len(
                    cast(list[int], cast(dict[str, Any], exposure).get("by_source", []))
                ) != _bank_size(state.carry):
                    raise ValueError(
                        "Saved sampled exposure does not match the source bank"
                    )
            if config.pqn is not None:
                from marl_battlegrounds.training.pqn_learner import (
                    _bank_size as _pqn_bank_size,  # pyright: ignore[reportPrivateUsage]
                )

                exposure = host.get("used_exposure")
                if not isinstance(exposure, dict) or len(
                    cast(list[int], cast(dict[str, Any], exposure).get("by_source", []))
                ) != _pqn_bank_size(state.carry):
                    raise ValueError(
                        "Saved used exposure does not match the source bank"
                    )
            validate_host_state(root, restored.details, panel=panel)
            if (root / "status.json").is_file():
                prior_status = json.loads((root / "status.json").read_text())
                prior_elapsed = prior_status.get("elapsed_seconds", 0.0)
                if (
                    prior_status.get("run_id") == metadata["run_id"]
                    and isinstance(prior_elapsed, (int, float))
                    and math.isfinite(prior_elapsed)
                ):
                    host["elapsed_seconds"] = max(
                        host["elapsed_seconds"], prior_elapsed
                    )
            writer = checkpoints.resume_recording(restored, root)
            if writer is not None:
                stack.callback(writer.close)
            abandoned = preserve_log_suffix(
                root / "training_updates.jsonl",
                cursors["training_updates.jsonl"],
                root / "attempts" / metadata["attempt_id"],
            )
            if abandoned is not None:
                append_jsonl(
                    root / "run_events.jsonl",
                    {
                        "event": "abandoned_updates_preserved",
                        "time_utc": utc_now(),
                        "attempt_id": metadata["attempt_id"],
                        "path": str(abandoned),
                        "restore_checkpoint": checkpoint.name,
                    },
                    durable=True,
                )
            restore_log_cursor(
                root / "training_updates.jsonl", cursors["training_updates.jsonl"]
            )
            _preserve_reports(root, metadata["attempt_id"])
            atomic_json(
                root / "validation_results.json",
                host.get("routine_results", []) + host.get("confirmation_results", []),
            )
            if config.random_diagnostic_seed_pairs is not None:
                atomic_json(root / "random_diagnostics.json", host["random_results"])
            atomic_json(root / "selection.json", host.get("selection"))
            atomic_json(root / "exposure.json", None)
            atomic_json(root / "slot_diagnostic.json", None)
            # Publish unfinished status before clearing recovery ownership. A
            # crash before the new attempt starts must not revive old completion.
            atomic_json(
                root / "status.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "attempt_id": metadata["attempt_id"],
                    "status": "incomplete",
                    "phase": "resuming",
                    "env_steps": host["env_steps"],
                    "total_env_steps": config.total_env_steps,
                    "completed_updates": host["completed_updates"],
                    "elapsed_seconds": host["elapsed_seconds"],
                    "latest_checkpoint": str(checkpoint),
                    "pending_task": host["pending"],
                    "process": process_identity(),
                    "updated_at": utc_now(),
                },
            )
            checkpoints.finish_checkpoint_recovery(restored, root)
            metadata["parent_checkpoint"] = checkpoint.name
            metadata["attempt_id"] = uuid.uuid4().hex
        else:
            metadata = {
                "run_id": uuid.uuid4().hex,
                "attempt_id": uuid.uuid4().hex,
                "parent_checkpoint": None,
                "config": config_to_dict(config),
                "source": source,
                "dependencies": dependencies,
                "execution": execution_identity_now,
                "host_state": {},
                "recording": None,
            }
            if config.recording:
                from marl_battlegrounds.evaluation.recording_identity import (
                    normalize_system_registration,
                )
                from marl_battlegrounds.evaluation.run_writer import RunWriter

                policies: dict[str, object] = {
                    name: normalize_system_registration(system, phase="training")[1]
                    for name, system in (
                        ("team_a", collection.actor),
                        ("team_b", collection.opponent),
                    )
                }
                # The pass names the pinned System; the writer and the
                # checkpoint must hold the same details, which resume compares.
                recording_details: dict[str, object] = (
                    {}
                    if collection.pinned_opponent is None
                    else {"pinned_opponent": collection.pinned_opponent}
                )
                writer = RunWriter(
                    root / "episodes",
                    phase="training",
                    policies=policies,
                    details=recording_details or None,
                )
                stack.callback(writer.close)
                metadata["recording"] = {
                    "relative_path": writer.run_dir.relative_to(root).as_posix(),
                    "phase": "training",
                    "pass_id": "1",
                    "policies": policies,
                    "checkpoint_id": None,
                    "details": recording_details,
                }
            atomic_json(
                root / "run_details.json",
                {
                    "schema_version": 1,
                    "run_id": metadata["run_id"],
                    "created_at": started_at,
                    "initial_setup_seconds": time.monotonic() - attempt_started,
                    "config": metadata["config"],
                    "source": source,
                    "dependencies": dependencies,
                    "schemas": checkpoints.checkpoint_schemas(config.method),
                    "execution": execution_identity_now,
                    "runtime": runtime,
                    "content_binding": collection.binding.model_dump(mode="json"),
                    "schedule": dict(schedule.rounding_report),
                    "panel_digest": None if panel is None else panel.digest,
                    "process": process_identity(),
                },
            )
        execution = _Run(
            root,
            config,
            collection,
            state,
            metadata,
            writer,
            panel,
            checkpoint,
            attempt_started=attempt_started,
            runtime=runtime,
        )
        # _Run now owns the learner. Drop these references so the first state
        # (for QMIX, a whole replay) is freed once the run moves past it.
        state = restored = None
        return execution.execute()


def _prepare_extension(
    config: TrainConfig,
    collection: TrainingCollection,
    restored: RestoredCheckpoint,
    panel: FrozenPanel | None,
    request: dict[str, Any],
    source: dict[str, Any],
) -> tuple[TrainConfig, TrainingCollection, Any, FrozenPanel | None, dict[str, Any]]:
    """Check and build a child boundary without writing or taking a real step.

    restored has passed full parent restore and host checks. All numerical
    changes are limited to declared segment accounting and future schedules.
    The result contains checked declarations; the runner publishes them before
    calling the existing execution loop. Parent files remain read only.
    """
    from copy import deepcopy

    import numpy as np

    from marl_battlegrounds.training import validation
    from marl_battlegrounds.training._continuation_schedules import (
        continuation_collection,
        continuation_state,
        learner_continuation,
        resolve_learner_continuation,
    )
    from marl_battlegrounds.training._selection_evidence import (
        freeze_inherited_candidates,
    )
    from marl_battlegrounds.training.collection import (
        _begin_training_segment,  # pyright: ignore[reportPrivateUsage]
        _continuation_history_thresholds,  # pyright: ignore[reportPrivateUsage]
    )
    from marl_battlegrounds.training.curriculum import (
        _continuation_details,  # pyright: ignore[reportPrivateUsage]
        _make_continuation_schedule,  # pyright: ignore[reportPrivateUsage]
        _restore_continuation_schedule,  # pyright: ignore[reportPrivateUsage]
    )

    state = restored.state
    saved = restored.details
    parent = saved["metadata"]
    prior = parent.get("continuation")
    start = saved["counters"]["env_steps"]
    total = start + request["additional_env_steps"]
    changes = request["changes"]
    history = state.carry.history
    occupied = int(history.count)
    frozen_epsilon = (
        ()
        if is_ppo_method(config.method)
        else tuple(
            float(value)
            for value in np.asarray(history.historical_variables.epsilon)[:occupied]
        )
    )
    learner_context = resolve_learner_continuation(
        method=config.method,
        settings=config_to_dict(config)[method_settings_field(config.method)],
        counters=saved["counters"],
        num_envs=config.num_envs,
        total_env_steps=total,
        changes=changes,
        parent=None if prior is None else prior["learner"],
        frozen_epsilon=frozen_epsilon,
        original_total_env_steps=config.total_env_steps,
    )
    parent_validation = validation.saved_validation_declaration(
        parent, panel
    ) or validation.run_validation_declaration(config_to_dict(config), panel)
    validation_changes: Any = changes.get("validation", {})
    if not isinstance(validation_changes, Mapping):
        raise ValueError("validation changes must be a mapping")
    validation_changes = cast(dict[str, Any], validation_changes)
    if "panel" in validation_changes:
        panel = validation.load_panel(
            Path(validation_changes["panel"]).resolve(),
            bindings=request.get("validation_bindings"),
            red_zone_depth=config.red_zone_depth,
        )
    root_total = (
        config.total_env_steps
        if prior is None
        else prior["segment"]["root_schedule"]["total_env_steps"]
    )
    config = _child_config(
        {
            **config_to_dict(config),
            "validation_panel": None if panel is None else str(panel.path.absolute()),
            "validation_opponents": config.validation_opponents
            if panel is None
            else None,
            "checkpoint_env_steps": [
                value for value in config.checkpoint_env_steps if start < value <= total
            ],
        },
        total_env_steps=total,
        root_total_env_steps=root_total,
    )
    context: dict[str, Any] = {
        "schema_version": 1,
        "parent_checkpoint": str(restored.path),
        "parent_checkpoint_id": saved["checkpoint_id"],
        "parent_run_id": parent["run_id"],
        "root_run_id": parent["run_id"] if prior is None else prior["root_run_id"],
        "parent_source": parent["source"],
        "child_source": source,
        "start_env_steps": start,
        "additional_env_steps": request["additional_env_steps"],
        "resulting_total_env_steps": total,
        "changes": changes,
        "learner": learner_context,
        "parent_validation_declaration": parent_validation,
        "parent_statistics": deepcopy(parent["host_state"]),
        "inherited_candidates": [],
    }
    declared = validation.run_validation_declaration(
        config_to_dict(config), panel, continuation=context
    )
    context["inherited_candidates"] = freeze_inherited_candidates(
        restored.path, declaration=declared
    )
    thresholds = None
    if "history_capture_env_steps" in changes:
        wanted_raw = changes["history_capture_env_steps"]
        if not isinstance(wanted_raw, list):
            raise ValueError("history_capture_env_steps must be a list")
        wanted = cast(list[object], wanted_raw)
        if any(type(value) is not int for value in wanted):
            raise ValueError("history_capture_env_steps must contain whole numbers")
        steps = cast(list[int], wanted)
        if any(
            not start < value <= total or value % config.num_envs for value in steps
        ):
            raise ValueError(
                "History targets must follow the saved step, fit the child end, "
                "and contain whole environment rounds"
            )
        thresholds = _continuation_history_thresholds(
            state.carry,
            future_rounds=tuple(value // config.num_envs for value in steps),
        )
    schedule = _make_continuation_schedule(
        collection.schedule,
        completed_rounds=start // config.num_envs,
        additional_env_steps=request["additional_env_steps"],
        history_threshold_rounds=thresholds,
    )
    collection, carry = _begin_training_segment(
        collection, state.carry, schedule=schedule
    )
    segment = _continuation_details(collection.schedule)
    assert segment is not None
    segment["learner"] = learner_context
    schedule = _restore_continuation_schedule(segment)
    collection = replace(collection, schedule=schedule)
    rules = learner_continuation(learner_context)
    collection = continuation_collection(collection, rules)
    state = state._replace(carry=carry)
    if "exploration" in changes:
        state = continuation_state(
            state,
            rules,
            num_envs=config.num_envs,
            initial_rounds=0 if config.pqn is None else config.pqn.initial_rounds,
        )
    context["segment"] = _continuation_details(schedule)
    return (
        config,
        collection,
        state,
        panel,
        {
            "continuation": context,
            "validation_declaration": declared,
            "training_lineage": {
                "root_run_id": context["root_run_id"],
                "parent_run_id": parent["run_id"],
            },
        },
    )


def _prepare_extension_recording(
    restored: RestoredCheckpoint,
    state: Any,  # noqa: ANN401
) -> PreparedRecordingFork:
    """Check a parent's full recording boundary and retain only unfinished games.

    restored is the checked full parent checkpoint; state is its prepared child
    state before another action. This reads the parent's files and uses owned
    temporary streams but creates no child path. The caller must close the
    returned preparation, including when child folder or lock setup fails.
    """
    import numpy as np

    from marl_battlegrounds.evaluation.recording_checkpoint import (
        prepare_recording_fork,
    )

    record = restored.details["metadata"]["recording"]
    mask = np.asarray(
        ~state.carry.state.done.done & state.carry.tracking.first_transition_seen
    )
    episodes = np.asarray(state.carry.state.episode_id)[mask].tolist()
    return prepare_recording_fork(
        restored.path.parent.parent / record["relative_path"],
        restored.details["recording_token"],
        episode_ids=episodes,
        policies=record["policies"],
    )


def _start_extension(
    *,
    root: Path,
    config: TrainConfig,
    collection: TrainingCollection,
    state: Any,  # noqa: ANN401
    panel: FrozenPanel | None,
    restored: RestoredCheckpoint,
    declarations: dict[str, Any],
    source: dict[str, Any],
    dependencies: object,
    execution: dict[str, Any],
    runtime: dict[str, Any],
    started_at: str,
    attempt_started: float,
    stack: ExitStack,
    prepared_recording: PreparedRecordingFork | None = None,
) -> _Run:
    """Publish one checked child setup and enter the shared execution owner.

    All declarations, parent numerical state and recording were checked before
    creating root. prepared_recording owns checked unfinished replay streams;
    stack owns it and any child recording writer. This never opens the parent
    writer for mutation or copies its completed replay/log history.
    """
    from copy import deepcopy

    from marl_battlegrounds.training import checkpoints

    host = deepcopy(restored.details["metadata"]["host_state"])
    host.update(
        routine_results=[],
        confirmation_results=[],
        random_results=[],
        validation_games=0,
        validation_seconds=0.0,
        actors={},
        pending=None,
        selected_actor=None,
        final_actor=None,
        slot_complete=False,
        selection=None,
        recovery_checkpoints=[],
    )
    # These totals use each game's original reset-time stage, like cumulative
    # exposure. Keep them across child accounting segments and add new finishes.
    metadata: dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "attempt_id": uuid.uuid4().hex,
        "parent_checkpoint": None,
        "config": config_to_dict(config),
        "source": source,
        "dependencies": dependencies,
        "execution": execution,
        "host_state": host,
        "recording": None,
        **declarations,
    }
    writer = None
    if config.recording:
        from marl_battlegrounds.evaluation.recording_checkpoint import (
            attach_recording_fork,
        )
        from marl_battlegrounds.evaluation.run_writer import RunWriter

        if prepared_recording is None:
            raise ValueError("Child recording must be checked before creating output")
        record = restored.details["metadata"]["recording"]
        writer = RunWriter(
            root / "episodes",
            phase="training",
            pass_id=record["pass_id"],
            policies=record["policies"],
            checkpoint_id=record["checkpoint_id"],
            details=record["details"] or None,
        )
        stack.callback(writer.close)
        metadata["continuation"]["recording"] = attach_recording_fork(
            writer, prepared_recording
        )
        metadata["recording"] = {
            **record,
            "relative_path": writer.run_dir.relative_to(root).as_posix(),
        }
    atomic_json(
        root / "run_details.json",
        {
            "schema_version": 1,
            "run_id": metadata["run_id"],
            "created_at": started_at,
            "initial_setup_seconds": time.monotonic() - attempt_started,
            "config": metadata["config"],
            "source": source,
            "dependencies": dependencies,
            "schemas": checkpoints.checkpoint_schemas(config.method),
            "execution": execution,
            "runtime": runtime,
            "content_binding": collection.binding.model_dump(mode="json"),
            "schedule": dict(collection.schedule.rounding_report),
            "panel_digest": None if panel is None else panel.digest,
            "process": process_identity(),
            **declarations,
        },
    )
    return _Run(
        root,
        config,
        collection,
        state,
        metadata,
        writer,
        panel,
        None,
        attempt_started=attempt_started,
        runtime=runtime,
    )


def _crosses_interval(before: int, after: int, interval: int) -> bool:
    """Say whether an update count crossed a multiple of the save interval.

    before and after are completed_updates around one block (PPO updates, or
    QMIX and PQN-VDN optimizer steps); interval is the positive save interval.
    PPO adds one update per block, so this is its historical
    ``after % interval == 0`` rule. A QMIX block adds epochs steps and a
    PQN-VDN learning block adds epochs times num_minibatches steps (an initial
    PQN-VDN chunk adds none), so 24 to 28 crosses 25 while 20 to 24 does not.

    Examples
    --------
    >>> _crosses_interval(24, 28, 25), _crosses_interval(20, 24, 25)
    (True, False)
    """
    return after // interval != before // interval


def _identity(value: Any) -> Any:  # noqa: ANN401
    """Return value unchanged; traced by jax.eval_shape for shape-only copies."""
    return value


def _runtime_details(package_version: str, num_envs: int) -> dict[str, Any]:
    """Read M8 runtime facts and declared device selectors once per attempt.

    package_version is the verified source version and num_envs is the declared
    positive batch. Reuse M8's backend/device/precision record after setup has
    selected JAX. Environment values describe selectors, not verified physical
    GPU identities; the frozen launch package owns its UUID/PCI verification.
    No model call, timing, transfer, file write or driver query occurs here.
    """
    from marl_battlegrounds.evaluation.runtime_provenance import (
        capture_runtime_provenance,
    )

    return {
        "provenance": capture_runtime_provenance(
            package_version, num_envs=num_envs, policy_execution_included=True
        ).model_dump(mode="json"),
        "declared_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_DISABLE_JIT",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
            )
        },
    }


def _memory_snapshot() -> dict[str, Any]:
    """Read process peak RAM and JAX allocator counters after an attempt ends.

    Return optional measurements and explicit read errors. RAM is the process
    lifetime high-water mark, including earlier work in a reused Python process.
    JAX counters exclude driver/display memory and CPU worker processes. Missing
    allocator statistics are None. This performs no GPU synchronization, array
    transfer, model call, polling or file write. A failed measurement cannot hide
    the attempt's original failure. Call only after the learner loop/report work.
    """
    result: dict[str, Any] = {
        "process_peak_ram_bytes": None,
        "jax_allocator": None,
        "errors": [],
        "scope": (
            "Process lifetime, including earlier work and compilation. JAX "
            "allocator counters exclude driver/display memory and CPU workers."
        ),
    }
    try:
        import resource
        import sys

        scale = 1 if sys.platform == "darwin" else 1024
        result["process_peak_ram_bytes"] = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
        )
    except Exception as error:
        result["errors"].append(f"Process RAM: {type(error).__name__}: {error}")
    try:
        import jax

        result["jax_allocator"] = jax.devices()[0].memory_stats()
    except Exception as error:
        result["errors"].append(f"JAX allocator: {type(error).__name__}: {error}")
    return result


def _random_progress(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize already verified Random games for the human progress display.

    results contains this run's completed Random captures, including an optional
    shared initialization. Return the latest capture's game counts, equal-map
    mean kills and deaths, and its change from initialization. Kills are read
    from the columns analysis._kill_columns names for each record: recorded
    kills for records that carry a Red Zone depth (a Red Zone death gives 2
    points but is still one kill), the score columns for older records, where
    points and kills were equal. Missing combat values remain None. Capture
    times are the times when the actor was saved, not the time when its
    evaluation finished. Empty results return None. This only reads small host
    dictionaries; it opens no files, touches no device arrays and changes
    neither evidence nor checkpoint state.
    """
    from marl_battlegrounds.training.analysis import (
        _kill_columns,  # pyright: ignore[reportPrivateUsage]
    )

    if not results:
        return None

    def mean_scores(record: dict[str, Any]) -> tuple[float | None, float | None]:
        """Average map kills for and against equally; missing values stay None."""
        cells = record.get("cells", [])
        if not cells:
            return None, None
        team_a, team_b = (f"mean_{name}" for name in _kill_columns(record))
        first = [cell.get(team_a) for cell in cells]
        second = [cell.get(team_b) for cell in cells]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in (*first, *second)
        ):
            return None, None
        return math.fsum(first) / len(cells), math.fsum(second) / len(cells)

    latest = max(results, key=lambda record: record["env_steps"])
    initial = next((record for record in results if record["env_steps"] == 0), None)
    kills, deaths = mean_scores(latest)
    margin = None if kills is None or deaths is None else kills - deaths
    initial_margin = None
    if initial is not None:
        initial_kills, initial_deaths = mean_scores(initial)
        if initial_kills is not None and initial_deaths is not None:
            initial_margin = initial_kills - initial_deaths
    return {
        "opponent": "Random",
        "task_id": latest["task_id"],
        "initial_task_id": None if initial is None else initial["task_id"],
        "env_steps": latest["env_steps"],
        "games": latest["games"],
        "wall_seconds": latest["wall_seconds"],
        "training_seconds": latest["training_seconds"],
        "score": latest["score"],
        **{
            name: sum(cell[name] for cell in latest["cells"])
            for name in ("wins", "draws", "losses")
        },
        "mean_kills_for": kills,
        "mean_kills_against": deaths,
        "kill_margin": margin,
        "initial_kill_margin": initial_margin,
        "kill_margin_change": None
        if margin is None or initial_margin is None
        else margin - initial_margin,
    }


def _finite_or_marked(value: object, name: str, markers: dict[str, str]) -> object:
    """Copy a float or nested list of floats, replacing nonfinite values by None.

    Parameters
    ----------
    value : object
        A Python float, bool or int, or a nested list of them (for example a
        rejected block's (epochs, minibatches) losses after ``tolist()``).
    name : str
        Field name used in the markers, such as ``"loss"``.
    markers : dict[str, str]
        Mapping this function adds to: one entry per replaced value, keyed
        by the name plus its indices, such as ``"loss[1,0]"``, with the value
        "NaN", "+Inf" or "-Inf". Existing entries are kept.

    Returns
    -------
    object
        The same structure with every nonfinite float replaced by None.
        Finite floats, booleans and integers are returned unchanged; an empty
        list stays empty.

    Notes
    -----
    Host-only and pure apart from adding to markers. The shared JSON writer
    refuses NaN, so the event keeps every value without disguising it.
    """

    def clean(item: object, index: tuple[int, ...]) -> object:
        """Replace one nonfinite float in place of its position."""
        if isinstance(item, list):
            return [
                clean(entry, (*index, position))
                for position, entry in enumerate(cast(list[object], item))
            ]
        if isinstance(item, float) and not math.isfinite(item):
            label = name if not index else f"{name}[{','.join(map(str, index))}]"
            markers[label] = (
                "NaN" if math.isnan(item) else ("+Inf" if item > 0 else "-Inf")
            )
            return None
        return item

    return clean(value, ())


def _rejected_block_facts(result: Any) -> dict[str, object]:  # noqa: ANN401
    """Describe a rejected PQN-VDN block for the learner_update_rejected event.

    Parameters
    ----------
    result : PQNUpdateResult
        Host copy of the rejected block's result.

    Returns
    -------
    dict
        reason (int32 failure code), generated_transitions and
        active_samples from the kept summary, task_reward_sum and
        shaping_reward_sum, and the attempted (epochs, minibatches) step
        flags (steps_performed, steps_finite) and losses. Finite values are
        written as they are; each nonfinite value is None and the
        "nonfinite" object names it with its marker.

    Notes
    -----
    Host-only; the runner writes the event, then attempt_failed, then raises.
    No retry happens and no host count changes.
    """
    import numpy as np

    summary, metrics = result.summary, result.metrics
    markers: dict[str, str] = {}
    facts: dict[str, object] = {
        "reason": int(result.failure_reason),
        "generated_transitions": int(summary.real_transitions),
        "active_samples": int(summary.active_samples),
        "task_reward_sum": _finite_or_marked(
            float(summary.task_reward_sum), "task_reward_sum", markers
        ),
        "shaping_reward_sum": _finite_or_marked(
            float(summary.shaping_reward_sum), "shaping_reward_sum", markers
        ),
        "steps_performed": np.asarray(metrics.performed).tolist(),
        "steps_finite": np.asarray(metrics.finite).tolist(),
        "losses": _finite_or_marked(
            np.asarray(metrics.loss, dtype=np.float64).tolist(), "losses", markers
        ),
    }
    facts["nonfinite"] = markers
    return facts


class _Run:
    """Own one locked process attempt and its small host bookkeeping.

    The public train function validates setup/recovery first. This object holds
    one learner state and stable compiled updater; it never stores a rollout
    after its update. Files and validation are performed between numerical calls.
    A QMIX run keeps exact Python-integer block, update and sampled-use totals
    in its host state instead of PPO's policy and value sample counts; a
    PQN-VDN run keeps block, update and used-row totals the same way.
    """

    def __init__(
        self,
        root: Path,
        config: TrainConfig,
        collection: TrainingCollection,
        state: Any,  # noqa: ANN401
        metadata: dict[str, Any],
        writer: RunWriter | None,
        panel: FrozenPanel | None,
        checkpoint: Path | None,
        *,
        attempt_started: float,
        runtime: dict[str, Any],
    ) -> None:
        """Attach validated owners and reconstruct host counters from a checkpoint."""
        from functools import partial

        import jax

        from marl_battlegrounds.training._compilation import training_compiler_options
        from marl_battlegrounds.training._continuation_schedules import (
            continuation_config,
            schedule_continuation,
        )
        from marl_battlegrounds.training._selection_evidence import (
            read_inherited_candidates,
        )
        from marl_battlegrounds.training.learner import update_learner
        from marl_battlegrounds.training.validation import (
            resolve_validation_schedule,
            saved_validation_declaration,
        )

        self.continuation = schedule_continuation(collection.schedule)
        self.validation_declaration = saved_validation_declaration(metadata, panel)
        self.inherited: dict[str, Any] = (
            {"records": [], "actors": {}, "used_roots": []}
            if self.validation_declaration is None
            else read_inherited_candidates(
                metadata["continuation"]["inherited_candidates"],
                declaration=self.validation_declaration,
            )
        )
        self.ppo_reporting = (
            continuation_config(config.ppo, self.continuation)
            if is_ppo_method(config.method)
            else config.ppo
        )
        self.qmix_reporting = (
            None
            if config.qmix is None
            else continuation_config(config.qmix, self.continuation)
        )
        self.qmix = config.qmix
        self.pqn = config.pqn
        self.rollout_length = (
            self.qmix.rollout_length
            if self.qmix is not None
            else self.pqn.rollout_length
            if self.pqn is not None
            else config.ppo.rollout_length
        )
        self.planned_learning_blocks = (
            0
            if self.pqn is None
            else cast(int, self.continuation.pqn_planned_learning_blocks)
            if self.continuation is not None
            else pqn_planned_learning_blocks(
                config.total_env_steps // config.num_envs, self.pqn
            )
        )
        update_context = (
            {} if self.continuation is None else {"continuation": self.continuation}
        )
        self.root, self.config = root, config
        self.collection, self.state = collection, state
        self.metadata, self.writer, self.panel = metadata, writer, panel
        self.runtime = runtime
        self.host = metadata["host_state"]
        defaults: dict[str, Any] = {
            "env_steps": 0,
            "completed_updates": 0,
            "actor_decisions": 0,
            "used_policy_samples": 0,
            "used_value_samples": 0,
            "training_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "routine_results": [],
            "confirmation_results": [],
            "random_results": [],
            "actors": {},
            "pending": None,
            "selected_actor": None,
            "final_actor": None,
            "validation_seconds": 0.0,
            "validation_games": 0,
            "save_seconds": 0.0,
            "saves": 0,
            "report_seconds": None,
            "slot_complete": False,
            "selection": None,
            "recovery_checkpoints": [],
        }
        if self.qmix is not None:
            from marl_battlegrounds.training.qmix_learner import (
                _bank_size,  # pyright: ignore[reportPrivateUsage]
            )

            del defaults["used_policy_samples"], defaults["used_value_samples"]
            defaults.update(
                completed_blocks=0,
                learning_blocks=0,
                sampled_sequences=0,
                used_td_pairs=0,
                used_agent_utilities=0,
                sampled_exposure={
                    "by_stage": [0] * 17,
                    "by_source": [0] * _bank_size(state.carry),
                    "by_opponent": [0] * 21,
                },
            )
        if self.pqn is not None:
            from marl_battlegrounds.training.pqn_learner import (
                _bank_size as _pqn_bank_size,  # pyright: ignore[reportPrivateUsage]
            )

            del defaults["used_policy_samples"], defaults["used_value_samples"]
            defaults.update(
                completed_blocks=0,
                learning_blocks=0,
                used_sequences=0,
                used_td_pairs=0,
                used_agent_utilities=0,
                used_prefix_td_pairs=0,
                used_exposure={
                    "by_stage": [0] * 17,
                    "by_source": [0] * _pqn_bank_size(state.carry),
                    "by_opponent": [0] * 21,
                },
            )
        for name, default in defaults.items():
            self.host.setdefault(name, default)
        self.checkpoint = checkpoint
        if (
            checkpoint is not None
            and checkpoint.name not in self.host["recovery_checkpoints"]
        ):
            self.host["recovery_checkpoints"].append(checkpoint.name)
        self.start = attempt_started
        self.prior_elapsed = self.host["elapsed_seconds"]
        details = json.loads((self.root / "run_details.json").read_text())
        self.created_at = datetime.fromisoformat(details["created_at"]).timestamp()
        self.reporter = ProgressReporter(enabled=config.verbose)
        self.speed_estimate = TrainingSpeedEstimate() if config.verbose else None
        if self.qmix is not None:
            from marl_battlegrounds.training.qmix_learner import update_qmix_learner

            self.updater = cast(
                "_Updater",
                jax.jit(
                    partial(update_qmix_learner, qmix=self.qmix, **update_context),
                    compiler_options=training_compiler_options(),
                ),
            )
        elif self.pqn is not None:
            from marl_battlegrounds.training.pqn_learner import update_pqn_learner

            # The planned count is a static Python int bound once per run.
            self.updater = cast(
                "_Updater",
                jax.jit(
                    partial(
                        update_pqn_learner,
                        pqn=self.pqn,
                        planned_learning_blocks=self.planned_learning_blocks,
                        **update_context,
                    ),
                    compiler_options=training_compiler_options(),
                ),
            )
        else:
            self.updater = cast(
                "_Updater",
                jax.jit(
                    partial(
                        update_learner,
                        ppo=config.ppo,
                        method=config.method,
                        **update_context,
                    ),
                    compiler_options=training_compiler_options(),
                ),
            )
        self.validation_steps: set[int] = (
            {point["env_steps"] for point in self.validation_declaration["points"]}
            if self.validation_declaration is not None
            else {
                point.env_steps
                for point in resolve_validation_schedule(
                    config.total_env_steps,
                    config.num_envs,
                    rollout_length=self.rollout_length,
                    fractions=config.validation_fractions,
                    initial_rounds=0 if self.pqn is None else self.pqn.initial_rounds,
                )
            }
            if panel is not None
            else set()
        )
        self.random_steps: set[int] = (
            {0, *config.checkpoint_env_steps, config.total_env_steps}
            if config.random_diagnostic_seed_pairs is not None
            else set()
        )
        if self.continuation is not None:
            self.random_steps = {
                value
                for value in self.random_steps
                if value > metadata["continuation"]["start_env_steps"]
            }
        self.status: dict[str, Any] = {}
        if config.random_diagnostic_seed_pairs is not None:
            self.status["random_validation"] = _random_progress(
                self.host["random_results"]
            )

    def event(self, event_name: str, **facts: object) -> None:
        """Durably append one lifecycle event without changing scientific counters."""
        append_jsonl(
            self.root / "run_events.jsonl",
            {
                "event": event_name,
                "time_utc": utc_now(),
                "attempt_id": self.metadata["attempt_id"],
                "env_steps": self.host["env_steps"],
                **facts,
            },
            durable=True,
        )

    def set_status(self, phase: str, **facts: object) -> None:
        """Publish host counters and the requested reset-time K without device reads."""
        self.host["elapsed_seconds"] = (
            self.prior_elapsed + time.monotonic() - self.start
        )
        training_seconds = self.host["training_seconds"]
        rate = self.host["env_steps"] / training_seconds if training_seconds else None
        report = self.collection.schedule.rounding_report
        thresholds = report.get("score_thresholds", (20,))
        ends = cast(tuple[int, ...], report["cumulative_round_ends"])
        stage = min(
            sum(end * self.config.num_envs <= self.host["env_steps"] for end in ends),
            len(ends) - 1,
        )
        requested_threshold = (
            cast(tuple[int, ...], thresholds)[stage]
            if self.config.score_threshold_curriculum
            else 20
        )
        self.status.update(
            {
                "schema_version": 1,
                "run_id": self.metadata["run_id"],
                "attempt_id": self.metadata["attempt_id"],
                "process": process_identity(),
                "phase": phase,
                "status": "running",
                "total_env_steps": self.config.total_env_steps,
                "env_steps": self.host["env_steps"],
                "score_threshold": requested_threshold,
                "completed_updates": self.host["completed_updates"],
                "final_actor": self.host["final_actor"],
                "selected_actor": self.host["selected_actor"],
                "elapsed_seconds": self.host["elapsed_seconds"],
                "wall_seconds": max(0.0, time.time() - self.created_at),
                "transitions_per_second": rate,
                "estimated_transitions_per_second": None
                if self.speed_estimate is None
                else self.speed_estimate.rate,
                "latest_checkpoint": None
                if self.checkpoint is None
                else str(self.checkpoint),
                "pending_task": self.host["pending"],
                "pending_work_seconds": 0.0
                if phase == "complete"
                else self.pending_seconds()
                if self.config.verbose
                else None,
                "updated_at": utc_now(),
                **facts,
            }
        )
        atomic_json(self.root / "status.json", self.status)
        self.reporter.report(self.status)

    def pending_seconds(self) -> float | None:
        """Estimate output time from completed phase timings, never new GPU work.

        Future confirmation allows its maximum three candidates. Completed
        validation games estimate final-diagnostic cost too; the ETA is an
        approximation, not a promised finish time. Unknown phase costs give None.
        """
        if not self.host["saves"] or self.host["report_seconds"] is None:
            return None
        if self.continuation is not None:
            from marl_battlegrounds.training._continuation_schedules import (
                continuation_counts,
            )

            _, learning = continuation_counts(
                self.config.total_env_steps // self.config.num_envs,
                continuation=self.continuation,
                rollout_length=self.rollout_length,
                initial_rounds=0 if self.pqn is None else self.pqn.initial_rounds,
                minimum=0 if self.qmix is None else self.qmix.min_buffer_size,
            )
            if self.pqn is not None:
                remaining_updates = (
                    (learning - self.host["learning_blocks"])
                    * self.pqn.epochs
                    * self.pqn.num_minibatches
                )
            elif self.qmix is not None:
                remaining_updates = (
                    learning - self.host["learning_blocks"]
                ) * self.qmix.epochs
            else:
                remaining_updates = learning - self.host["completed_updates"]
        elif self.pqn is not None:
            remaining_updates = (
                self.planned_learning_blocks - self.host["learning_blocks"]
            ) * (self.pqn.epochs * self.pqn.num_minibatches)
        elif self.qmix is None:
            remaining_updates = math.ceil(
                (self.config.total_env_steps - self.host["env_steps"])
                / (self.config.num_envs * self.config.ppo.rollout_length)
            )
        else:
            _, learning = qmix_fixed_block_counts(
                self.config.total_env_steps // self.config.num_envs,
                rollout_length=self.qmix.rollout_length,
                minimum=self.qmix.min_buffer_size,
            )
            remaining_updates = (
                learning - self.host["learning_blocks"]
            ) * self.qmix.epochs
        interval = cast(int, self.config.checkpoint_interval_updates)
        estimate = (
            math.ceil(remaining_updates / interval)
            * self.host["save_seconds"]
            / self.host["saves"]
            + self.host["report_seconds"]
        )
        random_done = {row["env_steps"] for row in self.host["random_results"]}
        random_games = (
            len(self.random_steps - random_done)
            * 10
            * (self.config.random_diagnostic_seed_pairs or 0)
        )
        if self.panel is None and not random_games:
            return float(estimate)
        if not self.host["validation_games"]:
            return None
        completed = {row["env_steps"] for row in self.host["routine_results"]}
        remaining_points = len(self.validation_steps - completed)
        games = random_games
        if self.panel is not None:
            games += (
                remaining_points
                * 5
                * len(self.panel.members)
                * 2
                * self.config.routine_seed_pairs
            )
            games += (
                max(0, 3 - len(self.host["confirmation_results"]))
                * 5
                * len(self.panel.members)
                * 2
                * self.config.confirmation_seed_pairs
            )
        if self.config.slot_diagnostic and not self.host["slot_complete"]:
            games += 3200
        return float(
            estimate
            + games * self.host["validation_seconds"] / self.host["validation_games"]
        )

    def save(self) -> Path:
        """Publish one accepted learner boundary with its exact host-log prefix."""
        from marl_battlegrounds.training.checkpoints import save_checkpoint

        self.host["elapsed_seconds"] = (
            self.prior_elapsed + time.monotonic() - self.start
        )
        self.metadata["log_cursors"] = {
            "training_updates.jsonl": log_cursor(self.root / "training_updates.jsonl")
        }
        started = time.monotonic()
        self.checkpoint = save_checkpoint(
            self.root,
            self.collection,
            self.state,
            metadata=self.metadata,
            writer=self.writer,
            ppo=self.config.ppo,
            method=self.config.method,
            qmix=self.qmix,
            pqn=self.pqn,
        )
        self.metadata["parent_checkpoint"] = self.checkpoint.name
        seconds = time.monotonic() - started
        self.host["save_seconds"] += seconds
        self.host["saves"] += 1
        self.event(
            "checkpoint_saved",
            checkpoint_id=self.checkpoint.name,
            seconds=seconds,
        )
        self.prune_recovery()
        return self.checkpoint

    def prune_recovery(self) -> None:
        """Keep two recent recovery payloads and every exported candidate boundary.

        Only payload directories of older checkpoints from this active attempt's
        saved recovery list are removed after a new durable pointer. Small
        ancestry descriptions, actor exports and all M8 recording bundles remain.
        """
        assert self.checkpoint is not None
        recent: list[str] = self.host["recovery_checkpoints"]
        if self.checkpoint.name not in recent:
            recent.append(self.checkpoint.name)
        while len(recent) > 2:
            identifier = recent.pop(0)
            if identifier in self.host["actors"]:
                continue
            if len(identifier) != 64 or any(
                char not in "0123456789abcdef" for char in identifier
            ):
                raise ValueError("Invalid checkpoint retention identity")
            directory = self.root / "checkpoints" / identifier
            if directory.is_symlink():
                raise ValueError("Checkpoint retention cannot follow a symlink")
            for name in ("actor", "state"):
                payload = directory / name
                if payload.is_symlink():
                    raise ValueError(
                        "Checkpoint retention cannot follow a payload symlink"
                    )
                if payload.exists():
                    shutil.rmtree(payload)
            self.event("checkpoint_payload_pruned", checkpoint_id=identifier)

    def actor(self) -> Path:
        """Export this boundary's exact actor; reuse an identical existing artifact.

        A QMIX export holds only the Q-network parameters and a PQN-VDN export
        only the network's parameters and statistics; both play greedily and
        record the optimizer steps behind them.
        """
        from marl_battlegrounds.training.checkpoints import export_system

        assert self.checkpoint is not None
        destination = self.root / "actors" / self.checkpoint.name
        destination.parent.mkdir(exist_ok=True)
        variables = self.state.carry.history.current_variables
        settings = self.qmix or self.pqn or self.config.ppo
        export_system(
            variables.params
            if self.qmix is not None
            else variables.network
            if self.pqn is not None
            else variables,
            destination,
            input_scale=settings.input_scale,
            spawn_frame=settings.spawn_frame,
            method=self.config.method,
            metadata={
                "run_id": self.metadata["run_id"],
                "seed": self.config.seed,
                "env_steps": self.host["env_steps"],
                "checkpoint_id": self.checkpoint.name,
                **(
                    {}
                    if is_ppo_method(self.config.method)
                    else {"optimizer_steps": self.host["completed_updates"]}
                ),
                # A later run that pins this export inherits its exposure.
                **(
                    {}
                    if self.collection.pinned_opponent is None
                    else {"pinned_opponent": self.collection.pinned_opponent}
                ),
            },
        )
        self.host["actors"][self.checkpoint.name] = str(destination)
        return destination

    def validate(self, actor: Path, purpose: str) -> dict[str, Any]:
        """Run or recover one frozen validation task and record its exact result."""
        import jax

        from marl_battlegrounds.training.validation import validate_checkpoint

        assert self.panel is not None
        self.set_status("validation")
        started = time.monotonic()
        validation_options: dict[str, Any] = {}
        declared_pairs = None
        if self.validation_declaration is not None:
            from marl_battlegrounds.training.validation import check_confirmation_roots

            root = self.validation_declaration["roots"][purpose]
            if purpose == "confirmation":
                check_confirmation_roots(
                    root,
                    self.host["routine_results"] + self.inherited["records"],
                    used_roots=self.inherited["used_roots"],
                )
            if self.panel.schema_version == 2:
                validation_options["root_seed"] = root
            declared_pairs = self.validation_declaration[
                "routine_seed_pairs"
                if purpose == "routine"
                else "confirmation_seed_pairs"
            ]
        summary = validate_checkpoint(
            actor,
            self.panel,
            output_dir=self.root / "validation" / f"{purpose}-{actor.name}",
            purpose=purpose,
            seed_pairs=declared_pairs
            if declared_pairs is not None
            else (
                self.config.routine_seed_pairs
                if purpose == "routine"
                else self.config.confirmation_seed_pairs
            ),
            num_envs=32
            if jax.default_backend() == "gpu"
            else min(32, self.config.num_envs),
            chunk_size=128,
            event_callback=lambda record: self.event("validation_segment", **record),
            red_zone_depth=self.config.red_zone_depth,
            **validation_options,
        )
        key = "routine_results" if purpose == "routine" else "confirmation_results"
        if not any(item["task_id"] == summary["task_id"] for item in self.host[key]):
            summary = {
                **summary,
                "elapsed_seconds": self.prior_elapsed + time.monotonic() - self.start,
            }
            self.host[key].append(summary)
            seconds = time.monotonic() - started
            self.host["validation_seconds"] += seconds
            pairs = (
                declared_pairs
                if declared_pairs is not None
                else (
                    self.config.routine_seed_pairs
                    if purpose == "routine"
                    else self.config.confirmation_seed_pairs
                )
            )
            self.host["validation_games"] += 5 * len(self.panel.members) * 2 * pairs
            self.event(
                "validation_complete",
                result=summary,
                seconds=seconds,
            )
        atomic_json(
            self.root / "validation_results.json",
            self.host["routine_results"] + self.host["confirmation_results"],
        )
        self.status["validation_score"] = summary["score"]
        return summary

    def finish_pending(self) -> None:
        """Finish the saved boundary's capture/validation without another decision."""
        pending = self.host["pending"]
        if pending is None:
            return
        actor = self.actor()
        if pending.get("routine"):
            self.validate(actor, "routine")
        if pending.get("random"):
            self.validate_random(actor)
        if self.host["env_steps"] == self.config.total_env_steps:
            self.host["final_actor"] = str(actor)
        self.host["pending"] = None
        if pending.get("routine"):
            self.report()

    def validate_random(self, actor: Path) -> None:
        """Complete one fixed Random capture without changing learner or its keys.

        actor is this saved boundary's immutable export. Initialization may reuse
        the fully verified original result configured by the caller. Capture
        timings describe this run just before evaluation/reuse. Original task,
        actor, checkpoint and M8 pass identities stay unchanged on reuse.
        Publish the accumulated records and existing host game and combat means.
        Combat changes compare equal-map mean kills minus deaths with the
        verified initialization. They are early learning clues, not native wins.
        Games use the run's red_zone_depth; a reused initialization must have
        been recorded at that same depth.
        """
        import jax

        from marl_battlegrounds.training import checkpoints, validation

        pairs = self.config.random_diagnostic_seed_pairs
        assert pairs is not None
        if any(
            row["env_steps"] == self.host["env_steps"]
            for row in self.host["random_results"]
        ):
            return
        self.set_status("random_validation")
        started = time.monotonic()
        elapsed = self.prior_elapsed + started - self.start
        wall = max(0.0, time.time() - self.created_at)
        reference = (
            self.config.random_initialization_result
            if self.host["env_steps"] == 0
            else None
        )
        record: dict[str, Any]
        if reference is None:
            directory = self.root / "validation" / f"random-{actor.name}"
            summary = validation.validate_random(
                actor,
                output_dir=directory,
                seed_pairs=pairs,
                num_envs=32
                if jax.default_backend() == "gpu"
                else min(32, self.config.num_envs),
                chunk_size=128,
                event_callback=lambda record: self.event(
                    "random_validation_segment", **record
                ),
                red_zone_depth=self.config.red_zone_depth,
            )
            record = {
                **summary,
                "actor_path": str(actor),
                "summary_path": str(directory / "validation_summary.json"),
                "reference_path": None,
                "reused_initialization": False,
            }
            self.host["validation_games"] += summary["games"]
        else:
            identity = checkpoints.artifact_identity(actor)
            record = {
                **validation.read_random_initialization(
                    reference,
                    actor_digest=identity["actor_digest"],
                    seed_pairs=pairs,
                    red_zone_depth=self.config.red_zone_depth,
                ),
                "reference_path": str(Path(reference).absolute()),
                "reused_initialization": True,
            }
        record.update(
            elapsed_seconds=elapsed,
            wall_seconds=wall,
            training_seconds=self.host["training_seconds"],
        )
        self.host["random_results"].append(record)
        seconds = time.monotonic() - started
        self.host["validation_seconds"] += seconds
        atomic_json(self.root / "random_diagnostics.json", self.host["random_results"])
        self.event("random_validation_complete", result=record, seconds=seconds)
        self.set_status(
            "random_validation_complete",
            validation_score=record["score"],
            random_validation=_random_progress(self.host["random_results"]),
            **{
                f"validation_{name}": sum(cell[name] for cell in record["cells"])
                for name in ("wins", "draws", "losses")
            },
        )

    def report(self) -> tuple[Path, ...]:
        """Refresh shared reports from durable source records without new evaluation."""
        from marl_battlegrounds.training.analysis import analyze

        self.set_status("reporting")
        started = time.monotonic()
        result = analyze([self.root], output_dir=self.root)
        self.host["report_seconds"] = time.monotonic() - started
        self.event("reports_written", seconds=self.host["report_seconds"])
        return tuple(Path(path) for path in result["artifacts"].values())

    def execute(self) -> TrainResult:
        """Run exact-budget work and preserve failures for explicit resume."""
        import jax
        import numpy as np

        from marl_battlegrounds.training.analysis import (
            confirmation_candidates,
            refresh_summary,
            select_checkpoint,
        )
        from marl_battlegrounds.training.collection import (
            collect_training_rollout,
            training_summary,
        )
        from marl_battlegrounds.training.validation import (
            selection_validation_results,
            validation_root_comparison,
        )

        self.event(
            "attempt_started",
            resumed_from=None if self.checkpoint is None else str(self.checkpoint),
            setup_seconds=time.monotonic() - self.start,
            runtime=self.runtime,
            execution=self.metadata["execution"],
        )
        try:
            if self.checkpoint is None:
                self.host["pending"] = {
                    "routine": 0 in self.validation_steps,
                    "random": 0 in self.random_steps,
                }
                self.set_status("initializing")
                self.save()
            self.finish_pending()
            if self.host["env_steps"] < self.config.total_env_steps:
                self.set_status(
                    "compiling_first_update"
                    if not self.host.get(
                        "completed_blocks", self.host["completed_updates"]
                    )
                    else "training"
                )
            interval = cast(int, self.config.checkpoint_interval_updates)
            while self.host["env_steps"] < self.config.total_env_steps:
                started = time.monotonic()
                collected, rollout = collect_training_rollout(
                    self.collection,
                    self.state.carry,
                    length=self.block_length(),
                    writer=self.writer,
                )
                # The required update-result transfer is the only learner sync.
                next_state, result = self.updater(self.state, collected, rollout)
                host_result = jax.device_get(result)
                seconds = time.monotonic() - started
                accepted = (
                    host_result.performed
                    if is_ppo_method(self.config.method)
                    else host_result.accepted
                )
                if bool(host_result.failed) or not bool(accepted):
                    if self.pqn is not None:
                        self.event(
                            "learner_update_rejected",
                            **_rejected_block_facts(host_result),
                        )
                    raise RuntimeError(
                        "Learner update failed: "
                        f"reason {int(host_result.failure_reason)}"
                    )
                self.state = next_state
                summary, metrics = host_result.summary, host_result.metrics
                updates_before = self.host["completed_updates"]
                self.host["env_steps"] += int(summary.real_transitions)
                self.host["actor_decisions"] += int(summary.live_actor_decisions)
                if self.qmix is not None:
                    self._count_qmix_block(host_result)
                elif self.pqn is not None:
                    self._count_pqn_block(host_result)
                else:
                    self.host["completed_updates"] += 1
                    self.host["used_policy_samples"] += sum(
                        int(x) for x in np.asarray(metrics.actor_samples).flat
                    )
                    self.host["used_value_samples"] += sum(
                        int(x) for x in np.asarray(metrics.critic_samples).flat
                    )
                for name in (
                    "stage_completed",
                    "stage_score_sums",
                    "stage_length_sum",
                    "stage_k20_count",
                ):
                    incoming = np.asarray(getattr(summary, name), dtype=object)
                    previous = np.asarray(
                        self.host.get(name, np.zeros(incoming.shape, dtype=object)),
                        dtype=object,
                    )
                    self.host[name] = (previous + incoming).tolist()
                self.host["training_seconds"] += seconds
                if self.speed_estimate is not None:
                    self.speed_estimate.observe(int(summary.real_transitions), seconds)
                self.host["elapsed_seconds"] = (
                    self.prior_elapsed + time.monotonic() - self.start
                )
                if self.qmix is not None or self.pqn is not None:
                    row = (
                        self._qmix_row(host_result, seconds)
                        if self.qmix is not None
                        else self._pqn_row(host_result, seconds)
                    )
                    append_jsonl(self.root / "training_updates.jsonl", row)
                    # Free the block's device arrays before saving or validating.
                    del rollout, collected, result, next_state
                    self._after_block(
                        row,
                        summary,
                        seconds,
                        updates_before,
                        interval,
                        phase="warmup"
                        if self.qmix is not None
                        else "initial_collection",
                    )
                    continue
                row = {
                    "schema_version": 1,
                    "attempt_id": self.metadata["attempt_id"],
                    "update_index": self.host["completed_updates"],
                    "wall_seconds": max(0.0, time.time() - self.created_at),
                    **{
                        key: self.host[key]
                        for key in (
                            "env_steps",
                            "actor_decisions",
                            "used_policy_samples",
                            "used_value_samples",
                            "training_seconds",
                            "elapsed_seconds",
                        )
                    },
                    "collection_update_seconds": seconds,
                    "task_reward_mean": float(summary.task_reward_sum)
                    / int(summary.active_samples)
                    if int(summary.active_samples)
                    else None,
                    "shaping_mean": float(summary.shaping_reward_sum)
                    / int(summary.real_transitions),
                    "policy_loss": float(
                        np.asarray(metrics.actor_loss)[
                            np.asarray(metrics.actor_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.actor_samples)
                    else None,
                    "value_loss": float(
                        np.asarray(metrics.value_loss)[
                            np.asarray(metrics.critic_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.critic_samples)
                    else None,
                    "entropy": float(
                        np.asarray(metrics.entropy)[
                            np.asarray(metrics.actor_samples) > 0
                        ].mean()
                    )
                    if np.any(metrics.actor_samples)
                    else None,
                    "learning_rate": self.ppo_reporting.actor_lr,
                    "value_loss_units": "normalized_squared"
                    if self.config.ppo.value_normalization
                    else "reward_squared",
                    **{
                        name: float(
                            np.asarray(getattr(metrics, name))[
                                np.asarray(samples) > 0
                            ].mean()
                        )
                        if np.any(samples)
                        else None
                        for samples, names in (
                            (
                                metrics.actor_samples,
                                (
                                    "approx_kl",
                                    "policy_clip_fraction",
                                    "actor_grad_norm",
                                ),
                            ),
                            (
                                metrics.critic_samples,
                                (
                                    "value_clip_fraction",
                                    "critic_grad_norm",
                                    "target_mean",
                                    "target_std",
                                    "value_mean",
                                    "value_rmse",
                                    "normalization_mean",
                                    "normalization_std",
                                ),
                            ),
                        )
                        for name in names
                    },
                }
                append_jsonl(self.root / "training_updates.jsonl", row)
                del rollout, collected, result, next_state
                steps = self.host["env_steps"]
                capture = (
                    steps in self.validation_steps
                    or steps in self.config.checkpoint_env_steps
                    or steps == self.config.total_env_steps
                )
                if capture:
                    self.host["pending"] = {
                        "routine": steps in self.validation_steps,
                        "random": steps in self.random_steps,
                    }
                if capture or _crosses_interval(
                    updates_before, self.host["completed_updates"], interval
                ):
                    self.save()
                    self.finish_pending()
                self.set_status(
                    "training",
                    recent_transitions_per_second=int(summary.real_transitions)
                    / seconds,
                    **{
                        key: row[key]
                        for key in (
                            "policy_loss",
                            "value_loss",
                            "entropy",
                            "task_reward_mean",
                            "shaping_mean",
                        )
                    },
                )
            if self.panel is not None:
                final = Path(self.host["final_actor"])
                actor_paths = {
                    **{
                        key: value["actor_path"]
                        for key, value in self.inherited["actors"].items()
                    },
                    **self.host["actors"],
                }
                routine = self.host["routine_results"]
                if self.validation_declaration is None:
                    candidate_ids = confirmation_candidates(
                        self.host["routine_results"], final_checkpoint_id=final.name
                    )
                    confirmations = self.host["confirmation_results"]
                else:
                    _, confirmations, candidate_ids = selection_validation_results(
                        self.host["routine_results"],
                        self.host["confirmation_results"],
                        self.inherited,
                        declaration=self.validation_declaration,
                        panel=self.panel,
                        final_checkpoint_id=final.name,
                    )
                completed = {row["checkpoint_id"] for row in confirmations}
                for identifier in candidate_ids:
                    if identifier not in completed:
                        self.validate(Path(actor_paths[identifier]), "confirmation")
                if self.validation_declaration is not None:
                    routine, confirmations, _ = selection_validation_results(
                        self.host["routine_results"],
                        self.host["confirmation_results"],
                        self.inherited,
                        declaration=self.validation_declaration,
                        panel=self.panel,
                        final_checkpoint_id=final.name,
                    )
                if candidate_ids:
                    selected = select_checkpoint(confirmations)
                    if self.validation_declaration is not None:
                        selected.update(
                            routine_comparison=validation_root_comparison(
                                routine,
                                allow_different_roots=self.validation_declaration[
                                    "allow_different_roots"
                                ],
                            ),
                            confirmation_comparison=validation_root_comparison(
                                confirmations,
                                allow_different_roots=self.validation_declaration[
                                    "allow_different_roots"
                                ],
                            ),
                        )
                    self.host["selection"] = selected
                    self.host["selected_actor"] = actor_paths[selected["checkpoint_id"]]
                    atomic_json(self.root / "selection.json", selected)
                else:
                    # A declared child may end before its first learning update.
                    # Its final actor still loads, but it is not selected evidence.
                    reason = "No eligible trained checkpoint"
                    self.status["selection_unavailable_reason"] = reason
                    self.event("selection_unavailable", reason=reason)
            if self.config.slot_diagnostic:
                self.run_slot_check()
            exposure = training_summary(self.collection, self.state.carry)
            if self.qmix is not None:
                exposure["sampled_exposure"] = {
                    "unit": "TD pairs used by optimizer steps; repeats count again",
                    "used_td_pairs": self.host["used_td_pairs"],
                    **self.host["sampled_exposure"],
                }
            if self.pqn is not None:
                exposure["used_exposure"] = {
                    "unit": (
                        "TD pairs used by optimizer steps, by left row; each "
                        "epoch counts again and kept rows are used again"
                    ),
                    "used_td_pairs": self.host["used_td_pairs"],
                    "used_prefix_td_pairs": self.host["used_prefix_td_pairs"],
                    **self.host["used_exposure"],
                }
            exposure.update(
                {
                    name: self.host.get(name)
                    for name in (
                        "stage_completed",
                        "stage_score_sums",
                        "stage_length_sum",
                        "stage_k20_count",
                    )
                }
            )
            thresholds = cast(list[int], exposure["score_thresholds_by_episode_stage"])
            completed = self.host.get("stage_completed")
            if completed is not None:
                for row in cast(
                    list[dict[str, Any]], exposure["exposure_by_score_threshold"]
                ):
                    counts = [
                        sum(
                            completed[i][outcome]
                            for i, threshold in enumerate(thresholds)
                            if threshold == row["score_threshold"]
                        )
                        for outcome in range(3)
                    ]
                    row.update(
                        completed_games=sum(counts),
                        wins=counts[0],
                        draws=counts[1],
                        losses=counts[2],
                    )
            atomic_json(self.root / "exposure.json", exposure)
            self.set_status("finalizing")
            paths = self.report()
            if not all(path.is_file() and path.stat().st_size for path in paths):
                raise ValueError("Required final reports are missing or empty")
            memory = _memory_snapshot()
            self.set_status(
                "complete",
                status="complete",
                pending_work_seconds=0.0,
                exit_code=0,
                memory=memory,
            )
            refresh_summary([self.root], output_dir=self.root)
            self.event("attempt_complete", memory=memory)
            return TrainResult(
                self.root,
                Path(self.host["final_actor"]),
                None
                if self.host["selected_actor"] is None
                else Path(self.host["selected_actor"]),
                self.host["env_steps"],
                self.host["completed_updates"],
                "complete",
                paths,
            )
        except BaseException as error:
            memory = _memory_snapshot()
            self.event(
                "attempt_failed",
                error=f"{type(error).__name__}: {error}",
                memory=memory,
            )
            self.set_status(
                "failed",
                status="failed",
                error=str(error),
                exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                memory=memory,
            )
            try:
                refresh_summary([self.root], output_dir=self.root)
            except Exception as report_error:
                self.event("failure_summary_unavailable", error=str(report_error))
            raise

    def _count_qmix_block(self, result: Any) -> None:  # noqa: ANN401
        """Add one accepted QMIX block to the exact host totals.

        result is the host copy of a QMIXUpdateResult. A warmup block adds a
        block only; a learning block also adds epochs optimizer steps and its
        sampled counts. Totals are Python integers and may exceed int32.
        """
        import numpy as np

        assert self.qmix is not None
        self.host["completed_blocks"] += 1
        if bool(result.performed):
            self.host["learning_blocks"] += 1
            self.host["completed_updates"] += self.qmix.epochs
        sampled = result.sampled
        for name in ("sampled_sequences", "used_td_pairs", "used_agent_utilities"):
            self.host[name] += int(getattr(sampled, name))
        exposure = self.host["sampled_exposure"]
        for key, field_name in (
            ("by_stage", "exposure_by_stage"),
            ("by_source", "exposure_by_source"),
            ("by_opponent", "exposure_by_opponent"),
        ):
            counts = cast(list[int], np.asarray(getattr(sampled, field_name)).tolist())
            totals = cast(list[int], exposure[key])
            exposure[key] = [
                total + int(count) for total, count in zip(totals, counts, strict=True)
            ]

    def _qmix_row(self, result: Any, seconds: float) -> dict[str, Any]:  # noqa: ANN401
        """Build one QMIX training-log row from host totals and a block result.

        result is the host copy of the block's QMIXUpdateResult, already added
        to the host totals by _count_qmix_block; seconds is the block's
        collection-plus-update wall time in seconds. Returns the JSON-ready row
        for training_updates.jsonl: the shared counters, the exact QMIX totals,
        task and shaping reward means, replay_rows_per_lane (rows stored per
        game, at most buffer_size), td_pairs_per_transition (used TD pairs per
        real transition; repeats count again), loss, mean_q and mean_target
        (averages over the block's optimizer steps; None for a warmup block),
        epsilon (the clock's rate after the block, a float64 host value) and
        learning_rate. No PPO loss or sample keys appear. Reads host values
        only; no device work.
        """
        import numpy as np

        from marl_battlegrounds.baselines.qmix import epsilon_reference

        assert self.qmix is not None
        summary, metrics = result.summary, result.metrics
        learned = bool(result.performed)
        rounds = self.host["env_steps"] // self.config.num_envs

        def mean(values: object) -> float | None:
            """Average one per-epoch metric over this block's optimizer steps."""
            return float(np.mean(np.asarray(values))) if learned else None

        return {
            "schema_version": 1,
            "attempt_id": self.metadata["attempt_id"],
            "wall_seconds": max(0.0, time.time() - self.created_at),
            **{
                key: self.host[key]
                for key in (
                    "env_steps",
                    "actor_decisions",
                    "completed_blocks",
                    "learning_blocks",
                    "completed_updates",
                    "sampled_sequences",
                    "used_td_pairs",
                    "used_agent_utilities",
                    "training_seconds",
                    "elapsed_seconds",
                )
            },
            "collection_update_seconds": seconds,
            "task_reward_mean": float(summary.task_reward_sum)
            / int(summary.active_samples)
            if int(summary.active_samples)
            else None,
            "shaping_mean": float(summary.shaping_reward_sum)
            / int(summary.real_transitions),
            "replay_rows_per_lane": min(rounds, self.qmix.buffer_size),
            "td_pairs_per_transition": self.host["used_td_pairs"]
            / self.host["env_steps"],
            "loss": mean(metrics.loss),
            "mean_q": mean(metrics.mean_q),
            "mean_target": mean(metrics.mean_target),
            "epsilon": self.continuation.exploration.at(self.host["env_steps"])
            if self.continuation is not None
            and self.continuation.exploration is not None
            else epsilon_reference(
                rounds, self.config.num_envs, self.qmix.eps_min, self.qmix.eps_decay
            ),
            "learning_rate": cast(QMIXConfig, self.qmix_reporting).q_lr,
        }

    def _count_pqn_block(self, result: Any) -> None:  # noqa: ANN401
        """Add one accepted PQN-VDN block to the exact host totals.

        result is the host copy of a PQNUpdateResult. Every accepted block
        adds one completed block; a learning block (performed) also adds a
        learning block, epochs * num_minibatches optimizer steps and its used
        counts and exposure. Totals are Python integers and may exceed int32.
        """
        import numpy as np

        assert self.pqn is not None
        self.host["completed_blocks"] += 1
        if bool(result.performed):
            self.host["learning_blocks"] += 1
            self.host["completed_updates"] += self.pqn.epochs * self.pqn.num_minibatches
        used = result.used
        for name in (
            "used_sequences",
            "used_td_pairs",
            "used_agent_utilities",
            "used_prefix_td_pairs",
        ):
            self.host[name] += int(getattr(used, name))
        exposure = self.host["used_exposure"]
        for key, field_name in (
            ("by_stage", "exposure_by_stage"),
            ("by_source", "exposure_by_source"),
            ("by_opponent", "exposure_by_opponent"),
        ):
            counts = cast(list[int], np.asarray(getattr(used, field_name)).tolist())
            totals = cast(list[int], exposure[key])
            exposure[key] = [
                total + int(count) for total, count in zip(totals, counts, strict=True)
            ]

    def _pqn_row(self, result: Any, seconds: float) -> dict[str, Any]:  # noqa: ANN401
        """Build one PQN-VDN training-log row from host totals and a block result.

        Parameters
        ----------
        result : PQNUpdateResult
            Host copy of the block's result, already added to the host totals
            by _count_pqn_block.
        seconds : float
            The block's collection-plus-update wall time in seconds.

        Returns
        -------
        dict
            JSON-ready row for training_updates.jsonl: the shared counters,
            completed_blocks, learning_blocks, completed_updates (optimizer
            steps), initial_rounds_completed (at most W), the exact used
            totals and used_exposure (cumulative, like the other counts),
            task and shaping reward means, and loss, mean_q and mean_target
            (averages over the block's minibatch steps; None for an initial
            chunk). epsilon is the rate current actors hold at this boundary
            (1.0 before W, then the block clock's value, a float64 host
            value); learning_rate is the scheduled rate of the block's last
            applied step (None for an initial chunk). No PPO or QMIX-only keys.

        Notes
        -----
        Reads host values only; adds no device work or synchronization.
        """
        import numpy as np

        from marl_battlegrounds.baselines.pqn import (
            pqn_epsilon_reference,
            pqn_learning_rate,
        )

        assert self.pqn is not None
        summary, metrics = result.summary, result.metrics
        learned = bool(result.performed)
        rounds = self.host["env_steps"] // self.config.num_envs
        planned = self.planned_learning_blocks

        def mean(values: object) -> float | None:
            """Average one (epochs, minibatches) metric over this block's steps."""
            return float(np.mean(np.asarray(values))) if learned else None

        return {
            "schema_version": 1,
            "attempt_id": self.metadata["attempt_id"],
            "wall_seconds": max(0.0, time.time() - self.created_at),
            **{
                key: self.host[key]
                for key in (
                    "env_steps",
                    "actor_decisions",
                    "completed_blocks",
                    "learning_blocks",
                    "completed_updates",
                )
            },
            "initial_rounds_completed": min(rounds, self.pqn.initial_rounds),
            **{
                key: self.host[key]
                for key in (
                    "used_sequences",
                    "used_td_pairs",
                    "used_agent_utilities",
                    "used_prefix_td_pairs",
                    "used_exposure",
                    "training_seconds",
                    "elapsed_seconds",
                )
            },
            "collection_update_seconds": seconds,
            "task_reward_mean": float(summary.task_reward_sum)
            / int(summary.active_samples)
            if int(summary.active_samples)
            else None,
            "shaping_mean": float(summary.shaping_reward_sum)
            / int(summary.real_transitions),
            "loss": mean(metrics.loss),
            "mean_q": mean(metrics.mean_q),
            "mean_target": mean(metrics.mean_target),
            "epsilon": 1.0
            if rounds < self.pqn.initial_rounds
            else self.continuation.exploration.at(self.host["learning_blocks"])
            if self.continuation is not None
            and self.continuation.exploration is not None
            else pqn_epsilon_reference(self.host["learning_blocks"], planned, self.pqn),
            "learning_rate": (
                self.continuation.pqn_rate.at(self.host["completed_updates"] - 1)
                if self.continuation is not None
                and self.continuation.pqn_rate is not None
                else pqn_learning_rate(
                    self.host["completed_updates"] - 1, planned, self.pqn
                )
            )
            if learned
            else None,
        }

    def block_length(self) -> int:
        """Return the rounds to request for the next collection block.

        PPO and QMIX always request rollout_length. PQN-VDN requests
        ``min(T, W - r)`` during its W initial rounds (a chunk of T, then one
        of H) and T afterwards; collection stops at the exact budget, so a
        final block may hold fewer real rounds. Reads host counts only.
        """
        if self.pqn is None:
            return self.rollout_length
        rounds = self.host["env_steps"] // self.config.num_envs
        initial = self.pqn.initial_rounds
        return (
            min(self.rollout_length, initial - rounds)
            if rounds < initial
            else self.rollout_length
        )

    def _after_block(
        self,
        row: dict[str, Any],
        summary: Any,  # noqa: ANN401
        seconds: float,
        updates_before: int,
        interval: int,
        *,
        phase: str,
    ) -> None:
        """Apply the shared capture and save rules after one QMIX or PQN-VDN block.

        row is the block's log row from _qmix_row or _pqn_row; summary is its
        host UpdateSummary; seconds is the block's collection-plus-update
        time; updates_before is the optimizer-step total before the block;
        interval is the resolved checkpoint_interval_updates; phase is the
        status name before the first learning block ("warmup" for QMIX,
        "initial_collection" for PQN-VDN). A capture point (a validation
        fraction, an extra save point or the final step) or a crossed
        multiple of the interval saves the learner and finishes pending
        captures, exactly as for PPO; blocks without optimizer steps never
        cross an interval. Status reports phase until the first learning
        block, then "training", with the row's loss, Q values, reward means,
        epsilon and the learning-block count. Writes checkpoint and status
        files.
        """
        steps = self.host["env_steps"]
        capture = (
            steps in self.validation_steps
            or steps in self.config.checkpoint_env_steps
            or steps == self.config.total_env_steps
        )
        if capture:
            self.host["pending"] = {
                "routine": steps in self.validation_steps,
                "random": steps in self.random_steps,
            }
        if capture or _crosses_interval(
            updates_before, self.host["completed_updates"], interval
        ):
            self.save()
            self.finish_pending()
        self.set_status(
            "training" if self.host["learning_blocks"] else phase,
            recent_transitions_per_second=int(summary.real_transitions) / seconds,
            learning_blocks=self.host["learning_blocks"],
            **{
                key: row[key]
                for key in (
                    "loss",
                    "mean_q",
                    "task_reward_mean",
                    "shaping_mean",
                    "epsilon",
                )
            },
        )

    def run_slot_check(self) -> None:
        """Run the declared trained comparison only after the final actor exists.

        The games use the run's red_zone_depth, which the slot task records.
        """
        from marl_battlegrounds.training.validation import run_slot_diagnostic

        assert self.panel is not None
        self.set_status("slot_diagnostic")
        started = time.monotonic()
        opponent = self.config.slot_diagnostic_actor or self.panel.members[-1].path
        assert opponent is not None
        result = run_slot_diagnostic(
            Path(self.host["final_actor"]),
            opponent,
            output_dir=self.root
            / "slot_diagnostic"
            / Path(self.host["final_actor"]).name,
            event_callback=lambda record: self.event("slot_segment", **record),
            red_zone_depth=self.config.red_zone_depth,
        )
        atomic_json(self.root / "slot_diagnostic.json", result)
        self.host["slot_complete"] = True
        self.event("slot_diagnostic_finished", seconds=time.monotonic() - started)
