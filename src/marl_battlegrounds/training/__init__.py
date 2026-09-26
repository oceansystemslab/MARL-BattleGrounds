"""Load optional training and collection helpers only when requested.

Content preparation, schedules, shaping, self-play and compact collection use
base dependencies. The package alone imports no JAX or PPO. Baseline actors need
the existing training extra. train owns complete PPO, QMIX and PQN-VDN runs;
load_system reads frozen actors and analyze uses the optional viz extra for
saved-result plots. reselect_checkpoint writes a separate decision from saved
actor and game evidence; it never replaces the original run selection.
"""

# pyright: reportUnsupportedDunderAll=false
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marl_battlegrounds.training._content import (
        PreparedTrainingContent as PreparedTrainingContent,
    )
    from marl_battlegrounds.training._content import (
        TrainingContentBinding as TrainingContentBinding,
    )
    from marl_battlegrounds.training._content import (
        prepare_training_content as prepare_training_content,
    )
    from marl_battlegrounds.training.analysis import analyze as analyze
    from marl_battlegrounds.training.analysis import analyze_screen as analyze_screen
    from marl_battlegrounds.training.checkpoints import load_system as load_system
    from marl_battlegrounds.training.collection import TrainingCarry as TrainingCarry
    from marl_battlegrounds.training.collection import (
        TrainingCollection as TrainingCollection,
    )
    from marl_battlegrounds.training.collection import (
        TrainingRollout as TrainingRollout,
    )
    from marl_battlegrounds.training.collection import (
        TrainingTransition as TrainingTransition,
    )
    from marl_battlegrounds.training.collection import (
        advance_training_step as advance_training_step,
    )
    from marl_battlegrounds.training.collection import (
        collect_training_rollout as collect_training_rollout,
    )
    from marl_battlegrounds.training.collection import (
        init_training_collection as init_training_collection,
    )
    from marl_battlegrounds.training.collection import (
        scan_training_rollout as scan_training_rollout,
    )
    from marl_battlegrounds.training.collection import (
        training_summary as training_summary,
    )
    from marl_battlegrounds.training.curriculum import ScheduleArrays as ScheduleArrays
    from marl_battlegrounds.training.curriculum import (
        TrainingProgress as TrainingProgress,
    )
    from marl_battlegrounds.training.curriculum import (
        TrainingSchedule as TrainingSchedule,
    )
    from marl_battlegrounds.training.curriculum import (
        make_training_schedule as make_training_schedule,
    )
    from marl_battlegrounds.training.distributions import (
        TRAINING_KEY_SCHEMA_VERSION as TRAINING_KEY_SCHEMA_VERSION,
    )
    from marl_battlegrounds.training.distributions import (
        SampledTrainingConfigs as SampledTrainingConfigs,
    )
    from marl_battlegrounds.training.distributions import (
        sample_training_configs as sample_training_configs,
    )
    from marl_battlegrounds.training.distributions import training_keys as training_keys
    from marl_battlegrounds.training.distributions import (
        validate_training_distribution as validate_training_distribution,
    )
    from marl_battlegrounds.training.opponents import OpponentHistory as OpponentHistory
    from marl_battlegrounds.training.opponents import SnapshotEvent as SnapshotEvent
    from marl_battlegrounds.training.opponents import (
        assign_opponents as assign_opponents,
    )
    from marl_battlegrounds.training.opponents import (
        init_opponent_history as init_opponent_history,
    )
    from marl_battlegrounds.training.opponents import (
        make_opponent_system as make_opponent_system,
    )
    from marl_battlegrounds.training.opponents import (
        refresh_opponents as refresh_opponents,
    )
    from marl_battlegrounds.training.runner import TrainConfig as TrainConfig
    from marl_battlegrounds.training.runner import TrainResult as TrainResult
    from marl_battlegrounds.training.runner import extend_training as extend_training
    from marl_battlegrounds.training.runner import train as train
    from marl_battlegrounds.training.screen import prepare_screen as prepare_screen
    from marl_battlegrounds.training.selection import (
        reselect_checkpoint as reselect_checkpoint,
    )
    from marl_battlegrounds.training.shaping import (
        team_potential_shaping as team_potential_shaping,
    )
    from marl_battlegrounds.training.shaping import (
        team_score_delta_shaping as team_score_delta_shaping,
    )
    from marl_battlegrounds.training.shaping import validate_shaping as validate_shaping
    from marl_battlegrounds.training.study import run_study as run_study
    from marl_battlegrounds.training.study import start_study as start_study
    from marl_battlegrounds.training.study import stop_study as stop_study
    from marl_battlegrounds.training.study import study_status as study_status

_OWNERS = {
    "TrainConfig": "runner",
    "TrainResult": "runner",
    "train": "runner",
    "extend_training": "runner",
    "load_system": "checkpoints",
    "analyze": "analysis",
    "analyze_screen": "analysis",
    "reselect_checkpoint": "selection",
    "prepare_screen": "screen",
    "run_study": "study",
    "start_study": "study",
    "stop_study": "study",
    "study_status": "study",
    "PreparedTrainingContent": "_content",
    "TrainingContentBinding": "_content",
    "prepare_training_content": "_content",
    "TRAINING_KEY_SCHEMA_VERSION": "distributions",
    "SampledTrainingConfigs": "distributions",
    "sample_training_configs": "distributions",
    "training_keys": "distributions",
    "validate_training_distribution": "distributions",
    "ScheduleArrays": "curriculum",
    "TrainingProgress": "curriculum",
    "TrainingSchedule": "curriculum",
    "make_training_schedule": "curriculum",
    "team_potential_shaping": "shaping",
    "team_score_delta_shaping": "shaping",
    "validate_shaping": "shaping",
    "OpponentHistory": "opponents",
    "SnapshotEvent": "opponents",
    "assign_opponents": "opponents",
    "init_opponent_history": "opponents",
    "make_opponent_system": "opponents",
    "refresh_opponents": "opponents",
    "TrainingCarry": "collection",
    "TrainingCollection": "collection",
    "TrainingRollout": "collection",
    "TrainingTransition": "collection",
    "advance_training_step": "collection",
    "collect_training_rollout": "collection",
    "init_training_collection": "collection",
    "scan_training_rollout": "collection",
    "training_summary": "collection",
}

__all__ = [
    "TRAINING_KEY_SCHEMA_VERSION",
    "OpponentHistory",
    "PreparedTrainingContent",
    "SampledTrainingConfigs",
    "ScheduleArrays",
    "SnapshotEvent",
    "TrainConfig",
    "TrainResult",
    "TrainingCarry",
    "TrainingCollection",
    "TrainingContentBinding",
    "TrainingProgress",
    "TrainingRollout",
    "TrainingSchedule",
    "TrainingTransition",
    "advance_training_step",
    "analyze",
    "analyze_screen",
    "assign_opponents",
    "collect_training_rollout",
    "extend_training",
    "init_opponent_history",
    "init_training_collection",
    "load_system",
    "make_opponent_system",
    "make_training_schedule",
    "prepare_screen",
    "prepare_training_content",
    "refresh_opponents",
    "reselect_checkpoint",
    "run_study",
    "sample_training_configs",
    "scan_training_rollout",
    "start_study",
    "stop_study",
    "study_status",
    "team_potential_shaping",
    "train",
    "training_keys",
    "training_summary",
    "validate_shaping",
    "validate_training_distribution",
]


def __getattr__(name: str) -> object:
    """Load and cache one public helper, record or schema constant.

    name must be in __all__; otherwise raise AttributeError. Import only the
    owning module on first access. Importing this package by itself starts no
    numerical backend, collection, file read or optional baseline dependency.
    """
    if name not in _OWNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_OWNERS[name]}"), name)
    globals()[name] = value
    return value
