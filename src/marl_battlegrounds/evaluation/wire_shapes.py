"""Frozen dimensions owned by the version-1 evaluation wire contract.

These values describe already-published evaluation records.  They intentionally
do not import the live simulator type module: a future simulator shape change
must introduce an explicit evaluation schema migration rather than silently
changing V1 artifact validation.
"""

from typing import Final

CONTEXT_FEATURES_V1: Final = 19
ENVIRONMENT_DIMENSIONS_V1: Final = 2
MAX_AGENT_SLOTS_V1: Final = 10
MAX_AGENTS_PER_TEAM_V1: Final = 5
MAX_OBJECTIVE_SLOTS_V1: Final = 8
MAX_OBSTACLE_SLOTS_V1: Final = 16
NUM_CLASSES_V1: Final = 6
NUM_MOVE_ACTIONS_V1: Final = 9
NUM_SLOW_CHANNELS_V1: Final = 3
NUM_STUN_CHANNELS_V1: Final = 3
NUM_TARGET_ACTIONS_V1: Final = 11
NUM_TEAMS_V1: Final = 2
NUM_ULTIMATE_ACTIONS_V1: Final = 2
OBJECTIVE_FEATURES_V1: Final = 12
OBSTACLE_FEATURES_V1: Final = 8
SELF_FEATURES_V1: Final = 58
UNIT_FEATURES_V1: Final = 58

__all__ = [
    "CONTEXT_FEATURES_V1",
    "ENVIRONMENT_DIMENSIONS_V1",
    "MAX_AGENTS_PER_TEAM_V1",
    "MAX_AGENT_SLOTS_V1",
    "MAX_OBJECTIVE_SLOTS_V1",
    "MAX_OBSTACLE_SLOTS_V1",
    "NUM_CLASSES_V1",
    "NUM_MOVE_ACTIONS_V1",
    "NUM_SLOW_CHANNELS_V1",
    "NUM_STUN_CHANNELS_V1",
    "NUM_TARGET_ACTIONS_V1",
    "NUM_TEAMS_V1",
    "NUM_ULTIMATE_ACTIONS_V1",
    "OBJECTIVE_FEATURES_V1",
    "OBSTACLE_FEATURES_V1",
    "SELF_FEATURES_V1",
    "UNIT_FEATURES_V1",
]
