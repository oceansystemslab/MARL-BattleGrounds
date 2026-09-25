"""Keep the fixed dimensions of version-1 evaluation records.

Model validators and historical replay readers use these constants without
importing live simulator types. Ten agent slots, five slots per team and all
feature/action axes describe the recorded wire layout, not a mutable runtime
configuration. Amendment A18 allowed the pre-alpha obstacle axis to grow from
16 to 32; later incompatible dimensions need an explicit schema migration.
Changing these values would change historical validation, so documentation
work must preserve them. CONTEXT_FEATURES_V2 is the 20-column context width
of frame V3 (the Team Deathmatch Red Zone depth appended as column 19); the
19-column CONTEXT_FEATURES_V1 still describes frames V1 and V2.
"""

from typing import Final

CONTEXT_FEATURES_V1: Final = 19
CONTEXT_FEATURES_V2: Final = 20
ENVIRONMENT_DIMENSIONS_V1: Final = 2
MAX_AGENT_SLOTS_V1: Final = 10
MAX_AGENTS_PER_TEAM_V1: Final = 5
MAX_OBJECTIVE_SLOTS_V1: Final = 8
MAX_OBSTACLE_SLOTS_V1: Final = 32
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
    "CONTEXT_FEATURES_V2",
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
