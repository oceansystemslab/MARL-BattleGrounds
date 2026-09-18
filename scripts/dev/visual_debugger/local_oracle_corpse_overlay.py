"""Keep this old import path bound to the package-owned Viewer module.

Objects and patched globals come from the installed Viewer authority. This
compatibility alias copies no classes or service state.
"""

import sys
from typing import TYPE_CHECKING

from marl_battlegrounds.viewer import local_oracle_corpse_overlay as _implementation

if TYPE_CHECKING:
    from marl_battlegrounds.viewer.local_oracle_corpse_overlay import *  # noqa: F403

    _host_has_clear_line_of_sight_v1 = (
        _implementation._host_has_clear_line_of_sight_v1  # pyright: ignore[reportPrivateUsage]
    )
    _host_within_observation_radius_v1 = (
        _implementation._host_within_observation_radius_v1  # pyright: ignore[reportPrivateUsage]
    )
    validate_local_oracle_corpse_overlay_against_source_v1 = (
        _implementation.validate_local_oracle_corpse_overlay_against_source_v1
    )

sys.modules[__name__] = _implementation
