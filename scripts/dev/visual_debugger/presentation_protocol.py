"""Keep this old import path bound to the package-owned Viewer module.

Objects and patched globals come from the installed Viewer authority. This
compatibility alias copies no classes or service state.
"""

import sys
from typing import TYPE_CHECKING

from marl_battlegrounds.viewer import presentation_protocol as _implementation

if TYPE_CHECKING:
    from marl_battlegrounds.viewer.presentation_protocol import *  # noqa: F403

    _seal_oracle_authorized_current_endpoint_v1 = (
        _implementation._seal_oracle_authorized_current_endpoint_v1  # pyright: ignore[reportPrivateUsage]
    )
    _validate_local_oracle_corpse_overlay = (
        _implementation._validate_local_oracle_corpse_overlay  # pyright: ignore[reportPrivateUsage]
    )

sys.modules[__name__] = _implementation
