"""Keep this old import path bound to the package-owned Viewer module.

Objects and patched globals come from the installed Viewer authority. This
compatibility alias copies no classes or service state.
"""

import sys
from typing import TYPE_CHECKING

from marl_battlegrounds.viewer import server as _implementation

if TYPE_CHECKING:
    from marl_battlegrounds.viewer.server import *  # noqa: F403
    from marl_battlegrounds.viewer.server import (
        GracefulCloseCallback as GracefulCloseCallback,
    )

    _legacy_live_binding = (
        _implementation._legacy_live_binding  # pyright: ignore[reportPrivateUsage]
    )
    webbrowser = _implementation.webbrowser

sys.modules[__name__] = _implementation
