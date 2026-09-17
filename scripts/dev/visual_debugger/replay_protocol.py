"""Keep this old import path bound to the package-owned Viewer module.

Objects and patched globals come from the installed Viewer authority. This
compatibility alias copies no classes or service state.
"""

import sys
from typing import TYPE_CHECKING

from marl_battlegrounds.viewer import replay_protocol as _implementation

if TYPE_CHECKING:
    from marl_battlegrounds.viewer.replay_protocol import *  # noqa: F403

sys.modules[__name__] = _implementation
