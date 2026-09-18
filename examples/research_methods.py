"""Load editable fixed methods for the Python and package-command examples.

Copy this file into your own importable research module or install that module.
For example, with this file on Python's import path, pass
``--system research_methods:load_custom`` to package evaluation. Each factory is
trusted local Python, takes no arguments and returns a fresh System. Replace the
factory body with your own checkpoint loading; MARL-BGs owns no model format.
These fixed policies illustrate integration, not trained checkpoints or learners.
"""

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.types import System


def load_custom() -> System:
    """Return two ordered independent Policies for the Mage/Priest example.

    Team-local slot zero uses tdm-alpha and slot one uses tdm-beta. Supply the
    matching ordered Mage/Priest roster when evaluating this System. No action,
    checkpoint read or file write occurs during construction. A different active
    roster size is rejected by the existing independent-Policy adapter.
    """
    return marl_bgs.independent_policies(
        (marl_bgs.policy("tdm-alpha"), marl_bgs.policy("tdm-beta"))
    )


def load_selected() -> System:
    """Return a fresh shared fixed Policy compatible with the canonical roster.

    This is the place to load your own previously selected checkpoint and return
    its System. The runnable example uses tdm-alpha and creates no files. It is
    neither a trained selection nor an official admission submission; a released
    snapshot can reject it if it duplicates an incumbent controller.
    """
    return marl_bgs.shared_policy(marl_bgs.policy("tdm-alpha"))
