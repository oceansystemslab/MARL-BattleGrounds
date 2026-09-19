"""Load optional training setup helpers without importing numerical dependencies.

prepare_training_content verifies the built-in content split and prepares a
shared map bank. Its models and numerical helpers use base dependencies only.
This package does not start a learner or import optional PPO dependencies.
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
    from marl_battlegrounds.training.distributions import (
        TRAINING_KEY_SCHEMA_VERSION as TRAINING_KEY_SCHEMA_VERSION,
    )
    from marl_battlegrounds.training.distributions import (
        SampledTrainingConfigs as SampledTrainingConfigs,
    )
    from marl_battlegrounds.training.distributions import (
        sample_training_configs as sample_training_configs,
    )
    from marl_battlegrounds.training.distributions import (
        training_keys as training_keys,
    )
    from marl_battlegrounds.training.distributions import (
        validate_training_distribution as validate_training_distribution,
    )

__all__ = [
    "TRAINING_KEY_SCHEMA_VERSION",
    "PreparedTrainingContent",
    "SampledTrainingConfigs",
    "TrainingContentBinding",
    "prepare_training_content",
    "sample_training_configs",
    "training_keys",
    "validate_training_distribution",
]


def __getattr__(name: str) -> object:
    """Load a public preparation or sampling helper when first requested.

    name must be a name in __all__; otherwise raise AttributeError. Return and
    cache the owning class or callable. Importing the package alone loads no JAX.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner = (
        "_content"
        if name
        in (
            "PreparedTrainingContent",
            "TrainingContentBinding",
            "prepare_training_content",
        )
        else "distributions"
    )
    value = getattr(import_module(f"{__name__}.{owner}"), name)
    globals()[name] = value
    return value
