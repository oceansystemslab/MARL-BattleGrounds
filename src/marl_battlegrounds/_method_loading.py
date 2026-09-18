"""Import explicitly named trusted Python callables without loading a runtime.

Command factories and tournament loaders share this grammar. Importing this
module uses only the standard library. Calling a selected function remains the
caller's responsibility; imported researcher code may have its own side effects.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast


def installed_callable(reference: str) -> Callable[..., object]:
    """Import a trusted ``module:function`` and return it without calling it.

    The attribute must be one module-level name, not a dotted traversal.
    Invalid names and noncallable attributes raise ValueError. Import and
    attribute errors retain their original cause. No files are downloaded and
    no path is added to Python's import search path.
    """
    if not isinstance(cast(object, reference), str) or reference.count(":") != 1:
        raise ValueError("Controller factory/loader must use module:function")
    module_name, attribute = reference.split(":")
    if not module_name or not attribute or "." in attribute:
        raise ValueError(
            "Controller factory/loader must name one module-level callable"
        )
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise ValueError("Controller factory/loader is not callable")
    return value
