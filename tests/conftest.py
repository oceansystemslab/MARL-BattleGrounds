"""Release memory between tests so one long shard process stays bounded.

Each CI shard runs all its tests in one pytest process. JAX keeps every compiled
program and glibc keeps freed buffers in its heap, so without a release the
process grows with every training test toward the 16 GB of a hosted runner.
After each test's protocol ends, when pytest has dropped its fixture values,
collect garbage and return free heap pages to the operating system. When the
next test comes from another file, or none follows, first clear the finished
file's own functools caches and JAX's compiled programs; tests in one file keep
sharing theirs. Objects that exist after collection are frozen once, so each
garbage collection skips the imported modules and collected items. This never
imports JAX and changes no test or assertion.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from collections.abc import Callable, Generator
from types import ModuleType

import pytest


def _find_malloc_trim() -> Callable[[int], int] | None:
    if sys.platform != "linux":
        return None
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except OSError, AttributeError:
        return None


_MALLOC_TRIM = _find_malloc_trim()


def _clear_module_caches(item: pytest.Item) -> None:
    module: ModuleType | None = (
        item.module if isinstance(item, pytest.Function) else None
    )
    if module is None:
        return
    values: list[object] = list(vars(module).values())
    for value in values:
        clear = getattr(value, "cache_clear", None)
        if getattr(value, "__module__", None) == module.__name__ and callable(clear):
            clear()


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session: pytest.Session) -> None:
    gc.collect()
    gc.freeze()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object | None, object | None]:
    result = yield
    if nextitem is None or nextitem.path != item.path:
        _clear_module_caches(item)
        jax = sys.modules.get("jax")
        if jax is not None:
            jax.clear_caches()
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)
    return result
