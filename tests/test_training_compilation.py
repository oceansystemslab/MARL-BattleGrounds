"""Check the shared outer-JIT policy and read-only continuation identity.

Each caller receives its own options dictionary. The identity records current
JAX settings and stable M8 runtime fields, including random-number settings that
can change the next draw. Capturing it does not alter process settings. These
host checks do not establish numerical equivalence or GPU compilation cost.
"""

import os
from typing import Any, cast

import jax
import pytest

from marl_battlegrounds.training._compilation import (
    execution_identity,
    training_compiler_options,
)


def test_options_are_independent_without_changing_global_settings() -> None:
    before = dict(os.environ)
    first = training_compiler_options()
    assert first == {"xla_gpu_autotune_level": 0}
    first["xla_gpu_autotune_level"] = 4
    assert training_compiler_options() == {"xla_gpu_autotune_level": 0}
    assert dict(os.environ) == before


def test_current_runtime_can_be_reused_without_recapturing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation import runtime_provenance

    def forbidden_capture(*args: object, **kwargs: object) -> None:
        pytest.fail("An already captured runtime must not be captured twice")

    monkeypatch.setattr(
        runtime_provenance, "capture_runtime_provenance", forbidden_capture
    )
    runtime: dict[str, Any] = {
        "backend": "cpu",
        "device": "test-kind",
        "runtime_version": "test-runtime",
        "precision": "float32",
        "batch_shape": [4],
        "timestamp": "not a continuation identity",
    }
    monkeypatch.setenv("XLA_FLAGS", "--declared-test-setting")
    before = dict(os.environ)
    result = execution_identity(runtime=runtime)
    assert result["runtime"] == {
        key: runtime[key]
        for key in ("backend", "device", "runtime_version", "precision")
    }
    assert result["compiler_options"] == training_compiler_options()
    assert result["declared_xla_flags"] == "--declared-test-setting"
    settings = cast(Any, jax.config)
    assert result["jax"] == {
        "enable_x64": settings.jax_enable_x64,
        "disable_jit": settings.jax_disable_jit,
        "default_matmul_precision": settings.jax_default_matmul_precision,
        "default_prng_impl": settings.jax_default_prng_impl,
        "threefry_partitionable": settings.jax_threefry_partitionable,
        "threefry_gpu_kernel_lowering": settings.jax_threefry_gpu_kernel_lowering,
        "enable_pgle": settings.jax_enable_pgle,
    }
    with jax.threefry_partitionable(not settings.jax_threefry_partitionable):
        changed = execution_identity(runtime=runtime)
    assert (
        changed["jax"]["threefry_partitionable"]
        != result["jax"]["threefry_partitionable"]
    )
    assert dict(os.environ) == before
