"""Own MAPPO compilation settings and checked continuation identity.

Training's outer JAX calls use training_compiler_options. Pure numerical helpers
remain composable and do not set compiler options themselves. execution_identity
records the active runtime and numerical settings used to check a learner resume.
Importing this module changes no JAX setting and initializes no device.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any, cast


def training_compiler_options() -> dict[str, int]:
    """Return a fresh compiler-options dictionary for an outer MAPPO JAX call.

    Disable GPU kernel timing-based selection so separate compilations use the
    same default choices. This changes no global JAX or environment setting.
    Pass the result only to a top-level jax.jit; JAX rejects options on nested
    jit calls. The setting controls GPU compilation and changes no CPU kernel
    choice. This policy alone makes no promise across different devices,
    libraries or compiler flags.
    """
    return {"xla_gpu_autotune_level": 0}


def execution_identity(
    *, runtime: Mapping[str, object] | None = None
) -> dict[str, Any]:
    """Describe the current compilation policy and numerical runtime for resume.

    Parameters
    ----------
    runtime : mapping or None, default None
        Current M8 runtime-provenance JSON, when already captured by the caller.
        Reuse its backend, device kind, runtime version and precision. None calls
        that shared capture owner and may initialize JAX's selected backend.
        Do not pass a saved or guessed record as the current runtime.

    Returns
    -------
    dict
        Fresh finite JSON with schema version, outer-JIT compiler options, stable
        runtime facts, current JAX precision/JIT/matmul/random settings, profile
        feedback use and declared XLA_FLAGS. No timestamp, process ID or compiled
        binary enters the identity. The device field is the actual JAX device
        kind, not a GPU UUID. The prepared launcher owns physical UUID/PCI checks.

    Notes
    -----
    Read-only host work; no model call, random key, file write or configuration
    change. XLA_FLAGS describes the current environment string. Changing that
    string after backend initialization does not prove parsed flags changed.
    Source and dependency identities remain owned by checkpoints.runtime_identity.
    """
    import jax

    if runtime is None:
        from marl_battlegrounds.evaluation.runtime_provenance import (
            capture_runtime_provenance,
        )

        runtime = capture_runtime_provenance(version("marl-battlegrounds")).model_dump(
            mode="json"
        )
    settings = cast(Any, jax.config)
    return {
        "schema_version": 1,
        "compiler_options": training_compiler_options(),
        "runtime": {
            name: runtime[name]
            for name in ("backend", "device", "runtime_version", "precision")
        },
        "jax": {
            "enable_x64": settings.jax_enable_x64,
            "disable_jit": settings.jax_disable_jit,
            "default_matmul_precision": settings.jax_default_matmul_precision,
            "default_prng_impl": settings.jax_default_prng_impl,
            "threefry_partitionable": settings.jax_threefry_partitionable,
            "threefry_gpu_kernel_lowering": settings.jax_threefry_gpu_kernel_lowering,
            "enable_pgle": settings.jax_enable_pgle,
        },
        "declared_xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
