"""Launch-only runtime provenance capture for debugger replay recording."""

from __future__ import annotations

import platform as host_platform
import sys
from importlib.metadata import version
from typing import Protocol, cast

from marl_battlegrounds.evaluation.models import CodeRevisionV1
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1


class _RuntimeClient(Protocol):
    platform_version: str


class _RuntimeDevice(Protocol):
    device_kind: str
    client: _RuntimeClient


def capture_debugger_runtime_provenance_v1(
    code_revision: CodeRevisionV1,
    *,
    policy_execution_included: bool = False,
) -> RuntimeProvenanceV1:
    """Capture one path-free host/runtime record for a recording launch.

    Imports of numerical runtimes stay inside this recording-only call so an
    ordinary CLI parse or a read-only replay launch does not acquire a new
    simulator/runtime dependency.
    """
    if type(code_revision) is not CodeRevisionV1:
        raise TypeError("code_revision must be exact CodeRevisionV1")
    if type(policy_execution_included) is not bool:
        raise TypeError("policy_execution_included must be an exact bool")

    import jax

    backend = jax.default_backend()
    runtime_devices = cast(
        list[_RuntimeDevice],
        cast(object, jax.devices(backend)),
    )
    devices = tuple(runtime_devices)
    if not devices:
        raise RuntimeError("the selected JAX backend exposes no runtime device")
    device = devices[0]
    device_name = getattr(device, "device_kind", None) or str(device)
    platform_version = device.client.platform_version
    runtime_version = " ".join(platform_version.split()) or None
    x64_enabled = cast(bool, cast(object, jax.config.read("jax_enable_x64")))
    if type(x64_enabled) is not bool:
        raise RuntimeError("JAX x64 configuration did not return a boolean")

    return RuntimeProvenanceV1(
        python_version=host_platform.python_version(),
        package_version=code_revision.package_version,
        jax_version=version("jax"),
        jaxlib_version=version("jaxlib"),
        numpy_version=version("numpy"),
        pydantic_version=version("pydantic"),
        platform=host_platform.system().lower() or sys.platform,
        machine=host_platform.machine() or "unknown",
        backend=backend,
        device=device_name,
        driver_version=None,
        runtime_version=runtime_version,
        precision="float64" if x64_enabled else "float32",
        environment_count=1,
        batch_shape=(1,),
        policy_execution_included=policy_execution_included,
    )


__all__ = ["capture_debugger_runtime_provenance_v1"]
