"""Describe the actual software and device used by a recorded run.

Runtime capture is host-only and may initialize JAX's default backend. JAX is
imported inside the capture call so ordinary metadata readers do not initialize
numerical devices just by importing this module. Records contain versions and
device descriptions, not machine paths, measured speed or guessed driver data.
"""

from __future__ import annotations

import platform as host_platform
import sys
from importlib.metadata import version
from typing import Protocol, cast

from marl_battlegrounds.evaluation.models import CodeRevisionV1
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1


class _RuntimeClient(Protocol):
    """Minimum runtime-client fields read when describing an active JAX backend."""

    platform_version: str


class _RuntimeDevice(Protocol):
    """Minimum device fields needed to record backend hardware and runtime version."""

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

    Parameters
    ----------
    code_revision : CodeRevisionV1
        Exact validated CodeRevisionV1 for this recording launch.
    policy_execution_included : bool
        Whether policy work belongs to the record,
        default False for manual debugger stepping.

    Returns
    -------
    RuntimeProvenanceV1
        RuntimeProvenanceV1 for one environment on the actual default JAX backend.

    Raises
    ------
    TypeError
        code_revision has another type or the inclusion flag is not bool.
    RuntimeError
        JAX exposes no selected device or invalid runtime metadata.

    This host-only call may initialize JAX devices. It reads installed package
    versions and returns metadata; it writes no files or local paths.
    """
    if type(code_revision) is not CodeRevisionV1:
        raise TypeError("code_revision must be exact CodeRevisionV1")
    if type(policy_execution_included) is not bool:
        raise TypeError("policy_execution_included must be an exact bool")
    return capture_runtime_provenance(
        code_revision.package_version,
        policy_execution_included=policy_execution_included,
    )


def capture_runtime_provenance(
    package_version: str,
    *,
    policy_execution_included: bool = True,
    num_envs: int = 1,
) -> RuntimeProvenanceV1:
    """Describe the active numerical runtime for one evaluation pass.

    Parameters
    ----------
    package_version : str
        Version of the MARL-BGs source/package being recorded.
    policy_execution_included : bool
        Whether policy execution is included in this
        run's declared work; default True.
    num_envs : int
        Environment count recorded in batch_shape=(num_envs,), default 1.

    Returns
    -------
    RuntimeProvenanceV1
        Validated RuntimeProvenanceV1 with Python/library versions, default
        backend, first device, runtime version and current JAX precision setting.
        driver_version is None because this function does not query the driver.

    Raises
    ------
    RuntimeError
        The selected backend has no device or its precision flag
        is not a boolean. JAX initialization errors also propagate.
    ValueError
        Supplied metadata violates the provenance model.

    Host-only: imports JAX here, may initialize the default backend, and reads
    installed distribution metadata. Missing distributions propagate their
    lookup error. No device timing, hardware speed claim or file write occurs.
    """

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
        package_version=package_version,
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
        environment_count=num_envs,
        batch_shape=(num_envs,),
        policy_execution_included=policy_execution_included,
    )


__all__ = ["capture_debugger_runtime_provenance_v1", "capture_runtime_provenance"]
