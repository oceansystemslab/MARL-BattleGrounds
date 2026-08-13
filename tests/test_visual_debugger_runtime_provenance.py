"""Launch-boundary proofs for debugger recording runtime provenance."""

from __future__ import annotations

import platform as host_platform
from importlib.metadata import version
from typing import cast

import jax
import pytest
import scripts.dev.visual_debugger.runtime_provenance as provenance_module
from scripts.dev.visual_debugger.runtime_provenance import (
    capture_debugger_runtime_provenance_v1,
)

from marl_battlegrounds.evaluation.models import CodeRevisionV1


def _revision() -> CodeRevisionV1:
    return CodeRevisionV1(
        package_version="9.8.7",
        commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        is_dirty=False,
        dirty_patch_digest=None,
    )


def test_runtime_provenance_captures_exact_single_environment_host_facts() -> None:
    provenance = capture_debugger_runtime_provenance_v1(_revision())

    assert provenance.package_version == "9.8.7"
    assert provenance.python_version == host_platform.python_version()
    assert provenance.jax_version == version("jax")
    assert provenance.jaxlib_version == version("jaxlib")
    assert provenance.numpy_version == version("numpy")
    assert provenance.pydantic_version == version("pydantic")
    assert provenance.backend == jax.default_backend()
    assert provenance.device
    x64_enabled = cast(bool, cast(object, jax.config.read("jax_enable_x64")))
    assert provenance.precision == ("float64" if x64_enabled else "float32")
    assert provenance.environment_count == 1
    assert provenance.batch_shape == (1,)
    assert provenance.policy_execution_included is False
    assert "path" not in provenance.model_dump_json().lower()
    assert "token" not in provenance.model_dump_json().lower()


def test_runtime_provenance_requires_exact_code_revision() -> None:
    with pytest.raises(TypeError, match="exact CodeRevisionV1"):
        capture_debugger_runtime_provenance_v1(
            object(),  # pyright: ignore[reportArgumentType]
        )


def test_runtime_provenance_normalizes_multiline_platform_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        platform_version = "PJRT C API\ncuda 13000\tbuild"

    class FakeDevice:
        device_kind = "test-device"
        client = FakeClient()

    def fake_devices(_backend: str) -> list[FakeDevice]:
        return [FakeDevice()]

    monkeypatch.setattr(jax, "devices", fake_devices)

    provenance = provenance_module.capture_debugger_runtime_provenance_v1(_revision())

    assert provenance.runtime_version == "PJRT C API cuda 13000 build"
