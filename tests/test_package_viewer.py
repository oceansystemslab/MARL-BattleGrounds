"""Check package Viewer ownership, lazy options and scoped resource launch.

Old developer imports must share exact objects and patched globals. Browser
resources stay alive until serving ends; invalid inputs cannot start a server.
Installed-wheel HTTP/browser evidence is collected separately at packet closeout.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from marl_battlegrounds.viewer.options import (
    resolve_playback_options,
    validate_playback_options,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundle


@pytest.mark.parametrize(
    "name",
    (
        "replay_service",
        "replay_protocol",
        "server",
        "presentation",
        "presentation_protocol",
        "match_summary",
        "no_shared_visual",
        "local_oracle_corpse_overlay",
    ),
)
def test_old_runtime_paths_are_exact_module_aliases(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = importlib.import_module(f"scripts.dev.visual_debugger.{name}")
    package = importlib.import_module(f"marl_battlegrounds.viewer.{name}")
    assert old is package
    marker = object()
    monkeypatch.setattr(old, "_packet8_alias_probe", marker, raising=False)
    assert package._packet8_alias_probe is marker


def test_shared_options_import_no_numerical_or_repository_runtime() -> None:
    code = """
import argparse
import sys
from marl_battlegrounds.viewer.options import resolve_playback_options
options = resolve_playback_options(argparse.Namespace(view='oracle'))
assert options.view == 'researcher'
assert options.supplied == frozenset({'view'})
for prefix in ('jax', 'numpy', 'matplotlib', 'scripts', 'marl_battlegrounds.core'):
    loaded = any(n == prefix or n.startswith(prefix + '.') for n in sys.modules)
    assert not loaded, prefix
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr


def test_playback_presence_and_static_compatibility() -> None:
    defaults = resolve_playback_options(argparse.Namespace())
    assert defaults.frame_index is None
    assert defaults.port == 0
    assert not defaults.ranges
    assert defaults.supplied == frozenset()
    validate_playback_options(defaults)
    with pytest.raises(ValueError, match="frame-index is required"):
        validate_playback_options(
            resolve_playback_options(argparse.Namespace(static=True))
        )
    with pytest.raises(ValueError, match="--port is unavailable"):
        validate_playback_options(
            resolve_playback_options(
                argparse.Namespace(static=True, frame_index=0, port=0)
            )
        )
    validate_playback_options(
        resolve_playback_options(
            argparse.Namespace(static=True, frame_index=0, ranges=False)
        )
    )


def test_loaded_replay_uses_resources_until_server_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds.viewer import launch, replay_service, server

    calls: list[str] = []
    active = False
    bundle = cast("LoadedReplayBundle", object())

    class Service:
        def __init__(self, received: object, **_kwargs: object) -> None:
            assert received is bundle
            calls.append("validate")

        def current_frame(self, **_kwargs: object) -> object:
            return None

        apply_command = current_frame
        current_timeline = current_frame
        current_presentation = current_frame
        current_metric_report = current_frame
        metric_analysis = current_frame
        metric_catalog = current_frame
        episode_details = current_frame

    @contextmanager
    def scoped_resource(_resource: object) -> Generator[Path]:
        nonlocal active
        active = True
        calls.append("resource")
        try:
            yield tmp_path
        finally:
            active = False
            calls.append("release")

    def serve(_service: object, **kwargs: object) -> int:
        assert active
        assert kwargs["asset_root"] == tmp_path
        assert kwargs["port"] == 0
        assert kwargs["open_browser"] is False
        calls.append("serve")
        return 17

    monkeypatch.setattr(replay_service, "ReplayViewerService", Service)
    monkeypatch.setattr(launch, "as_file", scoped_resource)
    monkeypatch.setattr(server, "serve_browser_debugger", serve)
    options = resolve_playback_options(argparse.Namespace(no_open=True))
    assert launch.launch_replay(None, options=options, loaded_bundle=bundle) == 17
    assert not active
    assert calls == ["validate", "resource", "serve", "release"]


def test_invalid_replay_does_not_resolve_assets_or_serve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from marl_battlegrounds.viewer import launch

    def fail_resources(_package: str) -> None:
        raise AssertionError("Invalid replay must fail before assets or serving")

    path = tmp_path / "invalid replay.json"
    path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(launch, "files", fail_resources)
    with pytest.raises(ValueError, match="Replay could not be loaded"):
        launch.launch_replay(
            path, options=resolve_playback_options(argparse.Namespace())
        )
