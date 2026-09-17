"""Check explicit CLI consent, cache handoff and unchanged preparation identity.

Temporary local assets exercise the real preparation authority. Controlled prompt
streams prove that dry runs, cancellation and noninteractive invocations cannot
silently start downloads or delete cache entries. Network integrity is tested by
the asset authority's separate controlled-server tests.
"""

from __future__ import annotations

import argparse

# pyright: reportPrivateUsage=false
import copy
import io
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds import _cli, _cli_assets
from marl_battlegrounds.evaluation import tournament_config as configs


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, missing: bool) -> Path:
    content = b"fixture weights"
    model = tmp_path / "model"
    if not missing:
        model.write_bytes(content)
    descriptor: dict[str, Any] = {
        "release": None,
        "assets": {
            "weights": {
                "sha256": sha256(content).hexdigest(),
                "size_bytes": len(content),
                "role": "model",
                "path": str(model),
                "url": "http://127.0.0.1:1/weights",
            }
        },
    }
    descriptor["snapshot_id"] = configs.snapshot_identity(descriptor)
    path = tmp_path / "source.json"
    path.write_text(json.dumps(descriptor))

    def resolve(value: object) -> dict[str, Any]:
        assert value == str(path)
        return copy.deepcopy(descriptor)

    monkeypatch.setattr(configs, "load_tournament_config", resolve)
    return path


class _Input(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize("answer", ("no\n", "", "maybe\n"))
def test_decline_or_eof_does_not_write_or_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    source = _config(tmp_path, monkeypatch, missing=True)
    monkeypatch.setattr(sys, "stdin", _Input(answer))
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert (
        _cli.main(
            [
                "models",
                "download",
                "--config",
                str(source),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output-config",
                str(tmp_path / "prepared.json"),
            ]
        )
        == 0
    )
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_noninteractive_missing_assets_require_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _config(tmp_path, monkeypatch, missing=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    with pytest.raises(SystemExit) as error:
        _cli.main(["models", "download", "--config", str(source)])
    assert error.value.code == 2


def test_dry_run_has_no_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _config(tmp_path, monkeypatch, missing=True)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert (
        _cli.main(
            [
                "models",
                "download",
                "--config",
                str(source),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_prepared_custom_cache_outputs_usable_priority_next_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _config(tmp_path, monkeypatch, missing=False)
    output = tmp_path / "prepared config.json"
    cache = tmp_path / "custom cache"
    assert (
        _cli.main(
            [
                "models",
                "download",
                "--config",
                str(source),
                "--cache-dir",
                str(cache),
                "--output-config",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["snapshot_id"] == json.loads(source.read_text())["snapshot_id"]
    assert result["assets"]["weights"]["path"] == str(tmp_path / "model")
    text = capsys.readouterr().out
    assert "Prepare Priority Reports" not in text
    assert "tournament --config" in text
    assert "--roles full_report" not in text
    assert not cache.exists()
    before = output.read_bytes()
    assert (
        _cli.main(
            [
                "models",
                "download",
                "--config",
                str(source),
                "--cache-dir",
                str(cache),
                "--output-config",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "roles, needs_reports",
    (
        (("model",), True),
        (("full_report",), True),
        (("outcomes_priority", "full_report"), False),
    ),
)
def test_reuse_next_command_requires_outcomes_separately(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    roles: tuple[str, ...],
    needs_reports: bool,
) -> None:
    namespace = argparse.Namespace(config=None, roles=roles, cache_dir=str(tmp_path))
    prepared = {
        "config": {
            "snapshot_id": "fixture",
            "release": None,
            "record_sources": [{"source_id": "fixture-source"}],
        }
    }
    output = tmp_path / "prepared config.json"
    _cli_assets._print_next(namespace, prepared, output)
    text = capsys.readouterr().out
    assert ("--roles outcomes_priority" in text) == needs_reports
    if needs_reports:
        assert "prepared config-reports.json" in text
    assert ("--metrics full" in text) == ("full_report" in roles)


@pytest.mark.parametrize(
    "options",
    (
        ["--cache-dir", "cache"],
        ["--dry-run", "--yes"],
        ["--dry-run", "--output-config", "out.json"],
        ["--roles", "model,model"],
        ["--roles", "unknown"],
    ),
)
def test_bad_preparation_usage_rejected_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
) -> None:
    from marl_battlegrounds.evaluation import tournament_assets

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Usage failure must precede asset reads")

    monkeypatch.setattr(tournament_assets, "_inspect_tournament_assets", forbidden)
    with pytest.raises(SystemExit) as error:
        _cli.main(["models", "download", *options])
    assert error.value.code == 2


def test_cleanup_preview_cancel_and_explicit_confirm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    content = b"shared fixture"
    digest = sha256(content).hexdigest()
    directory = tmp_path / "sha256"
    directory.mkdir()
    selected = directory / digest
    selected.write_bytes(content)
    other = directory / ("0" * 64)
    other.write_bytes(b"unselected")
    args = ["models", "clean", "--cache-dir", str(tmp_path), "--sha256", digest]
    assert _cli.main([*args, "--dry-run"]) == 0
    assert selected.read_bytes() == content
    monkeypatch.setattr(sys, "stdin", _Input("no\n"))
    assert _cli.main(args) == 0
    assert selected.exists()
    assert _cli.main([*args, "--yes"]) == 0
    assert not selected.exists() and other.read_bytes() == b"unselected"
    assert _cli.main([*args, "--yes"]) == 0
    assert "Already Absent" in capsys.readouterr().out
