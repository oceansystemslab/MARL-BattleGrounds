"""Check one-pass asset consent, prepared configs and selected cache cleanup.

These host tests use temporary bytes and controlled download responses. They
prove that confirmation bounds actual transfers, existing content is not hashed
again, prepared output never replaces source files, and cleanup touches only its
explicit regular cache entries. No games, released catalog or external network.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
from collections.abc import Callable
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.evaluation import _asset_commands as commands
from marl_battlegrounds.evaluation import tournament_assets as assets
from marl_battlegrounds.evaluation import tournament_config


def _descriptor(path: Path, data: bytes, *, present: bool = True) -> dict[str, Any]:
    if present:
        path.write_bytes(data)
    return {
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
        "role": "model",
        "path": str(path),
        "url": "https://fixture.invalid/" + path.name,
    }


def _resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolved(config: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(config)

    monkeypatch.setattr(tournament_config, "load_tournament_config", resolved)


def _opener(
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, bytes],
    *,
    before_open: Callable[[str], None] | None = None,
) -> list[str]:
    opened: list[str] = []

    class Opener:
        def open(self, url: str, *, timeout: int) -> BytesIO:
            assert timeout > 0
            opened.append(url)
            if before_open is not None:
                before_open(url)
            return BytesIO(payloads[url])

    def opener(*_args: object) -> Opener:
        return Opener()

    monkeypatch.setattr(assets, "build_opener", opener)
    return opened


def test_inspection_and_finish_share_hashes_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    weights = _descriptor(tmp_path / "weights", b"weights" * 100)
    metadata = _descriptor(tmp_path / "registration", b"{}")
    metadata["role"] = "registration"
    original = assets._hash_file
    hashes: list[Path] = []

    def measured(path: Path) -> tuple[str, int]:
        hashes.append(path)
        return original(path)

    monkeypatch.setattr(assets, "_hash_file", measured)
    context = assets._inspect_tournament_assets(
        {"assets": {"weights": weights, "registration": metadata}},
        cache_dir=tmp_path / "cache",
        roles=("model",),
    )
    assert set(hashes) == {tmp_path / "weights", tmp_path / "registration"}
    before = len(hashes)
    assert context.finish(download=True, confirmed=True)["missing"] == []
    assert context.finish()["bytes_downloaded"] == 0
    assert len(hashes) == before
    assert not (tmp_path / "cache").exists()


def test_confirmation_deduplicates_missing_aliases_and_transfers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    descriptor = _descriptor(tmp_path / "weights", b"abc", present=False)
    opened = _opener(monkeypatch, {descriptor["url"]: b"abc"})
    context = assets._inspect_tournament_assets(
        {"assets": {"a": descriptor, "b": descriptor}},
        cache_dir=tmp_path / "cache",
    )
    assert context.preview["missing"] == ["a", "b"]
    assert context.preview["bytes_missing"] == 3
    expected = context.expected_config
    prepared = context.finish(download=True, confirmed=True)
    assert prepared["bytes_downloaded"] == 3
    assert prepared["config"] == expected
    assert opened == [descriptor["url"]]
    assert prepared["verified"] == ["a", "b"]


@pytest.mark.parametrize("during_download", [False, True])
def test_present_content_disappearing_never_gets_unconfirmed_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    during_download: bool,
) -> None:
    _resolve(monkeypatch)
    missing = _descriptor(tmp_path / "missing", b"first", present=False)
    present = _descriptor(tmp_path / "present", b"second")

    def remove(_url: str) -> None:
        (tmp_path / "present").unlink()

    opened = _opener(
        monkeypatch,
        {missing["url"]: b"first", present["url"]: b"second"},
        before_open=remove if during_download else None,
    )
    context = assets._inspect_tournament_assets(
        {"assets": {"a-missing": missing, "z-present": present}},
        cache_dir=tmp_path / "cache",
    )
    if not during_download:
        remove("")
    with pytest.raises(ValueError, match="download changed after preview"):
        context.finish(download=True, confirmed=True)
    assert present["url"] not in opened
    assert opened == [missing["url"]]


def test_scope_rejects_already_checked_asset_removed_during_later_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    present = _descriptor(tmp_path / "present", b"first")
    missing = _descriptor(tmp_path / "missing", b"second", present=False)

    def remove(_url: str) -> None:
        (tmp_path / "present").unlink()

    _opener(monkeypatch, {missing["url"]: b"second"}, before_open=remove)
    context = assets._inspect_tournament_assets(
        {"assets": {"a-present": present, "z-missing": missing}},
        cache_dir=tmp_path / "cache",
    )
    with pytest.raises(ValueError, match="changed during verification"):
        context.finish(download=True, confirmed=True)


def test_download_guard_rejects_new_request_before_any_cache_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    present = _descriptor(tmp_path / "present", b"first")
    context = assets._inspect_tournament_assets(
        {"assets": {"present": present}},
        cache_dir=tmp_path / "cache",
    )
    opened = _opener(monkeypatch, {present["url"]: b"first"})
    (tmp_path / "present").unlink()
    with pytest.raises(ValueError, match="download changed"):
        context.finish(download=True, confirmed=True)
    assert opened == []
    assert not (tmp_path / "cache").exists()


def test_confirmed_content_cannot_be_transferred_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    descriptor = _descriptor(tmp_path / "weights", b"first", present=False)
    opened = _opener(monkeypatch, {descriptor["url"]: b"first"})
    context = assets._inspect_tournament_assets(
        {"assets": {"weights": descriptor}},
        cache_dir=tmp_path / "cache",
    )
    prepared = context.finish(download=True, confirmed=True)
    Path(prepared["config"]["assets"]["weights"]["path"]).unlink()
    with pytest.raises(ValueError, match="already had a download attempt"):
        context.finish(download=True, confirmed=True)
    assert len(opened) == 1


def test_prepared_output_is_atomic_idempotent_and_preserves_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    data = _descriptor(tmp_path / "weights", b"abc", present=False)
    config = {"assets": {"weights": data}}
    _opener(monkeypatch, {data["url"]: b"abc"})
    context = assets._inspect_tournament_assets(config, cache_dir=tmp_path / "cache")
    target = commands._preflight_prepared_config(
        tmp_path / "prepared.json",
        config=context.expected_config,
    )
    assert not target.path.exists()
    prepared = context.finish(download=True, confirmed=True)
    target.write(prepared["config"])
    before = target.path.stat()
    target.write(prepared["config"])
    assert target.path.stat() == before
    assert tournament_config.snapshot_identity(
        prepared["config"]
    ) == tournament_config.snapshot_identity(config)
    assert not list(tmp_path.glob(".prepared-*"))


@pytest.mark.parametrize("kind", ["input", "different", "symlink", "parent"])
def test_prepared_output_preflight_rejects_without_mutation(
    tmp_path: Path,
    kind: str,
) -> None:
    config: dict[str, Any] = {"assets": {}}
    path = tmp_path / "prepared.json"
    source = tmp_path / "source.json"
    source.write_bytes(b"source")
    if kind == "input":
        path = source
    elif kind == "different":
        path.write_bytes(b"different")
    elif kind == "symlink":
        path.symlink_to(source)
    else:
        path = tmp_path / "absent" / "config.json"
    with pytest.raises(ValueError):
        commands._preflight_prepared_config(path, config=config, input_config=source)
    assert source.read_bytes() == b"source"
    assert not (tmp_path / "absent").exists()


def test_prepared_output_late_conflict_never_overwrites(
    tmp_path: Path,
) -> None:
    config: dict[str, Any] = {"assets": {}}
    target = commands._preflight_prepared_config(
        tmp_path / "prepared.json", config=config
    )
    target.path.write_bytes(b"arrived later")
    with pytest.raises(ValueError, match="different"):
        target.write(config)
    assert target.path.read_bytes() == b"arrived later"


def test_prepared_output_fsync_failure_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config: dict[str, Any] = {"assets": {}}
    target = commands._preflight_prepared_config(
        tmp_path / "prepared.json", config=config
    )

    def fail(_fd: int) -> None:
        raise OSError("fixture sync failure")

    monkeypatch.setattr(commands.os, "fsync", fail)
    with pytest.raises(OSError, match="fixture sync"):
        target.write(config)
    assert not target.path.exists()
    assert not list(tmp_path.glob(".prepared-*"))


def _cache(tmp_path: Path) -> tuple[Path, str, str]:
    directory = tmp_path / "cache" / "sha256"
    directory.mkdir(parents=True)
    first, second = sha256(b"first").hexdigest(), sha256(b"second").hexdigest()
    (directory / first).write_bytes(b"first")
    (directory / second).write_bytes(b"second")
    return directory, first, second


def test_cleanup_preview_is_read_only_and_apply_is_selected_and_repeatable(
    tmp_path: Path,
) -> None:
    directory, first, second = _cache(tmp_path)
    absent = "a" * 64
    cleanup = commands._inspect_cache_cleanup(
        [first, first, absent],
        cache_dir=directory.parent,
    )
    assert cleanup.preview["bytes"] == 5
    assert len(cleanup.preview["entries"]) == 2
    assert (directory / first).exists()
    assert cleanup.apply() == {"deleted": [first], "absent": [absent]}
    assert (directory / second).read_bytes() == b"second"
    assert cleanup.apply() == {"deleted": [], "absent": [first, absent]}


@pytest.mark.parametrize("kind", ["link", "directory", "bad-digest", "root-link"])
def test_cleanup_checks_entire_set_and_never_follows_symlinks(
    tmp_path: Path,
    kind: str,
) -> None:
    directory, first, second = _cache(tmp_path)
    target = directory / second
    source = tmp_path / "source"
    source.write_bytes(b"keep")
    if kind == "link":
        target.unlink()
        target.symlink_to(source)
    elif kind == "directory":
        target.unlink()
        target.mkdir()
    elif kind == "root-link":
        other = tmp_path / "outside"
        directory.rename(other)
        directory.symlink_to(other, target_is_directory=True)
    else:
        second = "../" + second
    with pytest.raises(ValueError):
        commands._inspect_cache_cleanup([first, second], cache_dir=directory.parent)
    assert (directory / first).read_bytes() == b"first"
    assert source.read_bytes() == b"keep"


def test_cleanup_detected_later_change_blocks_first_delete(
    tmp_path: Path,
) -> None:
    directory, first, second = _cache(tmp_path)
    cleanup = commands._inspect_cache_cleanup(
        [first, second], cache_dir=directory.parent
    )
    (directory / second).write_bytes(b"changed")
    with pytest.raises(OSError, match=r"changed after preview.*Deleted entries: None"):
        cleanup.apply()
    assert (directory / first).exists()


def test_cleanup_partial_failure_reports_removed_entries_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, first, second = _cache(tmp_path)
    cleanup = commands._inspect_cache_cleanup(
        [first, second], cache_dir=directory.parent
    )
    original = commands.os.unlink

    def failing(path: str | Path, *, dir_fd: int | None = None) -> None:
        if path == second:
            raise OSError("fixture unlink failure")
        original(path, dir_fd=dir_fd)

    monkeypatch.setattr(commands.os, "unlink", failing)
    with pytest.raises(OSError, match=f"Deleted entries: {first}"):
        cleanup.apply()
    assert not (directory / first).exists()
    assert (directory / second).exists()
    monkeypatch.setattr(commands.os, "unlink", original)
    assert cleanup.apply() == {"deleted": [second], "absent": [first]}


def test_empty_cache_inspection_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "not-created"
    cleanup = commands._inspect_cache_cleanup(["a" * 64], cache_dir=path)
    assert cleanup.preview["bytes"] == 0
    assert cleanup.apply() == {"deleted": [], "absent": ["a" * 64]}
    assert not path.exists()


def test_no_download_finish_rejects_disappearance_before_prepared_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    present = _descriptor(tmp_path / "present", b"first")
    context = assets._inspect_tournament_assets(
        {"assets": {"present": present}},
        cache_dir=tmp_path / "cache",
    )
    commands._preflight_prepared_config(
        tmp_path / "prepared.json",
        config=context.expected_config,
    )
    (tmp_path / "present").unlink()
    with pytest.raises(ValueError, match="missing after preview"):
        context.finish(download=False, confirmed=True)
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "prepared.json").exists()


def test_final_request_guard_checks_after_temporary_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve(monkeypatch)
    descriptor = _descriptor(tmp_path / "missing", b"first", present=False)
    context = assets._inspect_tournament_assets(
        {"assets": {"missing": descriptor}},
        cache_dir=tmp_path / "cache",
    )
    opened = _opener(monkeypatch, {descriptor["url"]: b"first"})
    original = assets.tempfile.mkstemp

    def changed(*, prefix: str, dir: Path) -> tuple[int, str]:
        result = original(prefix=prefix, dir=dir)
        context.verifier.assets["missing"]["url"] += "-changed"
        return result

    monkeypatch.setattr(assets.tempfile, "mkstemp", changed)
    with pytest.raises(ValueError, match="download changed after preview"):
        context.finish(download=True, confirmed=True)
    assert opened == []
    assert not list((tmp_path / "cache").rglob(".download-*"))


def test_prepared_config_relocation_preserves_relative_sources(
    tmp_path: Path,
) -> None:
    from canonical_fixtures import config_descriptor
    from marl_battlegrounds.evaluation.tournament_config import canonical_json

    source = tmp_path / "source"
    source.mkdir()
    config = config_descriptor(root=source)
    for identifier, row in config["assets"].items():
        (source / identifier).write_bytes(b"{}")
        row.update(path=identifier, sha256=sha256(b"{}").hexdigest(), size_bytes=2)
    config["snapshot_id"] = tournament_config.snapshot_identity(config)
    config_path = source / "config.json"
    config_path.write_bytes(canonical_json(config))
    output = tmp_path / "elsewhere"
    output.mkdir()
    context = assets._inspect_tournament_assets(config_path, roles=())
    target = commands._preflight_prepared_config(
        output / "prepared.json",
        config=context.expected_config,
        input_config=config_path,
    )
    prepared = context.finish(confirmed=True)
    target.write(prepared["config"])
    reread = tournament_config.load_tournament_config(target.path)
    assert reread["snapshot_id"] == config["snapshot_id"]
    assert all(Path(row["path"]).parent == source for row in reread["assets"].values())
    assert config_path.read_bytes() == canonical_json(config)
