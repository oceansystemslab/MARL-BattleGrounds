"""Own prepared-config publication and deliberate content-cache maintenance.

Package commands ask for confirmation before using these host-only helpers.
They share tournament asset integrity and serialization authorities, import no
numerical runtime, and never download, load methods or modify source bundles.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.evaluation.tournament_assets import (
    _DIGEST,  # pyright: ignore[reportPrivateUsage]
    _json_bytes,  # pyright: ignore[reportPrivateUsage]
    _sync_directory,  # pyright: ignore[reportPrivateUsage]
    default_cache_dir,
)
from marl_battlegrounds.evaluation.tournament_config import snapshot_identity

__all__ = ["_inspect_cache_cleanup", "_preflight_prepared_config"]


def _existing_output(path: Path, expected: bytes) -> bool:
    """Return whether an identical regular prepared file exists at path.

    Missing files return False. Symlinks, non-files and different contents raise
    ValueError. This is repeated immediately before atomic no-overwrite install;
    a concurrent new entry is still protected by the hard-link publication.
    """
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Prepared config output must be a regular file: {path}")
    if path.read_bytes() != expected:
        raise ValueError(
            f"Prepared config output already contains different data: {path}"
        )
    return True


@dataclass(frozen=True)
class _PreparedConfig:
    """Hold a checked output path and its original scientific content identity.

    The parent already exists. write installs one complete file without replacing
    another file. Locations may change during verification; science may not.
    """

    path: Path
    identity: str

    def write(self, config: Mapping[str, Any]) -> Path:
        """Publish strict prepared JSON and synchronize its file and directory.

        config is the successful preparation result, including verified absolute
        path hints. Returns the absolute target; an identical existing file is a
        no-op. Different science or output bytes raise ValueError. I/O failures
        propagate, and no partial JSON or overwritten file is published.
        """
        if snapshot_identity(config) != self.identity:
            raise ValueError("Prepared configuration changed after output preflight")
        data = _json_bytes(dict(config)) + b"\n"
        if _existing_output(self.path, data):
            return self.path
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".prepared-", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, self.path)
            except FileExistsError:
                if not _existing_output(self.path, data):
                    raise FileExistsError(
                        "Prepared config destination changed during publication"
                    ) from None
            temporary.unlink()
            _sync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return self.path


def _preflight_prepared_config(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    input_config: str | Path | None = None,
) -> _PreparedConfig:
    """Check a prepared-output destination before download or cache mutation.

    path is relative to cwd or absolute; its parent must already exist. config
    contains expected final selected path hints. input_config, when a file was
    supplied, cannot be replaced even with identical bytes. Returns a private
    publication target. Existing different output, symlinks, invalid JSON and a
    missing parent fail without creating files. An identical output is allowed.
    """
    target = Path(path).expanduser().absolute()
    if not target.parent.is_dir():
        raise ValueError(
            f"Prepared config parent directory does not exist: {target.parent}"
        )
    if input_config is not None:
        source = Path(input_config).expanduser().resolve()
        if target.resolve() == source or (
            target.exists() and source.exists() and os.path.samefile(target, source)
        ):
            raise ValueError("Prepared config output must not replace its input config")
    _existing_output(target, _json_bytes(dict(config)) + b"\n")
    return _PreparedConfig(target, snapshot_identity(config))


def _cache_stamp(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return cache entry identity, size and mutation counters from lstat data."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _selected_entry(path: Path) -> tuple[int, int, int, int, int] | None:
    """Read one cache entry without following a final symlink; missing is None."""
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Selected cache entry must be a regular file: {path}")
    return _cache_stamp(value)


@dataclass(frozen=True)
class _CacheCleanup:
    """Hold one inspected deletion set, requiring exclusive local maintenance.

    Users must stop concurrent writers/downloads for these entries. Checking and
    unlinking a pathname cannot atomically protect against a hostile replacer.
    Only the named regular files inside this content directory may be removed.
    """

    cache_dir: Path
    entries: Mapping[str, tuple[int, int, int, int, int] | None]
    directory_identity: tuple[int, int] | None

    @property
    def preview(self) -> dict[str, Any]:
        """Return selected digest/size/presence rows and deduplicated total bytes."""
        return {
            "cache_dir": str(self.cache_dir),
            "entries": [
                {
                    "sha256": key,
                    "size_bytes": 0 if stamp is None else stamp[2],
                    "present": stamp is not None,
                }
                for key, stamp in self.entries.items()
            ],
            "bytes": sum(
                stamp[2] for stamp in self.entries.values() if stamp is not None
            ),
        }

    def apply(self) -> dict[str, list[str]]:
        """Delete confirmed entries with identity checks and directory syncs.

        Returns deleted and already absent digest lists. No source files, folders
        or symlink targets are deleted. Every supplied entry was checked before
        this call, and all are rechecked before the first removal. An I/O or
        detected-change failure reports earlier deletions in its OSError text;
        a later fresh inspection can retry. Several deletions are not atomic.
        """
        directory = self.cache_dir / "sha256"
        deleted: list[str] = []
        absent: list[str] = []
        descriptor: int | None = None
        try:
            if not directory.exists() and not directory.is_symlink():
                if self.directory_identity is not None:
                    raise ValueError("Cache content directory changed after preview")
                return {"deleted": [], "absent": list(self.entries)}
            descriptor = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            current = os.fstat(descriptor)
            if self.directory_identity != (current.st_dev, current.st_ino):
                raise ValueError("Cache content directory changed after preview")
            for key, expected in self.entries.items():
                actual = _selected_entry(directory / key)
                if actual != expected and actual is not None:
                    raise ValueError(f"Cache entry {key} changed after preview")
            for key, expected in self.entries.items():
                try:
                    current = os.stat(key, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    absent.append(key)
                    continue
                if (
                    not stat.S_ISREG(current.st_mode)
                    or _cache_stamp(current) != expected
                ):
                    raise ValueError(f"Cache entry {key} changed after preview")
                os.unlink(key, dir_fd=descriptor)
                deleted.append(key)
                os.fsync(descriptor)
        except (OSError, ValueError) as error:
            raise OSError(
                f"Cache cleanup stopped: {error}. Deleted entries: "
                + (", ".join(deleted) if deleted else "None")
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return {"deleted": deleted, "absent": absent}


def _inspect_cache_cleanup(
    digests: Sequence[str],
    *,
    cache_dir: str | Path | None = None,
) -> _CacheCleanup:
    """Inspect only explicit SHA256 cache filenames, without writing or deleting.

    digests must be a nonempty sequence of complete lowercase SHA256 strings;
    repeated values count once. cache_dir defaults to the platform user cache.
    Returns a preview/apply context. Missing files are allowed. Symlinks,
    nonregular entries and a symlinked content directory raise ValueError.
    Callers confirm the preview and stop concurrent writers before apply.
    """
    if (
        isinstance(digests, str)
        or not digests
        or any(
            not isinstance(cast(object, value), str) or _DIGEST.fullmatch(value) is None
            for value in digests
        )
    ):
        raise ValueError("Cleanup requires complete lowercase SHA256 digests")
    root = (
        default_cache_dir()
        if cache_dir is None
        else Path(cache_dir).expanduser().resolve()
    )
    directory = root / "sha256"
    identity = None
    try:
        value = directory.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISDIR(value.st_mode):
            raise ValueError("Cache content directory must be a real directory")
        identity = value.st_dev, value.st_ino
    entries = {key: _selected_entry(directory / key) for key in dict.fromkeys(digests)}
    return _CacheCleanup(root, entries, identity)
