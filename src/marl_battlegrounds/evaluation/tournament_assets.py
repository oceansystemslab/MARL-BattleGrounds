"""Prepare immutable tournament assets and load only the active competitors.

Configuration and result code use AssetVerifier for bounded, offline integrity
checks. prepare_tournament_assets is the explicit download route; runners never
download. Controller factories run only inside the loading helpers. Importing
this module does not import JAX, providers, model code or the simulator.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from graphlib import CycleError, TopologicalSorter
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, cast
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from marl_battlegrounds.evaluation.tournament_config import (
    ASSET_ROLES,
    validate_inline_asset,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import Policy, System

_BUFFER_BYTES = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PAYLOAD_ROLES = frozenset({"model", "outcomes_priority", "full_report", "replay"})
_METADATA_ROLES = frozenset(
    {
        "registration",
        "configuration",
        "schedule",
        "run_manifest",
        "dependencies",
        "qualification",
    }
)
_CONTENT_FIELDS = (
    "code",
    "parameters",
    "memory_template",
    "input_preparation",
    "decision_settings",
    "adapter_bindings",
    "external_state",
)


def _json_bytes(value: object) -> bytes:
    """Encode JSON facts using the configuration authority's stable byte format."""
    from marl_battlegrounds.evaluation.tournament_config import canonical_json

    return canonical_json(value)


def default_cache_dir() -> Path:
    """Return the platform user cache root without creating it or reading assets.

    Uses LOCALAPPDATA on Windows, Library/Caches on macOS and XDG_CACHE_HOME
    (or ~/.cache) elsewhere. MARL-BGs owns only its marl-battlegrounds child.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "marl-battlegrounds").expanduser().resolve()


def _file_stamp(path: Path) -> tuple[int, int, int, int, int]:
    """Read regular-file identity and change counters; reject non-file assets."""
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Tournament asset is not a regular file: {path}")
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_file(path: Path) -> tuple[str, int]:
    """Hash a file with a one-MiB working buffer, rejecting concurrent changes."""
    before = _file_stamp(path)
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_BUFFER_BYTES):
            size += len(chunk)
            digest.update(chunk)
    if before != _file_stamp(path) or size != before[2]:
        raise ValueError(f"Tournament asset changed while it was being read: {path}")
    return digest.hexdigest(), size


def _sync_directory(path: Path) -> None:
    """Synchronize a directory after publishing a complete cached file."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_cache_directory(path: Path) -> None:
    """Create missing cache parents and durably publish each new directory entry."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync_directory(directory.parent)


def _asset_descriptor(
    value: object, identifier: str, *, version: int = 1
) -> dict[str, Any]:
    """Check a file or v2 inline declaration without opening it or importing code."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Asset {identifier!r} needs a descriptor")
    result = dict(cast(Mapping[str, Any], value))
    size = result.get("size_bytes")
    if (
        not isinstance(result.get("sha256"), str)
        or _DIGEST.fullmatch(result["sha256"]) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or result.get("role") not in ASSET_ROLES
        or set(result)
        - (
            {"sha256", "size_bytes", "role", "path", "url", "inline"}
            if version == 2
            else {"sha256", "size_bytes", "role", "path", "url"}
        )
    ):
        raise ValueError(f"Asset {identifier!r} has invalid identity, size or role")
    for name in ("path", "url"):
        location = result.get(name)
        if location is not None and (not isinstance(location, str) or not location):
            raise ValueError(f"Asset {identifier!r} has an invalid {name}")
    validate_inline_asset(result, version=version)
    return result


class AssetVerifier:
    """Check resolved local files once per unchanged file identity within a run.

    Parameters
    ----------
    config : Mapping
        Resolved tournament configuration with an assets mapping. Relative paths
        need its absolute source_location. Config-relative paths may not escape
        that directory, including through symlinks. Explicit absolute paths are
        caller-selected files. Version-2 inline metadata has no physical path;
        read_json verifies and copies it in memory.
    cache_dir : path-like or None
        Content-cache root. None uses default_cache_dir(). This constructor
        creates no files, imports no controllers and uses no network.

    Notes
    -----
    This host object belongs to one run. A stamp includes inode, size, mtime and
    ctime; changed files are hashed again before use. It is not a protection
    against another process with control over both the file and its filesystem.
    """

    def __init__(
        self, config: Mapping[str, Any], *, cache_dir: str | Path | None = None
    ) -> None:
        """Copy declarations and create empty host caches without reading payloads.

        config and cache_dir follow the class contract. Malformed declarations
        raise ValueError. This neither creates cache directories nor imports a
        controller; each requested file is checked later by verify or require.
        """
        self.config = copy.deepcopy(dict(config))
        assets = self.config.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError("Tournament config needs an assets mapping")
        self.assets = {
            str(key): _asset_descriptor(
                value, str(key), version=self.config.get("version", 1)
            )
            for key, value in cast(Mapping[str, Any], assets).items()
        }
        self.cache_dir = (
            default_cache_dir()
            if cache_dir is None
            else Path(cache_dir).expanduser().resolve()
        )
        source = config.get("source_location")
        self.source_location = None if source is None else Path(str(source)).resolve()
        self._verified: dict[Path, tuple[tuple[int, int, int, int, int], str, int]] = {}
        self._json_values: dict[
            Path, tuple[tuple[int, int, int, int, int], object]
        ] = {}
        self._scope_depth = 0
        self._scope_failed = False
        self._scope_paths: dict[str, tuple[Path, tuple[int, int, int, int, int]]] = {}

    @contextmanager
    def verification_scope(self) -> Generator[None]:
        """Share file checks during one bounded read, then check for changes.

        Repeated verify calls for a present asset use its first verified path
        until the outermost scope exits. That exit checks every used asset's
        path and file stamp before returning. Changed, removed or replaced
        files raise ValueError, even when their original bytes were restored.
        An exception clears the short-lived cache; a caught nested exception
        still makes the outer scope fail.

        Callers must finish the scope before yielding data, publishing output
        or using a result outside the checked read. Nested scopes share the
        outer boundary. This host helper is not thread-safe and performs no
        downloads or writes. It retains one path and stamp per used asset,
        rather than caching rows or weakening ordinary out-of-scope checks.
        """
        self._scope_depth += 1
        try:
            yield
            if self._scope_depth == 1:
                if self._scope_failed:
                    raise ValueError(
                        "Tournament asset verification scope was interrupted"
                    )
                used = self._scope_paths
                self._scope_paths = {}
                for identifier, (path, stamp) in used.items():
                    try:
                        unchanged = (
                            self._candidate(identifier) == path
                            and _file_stamp(path) == stamp
                        )
                    except OSError:
                        unchanged = False
                    if not unchanged:
                        raise ValueError(
                            f"Tournament asset {identifier!r} "
                            "changed during verification"
                        )
        except BaseException:
            self._scope_failed = True
            self._scope_paths.clear()
            raise
        finally:
            self._scope_depth -= 1
            if self._scope_depth == 0:
                self._scope_paths.clear()
                self._scope_failed = False

    def _candidate(self, identifier: str) -> Path | None:
        """Choose the declared local file, then the content cache; never download."""
        descriptor = self.assets[identifier]
        supplied = descriptor.get("path")
        if supplied is not None:
            path = Path(supplied).expanduser()
            if not path.is_absolute():
                if self.source_location is None:
                    raise ValueError(
                        f"Relative asset {identifier!r} needs source_location"
                    )
                path = (self.source_location / path).resolve()
                if not path.is_relative_to(self.source_location):
                    raise ValueError(
                        f"Asset {identifier!r} escapes its bundle directory"
                    )
            if path.exists():
                return path.resolve()
        cached = self.cache_dir / "sha256" / descriptor["sha256"]
        if cached.exists():
            resolved = cached.resolve()
            if not resolved.is_relative_to(self.cache_dir):
                raise ValueError("Tournament cache entry escapes the cache directory")
            return resolved
        return None

    def verify(self, asset_id: str) -> Path | None:
        """Return a verified absolute path, or None when the declared asset is absent.

        Inline metadata has no path and raises ValueError; use read_json for it.
        Unknown IDs and incorrect bytes raise ValueError. Missing data is never
        downloaded. Unchanged previously checked files reuse their digest within
        this verifier; mutable file changes invalidate that result. Inside a
        verification_scope, repeated calls share the first check and the scope
        checks for changes before the caller may expose its result.
        """
        if asset_id not in self.assets:
            raise ValueError(f"Tournament asset {asset_id!r} is not declared")
        if "inline" in self.assets[asset_id]:
            raise ValueError(
                f"Tournament asset {asset_id!r} is inline metadata; use read_json"
            )
        if self._scope_depth and asset_id in self._scope_paths:
            return self._scope_paths[asset_id][0]
        path = self._candidate(asset_id)
        if path is None:
            descriptor = self.assets[asset_id]
            if descriptor.get("path") is None and descriptor.get("url") is None:
                raise ValueError(
                    f"Requested asset {asset_id!r} has no local path, URL "
                    "or cache entry"
                )
            return None
        descriptor = self.assets[asset_id]
        stamp = _file_stamp(path)
        expected = descriptor["sha256"], descriptor["size_bytes"]
        previous = self._verified.get(path)
        if previous is not None and previous == (stamp, *expected):
            if self._scope_depth:
                self._scope_paths[asset_id] = (path, stamp)
            return path
        actual = _hash_file(path)
        if actual != expected:
            raise ValueError(
                f"Tournament asset {asset_id!r} has changed or is corrupt: {path}"
            )
        self._verified[path] = (stamp, *expected)
        if self._scope_depth:
            self._scope_paths[asset_id] = (path, stamp)
        return path

    def require(self, asset_ids: Sequence[str]) -> dict[str, Path]:
        """Return all requested verified paths or report every missing ID and size.

        This method checks the complete selection before returning. It does not
        load models or create files. Callers explicitly prepare missing assets
        or choose a complete fresh execution plan instead.
        """
        found: dict[str, Path] = {}
        missing: list[str] = []
        for identifier in dict.fromkeys(asset_ids):
            path = self.verify(identifier)
            if path is None:
                missing.append(
                    f"{identifier} ({self.assets[identifier]['size_bytes']} bytes)"
                )
            else:
                found[identifier] = path
        if missing:
            raise ValueError(
                "Missing tournament assets: "
                + ", ".join(missing)
                + ". Prepare/download them explicitly; use rerun_existing=True "
                "only for a complete fresh run."
            )
        return found

    def read_json(self, asset_id: str) -> object:
        """Read a verified JSON evidence asset, checking duplicate keys and NaN.

        Returns an owned copy of the stored JSON value. Version-2 inline
        metadata is checked against canonical JSON bytes without file access.
        For file assets, hash verification precedes parsing and a
        second file stamp rejects a concurrent change. Numerical payload files
        and game JSONL streams must use their own bounded readers instead.
        """
        if asset_id not in self.assets:
            raise ValueError(f"Tournament asset {asset_id!r} is not declared")
        descriptor = self.assets[asset_id]
        if "inline" in descriptor:
            validate_inline_asset(descriptor, version=self.config.get("version", 1))
            return copy.deepcopy(descriptor["inline"])
        path = self.require((asset_id,))[asset_id]
        before = _file_stamp(path)
        if self._verified[path][0] != before:
            raise ValueError("Tournament JSON asset changed after verification")
        previous = self._json_values.get(path)
        if previous is not None and previous[0] == before:
            return copy.deepcopy(previous[1])

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            """Reject repeated JSON keys instead of accepting the final value."""
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(
                        f"Duplicate JSON key {key!r} in asset {asset_id!r}"
                    )
                result[key] = value
            return result

        def invalid(value: str) -> NoReturn:
            """Reject nonfinite JSON tokens in immutable evidence."""
            raise ValueError(f"Nonfinite JSON value {value!r} in asset {asset_id!r}")

        def finite(value: str) -> float:
            """Reject JSON exponents that overflow a finite Python float."""
            parsed = float(value)
            if not math.isfinite(parsed):
                invalid(value)
            return parsed

        with path.open(encoding="utf-8") as stream:
            result = json.load(
                stream,
                object_pairs_hook=pairs,
                parse_constant=invalid,
                parse_float=finite,
            )
        if before != _file_stamp(path):
            raise ValueError(f"Tournament asset changed while parsing: {asset_id}")
        self._json_values[path] = before, result
        return copy.deepcopy(result)


class _HTTPOnlyRedirect(HTTPRedirectHandler):
    """Keep explicit asset downloads on HTTP(S), including redirects."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        """Follow an HTTP(S) redirect, rejecting file and other local schemes."""
        if urlparse(newurl).scheme not in {"http", "https"}:
            raise ValueError("Tournament downloads support only HTTP and HTTPS URLs")
        return super().redirect_request(
            req, cast(Any, fp), code, msg, cast(Any, headers), newurl
        )


@dataclass
class _DownloadApproval:
    """Bind one confirmed download to its missing content, sizes and URLs.

    allowed holds the inspected digest/size/URL tuples. attempted stops a second
    transfer of a digest even if it disappears after its first transfer. This is
    private CLI consent state; ordinary explicit Python downloads do not need it.
    """

    allowed: frozenset[tuple[str, int, str | None]]
    attempted: set[str] = dataclass_field(default_factory=set[str])

    def check(self, descriptor: Mapping[str, Any], *, claim: bool) -> None:
        """Check permission before any temporary file or network operation.

        claim=True reserves the only allowed attempt for this digest. A later
        claim=False check confirms unchanged evidence immediately before opening
        its URL. Changed or previously unconfirmed content raises ValueError.
        """
        evidence = (
            descriptor["sha256"],
            descriptor["size_bytes"],
            descriptor.get("url"),
        )
        if evidence not in self.allowed:
            raise ValueError(
                "Required download changed after preview; run preparation again"
            )
        if claim:
            if evidence[0] in self.attempted:
                raise ValueError(
                    "A confirmed asset already had a download attempt; "
                    "run preparation again"
                )
            self.attempted.add(evidence[0])


def _download_asset(
    verifier: AssetVerifier,
    asset_id: str,
    *,
    approval: _DownloadApproval | None = None,
) -> int:
    """Stream one explicitly requested asset into a durable content-cache file.

    Returns downloaded bytes. Hash/length errors leave no complete cache entry.
    Uses bounded reads and only HTTP(S); no archives, code imports or retries.
    approval, when supplied by the CLI, limits each transfer to its inspected
    digest, size and URL before creating files and again before the request.
    """
    descriptor = verifier.assets[asset_id]
    url = descriptor.get("url")
    if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"Asset {asset_id!r} needs an explicit HTTP(S) download URL")
    if approval is not None:
        approval.check(descriptor, claim=True)
    directory = verifier.cache_dir / "sha256"
    _make_cache_directory(directory)
    if not directory.resolve().is_relative_to(verifier.cache_dir):
        raise ValueError("Tournament cache directory escapes its declared root")
    destination = directory / descriptor["sha256"]
    descriptor_fd, temporary_name = tempfile.mkstemp(prefix=".download-", dir=directory)
    temporary = Path(temporary_name)
    try:
        digest = sha256()
        size = 0
        with os.fdopen(descriptor_fd, "wb") as output:
            if approval is not None:
                approval.check(verifier.assets[asset_id], claim=False)
            with build_opener(_HTTPOnlyRedirect()).open(url, timeout=30) as response:
                while chunk := response.read(_BUFFER_BYTES):
                    size += len(chunk)
                    if size > descriptor["size_bytes"]:
                        raise ValueError(
                            f"Downloaded asset {asset_id!r} exceeds its declared size"
                        )
                    digest.update(chunk)
                    output.write(chunk)
            if (
                size != descriptor["size_bytes"]
                or digest.hexdigest() != descriptor["sha256"]
            ):
                raise ValueError(
                    f"Downloaded asset {asset_id!r} has the wrong size or hash"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        verifier._verified[destination] = (  # pyright: ignore[reportPrivateUsage]
            _file_stamp(destination),
            descriptor["sha256"],
            size,
        )
        _sync_directory(directory)
        _sync_directory(verifier.cache_dir)
        _sync_directory(verifier.cache_dir.parent)
        return size
    finally:
        temporary.unlink(missing_ok=True)


def _references(value: object) -> Iterator[str]:
    """Yield explicit asset references from metadata, ignoring arbitrary strings."""
    if isinstance(value, Mapping):
        for key, item in cast(Mapping[str, object], value).items():
            if key == "assets":
                continue
            if (str(key).endswith("_asset") or key == "asset_id") and isinstance(
                item, str
            ):
                yield item
            elif key in {"asset_ids"} and isinstance(item, list):
                yield from (
                    entry
                    for entry in cast(list[object], item)
                    if isinstance(entry, str)
                )
            else:
                yield from _references(item)
    elif isinstance(value, (list, tuple)):
        for item in cast(Sequence[object], value):
            yield from _references(item)


def _is_game_list(path: Path) -> bool:
    """Identify schedule JSONL from one bounded first-row read, without loading it.

    This only prevents expanding game lists during asset preparation. The shared
    schedule reader later validates every game. A pretty-printed manifest falls
    through to normal strict JSON parsing.
    """
    with path.open("rb") as stream:
        prefix = stream.readline(_BUFFER_BYTES)
    try:
        value: object = json.loads(prefix)
    except ValueError, UnicodeDecodeError:
        return False
    return isinstance(value, dict) and "logical_game_id" in value


class _AssetPreparation:
    """Keep one verifier and dependency graph through inspection and completion.

    Created only by _inspect_tournament_assets. preview is the ordinary helper
    result after read-only inspection. expected_config adds expected cache paths
    so output-file conflicts can fail before downloads. finish rechecks file
    stamps, reuses unchanged hashes/metadata and optionally downloads. This host
    object is private to one invocation and is not thread-safe.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        cache_dir: str | Path | None,
        roles: Sequence[str],
    ) -> None:
        """Inspect copied resolved declarations using the requested payload roles.

        cache_dir has AssetVerifier's meaning. roles also select all required
        metadata. Invalid assets/dependencies raise before any write; missing
        files remain in preview. No controller or numerical backend is loaded.
        """
        self.verifier = AssetVerifier(config, cache_dir=cache_dir)
        self._declarations = copy.deepcopy(self.verifier.assets)
        self._config = copy.deepcopy(dict(config))
        self._metadata = {
            key
            for key, value in self.verifier.assets.items()
            if value["role"] in _METADATA_ROLES
        }
        self._selected = tuple(
            sorted(
                self._metadata
                | {
                    key
                    for key, value in self.verifier.assets.items()
                    if value["role"] in roles
                }
            )
        )
        sizes: dict[str, int] = {}
        for key in self._selected:
            value = self.verifier.assets[key]
            if (
                sizes.setdefault(value["sha256"], value["size_bytes"])
                != value["size_bytes"]
            ):
                raise ValueError("Asset aliases declare different sizes for one digest")
        self._checked_metadata: dict[
            str, tuple[Path, tuple[int, int, int, int, int]]
        ] = {}
        self._graph: dict[str, set[str]] = {}
        self._graph_changed = False
        self._game_lists: set[str] = set()
        self._downloaded = 0
        self.preview = self._scan(download=False, approval=None)
        self._approval = _DownloadApproval(
            frozenset(
                (
                    self._declarations[key]["sha256"],
                    self._declarations[key]["size_bytes"],
                    self._declarations[key].get("url"),
                )
                for key in self.preview["missing"]
            )
        )

    @property
    def expected_config(self) -> dict[str, Any]:
        """Return path hints expected after completing the inspected preparation.

        This copy changes locations only. It does not assert missing files exist.
        It lets prepared-file preflight compare full intended bytes before I/O.
        """
        result = copy.deepcopy(self.preview["config"])
        for key in self.preview["missing"]:
            result["assets"][key]["path"] = str(
                self.verifier.cache_dir / "sha256" / self._declarations[key]["sha256"]
            )
        return result

    def _check_metadata(self, key: str, path: Path | None) -> None:
        """Check new metadata once; unchanged verified bytes reuse its references.

        Schedule game lists are bounded first-row checks, never expanded JSONL.
        Source-run manifests retain their historical dependency interpretation.
        """
        if path is not None:
            stamp = _file_stamp(path)
            if self._checked_metadata.get(key) == (path, stamp):
                return
            self._checked_metadata[key] = path, stamp
        if self.verifier.assets[key]["role"] == "run_manifest":
            return
        if key in self._game_lists or (
            path is not None
            and self.verifier.assets[key]["role"] == "schedule"
            and _is_game_list(path)
        ):
            return
        value = self.verifier.read_json(key)
        if self.verifier.assets[key]["role"] == "schedule" and isinstance(value, list):
            return
        if isinstance(value, dict):
            self._game_lists.update(
                str(cast(dict[str, object], value)[name])
                for name in ("games_asset", "challenger_games_asset")
                if value.get(name) is not None
            )
        refs = set(_references(cast(object, value)))
        undeclared = refs - self.verifier.assets.keys()
        if undeclared:
            raise ValueError(
                f"Asset {key!r} references undeclared assets: {sorted(undeclared)}"
            )
        self._graph[key] = refs & self._metadata
        self._graph_changed = True

    def _scan(
        self,
        *,
        download: bool,
        approval: _DownloadApproval | None,
    ) -> dict[str, Any]:
        """Verify selected files and return the public helper's plain mapping.

        Repeated scans inspect file stamps while retaining digests and parsed
        dependency evidence. Missing physical bytes count once per digest.
        Downloads use the supplied confirmation bound at their actual I/O site.
        """
        if self.verifier.assets != self._declarations:
            raise ValueError("Asset declarations changed after preview; prepare again")
        resolved = copy.deepcopy(self._config)
        verified: list[str] = []
        missing: list[str] = []
        missing_sizes: dict[str, int] = {}
        with self.verifier.verification_scope():
            for key in self._selected:
                if "inline" in self.verifier.assets[key]:
                    self._check_metadata(key, None)
                    verified.append(key)
                    continue
                path = self.verifier.verify(key)
                if path is None and download:
                    self._downloaded += _download_asset(
                        self.verifier, key, approval=approval
                    )
                    path = self.verifier.verify(key)
                if path is None:
                    missing.append(key)
                    descriptor = self.verifier.assets[key]
                    missing_sizes[descriptor["sha256"]] = descriptor["size_bytes"]
                    continue
                verified.append(key)
                resolved["assets"][key]["path"] = str(path)
                if key in self._metadata:
                    self._check_metadata(key, path)
        if self._graph_changed:
            try:
                TopologicalSorter(self._graph).prepare()
            except CycleError as error:
                raise ValueError(
                    "Tournament metadata contains a dependency cycle"
                ) from error
            self._graph_changed = False
        return {
            "config": resolved,
            "verified": verified,
            "missing": missing,
            "bytes_missing": sum(missing_sizes.values()),
            "bytes_downloaded": self._downloaded,
            "cache_dir": str(self.verifier.cache_dir),
        }

    def finish(
        self,
        *,
        download: bool = False,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Recheck inspected assets and optionally prepare their missing files.

        download=False performs no writes or network I/O. confirmed=True binds
        downloads to the initial preview and allows one attempt per digest. The
        public explicit-download helper leaves confirmed False. A confirmed
        completion also rejects remaining missing files, even with no download.
        Returns the same
        mapping as preview with current paths and cumulative downloaded bytes.
        Changed/corrupt content and invalid dependencies raise ValueError;
        filesystem and network errors propagate. Successful earlier copies stay.
        """
        if not isinstance(cast(object, download), bool) or not isinstance(
            cast(object, confirmed), bool
        ):
            raise ValueError("download and confirmed must be Booleans")
        result = self._scan(
            download=download,
            approval=self._approval if confirmed else None,
        )
        if confirmed and result["missing"]:
            raise ValueError(
                "Required assets are missing after preview; run preparation again"
            )
        return result


def _inspect_tournament_assets(
    config: str | Path | Mapping[str, Any] | None = None,
    *,
    cache_dir: str | Path | None = None,
    roles: Sequence[str] = ("model", "outcomes_priority", "full_report"),
) -> _AssetPreparation:
    """Resolve and inspect assets for one optional confirmation/completion cycle.

    Arguments follow prepare_tournament_assets. Returns its private preparation
    context; no file/network mutation occurs. Configuration and integrity errors
    propagate. Reuse the returned object instead of resolving or hashing twice.
    """
    from marl_battlegrounds.evaluation.tournament_config import load_tournament_config

    if isinstance(roles, str) or any(role not in ASSET_ROLES for role in roles):
        raise ValueError("Unknown tournament asset role")
    return _AssetPreparation(
        load_tournament_config(config),
        cache_dir=cache_dir,
        roles=roles,
    )


def prepare_tournament_assets(
    config: str | Path | Mapping[str, Any] | None = None,
    *,
    cache_dir: str | Path | None = None,
    roles: Sequence[str] = ("model", "outcomes_priority", "full_report"),
    download: bool = False,
) -> dict[str, Any]:
    """Inspect or explicitly download tournament files without running a method.

    Parameters
    ----------
    config : path, mapping or None
        Official or custom descriptor. None selects the installed official
        snapshot; a missing release fails clearly. The input is never changed.
    cache_dir : path or None
        Chosen content cache, default the platform user cache. Returned config
        paths let subsequent offline execution use a nondefault directory.
    roles : sequence of str
        Requested payload roles, default model, outcomes_priority and full_report.
        Required metadata dependencies are included. Replays are opt-in; report-
        only requests need no model payload or controller import.
    download : bool
        False only inspects. True explicitly allows HTTP(S) downloads of missing
        requested files. The caller/CLI owns confirmation and size display.

    Returns
    -------
    dict
        config (copied with verified absolute paths), verified and missing asset
        ID lists, bytes_missing, bytes_downloaded and absolute cache_dir. Missing
        bytes count each physical content digest once, even when several asset
        IDs name it. Version-2 inline metadata is verified in memory; it does not
        create cache files or acquire artificial file paths. Missing files remain
        explicit; corrupt files raise instead
        of becoming missing.

    Raises
    ------
    ValueError
        Configuration, roles, identity, dependencies, URL or file bytes are bad.
    OSError
        Reading, downloading, synchronizing or installing a file fails.

    Notes
    -----
    Host-only. Never loads Python model code, starts JAX or creates a run folder.
    Downloads use bounded buffers; metadata JSON is read at host setup. A data
    digest verifies bytes, not the safety or eligibility of a controller.
    """
    if not isinstance(cast(object, download), bool):
        raise ValueError("download must be a Boolean")
    preparation = _inspect_tournament_assets(config, cache_dir=cache_dir, roles=roles)
    return preparation.finish(download=True) if download else preparation.preview


def _known(value: object) -> bool:
    """Return whether all nested evidence is stated rather than null or unknown."""
    if value is None or value == "unknown":
        return False
    if isinstance(value, Mapping):
        return all(_known(item) for item in cast(Mapping[str, object], value).values())
    if isinstance(value, (list, tuple)):
        return all(_known(item) for item in cast(Sequence[object], value))
    return True


def controller_content_identity(
    controller: Mapping[str, Any], verifier: AssetVerifier
) -> tuple[str, bool]:
    """Hash verified controller facts without display names or asset aliases.

    Parameters
    ----------
    controller : Mapping
        Format-1 controller descriptor with its seven content fields. Evidence
        references identify immutable JSON assets by asset_id and sha256.
    verifier : AssetVerifier
        Same offline verifier used for the run's source/configuration assets.

    Returns
    -------
    tuple[str, bool]
        Versioned content SHA256 and whether all supplied evidence is known.
        Equal incomplete digests do not prove duplicate controllers. Qualification
        must still establish claims about external state and code dependencies.

    Raises
    ------
    ValueError
        Content references are missing, corrupt or contradict their declared hash.
    """
    raw_content = controller.get("content")
    if not isinstance(raw_content, Mapping):
        raise ValueError("Controller content must be an object")
    content = cast(Mapping[str, Any], raw_content)
    if set(content) != set(_CONTENT_FIELDS):
        raise ValueError("Controller content must name every format-1 evidence field")
    normalized: dict[str, Any] = {}
    for field in _CONTENT_FIELDS:
        item = content[field]
        if field == "adapter_bindings" or item is None:
            normalized[field] = copy.deepcopy(item)
            continue
        if not isinstance(item, Mapping) or set(cast(Mapping[str, Any], item)) != {
            "asset_id",
            "sha256",
        }:
            raise ValueError(f"Controller {field} requires an evidence reference")
        identifier = str(cast(Mapping[str, Any], item)["asset_id"])
        if (
            identifier not in verifier.assets
            or verifier.assets[identifier]["sha256"] != item["sha256"]
        ):
            raise ValueError(
                f"Controller {field} evidence hash does not match its asset"
            )
        normalized[field] = verifier.read_json(identifier)
    digest = sha256(
        b"marlbg-controller-content-v1\0" + _json_bytes(normalized)
    ).hexdigest()
    return digest, _known(normalized)


def loaded_controller_registration(method: object) -> dict[str, Any]:
    """Describe one already loaded frozen method without making an action call.

    Imports numerical identity code only here. Existing registration meanings
    remain unchanged, including honest unknown closures/provider state.
    """
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )

    return dict(
        normalize_system_registration(method, phase="tournament", frozen=True)[1]
    )


def _assert_known_facts(expected: object, actual: object, label: str) -> None:
    """Compare declared known registration facts, never filling unknown evidence."""
    if expected is None or expected == "unknown":
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ValueError(f"Loaded controller differs from {label}")
        for key, value in cast(Mapping[str, object], expected).items():
            _assert_known_facts(
                value, cast(Mapping[str, object], actual).get(key), f"{label}.{key}"
            )
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(cast(list[object], actual)) != len(
            cast(list[object], expected)
        ):
            raise ValueError(f"Loaded controller differs from {label}")
        for index, (left, right) in enumerate(
            zip(cast(list[object], expected), cast(list[object], actual), strict=True)
        ):
            _assert_known_facts(left, right, f"{label}[{index}]")
    elif expected != actual:
        raise ValueError(f"Loaded controller differs from {label}")


def policy_recording_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Project a frozen live Policy description to its existing writer format.

    Policy evaluation keeps its historical description and digest. That format
    records callable/controller identity and parameter/template digests, but has
    no hook or parameter-evidence fields. This host-only projection changes no
    stored record and does not invent those absent facts. System descriptions are
    returned unchanged. Loaded methods still require the full asset check before
    execution; release qualification owns evidence beyond old recorded fields.
    """
    if registration.get("kind") != "policy":
        return dict(registration)
    fields = (
        "name",
        "checkpoint",
        "execution",
        "variables_frozen",
        "callable_name",
        "controller_identity",
        "variables_digest",
        "initial_carry_digest",
        "kind",
        "components",
        "parameter_status",
    )
    return {field: registration[field] for field in fields if field in registration}


def duplicate_controller_id(
    method: object,
    participants: Sequence[Mapping[str, Any]],
    registrations: Mapping[str, Mapping[str, Any]],
    verifier: AssetVerifier,
) -> str | None:
    """Find a provable incumbent duplicate without loading any incumbent model.

    ``method`` is an already frozen challenger. Participant descriptors and
    registrations come from verified immutable assets. Registered built-ins have
    an exact installed callable identity; compare that identity, numerical
    variables, memory templates, execution mode and ordered adapter bindings.
    Display/checkpoint labels and locations do not affect this comparison.

    Generic Python/provider descriptions omit external behavior. Equal partial
    descriptions therefore return None, not an invented duplicate. Fully declared
    controller descriptors are compared by ``controller_content_identity`` by the
    resolver. No factory, initializer or action callback runs here.
    """
    from marl_battlegrounds.evaluation.policy_execution import Policy, System, policy

    def known_policy(value: Mapping[str, Any]) -> dict[str, Any] | None:
        """Keep only complete built-in behavior facts and numerical ownership."""
        controller = value.get("controller_identity")
        # Random has no historical scientific controller descriptor. Its exact
        # installed callable is checked separately for the live challenger.
        if controller is None:
            return None
        if any(
            value.get(key) is None
            for key in ("variables_digest", "initial_carry_digest")
        ):
            return None
        return {
            key: value.get(key)
            for key in (
                "controller_identity",
                "variables_digest",
                "initial_carry_digest",
                "execution",
            )
        }

    actual = loaded_controller_registration(method)
    if isinstance(method, Policy):
        for participant in participants:
            descriptor = participant["controller"]
            expected = registrations[participant["entrant_id"]]
            if (
                descriptor["kind"] == "builtin"
                and method.apply is policy(descriptor["name"]).apply
                and expected.get("kind") == "policy"
                and all(
                    actual.get(field) is not None
                    and actual.get(field) == expected.get(field)
                    for field in (
                        "execution",
                        "variables_digest",
                        "initial_carry_digest",
                    )
                )
            ):
                controller_content_identity(descriptor, verifier)
                return str(participant["controller_id"])
        known = known_policy(actual)
        if known is None:
            return None
        for participant in participants:
            expected = registrations[participant["entrant_id"]]
            if expected.get("kind") == "policy" and known_policy(expected) == known:
                # Verifying content again is a cached check, not a model load.
                controller_content_identity(participant["controller"], verifier)
                return str(participant["controller_id"])
        return None
    if isinstance(method, System) and method._policies:  # pyright: ignore[reportPrivateUsage]
        components = actual.get("adapter_policies", [])
        known_components = [known_policy(value) for value in components]
        if not known_components or any(value is None for value in known_components):
            return None
        for participant in participants:
            expected = registrations[participant["entrant_id"]]
            if (
                expected.get("kind") == "system"
                and expected.get("adapter_kind") == actual.get("adapter_kind")
                and expected.get("execution") == actual.get("execution")
                and expected.get("variables_digest") == actual.get("variables_digest")
                and [
                    known_policy(value)
                    for value in expected.get("adapter_policies", [])
                ]
                == known_components
            ):
                controller_content_identity(participant["controller"], verifier)
                return str(participant["controller_id"])
    return None


def validate_loaded_controller(
    method: object, participant: Mapping[str, Any], verifier: AssetVerifier
) -> None:
    """Check loaded registration and numerical evidence before a method can act.

    participant supplies name, controller, controller_id and registration_asset.
    Known expected registration facts must match. Parameter/template evidence
    may use existing registration fields, a digest string or {digest: value}.
    Adapter memory digests use the existing ordered memory-template authority.
    Generic Systems have no declared numerical template; their init hook is part
    of registration. No initializer, reset hook or action method runs. Unknown
    facts stay unknown; admission owns stronger scientific qualification.
    """
    actual = loaded_controller_registration(method)
    if actual.get("name") != participant.get("name"):
        raise ValueError("Loaded controller name differs from its participant label")
    expected = verifier.read_json(str(participant["registration_asset"]))
    _assert_known_facts(expected, actual, "registration")
    controller = cast(Mapping[str, Any], participant["controller"])
    identity, _ = controller_content_identity(controller, verifier)
    if identity != participant["controller_id"]:
        raise ValueError("Controller evidence differs from its content identity")
    content = controller["content"]
    for field, key in (
        ("parameters", "variables_digest"),
        ("memory_template", "initial_carry_digest"),
    ):
        reference = content[field]
        if reference is None:
            continue
        evidence = verifier.read_json(reference["asset_id"])
        if isinstance(evidence, str):
            expected_digest = evidence
        elif isinstance(evidence, Mapping) and "digest" in evidence:
            expected_digest = cast(Mapping[str, object], evidence)["digest"]
        else:
            _assert_known_facts(cast(object, evidence), actual, field)
            continue
        actual_digest = actual.get(key)
        if field == "memory_template" and actual.get("kind") == "system":
            from marl_battlegrounds.evaluation.policy_execution import (
                _adapter_template,  # pyright: ignore[reportPrivateUsage]
            )
            from marl_battlegrounds.evaluation.recording_identity import tree_digest

            actual_digest = tree_digest(_adapter_template(cast("System", method)))
        if expected_digest != actual_digest:
            raise ValueError(f"Loaded controller differs from declared {field}")


def _installed_callable(reference: str) -> Callable[..., object]:
    """Use the shared trusted module:callable importer without invoking it."""
    from marl_battlegrounds._method_loading import installed_callable

    return installed_callable(reference)


def load_tournament_controller(
    participant: Mapping[str, Any],
    verifier: AssetVerifier,
) -> System | Policy:
    """Load and freeze one declared competitor, then verify its actual evidence.

    Builtins use the existing registry. Factories take no arguments. Version-2
    references use the shared built-in/factory/actor-folder loader. Bundle
    loaders receive a read-only asset-ID-to-path mapping. Factories are trusted
    installed code and may have their own effects; exceptions propagate without
    retries. Construction stays on the calling thread. Researchers own provider
    processes and memory. This function never chooses an action or downloads a
    missing model.
    """
    from marl_battlegrounds.evaluation.system_evaluation import freeze_evaluation_method

    controller = participant["controller"]
    kind = controller["kind"]
    if kind == "builtin":
        method = freeze_evaluation_method(controller["name"])
    elif kind == "factory":
        method = freeze_evaluation_method(
            cast("System | Policy", _installed_callable(controller["factory"])())
        )
    elif kind == "reference":
        if verifier.config.get("version", 1) != 2:
            raise ValueError("Reference controllers need descriptor version 2")
        from marl_battlegrounds._method_loading import load_method

        method = freeze_evaluation_method(load_method(controller["reference"]))
    elif kind == "bundle":
        paths = verifier.require(controller["asset_ids"])
        method = freeze_evaluation_method(
            cast(
                "System | Policy",
                _installed_callable(controller["loader"])(MappingProxyType(paths)),
            )
        )
    else:
        raise ValueError(f"Unknown tournament controller kind: {kind!r}")
    validate_loaded_controller(method, participant, verifier)
    return method


@contextmanager
def active_tournament_pair(
    first: Mapping[str, Any], second: Mapping[str, Any], verifier: AssetVerifier
) -> Generator[tuple[System | Policy, System | Policy]]:
    """Own two loaded competitors for one matchup without an all-model cache.

    The caller runs all required games inside this context and must not retain
    returned methods afterward. It must finish/synchronize its result before
    leaving the context: this helper cannot discover caller-owned output trees.
    On exit, pending method-array creation and JAX callback effects finish before
    local references are released. Compilation caches stay; JAX may retain its
    device allocation pool. External sessions are not closed or rolled back.
    """
    first_method: System | Policy | None = None
    second_method: System | Policy | None = None
    try:
        first_method = load_tournament_controller(first, verifier)
        second_method = load_tournament_controller(second, verifier)
        yield first_method, second_method
    finally:
        if first_method is not None or second_method is not None:
            import jax

            for method in (first_method, second_method):
                if method is not None:
                    for leaf in jax.tree.leaves(method.variables):
                        if isinstance(leaf, jax.Array):
                            leaf.block_until_ready()
            jax.effects_barrier()
        first_method = second_method = None


def asset_location_config(
    config: Mapping[str, Any], locations: Mapping[str, Any]
) -> dict[str, Any]:
    """Overlay verified file hints without changing the immutable descriptor.

    locations maps known asset IDs to path/url hints only. Paths must be absolute
    strings or None; URLs are strings or None and are never fetched here. Reject
    extra keys and unknown assets. Return a shallow descriptor copy with copied
    asset entries. Digests, sizes and all scientific fields retain their values.
    The reader still verifies content before use; hints cannot certify an asset.
    """
    assets = dict(config["assets"])
    for identifier, hints in locations.items():
        if not isinstance(hints, Mapping):
            raise ValueError("tournament asset locations must be mappings")
        hints = cast(Mapping[str, Any], hints)
        if (
            identifier not in assets
            or set(hints) != {"path", "url"}
            or any(
                value is not None and not isinstance(value, str)
                for value in hints.values()
            )
            or (hints["path"] is not None and not Path(hints["path"]).is_absolute())
        ):
            raise ValueError(
                "tournament asset locations must be known absolute path/url hints"
            )
        assets[identifier] = {**assets[identifier], **hints}
    return {**config, "assets": assets}


def environment_source_manifest() -> dict[str, Any]:
    """Describe the installed Core and task source bytes without importing JAX.

    Paths are relative to the installed package and sorted. This format-1
    descriptor covers all Core Python files and tasks.py. Asset/configuration
    contents and controller code have separate identities. The result is host
    setup evidence, never a per-game or per-step operation.
    """
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.joinpath("core").glob("*.py"), root / "tasks.py"])
    return {
        "format": "marlbg-environment-sources",
        "version": 1,
        "files": [
            {"path": path.relative_to(root).as_posix(), "sha256": _hash_file(path)[0]}
            for path in paths
        ],
    }


def inference_dependency_lock(packages: Sequence[str] = ()) -> dict[str, Any]:
    """Record exact installed inference versions without importing their runtimes.

    packages optionally adds trusted provider/loader distributions to the base
    numerical and report dependencies. These are package names, not imports.
    Missing installed distributions raise PackageNotFoundError. This small lock
    describes inference only; training/optimizer checkpoints remain separate.
    """
    from importlib.metadata import version

    names = sorted({"jax", "jaxlib", "numpy", "pydantic", "scipy", *packages})
    return {
        "format": "marlbg-inference-dependencies",
        "version": 1,
        "python": list(sys.version_info[:2]),
        "packages": {name: version(name) for name in names},
    }


def verify_tournament_environment(
    config: Mapping[str, Any], verifier: AssetVerifier, *, execution: bool
) -> None:
    """Verify source/dependency evidence before output creation or model loading.

    config is the resolved tournament descriptor. The shared verifier checks its
    source manifest and inference lock assets. Both need supported format-1
    structures, exact ordered paths, hashes and versions. When execution is True,
    the current installed Core/task bytes, Python and declared distributions must
    match. False validates recorded evidence without requiring a simulation
    runtime: complete reuse reads games produced by that recorded environment.

    Missing/changed assets or incompatible evidence raise ValueError. This never
    imports JAX, queries devices, executes a factory or alters a run. Unknown
    source/dependency claims cannot be promoted into compatible execution.
    """
    from marl_battlegrounds.evaluation.tournament_config import canonical_json

    compatibility = config["compatibility"]
    raw_sources = verifier.read_json(compatibility["source_manifest_asset"])
    if not isinstance(raw_sources, dict):
        raise ValueError("Tournament needs a complete environment source manifest")
    sources = cast(dict[str, Any], raw_sources)
    if set(sources) != {"format", "version", "files"}:
        raise ValueError("Tournament needs a complete environment source manifest")
    if (
        sources["format"] != "marlbg-environment-sources"
        or type(sources["version"]) is not int
        or sources["version"] != 1
    ):
        raise ValueError("Unsupported tournament environment source format")
    files = sources["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Environment source manifest needs ordered files")
    paths: list[str] = []
    for raw_entry in cast(list[Any], files):
        if not isinstance(raw_entry, dict):
            raise ValueError("Environment source entries require path and sha256")
        entry = cast(dict[str, Any], raw_entry)
        if set(entry) != {"path", "sha256"}:
            raise ValueError("Environment source entries require path and sha256")
        path, digest = entry["path"], entry["sha256"]
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not _DIGEST.fullmatch(digest)
        ):
            raise ValueError("Environment source path or digest is invalid")
        if path != "tasks.py" and not (
            Path(path).parts[:1] == ("core",)
            and len(Path(path).parts) == 2
            and Path(path).suffix == ".py"
        ):
            raise ValueError("Environment source path leaves Core/task ownership")
        paths.append(path)
    if paths != sorted(set(paths)) or set(paths) != {
        "core/__init__.py",
        "core/axis_mappings.py",
        "core/combat.py",
        "core/config.py",
        "core/env.py",
        "core/geometry.py",
        "core/types.py",
        "tasks.py",
    }:
        raise ValueError(
            "Environment source paths must be complete, unique and ordered"
        )
    if sha256(canonical_json(sources)).hexdigest() != compatibility["environment_id"]:
        raise ValueError("Environment source manifest differs from environment_id")
    raw_lock = verifier.read_json(compatibility["dependency_lock_asset"])
    if not isinstance(raw_lock, dict):
        raise ValueError("Unsupported tournament inference dependency lock")
    lock = cast(dict[str, Any], raw_lock)
    if (
        set(lock) != {"format", "version", "python", "packages"}
        or lock["format"] != "marlbg-inference-dependencies"
        or type(lock["version"]) is not int
        or lock["version"] != 1
    ):
        raise ValueError("Unsupported tournament inference dependency lock")
    packages = lock["packages"]
    python = lock["python"]
    if (
        not isinstance(python, list)
        or len(cast(list[Any], python)) != 2
        or any(type(value) is not int or value < 0 for value in cast(list[Any], python))
    ):
        raise ValueError("Inference lock needs exact Python major/minor versions")
    if (
        not isinstance(packages, dict)
        or not {"jax", "jaxlib", "numpy", "pydantic", "scipy"} <= packages.keys()
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in cast(dict[Any, Any], packages).items()
        )
    ):
        raise ValueError("Inference lock lacks required distribution versions")
    if execution:
        if sources != environment_source_manifest():
            raise ValueError("Installed Core/task source differs from the snapshot")
        if lock != inference_dependency_lock(tuple(cast(dict[str, str], packages))):
            raise ValueError(
                "Installed inference dependencies differ from the snapshot"
            )
