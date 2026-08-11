"""Launch-scoped, path-free source revision discovery for debugger capture."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from marl_battlegrounds.evaluation.catalog import build_code_revision_v1
from marl_battlegrounds.evaluation.models import CodeRevisionV1

_PACKAGE_DISTRIBUTION = "marl-battlegrounds"


def _git(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", os.fspath(repository_root), *arguments),
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git revision discovery failed: {detail or 'unknown error'}")
    return completed.stdout


def _framed_update(digest: hashlib._Hash, label: bytes, payload: bytes) -> None:  # type: ignore[name-defined]
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _untracked_content_digest(repository_root: Path) -> bytes:
    paths = tuple(
        path
        for path in _git(
            repository_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ).split(b"\0")
        if path
    )
    digest = hashlib.sha256()
    for encoded_relative_path in sorted(paths):
        relative_path = os.fsdecode(encoded_relative_path)
        candidate = repository_root / relative_path
        metadata = candidate.lstat()
        _framed_update(digest, b"path", encoded_relative_path)
        _framed_update(digest, b"mode", str(stat.S_IFMT(metadata.st_mode)).encode())
        if stat.S_ISLNK(metadata.st_mode):
            _framed_update(
                digest,
                b"symlink-target",
                os.fsencode(os.readlink(candidate)),
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                "untracked source provenance supports only regular files and symlinks"
            )
        file_digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_digest.update(chunk)
        _framed_update(digest, b"file-sha256", file_digest.digest())
    return digest.digest()


def discover_debugger_code_revision_v1(
    repository_root: Path,
    *,
    package_version: str | None = None,
) -> CodeRevisionV1:
    """Resolve one truthful Git/source identity without storing a local path.

    The clean source-tree digest covers Git's ordered tree listing.  When the
    worktree differs from ``HEAD``, the dirty digest additionally covers the
    porcelain status, the binary tracked diff, and content hashes for every
    non-ignored untracked file.  The absolute repository path is used only as
    an input capability and never enters the returned scientific identity.
    """
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        repository_root, Path
    ):
        raise TypeError("repository_root must be a pathlib.Path")
    root = repository_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository_root must identify a local directory")

    commit_sha = _git(root, "rev-parse", "--verify", "HEAD").decode().strip()
    tree_listing = _git(root, "ls-tree", "-r", "--full-tree", "HEAD")
    source_tree_digest = hashlib.sha256(tree_listing).hexdigest()
    status_payload = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    is_dirty = bool(status_payload)
    dirty_patch_digest: str | None = None
    if is_dirty:
        dirty = hashlib.sha256()
        _framed_update(dirty, b"status-v1-z", status_payload)
        _framed_update(
            dirty,
            b"tracked-binary-diff-head",
            _git(root, "diff", "--binary", "HEAD", "--"),
        )
        _framed_update(
            dirty,
            b"untracked-content-sha256",
            _untracked_content_digest(root),
        )
        dirty_patch_digest = dirty.hexdigest()

    resolved_package_version = package_version
    if resolved_package_version is None:
        try:
            resolved_package_version = version(_PACKAGE_DISTRIBUTION)
        except PackageNotFoundError as error:
            raise ValueError(
                "the installed marl-battlegrounds package version is unavailable"
            ) from error
    return build_code_revision_v1(
        package_version=resolved_package_version,
        commit_sha=commit_sha,
        source_tree_digest=source_tree_digest,
        is_dirty=is_dirty,
        dirty_patch_digest=dirty_patch_digest,
    )


__all__ = ["discover_debugger_code_revision_v1"]
