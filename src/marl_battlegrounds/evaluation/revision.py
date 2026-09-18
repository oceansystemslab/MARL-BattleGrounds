"""Record source identity from a checkout or an installed package.

These host-only helpers read Git metadata and source bytes once at recording
setup. They return digests and versions, without embedding local paths. Dirty
checkout identity includes non-ignored untracked content as well as tracked
changes. They do not modify the repository or invent a Git commit for a wheel.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from marl_battlegrounds.evaluation.catalog import build_code_revision_v1
from marl_battlegrounds.evaluation.models import CodeRevisionV1, CodeRevisionV2

_PACKAGE_DISTRIBUTION = "marl-battlegrounds"


def _git(repository_root: Path, *arguments: str) -> bytes:
    """Run a read-only Git query and raise ValueError with stderr on failure."""
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
    """Hash a label and payload with lengths so adjacent fields cannot collide."""
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _untracked_content_digest(repository_root: Path) -> bytes:
    """Hash sorted, non-ignored untracked paths, file kinds and contents.

    Read symlink targets rather than following them. Reject other special file
    kinds. Chunk regular files to bound temporary host memory; filesystem errors
    propagate instead of producing an incomplete source identity.
    """
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


def discover_code_revision_v1(
    repository_root: Path,
    *,
    package_version: str | None = None,
) -> CodeRevisionV1:
    """Describe one Git checkout without placing its local path in the record.

    Parameters
    ----------
    repository_root : Path
        Existing pathlib.Path directory inside the intended Git
        checkout. The path is resolved before any query.
    package_version : str | None
        Optional explicit distribution version. None reads the
        installed marl-battlegrounds version.

    Returns
    -------
    CodeRevisionV1
        CodeRevisionV1 with the actual HEAD commit, a digest of Git's ordered tree
        listing, and dirty status. A dirty record also hashes porcelain status,
        the tracked binary diff against HEAD, and non-ignored untracked content.

    Raises
    ------
    TypeError
        repository_root is not a pathlib.Path.
    ValueError
        The root is not a directory, Git fails, the installed version
        is missing, or an untracked item is neither a regular file nor symlink.
    OSError
        The root or source content cannot be accessed.

    Notes
    -----
        Host-only and read-only. All Git queries refer to the supplied checkout.
        Source files must remain stable during discovery for a coherent snapshot.
        Only digests and package/commit identifiers enter the returned model.
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


def discover_code_revision_v2() -> CodeRevisionV1 | CodeRevisionV2:
    """Describe this imported package using its real source identity.

    Returns
    -------
    CodeRevisionV1 | CodeRevisionV2
        CodeRevisionV1 when this module is inside a detectable source checkout.
        Otherwise CodeRevisionV2 hashes installed package files in sorted relative
        path order, excluding __pycache__, and records the installed version.

    Raises
    ------
    ValueError
        Checkout revision discovery fails.
    OSError
        Package content cannot be read.

    Notes
    -----
        Takes no arguments. This host-only read may invoke Git or read package
        files. A missing distribution version propagates its lookup error.
        Installed content has no invented commit SHA. Keep files stable while
        capturing provenance; this function does not lock or edit them.
    """
    package = Path(__file__).resolve().parents[1]
    for parent in package.parents:
        if (parent / "src" / "marl_battlegrounds") == package and (
            parent / ".git"
        ).exists():
            return discover_code_revision_v1(parent)
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            _framed_update(
                digest, path.relative_to(package).as_posix().encode(), path.read_bytes()
            )
    return CodeRevisionV2(
        package_version=version(_PACKAGE_DISTRIBUTION),
        source_tree_digest=digest.hexdigest(),
    )


__all__ = ["discover_code_revision_v1", "discover_code_revision_v2"]
