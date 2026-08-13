"""Launch-scoped debugger source-revision discovery proofs."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest
from scripts.dev.visual_debugger.revision import (
    discover_debugger_code_revision_v1,
)

from marl_battlegrounds.evaluation.models import canonical_json_bytes


def _run_git(
    repository_root: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> bytes:
    completed = subprocess.run(
        ("git", "-C", os.fspath(repository_root), *arguments),
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={**os.environ, **(environment or {}), "LC_ALL": "C"},
    )
    return completed.stdout


def _initialize_repository(repository_root: Path) -> Path:
    repository_root.mkdir()
    _run_git(repository_root, "init", "--quiet")
    _run_git(repository_root, "config", "user.name", "Revision Test")
    _run_git(repository_root, "config", "user.email", "revision@example.invalid")
    tracked = repository_root / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    (repository_root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _run_git(repository_root, "add", "tracked.txt", ".gitignore")
    fixed_commit_environment = {
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    }
    _run_git(
        repository_root,
        "commit",
        "--quiet",
        "--message",
        "initial",
        environment=fixed_commit_environment,
    )
    return tracked


def _clone_repository(source: Path, destination: Path) -> None:
    subprocess.run(
        ("git", "clone", "--quiet", os.fspath(source), os.fspath(destination)),
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )


def test_clean_revision_matches_head_tree_and_ignores_ignored_files(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "clean"
    _initialize_repository(repository)
    (repository / "ignored.txt").write_text("local-only\n", encoding="utf-8")

    revision = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )

    assert (
        revision.commit_sha
        == _run_git(
            repository,
            "rev-parse",
            "--verify",
            "HEAD",
        )
        .decode()
        .strip()
    )
    assert (
        revision.source_tree_digest
        == hashlib.sha256(
            _run_git(repository, "ls-tree", "-r", "--full-tree", "HEAD")
        ).hexdigest()
    )
    assert not revision.is_dirty
    assert revision.dirty_patch_digest is None
    assert revision.package_version == "0.0.0"


def test_tracked_dirty_revision_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "tracked-dirty"
    tracked = _initialize_repository(repository)
    clean = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )
    tracked.write_text("first dirty value\n", encoding="utf-8")

    first = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )
    repeated = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )
    tracked.write_text("second dirty value\n", encoding="utf-8")
    changed = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )

    assert first == repeated
    assert first.is_dirty
    assert first.dirty_patch_digest is not None
    assert first.commit_sha == clean.commit_sha
    assert first.source_tree_digest == clean.source_tree_digest
    assert changed.dirty_patch_digest != first.dirty_patch_digest


def test_untracked_digest_is_order_independent_and_tracks_path_and_content(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "untracked"
    _initialize_repository(repository)
    first_path = repository / "alpha.txt"
    second_path = repository / "zeta.txt"
    first_path.write_bytes(b"alpha\x00payload")
    second_path.write_bytes(b"zeta payload")

    first = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )
    repeated = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )
    first_path.unlink()
    second_path.unlink()
    second_path.write_bytes(b"zeta payload")
    first_path.write_bytes(b"alpha\x00payload")
    recreated_in_reverse_order = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )
    second_path.write_bytes(b"different payload")
    changed_content = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )
    second_path.unlink()
    (repository / "renamed.txt").write_bytes(b"zeta payload")
    changed_path = discover_debugger_code_revision_v1(
        repository,
        package_version="0.0.0",
    )

    assert first == repeated == recreated_in_reverse_order
    assert first.is_dirty
    assert first.dirty_patch_digest is not None
    assert changed_content.dirty_patch_digest != first.dirty_patch_digest
    assert changed_path.dirty_patch_digest != first.dirty_patch_digest


def test_equal_checkouts_produce_equal_path_free_dirty_revision(
    tmp_path: Path,
) -> None:
    first_repository = tmp_path / "first-location"
    first_tracked = _initialize_repository(first_repository)
    second_repository = tmp_path / "unrelated" / "second-location"
    second_repository.parent.mkdir()
    _clone_repository(first_repository, second_repository)

    first_tracked.write_text("same tracked change\n", encoding="utf-8")
    (second_repository / "tracked.txt").write_text(
        "same tracked change\n",
        encoding="utf-8",
    )
    (first_repository / "new.bin").write_bytes(b"same untracked bytes\x00")
    (second_repository / "new.bin").write_bytes(b"same untracked bytes\x00")

    first = discover_debugger_code_revision_v1(
        first_repository,
        package_version="0.0.0",
    )
    second = discover_debugger_code_revision_v1(
        second_repository,
        package_version="0.0.0",
    )
    first_bytes = canonical_json_bytes(first)
    second_bytes = canonical_json_bytes(second)

    assert first == second
    assert first_bytes == second_bytes
    assert os.fsencode(first_repository) not in first_bytes
    assert os.fsencode(second_repository) not in second_bytes


def test_revision_discovery_rejects_non_path_and_non_repository(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match=r"must be a pathlib\.Path"):
        discover_debugger_code_revision_v1(
            os.fspath(tmp_path),  # type: ignore[arg-type]
            package_version="0.0.0",
        )
    with pytest.raises(ValueError, match="Git revision discovery failed"):
        discover_debugger_code_revision_v1(
            tmp_path,
            package_version="0.0.0",
        )
