"""Safe fixed-root persistence for private DevClient authoring assets."""

from __future__ import annotations

import fcntl
import os
import re
import stat
import tempfile
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from scripts.dev.visual_debugger.authoring_models import (
    MAX_DEV_ASSET_SEQUENCE,
    DevAuthoringProblemV1,
    DevMapDraftV1,
    DevScenarioDraftV1,
    SafeAssetId,
)

_ASSET_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

type DevAssetKind = Literal["map", "scenario"]
type DevDraft = DevMapDraftV1 | DevScenarioDraftV1


class DevAssetStoreError(ValueError):
    """Base class for deterministic persistence failures."""


class DevDraftRevisionConflictError(DevAssetStoreError):
    """A stale browser revision attempted to replace newer draft work."""


class DevAssetNotFoundError(DevAssetStoreError):
    """A safe identifier did not resolve to an existing persisted asset."""


class DevAssetIntegrityError(DevAssetStoreError):
    """Persisted content or its fixed-root storage failed strict validation."""

    def __init__(
        self,
        message: str,
        *,
        problems: tuple[DevAuthoringProblemV1, ...] = (),
    ) -> None:
        self.problems = problems
        super().__init__(message)


class DevAssetAlreadyExistsError(DevAssetStoreError):
    """An immutable target already exists and cannot be overwritten."""


@dataclass(frozen=True, slots=True)
class DevDraftReferenceV1:
    asset_kind: DevAssetKind
    asset_id: str
    revision: int


def _validate_asset_id(value: str) -> str:
    if (
        type(value) is not str
        or len(value) > 64
        or _ASSET_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("asset_id must be a safe lowercase kebab-case identifier")
    return value


def _stored_revision(path: Path) -> int | None:
    match = re.fullmatch(r"r([1-9][0-9]*)\.json", path.name)
    if not path.is_file() or match is None:
        return None
    revision = int(match.group(1))
    if revision > MAX_DEV_ASSET_SEQUENCE:
        raise DevAssetIntegrityError(
            "stored draft revision exceeds the supported range"
        )
    return revision


def _serialized(model: BaseModel) -> bytes:
    return (
        model.model_dump_json(
            by_alias=True,
            exclude_none=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _write_no_clobber_at(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    """Restore one regular file through an already verified directory handle."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        name,
        flags,
        0o600,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _guard_store_path(root: Path, path: Path) -> Path:
    """Reject every symlink or resolved escape below one configured store root."""
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise DevAssetIntegrityError(
            "persistence path escapes its configured root"
        ) from error
    cursor = root
    if cursor.is_symlink():
        raise DevAssetIntegrityError(
            "configured persistence root must not be a symlink"
        )
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise DevAssetIntegrityError("persistence paths must not contain symlinks")
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise DevAssetIntegrityError(
            "persistence path escapes its configured root"
        ) from error
    return path


def _atomic_no_clobber(root: Path, path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically while rejecting an existing target."""
    _guard_store_path(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _guard_store_path(root, path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _guard_store_path(root, path)
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise DevAssetAlreadyExistsError(
                f"immutable asset already exists: {path.name}"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


class DevAssetStore:
    """Own revisioned ignored DevClient map and scenario drafts."""

    def __init__(
        self,
        repository_root: Path,
        *,
        artifact_root: Path | None = None,
    ) -> None:
        root = repository_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repository_root must be an existing directory")
        self._repository_root = root
        self._artifact_root = (
            root / "artifacts" / "dev_client"
            if artifact_root is None
            else artifact_root.resolve(strict=False)
        )

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @contextmanager
    def _exclusive_store_lock(self) -> Generator[None]:
        """Serialize mutating asset operations across live DevClient processes."""
        path = _guard_store_path(
            self._artifact_root,
            self._artifact_root / "drafts" / ".store.lock",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _guard_store_path(self._artifact_root, path)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise DevAssetIntegrityError(
                "persistence lock could not be opened safely"
            ) from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise DevAssetIntegrityError("persistence lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _revision_fence_directory(
        self,
        kind: DevAssetKind,
        asset_id: str,
    ) -> Path:
        safe_id = _validate_asset_id(asset_id)
        collection = "maps" if kind == "map" else "scenarios"
        return _guard_store_path(
            self._artifact_root,
            self._artifact_root / "drafts" / ".revision_fences" / collection / safe_id,
        )

    def _latest_revision_fence(
        self,
        kind: DevAssetKind,
        asset_id: str,
    ) -> int | None:
        directory = self._revision_fence_directory(kind, asset_id)
        if not directory.exists():
            return None
        if not directory.is_dir():
            raise DevAssetIntegrityError(
                "saved revision fence path must be a directory"
            )
        revisions: list[int] = []
        for path in directory.iterdir():
            _guard_store_path(self._artifact_root, path)
            metadata = path.lstat()
            match = re.fullmatch(r"r([1-9][0-9]*)\.fence", path.name)
            if not stat.S_ISREG(metadata.st_mode) or match is None:
                raise DevAssetIntegrityError(
                    "saved revision fence contains an unexpected entry"
                )
            revision = int(match.group(1))
            if revision > MAX_DEV_ASSET_SEQUENCE:
                raise DevAssetIntegrityError(
                    "saved revision fence exceeds the supported range"
                )
            revisions.append(revision)
        return max(revisions, default=None)

    def _record_revision_fence(
        self,
        kind: DevAssetKind,
        asset_id: str,
        revision: int,
    ) -> None:
        latest = self._latest_revision_fence(kind, asset_id)
        if latest is not None and latest >= revision:
            return
        path = self._revision_fence_directory(kind, asset_id) / f"r{revision}.fence"
        _atomic_no_clobber(self._artifact_root, path, b"")

    def _draft_directory(self, kind: DevAssetKind, asset_id: str) -> Path:
        safe_id = _validate_asset_id(asset_id)
        collection = "maps" if kind == "map" else "scenarios"
        return _guard_store_path(
            self._artifact_root,
            self._artifact_root / "drafts" / collection / safe_id,
        )

    def _draft_path(
        self,
        kind: DevAssetKind,
        asset_id: str,
        revision: int,
    ) -> Path:
        if type(revision) is not int or not 1 <= revision <= MAX_DEV_ASSET_SEQUENCE:
            raise ValueError("saved draft revision must be a positive 32-bit integer")
        return self._draft_directory(kind, asset_id) / f"r{revision}.json"

    @staticmethod
    def _latest_revision(root: Path, directory: Path) -> int | None:
        _guard_store_path(root, directory)
        if not directory.is_dir():
            return None
        revisions: list[int] = []
        for path in directory.iterdir():
            _guard_store_path(root, path)
            revision = _stored_revision(path)
            if revision is not None:
                revisions.append(revision)
        return max(revisions, default=None)

    def save_draft(
        self,
        draft: DevDraft,
        *,
        expected_revision: int,
    ) -> DevDraft:
        """Atomically save the next whole revision or fail on stale state."""
        with self._exclusive_store_lock():
            return self._save_draft_locked(
                draft,
                expected_revision=expected_revision,
            )

    def _save_draft_locked(
        self,
        draft: DevDraft,
        *,
        expected_revision: int,
        allow_deleted_identity_reuse: bool = False,
    ) -> DevDraft:
        if (
            type(expected_revision) is not int
            or not 0 <= expected_revision < MAX_DEV_ASSET_SEQUENCE
        ):
            raise ValueError(
                "expected_revision must be a nonnegative 32-bit integer with "
                "room for the next revision"
            )
        kind: DevAssetKind = "map" if isinstance(draft, DevMapDraftV1) else "scenario"
        directory = self._draft_directory(kind, draft.asset_id)
        latest = self._latest_revision(self._artifact_root, directory)
        fenced = self._latest_revision_fence(kind, draft.asset_id)
        if latest is None and fenced is not None and not allow_deleted_identity_reuse:
            raise DevDraftRevisionConflictError(
                f"deleted {kind} draft {draft.asset_id!r} must be recreated "
                "through Save As"
            )
        if latest is not None and fenced is not None and fenced > latest:
            raise DevAssetIntegrityError(
                "saved revision fence is newer than persisted asset content"
            )
        observed_revision = max(latest or 0, fenced or 0)
        if (
            observed_revision != expected_revision
            or draft.revision != expected_revision
        ):
            raise DevDraftRevisionConflictError(
                f"stale {kind} draft {draft.asset_id!r}: expected revision "
                f"{expected_revision}, current revision is {observed_revision}"
            )
        saved = draft.model_copy(update={"revision": expected_revision + 1})
        path = self._draft_path(kind, saved.asset_id, saved.revision)
        _atomic_no_clobber(self._artifact_root, path, _serialized(saved))
        self._record_revision_fence(kind, saved.asset_id, saved.revision)
        return saved

    def save_draft_as(
        self,
        draft: DevDraft,
        *,
        asset_id: SafeAssetId,
    ) -> DevDraft:
        """Create the next safe revision under a new identity without overwriting."""
        with self._exclusive_store_lock():
            _validate_asset_id(asset_id)
            kind: DevAssetKind = (
                "map" if isinstance(draft, DevMapDraftV1) else "scenario"
            )
            if (
                self._latest_revision(
                    self._artifact_root,
                    self._draft_directory(kind, asset_id),
                )
                is not None
            ):
                raise DevAssetAlreadyExistsError(
                    f"{kind} draft {asset_id!r} already exists"
                )
            previous_revision = self._latest_revision_fence(kind, asset_id) or 0
            copied = draft.model_copy(
                update={"asset_id": asset_id, "revision": previous_revision}
            )
            return self._save_draft_locked(
                copied,
                expected_revision=previous_revision,
                allow_deleted_identity_reuse=True,
            )

    def load_draft(
        self,
        kind: DevAssetKind,
        asset_id: str,
        *,
        revision: int | None = None,
    ) -> DevDraft:
        """Strictly reopen one exact saved revision, defaulting to latest."""
        directory = self._draft_directory(kind, asset_id)
        selected_revision = revision
        if selected_revision is None:
            selected_revision = self._latest_revision(self._artifact_root, directory)
        if selected_revision is None:
            raise DevAssetNotFoundError(f"{kind} draft {asset_id!r} was not found")
        path = self._draft_path(kind, asset_id, selected_revision)
        try:
            _guard_store_path(self._artifact_root, path)
            payload = path.read_bytes()
        except FileNotFoundError as error:
            raise DevAssetNotFoundError(
                f"{kind} draft {asset_id!r} revision {selected_revision} was not found"
            ) from error
        model_type = DevMapDraftV1 if kind == "map" else DevScenarioDraftV1
        try:
            draft = model_type.model_validate_json(payload)
        except ValidationError as error:
            raise DevAssetIntegrityError(
                f"stored {kind} draft failed strict parsing"
            ) from error
        if draft.asset_id != asset_id or draft.revision != selected_revision:
            raise DevAssetIntegrityError(
                "stored draft identity does not match its path"
            )
        return draft

    def iter_draft_references(
        self,
        kind: DevAssetKind,
        *,
        latest_only: bool = False,
    ) -> tuple[DevDraftReferenceV1, ...]:
        collection = "maps" if kind == "map" else "scenarios"
        root = _guard_store_path(
            self._artifact_root,
            self._artifact_root / "drafts" / collection,
        )
        if not root.is_dir():
            return ()
        references: list[DevDraftReferenceV1] = []
        for directory in sorted(root.iterdir(), key=lambda path: path.name):
            _guard_store_path(self._artifact_root, directory)
            if (
                not directory.is_dir()
                or _ASSET_ID_PATTERN.fullmatch(directory.name) is None
            ):
                continue
            revisions: list[int] = []
            for path in directory.iterdir():
                _guard_store_path(self._artifact_root, path)
                revision = _stored_revision(path)
                if revision is not None:
                    revisions.append(revision)
            selected = (max(revisions),) if latest_only and revisions else revisions
            references.extend(
                DevDraftReferenceV1(kind, directory.name, revision)
                for revision in sorted(selected)
            )
        return tuple(references)

    def delete_draft(
        self,
        kind: DevAssetKind,
        asset_id: str,
        *,
        expected_revision: int,
    ) -> tuple[DevDraftReferenceV1, ...]:
        """Delete one exact saved identity after a complete, fail-closed preflight."""
        with self._exclusive_store_lock():
            return self._delete_draft_locked(
                kind,
                asset_id,
                expected_revision=expected_revision,
            )

    def _delete_draft_locked(
        self,
        kind: DevAssetKind,
        asset_id: str,
        *,
        expected_revision: int,
    ) -> tuple[DevDraftReferenceV1, ...]:
        if (
            type(expected_revision) is not int
            or not 1 <= expected_revision <= MAX_DEV_ASSET_SEQUENCE
        ):
            raise ValueError("expected_revision must be a positive 32-bit integer")

        directory = self._draft_directory(kind, asset_id)
        _guard_store_path(self._artifact_root, directory)
        if not directory.exists():
            raise DevAssetNotFoundError(f"{kind} draft {asset_id!r} was not found")
        if not directory.is_dir():
            raise DevAssetIntegrityError("saved asset path must be a directory")

        revision_paths: list[tuple[int, Path]] = []
        for path in sorted(directory.iterdir(), key=lambda value: value.name):
            _guard_store_path(self._artifact_root, path)
            try:
                metadata = path.lstat()
            except FileNotFoundError as error:
                raise DevAssetIntegrityError(
                    "saved asset changed during deletion preflight"
                ) from error
            match = re.fullmatch(r"r([1-9][0-9]*)\.json", path.name)
            if not stat.S_ISREG(metadata.st_mode) or match is None:
                raise DevAssetIntegrityError(
                    "saved asset contains an unexpected entry; nothing was deleted"
                )
            revision = int(match.group(1))
            if revision > MAX_DEV_ASSET_SEQUENCE:
                raise DevAssetIntegrityError(
                    "stored draft revision exceeds the supported range"
                )
            revision_paths.append((revision, path))
        revision_paths.sort(key=lambda item: item[0])

        if not revision_paths:
            raise DevAssetNotFoundError(f"{kind} draft {asset_id!r} was not found")
        latest_revision = max(revision for revision, _ in revision_paths)
        if latest_revision != expected_revision:
            raise DevDraftRevisionConflictError(
                f"stale {kind} draft {asset_id!r}: expected revision "
                f"{expected_revision}, current revision is {latest_revision}"
            )
        fenced_revision = self._latest_revision_fence(kind, asset_id)
        if fenced_revision is not None and fenced_revision > latest_revision:
            raise DevAssetIntegrityError(
                "saved revision fence is newer than persisted asset content"
            )

        # Strict parsing is part of deletion preflight. A syntactically valid
        # filename is insufficient: every payload must join back to the exact
        # kind, asset ID, and revision encoded by its path.
        validated_metadata: dict[int, tuple[int, int, int, int]] = {}
        revision_payloads: dict[int, bytes] = {}
        for revision, path in revision_paths:
            before = path.lstat()
            self.load_draft(kind, asset_id, revision=revision)
            revision_payloads[revision] = path.read_bytes()
            after = path.lstat()
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if before_identity != after_identity:
                raise DevAssetIntegrityError(
                    "saved asset changed during deletion preflight; nothing was deleted"
                )
            validated_metadata[revision] = after_identity

        # Recheck the complete directory immediately before the first unlink.
        observed_names = tuple(
            path.name
            for path in sorted(directory.iterdir(), key=lambda value: value.name)
        )
        expected_names = tuple(path.name for _, path in revision_paths)
        if observed_names != expected_names:
            raise DevAssetIntegrityError(
                "saved asset changed during deletion preflight; nothing was deleted"
            )
        for revision, path in revision_paths:
            metadata = path.lstat()
            identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
            if (
                not stat.S_ISREG(metadata.st_mode)
                or identity != validated_metadata[revision]
            ):
                raise DevAssetIntegrityError(
                    "saved asset changed during deletion preflight; nothing was deleted"
                )

        deleted = tuple(
            DevDraftReferenceV1(kind, asset_id, revision)
            for revision, _ in revision_paths
        )
        # Persist the generation boundary before removing content so another
        # live DevClient can never recreate this identity at an old revision.
        self._record_revision_fence(kind, asset_id, latest_revision)
        directory_identity = directory.lstat()
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(directory.parent, directory_flags | no_follow)
        directory_descriptor = os.open(
            directory.name,
            directory_flags | no_follow,
            dir_fd=parent_descriptor,
        )
        try:
            opened_identity = os.fstat(directory_descriptor)
            if (
                opened_identity.st_dev != directory_identity.st_dev
                or opened_identity.st_ino != directory_identity.st_ino
            ):
                raise DevAssetIntegrityError(
                    "saved asset changed before deletion; nothing was deleted"
                )
            try:
                for _, path in revision_paths:
                    os.unlink(path.name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
                os.rmdir(directory.name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except BaseException as error:
                rollback_errors: list[OSError] = []
                rollback_descriptor = directory_descriptor
                close_rollback_descriptor = False
                try:
                    try:
                        rollback_directory = os.stat(
                            directory.name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        try:
                            os.mkdir(
                                directory.name,
                                stat.S_IMODE(directory_identity.st_mode),
                                dir_fd=parent_descriptor,
                            )
                            os.fsync(parent_descriptor)
                            rollback_descriptor = os.open(
                                directory.name,
                                directory_flags | no_follow,
                                dir_fd=parent_descriptor,
                            )
                            close_rollback_descriptor = True
                        except OSError as rollback_error:
                            rollback_errors.append(rollback_error)
                    else:
                        if (
                            not stat.S_ISDIR(rollback_directory.st_mode)
                            or rollback_directory.st_dev != directory_identity.st_dev
                            or rollback_directory.st_ino != directory_identity.st_ino
                        ):
                            rollback_errors.append(
                                OSError("saved asset directory changed during rollback")
                            )
                    if not rollback_errors:
                        for revision, path in revision_paths:
                            try:
                                os.stat(
                                    path.name,
                                    dir_fd=rollback_descriptor,
                                    follow_symlinks=False,
                                )
                            except FileNotFoundError:
                                try:
                                    _write_no_clobber_at(
                                        rollback_descriptor,
                                        path.name,
                                        revision_payloads[revision],
                                    )
                                except OSError as rollback_error:
                                    rollback_errors.append(rollback_error)
                        if not rollback_errors:
                            os.fsync(rollback_descriptor)
                            os.fsync(parent_descriptor)
                finally:
                    if close_rollback_descriptor:
                        os.close(rollback_descriptor)
                if rollback_errors:
                    raise DevAssetIntegrityError(
                        "saved asset deletion failed and rollback was incomplete"
                    ) from error
                if isinstance(error, OSError):
                    raise DevAssetIntegrityError(
                        "saved asset deletion failed; original revisions were restored"
                    ) from error
                raise
        finally:
            # Both descriptors are read-only. The content commit or rollback
            # and its directory-entry durability are complete before cleanup;
            # a post-effect close error must not turn that settled result into
            # a false operation failure.
            for descriptor in (directory_descriptor, parent_descriptor):
                with suppress(OSError):
                    os.close(descriptor)
        return deleted


__all__ = [
    "DevAssetAlreadyExistsError",
    "DevAssetIntegrityError",
    "DevAssetKind",
    "DevAssetNotFoundError",
    "DevAssetStore",
    "DevAssetStoreError",
    "DevDraft",
    "DevDraftReferenceV1",
    "DevDraftRevisionConflictError",
]
