"""Safe fixed-root persistence for private DevClient authoring assets."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from marl_battlegrounds.evaluation.models import CodeRevisionV1, canonical_digest_sha256
from scripts.dev.visual_debugger.authoring_compiler import (
    DevAuthoringValidationError,
    compile_dev_map,
    compile_dev_scenario,
    map_semantic_digest,
    scenario_semantic_digest,
)
from scripts.dev.visual_debugger.authoring_models import (
    MAX_DEV_ASSET_SEQUENCE,
    DevAuthoringProblemV1,
    DevCandidateCompileEvidenceV1,
    DevMapCandidateV1,
    DevMapDraftV1,
    DevPromotedAssetV1,
    DevScenarioCandidateV1,
    DevScenarioDraftV1,
    SafeAssetId,
)

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

type DevAssetKind = Literal["map", "scenario"]
type DevDraft = DevMapDraftV1 | DevScenarioDraftV1
type DevCandidate = DevMapCandidateV1 | DevScenarioCandidateV1
_ASSET_KINDS: tuple[DevAssetKind, ...] = ("map", "scenario")


class DevAssetStoreError(ValueError):
    """Base class for deterministic persistence failures."""


class DevDraftRevisionConflictError(DevAssetStoreError):
    """A stale browser revision attempted to replace newer draft work."""


class DevAssetNotFoundError(DevAssetStoreError):
    """A safe identifier did not resolve to an existing persisted asset."""


class DevAssetIntegrityError(DevAssetStoreError):
    """Persisted content failed strict parsing, digest, or compile evidence."""

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


@dataclass(frozen=True, slots=True)
class DevCandidateReferenceV1:
    asset_kind: DevAssetKind
    candidate_id: str


def _validate_asset_id(value: str) -> str:
    if (
        type(value) is not str
        or len(value) > 64
        or _ASSET_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("asset_id must be a safe lowercase kebab-case identifier")
    return value


def _validate_digest(value: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError("candidate_id must be a lowercase SHA-256 digest")
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


def _candidate_content_digest(
    kind: DevAssetKind,
    *,
    content: BaseModel,
    evidence: DevCandidateCompileEvidenceV1,
) -> str:
    """Content-address candidate bytes independently from semantic identity."""
    return canonical_digest_sha256(
        {
            "schema": f"dev-{kind}-candidate-content@1",
            "content": content.model_dump(mode="json", by_alias=True),
            "evidence": evidence.model_dump(mode="json", by_alias=True),
        }
    )


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
    """Own ignored drafts/candidates and tracked promotion destinations."""

    def __init__(
        self,
        repository_root: Path,
        *,
        artifact_root: Path | None = None,
        configs_root: Path | None = None,
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
        self._configs_root = (
            root / "configs"
            if configs_root is None
            else configs_root.resolve(strict=False)
        )

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @property
    def configs_root(self) -> Path:
        return self._configs_root

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

    def _candidate_path(self, kind: DevAssetKind, candidate_id: str) -> Path:
        digest = _validate_digest(candidate_id)
        return _guard_store_path(
            self._artifact_root,
            self._artifact_root / "candidates" / f"{kind}-{digest}.json",
        )

    def _promotion_path(
        self,
        kind: DevAssetKind,
        asset_id: str,
        version: int,
    ) -> Path:
        safe_id = _validate_asset_id(asset_id)
        if type(version) is not int or not 1 <= version <= MAX_DEV_ASSET_SEQUENCE:
            raise ValueError("version must be a positive 32-bit integer")
        collection = "maps" if kind == "map" else "scenarios"
        return _guard_store_path(
            self._configs_root,
            self._configs_root / collection / safe_id / f"v{version}.json",
        )

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
        observed_revision = 0 if latest is None else latest
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
        return saved

    def save_draft_as(
        self,
        draft: DevDraft,
        *,
        asset_id: SafeAssetId,
    ) -> DevDraft:
        """Create revision one under a new safe identity without overwriting."""
        _validate_asset_id(asset_id)
        kind: DevAssetKind = "map" if isinstance(draft, DevMapDraftV1) else "scenario"
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
        copied = draft.model_copy(update={"asset_id": asset_id, "revision": 0})
        return self.save_draft(copied, expected_revision=0)

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

    def freeze_map(
        self,
        draft: DevMapDraftV1,
        *,
        code_revision: CodeRevisionV1,
    ) -> DevMapCandidateV1:
        compiled = compile_dev_map(draft)
        evidence = DevCandidateCompileEvidenceV1(
            semantic_digest=compiled.semantic_digest,
            code_revision=code_revision,
        )
        candidate = DevMapCandidateV1(
            candidate_id=_candidate_content_digest(
                "map",
                content=compiled.content,
                evidence=evidence,
            ),
            content=compiled.content,
            evidence=evidence,
        )
        self._publish_candidate("map", candidate)
        return candidate

    def freeze_scenario(
        self,
        draft: DevScenarioDraftV1,
        *,
        code_revision: CodeRevisionV1,
    ) -> DevScenarioCandidateV1:
        compiled = compile_dev_scenario(draft, require_freeze_qualified=True)
        evidence = DevCandidateCompileEvidenceV1(
            semantic_digest=compiled.semantic_digest,
            resolved_configuration_digest=compiled.resolved_configuration_digest,
            resolved_initial_state_digest=compiled.resolved_initial_state_digest,
            code_revision=code_revision,
        )
        candidate = DevScenarioCandidateV1(
            candidate_id=_candidate_content_digest(
                "scenario",
                content=compiled.content,
                evidence=evidence,
            ),
            content=compiled.content,
            evidence=evidence,
        )
        self._publish_candidate("scenario", candidate)
        return candidate

    def _publish_candidate(
        self,
        kind: DevAssetKind,
        candidate: DevCandidate,
    ) -> None:
        path = self._candidate_path(kind, candidate.candidate_id)
        payload = _serialized(candidate)
        _guard_store_path(self._artifact_root, path)
        if path.exists():
            existing = self.load_candidate(kind, candidate.candidate_id)
            if _serialized(existing) == payload:
                return
            raise DevAssetAlreadyExistsError(
                "candidate semantic identity already exists with different bytes"
            )
        _atomic_no_clobber(self._artifact_root, path, payload)

    def load_candidate(
        self,
        kind: DevAssetKind,
        candidate_id: str,
    ) -> DevCandidate:
        """Strictly reopen and recompile one immutable candidate."""
        path = self._candidate_path(kind, candidate_id)
        try:
            _guard_store_path(self._artifact_root, path)
            payload = path.read_bytes()
        except FileNotFoundError as error:
            raise DevAssetNotFoundError(
                f"{kind} candidate {candidate_id!r} was not found"
            ) from error
        model_type = DevMapCandidateV1 if kind == "map" else DevScenarioCandidateV1
        try:
            candidate = model_type.model_validate_json(payload)
        except ValidationError as error:
            raise DevAssetIntegrityError(
                f"stored {kind} candidate failed strict parsing"
            ) from error
        if candidate.candidate_id != candidate_id:
            raise DevAssetIntegrityError("candidate identity does not match its path")
        if candidate.candidate_id != _candidate_content_digest(
            kind,
            content=candidate.content,
            evidence=candidate.evidence,
        ):
            raise DevAssetIntegrityError("candidate content digest mismatch")
        try:
            if isinstance(candidate, DevMapCandidateV1):
                compiled_map = compile_dev_map(candidate.content)
                if compiled_map.semantic_digest != candidate.evidence.semantic_digest:
                    raise DevAssetIntegrityError(
                        "map candidate semantic digest mismatch"
                    )
            else:
                compiled_scenario = compile_dev_scenario(
                    candidate.content,
                    require_freeze_qualified=True,
                )
                if (
                    compiled_scenario.semantic_digest
                    != candidate.evidence.semantic_digest
                ):
                    raise DevAssetIntegrityError(
                        "scenario candidate semantic digest mismatch"
                    )
                if (
                    compiled_scenario.resolved_configuration_digest
                    != candidate.evidence.resolved_configuration_digest
                    or compiled_scenario.resolved_initial_state_digest
                    != candidate.evidence.resolved_initial_state_digest
                ):
                    raise DevAssetIntegrityError(
                        "scenario candidate compile evidence mismatch"
                    )
        except DevAuthoringValidationError as error:
            raise DevAssetIntegrityError(
                f"stored {kind} candidate no longer revalidates",
                problems=error.problems,
            ) from error
        return candidate

    def iter_candidate_references(
        self,
        kind: DevAssetKind,
    ) -> tuple[DevCandidateReferenceV1, ...]:
        root = _guard_store_path(
            self._artifact_root,
            self._artifact_root / "candidates",
        )
        if not root.is_dir():
            return ()
        pattern = re.compile(rf"^{kind}-([0-9a-f]{{64}})\.json$")
        references: list[DevCandidateReferenceV1] = []
        for path in sorted(root.iterdir(), key=lambda value: value.name):
            _guard_store_path(self._artifact_root, path)
            match = pattern.fullmatch(path.name)
            if path.is_file() and match is not None:
                references.append(DevCandidateReferenceV1(kind, match.group(1)))
        return tuple(references)

    def resolve_candidate_kind(self, candidate_id: str) -> DevAssetKind:
        digest = _validate_digest(candidate_id)
        matches: list[DevAssetKind] = []
        for kind in _ASSET_KINDS:
            path = self._candidate_path(kind, digest)
            _guard_store_path(self._artifact_root, path)
            if path.is_file():
                matches.append(kind)
        if not matches:
            raise DevAssetNotFoundError(f"candidate {digest!r} was not found")
        if len(matches) != 1:
            raise DevAssetIntegrityError(
                "candidate digest resolves to both a map and a scenario"
            )
        return matches[0]

    def promote_candidate(
        self,
        candidate_id: str,
        *,
        asset_id: SafeAssetId,
        version: int,
        approval_provenance: str,
    ) -> Path:
        """Revalidate and write one normalized tracked immutable asset version."""
        kind = self.resolve_candidate_kind(candidate_id)
        candidate = self.load_candidate(kind, candidate_id)
        semantic_digest = (
            map_semantic_digest(candidate.content)
            if isinstance(candidate, DevMapCandidateV1)
            else scenario_semantic_digest(candidate.content)
        )
        if semantic_digest != candidate.evidence.semantic_digest:
            raise DevAssetIntegrityError("candidate digest changed during promotion")
        promoted = DevPromotedAssetV1(
            asset_kind=kind,
            asset_id=asset_id,
            version=version,
            semantic_digest=semantic_digest,
            approved_candidate=candidate,
            approval_provenance=approval_provenance,
        )
        path = self._promotion_path(kind, asset_id, version)
        _atomic_no_clobber(self._configs_root, path, _serialized(promoted))
        return path


__all__ = [
    "DevAssetAlreadyExistsError",
    "DevAssetIntegrityError",
    "DevAssetKind",
    "DevAssetNotFoundError",
    "DevAssetStore",
    "DevAssetStoreError",
    "DevCandidate",
    "DevCandidateReferenceV1",
    "DevDraft",
    "DevDraftReferenceV1",
    "DevDraftRevisionConflictError",
]
