"""Canonical, bounded local persistence for semantic evaluation artifacts.

This module owns bytes and paths, not rollout semantics.  It accepts only the
strict replay/report/POV/scenario models, reuses their semantic validators, and
publishes a metric sidecar before the replay that references it.  It performs
no network, archive, plugin, simulator, policy, JAX, or device work.
"""

from __future__ import annotations

import json
import os
import stat
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path
from secrets import token_hex
from typing import Literal, cast

from pydantic import ValidationError

from marl_battlegrounds.evaluation.models import (
    EvaluationFrameV1,
    EvaluationTransitionV1,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import (
    ACTOR_POV_ARTIFACT_SCHEMA_ID,
    ActorPovReplayArtifactV1,
    validate_actor_pov_replay_against_replay_v1,
    validate_actor_pov_replay_artifact_v1,
)
from marl_battlegrounds.evaluation.replay import (
    METRIC_REPORT_ARTIFACT_SCHEMA_ID,
    REPLAY_ARTIFACT_SCHEMA_ID,
    REPLAY_SCHEMA_VERSION,
    EvaluationMetricReportArtifactV1,
    ReplayArtifactV1,
    ReplayBundleV1,
    _validate_metric_report_artifact_against_validated_replay_v1,  # pyright: ignore[reportPrivateUsage]
    validate_replay_artifact_v1,
)
from marl_battlegrounds.evaluation.scenario import (
    SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
    ScenarioEvaluationRecordV1,
    validate_scenario_evaluation_record_v1,
)
from marl_battlegrounds.evaluation.validation import validate_declared_model_tree

DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1 = 1024**3
DEFAULT_MAX_REPLAY_JSON_DEPTH_V1 = 128
REPLAY_FILE_SUFFIX_V1 = ".marlbg-replay.json"
METRIC_REPORT_FILE_SUFFIX_V1 = ".marlbg-metrics.json"
ACTOR_POV_FILE_SUFFIX_V1 = ".marlbg-pov.json"
SCENARIO_FILE_SUFFIX_V1 = ".marlbg-scenario.json"

type ReplayBundleLoadStatusV1 = Literal["complete", "metric_report_missing"]
type ReplayIOErrorCodeV1 = Literal[
    "invalid_argument",
    "unsupported_platform",
    "invalid_filename",
    "missing_parent",
    "path_not_found",
    "path_is_symlink",
    "path_not_regular_file",
    "path_not_directory",
    "file_too_large",
    "file_read_failed",
    "utf8_bom_forbidden",
    "invalid_utf8",
    "json_depth_exceeded",
    "duplicate_json_key",
    "nonfinite_json_number",
    "malformed_json",
    "wrong_root_schema",
    "unsupported_schema_version",
    "model_validation_failed",
    "semantic_validation_failed",
    "noncanonical_json",
    "metric_report_missing",
    "metric_report_mismatch",
    "replay_target_exists",
    "metric_report_conflict",
    "companion_target_exists",
    "temporary_write_failed",
    "atomic_publish_failed",
    "replay_publication_verification_failed",
]


class ReplayIOError(Exception):
    """Base error with a stable machine-readable code and affected path."""

    code: ReplayIOErrorCodeV1
    path: Path | None
    detail: str

    def __init__(
        self,
        code: ReplayIOErrorCodeV1,
        *,
        path: Path | None,
        detail: str,
    ) -> None:
        self.code = code
        self.path = path
        self.detail = detail
        location = "" if path is None else f" ({path})"
        super().__init__(f"{code}{location}: {detail}")


class ReplayLoadError(ReplayIOError):
    """A replay or metric sidecar could not be loaded safely."""


class ReplaySaveError(ReplayIOError):
    """A replay bundle could not be published safely."""


@dataclass(frozen=True, slots=True)
class LoadedReplayBundleV1:
    """One loaded replay and optional resolved metric-report sidecar."""

    replay: ReplayArtifactV1
    metric_report_artifact: EvaluationMetricReportArtifactV1 | None
    status: ReplayBundleLoadStatusV1

    def frame_at(self, frame_index: int) -> EvaluationFrameV1:
        """Return one canonical frame by O(1) artifact index lookup."""
        if type(frame_index) is not int:
            raise TypeError("frame index must be an integer")
        if frame_index < 0 or frame_index >= len(self.replay.frames):
            raise IndexError("frame index is outside the captured replay prefix")
        return self.replay.frames[frame_index]

    def incoming_transition_at(
        self,
        frame_index: int,
    ) -> EvaluationTransitionV1 | None:
        """Return the incoming transition for a frame, or ``None`` at frame zero."""
        self.frame_at(frame_index)
        if frame_index == 0:
            return None
        return self.replay.transitions[frame_index - 1]


@dataclass(frozen=True, slots=True)
class SavedReplayBundleV1:
    """Local publication result; paths never enter serialized artifacts."""

    replay_path: Path
    metric_report_path: Path
    replay_byte_length: int
    metric_report_byte_length: int
    metric_report_reused: bool


@dataclass(frozen=True, slots=True)
class ReplayBundleDestinationV1:
    """One structurally preflighted local replay/report filename pair."""

    replay_path: Path
    metric_report_path: Path

    def __post_init__(self) -> None:
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.replay_path, Path
        ) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.metric_report_path, Path
        ):
            raise TypeError("replay bundle destination paths must use pathlib.Path")
        if _metric_report_path_for_replay(self.replay_path) != self.metric_report_path:
            raise ValueError("metric report path must derive from the replay filename")


class _PreparedReplayBundleTooLargeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedReplayBundleV1:
    """Validated replay/report models and their immutable canonical bytes."""

    bundle: ReplayBundleV1
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1
    replay_json_bytes: bytes = field(init=False, repr=False)
    metric_report_json_bytes: bytes = field(init=False, repr=False)
    replay_byte_length: int = field(init=False)
    metric_report_byte_length: int = field(init=False)
    replay_payload_sha256: str = field(init=False)
    metric_report_payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.bundle) is not ReplayBundleV1:
            raise TypeError("prepared replay bundle requires exact ReplayBundleV1")
        size_limit = _require_positive_limit(
            self.max_file_size_bytes,
            name="max_file_size_bytes",
        )
        validate_replay_artifact_v1(self.bundle.replay)
        _validate_metric_report_artifact_against_validated_replay_v1(
            self.bundle.metric_report_artifact,
            self.bundle.replay,
        )
        replay_payload = canonical_json_bytes(self.bundle.replay)
        metric_report_payload = canonical_json_bytes(self.bundle.metric_report_artifact)
        replay_byte_length = len(replay_payload)
        metric_report_byte_length = len(metric_report_payload)
        if replay_byte_length > size_limit or metric_report_byte_length > size_limit:
            raise _PreparedReplayBundleTooLargeError(
                f"bundle member exceeds {size_limit} bytes"
            )
        object.__setattr__(self, "replay_json_bytes", replay_payload)
        object.__setattr__(
            self,
            "metric_report_json_bytes",
            metric_report_payload,
        )
        object.__setattr__(self, "replay_byte_length", replay_byte_length)
        object.__setattr__(
            self,
            "metric_report_byte_length",
            metric_report_byte_length,
        )
        object.__setattr__(
            self,
            "replay_payload_sha256",
            sha256(replay_payload).hexdigest(),
        )
        object.__setattr__(
            self,
            "metric_report_payload_sha256",
            sha256(metric_report_payload).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class SavedCompanionArtifactV1:
    """Publication result for one independently addressed companion artifact."""

    path: Path
    byte_length: int


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


def _require_positive_limit(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coerce_path(path: str | os.PathLike[str]) -> Path:
    if isinstance(path, str):
        if not path:
            raise ValueError("path must not be empty")
        return Path(path)
    try:
        return Path(path)
    except TypeError as error:
        raise TypeError("path must be a string or path-like value") from error


def _metric_report_path_for_replay(replay_path: Path) -> Path:
    name = replay_path.name
    if not name.endswith(REPLAY_FILE_SUFFIX_V1):
        raise ValueError(f"replay filename must end with {REPLAY_FILE_SUFFIX_V1}")
    base_name = name[: -len(REPLAY_FILE_SUFFIX_V1)]
    if not base_name:
        raise ValueError("replay filename must include an episode stem")
    return replay_path.with_name(f"{base_name}{METRIC_REPORT_FILE_SUFFIX_V1}")


def _require_artifact_suffix(path: Path, *, suffix: str, label: str) -> None:
    if not path.name.endswith(suffix) or path.name == suffix:
        raise ValueError(f"{label} filename must end with {suffix}")


def _require_secure_directory_fd_support(
    *,
    path: Path,
    error_type: type[ReplayLoadError] | type[ReplaySaveError],
) -> None:
    required_dir_fd_functions = (os.open, os.stat, os.link, os.unlink)
    if (
        os.name != "posix"
        or getattr(os, "O_DIRECTORY", 0) == 0
        or getattr(os, "O_NOFOLLOW", 0) == 0
        or any(
            function not in os.supports_dir_fd for function in required_dir_fd_functions
        )
        or os.stat not in os.supports_follow_symlinks
        or os.link not in os.supports_follow_symlinks
    ):
        raise error_type(
            "unsupported_platform",
            path=path,
            detail=(
                "secure replay paths require POSIX directory-fd and no-follow support"
            ),
        )


def _open_parent_directory(
    path: Path,
    *,
    error_type: type[ReplayLoadError] | type[ReplaySaveError],
) -> int:
    """Open every parent component without following links and return its fd."""
    _require_secure_directory_fd_support(path=path, error_type=error_type)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    start = path.anchor if path.is_absolute() else "."
    components = path.parts[1:-1] if path.is_absolute() else path.parts[:-1]
    try:
        descriptor = os.open(start, flags)
    except OSError as error:
        raise error_type(
            "file_read_failed",
            path=path.parent,
            detail="could not open the replay path root",
        ) from error
    current_path = Path(path.anchor) if path.is_absolute() else Path.cwd()
    try:
        for component in components:
            if component in ("", "."):
                continue
            component_path = current_path / component
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError as error:
                code: ReplayIOErrorCodeV1 = (
                    "missing_parent"
                    if error_type is ReplaySaveError
                    else "path_not_found"
                )
                raise error_type(
                    code,
                    path=component_path,
                    detail="replay path parent does not exist",
                ) from error
            except OSError as error:
                try:
                    component_status = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError:
                    component_status = None
                if component_status is not None and stat.S_ISLNK(
                    component_status.st_mode
                ):
                    raise error_type(
                        "path_is_symlink",
                        path=component_path,
                        detail="symlink paths are outside the replay contract",
                    ) from error
                raise error_type(
                    "path_not_directory",
                    path=component_path,
                    detail="replay path parent is not an accessible directory",
                ) from error
            os.close(descriptor)
            descriptor = next_descriptor
            current_path = component_path
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_bounded_regular_file_at(
    parent_descriptor: int,
    name: str,
    *,
    path: Path,
    max_file_size_bytes: int,
    error_type: type[ReplayLoadError] | type[ReplaySaveError] = ReplayLoadError,
    fsync_before_close: bool = False,
) -> bytes:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError as error:
        raise error_type(
            "path_not_found",
            path=path,
            detail="file does not exist",
        ) from error
    except OSError as error:
        try:
            target_status = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            target_status = None
        if target_status is not None and stat.S_ISLNK(target_status.st_mode):
            raise error_type(
                "path_is_symlink",
                path=path,
                detail="symlink paths are outside the replay contract",
            ) from error
        raise error_type(
            "file_read_failed",
            path=path,
            detail="file could not be opened",
        ) from error
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise error_type(
                "path_not_regular_file",
                path=path,
                detail="replay inputs must be regular files",
            )
        if file_status.st_size > max_file_size_bytes:
            raise error_type(
                "file_too_large",
                path=path,
                detail=f"file exceeds {max_file_size_bytes} bytes",
            )
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            payload = stream.read(max_file_size_bytes + 1)
            if fsync_before_close:
                try:
                    os.fsync(stream.fileno())
                except OSError as error:
                    raise error_type(
                        "atomic_publish_failed",
                        path=path,
                        detail="existing metric sidecar could not be made durable",
                    ) from error
    except ReplayIOError:
        raise
    except OSError as error:
        raise error_type(
            "file_read_failed",
            path=path,
            detail="file could not be read",
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > max_file_size_bytes:
        raise error_type(
            "file_too_large",
            path=path,
            detail=f"file exceeds {max_file_size_bytes} bytes",
        )
    return payload


def _read_bounded_regular_file(
    path: Path,
    *,
    max_file_size_bytes: int,
    error_type: type[ReplayLoadError] | type[ReplaySaveError] = ReplayLoadError,
) -> bytes:
    parent_descriptor = _open_parent_directory(path, error_type=error_type)
    try:
        return _read_bounded_regular_file_at(
            parent_descriptor,
            path.name,
            path=path,
            max_file_size_bytes=max_file_size_bytes,
            error_type=error_type,
        )
    finally:
        os.close(parent_descriptor)


def _require_json_depth(
    text: str,
    *,
    path: Path,
    max_json_depth: int,
) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > max_json_depth:
                raise ReplayLoadError(
                    "json_depth_exceeded",
                    path=path,
                    detail=f"JSON nesting exceeds {max_json_depth}",
                )
        elif character in "]}":
            depth -= 1


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise _NonFiniteNumberError(value)


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise _NonFiniteNumberError(value)
    return parsed


def _preflight_json(
    payload: bytes,
    *,
    path: Path,
    expected_schema_id: str,
    max_json_depth: int,
) -> None:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ReplayLoadError(
            "utf8_bom_forbidden",
            path=path,
            detail="canonical JSON must not contain a UTF-8 BOM",
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ReplayLoadError(
            "invalid_utf8",
            path=path,
            detail="file is not strict UTF-8",
        ) from error
    _require_json_depth(text, path=path, max_json_depth=max_json_depth)
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except _DuplicateKeyError as error:
        raise ReplayLoadError(
            "duplicate_json_key",
            path=path,
            detail=f"duplicate JSON key: {error}",
        ) from error
    except _NonFiniteNumberError as error:
        raise ReplayLoadError(
            "nonfinite_json_number",
            path=path,
            detail=f"non-finite JSON number: {error}",
        ) from error
    except (json.JSONDecodeError, RecursionError) as error:
        raise ReplayLoadError(
            "malformed_json",
            path=path,
            detail="file is not one complete JSON value",
        ) from error
    except ValueError as error:
        raise ReplayLoadError(
            "malformed_json",
            path=path,
            detail="JSON contains an unsupported numeric literal",
        ) from error
    if type(parsed) is not dict:
        raise ReplayLoadError(
            "wrong_root_schema",
            path=path,
            detail="artifact JSON root must be an object",
        )
    root = cast(dict[str, object], parsed)
    if root.get("schema_id") != expected_schema_id:
        raise ReplayLoadError(
            "wrong_root_schema",
            path=path,
            detail=f"expected root schema {expected_schema_id}",
        )
    schema_version = root.get("schema_version")
    if type(schema_version) is not int or schema_version != REPLAY_SCHEMA_VERSION:
        raise ReplayLoadError(
            "unsupported_schema_version",
            path=path,
            detail="only exact schema version 1 is supported",
        )


def _load_replay_bytes(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
) -> ReplayArtifactV1:
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=REPLAY_ARTIFACT_SCHEMA_ID,
        max_json_depth=max_json_depth,
    )
    try:
        replay = ReplayArtifactV1.model_validate_json(payload)
    except ValidationError as error:
        raise ReplayLoadError(
            "model_validation_failed",
            path=path,
            detail="replay does not satisfy its strict model contract",
        ) from error
    try:
        validate_replay_artifact_v1(replay)
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "semantic_validation_failed",
            path=path,
            detail="replay fails whole-artifact semantic validation",
        ) from error
    canonical = canonical_json_bytes(replay)
    if canonical != payload:
        raise ReplayLoadError(
            "noncanonical_json",
            path=path,
            detail="replay bytes are not the canonical V1 encoding",
        )
    return replay


def _load_metric_report_bytes(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
) -> EvaluationMetricReportArtifactV1:
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=METRIC_REPORT_ARTIFACT_SCHEMA_ID,
        max_json_depth=max_json_depth,
    )
    try:
        report = EvaluationMetricReportArtifactV1.model_validate_json(payload)
        canonical_report = cast(
            EvaluationMetricReportArtifactV1,
            validate_declared_model_tree(
                report,
                record_name="loaded metric report artifact",
                expected_type=EvaluationMetricReportArtifactV1,
            ),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ReplayLoadError(
            "model_validation_failed",
            path=path,
            detail="metric report does not satisfy its strict model contract",
        ) from error
    if canonical_json_bytes(canonical_report) != payload:
        raise ReplayLoadError(
            "noncanonical_json",
            path=path,
            detail="metric report bytes are not the canonical V1 encoding",
        )
    return canonical_report


def _load_actor_pov_bytes(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
) -> ActorPovReplayArtifactV1:
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=ACTOR_POV_ARTIFACT_SCHEMA_ID,
        max_json_depth=max_json_depth,
    )
    try:
        artifact = ActorPovReplayArtifactV1.model_validate_json(payload)
        validate_actor_pov_replay_artifact_v1(artifact)
    except (TypeError, ValueError, ValidationError) as error:
        raise ReplayLoadError(
            "model_validation_failed",
            path=path,
            detail="actor POV does not satisfy its strict artifact contract",
        ) from error
    if canonical_json_bytes(artifact) != payload:
        raise ReplayLoadError(
            "noncanonical_json",
            path=path,
            detail="actor POV bytes are not the canonical V1 encoding",
        )
    return artifact


def _load_scenario_record_bytes(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
) -> ScenarioEvaluationRecordV1:
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        max_json_depth=max_json_depth,
    )
    try:
        record = ScenarioEvaluationRecordV1.model_validate_json(payload)
        canonical_record = cast(
            ScenarioEvaluationRecordV1,
            validate_declared_model_tree(
                record,
                record_name="loaded scenario evaluation record",
                expected_type=ScenarioEvaluationRecordV1,
            ),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ReplayLoadError(
            "model_validation_failed",
            path=path,
            detail="scenario record does not satisfy its strict artifact contract",
        ) from error
    if canonical_json_bytes(canonical_record) != payload:
        raise ReplayLoadError(
            "noncanonical_json",
            path=path,
            detail="scenario bytes are not the canonical V1 encoding",
        )
    return canonical_record


def canonical_replay_json_bytes_v1(artifact: ReplayArtifactV1) -> bytes:
    """Return canonical replay bytes after full semantic validation."""
    validate_replay_artifact_v1(artifact)
    return canonical_json_bytes(artifact)


def canonical_metric_report_artifact_json_bytes_v1(
    artifact: EvaluationMetricReportArtifactV1,
) -> bytes:
    """Return canonical bytes for one strict local metric-report artifact."""
    canonical_artifact = cast(
        EvaluationMetricReportArtifactV1,
        validate_declared_model_tree(
            artifact,
            record_name="metric report artifact",
            expected_type=EvaluationMetricReportArtifactV1,
        ),
    )
    return canonical_json_bytes(canonical_artifact)


def canonical_scenario_evaluation_record_json_bytes_v1(
    record: ScenarioEvaluationRecordV1,
) -> bytes:
    """Return canonical bytes for one structurally valid scenario record."""
    canonical_record = cast(
        ScenarioEvaluationRecordV1,
        validate_declared_model_tree(
            record,
            record_name="scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV1,
        ),
    )
    return canonical_json_bytes(canonical_record)


def load_replay_artifact_v1(
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ReplayArtifactV1:
    """Load one canonical replay after byte, model, and semantic validation."""
    try:
        replay_path = _coerce_path(path)
        size_limit = _require_positive_limit(
            max_file_size_bytes,
            name="max_file_size_bytes",
        )
        depth_limit = _require_positive_limit(max_json_depth, name="max_json_depth")
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "invalid_argument",
            path=None,
            detail=str(error),
        ) from error
    try:
        _metric_report_path_for_replay(replay_path)
    except ValueError as error:
        raise ReplayLoadError(
            "invalid_filename",
            path=replay_path,
            detail=str(error),
        ) from error
    payload = _read_bounded_regular_file(
        replay_path,
        max_file_size_bytes=size_limit,
    )
    return _load_replay_bytes(
        payload,
        path=replay_path,
        max_json_depth=depth_limit,
    )


def load_replay_bundle_v1(
    path: str | os.PathLike[str],
    *,
    require_metric_report: bool = False,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> LoadedReplayBundleV1:
    """Load a replay and resolve its adjacent metric sidecar when available."""
    if type(require_metric_report) is not bool:
        raise ReplayLoadError(
            "invalid_argument",
            path=None,
            detail="require_metric_report must be a boolean",
        )
    try:
        replay_path = _coerce_path(path)
        size_limit = _require_positive_limit(
            max_file_size_bytes,
            name="max_file_size_bytes",
        )
        depth_limit = _require_positive_limit(max_json_depth, name="max_json_depth")
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "invalid_argument",
            path=None,
            detail=str(error),
        ) from error
    try:
        metric_report_path = _metric_report_path_for_replay(replay_path)
    except ValueError as error:
        raise ReplayLoadError(
            "invalid_filename",
            path=replay_path,
            detail=str(error),
        ) from error
    parent_descriptor = _open_parent_directory(
        replay_path,
        error_type=ReplayLoadError,
    )
    try:
        replay_payload = _read_bounded_regular_file_at(
            parent_descriptor,
            replay_path.name,
            path=replay_path,
            max_file_size_bytes=size_limit,
        )
        replay = _load_replay_bytes(
            replay_payload,
            path=replay_path,
            max_json_depth=depth_limit,
        )
        try:
            report_payload = _read_bounded_regular_file_at(
                parent_descriptor,
                metric_report_path.name,
                path=metric_report_path,
                max_file_size_bytes=size_limit,
            )
        except ReplayLoadError as error:
            if error.code != "path_not_found":
                raise
            if require_metric_report:
                raise ReplayLoadError(
                    "metric_report_missing",
                    path=metric_report_path,
                    detail="the replay's adjacent metric sidecar is missing",
                ) from error
            return LoadedReplayBundleV1(
                replay=replay,
                metric_report_artifact=None,
                status="metric_report_missing",
            )
    finally:
        os.close(parent_descriptor)
    metric_report = _load_metric_report_bytes(
        report_payload,
        path=metric_report_path,
        max_json_depth=depth_limit,
    )
    try:
        _validate_metric_report_artifact_against_validated_replay_v1(
            metric_report,
            replay,
        )
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "metric_report_mismatch",
            path=metric_report_path,
            detail="metric report does not join the loaded replay",
        ) from error
    return LoadedReplayBundleV1(
        replay=replay,
        metric_report_artifact=metric_report,
        status="complete",
    )


def load_actor_pov_replay_artifact_v1(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV1 | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ActorPovReplayArtifactV1:
    """Load one independently shareable POV artifact, optionally source-joined."""
    try:
        pov_path = _coerce_path(path)
        size_limit = _require_positive_limit(
            max_file_size_bytes,
            name="max_file_size_bytes",
        )
        depth_limit = _require_positive_limit(max_json_depth, name="max_json_depth")
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "invalid_argument",
            path=None,
            detail=str(error),
        ) from error
    try:
        _require_artifact_suffix(
            pov_path,
            suffix=ACTOR_POV_FILE_SUFFIX_V1,
            label="actor POV",
        )
    except ValueError as error:
        raise ReplayLoadError(
            "invalid_filename",
            path=pov_path,
            detail=str(error),
        ) from error
    payload = _read_bounded_regular_file(
        pov_path,
        max_file_size_bytes=size_limit,
    )
    artifact = _load_actor_pov_bytes(
        payload,
        path=pov_path,
        max_json_depth=depth_limit,
    )
    if source_replay is not None:
        try:
            validate_actor_pov_replay_against_replay_v1(
                artifact,
                source_replay,
            )
        except (TypeError, ValueError) as error:
            raise ReplayLoadError(
                "semantic_validation_failed",
                path=pov_path,
                detail="actor POV does not join its supplied source replay",
            ) from error
    return artifact


def load_scenario_evaluation_record_v1(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ScenarioEvaluationRecordV1:
    """Load one canonical scenario record and verify both evidence joins."""
    try:
        scenario_path = _coerce_path(path)
        size_limit = _require_positive_limit(
            max_file_size_bytes,
            name="max_file_size_bytes",
        )
        depth_limit = _require_positive_limit(max_json_depth, name="max_json_depth")
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "invalid_argument",
            path=None,
            detail=str(error),
        ) from error
    try:
        _require_artifact_suffix(
            scenario_path,
            suffix=SCENARIO_FILE_SUFFIX_V1,
            label="scenario",
        )
    except ValueError as error:
        raise ReplayLoadError(
            "invalid_filename",
            path=scenario_path,
            detail=str(error),
        ) from error
    payload = _read_bounded_regular_file(
        scenario_path,
        max_file_size_bytes=size_limit,
    )
    record = _load_scenario_record_bytes(
        payload,
        path=scenario_path,
        max_json_depth=depth_limit,
    )
    try:
        validate_scenario_evaluation_record_v1(
            record,
            source_replay,
            metric_report_artifact,
        )
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "semantic_validation_failed",
            path=scenario_path,
            detail="scenario record does not join its supplied replay/report",
        ) from error
    return record


def prepare_replay_bundle_v1(
    bundle: ReplayBundleV1,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> PreparedReplayBundleV1:
    """Validate one bundle and cache its exact canonical publication bytes."""
    try:
        return PreparedReplayBundleV1(
            bundle=bundle,
            max_file_size_bytes=max_file_size_bytes,
        )
    except _PreparedReplayBundleTooLargeError as error:
        raise ReplaySaveError(
            "file_too_large",
            path=None,
            detail=str(error),
        ) from error
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=None,
            detail="bundle fails replay/report validation",
        ) from error


def _validate_save_destination(replay_path: Path) -> Path:
    try:
        metric_report_path = _metric_report_path_for_replay(replay_path)
    except ValueError as error:
        raise ReplaySaveError(
            "invalid_filename",
            path=replay_path,
            detail=str(error),
        ) from error
    return metric_report_path


def _entry_status_at(
    parent_descriptor: int,
    name: str,
    *,
    path: Path,
    error_type: type[ReplayLoadError] | type[ReplaySaveError],
) -> os.stat_result | None:
    try:
        entry_status = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise error_type(
            "file_read_failed",
            path=path,
            detail="could not inspect replay path entry",
        ) from error
    if stat.S_ISLNK(entry_status.st_mode):
        raise error_type(
            "path_is_symlink",
            path=path,
            detail="symlink paths are outside the replay contract",
        )
    return entry_status


def preflight_replay_bundle_destination_v1(
    path: str | os.PathLike[str],
) -> ReplayBundleDestinationV1:
    """Validate an absent replay target and existing local parent without writes."""
    try:
        replay_path = _coerce_path(path)
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=None,
            detail=str(error),
        ) from error
    metric_report_path = _validate_save_destination(replay_path)
    parent_descriptor = _open_parent_directory(
        replay_path,
        error_type=ReplaySaveError,
    )
    try:
        if (
            _entry_status_at(
                parent_descriptor,
                replay_path.name,
                path=replay_path,
                error_type=ReplaySaveError,
            )
            is not None
        ):
            raise ReplaySaveError(
                "replay_target_exists",
                path=replay_path,
                detail="replay destinations are never overwritten",
            )
        metric_status = _entry_status_at(
            parent_descriptor,
            metric_report_path.name,
            path=metric_report_path,
            error_type=ReplaySaveError,
        )
        if metric_status is not None and not stat.S_ISREG(metric_status.st_mode):
            raise ReplaySaveError(
                "metric_report_conflict",
                path=metric_report_path,
                detail="existing metric sidecar is not a regular file",
            )
    finally:
        os.close(parent_descriptor)
    return ReplayBundleDestinationV1(
        replay_path=replay_path,
        metric_report_path=metric_report_path,
    )


def _fsync_directory(parent_descriptor: int) -> None:
    os.fsync(parent_descriptor)


def _rollback_published_link(
    parent_descriptor: int,
    target_name: str,
    temporary_name: str,
) -> None:
    """Remove only the target hard link created from this temporary inode."""
    try:
        target_stat = os.stat(
            target_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        temporary_stat = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError:
        return
    if (target_stat.st_dev, target_stat.st_ino) != (
        temporary_stat.st_dev,
        temporary_stat.st_ino,
    ):
        return
    with suppress(OSError):
        os.unlink(target_name, dir_fd=parent_descriptor)
    with suppress(OSError):
        _fsync_directory(parent_descriptor)


def _create_temporary_file_at(parent_descriptor: int) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        temporary_name = f".marlbg-artifact.{token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        return descriptor, temporary_name
    raise OSError("could not allocate a unique replay temporary name")


def _publish_bytes_no_clobber(
    target: Path,
    payload: bytes,
    *,
    existing_code: ReplayIOErrorCodeV1,
    parent_descriptor: int | None = None,
) -> None:
    owned_parent_descriptor = parent_descriptor is None
    if parent_descriptor is None:
        parent_descriptor = _open_parent_directory(
            target,
            error_type=ReplaySaveError,
        )
    temporary_name: str | None = None
    write_completed = False
    target_linked = False
    try:
        descriptor, temporary_name = _create_temporary_file_at(
            parent_descriptor,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            write_completed = True
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            raise
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ReplaySaveError(
                existing_code,
                path=target,
                detail="destination already exists",
            ) from error
        target_linked = True
        _fsync_directory(parent_descriptor)
    except ReplaySaveError:
        raise
    except OSError as error:
        if target_linked and temporary_name is not None:
            _rollback_published_link(
                parent_descriptor,
                target.name,
                temporary_name,
            )
        code: ReplayIOErrorCodeV1 = (
            "temporary_write_failed" if not write_completed else "atomic_publish_failed"
        )
        raise ReplaySaveError(
            code,
            path=target,
            detail="bundle member could not be atomically published",
        ) from error
    finally:
        if temporary_name is not None:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
        if owned_parent_descriptor:
            os.close(parent_descriptor)


def _publish_metric_report(
    path: Path,
    payload: bytes,
    *,
    max_file_size_bytes: int,
    parent_descriptor: int | None = None,
) -> bool:
    owned_parent_descriptor = parent_descriptor is None
    if parent_descriptor is None:
        parent_descriptor = _open_parent_directory(
            path,
            error_type=ReplaySaveError,
        )
    try:
        existing_status = _entry_status_at(
            parent_descriptor,
            path.name,
            path=path,
            error_type=ReplaySaveError,
        )
        if existing_status is not None:
            try:
                existing = _read_bounded_regular_file_at(
                    parent_descriptor,
                    path.name,
                    path=path,
                    max_file_size_bytes=max_file_size_bytes,
                    error_type=ReplaySaveError,
                    fsync_before_close=True,
                )
            except ReplaySaveError as error:
                if error.code == "atomic_publish_failed":
                    raise
                raise ReplaySaveError(
                    "metric_report_conflict",
                    path=path,
                    detail="existing metric sidecar is not an identical regular file",
                ) from error
            if existing != payload:
                raise ReplaySaveError(
                    "metric_report_conflict",
                    path=path,
                    detail="existing metric sidecar bytes differ",
                )
            try:
                _fsync_directory(parent_descriptor)
            except OSError as error:
                raise ReplaySaveError(
                    "atomic_publish_failed",
                    path=path,
                    detail="existing metric sidecar directory is not durable",
                ) from error
            return True
        try:
            _publish_bytes_no_clobber(
                path,
                payload,
                existing_code="metric_report_conflict",
                parent_descriptor=parent_descriptor,
            )
            return False
        except ReplaySaveError as error:
            if error.code != "metric_report_conflict":
                raise
            try:
                existing = _read_bounded_regular_file_at(
                    parent_descriptor,
                    path.name,
                    path=path,
                    max_file_size_bytes=max_file_size_bytes,
                    error_type=ReplaySaveError,
                    fsync_before_close=True,
                )
            except ReplaySaveError as read_error:
                if read_error.code == "atomic_publish_failed":
                    raise
                raise ReplaySaveError(
                    "metric_report_conflict",
                    path=path,
                    detail="racing metric sidecar is not an identical regular file",
                ) from read_error
            if existing != payload:
                raise
            try:
                _fsync_directory(parent_descriptor)
            except OSError as directory_error:
                raise ReplaySaveError(
                    "atomic_publish_failed",
                    path=path,
                    detail="racing metric sidecar directory is not durable",
                ) from directory_error
            return True
    finally:
        if owned_parent_descriptor:
            os.close(parent_descriptor)


def _verify_prepared_replay_bundle_at(
    prepared: PreparedReplayBundleV1,
    destination: ReplayBundleDestinationV1,
    *,
    parent_descriptor: int,
) -> None:
    """Require both published files to equal the cached canonical bytes."""
    try:
        replay_payload = _read_bounded_regular_file_at(
            parent_descriptor,
            destination.replay_path.name,
            path=destination.replay_path,
            max_file_size_bytes=prepared.max_file_size_bytes,
            error_type=ReplaySaveError,
            fsync_before_close=True,
        )
        metric_report_payload = _read_bounded_regular_file_at(
            parent_descriptor,
            destination.metric_report_path.name,
            path=destination.metric_report_path,
            max_file_size_bytes=prepared.max_file_size_bytes,
            error_type=ReplaySaveError,
            fsync_before_close=True,
        )
        if replay_payload != prepared.replay_json_bytes:
            raise ReplaySaveError(
                "replay_publication_verification_failed",
                path=destination.replay_path,
                detail="published replay bytes differ from the prepared bytes",
            )
        if metric_report_payload != prepared.metric_report_json_bytes:
            raise ReplaySaveError(
                "replay_publication_verification_failed",
                path=destination.metric_report_path,
                detail="published metric-report bytes differ from the prepared bytes",
            )
        _fsync_directory(parent_descriptor)
    except ReplaySaveError as error:
        if error.code == "replay_publication_verification_failed":
            raise
        raise ReplaySaveError(
            "replay_publication_verification_failed",
            path=error.path,
            detail="published replay bundle could not be verified",
        ) from error
    except OSError as error:
        raise ReplaySaveError(
            "replay_publication_verification_failed",
            path=destination.replay_path,
            detail="published replay bundle durability could not be verified",
        ) from error


def publish_prepared_replay_bundle_v1(
    prepared: PreparedReplayBundleV1,
    destination: ReplayBundleDestinationV1,
    *,
    verify_existing_replay: bool = False,
) -> SavedReplayBundleV1:
    """Publish cached bytes, or explicitly verify one prior uncertain publish."""
    if type(prepared) is not PreparedReplayBundleV1:
        raise ReplaySaveError(
            "invalid_argument",
            path=None,
            detail="publication requires the exact PreparedReplayBundleV1 type",
        )
    if type(destination) is not ReplayBundleDestinationV1:
        raise ReplaySaveError(
            "invalid_argument",
            path=None,
            detail="publication requires the exact ReplayBundleDestinationV1 type",
        )
    if type(verify_existing_replay) is not bool:
        raise ReplaySaveError(
            "invalid_argument",
            path=destination.replay_path,
            detail="verify_existing_replay must be a boolean",
        )

    parent_descriptor = _open_parent_directory(
        destination.replay_path,
        error_type=ReplaySaveError,
    )
    try:
        if verify_existing_replay:
            _verify_prepared_replay_bundle_at(
                prepared,
                destination,
                parent_descriptor=parent_descriptor,
            )
            metric_report_reused = True
        else:
            if (
                _entry_status_at(
                    parent_descriptor,
                    destination.replay_path.name,
                    path=destination.replay_path,
                    error_type=ReplaySaveError,
                )
                is not None
            ):
                raise ReplaySaveError(
                    "replay_target_exists",
                    path=destination.replay_path,
                    detail="replay destinations are never overwritten",
                )
            metric_report_reused = _publish_metric_report(
                destination.metric_report_path,
                prepared.metric_report_json_bytes,
                max_file_size_bytes=prepared.max_file_size_bytes,
                parent_descriptor=parent_descriptor,
            )
            _publish_bytes_no_clobber(
                destination.replay_path,
                prepared.replay_json_bytes,
                existing_code="replay_target_exists",
                parent_descriptor=parent_descriptor,
            )
            _verify_prepared_replay_bundle_at(
                prepared,
                destination,
                parent_descriptor=parent_descriptor,
            )
    finally:
        os.close(parent_descriptor)

    return SavedReplayBundleV1(
        replay_path=destination.replay_path,
        metric_report_path=destination.metric_report_path,
        replay_byte_length=prepared.replay_byte_length,
        metric_report_byte_length=prepared.metric_report_byte_length,
        metric_report_reused=metric_report_reused,
    )


def save_replay_bundle_v1(
    bundle: ReplayBundleV1,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedReplayBundleV1:
    """Publish metric bytes first and the referencing replay last, without overwrite."""
    prepared = prepare_replay_bundle_v1(
        bundle,
        max_file_size_bytes=max_file_size_bytes,
    )
    destination = preflight_replay_bundle_destination_v1(path)
    return publish_prepared_replay_bundle_v1(
        prepared,
        destination,
    )


def _save_companion_payload(
    path: Path,
    payload: bytes,
    *,
    max_file_size_bytes: int,
) -> SavedCompanionArtifactV1:
    if len(payload) > max_file_size_bytes:
        raise ReplaySaveError(
            "file_too_large",
            path=path,
            detail=f"companion artifact exceeds {max_file_size_bytes} bytes",
        )
    parent_descriptor = _open_parent_directory(
        path,
        error_type=ReplaySaveError,
    )
    try:
        if (
            _entry_status_at(
                parent_descriptor,
                path.name,
                path=path,
                error_type=ReplaySaveError,
            )
            is not None
        ):
            raise ReplaySaveError(
                "companion_target_exists",
                path=path,
                detail="companion artifacts are never overwritten",
            )
        _publish_bytes_no_clobber(
            path,
            payload,
            existing_code="companion_target_exists",
            parent_descriptor=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    return SavedCompanionArtifactV1(path=path, byte_length=len(payload))


def save_actor_pov_replay_artifact_v1(
    artifact: ActorPovReplayArtifactV1,
    source_replay: ReplayArtifactV1,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate, source-join, and publish one canonical actor-POV artifact."""
    try:
        pov_path = _coerce_path(path)
        size_limit = _require_positive_limit(
            max_file_size_bytes,
            name="max_file_size_bytes",
        )
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=None,
            detail=str(error),
        ) from error
    try:
        _require_artifact_suffix(
            pov_path,
            suffix=ACTOR_POV_FILE_SUFFIX_V1,
            label="actor POV",
        )
    except ValueError as error:
        raise ReplaySaveError(
            "invalid_filename",
            path=pov_path,
            detail=str(error),
        ) from error
    try:
        validate_actor_pov_replay_against_replay_v1(artifact, source_replay)
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=pov_path,
            detail="actor POV does not match its supplied source replay",
        ) from error
    return _save_companion_payload(
        pov_path,
        canonical_json_bytes(artifact),
        max_file_size_bytes=size_limit,
    )


def save_scenario_evaluation_record_v1(
    record: ScenarioEvaluationRecordV1,
    source_replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate both evidence joins and publish one canonical scenario record."""
    try:
        scenario_path = _coerce_path(path)
        size_limit = _require_positive_limit(
            max_file_size_bytes,
            name="max_file_size_bytes",
        )
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=None,
            detail=str(error),
        ) from error
    try:
        _require_artifact_suffix(
            scenario_path,
            suffix=SCENARIO_FILE_SUFFIX_V1,
            label="scenario",
        )
    except ValueError as error:
        raise ReplaySaveError(
            "invalid_filename",
            path=scenario_path,
            detail=str(error),
        ) from error
    try:
        validate_scenario_evaluation_record_v1(
            record,
            source_replay,
            metric_report_artifact,
        )
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=scenario_path,
            detail="scenario record does not match its supplied replay/report",
        ) from error
    return _save_companion_payload(
        scenario_path,
        canonical_json_bytes(record),
        max_file_size_bytes=size_limit,
    )


__all__ = [
    "ACTOR_POV_FILE_SUFFIX_V1",
    "DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1",
    "DEFAULT_MAX_REPLAY_JSON_DEPTH_V1",
    "METRIC_REPORT_FILE_SUFFIX_V1",
    "REPLAY_FILE_SUFFIX_V1",
    "SCENARIO_FILE_SUFFIX_V1",
    "LoadedReplayBundleV1",
    "PreparedReplayBundleV1",
    "ReplayBundleDestinationV1",
    "ReplayBundleLoadStatusV1",
    "ReplayIOError",
    "ReplayIOErrorCodeV1",
    "ReplayLoadError",
    "ReplaySaveError",
    "SavedCompanionArtifactV1",
    "SavedReplayBundleV1",
    "canonical_metric_report_artifact_json_bytes_v1",
    "canonical_replay_json_bytes_v1",
    "canonical_scenario_evaluation_record_json_bytes_v1",
    "load_actor_pov_replay_artifact_v1",
    "load_replay_artifact_v1",
    "load_replay_bundle_v1",
    "load_scenario_evaluation_record_v1",
    "preflight_replay_bundle_destination_v1",
    "prepare_replay_bundle_v1",
    "publish_prepared_replay_bundle_v1",
    "save_actor_pov_replay_artifact_v1",
    "save_replay_bundle_v1",
    "save_scenario_evaluation_record_v1",
]
