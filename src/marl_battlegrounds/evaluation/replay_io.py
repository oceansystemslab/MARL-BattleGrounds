"""Load and publish canonical local replay, scenario, and actor-view artifacts.

Readers require bounded regular files, strict UTF-8 without a BOM, finite JSON
with unique keys, exact schema versions, and canonical bytes. Paths are opened
through POSIX directory descriptors without following symlinks. Parent
directories must already exist; this module does not create them.

Replay V2-V4 are single files; V4 records the Team Deathmatch Red Zone rule
and older versions keep their original meaning. Legacy V1 bundles publish a
separate metric sidecar before the replay that references it. Publication
never overwrites an existing target; retries can explicitly verify previously
published cached bytes. Host validation and filesystem durability live here.
No simulator rollout, policy execution, device initialization, network, or
archive extraction occurs.
"""

from __future__ import annotations

import json
import os
import re
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
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    EvaluationEpisodeContext,
    EvaluationFrame,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import (
    ACTOR_POV_ARTIFACT_SCHEMA_ID,
    ActorPovReplayArtifactV1,
    ActorPovReplayArtifactV2,
    ActorPovReplayArtifactV3,
    validate_actor_pov_replay_against_replay_v1,
    validate_actor_pov_replay_against_replay_v2,
    validate_actor_pov_replay_against_replay_v3,
    validate_actor_pov_replay_artifact_v1,
    validate_actor_pov_replay_artifact_v2,
    validate_actor_pov_replay_artifact_v3,
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
from marl_battlegrounds.evaluation.replay_v2 import ReplayArtifactV2
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4
from marl_battlegrounds.evaluation.scenario import (
    SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
    SCENARIO_SCHEMA_VERSION_V2,
    ScenarioEvaluationRecordV1,
    ScenarioEvaluationRecordV2,
    ScenarioEvaluationRecordV3,
    ScenarioEvaluationRecordV4,
    ScenarioEvaluationRecordV5,
    validate_scenario_evaluation_record_v1,
    validate_scenario_evaluation_record_v2,
    validate_scenario_evaluation_record_v3,
    validate_scenario_evaluation_record_v4,
    validate_scenario_evaluation_record_v5,
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
    """Expose a stable error code, affected path, and readable I/O explanation.

    Attributes
    ----------
    code : ReplayIOErrorCodeV1
        Machine-readable ReplayIOErrorCodeV1 identifying the failure category.
    path : Path | None
        Affected local Path, or None when argument validation had no usable path.
    detail : str
        Human-readable explanation.

    Notes
    -----
    The exception message combines code, optional path, and detail. Callers can
    branch on code without parsing prose.
    """

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
        """Create a structured artifact I/O error.

        Parameters
        ----------
        code : ReplayIOErrorCodeV1
            Stable failure category.
        path : Path | None
            Affected local path, or None.
        detail : str
            Explanation included in the exception message.

        Returns
        -------
        None
            None.

        Notes
        -----
        Stores supplied values and initializes Exception; it performs no file access.
        """
        self.code = code
        self.path = path
        self.detail = detail
        location = "" if path is None else f" ({path})"
        super().__init__(f"{code}{location}: {detail}")


class ReplayLoadError(ReplayIOError):
    """Report an unsafe, missing, malformed, or inconsistent artifact load.

    Inherits code, path, and detail from ReplayIOError. A missing optional legacy
    sidecar is represented in the load result instead when permitted.
    """


class ReplaySaveError(ReplayIOError):
    """Report failed artifact preparation, destination admission, or publication.

    Inherits code, path, and detail from ReplayIOError. A verification failure can
    mean bytes were published but final durability could not be confirmed; retry
    with the prepared object and explicit verification when appropriate.
    """


@dataclass(frozen=True, slots=True)
class LoadedReplayBundleV1:
    """Keep a loaded historical replay and its optional metric sidecar.

    Attributes
    ----------
    replay : ReplayArtifactV1
        Validated ReplayArtifactV1.
    metric_report_artifact : EvaluationMetricReportArtifactV1 | None
        Matching V1 metric artifact, or None when absent.
    status : ReplayBundleLoadStatusV1
        complete when sidecar is loaded; metric_report_missing when absent.

    Notes
    -----
    Availability status describes the sidecar, not whether the simulated episode is
    complete.
    """

    replay: ReplayArtifactV1
    metric_report_artifact: EvaluationMetricReportArtifactV1 | None
    status: ReplayBundleLoadStatusV1

    def frame_at(self, frame_index: int) -> EvaluationFrameV1:
        """Return one captured frame by artifact index.

        Parameters
        ----------
        frame_index : int
            Exact Python int from zero through the final recorded frame.

        Returns
        -------
        EvaluationFrameV1
            The stored immutable frame, without a copy or simulator work.

        Raises
        ------
        TypeError
            The index is not an exact int, including bool.
        IndexError
            The index is negative or beyond the captured prefix.

        Notes
        -----
        Tuple lookup takes constant time; artifact index is not simulator tick.
        """
        if type(frame_index) is not int:
            raise TypeError("frame index must be an integer")
        if frame_index < 0 or frame_index >= len(self.replay.frames):
            raise IndexError("frame index is outside the captured replay prefix")
        return self.replay.frames[frame_index]

    def incoming_transition_at(
        self,
        frame_index: int,
    ) -> EvaluationTransitionV1 | None:
        """Return the transition that produced one captured frame.

        Parameters
        ----------
        frame_index : int
            Exact Python int within the recorded frame range.

        Returns
        -------
        EvaluationTransitionV1 | None
            None at frame zero; otherwise the transition at frame_index minus one.

        Raises
        ------
        TypeError
            The index is not an exact int.
        IndexError
            The index is outside the captured prefix.

        Notes
        -----
        Reuses stored records without copying or simulation.
        """
        self.frame_at(frame_index)
        if frame_index == 0:
            return None
        return self.replay.transitions[frame_index - 1]


@dataclass(frozen=True, slots=True)
class LoadedReplay:
    """Keep an exact supported replay and honest legacy metric availability.

    Attributes
    ----------
    replay : ReplayArtifactV1 | ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4
        ReplayArtifactV1, V2, V3 or V4 (V4 records the Red Zone rule).
    metric_report_artifact : EvaluationMetricReportArtifactV1 | None
        Optional V1 sidecar; defaults to None and is forbidden for V2-V4.
    status : Literal['complete', 'metric_report_missing', 'not_recorded']
        not_recorded for V2-V4 (default); complete or metric_report_missing for V1.

    Notes
    -----
    The loader validates content. Direct dataclass construction checks version/status
    pairing only.
    """

    replay: ReplayArtifactV1 | ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4
    metric_report_artifact: EvaluationMetricReportArtifactV1 | None = None
    status: Literal["complete", "metric_report_missing", "not_recorded"] = (
        "not_recorded"
    )

    def __post_init__(self) -> None:
        """Require exact replay roots and a sidecar status consistent with that version.

        V2-V4 forbid legacy sidecars and require not_recorded. V1 uses complete only
        when
        a sidecar is present. Raise TypeError or ValueError; do not load or validate
        files.
        """
        if type(self.replay) not in (
            ReplayArtifactV1,
            ReplayArtifactV2,
            ReplayArtifactV3,
            ReplayArtifactV4,
        ):
            raise TypeError("loaded replay requires an exact supported artifact")
        if type(self.replay) in (ReplayArtifactV2, ReplayArtifactV3, ReplayArtifactV4):
            if self.status != "not_recorded" or self.metric_report_artifact is not None:
                raise ValueError(
                    "Self-contained replay has no legacy metric-report sidecar"
                )
        elif self.status != (
            "complete"
            if self.metric_report_artifact is not None
            else "metric_report_missing"
        ):
            raise ValueError("legacy replay availability must match its loaded sidecar")

    def frame_at(self, frame_index: int) -> EvaluationFrame:
        """Return one captured frame by artifact index.

        Parameters
        ----------
        frame_index : int
            Exact Python int from zero through the final recorded frame.

        Returns
        -------
        EvaluationFrame
            The stored immutable frame, without a copy or simulator work.

        Raises
        ------
        TypeError
            The index is not an exact int, including bool.
        IndexError
            The index is negative or beyond the captured prefix.

        Notes
        -----
        Tuple lookup takes constant time; artifact index is not simulator tick.
        """
        if type(frame_index) is not int:
            raise TypeError("frame index must be an integer")
        if not 0 <= frame_index < len(self.replay.frames):
            raise IndexError("frame index is outside the captured replay prefix")
        return self.replay.frames[frame_index]

    def incoming_transition_at(self, frame_index: int) -> EvaluationTransitionV1 | None:
        """Return the transition that produced one captured frame.

        Parameters
        ----------
        frame_index : int
            Exact Python int within the recorded frame range.

        Returns
        -------
        EvaluationTransitionV1 | None
            None at frame zero; otherwise the transition at frame_index minus one.

        Raises
        ------
        TypeError
            The index is not an exact int.
        IndexError
            The index is outside the captured prefix.

        Notes
        -----
        Reuses stored records without copying or simulation.
        """
        self.frame_at(frame_index)
        return None if frame_index == 0 else self.replay.transitions[frame_index - 1]


type LoadedReplayBundle = LoadedReplayBundleV1 | LoadedReplay


@dataclass(frozen=True, slots=True)
class SavedReplay:
    """Report successful publication of one self-contained replay.

    Attributes
    ----------
    replay_path : Path
        Selected local output Path.
    replay_byte_length : int
        Canonical bytes written or verified.

    Notes
    -----
    These local paths never enter scientific artifact content.
    """

    replay_path: Path
    replay_byte_length: int


@dataclass(frozen=True, slots=True)
class ReplayDestination:
    """Keep the selected single-file replay destination for later publication.

    Attributes
    ----------
    replay_path : Path
        Path ending in a nonempty .marlbg-replay.json stem.

    Notes
    -----
    Use preflight_replay_destination to check the filesystem. This value reserves
    nothing; publication must still handle a racing writer.
    """

    replay_path: Path

    def __post_init__(self) -> None:
        """Require a Path and a nonempty .marlbg-replay.json filename stem.

        Direct construction checks structure only. Filesystem absence and parent safety
        are checked by preflight_replay_destination and again during publication.
        """
        if not isinstance(self.replay_path, Path):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("replay destination must use pathlib.Path")
        _metric_report_path_for_replay(self.replay_path)


@dataclass(frozen=True, slots=True)
class PreparedReplay:
    """Cache validated self-contained replay bytes for publication and retries.

    Attributes
    ----------
    replay : ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4
        Exact ReplayArtifactV2, V3 or V4.
    max_file_size_bytes : int
        Positive per-file cap, default 1 GiB (1024**3 bytes).
    replay_json_bytes : bytes
        Derived immutable canonical UTF-8 bytes, not a constructor argument.
    replay_payload_sha256 : str
        Derived SHA-256 of those full bytes, not a constructor argument.

    Notes
    -----
    Preparation validates and serializes on the host without writing files. The payload
    hash differs in purpose from the artifact's digest field, which excludes itself.
    """

    replay: ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1
    replay_json_bytes: bytes = field(init=False, repr=False)
    replay_payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate exact V2-V4 replay content and cache canonical bytes within the
        limit.

        Reject invalid limits/types/models before publishing anything. Oversized bytes
        raise ReplaySaveError with file_too_large.
        """
        size = _require_positive_limit(
            self.max_file_size_bytes, name="max_file_size_bytes"
        )
        if type(self.replay) not in (
            ReplayArtifactV2,
            ReplayArtifactV3,
            ReplayArtifactV4,
        ):
            raise TypeError("prepared replay requires an exact supported artifact")
        canonical = validate_declared_model_tree(
            self.replay, record_name="replay", expected_type=type(self.replay)
        )
        self._set_payload(canonical, size)

    @classmethod
    def _from_capture(
        cls, replay: ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4
    ) -> PreparedReplay:
        """Prepare freshly validated collector output without a duplicate model-tree
        pass.

        Only immediate capture-to-publication uses this trusted path. Require exact
        replay V2-V4, use the default 1 GiB limit, and cache canonical bytes. Arbitrary
        external models must use the normal strict constructor.
        """
        if type(replay) not in (ReplayArtifactV2, ReplayArtifactV3, ReplayArtifactV4):
            raise TypeError(
                "capture publication requires an exact self-contained replay"
            )
        prepared = object.__new__(cls)
        object.__setattr__(prepared, "replay", replay)
        object.__setattr__(
            prepared, "max_file_size_bytes", DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1
        )
        prepared._set_payload(replay, DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1)
        return prepared

    def _set_payload(self, canonical: EvaluationModel, size: int) -> None:
        """Cache canonical bytes and their payload SHA-256 after enforcing the byte cap.

        The canonical model is already validated by the caller. Mutate only frozen
        derived fields through object.__setattr__; oversize raises ReplaySaveError.
        """
        payload = canonical_json_bytes(canonical)
        if len(payload) > size:
            raise ReplaySaveError(
                "file_too_large", path=None, detail="replay exceeds byte limit"
            )
        object.__setattr__(self, "replay_json_bytes", payload)
        object.__setattr__(self, "replay_payload_sha256", sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True)
class SavedReplayBundleV1:
    """Report publication facts for a legacy replay/report file pair.

    Attributes
    ----------
    replay_path : Path
        Local replay Path.
    metric_report_path : Path
        Derived adjacent metric sidecar Path.
    replay_byte_length : int
        Canonical replay byte count.
    metric_report_byte_length : int
        Canonical sidecar byte count.
    metric_report_reused : bool
        True when an identical existing sidecar was reused or verified.

    Notes
    -----
    The two-file operation publishes the sidecar first; these paths remain outside
    artifact content.
    """

    replay_path: Path
    metric_report_path: Path
    replay_byte_length: int
    metric_report_byte_length: int
    metric_report_reused: bool


@dataclass(frozen=True, slots=True)
class ReplayBundleDestinationV1:
    """Keep a legacy replay filename and its derived adjacent sidecar path.

    Attributes
    ----------
    replay_path : Path
        Nonempty .marlbg-replay.json Path.
    metric_report_path : Path
        Same stem with .marlbg-metrics.json.

    Notes
    -----
    Preflight checks existing parent and replay absence without reserving either name.
    """

    replay_path: Path
    metric_report_path: Path

    def __post_init__(self) -> None:
        """Require Path objects and a metric filename derived from the replay stem.

        Raise TypeError or ValueError on structural mismatch. Direct construction does
        not establish that either filesystem entry is safe or absent.
        """
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.replay_path, Path
        ) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.metric_report_path, Path
        ):
            raise TypeError("replay bundle destination paths must use pathlib.Path")
        if _metric_report_path_for_replay(self.replay_path) != self.metric_report_path:
            raise ValueError("metric report path must derive from the replay filename")


class _PreparedReplayBundleTooLargeError(ValueError):
    """Mark a prepared legacy bundle member exceeding its per-file byte limit.

    The public preparation helper translates this internal ValueError into a
    ReplaySaveError with file_too_large.
    """

    pass


@dataclass(frozen=True, slots=True)
class PreparedReplayBundleV1:
    """Cache a validated historical bundle's two canonical byte strings.

    Attributes
    ----------
    bundle : ReplayBundleV1
        Exact ReplayBundleV1 with matching report sidecar.
    max_file_size_bytes : int
        Positive cap applied separately to each member; default 1 GiB.
    replay_json_bytes : bytes
        Derived immutable replay bytes.
    metric_report_json_bytes : bytes
        Derived immutable sidecar bytes.
    replay_byte_length : int
        Derived replay byte count.
    metric_report_byte_length : int
        Derived sidecar byte count.
    replay_payload_sha256 : str
        Derived SHA-256 of the full replay bytes.
    metric_report_payload_sha256 : str
        Derived SHA-256 of the full sidecar bytes.

    Notes
    -----
    Only bundle and limit are constructor arguments. Public prepare_replay_bundle_v1
    translates validation/size failures into stable save errors.
    """

    bundle: ReplayBundleV1
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1
    replay_json_bytes: bytes = field(init=False, repr=False)
    metric_report_json_bytes: bytes = field(init=False, repr=False)
    replay_byte_length: int = field(init=False)
    metric_report_byte_length: int = field(init=False)
    replay_payload_sha256: str = field(init=False)
    metric_report_payload_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate a V1 replay/report pair and cache both canonical byte strings once.

        Require exact bundle type, positive per-member size limit, full replay
        semantics,
        and matching report joins. Retain lengths and payload hashes for retry checks.
        Oversize raises the internal size-limit ValueError; no files are written.
        """
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
    """Report publication of one actor-view or scenario companion.

    Attributes
    ----------
    path : Path
        Selected local output Path.
    byte_length : int
        Canonical payload byte count.
    """

    path: Path
    byte_length: int


class _DuplicateKeyError(ValueError):
    """Signal a repeated JSON object key before model validation can hide it."""

    pass


class _NonFiniteNumberError(ValueError):
    """Signal NaN, infinity, or numeric overflow in a JSON number."""

    pass


def _require_positive_limit(value: int, *, name: str) -> int:
    """Return an exact positive Python int or raise ValueError naming the limit.

    Boolean and coercible numeric values are not accepted.
    """
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coerce_path(path: str | os.PathLike[str]) -> Path:
    """Convert a nonempty string or supported path-like object to Path.

    Do not resolve symlinks or create directories. Raise ValueError for empty strings
    and TypeError for unsupported path inputs.
    """
    if isinstance(path, str):
        if not path:
            raise ValueError("path must not be empty")
        return Path(path)
    try:
        return Path(path)
    except TypeError as error:
        raise TypeError("path must be a string or path-like value") from error


def _metric_report_path_for_replay(replay_path: Path) -> Path:
    """Derive a sibling .marlbg-metrics.json path from a valid replay filename.

    Require a nonempty stem before .marlbg-replay.json or raise ValueError.
    This naming helper does not inspect either filesystem entry.
    """
    name = replay_path.name
    if not name.endswith(REPLAY_FILE_SUFFIX_V1):
        raise ValueError(f"replay filename must end with {REPLAY_FILE_SUFFIX_V1}")
    base_name = name[: -len(REPLAY_FILE_SUFFIX_V1)]
    if not base_name:
        raise ValueError("replay filename must include an episode stem")
    return replay_path.with_name(f"{base_name}{METRIC_REPORT_FILE_SUFFIX_V1}")


def _require_artifact_suffix(path: Path, *, suffix: str, label: str) -> None:
    """Require a named artifact's exact suffix and nonempty stem.

    Raise ValueError using the label when the basename is invalid.
    """
    if not path.name.endswith(suffix) or path.name == suffix:
        raise ValueError(f"{label} filename must end with {suffix}")


def _require_secure_directory_fd_support(
    *,
    path: Path,
    error_type: type[ReplayLoadError] | type[ReplaySaveError],
) -> None:
    """Require POSIX no-follow and directory-descriptor filesystem operations.

    Raise the supplied load/save error type with unsupported_platform when the
    required open/stat/link/unlink support is missing.
    """
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
    """Open each existing parent directory without following symlinks.

    Return an owned descriptor for the final parent; the caller must close it.
    Close intermediate descriptors and all owned descriptors on failure. Raise the
    supplied load/save error type for unsafe or missing path components. No parent
    directory is created or retained by path resolution alone.
    """
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
    """Read at most the limit plus one byte through an owned parent descriptor.

    Reject symlinks, nonregular files, and files exceeding the cap, including growth
    after stat. Close the file but leave parent_descriptor owned by the caller.
    fsync_before_close optionally makes existing bytes durable for save retries;
    it defaults to false. Translate failures to the selected load/save error type.
    """
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
    """Open a secure parent, read one bounded regular file, and close the parent.

    Delegate entry validation and byte limits to the descriptor-based reader.
    No path component or final file is followed through a symlink.
    """
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
    """Reject object/array nesting beyond the declared depth before JSON parsing.

    Ignore brackets inside quoted strings and account for escapes. This scanner is
    only a depth guard; json.loads later checks syntax and balanced delimiters.
    """
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
    """Build a JSON object while rejecting duplicate keys before they can overwrite.

    Raise _DuplicateKeyError on the first duplicate and preserve each accepted value.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    """Reject JSON's nonstandard NaN/Infinity token through an internal typed error."""
    raise _NonFiniteNumberError(value)


def _parse_finite_float(value: str) -> float:
    """Parse a JSON floating literal and reject overflow to infinity.

    Return a finite Python float or raise _NonFiniteNumberError.
    """
    parsed = float(value)
    if not isfinite(parsed):
        raise _NonFiniteNumberError(value)
    return parsed


def _preflight_json(
    payload: bytes,
    *,
    path: Path,
    expected_schema_id: str,
    expected_schema_version: int | tuple[int, ...],
    max_json_depth: int,
) -> dict[str, object]:
    """Check UTF-8, depth, unique keys, finite numbers, and exact root schema/version.

    Return the parsed root dictionary. Reject BOMs, trailing malformed content,
    wrong root type, and bool/coerced schema versions with stable ReplayLoadError
    codes. This precedes strict model parsing and canonical-byte comparison.
    """
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
    versions = (
        (expected_schema_version,)
        if isinstance(expected_schema_version, int)
        else expected_schema_version
    )
    if type(schema_version) is not int or schema_version not in versions:
        raise ReplayLoadError(
            "unsupported_schema_version",
            path=path,
            detail=(
                f"only exact schema version {expected_schema_version} is supported"
            ),
        )
    return root


def _load_replay_bytes(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
) -> ReplayArtifactV1:
    """Admit canonical V1 replay bytes with full trajectory semantic validation.

    Check JSON limits/schema, strict model fields, O(T) replay joins, and exact
    canonical re-encoding. Raise ReplayLoadError with the owning failure stage.
    """
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=REPLAY_ARTIFACT_SCHEMA_ID,
        expected_schema_version=REPLAY_SCHEMA_VERSION,
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
    """Admit canonical V1 metric-artifact bytes without loading their replay.

    Check JSON/schema, exact model-tree validity, and byte equality. Actual trajectory
    and report-reference joins are checked after both artifacts are available.
    """
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=METRIC_REPORT_ARTIFACT_SCHEMA_ID,
        expected_schema_version=REPLAY_SCHEMA_VERSION,
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
    pov_version: Literal[1, 2, 3],
) -> ActorPovReplayArtifactV1 | ActorPovReplayArtifactV2 | ActorPovReplayArtifactV3:
    """Admit exact canonical POV bytes of the declared pov_version (1, 2 or 3).

    Check JSON/schema, complete standalone POV validity, and canonical byte equality.
    Bytes of any other POV version fail with unsupported_schema_version. The
    separate source-replay join is optional at the public load boundary.
    """
    model = (
        ActorPovReplayArtifactV3
        if pov_version == 3
        else ActorPovReplayArtifactV2
        if pov_version == 2
        else ActorPovReplayArtifactV1
    )
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=ACTOR_POV_ARTIFACT_SCHEMA_ID,
        expected_schema_version=pov_version,
        max_json_depth=max_json_depth,
    )
    try:
        artifact = model.model_validate_json(payload)
        if type(artifact) is ActorPovReplayArtifactV3:
            validate_actor_pov_replay_artifact_v3(artifact)
        elif type(artifact) is ActorPovReplayArtifactV2:
            validate_actor_pov_replay_artifact_v2(artifact)
        elif type(artifact) is ActorPovReplayArtifactV1:
            validate_actor_pov_replay_artifact_v1(artifact)
        else:
            raise TypeError("actor POV must use an exact supported root")
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
            detail="actor POV bytes are not the canonical declared-version encoding",
        )
    return artifact


def _load_scenario_record_bytes(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
) -> ScenarioEvaluationRecordV1:
    """Admit canonical scenario V1 bytes and strict local record structure.

    Do not infer endpoint truth or load referenced replay/report files.
    """
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        expected_schema_version=REPLAY_SCHEMA_VERSION,
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


def _load_scenario_record_bytes_v2(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
) -> ScenarioEvaluationRecordV2:
    """Admit canonical scenario V2 bytes and strict local record structure.

    Actual replay/report evidence joins are checked by the outer loader.
    """
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        expected_schema_version=SCENARIO_SCHEMA_VERSION_V2,
        max_json_depth=max_json_depth,
    )
    try:
        record = ScenarioEvaluationRecordV2.model_validate_json(payload)
        canonical_record = cast(
            ScenarioEvaluationRecordV2,
            validate_declared_model_tree(
                record,
                record_name="loaded V2 scenario evaluation record",
                expected_type=ScenarioEvaluationRecordV2,
            ),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ReplayLoadError(
            "model_validation_failed",
            path=path,
            detail="V2 scenario record does not satisfy its strict artifact contract",
        ) from error
    if canonical_json_bytes(canonical_record) != payload:
        raise ReplayLoadError(
            "noncanonical_json",
            path=path,
            detail="scenario bytes are not the canonical V2 encoding",
        )
    return canonical_record


def canonical_replay_json_bytes_v1(artifact: ReplayArtifactV1) -> bytes:
    """Validate and canonically encode one exact ReplayArtifactV1.

    Parameters
    ----------
    artifact : ReplayArtifactV1
        exact ReplayArtifactV1, including its complete trajectory.

    Returns
    -------
    bytes
        Compact sorted-key finite UTF-8 JSON bytes with normalized floating zeros.

    Raises
    ------
    ValueError
        Exact model revalidation or full replay semantics fails.

    Notes
    -----
    No file is written and no size cap is applied here. Replay validation is O(T).
    """
    validate_replay_artifact_v1(artifact)
    return canonical_json_bytes(artifact)


def canonical_metric_report_artifact_json_bytes_v1(
    artifact: EvaluationMetricReportArtifactV1,
) -> bytes:
    """Validate and canonically encode one exact EvaluationMetricReportArtifactV1.

    Parameters
    ----------
    artifact : EvaluationMetricReportArtifactV1
        exact EvaluationMetricReportArtifactV1.

    Returns
    -------
    bytes
        Compact sorted-key finite UTF-8 JSON bytes with normalized floating zeros.

    Raises
    ------
    ValueError
        Exact model revalidation fails.

    Notes
    -----
    No file is written and no size cap is applied here. Actual referenced evidence is
    not loaded or joined by this local encoder.
    """
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
    """Validate and canonically encode one exact ScenarioEvaluationRecordV1.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV1
        exact ScenarioEvaluationRecordV1.

    Returns
    -------
    bytes
        Compact sorted-key finite UTF-8 JSON bytes with normalized floating zeros.

    Raises
    ------
    ValueError
        Exact model revalidation fails.

    Notes
    -----
    No file is written and no size cap is applied here. Actual referenced evidence is
    not loaded or joined by this local encoder.
    """
    canonical_record = cast(
        ScenarioEvaluationRecordV1,
        validate_declared_model_tree(
            record,
            record_name="scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV1,
        ),
    )
    return canonical_json_bytes(canonical_record)


def canonical_scenario_evaluation_record_json_bytes_v2(
    record: ScenarioEvaluationRecordV2,
) -> bytes:
    """Validate and canonically encode one exact ScenarioEvaluationRecordV2.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV2
        exact ScenarioEvaluationRecordV2.

    Returns
    -------
    bytes
        Compact sorted-key finite UTF-8 JSON bytes with normalized floating zeros.

    Raises
    ------
    ValueError
        Exact model revalidation fails.

    Notes
    -----
    No file is written and no size cap is applied here. Actual referenced evidence is
    not loaded or joined by this local encoder.
    """
    canonical_record = cast(
        ScenarioEvaluationRecordV2,
        validate_declared_model_tree(
            record,
            record_name="V2 scenario evaluation record",
            expected_type=ScenarioEvaluationRecordV2,
        ),
    )
    return canonical_json_bytes(canonical_record)


def load_replay_artifact_v1(
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ReplayArtifactV1:
    """Load and fully validate one canonical historical replay file.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-replay.json stem.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ReplayArtifactV1
        ReplayArtifactV1 after strict bytes, model, and O(T) semantic validation.

    Raises
    ------
    ReplayLoadError
        Invalid arguments/path, unsupported platform, exceeded limits,
        malformed/noncanonical JSON, wrong version, or invalid replay semantics.

    Notes
    -----
    Rejects symlinks in parents and final entry. Does not load the adjacent metric
    sidecar; use load_replay_bundle_v1 when its evidence is also needed.
    """
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
    """Load a historical replay and its adjacent metric report when available.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular .marlbg-replay.json file.
    require_metric_report : bool
        Exact bool; if true, missing sidecar is an error.
        Defaults to false.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    LoadedReplayBundleV1
        LoadedReplayBundleV1 with status complete or metric_report_missing.

    Raises
    ------
    ReplayLoadError
        Replay/sidecar bytes, paths, limits, schemas, or joins are
        invalid, or a required metric report is missing.

    Notes
    -----
    Derives the sibling .marlbg-metrics.json path. Only absence is optional;
    an existing corrupt or mismatched sidecar always fails. Status describes
    sidecar availability, not episode termination. Reads both through one
    securely opened parent directory.
    """
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


def load_replay(
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> LoadedReplay:
    """Load a replay using its exact recorded schema version.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-replay.json stem.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    LoadedReplay
        LoadedReplay containing replay V1, V2, V3 or V4. V2-V4 use status
        not_recorded and no legacy sidecar. V1 resolves its sibling report or
        reports its absence. V4 records the Team Deathmatch Red Zone rule; older
        versions keep their original one-point meaning.

    Raises
    ------
    ReplayLoadError
        Unsafe path, invalid limits, unsupported schema, malformed
        or noncanonical bytes, invalid content, or mismatched legacy sidecar.

    Notes
    -----
    Host-only reader does not initialize JAX or rerun simulation. V2-V4 ignore
    nearby metric files because they are not part of those replay contracts.
    Existing invalid V1 sidecars fail rather than being treated as absent.
    """
    try:
        replay_path = _coerce_path(path)
        metric_path = _metric_report_path_for_replay(replay_path)
        size = _require_positive_limit(max_file_size_bytes, name="max_file_size_bytes")
        depth = _require_positive_limit(max_json_depth, name="max_json_depth")
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "invalid_argument", path=None, detail=str(error)
        ) from error
    parent = _open_parent_directory(replay_path, error_type=ReplayLoadError)
    try:
        payload = _read_bounded_regular_file_at(
            parent, replay_path.name, path=replay_path, max_file_size_bytes=size
        )
        root = _preflight_json(
            payload,
            path=replay_path,
            expected_schema_id=REPLAY_ARTIFACT_SCHEMA_ID,
            expected_schema_version=(1, 2, 3, 4),
            max_json_depth=depth,
        )
        if root["schema_version"] in (2, 3, 4):
            model: (
                type[ReplayArtifactV2] | type[ReplayArtifactV3] | type[ReplayArtifactV4]
            ) = {2: ReplayArtifactV2, 3: ReplayArtifactV3, 4: ReplayArtifactV4}[
                cast(int, root["schema_version"])
            ]
            try:
                replay = model.model_validate_json(payload)
            except (TypeError, ValueError) as error:
                raise ReplayLoadError(
                    "model_validation_failed",
                    path=replay_path,
                    detail="replay does not satisfy its declared version contract",
                ) from error
            if canonical_json_bytes(replay) != payload:
                raise ReplayLoadError(
                    "noncanonical_json",
                    path=replay_path,
                    detail="replay bytes are not canonical",
                )
            return LoadedReplay(replay)
        legacy = _load_replay_bytes(payload, path=replay_path, max_json_depth=depth)
        try:
            report_payload = _read_bounded_regular_file_at(
                parent, metric_path.name, path=metric_path, max_file_size_bytes=size
            )
        except ReplayLoadError as error:
            if error.code != "path_not_found":
                raise
            return LoadedReplay(legacy, status="metric_report_missing")
        report = _load_metric_report_bytes(
            report_payload, path=metric_path, max_json_depth=depth
        )
        try:
            _validate_metric_report_artifact_against_validated_replay_v1(report, legacy)
        except (TypeError, ValueError) as error:
            raise ReplayLoadError(
                "metric_report_mismatch",
                path=metric_path,
                detail="metric report does not join the loaded replay",
            ) from error
        return LoadedReplay(legacy, report, "complete")
    finally:
        os.close(parent)


def generated_replay_filename(
    context: EvaluationEpisodeContext, digest: str, *, episode_id: int
) -> str:
    """Build a readable, bounded filename without changing scientific identity.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid recorded context with map, policy, and seed identities.
    digest : str
        Full lowercase 64-character SHA-256 to include without truncation.
    episode_id : int
        Exact positive Python episode integer for the display label.

    Returns
    -------
    str
        ASCII basename at most 255 bytes, ending in .marlbg-replay.json, with map,
        episode, root/stream seeds, Team A/B policy labels, and full digest.

    Raises
    ------
    ValueError
        Digest/episode ID is invalid or fixed identity text cannot fit.

    Notes
    -----
    Sanitizes and shortens display policy labels only. Unknown seeds remain
    unknown. This helper neither reserves a path nor alters explicit caller paths.
    """
    from marl_battlegrounds.evaluation.map_identity import recorded_map

    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("replay filename requires its full SHA-256 digest")
    if type(episode_id) is not int or episode_id < 1:
        raise ValueError("replay filename requires a positive episode ID")

    def label(value: str, limit: int) -> str:
        """Make a bounded ASCII display label using letters, digits, and hyphens.

        Collapse nonalphanumeric runs, trim edge hyphens, fall back to unknown, then
        truncate. This affects filenames only, never scientific identities.
        """
        text = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-") or "unknown"
        return text[:limit]

    map_info = recorded_map(context)
    map_name = (
        map_info.technical_name
        if map_info.map_id is not None
        else (
            f"tdm_{'custom' if map_info.display_name == 'Custom Map' else 'recorded'}"
            f"_map_{context.identity.layout.canonical_digest[:12]}"
        )
    )
    seeds = context.seed_protocol
    root_seed = "unknown" if seeds.root_seed is None else str(seeds.root_seed)
    episode_seed = "unknown" if seeds.episode_seed is None else str(seeds.episode_seed)
    first = (
        f"{map_name}__episode-{label(str(episode_id), 20)}"
        f"__seed-{root_seed}__stream-{episode_seed}"
    )
    last = f"__{digest}{REPLAY_FILE_SUFFIX_V1}"
    # Reserve both team markers and shorten only display labels when needed.
    policy_limit = min(24, (255 - len(first) - len(last) - 8) // 2)
    if policy_limit < 1:
        raise ValueError("map identity is too long for a safe replay filename")
    teams = []
    for start in (0, 5):
        names = dict.fromkeys(
            row.policy_id
            for row in context.policy_assignments[start : start + 5]
            if isinstance(row, (AssignedPolicySlotV1, AssignedPolicySlotV2))
        )
        teams.append(label("-".join(names), policy_limit))
    filename = f"{first}__a-{teams[0]}__b-{teams[1]}{last}"
    if not filename.isascii() or len(filename.encode("ascii")) > 255:
        raise ValueError("generated replay filename exceeds the safe basename limit")
    return filename


def preflight_replay_destination(path: str | os.PathLike[str]) -> ReplayDestination:
    """Check a replay destination before recording without writing or reserving it.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Desired nonempty .marlbg-replay.json path whose parent already exists.

    Returns
    -------
    ReplayDestination
        ReplayDestination for one V2/V3/V4 replay file.

    Raises
    ------
    ReplaySaveError
        Invalid filename/path, unsupported secure filesystem support,
        missing/unsafe parent, existing replay target.

    Notes
    -----
    Rejects symlinks. No adjacent metric file is inspected.
    This preflight is advisory; exclusive publication still handles races.
    """
    try:
        replay_path = _coerce_path(path)
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument", path=None, detail=str(error)
        ) from error
    _validate_save_destination(replay_path)
    parent = _open_parent_directory(replay_path, error_type=ReplaySaveError)
    try:
        if (
            _entry_status_at(
                parent, replay_path.name, path=replay_path, error_type=ReplaySaveError
            )
            is not None
        ):
            raise ReplaySaveError(
                "replay_target_exists",
                path=replay_path,
                detail="replay destinations are never overwritten",
            )
    finally:
        os.close(parent)
    return ReplayDestination(replay_path)


def publish_prepared_replay(
    prepared: PreparedReplay,
    destination: ReplayDestination,
    *,
    verify_existing_replay: bool = False,
) -> SavedReplay:
    """Publish cached replay bytes or verify an uncertain earlier publication.

    Parameters
    ----------
    prepared : PreparedReplay
        Exact PreparedReplay with validated immutable bytes.
    destination : ReplayDestination
        Exact ReplayDestination naming the selected local output.
    verify_existing_replay : bool
        Exact bool, default false. If true, write nothing
        and require existing replay to equal the cached bytes.

    Returns
    -------
    SavedReplay
        SavedReplay with selected path and verified byte length.

    Raises
    ------
    ReplaySaveError
        Argument types, path safety, existing-target conflicts,
        temporary writes, exclusive publication, or final byte/durability verification
        fails.

    Notes
    -----
    Publish through a temporary file and exclusive hard link.
    Existing replay bytes are never overwritten. Files and parent metadata are
    fsynced and compared with cached bytes. Verification failure may leave a
    published file; explicit verification supports recovery without reserialization.
    """
    if (
        type(prepared) is not PreparedReplay
        or type(destination) is not ReplayDestination
        or type(verify_existing_replay) is not bool
    ):
        raise ReplaySaveError(
            "invalid_argument",
            path=None,
            detail="publication requires exact prepared replay and destination types",
        )
    path = destination.replay_path
    parent = _open_parent_directory(path, error_type=ReplaySaveError)
    try:
        if not verify_existing_replay:
            _publish_bytes_no_clobber(
                path,
                prepared.replay_json_bytes,
                existing_code="replay_target_exists",
                parent_descriptor=parent,
            )
        try:
            published = _read_bounded_regular_file_at(
                parent,
                path.name,
                path=path,
                max_file_size_bytes=prepared.max_file_size_bytes,
                error_type=ReplaySaveError,
                fsync_before_close=True,
            )
            if published != prepared.replay_json_bytes:
                raise ValueError("published replay differs from cached bytes")
            _fsync_directory(parent)
        except (ReplaySaveError, OSError, ValueError) as error:
            raise ReplaySaveError(
                "replay_publication_verification_failed",
                path=path,
                detail="published replay could not be verified against cached bytes",
            ) from error
    finally:
        os.close(parent)
    return SavedReplay(path, len(prepared.replay_json_bytes))


def _publish_staged_replays(  # pyright: ignore[reportUnusedFunction]
    files: list[tuple[Path, Path, int]],
) -> None:
    """Publish a whole checked family of privately staged replay files.

    Parameters
    ----------
    files : list of (Path, Path, int)
        Private source path, final replay path and expected byte length. Every
        source must already have been validated, serialized and made durable by
        publish_prepared_replay in a caller-owned directory on the same filesystem.
        The caller keeps those immutable sources until this function returns.

    Returns
    -------
    None
        All final files exist with the prepared bytes. Identical existing files
        are reused. No model is rebuilt and no payload is copied into memory.

    Raises
    ------
    ReplaySaveError
        A path is unsafe, staged evidence changed, an existing destination
        conflicts, or exclusive linking/durability fails. All known conflicts
        are checked before the first final link. Later filesystem failures can
        leave earlier files published, as in ordinary replay publication.

    Notes
    -----
    Internal writer route only. Existing files compare in 1 MiB blocks. New
    files use an exclusive hard link to the already durable private inode;
    linking cannot overwrite another writer's file or serialize a model twice.
    """
    if len({target for _, target, _ in files}) != len(files):
        raise ReplaySaveError(
            "invalid_argument", path=None, detail="replay destinations must be distinct"
        )
    checked: list[tuple[Path, Path, bool, tuple[int, int]]] = []
    for source, target, size in files:
        _validate_save_destination(target)
        source_parent = _open_parent_directory(source, error_type=ReplaySaveError)
        try:
            target_parent = _open_parent_directory(target, error_type=ReplaySaveError)
            try:
                source_stat = _entry_status_at(
                    source_parent, source.name, path=source, error_type=ReplaySaveError
                )
                if (
                    source_stat is None
                    or not stat.S_ISREG(source_stat.st_mode)
                    or source_stat.st_size != size
                ):
                    raise ReplaySaveError(
                        "replay_publication_verification_failed",
                        path=source,
                        detail="private replay bytes changed before publication",
                    )
                target_stat = _entry_status_at(
                    target_parent, target.name, path=target, error_type=ReplaySaveError
                )
                if target_stat is not None:
                    if (
                        not stat.S_ISREG(target_stat.st_mode)
                        or target_stat.st_size != size
                    ):
                        raise ReplaySaveError(
                            "replay_target_exists",
                            path=target,
                            detail="existing replay differs from prepared data",
                        )
                    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
                    with (
                        os.fdopen(
                            os.open(source.name, flags, dir_fd=source_parent), "rb"
                        ) as first,
                        os.fdopen(
                            os.open(target.name, flags, dir_fd=target_parent), "rb"
                        ) as second,
                    ):
                        while block := first.read(1024 * 1024):
                            if second.read(len(block)) != block:
                                raise ReplaySaveError(
                                    "replay_target_exists",
                                    path=target,
                                    detail="existing replay differs from prepared data",
                                )
                        if second.read(1):
                            raise ReplaySaveError(
                                "replay_target_exists",
                                path=target,
                                detail="existing replay grew during verification",
                            )
                        os.fsync(second.fileno())
                    _fsync_directory(target_parent)
                checked.append(
                    (
                        source,
                        target,
                        target_stat is not None,
                        (source_stat.st_dev, source_stat.st_ino),
                    )
                )
            finally:
                os.close(target_parent)
        except OSError as error:
            raise ReplaySaveError(
                "file_read_failed",
                path=target,
                detail="could not check staged replay destinations",
            ) from error
        finally:
            os.close(source_parent)
    for source, target, exists, identity in checked:
        if exists:
            continue
        source_parent = _open_parent_directory(source, error_type=ReplaySaveError)
        try:
            target_parent = _open_parent_directory(target, error_type=ReplaySaveError)
            linked = False
            try:
                os.link(
                    source.name,
                    target.name,
                    src_dir_fd=source_parent,
                    dst_dir_fd=target_parent,
                    follow_symlinks=False,
                )
                linked = True
                result = os.stat(
                    target.name, dir_fd=target_parent, follow_symlinks=False
                )
                if (result.st_dev, result.st_ino) != identity:
                    raise OSError("published replay differs from its private source")
                _fsync_directory(target_parent)
            except OSError as error:
                if linked:
                    with suppress(OSError):
                        result = os.stat(
                            target.name, dir_fd=target_parent, follow_symlinks=False
                        )
                        if (result.st_dev, result.st_ino) == identity:
                            os.unlink(target.name, dir_fd=target_parent)
                            _fsync_directory(target_parent)
                raise ReplaySaveError(
                    "replay_target_exists"
                    if isinstance(error, FileExistsError)
                    else "atomic_publish_failed",
                    path=target,
                    detail="staged replay could not be exclusively published",
                ) from error
            finally:
                os.close(target_parent)
        finally:
            os.close(source_parent)


def save_replay(
    replay: ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedReplay:
    """Validate and publish one self-contained replay file.

    Parameters
    ----------
    replay : ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4
        Exact ReplayArtifactV2, V3 or V4, including valid partial prefixes.
    path : str | os.PathLike[str]
        Absent nonempty .marlbg-replay.json destination in an existing parent.
    max_file_size_bytes : int
        Exact positive byte cap, default 1 GiB (1024**3).

    Returns
    -------
    SavedReplay
        SavedReplay with local path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid model/arguments, unsafe destination, size limit,
        existing target, or publication/durability verification failure.

    Notes
    -----
    Prepares bytes once, checks the destination, then publishes without overwrite.
    No metric sidecar is required or written. Use PreparedReplay with explicit
    publication for retries after an uncertain save.
    """
    try:
        prepared = PreparedReplay(replay, max_file_size_bytes=max_file_size_bytes)
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument", path=None, detail=str(error)
        ) from error
    return publish_prepared_replay(prepared, preflight_replay_destination(path))


def _load_actor_pov_replay_artifact(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV1 | ReplayArtifactV3 | ReplayArtifactV4 | None,
    pov_version: Literal[1, 2, 3],
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ActorPovReplayArtifactV1 | ActorPovReplayArtifactV2 | ActorPovReplayArtifactV3:
    """Read a bounded canonical POV file and optionally join supplied source evidence.

    pov_version selects the only accepted POV version (1, 2 or 3); a supplied
    source replay must be its paired version (replay V1, V3 or V4). Validate
    suffix and positive limits, then wrap byte/model/source-join errors as
    ReplayLoadError. No source replay is discovered from paths or downloaded.
    """
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
        pov_version=pov_version,
    )
    if source_replay is not None:
        try:
            _validate_actor_pov_source_join(
                artifact, source_replay, pov_version=pov_version
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
    """Load canonical scenario record V1 and check its supplied evidence.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-scenario.json stem.
    source_replay : ReplayArtifactV1
        Exact ReplayArtifactV1 referenced by the record.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Matching V1 report sidecar.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ScenarioEvaluationRecordV1
        Exact ScenarioEvaluationRecordV1 with validated local fields and evidence joins.

    Raises
    ------
    ReplayLoadError
        Unsafe path, byte/depth limit, wrong schema, noncanonical
        content, invalid model, or evidence mismatch.

    Notes
    -----
    Supplied replay/report objects are checked, not discovered from adjacent
    paths. This uses generic scenario validation without live official Core
    admission and does not recompute endpoint values or predicate truth.
    """
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


def load_scenario_evaluation_record_v2(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ScenarioEvaluationRecordV2:
    """Load canonical scenario record V2 and check its supplied evidence.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-scenario.json stem.
    source_replay : ReplayArtifactV1
        Exact ReplayArtifactV1 referenced by the record.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Matching V1 report sidecar.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ScenarioEvaluationRecordV2
        Exact ScenarioEvaluationRecordV2 with validated local fields and evidence joins.

    Raises
    ------
    ReplayLoadError
        Unsafe path, byte/depth limit, wrong schema, noncanonical
        content, invalid model, or evidence mismatch.

    Notes
    -----
    Supplied replay/report objects are checked, not discovered from adjacent
    paths. This uses generic scenario validation without live official Core
    admission and does not recompute endpoint values or predicate truth.
    """
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
    record = _load_scenario_record_bytes_v2(
        payload,
        path=scenario_path,
        max_json_depth=depth_limit,
    )
    try:
        validate_scenario_evaluation_record_v2(
            record,
            source_replay,
            metric_report_artifact,
        )
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "semantic_validation_failed",
            path=scenario_path,
            detail="V2 scenario record does not join its supplied replay/report",
        ) from error
    return record


def prepare_replay_bundle_v1(
    bundle: ReplayBundleV1,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> PreparedReplayBundleV1:
    """Validate a historical bundle and cache both canonical publication payloads.

    Parameters
    ----------
    bundle : ReplayBundleV1
        Exact ReplayBundleV1 with matching replay/report content.
    max_file_size_bytes : int
        Exact positive cap for each member, default 1 GiB.

    Returns
    -------
    PreparedReplayBundleV1
        PreparedReplayBundleV1 containing immutable bytes, lengths, and payload hashes.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/content or either member exceeds its cap.

    Notes
    -----
    Runs full V1 replay and sidecar joins on the host. It writes no files;
    reuse this result for publication attempts without repeating serialization.
    """
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
    """Return the derived sidecar path or raise an invalid_filename save error.

    Only filename structure is checked here; directory and target checks are separate.
    """
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
    """Inspect one entry without following links, returning None only when absent.

    Reject symlinks and translate stat failures to the supplied I/O error type.
    Leave the caller's parent descriptor open.
    """
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
    """Check a replay destination before recording without writing or reserving it.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Desired nonempty .marlbg-replay.json path whose parent already exists.

    Returns
    -------
    ReplayBundleDestinationV1
        ReplayBundleDestinationV1 with the derived adjacent metric path.

    Raises
    ------
    ReplaySaveError
        Invalid filename/path, unsupported secure filesystem support,
        missing/unsafe parent, existing replay target, or nonregular sidecar.

    Notes
    -----
    Rejects symlinks. An existing regular sidecar is allowed here; publication later
    requires identical bytes.
    This preflight is advisory; exclusive publication still handles races.
    """
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
    """Flush directory metadata to durable storage using the caller-owned descriptor.

    OSError propagates to the publication boundary that assigns its error code.
    """
    os.fsync(parent_descriptor)


def _rollback_published_link(
    parent_descriptor: int,
    target_name: str,
    temporary_name: str,
) -> None:
    """Best-effort remove only a target still sharing this temporary file's inode.

    Compare device and inode without following links. Never remove a replacement
    entry owned by another writer. Suppress cleanup/fsync errors during rollback.
    """
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
    """Create a private 0600 temporary file in the already opened parent directory.

    Use exclusive no-follow creation and up to 128 random names. Return owned file
    descriptor and relative basename; the caller writes, closes, and removes it.
    Raise OSError if no unique name can be allocated.
    """
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
    """Write/fsync temporary bytes, then publish with an exclusive hard link.

    Never replace an existing target. Use an optional caller-owned parent descriptor
    or open/close one locally. Flush directory metadata after linking; on failure
    remove only the link created from this temporary inode when possible. Always
    attempt temporary cleanup. Raise stable ReplaySaveError codes.
    """
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
    """Publish a V1 sidecar or reuse an existing regular file with identical bytes.

    Return true for reuse and false for new publication. Compare a racing existing
    file too, enforce the byte cap, and fsync file/directory before success.
    Differing bytes raise metric_report_conflict; no sidecar is overwritten.
    """
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
    """Require both published members to equal cached bytes and flush their durability.

    Use bounded no-follow reads through the supplied parent descriptor. Any missing,
    different, unreadable, or unflushable member becomes
    replay_publication_verification_failed. Leave the parent descriptor open.
    """
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
    """Publish cached replay/report bytes or verify an uncertain earlier publication.

    Parameters
    ----------
    prepared : PreparedReplayBundleV1
        Exact PreparedReplayBundleV1 with validated immutable bytes.
    destination : ReplayBundleDestinationV1
        Exact ReplayBundleDestinationV1 naming the selected local output.
    verify_existing_replay : bool
        Exact bool, default false. If true, write nothing
        and require existing members to equal the cached bytes.

    Returns
    -------
    SavedReplayBundleV1
        SavedReplayBundleV1 with both paths, lengths, and sidecar reuse flag.

    Raises
    ------
    ReplaySaveError
        Argument types, path safety, existing-target conflicts,
        temporary writes, exclusive publication, or final byte/durability verification
        fails.

    Notes
    -----
    Publish or reuse an identical sidecar first, then publish the replay. A later replay
    failure may leave the valid sidecar for retry.
    Existing replay bytes are never overwritten. Files and parent metadata are
    fsynced and compared with cached bytes. Verification failure may leave a
    published file; explicit verification supports recovery without reserialization.
    """
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
    """Publish a historical metric sidecar followed by its referencing replay.

    Parameters
    ----------
    bundle : ReplayBundleV1
        Exact valid ReplayBundleV1.
    path : str | os.PathLike[str]
        Absent .marlbg-replay.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive per-member byte cap; default 1 GiB.

    Returns
    -------
    SavedReplayBundleV1
        SavedReplayBundleV1 with paths, lengths, and sidecar reuse status.

    Raises
    ------
    ReplaySaveError
        Preparation, filename/path, conflict, size, write, or verification fails.

    Notes
    -----
    Reuses an existing sidecar only when bytes are identical. The replay is never
    overwritten. Failure after sidecar publication may leave that valid sidecar.
    Use separate prepare/preflight/publish calls for explicit retry verification.
    """
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
    """Publish one bounded companion byte string without replacing any existing entry.

    Require an existing secure parent, enforce size, and use atomic no-clobber
    publication. Return path and written byte length; the caller already validated
    suffix, model, and source joins.
    """
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


def _save_actor_pov_replay_artifact(
    artifact: ActorPovReplayArtifactV1
    | ActorPovReplayArtifactV2
    | ActorPovReplayArtifactV3,
    source_replay: ReplayArtifactV1 | ReplayArtifactV3 | ReplayArtifactV4,
    path: str | os.PathLike[str],
    *,
    pov_version: Literal[1, 2, 3],
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate the selected POV version and its source, then publish canonical bytes.

    pov_version selects the only accepted pair: 3 for POV V3 with replay V4, 2 for
    POV V2 with replay V3, or 1 for historical POV V1 with replay V1. Require a
    valid .pov suffix, positive byte cap, and absent destination. Wrap source
    mismatch as invalid_argument and leave source artifacts unchanged.
    """
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
        _validate_actor_pov_source_join(
            artifact, source_replay, pov_version=pov_version
        )
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
    """Validate evidence and publish scenario record V1 without overwrite.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV1
        Exact ScenarioEvaluationRecordV1 to publish.
    source_replay : ReplayArtifactV1
        Exact referenced ReplayArtifactV1.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Matching V1 metric sidecar.
    path : str | os.PathLike[str]
        Absent .marlbg-scenario.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with output path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/evidence, unsafe path, size limit,
        existing target, or atomic publication failure.

    Notes
    -----
    Checks generic evidence joins without live official Core admission. Writes
    only the companion; source replay/report files are not published or changed.
    Caller-computed measurements and predicate values are not recomputed.
    """
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


def save_scenario_evaluation_record_v2(
    record: ScenarioEvaluationRecordV2,
    source_replay: ReplayArtifactV1,
    metric_report_artifact: EvaluationMetricReportArtifactV1,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate evidence and publish scenario record V2 without overwrite.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV2
        Exact ScenarioEvaluationRecordV2 to publish.
    source_replay : ReplayArtifactV1
        Exact referenced ReplayArtifactV1.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Matching V1 metric sidecar.
    path : str | os.PathLike[str]
        Absent .marlbg-scenario.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with output path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/evidence, unsafe path, size limit,
        existing target, or atomic publication failure.

    Notes
    -----
    Checks generic evidence joins without live official Core admission. Writes
    only the companion; source replay/report files are not published or changed.
    Caller-computed measurements and predicate values are not recomputed.
    """
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
        validate_scenario_evaluation_record_v2(
            record,
            source_replay,
            metric_report_artifact,
        )
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=scenario_path,
            detail="V2 scenario record does not match its supplied replay/report",
        ) from error
    return _save_companion_payload(
        scenario_path,
        canonical_json_bytes(record),
        max_file_size_bytes=size_limit,
    )


type _ScenarioRecordVersion = Literal[3, 4, 5]
type _CurrentScenarioRecord = (
    ScenarioEvaluationRecordV3 | ScenarioEvaluationRecordV4 | ScenarioEvaluationRecordV5
)
type _CurrentScenarioReplay = ReplayArtifactV2 | ReplayArtifactV3 | ReplayArtifactV4
_SCENARIO_RECORD_MODELS: dict[
    int,
    type[ScenarioEvaluationRecordV3]
    | type[ScenarioEvaluationRecordV4]
    | type[ScenarioEvaluationRecordV5],
] = {
    3: ScenarioEvaluationRecordV3,
    4: ScenarioEvaluationRecordV4,
    5: ScenarioEvaluationRecordV5,
}


def _validate_current_scenario_record(
    record: _CurrentScenarioRecord,
    replay: _CurrentScenarioReplay,
) -> None:
    """Dispatch exact scenario V3/replay V2, V4/replay V3 or V5/replay V4 joins.

    Raise TypeError for other pairings. Use generic evidence checks, without live
    official Core admission or recomputing measurements.
    """
    if type(record) is ScenarioEvaluationRecordV5 and type(replay) is ReplayArtifactV4:
        validate_scenario_evaluation_record_v5(record, replay)
    elif (
        type(record) is ScenarioEvaluationRecordV4 and type(replay) is ReplayArtifactV3
    ):
        validate_scenario_evaluation_record_v4(record, replay)
    elif (
        type(record) is ScenarioEvaluationRecordV3 and type(replay) is ReplayArtifactV2
    ):
        validate_scenario_evaluation_record_v3(record, replay)
    else:
        raise TypeError("scenario and replay versions must match")


def _load_current_scenario_record_bytes(
    payload: bytes,
    *,
    path: Path,
    max_json_depth: int,
    version: _ScenarioRecordVersion,
) -> _CurrentScenarioRecord:
    """Admit canonical scenario record bytes of exactly the given version (3-5).

    Check strict JSON/schema/model structure and exact re-encoding before returning
    the record. Actual replay joins are checked by the outer loader.
    """
    model = _SCENARIO_RECORD_MODELS[version]
    _preflight_json(
        payload,
        path=path,
        expected_schema_id=SCENARIO_EVALUATION_RECORD_SCHEMA_ID,
        expected_schema_version=version,
        max_json_depth=max_json_depth,
    )
    try:
        record = model.model_validate_json(payload)
        canonical_record = cast(
            _CurrentScenarioRecord,
            validate_declared_model_tree(
                record,
                record_name="loaded Versioned scenario evaluation record",
                expected_type=model,
            ),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise ReplayLoadError(
            "model_validation_failed",
            path=path,
            detail="Scenario record does not satisfy its declared version contract",
        ) from error
    if canonical_json_bytes(canonical_record) != payload:
        raise ReplayLoadError(
            "noncanonical_json",
            path=path,
            detail="scenario bytes are not the canonical declared-version encoding",
        )
    return canonical_record


def _load_current_scenario_evaluation_record(
    path: str | os.PathLike[str],
    *,
    source_replay: _CurrentScenarioReplay,
    version: _ScenarioRecordVersion,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> _CurrentScenarioRecord:
    """Load a bounded V3-V5 scenario file and check its supplied replay evidence.

    version selects the exact record version. Require the scenario suffix and positive
    limits. Wrap structural/path/evidence failures in ReplayLoadError; no metric
    sidecar or live official Core validation is performed.
    """
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
    record = _load_current_scenario_record_bytes(
        payload,
        path=scenario_path,
        max_json_depth=depth_limit,
        version=version,
    )
    try:
        _validate_current_scenario_record(
            record,
            source_replay,
        )
    except (TypeError, ValueError) as error:
        raise ReplayLoadError(
            "semantic_validation_failed",
            path=scenario_path,
            detail="Versioned scenario record does not join its supplied replay",
        ) from error
    return record


def _save_current_scenario_evaluation_record(
    record: _CurrentScenarioRecord,
    source_replay: _CurrentScenarioReplay,
    version: _ScenarioRecordVersion,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate an exact V3-V5 record/replay pair and publish canonical scenario bytes.

    version fixes the allowed record version. Require safe suffix/parent/size and
    an absent destination. Generic evidence validation does not recompute endpoint
    values or rerun official simulator admission.
    """
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
        if type(record) is not _SCENARIO_RECORD_MODELS[version]:
            raise ValueError("scenario record must use its declared version")
        _validate_current_scenario_record(
            record,
            source_replay,
        )
    except (TypeError, ValueError) as error:
        raise ReplaySaveError(
            "invalid_argument",
            path=scenario_path,
            detail="Versioned scenario record does not match its supplied replay",
        ) from error
    return _save_companion_payload(
        scenario_path,
        canonical_json_bytes(record),
        max_file_size_bytes=size_limit,
    )


def load_scenario_evaluation_record_v3(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV2,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ScenarioEvaluationRecordV3:
    """Load canonical scenario record V3 and check its supplied evidence.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-scenario.json stem.
    source_replay : ReplayArtifactV2
        Exact ReplayArtifactV2 referenced by the record.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ScenarioEvaluationRecordV3
        Exact ScenarioEvaluationRecordV3 with validated local fields and evidence joins.

    Raises
    ------
    ReplayLoadError
        Unsafe path, byte/depth limit, wrong schema, noncanonical
        content, invalid model, or evidence mismatch.

    Notes
    -----
    Supplied replay/report objects are checked, not discovered from adjacent
    paths. This uses generic scenario validation without live official Core
    admission and does not recompute endpoint values or predicate truth.
    """
    return cast(
        ScenarioEvaluationRecordV3,
        _load_current_scenario_evaluation_record(
            path,
            source_replay=source_replay,
            version=3,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        ),
    )


def save_scenario_evaluation_record_v3(
    record: ScenarioEvaluationRecordV3,
    source_replay: ReplayArtifactV2,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate evidence and publish scenario record V3 without overwrite.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV3
        Exact ScenarioEvaluationRecordV3 to publish.
    source_replay : ReplayArtifactV2
        Exact referenced ReplayArtifactV2.
    path : str | os.PathLike[str]
        Absent .marlbg-scenario.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with output path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/evidence, unsafe path, size limit,
        existing target, or atomic publication failure.

    Notes
    -----
    Checks generic evidence joins without live official Core admission. Writes
    only the companion; source replay/report files are not published or changed.
    Caller-computed measurements and predicate values are not recomputed.
    """
    return _save_current_scenario_evaluation_record(
        record,
        source_replay,
        3,
        path,
        max_file_size_bytes=max_file_size_bytes,
    )


def load_scenario_evaluation_record_v4(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV3,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ScenarioEvaluationRecordV4:
    """Load canonical scenario record V4 and check its supplied evidence.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-scenario.json stem.
    source_replay : ReplayArtifactV3
        Exact ReplayArtifactV3 referenced by the record.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ScenarioEvaluationRecordV4
        Exact ScenarioEvaluationRecordV4 with validated local fields and evidence joins.

    Raises
    ------
    ReplayLoadError
        Unsafe path, byte/depth limit, wrong schema, noncanonical
        content, invalid model, or evidence mismatch.

    Notes
    -----
    Supplied replay/report objects are checked, not discovered from adjacent
    paths. This uses generic scenario validation without live official Core
    admission and does not recompute endpoint values or predicate truth.
    """
    return cast(
        ScenarioEvaluationRecordV4,
        _load_current_scenario_evaluation_record(
            path,
            source_replay=source_replay,
            version=4,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        ),
    )


def save_scenario_evaluation_record_v4(
    record: ScenarioEvaluationRecordV4,
    source_replay: ReplayArtifactV3,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate evidence and publish scenario record V4 without overwrite.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV4
        Exact ScenarioEvaluationRecordV4 to publish.
    source_replay : ReplayArtifactV3
        Exact referenced ReplayArtifactV3.
    path : str | os.PathLike[str]
        Absent .marlbg-scenario.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with output path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/evidence, unsafe path, size limit,
        existing target, or atomic publication failure.

    Notes
    -----
    Checks generic evidence joins without live official Core admission. Writes
    only the companion; source replay/report files are not published or changed.
    Caller-computed measurements and predicate values are not recomputed.
    """
    return _save_current_scenario_evaluation_record(
        record,
        source_replay,
        4,
        path,
        max_file_size_bytes=max_file_size_bytes,
    )


def load_scenario_evaluation_record_v5(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV4,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ScenarioEvaluationRecordV5:
    """Load canonical scenario record V5 and check its supplied evidence.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-scenario.json stem.
    source_replay : ReplayArtifactV4
        Exact ReplayArtifactV4 referenced by the record.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ScenarioEvaluationRecordV5
        Exact ScenarioEvaluationRecordV5 with validated local fields and evidence joins.

    Raises
    ------
    ReplayLoadError
        Unsafe path, byte/depth limit, wrong schema, noncanonical
        content, invalid model, or evidence mismatch.

    Notes
    -----
    Supplied replay/report objects are checked, not discovered from adjacent
    paths. This uses generic scenario validation without live official Core
    admission and does not recompute endpoint values or predicate truth.
    """
    return cast(
        ScenarioEvaluationRecordV5,
        _load_current_scenario_evaluation_record(
            path,
            source_replay=source_replay,
            version=5,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        ),
    )


def save_scenario_evaluation_record_v5(
    record: ScenarioEvaluationRecordV5,
    source_replay: ReplayArtifactV4,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Validate evidence and publish scenario record V5 without overwrite.

    Parameters
    ----------
    record : ScenarioEvaluationRecordV5
        Exact ScenarioEvaluationRecordV5 to publish.
    source_replay : ReplayArtifactV4
        Exact referenced ReplayArtifactV4.
    path : str | os.PathLike[str]
        Absent .marlbg-scenario.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with output path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/evidence, unsafe path, size limit,
        existing target, or atomic publication failure.

    Notes
    -----
    Checks generic evidence joins without live official Core admission. Writes
    only the companion; source replay/report files are not published or changed.
    Caller-computed measurements and predicate values are not recomputed.
    """
    return _save_current_scenario_evaluation_record(
        record,
        source_replay,
        5,
        path,
        max_file_size_bytes=max_file_size_bytes,
    )


def _validate_actor_pov_source_join(
    artifact: ActorPovReplayArtifactV1
    | ActorPovReplayArtifactV2
    | ActorPovReplayArtifactV3,
    replay: ReplayArtifactV1 | ReplayArtifactV3 | ReplayArtifactV4,
    *,
    pov_version: Literal[1, 2, 3],
) -> None:
    """Require the supported POV/source version pair and validate every source join.

    pov_version 3 means exact POV V3 with replay V4, 2 means exact POV V2 with
    replay V3, and 1 means exact POV V1 with replay V1. Raise ValueError for
    other pairings or projection/content disagreement.
    """
    if pov_version == 3:
        if (
            type(artifact) is not ActorPovReplayArtifactV3
            or type(replay) is not ReplayArtifactV4
        ):
            raise ValueError("current POV evidence requires POV V3 and replay V4")
        validate_actor_pov_replay_against_replay_v3(artifact, replay)
    elif pov_version == 2:
        if (
            type(artifact) is not ActorPovReplayArtifactV2
            or type(replay) is not ReplayArtifactV3
        ):
            raise ValueError("POV V2 evidence requires POV V2 and replay V3")
        validate_actor_pov_replay_against_replay_v2(artifact, replay)
    else:
        if (
            type(artifact) is not ActorPovReplayArtifactV1
            or type(replay) is not ReplayArtifactV1
        ):
            raise ValueError("historical POV evidence requires POV V1 and replay V1")
        validate_actor_pov_replay_against_replay_v1(artifact, replay)


def load_actor_pov_replay_artifact_v1(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV1 | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ActorPovReplayArtifactV1:
    """Load canonical actor-view artifact V1 with an optional source check.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-pov.json stem.
    source_replay : ReplayArtifactV1 | None
        Optional exact ReplayArtifactV1; defaults to None. If supplied,
        every projected source join is checked.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ActorPovReplayArtifactV1
        Exact ActorPovReplayArtifactV1 after standalone structure/content validation.

    Raises
    ------
    ReplayLoadError
        Unsafe path, size/depth limit, invalid/noncanonical content,
        wrong version, or mismatched optional source.

    Notes
    -----
    Standalone loading supports sharing only the actor view. Without source_replay
    it cannot prove equality to unavailable source evidence. No file is written.
    """
    return cast(
        ActorPovReplayArtifactV1,
        _load_actor_pov_replay_artifact(
            path,
            source_replay=source_replay,
            pov_version=1,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        ),
    )


def save_actor_pov_replay_artifact_v1(
    artifact: ActorPovReplayArtifactV1,
    source_replay: ReplayArtifactV1,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Check an actor-view artifact against its source and publish it without overwrite.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV1
        Exact ActorPovReplayArtifactV1.
    source_replay : ReplayArtifactV1
        Exact ReplayArtifactV1 from which this view was projected.
    path : str | os.PathLike[str]
        Absent .marlbg-pov.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/source join, unsafe path, size limit,
        existing target, or atomic publication failure.

    Notes
    -----
    Publishes only the actor-view bytes after checking their source. The full
    privileged replay is not embedded or copied to the destination.
    """
    return _save_actor_pov_replay_artifact(
        artifact,
        source_replay,
        path,
        pov_version=1,
        max_file_size_bytes=max_file_size_bytes,
    )


def load_actor_pov_replay_artifact_v2(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV3 | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ActorPovReplayArtifactV2:
    """Load canonical actor-view artifact V2 with an optional source check.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-pov.json stem.
    source_replay : ReplayArtifactV3 | None
        Optional exact ReplayArtifactV3; defaults to None. If supplied,
        every projected source join is checked.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ActorPovReplayArtifactV2
        Exact ActorPovReplayArtifactV2 after standalone structure/content validation.

    Raises
    ------
    ReplayLoadError
        Unsafe path, size/depth limit, invalid/noncanonical content,
        wrong version, or mismatched optional source.

    Notes
    -----
    Standalone loading supports sharing only the actor view. Without source_replay
    it cannot prove equality to unavailable source evidence. No file is written.
    """
    return cast(
        ActorPovReplayArtifactV2,
        _load_actor_pov_replay_artifact(
            path,
            source_replay=source_replay,
            pov_version=2,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        ),
    )


def save_actor_pov_replay_artifact_v2(
    artifact: ActorPovReplayArtifactV2,
    source_replay: ReplayArtifactV3,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Check an actor-view artifact against its source and publish it without overwrite.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV2
        Exact ActorPovReplayArtifactV2.
    source_replay : ReplayArtifactV3
        Exact ReplayArtifactV3 from which this view was projected.
    path : str | os.PathLike[str]
        Absent .marlbg-pov.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/source join, unsafe path, size limit,
        existing target, or atomic publication failure.

    Notes
    -----
    Publishes only the actor-view bytes after checking their source. The full
    privileged replay is not embedded or copied to the destination.
    """
    return _save_actor_pov_replay_artifact(
        artifact,
        source_replay,
        path,
        pov_version=2,
        max_file_size_bytes=max_file_size_bytes,
    )


def load_actor_pov_replay_artifact_v3(
    path: str | os.PathLike[str],
    *,
    source_replay: ReplayArtifactV4 | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    max_json_depth: int = DEFAULT_MAX_REPLAY_JSON_DEPTH_V1,
) -> ActorPovReplayArtifactV3:
    """Load canonical actor-view artifact V3 with an optional source check.

    Parameters
    ----------
    path : str | os.PathLike[str]
        Existing regular file with a nonempty .marlbg-pov.json stem.
    source_replay : ReplayArtifactV4 | None
        Optional exact ReplayArtifactV4; defaults to None. If supplied,
        every projected source join is checked.
    max_file_size_bytes : int
        Exact positive per-file byte cap; defaults to 1 GiB (1024**3).
    max_json_depth : int
        Exact positive nesting cap; defaults to 128.

    Returns
    -------
    ActorPovReplayArtifactV3
        Exact ActorPovReplayArtifactV3 after standalone structure/content validation.

    Raises
    ------
    ReplayLoadError
        Unsafe path, size/depth limit, invalid/noncanonical content,
        wrong version (POV V1 or V2 bytes are refused), or mismatched optional
        source.

    Notes
    -----
    POV V3 is the current actor view, exported from replay V4; each frame keeps
    context column 19, the recorded Red Zone depth. Standalone loading supports
    sharing only the actor view. Without source_replay it cannot prove equality
    to unavailable source evidence. No file is written.
    """
    return cast(
        ActorPovReplayArtifactV3,
        _load_actor_pov_replay_artifact(
            path,
            source_replay=source_replay,
            pov_version=3,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        ),
    )


def save_actor_pov_replay_artifact_v3(
    artifact: ActorPovReplayArtifactV3,
    source_replay: ReplayArtifactV4,
    path: str | os.PathLike[str],
    *,
    max_file_size_bytes: int = DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
) -> SavedCompanionArtifactV1:
    """Check a POV V3 artifact against its source and publish it without overwrite.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV3
        Exact ActorPovReplayArtifactV3.
    source_replay : ReplayArtifactV4
        Exact ReplayArtifactV4 from which this view was projected.
    path : str | os.PathLike[str]
        Absent .marlbg-pov.json path in an existing secure parent.
    max_file_size_bytes : int
        Exact positive byte cap; defaults to 1 GiB.

    Returns
    -------
    SavedCompanionArtifactV1
        SavedCompanionArtifactV1 with path and canonical byte length.

    Raises
    ------
    ReplaySaveError
        Invalid arguments/source join (including a POV V2 artifact or a replay
        V3 source), unsafe path, size limit, existing target, or atomic
        publication failure.

    Notes
    -----
    Publishes only the actor-view bytes after checking their source. The full
    privileged replay is not embedded or copied to the destination.
    """
    return _save_actor_pov_replay_artifact(
        artifact,
        source_replay,
        path,
        pov_version=3,
        max_file_size_bytes=max_file_size_bytes,
    )


__all__ = [
    "ACTOR_POV_FILE_SUFFIX_V1",
    "DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1",
    "DEFAULT_MAX_REPLAY_JSON_DEPTH_V1",
    "METRIC_REPORT_FILE_SUFFIX_V1",
    "REPLAY_FILE_SUFFIX_V1",
    "SCENARIO_FILE_SUFFIX_V1",
    "LoadedReplay",
    "LoadedReplayBundle",
    "LoadedReplayBundleV1",
    "PreparedReplay",
    "PreparedReplayBundleV1",
    "ReplayBundleDestinationV1",
    "ReplayBundleLoadStatusV1",
    "ReplayDestination",
    "ReplayIOError",
    "ReplayIOErrorCodeV1",
    "ReplayLoadError",
    "ReplaySaveError",
    "SavedCompanionArtifactV1",
    "SavedReplay",
    "SavedReplayBundleV1",
    "canonical_metric_report_artifact_json_bytes_v1",
    "canonical_replay_json_bytes_v1",
    "canonical_scenario_evaluation_record_json_bytes_v1",
    "canonical_scenario_evaluation_record_json_bytes_v2",
    "load_actor_pov_replay_artifact_v1",
    "load_actor_pov_replay_artifact_v2",
    "load_actor_pov_replay_artifact_v3",
    "load_replay",
    "load_replay_artifact_v1",
    "load_replay_bundle_v1",
    "load_scenario_evaluation_record_v1",
    "load_scenario_evaluation_record_v2",
    "load_scenario_evaluation_record_v3",
    "load_scenario_evaluation_record_v4",
    "load_scenario_evaluation_record_v5",
    "preflight_replay_bundle_destination_v1",
    "preflight_replay_destination",
    "prepare_replay_bundle_v1",
    "publish_prepared_replay",
    "publish_prepared_replay_bundle_v1",
    "save_actor_pov_replay_artifact_v1",
    "save_actor_pov_replay_artifact_v2",
    "save_actor_pov_replay_artifact_v3",
    "save_replay",
    "save_replay_bundle_v1",
    "save_scenario_evaluation_record_v1",
    "save_scenario_evaluation_record_v2",
    "save_scenario_evaluation_record_v3",
    "save_scenario_evaluation_record_v4",
    "save_scenario_evaluation_record_v5",
]
