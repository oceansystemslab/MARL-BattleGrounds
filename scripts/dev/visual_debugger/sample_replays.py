"""Import-light registry for checked-in Visual Debugger sample replays."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_REPLAY_DIRECTORY = _REPOSITORY_ROOT / "examples" / "replays" / "v1"
SAMPLE_REPLAY_MANIFEST_PATH = SAMPLE_REPLAY_DIRECTORY / "manifest.json"
SAMPLE_REPLAY_MANIFEST_SCHEMA_ID = (
    "marl_battlegrounds.visual_debugger.sample_replay_manifest"
)
SAMPLE_REPLAY_MANIFEST_SCHEMA_VERSION = 1
SAMPLE_REPLAY_GENERATOR_ID = "visual-debugger-sample-replays-v1"
SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE = (
    "Deterministic unofficial presentation demo; not a benchmark, policy "
    "evaluation, source-tree attestation, or host attestation."
)
SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES = 64 * 1024 * 1024
SAMPLE_REPLAY_MAX_MANIFEST_SIZE_BYTES = 1024 * 1024
_READ_CHUNK_SIZE_BYTES = 1024 * 1024
type _FileIdentity = tuple[int, int, int, int, int, int]


class SampleReplayVerificationError(ValueError):
    """A checked sample set failed an integrity or semantic proof."""


@dataclass(frozen=True, slots=True)
class SampleReplayDefinition:
    """One stable launch name and its immutable checked-in artifact pair."""

    name: str
    display_name: str
    description: str
    source_scenario: str
    replay_file_name: str
    metric_report_file_name: str
    seed: int = 0

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.name) is None:
            raise ValueError("sample replay names must be lower-kebab-case tokens")
        if not self.display_name or not self.description:
            raise ValueError("sample replay labels must be nonempty")
        if not self.source_scenario:
            raise ValueError("sample replay source scenario must be nonempty")
        expected_replay_name = f"{self.name}.marlbg-replay.json"
        expected_metric_name = f"{self.name}.marlbg-metrics.json"
        if self.replay_file_name != expected_replay_name:
            raise ValueError("sample replay filename must derive from its launch name")
        if self.metric_report_file_name != expected_metric_name:
            raise ValueError("sample metric filename must derive from its launch name")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("sample replay seed must be a nonnegative exact integer")

    def replay_path(self, directory: Path = SAMPLE_REPLAY_DIRECTORY) -> Path:
        """Resolve the registered replay member beneath one explicit directory."""
        return directory / self.replay_file_name

    def metric_report_path(
        self,
        directory: Path = SAMPLE_REPLAY_DIRECTORY,
    ) -> Path:
        """Resolve the registered metric member beneath one explicit directory."""
        return directory / self.metric_report_file_name

    def summary(self) -> str:
        """Return one stable, human-readable launcher listing row."""
        return f"{self.name:<31} {self.display_name} — {self.description}"


@dataclass(frozen=True, slots=True)
class VerifiedSampleReplaySetV1:
    """One manifest snapshot and its complete in-memory replay bundle set."""

    manifest: Mapping[str, object]
    sample_rows: tuple[Mapping[str, object], ...]
    bundles: tuple[LoadedReplayBundleV1, ...]
    file_names: frozenset[str]


SAMPLE_REPLAYS: tuple[SampleReplayDefinition, ...] = (
    SampleReplayDefinition(
        name="death-respawn-shield",
        display_name="Death, Respawn, and Spawn Shield",
        description="Lethal damage through the first post-shield interaction.",
        source_scenario="death_respawn_cycle",
        replay_file_name="death-respawn-shield.marlbg-replay.json",
        metric_report_file_name="death-respawn-shield.marlbg-metrics.json",
    ),
    SampleReplayDefinition(
        name="recovery-status-lifecycle",
        display_name="Recovery and Status Lifecycle",
        description="Recovery, rejection, refresh, break, reapply, and expiry.",
        source_scenario="recovery_refresh_cycle",
        replay_file_name="recovery-status-lifecycle.marlbg-replay.json",
        metric_report_file_name="recovery-status-lifecycle.marlbg-metrics.json",
    ),
    SampleReplayDefinition(
        name="mirrored-five-class-ultimates",
        display_name="Mirrored Five-Class Ultimates",
        description="Reciprocal demonstrations of every class Ultimate family.",
        source_scenario="mirrored_ultimates",
        replay_file_name="mirrored-five-class-ultimates.marlbg-replay.json",
        metric_report_file_name="mirrored-five-class-ultimates.marlbg-metrics.json",
    ),
)

_SAMPLE_REPLAY_BY_NAME = {sample.name: sample for sample in SAMPLE_REPLAYS}
if len(_SAMPLE_REPLAY_BY_NAME) != len(SAMPLE_REPLAYS):
    raise ValueError("sample replay launch names must be unique")


def iter_sample_replays() -> Iterator[SampleReplayDefinition]:
    """Yield the immutable checked-in registry in launcher display order."""
    return iter(SAMPLE_REPLAYS)


def get_sample_replay(name: str) -> SampleReplayDefinition:
    """Resolve one exact sample name without importing simulator authority."""
    try:
        return _SAMPLE_REPLAY_BY_NAME[name]
    except KeyError as error:
        available = ", ".join(_SAMPLE_REPLAY_BY_NAME)
        raise ValueError(
            f"unknown sample replay {name!r}; choose one of: {available}"
        ) from error


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SampleReplayVerificationError(
                f"sample manifest repeats JSON member {key!r}"
            )
        result[key] = value
    return result


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SampleReplayVerificationError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise SampleReplayVerificationError(
            f"{label} members differ from the V1 manifest contract"
        )


def _require_exact_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise SampleReplayVerificationError(
            f"{label} must be a nonnegative exact integer"
        )
    return value


def _reject_nonfinite_json_constant(value: str) -> object:
    raise SampleReplayVerificationError(
        f"sample manifest forbids nonfinite JSON constant {value}"
    )


def canonical_sample_replay_manifest_json_bytes_v1(
    manifest: Mapping[str, object],
) -> bytes:
    """Return the exact readable, sorted V1 manifest byte representation."""
    return (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_value_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _required_no_follow_flag() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if type(no_follow) is not int or no_follow == 0:
        raise SampleReplayVerificationError(
            "sample verification requires no-follow file opens on this platform"
        )
    return no_follow


def _read_open_bounded_regular_file(
    descriptor: int,
    *,
    max_size_bytes: int,
    label: str,
) -> tuple[bytes, _FileIdentity]:
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SampleReplayVerificationError(f"{label} must be a regular file")
        if metadata.st_size > max_size_bytes:
            raise SampleReplayVerificationError(
                f"{label} exceeds the {max_size_bytes}-byte size limit"
            )
        chunks: list[bytes] = []
        remaining = max_size_bytes + 1
        while remaining > 0:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_SIZE_BYTES, remaining),
            )
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_size_bytes:
            raise SampleReplayVerificationError(
                f"{label} exceeds the {max_size_bytes}-byte size limit"
            )
        final_metadata = os.fstat(descriptor)
        if (
            final_metadata.st_dev != metadata.st_dev
            or final_metadata.st_ino != metadata.st_ino
            or final_metadata.st_mode != metadata.st_mode
            or final_metadata.st_size != metadata.st_size
            or final_metadata.st_mtime_ns != metadata.st_mtime_ns
            or final_metadata.st_ctime_ns != metadata.st_ctime_ns
            or len(payload) != final_metadata.st_size
        ):
            raise SampleReplayVerificationError(f"{label} changed while it was read")
        return payload, _file_identity(final_metadata)
    except OSError as error:
        raise SampleReplayVerificationError(f"{label} could not be read") from error
    finally:
        try:
            os.close(descriptor)
        except OSError as error:
            raise SampleReplayVerificationError(
                f"{label} descriptor could not be closed"
            ) from error


def _read_bounded_regular_entry(
    entry: str | Path,
    *,
    directory_descriptor: int | None,
    max_size_bytes: int,
    label: str,
) -> tuple[bytes, _FileIdentity]:
    no_follow = _required_no_follow_flag()
    flags = os.O_RDONLY | no_follow
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        if directory_descriptor is None:
            descriptor = os.open(entry, flags)
        else:
            descriptor = os.open(entry, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise SampleReplayVerificationError(
            f"{label} could not be opened as a no-follow local file"
        ) from error
    return _read_open_bounded_regular_file(
        descriptor,
        max_size_bytes=max_size_bytes,
        label=label,
    )


def read_bounded_regular_file_v1(
    path: Path,
    *,
    max_size_bytes: int,
    label: str,
) -> bytes:
    """Read a bounded regular file through one no-follow descriptor.

    The descriptor is opened nonblocking where supported, so a crafted FIFO
    cannot stall the verifier.  Every inspection and byte read applies to the
    same descriptor; replacing the directory entry cannot redirect the read.
    """
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        path,
        Path,
    ):
        raise TypeError("sample member path must be pathlib.Path")
    if type(max_size_bytes) is not int or max_size_bytes <= 0:
        raise TypeError("sample member size limit must be a positive exact integer")
    payload, _identity = _read_bounded_regular_entry(
        path,
        directory_descriptor=None,
        max_size_bytes=max_size_bytes,
        label=label,
    )
    return payload


def _open_sample_replay_directory(path: Path) -> int:
    no_follow = _required_no_follow_flag()
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if type(directory_flag) is not int or directory_flag == 0:
        raise SampleReplayVerificationError(
            "sample verification requires directory-descriptor opens"
        )
    flags = os.O_RDONLY | no_follow | directory_flag
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SampleReplayVerificationError(
            "sample replay directory could not be opened as a no-follow directory"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise SampleReplayVerificationError(
                "sample replay directory must be an actual directory"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _close_sample_replay_directory(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError as error:
        raise SampleReplayVerificationError(
            "sample replay directory descriptor could not be closed"
        ) from error


def _file_identity(metadata: os.stat_result) -> _FileIdentity:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(descriptor: int) -> _FileIdentity:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise SampleReplayVerificationError(
            "sample replay directory could not be inspected"
        ) from error
    return _file_identity(metadata)


def _require_unchanged_directory(
    descriptor: int,
    expected_identity: _FileIdentity,
) -> None:
    if _directory_identity(descriptor) != expected_identity:
        raise SampleReplayVerificationError(
            "sample replay directory changed during verification"
        )


def _require_unchanged_regular_entry(
    directory_descriptor: int,
    entry_name: str,
    expected_identity: _FileIdentity,
    *,
    label: str,
) -> None:
    no_follow = _required_no_follow_flag()
    flags = os.O_RDONLY | no_follow
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(entry_name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise SampleReplayVerificationError(
            f"{label} changed during verification"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SampleReplayVerificationError(f"{label} changed during verification")
        if _file_identity(metadata) != expected_identity:
            raise SampleReplayVerificationError(f"{label} changed during verification")
    except OSError as error:
        raise SampleReplayVerificationError(
            f"{label} changed during verification"
        ) from error
    finally:
        try:
            os.close(descriptor)
        except OSError as error:
            raise SampleReplayVerificationError(
                f"{label} descriptor could not be closed"
            ) from error


def _validated_member_metadata(
    member: object,
    *,
    expected_file_name: str,
    max_size_bytes: int,
    label: str,
) -> Mapping[str, object]:
    row = _require_mapping(member, label=label)
    _require_exact_keys(row, {"file", "byte_length", "sha256"}, label=label)
    if row["file"] != expected_file_name:
        raise SampleReplayVerificationError(f"{label} filename is not registered")
    byte_length = _require_exact_nonnegative_int(
        row["byte_length"],
        label=f"{label} byte length",
    )
    if byte_length > max_size_bytes:
        raise SampleReplayVerificationError(f"{label} declares an oversized member")
    sha256 = row["sha256"]
    if type(sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise SampleReplayVerificationError(
            f"{label} SHA-256 must be lowercase hexadecimal"
        )
    return row


def _verified_member_bytes(
    member: object,
    *,
    expected_file_name: str,
    directory_descriptor: int,
    max_size_bytes: int,
    label: str,
) -> tuple[bytes, _FileIdentity]:
    row = _validated_member_metadata(
        member,
        expected_file_name=expected_file_name,
        max_size_bytes=max_size_bytes,
        label=label,
    )
    payload, identity = _read_bounded_regular_entry(
        expected_file_name,
        directory_descriptor=directory_descriptor,
        max_size_bytes=max_size_bytes,
        label=label,
    )
    if row["byte_length"] != len(payload):
        raise SampleReplayVerificationError(f"{label} byte length mismatch")
    if row["sha256"] != hashlib.sha256(payload).hexdigest():
        raise SampleReplayVerificationError(f"{label} SHA-256 mismatch")
    return payload, identity


def _validate_manifest_sample_row(
    sample: SampleReplayDefinition,
    raw_row: object,
) -> Mapping[str, object]:
    row = _require_mapping(raw_row, label=f"sample {sample.name}")
    _require_exact_keys(
        row,
        {
            "name",
            "display_name",
            "description",
            "source_scenario",
            "seed",
            "transition_count",
            "frame_count",
            "event_kind_coverage",
            "replay",
            "metric_report",
        },
        label=f"sample {sample.name}",
    )
    seed = _require_exact_nonnegative_int(
        row["seed"],
        label=f"sample {sample.name} seed",
    )
    transition_count = _require_exact_nonnegative_int(
        row["transition_count"],
        label=f"sample {sample.name} transition count",
    )
    frame_count = _require_exact_nonnegative_int(
        row["frame_count"],
        label=f"sample {sample.name} frame count",
    )
    coverage = row["event_kind_coverage"]
    if type(coverage) is not list:
        raise SampleReplayVerificationError(
            f"sample {sample.name} event coverage must be a JSON string array"
        )
    untyped_coverage = cast(list[object], coverage)
    if any(type(value) is not str for value in untyped_coverage):
        raise SampleReplayVerificationError(
            f"sample {sample.name} event coverage must be a JSON string array"
        )
    coverage_values = cast(list[str], coverage)
    if coverage_values != sorted(set(coverage_values)):
        raise SampleReplayVerificationError(
            f"sample {sample.name} event coverage must be sorted and unique"
        )
    if (
        row["name"] != sample.name
        or row["display_name"] != sample.display_name
        or row["description"] != sample.description
        or row["source_scenario"] != sample.source_scenario
        or seed != sample.seed
        or frame_count != transition_count + 1
    ):
        raise SampleReplayVerificationError(
            f"sample {sample.name!r} manifest metadata differs from the registry"
        )
    _validated_member_metadata(
        row["replay"],
        expected_file_name=sample.replay_file_name,
        max_size_bytes=SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
        label=f"sample {sample.name} replay",
    )
    _validated_member_metadata(
        row["metric_report"],
        expected_file_name=sample.metric_report_file_name,
        max_size_bytes=SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
        label=f"sample {sample.name} metric report",
    )
    return row


def _read_sample_replay_manifest_from_directory_descriptor(
    directory_descriptor: int,
) -> tuple[
    Mapping[str, object],
    tuple[Mapping[str, object], ...],
    _FileIdentity,
]:
    manifest_payload, manifest_identity = _read_bounded_regular_entry(
        SAMPLE_REPLAY_MANIFEST_PATH.name,
        directory_descriptor=directory_descriptor,
        max_size_bytes=SAMPLE_REPLAY_MAX_MANIFEST_SIZE_BYTES,
        label="sample replay manifest",
    )
    try:
        raw_manifest: object = json.loads(
            manifest_payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except SampleReplayVerificationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SampleReplayVerificationError(
            "sample replay manifest could not be read as strict JSON"
        ) from error
    manifest = _require_mapping(raw_manifest, label="sample manifest")
    try:
        canonical_manifest = canonical_sample_replay_manifest_json_bytes_v1(manifest)
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise SampleReplayVerificationError(
            "sample replay manifest could not be canonicalized"
        ) from error
    if canonical_manifest != manifest_payload:
        raise SampleReplayVerificationError(
            "sample replay manifest bytes are not canonical"
        )
    _require_exact_keys(
        manifest,
        {"schema_id", "schema_version", "generator_id", "demo_provenance", "samples"},
        label="sample manifest",
    )
    schema_version = _require_exact_nonnegative_int(
        manifest["schema_version"],
        label="sample manifest schema version",
    )
    if (
        manifest["schema_id"] != SAMPLE_REPLAY_MANIFEST_SCHEMA_ID
        or schema_version != SAMPLE_REPLAY_MANIFEST_SCHEMA_VERSION
        or manifest["generator_id"] != SAMPLE_REPLAY_GENERATOR_ID
    ):
        raise SampleReplayVerificationError(
            "sample replay manifest identity is unsupported"
        )
    provenance = _require_mapping(
        manifest["demo_provenance"],
        label="demo provenance",
    )
    _require_exact_keys(
        provenance,
        {
            "official",
            "benchmark_eligible",
            "source_tree_attestation",
            "host_attestation",
            "policy_execution_included",
            "notice",
            "code_revision",
            "runtime_provenance",
        },
        label="demo provenance",
    )
    if (
        provenance["official"] is not False
        or provenance["benchmark_eligible"] is not False
        or provenance["source_tree_attestation"] is not False
        or provenance["host_attestation"] is not False
        or provenance["policy_execution_included"] is not False
        or provenance["notice"] != SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE
    ):
        raise SampleReplayVerificationError(
            "sample manifest must retain explicit unofficial demo provenance"
        )
    raw_code_revision = _require_mapping(
        provenance["code_revision"], label="demo code revision"
    )
    raw_runtime_provenance = _require_mapping(
        provenance["runtime_provenance"], label="demo runtime provenance"
    )
    from pydantic import ValidationError

    from marl_battlegrounds.evaluation.models import CodeRevisionV1
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1

    try:
        code_revision = CodeRevisionV1.model_validate(raw_code_revision)
        runtime_provenance = RuntimeProvenanceV1.model_validate_json(
            _canonical_json_value_bytes(raw_runtime_provenance)
        )
    except ValidationError as error:
        raise SampleReplayVerificationError(
            "sample manifest provenance does not satisfy the strict V1 schemas"
        ) from error
    if (
        _canonical_json_value_bytes(code_revision.model_dump(mode="json"))
        != _canonical_json_value_bytes(raw_code_revision)
        or _canonical_json_value_bytes(runtime_provenance.model_dump(mode="json"))
        != _canonical_json_value_bytes(raw_runtime_provenance)
        or runtime_provenance.backend != "cpu"
        or runtime_provenance.policy_execution_included
    ):
        raise SampleReplayVerificationError(
            "sample manifest provenance is not a truthful CPU demo capture"
        )
    raw_samples = manifest["samples"]
    if type(raw_samples) is not list:
        raise SampleReplayVerificationError("sample manifest rows must be a JSON array")
    sample_values = cast(list[object], raw_samples)
    if len(sample_values) != len(SAMPLE_REPLAYS):
        raise SampleReplayVerificationError("sample manifest registry size mismatch")
    rows = tuple(
        _validate_manifest_sample_row(sample, raw_row)
        for sample, raw_row in zip(
            SAMPLE_REPLAYS,
            sample_values,
            strict=True,
        )
    )
    return manifest, rows, manifest_identity


def read_sample_replay_manifest_v1(
    directory: Path = SAMPLE_REPLAY_DIRECTORY,
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    """Read one strict V1 manifest snapshot from a held directory descriptor."""
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        directory,
        Path,
    ):
        raise TypeError("sample replay directory must be pathlib.Path")
    directory_descriptor = _open_sample_replay_directory(directory)
    try:
        directory_identity = _directory_identity(directory_descriptor)
        manifest, rows, manifest_identity = (
            _read_sample_replay_manifest_from_directory_descriptor(directory_descriptor)
        )
        _require_unchanged_regular_entry(
            directory_descriptor,
            SAMPLE_REPLAY_MANIFEST_PATH.name,
            manifest_identity,
            label="sample replay manifest",
        )
        _require_unchanged_directory(directory_descriptor, directory_identity)
        return manifest, rows
    finally:
        _close_sample_replay_directory(directory_descriptor)


def _write_private_member(path: Path, payload: bytes) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if type(no_follow) is not int or no_follow == 0:
        raise SampleReplayVerificationError(
            "sample verification requires no-follow file creation"
        )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("private sample member write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_verified_sample_replay_from_snapshot(
    sample: SampleReplayDefinition,
    *,
    manifest: Mapping[str, object],
    row: Mapping[str, object],
    directory_descriptor: int,
) -> tuple[LoadedReplayBundleV1, tuple[tuple[str, _FileIdentity, str], ...]]:
    name = sample.name
    replay_payload, replay_identity = _verified_member_bytes(
        row["replay"],
        expected_file_name=sample.replay_file_name,
        directory_descriptor=directory_descriptor,
        max_size_bytes=SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
        label=f"sample {name} replay",
    )
    metric_payload, metric_identity = _verified_member_bytes(
        row["metric_report"],
        expected_file_name=sample.metric_report_file_name,
        directory_descriptor=directory_descriptor,
        max_size_bytes=SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
        label=f"sample {name} metric report",
    )

    from marl_battlegrounds.evaluation.replay_io import (
        ReplayLoadError,
        canonical_metric_report_artifact_json_bytes_v1,
        canonical_replay_json_bytes_v1,
        load_replay_bundle_v1,
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="marl-sample-replay-load-"
        ) as temporary:
            private_directory = Path(temporary)
            private_replay_path = private_directory / sample.replay_file_name
            private_metric_path = private_directory / sample.metric_report_file_name
            _write_private_member(private_metric_path, metric_payload)
            _write_private_member(private_replay_path, replay_payload)
            loaded = load_replay_bundle_v1(
                private_replay_path,
                require_metric_report=True,
                max_file_size_bytes=SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
            )
    except ReplayLoadError as error:
        raise SampleReplayVerificationError(
            f"sample {name!r} failed public replay validation ({error.code})"
        ) from error
    except OSError as error:
        raise SampleReplayVerificationError(
            f"sample {name!r} could not enter the private validation boundary"
        ) from error
    if loaded.status != "complete" or loaded.metric_report_artifact is None:
        raise SampleReplayVerificationError(
            f"sample {name!r} requires a complete replay and metric sidecar"
        )
    if canonical_replay_json_bytes_v1(loaded.replay) != replay_payload:
        raise SampleReplayVerificationError(
            f"sample {name!r} replay changed during public validation"
        )
    if (
        canonical_metric_report_artifact_json_bytes_v1(loaded.metric_report_artifact)
        != metric_payload
    ):
        raise SampleReplayVerificationError(
            f"sample {name!r} metric report changed during public validation"
        )
    replay = loaded.replay
    scenario_rows = tuple(
        aggregation.value
        for aggregation in replay.header.context.aggregation_keys
        if aggregation.name == "scenario"
    )
    event_kinds = sorted(
        {
            event.event_type
            for transition in replay.transitions
            for event in transition.events
        }
    )
    if (
        len(scenario_rows) != 1
        or scenario_rows[0] != sample.source_scenario
        or row["transition_count"] != len(replay.transitions)
        or row["frame_count"] != len(replay.frames)
        or row["event_kind_coverage"] != event_kinds
    ):
        raise SampleReplayVerificationError(
            f"sample {name!r} scientific manifest facts are stale"
        )
    provenance = cast(Mapping[str, object], manifest["demo_provenance"])
    if _canonical_json_value_bytes(
        replay.header.context.code_revision.model_dump(mode="json")
    ) != _canonical_json_value_bytes(
        provenance["code_revision"]
    ) or _canonical_json_value_bytes(
        replay.header.runtime_provenance.model_dump(mode="json")
    ) != _canonical_json_value_bytes(provenance["runtime_provenance"]):
        raise SampleReplayVerificationError(
            f"sample {name!r} artifact provenance differs from its manifest"
        )
    _require_unchanged_regular_entry(
        directory_descriptor,
        sample.replay_file_name,
        replay_identity,
        label=f"sample {name} replay",
    )
    _require_unchanged_regular_entry(
        directory_descriptor,
        sample.metric_report_file_name,
        metric_identity,
        label=f"sample {name} metric report",
    )
    return loaded, (
        (sample.replay_file_name, replay_identity, f"sample {name} replay"),
        (
            sample.metric_report_file_name,
            metric_identity,
            f"sample {name} metric report",
        ),
    )


def load_verified_sample_replay(
    name: str,
    *,
    directory: Path = SAMPLE_REPLAY_DIRECTORY,
) -> LoadedReplayBundleV1:
    """Return one complete in-memory bundle bound to one manifest snapshot."""
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        directory,
        Path,
    ):
        raise TypeError("sample replay directory must be pathlib.Path")
    sample = get_sample_replay(name)
    directory_descriptor = _open_sample_replay_directory(directory)
    try:
        directory_identity = _directory_identity(directory_descriptor)
        manifest, rows, manifest_identity = (
            _read_sample_replay_manifest_from_directory_descriptor(directory_descriptor)
        )
        row = rows[tuple(entry.name for entry in SAMPLE_REPLAYS).index(name)]
        loaded, member_identities = _load_verified_sample_replay_from_snapshot(
            sample,
            manifest=manifest,
            row=row,
            directory_descriptor=directory_descriptor,
        )
        for entry_name, identity, label in member_identities:
            _require_unchanged_regular_entry(
                directory_descriptor,
                entry_name,
                identity,
                label=label,
            )
        _require_unchanged_regular_entry(
            directory_descriptor,
            SAMPLE_REPLAY_MANIFEST_PATH.name,
            manifest_identity,
            label="sample replay manifest",
        )
        _require_unchanged_directory(directory_descriptor, directory_identity)
        return loaded
    finally:
        _close_sample_replay_directory(directory_descriptor)


def load_verified_sample_replay_set_v1(
    directory: Path = SAMPLE_REPLAY_DIRECTORY,
) -> VerifiedSampleReplaySetV1:
    """Load the complete registered set against one immutable directory view."""
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        directory,
        Path,
    ):
        raise TypeError("sample replay directory must be pathlib.Path")
    directory_descriptor = _open_sample_replay_directory(directory)
    try:
        directory_identity = _directory_identity(directory_descriptor)
        manifest, rows, manifest_identity = (
            _read_sample_replay_manifest_from_directory_descriptor(directory_descriptor)
        )
        loaded_rows = tuple(
            _load_verified_sample_replay_from_snapshot(
                sample,
                manifest=manifest,
                row=row,
                directory_descriptor=directory_descriptor,
            )
            for sample, row in zip(SAMPLE_REPLAYS, rows, strict=True)
        )
        bundles = tuple(loaded for loaded, _identities in loaded_rows)
        member_identities = tuple(
            identity_row
            for _loaded, identities in loaded_rows
            for identity_row in identities
        )
        try:
            file_names = frozenset(os.listdir(directory_descriptor))
        except OSError as error:
            raise SampleReplayVerificationError(
                "sample replay directory could not be enumerated"
            ) from error
        for entry_name, identity, label in member_identities:
            _require_unchanged_regular_entry(
                directory_descriptor,
                entry_name,
                identity,
                label=label,
            )
        _require_unchanged_regular_entry(
            directory_descriptor,
            SAMPLE_REPLAY_MANIFEST_PATH.name,
            manifest_identity,
            label="sample replay manifest",
        )
        _require_unchanged_directory(directory_descriptor, directory_identity)
        return VerifiedSampleReplaySetV1(
            manifest=manifest,
            sample_rows=rows,
            bundles=bundles,
            file_names=file_names,
        )
    finally:
        _close_sample_replay_directory(directory_descriptor)


__all__ = [
    "SAMPLE_REPLAYS",
    "SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE",
    "SAMPLE_REPLAY_DIRECTORY",
    "SAMPLE_REPLAY_GENERATOR_ID",
    "SAMPLE_REPLAY_MANIFEST_PATH",
    "SAMPLE_REPLAY_MANIFEST_SCHEMA_ID",
    "SAMPLE_REPLAY_MANIFEST_SCHEMA_VERSION",
    "SAMPLE_REPLAY_MAX_MANIFEST_SIZE_BYTES",
    "SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES",
    "SampleReplayDefinition",
    "SampleReplayVerificationError",
    "VerifiedSampleReplaySetV1",
    "canonical_sample_replay_manifest_json_bytes_v1",
    "get_sample_replay",
    "iter_sample_replays",
    "load_verified_sample_replay",
    "load_verified_sample_replay_set_v1",
    "read_bounded_regular_file_v1",
    "read_sample_replay_manifest_v1",
]
