"""Generate or verify the checked-in Visual Debugger sample replay bundle set.

Generation deliberately traverses the same scripted session, recording
coordinator, canonical persistence, and public replay loader used by the live
debugger.  Scientific provenance records the actual source and CPU runtime;
separate manifest flags keep these episodes explicitly unofficial demos.
"""

# This file is also a directly executable repository script, so it establishes
# the repository import root before loading project modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import jax

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from marl_battlegrounds.evaluation.models import CodeRevisionV1
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.replay_io import (
    LoadedReplayBundleV1,
    canonical_metric_report_artifact_json_bytes_v1,
    canonical_replay_json_bytes_v1,
    load_replay_bundle_v1,
    preflight_replay_bundle_destination_v1,
)
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.evaluation_bridge import (
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.protocol import (
    CommandRequestV1,
    CommandResponseV2,
    KeyboardCommandV1,
)
from scripts.dev.visual_debugger.recording import (
    DebuggerReplayRecorderV1,
    build_debugger_recording_specification_v1,
)
from scripts.dev.visual_debugger.recording_coordinator import (
    RecordingDebuggerCoordinator,
)
from scripts.dev.visual_debugger.revision import discover_debugger_code_revision_v1
from scripts.dev.visual_debugger.runtime_provenance import (
    capture_debugger_runtime_provenance_v1,
)
from scripts.dev.visual_debugger.sample_replays import (
    SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE,
    SAMPLE_REPLAY_DIRECTORY,
    SAMPLE_REPLAY_GENERATOR_ID,
    SAMPLE_REPLAY_MANIFEST_PATH,
    SAMPLE_REPLAY_MANIFEST_SCHEMA_ID,
    SAMPLE_REPLAY_MANIFEST_SCHEMA_VERSION,
    SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
    SAMPLE_REPLAYS,
    SampleReplayDefinition,
    SampleReplayVerificationError,
    canonical_sample_replay_manifest_json_bytes_v1,
    load_verified_sample_replay_set_v1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService


def _path_entry_exists(path: Path) -> bool:
    """Return lexical entry existence, including dangling symbolic links."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _refuse_existing_output(path: Path) -> None:
    """Fence every pre-existing directory entry before no-clobber publication."""
    if _path_entry_exists(path):
        raise FileExistsError(
            f"refusing to overwrite existing sample directory: {path}"
        )


def _atomic_publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a complete directory without replacing any entry.

    Linux ``renameat2(RENAME_NOREPLACE)`` is the required primitive.  Failing
    closed on hosts without it preserves the generator's no-clobber contract.
    """
    if sys.platform != "linux":
        raise RuntimeError(
            "atomic sample publication requires Linux renameat2 no-replace support"
        )
    try:
        library = ctypes.CDLL(None, use_errno=True)
        rename_at2 = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
            use_errno=True,
        )(("renameat2", library))
    except (AttributeError, OSError) as error:
        raise RuntimeError(
            "atomic sample publication requires renameat2 no-replace support"
        ) from error
    at_current_working_directory = -100
    rename_noreplace = 1
    result = rename_at2(
        at_current_working_directory,
        os.fsencode(source),
        at_current_working_directory,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"refusing to overwrite existing sample directory: {destination}",
            destination,
        )
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
        raise RuntimeError(
            "atomic sample publication requires renameat2 no-replace support"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination,
    )


def _require_cpu_backend() -> None:
    """Reject generation when the actual selected JAX backend is not CPU."""
    backend = jax.default_backend()
    if backend != "cpu":
        raise RuntimeError(
            "sample replay generation requires the JAX default backend to be cpu; "
            f"actual backend is {backend!r}"
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_member(
    *,
    file_name: str,
    payload: bytes,
) -> dict[str, object]:
    return {
        "file": file_name,
        "byte_length": len(payload),
        "sha256": _sha256(payload),
    }


def _scenario_name_from_replay(loaded: LoadedReplayBundleV1) -> str:
    replay = loaded.replay
    rows = tuple(
        row.value
        for row in replay.header.context.aggregation_keys
        if row.name == "scenario"
    )
    if len(rows) != 1:
        raise SampleReplayVerificationError(
            "sample replay requires exactly one scenario aggregation key"
        )
    return rows[0]


def _record_one_sample(
    sample: SampleReplayDefinition,
    *,
    output_directory: Path,
    code_revision: CodeRevisionV1,
    runtime_provenance: RuntimeProvenanceV1,
) -> dict[str, object]:
    scenario = get_scenario(sample.source_scenario)
    if scenario.mode != "scripted" or scenario.audience != "researcher":
        raise ValueError("sample replay sources must be scripted researcher scenarios")
    if not scenario.frames:
        raise ValueError("sample replay source scenarios must contain transitions")

    launch = build_debugger_evaluation_launch_specification_v1(
        root_seed=sample.seed,
        code_revision=code_revision,
        capture_profile="evaluation_metric_complete",
    )
    session = create_session(
        scenario,
        seed=sample.seed,
        evaluation_launch_specification=launch,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    recorder = DebuggerReplayRecorderV1(
        specification=build_debugger_recording_specification_v1(
            action_source_kind="scripted",
            runtime_provenance=runtime_provenance,
        ),
        destination=preflight_replay_bundle_destination_v1(
            sample.replay_path(output_directory)
        ),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    coordinator = RecordingDebuggerCoordinator(
        DebuggerService(
            session,
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
            session_id=f"sample-replay-generator-{sample.name}",
            recorder=recorder,
        )
    )

    for frame_index in range(len(scenario.frames)):
        result = coordinator.apply_command(
            CommandRequestV1(
                client_id="sample-replay-generator",
                command_id=f"{sample.name}-transition-{frame_index}",
                base_revision=coordinator.service.revision,
                command=KeyboardCommandV1(key="n"),
            )
        )
        if result.outcome != "response" or not isinstance(
            result.payload,
            CommandResponseV2,
        ):
            raise RuntimeError(
                f"sample {sample.name!r} did not accept scripted transition "
                f"{frame_index}"
            )
        if result.payload.result != "applied":
            raise RuntimeError(
                f"sample {sample.name!r} transition {frame_index} was not applied"
            )

    if recorder.lifecycle != "saved":
        raise RuntimeError(
            f"sample {sample.name!r} did not reach a complete saved recording"
        )
    saved = recorder.saved_bundle
    if saved is None:
        raise RuntimeError(f"sample {sample.name!r} has no saved artifact pair")
    if saved.replay_path != sample.replay_path(
        output_directory
    ) or saved.metric_report_path != sample.metric_report_path(output_directory):
        raise RuntimeError("recording publisher returned an unexpected sample path")

    loaded = load_replay_bundle_v1(
        saved.replay_path,
        require_metric_report=True,
        max_file_size_bytes=SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
    )
    if loaded.status != "complete" or loaded.metric_report_artifact is None:
        raise RuntimeError("public sample reload did not resolve a complete pair")
    replay = loaded.replay
    transition_count = len(replay.transitions)
    if transition_count != len(scenario.frames):
        raise RuntimeError("sample transition count differs from its scripted source")
    if _scenario_name_from_replay(loaded) != sample.source_scenario:
        raise RuntimeError("sample replay lost its source-scenario identity")

    replay_payload = canonical_replay_json_bytes_v1(replay)
    metric_payload = canonical_metric_report_artifact_json_bytes_v1(
        loaded.metric_report_artifact
    )
    event_kinds = sorted(
        {
            event.event_type
            for transition in replay.transitions
            for event in transition.events
        }
    )
    return {
        "name": sample.name,
        "display_name": sample.display_name,
        "description": sample.description,
        "source_scenario": sample.source_scenario,
        "seed": sample.seed,
        "transition_count": transition_count,
        "frame_count": len(replay.frames),
        "event_kind_coverage": event_kinds,
        "replay": _artifact_member(
            file_name=sample.replay_file_name,
            payload=replay_payload,
        ),
        "metric_report": _artifact_member(
            file_name=sample.metric_report_file_name,
            payload=metric_payload,
        ),
    }


def _manifest_payload(
    *,
    code_revision: CodeRevisionV1,
    runtime_provenance: RuntimeProvenanceV1,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_id": SAMPLE_REPLAY_MANIFEST_SCHEMA_ID,
        "schema_version": SAMPLE_REPLAY_MANIFEST_SCHEMA_VERSION,
        "generator_id": SAMPLE_REPLAY_GENERATOR_ID,
        "demo_provenance": {
            "official": False,
            "benchmark_eligible": False,
            "source_tree_attestation": False,
            "host_attestation": False,
            "policy_execution_included": False,
            "notice": SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE,
            "code_revision": code_revision.model_dump(mode="json"),
            "runtime_provenance": runtime_provenance.model_dump(mode="json"),
        },
        "samples": samples,
    }


def generate_sample_replays(
    output_directory: Path,
) -> dict[str, object]:
    """Atomically publish one new sample set, refusing every existing target."""
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        output_directory,
        Path,
    ):
        raise TypeError("output_directory must be pathlib.Path")
    _refuse_existing_output(output_directory)
    _require_cpu_backend()
    resolved_revision = discover_debugger_code_revision_v1(
        _REPOSITORY_ROOT,
        package_version="0.0.0",
    )
    runtime_provenance = capture_debugger_runtime_provenance_v1(resolved_revision)
    if runtime_provenance.backend != "cpu":
        raise RuntimeError("sample replay runtime provenance must capture CPU")
    if runtime_provenance.policy_execution_included:
        raise RuntimeError(
            "sample replay runtime provenance must exclude policy execution"
        )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.tmp-",
            dir=output_directory.parent,
        )
    )
    published = False
    try:
        samples = [
            _record_one_sample(
                sample,
                output_directory=staging_directory,
                code_revision=resolved_revision,
                runtime_provenance=runtime_provenance,
            )
            for sample in SAMPLE_REPLAYS
        ]
        manifest = _manifest_payload(
            code_revision=resolved_revision,
            runtime_provenance=runtime_provenance,
            samples=samples,
        )
        (staging_directory / SAMPLE_REPLAY_MANIFEST_PATH.name).write_bytes(
            canonical_sample_replay_manifest_json_bytes_v1(manifest)
        )
        verify_sample_replays(staging_directory)
        _refuse_existing_output(output_directory)
        _atomic_publish_directory_noreplace(staging_directory, output_directory)
        published = True
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging_directory, ignore_errors=True)


def verify_sample_replays(
    directory: Path = SAMPLE_REPLAY_DIRECTORY,
) -> dict[str, object]:
    """Verify manifest hashes, canonical bytes, semantic joins, and provenance."""
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        directory,
        Path,
    ):
        raise TypeError("directory must be pathlib.Path")
    verified_set = load_verified_sample_replay_set_v1(directory)
    manifest = verified_set.manifest
    sample_rows = verified_set.sample_rows

    expected_files = {SAMPLE_REPLAY_MANIFEST_PATH.name}
    for sample, row, loaded in zip(
        SAMPLE_REPLAYS,
        sample_rows,
        verified_set.bundles,
        strict=True,
    ):
        expected_files.update((sample.replay_file_name, sample.metric_report_file_name))
        replay = loaded.replay
        transition_count = len(replay.transitions)
        event_kinds = sorted(
            {
                event.event_type
                for transition in replay.transitions
                for event in transition.events
            }
        )
        if (
            row["transition_count"] != transition_count
            or row["frame_count"] != len(replay.frames)
            or row["event_kind_coverage"] != event_kinds
            or _scenario_name_from_replay(loaded) != sample.source_scenario
        ):
            raise SampleReplayVerificationError(
                f"sample {sample.name!r} scientific manifest facts are stale"
            )

    if verified_set.file_names != frozenset(expected_files):
        raise SampleReplayVerificationError(
            "sample replay directory contains missing or unregistered files"
        )
    return dict(manifest)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit no-overwrite generator/check command."""
    parser = argparse.ArgumentParser(
        description="Generate or verify deterministic Visual Debugger sample replays.",
        allow_abbrev=False,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--generate",
        action="store_true",
        help="create one new sample directory; existing targets are refused",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify checked-in hashes, public reload, and canonical byte stability",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=SAMPLE_REPLAY_DIRECTORY,
        help="sample directory (default: examples/replays/v1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run generation or verification with concise command-line diagnostics."""
    options = build_parser().parse_args(argv)
    try:
        if options.generate:
            manifest = generate_sample_replays(options.output_directory)
            action = "generated"
        else:
            manifest = verify_sample_replays(options.output_directory)
            action = "verified"
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{action} {len(cast(list[object], manifest['samples']))} sample replays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
