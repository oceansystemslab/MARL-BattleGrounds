"""Canonical replay persistence, hostile-input, and loaded-parity proofs."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
)

from marl_battlegrounds.evaluation import replay_io
from marl_battlegrounds.evaluation.metrics import (
    EvaluationEpisodeObserverV1,
    EvaluationMetricReportV1,
    build_evaluation_observer_v1,
)
from marl_battlegrounds.evaluation.models import (
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactV1,
    ReplayBundleV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
    iter_replay_transition_views_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1,
    METRIC_REPORT_FILE_SUFFIX_V1,
    REPLAY_FILE_SUFFIX_V1,
    LoadedReplayBundleV1,
    ReplayIOErrorCodeV1,
    ReplayLoadError,
    ReplaySaveError,
    canonical_metric_report_artifact_json_bytes_v1,
    canonical_replay_json_bytes_v1,
    load_replay_artifact_v1,
    load_replay_bundle_v1,
    save_replay_bundle_v1,
)


@dataclass(frozen=True, slots=True)
class _ReplayIoCase:
    """One live CP2/CP3 trajectory and its immutable Step 6 bundle."""

    trajectory: CapturedEvaluationTrajectory
    observer: EvaluationEpisodeObserverV1
    report: EvaluationMetricReportV1
    bundle: ReplayBundleV1


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.13.0",
        package_version="0.0.0",
        jax_version="0.7.0",
        jaxlib_version="0.7.0",
        numpy_version="2.3.0",
        pydantic_version="2.11.0",
        platform="linux",
        machine="x86_64",
        backend="cpu",
        device="generic-cpu",
        precision="float32",
        environment_count=1,
        batch_shape=(1,),
        policy_execution_included=False,
    )


def _build_io_case(
    *,
    transition_count: int = 2,
    episode_id: str = "episode-001",
) -> _ReplayIoCase:
    trajectory = captured_evaluation_trajectory(
        transition_count=transition_count,
        expected_horizon=transition_count,
        episode_id=episode_id,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    for transition, successor_frame in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor_frame)
    report = observer.finalize(completion_state="complete")
    bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )
    return _ReplayIoCase(
        trajectory=trajectory,
        observer=observer,
        report=report,
        bundle=bundle,
    )


@pytest.fixture(scope="module")
def replay_io_case() -> _ReplayIoCase:
    """Build one real public-core trajectory for every persistence proof."""
    return _build_io_case()


def _replay_path(parent: Path, *, stem: str = "episode") -> Path:
    return parent / f"{stem}{REPLAY_FILE_SUFFIX_V1}"


def _metric_path(parent: Path, *, stem: str = "episode") -> Path:
    return parent / f"{stem}{METRIC_REPORT_FILE_SUFFIX_V1}"


def _write_canonical_bundle(
    parent: Path,
    bundle: ReplayBundleV1,
    *,
    stem: str = "episode",
) -> tuple[Path, Path]:
    replay_path = _replay_path(parent, stem=stem)
    metric_path = _metric_path(parent, stem=stem)
    replay_path.write_bytes(canonical_replay_json_bytes_v1(bundle.replay))
    metric_path.write_bytes(
        canonical_metric_report_artifact_json_bytes_v1(bundle.metric_report_artifact)
    )
    return replay_path, metric_path


def _readdress_metric_reference(
    replay: ReplayArtifactV1,
    **reference_updates: object,
) -> ReplayArtifactV1:
    reference = replay.metric_report_reference.model_copy(
        update=reference_updates,
    )
    payload = replay.model_dump(
        mode="python",
        exclude={"canonical_digest_sha256"},
    )
    payload["metric_report_reference"] = reference
    return ReplayArtifactV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _assert_load_error(
    path: Path,
    code: str,
    **load_options: object,
) -> ReplayLoadError:
    with pytest.raises(ReplayLoadError) as caught:
        load_replay_artifact_v1(path, **load_options)  # type: ignore[arg-type]
    assert caught.value.code == code
    return caught.value


def test_canonical_bytes_are_exact_and_save_load_save_is_stable(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    """Both sidecars persist exactly once and survive a byte-identical cycle."""
    replay_bytes = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    metric_bytes = canonical_metric_report_artifact_json_bytes_v1(
        replay_io_case.bundle.metric_report_artifact
    )
    assert replay_bytes == canonical_json_bytes(replay_io_case.bundle.replay)
    assert metric_bytes == canonical_json_bytes(
        replay_io_case.bundle.metric_report_artifact
    )

    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    first_path = _replay_path(first_directory)
    saved = save_replay_bundle_v1(replay_io_case.bundle, first_path)

    assert saved.replay_path == first_path
    assert saved.metric_report_path == _metric_path(first_directory)
    assert saved.replay_byte_length == len(replay_bytes)
    assert saved.metric_report_byte_length == len(metric_bytes)
    assert saved.metric_report_reused is False
    assert first_path.read_bytes() == replay_bytes
    assert saved.metric_report_path.read_bytes() == metric_bytes

    loaded = load_replay_bundle_v1(first_path, require_metric_report=True)
    assert loaded.status == "complete"
    assert loaded.replay == replay_io_case.bundle.replay
    assert loaded.metric_report_artifact == (
        replay_io_case.bundle.metric_report_artifact
    )
    assert loaded.metric_report_artifact is not None
    loaded_bundle = ReplayBundleV1(
        replay=loaded.replay,
        metric_report_artifact=loaded.metric_report_artifact,
    )
    second_path = _replay_path(second_directory)
    save_replay_bundle_v1(loaded_bundle, second_path)

    assert second_path.read_bytes() == first_path.read_bytes()
    assert _metric_path(second_directory).read_bytes() == metric_bytes


def test_relative_parent_traversal_remains_descriptor_bound_and_supported(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_directory = tmp_path / "captures"
    capture_directory.mkdir()
    monkeypatch.chdir(tmp_path)
    replay_path = (
        Path("captures") / ".." / "captures" / (f"episode{REPLAY_FILE_SUFFIX_V1}")
    )

    save_replay_bundle_v1(replay_io_case.bundle, replay_path)
    loaded = load_replay_bundle_v1(replay_path, require_metric_report=True)

    assert loaded.replay == replay_io_case.bundle.replay
    assert loaded.metric_report_artifact == replay_io_case.bundle.metric_report_artifact


def test_maximum_portable_artifact_basename_does_not_expand_temporary_name(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    stem_length = 255 - max(
        len(REPLAY_FILE_SUFFIX_V1),
        len(METRIC_REPORT_FILE_SUFFIX_V1),
    )
    replay_path = _replay_path(tmp_path, stem="e" * stem_length)

    saved = save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert saved.replay_path.name == replay_path.name
    assert saved.replay_path.exists()
    assert saved.metric_report_path.exists()


def test_missing_metric_sidecar_is_renderable_unless_explicitly_required(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    replay_path = _replay_path(tmp_path)
    replay_path.write_bytes(
        canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    )

    replay = load_replay_artifact_v1(replay_path)
    loaded = load_replay_bundle_v1(replay_path)

    assert replay == replay_io_case.bundle.replay
    assert loaded == LoadedReplayBundleV1(
        replay=replay_io_case.bundle.replay,
        metric_report_artifact=None,
        status="metric_report_missing",
    )
    with pytest.raises(ReplayLoadError) as caught:
        load_replay_bundle_v1(replay_path, require_metric_report=True)
    assert caught.value.code == "metric_report_missing"
    assert caught.value.path == _metric_path(tmp_path)


def test_loaded_views_report_and_constant_time_cursor_equal_live_inputs(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One load validates once; cursor reads never repeat the O(T) pass."""
    replay_path = _replay_path(tmp_path)
    save_replay_bundle_v1(replay_io_case.bundle, replay_path)
    validation_count = 0
    real_validator = replay_io.validate_replay_artifact_v1

    def counted_validator(artifact: ReplayArtifactV1) -> None:
        nonlocal validation_count
        validation_count += 1
        real_validator(artifact)

    monkeypatch.setattr(replay_io, "validate_replay_artifact_v1", counted_validator)
    loaded = load_replay_bundle_v1(replay_path, require_metric_report=True)

    assert validation_count == 1
    assert tuple(iter_replay_transition_views_v1(loaded.replay)) == tuple(
        iter_replay_transition_views_v1(replay_io_case.bundle.replay)
    )
    assert loaded.metric_report_artifact is not None
    assert loaded.metric_report_artifact.report == replay_io_case.report
    assert loaded.frame_at(0) is loaded.replay.frames[0]
    assert loaded.incoming_transition_at(0) is None
    for frame_index in range(len(loaded.replay.frames)):
        assert loaded.frame_at(frame_index) is loaded.replay.frames[frame_index]
        if frame_index:
            assert (
                loaded.incoming_transition_at(frame_index)
                is (loaded.replay.transitions[frame_index - 1])
            )
    assert validation_count == 1

    with pytest.raises(TypeError, match="integer"):
        loaded.frame_at(True)
    with pytest.raises(IndexError, match="outside"):
        loaded.frame_at(-1)
    with pytest.raises(IndexError, match="outside"):
        loaded.incoming_transition_at(len(loaded.replay.frames))


def test_loading_in_a_fresh_process_imports_no_array_core_policy_or_capture_code(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    """Artifact reading stays a host-only operation without backend discovery."""
    replay_path = _replay_path(tmp_path)
    replay_path.write_bytes(
        canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    )
    script = f"""
import importlib
import sys

replay_io = importlib.import_module("marl_battlegrounds.evaluation.replay_io")
replay_io.load_replay_artifact_v1({str(replay_path)!r})

for loaded_name in sys.modules:
    if loaded_name in {{
        "marl_battlegrounds.evaluation.capture",
        "marl_battlegrounds.evaluation.catalog",
        "marl_battlegrounds.core.env",
        "marl_battlegrounds.core.types",
    }}:
        raise SystemExit("unexpected replay-load dependency: " + loaded_name)
    if any(
        loaded_name == prefix or loaded_name.startswith(prefix + ".")
        for prefix in ("jax", "jaxlib", "numpy", "marl_battlegrounds.baselines")
    ):
        raise SystemExit("unexpected replay-load runtime: " + loaded_name)
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_bundle_publication_orders_metric_report_before_replay(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[Path] = []
    real_publish = replay_io._publish_bytes_no_clobber  # pyright: ignore[reportPrivateUsage]

    def tracked_publish(
        target: Path,
        payload: bytes,
        *,
        existing_code: ReplayIOErrorCodeV1,
        parent_descriptor: int | None = None,
    ) -> None:
        published.append(target)
        real_publish(
            target,
            payload,
            existing_code=existing_code,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(
        replay_io,
        "_publish_bytes_no_clobber",
        tracked_publish,
    )
    replay_path = _replay_path(tmp_path)

    save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert published == [_metric_path(tmp_path), replay_path]


def test_existing_replay_is_never_clobbered_or_allowed_to_publish_a_sidecar(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    replay_path = _replay_path(tmp_path)
    sentinel = b"existing replay must survive"
    replay_path.write_bytes(sentinel)

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert caught.value.code == "replay_target_exists"
    assert replay_path.read_bytes() == sentinel
    assert not _metric_path(tmp_path).exists()


def test_identical_orphan_metric_report_is_reused_before_replay_publication(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    metric_path = _metric_path(tmp_path)
    metric_bytes = canonical_metric_report_artifact_json_bytes_v1(
        replay_io_case.bundle.metric_report_artifact
    )
    metric_path.write_bytes(metric_bytes)

    saved = save_replay_bundle_v1(
        replay_io_case.bundle,
        _replay_path(tmp_path),
    )

    assert saved.metric_report_reused is True
    assert metric_path.read_bytes() == metric_bytes
    assert saved.replay_path.exists()


def test_existing_identical_report_must_be_durable_before_replay_publication(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric_path = _metric_path(tmp_path)
    metric_bytes = canonical_metric_report_artifact_json_bytes_v1(
        replay_io_case.bundle.metric_report_artifact
    )
    metric_path.write_bytes(metric_bytes)
    real_fsync = replay_io.os.fsync

    def fail_regular_file_fsync(descriptor: int) -> None:
        if stat.S_ISREG(replay_io.os.fstat(descriptor).st_mode):
            raise OSError("injected existing-report fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(replay_io.os, "fsync", fail_regular_file_fsync)
    replay_path = _replay_path(tmp_path)

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert caught.value.code == "atomic_publish_failed"
    assert not replay_path.exists()
    assert metric_path.read_bytes() == metric_bytes


def test_conflicting_metric_report_preserves_existing_bytes_and_publishes_no_replay(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    metric_path = _metric_path(tmp_path)
    sentinel = b"not the referenced metric artifact"
    metric_path.write_bytes(sentinel)

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, _replay_path(tmp_path))

    assert caught.value.code == "metric_report_conflict"
    assert metric_path.read_bytes() == sentinel
    assert not _replay_path(tmp_path).exists()


def test_nonregular_existing_metric_target_is_a_typed_save_conflict(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    """Save never leaks a loader exception for an occupied sidecar target."""
    _metric_path(tmp_path).mkdir()

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, _replay_path(tmp_path))

    assert caught.value.code == "metric_report_conflict"
    assert not _replay_path(tmp_path).exists()


def test_metric_publication_failure_never_publishes_referencing_replay(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_metric_publish(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        raise ReplaySaveError(
            "atomic_publish_failed",
            path=_metric_path(tmp_path),
            detail="injected report publication failure",
        )

    monkeypatch.setattr(replay_io, "_publish_metric_report", fail_metric_publish)

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, _replay_path(tmp_path))

    assert caught.value.code == "atomic_publish_failed"
    assert not _replay_path(tmp_path).exists()
    assert not _metric_path(tmp_path).exists()


def test_replay_publication_failure_leaves_only_reusable_report_orphan(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_publish = replay_io._publish_bytes_no_clobber  # pyright: ignore[reportPrivateUsage]
    replay_path = _replay_path(tmp_path)

    def fail_replay_publish(
        target: Path,
        payload: bytes,
        *,
        existing_code: ReplayIOErrorCodeV1,
        parent_descriptor: int | None = None,
    ) -> None:
        if target == replay_path:
            raise ReplaySaveError(
                "atomic_publish_failed",
                path=target,
                detail="injected replay publication failure",
            )
        real_publish(
            target,
            payload,
            existing_code=existing_code,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(
        replay_io,
        "_publish_bytes_no_clobber",
        fail_replay_publish,
    )

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert caught.value.code == "atomic_publish_failed"
    assert not replay_path.exists()
    assert _metric_path(tmp_path).read_bytes() == (
        canonical_metric_report_artifact_json_bytes_v1(
            replay_io_case.bundle.metric_report_artifact
        )
    )


def test_replay_directory_fsync_failure_rolls_back_target_and_allows_retry(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed durability barrier cannot leave a falsely failed replay link."""
    real_fsync_directory = replay_io._fsync_directory  # pyright: ignore[reportPrivateUsage]
    fsync_call_count = 0

    def fail_first_replay_directory_fsync(directory: int) -> None:
        nonlocal fsync_call_count
        fsync_call_count += 1
        if fsync_call_count == 2:
            raise OSError("injected replay directory fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(
        replay_io,
        "_fsync_directory",
        fail_first_replay_directory_fsync,
    )
    replay_path = _replay_path(tmp_path)

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert caught.value.code == "atomic_publish_failed"
    assert caught.value.path == replay_path
    assert not replay_path.exists()
    metric_path = _metric_path(tmp_path)
    metric_bytes = canonical_metric_report_artifact_json_bytes_v1(
        replay_io_case.bundle.metric_report_artifact
    )
    assert metric_path.read_bytes() == metric_bytes

    retried = save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert retried.metric_report_reused is True
    assert replay_path.read_bytes() == canonical_replay_json_bytes_v1(
        replay_io_case.bundle.replay
    )
    assert metric_path.read_bytes() == metric_bytes


def test_temporary_file_creation_failure_publishes_neither_bundle_member(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_temporary_creation(*args: object, **kwargs: object) -> tuple[int, str]:
        del args, kwargs
        raise OSError("injected temporary-file failure")

    monkeypatch.setattr(
        replay_io,
        "_create_temporary_file_at",
        fail_temporary_creation,
    )

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(replay_io_case.bundle, _replay_path(tmp_path))

    assert caught.value.code == "temporary_write_failed"
    assert not _replay_path(tmp_path).exists()
    assert not _metric_path(tmp_path).exists()


def test_loader_requires_a_regular_local_file_with_the_replay_suffix(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    directory_path = _replay_path(tmp_path, stem="directory")
    directory_path.mkdir()
    _assert_load_error(directory_path, "path_not_regular_file")

    missing_path = _replay_path(tmp_path, stem="missing")
    missing_error = _assert_load_error(missing_path, "path_not_found")
    assert missing_error.path == missing_path

    wrong_suffix = tmp_path / "episode.json"
    wrong_suffix.write_bytes(
        canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    )
    _assert_load_error(wrong_suffix, "invalid_filename")


def test_loader_rejects_final_and_parent_component_symlinks(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    real_path, _ = _write_canonical_bundle(
        real_directory,
        replay_io_case.bundle,
    )
    final_link = _replay_path(tmp_path, stem="final-link")
    final_link.symlink_to(real_path)
    _assert_load_error(final_link, "path_is_symlink")

    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_directory, target_is_directory=True)
    _assert_load_error(_replay_path(parent_link), "path_is_symlink")


def test_loader_does_not_lexically_erase_a_symlink_before_parent_traversal(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    safe_directory = tmp_path / "safe"
    attacker_directory = tmp_path / "attacker"
    safe_directory.mkdir()
    attacker_directory.mkdir()
    _write_canonical_bundle(safe_directory, replay_io_case.bundle)
    parent_link = tmp_path / "link"
    parent_link.symlink_to(attacker_directory, target_is_directory=True)
    deceptive_path = (
        parent_link / ".." / safe_directory.name / _replay_path(safe_directory).name
    )

    _assert_load_error(deceptive_path, "path_is_symlink")


def test_loader_is_bound_to_the_opened_parent_during_ancestor_swap(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checked parent cannot be swapped to redirect the final file open."""
    source_directory = tmp_path / "source"
    moved_source_directory = tmp_path / "source-opened"
    attacker_directory = tmp_path / "attacker"
    source_directory.mkdir()
    attacker_directory.mkdir()
    source_path, _ = _write_canonical_bundle(
        source_directory,
        replay_io_case.bundle,
    )
    attacker_case = _build_io_case(episode_id="episode-attacker")
    _write_canonical_bundle(attacker_directory, attacker_case.bundle)
    real_read = replay_io._read_bounded_regular_file_at  # pyright: ignore[reportPrivateUsage]
    swapped = False

    def swap_then_read(
        parent_descriptor: int,
        name: str,
        *,
        path: Path,
        max_file_size_bytes: int,
        error_type: type[ReplayLoadError] | type[ReplaySaveError] = ReplayLoadError,
    ) -> bytes:
        nonlocal swapped
        if not swapped:
            source_directory.rename(moved_source_directory)
            source_directory.symlink_to(attacker_directory, target_is_directory=True)
            swapped = True
        return real_read(
            parent_descriptor,
            name,
            path=path,
            max_file_size_bytes=max_file_size_bytes,
            error_type=error_type,
        )

    monkeypatch.setattr(
        replay_io,
        "_read_bounded_regular_file_at",
        swap_then_read,
    )

    loaded = load_replay_artifact_v1(source_path)

    assert loaded == replay_io_case.bundle.replay
    assert loaded != attacker_case.bundle.replay


def test_bundle_loader_reads_report_from_the_same_parent_after_swap(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay and sidecar resolution share one descriptor-bound directory."""
    source_directory = tmp_path / "source"
    moved_source_directory = tmp_path / "source-opened"
    attacker_directory = tmp_path / "attacker"
    source_directory.mkdir()
    attacker_directory.mkdir()
    source_path, _ = _write_canonical_bundle(
        source_directory,
        replay_io_case.bundle,
    )
    attacker_case = _build_io_case(episode_id="episode-attacker")
    _write_canonical_bundle(attacker_directory, attacker_case.bundle)
    real_load_replay_bytes = replay_io._load_replay_bytes  # pyright: ignore[reportPrivateUsage]
    swapped = False

    def load_then_swap(
        payload: bytes,
        *,
        path: Path,
        max_json_depth: int,
    ) -> ReplayArtifactV1:
        nonlocal swapped
        replay = real_load_replay_bytes(
            payload,
            path=path,
            max_json_depth=max_json_depth,
        )
        if not swapped:
            source_directory.rename(moved_source_directory)
            source_directory.symlink_to(attacker_directory, target_is_directory=True)
            swapped = True
        return replay

    monkeypatch.setattr(replay_io, "_load_replay_bytes", load_then_swap)

    loaded = load_replay_bundle_v1(source_path, require_metric_report=True)

    assert loaded.replay == replay_io_case.bundle.replay
    assert loaded.metric_report_artifact == replay_io_case.bundle.metric_report_artifact
    assert loaded.replay != attacker_case.bundle.replay


def test_loader_fails_closed_without_secure_directory_fd_support(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path, _ = _write_canonical_bundle(tmp_path, replay_io_case.bundle)
    monkeypatch.setattr(replay_io.os, "supports_dir_fd", set[object]())

    error = _assert_load_error(replay_path, "unsupported_platform")

    assert "directory-fd" in error.detail


def test_bundle_loader_rejects_a_symlinked_metric_sidecar(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    replay_path = _replay_path(tmp_path)
    replay_path.write_bytes(
        canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    )
    real_metric = tmp_path / "real-metric.json"
    real_metric.write_bytes(
        canonical_metric_report_artifact_json_bytes_v1(
            replay_io_case.bundle.metric_report_artifact
        )
    )
    _metric_path(tmp_path).symlink_to(real_metric)

    with pytest.raises(ReplayLoadError) as caught:
        load_replay_bundle_v1(replay_path)

    assert caught.value.code == "path_is_symlink"
    assert caught.value.path == _metric_path(tmp_path)


@pytest.mark.parametrize("invalid_limit", (0, -1, True, 1.5))
def test_loader_rejects_invalid_explicit_resource_limits(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    invalid_limit: object,
) -> None:
    replay_path, _ = _write_canonical_bundle(tmp_path, replay_io_case.bundle)

    with pytest.raises(ReplayLoadError) as caught:
        load_replay_artifact_v1(
            replay_path,
            max_file_size_bytes=invalid_limit,  # type: ignore[arg-type]
        )

    assert caught.value.code == "invalid_argument"


def test_saver_rejects_bad_suffix_missing_parent_and_nondirectory_parent(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    with pytest.raises(ReplaySaveError) as suffix_error:
        save_replay_bundle_v1(replay_io_case.bundle, tmp_path / "episode.json")
    assert suffix_error.value.code == "invalid_filename"

    missing_parent_path = _replay_path(tmp_path / "missing")
    with pytest.raises(ReplaySaveError) as parent_error:
        save_replay_bundle_v1(replay_io_case.bundle, missing_parent_path)
    assert parent_error.value.code == "missing_parent"
    assert not missing_parent_path.parent.exists()

    parent_file = tmp_path / "parent-file"
    parent_file.write_bytes(b"not a directory")
    with pytest.raises(ReplaySaveError) as directory_error:
        save_replay_bundle_v1(
            replay_io_case.bundle,
            parent_file / f"episode{REPLAY_FILE_SUFFIX_V1}",
        )
    assert directory_error.value.code == "path_not_directory"


def test_saver_rejects_symlinked_parent_replay_and_metric_paths(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ReplaySaveError) as parent_error:
        save_replay_bundle_v1(replay_io_case.bundle, _replay_path(parent_link))
    assert parent_error.value.code == "path_is_symlink"

    real_replay = tmp_path / "real-replay"
    real_replay.write_bytes(b"sentinel")
    replay_link = _replay_path(tmp_path, stem="replay-link")
    replay_link.symlink_to(real_replay)
    with pytest.raises(ReplaySaveError) as replay_error:
        save_replay_bundle_v1(replay_io_case.bundle, replay_link)
    assert replay_error.value.code == "path_is_symlink"

    real_metric = tmp_path / "real-metric"
    real_metric.write_bytes(b"sentinel")
    metric_link = _metric_path(tmp_path, stem="metric-link")
    metric_link.symlink_to(real_metric)
    with pytest.raises(ReplaySaveError) as metric_error:
        save_replay_bundle_v1(
            replay_io_case.bundle,
            _replay_path(tmp_path, stem="metric-link"),
        )
    assert metric_error.value.code == "path_is_symlink"


def test_saver_keeps_report_and_replay_on_one_bound_parent_during_swap(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ancestor swap cannot redirect either half of bundle publication."""
    destination_directory = tmp_path / "destination"
    moved_destination_directory = tmp_path / "destination-opened"
    attacker_directory = tmp_path / "attacker"
    destination_directory.mkdir()
    attacker_directory.mkdir()
    replay_path = _replay_path(destination_directory)
    real_publish_metric = replay_io._publish_metric_report  # pyright: ignore[reportPrivateUsage]
    swapped = False

    def swap_then_publish_metric(
        path: Path,
        payload: bytes,
        *,
        max_file_size_bytes: int,
        parent_descriptor: int | None = None,
    ) -> bool:
        nonlocal swapped
        assert parent_descriptor is not None
        if not swapped:
            destination_directory.rename(moved_destination_directory)
            destination_directory.symlink_to(
                attacker_directory, target_is_directory=True
            )
            swapped = True
        return real_publish_metric(
            path,
            payload,
            max_file_size_bytes=max_file_size_bytes,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(
        replay_io,
        "_publish_metric_report",
        swap_then_publish_metric,
    )

    saved = save_replay_bundle_v1(replay_io_case.bundle, replay_path)

    assert saved.replay_path == replay_path
    assert _replay_path(moved_destination_directory).read_bytes() == (
        canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    )
    assert _metric_path(moved_destination_directory).read_bytes() == (
        canonical_metric_report_artifact_json_bytes_v1(
            replay_io_case.bundle.metric_report_artifact
        )
    )
    assert list(attacker_directory.iterdir()) == []


def test_saver_honors_explicit_size_ceiling_before_any_publication(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    replay_bytes = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    metric_bytes = canonical_metric_report_artifact_json_bytes_v1(
        replay_io_case.bundle.metric_report_artifact
    )
    limit = max(len(replay_bytes), len(metric_bytes)) - 1

    with pytest.raises(ReplaySaveError) as caught:
        save_replay_bundle_v1(
            replay_io_case.bundle,
            _replay_path(tmp_path),
            max_file_size_bytes=limit,
        )

    assert caught.value.code == "file_too_large"
    assert not _replay_path(tmp_path).exists()
    assert not _metric_path(tmp_path).exists()


def test_loader_enforces_size_limit_at_and_below_the_exact_byte_boundary(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    replay_path = _replay_path(tmp_path)
    payload = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    replay_path.write_bytes(payload)

    assert (
        load_replay_artifact_v1(
            replay_path,
            max_file_size_bytes=len(payload),
        )
        == replay_io_case.bundle.replay
    )
    _assert_load_error(
        replay_path,
        "file_too_large",
        max_file_size_bytes=len(payload) - 1,
    )
    assert len(payload) < DEFAULT_MAX_REPLAY_FILE_SIZE_BYTES_V1


def test_loader_rejects_bom_invalid_utf8_and_excessive_nesting(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    canonical = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    bom_path = _replay_path(tmp_path, stem="bom")
    bom_path.write_bytes(b"\xef\xbb\xbf" + canonical)
    _assert_load_error(bom_path, "utf8_bom_forbidden")

    utf8_path = _replay_path(tmp_path, stem="utf8")
    utf8_path.write_bytes(canonical + b"\xff")
    _assert_load_error(utf8_path, "invalid_utf8")

    depth_path = _replay_path(tmp_path, stem="depth")
    depth_path.write_bytes(
        b'{"nested":'
        + (b"[" * 129)
        + b"0"
        + (b"]" * 129)
        + b',"schema_id":"marl_battlegrounds.evaluation.replay_artifact",'
        + b'"schema_version":1}'
    )
    _assert_load_error(depth_path, "json_depth_exceeded", max_json_depth=128)


def test_loader_rejects_duplicate_keys_nonfinite_numbers_and_trailing_content(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    canonical = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)

    duplicate_path = _replay_path(tmp_path, stem="duplicate")
    duplicate_path.write_bytes(
        canonical.replace(
            b'{"artifact_id":',
            b'{"artifact_id":"duplicate","artifact_id":',
            1,
        )
    )
    _assert_load_error(duplicate_path, "duplicate_json_key")

    for stem, token in (("nan", b"NaN"), ("positive-infinity", b"Infinity")):
        nonfinite_path = _replay_path(tmp_path, stem=stem)
        nonfinite_path.write_bytes(b'{"probe":' + token + b"," + canonical[1:])
        _assert_load_error(nonfinite_path, "nonfinite_json_number")

    overflow_path = _replay_path(tmp_path, stem="overflow-number")
    overflow_path.write_bytes(b'{"probe":1e9999,' + canonical[1:])
    _assert_load_error(overflow_path, "nonfinite_json_number")

    trailing_path = _replay_path(tmp_path, stem="trailing")
    trailing_path.write_bytes(canonical + b"{}")
    _assert_load_error(trailing_path, "malformed_json")


def test_loader_wraps_pathological_integer_and_malformed_json_as_typed_errors(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    canonical = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    huge_integer_path = _replay_path(tmp_path, stem="huge-integer")
    huge_integer_path.write_bytes(b'{"probe":' + (b"9" * 5_000) + b"," + canonical[1:])
    _assert_load_error(huge_integer_path, "malformed_json")

    malformed_path = _replay_path(tmp_path, stem="malformed")
    malformed_path.write_bytes(canonical[:-1])
    _assert_load_error(malformed_path, "malformed_json")


def test_loader_rejects_wrong_roots_and_future_root_versions_before_models(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    canonical = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    canonical_payload = json.loads(canonical)

    scalar_path = _replay_path(tmp_path, stem="scalar")
    scalar_path.write_bytes(b"[]")
    _assert_load_error(scalar_path, "wrong_root_schema")

    wrong_schema_payload = dict(canonical_payload)
    wrong_schema_payload["schema_id"] = "marl_battlegrounds.evaluation.unknown"
    wrong_schema_path = _replay_path(tmp_path, stem="wrong-schema")
    wrong_schema_path.write_bytes(
        json.dumps(
            wrong_schema_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    _assert_load_error(wrong_schema_path, "wrong_root_schema")

    for stem, version in (("future", 2), ("boolean", True)):
        future_payload = dict(canonical_payload)
        future_payload["schema_version"] = version
        future_path = _replay_path(tmp_path, stem=stem)
        future_path.write_bytes(
            json.dumps(
                future_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        _assert_load_error(future_path, "unsupported_schema_version")


def test_loader_rejects_extra_fields_and_unknown_nested_versions(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    payload = json.loads(canonical_replay_json_bytes_v1(replay_io_case.bundle.replay))
    payload["unexpected"] = "not in the replay schema"
    extra_path = _replay_path(tmp_path, stem="extra")
    extra_path.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    _assert_load_error(extra_path, "model_validation_failed")

    payload = json.loads(canonical_replay_json_bytes_v1(replay_io_case.bundle.replay))
    payload["header"]["schema_version"] = 2
    nested_version_path = _replay_path(tmp_path, stem="nested-version")
    nested_version_path.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    _assert_load_error(nested_version_path, "model_validation_failed")


def test_loader_rejects_parseable_but_noncanonical_whitespace_and_key_order(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    canonical = canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    for stem, payload in (
        ("leading-space", b" " + canonical),
        ("trailing-newline", canonical + b"\n"),
    ):
        path = _replay_path(tmp_path, stem=stem)
        path.write_bytes(payload)
        _assert_load_error(path, "noncanonical_json")

    parsed = json.loads(canonical)
    reversed_root = {key: parsed[key] for key in reversed(tuple(parsed))}
    reordered_path = _replay_path(tmp_path, stem="reordered")
    reordered_path.write_bytes(
        json.dumps(
            reversed_root,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        ).encode()
    )
    _assert_load_error(reordered_path, "noncanonical_json")


@pytest.mark.parametrize(
    ("reference_field", "replacement"),
    (
        ("canonical_digest_sha256", "0" * 64),
        ("canonical_byte_length", None),
    ),
)
def test_bundle_loader_rejects_metric_reference_digest_and_length_mismatch(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    reference_field: str,
    replacement: object,
) -> None:
    replay = replay_io_case.bundle.replay
    if replacement is None:
        replacement = replay.metric_report_reference.canonical_byte_length + 1
    tampered_replay = _readdress_metric_reference(
        replay,
        **{reference_field: replacement},
    )
    replay_path = _replay_path(tmp_path)
    replay_path.write_bytes(canonical_replay_json_bytes_v1(tampered_replay))
    _metric_path(tmp_path).write_bytes(
        canonical_metric_report_artifact_json_bytes_v1(
            replay_io_case.bundle.metric_report_artifact
        )
    )

    assert load_replay_artifact_v1(replay_path) == tampered_replay
    with pytest.raises(ReplayLoadError) as caught:
        load_replay_bundle_v1(replay_path)

    assert caught.value.code == "metric_report_mismatch"
    assert caught.value.path == _metric_path(tmp_path)


def test_bundle_loader_rejects_canonical_sidecar_from_another_replay(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    other = _build_io_case(episode_id="episode-foreign")
    replay_path = _replay_path(tmp_path)
    replay_path.write_bytes(
        canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    )
    _metric_path(tmp_path).write_bytes(
        canonical_metric_report_artifact_json_bytes_v1(
            other.bundle.metric_report_artifact
        )
    )

    with pytest.raises(ReplayLoadError) as caught:
        load_replay_bundle_v1(replay_path)

    assert caught.value.code == "metric_report_mismatch"


@pytest.mark.parametrize(
    ("member", "field"),
    (
        ("replay", "canonical_digest_sha256"),
        ("metric", "canonical_digest_sha256"),
    ),
)
def test_loader_rejects_tampered_root_digests_before_publication_use(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
    member: str,
    field: str,
) -> None:
    replay_payload = json.loads(
        canonical_replay_json_bytes_v1(replay_io_case.bundle.replay)
    )
    metric_payload = json.loads(
        canonical_metric_report_artifact_json_bytes_v1(
            replay_io_case.bundle.metric_report_artifact
        )
    )
    selected = replay_payload if member == "replay" else metric_payload
    selected[field] = "0" * 64
    replay_path = _replay_path(tmp_path)
    replay_path.write_bytes(
        json.dumps(
            replay_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    _metric_path(tmp_path).write_bytes(
        json.dumps(
            metric_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )

    if member == "replay":
        with pytest.raises(ReplayLoadError) as caught:
            load_replay_artifact_v1(replay_path)
    else:
        with pytest.raises(ReplayLoadError) as caught:
            load_replay_bundle_v1(replay_path)
    assert caught.value.code == "model_validation_failed"


def test_metric_sidecar_has_its_own_root_version_and_canonical_byte_gate(
    replay_io_case: _ReplayIoCase,
    tmp_path: Path,
) -> None:
    replay_path, metric_path = _write_canonical_bundle(
        tmp_path,
        replay_io_case.bundle,
    )
    canonical_metric = metric_path.read_bytes()
    metric_payload = json.loads(canonical_metric)

    metric_payload["schema_version"] = 2
    metric_path.write_bytes(
        json.dumps(
            metric_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    with pytest.raises(ReplayLoadError) as version_error:
        load_replay_bundle_v1(replay_path)
    assert version_error.value.code == "unsupported_schema_version"

    metric_path.write_bytes(b" " + canonical_metric)
    with pytest.raises(ReplayLoadError) as canonical_error:
        load_replay_bundle_v1(replay_path)
    assert canonical_error.value.code == "noncanonical_json"
