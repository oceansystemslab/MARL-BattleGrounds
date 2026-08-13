"""Canonical POV/scenario companion persistence and path-security proofs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from tests.evaluation_fixtures import captured_evaluation_trajectory

import marl_battlegrounds.evaluation.replay_io as replay_io
from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.models import (
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeIdentityV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovReplayArtifactV1,
    canonical_actor_pov_replay_json_bytes_v1,
    export_actor_pov_replay_v1,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayBundleV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    ACTOR_POV_FILE_SUFFIX_V1,
    SCENARIO_FILE_SUFFIX_V1,
    ReplayIOErrorCodeV1,
    ReplayLoadError,
    ReplaySaveError,
    canonical_scenario_evaluation_record_json_bytes_v1,
    load_actor_pov_replay_artifact_v1,
    load_scenario_evaluation_record_v1,
    save_actor_pov_replay_artifact_v1,
    save_scenario_evaluation_record_v1,
)
from marl_battlegrounds.evaluation.scenario import (
    SCENARIO_SPECIFICATION_SCHEMA_ID,
    ResolvedScenarioSpecificationV1,
    ScenarioBooleanValueV1,
    ScenarioEvaluationRecordV1,
    ScenarioMeasurementDefinitionV1,
    ScenarioMeasurementResultV1,
    ScenarioPredicateResultV1,
    build_scenario_evaluation_record_v1,
)

type CompanionKind = Literal["pov", "scenario"]


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.14",
        package_version="0.0.0",
        jax_version="test",
        jaxlib_version="test",
        numpy_version="test",
        pydantic_version="test",
        platform="test-platform",
        machine="test-machine",
        backend="cpu",
        device="test-device",
        precision="float32",
        environment_count=1,
        batch_shape=(),
        policy_execution_included=False,
    )


def _scenario_specification(
    context: EvaluationEpisodeContextV1,
) -> ResolvedScenarioSpecificationV1:
    payload: dict[str, object] = {
        "schema_id": SCENARIO_SPECIFICATION_SCHEMA_ID,
        "schema_version": 1,
        "scenario_id": "scenario",
        "scenario_version": 1,
        "classification": "custom",
        "hypothesis": "The supplied endpoint is evaluated without inference.",
        "eligible_roles": ("focal",),
        "authored_initial_condition": ContentAddressedIdentityV1(
            identifier="initial-condition",
            version=1,
            canonical_digest="1" * 64,
        ),
        "parameters": (),
        "resolved_config_digest_sha256": (
            context.resolved_env_config.canonical_digest_sha256
        ),
        "horizon": context.expected_horizon,
        "pressure_protocol": None,
        "primary_measurement": ScenarioMeasurementDefinitionV1(
            measurement_id="test.endpoint",
            measurement_version=1,
            role="primary",
            value_type="boolean",
            units="boolean",
            completion_scope="complete_episode",
            supports_right_censoring=False,
        ),
        "secondary_measurements": (),
        "violations": (),
        "success_predicate": VersionedIdentityV1(
            identifier="test.success",
            version=1,
        ),
        "completion_policy": VersionedIdentityV1(
            identifier="test.completion",
            version=1,
        ),
        "partial_result_policy": VersionedIdentityV1(
            identifier="test.partial",
            version=1,
        ),
    }
    return ResolvedScenarioSpecificationV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _context_with_specification(
    context: EvaluationEpisodeContextV1,
    specification: ResolvedScenarioSpecificationV1,
) -> EvaluationEpisodeContextV1:
    identity_payload = context.identity.model_dump(mode="python")
    identity_payload["scenario"] = ContentAddressedIdentityV1(
        identifier=specification.scenario_id,
        version=specification.scenario_version,
        canonical_digest=specification.canonical_digest_sha256,
    )
    identity = EvaluationEpisodeIdentityV1.model_validate(identity_payload)
    context_payload = context.model_dump(mode="python")
    context_payload["identity"] = identity
    return EvaluationEpisodeContextV1.model_validate(context_payload)


@dataclass(frozen=True, slots=True)
class _CompanionCase:
    bundle: ReplayBundleV1
    pov: ActorPovReplayArtifactV1
    scenario: ScenarioEvaluationRecordV1


def _build_case(episode_id: str) -> _CompanionCase:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        capture_profile="scenario_metric_complete",
        expected_horizon=1,
        with_scenario=True,
        episode_id=episode_id,
    )
    specification = _scenario_specification(trajectory.context)
    context = _context_with_specification(trajectory.context, specification)
    observer = build_evaluation_observer_v1(context)
    observer.start(trajectory.frames[0])
    observer.append(trajectory.transitions[0], trajectory.frames[1])
    report = observer.finalize(completion_state="complete")
    bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    )
    scenario = build_scenario_evaluation_record_v1(
        specification,
        bundle.replay,
        bundle.metric_report_artifact,
        measurement_results=(
            ScenarioMeasurementResultV1(
                measurement_id="test.endpoint",
                measurement_version=1,
                result_status="defined",
                endpoint_observation_status="observed",
                value=ScenarioBooleanValueV1(value=True),
            ),
        ),
        violation_results=(),
        predicate_result=ScenarioPredicateResultV1(
            predicate_id="test.success",
            predicate_version=1,
            status="satisfied",
        ),
    )
    return _CompanionCase(
        bundle=bundle,
        pov=export_actor_pov_replay_v1(bundle.replay, global_slot=0),
        scenario=scenario,
    )


@pytest.fixture(scope="module")
def companion_case() -> _CompanionCase:
    return _build_case("episode-companion")


@pytest.fixture(scope="module")
def other_case() -> _CompanionCase:
    return _build_case("episode-companion-other")


def _path(
    directory: Path,
    kind: CompanionKind,
    *,
    stem: str | None = None,
) -> Path:
    suffix = ACTOR_POV_FILE_SUFFIX_V1 if kind == "pov" else SCENARIO_FILE_SUFFIX_V1
    canonical_stem = (
        ("episode.agent-agent-slot-0" if kind == "pov" else "episode")
        if stem is None
        else stem
    )
    return directory / f"{canonical_stem}{suffix}"


def _artifact(case: _CompanionCase, kind: CompanionKind) -> object:
    return case.pov if kind == "pov" else case.scenario


def _canonical_bytes(case: _CompanionCase, kind: CompanionKind) -> bytes:
    if kind == "pov":
        return canonical_actor_pov_replay_json_bytes_v1(case.pov)
    return canonical_scenario_evaluation_record_json_bytes_v1(case.scenario)


def _save(
    case: _CompanionCase,
    kind: CompanionKind,
    path: Path,
    *,
    max_file_size_bytes: int = 1024**3,
) -> object:
    if kind == "pov":
        return save_actor_pov_replay_artifact_v1(
            case.pov,
            case.bundle.replay,
            path,
            max_file_size_bytes=max_file_size_bytes,
        )
    return save_scenario_evaluation_record_v1(
        case.scenario,
        case.bundle.replay,
        case.bundle.metric_report_artifact,
        path,
        max_file_size_bytes=max_file_size_bytes,
    )


def _load(
    case: _CompanionCase,
    kind: CompanionKind,
    path: Path,
    *,
    max_file_size_bytes: int = 1024**3,
    max_json_depth: int = 128,
) -> object:
    if kind == "pov":
        return load_actor_pov_replay_artifact_v1(
            path,
            source_replay=case.bundle.replay,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        )
    return load_scenario_evaluation_record_v1(
        path,
        source_replay=case.bundle.replay,
        metric_report_artifact=case.bundle.metric_report_artifact,
        max_file_size_bytes=max_file_size_bytes,
        max_json_depth=max_json_depth,
    )


def _assert_load_error(
    case: _CompanionCase,
    kind: CompanionKind,
    path: Path,
    code: ReplayIOErrorCodeV1,
    *,
    max_file_size_bytes: int = 1024**3,
    max_json_depth: int = 128,
) -> ReplayLoadError:
    with pytest.raises(ReplayLoadError) as caught:
        _load(
            case,
            kind,
            path,
            max_file_size_bytes=max_file_size_bytes,
            max_json_depth=max_json_depth,
        )
    assert caught.value.code == code
    return caught.value


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_save_load_save_bytes_are_exact(
    companion_case: _CompanionCase,
    tmp_path: Path,
    kind: CompanionKind,
) -> None:
    path = _path(tmp_path, kind)
    saved = _save(companion_case, kind, path)
    expected = _canonical_bytes(companion_case, kind)
    assert path.read_bytes() == expected
    assert saved.path == path  # type: ignore[union-attr]
    assert saved.byte_length == len(expected)  # type: ignore[union-attr]
    loaded = _load(companion_case, kind, path)
    assert loaded == _artifact(companion_case, kind)

    second = _path(tmp_path, kind, stem="second")
    _save(companion_case, kind, second)
    assert second.read_bytes() == expected


def test_scenario_canonical_bytes_round_trip(
    companion_case: _CompanionCase,
) -> None:
    payload = canonical_scenario_evaluation_record_json_bytes_v1(
        companion_case.scenario
    )
    assert ScenarioEvaluationRecordV1.model_validate_json(payload) == (
        companion_case.scenario
    )
    assert payload == canonical_scenario_evaluation_record_json_bytes_v1(
        companion_case.scenario
    )


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_suffix_and_size_limits_reject(
    companion_case: _CompanionCase,
    tmp_path: Path,
    kind: CompanionKind,
) -> None:
    wrong_suffix = tmp_path / f"{kind}.json"
    wrong_suffix.write_bytes(_canonical_bytes(companion_case, kind))
    _assert_load_error(
        companion_case,
        kind,
        wrong_suffix,
        "invalid_filename",
    )
    with pytest.raises(ReplaySaveError) as suffix_error:
        _save(companion_case, kind, wrong_suffix)
    assert suffix_error.value.code == "invalid_filename"

    path = _path(tmp_path, kind)
    payload = _canonical_bytes(companion_case, kind)
    path.write_bytes(payload)
    _assert_load_error(
        companion_case,
        kind,
        path,
        "file_too_large",
        max_file_size_bytes=len(payload) - 1,
    )
    save_target = _path(tmp_path, kind, stem="too-large")
    with pytest.raises(ReplaySaveError) as size_error:
        _save(
            companion_case,
            kind,
            save_target,
            max_file_size_bytes=len(payload) - 1,
        )
    assert size_error.value.code == "file_too_large"
    assert not save_target.exists()


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_depth_duplicate_nonfinite_and_noncanonical_reject(
    companion_case: _CompanionCase,
    tmp_path: Path,
    kind: CompanionKind,
) -> None:
    canonical = _canonical_bytes(companion_case, kind)

    depth_path = _path(tmp_path, kind, stem="depth")
    depth_path.write_bytes(canonical)
    _assert_load_error(
        companion_case,
        kind,
        depth_path,
        "json_depth_exceeded",
        max_json_depth=1,
    )

    expected_schema = json.loads(canonical)["schema_id"]
    duplicate = _path(tmp_path, kind, stem="duplicate")
    duplicate.write_bytes(
        b'{"schema_id":'
        + json.dumps(expected_schema).encode("utf-8")
        + b","
        + canonical[1:]
    )
    _assert_load_error(companion_case, kind, duplicate, "duplicate_json_key")

    for stem, token in (
        ("nan", b"NaN"),
        ("infinity", b"Infinity"),
        ("overflow", b"1e9999"),
    ):
        nonfinite = _path(tmp_path, kind, stem=stem)
        nonfinite.write_bytes(b'{"probe":' + token + b"," + canonical[1:])
        _assert_load_error(
            companion_case,
            kind,
            nonfinite,
            "nonfinite_json_number",
        )

    noncanonical = _path(tmp_path, kind, stem="noncanonical")
    noncanonical.write_bytes(canonical + b"\n")
    _assert_load_error(companion_case, kind, noncanonical, "noncanonical_json")


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_future_version_and_wrong_root_reject(
    companion_case: _CompanionCase,
    tmp_path: Path,
    kind: CompanionKind,
) -> None:
    payload = json.loads(_canonical_bytes(companion_case, kind))
    payload["schema_version"] = 2
    future = _path(tmp_path, kind, stem="future")
    future.write_bytes(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    _assert_load_error(
        companion_case,
        kind,
        future,
        "unsupported_schema_version",
    )

    payload["schema_version"] = 1
    payload["schema_id"] = "marl_battlegrounds.evaluation.wrong_root"
    wrong_root = _path(tmp_path, kind, stem="wrong-root")
    wrong_root.write_bytes(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    _assert_load_error(companion_case, kind, wrong_root, "wrong_root_schema")


def test_pov_source_mismatch_rejects_save_and_load(
    companion_case: _CompanionCase,
    other_case: _CompanionCase,
    tmp_path: Path,
) -> None:
    load_path = _path(tmp_path, "pov")
    load_path.write_bytes(_canonical_bytes(companion_case, "pov"))
    with pytest.raises(ReplayLoadError) as load_error:
        load_actor_pov_replay_artifact_v1(
            load_path,
            source_replay=other_case.bundle.replay,
        )
    assert load_error.value.code == "semantic_validation_failed"

    save_path = _path(tmp_path, "pov", stem="mismatch")
    with pytest.raises(ReplaySaveError) as save_error:
        save_actor_pov_replay_artifact_v1(
            companion_case.pov,
            other_case.bundle.replay,
            save_path,
        )
    assert save_error.value.code == "invalid_argument"
    assert not save_path.exists()


def test_scenario_replay_and_report_mismatches_reject(
    companion_case: _CompanionCase,
    other_case: _CompanionCase,
    tmp_path: Path,
) -> None:
    path = _path(tmp_path, "scenario")
    path.write_bytes(_canonical_bytes(companion_case, "scenario"))
    for replay, report in (
        (other_case.bundle.replay, other_case.bundle.metric_report_artifact),
        (companion_case.bundle.replay, other_case.bundle.metric_report_artifact),
    ):
        with pytest.raises(ReplayLoadError) as load_error:
            load_scenario_evaluation_record_v1(
                path,
                source_replay=replay,
                metric_report_artifact=report,
            )
        assert load_error.value.code == "semantic_validation_failed"

    save_path = _path(tmp_path, "scenario", stem="mismatch")
    with pytest.raises(ReplaySaveError) as save_error:
        save_scenario_evaluation_record_v1(
            companion_case.scenario,
            other_case.bundle.replay,
            other_case.bundle.metric_report_artifact,
            save_path,
        )
    assert save_error.value.code == "invalid_argument"
    assert not save_path.exists()


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_publication_is_no_clobber(
    companion_case: _CompanionCase,
    tmp_path: Path,
    kind: CompanionKind,
) -> None:
    path = _path(tmp_path, kind)
    first = _save(companion_case, kind, path)
    sentinel = path.read_bytes()
    with pytest.raises(ReplaySaveError) as caught:
        _save(companion_case, kind, path)
    assert caught.value.code == "companion_target_exists"
    assert path.read_bytes() == sentinel
    assert first.byte_length == len(sentinel)  # type: ignore[union-attr]


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_load_and_save_reject_symlink_paths(
    companion_case: _CompanionCase,
    tmp_path: Path,
    kind: CompanionKind,
) -> None:
    real = tmp_path / f"real-{kind}.json"
    real.write_bytes(_canonical_bytes(companion_case, kind))
    final_link = _path(tmp_path, kind, stem="final-link")
    final_link.symlink_to(real)
    _assert_load_error(companion_case, kind, final_link, "path_is_symlink")
    with pytest.raises(ReplaySaveError) as save_error:
        _save(companion_case, kind, final_link)
    assert save_error.value.code == "path_is_symlink"
    assert real.read_bytes() == _canonical_bytes(companion_case, kind)

    real_parent = tmp_path / f"real-parent-{kind}"
    real_parent.mkdir()
    parent_link = tmp_path / f"parent-link-{kind}"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    linked_path = _path(parent_link, kind)
    with pytest.raises(ReplaySaveError) as parent_save_error:
        _save(companion_case, kind, linked_path)
    assert parent_save_error.value.code == "path_is_symlink"
    assert not _path(real_parent, kind).exists()


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_loader_is_bound_to_opened_parent_during_swap(
    companion_case: _CompanionCase,
    other_case: _CompanionCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: CompanionKind,
) -> None:
    source_directory = tmp_path / f"source-{kind}"
    moved_source_directory = tmp_path / f"source-opened-{kind}"
    attacker_directory = tmp_path / f"attacker-{kind}"
    source_directory.mkdir()
    attacker_directory.mkdir()
    source_path = _path(source_directory, kind)
    source_path.write_bytes(_canonical_bytes(companion_case, kind))
    _path(attacker_directory, kind).write_bytes(_canonical_bytes(other_case, kind))
    real_read = replay_io._read_bounded_regular_file_at  # pyright: ignore[reportPrivateUsage]
    swapped = False

    def swap_then_read(
        parent_descriptor: int,
        name: str,
        *,
        path: Path,
        max_file_size_bytes: int,
        error_type: type[ReplayLoadError] | type[ReplaySaveError] = ReplayLoadError,
        fsync_before_close: bool = False,
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
            fsync_before_close=fsync_before_close,
        )

    monkeypatch.setattr(replay_io, "_read_bounded_regular_file_at", swap_then_read)
    loaded = _load(companion_case, kind, source_path)
    assert loaded == _artifact(companion_case, kind)
    assert loaded != _artifact(other_case, kind)


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_companion_saver_is_bound_to_opened_parent_during_swap(
    companion_case: _CompanionCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: CompanionKind,
) -> None:
    destination = tmp_path / f"destination-{kind}"
    moved_destination = tmp_path / f"destination-opened-{kind}"
    attacker = tmp_path / f"attacker-save-{kind}"
    destination.mkdir()
    attacker.mkdir()
    path = _path(destination, kind)
    real_publish = replay_io._publish_bytes_no_clobber  # pyright: ignore[reportPrivateUsage]
    swapped = False

    def swap_then_publish(
        target: Path,
        payload: bytes,
        *,
        existing_code: ReplayIOErrorCodeV1,
        parent_descriptor: int | None = None,
    ) -> None:
        nonlocal swapped
        assert parent_descriptor is not None
        if not swapped:
            destination.rename(moved_destination)
            destination.symlink_to(attacker, target_is_directory=True)
            swapped = True
        real_publish(
            target,
            payload,
            existing_code=existing_code,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(replay_io, "_publish_bytes_no_clobber", swap_then_publish)
    _save(companion_case, kind, path)
    assert _path(moved_destination, kind).read_bytes() == _canonical_bytes(
        companion_case,
        kind,
    )
    assert not _path(attacker, kind).exists()


@pytest.mark.parametrize("kind", ("pov", "scenario"))
def test_injected_directory_publish_failure_rolls_back_target(
    companion_case: _CompanionCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: CompanionKind,
) -> None:
    target = _path(tmp_path, kind)
    unrelated = tmp_path / f"unrelated-{kind}.txt"
    unrelated.write_bytes(b"preserve-me")
    real_fsync_directory = replay_io._fsync_directory  # pyright: ignore[reportPrivateUsage]
    call_count = 0

    def fail_first_directory_fsync(directory: int) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("injected companion directory fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(
        replay_io,
        "_fsync_directory",
        fail_first_directory_fsync,
    )
    with pytest.raises(ReplaySaveError) as caught:
        _save(companion_case, kind, target)
    assert caught.value.code == "atomic_publish_failed"
    assert not target.exists()
    assert unrelated.read_bytes() == b"preserve-me"

    monkeypatch.setattr(replay_io, "_fsync_directory", real_fsync_directory)
    _save(companion_case, kind, target)
    assert target.read_bytes() == _canonical_bytes(companion_case, kind)
