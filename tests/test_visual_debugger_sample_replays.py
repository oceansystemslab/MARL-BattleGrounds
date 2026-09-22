"""Check the deterministic bundled sample replays."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import scripts.dev.generate_visual_debugger_sample_replays as generator_module
import scripts.dev.visual_debugger.sample_replays as sample_replays_module
from scripts.dev.generate_visual_debugger_sample_replays import (
    generate_sample_replays,
    verify_sample_replays,
)
from scripts.dev.visual_debugger.model import DebuggerScenario
from scripts.dev.visual_debugger.sample_replays import (
    SAMPLE_REPLAY_DIRECTORY,
    SAMPLE_REPLAY_MANIFEST_PATH,
    SAMPLE_REPLAY_MAX_MANIFEST_SIZE_BYTES,
    SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES,
    SAMPLE_REPLAYS,
    SampleReplayVerificationError,
    canonical_sample_replay_manifest_json_bytes_v1,
    load_verified_sample_replay,
    read_bounded_regular_file_v1,
)

import marl_battlegrounds.evaluation.replay_io as replay_io_module
from marl_battlegrounds.evaluation.analysis import analyze_replay
from marl_battlegrounds.evaluation.models import CodeRevisionV1, CodeRevisionV2
from marl_battlegrounds.evaluation.replay import ReplayArtifactV1, RuntimeProvenanceV1

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR_SCRIPT = (
    _REPOSITORY_ROOT / "scripts" / "dev" / "generate_visual_debugger_sample_replays.py"
)
_PRE_M7_EVENT_KIND_UNION = frozenset(
    {
        "ability_activated",
        "action_rejected",
        "agent_died",
        "agent_left_combat",
        "agent_respawned",
        "charge_phase_displacement",
        "combat_countdown_reset",
        "cooldown_ready",
        "cooldown_started",
        "health_regenerated",
        "lethal_damage_contribution",
        "ordinary_movement_phase_displacement",
        "recipient_health_resolution",
        "respawn_wave_occurred",
        "source_damage_output",
        "source_healing_output",
        "spawn_shield_expired",
        "status_aged_to_zero",
        "status_applied",
        "status_broken_by_damage",
        "status_cleared_by_new_death",
        "status_refreshed_or_extended",
    }
)
_CONTROLLED_PROVENANCE_GENERATION_CHILD = r"""
import json
import sys
from pathlib import Path

import scripts.dev.generate_visual_debugger_sample_replays as generator
import scripts.dev.visual_debugger.control as control
from marl_battlegrounds.evaluation.models import CodeRevisionV1
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from scripts.dev.visual_debugger.sample_replays import (
    SAMPLE_REPLAYS,
    load_verified_sample_replay,
    read_sample_replay_manifest,
)

historical_directory = Path(sys.argv[1])
output_directory = Path(sys.argv[2])
manifest, _rows = read_sample_replay_manifest(historical_directory)
provenance = manifest["demo_provenance"]
code_payload = dict(provenance["code_revision"])
if manifest["schema_version"] == 2:
    if code_payload.pop("schema_version") != 2:
        raise SystemExit("current source schema marker is invalid")
runtime_payload = provenance["runtime_provenance"]
code_revision = CodeRevisionV1.model_validate_json(
    json.dumps(code_payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
)
runtime_provenance = RuntimeProvenanceV1.model_validate_json(
    json.dumps(runtime_payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
)
if code_revision.model_dump(mode="json") != code_payload:
    raise SystemExit("historical code provenance is not canonical")
if runtime_provenance.model_dump(mode="json") != runtime_payload:
    raise SystemExit("historical runtime provenance is not canonical")

historical_map_origin = {
    sample.source_scenario: any(
        row.name == "map_origin"
        for row in load_verified_sample_replay(
            sample.name, directory=historical_directory
        ).replay.header.context.aggregation_keys
    )
    for sample in SAMPLE_REPLAYS
}
build_context = control.build_debugger_evaluation_context_v1


def controlled_context(*args, **kwargs):
    context = build_context(*args, **kwargs)
    scenario = next(
        row.value for row in context.aggregation_keys if row.name == "scenario"
    )
    if not historical_map_origin[scenario]:
        # Preserve the archived metadata contract in this historical reproduction
        # only. Normal current generation still records its explicit custom origin.
        origin = next(
            row.value for row in context.aggregation_keys if row.name == "map_origin"
        )
        if origin != "custom":
            raise RuntimeError(
                "historical diagnostic unexpectedly claims an approved map"
            )
        context = context.model_copy(update={
            "aggregation_keys": tuple(
                row for row in context.aggregation_keys if row.name != "map_origin"
            )
        })
    return context


def controlled_revision(
    _repository_root: Path,
    *,
    package_version: str | None = None,
) -> CodeRevisionV1:
    if package_version != code_revision.package_version:
        raise RuntimeError("controlled package version differs from history")
    return code_revision


def controlled_runtime(
    revision: CodeRevisionV1,
    *,
    policy_execution_included: bool,
) -> RuntimeProvenanceV1:
    if revision != code_revision:
        raise RuntimeError("controlled runtime received a different revision")
    if policy_execution_included:
        raise RuntimeError("sample generation unexpectedly included policy execution")
    return runtime_provenance


generator.discover_code_revision_v1 = controlled_revision
generator.capture_debugger_runtime_provenance_v1 = controlled_runtime
control.build_debugger_evaluation_context_v1 = controlled_context
generated = generator.generate_sample_replays(output_directory)
print(f"generated {len(generated['samples'])} sample replays")
"""


def _cpu_child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return environment


def _generate_samples_in_cpu_child(
    output_directory: Path,
    *,
    historical_provenance_directory: Path | None = None,
) -> None:
    if historical_provenance_directory is None:
        command = (
            sys.executable,
            os.fspath(_GENERATOR_SCRIPT),
            "--generate",
            "--output-directory",
            os.fspath(output_directory),
        )
    else:
        command = (
            sys.executable,
            "-c",
            _CONTROLLED_PROVENANCE_GENERATION_CHILD,
            os.fspath(historical_provenance_directory),
            os.fspath(output_directory),
        )
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        env=_cpu_child_environment(),
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "generated 3 sample replays"
    assert completed.stderr == ""


@pytest.fixture(scope="module")
def generated_sample_directories(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("visual-debugger-samples")
    first = root / "first"
    second = root / "second"
    parent_platform_selection = os.environ.get("JAX_PLATFORMS")
    parent_default_backend = generator_module.jax.default_backend()
    _generate_samples_in_cpu_child(first)
    _generate_samples_in_cpu_child(
        second,
        historical_provenance_directory=first,
    )
    assert os.environ.get("JAX_PLATFORMS") == parent_platform_selection
    assert generator_module.jax.default_backend() == parent_default_backend
    return first, second


@pytest.fixture(scope="module")
def historical_sample_directories() -> tuple[Path, Path]:
    return SAMPLE_REPLAY_DIRECTORY, SAMPLE_REPLAY_DIRECTORY


def _file_bytes_by_name(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir(), key=lambda row: row.name)
    }


def _manifest_object(directory: Path) -> dict[str, object]:
    raw_manifest: object = json.loads(
        (directory / SAMPLE_REPLAY_MANIFEST_PATH.name).read_text(encoding="utf-8")
    )
    assert type(raw_manifest) is dict
    return cast(dict[str, object], raw_manifest)


def _first_sample_row(manifest: dict[str, object]) -> dict[str, object]:
    raw_samples = manifest["samples"]
    assert type(raw_samples) is list
    samples = cast(list[object], raw_samples)
    assert samples and type(samples[0]) is dict
    return cast(dict[str, object], samples[0])


def _write_manifest(directory: Path, manifest: dict[str, object]) -> None:
    (directory / SAMPLE_REPLAY_MANIFEST_PATH.name).write_bytes(
        canonical_sample_replay_manifest_json_bytes_v1(manifest)
    )


def _member_path(directory: Path, member: str) -> Path:
    sample = SAMPLE_REPLAYS[0]
    if member == "manifest":
        return directory / SAMPLE_REPLAY_MANIFEST_PATH.name
    if member == "replay":
        return sample.replay_path(directory)
    if member == "metric":
        return sample.metric_report_path(directory)
    raise AssertionError(f"unknown test member: {member}")


def _verify_with_boundary(boundary: str, directory: Path) -> None:
    if boundary == "light":
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=directory)
        return
    if boundary == "heavy":
        verify_sample_replays(directory)
        return
    raise AssertionError(f"unknown test boundary: {boundary}")


def _stub_truthful_provenance_capture(
    monkeypatch: pytest.MonkeyPatch,
    generated_sample_directories: tuple[Path, Path],
) -> None:
    first, _second = generated_sample_directories
    loaded = load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=first)

    def fake_revision(
        _repository_root: Path,
        *,
        package_version: str | None = None,
    ) -> CodeRevisionV1:
        assert package_version == "0.0.0"
        return CodeRevisionV1.model_validate(
            loaded.replay.header.context.code_revision.model_dump(
                exclude={"schema_version"}
            )
        )

    def fake_runtime(
        _revision: CodeRevisionV1,
        *,
        policy_execution_included: bool,
    ) -> RuntimeProvenanceV1:
        assert not policy_execution_included
        return loaded.replay.header.runtime_provenance

    monkeypatch.setattr(
        generator_module,
        "discover_code_revision_v1",
        fake_revision,
    )
    monkeypatch.setattr(
        generator_module,
        "capture_debugger_runtime_provenance_v1",
        fake_runtime,
    )


def test_registry_has_three_stable_lower_kebab_sample_names() -> None:
    assert tuple(sample.name for sample in SAMPLE_REPLAYS) == (
        "death-respawn-shield",
        "recovery-status-lifecycle",
        "mirrored-five-class-ultimates",
    )
    assert tuple(sample.source_scenario for sample in SAMPLE_REPLAYS) == (
        "death_respawn_cycle",
        "recovery_refresh_cycle",
        "mirrored_ultimates",
    )
    assert len({sample.replay_file_name for sample in SAMPLE_REPLAYS}) == 3
    assert len({sample.metric_report_file_name for sample in SAMPLE_REPLAYS}) == 3


def test_real_generator_is_byte_stable_and_publicly_reloadable(
    generated_sample_directories: tuple[Path, Path],
) -> None:
    first, second = generated_sample_directories

    manifest = _manifest_object(first)
    provenance = cast(dict[str, object], manifest["demo_provenance"])
    assert _file_bytes_by_name(first) == _file_bytes_by_name(second)
    assert verify_sample_replays(first) == verify_sample_replays(second)
    expected_revision = CodeRevisionV2.model_validate(provenance["code_revision"])
    expected_runtime = RuntimeProvenanceV1.model_validate_json(
        json.dumps(provenance["runtime_provenance"])
    )
    for sample in SAMPLE_REPLAYS:
        loaded = load_verified_sample_replay(sample.name, directory=first)
        assert loaded.status == "not_recorded"
        assert loaded.metric_report_artifact is None
        assert loaded.replay.schema_version == 3
        assert loaded.replay.completion.completion_state == "complete"
        assert len(loaded.replay.transitions) > 0
        assert len(loaded.replay.frames) == len(loaded.replay.transitions) + 1
        assert (
            CodeRevisionV2.model_validate(
                loaded.replay.header.context.code_revision.model_dump(mode="python")
            )
            == expected_revision
        )
        assert loaded.replay.header.runtime_provenance == expected_runtime
        assert expected_runtime.backend == "cpu"
        assert not expected_runtime.policy_execution_included


def test_fresh_samples_use_exact_researcher_geometry_and_event_union(
    generated_sample_directories: tuple[Path, Path],
) -> None:
    first, _second = generated_sample_directories
    assert len(_file_bytes_by_name(first)) == 4

    observed_event_kinds: set[str] = set()
    for sample in SAMPLE_REPLAYS:
        loaded = load_verified_sample_replay(sample.name, directory=first)
        resolved_config = loaded.replay.header.context.resolved_env_config
        assert (resolved_config.map_width, resolved_config.map_height) == (
            18.0,
            12.0,
        )
        observed_event_kinds.update(
            event.event_type
            for transition in loaded.replay.transitions
            for event in transition.events
        )

    assert observed_event_kinds == _PRE_M7_EVENT_KIND_UNION


def test_checked_samples_match_fresh_cpu_generation_scientific_truth(
    tmp_path: Path,
) -> None:
    expected = _file_bytes_by_name(SAMPLE_REPLAY_DIRECTORY)
    assert len(expected) == 7
    historical_manifest = verify_sample_replays(SAMPLE_REPLAY_DIRECTORY)
    fresh = tmp_path / "fresh"
    _generate_samples_in_cpu_child(
        fresh,
        historical_provenance_directory=SAMPLE_REPLAY_DIRECTORY,
    )
    fresh_manifest = verify_sample_replays(fresh)
    assert fresh_manifest["schema_version"] == 2
    assert fresh_manifest["generator_id"] == "visual-debugger-sample-replays-v2"
    assert len(_file_bytes_by_name(fresh)) == 4
    for sample in SAMPLE_REPLAYS:
        historical = load_verified_sample_replay(sample.name)
        current = load_verified_sample_replay(sample.name, directory=fresh)
        assert type(historical.replay) is ReplayArtifactV1
        assert historical.metric_report_artifact is not None
        assert historical.metric_report_artifact.report.statistics == ()
        assert current.metric_report_artifact is None
        assert current.replay.schema_version == 3
        assert current.replay.transitions == historical.replay.transitions
        assert current.replay.completion == historical.replay.completion

        old_context = historical.replay.header.context.model_dump(mode="python")
        expected_context = historical.replay.header.context.model_dump(mode="python")
        expected_context["schema_version"] = 3
        expected_context["scenario_name"] = sample.source_scenario
        expected_context["actor_projection"]["version"] = 3
        for binding in expected_context["schema_versions"]:
            if binding["schema_id"].endswith("episode_context"):
                binding["schema_version"] = 3
            elif binding["schema_id"].endswith(".frame"):
                binding["schema_version"] = 2
        for assignment in expected_context["policy_assignments"]:
            if assignment["assignment_status"] == "assigned":
                assignment["callable_name"] = None
                assignment["lifecycle"] = "frozen"
                assignment["preprocessing"]["version"] = 3
        assert (
            current.replay.header.context.model_dump(mode="python") == expected_context
        )
        assert old_context["code_revision"] == expected_context["code_revision"]
        for old_frame, new_frame in zip(
            historical.replay.frames, current.replay.frames, strict=True
        ):
            expected_frame = old_frame.model_dump(mode="python")
            expected_frame["schema_version"] = 2
            observation = expected_frame["base_observation"]
            observation["self_ally_index"] = tuple(
                row.team_local_slot if row.configured_active else 0
                for row in historical.replay.header.context.roster
            )
            for field in ("self_features", "ally_unit_features", "enemy_unit_features"):
                material = observation[field]
                if field == "self_features":
                    observation[field] = tuple(
                        (*row[:3], 0.0, *row[4:]) for row in material
                    )
                else:
                    visibility = observation[
                        "enemy_visibility_mask"
                        if field == "enemy_unit_features"
                        else "ally_visibility_mask"
                    ]
                    observation[field] = tuple(
                        tuple(
                            (
                                *row[:3],
                                float(field == "enemy_unit_features" and visible),
                                *row[4:],
                            )
                            for row, visible in zip(rows, mask, strict=True)
                        )
                        for rows, mask in zip(material, visibility, strict=True)
                    )
            assert new_frame.model_dump(mode="python") == expected_frame

        old_analysis = analyze_replay(historical)
        new_analysis = analyze_replay(current)
        assert old_analysis.original_metric_status == "empty"
        assert new_analysis.original_metric_status == "not_recorded"
        assert (
            old_analysis.source_replay_digest
            == historical.replay.canonical_digest_sha256
        )
        assert (
            new_analysis.source_replay_digest == current.replay.canonical_digest_sha256
        )
        assert (
            old_analysis.summary(old_analysis.frame_count - 1)["statistics"]
            == new_analysis.summary(new_analysis.frame_count - 1)["statistics"]
        )

    old_provenance = cast(dict[str, object], historical_manifest["demo_provenance"])
    new_provenance = cast(dict[str, object], fresh_manifest["demo_provenance"])
    expected_provenance = dict(old_provenance)
    expected_provenance["code_revision"] = CodeRevisionV2.model_validate(
        old_provenance["code_revision"]
    ).model_dump(mode="json")
    assert new_provenance == expected_provenance
    old_rows = cast(list[dict[str, object]], historical_manifest["samples"])
    new_rows = cast(list[dict[str, object]], fresh_manifest["samples"])
    for old_row, new_row in zip(old_rows, new_rows, strict=True):
        assert {
            key: value
            for key, value in old_row.items()
            if key not in ("replay", "metric_report")
        } == {key: value for key, value in new_row.items() if key != "replay"}
    assert _file_bytes_by_name(SAMPLE_REPLAY_DIRECTORY) == expected


def test_generator_refuses_to_overwrite_an_existing_directory(
    generated_sample_directories: tuple[Path, Path],
) -> None:
    first, _second = generated_sample_directories
    before = _file_bytes_by_name(first)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_sample_replays(first)

    assert _file_bytes_by_name(first) == before


def test_generator_refuses_dangling_symlink_output_without_staging(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "dangling-output"
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    assert destination.is_symlink()
    assert not destination.exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_sample_replays(destination)

    assert destination.is_symlink()
    assert tuple(tmp_path.iterdir()) == (destination,)


def test_generator_rejects_non_cpu_backend_before_recording_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "gpu-output"

    monkeypatch.setattr(generator_module.jax, "default_backend", lambda: "gpu")

    def fail_if_recorded(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("non-CPU generation must not begin recording")

    monkeypatch.setattr(generator_module, "_record_one_sample", fail_if_recorded)

    with pytest.raises(RuntimeError, match="actual backend is 'gpu'"):
        generate_sample_replays(destination)

    assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_generator_captures_truthful_provenance_before_creating_output_parent(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second = generated_sample_directories
    loaded = load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=first)
    destination = tmp_path / "absent-parent" / "truthful-output"
    events: list[str] = []

    def fake_revision(
        repository_root: Path,
        *,
        package_version: str | None = None,
    ) -> CodeRevisionV1:
        assert repository_root == _REPOSITORY_ROOT
        assert package_version == "0.0.0"
        assert not destination.parent.exists()
        events.append("source")
        return CodeRevisionV1.model_validate(
            loaded.replay.header.context.code_revision.model_dump(
                exclude={"schema_version"}
            )
        )

    def fake_runtime(
        revision: CodeRevisionV1,
        *,
        policy_execution_included: bool,
    ) -> RuntimeProvenanceV1:
        assert revision == loaded.replay.header.context.code_revision
        assert not policy_execution_included
        assert not destination.parent.exists()
        events.append("runtime")
        return loaded.replay.header.runtime_provenance

    def fail_before_record(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("record")
        raise RuntimeError("injected post-provenance failure")

    monkeypatch.setattr(generator_module, "_require_cpu_backend", lambda: None)
    monkeypatch.setattr(
        generator_module,
        "discover_code_revision_v1",
        fake_revision,
    )
    monkeypatch.setattr(
        generator_module,
        "capture_debugger_runtime_provenance_v1",
        fake_runtime,
    )
    monkeypatch.setattr(generator_module, "_record_one_sample", fail_before_record)

    with pytest.raises(RuntimeError, match="post-provenance failure"):
        generate_sample_replays(destination)

    assert events == ["source", "runtime", "record"]
    assert not destination.exists()
    assert destination.parent.is_dir()
    assert tuple(destination.parent.iterdir()) == ()


def test_generator_rechecks_dangling_symlink_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_sample_directories: tuple[Path, Path],
) -> None:
    destination = tmp_path / "late-output"

    def fake_record(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    def inject_late_target(_directory: Path) -> dict[str, object]:
        destination.symlink_to(
            tmp_path / "late-missing-target",
            target_is_directory=True,
        )
        return {}

    monkeypatch.setattr(generator_module, "_record_one_sample", fake_record)
    monkeypatch.setattr(generator_module, "verify_sample_replays", inject_late_target)
    monkeypatch.setattr(generator_module, "_require_cpu_backend", lambda: None)
    _stub_truthful_provenance_capture(monkeypatch, generated_sample_directories)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_sample_replays(destination)

    assert destination.is_symlink()
    assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == (destination,)


@pytest.mark.parametrize("attacker_entry", ("empty-directory", "dangling-symlink"))
def test_atomic_publication_cannot_replace_entry_inserted_after_final_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attacker_entry: str,
    generated_sample_directories: tuple[Path, Path],
) -> None:
    destination = tmp_path / "publication-race"
    actual_publish = cast(
        Callable[[Path, Path], None],
        vars(generator_module)["_atomic_publish_directory_noreplace"],
    )

    def fake_record(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    def fake_verify(_path: Path) -> dict[str, object]:
        return {}

    def inject_at_publish(source: Path, target: Path) -> None:
        assert target == destination
        if attacker_entry == "empty-directory":
            destination.mkdir()
        else:
            destination.symlink_to(
                tmp_path / "attacker-missing-target",
                target_is_directory=True,
            )
        actual_publish(source, target)

    monkeypatch.setattr(generator_module, "_record_one_sample", fake_record)
    monkeypatch.setattr(generator_module, "verify_sample_replays", fake_verify)
    monkeypatch.setattr(generator_module, "_require_cpu_backend", lambda: None)
    _stub_truthful_provenance_capture(monkeypatch, generated_sample_directories)
    monkeypatch.setattr(
        generator_module,
        "_atomic_publish_directory_noreplace",
        inject_at_publish,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_sample_replays(destination)

    if attacker_entry == "empty-directory":
        assert destination.is_dir()
        assert tuple(destination.iterdir()) == ()
    else:
        assert destination.is_symlink()
        assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == (destination,)


def test_generation_failure_cleans_owned_staging_without_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_sample_directories: tuple[Path, Path],
) -> None:
    destination = tmp_path / "failed-set"

    def fail_record(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(generator_module, "_record_one_sample", fail_record)
    monkeypatch.setattr(generator_module, "_require_cpu_backend", lambda: None)
    _stub_truthful_provenance_capture(monkeypatch, generated_sample_directories)

    with pytest.raises(RuntimeError, match="injected generation failure"):
        generate_sample_replays(destination)

    assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_load_boundary_rejects_tampered_member_before_public_load(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "tampered"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    sample.replay_path(copied).write_bytes(
        sample.replay_path(copied).read_bytes() + b"\n"
    )

    with pytest.raises(SampleReplayVerificationError, match="byte length mismatch"):
        load_verified_sample_replay(sample.name, directory=copied)


def test_load_boundary_requires_the_registered_metric_sidecar(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "missing-sidecar"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    sample.metric_report_path(copied).unlink()

    with pytest.raises(SampleReplayVerificationError, match="could not be opened"):
        load_verified_sample_replay(sample.name, directory=copied)


@pytest.mark.parametrize("boundary", ("light", "heavy"))
@pytest.mark.parametrize("member", ("manifest", "replay", "metric"))
def test_verifiers_reject_stationary_symlink_members_before_read(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    boundary: str,
    member: str,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / f"{boundary}-symlink-{member}"
    shutil.copytree(first, copied)
    attacked = _member_path(copied, member)
    source = _member_path(first, member)
    attacked.unlink()
    attacked.symlink_to(source)

    with pytest.raises(SampleReplayVerificationError, match="no-follow"):
        _verify_with_boundary(boundary, copied)


@pytest.mark.parametrize("boundary", ("light", "heavy"))
@pytest.mark.parametrize(
    ("member", "size_limit"),
    (
        ("manifest", SAMPLE_REPLAY_MAX_MANIFEST_SIZE_BYTES),
        ("replay", SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES),
        ("metric", SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES),
    ),
)
def test_verifiers_reject_oversized_members_before_read(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    boundary: str,
    member: str,
    size_limit: int,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / f"{boundary}-oversized-{member}"
    shutil.copytree(first, copied)
    attacked = _member_path(copied, member)
    with attacked.open("wb") as stream:
        stream.truncate(size_limit + 1)

    with pytest.raises(SampleReplayVerificationError, match="size limit"):
        _verify_with_boundary(boundary, copied)


@pytest.mark.parametrize("boundary", ("light", "heavy"))
@pytest.mark.parametrize("member", ("manifest", "replay", "metric"))
def test_verifiers_reject_fifo_members_without_blocking(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    boundary: str,
    member: str,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / f"{boundary}-fifo-{member}"
    shutil.copytree(first, copied)
    attacked = _member_path(copied, member)
    attacked.unlink()
    os.mkfifo(attacked)
    child_code = """
import sys
from pathlib import Path
from scripts.dev.generate_visual_debugger_sample_replays import verify_sample_replays
from scripts.dev.visual_debugger.sample_replays import (
    SampleReplayVerificationError,
    load_verified_sample_replay,
)
directory = Path(sys.argv[1])
try:
    if sys.argv[2] == "light":
        load_verified_sample_replay(sys.argv[3], directory=directory)
    else:
        verify_sample_replays(directory)
except SampleReplayVerificationError as error:
    print(f"REJECTED: {error}")
    raise SystemExit(0)
raise SystemExit("FIFO was unexpectedly accepted")
"""

    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            child_code,
            os.fspath(copied),
            boundary,
            SAMPLE_REPLAYS[0].name,
        ),
        cwd=_REPOSITORY_ROOT,
        env=_cpu_child_environment(),
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REJECTED:")


def test_bounded_reader_rejects_growth_after_initial_fstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacked = tmp_path / "growing.json"
    attacked.write_bytes(b"x" * 4096)
    attacked_metadata = attacked.stat()
    size_limit = 8192
    real_read = os.read
    grew = False

    def grow_before_read(descriptor: int, size: int) -> bytes:
        nonlocal grew
        metadata = os.fstat(descriptor)
        if (
            not grew
            and metadata.st_dev == attacked_metadata.st_dev
            and metadata.st_ino == attacked_metadata.st_ino
        ):
            with attacked.open("r+b") as stream:
                stream.truncate(size_limit + 1)
            grew = True
        return real_read(descriptor, size)

    monkeypatch.setattr(sample_replays_module.os, "read", grow_before_read)

    with pytest.raises(SampleReplayVerificationError, match="size limit"):
        read_bounded_regular_file_v1(
            attacked,
            max_size_bytes=size_limit,
            label="growing test member",
        )

    assert grew


def test_bounded_reader_rejects_same_length_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacked = tmp_path / "mutating.json"
    attacked.write_bytes(b"x" * 4096)
    attacked_metadata = attacked.stat()
    real_read = os.read
    mutated = False

    def mutate_before_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        metadata = os.fstat(descriptor)
        if (
            not mutated
            and metadata.st_dev == attacked_metadata.st_dev
            and metadata.st_ino == attacked_metadata.st_ino
        ):
            with attacked.open("r+b") as stream:
                stream.write(b"y" * 4096)
            os.utime(
                attacked,
                ns=(attacked_metadata.st_atime_ns, attacked_metadata.st_mtime_ns + 1),
            )
            mutated = True
        return real_read(descriptor, size)

    monkeypatch.setattr(sample_replays_module.os, "read", mutate_before_read)

    with pytest.raises(
        SampleReplayVerificationError, match="changed while it was read"
    ):
        read_bounded_regular_file_v1(
            attacked,
            max_size_bytes=8192,
            label="same-length mutation test member",
        )

    assert mutated


def test_bounded_reader_fails_closed_without_no_follow_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = tmp_path / "member.json"
    member.write_bytes(b"{}\n")
    monkeypatch.setattr(sample_replays_module.os, "O_NOFOLLOW", 0)

    with pytest.raises(SampleReplayVerificationError, match="requires no-follow"):
        read_bounded_regular_file_v1(
            member,
            max_size_bytes=1024,
            label="unsupported-platform test member",
        )


def test_sample_load_stays_bound_to_opened_bytes_during_path_replacement(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "replacement-race"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    attacked = sample.replay_path(copied)
    attacked_metadata = attacked.stat()
    held = copied / "opened-replay-held-by-test.json"
    replacement = SAMPLE_REPLAYS[1].replay_path(copied)
    real_read = os.read
    replaced = False

    def replace_before_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        metadata = os.fstat(descriptor)
        if (
            not replaced
            and metadata.st_dev == attacked_metadata.st_dev
            and metadata.st_ino == attacked_metadata.st_ino
        ):
            attacked.rename(held)
            attacked.symlink_to(replacement)
            replaced = True
        return real_read(descriptor, size)

    monkeypatch.setattr(sample_replays_module.os, "read", replace_before_read)

    with pytest.raises(
        SampleReplayVerificationError, match="changed while it was read"
    ):
        load_verified_sample_replay(sample.name, directory=copied)

    assert replaced
    assert attacked.is_symlink()


def test_sample_load_rejects_member_replacement_after_verified_read(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "post-read-replacement-race"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    metric = sample.metric_report_path(copied)
    replacement = SAMPLE_REPLAYS[1].metric_report_path(copied)
    actual_write = cast(
        Callable[[Path, bytes], None],
        vars(sample_replays_module)["_write_private_member"],
    )
    replaced = False

    def replace_after_source_reads(path: Path, payload: bytes) -> None:
        nonlocal replaced
        if not replaced:
            metric.unlink()
            metric.symlink_to(replacement)
            replaced = True
        actual_write(path, payload)

    monkeypatch.setattr(
        sample_replays_module,
        "_write_private_member",
        replace_after_source_reads,
    )

    with pytest.raises(SampleReplayVerificationError, match="changed during"):
        load_verified_sample_replay(sample.name, directory=copied)

    assert replaced
    assert metric.is_symlink()


def test_sample_load_stays_bound_to_held_directory_during_root_replacement(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "directory-race"
    held = tmp_path / "directory-race-held"
    attacker = tmp_path / "directory-race-attacker"
    attacker.mkdir()
    shutil.copytree(first, copied)
    manifest_path = copied / SAMPLE_REPLAY_MANIFEST_PATH.name
    manifest_metadata = manifest_path.stat()
    real_read = os.read
    replaced = False

    def replace_root_before_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        metadata = os.fstat(descriptor)
        if (
            not replaced
            and metadata.st_dev == manifest_metadata.st_dev
            and metadata.st_ino == manifest_metadata.st_ino
        ):
            copied.rename(held)
            copied.symlink_to(attacker, target_is_directory=True)
            replaced = True
        return real_read(descriptor, size)

    monkeypatch.setattr(sample_replays_module.os, "read", replace_root_before_read)

    with pytest.raises(SampleReplayVerificationError, match="directory changed"):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=copied)

    assert replaced
    assert copied.is_symlink()


def test_heavy_verifier_uses_one_manifest_snapshot_for_the_complete_set(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "manifest-snapshot-race"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    replay_metadata = sample.replay_path(copied).stat()
    replacement_manifest = tmp_path / "replacement-manifest.json"
    manifest = _manifest_object(copied)
    provenance = cast(dict[str, object], manifest["demo_provenance"])
    provenance["official"] = True
    replacement_manifest.write_bytes(
        canonical_sample_replay_manifest_json_bytes_v1(manifest)
    )
    manifest_path = copied / SAMPLE_REPLAY_MANIFEST_PATH.name
    real_read = os.read
    replaced = False

    def replace_manifest_before_replay_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        metadata = os.fstat(descriptor)
        if (
            not replaced
            and metadata.st_dev == replay_metadata.st_dev
            and metadata.st_ino == replay_metadata.st_ino
        ):
            replacement_manifest.replace(manifest_path)
            replaced = True
        return real_read(descriptor, size)

    monkeypatch.setattr(
        sample_replays_module.os,
        "read",
        replace_manifest_before_replay_read,
    )

    with pytest.raises(
        SampleReplayVerificationError,
        match=r"manifest changed|directory changed",
    ):
        verify_sample_replays(copied)

    assert replaced
    assert (
        cast(dict[str, object], _manifest_object(copied)["demo_provenance"])["official"]
        is True
    )


def test_heavy_verifier_rechecks_earlier_members_after_later_sample_load(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "complete-set-member-race"
    shutil.copytree(first, copied)
    earlier = SAMPLE_REPLAYS[0].replay_path(copied)
    later = SAMPLE_REPLAYS[1].replay_path(copied)
    later_metadata = later.stat()
    real_read = os.read
    mutated = False

    def mutate_earlier_when_later_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        metadata = os.fstat(descriptor)
        if (
            not mutated
            and metadata.st_dev == later_metadata.st_dev
            and metadata.st_ino == later_metadata.st_ino
        ):
            payload = earlier.read_bytes()
            with earlier.open("r+b") as stream:
                stream.write(payload)
            mutated = True
        return real_read(descriptor, size)

    monkeypatch.setattr(
        sample_replays_module.os,
        "read",
        mutate_earlier_when_later_read,
    )

    with pytest.raises(SampleReplayVerificationError, match="changed during"):
        verify_sample_replays(copied)

    assert mutated


def test_load_boundary_wraps_public_replay_validation_errors(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "invalid-public-replay"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    invalid_replay = b"{}\n"
    sample.replay_path(copied).write_bytes(invalid_replay)
    manifest = _manifest_object(copied)
    replay_row = cast(dict[str, object], _first_sample_row(manifest)["replay"])
    replay_row["byte_length"] = len(invalid_replay)
    replay_row["sha256"] = hashlib.sha256(invalid_replay).hexdigest()
    _write_manifest(copied, manifest)

    with pytest.raises(
        SampleReplayVerificationError,
        match="failed public replay validation",
    ):
        load_verified_sample_replay(sample.name, directory=copied)


def test_load_boundary_wraps_public_loader_io_errors(
    historical_sample_directories: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _second = historical_sample_directories

    def fail_public_load(*_args: object, **_kwargs: object) -> object:
        raise OSError("injected public-loader I/O failure")

    monkeypatch.setattr(replay_io_module, "load_replay_bundle_v1", fail_public_load)

    with pytest.raises(
        SampleReplayVerificationError,
        match="private validation boundary",
    ):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=first)


def test_load_boundary_rejects_false_official_demo_claim(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "false-official-claim"
    shutil.copytree(first, copied)
    manifest = _manifest_object(copied)
    provenance = cast(dict[str, object], manifest["demo_provenance"])
    provenance["official"] = True
    _write_manifest(copied, manifest)

    with pytest.raises(
        SampleReplayVerificationError,
        match="explicit unofficial demo provenance",
    ):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=copied)


@pytest.mark.parametrize(
    "numeric_field",
    (
        "schema-version",
        "seed",
        "transition-count",
        "frame-count",
        "replay-byte-length",
        "metric-byte-length",
    ),
)
def test_manifest_rejects_non_exact_integer_fields(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    numeric_field: str,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / f"non-exact-{numeric_field}"
    shutil.copytree(first, copied)
    manifest = _manifest_object(copied)
    sample_row = _first_sample_row(manifest)
    if numeric_field == "schema-version":
        manifest["schema_version"] = True
    elif numeric_field == "seed":
        sample_row["seed"] = False
    elif numeric_field == "transition-count":
        sample_row["transition_count"] = 1.0
    elif numeric_field == "frame-count":
        sample_row["frame_count"] = True
    elif numeric_field == "replay-byte-length":
        replay_row = cast(dict[str, object], sample_row["replay"])
        replay_row["byte_length"] = False
    else:
        metric_row = cast(dict[str, object], sample_row["metric_report"])
        metric_row["byte_length"] = 1.0
    _write_manifest(copied, manifest)

    with pytest.raises(SampleReplayVerificationError, match="exact integer"):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=copied)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_manifest_rejects_nonfinite_json_constant(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    constant: str,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / f"nonfinite-manifest-{constant}"
    shutil.copytree(first, copied)
    manifest_path = copied / SAMPLE_REPLAY_MANIFEST_PATH.name
    payload = manifest_path.read_bytes().replace(
        b'"schema_version": 1',
        f'"schema_version": {constant}'.encode(),
        1,
    )
    manifest_path.write_bytes(payload)

    with pytest.raises(SampleReplayVerificationError, match="nonfinite"):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=copied)


def test_manifest_rejects_duplicate_json_members(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "duplicate-manifest-member"
    copied.mkdir()
    manifest_path = copied / SAMPLE_REPLAY_MANIFEST_PATH.name
    manifest_path.write_bytes(b'{"schema_version": 1, "schema_version": 1}\n')

    with pytest.raises(SampleReplayVerificationError, match="repeats JSON member"):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=copied)


def test_manifest_rejects_noncanonical_bytes(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "noncanonical-manifest"
    shutil.copytree(first, copied)
    manifest_path = copied / SAMPLE_REPLAY_MANIFEST_PATH.name
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(SampleReplayVerificationError, match="not canonical"):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=copied)


@pytest.mark.parametrize(
    ("container_name", "field_name"),
    (
        ("code_revision", "is_dirty"),
        ("runtime_provenance", "schema_version"),
        ("runtime_provenance", "environment_count"),
    ),
)
def test_manifest_rejects_nested_provenance_numeric_aliases(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    container_name: str,
    field_name: str,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / f"provenance-alias-{container_name}-{field_name}"
    shutil.copytree(first, copied)
    manifest = _manifest_object(copied)
    provenance = cast(dict[str, object], manifest["demo_provenance"])
    container = cast(dict[str, object], provenance[container_name])
    container[field_name] = 1.0
    _write_manifest(copied, manifest)

    with pytest.raises(SampleReplayVerificationError, match="strict versioned schemas"):
        load_verified_sample_replay(SAMPLE_REPLAYS[0].name, directory=copied)


def test_heavy_verifier_rejects_stale_manifest_scientific_facts(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _second = historical_sample_directories
    copied = tmp_path / "stale-manifest"
    shutil.copytree(first, copied)
    manifest = _manifest_object(copied)
    sample_row = _first_sample_row(manifest)
    sample_row["transition_count"] = cast(int, sample_row["transition_count"]) + 1
    sample_row["frame_count"] = cast(int, sample_row["frame_count"]) + 1
    _write_manifest(copied, manifest)

    with pytest.raises(
        SampleReplayVerificationError,
        match="scientific manifest facts are stale",
    ):
        verify_sample_replays(copied)


@pytest.mark.parametrize("boundary", ("light", "heavy"))
def test_verifiers_reject_symlink_directory_root(
    historical_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    boundary: str,
) -> None:
    first, _second = historical_sample_directories
    linked = tmp_path / "linked-set"
    linked.symlink_to(first, target_is_directory=True)

    with pytest.raises(SampleReplayVerificationError, match="no-follow directory"):
        _verify_with_boundary(boundary, linked)


@pytest.mark.parametrize("boundary", ("light", "heavy"))
@pytest.mark.parametrize("member", ("manifest", "replay"))
@pytest.mark.parametrize("hazard", ("symlink", "oversized", "fifo"))
def test_current_sample_members_keep_bounded_no_follow_checks(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    boundary: str,
    member: str,
    hazard: str,
) -> None:
    if hazard == "symlink":
        test_verifiers_reject_stationary_symlink_members_before_read(
            generated_sample_directories, tmp_path, boundary, member
        )
    elif hazard == "oversized":
        limit = (
            SAMPLE_REPLAY_MAX_MANIFEST_SIZE_BYTES
            if member == "manifest"
            else SAMPLE_REPLAY_MAX_MEMBER_SIZE_BYTES
        )
        test_verifiers_reject_oversized_members_before_read(
            generated_sample_directories, tmp_path, boundary, member, limit
        )
    else:
        test_verifiers_reject_fifo_members_without_blocking(
            generated_sample_directories, tmp_path, boundary, member
        )


@pytest.mark.parametrize(
    "exercise",
    (
        test_sample_load_stays_bound_to_opened_bytes_during_path_replacement,
        test_sample_load_stays_bound_to_held_directory_during_root_replacement,
        test_heavy_verifier_uses_one_manifest_snapshot_for_the_complete_set,
        test_heavy_verifier_rechecks_earlier_members_after_later_sample_load,
    ),
    ids=lambda exercise: exercise.__name__,
)
def test_current_samples_keep_snapshot_race_checks(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exercise: Callable[[tuple[Path, Path], Path, pytest.MonkeyPatch], None],
) -> None:
    exercise(generated_sample_directories, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "exercise",
    (
        test_load_boundary_rejects_tampered_member_before_public_load,
        test_load_boundary_wraps_public_replay_validation_errors,
        test_load_boundary_rejects_false_official_demo_claim,
    ),
    ids=lambda exercise: exercise.__name__,
)
def test_current_samples_keep_integrity_and_claim_checks(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    exercise: Callable[[tuple[Path, Path], Path], None],
) -> None:
    exercise(generated_sample_directories, tmp_path)


def test_current_sample_rejects_rehashed_event_fact_disagreement(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.models import (
        canonical_digest_sha256,
        canonical_json_bytes,
    )
    from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3

    first, _second = generated_sample_directories
    copied = tmp_path / "event-disagreement"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    loaded = load_verified_sample_replay(sample.name, directory=copied)
    payload = loaded.replay.model_dump(mode="python")
    event = next(
        row
        for row in payload["transitions"][0]["events"]
        if row["event_type"] == "source_damage_output"
    )
    event["raw_damage_output"] += 1.0
    payload["trajectory_content_digest_sha256"] = canonical_digest_sha256(
        {
            name: payload[name]
            for name in ("header", "completion", "frames", "transitions")
        }
    )
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload, exclude={"canonical_digest_sha256"}
    )
    forged = ReplayArtifactV3.model_validate(payload)
    raw = canonical_json_bytes(forged)
    sample.replay_path(copied).write_bytes(raw)
    manifest = _manifest_object(copied)
    member = cast(dict[str, object], _first_sample_row(manifest)["replay"])
    member["byte_length"] = len(raw)
    member["sha256"] = hashlib.sha256(raw).hexdigest()
    _write_manifest(copied, manifest)
    with pytest.raises(SampleReplayVerificationError, match="semantic validation"):
        load_verified_sample_replay(sample.name, directory=copied)


def test_current_manifest_rejects_a_historical_replay_root(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    first, _second = generated_sample_directories
    copied = tmp_path / "wrong-replay-version"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    raw = sample.replay_path().read_bytes()
    sample.replay_path(copied).write_bytes(raw)
    manifest = _manifest_object(copied)
    member = cast(dict[str, object], _first_sample_row(manifest)["replay"])
    member["byte_length"] = len(raw)
    member["sha256"] = hashlib.sha256(raw).hexdigest()
    _write_manifest(copied, manifest)
    with pytest.raises(SampleReplayVerificationError, match="replay version"):
        load_verified_sample_replay(sample.name, directory=copied)


def test_current_sample_rejects_rehashed_root_seed_disagreement(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.models import (
        canonical_digest_sha256,
        canonical_json_bytes,
    )
    from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3

    first, _second = generated_sample_directories
    copied = tmp_path / "seed-disagreement"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    loaded = load_verified_sample_replay(sample.name, directory=copied)
    assert loaded.replay.header.context.seed_protocol.root_seed == sample.seed
    payload = loaded.replay.model_dump(mode="python")
    payload["header"]["context"]["seed_protocol"]["root_seed"] = sample.seed + 1
    payload["header"]["context_digest_sha256"] = canonical_digest_sha256(
        payload["header"]["context"]
    )
    payload["trajectory_content_digest_sha256"] = canonical_digest_sha256(
        {
            name: payload[name]
            for name in ("header", "completion", "frames", "transitions")
        }
    )
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload, exclude={"canonical_digest_sha256"}
    )
    forged = ReplayArtifactV3.model_validate(payload)
    raw = canonical_json_bytes(forged)
    sample.replay_path(copied).write_bytes(raw)
    manifest = _manifest_object(copied)
    member = cast(dict[str, object], _first_sample_row(manifest)["replay"])
    member["byte_length"] = len(raw)
    member["sha256"] = hashlib.sha256(raw).hexdigest()
    _write_manifest(copied, manifest)
    with pytest.raises(
        SampleReplayVerificationError, match="scientific manifest facts"
    ):
        load_verified_sample_replay(sample.name, directory=copied)


def test_current_sample_rejects_rehashed_completion_reason(
    generated_sample_directories: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.core.types import TASK_MODE_TDM
    from marl_battlegrounds.evaluation.models import (
        canonical_digest_sha256,
        canonical_json_bytes,
    )
    from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3

    first, _second = generated_sample_directories
    copied = tmp_path / "completion-disagreement"
    shutil.copytree(first, copied)
    sample = SAMPLE_REPLAYS[0]
    original = load_verified_sample_replay(sample.name, directory=copied)
    scenario = generator_module.get_scenario(sample.source_scenario)
    config, initial_state = scenario.build_scenario()
    ending_config = config._replace(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=99,
        max_steps=len(scenario.frames),
    )
    ending_scenario = replace(
        scenario, build_scenario=lambda: (ending_config, initial_state)
    )

    def get_ending_scenario(_name: str) -> DebuggerScenario:
        return ending_scenario

    monkeypatch.setattr(generator_module, "get_scenario", get_ending_scenario)
    sample.replay_path(copied).unlink()
    row = generator_module._record_one_sample(  # pyright: ignore[reportPrivateUsage]
        sample,
        output_directory=copied,
        code_revision=CodeRevisionV1.model_validate(
            original.replay.header.context.code_revision.model_dump(
                exclude={"schema_version"}
            )
        ),
        runtime_provenance=original.replay.header.runtime_provenance,
    )
    manifest = _manifest_object(copied)
    rows = cast(list[dict[str, object]], manifest["samples"])
    rows[0] = row
    _write_manifest(copied, manifest)
    valid = load_verified_sample_replay(sample.name, directory=copied)
    reason = valid.replay.transitions[-1].owning_task_end_reason
    assert reason is not None
    assert valid.replay.completion.end_or_failure_reason == reason
    payload = valid.replay.model_dump(mode="python")
    payload["completion"]["end_or_failure_reason"] = "contradictory test reason"
    payload["trajectory_content_digest_sha256"] = canonical_digest_sha256(
        {
            name: payload[name]
            for name in ("header", "completion", "frames", "transitions")
        }
    )
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload, exclude={"canonical_digest_sha256"}
    )
    forged = ReplayArtifactV3.model_validate(payload)
    raw = canonical_json_bytes(forged)
    sample.replay_path(copied).write_bytes(raw)
    member = cast(dict[str, object], row["replay"])
    member["byte_length"] = len(raw)
    member["sha256"] = hashlib.sha256(raw).hexdigest()
    _write_manifest(copied, manifest)
    with pytest.raises(SampleReplayVerificationError, match="semantic validation"):
        load_verified_sample_replay(sample.name, directory=copied)
