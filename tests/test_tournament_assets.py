"""Check explicit asset preparation, immutable identity and active model loading.

These tests use temporary local files and a loopback HTTP server. They check
bounded integrity reads, missing/corrupt data, custom caches, metadata closure,
truthful controller evidence and failure before actions. No released Big 12,
external downloads, games or public catalog mutation are needed.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Never

import numpy as np
import pytest

from marl_battlegrounds.evaluation import tournament_assets as assets


def _asset(
    root: Path, identifier: str, value: bytes, role: str = "model"
) -> dict[str, Any]:
    path = root / identifier
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {
        "sha256": sha256(value).hexdigest(),
        "size_bytes": len(value),
        "role": role,
        "path": str(path),
        "url": None,
    }


def _config(**descriptors: dict[str, Any]) -> dict[str, Any]:
    return {"assets": descriptors}


def _identity_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    from marl_battlegrounds.evaluation import tournament_config

    def resolve(config: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(config)

    monkeypatch.setattr(tournament_config, "load_tournament_config", resolve)


@contextmanager
def _server(root: Path) -> Generator[str]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join()


def test_verification_reuses_hash_and_rejects_same_size_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _asset(tmp_path, "weights", b"abc")
    verifier = assets.AssetVerifier(
        _config(weights=descriptor), cache_dir=tmp_path / "cache"
    )
    original = assets._hash_file
    calls: list[Path] = []

    def measured(path: Path) -> tuple[str, int]:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(assets, "_hash_file", measured)
    path = verifier.verify("weights")
    assert path == tmp_path / "weights"
    assert verifier.verify("weights") == path
    assert len(calls) == 1
    assert path is not None
    before = path.stat()
    path.write_bytes(b"xyz")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="changed or is corrupt"):
        verifier.verify("weights")
    assert len(calls) == 2


def test_missing_reports_all_assets_without_creating_cache(tmp_path: Path) -> None:
    descriptor = _asset(tmp_path, "gone", b"one")
    Path(descriptor["path"]).unlink()
    cache = tmp_path / "absent-cache"
    verifier = assets.AssetVerifier(
        _config(a=descriptor, b=descriptor), cache_dir=cache
    )
    assert verifier.verify("a") is None
    with pytest.raises(ValueError, match=r"a \(3 bytes\).*b \(3 bytes\)"):
        verifier.require(("a", "b"))
    assert not cache.exists()


def test_requested_asset_needs_location_or_prepared_cache(tmp_path: Path) -> None:
    descriptor = _asset(tmp_path, "source", b"values")
    descriptor["path"] = None
    cache = tmp_path / "cache"
    verifier = assets.AssetVerifier(_config(model=descriptor), cache_dir=cache)
    with pytest.raises(ValueError, match="no local path, URL or cache"):
        verifier.verify("model")
    (cache / "sha256").mkdir(parents=True)
    expected = cache / "sha256" / descriptor["sha256"]
    expected.write_bytes(b"values")
    assert verifier.verify("model") == expected


@pytest.mark.parametrize("kind", ["parent", "symlink"])
def test_relative_assets_cannot_escape_bundle(tmp_path: Path, kind: str) -> None:
    descriptor = _asset(tmp_path, "outside", b"a")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    if kind == "symlink":
        (bundle / "shortcut").symlink_to(tmp_path / "outside")
        descriptor["path"] = "shortcut"
    else:
        descriptor["path"] = "../outside"
    config = _config(a=descriptor)
    config["source_location"] = str(bundle)
    with pytest.raises(ValueError, match="escapes"):
        assets.AssetVerifier(config).verify("a")
    descriptor["path"] = str(tmp_path / "outside")
    assert assets.AssetVerifier(config).verify("a") == tmp_path / "outside"


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_evidence_json_is_strict(tmp_path: Path, body: bytes) -> None:
    verifier = assets.AssetVerifier(
        _config(meta=_asset(tmp_path, "meta", body, "registration"))
    )
    with pytest.raises(ValueError):
        verifier.read_json("meta")


@pytest.mark.parametrize("size", [True, -1, 1.0])
def test_asset_sizes_are_exact_nonnegative_integers(
    tmp_path: Path, size: object
) -> None:
    descriptor = _asset(tmp_path, "weights", b"a")
    descriptor["size_bytes"] = size
    with pytest.raises(ValueError, match="identity, size or role"):
        assets.AssetVerifier(_config(weights=descriptor))


def test_explicit_download_is_atomic_and_returns_custom_cache_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _identity_resolver(monkeypatch)
    server_root = tmp_path / "http"
    server_root.mkdir()
    descriptor = _asset(server_root, "report", b"a,b\n1,2\n", "outcomes_priority")
    descriptor["path"] = None
    cache = tmp_path / "elsewhere" / "cache"
    with _server(server_root) as address:
        descriptor["url"] = address + "/report"
        config = _config(report=descriptor)
        inspected = assets.prepare_tournament_assets(
            config, cache_dir=cache, roles=("outcomes_priority",)
        )
        assert inspected["missing"] == ["report"]
        assert not cache.exists()
        prepared = assets.prepare_tournament_assets(
            config, cache_dir=cache, roles=("outcomes_priority",), download=True
        )
    assert prepared["missing"] == []
    assert prepared["bytes_downloaded"] == descriptor["size_bytes"]
    expected = cache / "sha256" / descriptor["sha256"]
    assert prepared["config"]["assets"]["report"]["path"] == str(expected)
    assert assets.AssetVerifier(prepared["config"]).verify("report") == expected
    assert config["assets"]["report"]["path"] is None
    assert not list(cache.rglob(".download-*"))


@pytest.mark.parametrize("change", ["hash", "short", "long"])
def test_bad_download_never_installs_a_complete_entry(
    tmp_path: Path, change: str
) -> None:
    server_root = tmp_path / "http"
    server_root.mkdir()
    descriptor = _asset(server_root, "weights", b"abc")
    descriptor["path"] = None
    if change == "hash":
        descriptor["sha256"] = "f" * 64
    elif change == "short":
        descriptor["size_bytes"] += 1
    else:
        descriptor["size_bytes"] -= 1
    with _server(server_root) as address:
        descriptor["url"] = address + "/weights"
        verifier = assets.AssetVerifier(
            _config(weights=descriptor), cache_dir=tmp_path / "cache"
        )
        with pytest.raises(ValueError, match=r"size or hash|declared size"):
            assets._download_asset(verifier, "weights")
    assert not (verifier.cache_dir / "sha256" / descriptor["sha256"]).exists()
    assert not list(verifier.cache_dir.rglob(".download-*"))


def test_metadata_is_included_but_models_and_replays_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _identity_resolver(monkeypatch)
    descriptor = _asset(tmp_path, "report", b"data", "outcomes_priority")
    metadata = _asset(
        tmp_path, "schedule", b'{"source_config_asset":"config"}', "schedule"
    )
    config_data = _asset(tmp_path, "config", b"{}", "configuration")
    model = _asset(tmp_path, "model", b"weights")
    replay = _asset(tmp_path, "replay", b"replay", "replay")
    Path(model["path"]).unlink()
    Path(replay["path"]).unlink()
    result = assets.prepare_tournament_assets(
        _config(
            report=descriptor,
            schedule=metadata,
            config=config_data,
            model=model,
            replay=replay,
        ),
        roles=("outcomes_priority",),
    )
    assert set(result["verified"]) == {"report", "schedule", "config"}
    assert result["missing"] == []


def test_metadata_cycles_and_unknown_references_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _identity_resolver(monkeypatch)
    first = _asset(tmp_path, "a", b'{"registration_asset":"b"}', "registration")
    second = _asset(tmp_path, "b", b'{"registration_asset":"a"}', "registration")
    with pytest.raises(ValueError, match="dependency cycle"):
        assets.prepare_tournament_assets(_config(a=first, b=second))
    with pytest.raises(ValueError, match="undeclared"):
        assets.prepare_tournament_assets(_config(a=first))


def test_controller_identity_ignores_asset_aliases_and_names(tmp_path: Path) -> None:
    data = _asset(tmp_path, "evidence", b'{"value":3}', "registration")
    config = _config(first=data, alias=data)
    verifier = assets.AssetVerifier(config)
    content: dict[str, Any] = {
        field: {"asset_id": "first", "sha256": data["sha256"]}
        for field in assets._CONTENT_FIELDS
        if field != "adapter_bindings"
    }
    content["adapter_bindings"] = []
    first: dict[str, Any] = {"kind": "builtin", "name": "one", "content": content}
    second = copy.deepcopy(first)
    second["name"] = "two"
    for key, value in second["content"].items():
        if key != "adapter_bindings":
            value["asset_id"] = "alias"
    assert assets.controller_content_identity(
        first, verifier
    ) == assets.controller_content_identity(second, verifier)
    digest, known = assets.controller_content_identity(first, verifier)
    assert len(digest) == 64 and known
    second["content"]["external_state"] = None
    assert assets.controller_content_identity(second, verifier)[1] is False


def _participant(tmp_path: Path) -> tuple[dict[str, Any], assets.AssetVerifier]:
    from marl_battlegrounds.evaluation.policy_execution import policy

    method = policy("random")
    registration = assets.loaded_controller_registration(method)
    descriptor = _asset(
        tmp_path, "registration", json.dumps(registration).encode(), "registration"
    )
    verifier = assets.AssetVerifier(_config(registration=descriptor))
    controller: dict[str, Any] = {
        "kind": "builtin",
        "name": "random",
        "content": {field: None for field in assets._CONTENT_FIELDS},
    }
    return {
        "name": method.name,
        "controller": controller,
        "controller_id": assets.controller_content_identity(controller, verifier)[0],
        "registration_asset": "registration",
    }, verifier


def test_loaded_method_matches_registration_without_an_action(tmp_path: Path) -> None:
    from marl_battlegrounds.evaluation.policy_execution import Policy

    participant, verifier = _participant(tmp_path)
    method = assets.load_tournament_controller(participant, verifier)
    assert isinstance(method, Policy)
    changed = replace(method, variables={"wrong": 3})
    with pytest.raises(ValueError, match=r"registration\.variables_digest"):
        assets.validate_loaded_controller(changed, participant, verifier)


def test_active_pair_loads_only_two_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    participant, verifier = _participant(tmp_path)
    original = assets.load_tournament_controller
    calls: list[str] = []

    def measured(value: dict[str, Any], check: assets.AssetVerifier) -> object:
        calls.append(value["name"])
        return original(value, check)

    monkeypatch.setattr(assets, "load_tournament_controller", measured)
    with (
        pytest.raises(RuntimeError, match="caller failed"),
        assets.active_tournament_pair(participant, participant, verifier) as methods,
    ):
        assert len(methods) == 2
        raise RuntimeError("caller failed")
    assert calls == [participant["name"], participant["name"]]


def test_assets_import_keeps_numerical_runtime_unloaded() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import marl_battlegrounds.evaluation.tournament_assets; "
            "assert 'jax' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_preparation_uses_real_config_resolver_and_keeps_identity(
    tmp_path: Path,
) -> None:
    from canonical_fixtures import config_descriptor
    from marl_battlegrounds.evaluation.tournament_config import snapshot_identity

    config = config_descriptor(root=tmp_path)
    for identifier, descriptor in config["assets"].items():
        config["assets"][identifier] = _asset(
            tmp_path, identifier, b"{}", descriptor["role"]
        )
    config["snapshot_id"] = snapshot_identity(config)
    before = copy.deepcopy(config)
    result = assets.prepare_tournament_assets(config, roles=("outcomes_priority",))
    assert set(result["verified"]) == set(config["assets"])
    assert result["missing"] == []
    assert result["config"]["snapshot_id"] == before["snapshot_id"]
    assert config == before


def test_schedule_game_lists_are_not_expanded_for_asset_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _identity_resolver(monkeypatch)
    games = _asset(
        tmp_path,
        "a-games",
        b'{"logical_game_id":"one"}\n{"logical_game_id":"two"}\n',
        "schedule",
    )
    manifest = _asset(tmp_path, "z-manifest", b'{"games_asset":"a-games"}', "schedule")
    original = assets.AssetVerifier.read_json
    parsed: list[str] = []

    def checked(verifier: assets.AssetVerifier, identifier: str) -> object:
        parsed.append(identifier)
        assert identifier != "a-games"
        return original(verifier, identifier)

    monkeypatch.setattr(assets.AssetVerifier, "read_json", checked)
    result = assets.prepare_tournament_assets(
        _config(**{"a-games": games, "z-manifest": manifest}), roles=()
    )
    assert result["verified"] == ["a-games", "z-manifest"]
    assert parsed == ["z-manifest"]


def test_json_cache_does_not_expose_mutable_cached_evidence(tmp_path: Path) -> None:
    verifier = assets.AssetVerifier(
        _config(meta=_asset(tmp_path, "meta", b'{"items":[1]}', "registration"))
    )
    value = verifier.read_json("meta")
    assert isinstance(value, dict)
    value["items"] = [100]
    assert verifier.read_json("meta") == {"items": [1]}


def test_download_reads_bounded_chunks_and_file_sync_failure_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x" * (assets._BUFFER_BYTES * 2 + 19)
    descriptor = _asset(tmp_path, "weights", payload)
    descriptor.update(path=None, url="https://fixture.invalid/weights")
    requested: list[int] = []

    class Response(BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 < size <= 1024 * 1024
            requested.append(size)
            return super().read(size)

    class Opener:
        def open(self, _url: str, *, timeout: int) -> Response:
            assert timeout > 0
            return Response(payload)

    def opener(*_args: object) -> Opener:
        return Opener()

    monkeypatch.setattr(assets, "build_opener", opener)
    verifier = assets.AssetVerifier(
        _config(weights=descriptor), cache_dir=tmp_path / "cache"
    )
    assert assets._download_asset(verifier, "weights") == len(payload)
    assert len(requested) >= 3
    assert verifier.require(("weights",))["weights"].read_bytes() == payload
    target = verifier.cache_dir / "sha256" / descriptor["sha256"]
    target.unlink()

    def failed_sync(_fd: int) -> None:
        raise OSError("injected file sync failure")

    monkeypatch.setattr(assets.os, "fsync", failed_sync)
    with pytest.raises(OSError, match="injected file sync failure"):
        assets._download_asset(verifier, "weights")
    assert not target.exists()
    assert not list(verifier.cache_dir.rglob(".download-*"))


def test_download_rejects_local_urls_and_redirects(tmp_path: Path) -> None:
    from urllib.request import Request

    descriptor = _asset(tmp_path, "weights", b"a")
    descriptor.update(path=None, url="file:///tmp/weights")
    verifier = assets.AssetVerifier(
        _config(weights=descriptor), cache_dir=tmp_path / "cache"
    )
    with pytest.raises(ValueError, match="HTTP"):
        assets._download_asset(verifier, "weights")
    assert not verifier.cache_dir.exists()
    with pytest.raises(ValueError, match="HTTP"):
        assets._HTTPOnlyRedirect().redirect_request(
            Request("https://fixture.invalid/file"),
            None,
            302,
            "Found",
            {},
            "file:///tmp/weights",
        )


@pytest.mark.parametrize("kind", ["factory", "bundle"])
def test_declared_loaders_run_once_with_verified_read_only_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from marl_battlegrounds.evaluation.policy_execution import policy

    participant, verifier = _participant(tmp_path)
    descriptor = _asset(tmp_path, "weights", b"values")
    verifier = assets.AssetVerifier(_config(**verifier.assets, weights=descriptor))
    content = participant["controller"]["content"]
    calls: list[object] = []

    def factory() -> object:
        calls.append(None)
        return policy("random")

    def loader(paths: dict[str, Path]) -> object:
        assert paths["weights"].read_bytes() == b"values"
        with pytest.raises(TypeError):
            paths["weights"] = tmp_path
        calls.append(dict(paths))
        return policy("random")

    if kind == "factory":
        participant["controller"] = {
            "kind": kind,
            "factory": "fixture:factory",
            "content": content,
        }
    else:
        participant["controller"] = {
            "kind": kind,
            "loader": "fixture:loader",
            "asset_ids": ["weights"],
            "content": content,
        }

    def installed(_name: str) -> Callable[..., object]:
        return factory if kind == "factory" else loader

    monkeypatch.setattr(assets, "_installed_callable", installed)
    result = assets.load_tournament_controller(participant, verifier)
    assert result.name == participant["name"]
    assert len(calls) == 1


def test_system_memory_evidence_matches_ordered_adapter_template(
    tmp_path: Path,
) -> None:
    import numpy as np

    from marl_battlegrounds.evaluation.policy_execution import policy, shared_policy
    from marl_battlegrounds.evaluation.recording_identity import tree_digest

    template = np.array([3, 7], dtype=np.int32)
    method = shared_policy(replace(policy("random"), initial_carry=template))
    registration = assets.loaded_controller_registration(method)
    metadata = _asset(
        tmp_path, "registration", json.dumps(registration).encode(), "registration"
    )
    evidence = _asset(
        tmp_path, "template", json.dumps(tree_digest(template)).encode(), "registration"
    )
    verifier = assets.AssetVerifier(_config(registration=metadata, template=evidence))
    content: dict[str, Any] = {field: None for field in assets._CONTENT_FIELDS}
    content["memory_template"] = {"asset_id": "template", "sha256": evidence["sha256"]}
    controller: dict[str, Any] = {"kind": "factory", "content": content}
    participant = {
        "name": method.name,
        "controller": controller,
        "registration_asset": "registration",
        "controller_id": assets.controller_content_identity(controller, verifier)[0],
    }
    assets.validate_loaded_controller(method, participant, verifier)
    wrong = _asset(
        tmp_path,
        "wrong",
        json.dumps(tree_digest(template + 1)).encode(),
        "registration",
    )
    verifier = assets.AssetVerifier(_config(registration=metadata, wrong=wrong))
    controller["content"]["memory_template"] = {
        "asset_id": "wrong",
        "sha256": wrong["sha256"],
    }
    participant["controller_id"] = assets.controller_content_identity(
        controller, verifier
    )[0]
    with pytest.raises(ValueError, match="declared memory_template"):
        assets.validate_loaded_controller(method, participant, verifier)


def test_known_builtin_duplicate_ignores_display_and_checkpoint_labels() -> None:
    from dataclasses import replace

    from marl_battlegrounds.evaluation.policy_execution import policy

    original = policy("tdm-alpha")
    renamed = replace(original, name="my-version", checkpoint="renamed-file")
    registration = assets.loaded_controller_registration(original)
    controller: dict[str, Any] = {
        "kind": "builtin",
        "name": "tdm-alpha",
        "content": {
            key: [] if key == "adapter_bindings" else None
            for key in (
                "code",
                "parameters",
                "memory_template",
                "input_preparation",
                "decision_settings",
                "adapter_bindings",
                "external_state",
            )
        },
    }
    verifier = assets.AssetVerifier({"assets": {}})
    identifier, _ = assets.controller_content_identity(controller, verifier)
    entrant: dict[str, Any] = {
        "entrant_id": "old",
        "controller_id": identifier,
        "controller": controller,
    }
    assert (
        assets.duplicate_controller_id(
            renamed, [entrant], {"old": registration}, verifier
        )
        == identifier
    )
    different = replace(renamed, variables=np.asarray(1.0, np.float32))
    assert (
        assets.duplicate_controller_id(
            different, [entrant], {"old": registration}, verifier
        )
        is None
    )


def test_equal_unknown_methods_are_not_declared_exact_duplicates() -> None:
    from marl_battlegrounds.evaluation.policy_execution import Policy

    def choose(*args: object) -> Never:
        raise AssertionError("Identity checking must not choose an action")

    method = Policy("unknown", choose)
    registration = assets.loaded_controller_registration(method)
    assert (
        assets.duplicate_controller_id(
            method,
            [{"entrant_id": "old", "controller": {"kind": "factory"}}],
            {"old": registration},
            assets.AssetVerifier({"assets": {}}),
        )
        is None
    )


def test_legacy_policy_projection_matches_existing_writer_digest() -> None:
    from marl_battlegrounds.evaluation.policy_execution import policy
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
        policy_description,
    )

    method = policy("tdm-beta")
    actual = assets.loaded_controller_registration(method)
    legacy = policy_description(
        method, method.variables, method.initial_carry, include_digests=True
    )
    _, recorded = normalize_system_registration(legacy, phase="tournament")
    assert assets.policy_recording_registration(actual) == recorded
    assert "hooks" not in recorded
    assert "hooks" in actual


def _environment_config(tmp_path: Path) -> dict[str, Any]:
    from marl_battlegrounds.evaluation.tournament_config import canonical_json

    sources = assets.environment_source_manifest()
    lock = assets.inference_dependency_lock()
    return {
        "assets": {
            "source": _asset(
                tmp_path, "source", canonical_json(sources), "qualification"
            ),
            "lock": _asset(tmp_path, "lock", canonical_json(lock), "dependencies"),
        },
        "compatibility": {
            "source_manifest_asset": "source",
            "dependency_lock_asset": "lock",
            "environment_id": sha256(canonical_json(sources)).hexdigest(),
        },
    }


def test_environment_execution_checks_source_and_dependency_identity(
    tmp_path: Path,
) -> None:
    config = _environment_config(tmp_path)
    verifier = assets.AssetVerifier(config)
    assets.verify_tournament_environment(config, verifier, execution=True)
    config["compatibility"]["environment_id"] = "0" * 64
    with pytest.raises(ValueError, match="differs from environment_id"):
        assets.verify_tournament_environment(config, verifier, execution=False)


def test_reuse_can_analyze_recorded_versions_without_importing_the_runtime(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.tournament_config import canonical_json

    config = _environment_config(tmp_path)
    lock = assets.inference_dependency_lock()
    lock["packages"]["jax"] = "0.0.1"
    config["assets"]["lock"] = _asset(
        tmp_path, "lock", canonical_json(lock), "dependencies"
    )
    verifier = assets.AssetVerifier(config)
    assets.verify_tournament_environment(config, verifier, execution=False)
    with pytest.raises(ValueError, match="dependencies differ"):
        assets.verify_tournament_environment(config, verifier, execution=True)
    script = """
import json, sys
from marl_battlegrounds.evaluation import tournament_assets as assets
config = json.loads(sys.argv[1])
assets.verify_tournament_environment(
    config, assets.AssetVerifier(config), execution=False)
assert 'jax' not in sys.modules
assert 'marl_battlegrounds.evaluation.policy_execution' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, json.dumps(config)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_manifest_history_is_not_a_new_dependency_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _identity_resolver(monkeypatch)
    descriptor = _asset(
        tmp_path,
        "source-manifest",
        json.dumps(
            {"details": {"old_config": {"asset_id": "removed-history"}}}
        ).encode(),
        "run_manifest",
    )
    prepared = assets.prepare_tournament_assets(_config(source=descriptor), roles=())
    assert prepared["verified"] == ["source"]
    assert prepared["missing"] == []


def test_verified_json_rejects_change_before_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _asset(tmp_path, "meta", b'{"value":1}', "registration")
    verifier = assets.AssetVerifier({"assets": {"meta": descriptor}})
    original = verifier.require

    def changed(ids: object) -> object:
        result = original(ids)  # type: ignore[arg-type]
        Path(descriptor["path"]).write_bytes(b'{"value":2}')
        return result

    monkeypatch.setattr(verifier, "require", changed)
    with pytest.raises(ValueError, match="changed after verification"):
        verifier.read_json("meta")


def test_verified_json_rejects_exponent_overflow(tmp_path: Path) -> None:
    descriptor = _asset(tmp_path, "meta", b'{"value":1e999}', "registration")
    verifier = assets.AssetVerifier({"assets": {"meta": descriptor}})
    with pytest.raises(ValueError, match="Nonfinite"):
        verifier.read_json("meta")


def test_verification_scope_checks_each_used_path_at_both_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = assets.AssetVerifier(
        _config(weights=_asset(tmp_path, "weights", b"abc"))
    )
    calls: list[str] = []
    original = verifier._candidate

    def counted(identifier: str) -> Path | None:
        calls.append(identifier)
        return original(identifier)

    monkeypatch.setattr(verifier, "_candidate", counted)
    with verifier.verification_scope():
        assert verifier.verify("weights") == tmp_path / "weights"
        with verifier.verification_scope():
            for _ in range(10):
                assert verifier.verify("weights") == tmp_path / "weights"
        assert calls == ["weights"]
    assert calls == ["weights", "weights"]
    verifier.verify("weights")
    assert calls == ["weights"] * 3


@pytest.mark.parametrize("change", ["different", "reversal", "replacement", "removed"])
def test_verification_scope_rejects_changed_assets_before_return(
    tmp_path: Path, change: str
) -> None:
    verifier = assets.AssetVerifier(
        _config(weights=_asset(tmp_path, "weights", b"abc"))
    )
    path = tmp_path / "weights"
    with (
        pytest.raises(ValueError, match="changed during verification"),
        verifier.verification_scope(),
    ):
        assert verifier.verify("weights") == path
        before = path.stat()
        if change == "removed":
            path.unlink()
        elif change == "replacement":
            other = tmp_path / "replacement"
            other.write_bytes(b"abc")
            other.replace(path)
        else:
            path.write_bytes(b"xyz")
            if change == "reversal":
                path.write_bytes(b"abc")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert verifier.verify("weights") == path
    assert verifier._scope_depth == 0
    assert not verifier._scope_paths
    path.write_bytes(b"abc")
    with verifier.verification_scope():
        assert verifier.verify("weights") == path


def test_verification_scope_rejects_changed_symlink_target(tmp_path: Path) -> None:
    descriptor = _asset(tmp_path, "first", b"abc")
    _asset(tmp_path, "second", b"abc")
    link = tmp_path / "selected"
    link.symlink_to(tmp_path / "first")
    descriptor["path"] = str(link)
    verifier = assets.AssetVerifier(_config(weights=descriptor))
    with (
        pytest.raises(ValueError, match="changed during verification"),
        verifier.verification_scope(),
    ):
        assert verifier.verify("weights") == tmp_path / "first"
        link.unlink()
        link.symlink_to(tmp_path / "second")


def test_verification_scope_nested_failure_aborts_outer_read(tmp_path: Path) -> None:
    verifier = assets.AssetVerifier(
        _config(weights=_asset(tmp_path, "weights", b"abc"))
    )
    with (
        pytest.raises(ValueError, match="scope was interrupted"),
        verifier.verification_scope(),
    ):
        verifier.verify("weights")
        with (
            pytest.raises(RuntimeError, match="reader failed"),
            verifier.verification_scope(),
        ):
            raise RuntimeError("reader failed")
        assert not verifier._scope_paths
    assert verifier._scope_depth == 0
    with verifier.verification_scope():
        assert verifier.verify("weights") == tmp_path / "weights"


def test_second_loader_failure_closes_the_first_owned_system_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.evaluation.policy_execution import Policy, shared_policy

    participant, verifier = _participant(tmp_path)
    loaded = assets.load_tournament_controller(participant, verifier)
    assert isinstance(loaded, Policy)
    base = shared_policy(loaded)
    events: list[str] = []

    @contextmanager
    def scope(recording: bool) -> Generator[None]:
        events.append("open")
        try:
            yield
        finally:
            events.append("close")

    owned = replace(base, resource_scope=scope)

    def load(*args: object) -> object:
        if events:
            raise RuntimeError("Second load failed")
        return owned

    monkeypatch.setattr(assets, "load_tournament_controller", load)
    with (
        pytest.raises(RuntimeError, match="Second load failed"),
        assets.active_tournament_pair(participant, participant, verifier),
    ):
        raise AssertionError("A failed pair must not reach execution")
    assert events == ["open", "close"]
