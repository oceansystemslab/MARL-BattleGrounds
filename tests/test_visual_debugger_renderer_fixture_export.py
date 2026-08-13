"""Focused tests for the synthetic renderer-fixture JSON export seam."""

import inspect
import json

import pytest
import scripts.dev.visual_debugger.export_renderer_fixture as export_module
from scripts.dev.visual_debugger.export_renderer_fixture import main
from scripts.dev.visual_debugger.renderer_fixtures import (
    get_renderer_fixture,
    renderer_fixture_to_jsonable,
)


def test_export_emits_one_exact_registered_synthetic_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = get_renderer_fixture("mixed_net_zero")

    assert main(("mixed_net_zero",)) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == f"{json.dumps(renderer_fixture_to_jsonable(fixture))}\n"
    payload = json.loads(captured.out)
    assert payload["live_frame"]["schema_version"] == 2
    assert payload["live_frame"]["frame_kind"] == "researcher_live_debugger"


def test_unknown_fixture_name_is_an_argparse_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(("not-a-renderer-fixture",))

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert captured.out == ""
    assert "invalid choice" in captured.err


def test_exporter_source_has_no_authoritative_or_runtime_dependencies() -> None:
    source = inspect.getsource(export_module)

    for forbidden_identifier in (
        "DebuggerSession",
        "ActionMask",
        "DebuggerService",
        "build_debugger_frame",
        "server",
        "web/visual_debugger",
    ):
        assert forbidden_identifier not in source
