"""Expose the runnable example's artificial bundle for focused host tests.

The example and tests share one fixture builder. Loading this local support
module does not run a game, fit ratings or install an official catalog.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

_path = Path(__file__).resolve().parents[1] / "examples" / "canonical_fixture.py"
_spec = importlib.util.spec_from_file_location("canonical_fixture", _path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["canonical_fixture"] = _module
_spec.loader.exec_module(_module)
build_record_bundle: Any = _module.build_record_bundle
fixture_policy: Any = _module.fixture_policy
