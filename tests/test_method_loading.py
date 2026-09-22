"""Check the one reference loader and the pinned-opponent evidence classes.

load_method turns the three built-in names into their Policies, an exported
actor directory into its recurrent MAPPO System, and a module:function factory
into the Policy or System it returns; it refuses a learner checkpoint
directory, an empty or unknown reference and a factory that returns anything
else, and a factory's own error keeps its type. The command line's factory
branch calls the same factory loader, load_factory, and wraps every error with
its cause kept.
pinned_opponent_evidence records the built-in Random team as installed with no
controller exposure, tdm-alpha and tdm-beta as installed with known exposure
that makes all eight protected scenarios familiar, a researcher factory as
unknown exposure, and an export without its sibling learner description as a
declared export with unknown exposure.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import sys
import types
from pathlib import Path

import jax
import pytest

from marl_battlegrounds import _cli, _method_loading
from marl_battlegrounds._method_loading import load_method
from marl_battlegrounds.baselines.ppo import initialize_ppo
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    SystemInput,
    policy,
)
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    pinned_opponent_evidence,
    prepare_training_content,
)
from marl_battlegrounds.training.checkpoints import export_system


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("loading") / "actors"
    root.mkdir()
    return export_system(
        initialize_ppo(jax.random.key(3)).actor_params,
        root / "actor",
        metadata={
            "run_id": "loading",
            "seed": 3,
            "env_steps": 0,
            "checkpoint_id": "b" * 64,
        },
        spawn_frame="left",
    )


def _unused_apply(
    variables: object, memory: object, inputs: SystemInput, keys: object
) -> tuple[object, object]:
    del inputs, keys
    return variables, memory


def _factory_module(monkeypatch: pytest.MonkeyPatch, value: object) -> str:
    module = types.ModuleType("pinned_loading_factory")
    module.make = lambda: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}:make"


def test_built_in_names_exports_and_factories_resolve(
    exported: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("random", "tdm-alpha", "tdm-beta"):
        value = load_method(name)
        assert isinstance(value, Policy) and value.name == name
    loaded = load_method(str(exported))
    assert isinstance(loaded, System) and loaded.execution == "jax"
    host = System("Host Team", _unused_apply, execution="host")
    assert load_method(_factory_module(monkeypatch, host)) is host


def test_bad_references_are_refused_with_clear_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    learner = tmp_path / "learner"
    learner.mkdir()
    (learner / "checkpoint_details.json").write_text("{}")
    with pytest.raises(ValueError, match="export the actor first"):
        load_method(str(learner))
    for text in ("", "   ", "unknown-team", "a:b:c"):
        with pytest.raises(ValueError):
            load_method(text)
    with pytest.raises(TypeError, match="System or Policy"):
        load_method(_factory_module(monkeypatch, object()))
    with pytest.raises(TypeError, match="string"):
        load_method(3)  # pyright: ignore[reportArgumentType]


def test_factory_errors_keep_their_type_and_the_cli_keeps_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("pinned_failing_factory")

    def fail() -> object:
        raise RuntimeError("provider unavailable")

    module.make = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        load_method("pinned_failing_factory:make")
    with pytest.raises(_cli._FactoryError) as error:
        _cli._load_method("pinned_failing_factory:make")
    assert isinstance(error.value.__cause__, RuntimeError)
    assert _cli._load_method("tdm-alpha") == "tdm-alpha"
    seen: list[str] = []
    chosen = policy("random")

    def spy(reference: str) -> Policy:
        seen.append(reference)
        return chosen

    monkeypatch.setattr(_method_loading, "load_factory", spy)
    assert _cli._load_method("any_module:make") is chosen
    assert seen == ["any_module:make"]


def test_evidence_classes_for_built_ins_factories_and_unlinked_exports(
    prepared: PreparedTrainingContent, exported: Path
) -> None:
    binding = prepared.binding
    random = pinned_opponent_evidence(binding, policy("random"))
    assert random["source"] == "installed" and random["exposure"] == "none"
    assert random["familiar_scenarios"] == []
    for name in ("tdm-alpha", "tdm-beta"):
        record = pinned_opponent_evidence(binding, policy(name))
        assert record["source"] == "installed" and record["exposure"] == "known"
        assert record["familiar_scenarios"] == list(range(1, 9))
        assert len(record["controllers"]) == 1  # pyright: ignore[reportArgumentType]
    researcher = pinned_opponent_evidence(binding, System("Researcher", _unused_apply))
    assert researcher["source"] == "researcher method"
    assert researcher["exposure"] == "unknown"
    declared = pinned_opponent_evidence(
        binding, load_method(str(exported)), export=exported
    )
    assert declared["source"] == "declared export"
    assert declared["exposure"] == "unknown"
    assert declared["familiar_scenarios"] == list(range(1, 9))
