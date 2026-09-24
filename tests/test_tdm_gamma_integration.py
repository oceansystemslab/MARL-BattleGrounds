"""Check how the built-in tdm-gamma controller is loaded, identified and recorded.

policy("tdm-gamma") is the registered GAMMA adapter (its named-versus-direct
agreement is checked with the other built-ins in test_policy_execution.py);
changing observation, key and spawn values of the same shape reuses one trace.
Its descriptor is version 2 with exactly the keys policy_id, version,
information, execution, inherited_controller, base, search, trap_hold,
hunter_trap and restrictions. Its Trap-hold limit (damage allowed at 1 Trap
tick or fewer) and its Trap order (Priest, Mage, Rogue, Warrior, Hunter) match
the module's TRAP_DAMAGE_MAX_TICKS and TRAP_ORDER. The descriptor is fresh on
every call, nested parts included, stable, distinct from ALPHA's and BETA's,
and records an unmodified BETA descriptor as ancestry.
controller_identity names GAMMA version 2 only for the registered callable; a
Policy that merely carries the name, or wraps the same function in a new
adapter, gets no GAMMA identity. pinned_opponent_evidence records direct GAMMA
as installed with known exposure through its verified BETA ancestry (all eight
protected scenarios familiar), a mixed System containing GAMMA in the existing
researcher-method format with known exposure, and GAMMA whose recorded ancestry
does not match the binding as unknown with all eight scenarios, never "none";
ALPHA, BETA and Random keep their exact earlier records.
The name works through load_method, freeze_evaluation_method, the command
line's name route and one small CPU evaluate call.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import pytest

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import _cli
from marl_battlegrounds._method_loading import load_method
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_X,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
)
from marl_battlegrounds.environment import Environment
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    _shared_apply,
    apply_policies,
    controller_identity,
    independent_policies,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_identity import canonical_digest_sha256
from marl_battlegrounds.evaluation.system_evaluation import freeze_evaluation_method
from marl_battlegrounds.policies import reactive_tdm_gamma
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
)
from marl_battlegrounds.policies.reactive_tdm_gamma import (
    reactive_tdm_gamma_controller_descriptor,
    reactive_tdm_gamma_policy,
)
from marl_battlegrounds.training._content import (
    PreparedTrainingContent,
    pinned_opponent_evidence,
    prepare_training_content,
)

_BETA_PRESSURE = "scenario-5-pressure-controller@5"


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def env() -> Environment:
    return marl_bgs.make("tdm", map_id=0, num_envs=2, metrics="none")


def test_same_shape_changed_inputs_reuse_one_trace(env: Environment) -> None:
    traces: list[int] = []

    def choose(observations: Observations, mask: ActionMask, keys: jax.Array) -> Action:
        traces.append(1)
        gamma = policy("tdm-gamma")
        return apply_policies(
            gamma.apply, gamma.apply, (), (), (), (), observations, mask, keys
        )[0]

    run = cast(Callable[..., Action], jax.jit(jax.vmap(choose)))
    observations, state = env.reset(jax.random.key(5))
    first = run(observations, state.action_mask, jnp.zeros((2, 10, 2), jnp.uint32))
    moved = observations._replace(
        observation=observations.observation._replace(
            self_features=observations.observation.self_features.at[
                ..., AGENT_FEATURE_X
            ].add(0.25),
            enemy_unit_features=observations.observation.enemy_unit_features.at[
                ..., AGENT_FEATURE_CURRENT_HEALTH
            ].multiply(0.5),
            spawn_lifecycle=observations.observation.spawn_lifecycle._replace(
                spawn_pad_positions_by_agent_by_team=observations.observation.spawn_lifecycle.spawn_pad_positions_by_agent_by_team
                + 0.5
            ),
        )
    )
    second = run(moved, state.action_mask, jnp.ones((2, 10, 2), jnp.uint32))
    jax.block_until_ready((first, second))
    assert len(traces) == 1


def test_descriptor_v2_is_fresh_stable_distinct_with_beta_ancestry() -> None:
    first = reactive_tdm_gamma_controller_descriptor()
    second = reactive_tdm_gamma_controller_descriptor()
    assert first == second and first is not second
    cast(dict[str, object], first["trap_hold"])["rule"] = "changed"
    order = cast(dict[str, object], first["hunter_trap"])["order"]
    cast(list[str], order).append("changed")
    cast(dict[str, object], first["inherited_controller"])["version"] = 99
    assert reactive_tdm_gamma_controller_descriptor() == second
    assert set(second) == {
        "policy_id",
        "version",
        "information",
        "execution",
        "inherited_controller",
        "base",
        "search",
        "trap_hold",
        "hunter_trap",
        "restrictions",
    }
    assert second["policy_id"] == "reactive-team-deathmatch-gamma-controller"
    assert second["version"] == 2
    assert second["inherited_controller"] == reactive_tdm_beta_controller_descriptor()
    trap_hold = cast(dict[str, object], second["trap_hold"])
    assert trap_hold["damage_max_trap_ticks_inclusive"] == 1
    assert reactive_tdm_gamma.TRAP_DAMAGE_MAX_TICKS == 1
    hunter_trap = cast(dict[str, object], second["hunter_trap"])
    assert hunter_trap["order"] == ["Priest", "Mage", "Rogue", "Warrior", "Hunter"]
    assert reactive_tdm_gamma.TRAP_ORDER == (
        PRIEST_CLASS_ID,
        MAGE_CLASS_ID,
        ROGUE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
    )
    digests = {
        canonical_digest_sha256(describe())
        for describe in (
            reactive_tdm_alpha_controller_descriptor,
            reactive_tdm_beta_controller_descriptor,
            reactive_tdm_gamma_controller_descriptor,
        )
    }
    assert len(digests) == 3


def test_identity_follows_the_registered_callable_not_the_name() -> None:
    identity = controller_identity(policy("tdm-gamma"))
    assert identity == {
        "identifier": "reactive-team-deathmatch-gamma-controller",
        "version": 2,
        "canonical_digest": canonical_digest_sha256(
            reactive_tdm_gamma_controller_descriptor()
        ),
    }
    renamed = Policy(name="tdm-gamma", apply=policy("random").apply)
    rewrapped = Policy(name="tdm-gamma", apply=_shared_apply(reactive_tdm_gamma_policy))
    assert controller_identity(renamed) is None
    assert controller_identity(rewrapped) is None


def test_direct_and_mixed_gamma_record_known_beta_exposure(
    prepared: PreparedTrainingContent,
) -> None:
    binding = prepared.binding
    direct = pinned_opponent_evidence(binding, policy("tdm-gamma"))
    assert direct == {
        "source": "installed",
        "exposure": "known",
        "controllers": [_BETA_PRESSURE],
        "familiar_scenarios": list(range(1, 9)),
    }
    for mixed in (
        independent_policies((policy("tdm-gamma"), policy("random"))),
        shared_policy(policy("tdm-gamma")),
    ):
        record = pinned_opponent_evidence(binding, mixed)
        assert record["source"] == "researcher method"
        assert record["exposure"] == "known"
        assert record["controllers"] == [_BETA_PRESSURE]
        assert record["familiar_scenarios"] == list(range(1, 9))
        assert mixed.components is not None
        assert record["declared_components"] == [
            dict(entry) for entry in mixed.components
        ]


def test_spoofed_or_unverifiable_gamma_stays_unknown(
    prepared: PreparedTrainingContent, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = prepared.binding
    spoofed = Policy(name="tdm-gamma", apply=_shared_apply(reactive_tdm_gamma_policy))
    unknown = {
        "source": "researcher method",
        "exposure": "unknown",
        "controllers": [],
        "familiar_scenarios": list(range(1, 9)),
    }
    assert pinned_opponent_evidence(binding, spoofed) == unknown
    original = reactive_tdm_gamma.reactive_tdm_gamma_controller_descriptor

    def drifted() -> dict[str, object]:
        descriptor = original()
        ancestry = cast(dict[str, object], descriptor["inherited_controller"])
        descriptor["inherited_controller"] = {**ancestry, "version": 99}
        return descriptor

    monkeypatch.setattr(
        reactive_tdm_gamma, "reactive_tdm_gamma_controller_descriptor", drifted
    )
    assert pinned_opponent_evidence(binding, policy("tdm-gamma")) == unknown


def test_name_route_loader_freezer_and_cpu_evaluate() -> None:
    assert _cli._load_method("tdm-gamma") == "tdm-gamma"
    loaded = load_method("tdm-gamma")
    frozen = freeze_evaluation_method("tdm-gamma")
    assert isinstance(loaded, Policy) and isinstance(frozen, Policy)
    assert loaded.apply is frozen.apply is policy("tdm-gamma").apply
    result = marl_bgs.evaluate(
        "tdm-gamma",
        "tdm-beta",
        num_episodes=2,
        maps=(0,),
        max_steps=4,
        num_envs=2,
        chunk_size=2,
        metrics="priority",
    )
    assert len(result.episodes) == 2


def test_alpha_beta_and_random_records_are_unchanged(
    prepared: PreparedTrainingContent,
) -> None:
    binding = prepared.binding
    expected = {
        "tdm-alpha": ["reactive-team-deathmatch-controller@3"],
        "tdm-beta": [_BETA_PRESSURE],
    }
    for name, controllers in expected.items():
        assert pinned_opponent_evidence(binding, policy(name)) == {
            "source": "installed",
            "exposure": "known",
            "controllers": controllers,
            "familiar_scenarios": list(range(1, 9)),
        }
    assert pinned_opponent_evidence(binding, policy("random")) == {
        "source": "installed",
        "exposure": "none",
        "controllers": [],
        "familiar_scenarios": [],
    }
