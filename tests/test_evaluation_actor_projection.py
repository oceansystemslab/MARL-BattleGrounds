"""Check reconstruction of each NoSharedObs actor's allowed input.

Historical projection V2 rebuilds public class rows from its context. The
relative-input projection version follows the context version: context V3
records NoSharedObs V3 / SharedObs V2, and the current context V4 records
NoSharedObs V4 / SharedObs V3 (the current versions are 4 and 3). Current class
reconstruction accepts exactly those pairs, gives the same public rows for both,
and refuses any other context type or projection.
"""

import jax.numpy as jnp
import pytest
from tests.evaluation_fixtures import evaluation_context, evaluation_env_config

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    EnvConfig,
)
from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_ID,
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
    NO_SHARED_OBS_ACTOR_PROJECTION_V3,
    NO_SHARED_OBS_ACTOR_PROJECTION_V4,
    NO_SHARED_OBS_ACTOR_PROJECTION_VERSION,
    SHARED_OBS_ACTOR_PROJECTION_ID,
    SHARED_OBS_ACTOR_PROJECTION_V3,
    SHARED_OBS_ACTOR_PROJECTION_VERSION,
    reconstruct_actor_class_ids_by_team_v2,
    reconstruct_actor_class_ids_by_team_v3,
    reconstruct_class_ids_by_agent_by_team_v2,
    reconstruct_class_ids_by_agent_by_team_v3,
    validate_class_ids_by_agent_by_team_against_context_v1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    ExecutionInformationMode,
    VersionedIdentityV1,
)


def _projection_context(
    *,
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
    config: EnvConfig | None = None,
) -> EvaluationEpisodeContextV1:
    context = evaluation_context(
        execution_information_mode=execution_information_mode,
        config=evaluation_env_config() if config is None else config,
    )
    return context.model_copy(
        update={"actor_projection": NO_SHARED_OBS_ACTOR_PROJECTION_V2}
    )


def test_no_shared_obs_actor_projection_v2_has_exact_identity() -> None:
    assert NO_SHARED_OBS_ACTOR_PROJECTION_ID == "base-observation-no-shared-obs"
    assert NO_SHARED_OBS_ACTOR_PROJECTION_VERSION == 4
    assert (
        VersionedIdentityV1(
            identifier="base-observation-no-shared-obs",
            version=2,
        )
        == NO_SHARED_OBS_ACTOR_PROJECTION_V2
    )


def test_full_and_scalar_projection_reconstruct_team_relative_public_classes() -> None:
    context = _projection_context()
    full = reconstruct_class_ids_by_agent_by_team_v2(context)
    team_a = (MAGE_CLASS_ID, WARRIOR_CLASS_ID, PRIEST_CLASS_ID, 0, 0)
    team_b = (HUNTER_CLASS_ID, ROGUE_CLASS_ID, 0, 0, 0)
    zero_row = (0, 0, 0, 0, 0)

    assert len(full) == MAX_AGENT_SLOTS
    assert full[0] == (team_a, team_b)
    assert full[2] == (team_a, team_b)
    assert full[5] == (team_b, team_a)
    assert full[6] == (team_b, team_a)
    assert full[3] == (zero_row, zero_row)
    assert full[9] == (zero_row, zero_row)
    assert full[0][0][3:] == (0, 0)
    assert full[0][1][2:] == (0, 0, 0)

    for global_slot in range(MAX_AGENT_SLOTS):
        assert (
            reconstruct_actor_class_ids_by_team_v2(context, global_slot)
            == full[global_slot]
        )
    assert all(
        type(class_id) is int
        for actor_rows in full
        for team_row in actor_rows
        for class_id in team_row
    )
    validate_class_ids_by_agent_by_team_against_context_v1(context, full)


def test_projection_preserves_arbitrary_class_to_slot_permutations() -> None:
    team_a = (
        PRIEST_CLASS_ID,
        ROGUE_CLASS_ID,
        MAGE_CLASS_ID,
        HUNTER_CLASS_ID,
        WARRIOR_CLASS_ID,
    )
    team_b = (
        WARRIOR_CLASS_ID,
        MAGE_CLASS_ID,
        PRIEST_CLASS_ID,
        ROGUE_CLASS_ID,
        HUNTER_CLASS_ID,
    )
    config = evaluation_env_config(team_sizes=(5, 5))._replace(
        agent_profile=resolve_agent_profile(
            jnp.asarray((*team_a, *team_b), dtype=jnp.int32),
            jnp.asarray((5, 5), dtype=jnp.int32),
        )
    )
    full = reconstruct_class_ids_by_agent_by_team_v2(_projection_context(config=config))

    assert full[:5] == ((team_a, team_b),) * 5
    assert full[5:] == ((team_b, team_a),) * 5


@pytest.mark.parametrize(
    ("mode", "identity"),
    (
        pytest.param("shared_obs", NO_SHARED_OBS_ACTOR_PROJECTION_V2, id="mode"),
        pytest.param(
            "no_shared_obs",
            VersionedIdentityV1(identifier="wrong-projection", version=2),
            id="identifier",
        ),
        pytest.param(
            "no_shared_obs",
            VersionedIdentityV1(
                identifier=NO_SHARED_OBS_ACTOR_PROJECTION_ID,
                version=1,
            ),
            id="version",
        ),
    ),
)
def test_projection_rejects_unsupported_mode_or_identity(
    mode: ExecutionInformationMode,
    identity: VersionedIdentityV1,
) -> None:
    context = evaluation_context(execution_information_mode=mode).model_copy(
        update={"actor_projection": identity}
    )

    with pytest.raises(ValueError, match="projection V2"):
        reconstruct_class_ids_by_agent_by_team_v2(context)


@pytest.mark.parametrize("global_slot", (-1, MAX_AGENT_SLOTS, True, 1.0))
def test_scalar_projection_rejects_invalid_global_slots(global_slot: object) -> None:
    with pytest.raises(ValueError, match="global_slot"):
        reconstruct_actor_class_ids_by_team_v2(
            _projection_context(),
            global_slot,  # type: ignore[arg-type]
        )


def test_context_validation_rejects_a_different_live_class_map() -> None:
    context = _projection_context()
    full = reconstruct_class_ids_by_agent_by_team_v2(context)
    changed_first_team = (ROGUE_CLASS_ID, *full[0][0][1:])
    changed = ((changed_first_team, full[0][1]), *full[1:])

    with pytest.raises(ValueError, match="do not match episode roster context"):
        validate_class_ids_by_agent_by_team_against_context_v1(context, changed)


def test_relative_projection_version_follows_the_context_version() -> None:
    from tests.evaluation_fixtures import (
        current_evaluation_context,
        pre_red_zone_evaluation_context,
    )

    assert SHARED_OBS_ACTOR_PROJECTION_VERSION == 3
    assert (
        VersionedIdentityV1(identifier=NO_SHARED_OBS_ACTOR_PROJECTION_ID, version=4)
        == NO_SHARED_OBS_ACTOR_PROJECTION_V4
    )
    assert (
        VersionedIdentityV1(identifier=SHARED_OBS_ACTOR_PROJECTION_ID, version=3)
        == SHARED_OBS_ACTOR_PROJECTION_V3
    )
    config = evaluation_env_config()
    current = current_evaluation_context(config)
    older = pre_red_zone_evaluation_context(config)
    assert current.actor_projection == NO_SHARED_OBS_ACTOR_PROJECTION_V4
    assert older.actor_projection == NO_SHARED_OBS_ACTOR_PROJECTION_V3
    assert (
        current_evaluation_context(
            config, execution_information_mode="shared_obs"
        ).actor_projection
        == SHARED_OBS_ACTOR_PROJECTION_V3
    )
    rows = reconstruct_class_ids_by_agent_by_team_v3(current)
    assert rows == reconstruct_class_ids_by_agent_by_team_v3(older)
    for slot in range(MAX_AGENT_SLOTS):
        assert reconstruct_actor_class_ids_by_team_v3(current, slot) == rows[slot]

    wrong_projection = current.model_copy(
        update={"actor_projection": NO_SHARED_OBS_ACTOR_PROJECTION_V3}
    )
    with pytest.raises(ValueError, match="does not match its information mode"):
        reconstruct_class_ids_by_agent_by_team_v3(wrong_projection)
    with pytest.raises(TypeError, match="requires context V3 or V4"):
        reconstruct_class_ids_by_agent_by_team_v3(_projection_context())
