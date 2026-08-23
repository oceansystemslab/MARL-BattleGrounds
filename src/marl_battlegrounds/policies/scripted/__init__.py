"""Canonical scripted policies implemented for current battleground tasks."""

from marl_battlegrounds.policies.scripted.no_shared_obs import (
    NO_SHARED_OBS_ADAPTER_ID,
    NO_SHARED_OBS_ADAPTER_VERSION,
    decide_team_deathmatch_no_shared_obs,
    team_deathmatch_no_shared_obs_policy,
)
from marl_battlegrounds.policies.scripted.team_deathmatch import (
    NUMERIC_PROFILE_ID,
    POLICY_ID,
    POLICY_SEMANTIC_VERSION,
    SEMANTIC_PROFILE_ID,
    TASK_HEAD_VERSION,
    TEAM_DEATHMATCH_PROFILE,
    TRACE_ONTOLOGY_VERSION,
)

# Keep unsupported regimes and future task modules outside the public surface.
__all__ = (
    "NO_SHARED_OBS_ADAPTER_ID",
    "NO_SHARED_OBS_ADAPTER_VERSION",
    "NUMERIC_PROFILE_ID",
    "POLICY_ID",
    "POLICY_SEMANTIC_VERSION",
    "SEMANTIC_PROFILE_ID",
    "TASK_HEAD_VERSION",
    "TEAM_DEATHMATCH_PROFILE",
    "TRACE_ONTOLOGY_VERSION",
    "decide_team_deathmatch_no_shared_obs",
    "team_deathmatch_no_shared_obs_policy",
)
