"""Keep this old import path bound to the package-owned Viewer module.

Objects and patched globals come from the installed Viewer authority. This
compatibility alias copies no classes or service state.
"""

import sys
from typing import TYPE_CHECKING

from marl_battlegrounds.viewer import replay_service as _implementation

if TYPE_CHECKING:
    from marl_battlegrounds.viewer.replay_service import *  # noqa: F403

    EvaluationTransitionViewV1 = _implementation.EvaluationTransitionViewV1
    METRIC_SCHEMA_VERSION = _implementation.METRIC_SCHEMA_VERSION
    ReplayApiErrorV1 = _implementation.ReplayApiErrorV1
    ReplayCommandResponseV1 = _implementation.ReplayCommandResponseV1
    ReplayMetricReportResultV1 = _implementation.ReplayMetricReportResultV1
    _safe_metric_report_filename = (
        _implementation._safe_metric_report_filename  # pyright: ignore[reportPrivateUsage]
    )
    build_shared_obs_authority_source_material_projection_v1 = (
        _implementation.build_shared_obs_authority_source_material_projection_v1
    )
    build_status_source_evidence_index_v2 = (
        _implementation.build_status_source_evidence_index_v2
    )

sys.modules[__name__] = _implementation
