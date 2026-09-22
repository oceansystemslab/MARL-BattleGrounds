"""Verify the built-in training split before collection or recording recovery.

Preparation reads installed TDM resources and their scientific authorities once
at host setup. The frozen binding records the finite built-in content closure;
it admits no external datasets, replays, feedback or seed schedules.
pinned_opponent_evidence records what is known about a pinned training
opponent's own training content and controller exposure; it never refuses a
method, and it cannot certify a researcher's undeclared outside influences. The returned
threshold-major numerical bank is separate from this host evidence and shared by
compiled samplers. This module uses base dependencies only and runs no episode.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import (
    SerializerFunctionWrapHandler,
    StringConstraints,
    model_serializer,
    model_validator,
)

from marl_battlegrounds._tdm_assets import (
    MapGeometry,
    TDMMapInfo,
    TDMScenarioInfo,
    asset_manifest,
    scenario_content,
)
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.catalog import (
    build_resolved_env_config_v1,
    build_static_mechanics_catalog_v1,
)
from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V3,
    ContentAddressedIdentityV1,
    EvaluationModel,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.recording_identity import (
    ordered_source_bank_identity,
)
from marl_battlegrounds.evaluation.scenario import ResolvedScenarioSpecificationV3
from marl_battlegrounds.evaluation.tdm_scenarios import (
    _prepared_scenario_specification,  # pyright: ignore[reportPrivateUsage]
    build_tdm_qualification_seed_schedule,
)
from marl_battlegrounds.tasks import (
    _load_tdm_scenario,  # pyright: ignore[reportPrivateUsage]
    balanced_spawn_configs,
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)

type _Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_SHARED_NAMES = ("static-mechanics", "evaluation-schemas", "simulator-rules")


class _MapContent(EvaluationModel):
    """Bind one installed map's verified layout and ordinary configuration.

    info retains the catalog's resource hash and unverified authored-file
    provenance. layout hashes resolved float32 geometry only. configuration
    hashes the full canonical 5v5 configuration. Neither identity grants a use;
    the parent binding checks the ordered split independently of display labels.
    """

    kind: Literal["map"] = "map"
    info: TDMMapInfo
    layout: ContentAddressedIdentityV1
    configuration: ContentAddressedIdentityV1


class _ProtectedScenario(EvaluationModel):
    """Retain one protected start and its existing recording-check instrument.

    info retains resource and authoring evidence. layout uses the same format
    as ordinary maps. root identifies the configuration, exact state, roster
    and horizon independently of a seed schedule. pressure is the verified
    controller identity. qualification_specification is the complete existing
    two-coordinate recording check, not a full evaluation sampling plan.
    """

    kind: Literal["protected-scenario"] = "protected-scenario"
    info: TDMScenarioInfo
    layout: ContentAddressedIdentityV1
    root: ContentAddressedIdentityV1
    pressure: ContentAddressedIdentityV1
    qualification_specification: ResolvedScenarioSpecificationV3


class TrainingContentBinding(EvaluationModel):
    """Keep immutable, versioned evidence for the built-in content selection.

    Attributes
    ----------
    schema_id, schema_version, eligibility_version
        Version 1 keeps the original 42 K20 sources. Version 2 adds the ordered
        score_thresholds declaration; the content use rules remain version 1.
    maps
        Ordered 52 verified map records. IDs 0-41 are training, 42-46 validation,
        and 47-51 test. Scientific layouts are distinct across all map roots.
    protected_scenarios
        Ordered eight protected starts, controllers and recording-check
        specifications. Their layouts cannot occur in training or validation.
    shared_definitions
        Ordered identities for mechanics, supported schemas and executable Core
        rules. These shared dependencies are allowed in every content role.
    source_bank
        Full 256-bit identity of the ordered configuration bank. Each threshold
        contributes 42 rows in map order; threshold blocks keep declared order.
    source_configurations
        Ordered source configuration identities under the recording bank format.
    canonical_digest
        Canonical SHA-256 of this entire record, excluding this field itself.
        Includes audit metadata. Resume instead compares scientific_projection().
    score_thresholds
        Distinct positive integer winning scores, default (20,). Version 1
        omits this field when saved, preserving its historical bytes and hash.

    Notes
    -----
    Tuple-backed strict models reject unknown kinds, fields and versions. There
    is no external dependency-reference surface: required dependencies are fully
    embedded, so unresolved or cyclic external references are rejected. Original
    authored source paths/hashes are provenance; their files are never opened.
    Current installed resource hashes are verified during preparation. Historical
    audit bytes need not equal harmless repackaging of current resources.
    """

    schema_id: Literal["marl_battlegrounds.training_content"] = (
        "marl_battlegrounds.training_content"
    )
    schema_version: Literal[1, 2] = 1
    eligibility_version: Literal[1] = 1
    maps: tuple[_MapContent, ...]
    protected_scenarios: tuple[_ProtectedScenario, ...]
    shared_definitions: tuple[ContentAddressedIdentityV1, ...]
    source_bank: ContentAddressedIdentityV1
    source_configurations: tuple[_Digest, ...]
    canonical_digest: _Digest
    score_thresholds: tuple[int, ...] = (20,)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Keep version-1 serialization unchanged, including its original digest."""
        result = handler(self)
        if self.schema_version == 1:
            result.pop("score_thresholds", None)
        return result

    @model_validator(mode="after")
    def _validate_binding(self) -> Self:
        """Reject incomplete closure, forbidden overlap or a false record digest.

        Validation reads no files. Fresh preparation separately verifies current
        resources. Return this frozen record; raise ValueError on disagreement.
        """
        _validate_score_thresholds(self.score_thresholds)
        if self.schema_version == 1 and self.score_thresholds != (20,):
            raise ValueError("Version-1 training content requires K20")
        if self.schema_version == 2 and self.score_thresholds == (20,):
            raise ValueError("K20-only training content uses version 1")
        if tuple(row.info.map_id for row in self.maps) != tuple(range(52)):
            raise ValueError("training content requires ordered maps 0 through 51")
        if tuple(row.info.scenario_id for row in self.protected_scenarios) != tuple(
            range(1, 9)
        ):
            raise ValueError(
                "training content requires protected scenarios 1 through 8"
            )
        for row in self.maps:
            expected = (
                "training"
                if row.info.map_id < 42
                else "validation"
                if row.info.map_id < 47
                else "test"
            )
            if row.info.split != expected:
                raise ValueError("map split disagrees with approved training selection")
            if (
                row.layout.identifier != "training-resolved-layout"
                or row.layout.version != 1
            ):
                raise ValueError("unsupported resolved layout identity")
            if (
                row.configuration.identifier != "resolved-env-config"
                or row.configuration.version != 1
            ):
                raise ValueError("unsupported resolved configuration identity")
        layouts = tuple(row.layout.canonical_digest for row in self.maps)
        if len(set(layouts)) != 52:
            raise ValueError(
                "training, validation and test map layouts must be distinct"
            )
        for row in self.protected_scenarios:
            if (
                row.layout.identifier != "training-resolved-layout"
                or row.layout.version != 1
            ):
                raise ValueError("unsupported protected layout identity")
            if row.layout.canonical_digest in layouts[:47]:
                raise ValueError(
                    "protected scenario layout appears in training or validation"
                )
            if row.root != _scenario_root(row.info):
                raise ValueError(
                    "protected scenario root disagrees with its resolved start"
                )
            spec = row.qualification_specification
            if (
                spec.pressure_protocol != row.pressure
                or spec.resolved_config_digest_sha256
                != row.info.resolved_configuration_digest
                or spec.horizon != row.info.horizon
                or tuple(item.class_id for item in spec.roster_template)
                != row.info.class_ids
                or spec.seed_schedule != build_tdm_qualification_seed_schedule()
            ):
                raise ValueError(
                    "protected scenario dependencies disagree with the root"
                )
        if tuple(row.identifier for row in self.shared_definitions) != _SHARED_NAMES:
            raise ValueError(
                "required shared definitions are missing or have a forbidden role"
            )
        if any(row.version != 1 for row in self.shared_definitions):
            raise ValueError("unsupported shared definition version")
        if (
            self.source_bank.identifier != "ordered-source-bank"
            or self.source_bank.version != 1
            or len(self.source_configurations) != 42 * len(self.score_thresholds)
        ):
            raise ValueError(
                "Source bank requires 42 configuration identities per threshold"
            )
        # This existing recording format hashes compact sorted JSON plus newline.
        from hashlib import sha256

        from marl_battlegrounds.evaluation.run_writer import (
            _json_bytes,  # pyright: ignore[reportPrivateUsage]
        )

        if (
            sha256(_json_bytes(list(self.source_configurations))).hexdigest()
            != self.source_bank.canonical_digest
        ):
            raise ValueError("source bank order disagrees with its identity")
        if (
            canonical_digest_sha256(self, exclude={"canonical_digest"})
            != self.canonical_digest
        ):
            raise ValueError("training content binding digest mismatch")
        return self

    def scientific_projection(self) -> dict[str, object]:
        """Return the path-independent comparison used for training resume.

        Preserve resolved content, use roles, source order, controller rules,
        seed membership and specification endpoints. Exclude display wording,
        authored paths/revisions and raw-resource packaging. No files are read.
        """
        protected: list[dict[str, object]] = []
        for row in self.protected_scenarios:
            specification = row.qualification_specification.model_dump(mode="json")
            for name in (
                "canonical_digest_sha256",
                "authored_initial_condition",
            ):
                specification.pop(name)
            specification["layout"] = row.layout.model_dump(mode="json")
            protected.append(
                {
                    "kind": row.kind,
                    "scenario_id": row.info.scenario_id,
                    "root": row.root,
                    "specification": specification,
                }
            )
        result: dict[str, object] = {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "eligibility_version": self.eligibility_version,
            "maps": tuple(
                {
                    "kind": row.kind,
                    "map_id": row.info.map_id,
                    "role": row.info.split,
                    "layout": row.layout,
                    "configuration": row.configuration,
                }
                for row in self.maps
            ),
            "protected_scenarios": protected,
            "shared_definitions": self.shared_definitions,
            "source_bank": self.source_bank,
            "source_configurations": self.source_configurations,
        }
        if self.schema_version == 2:
            result["score_thresholds"] = self.score_thresholds
        return result


@dataclass(frozen=True)
class PreparedTrainingContent:
    """Separate verified host evidence from the shared numerical source bank.

    Attributes
    ----------
    binding : TrainingContentBinding
        Immutable version-1 or version-2 setup and resume evidence. Keep outside
        JAX loops. Only the extended threshold declaration needs version 2.
    source_configs : EnvConfig
        Canonical 5v5 source bank. Every array leaf starts with axis 42 times
        the threshold count. Each threshold block holds maps 0 through 41; row i
        belongs to training map i % 42. Original spawn banks and other task
        defaults remain intact. Both spawn arrangements passed host validation. Do not
        mutate or donate the shared bank while samplers or recorders use it.
    """

    binding: TrainingContentBinding
    source_configs: EnvConfig


def _identity(name: str, payload: Mapping[str, object]) -> ContentAddressedIdentityV1:
    """Hash a finite payload into a named version-1 scientific identity."""
    return ContentAddressedIdentityV1(
        identifier=name, version=1, canonical_digest=canonical_digest_sha256(payload)
    )


def _layout_identity(geometry: MapGeometry) -> ContentAddressedIdentityV1:
    """Hash ordered map geometry after rounding every numerical value to float32.

    geometry uses world units and radians. Include unused obstacle and pad rows;
    exclude roster, rules, labels and paths. Reject nonfinite float32 results.
    Both ordinary maps and scenario configurations use this same representation.
    """
    payload = {
        name: np.asarray(value, dtype=np.float32).tolist()
        for name, value in geometry.model_dump().items()
    }
    return _identity(
        "training-resolved-layout", {"format": "training-resolved-layout@1", **payload}
    )


def _config_geometry(config: EnvConfig) -> MapGeometry:
    """Extract the ordered scalar layout from a concrete configuration on the host."""
    return MapGeometry.model_validate_json(
        canonical_json_bytes(
            {
                name: np.asarray(getattr(config, name), dtype=np.float32).tolist()
                for name in (
                    "map_width",
                    "map_height",
                    "obstacles",
                    "team_spawn_pad_positions",
                )
            }
        )
    )


def _scenario_root(info: TDMScenarioInfo) -> ContentAddressedIdentityV1:
    """Identify a validated scenario's owned start independently of its schedule."""
    return _identity(
        "protected-scenario-root",
        {
            "configuration": info.resolved_configuration_digest,
            "initial_state": info.resolved_initial_state_digest,
            "class_ids": info.class_ids,
            "team_sizes": info.team_sizes,
            "horizon": info.horizon,
        },
    )


def _shared_definitions() -> tuple[ContentAddressedIdentityV1, ...]:
    """Identify common rules without treating shared use as protected leakage.

    Read six installed Core Python sources and remove documentation strings before
    hashing their syntax trees. Formatting, comments and docstrings do not change
    this executable identity. The mechanics catalog and schema versions retain
    their existing owners. This finite read belongs only at setup.
    """
    bodies: dict[str, str] = {}
    for name in ("types", "axis_mappings", "combat", "config", "geometry", "env"):
        source = files("marl_battlegrounds").joinpath("core", f"{name}.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and node.body
                and isinstance(node.body[0], ast.Expr)
            ):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
        bodies[name] = ast.dump(tree, include_attributes=False)
    return (
        _identity("static-mechanics", {"catalog": build_static_mechanics_catalog_v1()}),
        _identity("evaluation-schemas", {"bindings": REQUIRED_SCHEMA_BINDINGS_V3}),
        _identity("simulator-rules", bodies),
    )


def _validate_score_thresholds(values: tuple[int, ...]) -> None:
    """Require a nonempty tuple of distinct supported positive integer scores.

    Reject bool and values outside Core's existing exact-float score range.
    This host check changes no files and allocates no numerical arrays.
    """
    if not isinstance(cast(object, values), tuple) or not values:
        raise TypeError("score_thresholds must be a nonempty tuple")
    if any(type(value) is not int for value in values):
        raise TypeError("score_thresholds must contain Python integers, not bool")
    if len(set(values)) != len(values) or any(
        not 1 <= value <= 2**24 - 4 for value in values
    ):
        raise ValueError("score_thresholds must be distinct scores in [1, 2**24 - 4]")


def prepare_training_content(
    *,
    expected: TrainingContentBinding | Mapping[str, object] | None = None,
    score_thresholds: tuple[int, ...] | None = None,
) -> PreparedTrainingContent:
    """Verify built-in content and prepare canonical sources for declared scores.

    Parameters
    ----------
    expected : TrainingContentBinding | Mapping[str, object] | None
        Saved version-1 or version-2 binding, or its JSON mapping. Defaults to
        a fresh setup.
        Revalidate its structure and digest, then compare scientific content with
        the installed package. Historical locations and harmless labels may differ.
    score_thresholds : tuple[int, ...] or None, default=None
        Distinct supported positive integer winning scores, in source-bank order.
        None uses expected.score_thresholds when resuming, otherwise (20,).
        Each score adds maps 0..41 as a complete block. Other rules stay unchanged.
        K20-only output keeps the original version-1 binding and bank identity.

    Returns
    -------
    PreparedTrainingContent
        Current immutable binding and shared EnvConfig bank with float32,
        int32 and Boolean leaves. Source order is threshold, then map 0..41.

    Raises
    ------
    ValueError
        Content, split declarations, dependency evidence or a saved binding is
        invalid, overlaps protected content or is scientifically incompatible.
    OSError
        A required installed resource or shared rule source cannot be read.

    Notes
    -----
    Host-only: reads package files, verifies hashes, resolves configurations and
    synchronizes arrays for validation. Uses base dependencies only. Runs no
    episode, policy or learner and writes no files. Call before constructing a
    resumed RunWriter, whose constructor can change existing recording files.
    The closed surface admits no external controllers, data, replays or feedback.
    Those inputs require future extensions to this same gate before consumption.
    """
    saved = None
    if expected is not None:
        saved = TrainingContentBinding.model_validate_json(
            canonical_json_bytes(expected)
        )
    if score_thresholds is None:
        score_thresholds = saved.score_thresholds if saved is not None else (20,)
    _validate_score_thresholds(score_thresholds)
    manifest = asset_manifest()
    team_a, team_b = canonical_tournament_rosters()
    maps: list[_MapContent] = []
    configs: list[EnvConfig] = []
    for info in manifest.maps:
        config = make_standard_team_deathmatch_config(
            map_id=info.map_id,
            team_a_roster=team_a,
            team_b_roster=team_b,
        )
        # The task factory verifies resource bytes and resolves this exact layout.
        layout = _layout_identity(_config_geometry(config))
        if info.map_id < 42:
            balanced_spawn_configs(config, num_envs=2)
            configs.append(config)
        maps.append(
            _MapContent(
                info=info,
                layout=layout,
                configuration=ContentAddressedIdentityV1(
                    identifier="resolved-env-config",
                    version=1,
                    canonical_digest=build_resolved_env_config_v1(
                        config
                    ).canonical_digest_sha256,
                ),
            )
        )
    schedule = build_tdm_qualification_seed_schedule()
    scenarios: list[_ProtectedScenario] = []
    for info in manifest.scenarios:
        content = scenario_content(info)
        scenario = _load_tdm_scenario(info, content)
        specification = _prepared_scenario_specification(scenario, content, schedule)
        pressure = specification.pressure_protocol
        if pressure is None:
            raise ValueError("protected scenario requires its pressure controller")
        scenarios.append(
            _ProtectedScenario(
                info=info,
                layout=_layout_identity(_config_geometry(scenario.config)),
                root=_scenario_root(info),
                pressure=pressure,
                qualification_specification=specification,
            )
        )
    if score_thresholds != (20,):
        configs = [
            config._replace(team_deathmatch_score_threshold=threshold)
            for threshold in score_thresholds
            for config in configs
        ]
    bank = jax.tree.map(lambda *rows: jnp.stack(rows), *configs)
    bank_digest, references, _ = ordered_source_bank_identity(bank)
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.training_content",
        "schema_version": 1 if score_thresholds == (20,) else 2,
        "eligibility_version": 1,
        "maps": tuple(maps),
        "protected_scenarios": tuple(scenarios),
        "shared_definitions": _shared_definitions(),
        "source_bank": ContentAddressedIdentityV1(
            identifier="ordered-source-bank",
            version=1,
            canonical_digest=bank_digest,
        ),
        "source_configurations": tuple(references),
    }
    if score_thresholds != (20,):
        payload["score_thresholds"] = score_thresholds
    binding = TrainingContentBinding.model_validate(
        {
            **payload,
            "canonical_digest": canonical_digest_sha256(payload),
        }
    )
    if saved is not None and canonical_digest_sha256(
        saved.scientific_projection()
    ) != canonical_digest_sha256(binding.scientific_projection()):
        raise ValueError(
            "saved training content is incompatible with installed scientific content"
        )
    return PreparedTrainingContent(binding=binding, source_configs=bank)


def _leakage_projection(binding: TrainingContentBinding) -> dict[str, object]:
    """Keep the part of a binding that decides protected-controller exposure.

    The protected scenario closure: each scenario's ID, root and layout.
    Ordinary map revisions, threshold banks and controller rule descriptors are
    left out, so a source run on older maps or another threshold bank still
    links, while a run made under a different protected closure does not. The
    source run's own binding was verified by this gate when it trained, which
    already kept protected content out of its training. No files are read.
    """
    return {
        "protected": [
            [
                row.info.scenario_id,
                row.root.model_dump(mode="json"),
                row.layout.model_dump(mode="json"),
            ]
            for row in binding.protected_scenarios
        ],
    }


def _export_origin(
    binding: TrainingContentBinding, export: Path
) -> tuple[str, dict[str, object] | None]:
    """Decide whether an actor export is proven to come from its claimed checkpoint.

    Parameters
    ----------
    binding : TrainingContentBinding
        Today's verified training content.
    export : Path
        Actor export directory, already loaded and hash-checked by load_system.

    Returns
    -------
    tuple[str, dict or None]
        ("verified", the source learner's saved pinned-opponent record or None)
        when every linking field agrees and the learner's protected scenario
        closure matches today's; otherwise ("declared", None). A pinned record
        of None means the source run played self-play only.

    Notes
    -----
    Reads the export's description and its sibling learner description
    ``<run>/checkpoints/<checkpoint_id>/`` without payloads; each description's
    own hash is verified. Writes nothing.
    """
    from marl_battlegrounds.training import checkpoints

    details = checkpoints.read_checkpoint_description(export)
    metadata = cast(dict[str, object], details.get("metadata", {}))
    identifier = metadata.get("checkpoint_id")
    if not isinstance(identifier, str):
        return "declared", None
    source = export.resolve().parent.parent / "checkpoints" / identifier
    try:
        learner = checkpoints.read_checkpoint_description(source)
    except OSError, ValueError:
        return "declared", None
    saved = cast(dict[str, object], learner.get("metadata", {}))
    config = cast(dict[str, object], saved.get("config", {}))
    counters = cast(dict[str, object], learner.get("counters", {}))
    collection = cast(dict[str, object], learner.get("collection", {}))
    linked = (
        learner.get("checkpoint_id") == identifier
        and learner.get("actor_digest") == details.get("actor_digest")
        and saved.get("run_id") == metadata.get("run_id")
        and config.get("seed") == metadata.get("seed")
        and counters.get("env_steps") == metadata.get("env_steps")
        and checkpoints._config_input_scale(config)  # pyright: ignore[reportPrivateUsage]
        == checkpoints._actor_input_scale(details)  # pyright: ignore[reportPrivateUsage]
        and checkpoints._config_spawn_frame(config)  # pyright: ignore[reportPrivateUsage]
        == checkpoints._actor_spawn_frame(details)  # pyright: ignore[reportPrivateUsage]
        and collection.get("pinned_opponent") == metadata.get("pinned_opponent")
    )
    if not linked:
        return "declared", None
    try:
        # The same JSON route prepare_training_content uses for saved bindings.
        source_binding = TrainingContentBinding.model_validate_json(
            canonical_json_bytes(
                cast(Mapping[str, object], collection.get("content_binding"))
            )
        )
    except ValueError:
        return "declared", None
    if _leakage_projection(source_binding) != _leakage_projection(binding):
        return "declared", None
    return "verified", cast(dict[str, object] | None, collection.get("pinned_opponent"))


def pinned_opponent_evidence(
    binding: TrainingContentBinding, method: object, *, export: Path | None = None
) -> dict[str, object]:
    """Record what is known about a pinned training opponent's training history.

    Parameters
    ----------
    binding : TrainingContentBinding
        Today's verified training content.
    method : Policy or System
        The frozen pinned method.
    export : Path or None, default None
        Its actor export directory when it was loaded from one.

    Returns
    -------
    dict
        JSON-ready. ``source`` is "installed" (a built-in Policy), "verified
        export", "declared export" or "researcher method". ``exposure`` is
        "none", "known" or "unknown": whether the method, or anything it was
        trained against through a verified chain, is one of the protected
        scenario pressure controllers. ``controllers`` names the known ones by
        identifier and version. ``familiar_scenarios`` lists the scenarios whose
        results become familiar-opponent results or cannot count as protected
        evidence: all eight whenever any pressure controller is known, because
        Beta's rules include Alpha's, and all eight when exposure is unknown.
        Unknown history is never reported as "none". A researcher System that
        declares components also gets ``declared_components``, a copy of them.

    Notes
    -----
    Host-only. Never refuses a method: execution is allowed for every valid
    System; this record limits what its results may claim. Controller identity
    is descriptor identity, so a controller update must change its descriptor.
    """
    from marl_battlegrounds.evaluation.policy_execution import (
        Policy,
        System,
        controller_identity,
        policy,
    )

    pressures = {
        (row.pressure.identifier, row.pressure.version, row.pressure.canonical_digest)
        for row in binding.protected_scenarios
    }
    all_scenarios = sorted(row.info.scenario_id for row in binding.protected_scenarios)

    def known(controllers: list[str]) -> dict[str, object]:
        """Exposure to named pressure controllers makes every scenario familiar."""
        return {
            "exposure": "known",
            "controllers": sorted(set(controllers)),
            "familiar_scenarios": all_scenarios,
        }

    policies: tuple[Policy, ...] = ()
    if isinstance(method, Policy):
        policies = (method,)
    elif isinstance(method, System):
        policies = method._policies  # pyright: ignore[reportPrivateUsage]
    controllers: list[str] = []
    for entry in policies:
        identity = controller_identity(entry)
        if (
            identity is not None
            and (
                identity["identifier"],
                identity["version"],
                identity["canonical_digest"],
            )
            in pressures
        ):
            controllers.append(f"{identity['identifier']}@{identity['version']}")
    random_apply = policy("random").apply
    if isinstance(method, Policy) and (controllers or method.apply is random_apply):
        base: dict[str, object] = {"source": "installed"}
        base.update(
            known(controllers)
            if controllers
            else {"exposure": "none", "controllers": [], "familiar_scenarios": []}
        )
        return base
    if export is not None:
        status, inherited = _export_origin(binding, export)
        if status == "verified":
            record: dict[str, object] = {"source": "verified export"}
            if inherited is None:
                record.update(
                    {"exposure": "none", "controllers": [], "familiar_scenarios": []}
                )
            else:
                parent = cast(dict[str, object], inherited.get("evidence", {}))
                if parent.get("exposure") == "known":
                    record.update(known(cast(list[str], parent.get("controllers", []))))
                elif parent.get("exposure") == "none":
                    record.update(
                        {
                            "exposure": "none",
                            "controllers": [],
                            "familiar_scenarios": [],
                        }
                    )
                else:
                    record.update(
                        {
                            "exposure": "unknown",
                            "controllers": [],
                            "familiar_scenarios": all_scenarios,
                        }
                    )
            return record
        return {
            "source": "declared export",
            "exposure": "unknown",
            "controllers": [],
            "familiar_scenarios": all_scenarios,
        }
    record = {"source": "researcher method"}
    record.update(
        known(controllers)
        if controllers
        else {
            "exposure": "unknown",
            "controllers": [],
            "familiar_scenarios": all_scenarios,
        }
    )
    if isinstance(method, System) and method.components:
        record["declared_components"] = [dict(entry) for entry in method.components]
    return record
