# Standard Evaluation Replay Format

This document defines the version-1 semantic replay normal form introduced in
Milestone 6. It is a contract for evaluation evidence, not a renderer frame,
simulator checkpoint, policy-state dump, or offline-RL dataset.

## Authority and scope

A replay serializes already accepted evaluation records:

```text
EvaluationEpisodeContextV1
  + EvaluationFrameV1[T + 1]
  + EvaluationTransitionV1[T]
  + EvaluationEpisodeCompletionV1
  + EvaluationProcessingStatusV1
```

The episode context remains the single authority for resolved configuration,
catalog, roster, policies, seeds, information regimes, reward/shaping identity,
and source revision. Its `schema_versions` tuple remains exactly the eight CP2
roots. Replay-specific schema bindings live in `ReplayArtifactHeaderV1` and do
not mutate that historical V1 contract.

V1 is homogeneous with respect to execution information. Its context has one
episode-global `execution_information_mode` and one episode-global
`actor_projection`, and its frame validation requires SharedObs availability
for a `shared_obs` episode and forbids it for a `no_shared_obs` episode. A V1
policy-assignment row does not carry a per-slot information regime. Therefore,
V1 cannot represent SharedObs and NoSharedObs actors in the same match; a
policy ID, a false availability row, or an "at least one actor shares"
reinterpretation must not be used as an implicit mixed-regime encoding.

Milestone 7 makes one approved in-place expansion of the unreleased pre-alpha
V1 evaluation/replay models. Resolved configuration now carries numeric task
mode and Team Deathmatch score threshold, analysis frames carry authoritative
team scores, and transition facts carry the task outcome. The discriminated V1
event union has exactly 24 variants. It retains
`AgentLeftCombatEventV1` at rank 50 and adds
`TeamDeathmatchScoreChangedEventV1` at rank 130 for each positive score edge
plus `TeamDeathmatchCompletedEventV1` at rank 140 for the result and one of
`score_threshold`, `horizon`, or `score_threshold_at_horizon`. Existing
development artifacts and fixtures are disposable and must be regenerated.
There is no legacy loader, optional
fallback, compatibility shim, or parallel V2 model family for that approved
in-place expansion. After the alpha schema freeze, any incompatible wire
change requires a schema-version bump.

[Amendment A18](../design/specification_amendments.md#a18-final-pre-alpha-obstacle-capacity-expansion)
authorizes the final pre-alpha V1 shape expansion: the ordered obstacle axis is
now exactly 32 rows in resolved configuration and exactly `[10, 32, 8]` in
captured base observations. Active walls and pillars occupy a contiguous
ordered prefix and every inactive row is canonical all-zero padding.
Development V1 artifacts created with 16 rows are disposable and are
regenerated in place; there is no V2 alias, legacy loader, or compatibility
shim. V1 is refrozen at 32 rows after this migration.

The current V1 writers, readers, validators, canonical bytes, and digests stay
unchanged. Milestone 10 must introduce an explicit future V2 context and replay
binding before mixed per-slot execution-information assignments become an
official artifact. V2 must use version dispatch rather than field guessing or
silent V1 reinterpretation; post-A18 V1 artifacts retain their original bytes,
identities, and digests.

That future mixed-capable normal form remains one physical episode record: one
context, exactly `T + 1` frames, and exactly `T` transitions. It stores the
same-epoch base observations once per frame, together with stable source
provenance and required recipient-by-source availability, and never stores a
materialized SharedObs tensor. The context records each configured-active
slot's regime and versioned compositor/projection provenance; inactive slots
remain not applicable. The availability matrix may be absent only when every
configured active slot is NoSharedObs. When present, every NoSharedObs or
inactive-recipient row is all false, and diagonal, cross-team, and
inactive-source entries are false. Capture records the exact same-epoch matrix
used by actor-input composition; replay never infers it. Reconstruction is
selected-slot-specific: a NoSharedObs slot derives from its own base-observation
row under its recorded projection, while a SharedObs slot uses the recorded
same-epoch source material, that recipient's availability row, the stable
source mapping, and its recorded compositor/projection versions. Exact
SharedObs actor-input reconstruction still fails closed until the Milestone 12
compositor contract exists.

The V2 homogeneous/mixed episode profile is derived from configured-active
slot assignments and is never a competing stored authority. An explicit
homogeneous-only V1-to-V2 migrator copies V1's episode-global mode and
projection to every configured-active V2 slot, marks inactive slots not
applicable, and preserves compatible recorded availability. It fails if the
required versioned provenance cannot be established, cannot synthesize a mixed
profile, emits a new V2 artifact identity/digest, and leaves the source V1
bytes and digest unchanged. Loaders never perform this migration silently.

The standard replay never contains raw `EnvState`, PRNG keys, policy logits or
hidden state, optimizer state, learner batches, renderer summaries, local file
paths, browser capability tokens, or private debugger state. Loading and
rendering never rerun a policy or simulator and never recreate geometry,
visibility, masks, combat, lifecycle, reward, termination, or task outcome.

## Artifact graph

The version-1 artifact graph is deliberately one-way:

```text
trajectory content
  <- EvaluationMetricReportArtifactV1
  <- MetricReportReferenceV1 embedded in ReplayArtifactV1

ReplayArtifactV1
  <- ReplayArtifactReferenceV1
  <- later scenario and actor-POV artifacts
```

`ReplayTrajectoryContentReferenceV1` is the pre-link reference used by the
metric-report artifact. It carries replay/episode/schema identity plus the
context and trajectory-content digests; it does not carry the replay's outer
digest or byte length. This prevents a content-hash cycle.

The build order is:

1. validate a finalized metric-complete observer and its exact metric report;
2. construct the replay header and compute the trajectory-content digest;
3. construct and address the metric-report artifact against that content;
4. place the report artifact's identity, digest, and canonical byte length in
   `MetricReportReferenceV1`;
5. compute the replay's outer digest; and
6. construct path-free replay references for scenario and POV artifacts.

The report remains a companion artifact. A replay is semantically loadable and
renderable without the report bytes; bundle validation must expose a missing or
mismatched report rather than pretending that metric evidence is complete.

## Replay normal form

`ReplayArtifactV1` contains:

- canonical artifact identity `{episode_id}:replay`;
- its schema ID/version and outer SHA-256 digest;
- the trajectory-content SHA-256 digest;
- one `ReplayArtifactHeaderV1`;
- rollout completion and independent metric-processing status;
- one path-free metric-report reference;
- exactly `T + 1` ordered frames; and
- exactly `T` ordered transitions.

The header contains the context exactly once, its digest, the exact CP2 source
bindings, the exact replay-envelope bindings, expected and recorded lengths,
first/last frame identities, runtime provenance, and an ordered wrapper/adapter
stack. Runtime and wrapper data are typed records, never arbitrary dictionaries.

The trajectory-content digest is the canonical digest of:

```text
header + completion + processing_status + frames + transitions
```

It excludes the metric-report reference and both replay-level digest fields.
The outer replay digest covers every replay field except its own digest. Thus a
report-link change preserves the trajectory-content identity but changes the
outer replay identity.

## Semantic validation

Pydantic model validation establishes strict field structure. It is not the
whole replay validation claim. `validate_replay_artifact_v1` performs one
explicit O(T) semantic pass that:

- rejects undeclared root or nested model types and hidden mutable state;
- verifies all schema bindings and content digests;
- requires frame/transition tuple positions to equal artifact indices;
- validates frame zero independently of the simulator starting epoch;
- constructs every adjacent CP3 `EvaluationTransitionViewV1`, thereby reusing
  the accepted CP2 four-record validator and canonical event re-decoding;
- rejects continuation after termination, truncation, or the declared horizon;
- joins completion length, bases, done flags, last-frame identity, and any
  authoritative tail end reason;
- joins processing status to validated progress through the public CP3
  processing-progress validator; and
- preserves valid gap-free complete, partial, interrupted, failed, and
  zero-transition prefixes without converting missing outcomes to zero.

`iter_replay_transition_views_v1` exposes the same coherent views consumed by
live CP3 reducers. It does not construct alternative facts or event links.
Score-change and completion events remain direct projections of adjacent
authoritative frames and task facts; replay never derives them from deaths,
reward, or done flags. The renderer and web transport preserve these events
losslessly and add no second scoring or outcome path.

Replay construction accepts only a finalized
`EvaluationEpisodeObserverV1` and its exact `EvaluationMetricReportV1`.
`training_light` and `debug` observers intentionally retain no trajectory and
cannot create a replay. `evaluation_metric_complete` and
`scenario_metric_complete` are the retaining profiles.

## Completion, processing, and endpoint truth

The replay preserves three independent dimensions:

1. rollout completion: `complete | partial | interrupted | failed`;
2. evaluation processing: `succeeded | failed`; and
3. statistic-specific endpoint observation in metric rows.

A physically complete rollout remains complete if reducer processing failed.
An infrastructure interruption is not scientific right censoring. Early core
truncation is preserved as transition-tail truth; its partial, interrupted, or
failed classification remains runner/report-authored and replay validation does
not infer it. Exact-horizon and task-terminal evidence remain separate
completion bases and may coexist.

Team Deathmatch is threshold-victory. If neither team reaches the configured
threshold by the horizon, the authoritative result is a draw regardless of
terminal score differential. If both teams cross on one simultaneous
transition, complete successor scores decide the result and equality draws.
Threshold completion sets termination, horizon completion sets truncation, and
both truths are preserved on coincidence. A completed transition carries the
matching authoritative completion event, and replay rejects every subsequent
transition.

## Scenario and POV companions

Scenario specifications declare identities, authored initial conditions,
horizon, measurements, violations, predicate identity, and partial-result
policy before rollout. Scenario records reference the completed replay and
metric report and carry supplied results. Their validators join evidence and
identities; they never execute a metric formula or success predicate.

`ResolvedScenarioSpecificationV2` additionally freezes the promoted layout and
authored-initial-condition identities, exact fixed-slot roster and role
template, resolved configuration, one ordered matched seed schedule, and one
stable compiled initial-state digest. That compiled digest covers only a tagged
projection of simulator step count plus `GlobalAnalysisSnapshotV1`; the record's
full frame-zero digest remains separate attempt-specific replay evidence.

Every `right_censored` scenario result requires a censor-aware definition and a
complete declared-horizon replay, independently of whether its estimate is
`defined` or `insufficient_data`. A `competing_event` endpoint likewise
requires a complete rollout. This is the same endpoint algebra used by CP3
metric rows; a scenario record cannot weaken it.

The replay header's exact envelope binding map is deliberately closed over the
replay and its directly linked metric-report graph. Downstream scenario and POV
artifacts carry their own schema IDs/versions and a typed replay reference.
Adding a downstream companion therefore never mutates the Replay V1 header
contract.

Actor POV artifacts use dedicated recipient-sliced records. They do not reuse
the privileged replay context, world snapshot, transition, or canonical CP2
event tuple. Exact actor-input export is supported for `no_shared_obs`. Under
`shared_obs`, Milestone 6 may expose a labelled base-sensor/source-availability
view, but exact materialized input export fails closed until the Milestone 12
compositor exists. Those alternatives are episode-global in V1. A future
mixed-capable V2 selects the branch from the chosen actor's slot assignment;
another actor's regime never authorizes or blocks that selected slot's export.

`ActorPovReplayContentV1` is the independently privacy-testable payload. It
contains one active actor's exact recorded base-observation and action-mask row,
own submitted int32 and accepted category-bounded actions, own rejection flags,
own canonical reward, public done truth, a minimal rollout-completion row, and
recipient-local presentation cues derived only from adjacent authorized rows.
Its compact axis map supplies public recipient IDs, relation rows, action names,
movement directions, actor-relative spawn axes, and source projection/schema
identities without embedding the full roster, policy table, seed protocol, or
mechanics catalog. Hidden CP2 event IDs, ordinals, counts, sources, aura
emitters, contributors, other-agent actions/rewards, snapshots, and policies
remain absent.

`ActorPovReplayArtifactV1` wraps that content with a mandatory path-free
`ReplayArtifactReferenceV1` and an outer digest. Privacy noninterference is
therefore evaluated over canonical recipient-content bytes/digest: two source
replays with equal authorized actor inputs but different hidden truth produce
equal content, while their outer artifacts may differ because truthful source
provenance differs. The source reference is never omitted or forged merely to
make the outer bytes equal.

## Canonical local persistence

`marl_battlegrounds.evaluation.replay_io` owns paths and bytes while
`evaluation.replay` remains the semantic model boundary. The stable library
surface is:

- `canonical_replay_json_bytes_v1` and
  `canonical_metric_report_artifact_json_bytes_v1` for exact target bytes;
- `prepare_replay_bundle_v1` for one path-independent validation and canonical
  serialization pass whose immutable bytes can be retried without rebuilding
  scientific evidence;
- `preflight_replay_bundle_destination_v1` and
  `publish_prepared_replay_bundle_v1` for an explicitly authorized local
  destination and no-clobber publication of those prepared bytes;
- `save_replay_bundle_v1` for report-first, replay-last publication;
- `load_replay_artifact_v1` for a standalone trajectory; and
- `load_replay_bundle_v1` for optional or required metric-sidecar resolution;
- `save_actor_pov_replay_artifact_v1` and
  `load_actor_pov_replay_artifact_v1` for source-validated, independently
  shareable POV companions; and
- `save_scenario_evaluation_record_v1` and
  `load_scenario_evaluation_record_v1` for historical V1 scenario records whose
  replay and metric-report evidence joins are mandatory; and
- `save_scenario_evaluation_record_v2` and
  `load_scenario_evaluation_record_v2` for the versioned fixed-slot companion,
  plus `validate_official_scenario_evaluation_record_v2` for the mandatory
  core-aware product and curated-state acceptance gate.

The filename pair is derived locally, never serialized:

```text
episode.marlbg-replay.json
episode.marlbg-metrics.json
episode.marlbg-scenario.json
episode.agent-<public-id>.marlbg-pov.json
```

Version 1 accepts finite plain UTF-8 JSON only. Loading rejects a byte-order
mark, invalid UTF-8, duplicate object keys, non-finite or overflowing numeric
literals, trailing content, excessive nesting, unknown/future roots, extra
model fields, digest mismatches, and noncanonical whitespace or key order.
Files are bounded before parsing (1 GiB by default, with an explicit positive
library override) and the JSON nesting limit defaults to 128. A direct
`model_validate_json` call is never advertised as proof of canonical bytes,
digest validity, or whole-artifact semantics: the loader performs byte
preflight, strict model validation, the O(T) replay validator, and exact
canonical reserialization in that order.

Only existing local directories and regular nonsymlink files participate.
URLs, symlinks in any path component, archives, compression, pickle, dynamic
imports, implicit directory creation, and network resolution are outside the
contract. Replay and companion loading import no JAX, backend, simulator,
policy, capture, or device-transfer path. V2 scenario loading therefore
establishes canonical bytes and replay/report/scenario evidence joins but does
not by itself confer official status. Official consumers must subsequently call
`validate_official_scenario_evaluation_record_v2`, whose deliberately separate
host boundary rehydrates and checks the carried configuration and curated state
against current core authorities. Evaluation-owned V1 wire dimensions are
frozen at the post-A18 values for artifact decoding and checked against the
current core dimensions in ordinary tests; changing them again requires an
explicit schema migration. The V1
filesystem backend requires POSIX directory-descriptor and no-follow support so
every component and final operation remain bound to one opened directory inode;
it fails closed with `unsupported_platform` when those guarantees are unavailable.

Saving writes each member to a same-directory temporary file, flushes and
`fsync`s it, then publishes by atomic no-clobber hard link and `fsync`s the
directory. The content-addressed metric report is published first and may be
reused only when existing bytes are identical. The replay is published last
and is never overwritten. A publication failure removes only a link proven to
belong to that temporary inode, preserves every pre-existing target, and leaves
immutable bytes available for a safe retry. A report orphan created before a
failed replay publication is recoverable by identical-byte reuse; no published
replay can point to an unpublished report.

`PreparedReplayBundleV1` derives its byte payloads, lengths, and SHA-256
digests from one exact validated `ReplayBundleV1`; callers cannot supply or
forge those cache fields. Normal publication still rejects an existing replay.
The explicit recovery mode of `publish_prepared_replay_bundle_v1` is narrower:
it performs no serialization or overwrite and succeeds only when both existing
files are byte-for-byte identical to the prepared replay and report. This is
the retry seam used by live debugger recording after an uncertain publication
or verification outcome.

`load_replay_bundle_v1(..., require_metric_report=False)` returns a typed
`metric_report_missing` status when the trajectory is valid but its sidecar is
absent. Requiring the sidecar makes absence an error. A present but malformed,
noncanonical, foreign, or digest-mismatched sidecar always fails; it is never
downgraded to “missing.” Loaded frame and incoming-transition selection is a
direct tuple lookup after the one-time O(T) semantic validation.
