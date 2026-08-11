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

## Scenario and POV companions

Scenario specifications declare identities, authored initial conditions,
horizon, measurements, violations, predicate identity, and partial-result
policy before rollout. Scenario records reference the completed replay and
metric report and carry supplied results. Their validators join evidence and
identities; they never execute a metric formula or success predicate.
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
compositor exists.

## Persistence ownership

Canonical local JSON persistence, bounded parsing, atomic publication, and
sidecar resolution are owned by the replay I/O layer defined in the next
contract checkpoint. The in-memory model contract does not contain paths and
does not treat a direct `model_validate_json` call as proof of canonical bytes,
digest validity, or whole-artifact semantics.

Version 1 permits only finite plain UTF-8 JSON. URLs, symlinks, archives,
compression, pickle, dynamic imports, and network resolution are outside the
standard replay contract.
