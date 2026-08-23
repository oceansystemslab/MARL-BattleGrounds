# MARL-BattleGrounds Specification Amendments

> **NORMATIVE CONTRACT — ACTIVATED 2026-08-10.** This document records the
> accepted amendments to the historical design PDF.

This document changes `MARL_BGs_Design_Document.pdf`. The PDF remains the
unaltered historical architectural blueprint; this file is the controlling
public source where the two disagree. An amendment changes only the clauses it
names. Unmentioned PDF requirements remain in force.

The amendments below were drafted before Milestone 6 Step 5 implementation.
They deliberately favor the four project North Stars: researcher-centricity,
low sample complexity, meaningful tactical and strategic team behavior, and
professional MARL/software engineering.

## A1. Actor execution-information regimes

**Classification:** risky, accepted design drift.
**Supersedes:** R20; Sections 2.4, 2.4.1, 2.4.8, 2.4.11, 2.6.18, 2.8.5,
2.13.8, 2.15.7, 2.16.2, and 2.16.19; Appendix A.11; and baseline or paper
clauses that use the historical disabled-sharing label, make that regime the
default, or make SharedObs merely optional.

SharedObs is the approved future default execution-time actor-information
regime. It is not current runtime behavior and does not become active until its
versioned learner-input projection is implemented, performance-tested, and
accepted before canonical baseline training.

NoSharedObs remains an official first-class regime. It must remain selectable
and must be evaluated and reported separately from SharedObs. Results from the
two regimes must never be pooled into one benchmark cell or summary.

The editable specification uses **NoSharedObs** and `no_shared_obs`. The
historical PDF's earlier disabled-sharing wording is superseded rather than
rewritten in the PDF itself. NoSharedObs means only that the SharedObs
compositor is disabled; it does not claim that an algorithm has no
communication mechanism of its own.

The canonical implementation boundary is:

- `execution_information_mode` is the extensible actor-side setting whose
  current values are `shared_obs` and `no_shared_obs`; each learned checkpoint
  fixes exactly one value;
- a command-line Boolean may be a convenience alias, but is not the canonical
  persisted representation;
- simulator state, dynamics, rewards, action semantics, base observations, and
  masks do not change with this selection;
- in `shared_obs`, each already-authored same-decision-epoch base sensor
  projection is factored once into a fixed-shape, roster-joined sensor-source
  bank that preserves stable source identity and each source's normal
  visibility, line of sight, padding, and lifecycle redaction;
- each actor's shared input applies a versioned Boolean
  recipient-by-source information-availability mask to that bank, so teammates
  may receive different source subsets rather than one identical team tensor;
- the compositor never recomputes geometry, visibility, line of sight, or
  redaction, and the actor's own base observation remains separate;
- in `no_shared_obs`, the learner returns the actor's base observation and
  bypasses material shared-bank and recipient-by-source-mask construction;
- teammate masks, previous-action/history fields, recurrent state, policy
  memory, rewards, transition facts, raw state, analysis snapshots, and critic
  world state are excluded;
- observer-invariant content is factored rather than repeatedly copied; and
- actor SharedObs, team-observation critic input, and privileged world-state
  critic input remain separate versioned contracts.

Replay and evaluation records store base observations once, together with the
source-axis/provenance mapping, any information-availability input not
losslessly derivable from those observations, `execution_information_mode`,
and the actor-input projection version. They do not persist a second copy of
materialized SharedObs tensors.

The Milestone 6 V1 frame schema supports either information regime without
implementing the future compositor. `EvaluationEpisodeContextV1` records one
episode-wide `execution_information_mode` and one actor-input projection, so
every configured active policy assignment in a V1 episode is homogeneous in
those contracts. The current V1 schema is immutable: it cannot truthfully
represent SharedObs and NoSharedObs assignments in the same episode. Such
mixed execution is ineligible for official evaluation, scenario, replay, or
metric output until the Milestone 10 V2 contract in A12 exists.

The V1 availability field is an optional, explicit Boolean matrix with axes
`(recipient_global_slot, sensor_source_global_slot)` and shape `(10, 10)`.
Conditional validation requires that matrix for `shared_obs` and forbids it
for `no_shared_obs`. Its diagonal, cross-team cells, inactive-recipient rows,
and inactive-source columns are false. Neither regime stores a materialized
SharedObs tensor. SharedObs-versus-NoSharedObs and the reverse assignment are
future directional strata, not values to pool with each other or with either
homogeneous regime.

## A2. Metric architecture and simulator cost

**Classification:** required architectural clarification.
**Supersedes:** R24 and Sections 2.6.16, 2.12.10, 2.13.18, 2.16.15, and
2.17.8 wherever they imply cumulative combat metrics or a full metric suite in
the simulator or ordinary training loop.

MARL-BattleGrounds uses three deliberately separate data planes:

1. The JAX simulator emits only fixed-shape authoritative facts for causes or
   resolved phase outcomes that would otherwise disappear.
2. Opt-in evaluation and scenario capture produces a complete semantic
   trajectory with `T + 1` frames and `T` transitions.
3. Trainer-owned telemetry contains algorithm-specific values such as policy
   statistics, optimizer state, runtime timing, and optional reward-shaping
   components.

These are payload, authority, and cost boundaries inside one lifecycle; they
are not separate regime-specific runners, trainers, evaluators, RNG paths,
action-realization paths, or metric systems. A12 defines the common Milestone
10–12 spine and the narrow seams at which actor information may differ.

Milestone 6 CP2 normalizes those planes through the following accepted host
boundary:

- `EvaluationEpisodeContextV1` owns the roster's fixed-slot identity/topology
  fields and the resolved configuration's per-slot body radius, movement,
  interaction ranges, maximum health, and recovery mechanics. It also owns
  policy, catalog, seed, one episode-wide homogeneous
  `execution_information_mode`, one compatible `actor_projection`,
  `critic_information_regime`, `canonical_reward_mode`,
  `shaping_configuration`, and code provenance. It carries exactly ten
  discriminated policy-assignment rows; inactive slots use an explicit
  `not_applicable` variant.
- `GlobalAnalysisSnapshotV1` contains dynamic state only. It does not repeat
  configured-active, team, class, identity, body radius, maximum health, or
  any other static context/catalog value.
- Submitted and accepted actions occur exactly once, inside
  `TransitionFactsV1.action_acceptance_facts`. `EvaluationTransitionV1` may
  expose read-only conveniences but must not serialize a second action
  authority. `TransitionFactsV1` mirrors every core subtree and leaf name
  exactly rather than introducing shortened host aliases.
- Transition rewards are named `canonical_reward_by_agent` with fixed length
  ten and optional `canonical_reward_by_team` with fixed length two. Shaping
  values are excluded; only the shaping configuration identity belongs in
  context.
- A core recipient sentinel of `-1` normalizes to JSON `null`, agrees with the
  corresponding `has_recipient` flag in both directions, and reverses
  losslessly.
- Sparse events form an exactly 23-variant discriminated V1 union derived from
  facts and adjacent frames. The one-time Milestone 7 pre-alpha expansion adds
  authoritative Team Deathmatch score-change and completion variants at phase
  ranks 130 and 140. `AgentDiedEventV1` records the newly dead recipient;
  one `LethalDamageContributionEventV1` separately records each authoritative
  positive source contribution on that lethal transition. Rank 90 orders the
  death event before its contribution events. Neither record claims a killer,
  last hit, or complete historical elimination credit. Aura attachments name
  direct transition-start covering emitters, not causal credit.
- Rank 120 uses family-specific canonical coordinates: a team-wave event sorts
  by `(120, team_index, -1, wave_subtype, neutral_source)`, while each realized
  agent respawn sorts by `(120, configured_team_index, agent_global_slot,
  respawn_subtype, neutral_source)`. This groups teams and places each wave
  before that team's realized agents.
- Catalog digests use finite-only canonical JSON with ASCII identifiers,
  recursive `-0.0` normalization, sorted keys, compact separators, and UTF-8
  encoding. Every code revision requires `source_tree_digest`; additionally,
  `dirty_patch_digest` is required exactly when `is_dirty` is true. Local paths
  are never durable policy or code identities.
- One public semantic validator consumes the context, transition-start frame,
  transition, and successor frame together, then re-decodes and exactly
  compares the deterministic event sequence. Structural model validation is
  not misrepresented as cross-record semantic validation.

Milestone 6 CP3 adds the following host-only streaming boundary:

- `validate_initial_evaluation_frame_v1` separately validates context join,
  artifact index zero, canonical frame identity, information-regime rules, and
  inactive dynamic padding. Artifact frame zero may correspond to any
  nonnegative simulator epoch; capture indices are never inferred from an
  arbitrary simulator step.
- An opt-in `EvaluationEpisodeObserverV1` accepts exactly one initial frame and
  then gap-free coherent context/start-frame/transition/successor-frame views.
  It tracks validated and successfully processed transition counts separately.
- Versioned reducers are immutable and copy-on-write. All reducer replacements
  and final rows validate before one atomic observer commit. A failed append,
  reducer, or final report poisons the observer, preserves the last validated
  prefix, and never publishes a partially updated statistic set.
- `EvaluationEpisodeCompletionV1` records only rollout completion
  (`complete`, `partial`, `interrupted`, or `failed`) and its authoritative
  basis or failure origin. `EvaluationProcessingStatusV1` independently records
  host processing success/failure. A processing failure after task terminal or
  declared horizon does not relabel the completed rollout as failed.
- Scientific endpoint observation is per statistic, using `not_applicable`,
  `observed`, `right_censored`, `competing_event`, or `unavailable`. It is not
  an episode completion state. Missing or ineligible data never become zero.
- Raw count, sum, numerator/denominator, duration, opportunity, and distribution
  components join one immutable episode context in
  `EvaluationMetricReportV1`. They retain exact subjects and dimensions but do
  not duplicate policy, seed, task/scenario, information-regime, reward, or
  shaping provenance on every row.
- An absent observer is the only disabled path. Explicit `training_light` and
  `debug` observers stream the standard semantic view without history;
  evaluation/scenario metric-complete observers retain exact `T + 1` frame /
  `T` transition prefixes. Debug adds no private-state payload.

CP3 adds no core callback, runner, logging sink, file I/O, replay artifact,
official metric formula, universal registry, materialized SharedObs tensor, or
private debug-state format. Replay persistence and loaded-file parity remain
the next milestone step.

The core mapping change permitted by CP2 is limited to a pure public
`core.axis_mappings` authority and bounded mechanical imports in the current
core environment/configuration consumers. Any changed mapping value, traced
operation, simulator result, or wider core edit stops the checkpoint for
review; host code never recreates an actor-relative indexing formula.

Derived totals, ratios, windows, engagement groupings, distributions,
leaderboards, ratings, and population summaries do not belong in `EnvState` or
the traced transition kernel. They are computed deterministically by host or
offline consumers. Ordinary training performs no host transfer, Pydantic
validation, event decoding, trajectory retention, or full metric calculation
unless an experiment explicitly enables a narrow diagnostic.

Researchers may construct optional JAX-native reward-shaping or auxiliary
reward components from ordinary transition outputs. Such components are
training interventions, not official evaluation metrics or canonical task
reward. A component using privileged facts must be labeled a privileged
training signal and must never become actor input.

## A3. Metric constitution and reporting surfaces

**Classification:** required scientific clarification.
**Supersedes:** Section 2.16 wherever a candidate is treated as official merely
because it is inexpensive or derivable.

An official metric must answer a real research question, be explainable in one
sentence, use genuine opportunities, preserve its raw sufficient components,
state its direction or descriptive status, avoid unsupported causal claims,
remain meaningful across algorithms and seeds, add material information, and
justify its implementation and cognitive cost.

The suite has separate surfaces:

- a compact primary team scorecard;
- a compact descriptive per-agent/per-class scorecard;
- an advanced long-form analysis export;
- quantitative controlled-scenario scorecards;
- diagnostics and quality-control outputs;
- cross-play, learning, runtime, and population summaries.

There is no global tactical score, opaque coordination score, or universal
agent-quality ranking. Presentation requirements such as sorting, tooltips,
quantiles, or CSV export do not create new simulator facts or primitive metric
columns.

Stable metric semantics are separate from evaluation-suite populations and
weights, experiment-manifest checkpoint/inference choices, result rows, and
presentation metadata. A paper may change an opponent pool or confidence
method without changing what a metric means or forcing a metric-version bump.

## A4. Combat terminology and attribution

**Classification:** corrective semantic clarification.
**Supersedes:** R24 and Sections 2.16.4, 2.16.5, 2.16.6, 2.16.7, and 2.16.8
where they use kills, last hits, K/D, solo kills, pentakills, source-level
effective-healing/overhealing credit, or ambiguous "effective" amounts.

The simulator records newly dead recipients and positive-damage contributors
on the lethal transition; it does not retain an unbounded damage history or
select a killer or last hitter. Official analysis therefore uses:

- **lethal-transition damage contribution:** an agent contributed positive
  recipient-modified gross damage on the transition where an enemy became
  dead;
- **lethal-transition contribution rate:** those contributions divided by the
  team's enemy deaths, with no-team-death episodes represented by a zero
  opportunity rather than a fabricated zero rate;
- **single-contributor lethal transition:** exactly one positive-damage source
  was recorded on that transition; this does not claim that the entire fight
  was a solo kill or that earlier damage did not matter; and
- **team wipe:** the task-defined complete elimination of a team, without
  assigning a pentakill.

K/D, last-hit kills, general elimination participation, solo kills,
pentakills, and arbitrary killer selection are not official metrics without a
separate versioned historical-contribution definition.

Every health-effect metric names its amount or health-resolution stage:

1. raw source output;
2. source-modified gross output;
3. recipient-modified gross effect entering simultaneous health resolution;
4. combat-resolution health after simultaneous resolution and clamping, before
   regeneration;
5. realized recipient net health change from transition-start health to that
   combat-resolution health; and
6. separately authored actual regeneration.

Recipient-modified gross damage may exceed remaining health. Simultaneous
damage, healing, and clamping do not support unique per-source attribution of
realized health loss or restoration. Overlapping equal-class auras and
anti-heal sources similarly support combined team/class effects, not invented
per-emitter causal credit.

## A5. Teamfights, engagements, and contextual behavior

**Classification:** corrective scientific clarification.
**Supersedes:** Sections 2.10.7, 2.12.11, 2.16.4–2.16.5, and 2.16.11, plus any
scenario or metric clause that assumes an observation-radius clique, team
centroid, out-of-combat timer, or casualty count is an authoritative teamfight
definition or that equates “single-shot” with one stochastic episode.

MARL-BattleGrounds does not define or roadmap a generic teamfight or engagement
detector, validator, or conditioned metric. All such candidates are rejected,
not deferred or validation-pending. Observation radius is a policy-information
mechanic rather than a fight boundary; a team centroid can merge separate
skirmishes; and the out-of-combat countdown is regeneration bookkeeping. No
core fact, host module, scenario validator, or milestone is reserved for
generic teamfight segmentation.

Peeling, kiting, flanking, body blocking, backline access, healing triage,
regrouping, rotations, escort/interception quality, trap discipline, and Burst
synchronization are evaluated through controlled quantitative scenarios rather
than vague episode-wide quality scores. A scenario has one primary quantitative
endpoint, at most two secondary margins, explicit violations, a horizon,
censoring semantics, frozen policy/configuration provenance, and replay
evidence.

"Single-shot" means that the evaluated policy does not adapt across attempts.
It does not mean that one stochastic episode constitutes an algorithm-level
sample.

## A6. Objective and class-specific metric corrections

**Classification:** required task-ownership clarification.
**Supersedes:** Sections 2.16.8–2.16.10 where formulas presuppose mechanics or
causal credit not yet owned by a task.

Task-specific metric formulas remain inactive until the owning task exposes
authoritative score, objective, reward, terminal, and event/snapshot truth.

In particular:

- KoTH control and contest denominators use eligible hill-timesteps, not bare
  episode horizon;
- per-agent hill occupancy or participation may be reported, but team score is
  not arbitrarily apportioned to agents;
- CTF class-specific carrying time is descriptive; there is no
  designer-preferred-carrier quality score;
- capture conversion uses eligible enemy-flag pickups and is always shown with
  that pickup exposure;
- task trades are reported through score/objective swing and casualty facts,
  not a universal teamfight-win label;
- Hunter Trap is an immediate targeted status/damage mechanic, so applications,
  control duration, lifecycle break rate, and follow-up associations replace
  the PDF's placed-trap uptime and trigger-rate language; and
- Priest Freedom application, uptime, and binding coverage are derivable, but
  exact counterfactual movement recovered is not an official metric.

## A7. Evaluation statistics and leakage control

**Classification:** required research-protocol correction.
**Supersedes:** Sections 2.15.17 and 2.16.16 wherever they prescribe a generic
three/five-seed target or leave “standard error or confidence interval”
unspecified.

Evaluation episodes estimate one trained checkpoint. Each learned checkpoint
is fixed to one `execution_information_mode`, actor-input projection version,
and compatible compiled actor front end. Compatibility validation must reject
a mismatch before compilation, device allocation, or execution. An explicitly
declared cross-regime transfer or out-of-distribution study is a separate
estimand and may not reinterpret the source checkpoint as native to the
destination regime.

Independent training seeds—not episodes, teams, agents, deaths, or repeated
matches—are the default experimental units for algorithm-level claims. Results
aggregate in this order:

1. episodes within one homogeneous training-seed/evaluation cell;
2. evaluation cells under frozen predeclared weights;
3. independent training seeds with equal seed weight.

Rates pool raw numerator and denominator within a cell, but do not pool
opportunities across training seeds. Both teams from one match are paired;
agents are nested observations. Zero opportunities produce `N/A`, not zero.

Across frozen cells, opportunity metrics use the ratio of weighted raw
components, not the average of cell rates. Cell weights are not renormalized
around cells that happen to produce no opportunities. Independent training
runs receive equal weight, with seed-level results and uncertainty reported.

Official evaluation separates training, development/validation, and locked
test layouts, scenarios, opponents, partner pools, and seeds. Locked-test
results must not select metrics, tune predeclared analysis thresholds, choose checkpoints,
or shape rewards. Each experiment manifest predeclares its seed budget,
precision rule, checkpoint rule, endpoint hierarchy, uncertainty method, and
failure/missingness policy.

Learning/sample-efficiency results report both environment transitions and
active-agent decision transitions, plus wall-clock/compute provenance. A 1v1
and a 5v5 environment transition do not represent equal agent experience.

Under the immutable V1 evaluation contract, every cell is also homogeneous in
the episode-wide execution-information mode and compatible actor projection.
Mixed-regime execution is ineligible until Milestone 10 introduces the V2
per-active-slot provenance contract in A12. After that contract exists,
SharedObs-versus-NoSharedObs and NoSharedObs-versus-SharedObs are distinct
assignment directions, each requiring task-appropriate side swaps. They are
never pooled with each other or with homogeneous-regime cells.

## A8. Milestone 6 Step 5 fact budget

**Classification:** efficiency correction to an unimplemented design.
**Supersedes:** Milestone 6 planning prose that approves 48 leaves / 1,677 raw
bytes or a `CooldownTransitionFacts` subtree.

The accepted pre-Step-5 baseline is 37 leaves / 897 raw bytes. Step 5 adds nine
leaves / 760 raw bytes:

- one `float32 (10,)` post-combat, pre-regeneration health stage;
- two `float32 (10, 2)` realized Charge- and ordinary-movement-phase
  displacement arrays;
- two `bool (10, 10)` aura emitter/beneficiary coverage relations; and
- four `bool (10, 9)` independent status-lifecycle cause matrices.

The resulting Milestone 6 target was **46 leaves / 1,657 raw bytes**. The
approved Milestone 7 pre-alpha expansion in A11 adds one scalar `int32` task
outcome leaf, so the current target is **47 leaves / 1,661 raw bytes**.

Cooldown start is already the accepted Ultimate action. Cooldown readiness is
the positive-to-zero change between adjacent semantic frames. Deriving either
does not recreate simulator rules, so permanent cooldown fact leaves would be
redundant. Host evaluation events may still present both lifecycle events from
those authoritative inputs.

Ordinary-movement displacement is algebraically redundant only in ideal real
arithmetic. In the actual `float32` trajectory, subtracting a rounded Charge
displacement from adjacent positions is not guaranteed to reproduce the
authoritative post-Charge boundary, can fabricate tiny nonzero sparse events,
and requires the host to import the rule that transition-start dead respawned
rows did not move. Retaining the 80-byte leaf is therefore an approved
phase-fidelity tradeoff: it preserves the exact phase-local result and keeps
respawn/liveness semantics in the simulator. It does not authorize a duplicate
geometry pass; the leaf must reuse the already resolved phase positions.

No accepted production core defect prompted this amendment. It removes
avoidable telemetry before implementation.

## A9. Spawn protection and out-of-combat regeneration

**Classification:** documentation of already accepted simulator drift.
**Supersedes:** R18, R19, Sections 2.5.8, 2.6.13, 2.7.9, 2.14.6, 2.14.7,
2.17.7, and any dependent wording that requires geometric spawn sanctuaries or
stationary-only regeneration.

Geometric spawn sanctuaries are replaced by deterministic team respawn waves
and timed spawn shielding. Shielded agents remain mobile and use the accepted
shield-specific collision, targeting, and visibility contract; the simulator
does not maintain a sanctuary region or sanctuary-egress subsystem.

Stationary-only regeneration is replaced by class-specific mobile
out-of-combat regeneration. Accepted positive combat interaction resets the
public countdown for its participants; an eligible living agent regenerates
according to its resolved class profile when the countdown permits it.
Movement is not itself a regeneration blocker. The authoritative transition
facts preserve countdown resets and actual realized regeneration rather than a
cumulative recovery metric.

## A10. Standard semantic replay ownership

**Classification:** required artifact-boundary clarification.
**Supersedes:** any historical wording that treats a raw simulator state,
renderer frame, or debugger session as the durable replay authority.

Milestone 6 version-1 replay stores the accepted evaluation context once plus
exactly `T + 1` semantic frames and `T` adjacent transitions, rollout
completion, independent evaluation-processing status, and a path-free
content-addressed metric-report reference. It does not store `EnvState`, policy
state, renderer summaries, local paths, or browser/session authority.

The episode context keeps exactly its eight CP2 schema bindings. The closed
replay/metric-report envelope has a separate exact binding map in the replay
header. Downstream scenario and actor-POV companions self-version and carry a
typed replay reference; their schemas are not retroactively inserted into the
replay header. A pre-link trajectory-content digest covers the header,
completion, processing status, frames, and transitions while excluding the
report reference and replay-level digests. The report artifact references that
pre-link content; the completed replay references the report artifact; later
scenario and actor-POV artifacts reference the completed replay. This one-way
graph prevents content-hash cycles.

Replay validity requires an explicit O(T) semantic pass using the public
initial-frame and four-record validators. Direct Pydantic revalidation proves
structure only. Offline analysis and presentation may consume only serialized
catalog mappings and captured semantic records; they never rerun a simulator
or recreate mechanics.

Canonical V1 persistence is finite local UTF-8 JSON with exact canonical-byte
equality after strict model and whole-replay validation. It rejects symlinks,
nonregular files, duplicate keys, non-finite or oversized/deep inputs, unknown
versions, and mismatched content references. Publication is atomic and
no-clobber: the metric-report object is durable before the referencing replay,
an existing report is reused only when bytes are identical, and replay targets
are never overwritten. The host loader owns frozen V1 wire dimensions and must
not import or initialize JAX, a backend, simulator, policy, or capture path.
The V1 filesystem backend fails closed unless POSIX directory-descriptor and
no-follow operations can prevent ancestor-symlink races; report reuse
synchronizes the exact compared file descriptor before a referencing replay is
published.

Scenario and actor-POV companions use the same bounded canonical JSON and
descriptor-bound, no-clobber publication contract. Scenario records are valid
only against their referenced replay and metric-report evidence. Exact
NoSharedObs POV exports use recipient-sliced schemas and keep submitted int32
intent distinct from category-bounded accepted actions. Their privacy claim is
defined over the recipient-content bytes; the outer artifact retains a truthful
completed-replay reference and may therefore differ when hidden source truth
differs.

Exact materialized SharedObs export remains unavailable until the Milestone 12
compositor exists. Milestone 6 may instead project a prominently labelled
`source_material_only` view containing the selected recipient's recorded base
sensor row and the recorded recipient-by-source availability inputs. That view
is not an actor-input artifact and may never be described as the composed
SharedObs tensor.

Offline presentation consumes these canonical records through a pure,
renderer-neutral scene projection. It may expose recorded durable state,
catalog mechanics, actor-relative mappings, and direct event evidence, but it
must not call simulator, geometry, visibility, masking, policy, or mechanic
helpers. Researcher and actor-authorized presentation roots remain
structurally distinct.

## A11. Team Deathmatch outcome and pre-alpha V1 schema expansion

**Classification:** accepted task-semantic and pre-alpha schema amendment.
**Supersedes:** Sections 2.3.6 and 2.8.2 where they award a Team Deathmatch
score-decision win at the maximum horizon, plus A2's former 21-event closure
and any V1 wording that forbids this explicitly approved pre-alpha expansion.

Team Deathmatch is a threshold-victory task. Each newly dead configured Team A
recipient increments Team B's score once, and vice versa. Contributor,
killer, and last-hit identity never affect scoring. Both score increments from
one simultaneous transition are applied before result selection. If either
complete successor score reaches the configured threshold, the higher score
wins and equal scores draw. If neither team reaches the threshold by the final
allowed action, the authoritative result is a draw regardless of score
differential. Threshold completion sets `terminated`; the horizon sets
`truncated`; both flags remain true when the two bases coincide.

The canonical terminal reward is team-shared and sparse: Team A win is
`[+1, -1]` by team, Team B win is `[-1, +1]`, and draw or ongoing play is
`[0, 0]`. Every configured active teammate receives its team's terminal value
even when dead; inactive padded slots remain zero. Callers stop or reset after
completion. The core API does not add a terminal latch or an absorbing
post-completion transition.

The categorical result encoding is shared task vocabulary rather than
Team-Deathmatch-specific vocabulary: `0` is ongoing, `1` is a Team A win, `2`
is a Team B win, and `3` is a draw. Core task selection is one fixed numeric
`jax.lax.switch` over the four known v1 mode slots: neutral, Team Deathmatch,
King of the Hill, and Capture the Flag. KoTH and CTF branches remain canonical
neutral placeholders and host validation rejects both modes until their own
milestones implement them. This fixed dispatch is not a task registry, plugin
system, generic objective framework, or authorization to add speculative KoTH
or CTF state and dynamics.

Milestone 7 performs one approved in-place expansion of the unreleased V1
evaluation and replay models. Resolved configuration records add numeric task
mode and Team Deathmatch threshold, analysis snapshots add the two
authoritative team scores, and transition facts add the task outcome. The
event union expands from 21 to exactly 23 variants:

- `team_deathmatch_score_changed`, phase rank 130, records the zero-based team
  index, public team ID, positive score increment, previous score, and
  successor score; and
- `team_deathmatch_completed`, phase rank 140, records the authoritative
  Team A win, Team B win, or draw plus `score_threshold`, `horizon`, or
  `score_threshold_at_horizon` completion basis.

Score events follow lifecycle events and precede the completion event. Host
capture derives them from adjacent authoritative snapshots and core facts; it
does not rerun combat or infer an outcome from reward. The renderer-neutral
Scene/Event V2 and browser transports preserve both events losslessly without
adding Milestone 7 presentation behavior.

Existing development artifacts and fixtures are disposable and are
regenerated under the expanded V1 contract. For this approved pre-freeze
in-place expansion, there is no V2 alias, legacy loader, optional fallback,
dual-schema root, or compatibility shim. After the alpha schema freeze, any
incompatible wire change—including the mixed-regime contract required below—
requires a version bump and an explicit migration policy rather than another
in-place mutation.

## A12. Common Milestone 10–12 policy pipeline spine

**Classification:** required policy, training, and evaluation architecture
clarification.
**Supersedes:** Sections 2.12.13–2.12.18, 2.13.4–2.13.5, 2.15.7,
2.15.10–2.15.11, and 4.3.10–4.3.12, plus Appendices A.12, A.15, and A.16,
wherever they permit separate execution-regime pipelines, mutable checkpoint
regimes, mixed-regime V1 provenance, or duplicated lifecycle ownership.

Milestones 10–12 extend one common policy-to-transition pipeline spine. An
episode specification is either selected by a training distribution or fixed
by an evaluation suite or scenario; that selection ownership does not create
separate task/policy semantics or a selector-owned runner:

```text
training-selected or evaluation/scenario-fixed episode specification
  -> versioned policy assignments and seed schedule
  -> evaluation/scenario host adapter or JAX training adapter
  -> reset, base observations, and exact action masks
  -> selected authorized-input composition and learner projection
  -> regime-compatible compiled actor front end
  -> shared exact-mask action realization and one joint-action assembler
  -> core step and common transition/rollout semantics
  -> training-only batch/update/checkpoint lifecycle
  -> the same capture/replay/metric authority wherever applicable
```

Execution-information regimes may differ only at these explicit seams:

1. authorized-input composition;
2. learner-input projection and the compatible compiled actor front end;
3. checkpoint compatibility validation;
4. separately measured compute and resource cost; and
5. manifest, evaluation-cell, and report stratum.

Regime selection does not authorize a second policy-specification resolver,
environment or scenario runner, trainer or update loop, evaluator, RNG
protocol, mask consumer, legal-action realization, joint-action assembler,
capture path, replay format, or metric implementation. SharedObs and
NoSharedObs therefore share lifecycle and semantic ownership even when their
actor inputs and compatible compiled front ends differ.

Milestone 10 owns the common versioned episode-specification and
policy-assignment contracts, seed-schedule schema and named derivation
protocol, evaluation/scenario host runner,
capture, replay, and failure-semantics integration. Milestone 11 owns every
training selector: stateless direct or custom training distributions and
optional stateful, checkpointable curricula. Each emits ordinary episode and
policy specifications into the common training spine; a distribution or
curriculum is not a second trainer or rollout path. Milestone 12 consumes those
selections through the common JAX rollout, batch, update, and checkpoint
lifecycle while owning no roster catalog or selection policy. It also owns the
SharedObs compositor, versioned learner projections, and compatible compiled
actor front ends. These ownership boundaries create extension seams, not
parallel products.

The M10 evaluation/scenario host adapter and M12 pure-JAX training adapter may
differ mechanically because fallible heterogeneous provider orchestration and
compiled learner batching have different constraints. M11 training selections
enter the shared episode contract and the M12 adapter; they do not route through
the M10 host runner. Both adapters implement the same policy epoch, action,
transition, completion, and reproducibility contracts and require shared
conformance evidence. This is one semantic pipeline with purpose-appropriate
adapters, not two interpretations of the environment.

Every learned checkpoint declares exactly one `execution_information_mode`,
one actor-input projection version, and one compatible compiled actor-front-end
contract. Standard compatibility validation rejects incompatible combinations
before compilation, device allocation, or execution. Cross-regime
initialization is permitted only as an explicitly declared transfer or
out-of-distribution intervention with separate provenance and reporting; it
does not make one checkpoint switch regimes in place.

The current `EvaluationEpisodeContextV1` and its replay family are homogeneous
and immutable: their single episode-wide `execution_information_mode` and
`actor_projection` apply to every configured active policy assignment. A
runtime may not label a mixed SharedObs/NoSharedObs match as a valid V1 episode,
even if it can mechanically produce a joint action. Such a match is ineligible
for official evaluation, controlled-scenario evidence, replay publication, and
metric reporting.

Mixed-regime execution is deferred to a Milestone 10 V2 contract. V2 must add
per-active-slot execution-information and actor-projection provenance, preserve
the availability authority required to reconstruct every recipient's input,
and feed the same common runner, action, capture, replay, and metric lifecycle.
It must not mutate or reinterpret V1. Its recipient-by-source availability
matrix may be absent only when every configured active assignment is
NoSharedObs. When the matrix is present, every NoSharedObs or inactive-recipient
row is all false; diagonal, cross-team, and inactive-source entries are also
false. The matrix is the exact same-epoch authority used by input composition
and capture, not a replay-side reconstruction. A homogeneous/mixed episode
profile is derived from the configured-active per-slot assignments; it is not
a second editable authority. An explicit V1-to-V2 migration may copy V1's
episode-wide mode and projection to each configured-active V2 slot, mark
inactive slots not applicable, and preserve compatible recorded availability,
but it can produce only a homogeneous V2 profile. It fails if required
versioned provenance cannot be established, gives the migrated V2 artifact a
new identity, and never mutates V1 bytes or digests. After V2 exists, focal
SharedObs versus opponent NoSharedObs and focal NoSharedObs versus opponent
SharedObs are separate directional cells with task-appropriate side swaps.
Neither direction is pooled with the other or with homogeneous SharedObs or
NoSharedObs results.

## A13. Canonical scripted-policy identity and task/regime separation

**Classification:** required baseline-architecture correction.
**Supersedes:** historical baseline clauses that define easy, medium, hard,
expert, or any other difficulty-indexed scripted-policy family; and any
Milestone 7–12 planning language that permits separate task behavior or
parameter identities merely because SharedObs and NoSharedObs expose different
authorized actor inputs.

Each implemented task owns exactly one canonical scripted-policy behavioral
identity and one immutable parameter profile for that semantic version. The
historical easy/medium/hard/expert family is permanently replaced rather than
retained as aliases, presets, evaluation strata, or hidden parameter variants.
Episode configuration, training distribution, evaluation suite, and scenario
are separate host concepts and do not create additional scripted-policy
profiles. A roster is resolved by the owning training distribution,
evaluation suite, or scenario before the common policy pipeline invokes the
scripted policy; it is not a scripted-policy identity or difficulty profile.

Information regime is provenance and input availability, not a second
behavioral specification. SharedObs and NoSharedObs adapters must project their
authorized same-epoch sources into the same versioned policy-fact contract and
feed the same task scorer, class semantics, weights, thresholds, exact-mask
handling, tie protocol, and trace ontology. The scorer may respond differently
when SharedObs makes additional facts valid, but it must not branch on the
regime identifier or substitute regime-specific behavioral parameters.

The scripted policy uses direct combat-pair and movement-candidate scoring. It
does not introduce persistent attack, retreat, engage, flank, recovery,
guardian, carrier, escort, allocation, or other tactical modes. Common
mechanic and class-role terms form the stable semantic core; a thin task head
adds only bounded current-objective contributions authorized by that task's
public state. A task that requires different base mechanic weights, class
triggers, or causal semantics must first explicitly reopen the owning common
decision and then declare a new task-policy semantic version rather than
hiding the change in an adapter.

The Milestone 7 questionnaire freezes that reusable semantic core for every
scripted task policy: authorized mechanic facts, class roles, combat and
movement score meanings, Ultimate triggers, mask authority, causal epochs,
class-prior semantics, and exact-peer tie handling. Team Deathmatch supplies a
zero objective contribution. Later King of the Hill and Capture the Flag
questionnaires may add only their authorized objective facts and bounded
current-objective contributions. There is no pre-authorized task-mechanic
exception bucket. A task that cannot obey the common mechanic/class contract
must first obtain an explicit user reopening of the owning common decision,
with a rationale, semantic version bump, and cross-task compatibility audit;
until then, the proposed divergence is forbidden.

Within this contract, **bounded** means finite and overridable, not
necessarily weak. A declared class prior or objective contribution may
materially influence its scorer while remaining subject to stronger current
threat, vulnerability, finishing, control, effectful recipient-bound team
value, and the other declared direct-score components.

Milestone 7 implements only the Team Deathmatch task policy with its
NoSharedObs adapter.
SharedObs is added only after its authorized source-bank and projection
contracts exist. King of the Hill and Capture the Flag policy heads are added
only after those tasks provide implemented observation, mask, transition,
replay, and metric authority. No placeholder adapter, empty task head, or
future-module stub is required. The first implementation may keep the logical
common scorer inside the Team Deathmatch policy module; extraction to a common
module occurs only when a second real task consumes it and equivalence proof
shows the refactor is behavior-preserving.

The canonical Team Deathmatch scripted policy treats score, score threshold,
remaining kills, match point, and horizon as behaviorally inert. It selects
the best current combat and movement action from current authorized mechanics;
it does not switch personalities because the match is early, late, close, or
at match point.

## A14. Public configured class-to-slot observation metadata

**Classification:** accepted actor-observation contract clarification.
**Supersedes:** any clause or planning assumption that treats a configured
unit's class, its class-to-roster-slot association, or configured-class
presence/absence as private dynamic sensor information.

`SpawnLifecycleObservation` includes the configured roster field:

```text
class_ids_by_agent_by_team
full environment shape: (10, 2, 5) int32
scalar actor shape:          (2, 5) int32
relation row 0: observer's own team
relation row 1: observer's opponent
```

For each configured-active observer, relation slot `j` aligns with that
observer's unit, configured-active, alive, spawn-shield, respawn, and spawn-pad
relation slot `j`. Team A observers receive `[Team A, Team B]`; Team B
observers receive `[Team B, Team A]`. Configured-active slots retain their
class ID through occlusion, death, spawn shielding, and respawn. A
configured-inactive candidate slot uses neutral class ID `0`, and every
configured-inactive observer row is canonical zero.

This field makes both configured class-to-slot mappings public. It may be
joined with already-public lifecycle rows, so an actor may know which
configured class is alive, dead, shielded, or awaiting respawn. It does not
unmask position, health, status, cooldown, selected action, action history, or
any other visibility-gated dynamic unit value. Class equality is not a
guaranteed focal-row decoder because duplicate-class rosters are legal; focal
truth continues to come from the dedicated self projection and exact focal
mask.

The class field is identical for learned and scripted actors and in SharedObs
and NoSharedObs. It is observer-invariant public roster metadata carried by the
base observation, not teammate-sensor material and not an input that the
SharedObs source bank owns or duplicates. SharedObs versus NoSharedObs measures
additional authorized dynamic teammate sensing, not discovery of the public
opposing roster.

The immutable V1 evaluation/replay schema is not mutated to serialize a new
leaf. Its existing episode roster context contains the configured slot/class
authority needed to reconstruct this base-observation field losslessly. Any
consumer that needs the field must use a newly versioned actor projection/POV
contract that performs and validates that reconstruction, or fail closed.
Mixed-regime execution remains subject to A12's separate V2 gate.

## A15. Episode configuration and experiment-distribution ownership

**Classification:** required experiment-architecture correction.
**Supersedes:** Sections 2.3.1, 2.3.2, 2.3.6, 2.9.1–2.9.4,
2.12.6–2.12.7, 2.12.17, 2.13.3, 2.17.9, 2.17.15, 4.3.7, and 4.3.11;
Appendices A.3, A.12, A.20, and A.21; and any
roadmap, deliverable, or planning clause, only where it requires the 136-cell
no-duplicate composition grid, roster-bearing `1v1`–`5v5` task identifiers,
curriculum discovery through a generic task registry, fixed smaller-team
training rosters, or Stage/Milestone 7 completion of such a grid. It also
supersedes those sources wherever they place training-distribution,
information-regime, actor/critic observation-schema, action-schema,
reward-mode or shaping,
logging, replay, evaluation-suite, scenario, policy-assignment, seed-schedule,
or reset-randomization ownership inside a task/episode configuration rather
than the owning contracts named below. `EnvConfig` continues to own resolved
simulator inputs and task mechanics; this amendment reassigns experiment and
artifact orchestration, not transition semantics. In particular, historical
formal-model assumptions of symmetric team topology do not constrain the two
resolved rosters, and reset-layout randomization belongs to the selecting
training distribution, evaluation suite, or scenario rather than a roster-
bearing task identity.

`EnvConfig` is a resolved configuration for one reproducible episode. It owns
the immutable simulator inputs consumed by reset and step, including the task
mode, padded roster and active slots, resolved map geometry, task constants,
lifecycle mechanics, and horizon. Joined policy assignments, seed schedule,
catalog and code provenance complete the reproducibility record. `EnvConfig`
is not a roster whitelist, named matchup, training preset, sampling
distribution, curriculum stage, evaluation suite, or scenario definition.

These terms are deliberately distinct:

- **structurally valid** means that an episode configuration satisfies the
  simulator's supported task, schema, dtype, shape, catalog, padding,
  geometry, threshold, horizon, and active-team invariants;
- **default** identifies a convenience selected by an owning workflow and
  imposes no restriction on other valid configurations;
- **official** identifies a versioned benchmark-owned evaluation or scenario
  population with frozen provenance; and
- **canonical** identifies the benchmark's primary fixed task semantics or
  evaluation condition, not every condition on which a policy may train.

Ownership is explicit and non-overlapping:

| Concern | Decision owner | Executor or consumer |
| --- | --- | --- |
| Structural validity and resolved inputs for one episode | `EnvConfig` construction plus core host validation | `reset` and `step` |
| Default direct, researcher-defined custom, and optional curriculum training selection | Milestone 11 | Milestone 12 JAX training adapter |
| Official or custom evaluation population and reporting identity | Versioned evaluation suite under the evaluation protocol | Milestone 10 evaluation host adapter |
| Controlled setup, roster, initial state, fixed-slot roles, matched seed schedule and realized-coordinate rule, horizon, and endpoints | Versioned scenario definition | Milestone 10 scenario host adapter |
| Shared episode/policy-assignment and seed-derivation schemas | Milestone 10 contracts | M11 selectors, M12 training, and evaluation/scenario definitions |

Owning a schema does not transfer ownership of the values selected under it:
M10 defines the shared contracts, M11 selects training populations, and each
evaluation suite or scenario freezes its own evidence population.

For Team Deathmatch, every roster with one through five configured active
members on each team is structurally eligible. The two team sizes and class
sequences may differ. Duplicate classes, Priest in 1v1, all-Priest teams, and
any other catalog-valid composition are legal. Validation rejects malformed,
out-of-domain, geometrically impossible, or unimplemented configurations; it
does not reject an experiment because its roster is noncanonical, asymmetric,
duplicated, strategically weak, or likely to draw. The fixed maximum shape and
inactive-slot masks preserve one learner-facing schema across those choices.

Milestone 7 exposes construction, not an experiment catalog. Its planned
Team Deathmatch boundary is:

```python
make_standard_team_deathmatch_config(
    *,
    team_a_roster: Sequence[AgentClassName],
    team_b_roster: Sequence[AgentClassName],
    score_threshold: int,
    max_steps: int,
) -> EnvConfig

make_canonical_team_deathmatch_evaluation_config() -> EnvConfig
```

`AgentClassName` denotes one exact supported class token (`mage`, `warrior`,
`hunter`, `rogue`, or `priest`); it is not a roster-combination enum or
whitelist. Its host type definition is part of the M7 construction surface.

The standard-layout factory accepts every structurally valid Team Deathmatch
roster and applies the approved standard layout and lifecycle mechanics. It is
a focused episode-construction convenience, not the universal training API and
not authority over future training-map selection. The canonical evaluation
factory fixes the mirrored 5v5 roster with exactly one Mage, Warrior, Hunter,
Rogue, and Priest per team, approved canonical evaluation layout, score
threshold, horizon, and lifecycle rules. Canonical Team Deathmatch reward
semantics remain simulator behavior; the evaluation suite/context separately
records the canonical reward-mode identity and other joined provenance because
`EnvConfig` carries no reward-mode identifier.
`team_deathmatch` remains the battleground identity. Stable
training-preset, evaluation-suite, and scenario IDs, if introduced, belong to
their respective layers rather than a roster-resolving task registry.

Milestone 11 owns training selection. Default direct Team Deathmatch training
selects the canonical mirrored five-class 5v5 roster with the same task
mechanics, lifecycle rules, score threshold `K`, horizon `H`, and canonical
reward as canonical evaluation, but uses the separately approved training-map
distribution and training seed schedule. Researchers may instead define any
distribution over structurally valid episode configurations, rosters, maps,
opponents, and supported policy contracts. Optional curriculum training is a
stateful selector over the same episode contract; the benchmark curriculum
will use explicitly reviewed, handpicked 1v1–5v5 rosters rather than an
exhaustive composition product. Its exact rosters, maps, weights, retention,
opponents, and transition rules remain Milestone 11 decisions and require
explicit scientific approval. Milestone 12 executes the selected episodes
through the common JAX training spine and must not reconstruct, enumerate, or
own a roster distribution.

Official canonical Team Deathmatch evaluation is a separately versioned,
frozen mirrored 5v5 population with exactly one Mage, Warrior, Hunter, Rogue,
and Priest per team. A custom evaluation suite may use any structurally valid
roster but must identify itself as custom and freeze the same reproducibility
dimensions. A scenario owns its resolved episode configuration, explicit
fixed-slot roster, initial state, role template, horizon, matched seed schedule,
and endpoints. The schedule is a stable multi-attempt definition; each
episode's realized seed record joins to exactly one declared schedule
coordinate. Runtime assignment of Team A/Team B or per-slot policies binds
policies to those frozen slots; it cannot replace the scenario roster.

This scenario contract is a required M7 C2 implementation gate, not a claim
that the current V1 schema already proves every join. The current
`ResolvedScenarioSpecificationV1` binds scenario identity, resolved-config
digest, horizon, and eligible role names, but it does not bind an explicit
fixed-slot roster, exact role template, or matched seed-schedule identity and
membership rule. The current resolved-config record does not contain roster
rows, and one `EvaluationSeedProtocolV1` is only one episode's realized seed
record rather than a schedule. C2 must version the resolved scenario contract,
bind the explicit roster and role template exactly to episode context, bind a
stable schedule definition, and prove that each realized episode seed record
occupies exactly one declared schedule coordinate. Its official-scenario
validator must also enforce parity with the core structural/product config
invariants on loaded context. Team A/Team B and per-slot policy binding cannot
alter the resolved configuration, roster, configured-active slots, role
template, schedule, or realized coordinate. Until those proofs pass, no
scenario receives an official frozen-scenario claim under A15.

Training distributions are never inferred from an official evaluation suite
or scenario, and training may not consume locked evaluation or scenario maps,
seeds, opponents, or other held-out material. Direct and curriculum training
share task semantics and one learner lifecycle; curriculum adds selection
state, not a simulator mode or execution path.

The historical 136-cell no-duplicate Team Deathmatch grid is retired
provenance. It is not a required preset, registry surface, evaluation grid,
curriculum commitment, test count, acceptance criterion, or implied default.
Any future proposal to ship that grid as a benchmark-owned training preset or
official evaluation suite requires a new tracked amendment and explicit
scientific approval.

## A16. Development scene authoring and asset-promotion boundary

**Classification:** required experiment-architecture clarification.
**Clarifies:** Sections 2.3.1, 2.3.6, 2.8.3-2.8.4, 2.9, 2.12.6-2.12.7,
and 4.3.7; Amendment A15; and the future Milestone 8-12 handoffs.
**Activation:** this amendment authorizes only development-time authoring,
validation, and content identity. It does not activate King of the Hill or
Capture the Flag configuration, state, observations, transitions, collision,
scoring, reward, or policy behavior.

MARL-BattleGrounds may provide one local, developer-only scene-authoring tool
for expressing map and controlled-scenario designs precisely. The tool is a
development communication and validation surface, not a researcher-facing
trainer, evaluation runner, scenario host, task registry, or second simulator.
Its browser or presentation layer never owns catalog mechanics, simulator
truth, or JAX arrays. Authoritative compilation and validation remain host-side
and reuse the existing catalog, configuration, reset, and curated-state
authorities.

Map and scenario authoring are separate semantic concerns. A map owns reusable
static spatial inputs: dimensions, blocking walls and pillars, the exact five
respawn-pad centers for each team, and development annotations for future
objective regions. A scenario references one map and adds ordered Team A and
Team B fixed-slot rosters, initial agent centers, episode rules, sparse
curated-state overrides, roles, schedule, horizon, and endpoints as required by
its owning scenario contract. Duplicate classes and asymmetric
one-to-five-agent rosters remain structurally legal; omitted team-local rows
resolve to inactive fixed-shape padding.

Development authoring has three deliberately different lifecycle stages:

1. A **mutable development draft** preserves editable semantic intent. The
   working names `DevMapDraftV1` and `DevScenarioDraftV1` are explicitly
   non-public and do not pre-approve permanent M10 or M11 type names.
2. A **validated candidate** contains normalized semantic content with its own
   content digest. Its immutable candidate record separately binds that digest
   to versioned compile and validation evidence produced by current product
   authorities. It is not thereby official, canonical, training, validation,
   evaluation, or locked-test material.
3. An **owner-promoted asset** retains that inherent semantic content digest
   and receives the durable public identity, version, and approval record
   assigned by its owning future contract. Evaluation suites, experiment
   manifests, scenarios, or M11 training distributions then select it and
   assign its use.

Mutable drafts are never direct production inputs to training, validation,
official evaluation, or scenario-evaluation pipelines. Future consumers load
owner-promoted semantic assets or the common resolved episode/scenario
contracts. They may subsume the development compiler or retain a thin adapter,
but they must not trust browser state, filenames, mutable draft IDs, or a
parallel development loader as scientific authority.

Maps are partition-neutral. A map document does not declare itself `training`,
`development`, `validation`, `evaluation`, `locked-test`, `official`, or
`canonical`. Those classifications, weights, and leakage constraints belong to
the selecting distribution, suite, scenario, or experiment manifest. Reusing a
map across populations is therefore an explicit manifest decision rather than
a property inferred from its filename or authoring history.

The first authoring boundary exposes only authored facts. Agent class, team and
fixed team-local slot, initial center, alive state, current health, remaining
cooldown, named status durations, spawn-shield duration, out-of-combat
countdown, team scores, timestep, and respawn-wave clocks may be curated within
their existing validation contracts. Body radius, movement speed, observation
and interaction ranges, maximum health, recovery mechanics, damage, healing,
cooldown maxima, and status magnitudes remain catalog-derived and read-only.
Previous-action history remains the canonical neutral initialization and is
not an ordinary first-version authoring control.

The authoritative development path is semantic draft parsing, immutable
profile resolution, padded configuration construction, ordinary reset defaults,
sparse curated-state overlay, product and scenario validation, and normalized
content identity. Map resizing never silently moves authored content; geometry
that becomes invalid remains invalid until the author corrects it. The exact
frontend package, reuse boundary, endpoints, persistence paths, canvas
mechanics, components, wire shapes, and detailed proofs remain deferred until
the accepted replay-viewer foundation has been integrated.

Future-objective geometry is annotation-only until its task milestone owns the
complete semantics. King of the Hill annotations carry hill center and radius.
Capture the Flag carries exactly one static `CTF Team Base` per team: the base
center is that team's flag-home point, and the circle at that center is that
team's capture zone. One geometry record therefore serves both spatial roles
without duplicating coordinates. Pickup radius, dynamic home/carried/dropped
flag state, current flag position, carrier identity, return timing, capture
eligibility, observations, and transition ordering remain separate Milestone 9
decisions. Neither hills nor CTF bases enter current Team Deathmatch
`EnvConfig`, `EnvState`, observations, collision, policy inputs, transitions,
scoring, or reward authority.

The bounded M7 C2 scenario precedent must anticipate promoted authored content
without depending on development draft schemas. Its resolved scenario version
independently freezes and joins a content-addressed layout identity, a
content-addressed authored-initial-condition identity, the resolved
configuration digest, explicit fixed-slot roster and role template, matched
seed schedule and realized coordinate, horizon, and scenario identity. M10
must later subsume or explicitly version this precedent when it defines the
permanent shared contracts and promotion adapters.
