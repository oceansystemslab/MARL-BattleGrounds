# MARL-BattleGrounds Specification Amendments

> **NORMATIVE CONTRACT — ACTIVATED 2026-08-10.** This document records the
> accepted amendments to the historical design PDF.

This is the dated decision record for changes to the historical design PDF.
Each amendment changes only the clauses it names. Later explicit decisions
supersede conflicting earlier clauses; an old future-tense sentence does not
mean that feature is still unimplemented. Keep historical wording, numbers,
source identities and evidence intact when reading later decisions.

Use the [workflow guide](../evaluation/workflows.md) for current executable API
examples and the [protocol](../evaluation/protocol.md) for current scientific
rules. These distinguish implemented behavior, accepted future design and
historical evidence. The current monthly Big 12 snapshot direction supersedes
A27's weekly schedule and fixed numerical example; the final official budget
and remaining admission gates are still unresolved.

The four North Stars remain researcher usability, low sample complexity,
meaningful tactical/strategic team behavior and professional engineering.
Design acceptance is separate from correctness, GPU efficiency and learning
proof. The [documentation standard](../dev/documentation_standard.md) governs
new explanatory prose; it does not authorize changing historical evidence.

**Roadmap numbering:** A36 records the executive mapping. Older titles and file
names keep their original milestone numbers. **Current actor inputs:** A37 owns
the later relative-identity/version changes. A38 records the separate geometry
work and its evidence limits. Neither is silently requalified by a docs edit.

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
- Sparse events form an exactly 24-variant discriminated V1 union derived from
  facts and adjacent frames. The Replay/Debugger baseline contributes the
  rank-50 `AgentLeftCombatEventV1`; the one-time Milestone 7 pre-alpha
  expansion adds authoritative Team Deathmatch score-change and completion
  variants at phase ranks 130 and 140. `AgentDiedEventV1` records the newly dead recipient;
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
integrated event union expands from A2's original 21 to exactly 24 variants:

- `agent_left_combat`, phase rank 50, retained from the authoritative
  Replay/Debugger baseline, records an alive recipient's countdown edge from
  one to zero after countdown reset and before regeneration;
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

**Historical training-selection decision:** the following paragraph records
A15's original plan. [A39](#a39-sampled-training-maps-and-rosters) replaces its
handpicked-roster proposal and resolves the training, validation and test map
selections. [A36](#a36-submission-roadmap-approved-tdm-content-and-m7-closeout)
maps the old milestone numbers to the current roadmap.

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
## A17. SharedObs recorded visual-union presentation

**Classification:** harmless clarification of the rendering and learner-input
boundary established by A1 and A10.
**Clarifies:** A1 and A10; the Milestone 6 presentation contract; and the
conceptual name `shared_obs_recorded_visual_union`.

`shared_obs_recorded_visual_union` names a rendering-only authorized view. Its
frozen V1 wire literals are `shared_obs_visual_union` for the observation mode
and `authorized_same_epoch_sensor_source_visual_union` for the construction
basis. These names describe the same presentation contract; existing V1 wire
literals are unchanged.

The view keeps the selected recipient's own recorded base-sensor row separate
and may add only recorded, same-decision-epoch sensor rows from same-team,
configured-active sources for which the recipient's recorded source
availability is true. It is an authorized visual union, not a recomputed
observation: presentation code must not recreate geometry, visibility, line of
sight, masks, mechanics, or simulator state.

The visual union excludes teammate action masks, prior-action or other history,
rewards, recurrent or policy state, transition facts, critic input, and hidden
Oracle state. The diagnostic `source_material_only` view described by A10 is
not a product Agent-POV presentation root and must not be installed or labelled
as one.

One separate presentation-only exception is the
`local_oracle_corpse_overlay`. Dead bodies are not actor-input observations, so
the trusted Python presentation producer may project configured-active corpses
from the same authoritative epoch when at least one authorized living sensor
places the corpse within its recorded observation radius and static line of
sight. NoSharedObs uses only the selected recipient; SharedObs uses the
deduplicated union of authorized living allied sensors. This exception may add
only Oracle-matching public corpse facts, the public sensor identities that
authorized them, and same-epoch join fields already present in or derivable
from the Agent source. It is consumed for corpse painting and inspection, plus
one narrowly bound lifecycle presentation: paired prior/current projections
may admit only an `agent_died` or `agent_respawned` cue and the phase endpoint
owned by that cue. An overlay-only endpoint must not move an ability route or
admit any other event. The overlay must not enter actor observations, masks,
targeting, legality, accepted actions, simulator or recorded transition
semantics, or hidden-geometry reconstruction in the browser. Dead sensors
authorize no additional visibility.

That exclusion governs the durable Agent scene and learner-facing material; it
does not make visible battlefield actions disappear from replay or debugger
playback.  The presentation layer may carry a separate, ephemeral visual-event
projection derived from the canonical incoming transition.  Every agent and
phase anchor in an emitted ordinary row must already be authorized by the
recipient's fog-filtered start or successor scene. The only exception is the
death/respawn cue-owned corpse endpoint defined above. A payload fact that
describes or derives from an endpoint requires that endpoint to be authorized
even when its canonical coordinate anchor belongs to an earlier phase; for
example, hidden successor health, regeneration, and cooldown outcomes cannot be
disclosed through a transition-start anchor. Hidden sources, targets, aura
emitters, inactive identities, and global-only pulses are omitted server-side.
Surviving event rows use a dense recipient-local identity axis, so canonical
event IDs, hidden counts, ordering gaps, slots, and Oracle transition identities
never cross the Agent presentation boundary. The corpse-only exception may
carry the canonical Oracle frame ID because it is exactly derivable from the
already-disclosed episode and frame index; it carries no additional frame fact.
The event projection exists only to render the same visible action semantics as
Oracle View under fog; it is not stored in or derived from the learner's
actor-input artifact.

Neither representation is a materialized SharedObs learner input. Exact
SharedObs actor-input export remains unavailable until the Milestone 12
compositor is implemented, performance-tested, and accepted. Presentation and
replay support therefore do not activate SharedObs training or authorize any
claim that the composed learner tensor is available.

## A18. Final pre-alpha obstacle-capacity expansion

**Classification:** one-time, approved pre-alpha V1 schema migration.
**Supersedes:** V1 immutability and schema-freeze clauses only where they fix
the obstacle-slot axis at 16. All other V1 identities and semantic contracts
remain unchanged.

The combined wall-and-pillar capacity expands from 16 to 32. Resolved
configuration therefore carries exactly `[32, 8]` obstacle rows, and each
captured base observation carries exactly `[10, 32, 8]`. The independent V1
evaluation wire constant advances to the same capacity without importing the
live simulator constant.

Active obstacles occupy one contiguous, semantically ordered prefix. Every
inactive row is canonical all-zero padding. Appending places an obstacle at the
end of the active prefix, deletion compacts that prefix, and explicit reordering
changes fixed-slot order. Rows 16 through 31 must reach policy observation,
capture, replay, presentation, and rendering whenever the simulator consumes
them for collision, movement legality, or line of sight.

This is the final approved pre-alpha in-place V1 break. Existing development
replays, fixtures, schemas, manifests, metric companions, and content digests
are disposable and must be regenerated at 32 rows. No V2 family, legacy loader,
dual-schema branch, padding compatibility shim, or alias for 16-row artifacts
is authorized. V1 refreezes at 32 immediately after the migration.

For every pre-migration layout with at most 16 active obstacles, the original
obstacle prefix, state, action masks, rewards, and transitions remain unchanged;
the observation change is only the added canonical zero tail and consequent
artifact identities. Obstacle geometry, overlap, ordering, collision, movement,
visibility, line-of-sight, observation-redaction, and rendering semantics do not
otherwise change.

## A19. SharedObs structured runtime advancement

The global-slot policy payload described below is historical. A37 replaces its
current execution layout and callback identity argument. V1 recordings keep the
layout below for exact historical reconstruction.

**Classification:** accepted milestone-ownership advancement without a core
observation or simulator change.
**Supersedes:** the timing clauses in A1, A12, A13, and A17 that defer the
entire SharedObs compositor and every executable SharedObs policy path to
Milestone 12. It does not supersede their information boundaries,
homogeneous-episode requirement, or separate-reporting requirement.

The Milestone 7 interlude activates the canonical structured SharedObs source
bank for scripted/reference policy execution. SharedObs is the
researcher-facing default information regime. NoSharedObs remains an equally
legitimate, first-class regime selected by the same high-level
`execution_information_mode` setting and reported in a separate stratum. It
is not reduced to an ablation label, and results from the two regimes remain
ineligible for pooling.

At each decision epoch, the runtime factors the already-authored base
observation into exactly one source bank:

```text
SharedObsSensorSourceBankV1
  unit_features_by_sensor_source_and_global_slot   float32[10, 10, 58]
  unit_visibility_by_sensor_source_and_global_slot bool[10, 10]
  objective_features_by_sensor_source              float32[10, 8, 12]
```

The source bank is a stable global-slot remapping of ordinary, same-epoch,
visibility-redacted base-sensor rows. It does not recompute geometry,
visibility, line of sight, lifecycle redaction, or any simulator mechanic.
Configured-active dead sources remain present in the static authorization
topology but contribute no ordinary unit or objective sensor material. The
recipient's own complete base observation remains separate and authoritative.

The default recipient-by-source availability matrix has shape `(10, 10)` and
authorizes only configured-active, same-team, off-diagonal sources. Its
diagonal, cross-team, inactive-recipient, and inactive-source cells are false.
The bank contains no teammate action mask, prior-action/history field, reward,
transition or successor fact, raw state, critic state, policy memory, or
renderer data. A SharedObs scalar policy receives the recipient base
observation, recipient action mask, actor key, source bank, recipient
availability row, and recipient global slot. The focal action mask remains the
sole action-legality authority.

The versioned SharedObs actor projection is
`base-observation-plus-authorized-sensor-source-bank@1`. Evaluation and replay
records retain base observations, exact availability, execution mode,
projection identity, and stable source/global-slot mapping. They never persist
a second materialized SharedObs tensor; exact structured source-bank material
is reconstructed on demand from those authorities.

The existing reference rollout owns one statically selected, homogeneous
episode mode. Both teams use that mode and choose all actions from the same
decision epoch before one unchanged joint-action assembly and one unchanged
core step. SharedObs and NoSharedObs share reset, actor/environment key
protocol, scan, done padding, capture, replay, metric, and failure semantics.
Researcher-facing launch surfaces default to SharedObs, while the low-level
reference rollout requires the mode explicitly and rejects an opposite scalar
policy ABI before JIT. Its result carries the exact availability consumed by
the episode (`None` for NoSharedObs), which is the authority passed to capture.
Before any scalar SharedObs callable runs, every unavailable source row is
zeroed across all three bank fields. The compiled NoSharedObs branch bypasses
source-bank and availability construction. V1 still cannot represent mixed
SharedObs/NoSharedObs actors in one episode; A12's future mixed-regime V2 gate
remains unchanged.

Milestone 12 continues to own learned neural input front ends, encoders,
critics, trainers, updates, learned-policy integration, checkpoint
compatibility, and baseline training. It must provide two compatible neural
front ends behind the same high-level regime setting while retaining the
common trainer, trunk, heads, masking, rollout lifecycle, and update loop.
Checkpoints remain regime-tagged rather than switching regime in place. This
amendment changes no `EnvConfig`, `EnvState`, `Observation`, `Action`, mask,
reset, transition, reward, termination, geometry, or task semantic.

## A20. DevClient authoring and saved-scenario execution boundary

**Classification:** required product and experiment-architecture clarification.
**Supersedes:** the product name and live-inheritance assumptions in A16, plus
the requirement to author speculative King of the Hill and Capture the Flag
objective annotations before those task milestones resume.
**Clarifies:** Sections 2.3.1, 2.3.6, 2.8.3-2.8.4, 2.9, 2.12.6-2.12.7,
and 4.3.7; Amendments A15-A17; and the Milestone 7-12 handoffs.

The live developer product is the **MARL-BattleGrounds DevClient**. It contains
three deliberately small areas: the existing Combat Debugger, a reusable Map
Author, and a task-controlled Scenario Author. Internal
`visual_debugger` package paths and the `combat_debugger` wire identity may
remain unchanged as implementation details. The read-only Replay Viewer is a
separate researcher-facing application. It retains its own launcher, product
identity, routes, lifecycle, title, controls, and artifact responsibilities and
must never receive DevClient authoring authority or navigation.

The Map Author and Scenario Author reuse the DevClient's established native
HTML, CSS, JavaScript, SVG coordinates, icons, viewport behavior, authenticated
loopback service, and visual language. Authoring uses one selection, direct
center dragging, exact numeric fields, fixed snapping with an explicit bypass,
keyboard nudging, local undo/redo, ordered obstacle controls, and linked host
problems. It deliberately omits a frontend framework, multi-selection,
alignment guides, transform handles, layers, groups, freehand geometry,
configurable grids, collaboration, plugins, and generic asset-management
machinery. The existing read-only combat renderer remains read-only; a small
authoring renderer/controller sits beside it.

A reusable map contains positive finite decimal dimensions, an ordered list of
walls and pillars, and exactly five fixed spawn pads per team. Obstacle order is
semantic fixed-slot order. The one-world-unit visual grid is not persisted.
Wall rotation is authored in degrees and normalized host-side to float32
radians. Ten pads always exist. Obstacle overlap is permitted, and out-of-bounds
obstacles produce a warning rather than a new simulator-invalid condition.

A scenario owns an embedded, independently editable copy of complete map
content. Optional source-map identity, revision, or digest is nonsemantic
provenance only: later map changes never propagate into the scenario, and
scenario edits never mutate the source map. The scenario task section is a
discriminated contract with only `team_deathmatch` in this version. King of the
Hill and Capture the Flag remain specified future tasks, but their runtime and
authoring work is postponed until after the other roadmap milestones. Their
history is not removed, and their real task-owned union members are added only
when those milestones resume; no speculative objective controls, plugin
framework, or generic objective abstraction is authorized now.

Team Deathmatch scenarios carry complete fixed Team A and Team B roster blocks,
one-to-five-agent contiguous active prefixes, per-slot initial state, episode
configuration, current global state, and bounded controlled-study metadata.
Physical scenario content is independent of controller assignment and
information regime. The same saved scenario can therefore be reset to the same
immutable state and seed for manual/manual or manual/scripted comparison under
SharedObs or NoSharedObs without duplicating the asset. Previous-action history
is always the canonical neutral initialization and is not authorable.

The authoritative host pipeline is strict whole-draft parsing, ordered map
normalization, fixed-shape obstacle padding, catalog profile resolution,
ordinary reset, overlay of only authorized state fields, forced neutral
previous-action history, existing product configuration validation, existing
scenario initial-state validation, and authored-state initialization. Browser
code never constructs JAX arrays, recreates mechanics, repairs invalid content,
or becomes a second simulator. No change to core configuration, state,
observation, action, reset, step, geometry, combat, reward, termination, or task
semantics is authorized by this amendment.

Development assets use three explicit lifecycle stages: a mutable
revision-fenced draft, an immutable content-addressed candidate produced by
validation, and an explicit owner-only promotion to normalized tracked content.
Draft saves are atomic and stale revisions fail closed. Semantic identity
excludes names, timestamps, draft revisions, browser object IDs, and source-map
provenance, while retaining ordered geometry, embedded scenario map content,
roster, configuration, state, and study contract. Freeze and promotion never
overwrite an existing identity. A map remains partition-neutral; selecting
training or evaluation populations owns partition and leakage decisions.

The Combat Debugger includes a simple persistent saved-scenario loader. On a
later DevClient session it lists saved scenario drafts whose latest revisions
are execution-valid and frozen scenario candidates. Loading reopens the exact
saved revision or candidate, strictly parses it, recompiles it through current
authorities, and revalidates it before atomically replacing the Debug session.
A failed revalidation returns linked problems and leaves the current session
untouched. `Open in Debug` from the Scenario Author calls this same host loading
service; it is a convenience route, not a separate execution model and not the
only way to load saved work. The current authoring buffer is first compiled and
validated into an immutable in-memory candidate snapshot; mutable browser
state is never executed directly.

Combat execution continues to use one current decision epoch, one joint action,
and one unchanged simulator step. Team A remains manually controlled in this
version. Team B is selectable between manual control and the scripted Team
Deathmatch policy. SharedObs and NoSharedObs are selectable run configuration,
with SharedObs the researcher-facing default and NoSharedObs a separately
reported first-class regime. Changing controller or information regime resets
the exact loaded scenario snapshot and seed. Manual runs are diagnostic and do
not become official policy-evaluation evidence.

## A21. Minimal DevClient scenario and asset-workflow correction

**Classification:** approved private pre-alpha authoring-contract cutover and
product-workflow clarification.
**Supersedes:** only A16 and A20 clauses that require authored agent-role labels,
controlled-study metadata, or a study contract in DevClient scenario content or
semantic identity. It does not supersede the separate controlled-evaluation,
policy-assignment, evidence-role, seed-schedule, endpoint, Replay, or promotion
contracts owned by later evaluation milestones.

The DevClient Scenario Author defines executable episode content: name,
description, optional ordinary notes, an embedded map snapshot, task
configuration, fixed-slot roster and classes, per-slot initial state, episode
configuration, and current global state. It does not author a privileged focal
agent, cooperative/adversarial evidence roles, hypotheses, seed schedules,
measurements, endpoint policies, pressure protocols, versioned predicates, or
other controlled-study declarations. A future experiment or evaluation
definition may reference a promoted scenario and independently assign those
run- and evidence-level facts.

The private authoring V1 contract therefore removes `role` from roster rows and
removes the complete `study` member. It adds optional plain `notes`, bounded to
8,000 characters. The strict parser rejects obsolete authoring documents that
still contain `role` or `study`; no compatibility shim, silent migration, or
dual-read path is authorized. This is an in-place pre-alpha cutover because no
scenario draft or candidate has entered durable use. Existing map drafts and
candidates remain compatible and unchanged.

Scenario semantic identity excludes name, description, notes, browser object
identities, timestamps, draft revision, and source-map provenance. It continues
to include the ordered embedded map, roster topology and classes, task and
episode configuration, initial state, and current global state. Successful
authoritative compilation and existing validation are sufficient for both
execution validity and candidate freeze qualification. Freezing creates an
immutable content-addressed snapshot only; it does not promote, partition,
canonize, train on, or evaluate the asset.

The Combat Debugger initially controls the first active Team A slot selected by
the fixed physical roster topology. That selection is not persisted as an
authored role. Existing evaluation and Replay layers may continue to derive
their independent focal/cooperative/adversarial evidence vocabulary from the
controlled slot and policy assignments; this amendment changes none of those
public contracts or recorded bytes.

Maps may be inspected in Combat through an explicit **Default TDM map preview**.
The authoritative host reopens the exact current buffer, saved revision, or
frozen candidate; independently copies its map content into the ordinary
default Team Deathmatch scenario factory; and compiles and validates the
result before atomically replacing the current Debug session. The transient
preview uses mirrored Mage/Warrior/Hunter/Rogue/Priest five-agent teams at the
authored pads, full catalog health, threshold 5, 300 maximum steps, spawn
shield duration 3 at speed 2.0, respawn periods 5 with countdowns 4, zero
scores/step/status/cooldown state, and neutral previous-action history. It is
identified by `default-tdm-map-preview@1`, is visibly distinguished from an
authored scenario, never mutates or implicitly saves the source map, and leaves
the current Debug session untouched on any resolution or validation failure.

Spawn pads remain simulator coordinate points with no core radius. DevClient
authoring renders their validation footprint using the maximum positive body
radius in the live catalog, with a defensive `0.5` fallback. Humanized
mechanics labels, rounded read-only presentation, obstacle display identities,
saved-asset selectors, and save/freeze location feedback are browser
presentation and workflow concerns only. Editable numeric values, persisted
content, simulator mechanics, Replay behavior, and evaluation semantics remain
exact and unchanged.

## A22. Minimal saved-asset and symmetric-controller workflow

**Classification:** approved private pre-alpha product-workflow simplification.
**Supersedes:** A16, A20, and A21 only where they require a DevClient candidate,
Freeze, freeze qualification, owner-promotion stage, promoted source, or
Team-A-manual-only Combat configuration. It preserves their authoring/compiler,
map-copy, scenario-content, in-memory snapshot, Replay-separation, and future
evaluation boundaries.

An exact revisioned Save is the sole DevClient content-persistence operation.
Save and Save As atomically create immutable numbered revisions under the
existing ignored map or scenario asset identity; an exact revision may be
reopened, copied into a scenario, loaded into Combat, or supplied as a
deterministic map preview source immediately. The revision fence remains an
invisible concurrency guard so stale browser state cannot overwrite or delete
newer work.
The DevClient has no Freeze command, freeze-qualified state, candidate schema,
candidate storage, candidate source, promotion command, promotion wrapper, or
tracked-configuration publication path. No renamed or hidden equivalent is
authorized.

Both authoring areas expose one confirmed saved-asset deletion. Deletion names
the exact map or scenario identity and expected latest revision, then removes
all saved revisions for that identity. Cancellation changes nothing; a missing,
stale, unsafe, symlinked, or malformed target fails closed. Successful deletion
refreshes Map, Scenario, scenario-source, and Combat discovery. If the deleted
asset is open, its current browser content remains an unsaved recovery copy;
an already loaded immutable Combat snapshot remains runnable and resettable.
Deletion never reaches outside the fixed ignored DevClient draft roots.

Each obstacle retains one immutable, unique structural object ID for joins,
selection, ordering, dragging, and linked validation. Its separate display name
is editable, need not be unique, and falls back visually to that object ID when
blank. Renaming changes neither semantic identity nor the obstacle's ordered
fixed-slot position or geometry. Maps and embedded scenario maps use the same
editor and validation behavior.

Team A and Team B each select a controller from the same run-configuration
boundary. V1 exposes `manual` and the existing scripted Team Deathmatch policy
for either team. Both scripted teams choose actions from the same current
decision epoch and still enter one joint action and one unchanged simulator
step. Switching either team controller or SharedObs/NoSharedObs resets the exact
loaded snapshot and seed. Scenario documents remain controller- and
information-regime-independent. The selector contract may later admit real
saved policy/controller identities, but this amendment does not invent policy
storage, a plugin system, learned-policy loading, checkpoint semantics, or a
generic controller registry now.

Future training, experiment, or evaluation contracts may import an exact saved
DevClient revision. Their owning manifest or pipeline must independently
reopen, normalize, validate, content-address, approve, partition, and bind that
content to its resolved scientific identities. A moving latest-revision alias,
mutable browser buffer, filename, or successful Combat preview remains
insufficient scientific evidence. This later lifecycle is not reintroduced as
a DevClient promotion stage. Public evaluation roles and Replay contracts are
unchanged.

## A23. Editable obstacle identities and random controller execution

**Classification:** approved private pre-alpha authoring and diagnostic-control
correction.
**Supersedes:** A22 only where it requires immutable obstacle object IDs, a
separate obstacle display name, or exactly two interactive controller values.
It preserves A22's revisioned Save/Delete lifecycle, symmetric team ownership,
single simulator path, Replay separation, and future learned-policy boundary.

An obstacle has one visible identity: its `object_id`. The Map and Scenario
authors permit that ID to be edited directly for walls and pillars. Spawn-pad
and agent IDs remain fixed. One rename trims surrounding whitespace, enforces
the existing host `ObjectId` grammar, rejects every collision with an obstacle,
pad, or scenario agent before mutation, and then updates selection and linked
validation through one undoable transaction. Invalid, duplicate, and
same-value requests have no persistence or simulator effect. Deletion does not
renumber surviving obstacles; the author may explicitly close a numbering gap.

The redundant obstacle `name` member is removed from the strict private V1
authoring contract and browser. Obsolete input carrying that member is rejected
without a compatibility shim. Obstacle IDs remain browser/validation join
metadata: ordered obstacle rows still own fixed-slot semantics, while object
IDs remain excluded from semantic map/scenario digests and compiled physics
arrays. Save/Open, Reset, map copy, and scenario duplication preserve the exact
authored IDs.

Both interactive team selectors admit `manual`, `scripted_tdm`, and
`random_valid`. `random_valid` invokes the existing team-agnostic policy that
samples only the exact valid action support; this amendment does not change
that policy or create a registry, checkpoint format, plugin system, or saved
controller lifecycle. A private SharedObs ABI adapter deliberately ignores
shared sensor material and delegates to the same random policy, so equal keys
and masks produce equal random actions in SharedObs and NoSharedObs. Both teams
still choose from one current observation/mask epoch, assemble one joint
action, and enter one unchanged simulator step. Shared execution constructs at
most one source bank for the epoch; NoSharedObs constructs none.

Per-slot provenance records `random_valid`, algorithm
`canonical-random-valid`, and stochastic execution. Manual plus any policy is
`mixed`; Scripted TDM on both teams remains `scripted`; a fully
policy-controlled pair containing Random is `policy`. Actual policy execution
is true whenever either interactive controller is nonmanual. Existing Replay
artifacts and Replay Viewer behavior remain unchanged; the strict shared live
presentation and debugger-recording validators merely admit the new truthful
controller and action-source identities.

## A24. Complete saved-asset discovery and snake-case identities

**Classification:** approved private pre-alpha identity cutover and authoring
presentation correction.
**Supersedes:** A22 and A23 only where they permit another Map or Scenario asset
ID convention, generate hyphenated obstacle IDs, or leave saved-asset ordering
lexicographic. It preserves revisioned Save/Delete, editable obstacle identity,
authoritative compilation, Replay separation, and every simulator contract.

The DevClient has no saved-asset list limit. Each Maps, Scenarios, scenario
source, and Combat selector presents every discoverable latest saved revision
for its applicable asset kind. Presentation uses stable numeric-aware asset-ID
ordering, so `map_9` precedes `map_10`, and places the exact persistent asset ID
first in each label. The existing native single-select control owns scrolling
and keyboard navigation; no pagination, custom combobox, or asset-manager
surface is introduced.

Private Map and Scenario asset IDs now use lowercase snake case, with at most
64 characters and grammar `[a-z0-9]+(?:_[a-z0-9]+)*`. Human-facing map and
scenario names remain ordinary free-form prose. Newly allocated obstacle IDs
use `obstacle_0`, `obstacle_1`, and so on. The general editable `ObjectId`
contract remains unchanged, as do authored legacy-shaped obstacle identities,
spawn-pad IDs, and agent IDs. This convention changes browser and persistence
identity only; obstacle order continues to own compiled fixed-slot semantics.

Save remains an explicit user operation and no autosave is authorized. A
successful Save or Save As atomically persists an immutable numbered revision
under ignored `artifacts/dev_client/` storage; shutting down or restarting the
DevClient does not remove it. Every successful save refreshes complete asset
discovery immediately.

As a one-time pre-alpha exception to A22's byte-immutable local revisions, the
existing ignored Map and Scenario revision histories may be migrated in place
from hyphenated asset IDs to snake case, with generated `obstacle-N` identities
converted to `obstacle_N`. The maintenance operation must stop the DevClient,
hold the store lock, reject unsafe or colliding input, create an exact ignored
backup and checksum manifest, construct and validate the complete replacement
before an atomic swap, preserve every revision number and semantic field, and
roll back wholly on failure. Any saved-map provenance in existing scenarios is
migrated consistently. Post-migration strict parsing, compilation, validation,
revision counts, and semantic digests must match the pre-migration authorities.
No automatic startup migration, alias, dual-read path, or compatibility shim is
created. The private authoring contract refreezes immediately after this
single migration; deletion of its backup requires separate owner approval.

This amendment changes no `core/` source or contract, simulator behavior,
Replay Viewer production code, Replay schema, replay fixture, or replay
artifact.

## A25. SharedObs-only canonical benchmark execution

**Classification:** approved Paper 1 benchmark-contract consolidation and
official-evidence qualification.
**Supersedes:** A1, A7, A11, A12, A14, A19, and A20 only where their forward
requirements make NoSharedObs a separately reported official regime, require
dual-regime official cells or checkpoints, or require a mixed-regime V2
artifact family. It preserves their implemented SharedObs semantics, common
rollout lifecycle, local observations, replay reconstruction, historical
calibration record, generic dual-mode compatibility, and regime-independent
saved scenarios.

Paper 1 has one canonical execution-time actor-information contract:

```text
execution_information_mode = shared_obs
actor_projection = base-observation-plus-authorized-sensor-source-bank@1
```

Every frame of an official replay carries the exact Boolean availability
matrix implied by its frozen configured roster. For recipient slot `r` and
source slot `s`, availability is true exactly when both slots are
configured-active, both have the same configured team identity, and `r != s`.
All other entries are false. This is a complete equality requirement, not only
a forbidden-cell check: an arbitrary subset of permitted teammate edges, an
all-false matrix for a multi-agent team, or any mid-episode change is
noncanonical. An all-false matrix remains canonical when the configured roster
genuinely has no same-team off-diagonal pair, such as a 1v1 episode.

The matrix is derived from immutable configured-roster topology, never from
current alive state, health, visibility, score, controller identity, or frame
index. A configured-but-dead teammate remains an authorized source; its
ordinary sensor material remains zeroed by the existing lifecycle contract.
Official validation checks the exact matrix on every replay frame so the
recorded actor-information topology cannot silently narrow or change during an
episode.

Official baseline training and evaluation, controlled-scenario evaluation,
tournament and cross-play evaluation, Elo or other rating inputs, leaderboards,
and Paper 1 reporting use this canonical SharedObs contract exclusively.
Execution information is invariant provenance for those results, not an
evaluation cell, aggregation stratum, ablation axis, or symmetry direction.
There are no official NoSharedObs baselines, checkpoints, scenario strata,
tournament cells, or Paper 1 claims, and the current roadmap contains no
mixed-regime V2 work.

NoSharedObs remains a supported generic execution mode for diagnostics, custom
research, and historical replay compatibility. Existing NoSharedObs evidence
remains parseable, semantically valid under its versioned generic contract, and
renderable, but it is ineligible as new official benchmark evidence. Generic
SharedObs validation may likewise accept a structurally permitted subset
topology; only the official eligibility boundary requires the complete matrix.
The V1 wire format, its episode-global mode and projection fields, its
dual-mode readers, and its prohibition on implicit mixed execution remain
unchanged. A future mixed-regime artifact would require a new, separately
approved amendment and versioned contract rather than being inferred from this
amendment.

Maps and scenarios remain information-regime-independent authored assets.
Loading one does not mutate or duplicate it; an official execution owner binds
the canonical SharedObs mode, projection, and availability topology after the
asset has been independently normalized and validated. KOTH and CTF inherit
this same task-independent actor-information contract when their work resumes
after Paper 1.

Centralized training is a separate authority. A critic may consume explicitly
authorized training-only team or privileged information, but actor action
selection still consumes only the canonical SharedObs input above. Critic
inputs may not be relabelled as actor availability or enter an evaluation
frame's actor projection.

Milestone 12 freezes the actor-input semantic contract: categorical identities,
domains, sentinel meanings, reference-encoder conventions, fixed shapes,
slot/source identities, masks, and provenance. Its canonical learned-policy
boundary is:

```text
recipient base observation
+ authorized same-epoch teammate source bank
+ exact recipient availability row
+ recipient/global-slot identity
→ researcher-selected or benchmark reference encoder
→ policy network
```

The reference encoder supports shipped baselines; researchers may replace it
with one-hot, embedding, attention, or another representation without changing
the authorized input contract. This amendment therefore standardizes
information, not neural architecture.

The built-in Random controller remains a diagnostic and quality-control policy.
It is not an official baseline unless a later amendment explicitly promotes it
into an official reported baseline family. Milestone 10 must reconcile
delivered common-pipeline work, work superseded here, and any genuinely
unfinished runner contract; removal of obsolete dual-regime requirements alone
does not complete that milestone.

This amendment changes no `core/` source or contract, simulator behavior,
observation, mask, transition, reward, saved DevClient map or scenario, replay
schema or artifact, generic `ExecutionInformationMode` union, SharedObs or
NoSharedObs policy implementation, or Replay Viewer behavior.

## A26. Scenario pressure controllers and behavioral ablations

**Classification:** approved controlled-evaluation architecture and scientific-
integrity clarification.
**Supersedes:** A13 only where it classifies or implies that the generic
scripted Team Deathmatch policy is a published benchmark baseline, or prohibits
a controlled scenario from binding a separate scenario-pressure-controller
identity. It preserves A13's generic Team Deathmatch scorer and parameter
profile, SharedObs adapter, team-agnostic execution, authoritative-mask and
same-epoch causal semantics, trace ontology, and historical calibration
evidence. It also preserves A25's SharedObs-only official actor-information
contract.

The existing generic Scripted SharedObs Team Deathmatch policy is a DevClient,
debugging, regression, and behavior-inspection controller. It is not an
official baseline, a Big 12 entrant, or a source of general-strength claims.
The built-in Random controller remains diagnostic and quality-control tooling.
A materially improved general scripted policy may become an official baseline
only through a later explicit qualification and amendment; its present
availability does not grant that status.

Official controlled-scenario evaluation instead binds an explicit
scenario-pressure-controller identity through the resolved evaluation
definition's `pressure_protocol`. A pressure controller is an evaluation
fixture, not another simulator, task policy family, learned baseline, or
authored-scenario field. Saved DevClient scenarios remain independent of
controller, policy, checkpoint, and information-regime identity. The
evaluation definition joins an exact saved scenario revision, the pressure
protocol, policy assignments, seeds, sides, endpoints, and other evidence
authorities without mutating that scenario.

Each pressure controller is:

- deterministic and reactive, selecting from the current decision epoch rather
  than replaying a stored sequence of exact actions;
- versioned and content-addressed so its complete behavior is frozen by the
  evaluation definition;
- limited to the canonical same-epoch SharedObs input authorized by A25 and the
  exact current action mask;
- explicit about every controlled slot, observation-dependent target choice,
  deterministic tie break, and invalid- or unavailable-action fallback; and
- held identical across every treatment and ablation policy in the comparison.

Given identical authorized inputs, mask, initial state, and protocol identity,
the controller must return identical actions without hidden randomness, raw
state, successor facts, replay facts, result feedback, or treatment identity.
It may express a simple behavioral rule such as following an observed
lowest-health ally while selecting a valid heal, but it must not encode a
frame-indexed action tape or inspect facts outside the actor contract. All
controlled actions are chosen from the same `o_t` and `m_t` as the evaluated
actors, assembled into one joint action, and enter one ordinary simulator step.

The approved initial suite contains eight TDM scenarios. Scenarios 1, 2 and
4–8 have five-transition evidence horizons. Scenario 3 alone has an approved
ten-transition horizon to examine sustained body blocking; its canonical
respawn-wave period remains five transitions. These are properties of the
frozen evaluation definitions, not a global scenario-schema restriction.
See A36 for the executive acceptance and supersession record.

The primary scientific role of an official scenario is a matched behavioral-
ablation comparison:

```text
full method
vs.
matched ablation

under the same:
- exact scenario revision, embedded map, and initial state
- deterministic pressure-controller identity
- canonical SharedObs contract
- evaluation seeds and side assignments
- training budget and checkpoint-selection rule
- primary behavioral endpoint
```

Each evaluation definition declares one primary behavioral claim and one
corresponding primary endpoint. It may declare no more than two supporting
secondary margins. Scenario results are controlled evidence that the tested
component changes the named behavior under the frozen conditions; they are not
universal proof of causality outside those conditions, a general-strength
score, or tournament evidence. Scenario outcomes never contribute to Elo or
another Big 12 rating.

The comparison uses multiple independently trained treatment/control pairs.
The independently trained run is the replication unit. Agents, slots, ticks,
scenario episodes, maps, and repeated observations within one run are not
independent replications and may not be counted as such. Treatment and control
must retain matched budgets, validation-only checkpoint-selection authority,
scenario schedule, seeds, sides, pressure protocol, SharedObs provenance, and
endpoint semantics. Any deliberate mismatch must be declared as a different
experiment rather than hidden inside the ablation result.

The accepted scenario suite is public evaluation material. `Locked` means
protocol-frozen and content-addressed, not secret. Upon acceptance, the exact
scenario revision and its full content closure are published immediately in an
immutable evaluation manifest. This documentation amendment does not itself
publish or alter unfinished local authoring assets.

No official scenario material may influence an adaptive decision. The
prohibition includes gradients, online or offline learning, reinforcement
learning, imitation, behavioral cloning, distillation, curriculum or opponent
adaptation, reward or heuristic design, architecture or hyperparameter choice,
prompt or decoding choice, checkpoint selection, early stopping, repeated-
submission selection, population weighting, or any other decision adapted in
response to the evaluation content or its results.

The protected content closure includes the scenario's embedded map, initial
state, roster and configuration, pressure controller, seed schedule, endpoints,
replays, and result feedback. A map embedded in an official scenario is
therefore ineligible for an official training or validation manifest even if a
separate saved-map revision contains identical content. Identity aliases and
different filenames do not defeat content-digest disjointness.

Exception of 22 September 2026: the ordinary built-in `tdm-alpha` and `tdm-beta`
teams may be training opponents (the pinned opponent of a training run), even
though their rules drive the scenario pressure controllers. Every other part of
the scenario closure stays protected. All eight scenario results of a run whose
training met either team, directly or through a pinned export trained that way,
are familiar-opponent results and do not count as protected-scenario evidence;
the run records that exposure. The same rule applies whenever a run's recorded
exposure is unknown, as for a researcher method or an export whose training
history cannot be linked: its eight scenario results do not count as
protected-scenario evidence.

Milestones 11 and 12 enforce this boundary proportionately through
content-addressed and mutually disjoint training, validation, and evaluation
manifests; complete training, checkpoint-selection, and controller provenance;
and maintainer reproduction of Big 12 systems. A claimed result that cannot be
reproduced within the frozen protocol is ineligible for official reporting.
Non-reproduction alone is not proof of fraud, while intentional undisclosed use
of protected evaluation material for adaptation is scientific misconduct.
Open-source software cannot make such misconduct technically impossible, and
the current roadmap does not add a secret-scenario service, private hosted
evaluator, attestation system, or submission sandbox.

Scenario pressure controllers remain outside the Big 12 and the cumulative
Baseline Library defined by A27. Their content identities and outputs may be
retained as evaluation provenance, but they are not training opponents,
published baselines, or independently rated systems.

## A27. Rolling Big 12 and Baseline Library governance

**Classification:** approved Paper 1 baseline-roster, checkpoint-selection,
rating, and longitudinal-governance contract.
**Supersedes:** historical or planning language that treats independently
trained runs, checkpoint histories, population members, diagnostic policies,
or scenario controllers as separate tournament entrants; specifies a Big 12
cardinality other than twelve; gives the current generic scripted policy
official baseline status; or permits a mutable latest-roster alias to select
training inputs. It
preserves A25's SharedObs-only official execution and A26's independent
scenario-ablation boundary.

The Big 12 contains exactly twelve method-level entrants. Each numbered,
versioned method/configuration row is one entrant identity, so separately
declared variants of the same broad algorithm may occupy distinct rows. Each
entrant supplies one validation-selected, fixed executable system to the active
tournament and receives one Elo. Independently trained runs and their
checkpoint histories are selection and reproducibility evidence, not
additional entrants or rating subjects.

The tentative initial method roster is:

1. RNN-IPPO, parameter-shared;
2. RNN-MAPPO, parameter-shared;
3. RNN-MAPPO, class-specific actors;
4. RNN-HAPPO;
5. HyperMARL-PPO;
6. RNN-QMIX;
7. RNN-PQN-VDN;
8. MAPPO-PFSP League;
9. MAPPO-PSRO;
10. S*-Curriculum;
11. S*-Curriculum-Shaped; and
12. Qwen-Five.

This roster is a planned target, not a claim that the methods, training
pipeline, or tournament are already implemented or qualified. Exact HyperMARL
method identity, PFSP and PSRO training/runtime semantics, stochastic
evaluation policy, compatibility requirements, and other unresolved method
details must be frozen and qualified before activation rather than guessed from
the labels above.

For each of rows 1–11, training produces three independent runs and retains all
three checkpoint histories. Before training begins, the owning protocol freezes
the validation distribution and metric, checkpoint cadence, checkpoint
eligibility rules, run-comparison rule, and every tie break. Each run first
selects its eligible checkpoint using only that frozen validation rule. The
highest validation-scoring selected checkpoint among the three runs becomes
the method's one fixed tournament system, with the frozen tie break applied
when necessary. All run identities, seeds, configurations, checkpoints,
validation evidence, exclusions, and selection evidence remain inspectable.
Official scenarios, scenario results, tournament games, tournament ratings,
and other locked evaluation feedback may not select or alter the system.

Qwen-Five contributes at most one fixed executable system. It is tentative and
cost-gated: measured throughput, latency, resource consumption, execution
stability, reproducibility, interface compatibility, and tournament-budget
feasibility must pass before it may occupy the twelfth active position. It is
not presumed compatible with the JAX learner loop, and its inference or
artifact costs may not be hidden. Failure of the feasibility gate leaves the
roster position unresolved until an explicit governance decision; no substitute
is silently invented or admitted.

The Paper 1 Big 12 tournament therefore contains:

```text
12 method entrants
12 fixed tournament systems
66 unordered pairings
100 episodes per pairing
6,600 tournament episodes
1 Elo per method
```

Each unordered pairing evaluates five maps by ten evaluation coordinates by
two side assignments. The side assignments belong to the same unordered
pairing and do not create more entrants. Every episode uses A25's canonical
SharedObs actor-information contract. The raw win/draw/loss matrix, exact
episode manifests, failures, and provenance are the authoritative result; a
compact rating is a derived presentation.

The planned compact rating is a jointly fitted, draw-aware
Bradley–Terry–Davidson rating centred at 1000, reported as one Elo-style value
per method. Exact estimator parameterization, numerical implementation,
uncertainty reporting, convergence criteria, missing/failing-episode handling,
and validation against the raw matrix are mandatory pre-tournament activation
gates. The word `Elo` does not authorize sequential order dependence or permit
the derived rating to replace the raw matrix.

After Paper 1, the live Big 12 is governed through immutable dated weekly
snapshots. A challenger becomes eligible only after peer-reviewed publication,
a compatible versioned implementation, reproducible qualification, and all
then-current safety, resource, and protocol gates. When a qualified challenger
is admitted, promotion and relegation apply to one exact versioned entrant
identity, never an individual checkpoint or an entire broad algorithm family,
and the bottom entrant leaves active Big 12 status so the active roster remains
exactly twelve. If no qualified challenger exists, the roster does not change
merely because a weekly review occurred.

The Paper 1 roster, manifests, systems, raw results, and ratings remain
permanently frozen after publication. Every later weekly snapshot likewise
retains its exact member systems, tournament/evaluation contract, inputs,
outputs, and governance decision. Ratings centred within different weekly
pools are not directly comparable as a longitudinal measure when membership or
systems change. A separate explicitly validated longitudinal model would be
required for that claim. Exact challenger scheduling, multiple simultaneous
challengers, ties at promotion or relegation boundaries, weekly refitting, and
exception handling remain mandatory pre-launch governance gates rather than
implicit implementation choices.

Relegation removes only active Big 12 membership. The cumulative Baseline
Library monotonically retains every current and former Big 12 method's:

- compatible implementation and official configuration;
- complete provenance and qualification record;
- designated retained checkpoints and its selected final system;
- interface and environment-version compatibility metadata; and
- historical Big 12 membership, dated snapshots, and results.

An authorized training workflow may consume compatible Baseline Library
material only through an immutable, content-addressed population manifest that
declares exact method/system/checkpoint identities and sampling weights. It may
not resolve a mutable `latest_big_12`, `current_champion`, or equivalent moving
alias. This includes population-based training and any other adaptive opponent
or curriculum selection. Use of a retained method for training does not restore
active Big 12 membership or create another tournament entrant.

Scenario pressure controllers, generic Scripted TDM, Random, privileged or
oracle policies, intermediate or otherwise unqualified checkpoints, and a
PSRO method's internal population members are not independent Big 12 entrants.
They do not receive Big 12 ratings merely because they support development,
training, or evaluation. If Qwen-Five is admitted, its approved artifacts and
history remain retained by the Baseline Library after relegation; using them in
a training population remains separately capability- and cost-gated rather
than assumed to fit the common JAX training loop.

All official Big 12 training, validation, selection, tournament, rating, and
weekly-governance evidence must retain canonical SharedObs provenance, exact
content identities, code and environment revisions, seeds, resource and
failure records, and the immutable manifests that joined them. These controls
support maintainer reproduction and eligibility assessment; they do not claim
to make scientific misconduct technically impossible.

## A28. Scenario pressure controllers in the DevClient

**Classification:** narrow developer-tool integration of A26's reactive
pressure controllers.
**Supersedes:** A22/A23's symmetric controller-choice envelope only to allow
explicit scenario-pressure choices on Team B. Existing Manual, Scripted TDM,
and Random choices remain symmetric. A25/A26's information, evaluation, and
scientific-integrity boundaries remain unchanged.

The DevClient now exposes **Scenario 1 Controller** on Team B only. It is a
deterministic, reactive SharedObs policy, not a fixed action tape or a generic
Scripted TDM alias. Team A cannot select it. NoSharedObs cannot execute it.
The controller consumes only canonical same-epoch SharedObs and each actor's
exact current mask. Both teams' actions still enter one ordinary simulator step.

Scenario 1's controller supports interactive TDM snapshots with one to five
remaining transitions and Mage/Rogue/Priest Team B decision-making slots.
Configured Warrior/Hunter slots must start dead and cannot revive before the
final successor. Compatibility is checked against the immutable initial
snapshot, not the asset name, ID, or revision. Renamed and copied compatible
scenarios work. Incompatible controller or scenario replacement fails without
changing the installed session; no controller or regime is silently substituted.

Saved scenarios remain controller-independent. Loading one does not install a
pressure controller automatically. Controller changes and Reset retain the
existing exact-snapshot and seed semantics. Policy-controlled agents remain
inspectable while their action inputs are read-only.

The controller has its own versioned, content-addressed rule identity bound to
the existing source revision. Diagnostic provenance distinguishes it from
generic Scripted TDM, Random, and manual input and records deterministic policy
execution. Existing controllers retain their previous content identities.
Replay wire contracts and Replay Viewer responsibilities are unchanged.

The checked-in Scenario 1 setup is a test-only physical fixture, not an official
evaluation-suite release. Neither DevClient play nor a diagnostic recording
automatically qualifies as official scenario evidence. Future evaluation
definitions bind the controller through `pressure_protocol` as specified by
A26; this amendment does not implement those evaluation pipelines.

Small reusable target-selection and movement helpers may support later
diagnostic policy development. Scenario-derived rules are not silently admitted
to official training, and performance on the originating scenario must not be
presented as uncontaminated evaluation of a policy engineered from it.

## A29. Unrestricted interactive Reactive MRP selection

**Classification:** narrow developer-tool availability and error-handling fix.
**Supersedes:** A28's TDM-only, one-to-five-transition, and restricted
Warrior/Hunter lifecycle requirements. Team B-only and SharedObs-only execution,
the versioned policy rules, and A25/A26's evaluation boundaries remain intact.

The DevClient's **Reactive MRP Controller** is available in every valid
interactive setup, including no-task diagnostic arenas, TDM map previews,
current authoring snapshots, and saved scenarios of any valid horizon or roster.
It is not a replacement for registered fixed-frame diagnostics. No scenario
name, score, remaining horizon, or respawn schedule gates its selection.

Mage, Rogue, and Priest use their existing deterministic reactive rules.
Warrior and Hunter retain the existing Stay/no-combat fallback, even when alive
or revived during play; their future rules require separate design approval.
The selector explains this limitation. The internal `scenario_1` identifier,
controller version, canonical descriptor, and action-source contract are
unchanged. Task provenance continues to identify the actual loaded task;
diagnostic arenas are not relabelled TDM.

Expected controller-configuration rejections return the existing nonfatal
unchanged-frame notice response. They preserve the installed session, pending
actions, revision, seed, generation, and recording progress without faulting
the service or requiring reconnect. This does not suppress unexpected policy,
simulation, capture, or presentation failures. Invalid protocol requests remain
rejected before command execution.

Controller changes and Reset retain exact initial-snapshot/seed behavior.
Loading another asset preserves installed controllers; no controller or regime
is silently substituted. Recorded-progress discard confirmation remains
explicit. Saved assets, schemas, Core, policy algorithms, and Replay Viewer
responsibilities are unchanged. Wider diagnostic availability neither qualifies
Scenario 3 nor extends the measured Scenario 1/2 witnesses to other setups.

## A30. Reactive TDM and specialist scenario controllers

**Classification:** explicit pre-alpha diagnostic-policy replacement and bounded
specialist-controller addition.
**Supersedes:** A13/A19's executable legacy TDM scorer, profile, trace, and
dual-adapter requirements; A22/A23's legacy live controller choices; and
A28/A29's separate executable Reactive MRP/Scenario 1 identity and Team B-only
general-controller restriction. A25's canonical SharedObs contract and
A26/A27's scientific-integrity, evaluation-role, and baseline boundaries remain
unchanged. Historical amendments and recorded evidence are preserved.

The old Scripted TDM scorer, profiles, traces, policy-specific SharedObs and
NoSharedObs adapters, exports, and dedicated algorithm tests are retired
completely. The separate Reactive MRP/Scenario 1 executable interface is also
retired after its useful small helpers move into the replacement. No executable
aliases, wrappers, dormant copies, or historical-policy dispatch remain.
Generic source composition, team execution, Random, dual-mode rollout, and
their causal, mask, RNG, capture, and replay proofs are retained. Fixed-frame
scripted diagnostics and future KOTH/CTF specifications are not legacy TDM
policy code and are not removed.

**Reactive TDM** is one deterministic, team-agnostic five-class policy, available
on either DevClient team under SharedObs. **Scenario 3 Controller** is a separate
deterministic Rogue specialist, available on Team B under SharedObs; its other
classes Stay/no-combat, including after revival. Both work in every valid
interactive setup without asset-name, task, horizon, roster, or respawn gates.
The general controller contains no scenario-ID branch. Specialist behavior is
an explicit small policy, not a controller registry, planner, behavior tree,
plugin, or saved-policy format.

Both policies consume only authorized current SharedObs and each recipient's
exact mask. They accept but ignore actor keys. Eligible targets are observed,
active, living, positive-health units. Movement and combat select independently;
Ultimate replaces Basic for the transition. Health ties use global slot, except
Priest health ties first use maximum health. Distance ties use global slot;
movement ties use the existing movement-action order. No raw state, pending
actions, history, future information, or scenario identity enters a decision.
All actors use the same epoch, with at most one source bank, one joint-action
assembly, and one unchanged simulator step. Exact masks and lifecycle no-ops
override all preferences.

Reactive TDM's class rules use center distances and existing float32 conventions:

| Class | Movement with observed enemies | Combat priority |
| --- | --- | --- |
| Priest | Select lowest-health ally including self; approach a nonself selected ally beyond 3, otherwise retreat from nearest enemy | Lowest-health legal Ultimate ally at HP ≤30; otherwise lowest-health legal Basic ally, including self/full-health allies |
| Mage | Nearest enemy: approach above 2, retreat below 2, Stay at exactly 2 | Legal Burst when a legal Basic enemy exists; otherwise lowest-health legal Basic enemy |
| Rogue | Approach nearest enemy center, including at body contact | Lowest-health legal Ultimate enemy; otherwise lowest-health legal Basic enemy |
| Hunter | Nearest enemy: retreat at/below 3, Stay in (3, 3.5], approach above 3.5 | Nearest legal Trap enemy within distance ≤2; otherwise lowest-health legal Basic enemy |
| Warrior | Approach nearest enemy center, including at body contact | Lowest-health legal Charge enemy with HP strictly <40; otherwise lowest-health legal Basic enemy |

Without observed enemies, Mage/Rogue/Hunter/Warrior move toward map center.
Priest instead follows the nearest observed living ally excluding self:
retreat at/below 1.5, Stay in (1.5, 2], and approach above 2. A Priest without
another observed living ally also moves toward map center. This peaceful band
does not change Priest's enemy-visible threshold of 3. No legal combat target
means no-combat. Exact center or zero movement intention means Stay.

General movement retains the existing eight-direction alignment ranking,
obstacle/bounds projection, inclusive 10% useful-stride threshold, and fixed
tie order, without estimated body interference. Clipping, sliding, contact,
and desired oscillation remain legal. These are local intentions, not final
distance guarantees; map-center rendezvous is not pathfinding or guaranteed
exploration. Core remains authoritative for every actual collision.

Scenario 3's Rogue pursues the lowest-health observed living enemy, recomputed
each epoch. It independently uses Ultimate, then Basic, on the lowest-health
legal enemy within Basic interaction radius. Without an observed enemy it
Stays/no-combat. Its separate movement helper evaluates eight legal static-world
projected moves retaining at least 10% stride. It screens each entire straight
segment against other observed living body discs, excluding self and the
selected prey, using summed radii with tangency allowed and no added margin.
An already overlapping blocker permits only non-deepening movement ending
farther from that blocker. The safe endpoint nearest the prey wins, with fixed
action-order ties; the best move may temporarily increase prey distance.
Stay is the fallback only when none qualifies, not a competing endpoint.
This stationary-body estimate does not predict opposing actions, guarantee
navigation against a moving blocker, or alter general Reactive TDM movement.

Private live controller values become `manual`, `reactive_tdm`, `random_valid`,
plus Team B's `scenario_3`. Obsolete live values fail strict validation rather
than silently selecting a replacement. NoSharedObs remains available for
Manual/Random and generic custom research, but neither reactive policy receives
a NoSharedObs adapter. Selectors do not silently change information mode.
Expected configuration rejection remains nonfatal and preserves the current
session, pending actions, keys, generation, revision, and recording progress.
Reset and controller changes preserve exact-snapshot/seed semantics; loading
an asset preserves the installed choices and never installs a controller.

New policy identities are `reactive-team-deathmatch-controller@1` and
`scenario-3-pressure-controller@1`, with fresh canonical descriptors bound to
the launch-captured source revision. New interactive action-source payloads use
V4 and identify both controllers separately from the loaded map/scenario.
Manual/manual is `manual`, exactly one manual team is `mixed`, and fully
policy-controlled interactive execution is `policy`. Fixed-frame diagnostics
retain their existing `scripted` payload. Existing replay formats, artifacts,
and readers remain unchanged; historical policy identifiers do not require
retaining their executable implementations. New interactive digests change
without changing Random's keys or actions.

Both reactive policies and Random remain diagnostic/scenario-pressure tooling,
not official baselines, Big 12 entrants, or Baseline Library members. An official
evaluation must separately freeze and bind the chosen controller through
`pressure_protocol`, identically for treatment and matched ablation. Saved
scenarios remain controller-independent. Scenario 3's physical regression
fixture and development-stage notes do not release or qualify an official
scenario or impose a five-transition horizon. Scenario-derived rules must not
silently enter official training, and their originating scenario results must
not be presented as uncontaminated evaluation. Core and public evaluation-role
contracts are unchanged.

## A31. Scenario 5 Priest-pursuit controller

**Classification:** bounded diagnostic/scenario-pressure controller addition.
**Extends:** A30's explicit Team B specialist choices and private live action-source
contract. Existing Reactive TDM/Scenario 3 behavior, historical amendments,
A25's information contract and A26/A27's evaluation-integrity boundaries remain.

**Scenario 5 Controller** (`scenario_5`) is available only on Team B under
SharedObs in every valid interactive setup, without asset, task, horizon,
roster or respawn restrictions. Mage, Warrior, Hunter and Priest delegate to
the unchanged Reactive TDM policy. Active living Rogues pursue the observed
living positive-health enemy Priest with lowest current HP, breaking ties by
global slot. The target is recomputed every epoch, without memory or hidden
positions. If no Priest is observed, movement falls back to ordinary Reactive
TDM: nearest observed enemy, otherwise map center, with its ordinary static-world
refinement rather than specialist body avoidance.

While pursuing Priest, Rogue reuses Scenario 3's exact body-aware movement:
eight legal static-world projected candidates, inclusive 10% useful stride,
full displacement-segment screening against stationary observed living bodies,
summed radii, tangency allowed, and no added clearance margin. Self and the
selected Priest are exempt. Initial overlaps permit non-deepening outward
escape. Choose the safe moving endpoint closest to Priest with action-order
ties, allowing temporary retreat; otherwise Stay. This is neither a planner nor
a guarantee against moving blockers. General Reactive TDM steering is unchanged.

Combat selects independently within Basic interaction radius: lowest-current-HP
legal Ultimate enemy, otherwise lowest-current-HP legal Basic enemy, otherwise
no-combat. Health ties use global slot. Rogue can attack a blocker while moving
toward Priest. Ultimate replaces Basic; exact recipient masks and lifecycle
no-ops override preferences. The actor key is ignored. All decisions use current
authorized SharedObs, at most one bank, one assembler and one unchanged step;
no raw state, pending actions, history or successor data is consumed.

The algorithm identity is `scenario-5-pressure-controller@1`, deterministic,
with a fresh descriptor covering inherited rules, pursuit, combat and fallbacks,
bound to the launch-captured source revision. All configured-active Team B rows
identify their installed policy as `scenario_5`, including non-Rogues. Manual
versus Scenario 5 reports `mixed`; fully policy-controlled pairs report `policy`.
Private action-source V5 is used only for Scenario 5 combinations and includes
truthful execution identification; existing V4 combinations and fixed-frame V1
payloads are preserved. Replay schemas/readers and historical artifacts need
no changes. No checkpoint or training claim is made.

Selection remains authoritative-frame-confirmed; incompatible requests fail
without damaging session health or pending state. Loading never installs a
controller automatically. Reset/load preserve the chosen controller pair and
exact snapshot/seed. Team B actions remain read-only while inspectable.

Saved scenarios remain controller-independent. Development notes and diagnostic
traces do not establish an official suite release or a winning Team A solution.
Official use still requires a separately frozen evaluation definition and the
same pressure controller across matched treatments; scenario-derived behavior
must not silently influence official training. No Core, public evaluation-role,
policy-registry, Replay behavior or physical asset change is authorized here.

## A32. Scenario 5 shoulder bypass and fallback prey

**Classification:** bounded diagnostic/scenario-pressure behavior revision.
**Revises:** A31's Scenario 5 pursuit and body-contact preference. A1–A31 remain
the historical record; A25's information contract and A26/A27's evaluation
boundaries continue to apply. Scenario 3 retains its strict movement preference
and behavior v1. General Reactive TDM, Core physics, physical assets, action
masks, actor inputs, and Replay schemas/readers are unchanged.

Scenario 5 (`scenario_5`) remains Team B/SharedObs-only. Active living Rogues
choose an observed active living positive-health enemy Priest first, otherwise
a Hunter. Within the selected class, choose lowest current HP, then lowest
global slot. Recompute every decision. If neither class supplies a candidate,
use ordinary Reactive TDM Rogue movement toward the nearest observed enemy or
map center. Non-Rogues still delegate unchanged to Reactive TDM.

While pursuing either prey class, examine the same eight legal static-world
projected moves with the inclusive 10% useful-stride threshold. Screen each
projected displacement segment against currently observed living bodies, with
self and selected prey excluded and full physical radii retained. Admit a clear
path or glancing contact at least 45 degrees from the inward normal at first
contact with every contacted blocker. As in strict screening, tangency or an
endpoint that merely reaches a body without entering it counts as clear.
An inward head-on stride from existing contact does not qualify, however short.
Initial overlap within the existing geometry tolerance counts as contact;
deeper initial overlap requires non-deepening outward motion ending farther
from the blocker. Choose the admissible endpoint closest to prey, with
movement-action-order ties and no preference for contact-free detours. Temporary
retreat remains allowed; without a useful admissible move, Stay.

This is a local steering preference. Ordinary simulator collision handling
resolves attempted contact and can displace either body. The controller treats
observed bodies as stationary for its estimate; it does not inspect pending
opponent actions, simulate a successor, remember a route, or guarantee bypass
against moving or pinned defenders. Scenario 3 continues to use its existing
strict clearance rule through the shared helper's default behavior.

Rogue combat remains independent of pursuit: within Basic interaction radius,
choose the lowest-health legal Ultimate enemy, otherwise the lowest-health
legal Basic enemy, otherwise no-combat, with global-slot ties. Exact masks,
Ultimate priority, and lifecycle no-ops remain authoritative. Current authorized
observation/mask and one shared sensor bank feed precommitted simultaneous
actions, one joint-action assembly, and one unchanged simulator transition;
the next observation supplies the next decision. The actor key remains ignored.

The installed identity is `scenario-5-pressure-controller@2`. The fresh behavior
descriptor records prey priority, contact angle, overlap handling, selection,
combat, inherited rules and fallbacks, and remains bound to the launch-captured
source revision. The existing private V5 action-source payload carries this
identity without a schema revision. Scenario 3/Reactive TDM algorithm IDs and
behavior versions, other interactive V4 contracts, fixed-frame V1 payloads,
and historical recordings remain unchanged.

Qualification requires focused contact/overlap and prey-selection regressions,
actual shoulder-contact trajectories through the public simulator, unchanged
Scenario 3/Reactive TDM proof, version/digest and recording-reopen checks, and
bounded policy-cost evidence. Diagnostic progress or a finishing attack does
not establish an official scenario win or release. Official use still needs a
separately frozen evaluation definition and identical controller identity across
matched treatments. No saved scenario automatically selects this controller.

## A33. One controller for Scenarios 3 and 5

**Classification:** controller retirement and shared diagnostic usage.
**Revises:** A30–A32's forward requirement to retain a separate executable
Scenario 3 controller. A1–A32 and their measured historical evidence remain
intact. This is not a change to the surviving Scenario 5 algorithm.

DevClient exposes one **Scenario 3 and 5 Controller** on Team B under SharedObs.
It uses existing `scenario_5`, `scenario-5-pressure-controller` behavior version
2 unchanged: Rogue pursues observed living Priest first, otherwise Hunter,
with same-epoch HP/slot ties and shoulder bypass. Without either class, movement
falls back to ordinary Reactive TDM. Combat independently chooses the
lowest-health legal enemy within Basic radius, Ultimate before Basic.
Non-Rogues use ordinary Reactive TDM, including after revival. The actor key
remains ignored. General Reactive TDM, Random and Core are unchanged.

Remove the standalone Scenario 3 executable, live selector, dispatch, unused
strict-steering branch and retired-only tests. Retain no aliases or historical
executable copies. Preserve useful generic regression proof, physical test
fixtures, historical documents and existing replay bytes. Retired live
`scenario_3` requests fail ordinary validation without faulting the session;
historical replay provenance remains readable without executing the old policy.

The surviving controller literal, algorithm identity, descriptor version and
private V5 payload remain unchanged. Preserve surviving V4/V5 payload fields;
`scenario_3_execution_included` is fixed false compatibility metadata, not a
live controller. No new wire format, registry, schema or policy alias is added.
Launch-source provenance still identifies the actual code revision.

Saved scenarios remain controller-independent. Update their prose using normal
revision-fenced saves without changing physical content or historical bytes.
Earlier Scenario 3 results remain tied to the retired opponent and are not
claims of success against the shared controller. New usage is diagnostic until
separately qualified; official evaluation still freezes one exact pressure
protocol across matched treatments. Core, public evaluation roles, Replay
behavior, scenario physical semantics and A25's SharedObs contract are unchanged.

## A34. Reactive TDM ALPHA and BETA

**Classification:** diagnostic presentation and narrow Rogue pursuit revision.
**Revises:** A33's current display names and two-class pursuit order only.
Historical amendments, saved scenarios and replay evidence remain unchanged.

DevClient displays the existing general `reactive_tdm` controller as
**Reactive TDM ALPHA**, on either team. The surviving `scenario_5` variant is
displayed as **Reactive TDM BETA**, still Team B-only. Both require SharedObs;
there is no new registry, controller literal, observation contract or schema.

The general module and callable names become `reactive_tdm_alpha.py`,
`reactive_tdm_alpha_policy` and `reactive_tdm_alpha_controller_descriptor`;
this is a behavior-preserving rename. The variant uses `reactive_tdm_beta.py`,
`reactive_tdm_beta_policy` and `reactive_tdm_beta_controller_descriptor`.
Update imports directly; retain no `reactive_tdm.py` or `scenario_5.py` alias.
Stable persisted/live identity strings remain separate from Python module names.

BETA behavior version 3 selects observed, active, living, positive-health enemy
Priests first, otherwise Mages, otherwise Hunters. Within the selected class,
lowest current health wins, then lowest global slot. Recompute at every current
decision epoch. Use the existing glancing shoulder-bypass movement unchanged
for all three prey classes, exempting self and selected prey from body screening.
Without any priority prey, Rogue movement falls back to ALPHA: nearest observed
enemy, otherwise map center, with ordinary obstacle refinement.

Combat remains independent of pursuit: lowest-health legal Ultimate enemy
within Basic radius, otherwise lowest-health legal Basic enemy there, otherwise
no-combat. This radius check remains even during the movement fallback; BETA
does not simply return the entire ALPHA Rogue action when prey is absent.
Every non-Rogue still delegates directly to unchanged ALPHA. Body geometry,
movement masks, action ordering, RNG handling and simulator execution do not
change. The new priority neither implies access to hidden prey nor guarantees
a route past a moving defender.

The algorithm identity remains `scenario-5-pressure-controller`; its fresh
descriptor advances to version 3 and records the three-class order. Existing
V5 provenance carries the updated descriptor/source-bound digest. ALPHA's
algorithm/version and all prior recording bytes remain unchanged. Old versions
remain historical evidence, not assertions about the newly selected controller.

ALPHA and BETA remain diagnostic/scenario-pressure tools, not official baselines
or Big 12 entrants. A recorded full match is inspectable diagnostic evidence,
not a tournament comparison or scenario qualification. Official scenario use
still freezes one exact pressure-controller identity across matched treatments.

## A35. Reactive TDM wall steering

**Classification:** diagnostic navigation improvement with an accepted residual
limitation. **Revises:** A30/A34's forward obstacle-refinement behavior and
controller versions only; historical amendments and recordings remain intact.

ALPHA advances to behavior version 2 and BETA to version 4. Class movement
goals, retreat/Stay intentions, spacing bands, combat priorities, authoritative
masks and SharedObs inputs remain unchanged. Only approach refinement changes:
near a qualifying axis-aligned vertical wall, prefer its SOUTH end, otherwise
NORTH when the lower passage cannot fit. Equivalent quarter turns qualify;
pillars, horizontal walls and genuinely angled walls are not steering anchors.
Use body-sized geometric strips, nearest eligible wall-center/slot selection,
projected corner progress and far-face release. ALPHA checks static exit
clearance and retains positive partial phase progress if no full useful phase
move exists. It does not predict body interference.

BETA's priority-pursuing Rogue retains unchanged 45-degree shoulder screening
and self/prey exemptions. Its local wall envelope includes directly overlapping
body-expanded vertical walls, not a transitive route graph. Already cleared
groups are excluded before anchor selection. Prey at or beyond the expanded
East face selects East (with existing geometry tolerance); otherwise West,
including prey inside the group. Boundary-width passages account for currently
observed blockers, while an admissible SOUTH corner move can establish progress.
Prefer phase-aligned useful moves, then prey distance and action order. With a
fitting end but no preferred move, restore the existing body-admissible
prey-distance choice, including during corner crossing. Neither fitting end
means Stay; screening is never relaxed. Non-Rogues and ordinary Rogue movement
fallback continue delegating to ALPHA.

All estimates use current authorized inputs, eight moving candidates and the
existing 10% useful-static-displacement threshold. No route memory, lookahead,
opponent-action prediction or simulator invocation enters a policy. Actions
remain simultaneous: current observations/masks and one SharedObs bank feed
actor choices, one joint assembly and one ordinary simulator transition.
Core, body radii, movement speeds, Replay schemas/behavior and saved assets do
not change. Stable controller literals, algorithm IDs and V4/V5 recording
structures remain; source-bound descriptors identify the new behavior versions.

This is a user-tested partial improvement, not a deadlock-freedom guarantee.
Matched diagnostic windows resolve the tested BETA pockets and captured
scenario decisions/transitions remain exact. A three-decision ALPHA allied-body
congestion stall remains known; no complete final-version Foxhole rollout or
measured speedup is claimed. Fixed-side choices may take longer routes, moving
defenders can re-block, and other local jams remain possible. The user accepts
these limits for diagnostic use. (Superseded, 22 September 2026: this assessment
of ALPHA version 2 and BETA version 4 is historical only; later study found
longer stalls.) Official scenario evidence must still freeze
and separately qualify its exact controller identity; this change neither
completes M7 nor qualifies a scientific scenario.

## A36. Submission roadmap, approved TDM content and M7 closeout

**Authority:** explicit executive decisions from the user on 2026-09-07.
**Classification:** approved roadmap/scope override and developer-tool contract.
**Supersedes:** the historical PDF's remaining milestone ordering and scenario
count; A26's twelve-scenario target and uniform five-transition horizon;
private plans requiring further scenario/map design approval or Scenario 3
re-approval; and the provisional post-manuscript placement of LLM integration.
This amendment records decisions and planned work, not implementation completion.

### Current milestone names and numbering

| Current milestone | Name | Historical milestone |
| --- | --- | --- |
| M7 | TDM Benchmark and Researcher Tools | M7 |
| M8 | Policy Execution and Evaluation | M10 |
| M9 | Training Distributions and Curriculum | M11 |
| M10 | Learning Platform and Baselines | M12 |
| M11 | LLM-Agent Integration | M14 |
| M12 | Manuscript Experiments and Release | M13 |
| M13 | King of the Hill | M8 |
| M14 | Capture the Flag | M9 |

The submission sequence is M7 through M12. M13 and M14 begin after manuscript
submission and the optimization audit below. Milestones 1–6 retain their identities.
The current M12 owns full manuscript training runs, behavioral ablations, the frozen Paper 1 tournament,
analysis and release artifacts. Earlier milestones qualify their machinery
with focused tests and bounded pilots; protocols and selection rules are
frozen before the dependent full runs.

Historical PDF/amendment references, commit history, private filenames and
versioned schema identifiers are not mechanically renumbered. Their old numbers
are aliases under this table. New plans use the current name/number and give
the historical alias when needed to disambiguate a source. Existing private
handoffs remain at their original paths with an explicit current-name notice.

### Post-manuscript optimization audit

The user's 2026-09-08 decision requires a dedicated optimization phase after
manuscript completion and before KOTH/CTF implementation. Profile representative
training, validation, evaluation, metrics, replay and persistence workloads across
realistic batch sizes and rollout lengths. Examine runtime, peak VRAM/RAM, data
transfer, allocation and disk costs; remove repeated computation and serialization.
Address every identified material improvement and verify unchanged results with
before/after measurements. Close the audit only when no identified material
optimization remains unresolved for the declared workloads. Keep researcher
interfaces simple and require explicit approval for concrete Core changes.

The ambition is to make MARL-BGs one of the most efficient MARL benchmarks.
Substantiate comparisons with equivalent workloads, declared hardware/precision,
warm and compilation timings, memory measurements and preserved semantics. This
phase does not renumber milestones or defer optimization during active work.

### Approved scenarios and maps

The TDM suite is exactly Scenarios 1–8. Scenario authoring and design approval
are complete; no Scenarios 9–12 or extra winning-witness approval gates are
required for M7. The user confirms Scenario 3 r24 has been reviewed and
accepted. Its ten-transition horizon is the sole suite exception, intentionally
testing sustained body blocking. The respawn-wave period is five, not ten;
an authored current countdown of four is compatible with that period and must
not be changed merely to match its name. Stale r23/twelve-wall/review-pending
prose must be corrected without silently changing approved physical content.
Do not describe the user's acceptance as a newly executed Codex witness.

All forty base maps and twelve curriculum maps are approved. No additional
map-design approval, general balance campaign or reachability re-approval is
introduced by M7 closeout. A read-only vertical-reflection audit is the sole
additional map approval check the user permits. Exact revision/digest capture,
loading validation and manifest packaging implement those existing decisions;
they do not reopen approval or authorize physical asset edits.

Canonical evaluation uses mirrored five-class 5v5 teams: Mage faces Mage,
Warrior faces Warrior, Hunter faces Hunter, Rogue faces Rogue and Priest faces
Priest at reflected same-class starting pads. Reflection is about the vertical
map centerline (x = width/2 in the stored coordinate system). Preserve the
existing paired side assignments. Geometric symmetry supports equal starting
geometry; it does not by itself prove every numerical or policy behavior is
side-invariant.

At the earlier M7 closeout, the permitted audit passed all 52 maps: exact
authored obstacle reflection and exact opposing same-slot spawn pads. The user
corrected Darkspear in r4, moving only `obstacle_6` from x=14.5 to x=13.5 to
reflect `obstacle_1` at x=6.5.
All other audited map bytes were unchanged at that checkpoint. Compiled geometry
agreed within 1e-5 world units after float32 conversion (largest discrepancy
approximately 3.09e-8). Scenario 3 r25 is a prose-only successor to the accepted r24; its
physical semantic digest is unchanged. These provenance updates implement the
existing approvals and do not reopen them. This is historical evidence; the map
revision selection below supersedes it for new games.

### Approved map publication on 2026-09-16

**Status:** published locally and verified on 2026-09-16; uncommitted. All 137
focused GPU tests passed, including the nine winning scenario lines. A separate
GPU batch checked reset and movement on all 52 maps. Two exports produced
identical bytes. Installed-wheel checks passed outside the repository, including
current map loading, retained history and rejection of stale catalog entries.

Publish the exact audited set: 39 maps have physical changes, Map 3 changes only
obstacle IDs, and 12 maps are unchanged. Keep the 52 map IDs, names, splits,
aliases and scenario content. The packaged
[manifest](../../src/marl_battlegrounds/data/tdm/manifest.json) records the selected
saved revision, source hash and compiled resource hash for each map. New games
must use that selection, not a moving latest-draft lookup.

The repeated source audit passed exact vertical reflection and consecutive
obstacle IDs `0` through `N - 1` on all 52 maps. The clearance check measures
the shortest distance between shape edges in map units, including rotated
rectangles. It uses these approved limits:

- A rectangle's open gap to a boundary wall must be at least 1.095.
- A circle's open gap to a boundary wall must be at least 1.05. The user accepts
  1.04999 when it rounds to 1.05.
- A gap of at most 0.01 to a boundary wall counts as contact with that wall.
  Check each wall separately; touching one wall does not excuse another gap.
- A positive gap between two obstacles must be at least 1.05. Touching or
  overlapping obstacles and the exact exceptions below are allowed.

The following exceptions apply only to the named map, obstacle and wall or
pair. Numbers refer to the audited obstacle IDs, not arbitrary later row order.
An exception does not approve a wider set of gaps.

| Map ID | Boundary-wall exceptions | Obstacle-pair exceptions |
| --- | --- | --- |
| 11 | 23 and 24: bottom; 25 and 26: top | None |
| 12 | None | (6, 8), (9, 11), (17, 20), (18, 19) |
| 13 | 10 and 17: bottom | None |
| 31 | 4 and 7: bottom | None |
| 40 | 22: bottom; 23: top | None |
| 42 | 15, 16, 17 and 18: top | None |
| 47 | 26: bottom; 27: top | (1, 31), (3, 30) |
| 49 | 17 and 18: top; 19 and 20: bottom | None |

GPU clearance checks of the repaired Maps 21, 36 and 48 reported minimum
non-exempt gaps of approximately 1.142892647, 1.099999905 and 1.216316939,
respectively. These measurements and the source audit support the declared
geometry checks. They do not prove every crowded route is traversable or
requalify solver performance on the changed maps.

Retain prior packaged map identities in one immutable bundled
[history resource](../../src/marl_battlegrounds/data/tdm/map_history.json).
Old replays must keep their recorded layout, revision and identity; a new map
selection must not relabel them as the new geometry. History supports reading
old records and does not change the current map catalog used for new games.
Restart running DevClient, Replay Viewer and Python processes after updating
the package so their cached catalog is refreshed. Restarting does not change
an already recorded game.

Earlier replays, solver tests and speed measurements remain evidence for their
recorded map versions. Do not present them as measurements of these revised
maps. This publication does not change the approved eight scenarios, their
embedded geometry or their winning-command witnesses.

### UFO map update on 2026-09-22

Map 39, `tdm_map_id_39_ufo_training`, now uses saved revision 7. Revision 7
removes one obstacle from revision 6: the thin wall that ran from the centre
circle at (10, 6) up to the circle at (10, 8.45). Every other obstacle keeps
its shape and position. The obstacles after the removed wall are renumbered,
so the map has 22 obstacles with IDs 0 through 21. The map keeps its ID, name,
training split and mirror symmetry. The manifest records revision 7, its
source hash and its compiled resource hash. The other 51 maps, the aliases,
the history resource and the eight scenarios are unchanged.

Why: with that wall in place, the pocket between it and the left slanted wall
closed toward the centre. The scripted controllers (seen with ALPHA version 2,
BETA version 4 and a candidate revision of them) do not steer around other
bodies, so the leading agent was pinned in that corner by the teammates behind
it, and six agents stood still for whole games. With the wall removed, the
agents go round the centre circle.

The clearance table above no longer lists Map 39. Its two exceptions were the
gaps between the removed wall and the two slanted walls. Removing an obstacle
changes no other gap, so the remaining obstacles meet the limits with no
exception.

This update does not add revision 6 to the history resource, which keeps one
earlier version per map; for Map 39 that is revision 3. This is an accepted
exception to the rule above that old replays keep their recorded version.
Replay files recorded on revision 6 of Map 39 still pass the replay loader,
but the Replay Viewer cannot present them, and every check of a recorded map's
identity rejects them. Training checkpoints saved before this update cannot
resume, because their saved content binding covers every map. Results from
revision 6 remain evidence for that version only.

### Final M7 DevClient and Replay Viewer work

- DevClient has a task selector containing TDM only. Replay Viewer derives
  task identity from the recorded configuration.
- Both display authoritative current scores and configured threshold, Team A
  in blue and Team B in red, plus truthful controller/model identities. At
  authoritative completion, each team's label is VICTORY (green), DRAW (light
  gray) or DEFEAT (dark red). This applies to scenarios too; an unfinished or
  interrupted recording is not a completed loss or draw.
- Replay Viewer gets a TDM evaluation-metrics disclosure immediately below
  Comprehensive Agent Class Details and a usable CSV export. Keep canonical
  JSON provenance available. Show all applicable implemented episode metrics,
  with explicit pending, unavailable, conditional and zero-opportunity states.
  Do not invent single-replay population/learning/rating values or activate
  rejected/pending scientific metrics merely to fill the panel.
- Prefer a single host analysis pass with cached cursor-prefix summaries and
  a separate final report. Prefix values consume no future frames; complete-only
  results remain pending until completion. Measure preparation, memory and
  lookup cost before claiming cheap progressive inspection. DevClient has no
  full metric panel and does not run the full suite every simulation tick.
- Initially enable exactly Ultimate Ability Effects, Spawn Shield, Basic
  Ability Effects, Regeneration Effects, Death Effects, Resurrection Effects,
  Scrolling Battle Text and Respawn Wave. Initially open both Visual Filters
  and Roster. Preserve the existing filter meanings and controls; Enable All
  must still enable every filter.
- Oracle and Agent POV switching must work at any replay cursor without a
  crash/reconnect or stale-authority leakage. Preserve the durable timestep and
  subsequent playback behavior.
- Use one visual/layout authority for status overflow. When one status remains,
  display that status and its duration rather than `+1`. For two or more, use
  the existing readable solid gray, black-filled, centered `+N` badge. Preserve
  collision handling, owner association, authorized visibility and accessibility.

Implementation should favor small, cohesive extensions to existing authorities.
The metrics panel is researcher analysis, not actor input; changing battlefield
POV does not grant a policy access to it or to Oracle facts.

### Metric computation and training persistence

The user's subsequent 2026-09-07 scalability decision removes repeated deep
validation of trusted computed metric state and final reports. The tested
reducer pipeline owns its formulas and immutable output contract. Runtime keeps
cheap type/identity, progress, eligibility, row uniqueness and provenance checks;
external artifact ingestion and independent semantic tests remain authoritative
verification boundaries. Do not retain a second expensive audit-mode pipeline.
Completion and failure metadata remain explicit; this decision changes cost,
not scientific metric definitions or actor information.

Metric computation returns data without automatically writing files. Current
M10 training defaults to lightweight episode statistics. Full evaluation metrics
are off by default throughout the metric APIs and ordinary recording. The
critical default set is outcome distribution, terminal score differential,
evaluation return, episode length, and completion/failure metadata. Absolute
scores retain their existing replay/scoreboard authority. Detailed TDM metrics
require explicit opt-in, including requests for Replay Viewer detailed analysis.
Periodic diagnostics and validation-map evaluation must reuse the same evaluator.
The proposed downstream interface is one interval, a selected validation-map set,
a match count, and critical/full metrics. The 2026-09-12 catalogue reorder
replaces public validation IDs 24, 29, 30, 32 and 37 with 42–46. Numbered map names,
authored IDs and folders follow the new IDs. Geometry and split membership stay
unchanged; frozen manifests remain the authority. The
user requests measured costs before finalizing cadence/API details, so this is
a provisional handoff, not an activated scheduler. Log any eventual schedule and
selected episode identities.
Replay recording
is independently opt-in or sampled. Training logging appends batches to one
CSV per run and flushes periodically, avoiding both per-episode file proliferation
and an end-only write that loses the entire run on interruption. Ladder evaluation
retains one results CSV with match identities and sufficient counts/sums, plus
running summaries over the declared N matches grouped by matchup, roster/map and
side. Pool ratio numerators and opportunities; do not average percentages with
unequal denominators or retain only an irreversible grand mean. Preserve failure,
draw and exclusion counts and sufficient evidence for the declared uncertainty
analysis. M7 provides
reusable computation; the training control and writer belong to the learning
platform. Measure computation, artifact loading, replay indexing and persistence
separately. Loading UI must describe actual work, without claiming that removed
validation establishes trust or normalizing a minute-long metrics budget.

### Public ladder display

The public ladder displays Elo, win/loss/draw percentages, matches played and
the identities of the evaluated systems. It does not broadcast the full tactical
metric suite. Reproducing researchers may explicitly enable full diagnostics
when running the ladder locally. The user prefers one secondary team K/D column: total tournament kills divided
by total tournament deaths, with no-death results displayed as unavailable.
Retain raw totals internally; do not average per-match ratios or use K/D as an
additional rating input. This narrow team-level display supersedes earlier
broad K/D rejection; individual killer attribution and agent K/D remain rejected.
TDM team totals use authoritative score increments
or recorded death counts, respecting any nonzero initial score. Selected
manuscript analyses remain a separate reporting surface.

### Implementation boundary

M7's planned sweep includes explicitly identified host/evaluation, transport,
presentation, browser, test and documentation code. It does not authorize changes
under `src/marl_battlegrounds/core/`. If a missing fact or defect requires Core
work, Codex must stop the affected slice and ask for express approval, naming
the exact file/function, reason, behavioral impact and proposed scope. A host
reimplementation of simulator rules is not an acceptable way around that gate.
No Core change is currently planned. Existing action/observation, policy,
controller and simulator semantics remain authoritative.

## A37. Relative Policy Identity And Versioned Recordings

**Accepted change — 2026-09-12.** This supersedes earlier clauses that expose a
hard team ID or global slot in current policy inputs. Simulator routing,
configuration, transition facts and recording metadata retain their existing
identities. This amendment does not alter collision, combat, masks, action
meanings, rewards, random-key assignment or termination.

The approved Core scope is limited to `core/types.py` and the observation
builders in `core/env.py`. Feature column 3 becomes `AGENT_FEATURE_IS_ENEMY`:
self and allies have zero, visible enemies have one, and the existing visibility
mask still clears hidden rows. The feature width remains 58 and every other
column keeps its meaning. `Observation.self_ally_index` is an `int32` local row
0–4 for active actors, stable through death and respawn; inactive rows use zero.
Activity masks distinguish padding. `self_features` remains the normal way a
shared network conditions on its own class and state. The local index supports
row lookup and is not automatically added as a network feature.

`SpawnLifecycleObservation` is unchanged. Its own-team/opponent groups and
existing active/alive/visibility information already distinguish hidden units
and padding. Absolute world positions and map/spawn geometry retain their
existing visibility contracts; removing simulator labels does not hide physical
location or promise that a method cannot learn a side preference.

Current `ActorInput` contains observation, source bank and source availability.
SharedObs callbacks take five arguments: observation, action mask, key, bank and
availability. NoSharedObs retains three arguments: observation, mask and key.
No policy receives a global slot or hard team ID from these adapters.

`SharedObsSensorSourceBankV2` gives each actor features `(5, 10, 58)`, visibility
`(5, 10)` and objectives `(5, 8, 12)`, with availability `(5,)`. Sources are the
five stable own-team positions; candidates are five allies followed by five
enemies. Self is an unavailable shared source because its own view is supplied
separately. Build team source data once from existing relative rows, then mask
all unavailable material per recipient. Preserve authorized subsets, dead-source
redaction, own-view precedence and lowest-source selection. Store compact
observations during rollouts, not expanded actor banks. The recorder's existing
global availability matrix remains metadata used to reconstruct local inputs.

Changed payloads have explicit versions: base observation/frame V2, episode
context V3, replay header/artifact/reference V3, SharedObs projection V2,
NoSharedObs projection V3 and actor-POV V2. Scenario evaluation V4 binds the new
replay reference while reusing unchanged scenario definitions and predicates.
Other unchanged records retain their versions. Old V1/V2 readers preserve the
original team-ID column, global source layout, identities and bytes. Current
capture records actual delivered observation values, with no translation back
to pretend historical inputs. Viewer ownership comes from roster metadata.
Reject incompatible version combinations and changed-contract resumes.

Ordinary evaluation continues to assign the first policy to simulator Team A.
Team B training and diagnostic use remain supported. Raw team and agent metrics
retain their physical identities. Broader tournament rules remain unresolved.
[A38](#a38-neutral-collision-handling) supersedes the earlier decision to defer
collision ordering work until after training the 12 policies. Equal real
training steps at both spawn ends do not themselves prove slot fairness.

Acceptance requires exact preserved results for the two saved 2,000-game GPU
schedules, separate matched-shape action/state/input comparisons, historical and
current recording tests, and measured computation/storage costs. Source review
alone cannot establish speed, sample efficiency, learned behavior or fairness.

Addition of 2026-09-21: the recurrent MAPPO baseline gains an optional
spawn frame, `PPOConfig.spawn_frame` with values "world" (default), "left" and
"right". In a left or right frame the baseline reflects its own team's permitted
view about the vertical map centerline whenever that team starts on the other
bank and reflects the chosen move back before submitting it. This is an input
convention inside the researcher's System, offered through optional public
helpers beside the team-view builder; Core observations keep the world frame
exactly as this amendment states, and the environment, evaluator and tournament
apply nothing. The frame is saved with exported actors and is part of their
inference identity, so equal weights played in different frames are different
policies; a saved frame is restored, never inferred. The reflection is exact
geometry on the left-right symmetric maps, and because an obstacle row whose
mirror image is already in the table stays as authored, both spawn ends produce
the same encoded view there; it does not claim that reflected games reproduce
outcomes bit for bit, which the reflection diagnostics recorded elsewhere in
this document show they do not.

Clarification of 22 September 2026: "left" is now the default and "right" is
removed. A saved checkpoint, exported actor or run record without a frame still
means "world", and resuming such a run keeps it; an old training config file
started as a new run takes the new default unless it names "world". Exporting
raw weights must name the frame. A default-built recurrent MAPPO System now
registers the left hook, so an evaluation interrupted before this change and
started with a default-built System is refused on resume as a different System.

## A38. Neutral Collision Handling

**Accepted change — 2026-09-14.** This supersedes A37's collision-ordering
deferral and earlier requirements for sequential global-slot or obstacle-row
collision resolution. It authorizes the reviewed replacement in
`core/geometry.py`, its ordinary/Charge calls in `core/env.py`, and the two
explicit static-only policy query budgets in `policies/reactive_common.py`.
The replacement is integrated. The evidence below supports this change;
release qualification remains separate from the accepted collision contract.

Reordering agent rows, with every per-agent input reordered to match, must only
reorder the result. Reordering obstacle rows must leave the result unchanged.
These are exact float32 storage contracts within the same backend, array shapes
and execution settings. Team IDs, global slots and obstacle IDs must not choose
a contact winner or a separating direction. Changing physical coordinates is
a different operation: ordinary floating-point differences under reflection,
including their growth through later contact and policy decisions, remain
diagnostic. This evidence does not assert exact CPU/GPU numerical identity.
Neither this contract nor a symmetric map requires every self-play game to draw.

The movement solver keeps separate body and static corrections in repeated
rounds. Its default is four physical movement substeps, with 28 literal
collision rounds per substep and one body sweep per round. Correction strength
is fixed at 1. Each substep keeps one starting position; numerical rounds repair
proposed endpoints from that start. Bounds, actual pillar circles and actual
rotated rectangles remain authoritative. A rectangle's disc-clearance region
includes its rounded corners. Static correction retains useful sliding, and a
whole-disc travel check guards the committed straight segment. This does not
claim continuous collision detection between moving bodies.

Each body pair shares its correction equally. Geometry supplies contact and
tie directions, using separation and movement intent rather than row identity.
A body that becomes a collision participant at the final substep may already
overlap another body after its earlier intangible movement. That narrow case
uses the current radial separation for recovery. Already blocking pairs retain
their incoming-side rule. Equal-radius bodies with exactly equal positions and
equal intended movement have no physical separating direction. This narrow
anonymous coincidence may remain coincident; it must not acquire an arbitrary
axis or slot-based push. Static validity still applies. A later distinct
movement intent must receive normal collision handling.

Charge uses the explicit `project_charge_endpoints_with_geometry` helper.
Its requested relocation may pass intervening bodies or obstacles, as before.
It first repairs the arrival's static contacts, then resolves body contacts
using the arrival geometry; earlier separation only breaks a direction tie.
The default performs 28 arrival-recovery rounds and 28 collision rounds in one
endpoint step. Ordinary movement then runs from the realized Charge positions
using the already chosen actions. Intermediate Charge body overlap is a
diagnostic; public body acceptance applies after Charge and ordinary movement.

The accepted general body-overlap ceiling is exactly **0.135 map units** for
participating bodies in the valid-start qualification scope. This is the amount
by which the sum of two radii exceeds their center distance. Zero overlap remains
the preferred result. There is no added tolerance on this ceiling. Named simple
controls keep their stricter limits,
and the anonymous case above is the explicit exception. Finite values, bounds,
obstacle clearance and static travel checks remain separate hard requirements.
The common ceiling replaces historical per-stratum body limits; those older
measurements remain diagnostics. Respawn still places each body on its assigned
pad at the end of the transition. A zero-shield respawn can create the exact
anonymous overlap described above.

The ordinary helper keeps its existing signature; explicit round counts are
literal, and zero body sweeps disables body correction. The two static-only
policy queries explicitly retain four collision rounds. State, observation,
mask, action, reward, transition-fact and recording schemas are unchanged.
Action choice, Charge, ordinary movement, death, shield and respawn timing are
unchanged. No solver memory, public precision setting or coordinate grid is
added. Existing line-of-sight and obstacle-query behavior is preserved.

The default briefly increased to 64 rounds on 2026-09-15, then returned to 28
after the GPU cost and scenario comparisons. A commented 64-round option remains
beside the default for excessive body overlap found during training. R28 was
about 2.2–2.3 times as fast as R64 in the isolated RTX 5090 movement and Charge
checks at batches 1, 64 and 1024. This does not measure full training speed.
The following evidence describes the original 28-round integration and
acceptance decisions; changing the default back does not rerun those checks.

Completed RTX 5090 evidence for that integration includes 208 aligned team/slot
pairs across 54 recorded fields in the 416-condition full-game census, plus its exact repeated
run with reordered world lanes. The final integrated source reproduces all 92
saved arrays from those two executions exactly, including captured collision
inputs and outputs. The integrated movement helper also matches the frozen
candidate exactly on endpoints and all four committed substeps for the full
13,312-input bank on CPU and GPU. Focused comparisons cover Charge and the
static-only policy queries. A final type-only change preserves the calculation
and passes the full GPU game comparison. No full CPU game comparison between
the candidate and integrated versions is claimed.

Three further public contacts found in the CPU census were replayed with the
same saved inputs on GPU. Their GPU overlap depths are 0.134743, 0.095014 and
0.107966 map units. The user inspected these contacts and accepted the 0.135
ceiling. All three pass that ceiling; their earlier failures under 0.076 remain
recorded. This acceptance change does not change the 28-round solver or its cost.

The separate controlled-reflection study completes 2,496 games with no assigned
physical failures and exact retained team/slot comparisons in both orientations.
Reflected game outcomes often differ: 194 of 416 matched outcomes agree when
controllers choose actions throughout each game. These are spatial-reflection
diagnostics, not evidence that all mirrored fights agree. The checks support
the stated comparisons; they do not establish a universal speedup or
sample-efficiency claim. (Workload note, 22 September 2026: the census and reflection
games used ALPHA version 2 and BETA version 4 as their workload. They remain
solver evidence and say nothing current about either controller.)

**Scenario repair — 2026-09-15.** The user supplied revised winning commands
for Scenarios 1, 3, 5 and 8 and three small physical edits for Scenario 4.
Its Mage-B starts at y=5.8, obstacle_0 moves to x=7.2, and obstacle_2 moves
to y=3.4. Scenarios 2, 6 and 7 retain their commands and physical setup.
All eight keep their goals, rules and intended mechanics. Their nine winning
lines, including both Scenario 1 healing choices, now win 20–19 on GPU over
41 real transitions. Expected per-turn records were updated only after those
user-supplied routes were verified with live opponent decisions. Scenario 3's
revised missed-move control loses Hunter on turn 7; its older draw result
belongs to the earlier route.

The packaged scenarios and Notes now use revisions 41/19/26/14/15/16/29/18.
Scenario 3 retains approved r24 physics. Twenty-four focused GPU checks pass
for the packaged solutions, their identities and public initialization.
Standalone solution tests use the revised commands and the current packaged
setups where those setups changed. Historical fixture checks remain separate.
Scenario 2 demonstrates its within-range wall protection on turn 4, after its
turn-3 Ultimate heal. Delaying Charge in the revised Scenario 8 lets Priest-B
complete its chosen self-heal and produces a 19–19 draw. This retains the
action-timing lesson; the older dying-healer rescue belongs to the older route.
CPU/GPU position and displacement snapshots allow 0.00016 map units of
rounding difference; health and damage retain 0.00001, and actions, life flags,
scores and outcomes remain exact. Storage permutations remain exact on the
same backend. These focused
results do not claim a complete regression-gate or release-qualification pass;
those gates must be run on the final candidate.

## A39. Sampled Training Maps And Rosters

**Accepted training-distribution rule — 2026-09-19.** This replaces A15's
handpicked benchmark curriculum rosters. Its retired exhaustive composition
grid remains retired. Episode construction and simulator roster validity
retain A15's wider contract; the rules below belong to training selection.

The approved installed map selections are training 0–41, validation 42–46 and
locked test 47–51. Eligibility follows verified content and declared use, not
filenames or numbers alone. Protected scenario content remains separate from
adaptive training inputs. Shared class catalogs, schemas and simulator rules
remain permitted dependencies. Direct training draws uniformly from the full
training set. A curriculum may select an eligible subset; map draws within
that subset remain uniform.

Both teams have the same selected size. At 1v1, each team independently draws
one of Mage, Warrior, Hunter and Rogue with equal probability. At 2v2–4v4,
each team independently draws a uniform subset of the five classes without
replacement. The teams may share classes. Within a team, selected classes use
the relative order Mage, Warrior, Hunter, Rogue, Priest and occupy the first
slots; unused slots remain inactive. At 5v5, both teams use exactly that
five-class canonical roster. There is no production table of roster
combinations or handpicked smaller-team roster list.

Maps and rosters change only at episode reset. Continuing games keep their
configuration, state and recurrent memory. These distribution rules do not
change task mechanics, action meanings, score thresholds, horizons or the
simulator's support for custom rosters.

M9 owns the distributions and later curriculum execution. M10 owns learners
and learning experiments. Preparation, sampling and episode-provenance tools
do not establish a working trainer or a learning result. Curriculum schedules,
shaping and opponent history are separate implementation work. The
[evaluation protocol](../evaluation/protocol.md#episode-training-evaluation-and-scenario-ownership)
states the active experiment contract.

## A40. Training Collection Boundaries

**Accepted training-execution rule — 2026-09-19.** This extends A39's reset-time
selection with curriculum accounting, optional team potential shaping and
same-run self-play history. It changes no Core transition, observation, action,
reward or metric rule. Basic environment/System loops remain available.

Plain and RS request canonical 5v5 on all 42 training maps. C and C-RS request
17 stages: sizes 1–5 on map 0 at 4% each; eleven 5v5 pools from maps 0–1 through
0–11 at 20/11% each; then all 42 maps at 60%. C denotes curriculum; RS denotes
shaping. All four keep K20/H300. These are starting experiment settings whose
learning value remains unproven.

A round advances every lane once. The positive even batch stays fixed, and the
total real-transition budget must divide by it exactly. Assign all requested
shares together using largest remainders, with earlier stages winning exact
ties. Reject a budget that gives any active stage zero rounds. Do not enlarge
the total or silently remove stages. Stage completion checks exact real counts
and equal exposure to the two fixed spawn arrangements.

Stage changes apply to future resets. Existing games retain their configuration
and memory. Reset finished games immediately before their next real action,
after any intervening stage change or completed learner update. A final budget
cutoff performs no replacement reset. Report requested budgets separately from
played distributions; a requested stage can receive no new games. Reset calls
and output padding count as neither experience nor played games.

Optional shaping uses coefficient times own score minus opponent score as the
team potential. The adjustment is the configured learner discount times next
potential, minus current potential. Real wins, losses and horizon draws set
next potential to zero; collection/stage cutoffs and actor death do not. Use
the producing game's pre-action and after-action scores, including authored
starts. Keep one team-scale signal, task rewards and official scores separate.
The default coefficient 0.01 is a starting setting. Full discounted adjustment
is constant for a fixed start; this is not a faster-learning or undiscounted
win-rate claim. Disabled shaping skips its calculation.

With empty history, opponents use current self-play. Otherwise each reset picks
current weights with probability 0.8 or a uniformly chosen occupied historical
slot with probability 0.2. Historical weights and all inference variables remain
fixed for that game. Addition of 2026-09-21: an optional declared
pinned share, zero by default, makes the first capture follow the first
completed update and keeps that actor in slot 0 for the run; each reset then
picks slot 0 with the declared share, another occupied slot with total
probability 0.2 when any exists, and current weights otherwise. The default
preserves this rule exactly. Addition of 22 September 2026: the pinned share
may instead play a named System (a built-in team, an exported actor, or a
researcher's JAX or host method), which plays every lane assigned to slot 0
under the same one-team rules as a Team B method in evaluation. Its identity,
registration and recorded controller exposure are saved with the run, a JAX
method's memory is saved with each checkpoint, and a host method's memory is
not, so a resume that would cut one of its unfinished games is refused. Results
against a pinned System are familiar-opponent results (see A26). Current
weights refresh only after a completed learner update, before the next block,
while game memory continues. Teams own separate memory and retain existing
actor information limits.

The bank holds at most 20 immutable snapshots without eviction. Requested
thresholds are 5%, 10%, ..., 100% of the exact real budget, rounded upward to
whole rounds; with a positive pinned share (2026-09-21) they are round
1, then 5% through 95%, and the 100% capture that no later game can play is
dropped. The first completed update reaching unmet thresholds captures one
actor and maps all those thresholds to it. Report the actual capture round and
update; do not create duplicate stored entries to fill threshold labels. The
final capture may receive no later training exposure.

Collection returns compact permitted inputs, same-call actor outputs, separate
rewards, real-row masks, producing identities and the true final successor.
An optional separate physical-state view is training-only and stored once per
game. Critic memory, learner targets, optimization, model selection and durable
learner restart remain outside this collection boundary. Correct collection,
cost measurements and learning results need their own evidence. See the
[training guide](../training/README.md#collect-an-exact-experience-budget).

## A41. Complete Recurrent MAPPO Runs

**Accepted training-workflow rule — 2026-09-19.** Reuse A39 content admission
and A40 collection. The optional trainer joins the existing donor PPO update,
separate training-only critic, exact budgets, curriculum and opponent history.
Actor information rights and simulator rules remain unchanged.

The public owners are `training.train`, `training.load_system` and
`training.analyze`. CLI commands call those same functions. A run owns its
settings, source/content identities, complete learner checkpoints, optional
episode recording, validation tasks and reports. Default training uses plain
recurrent MAPPO, B32/T128 and the recorded donor settings. Other learners remain
separate implementation work.

Keep action-time actor outputs. Real endings receive zero bootstrap; ordinary
collection cutoffs retain it. Critic bootstrap reads the true successor without
advancing retained recurrent memory twice. Death/respawn does not clear memory.
Padding contributes no experience or samples, and empty updates change no state.
Each accepted update refreshes current self-play/history exactly once.

Save complete state at initialization and completed update boundaries. Validate
all payloads, scientific settings and continuation state before recording rewind.
Checkpoint identities include their actual payload and continuation ancestry.
Preserve the original training-writer registrations. Frozen actor exports retain
sampled deployment behavior and exact parameter identity without critic data.

Built-in MAPPO fixes GPU autotuning at level zero on the outer collection and
update compilations, including recorded collection. It changes no global JAX
setting and leaves raw numerical helpers composable. Checkpoints bind the
compiler policy and numerical runtime before any recording or log recovery.
Historical actors and reports remain readable; missing old execution identity
does not qualify strict learner continuation. See the
[compiler policy](../training/source_reuse.md#training-compiler-policy) for the
reason, evidence and limits. This does not promise equality across hardware or
library changes.

The provisional panel contains fixed halfway/final actors from a separate Plain
development run. Validation uses maps 42–46, equal map/opponent weights and paired
spawn ends. Routine checks use 200 games; fresh confirmations use 1,000. Confirm
the best two eligible routine checkpoints plus final when distinct, then select
by confirmation score and earlier step. Incomplete results cannot select a model.
Keep shared-seed opponent/spawn vectors together when estimating uncertainty.
This smaller panel does not replace the later four-family comparison panel.

An unattended command must finish declared validation, selection, export, final
diagnostics and reports after the exact training budget. It must resume pending
work without duplicate games or extra training. The trained-model slot diagnostic
runs only after the full-run final actor exists. Progress uses existing host
records; added terminal output is retained only with no measurable slowdown.

Working execution, efficient computation, sample efficiency and learned team
behavior need separate evidence. See the complete
[training workflow](../training/README.md#train-resume-load-and-analyze).
