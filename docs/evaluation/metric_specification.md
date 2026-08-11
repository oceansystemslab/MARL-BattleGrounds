# Evaluation Metric Specification

## Status and authority

> **NORMATIVE CONTRACT — ACTIVATED 2026-08-10.** Metric rows remain subject to
> their stated readiness and validity gates.

This document is the normative MARL-BattleGrounds metric contract. It owns
metric identities, meanings, dispositions, amount stages, opportunity rules,
attribution limits, and presentation tiers. The companion
[evaluation protocol](protocol.md) owns evaluation cells, aggregation,
uncertainty, cross-play, scenarios, and leakage control. Accepted departures
from the original PDF are recorded in the
[specification amendments](../design/specification_amendments.md).

The specification is deliberately broader than the initial implementation.
`derivable_now` means the completed mechanics and Milestone 6 evaluation seam
can support the metric. `requires_future_task_authority` means the disposition
is settled but the formula cannot activate until the owning task defines its
score, objective, reward, and terminal facts. A deferred or blocked row is not
an official result.

This document does not create a production metric registry. Stable IDs are
public semantic references for documentation, artifacts, tests, and future
implementations. Formulas may be implemented only by their named owner.

## Metric constitution

A MARL-BattleGrounds metric must pass all of these tests:

1. A researcher can explain it in one sentence.
2. It answers a real behavioral question rather than merely reporting an
   available statistic.
3. Its denominator represents genuine opportunities.
4. Its direction is clear, or it is explicitly labeled descriptive.
5. It does not claim causality or individual credit that the trajectory does
   not establish.
6. It remains meaningful across algorithms, model architectures, and seeds.
7. Its raw numerator, denominator, counts, sums, durations, and stratification
   keys survive downstream aggregation.
8. It adds material information beyond simpler retained measurements.
9. Its scientific value justifies implementation, compute, storage,
   maintenance, and presentation cost.
10. Its aggregation is statistically valid and its denominator cannot be
    trivially gamed without a visible exposure companion or countermetric.
11. It is replay-verifiable from an authoritative owner without reconstructing
    simulator semantics.

Failure does not always mean deletion. A candidate may move to advanced
description, a controlled scenario, diagnostics, validation-pending research,
or an explicitly rejected ledger.

## Data and interpretation vocabulary

### Health-effect and resolution stages

Every damage, healing, or health-resolution value must use one of these exact
stages:

| Stage | Meaning | Attribution |
| --- | --- | --- |
| `raw_source` | Accepted source output before source and recipient modifiers | Source-aligned |
| `source_modified_gross` | Output after source-side mechanics such as Mage Burst and Mage aura | Source-aligned |
| `recipient_modified_gross` | Effect after recipient-side mitigation or anti-heal, entering simultaneous health resolution | Source-aligned when routing is unique; otherwise recipient total |
| `combat_resolution_health` | Clamped recipient health after simultaneous combat damage/healing and before regeneration | Recipient state, not an effect allocation |
| `realized_net_health_change` | Combat-resolution health minus transition-start health | Recipient only; not uniquely attributable to damage or healing sources |
| `actual_regeneration` | Separately authored post-combat recovery applied after combat resolution | Recipient/lifecycle source |

The word **effective** is not an amount stage. Existing code fields containing
that word are interpreted through the table above.

Milestone 6 CP2 preserves the fixed transition facts losslessly and exposes
sparse events only as a deterministic semantic view. Event absence never
erases false, zero, padded, or continuously active normalized facts. Submitted
and accepted actions remain one normalized authority inside action-acceptance
facts. Evaluation rewards are explicitly `canonical_reward_by_agent` and,
when task-authored, `canonical_reward_by_team`; shaped or auxiliary rewards do
not enter these fields.

The discriminated V1 event union has exactly 21 atomic variants. Newly dead
recipient truth belongs to `AgentDiedEventV1`; each authoritative positive damage
source on that lethal transition receives a separate
`LethalDamageContributionEventV1`. Death sorts before its contribution records
at phase rank 90. A contribution is not a killer, last hit, or complete
historical elimination attribution.

Rank-120 ordering uses family-specific coordinates: team waves sort by `(120,
team_index, -1, wave_subtype, neutral_source)` and realized agent respawns by
`(120, configured_team_index, agent_global_slot, respawn_subtype,
neutral_source)`. Each team wave therefore precedes its realized agents, with
teams kept in canonical order.

Recipient-modified gross damage can exceed remaining health. Gross healing can
offset simultaneous damage without producing a positive net-health change.
Consequently, individual realized damage, individual realized healing, and
individual overhealing are undefined without an arbitrary apportionment rule.
MARL-BattleGrounds does not invent one.

### Attribution grades

Every effect-derived metric declares the strongest attribution supported by
the recorded trajectory:

| Grade | Meaning |
| --- | --- |
| `source_exact` | The recorded source-to-recipient route uniquely owns the gross effect or application |
| `recipient_exact` | The recipient outcome is exact, but source allocation is not |
| `unique_emitter_exact` | One eligible emitter uniquely owns a modifier/application in this roster and transition |
| `combined_emitter_set_exact` | The combined team/class effect is exact but division among overlapping emitters is not |
| `attribution_ambiguous` | The requested subject credit is not identifiable and the value is `N/A` |

Canonical non-duplicate rosters can make some aura, anti-heal, Trap, and rescue
credit unique. Duplicate-source rosters fall back to combined team/class
attribution or `N/A`; offline analysis must not invent an apportionment rule.

### Causal-support tiers

| Tier | Meaning | Permitted wording |
| --- | --- | --- |
| `authoritative_outcome` | Direct phase-authored fact or task result | “occurred,” “applied,” “ended because” |
| `deterministic_derived` | Exact transformation of recorded semantic frames/facts | “derived,” “covered,” “was active while” |
| `associational` | Temporal or contextual relation without counterfactual identification | “during,” “followed by,” “associated with” |
| `counterfactual_unsupported` | Requires an unobserved alternate trajectory or arbitrary credit | No official causal claim |

“Within K steps” is an association unless K is a mechanically owned duration
and the claim remains phrased as co-occurrence during that window.

### Subject and symmetry

Every symmetric statistic is computed for Team A and Team B from the same
definition. A difference is exposed only when subtraction is meaningful. Two
teams from one match are paired observations, not independent experimental
units. Agent and class rows are nested descriptive measurements; an absent
class is `N/A`, never zero.

### Disposition axes

Each canonical candidate has four independent labels.

| Axis | Values |
| --- | --- |
| Endpoint role | `primary_confirmatory`, `key_secondary`, `exploratory_descriptive`, `diagnostic_qc`, `rejected` |
| Surface | `primary_team`, `primary_agent_class`, `advanced`, `scenario`, `cross_play_population`, `learning_runtime`, `reward_research`, `none` |
| Readiness | `derivable_now`, `requires_future_task_authority`, `requires_policy_sidecar`, `validation_pending`, `counterfactual_unsupported` |
| Validity | `pass`, `conditional`, `blocked` |

The implementation owner is separately one of: simulator fact, host
evaluation, offline metrics, task, scenario runner, trainer/evaluation harness,
or UI/export. Presentation never owns semantic computation.

### Stable IDs and versioning

IDs use `marlbg.<family>.<metric>.vN`. A change to formula, denominator,
eligibility, subject, direction, amount stage, or attribution semantics creates
a new version. Display-text or formatting-only corrections do not.

One semantic metric may carry long-form dimensions such as team, agent, class,
ability, target class, status channel, task, map, information regime, and
window. Identical slices do not become separate metric IDs.

Task mechanics that do not exist yet receive a `candidate.<task>.<name>` key,
not a normative `.v1` metric ID. The owning task replaces that key with a
versioned metric definition only after its authoritative state, events,
eligibility, and edge semantics exist.

### Contract layering

Stable metric semantics must not change merely because a new paper uses a
different opponent pool or confidence interval. The public contract is split
conceptually into four layers:

- **Metric definition:** ID/version, question, scope, data authority, amount
  stage, eligibility, sufficient components, reduction kind, zero-opportunity
  result, attribution, interpretation, gameability, shaping properties, and
  validation state.
- **Evaluation suite:** task and information-regime strata, metric selection,
  endpoint hierarchy, layouts, scenarios, rosters, cooperative partners,
  adversarial opponents, sides, frozen joint weights, completion policy, and
  artifact retention.
- **Experiment manifest:** evaluated algorithms/checkpoints, independent
  training runs, seed schedule, checkpoint selection, comparisons,
  uncertainty method, confidence level, multiplicity family, runtime protocol,
  and train/validation/locked-test partition.
- **Metric result:** metric/suite/manifest identities, complete cell and subject
  coordinates, raw sufficient components, computed value or `null`, result
  status, rollout completion, observer-processing status, per-statistic
  endpoint observation where applicable, and source-schema versions.

Only a semantic-definition change increments a metric version. Population,
weighting, comparison, or inferential changes increment the suite or manifest
version instead. These are documentation contracts in the current milestone,
not a request for a universal production registry.

## Presentation budgets

### Primary team card

Each task and `execution_information_mode` receives at most four endpoint
blocks; this is a ceiling, not a quota:

1. win/draw/loss as one outcome distribution;
2. terminal canonical score differential;
3. one non-redundant task-native signature; and
4. at most one validated coordination descriptor, displayed with its exposure.

The footer always states independent training seeds, evaluation cells,
episodes, failures/truncations, cell weighting, and uncertainty method.

### Primary agent/class card

The compact card contains four universal descriptive blocks and at most one
class signature:

- recipient-modified gross damage output and team share;
- recipient-modified gross healing output and team share, otherwise `N/A`;
- recipient-modified gross damage received and team share;
- deaths and team death share; and
- one validated class signature.

These columns describe role usage. They do not create a universal ranking of
heterogeneous classes. Lethal-transition contribution remains available in the
advanced export, but it is too outcome-local and gameable to be a universal
primary role fact.

### Advanced, scenario, and diagnostic surfaces

The advanced export preserves tidy long-form observations and raw sufficient
statistics. It favors distributions, medians, and interquartile ranges over
default min/max columns. Scenario cards contain one primary quantitative
endpoint, at most two secondary margins, explicit violations, and replay.
Diagnostics remain outside tactical leaderboards.

Every displayed metric exposes its ID/version, units or health-effect stage,
subject, direction or descriptive label, opportunity and `N/A` behavior,
aggregation protocol, defined/undefined counts, and concise allowed
interpretation. Tooltips are reachable by hover, keyboard focus, and click.
Sortable columns use one consistent interaction and preserve team/opponent and
information-regime labels. CSV and JSON export tidy raw sufficient components,
not only rounded dashboard values. UI/export code references this contract and
does not own a second formula.

## Raw sufficient-component defaults

This document owns what each semantic metric must preserve; the
[evaluation protocol](protocol.md) exclusively owns reduction order, cell
weighting, inferential units, and uncertainty. Unless a metric row states
otherwise, preserve:

- a total's sum and eligible episode count;
- a rate's numerator, genuine-opportunity denominator, and zero-opportunity
  count;
- a share's subject component and corresponding team total;
- a duration's qualifying and eligible agent-steps; and
- a distribution's event- or episode-level observations at its declared unit.

A zero opportunity produces `N/A`, not zero. Partial prefixes are excluded
from every official endpoint estimator unless the metric explicitly declares
prefix validity. Scientific censoring is a separate per-statistic endpoint
observation, not an episode completion state. Prefix-valid diagnostic or
descriptive components may be exported only with rollout and processing status
and remain outside the official estimator.

Result status follows this precedence: `invalid_artifact`,
`structurally_inapplicable`, `ambiguous_attribution`, `insufficient_data`,
`zero_opportunity`, then `defined`. A currently zero denominator on an
ineligible prefix is insufficient data rather than zero opportunity. A
right-censored result is defined only when that versioned metric declares a
censoring estimand and preserves the required censoring component.

Milestone 6 CP3 supplies strict generic count, sum, ratio-component,
duration-component, opportunity, and distribution-observation records. It does
not implement the metric formulas below. One immutable
`SufficientStatisticAccumulatorV1` combines only drafts with the same complete
semantic key and preserves raw components; ratios, means, ratings, uncertainty,
and presentation values remain downstream derivations. Agent and policy
subjects must join configured-active context rows. An absent class may appear
as `structurally_inapplicable`, never as a fabricated zero, and padded actors
never enter opportunity denominators merely because their canonical no-op mask
has a valid category.

The CP3 accumulator is episode-local. Its `eligible_episode_count` is therefore
exactly `0` or `1`, and a ratio's `zero_opportunity_occurrence` is the final
episode-level `0` or `1` incidence after local contributions merge. Later
cross-episode reduction counts those finalized raw rows; it does not merge CP3
draft accumulators and cannot erase zero-opportunity episodes.
At final materialization, an incomplete or otherwise ineligible row carries
`eligible_episode_count = 0`; `defined` and eligible `zero_opportunity` rows
carry `1`. A ratio with a recorded zero-opportunity occurrence finalizes as
`zero_opportunity`, never `defined`. Other zero-valued component families do
not by themselves reveal metric-specific opportunity semantics. A separately
recorded observer-processing failure does not erase a fully consumed prefix:
complete-only eligibility requires rollout completion and equal validated and
processed transition counts, not a successful-status label by itself.

For standard replay, `EvaluationMetricReportV1` is wrapped in its own
content-addressed artifact and joined to the replay's pre-link trajectory
content. The replay stores only the report artifact's path-free identity,
schema, digest, and canonical byte length. Missing sidecar bytes do not make
the semantic trajectory unrenderable, but they do make metric-bundle evidence
incomplete and must be reported as such. See the
[standard replay format](replay_format.md).

## Canonical task-independent metrics

### Outcome and completion templates

| ID | Question and sufficient statistic | Role / surface | Direction | Readiness / owner |
| --- | --- | --- | --- | --- |
| `marlbg.task.outcome_distribution.v1` | How often did the team win, draw, or lose? Preserve the three mutually exclusive counts and total eligible games. | `primary_confirmatory` / `primary_team` | win higher, loss lower; one multinomial endpoint | `requires_future_task_authority` / task |
| `marlbg.task.terminal_score_differential.v1` | By how much did the team lead at the terminal frame? Preserve `team_score - opponent_score` per complete episode. | `primary_confirmatory` / `primary_team` | higher | `requires_future_task_authority` / task |
| `marlbg.task.evaluation_return.v1` | What canonical evaluation reward did the team/agent receive? Preserve the unshaped episode return and reward-mode identity. | `key_secondary` / `advanced` | task-defined | `requires_future_task_authority` / task |
| `marlbg.task.episode_length.v1` | How many valid transitions occurred before completion? Preserve transition count and end reason. | `exploratory_descriptive` / `advanced` | descriptive | `derivable_now` / host evaluation |
| `marlbg.artifact.completion.v1` | Was the rollout complete, partial, interrupted, or failed, and did host processing succeed? Preserve completion basis, validated/processed prefix lengths, processing failure, and end/failure reason. | `diagnostic_qc` / `none` | descriptive | `derivable_now` / host evaluation |

Outcome rows are complete-only. A missing outcome, failed run, truncation, or
right-censored endpoint never silently becomes a draw, loss, zero, or excluded
row.

### Combat output and exposure

| ID | Exact definition and raw components | Role / surface | Direction | Validity and caveat |
| --- | --- | --- | --- | --- |
| `marlbg.combat.recipient_modified_gross_damage_output.v1` | Sum, by source/team, of source-modified gross damage times the recipient modifier for accepted routed damage. Store HP and Basic/Ultimate dimensions. | `key_secondary` / primary cards + advanced | descriptive | `pass`; not realized HP loss |
| `marlbg.combat.recipient_modified_gross_damage_received.v1` | Sum authoritative recipient-modified gross damage by recipient/team. | `key_secondary` / primary cards + advanced | descriptive | `pass`; exposure is not automatically poor positioning |
| `marlbg.support.recipient_modified_gross_healing_output.v1` | Sum, by source/team, of source-modified gross healing times the recipient modifier for accepted routed healing. | `key_secondary` / primary cards + advanced | descriptive | `pass`; do not label “effective healing” or realized restoration |
| `marlbg.combat.realized_net_health_change.v1` | For each recipient-transition, preserve post-combat health minus transition-start health, plus gross damage and healing. | `exploratory_descriptive` / advanced | descriptive | `pass`; recipient-only net result |
| `marlbg.combat.upper_health_clamp_overflow.v1` | Recipient-level positive health amount discarded only by the upper health clamp after simultaneous netting. Preserve overflow and gross-healing exposure. | `exploratory_descriptive` / advanced | lower may indicate less saturation, but conditional | `conditional`; never attribute to one healer under overlap |
| `marlbg.combat.death_count.v1` | Count authoritative `AgentDiedEventV1` newly dead recipients for each team and class. | `key_secondary` / primary cards + advanced | lower for own team, higher for opponent, task-context dependent | `pass` |

Shares use the pooled agent/class component divided by its pooled team total
within the same cell. A team total of zero yields `N/A`. Damage/healing totals
are always accompanied by episode count and exposure opportunities.

### Lethal-transition damage contribution and coordinated offense

| ID | Exact definition and raw components | Role / surface | Direction | Validity and caveat |
| --- | --- | --- | --- | --- |
| `marlbg.combat.lethal_transition_damage_contribution.v1` | Count atomic `LethalDamageContributionEventV1` records where the source dealt authoritative positive recipient-modified gross damage on an enemy's lethal transition. Preserve source, recipient, class, team, and team enemy-death count. | `exploratory_descriptive` / advanced | descriptive | `pass`; not a kill, last hit, or complete historical elimination contribution |
| `marlbg.combat.lethal_transition_contribution_rate.v1` | Lethal-transition damage contributions divided by enemy deaths caused by the source's team, using the protocol's raw-component reduction. | `exploratory_descriptive` / advanced | descriptive | `conditional`; `N/A` when the team caused no deaths and susceptible to last-transition crowding |
| `marlbg.coordination.single_contributor_lethal_transition_count.v1` | Count enemy deaths having exactly one authoritative positive-damage source on the lethal transition. | `exploratory_descriptive` / advanced | descriptive | `pass`; earlier damage may have occurred, so this is not a “solo kill” |
| `marlbg.coordination.multi_contributor_lethal_transition_rate.v1` | Enemy deaths with at least two allied positive-damage sources on the lethal transition divided by all team-caused enemy deaths; always show the denominator. | `key_secondary` / primary-team candidate + advanced | descriptive | `conditional`; outcome-conditioned and gameable without death exposure |
| `marlbg.coordination.focus_fire_concentration.v1` | On a transition with at least two allied positive-damage sources, let `n_r` be sources damaging enemy recipient `r`; retain `max_r(n_r) / sum_r(n_r)` and one opportunity. Pool sum and opportunity count. | `key_secondary` / primary-team candidate + advanced | descriptive | `conditional` until construct validation; no eligible transition gives `N/A` |

Focus-fire concentration measures same-transition target concentration, not
whether the chosen target or degree of concentration was strategically
correct. It may enter a primary card only after blinded replay validation and
counterexample review.

### Abilities, cooldowns, and movement

| ID | Exact definition and raw components | Role / surface | Direction | Readiness / caveat |
| --- | --- | --- | --- | --- |
| `marlbg.ability.activation_count.v1` | Count accepted Basic and Ultimate activations by source/class/ability; rejected submissions are separate. | `exploratory_descriptive` / advanced | descriptive | `derivable_now` |
| `marlbg.ability.ultimate_cooldown_start_count.v1` | Count accepted actions with Ultimate enabled. | `exploratory_descriptive` / advanced | descriptive | `derivable_now`; no cooldown fact leaf |
| `marlbg.ability.ultimate_ready_transition_count.v1` | Count adjacent-frame positive-to-zero Ultimate cooldown edges. | `exploratory_descriptive` / advanced | descriptive | `derivable_now`; no cooldown fact leaf |
| `marlbg.movement.phase_displacement.v1` | Preserve exact vector and distance distributions separately for Charge-phase and ordinary-movement-phase realized displacement. | `exploratory_descriptive` / advanced | descriptive | `derivable_now` after M6 CP1; both vectors are phase-authored without a second geometry pass and are not positioning-quality scores |

Cooldown use rates require a task/experiment-owned opportunity definition.
Readiness alone is not an instruction to activate, so “ultimate efficiency” is
not a universal quality metric.

### Status and crowd-control lifecycle

| ID | Exact definition and raw components | Role / surface | Direction | Validity and caveat |
| --- | --- | --- | --- | --- |
| `marlbg.status.application_count.v1` | Count authoritative applications by source, recipient, and stable status channel. | `exploratory_descriptive` / advanced | descriptive | `pass` |
| `marlbg.status.active_recipient_steps.v1` | Count frame-level active status recipient-steps; preserve eligible active/alive recipient-steps and channel. | `exploratory_descriptive` / advanced | descriptive | `pass`; “uptime” is the derived ratio |
| `marlbg.status.lifecycle_cause_count.v1` | Count age-to-zero, refresh/extension, damage-break, and new-death-clear causes independently by recipient/channel. | `exploratory_descriptive` / advanced | descriptive | `pass`; causes may coexist |
| `marlbg.control.damage_to_controlled_recipient.v1` | Recipient-modified gross damage routed while the recipient had a named transition-start control status. | `exploratory_descriptive` / advanced | descriptive | `pass`; context association, not proof of follow-up quality |
| `marlbg.control.enemy_death_while_controlled.v1` | Enemy deaths occurring while a named transition-start control status was active. | `exploratory_descriptive` / advanced | descriptive | `pass`; associational |

Do not sum heterogeneous slow, stun, anti-heal, Burst, and Freedom durations
into one “CC score.” Channel-level durations and exact lifecycle causes remain
available in the long-form export.

### Class-specific retained metrics

| ID | Exact definition and raw components | Role / surface | Direction | Attribution and readiness |
| --- | --- | --- | --- | --- |
| `marlbg.mage.burst_window_damage.v1` | Recipient-modified gross damage from the Mage while its Burst status is active; preserve activations, active Mage-steps, and damage exposure. | `key_secondary` / primary class candidate + advanced | descriptive | `deterministic_derived`; damage during Burst, not necessarily caused by activation |
| `marlbg.mage.burst_enemy_death_association.v1` | Enemy deaths where the Mage dealt positive damage on the lethal transition while Burst was active. Preserve Burst activations and team enemy deaths. | `exploratory_descriptive` / advanced | descriptive | associational; not “kills caused by Burst” or complete historical contribution |
| `marlbg.mage.aura_coverage.v1` | For each Mage emitter, covered eligible emitter-beneficiary steps divided by all same-team steps where both emitter and beneficiary are active, alive, and unshielded. Preserve emitter/beneficiary counts. | `key_secondary` / primary class candidate + advanced | descriptive | exact per-emitter coverage after M6 CP1 |
| `marlbg.mage.combined_aura_amplification.v1` | Combined recipient-modified gross damage increment attributable to the recorded Mage-aura multiplier, aggregated by team/class. | `exploratory_descriptive` / advanced | descriptive | exact combined value; per-emitter value blocked when emitters overlap |
| `marlbg.warrior.aura_coverage.v1` | For each Warrior emitter, covered eligible emitter-beneficiary steps divided by all same-team steps where both emitter and beneficiary are active, alive, and unshielded. | `key_secondary` / primary class candidate + advanced | descriptive | exact per-emitter coverage after M6 CP1 |
| `marlbg.warrior.combined_aura_mitigation.v1` | Sum pre-recipient damage minus recipient-modified gross damage where the recorded Warrior aura modifier applies. | `key_secondary` / primary class candidate + advanced | descriptive | exact combined team/class value; per-emitter value blocked under overlap |
| `marlbg.hunter.trap_active_steps.v1` | Active Hunter Trap stun recipient-steps with application and eligible-recipient exposure. | `key_secondary` / primary class candidate + advanced | descriptive | exact; replaces placed-trap uptime terminology |
| `marlbg.hunter.trap_status_episode_end.v1` | Segment Trap status episodes from authoritative lifecycle causes in simulator phase order; record whether each completed episode ended by age, damage break, or death clear. An ordinary refresh without an end remains in the same episode. An end followed by same-transition reapplication closes the old episode and starts a new one even though no zero-duration frame exists. | `exploratory_descriptive` / advanced | descriptive | exact after M6 CP1; multiple episode edges may occur in one transition and artifact-end censoring is explicit |
| `marlbg.hunter.trap_damage_break_rate.v1` | Completed Trap status episodes ending by damage divided by all completed Trap status episodes; report censored active episodes separately. | `key_secondary` / primary class candidate + advanced | descriptive | conditional; never use casts as a silent denominator |
| `marlbg.rogue.combined_anti_heal_reduction.v1` | Sum source-modified healing minus recipient-modified gross healing while anti-heal applies. Preserve healing exposure and active steps. | `key_secondary` / primary class candidate + advanced | descriptive | exact combined reduction; per-Rogue credit blocked under overlap |
| `marlbg.rogue.priority_target_damage_share.v1` | Recipient-modified gross Rogue damage to a task-declared priority-target class/state divided by all Rogue gross damage. | `exploratory_descriptive` / advanced | descriptive | target class available now; flag-carrier state requires CTF authority |
| `marlbg.priest.same_transition_lethal_damage_rescue.v1` | Count living recipients where gross damage alone reached/exceeded start health, gross healing was positive, and post-combat health remained positive. Preserve recipient opportunities and unique/multiple healer count. | `key_secondary` / primary class candidate + advanced | higher with exposure companion | recipient/team causal outcome; source attribution only when exactly one healer contributed |
| `marlbg.priest.freedom_binding_coverage.v1` | Among active, alive, unshielded, unstunned recipient-frames with Freedom active, the fraction satisfying `max(canonical_float32_slow_product, global_slow_floor) < freedom_floor`. The product uses the catalog's stable slow-channel order. Preserve binding and eligible frames separately; return `N/A` at zero eligibility. Frame `t` governs movement in `t -> t+1`; stay actions remain eligible. | `exploratory_descriptive` / advanced | descriptive | exact frame/catalog derivation; permits only “Freedom was mechanically binding,” not distance recovered, movement caused, or tactical value |

No class signature is guaranteed a primary slot merely because it is
derivable. Each must pass construct validation on realistic 5v5 replays and
must remain useful when duplicate-class ablations are enabled.

For Trap episode reconstruction, lifecycle causes are consumed in their
authoritative order. Age or damage break closes the currently active episode;
application can then start a new episode on the same transition; new-death
clearing can close that newly applied episode immediately. Thus
break–reapplication–death-clear may yield two completed episodes in one
transition. One episode end carries the set of coexisting authoritative causes
rather than a forced precedence; it contributes once to the completed-episode
denominator and to the damage-break numerator when that set contains
damage-break. An episode still active at artifact end is right-censored and is
reported separately from the completed-episode denominator.

### Lifecycle and diagnostic metrics

| ID | Exact definition and raw components | Role / surface | Direction | Readiness / caveat |
| --- | --- | --- | --- | --- |
| `marlbg.recovery.realized_regeneration.v1` | Sum actual post-clamp out-of-combat regeneration separately from combat healing. | `exploratory_descriptive` / advanced | descriptive | `derivable_now` |
| `marlbg.recovery.combat_countdown_reset_count.v1` | Count authoritative combat-countdown resets by agent and preserve current/next countdown context. | `diagnostic_qc` / advanced | descriptive | lifecycle diagnostic; not an engagement boundary |
| `marlbg.lifecycle.respawn_wave.v1` | Count authoritative team waves and realized agent respawns, preserving team, agent, and countdown context. | `diagnostic_qc` / advanced | descriptive | lifecycle diagnostic, not policy quality by itself |
| `marlbg.lifecycle.dead_agent_steps.v1` | Count configured active frame-agent rows that are dead; preserve new-death, respawn, and episode-end boundaries for life/dead-duration distributions. | `exploratory_descriptive` / advanced | descriptive | an unrespawned terminal life/dead interval is censored for duration analysis |
| `marlbg.lifecycle.spawn_shield_expiry.v1` | Count ordinary shield expiries and preserve active-at-start exposure. | `diagnostic_qc` / advanced | descriptive | lifecycle diagnostic |
| `marlbg.diagnostic.action_acceptance_rate.v1` | Configured active actor-transitions with no tuple-domain, movement, or combat-pair rejection divided by all configured active actor-transitions; preserve per-component accepted/rejected counts. | `diagnostic_qc` / none | higher usually indicates healthier policy plumbing | dead/stunned canonical choices remain real masked decisions; padded slots are excluded |
| `marlbg.diagnostic.action_rejection.v1` | Count submitted tuple-domain, movement-mask, and combat-pair rejection facts with submitted/accepted actions and opportunity count. | `diagnostic_qc` / none | lower usually indicates healthier policy plumbing | no invented LOS/range/cooldown reason |
| `marlbg.formation.ally_distance_distribution.v1` | Distribution of pairwise distances between eligible active/alive allies from semantic frame positions. | `exploratory_descriptive` / advanced | descriptive | not a cohesion-quality score; no new core distance fact |

Time dead, death-to-respawn duration, life duration, wave size, countdown
history, and recovery timing remain available as lifecycle diagnostics when a
named analysis needs them. They do not enter the primary tactical scorecard.

## Future task-owned metrics

These dispositions are stable, but the rows remain inactive until their task
milestone defines authoritative state and edge semantics.

### Team Deathmatch

| ID | Required definition | Role / surface | Activation dependency |
| --- | --- | --- | --- |
| `candidate.tdm.elimination_differential` | Team enemy deaths minus allied deaths under the task's official scoring eligibility. Omit if mathematically identical to terminal score differential. | `primary_confirmatory` / primary team | M7 score/outcome facts |
| `candidate.tdm.team_wipe_count` | Task-defined transitions or intervals where every eligible opposing agent is dead; preserve respawn-wave context. | `exploratory_descriptive` / advanced | M7 task semantics |

TDM does not create killer ownership, K/D, or generic teamfight victories.

### Three-hill King of the Hill

| ID | Required definition | Role / surface | Activation dependency |
| --- | --- | --- | --- |
| `candidate.koth.eligible_hill_control_share` | Team-controlled eligible hill-timesteps divided by all eligible team-control hill-timesteps; preserve neutral and contested states separately. | `primary_confirmatory` / primary team | authoritative per-hill control state |
| `candidate.koth.contest_share` | Contested eligible hill-timesteps divided by all eligible hill-timesteps. | `exploratory_descriptive` / advanced | authoritative contest state |
| `candidate.koth.control_transition_outcome` | Versioned counts of neutral/enemy/ally control transitions, including defense retention and capture/steal outcomes defined by the task. | `key_secondary` / advanced | task-authored transition events |
| `candidate.koth.hill_occupancy` | Per-agent/class eligible occupancy steps by ally/neutral/enemy hill and control state. | `exploratory_descriptive` / advanced | authoritative membership state |
| `candidate.koth.allocation_profile` | Long-form distribution of eligible agents across the three stable hill identities; no universal higher-is-better entropy score. | `exploratory_descriptive` / advanced | task and layout identities |

Control, contest, and occupancy use hill-timesteps, not bare episode horizon.
There is no arbitrary per-agent score contribution and no deaths/score
“objective fight cost” ratio.

### Capture the Flag

| ID | Required definition | Role / surface | Activation dependency |
| --- | --- | --- | --- |
| `candidate.ctf.capture_count` | Authoritative captures by team. | `primary_confirmatory` / primary team | flag/capture events and score |
| `candidate.ctf.capture_conversion` | Captures divided by eligible enemy-flag pickups, always displayed with that pickup exposure. | `primary_confirmatory` / primary team | pickup/capture identity |
| `candidate.ctf.forced_drop_interception` | Task-defined enemy-carrier forced drops/interceptions divided by eligible opposing carry episodes; preserve raw episodes and causes. | `key_secondary` / primary candidate + advanced | carrier/drop cause semantics |
| `candidate.ctf.friendly_return` | Task-defined friendly returns divided by eligible friendly dropped-flag episodes; report auto-return separately. | `key_secondary` / advanced | return causes |
| `candidate.ctf.flag_possession` | Carrier agent-steps by team/class and share of eligible possession time. | `exploratory_descriptive` / advanced | carrier state |
| `candidate.ctf.escort_coverage` | Carrier steps with at least one eligible allied non-carrier in a declared escort relation divided by carrier steps. | `exploratory_descriptive` / advanced | versioned relation and carrier state; not escort quality |

Carry time by class is descriptive. “Preferred carrier class share,” arbitrary
agent score credit, and universal escort/interception quality are rejected;
decision quality belongs in controlled scenarios.

On the compact team card, capture count, pickup-to-capture conversion, and the
pickup exposure supporting that conversion form one compound CTF task-native
block. They do not consume multiple task-signature slots or become independent
confirmatory families merely because all raw components are visible.

## Scenario-owned behavior matrix

| Behavior | Why no episode-wide quality scalar | Required scenario evidence |
| --- | --- | --- |
| Peeling / backline protection | Value depends on threat, protected ally, and resulting trade | protected-ally survival or health margin; threat displacement/control; violations |
| Kiting / disengagement / re-engagement | Low damage taken can also mean non-participation | survival or damage-trade endpoint under a fixed pursuer; distance/time margin; participation constraint |
| Flanking / backline access | Geometry and timing make angle alone ambiguous | priority-target access/effect endpoint; time or health margin; route/visibility constraints |
| Body blocking / escape denial | Contact alone may be accidental or harmful | protected route or interception outcome; progress margin; collision/position replay |
| Healing triage | Correct recipient depends on synchronized threats | weighted survival/health endpoint under fixed simultaneous pressure; response margin; invalid-target violations |
| Regrouping | Fast cohesion can be strategically wrong | survival/objective readiness after a fixed respawn split; time margin; premature-engagement violations |
| Rotations / multi-objective allocation | Entropy and rotation count have no universal direction | objective conversion under fixed multi-hill pressure; score/time margin; overcommit violations |
| Escort / interception | Proximity does not establish useful protection | capture/stop endpoint under fixed carrier route; completion/censored time; role violations |
| Trap discipline | Aggregate uptime cannot identify correct tactical timing | fixed threat-specific denial/peel endpoint; control/follow-up margin; misuse violations |
| Burst synchronization | Damage during Burst does not prove good timing | fixed coordinated damage/objective endpoint; activation timing margin; survival/position constraint |
| Freedom-assisted movement | Exact counterfactual movement is not in the trajectory | fixed slowed traversal/rescue endpoint; completion/progress margin; status evidence |

Every scenario must follow the scenario protocol. Replay is mandatory evidence
but never substitutes for the quantitative endpoint.

## Population, learning, and runtime metrics

| ID | Definition | Role / surface | Owner |
| --- | --- | --- | --- |
| `marlbg.population.matched_partner_performance.v1` | Task performance with the declared training-related cooperative partner under a frozen adversarial-opponent distribution. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.held_out_partner_performance.v1` | The same task endpoint with a disjoint cooperative-partner pool while holding the adversarial-opponent distribution fixed. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.held_out_opponent_performance.v1` | The same task endpoint against a disjoint adversarial-opponent pool while holding the cooperative-partner distribution fixed. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.partner_generalization_gap.v1` | Matched-partner minus held-out-partner performance under the same opponent panel and frozen joint cell weights; always report both absolute components. | `exploratory_descriptive` / cross-play population | evaluation harness |
| `marlbg.population.lower_tail_performance.v1` | Predeclared lower quantile over one explicitly named partner or opponent population dimension, never an unstable unqualified minimum. | `key_secondary` / cross-play population | evaluation harness |
| `marlbg.population.rating.v1` | Secondary rating with pool, protocol, side assignments, and rating-system version. | `exploratory_descriptive` / cross-play population | evaluation harness |
| `marlbg.learning.fixed_budget_performance.v1` | Official evaluation endpoint at a predeclared budget; report both environment transitions and active-agent decision transitions. | `primary_confirmatory` / learning runtime | trainer/evaluation harness |
| `marlbg.learning.curve_auc.v1` | Area under a predeclared held-out evaluation curve with fixed x-axis, horizon, checkpoint schedule, and interpolation rule. | `key_secondary` / learning runtime | trainer/evaluation harness |
| `marlbg.learning.time_to_threshold.v1` | First predeclared evaluation checkpoint reaching a frozen threshold; non-reaching runs remain censored. | `exploratory_descriptive` / learning runtime | trainer/evaluation harness |
| `marlbg.runtime.environment_throughput.v1` | Environment transitions per second under a versioned hardware/batch/JIT protocol. | `diagnostic_qc` / learning runtime | trainer/runtime harness |
| `marlbg.runtime.policy_inference.v1` | Policy inference latency/throughput under the same declared measurement protocol. | `diagnostic_qc` / learning runtime | trainer/runtime harness |

Cross-play summaries never discard the underlying focal-by-partner-by-opponent
tensor. Matched, held-out, cooperative-partner, and adversarial-opponent
experiments are distinct populations and cannot share an unlabeled
“robustness” number.

## Reward-shaping classification

Metric definitions do not automatically become rewards. A single label would
conflate availability, information privilege, credit, and objective impact, so
future M12 tooling records four independent axes:

| Axis | Values |
| --- | --- |
| Availability | `transition_local`, `wrapper_memory`, `episode_terminal`, `offline_only` |
| Information | `actor_observable`, `shared_obs_observable`, `centralized_training_privileged`, `external_annotation` |
| Credit scope | `team`, `agent`, `pair`, `population` |
| Objective effect | `potential_based_candidate`, `objective_changing`, `unknown`, `unsuitable` |

Each proposed component additionally records whether it is available on the
JAX hot path and its known reward-hacking risks.

The following classification covers every current retained family. It states
what an experiment could access, not what MARL-BattleGrounds recommends as a
default reward.

| Metric IDs | Availability | Information | Credit | Objective effect and principal risk |
| --- | --- | --- | --- | --- |
| `marlbg.task.outcome_distribution.v1`, `marlbg.task.terminal_score_differential.v1`, `marlbg.task.evaluation_return.v1` | `episode_terminal` | `centralized_training_privileged` | team | `objective_changing`; can duplicate or replace canonical task intent |
| `marlbg.task.episode_length.v1`, `marlbg.artifact.completion.v1` | `episode_terminal` | `external_annotation` | team | `unsuitable`; optimizing duration/completion can reward premature endings or infrastructure artifacts |
| gross damage/healing, realized health, clamp overflow, and death-count IDs | `transition_local` | `centralized_training_privileged` | agent or team | `objective_changing`; throughput farming and suicidal trades require outcome companions |
| lethal-transition contribution and single-/multi-contributor IDs | `transition_local` | `centralized_training_privileged` | agent or team | `objective_changing`; outcome-conditioned credit can encourage last-transition crowding |
| `marlbg.coordination.focus_fire_concentration.v1` | `transition_local` | `centralized_training_privileged` | team | `unknown`; concentration is gameable and target quality is absent |
| activation and cooldown IDs | `transition_local` | `actor_observable` for own values | agent | `objective_changing`; casting or holding is not inherently good |
| `marlbg.movement.phase_displacement.v1` | `transition_local` | `centralized_training_privileged` | agent | `unknown`; distance can reward purposeless motion |
| status application/lifecycle and controlled-recipient IDs | `transition_local`; `wrapper_memory` only for longer windows | `centralized_training_privileged` | agent, pair, or team | `objective_changing`; raw uptime/application can reward low-value control |
| Mage, Warrior, Hunter, Rogue, and Priest class IDs | `transition_local`; `wrapper_memory` for status episodes | `centralized_training_privileged` | agent or team | `objective_changing` or `unknown`; each requires its exposure and task outcome companions |
| regeneration, respawn-wave, spawn-shield, action-rejection, and ally-distance IDs | `transition_local` | privileged facts/frames or `external_annotation` for QC | agent or team | `unsuitable` for canonical shaping; easily rewards stalling, dying, rejection suppression, or arbitrary spacing |
| future TDM/KoTH/CTF candidate keys | task-owned after implementation | task-dependent | team, agent, or pair | `unknown` until task reward and opportunity semantics are authoritative |
| scenario-owned behavior | `offline_only` | `external_annotation` plus replay | scenario subject | `unsuitable` for generic rollout shaping; may become a separately designed curriculum objective |
| population, learning, and runtime IDs | `offline_only` | `external_annotation` | population | `unsuitable`; not transition credit signals |

“Actor observable” means the actor legitimately receives the required
same-epoch or adjacent own values; it does not authorize privileged team facts
as actor input. Any implementation promotes an ID to a shaping component only
through its own versioned configuration and reward-hacking review.

The canonical task reward and each named shaping component remain separately
logged. Host Pydantic models, sparse events, metric formulas, and replay files
never feed the training hot path.

## Consolidated disposition index

This index consolidates aliases from the design PDF and metric brainstorm. The
private M6 source ledger preserves the line-level source trace.

| Candidate family or alias | Final disposition | Canonical replacement or reason |
| --- | --- | --- |
| Win/loss/draw, win rate | Retain, primary | One multinomial outcome distribution |
| Score, score differential, final margin | Retain one dominance endpoint | Terminal score differential; omit redundant fields per task |
| Return | Retain, secondary | Evaluation return separated from training/shaped return |
| Episode length, time to win/defeat | Advanced | Completion-aware duration; conditional times require outcome exposure/censoring |
| Comebacks, recovered deficits, surrendered leads, lead changes, first-score effects, close/decisive wins | Advanced descriptive | Score-trajectory slices; post-treatment and gameable, never primary |
| Kills, last hits, K/D, general elimination participation, solo kills, pentakills | Reject | Lethal-transition damage-source truth, single-/multi-contributor lethal transitions, team wipes |
| Damage dealt/taken | Retain | Recipient-modified gross stages with exact units and shares |
| “Effective healing,” overheal | Correct and retain selectively | Gross healing primary; recipient-level net/clamp outcomes advanced; no arbitrary source realization |
| Damage/healing per engagement or teamfight | Reject | No generic engagement/teamfight segmentation is planned; use authoritative task context or controlled scenarios for a named question |
| Damage by target class/status/ability/context | Retain as dimensions | Long-form slices of one amount metric |
| Focus-fire events/concentration | Conditional key secondary | Same-transition eligible multi-attacker concentration plus exposure |
| Contributors per death, multi-agent deaths | Retain with narrowed meaning | Lethal-transition source-count distribution and multi-contributor lethal-transition rate |
| Teamfight W/D/L, participation, damage, healing, CC, numerical advantage | Reject | No generic teamfight detector is planned and no universal strategic win definition exists |
| Damage mitigation/amplification | Retain combined value | Exact team/class combined effect; duplicate emitter credit blocked |
| Survival rate or deaths per engagement/teamfight | Reject | Depends on the rejected generic segmentation and encourages unsupported quality claims |
| Time alive/dead, life duration, respawn time, wave size | Advanced diagnostics | Authoritative lifecycle quantities, not compact tactical-quality endpoints |
| Critical-health escapes | Scenario/advanced association | Low-health state alone does not prove skill |
| Lethal/clutch saves | Retain corrected definition | Same-transition lethal-damage rescue; critical-health survival is a separate association |
| Healing by recipient class, share, received | Retain as dimensions | Gross-healing long-form slices |
| Ally deaths in heal radius / while Ultimate ready | Reject as headline | Exposure and counterfactual quality are undefined; scenario if needed |
| CC duration/uptime/applications/lifecycle | Retain advanced | Channel-specific; no universal CC score |
| Damage/death under CC | Retain advanced association | Do not claim causal combo quality |
| Trap casts/triggers/placed uptime | Correct | Targeted applications, active steps, lifecycle episode endings, damage-break rate |
| Anti-heal uptime/healing prevented | Retain combined value | Preserve healing exposure; no per-Rogue credit under overlap |
| Ultimate use/conversion/kill within K | Activations retained; conversion conditional | Mechanic-native active-window association only |
| Burst uptime/damage/deaths/waste | Retain active-window amounts; reject waste counterfactual | No hypothetical attacks or “kills caused” wording |
| Freedom uptime/effective time/movement recovered | Retain binding coverage; scenario for utility | No counterfactual distance fact |
| Warrior tanking, Hunter low damage taken | Retain descriptive exposure | Not proof of tanking quality or kiting |
| Peeling, kiting, flanking, body blocking, backline access, cornering, LOS/choke use, escape denial | Scenario | Context-dependent geometry behavior |
| Regeneration/recovery | Retain advanced diagnostic | Separate from Priest healing; mobile class-specific mechanic |
| KoTH control/contest/score/captures/defenses/allocations | Future task owner | Eligible hill-timestep denominators; no per-agent score apportionment |
| Objective fight cost deaths/score | Reject | Unstable at low/zero score and strategically ambiguous |
| CTF pickups/captures/drops/returns/possession/interceptions | Future task owner | Exact task-authored edges and opportunity denominators |
| Preferred carrier quality | Reject | Carry-by-class is descriptive; do not encode designer preference |
| Escort/bodyguard/kill-squad size | Advanced descriptive | Scenario owns decision quality |
| Local numerical advantage, cohesion, pairwise distance, centroids | Advanced descriptive/scenario | Not universal quality and no new core fact |
| Action acceptance/rejection | Diagnostic | Upstream policy/sampler and legality quality control |
| Temporal slices while ahead/behind/contesting | Retain as dimensions | Do not create primitive metric explosion |
| Cross-play gap, partner robustness, worst partner, ratings | Retain with corrections | Frozen pools/full matrix; lower-tail not raw minimum; ratings secondary |
| Sample efficiency and runtime | Retain protocol-owned | Fixed held-out evaluations and declared hardware/JIT protocol |
| Opaque tactical/coordination score | Reject | Conflicts with interpretability and hides behavior tradeoffs |

## Verification requirements

Before a metric becomes active, its owner must provide:

- hand-constructed neutral, positive, negative, and zero-opportunity traces;
- team-swap and side-swap invariance where applicable;
- complete/partial/interrupted/failed rollout tests plus independent observed,
  right-censored, competing-event, unavailable, and not-applicable endpoint
  tests where relevant;
- simultaneous damage/healing/clamp and duplicate-class overlap cases;
- live-model and replay-loaded parity once file replay exists;
- raw sufficient-statistic and presentation-roundtrip checks;
- adversarial gaming/counterexample review;
- construct validation for any coordination or scenario interpretation; and
- explicit Four-North-Star verdicts.

No unresolved `blocked` or `validation_pending` row may appear as an official
paper endpoint, benchmark score, reward preset, or release claim.
