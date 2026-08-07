# Visual Debugger and Analyzer — Visual Acceptance Evidence

Status: **ready for human review; Milestone 5 remains open until the user
explicitly accepts the visuals.**

This gallery is the review index for the Milestone 5 surgical polish pass.
Every link opens the committed image at its original captured resolution.
Simulator-backed scenarios are identified separately from synthetic
renderer-only fixtures.

Only the repository-owned images linked from this page belong to the
acceptance set. Temporary `/tmp` captures used to compare historical Git
baselines are deliberately excluded, so an older presentation cannot be
mistaken for current evidence. The Warrior shield source path was retained
unchanged during this polish pass.

## Start here

- [Five-class icon and action vocabulary][vocabulary] — all class crests,
  fallback letters, target selection, observation/Basic/Ultimate ranges,
  cooldowns, class-owned Basics, distinct Ultimates, damage, healing, team
  rings, and tint-only auras.
- [Canonical durable Stun and Slow vocabulary][durable-controls] — the same
  stun glyph across Warrior/Hunter/Rogue channels and the same swirl across all
  Slow channels, while source-class accents remain distinct.
- [Professional joint-action console at 1440×900][console-desktop] and
  [960×600][console-minimum] — the numbered Movement/Action composer, dedicated
  Submit rail, and quieter Inspect/Session utilities.
- [Ten independently staged agent rows before one submit][joint-turn] — the
  authoritative pending inventory used by the single joint transition.
- [Maximum supported status density][max-status] and
  [minimum-viewport crowded combat][crowded] — collision, association,
  duration, cooldown, modifier, and NET-outcome stress evidence. During active
  minimum-viewport choreography, accepted routes and exact NET outcomes take
  battlefield priority while the roster and tooltip retain every exact
  status. Actor-labelled `id_N / +N` summaries with leaders return as soon as
  choreography settles or is skipped.

## Class-icon review list

| Class | Current visual identity | Review evidence |
| --- | --- | --- |
| Mage | Cyan arena star with `M` fallback | [Contact sheet][vocabulary] |
| Warrior | Amber shield with `W` fallback | [Contact sheet][vocabulary] |
| Hunter | Lime bow, straight string, and arrow with `H` fallback | [Contact sheet][vocabulary] |
| Rogue | Yellow crossed blades with `R` fallback | [Contact sheet][vocabulary] |
| Priest | Magenta equal-arm medic cross with `P` fallback | [Contact sheet][vocabulary] |

The magenta corner reticle is the class-independent selected-target treatment.
The contact sheet deliberately places each crest beside its Basic and Ultimate
grammar so an icon change can be judged in tactical context rather than in
isolation.

## Requirement-to-evidence matrix

| Requested visual result | Automated proof | Original-resolution visual evidence |
| --- | --- | --- |
| Fallback letters sit below, rather than on top of, class icons | `renderer-fixture.spec.js` measures distinct icon/letter bounds | [Five-class vocabulary][vocabulary] |
| Hunter uses a bow/string/arrow and is visually separate from healing | Icon vocabulary assertions; Hunter palette assertion | [Five-class vocabulary][vocabulary] |
| Priest uses a thick symmetric medic cross | Icon path/geometry assertions | [Five-class vocabulary][vocabulary] |
| All durable Stuns share the Warrior stun glyph | Exact token-family DOM and vocabulary assertions | [Canonical control vocabulary][durable-controls] |
| All durable Slows share one swirl | Exact token-family DOM and vocabulary assertions | [Canonical control vocabulary][durable-controls] |
| Ultimate application symbols remain class-specific | Five activation-token and effect-signature cases | [Mage][ultimate-mage], [Warrior][ultimate-warrior], [Hunter][ultimate-hunter], [Rogue][ultimate-rogue], [Priest][ultimate-priest] |
| Icon and duration/value compartments never overlap | DOM bounding-box assertions for ordinary, maximum-density, and required-fallback layouts | [Maximum status stack][max-status], [required-dock fallback][required-fallback] |
| Human-facing floats use no more than two decimals | Visible-text precision scan and formatter unit cases | [Number-format fixture][number-format] |
| Overflow is neutral rather than class-colored | CSS/token assertions and tooltip inventory assertion | [Neutral overflow tooltip][overflow-tooltip] |
| Cooldown cue appears only above zero with exact ticks | Cooldown presence/absence and compartment assertions | [Five-class vocabulary][vocabulary] |
| Mage and Warrior auras are tint-only, without borders | SVG stroke-absence assertions | [Five-class vocabulary][vocabulary], [focus-fire aura pressure][focus-fire] |
| Both team rings are solid; Team B retains a non-color marker | Computed-stroke and Team-B chevron assertions | [Five-class vocabulary][vocabulary] |
| Observation is white dotted, Basic is class-colored dashed, and Ultimate is purple dash-dot | Computed SVG stroke/dash assertions | [Five-class vocabulary][vocabulary] |
| Basic routes use the acting class color | Token, marker, and static-painter color assertions | [Five-class vocabulary][vocabulary], [static fallback][static-vocabulary] |
| Damage ends in a red minus; healing ends in a green plus | Impact semantic unit tests and DOM marker assertions | [Five-class vocabulary][vocabulary], [moving Basic crossfire][moving-basic] |
| Target selection is clearly separate from controlled state | Selected-reticle and controlled-halo assertions | [Five-class vocabulary][vocabulary] |
| Aura multiplier values do not overlap icons | Modifier-cell geometry assertions | [Maximum status stack][max-status], [crowded minimum viewport][crowded] |
| Repeated/focus damage and coordinated healing remain readable | Exact activation, route, recipient-NET, and collision assertions | [Focus fire and healing][focus-fire], [moving focus at 960×600][moving-focus] |
| Close, reciprocal, crossing, and Charge routes keep source-to-recipient direction | Route tangent/unit tests and scenario endpoint assertions | [Moving Basic crossfire][moving-basic], [Charge convergence][charge-convergence] |
| Trap damage impact, application, exact break, break plus reapplication, ambiguous end, and expiry remain distinct | Activation-impact grammar, lifecycle classification, and event-ID assertions | [Applied][trap-applied], [exact break][trap-break], [damage plus reapplication][trap-reapplication], [ambiguous end and expiry][trap-end] |
| One tooltip wins overlaps, clamps to the viewport, and exposes full neutral overflow | Tooltip arbitration, content, leave, focus, edge, and POV tests | [Agent over broad fields][agent-tooltip], [edge-clamped overflow][overflow-tooltip] |
| Agent POV omits hidden entities and route endpoints server-side | Python payload projection tests and browser DOM absence assertions | [Redacted POV fixture][pov] |
| Movement scale is exact and professionally integrated | Protocol/service zero-step reset tests and slider E2E | [`0.10` override][scale-010], [authored default restored][scale-default] |
| Unavailable actions are visible, dim/red, explained, and inert | Mask-parity, no-mutation, and disabled-control E2E | [Minimum-width command console][console-minimum] |
| All agents can be staged before one Enter/Submit transition | Exactly-one-step Python/service proof and ten-row E2E | [Ten-agent pending inventory][joint-turn] |
| Movement/Action/Inspect/Session controls have a clear professional hierarchy | Responsive order, primary-submit, focus, and overflow assertions at both supported viewports | [1440×900 console][console-desktop], [960×600 console][console-minimum] |
| The controlled actor remains exact and prominent in the composer | Roster-control, Shift-click, viewport-clipping, and live-label assertions | [1440×900 console][console-desktop], [960×600 console][console-minimum] |
| Reduced motion and Motion Off retain the same truthful event inventory without travel animation | Choreography-controller lifecycle, static-identity, bounded-cleanup, and DOM event-count assertions | [Reduced-motion mixed outcome][reduced-motion], [Motion-Off route batch][motion-off] |
| Static/headless Matplotlib remains semantically aligned without claiming browser-quality parity | Focused scene-painter assertions and static exporter test | [Static visual vocabulary][static-vocabulary] |

## Scenario evidence

### Simulator-backed

- [Moving Basic crossfire][moving-basic] — all five classes participate across
  changing successor anchors.
- [Moving focus fire and healing at 960×600][moving-focus] — simultaneous
  convergence, damage, healing, auras, and exact recipient NET.
- [Four-source focus fire plus three healers][focus-fire] — opposing intent
  remains directional without per-source numeric attribution.
- [Three converging and reciprocal Charges][charge-convergence].
- [Maximum status stack][max-status].
- Mirrored Ultimates:
  [Mage Burst][ultimate-mage],
  [Warrior Charge][ultimate-warrior],
  [Hunter Trap][ultimate-hunter],
  [Rogue Poison][ultimate-rogue], and
  [Priest Holy Word][ultimate-priest].
- Trap lifecycle:
  [application][trap-applied],
  [exact break][trap-break],
  [break plus reapplication][trap-reapplication], and
  [ambiguous ending beside natural expiry][trap-end].

### Synthetic presentation fixtures

These images test renderer contracts only; they are not represented as
simulator histories.

- [Five-class visual vocabulary][vocabulary].
- [Canonical durable controls][durable-controls].
- [Crowded minimum viewport][crowded].
- [Required-dock measured fallback][required-fallback].
- [Human number formatting][number-format].
- [Agent tooltip over overlapping broad fields][agent-tooltip].
- [Neutral overflow tooltip at the viewport edge][overflow-tooltip].
- [Strict POV redaction][pov].
- [Reduced-motion mixed damage/healing][reduced-motion].
- [Motion-Off static route batch][motion-off].

The Matplotlib artifact is compatibility evidence for the retained
scene-native, static/headless path. Its axes-based composition is intentionally
not cited as evidence of the browser product's responsive layout or animation
quality.

## Human review checklist

Review the linked originals at normal zoom and answer these questions:

1. Are the five class crests acceptable, or should a named crest change?
2. Are canonical Stun and Slow consequences immediately recognizable while
   their source-class accents remain clear?
3. Is the joint-action console sleek enough for sustained researcher use at
   both supported viewport sizes?
4. Can every shown source, recipient, action family, NET outcome, and durable
   consequence be followed without relying on terminal logs?
5. Does any number, icon, route, status dock, cue, or tooltip visibly collide
   or imply the wrong mechanic?

Automated checks and Codex review do not substitute for this explicit human
acceptance.

[agent-tooltip]: ../../web/visual_debugger/e2e/tooltip.spec.js-snapshots/agent-tooltip-overlapping-fields-960x600-linux.png
[charge-convergence]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/charge-convergence-t1-mid-impact-1440x900-linux.png
[console-desktop]: ../../web/visual_debugger/e2e/control-parity.spec.js-snapshots/command-console-1440x900-linux.png
[console-minimum]: ../../web/visual_debugger/e2e/control-parity.spec.js-snapshots/command-console-960x600-linux.png
[crowded]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/crowded-teamfight-synthetic-mid-impact-960x600-linux.png
[durable-controls]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/durable-control-vocabulary-synthetic-settled-1440x900-linux.png
[focus-fire]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/team-focus-crossfire-t3-mid-impact-1440x900-linux.png
[joint-turn]: ../../web/visual_debugger/e2e/control-parity.spec.js-snapshots/joint-turn-ten-agent-pending-inventory-1440x1600-linux.png
[max-status]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/max-status-stack-t1-settled-1440x900-linux.png
[moving-basic]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/moving-basic-crossfire-t1-mid-impact-1440x900-linux.png
[moving-focus]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/moving-focus-crossfire-t1-mid-impact-960x600-linux.png
[motion-off]: ../../web/visual_debugger/e2e/combat-choreography.spec.js-snapshots/motion-off-static-route-batch-1440x900-linux.png
[number-format]: ../../web/visual_debugger/e2e/renderer-fixture.spec.js-snapshots/human-number-formatting-synthetic-1440x900-linux.png
[overflow-tooltip]: ../../web/visual_debugger/e2e/tooltip.spec.js-snapshots/neutral-status-overflow-tooltip-edge-clamped-960x600-linux.png
[pov]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/pov-redaction-synthetic-debug-mid-impact-1440x900-linux.png
[required-fallback]: ../../web/visual_debugger/e2e/renderer-fixture.spec.js-snapshots/required-dock-fallback-focus-synthetic-1440x900-linux.png
[reduced-motion]: ../../web/visual_debugger/e2e/combat-choreography.spec.js-snapshots/reduced-motion-mixed-net-1440x900-linux.png
[scale-010]: ../../web/visual_debugger/e2e/control-parity.spec.js-snapshots/movement-scale-010-override-960x600-linux.png
[scale-default]: ../../web/visual_debugger/e2e/control-parity.spec.js-snapshots/movement-scale-default-restored-960x600-linux.png
[static-vocabulary]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/static-renderer-visual-vocabulary-1440x900.png
[trap-applied]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/trap-lifecycle-t1-applied-1440x900-linux.png
[trap-break]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/trap-lifecycle-t2-broken-1440x900-linux.png
[trap-end]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/trap-lifecycle-t5-ambiguous-and-expired-1440x900-linux.png
[trap-reapplication]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/trap-lifecycle-t4-broken-and-reapplied-1440x900-linux.png
[ultimate-hunter]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/mirrored-ultimates-hunter-trap-t3-mid-impact-1440x900-linux.png
[ultimate-mage]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/mirrored-ultimates-mage-burst-t1-mid-impact-1440x900-linux.png
[ultimate-priest]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/mirrored-ultimates-holy-word-t5-mid-impact-1440x900-linux.png
[ultimate-rogue]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/mirrored-ultimates-rogue-poison-t4-mid-impact-1440x900-linux.png
[ultimate-warrior]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/mirrored-ultimates-warrior-charge-t2-mid-impact-1440x900-linux.png
[vocabulary]: ../../web/visual_debugger/e2e/visual-regression.spec.js-snapshots/visual-vocabulary-synthetic-mid-impact-1440x900-linux.png
