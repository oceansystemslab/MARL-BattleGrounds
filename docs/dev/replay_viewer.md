# Replay Viewer

The Replay Viewer is the read-only browser product for immutable semantic
replays, checked demonstration samples, and scripted demonstrations
materialized as replay bundles. It always uses the fixed Analysis presentation
and cannot stage actions, submit a simulator transition, reset an episode, or
record to a user-selected destination.

Manual live work belongs to the [Combat Debugger](combat_debugger.md).

## Select exactly one input

Every invocation must choose exactly one artifact, sample, scripted scenario,
or list operation:

| Selector | Purpose |
| --- | --- |
| `--replay PATH` | Validate and open a local canonical-format replay bundle. |
| `--sample-replay NAME` | Verify and open one checked-in sample by stable launch name. |
| `--scenario NAME` | Materialize one registered scripted demonstration in isolation, then open its validated bundle. |
| `--list-scenarios` | List default scripted demonstrations and exit. |
| `--list-sample-replays` | List checked-in sample names and descriptions and exit. |

Open a local artifact:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json
```

Choose the initial frame and audience without opening the browser
automatically:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json \
  --frame-index 12 --view pov --pov-slot 5 --no-open
```

List and open a checked sample:

```bash
./scripts/dev/run_replay_viewer.sh --list-sample-replays
./scripts/dev/run_replay_viewer.sh \
  --sample-replay death-respawn-shield
```

List and materialize a scripted demonstration:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios
./scripts/dev/run_replay_viewer.sh --scenario stacked_team_auras
```

The public browser options are `--frame-index`, `--pov-slot`,
`--view oracle|pov`, `--ranges`/`--no-ranges`, `--port`, and `--no-open`.
Ranges start hidden; `--ranges` opts in without changing the eight default effects.
`--seed` applies only to scenario materialization. List operations reject
unrelated options. `--static` has its own narrow matrix described below.
Option abbreviations are rejected.

## Scripted-scenario isolation

`--list-scenarios` shows researcher scripted demonstrations and omits the
manual `arena_5v5` laboratory. `stacked_team_auras` follows `aura_crossfire` in
the default catalog and demonstrates simultaneous Basics under two same-team
Mage and two same-team Warrior emitters per team.

Developer visual-stress demonstrations are excluded by default:

```bash
./scripts/dev/run_replay_viewer.sh --list-scenarios --include-stress
./scripts/dev/run_replay_viewer.sh \
  --scenario max_status_stack --include-stress
```

`--include-stress` is only a catalog-discovery/authorization input for
`--list-scenarios` or `--scenario`; it is not a browser selector and is
unavailable with local artifacts or checked samples.

Scenario materialization runs in a temporary child process with
`JAX_PLATFORMS=cpu`. The child executes the registered commands and publishes one
V2 replay containing the captured frames, facts and episode provenance. No metric
sidecar or full metric computation is required. The parent opens those bytes
through the public loader before starting the Replay Viewer.
The read-only viewer process does not import or run simulator control.

### Authoritative-battlefield visual coverage rule

Every authoritative-battlefield visual mechanic must have executable semantic
coverage in both one default researcher scenario and one opt-in stress
scenario. A mechanic is not covered merely because a frame label or scenario
description names it: the scenario tests must derive the corresponding fact
from validated transition events or normalized Scene V2 data. Status mechanics
must additionally prove their stable catalog token through natural expiry.

Any change that adds a battlefield effect, durable badge, status, aura,
ability-family presentation, or lifecycle presentation must extend the regular
and stress trajectories—and the paired semantic coverage contract—in the same
change. Prefer extending an existing coherent trajectory; add a new scenario
only when doing so keeps the demonstration readable. Research-space panels and
controls outside the authoritative battlefield snapshot are not governed by
this scenario-pair rule.

## Checked sample replays

The checked V1 bundle under `examples/replays/v1/` contains exactly three
replay/metric pairs plus `manifest.json`:

| Launch name | Source scenario | Coverage focus |
| --- | --- | --- |
| `death-respawn-shield` | `death_respawn_cycle` | Lethal damage through the first post-shield interaction. |
| `recovery-status-lifecycle` | `recovery_refresh_cycle` | Recovery, rejection, refresh, break, reapply, and expiry. |
| `mirrored-five-class-ultimates` | `mirrored_ultimates` | Reciprocal demonstrations of every class Ultimate family. |

Every sample uses an 18×12 map. The manifest records stable names, source
scenarios, transition/frame counts, event-kind coverage, byte lengths, hashes,
and actual source/runtime provenance. Samples are deterministic unofficial
presentation demonstrations—not benchmarks, policy evaluations, source-tree
attestations, or host attestations.

Verify the complete seven-file set through the public loaders:

```bash
JAX_PLATFORMS=cpu uv run python \
  scripts/dev/generate_visual_debugger_sample_replays.py \
  --check --output-directory examples/replays/v1
```

Generation is maintainer-only and refuses overwrite. Generate into one new,
absent directory only after scenario source, tests, and public documentation
are frozen; verify the complete set before an explicit reviewed publication.
Never hand-edit one member or delete the checked directory merely to bypass the
no-overwrite boundary.

## Transport and exact-frame summaries

| Control | Behavior |
| --- | --- |
| **Start** / **End** | Seek to the first or final captured frame. |
| **−10** / **−1** / **+1** / **+10** | Issue one clamped absolute seek. |
| **Play** / **Pause** | Serialize playback with at most one replay request and one presentation in flight. |
| Frame slider | Preview the target tick locally without a request; commit one exact absolute seek when the value is committed. |
| Tick label | Show the authoritative current and final simulator ticks joined from the recorded timeline. |

Button seeks, committed slider seeks, view/range changes, reconnects, and other
non-play navigation install a settled exact-frame summary. The summary retains
the enabled durable and incoming-transition cues together without animation or
hidden timer state. Pressing Play restarts the displayed incoming transition at
logical time zero when one exists, waits for its scaled presentation to settle,
and then requests the next frame. Each accepted exact successor advance may
animate only that successor's recorded incoming transition.

**Latest Transition** is the recorded incoming transition `T_(n-1)` that
produced the displayed frame `s_n`; it is absent at frame zero. **Upcoming
Transition** is the recorded `T_n` out of `s_n` in the same Submitted /
Accepted row grammar. Both panels remain global researcher-space evidence in
Oracle View and Agent POV: they show every configured-active actor in canonical
roster order and never stage, predict, or execute an action. Agent POV fog of
war applies only inside the battlefield snapshot and its choreography. The
roster likewise remains a global researcher control, partitioning active agents
into **Visible** and **Not Visible** groups so any agent's POV can be selected at
the same replay tick. Upcoming is absent only when the displayed frame has no
recorded successor.

The eight supported rates are exactly **0.25×, 0.50×, 0.75×, 1.00×, 1.25×,
1.50×, 1.75×, and 2.00×**. A rate scales the complete presentation clock,
including animation phases, waits, and the replay terminal hold. It never
changes simulator ticks or artifact contents.

### Document keyboard shortcuts and exclusions

Unmodified Left, Right, and Space are document-level shortcuts for previous,
next, and play/pause. They do not require the battlefield or timeline to be
active. The viewer deliberately issues no replay command when:

- Shift, Control, Alt, or Meta is held;
- Space is an auto-repeat event;
- the event originates in or under a button, input, select, textarea, link,
  disclosure summary, dialog, editable region, or ARIA widget such as a
  slider, textbox, combobox, spinbutton, or menu item; or
- replay authority is absent, offline, hidden, or not yet installed.

Modified shortcuts, events from native controls, and absent, offline, hidden, or
uninstalled replay authority preserve ordinary browser and assistive-technology
behavior. When complete replay authority remains visibly installed, Space
auto-repeat and Left, Right, or Space during a temporary in-flight fence are
consumed without issuing another command. This prevents the document or an open
disclosure from scrolling while the retained replay scene is still active.

## Audiences and recorded authority

Oracle View exposes the full authorized battlefield presentation. Agent POV
applies the selected recipient's fog of war to that battlefield: NoSharedObs
shows the recipient's local view, while SharedObs shows the same-epoch visual
union authorized by its team observations. Activating a visible agent in Replay
Agent POV switches the recipient at the current tick; it never advances or
mutates the artifact. Artifact identity, completion/processing evidence, PNG
export, and metric download remain capabilities of the researcher tool in every
visual POV.

Under
[amendment A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution),
this dual-mode presentation is a reader-compatibility contract, not benchmark
eligibility. NoSharedObs material is historical, diagnostic, or custom evidence
and is noncanonical for new official claims. SharedObs rendering likewise does
not confer official status: official evidence must separately prove the
canonical actor projection and the exact configured-active, same-team,
off-diagonal availability matrix on every recorded frame. The Replay Viewer
continues to render both regimes and does not mutate, migrate, or relabel their
artifacts.

The SharedObs visual union follows
[specification amendment A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).
It may join only recorded same-decision-epoch rows from authorized same-team
sensor sources. It does not recompute geometry, visibility, line of sight,
masks, mechanics, or state; include teammate masks/history, rewards, policy or
critic state, transition facts, or hidden Oracle truth; or claim to be a
materialized SharedObs learner input.

A separate server-authorized corpse overlay keeps locally visible dead bodies
consistent with Oracle View. Python admits a corpse only from the same epoch
and only when an authorized living sensor has it within recorded observation
radius and static line of sight. The overlay is used for corpse painting and
inspection and may additionally admit only a death/respawn presentation cue
and that cue's owned endpoint. It never admits another event, moves an ability
route, or changes policy input, masks, targeting, actions, simulator or recorded
transition semantics, or the replay artifact.

### Technical Frame

Replay Oracle uses an exact five-leaf Technical Frame allowlist:

1. Artifact digest prefix
2. Frame
3. Simulator step
4. Incoming transition, omitted at frame zero
5. Ordinary movement distance scale

Agent POV receives only Frame, Simulator step, and its conditional authorized
Incoming transition. It never receives the canonical artifact digest or
movement scale through this panel. Completion and Processing remain distinct
rollout/host-processing facts on their authorized replay surface; neither is a
Technical Frame expansion.

## Visual filters

Visual Filters contains 19 browser-local controls plus Ranges. Initially
enable Ultimate Ability Effects, Spawn Shield, Basic Ability Effects,
Regeneration Effects, Death Effects, Resurrection Effects, Scrolling Battle
Text and Respawn Wave. Ranges and Death Announcer start off, so the initial
count is `8 enabled`. Visual Filters and Roster start open. **Default** restores
this configuration. The complete filter inventory is:

1. Aura Fields
2. Aura Modifier Badges
3. Duration Status Badges
4. Spawn Shield
5. Target Selection Visuals
6. Basic Ability Effects
7. Ultimate Ability Effects
8. Regeneration Effects
9. Cooldown Effects
10. Status Application
11. Natural Status Expiry
12. Freezing Trap Break
13. Status Clear on Death
14. Death Effects
15. Respawn Wave
16. Resurrection Effects
17. Spawn-Shield Expiry
18. Scrolling Battle Text
19. Death Announcer

Death Announcer displays concise authoritative death notifications while the
replay advances. It follows the existing presentation clock and clears when
seeking or reinstalling a settled frame.

Duration Status Badges includes the white crossed-swords **In Combat** countdown.
Basic Ability Effects and Ultimate Ability Effects each own their corresponding
activation presentation and damage/healing impact glyphs. Scrolling Battle
Text owns the complete net-health unit: outcome glyph, signed value, recipient
label, and connector. Those parts are enabled or disabled together; damage and
healing remain distinguished by their outcome sign and color rather than by
separate filters.

A filter change pauses playback and reinstalls the current settled summary
after filtering, so disabled paint never consumes layout space. Filters and
Ranges do not change authorized data or authorized event data used by
battlefield choreography. **Enable All** enables all 19 controls plus Ranges;
**Disable All** disables all 20 visible controls.

## PNG export and metrics

**Export PNG** is enabled only when one coherent replay frame is connected,
visible, settled, and free of pending replay/presentation work. It exports the
battlefield alone—not the toolbar, timeline, or inspectors—at exactly twice
its displayed pixel dimensions. The result uses the bundled fonts and locked
battlefield background, reflects the current audience, selection, Ranges, and
visual-filter states, and embeds one canonical
`MARL-BattleGrounds Replay Provenance` iTXt record. Export does not navigate the
replay or request another replay frame.

The scoreboard derives the task, participant identities, current scores and
configured target from the captured episode. Victory, draw and defeat appear
only when the current frame contains the task completion event. Seeking backward
removes the later result. Legacy task-zero replays say **Combat diagnostic**.
Changing between Oracle and Agent POV preserves the cursor and playing intent.

Open **TDM Evaluation Metrics**, below Comprehensive Agent Class Details, to
prepare the offline analysis once. **Up to Current Tick** is the default: local
frame k uses exactly k captured transitions. The displayed tick comes from the
captured simulator state, which may start above zero. **Final Episode** selects the complete
captured prefix; interrupted recordings do not become completed results.
An indeterminate progress indicator appears while metrics are prepared. Longer
replays can take longer; the indicator makes no fixed-duration promise. Playback,
seeking and POV changes remain available while analysis prepares. CSV preparation
also shows an indicator when the metrics panel is closed.
The selector groups the complete TDM scalar catalog into readable families and
shows the selected family's definition outside the scrolling table. Rows show
**Subject**, **Measure** and **Value**, with teams first and numeric agent IDs
using their recorded classes. Unavailable values appear as a dash; hover or focus
a measure for its definition, units and missing-value rules. Formation summaries
include mean distance and the number of eligible pair observations.
Conditional coordination/class associations remain descriptive, not validated
proof of strategic reasoning.

**Download Metrics CSV** exports one wide row for the selected boundary, using
the same scalar names, order and values as the run tables; unavailable cells are
empty. Boundary provenance includes scope, local frame index, actual simulator
tick and captured roster/policy identities. **Episode Details** downloads the
recorded episode, policy, completion and runtime metadata as JSON without copying
the trajectory or requiring a metrics sidecar. Older V1 replays can be analyzed
from captured facts without rewriting their artifacts. Derived analysis records
its own source fingerprint.
Metrics remain researcher analysis in every visual POV and never enter policy
inputs, recipient tooltips or battlefield visibility. Preparing analysis does not
hold the replay command lock; subsequent seeks reuse cached prefix components.
Metric computation uses the captured game directly and retains inexpensive
consistency checks. Imported artifacts still receive the input checks described
below; computing metrics does not repeatedly revalidate that same data.

## Static Matplotlib frame

Render one exact frame without opening the browser or starting an HTTP server:

```bash
./scripts/dev/run_replay_viewer.sh \
  --replay episode.marlbg-replay.json --static --frame-index 12

./scripts/dev/run_replay_viewer.sh \
  --sample-replay recovery-status-lifecycle --static --frame-index 3

./scripts/dev/run_replay_viewer.sh \
  --scenario stacked_team_auras --static --frame-index 1
```

`--frame-index` is required with `--static`. Only the selected input, range
state, and—for a scripted scenario—seed/stress authorization are accepted in
this mode; browser audience, POV, port, and opening options are rejected. The
shell activates the optional `viz` dependency. Static mode validates the
complete bundle and paints the exact researcher frame through the stateless
scene-native Matplotlib adapter.

## Loopback and artifact safety

Local artifacts, checked samples, and materialized replays pass whole-artifact
validation before the server binds or a browser is opened. Historical V1
companions are checked when present or required by the checked-sample contract;
current V2 replays are self-contained.
Invalid schemas, canonical bytes, hashes, event/frame joins, frame indices,
POV recipients, symlinks, and unsupported paths fail closed.

The browser server binds only to `127.0.0.1`, uses a random fragment-delivered
capability token, validates request headers and origins, serves an explicit
asset allowlist under restrictive Content Security Policy and `no-store`, and
serializes revisioned/idempotent commands. Replay commands change only the
selected recorded authority; none can call `step` or modify the source files.

Refresh and **Reconnect** fetch current authority without repeating a seek.
Closing the tab does not stop Python; use **Exit Replay Viewer** or `Ctrl-C`.

## Troubleshooting

- **No selector:** choose exactly one replay, sample, scenario, or list
  operation.
- **Artifact rejected before a URL appears:** inspect the reported path,
  canonicality, companion, frame, or POV error; the launcher intentionally did
  not bind a server.
- **Manual arena selected as a scenario:** open it with the
  [Combat Debugger](combat_debugger.md).
- **Stress scenario rejected:** add `--include-stress` to the scenario command.
- **Transport disabled:** reconnect if offline; Start/negative moves stop at the
  lower bound, positive moves/End stop at the captured endpoint, and artifact
  actions wait for a settled frame.
- **Metrics unavailable:** inspect the analysis panel's error. A missing V1
  metrics sidecar does not prevent analysis from recorded facts or the Episode
  Details download; current V2 recordings do not need a sidecar.
- **PNG export disabled:** pause playback and wait for the exact-frame summary
  to settle in a visible connected tab.
- **Static Matplotlib import failed:** run `uv sync --extra viz`.
- **Server remains after tab closure:** use the in-page Exit action or `Ctrl-C`.

The replay format itself is documented in
[replay_format.md](../evaluation/replay_format.md). Return to the
[browser-tools migration page](visual_debugger.md) or the
[project README](../../README.md).

Current DevClient and scripted-scenario recording captures the replay without
computing full diagnostics. Opening Evaluation Metrics or requesting CSV opts
into full replay analysis once; the result is cached across cursor and POV
changes. Library callers likewise opt in with `analyze_replay(bundle, full=True)`.
Research training and evaluation independently select priority/full metrics and
replay saving through the [evaluation workflow](../evaluation/workflows.md).
