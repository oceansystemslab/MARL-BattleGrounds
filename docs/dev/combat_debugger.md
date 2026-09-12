# DevClient and Combat Debugger

The DevClient is the local developer workspace for MARL-BattleGrounds. It has
three deliberately small areas: Combat Debugger, Maps, and Scenarios. Use the
Combat Debugger to inspect same-epoch authority, compose simultaneous actions,
load an execution-valid saved scenario, submit transitions, and optionally
record one manual episode. Use Maps and Scenarios to author local reusable maps
and complete Team Deathmatch starting states without creating a second
simulator in the browser.

Scripted demonstrations, checked samples, and existing replay artifacts belong
to the separate [Replay Viewer](replay_viewer.md). It receives no DevClient
navigation or authoring authority.

The task selector currently offers TDM. Its scoreboard shows Team A/Team B
controller identities, authoritative scores and the configured target, including
loaded scenarios. Task completion adds each team's victory, draw or defeat.
Legacy neutral debugging configurations are labelled Combat diagnostic.
Recording preserves authoritative replay facts without running the full metric
suite. Metric inspection and CSV export belong to Replay Viewer and are computed
on request from that capture.
Finishing a recording displays an indeterminate progress indicator while the
recording is prepared and saved. Longer recordings can take longer; the indicator
makes no fixed-duration promise. Ordinary live steps do not compute the full suite.

## Launch

Open the default Oracle view:

```bash
./scripts/dev/run_dev_client.sh
```

Useful launch variants:

```bash
# Start with one active global slot selected and hide ranges.
./scripts/dev/run_dev_client.sh --controlled-slot 5 --no-ranges

# Start in the selected actor's authorized POV.
./scripts/dev/run_dev_client.sh --view pov --controlled-slot 5

# Print the loopback URL without asking the operating system to open it.
./scripts/dev/run_dev_client.sh --no-open --port 8123

# Show the executable CLI contract.
./scripts/dev/run_dev_client.sh --help
```

The launcher resolves the repository from its own location, binds only to
`127.0.0.1`, chooses an ephemeral port by default, prints the URL, and attempts
to open a modern browser. Node.js and npm are not runtime requirements.

`run_debug_renderer.sh` remains a thin compatibility redirect to this launcher.

The public options are:

| Option | Meaning |
| --- | --- |
| `--record-replay PATH` | Record one manual episode to a self-contained V3 replay, then offer read-only review. |
| `--seed N` | Set the deterministic reset/step seed; default `0`. |
| `--controlled-slot N` | Select an initially active global slot; otherwise use the arena default. |
| `--static` | Render one stateless Matplotlib reset snapshot without a browser server. |
| `--no-open` | Print the URL without opening a browser automatically. |
| `--port N` | Select a loopback port; `0` requests an ephemeral port. |
| `--view oracle\|pov` | Select the initial authorization; default `oracle`. |
| `--ranges` / `--no-ranges` | Show or hide controlled-actor ranges initially; default hidden. |

Option abbreviations are rejected. Replay, sample, scripted-scenario,
frame-index, and replay-POV-slot options are rejected with the Replay Viewer
launcher named in the error.

## Maps and scenarios

The Maps and Scenarios areas share one small authoring surface: an object or
roster list, the SVG map, an exact numeric inspector, and linked host problems.
Opening either area immediately creates its prompt-free untitled draft. Choose
a lowercase snake-case durable asset ID, such as `training_map_10`, on the first
Save; later Saves update that exact asset under its revision fence. The visible
Name field remains free-form. Drag centers for quick placement, hold Alt to
bypass the fixed 0.5-world-unit snap, use arrow keys for exact nudging, and use
the inspector for dimensions, wall size/rotation, roster, episode, current
state, name, description, and ordinary notes. Experiment hypotheses, evidence
roles, seed schedules, and measurements belong to a separate future evaluation
definition, not the physical scenario document. The mouse wheel zooms only the
authoring canvas; Space-drag or middle-button drag pans it. **Recenter** restores
the complete-map view without changing the draft. **Reset (R)** restores the
latest New, Open, or successfully saved baseline and can be undone immediately.
The `R` shortcut does not fire while typing in a form control. Ten spawn pads
are fixed identities. New walls and pillars receive `obstacle_0`, `obstacle_1`,
and so on. Their IDs are editable directly under the general safe Object ID
rules; deletion does not renumber survivors, so an author may explicitly close
a numbering gap. Obstacles also have duplicate, delete, and up/down order
controls.

Drafts are local files under ignored `artifacts/dev_client/` storage. Each
compact selector includes every applicable latest saved revision, orders exact
asset IDs numerically (`map_9` before `map_10`), and uses the native browser
control for scrolling and keyboard navigation. The status line names the exact
saved revision and repository-relative path, for example
`artifacts/dev_client/drafts/maps/training_arena/r2.json`, and reports unsaved
changes without hiding the last truthful location. Save is explicit; there is
no autosave. A successful Save persists through DevClient shutdown and restart
and refreshes every applicable selector immediately. Save uses an exact
revision fence, so an older browser cannot overwrite a newer revision.
**Delete Saved** removes the selected map or scenario identity and all of its
saved revisions only after an explicit confirmation. A stale revision cannot
delete newer work. If the deleted asset is open, its current browser content
remains as an unsaved copy that may be saved again; deletion does not interrupt
an already loaded Combat snapshot.

A new scenario can start blank, copy a saved map, or duplicate a saved
scenario. The adjacent source selector carries the exact saved revision, and
successful Save, Save As, and Delete refresh discovery immediately. Copied map
content is independent: later changes never propagate in either direction. The
browser edits JSON-shaped fields only. Python compiles the whole draft into
existing `EnvConfig` and `EnvState` authorities and runs the existing
validators before any scenario can enter the Combat Debugger. Save is the only
way the DevClient persists asset content, and Delete Saved is its only removal
operation. Experiment and evaluation manifests own any later approval,
partition, or normalized scientific identity.

## Loading saved scenarios and map previews

The Combat Debugger selector lists every latest execution-valid saved map and
scenario revision in numeric-aware asset-ID order. Scenario rows load the
authored starting state. Map rows are explicitly labelled as default 5v5 TDM
previews:
the host independently copies the exact map into the ordinary default scenario,
compiles it, and validates it without modifying or saving the map. `Open in
Debug` in either authoring area calls this same loading service for its current
buffer. Every path strictly parses and revalidates before the current session
changes; failure leaves the current session untouched and returns linked
problems. Reset restores the immutable loaded snapshot and seed, including its
map, roster, scores, timers, and current timestep.

Team A and Team B offer **Manual**, **Reactive TDM ALPHA**, and **Random**.
ALPHA is the existing deterministic, five-class SharedObs policy, with independent movement
and combat priorities. Random samples only the exact current valid action
support and ignores observation features; the same key and mask therefore
produce the same action under SharedObs and NoSharedObs. Neither controller is
an official baseline or Big 12 entrant. There is no policy registry or
checkpoint loader. Under
[amendment A25](../design/specification_amendments.md#a25-sharedobs-only-canonical-benchmark-execution),
SharedObs is the default and the only evaluation-eligible information regime.
NoSharedObs remains available when both teams use Manual or Random, and in
generic custom research. It is disabled while either team uses a reactive
controller; the application never silently substitutes a regime or controller.

Team B additionally offers **Reactive TDM BETA**, a deterministic
SharedObs-only specialist. Mage, Warrior, Hunter
and Priest use Reactive TDM unchanged. Rogue pursues an observed living enemy
Priest first, otherwise Mage, otherwise Hunter, choosing lowest current HP within
the selected class and then lowest global slot. If none is observed alive, Rogue
uses ordinary Reactive TDM movement toward the nearest observed enemy or,
without enemies, map center.

The shared controller admits clear paths and glancing shoulder contact at least
45 degrees from head-on at the first contact with each intervening observed living body.
Self and the selected prey are exempt. Tiny initial overlap within the existing
geometry tolerance counts as contact; deeper overlap permits only non-deepening
outward escape. Outside wall steering, choose the useful legal projected endpoint
closest to prey, with action-order ties and no bonus for avoiding contact entirely;
temporary retreat is allowed and no qualifying move means Stay. Ordinary
simulator collision response resolves the attempted contact. Both scenarios use
`scenario_5` behavior v4, displayed as BETA; the old standalone Scenario 3 controller
is retired, not an alias. Historical recordings retain their original identities.

Combat is independent: Ultimate the lowest-health legal enemy within Basic
radius, otherwise Basic the lowest-health legal enemy there, otherwise no-combat.
The selected prey determines the destination, not necessarily the attack target.
No target memory, hidden prey position, opponent action prediction or guaranteed
route is available. Dead/inactive agents no-op and exact masks override preferences.

ALPHA v2 and BETA v4 add SOUTH-preferred steering around nearby vertical walls,
with NORTH used when the lower passage is unavailable and a turn across the
wall end instead of continued boundary marching. BETA keeps its shoulder
screening: prefer useful wall-phase progress, then distance/action-order ties;
if none exists and an end fits, allow a body-admissible detour. See
[A35](../design/specification_amendments.md#a35-reactive-tdm-wall-steering)
for the precise local rules. This improves tested wall pockets but does not
guarantee navigation: a three-tick ALPHA allied-congestion stall remains known,
and fixed-side routing may take a longer path. Combat and class goals are unchanged.

In a body-blocking study, the defending Warrior can screen a vulnerable Hunter
and provide a moving body around which the Hunter kites, while both maintain
offensive pressure. The specialist can attack that Warrior while pursuing the
Hunter; body avoidance does not imply that it ignores the blocker in combat.
For a TDM finishing scenario, classify the terminal Team A reward, not a
particular movement sequence or minimum attack count: +1 is success, 0 is a
draw, and −1 is failure. A zero reward before termination/truncation is not a
draw. Keep author-observed wins distinct from exact replay-verified witnesses.

To play against a reactive controller:

1. Load a valid scenario or map preview, or keep the default diagnostic arena.
2. Keep Team A Manual and select SharedObs.
3. Select Reactive TDM ALPHA or Reactive TDM BETA for Team B.
4. Stage Team A's actions and Submit; Reset restores the exact starting state.

After editing and saving a scenario, explicitly load its new saved revision to
test that update. Reset restores the already loaded snapshot; it does not pick
up later authoring changes. Saving revised Description/Notes alone does not
change the physical scenario or the controller's behavior identity.

Reactive choices have no task, horizon, roster, score, asset-ID, or respawn
restriction on valid interactive setups. Changing either controller or the
information regime resets the exact loaded snapshot and seed before comparison.
Policy-controlled teams remain inspectable but their action inputs are read-only;
Submit works with both teams policy-controlled. Expected configuration rejection
shows a notice and preserves the session and recording progress without requiring
reconnect. Genuine internal failures retain their fail-closed protections.

The old Scripted TDM scorer, separate Reactive MRP/`scenario_1` executable
interface and standalone Scenario 3 controller are removed. Current reactive
identities are `reactive-team-deathmatch-controller@1` and
`scenario-5-pressure-controller@2`. Scenario 5 combinations retain private
action-source V5; other interactive combinations retain V4. Both record installed
controllers separately from scenario/map identity. Historical recordings retain their old
identities and remain readable; fixed-frame scripted diagnostics are unchanged.
Availability on a setup does not qualify it scientifically or promise the
outcome of another scenario's verified witness.

Loading a scenario does not select its pressure controller automatically, and
saved assets remain controller-independent. Future official evaluations bind
the versioned controller through a separate evaluation definition and apply the
same controller to every treatment and matched-ablation arm. DevClient play
remains diagnostic, not official evidence. See
[A26](../design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations)
and [A33](../design/specification_amendments.md#a33-one-controller-for-scenarios-3-and-5).
Scenario-derived controller rules are not silently eligible for official
training, and originating-scenario performance is not uncontaminated evaluation.

Saved maps and scenarios do not encode an information regime. Loading one
preserves its authored bytes and binds the selected regime only for that run.
New official evidence must bind `shared_obs`, actor projection
`base-observation-plus-authorized-sensor-source-bank@1`, and the exact
configured-active, same-team, off-diagonal availability matrix on every replay
frame. A successful DevClient run or recording is not by itself official
qualification; the evaluation owner applies the separate official evidence
gate.

## Authority and views

Python owns the environment configuration, JAX state and key, observations,
action masks, staged drafts, hit testing, target mapping, legality, accepted
actions, the single authoritative `step`, canonical transition capture, and
the audience-specific presentation root. The browser owns input capture,
pointer-to-world projection, responsive layout, SVG/HTML paint, panels, help,
and presentation timing. Browser-only activity never advances the simulator.

Oracle View can inspect every authorized actor and stages one independent draft
per active slot. A submit precommits all staged actions and applies one
simultaneous joint transition. Agent POV uses the same global researcher
controls and panels, including the complete roster, target selector, Pending
Authorized Draft, Latest Transition, and joint submission. The selected actor
is also the POV recipient. Only the authoritative battlefield snapshot,
hit-testing, ranges, routes, and choreography are filtered by that actor's fog
of war.

The View control may switch between these authorities. Analysis is the only
public presentation: there is no user-selectable density mode. The
rendering-only SharedObs visual-union boundary is defined by
[specification amendment A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation).
It does not provide a materialized SharedObs learner tensor or authorize the
browser to reconstruct geometry, visibility, masks, history, rewards, policy
state, or hidden Oracle facts.

Agent POV also accepts one separate Python-authorized corpse overlay so a dead
body visible to an authorized living sensor remains visible exactly as it is in
Oracle View. That overlay is same-epoch, paint-and-inspection-only evidence; it
may additionally admit only a death/respawn presentation cue and that cue's
owned endpoint. It cannot admit another event, move an ability route, or change
policy input, masks, targeting, legality, accepted actions, or simulator
transitions.

## Manual input and joint turns

Click the battlefield before using live command keys. Inspector controls and
form fields retain ordinary browser keyboard and Tab behavior.

| Input | Oracle View | Agent POV |
| --- | --- | --- |
| Left click an authorized actor | Control that actor. | Control that visible actor and switch to its POV. |
| Activate a Roster row | Control that actor. | Control any active actor and switch to its POV, whether or not its body is currently visible. |
| Shift+left click an active authorized actor | Select it as the controlled actor's target. | Select that visible actor as the controlled actor's target. |
| Target selector | Stage any globally authorized target. | Stage any globally authorized target. |
| `Escape` | Clear the target and leave battlefield command focus. | Clear the target and leave battlefield command focus. |
| `Tab` / `Shift+Tab` | Cycle active actors without discarding drafts. | Cycle active actors and POV recipients without discarding drafts. |
| `W A S D` / arrow keys | Stage cardinal movement. | Stage cardinal movement. |
| `Q E Z C` | Stage diagonal movement. | Stage diagonal movement. |
| `X` | Stage Stay. | Stage Stay. |
| `0` / `1` / `2` | Stage no combat / Basic / Ultimate. | Stage no combat / Basic / Ultimate. |
| `Space` / `Enter` | Submit every staged actor as one joint turn. | Submit every staged actor as one joint turn. |
| `R` | Reset the arena deterministically. | Reset the arena deterministically. |
| `G` | Toggle Oracle controlled-actor ranges. | Toggle fog-authorized controlled-actor ranges without a simulator command. |
| `?` | Open browser help. | Open browser help. |

Each active actor has an independent movement, target, and combat-lane draft.
Changing the controlled actor does not erase the other drafts. The Pending
Action and legality surfaces show the exact same-decision-epoch axis and joint
target/lane result. Latest Transition reports Submitted and Accepted actions
after Python applies the transition.

Network-busy duplicate submissions are blocked. If readable choreography is
still active, Submit settles that presentation before sending the current
revision-fenced draft once. While a live response is still installing, both
views retain at most the first fresh battlefield staging key and one following
fresh Enter, then apply them in that order after the confirmed update. Once
Enter is queued, later staging keys cannot move ahead of it. Retained input is
discarded if the update is stale, fails, requires reconnection, ends the
episode, or shuts down the debugger; a retained Escape still performs its local
focus release without replaying a simulator command. Animation, hover, panels,
help, selection, and filters do not change scientific authority.

## Inspection and visual filters

The selected-target inspector reports authorized identity, relation, distance,
and public geometry. Roster, Comprehensive Agent Class Details, Pending Joint
Action, Latest Transition, Technical Frame, and the target selector remain
global researcher-space in both views. Pending Joint Action reports
movement, ability, target, and exact lane legality without carrying battlefield
anchors. Basic Legality is false when no target is selected. Only SVG-local
inspection and choreography consume fog-filtered geometry. The live Technical
Frame is allowlisted by authority: Episode, Frame, Simulator step, and
conditional Incoming transition. The initial frame has no incoming-transition
row.

Visual Filters contains 19 independently controlled paint families plus Ranges.
Initially enable Ultimate Ability Effects, Spawn Shield, Basic Ability Effects,
Regeneration Effects, Death Effects, Resurrection Effects, Scrolling Battle
Text, Respawn Wave and Death Announcer. Ranges start off, giving `9 enabled`.
Visual Filters and Roster start open; Enable All selects all 20 controls, and
Default Configuration restores these nine effects with Ranges off.
The complete inventory is:

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

Death Announcer names the killing team and shows a compact list of victims.
Hover or focus a victim to see every associated **Kill Contributor**, including
useful same-tick Priest support. Team A's kills appear in blue on the left and
Team B's in red on the right. Contributor details use the same accessible
event-information tooltip machinery as Respawn Wave. Missing historical credit
is identified as unavailable rather than guessed.

Notices last 1.5 seconds at normal presentation speed. Enabled cues remain visible
when inspecting their paused/static transition; seek, restart and replacement
clear the previous transition's notices. This global researcher HUD appears in
both POVs and adds no actor observations or battlefield visibility.

Duration Status Badges includes the white crossed-swords **In Combat** countdown.
An overflow badge represents at least two hidden statuses. If only one remains,
its actual status and duration occupy the cell. Compact and ordinary overflow
use the same centered solid badge.
Basic Ability Effects and Ultimate Ability Effects each own their corresponding
activation presentation and damage/healing impact glyphs. Scrolling Battle
Text owns the complete net-health unit: outcome glyph, signed value, recipient
label, and connector. Those parts are enabled or disabled together; damage and
healing remain distinguished by their outcome sign and color rather than by
separate filters.

These switches affect browser paint, accessible descriptions belonging to that
paint, and nothing else. They do not redact source data, change simulator state,
or alter authorized event data used by battlefield choreography. Ranges uses
its existing service/local authority path rather than the 19-entry paint schema.
**Enable All** and **Disable All** govern all 19 paint families and the active
Ranges control together.

## Recording and recovery

Create the destination parent, then opt into recording:

```bash
mkdir -p recordings
./scripts/dev/run_dev_client.sh \
  --record-replay recordings/episode.marlbg-replay.json
```

The target must end in `.marlbg-replay.json`; its parent must already exist,
and the destination must not already exist. The
launcher preflights the destination before building the scenario, discovering
runtime provenance, binding a server, or opening a browser.

Recording retains one authoritative trajectory in memory. Each
accepted submit still performs exactly one transition and one canonical
capture; there is no per-transition replay-file write.

- **Finish & Review** closes an open prefix, publishes its self-contained V3
  replay without a metrics sidecar, and changes the same loopback
  page to settled read-only review at frame zero.
- Task termination or the declared horizon closes and saves automatically.
  **Review Replay** performs the frame-zero handoff when requested.
- **Retry save** republishes the exact cached bytes or verifies an already
  successful exact publication; it never overwrites different bytes.
- **Save As** accepts only a new safe replay basename in the original parent.
  It cannot select another directory, traverse paths, follow symlinks, or
  overwrite an existing target.
- Resetting a nonempty recording prefix requires explicit discard
  confirmation. Cancelling preserves the current recording.
- **Exit Combat Debugger** and `Ctrl-C` attempt durable closeout before server
  shutdown. If ordinary publication fails, terminal closeout attempts the
  deterministic no-clobber recovery sibling reported in the terminal.

The page stays online after a persistence failure so bounded recovery actions
remain available. Closing a browser tab alone neither saves nor stops Python.

## Static reset snapshot

Render the manual arena's authorized reset state without a browser server:

```bash
./scripts/dev/run_dev_client.sh --static
./scripts/dev/run_dev_client.sh \
  --static --seed 7 --controlled-slot 5 --no-ranges
```

The shell activates the optional `viz` dependency for `--static`. Static mode
creates one reset session, calls the scene-native Matplotlib adapter, registers
no callbacks, and calls `step` zero times. For an exact recorded frame, use the
[Replay Viewer static path](replay_viewer.md#static-matplotlib-frame).

## Loopback safety

Before binding, Python validates an explicit runtime-asset allowlist. The
server binds only to `127.0.0.1`, places a random capability token in the URL
fragment, requires the corresponding request header, validates Host/Origin/
fetch-site metadata, applies a restrictive Content Security Policy and
`no-store`, and serializes commands under one service lock. Malformed,
conflicting-duplicate, and stale commands fail closed.

Refresh reconnects to the existing session. A stale tab receives current
authority rather than replaying a submit. Use **Reconnect** after a transient
connection loss, and use **Exit Combat Debugger** or `Ctrl-C` to stop Python.

## Troubleshooting

- **Browser did not open:** use the printed URL or pass `--no-open`.
- **Port unavailable:** omit `--port` or choose another loopback port.
- **Moved-option error:** use the [Replay Viewer](replay_viewer.md) for replay,
  sample, or scripted-scenario work.
- **Inactive controlled slot:** choose a configured-active `arena_5v5` slot.
- **Initial submit is slow:** the initial JAX transition may compile; no warm-up
  transition is hidden from the episode.
- **Static Matplotlib import failed:** run `uv sync --extra viz`.
- **Server remains after tab closure:** use the in-page Exit action or `Ctrl-C`.

Return to the [browser-tools migration page](visual_debugger.md) or the
[project README](../../README.md).

Recorded games preserve the captured states, actions, rewards and task outcomes.
Open Replay Viewer detailed analysis to compute the complete metric suite from
those facts and export CSV. New recordings need no metric sidecar; historical V1
replays and their existing sidecars remain readable.
