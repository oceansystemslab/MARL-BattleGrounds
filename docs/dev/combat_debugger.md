# Combat Debugger

The Combat Debugger is the live, manual MARL-BattleGrounds laboratory. It
always opens `arena_5v5`, whose map is 18×12, and always uses the fixed Analysis
presentation. Use it to inspect same-epoch authority, compose simultaneous
actions, submit transitions, and optionally record one manual episode.

Scripted demonstrations, checked samples, and existing artifacts belong to the
[Replay Viewer](replay_viewer.md). The debugger has no public scenario or
replay selector.

## Launch

Open the default Oracle view:

```bash
./scripts/dev/run_debug_renderer.sh
```

Useful launch variants:

```bash
# Start with one active global slot selected and hide ranges.
./scripts/dev/run_debug_renderer.sh --controlled-slot 5 --no-ranges

# Start in the selected actor's authorized POV.
./scripts/dev/run_debug_renderer.sh --view pov --controlled-slot 5

# Print the loopback URL without asking the operating system to open it.
./scripts/dev/run_debug_renderer.sh --no-open --port 8123

# Show the executable CLI contract.
./scripts/dev/run_debug_renderer.sh --help
```

The launcher resolves the repository from its own location, binds only to
`127.0.0.1`, chooses an ephemeral port by default, prints the URL, and attempts
to open a modern browser. Node.js and npm are not runtime requirements.

The public options are:

| Option | Meaning |
| --- | --- |
| `--record-replay PATH` | Record one manual episode to a canonical replay and adjacent metric sidecar, then offer read-only review. |
| `--seed N` | Set the deterministic reset/step seed; default `0`. |
| `--controlled-slot N` | Select an initially active global slot; otherwise use the arena default. |
| `--static` | Render one stateless Matplotlib reset snapshot without a browser server. |
| `--no-open` | Print the URL without opening a browser automatically. |
| `--port N` | Select a loopback port; `0` requests an ephemeral port. |
| `--view oracle\|pov` | Select the initial authorization; default `oracle`. |
| `--ranges` / `--no-ranges` | Show or hide controlled-actor ranges initially. |

Option abbreviations are rejected. Replay, sample, scripted-scenario,
frame-index, and replay-POV-slot options are rejected with the Replay Viewer
launcher named in the error.

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
[specification amendment A11](../design/specification_amendments.md#a11-sharedobs-recorded-visual-union-presentation).
It does not provide a materialized SharedObs learner tensor or authorize the
browser to reconstruct geometry, visibility, masks, history, rewards, policy
state, or hidden Oracle facts.

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
and public geometry. Roster, Comprehensive Agent Class Details, Pending
Authorized Draft, Latest Transition, Technical Frame, and the target selector
remain global researcher-space in both views. Pending Authorized Draft reports
movement, ability, target, and exact lane legality without carrying battlefield
anchors. Basic Legality is false when no target is selected. Only SVG-local
inspection and choreography consume fog-filtered geometry. The live Technical
Frame is allowlisted by authority: Episode, Frame, Simulator step, and
conditional Incoming transition. The initial frame has no incoming-transition
row.

Visual Filters contains exactly 18 independently controlled paint families,
all enabled by default:

1. Aura Fields
2. Aura Modifier Badges
3. Duration Status Badges
4. Spawn Shield
5. Rejected Action Feedback
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

Duration Status Badges includes the white crossed-swords **In combat** countdown.
Basic Ability Effects and Ultimate Ability Effects each own their corresponding
activation presentation and damage/healing impact glyphs. Scrolling Battle
Text owns the complete net-health unit: outcome glyph, signed value, recipient
label, and connector. Those parts are enabled or disabled together; damage and
healing remain distinguished by their outcome sign and color rather than by
separate filters.

These switches affect browser paint, accessible descriptions belonging to that
paint, and nothing else. They do not redact source data, change simulator
state, alter authorized event data used by battlefield choreography, or replace
the separate Ranges control. **Enable All** enables all 18; **Disable All**
disables all 18.

## Recording and recovery

Create the destination parent, then opt into recording:

```bash
mkdir -p recordings
./scripts/dev/run_debug_renderer.sh \
  --record-replay recordings/episode.marlbg-replay.json
```

The target must end in `.marlbg-replay.json`; its parent must already exist;
and neither it nor an incompatible companion target may already exist. The
launcher preflights the destination before building the scenario, discovering
runtime provenance, binding a server, or opening a browser.

Recording retains one canonical metric-complete trajectory in memory. Each
accepted submit still performs exactly one transition and one canonical
capture; there is no per-transition replay-file write.

- **Finish & Review** closes an open prefix, publishes and validates the replay
  plus adjacent `.marlbg-metrics.json`, and changes the same loopback page to
  settled read-only review at frame zero.
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
./scripts/dev/run_debug_renderer.sh --static
./scripts/dev/run_debug_renderer.sh \
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
