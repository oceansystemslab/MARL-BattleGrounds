# Milestone 5 Visual Debugger and Analyzer

The Milestone 5 Visual Debugger and Analyzer is an explicit-submit research
tool for
inspecting battlefield geometry, exact action-mask values, pending actions,
accepted actions, combat consequences, statuses, and selected-target facts.
The live application uses a local browser. A stateless Matplotlib snapshot
remains available for compatibility and headless use.

## Authority boundary

Python is the sole authority for:

- `EnvConfig`, JAX state, observations, action masks, and PRNG keys;
- `DebuggerSession`, every active agent's staged action, and scenario state;
- simulator-command normalization, authorized agent hit testing,
  global/relative target mapping, and exact legality;
- accepted actions, transition diagnostics, status lifecycle classification,
  and reset/switch behavior;
- one submission, one key split, and one `core.step` call.

The browser owns input capture, pointer-to-world coordinate projection,
responsive layout, SVG/HTML rendering, hover/help/panel state, local
presentation keys such as `P` and `?`, focus release after Escape, and
presentation-only animation time. It receives a small allowlisted
`DebuggerFrameV1`; it never receives raw JAX arrays or recomputes combat,
visibility, targetability, agent hit testing, or legality.

The live path is:

```text
browser input
  -> revisioned loopback command
  -> Python DebuggerService
  -> shared input/control/targeting helpers
  -> optional single authoritative submit
  -> allowlisted scene, HUD, and latest event batch
  -> SVG battlefield and HTML inspector
```

UI-only activity never calls `step`, splits a key, or restarts an already
consumed transition animation.

## Launch

Run the default researcher laboratory:

```bash
./scripts/dev/run_debug_renderer.sh
```

The shell launcher resolves the repository root from its own path, so it also
works from another current working directory. It binds only to `127.0.0.1`,
selects an ephemeral port by default, prints the URL, and attempts to open the
system browser. If automatic opening fails, copy the printed URL into a modern
browser.

Node.js and npm are not runtime requirements. Browser assets are tracked native
HTML, CSS, SVG, and JavaScript modules served directly by Python.

Useful invocations:

```bash
./scripts/dev/run_debug_renderer.sh --help
./scripts/dev/run_debug_renderer.sh --list-scenarios
./scripts/dev/run_debug_renderer.sh --scenario team_focus_crossfire
./scripts/dev/run_debug_renderer.sh --scenario max_status_stack --include-stress
./scripts/dev/run_debug_renderer.sh --controlled-slot 5 --no-ranges
./scripts/dev/run_debug_renderer.sh --view pov --preset debug
./scripts/dev/run_debug_renderer.sh --no-open --port 8123
```

### Options

| Option | Meaning |
| --- | --- |
| `--scenario NAME` | Initial registered scenario. Default: `arena_5v5`. |
| `--list-scenarios` | Print scenario names, modes, and descriptions, then exit. |
| `--include-stress` | Include developer stress scenarios in lookup and the browser selector. |
| `--seed N` | Deterministic reset/step key seed. Default: `0`. |
| `--controlled-slot N` | Initially control an active fixed global slot. |
| `--no-open` | Print the URL without asking the operating system to open it. |
| `--port N` | Loopback port; `0` selects an ephemeral port. |
| `--view researcher\|pov` | Initial authorization mode. Default: `researcher`. |
| `--preset presentation\|analysis\|debug` | Initial visual-density preset. Default: `analysis`. |
| `--verbose` | Enable expanded diagnostics. |
| `--ranges` / `--no-ranges` | Initially show or hide controlled-actor ranges. |
| `--static` | Render one Matplotlib reset snapshot; start no server and register no callbacks. |

Option abbreviations are rejected. Unknown scenarios, invalid ports or
arguments, and inactive controlled slots exit with code `2`.
`--no-open`, `--port`, `--view`, and `--preset` are browser-only and have no
effect on a `--static` snapshot.

## Loopback lifecycle and safety

Before binding or opening a browser, Python validates the complete runtime asset
manifest. Static files come from an explicit allowlist; symlinks, path escape,
unsupported types, and missing assets are rejected.

The server:

- binds only to `127.0.0.1`;
- places a random capability token in the URL fragment and requires it in a
  request header;
- validates Host, Origin, and fetch-site metadata;
- applies a restrictive Content Security Policy and `no-store` caching;
- serializes commands under one service lock;
- rejects malformed, duplicate-conflicting, and stale commands without
  automatically retrying a submission.

Command IDs provide mutation idempotency: an identical duplicate is never
applied twice and receives the current authoritative frame, rather than a replay
of the original transport outcome.

Refresh reconnects to the existing Python session. A stale tab receives the
latest frame and a notice that another client advanced the session. Connection
loss shows an offline state; Reconnect fetches current authority and never
replays a submit.

Closing the browser tab does not stop Python. Use **Exit analyzer** or `Ctrl-C`
in the launching terminal.

## Focus and controls

Keyboard shortcuts apply only while the battlefield SVG has focus. Click the
battlefield once to give it command focus. Controls and form fields in the
inspector retain normal browser keyboard and Tab behavior.

| Input | Action |
| --- | --- |
| `Tab` / `Shift+Tab` | Cycle the controlled actor without discarding any staged draft. |
| Left click | Select a visible active target. |
| `Shift` + left click | Control the clicked visible active actor. |
| Right click | Clear the selected target. |
| `Escape` | Clear the target and leave battlefield focus. |
| `W A S D` / arrow keys | Choose cardinal movement for the controlled actor. |
| `Q E Z C` | Choose diagonal movement. |
| `X` | Set the controlled actor to Stay. |
| `1` / `2` | Arm Basic lane 0 / Ultimate lane 1. |
| `Space` / `Enter` | Submit the staged joint turn in researcher view, or the controlled actor only in agent POV. |
| `N` | Advance the next registered scripted frame. |
| `R` | Reset the scenario deterministically. |
| `Shift+R` | Explain why cooldown-only clearing is unavailable; state is unchanged. |
| `G` / `V` | Toggle controlled-actor ranges / diagnostic verbosity. |
| `[` / `]` | Previous / next scenario. |
| `P` | Pause or resume presentation-only motion. |
| `?` | Open in-app help. |

The joint-action console presents numbered **Movement** and **Action** composer
cards, followed by a visually dominant Submit rail. Lower-emphasis
**Inspect** and **Session** utility rows provide scripted advance, actor
cycling, target clearing, range and verbosity controls, reset, and scenario
navigation without competing with turn composition. Each authorized roster row
also provides **Target** and **Control** buttons. The toolbar provides
Scenario, View, Preset, Reconnect, Help, Exit, motion pause, `0.5×`, `1×`,
`2×`, Off, and Skip.

Normal animation briefly gates only the next Submit or scripted-frame command
during its explanatory phase. Skip, reduced-motion preference, or Off releases
that gate immediately. Animation state never changes simulator authority.

## Joint-turn planning

Every active actor owns an independent fixed-slot draft containing movement,
selected target, and armed lane. Reset initializes all drafts to Stay and
target-none; Basic auto-arms only where the exact same-epoch lane-0 pair is
available. Ultimate never auto-arms.

Cycling or directly controlling an actor changes only the row being edited.
Previously staged actions remain visible in the **Pending joint turn** card.
In an interactive researcher scenario, one Enter packages every active row and
submits one joint action through the single transition seam.

Agent POV authorizes only the controlled actor; all other rows are neutralized
server-side. Scripted scenarios are inspection-only for manual drafts, and `N`
advances their preflighted trajectory.

Unavailable movement and lane choices remain visible, red/dim, and
explainable, but normal browser input cannot execute them. Python repeats the
exact mask check before changing a pending row. Low-level diagnostic helpers
retain the deliberate masked-draft path used by rejection-analysis tests; the
live deck does not expose that path or invent a rejection cause.

After a successful interactive transition:

- each target selection persists where still authorized;
- movement returns to Stay;
- Ultimate disarms;
- Basic re-arms only when the successor exact lane-0 pair is available;
- all displayed geometry, visibility, ranges, and masks come from the successor
  decision epoch.

Terminal or truncated sessions block submissions before key splitting while
leaving inspection, reset, view, preset, and scenario controls available.

### Movement-scale reset

The toolbar exposes the authoritative ordinary-movement scale from `0.01`
through `1.00` in hundredth increments, plus shortcuts for `0.10` and the
scenario-authored default. Dragging previews the two-decimal value locally;
one command is sent only when the value is committed.

Changing scale rebuilds a coherent reset epoch and calls `step` zero times. It
preserves the seed, view, preset, range visibility, and a still-valid
controlled slot while clearing transition events, reward, diagnostics, pending
rows, and scripted progress. Ordinary Reset preserves the current override;
switching scenarios restores the destination scenario's authored default.

## Views and presets

### Researcher

The default is an explicitly labelled **PRIVILEGED RESEARCHER VIEW**. It
contains the allowlisted omniscient battlefield and can add an
observer-visibility analysis overlay. It is never described as policy POV.

### Agent POV

Agent POV is constructed server-side from the controlled actor's authorized
observation and action-mask row. Hidden dynamic positions, health, statuses,
identities, and endpoints are omitted from the payload rather than hidden with
CSS. A submitted action may remain visible while its hidden combat result is
reported as undisclosed.

### Presentation presets

- **Presentation:** durable geometry and semantic events with minimal analysis
  decoration.
- **Analysis:** default researcher layout with roster, selected facts, event
  feed, selected ranges, and selected legality.
- **Debug:** privileged visibility, expanded candidate legality, geometry, and
  technical frame details.

## Responsive battlefield and inspector

The battlefield is native SVG. The inspector is responsive HTML with:

1. scenario, revision, step, transition, view, and preset state;
2. exact-ID team rosters with health, cooldown, statuses, and selection;
3. controlled/selected comparison;
4. **Pending joint turn** with explicit submission scope;
5. **Latest transition** with submitted/accepted/rejected facts;
6. latest semantic event feed;
7. visual key and collapsible technical frame.

At the minimum supported viewport the two-column regions compress and scroll
independently. The primary review viewport is `1440×900`; the minimum supported
viewport is `960×600`. A stacked convenience layout exists below `960px`, but
it is outside the supported review contract.

Exact IDs are durable in the roster. Battlefield identity tags appear for
selection/hover when space permits, avoiding permanent `id_N` clutter.
One delegated tooltip explains the highest-priority authorized fact beneath
the pointer or keyboard focus. Statuses, modifiers, overflow, legality, and
cooldowns outrank agents; agents outrank event routes, obstacles, ranges, and
auras. The tooltip switches immediately, stays within the viewport, and cannot
explain a fact omitted from an agent-POV payload.

## Visual vocabulary

The [visual acceptance evidence gallery](visual_debugger_visual_evidence.md)
maps every requested visual rule to its automated proof and
original-resolution screenshot.

### Durable identity and geometry

| Fact | Presentation |
| --- | --- |
| Team A | Solid blue physical perimeter. |
| Team B | Solid red physical perimeter plus a right-edge chevron, so team is not color-only. |
| Mage | Aqua arena-star glyph with `M` fallback. |
| Warrior | Bronze shield glyph with `W` fallback. |
| Hunter | Lime (`#84CC16`) bow, straight string, and arrow glyph with `H` fallback. |
| Rogue | Yellow twin-blade glyph with `R` fallback. |
| Priest | Pink healing-cross glyph with `P` fallback. |
| Health | Inset successor-health ring; exact value remains in the roster. |
| Controlled actor | Bright outer halo. |
| Selected target | Magenta corner reticle. |
| Observation range | White dotted circle. |
| Basic range | Controlled actor's class color with a dashed stroke. |
| Ultimate range | Purple dash-dot stroke. |
| Mage/Warrior aura | Low-alpha cyan/bronze tint only, with no border. |
| Basic/Ultimate legality | Detached selected-target `0/B` and `1/U` pills using exact mask values. |
| Ultimate cooldown | Class-specific Ultimate icon plus exact positive tick count in a separate collision-aware dock; absent at zero. |

Aura tint has no perimeter so it cannot be confused with observation, Basic,
Ultimate, team, health, controlled, or target boundaries.

### Statuses and modifiers

Statuses use compact glyph-duration cells in stable semantic order: hard
control, slows, then combat modifiers. Source-class channels share a family
glyph but retain an accent and exact accessible name. Candidate docks use
deterministic north/east/west/south anchors, bounded leader ticks, and overflow
accounting; the roster always retains complete exact status facts.

At the `960×600` stress limit, a projected battlefield of at most `600×420`
uses an actor-owned two-line status summary (`id_N` and `+N`) with a mandatory
leader instead of placing a full status matrix beside every body. This applies
to controlled and selected agents as well as ordinary agents because retaining
all matrices made ownership ambiguous in dense combat. The exact token list
remains available in the roster and delegated tooltip; desktop docks remain
fully expanded. This is a responsive presentation policy, not status
suppression or a change to simulator truth.

While choreography is actively playing at that compact size, accepted routes,
recipient NET outcomes, bodies, modifiers, and cooldowns take priority.
Presentation-only range rings, the pending route, selected-legality pills, and
the compact status summaries are temporarily hidden; their authoritative DOM
and HUD facts remain intact. They return automatically when choreography
settles or immediately when **Skip** is used. This does not dispatch a command
or change simulator state.

The supported durable vocabulary includes three source-class stun channels
sharing one canonical stun glyph, three source-class slow channels sharing one
canonical swirl, anti-heal, Freedom, Burst, and effective aura modifiers.
Source identity remains in each token's accent and accessible name. Overflow
is neutral monochrome and its tooltip enumerates every hidden fact.

The static/headless painter has no interactive tooltip, so its compact status
labels also append the source-class fallback letter (for example, `⬢W` or
`↻H`). The family glyph remains canonical while grayscale and print exports
retain non-color source identity.

Human-facing floating values, health values, durations, and multipliers show at
most two decimal places. Final font geometry is measured before placement.
Supported durations and cooldowns reserve separate icon and number
compartments. Extraordinary future values may use a measured neutral fallback,
but a number never overlaps an icon or escapes its cell.

### Damage, healing, targeting, and class actions

| Class | Basic | Ultimate activation |
| --- | --- | --- |
| Mage | Directional damage route terminating in a red minus. | Source-local expanding Burst ring and arena-star flare; durable Burst remains a separate status token. |
| Warrior | Directional damage route terminating in a red minus. | Directional Charge impact/flare plus the exact public before/after displacement chord. |
| Hunter | Directional damage route terminating in a red minus. | Directional Trap delivery with a neutral lattice/diamond impact; durable Trap and its ending lifecycle remain separate. |
| Rogue | Directional damage route terminating in a red minus. | Directional Poison delivery, red-minus impact, target splash, and durable consequences. |
| Priest | Rounded directional healing tether terminating in a green plus. | Stronger Holy Word route with green-plus impact and dual healing flare. |

Selection is always the magenta corner reticle; targeting intent is a thin
pending preview. Ordinary completed Basics and non-Charge Ultimates use
successor source/recipient anchors so routes agree with displayed bodies.
Charge activation remains pre-transition and its displacement joins the
Warrior's pre-position to its successor position. Routes are clipped at body
radii. Reciprocal routes bend in opposite directions, same-direction
multiplicity receives stable parallel offsets, close distinct centers preserve
the actual source-to-recipient bearing, and all accepted activations begin
together.

In extremely dense static frames, the event feed is the definitive direction
and identity fallback; live particles and route markers carry direction more
clearly than a frozen overlapping screenshot.

## Outcomes, animation, and honest attribution

Accepted source/target activations and recipient health consequences are
different facts:

- one activation event/route exists for every exact accepted activation;
- one recipient-level `NET −N.NN`, `NET +N.NN`, or `HP unchanged` cue reports
  exact before/after health;
- no source route carries a fabricated damage or healing amount.

NET cues place before lifecycle decoration and search deterministic local,
protected-edge, and whole-viewport candidates. Bodies, status docks, selection
marks, activation icons, and existing outcome cues are protected regions. If
no collision-free location exists, the transient node is retained but hidden
so resize/preset reprojection can reveal it without replaying the event.

At the `960×600` crowded stress limit, lower-priority lifecycle decoration can
be suppressed under this explicit policy; exact NET outcomes, durable statuses,
and the structured event feed retain the authoritative story. This density case
remains part of final human acceptance rather than being presented as unlimited
screen capacity.

All accepted activations start in one shared phase; impact and NET cues share
one impact phase. Ordinary choreography is bounded and uses the latest
transition only. Hover, help, panel changes, and redraw do not restart it.

### Charge

Charge shows only the exact public pre-transition source position and successor
source position. It never reconstructs a private collision-resolved
intermediate landing or claims a literal continuous physical path. The
displacement remains through UI-only activity and is replaced by the next
successful transition, reset, or scenario switch.

### Trap

Trap lifecycle language remains conservative:

| Public evidence | Classification |
| --- | --- |
| `0 -> full` with accepted Trap | Applied |
| Positive duration `-> full` with accepted Trap | Refreshed/reapplied |
| `before > 1 -> 0`, no new Trap, accepted positive raw-damage action | Exact break/shatter |
| `1 -> 0` with accepted damage | Ambiguous end; neutral dissolve, never “break” |
| `1 -> 0` without damage/application | Natural expiry |
| Unexpected clear | Unclassified neutral ending |
| Defensible break plus reapplication | Composite break-and-reapply |

The debugger does not invent a breaker when several damage sources were
accepted.

## Scenarios

### Researcher menu

- `arena_5v5`: interactive geometry and combat laboratory.
- `basic_support`: simultaneous Basic damage, healing, passives, and zero-net
  attribution.
- `ultimate_showcase`: all five class Ultimates and lifecycle follow-up.
- `aura_crossfire`: reciprocal Basics under Mage and Warrior auras.
- `status_stack`: stacked control, mitigation, break, movement, and expiry.
- `team_focus_crossfire`: repeated damage, four-way focus fire, three-way
  healing, anti-heal, and multiple Holy Words.
- `mirrored_ultimates`: reciprocal mirrored activation of all Ultimate
  families.

### Developer stress menu

Pass `--include-stress` to expose:

- `charge_convergence`;
- `trap_lifecycle`;
- `max_status_stack`;
- `moving_basic_crossfire`;
- `moving_focus_crossfire`.

Every simulator-backed scripted command is preflighted against its actual
pre-state mask and accepted action.

### Renderer-only fixtures

`visual_vocabulary`, `crowded_teamfight`, `route_collision`, `mixed_net_zero`,
`viewport_matrix`, and `pov_redaction` are explicitly synthetic presentation
fixtures. They are never submitted to the simulator and must not be described
as valid histories.

## Static Matplotlib snapshot

Install the optional visualization dependency and render one reset frame:

```bash
uv sync --extra viz
./scripts/dev/run_debug_renderer.sh --scenario arena_5v5 --static
```

The shell activates `viz` only for the exact `--static` flag. Static mode
creates one reset session, builds one authorized researcher scene, calls the
scene-native stateless Matplotlib painter, opens no HTTP server, registers no
callbacks, and calls `step` zero times.

The public scene-native `draw_scene_geometry`, `render_scene_geometry`, and
`redraw_scene_geometry` APIs remain lazy-import and headless-capable.
Matplotlib does not reproduce browser animation or the live inspector.

## Contributor visual checks

Browser source is native JavaScript with strict JSDoc checking and no build
step. See [quality_gates.md](quality_gates.md) for the impact-based selection
policy and complete closeout commands.

Curated Playwright baselines use pinned Chromium, bundled fonts, fixed
viewports/device scale/locale, and a deterministic paused animation clock:

```bash
npm run test:visual --prefix web/visual_debugger
```

Updating baselines is exceptional:

```bash
npm run test:visual:update --prefix web/visual_debugger
```

Review every changed image at original resolution and its semantic companion
assertions before committing. Never update snapshots merely to make a failure
green.

## Troubleshooting

- **Browser did not open:** use the printed loopback URL or launch with
  `--no-open`.
- **Port is unavailable:** omit `--port` for an ephemeral port or choose
  another explicit port.
- **Runtime asset error:** restore the named tracked asset; the server fails
  before binding.
- **Offline/stale banner:** use Reconnect to fetch current authority; never
  retry a submit by hand unless the latest frame proves it was not applied.
- **First submit is slow:** the first JAX transition may compile; the busy state
  is immediate and no warm-up step is performed.
- **Static Matplotlib missing:** run `uv sync --extra viz`.
- **Server remains after tab close:** use Exit analyzer or `Ctrl-C`.

## Replay reuse boundary

The reusable boundary is the renderer-neutral scene/event vocabulary, SVG
painter, layout, animation controller, presets, and accessibility conventions.
`DebuggerFrameV1`, pending actions, revision/idempotency handling, and the
loopback command protocol are live-analyzer-only.

A future replay product may provide the same scene/event primitives from a
recorded artifact, but it will own a separate replay envelope, timeline,
schema-migration, integrity, and export contract. The live analyzer does not
load or simulate replay data.
