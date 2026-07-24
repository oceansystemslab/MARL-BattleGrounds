# Milestone 5 visual debugger

The Milestone 5 visual debugger is an explicit-submit development tool for
inspecting the simulator's battlefield geometry, action mask, accepted action,
combat state, and selected-target facts. It uses ordinary public `reset` and
`step` calls. It does not add debugger behavior to the simulator core.

The renderer is stateless. Keyboard and mouse input, deterministic scenarios,
transition comparison, logs, and transient-effect history live under
`scripts/dev/visual_debugger`.

## Setup and launch

Install the development and optional visualization dependencies:

```bash
uv sync --extra viz --extra dev
```

Run the default 5v5 laboratory:

```bash
./scripts/dev/run_geometry_renderer.sh
```

The shell launcher resolves the repository root from its own path, so the
command also works when invoked from another current working directory.

Useful invocations:

```bash
./scripts/dev/run_geometry_renderer.sh --help
./scripts/dev/run_geometry_renderer.sh --list-scenarios
./scripts/dev/run_geometry_renderer.sh --scenario acceptance_lane_lab
./scripts/dev/run_geometry_renderer.sh --scenario status_stack --verbose
./scripts/dev/run_geometry_renderer.sh --scenario ultimate_showcase --static
./scripts/dev/run_geometry_renderer.sh --controlled-slot 5 --no-ranges
```

`--static` renders the deterministic reset snapshot without registering input
callbacks and without calling `step`.

## Command-line options

| Option | Meaning |
|---|---|
| `--scenario NAME` | Open a registered scenario. Default: `arena_5v5`. |
| `--list-scenarios` | Print names, modes, and descriptions, then exit without importing Matplotlib. |
| `--seed N` | Set the deterministic reset/step key seed. Default: `0`. |
| `--controlled-slot N` | Initially control an active fixed global slot instead of the scenario default. |
| `--static` | Draw only the reset snapshot; do not register callbacks. |
| `--verbose` | Print the expanded geometry, mask, aura, speed, and episode diagnostics. |
| `--ranges` / `--no-ranges` | Show or hide the controlled actor's range circles. Ranges are shown by default. |

An unknown scenario, invalid argument, or inactive controlled slot exits with
code `2`. A missing `uv` executable exits with code `127`. A missing Matplotlib
dependency exits with code `2` and prints the exact installation command.

## Controls

| Input | Action |
|---|---|
| `Tab` / `Shift+Tab` | Cycle forward/backward through active global slots. |
| Left click | Select an active global target for inspection. |
| Right click / `Escape` | Clear the target to target-none. |
| `1` | Explicitly arm Basic lane 0, even when the pair is unavailable. |
| `2` | Explicitly arm Ultimate lane 1, even when the pair is unavailable. |
| `W A S D` | Select cardinal movement. |
| `Q E Z C` | Select diagonal movement. |
| `Space` / `Enter` | Submit the manual action, or the next frame in a scripted scenario. |
| `N` | Submit the next registered reference/script frame. |
| `R` | Reset the current scenario deterministically. |
| `G` | Toggle the controlled actor's range circles. |
| `V` | Toggle concise/verbose transition logs. |
| `[` / `]` | Switch to the previous/next scenario. |
| Close window | Disconnect callbacks, restore Matplotlib key mappings, and exit. |

There is no simulation timer. Selecting an actor, changing movement, arming a
lane, cycling the controlled actor, toggling presentation, or redrawing never
advances the simulator or consumes a PRNG key.

## Pending-action behavior

The reset pending action is Stay, target-none, and automatically armed Basic
lane 0. A clicked target is retained even when no combat lane is available.
Basic auto-arms only when the exact current lane-0 value for that actor-target
pair is true. Ultimate never auto-arms.

Keys `1` and `2` deliberately permit an unavailable in-domain pair to be
submitted. This lets the authoritative simulator demonstrate canonical
rejection. The debugger does not pre-filter it.

Mage Burst is a target-none Ultimate. Pressing `2` while controlling a Mage
clears a selected target and arms lane 1. It creates no target link or
Ultimate-radius circle. It can still be armed during cooldown or stun so its
rejection can be inspected.

After every submitted transition:

- the selected global target persists;
- movement resets to Stay;
- Ultimate disarms;
- Basic re-arms only if successor lane 0 is available for the retained target;
- otherwise the target remains selected with no lane armed;
- distance, relation, line of sight, visibility, ranges, and lanes are derived
  again from the successor decision epoch.

At termination or truncation, submissions are blocked before key splitting.
Inspection, actor cycling, presentation toggles, reset, and scenario switching
remain available.

## Selected-target inspector

The HUD separates three different kinds of fact:

```text
TARGET g6/t7 relation=enemy distance=4.25
GEOMETRY los=1 visible=1 observation_range=1 basic_range=1 ultimate_range=0
LEGALITY lane0=1 lane1=0 selected=Basic pending_legal=1
```

`TARGET` reports the fixed global slot, actor-relative target category,
relationship, and center distance.

`GEOMETRY` reports independent facts:

- line of sight from the public geometry authority;
- observer-relative visibility from the current observation mask;
- inclusive membership in the controlled actor's observation radius;
- inclusive membership in the controlled actor's Basic radius;
- inclusive membership in a targeted Ultimate radius.

A target-none or non-targeted Ultimate displays `n/a` for target geometry.

`LEGALITY` reports lane 0 and lane 1 directly from
`select_target_use_ultimate_joint_mask[actor, target, lane]`. The debugger does
not reconstruct those values from geometry. Displayed geometry and visibility
facts are never claimed to be the cause of an unavailable lane.

While a target is selected, non-visible active agents may be dimmed, but their
team outline, global-slot label, and selection cues remain locatable. Clearing
the target clears this visibility treatment. Visibility is supplied to the
renderer as presentation data; the renderer does not derive it.

## Battlefield legend

### Durable snapshot presentation

| Meaning | Presentation |
|---|---|
| Mage / Warrior / Hunter / Rogue / Priest | Class-filled body and `M/W/H/R/P` label using aqua, brown, green, mustard, and pink. |
| Team A / Team B | Blue/red circumference at the true physical body radius. |
| Health | Inset green/amber/red annulus; it never changes the collision outline. |
| Controlled actor | Four inward white chevrons with a dark under-stroke. |
| Selected target | Magenta crosshair and bullseye. |
| Lane 0 / lane 1 | Lower-left green and lower-right violet arcs; unavailable lanes are gray and the armed lane is thicker. |
| Observation / Basic / Ultimate range | Gray dashed, green solid, and violet dotted world-space circles. |
| Warrior/Hunter/Rogue stun | Three fixed top status anchors in source-class colors. |
| Warrior/Hunter/Rogue slow | Three fixed bottom double-chevron anchors in source-class colors. |
| Mage aura / Warrior aura | Aqua dotted upper and bronze hatched lower semicircular bands. |
| Rogue anti-heal | Broken medical cross. |
| Blessing of Freedom | Pink shield/wing cue. |
| Mage Burst active | Aqua eight-ray inner starburst. |
| Previous accepted action | Graphite movement arrow and compact Basic/Ultimate badge. |

All persistent body-local presentation stays within the true body radius. Fixed
anchors allow all three stun channels, all three slow channels, two auras,
anti-heal, Freedom, Burst, health, lanes, selection, and previous-action
history to coexist without allocating additional collision-like rings.

### Transient presentation

| Event | Presentation and lifetime |
|---|---|
| Net health change | One signed red/green public net delta for one submitted transition. |
| Basic damage | Source-colored action link and red impact flash. |
| Basic healing | Priest-colored link and one green recipient pulse. |
| Holy Word | Strong link and two green/pink recipient pulses. |
| Mage Burst activation | Target-none aqua expanding starburst. |
| Warrior Charge | Bronze arrowed realized path and impact wedge. |
| Hunter Trap | Green square-jaw/cage flash. |
| Rogue Poison | Three mustard droplet/needle marks. |
| Multiple recipient effects | One segmented source-colored composite marker; exact sources remain in the HUD/log. |
| Rejection | Red `M×`, `C×`, or `TUPLE×`; rejected combat also has a dashed target link, and simultaneous components share one label. |

One-step visuals expire on the next submitted transition. Charge traces are
shown at opacity `1.00`, `0.65`, and `0.35`, then removed. Redrawing does not
age effects. Reset and scenario switching clear all transient history.

Health labels show only `after_health - before_health`. Accepted effect labels
identify capability activations, but do not invent gross per-source damage or
healing when effects aggregate, clip, or cancel.

## Deterministic scenarios

Every scenario builds a validated `EnvConfig`, uses ordinary `reset`, sets
`ordinary_movement_distance_scale=1.0`, and starts at maximum health with zero
cooldowns, statuses, and action history.

### `arena_5v5`

Interactive `18×12` laboratory with all five classes per team, a pillar at
`(9,3)` with radius `0.9`, and a rotated `3×0.5` wall at `(9,7.8)` with angle
`0.45`. Use it to inspect open/blocked LOS, observer visibility, relations,
range boundaries, collision, and pair masks.

### `acceptance_lane_lab`

Interactive `16×12` Hunter-versus-Mage boundary. Six reference frames move the
Hunter from distance `9` to `4`. Visibility becomes true at distance `6`,
Basic becomes available at `5`, and Trap becomes available at `4`. Early
frames demonstrate accepted East movement with rejected combat. The last frame
applies Hunter Trap with cooldown `30` and stun duration `4`.

### `basic_support`

Two scripted frames. The first applies three simultaneous Basics and exposes
Mage amplification, Warrior mitigation, Hunter slows, and exact health
successors. The second combines Priest self-healing with simultaneous Hunter
damage, retaining the public net `0.00` without fabricating gross
contributions.

### `ultimate_showcase`

Three scripted frames. A Mage Basic prepares the allied Hunter; then Mage
Burst, Warrior Charge, Hunter Trap, Rogue Poison, and Holy Word activate
simultaneously. The final Hunter Basic demonstrates an unambiguous Trap break,
status decrements/expirations, and cooldown decrement.

### `aura_crossfire`

One scripted reciprocal Hunter-Basic frame while both Hunters are inside Mage
amplification and Warrior mitigation. Both health values end at `92.18`, both
aura bands remain visible, and both Hunter slow channels apply.

### `status_stack`

Four scripted frames that combine Charge plus precommitted movement, Trap,
Poison, Priest healing/Freedom, rejected stunned movement, Trap break, Freedom
speed floor, stacked slow movement, duration expiration, and cooldown
decrement. A Charge-plus-movement trail is labeled as combined realized
displacement; it does not claim a private intermediate landing.

## Terminal diagnostics

Concise mode prints:

```text
STEP scenario=<name> <before>-><after> terminated=<0|1> truncated=<0|1>
ACTOR g<slot> <team>/<class> target=<none|gN/tN> lanes=0:<0|1>,1:<0|1> submitted=<...> accepted=<...> movement=<...> combat=<...>
POSITION gN (<before>)->(<after>) delta=(<dx>,<dy>)
HEALTH gN <before>-><after> net=<signed delta>
COOLDOWN gN <before>-><after> <started|decremented|expired>
STATUS gN <kind> <before>-><after> <classification>
EVENT <kind> source=gN recipient=<gN|none> [detail]
```

Verbose mode adds one `TARGET`, `GEOMETRY`, `MASK`, `SUBMITTED`, `ACCEPTED`,
`AURA`, `SPEED`, and `EPISODE` record per report actor. It prints named channels
and scalar facts, not raw tensor dumps.

Accepted actions come only from the successor state's previous-action fields.
Movement and combat-pair acceptance are reported independently. Rejections show
domain and current-mask facts, never guessed causal explanations.

Status changes are conservatively classified as application, refresh,
decrement, expiration, unambiguous Trap break, or unclassified clear. A Trap
transition from `1` to `0` alongside accepted damage remains ambiguous because
natural expiration and break cannot be distinguished from public artifacts.

## Reset and reproducibility

The CLI seed initializes a master JAX PRNG key. Reset and scenario switching
reconstruct the same deterministic key order. Each successful submission splits
the current key once and calls `step` once with the action mask paired to the
current state. UI-only actions never split the key.

Scripted frames can command multiple actors. Manual mode builds a joint action
for only the controlled actor. Both paths meet at the same submission boundary;
scripted frames do not pass through the single-actor pending-action builder.

## Architectural boundary and non-goals

Reusable code under `marl_battlegrounds.rendering` draws snapshots and
already-described cues. It has no callbacks, timers, simulator calls, inferred
event model, or history.

The debugger's `SelectedTargetFacts`, before/after transition comparison, and
transient aging are development diagnostics. They are not a production replay
event schema and are not serialized. An eventual replay viewer may naturally
reuse the stateless battlefield renderer and visual descriptions while
consuming recorded state or event data through its own durable contract.

The debugger does not:

- modify simulator action, observation, reward, status, or termination
  semantics;
- duplicate LOS, visibility, or action-mask algorithms;
- attribute an unavailable action to one displayed geometric fact;
- recover gross health contributions from an aggregated net transition;
- reconstruct the private intermediate Charge landing;
- create a replay timeline or production event protocol.
