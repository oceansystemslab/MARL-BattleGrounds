# Milestone 4: Geometry, Movement, Visibility, and Masks

Status: complete

Important Git note: this file is now completed milestone documentation. Step
packet files remain local apprenticeship guides and must not be staged or
committed.

## Milestone Objective

Milestone 4 makes the Milestone 3 environment spine spatially meaningful.

This milestone delivers:

- rectangular map boundaries;
- finite-radius disc agents;
- fixed padded obstacle slots for circular pillars and rotated rectangular walls;
- movement commands mapped to continuous displacement;
- movement projection against boundaries, pillars, walls, and other agents;
- line-of-sight checks against pillars and walls;
- LOS-gated observation-radius logic for dynamic unit visibility;
- structured self, ally, enemy, map, objective-placeholder, and context
  observations;
- selection masks derived from state and relation-specific effective stat
  families;
- neutral class identity and slot-aligned effective stat arrays that preserve
  future class heterogeneity without implementing class mechanics yet;
- ultimate-mask placeholders that keep ultimate use unavailable until class
  mechanics exist;
- global map observation slots exposed to every agent;
- deterministic geometry, line-of-sight, movement, observation, and mask tests;
- a minimal Python geometry renderer and movement-only manual-control debug
  harness for visual inspection.

This milestone deliberately does not deliver:

- class identity beyond neutral placeholders needed to preserve schema shape;
- damage, healing, ultimates, cooldown semantics, passives, or status effects;
- death, respawn waves, full spawn-sanctuary behavior, regeneration, or events;
- Team Deathmatch, King of the Hill, Capture the Flag, task registry, curriculum,
  baselines, wrappers, replay export, metrics, or logging;
- randomized map generation beyond fixed padded geometry accepted through
  configuration;
- polished visualization or browser replay.

The milestone should end with a runnable, testable, visually inspectable spatial
slice. The benchmark will not yet be a game, but agents should have meaningful
positions, collision constraints, visibility relationships, and legal selection
masks.

## Source Anchors

- `MARL_BGs_Design_Document.pdf`, section 4.3.4, "Milestone 4: Geometry,
  Movement, Visibility, and Masks": requires map boundaries, disc agents, walls
  and pillars, collision projection, line-of-sight checks, observation-radius
  logic, selection masks, global map observation, minimal renderer, and
  deterministic geometry and masking tests.
- `MARL_BGs_Design_Document.pdf`, section 2.11.16, "Manual-Control Debugging
  Interface": supports a development-only human-control harness for debugging
  one controlled agent while other agents can remain scripted or no-op.
- `MARL_BGs_Design_Document.pdf`, section 2.17.4, "Stage 2: Geometry,
  Movement, Collision, and Minimal Rendering": requires boundaries, circular
  agent bodies, command-to-displacement movement, boundary projection,
  agent-agent body blocking, pillar collision, rotated wall collision, and a
  minimal renderer that visualizes simulator geometry. Milestone 4 implements
  agent-agent body blocking as deterministic fixed-pass projection with bounded
  residual overlap rather than as a complete global constraint solve.
- `MARL_BGs_Design_Document.pdf`, section 2.17.5, "Stage 3: Observation,
  Targetability, Line of Sight, and Masks": requires self observations, allied
  and enemy slots, padded slots, map slots, placeholder objective slots,
  observation-radius logic, line-of-sight checks, targetability predicates,
  movement masks, selection masks, ultimate-mask placeholders, context
  indicators, and valid `none` selection.
- `MARL_BGs_Design_Document.pdf`, requirements R11-R14: define map geometry,
  collision, line of sight, targetability, global map observation, and partial
  observability over dynamic units.
- `MARL_BGs_Design_Document.pdf`, section 2.3.3, "Observation Space": defines
  the mode-agnostic observation families, fixed allied/enemy slots, observation
  radius, targetability, global map observation, padding, and masks.
- `MARL_BGs_Design_Document.pdf`, section 2.3.4, "Action Space": defines the
  factored movement, selection, and ultimate heads, including selection index
  alignment and valid `none`.
- `MARL_BGs_Design_Document.pdf`, section 2.3.5, "Transition Dynamics": places
  movement and collision before later objective and combat logic. It also
  identifies class identity, body radius, cooldown timers, status effects,
  temporary modifiers, and modifier durations as agent state.
- `MARL_BGs_Design_Document.pdf`, requirements R8-R9: require class differences
  to be represented through parameters, masks, and transition semantics, with
  distinct health, interaction radius, movement speed, damage/healing, passives,
  ultimates, cooldowns, and status-effect interactions.
- `MARL_BGs_Design_Document.pdf`, requirement R12: requires targetability to
  depend on interaction radius and clear line of sight.
- `MARL_BGs_Design_Document.pdf`, section 2.7.3, "Class Parameters": defines
  class-owned max health, body radius, base movement speed, basic interaction
  radius, base damage/healing, ultimate parameters, ultimate interaction radius,
  and ultimate cooldown duration.
- `MARL_BGs_Design_Document.pdf`, section 2.5, "Map Geometry, Collision, and
  Visibility": defines map boundaries, static geometry as configuration,
  obstacle slots, movement projection, collision semantics, line of sight,
  observation, visibility, and implementation requirements.
- `MARL_BGs_Design_Document.pdf`, section 2.6.1, "Feasibility Through Action
  Masking": explains why masks remove impossible choices while movement
  feasibility is resolved through projection.
- `MARL_BGs_Design_Document.pdf`, section 2.6.4, "Global Geometry and Objective
  Observation": requires globally exposed static geometry without removing
  dynamic-unit partial observability.
- `MARL_BGs_Design_Document.pdf`, section 2.12.9, "Shared Geometry and
  Targetability": requires one shared geometry subsystem used by transition
  dynamics, observation construction, action masks, objective logic, replay,
  scenario validation, and debugging tools; the renderer must not define
  semantics.
- `MARL_BGs_Design_Document.pdf`, sections 2.14.4-2.14.5: require
  deterministic geometry, line-of-sight, observation, and action-mask tests.
- `MARL_BGs_Design_Document.pdf`, Appendix A.7-A.8: define the public
  observation schema and action-mask contracts.
- `SKILL.md`, "Mandatory milestone planning protocol": requires this dedicated
  milestone plan before implementation continues.
- `SKILL.md`, "The code-ownership rule": reserves geometry, collision, line of
  sight, visibility, masking algorithms, observation construction, transition
  logic, and semantic tests for the user to implement.

## Design-Doc Clarification

There is a useful tension between section 2.17.4 and milestone 4.3.4.
Section 2.17.4 mentions spawn-sanctuary boundaries in the geometry stage, but
milestone 4.3.6 explicitly assigns full sanctuary behavior to Milestone 6:
death, respawn, sanctuaries, re-entry restrictions, flag-carrier restrictions,
regeneration, events, and replay skeleton.

Classification: harmless clarification.

Recommendation: Milestone 4 may reserve schema or renderer space for sanctuary
geometry only if doing so avoids churn, but it must not implement or claim full
sanctuary behavior. Full sanctuary interaction rules remain Milestone 6 work.

Documentation patch: this clarification lives in this milestone plan. If future
implementation discovers that sanctuary geometry must be represented earlier,
update this section and the relevant type comments before implementing behavior.

## Design Drift Correction: Per-Agent Effective Stat Families

Drift: earlier Milestone 4 scaffolding treated movement speed, observation
radius, and target radius as scalar simulator-wide config values. That conflicts
with the original design document's heterogeneous class model.

Evidence: requirements R8-R9, requirement R12, section 2.3.5, section 2.7.3,
section 2.17.5, and Appendix A.8 all assume class-specific and status-modified
stat families. Body radius, movement speed, observation radius, basic
interaction radius, ultimate interaction radius, damage/healing, cooldowns, and
future buffs/debuffs are independent concerns.

Classification: unacceptable drift, corrected before continuing Step 4
visibility.

Corrected contract:

- `EnvConfig.default_*` stat fields are reset/scenario construction defaults
  only. They are not simulator semantic truth after an `EnvState` exists.
- `EnvState` owns the authoritative current per-slot stat arrays consumed by
  transition, observation, and masks.
- `agent_radii` remains the authoritative per-slot body-radius array.
- `movement_speeds` is consumed by movement transition logic.
- `observation_radii` must be consumed by Step 4 visibility.
- `basic_interaction_radii` must be consumed by basic attack/heal
  targetability.
- `ultimate_interaction_radii` is introduced now as a placeholder for later
  ultimate-mask semantics, but it must not affect basic targetability.
- `target_radius` and `target_radii` are banned as simulator contract names
  because targetability radius is interaction-specific.

Milestone 4 still does not implement real classes, cooldowns, passives,
ultimates, damage/healing, or class-specific legality. It uses neutral class
identity and default stat initialization so future mechanics can specialize
without migrating away from scalar simulator truth.

Documentation patch: this correction lives here and in the Step 4 packet until
the design document is revised. Future implementation packets must preserve the
separation between config defaults and effective per-slot state.

## Design Override: LOS-Gated Dynamic Observability

Decision: line of sight gates dynamic-unit observability in MARL-BattleGrounds
v1. A unit that is inside observation radius but behind an active wall or active
pillar is not currently observable to that observer.

This intentionally overrides the earlier design-doc wording in section 2.5.10
that allowed observed-but-LOS-blocked units. That older rule made targetability
depend on LOS while visibility depended only on observation radius. The new rule
makes both dynamic-unit visibility and targetability depend on LOS, while still
keeping static map geometry globally observed.

For observer slot `i` and candidate unit slot `j`, future observation
construction should compute dynamic-unit visibility as:

```text
visible(i, j) =
    observer is active and alive
    AND candidate is active and alive
    AND candidate is within observer i's current observation radius
    AND line of sight from observer to candidate is clear
```

Active pillars and active walls block LOS. Inactive obstacle rows do not block
LOS. Other agents do not block LOS in v1, though they still block movement.

Observation semantics:

- If `visible(i, j)` is false, candidate `j`'s current dynamic-unit feature row
  must be masked for observer `i`.
- At minimum, current position must not be exposed when LOS is broken.
- Preferred v1 rule: mask the whole dynamic unit feature row, not only position,
  to avoid leaking current health, cooldowns, class state, or other dynamic
  information.
- The corresponding visibility-mask entry must be false.
- The corresponding targetability-mask entry must be false unless a future
  ability explicitly overrides LOS.
- The true `EnvState` still contains full positions and unit state. Only the
  observer-specific `Observation` is partial.
- The environment should not provide last-seen position memory in v1. Agents
  must learn object permanence, belief state, and opponent modelling through
  policy memory or recurrence.

Strategic rationale: this enables tactical hiding, ganking, bluffing, ambushes,
and use of pillars and walls as cover. It also makes the benchmark more
meaningfully partially observable because agents must reason over stale
information after LOS is broken.

Classification: risky design override, accepted.

Documentation patch: this override lives in the Milestone 4 plan until the
design document itself is revised. Future implementation packets and milestone
docs must follow this rule.

## Design Override: Pre-Movement Target and Effect Validity

Decision: target/effect validity is computed from the state the policy observes
before ordinary movement, and selected unit effects resolve from that
pre-movement validity. Ordinary movement then updates positions, unless the
selected effect suppresses ordinary movement for that timestep.

This intentionally overrides the transition-order wording in design-document
section 2.3.5, which places movement and spatial objectives before unit
selection, basic class interactions, and ultimates. The accepted v1 rule is:

```text
target/effect validity from pre-movement state
selected basic or ultimate effect resolves from that validity
effect-specific displacement resolves when relevant
ordinary movement resolves afterward unless suppressed by the selected effect
next observations and masks are built from the resulting state
```

Rationale: move-conditioned target masks would make the action interface harder
to use and would increase invalid or no-op exploration. Pre-movement validity
keeps masks aligned with the state visible to the policy, avoids separate basic
and ultimate target heads, and supports target-first conditional masking in
Milestone 5.

Milestone 4 effect: this override does not add combat or ultimate behavior to
Milestone 4. Milestone 4 targetability remains basic-interaction targetability
only:

```text
basic targetable =
    visible
    AND within observer basic_interaction_radii
    AND observer/candidate active and alive
    AND placeholder class legality
```

Milestone 4 keeps:

- `ActionMask.target[agent, none]` valid iff the acting slot is active and
  alive;
- unit target columns valid iff the acting slot is active/alive and the
  relation-specific basic targetability mask is true;
- `ActionMask.use_ultimate[:, 1]` false;
- `state.ultimate_interaction_radii` preserved but unused by basic
  targetability.

Milestone 5 effect: class and ultimate work must use target-first conditional
masking. The target head selects any target valid for at least one currently
available selected-target effect, and the ultimate head is then masked
conditioned on the selected target. `use_ultimate = 1` means ultimate only; it
does not also auto-apply the basic effect.

Movement-displacing ultimates, including Warrior Charge, suppress ordinary
movement for the acting agent during that timestep unless a future class design
explicitly says otherwise. Charge displacement must use the shared geometry
projection primitive rather than inventing a separate movement path.

Classification: risky design override, accepted. This is a deliberate
sample-complexity and interface decision, not an implementation detail.

Documentation patch: this override lives in the Milestone 4 plan and the
Milestone 5 plan until the design document itself is revised. Future
implementation packets must not reintroduce move-conditioned target masks or
separate basic-target and ultimate-target action heads without a new explicit
design decision.

## Design Clarification: Movement Projection Constraint Priority

Decision: Milestone 4 treats static world validity as the final hard constraint
priority and agent-agent body blocking as deterministic fixed-pass best-effort
projection.

Static geometry remains a hard simulator invariant:

- active alive agents must finish inside the map bounds;
- active alive agents must finish outside active pillars to the extent possible
  for sane scenario geometry;
- active alive agents must finish outside active walls to the extent possible for
  sane scenario geometry;
- static projection must stay finite, deterministic, and JAX-compatible.

Checkpoint 12 revealed the coupled-constraint ordering issue that motivates this
priority. No fixed ordering of local projection primitives can guarantee all
constraints simultaneously in every constrained scenario. Agent-agent projection
can push an agent out of bounds or into a wall or pillar. A final bounds
projection can push an agent back into another agent. Obstacle projection can
move an agent in a way that affects both boundary validity and agent-agent
spacing. A stronger all-constraints-satisfied guarantee would require a more
sophisticated constraint-aware body-blocking solver, such as pinned-agent-aware
correction transfer or another global constraint method. That solver is out of
scope for Milestone 4.

Agent-agent overlap is handled as a bounded residual constraint rather than a
mathematically complete global solve. The design document's broad "hard
non-overlap" language is an idealized tactical requirement, but a universal hard
guarantee would require a heavier constraint solver, explicit priority rules for
impossible states, and more complex JAX control flow than Milestone 4 should
take on. Milestone 4 instead uses fixed substeps and fixed projection passes.
Increasing the outer collision-projection pass count should reduce ordinary
agent-agent residual overlap where surrounding static constraints leave room to
do so, but crowded or overconstrained states may retain finite residual overlap.

The required safety contract is:

- residual agent-agent overlap must never produce NaNs or infinities;
- residual overlap must never push agents out of bounds or into active static
  obstacles in the final projected positions;
- residual overlap must never create invalid masks, dynamic shapes, exploding
  corrections, hidden state, or non-deterministic behavior;
- same-team and enemy body blocking use the same fixed-pass projection rule.
- free-space agent-agent overlap tests should still expect complete separation
  after sufficient fixed passes;
- boundary-pinned, obstacle-pinned, crowded, or overconstrained tests should
  assert finite positions, hard static validity, inactive/dead preservation, and
  bounded residual overlap rather than exact agent-agent separation.

Classification: harmless clarification of the Milestone 4 implementation
contract, replacing an idealized hard-solver reading with a deterministic
fixed-pass residual contract.

Documentation patch: this clarification lives in the Milestone 4 plan until the
design document itself is revised. Future movement, mask, renderer, and scenario
validation work should treat active alive agent-agent overlap as a possible
bounded residual, while continuing to treat static geometry validity as hard.

## Architecture Context

The current Milestone 3 spine is intentionally small:

- `marl_battlegrounds.core.types` owns fixed data contracts.
- `marl_battlegrounds.core.env` owns functional `reset` and `step`.
- `tests/test_core_spine.py` protects construction, reset, placeholder step,
  rewards, done flags, JIT, and scanned rollout smoke coverage.

Milestone 4 should add spatial semantics without breaking these principles:

- Static shapes remain fixed. Smaller team sizes and unused geometry are
  represented through masks and padded slots, not variable-length arrays.
- Static geometry belongs in `EnvConfig`; dynamic positions, radii, active masks,
  and alive masks belong in `EnvState`.
- Geometry helpers should be pure, deterministic, JAX-compatible functions.
- `step` remains functional: it receives config, state, joint action, and key;
  it returns the complete next state, observations, rewards, done flags, masks,
  and info.
- The renderer and manual-control harness may inspect and display state/config,
  but they must not own geometry, action legality, or transition truth.
- Scenario loaders remain future work. Milestone 4 tests may hand-construct
  states directly to validate geometry and masks.

Expected module boundaries:

- `core.types`: constants and `NamedTuple` contracts.
- `core.env`: reset, step, observation construction, and action-mask assembly.
- `core.geometry`: pure geometry predicates and projection helpers.
- `rendering.geometry`: minimal Python renderer for debug images.
- `rendering.manual_control`: development-only manual-control harness that
  constructs actions and calls the real simulator step.
- tests: focused test files for geometry, line of sight, movement integration,
  observations, masks, renderer smoke, and existing core spine compatibility.

Important invariants:

- All per-agent arrays use shape `(MAX_AGENT_SLOTS, ...)`.
- All obstacle arrays use shape `(MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES)`.
- `MAX_OBSTACLE_SLOTS = 16`.
- Obstacle feature order is `(type, x, y, r, w, h, theta, active)`.
- Obstacle types are `0 = none`, `1 = pillar`, `2 = wall`.
- Selection index `0` is `none`.
- Selection indices `1..5` resolve to allied slots.
- Selection indices `6..10` resolve to enemy slots.
- Observation targetability is relation-specific:
  `ally_targetability_mask` has shape
  `(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM)` and
  `enemy_targetability_mask` has shape
  `(MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM)`.
- In Milestone 4, relation-specific targetability masks mean basic-interaction
  targetability only. Ultimate targetability is Milestone 5 work.
- `ActionMask.target` remains the flat action-head mask with shape
  `(MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)` and should be derived from
  `[none, ally_targetability_mask, enemy_targetability_mask]` rather than
  maintained independently.
- `none` selection is always valid for active alive agents.
- Inactive/padded slots are never selectable.
- Dead slots are not selectable. Full dead-agent behavior waits for Milestone 6,
  but masks should already respect `alive_mask`.
- Static geometry is globally observed by all agents.
- Dynamic-unit visibility is gated by active/alive state, observation radius,
  and line of sight.
- `self_features`, `ally_unit_features`, and `enemy_unit_features` use one
  shared agent-feature schema. `SELF_FEATURES == UNIT_FEATURES` remains true
  unless a future explicit schema decision changes it.
- Both self rows and unit-candidate rows use the shared `AGENT_FEATURE_*` index
  contract. The self vector exists only to condition the acting policy
  conveniently; it is not a separate schema.
- `self_features` remain canonical fixed-slot rows: active rows expose true
  state, while inactive/padded rows expose deterministic dummy inactive state.
  Do not add duplicate `self_active_mask`, `self_actionable_mask`, or
  `actor_valid_mask` fields to `Observation`; `AGENT_FEATURE_ACTIVE` identifies
  padding versus configured self rows, `AGENT_FEATURE_ALIVE` identifies
  alive/dead state, and action masks own action legality.
- Future public wrappers must use active self rows to hide padded inactive
  agents by default. Dead agents remain public agents with restricted action
  masks because death is state, not deletion.
- Step 1 placeholder visibility exposes only self in the allied visibility mask;
  enemy visibility remains false until LOS-gated visibility is implemented.
- Line of sight affects both dynamic-unit visibility and targetability.
- Hidden dynamic units remain in `EnvState`, but their observer-specific
  `Observation` feature rows must be masked.
- The environment does not provide last-seen position memory in v1.
- Agents block movement but do not block line of sight in v1.
- Ultimate action `1` remains invalid until Milestone 5 defines classes and
  ultimate semantics.
- Rewards remain zero placeholders in Milestone 4.
- Termination remains false and truncation remains horizon-based.

## Dependency Order

1. Branch and planning setup.
   - Branch from updated `main`.
   - Create this milestone plan before Milestone 4 implementation begins.
   - Confirm `origin/main` includes accepted Milestone 3 work.

2. Schema and configuration expansion.
   - Extend core contracts with map dimensions, default stat values, neutral
     class identity, authoritative per-slot effective stat arrays, agent radii,
     and obstacle slots.
   - Replace the single dummy observation vector with structured observation
     families.
   - Preserve compatibility intent while intentionally updating tests that
     asserted the Milestone 3 dummy schema.

3. Geometry helper skeleton and tests.
   - Add pure helper entry points and focused test scaffolds.
   - The user implements the actual projection and intersection logic.
   - Step 2 provides LOS geometry primitives only; it must not wire LOS into
     observation construction or masks.

4. Movement integration.
   - Map movement IDs to continuous displacement.
   - Apply fixed-substep movement projection in `step`.
   - Preserve zero reward and existing done semantics.

5. Line of sight, visibility, and targetability.
   - Add LOS checks against active pillars and walls.
- Add dynamic-unit visibility that combines active state, alive state,
  observer-specific observation radius, and clear LOS.
- Add basic targetability predicates that combine visibility, the observer's
  `state.basic_interaction_radii`, active state, alive state, and placeholder
  class legality.
- Preserve `state.ultimate_interaction_radii` for Milestone 5, but do not use it
  for Milestone 4 basic targetability.

6. Observation and action masks.
   - Build structured observations from state/config.
   - Build movement, selection, and ultimate masks from state/config.
   - Keep `none` valid and ultimate use unavailable.

7. Geometry renderer and manual-control harness.
   - Add a debug renderer that displays map boundaries, agents, radii, pillars,
     walls, and optional debug overlays.
   - Add a movement-only manual-control harness for one controlled agent.
   - Keep renderer semantics read-only and route all state changes through
     `core.env.step`.

8. Milestone review.
   - Run the quality gate.
   - Review for design drift.
   - Finalize this document only after implementation and review are complete.

## Step-by-Step Implementation Plan

### Step 0: Branch and Milestone Plan

Status: complete.

Acceptance:

- Worktree is on `feature/m4-geometry-masks`.
- `main` has been updated to include the accepted Milestone 3 merge.
- `docs/dev/milestone_4.md` exists.
- This file is not staged or committed while implementation is in progress.

### Step 1: Expand Core Contracts

Status: complete.

Target files:

- `src/marl_battlegrounds/core/types.py`
- `src/marl_battlegrounds/core/env.py`
- `tests/test_core_spine.py`

Required contract changes:

- Add constants:
  - `MAX_OBSTACLE_SLOTS = 16`;
  - `OBSTACLE_FEATURES = 8`;
  - `OBSTACLE_TYPE_NONE = 0`;
  - `OBSTACLE_TYPE_PILLAR = 1`;
  - `OBSTACLE_TYPE_WALL = 2`;
  - explicit movement IDs or a documented movement-order constant.
- Extend `EnvConfig` with:
  - map width and height;
  - reset/scenario defaults for movement speed, observation radius, basic
    interaction radius, ultimate interaction radius, and agent radius;
  - padded obstacle array.
- Extend `EnvState` with:
  - neutral `class_ids`;
  - `agent_radii`;
  - `movement_speeds`;
  - `observation_radii`;
  - `basic_interaction_radii`;
  - `ultimate_interaction_radii`;
  - any minimal spatial fields required by Milestone 4 and still defensible as
    dynamic state.
- Replace `Observation.observation_vectors` with fixed-shape families:
  - self features;
  - allied unit slots;
  - enemy unit slots;
  - map obstacle slots;
  - placeholder objective slots;
  - context features;
  - visibility masks, with Step 1 exposing only self and keeping enemy
    visibility false;
  - relation-specific ally and enemy targetability masks.
- Keep `ActionMask.target` as the flat selection-head mask aligned to selection
  indices: column `0` is `none`, columns `1..5` are allied slots, and columns
  `6..10` are enemy slots.

Implementation ownership:

- Codex may help name fields, document contracts, and review shapes.
- The user writes the contract changes that determine simulator semantics.

Acceptance:

- Core construction tests cover every new field shape and dtype.
- Reset returns fixed-shape state, observation, masks, and info.
- Existing M3 tests are updated to assert the new M4 schema, not dummy vectors.
- No movement or geometry semantics are implemented in this step unless needed
  for reset construction.

### Step 2: Add Geometry Helper Entry Points

Status: complete after Checkpoint 13 review.

Target files:

- `src/marl_battlegrounds/core/geometry.py`
- `tests/test_geometry.py`

Required helper responsibilities:

- boundary projection;
- disc-pillar overlap/projection;
- disc-rotated-rectangle overlap/projection;
- fixed-pass best-effort agent-agent body blocking with bounded residual;
- fixed-substep movement projection;
- segment-circle intersection for LOS;
- segment-rotated-rectangle intersection for LOS;
- obstacle active-slot handling.

Step 2 scope clarification:

- Step 2 creates LOS geometry primitives only.
- Step 2 must not wire LOS into `Observation`, visibility masks,
  targetability masks, or `env.step`.
- Later observation/visibility work must consume these core geometry helpers
  instead of reimplementing LOS.

Implementation ownership:

- The user writes the geometry algorithms.
- Codex may create empty skeletons with TODOs if needed, but should not fill in
  projection or intersection logic.

Acceptance:

- Low-level tests exist and describe expected behavior.
- Continuous geometry assertions use explicit tolerances.
- Inactive obstacle slots are ignored.
- Zero-distance fallback cases are tested.
- Helpers are JAX-compatible and avoid Python data-dependent control flow.

### Step 3: Integrate Movement Into `step`

Status: complete.

Target files:

- `src/marl_battlegrounds/core/env.py`
- `src/marl_battlegrounds/core/geometry.py`
- `tests/test_movement_integration.py`

Required behavior:

- Active alive agents map movement IDs to intended displacement.
- `stay` preserves position.
- Cardinal directions move by placeholder speed.
- Diagonal movement has normalized displacement.
- Movement projection prevents final invalid positions.
- Padded and inactive agents do not move.
- Rewards remain zero placeholders.
- `terminated` remains false.
- `truncated` remains horizon-based.

Implementation ownership:

- The user writes movement and projection integration.
- Codex reviews for JAX compatibility, state ownership, and shape stability.

Acceptance:

- Step movement tests pass under ordinary execution.
- JIT step smoke still passes.
- Scanned rollout smoke still passes.
- Projected positions remain valid after movement.

Current reality:

- `env.step` maps discrete movement IDs to per-slot continuous deltas.
- Movement uses authoritative `state.movement_speeds`, not config defaults after
  state construction.
- Movement is projected through `project_movement_with_geometry`.
- Step-level movement integration tests, non-stay JIT coverage, scanned rollout
  coverage, and the full quality gate have passed.

### Step 4: Implement Line of Sight, Visibility, and Targetability

Status: complete.

Target files:

- `src/marl_battlegrounds/core/env.py`
- `src/marl_battlegrounds/core/types.py`
- `tests/test_observation_masks.py`

Inspected but not ordinarily modified:

- `src/marl_battlegrounds/core/geometry.py`
- `tests/test_geometry.py`

Planning clarification: low-level LOS primitive tests live in
`tests/test_geometry.py`. Step 4 owns env-level consumption of
`has_clear_line_of_sight`, not duplicate LOS formula tests in a separate
`tests/test_line_of_sight.py` file.

Required behavior:

- Pillars and walls block LOS.
- Agents do not block LOS.
- Dynamic units are visible only when they are active, alive, inside observation
  radius, and have clear LOS from the observer.
- Units outside observation radius or behind LOS-blocking geometry are hidden or
  zero-filled according to the documented observation convention.
- When LOS is blocked, the preferred v1 behavior is to mask the whole dynamic
  unit feature row, not only position.
- Basic targetability requires visibility, the observer's
  `state.basic_interaction_radii`, active state, alive state, and placeholder
  class legality.
- `state.ultimate_interaction_radii` must not affect Milestone 4 targetability.
- Placeholder class legality should be deliberately simple and documented until
  Milestone 5 replaces it with class semantics.

Implementation ownership:

- The user writes LOS, visibility, targetability, and observation construction.
- Codex reviews contracts and tests.

Acceptance:

- Clear LOS, pillar-blocked LOS, wall-blocked LOS, rotated-wall-blocked LOS,
  tangent/near-tangent cases, and "agents do not block LOS" cases are tested.
- LOS-blocked units are not visible and are not targetable.
- Dynamic unit feature rows are masked when LOS is blocked.
- Inactive obstacle rows do not block visibility.
- Outside-radius visible targets are not targetable.

Current reality:

- Shared `AGENT_FEATURE_*` indices exist for self rows and unit-candidate rows.
- `reset` and `step` use shared observation and action-mask builders.
- Dynamic visibility combines active state, alive state, per-slot observation
  radius, and clear LOS.
- Visible ally/enemy candidate rows use the shared agent-feature schema.
- Hidden, out-of-radius, inactive, dead, and padded candidate rows are fully
  zeroed.
- Static map obstacle features remain globally observed.
- Basic targetability combines visibility, observer-specific
  `state.basic_interaction_radii`, active/alive state, and neutral placeholder
  class legality.
- `ActionMask.target` derives from `none`, ally targetability, and enemy
  targetability.
- `state.ultimate_interaction_radii` remains preserved for Milestone 5 but does
  not affect Milestone 4 basic targetability.

### Step 5: Build Selection, Movement, and Ultimate Masks

Status: complete.

Target files:

- `src/marl_battlegrounds/core/env.py`
- `tests/test_observation_masks.py`

Required behavior:

- Build masks from `state.active_mask`, `state.alive_mask`, positions, radii, and
  config geometry.
- Keep movement mask simple unless a movement command is explicitly disallowed by
  task constraints; ordinary collision feasibility is handled by projection.
- Keep selection index `0` valid for active alive agents.
- Mask padded, inactive, dead, hidden, outside-radius, and LOS-blocked unit slots.
- Keep ultimate action `0` valid for active alive agents.
- Keep ultimate action `1` invalid until Milestone 5.

Implementation ownership:

- The user writes mask semantics and semantic assertions.
- Codex may write helper comments, review tests, and update this progress log.

Acceptance:

- Padded slots are masked.
- Dead slots are masked.
- Hidden slots are masked.
- Outside-radius slots are masked.
- LOS-blocked slots are masked.
- `none` remains valid.
- Ultimate use remains unavailable.

Current reality:

- Movement masks remain active/alive-gated placeholders; collision feasibility
  is still handled by projection rather than action masking.
- `ActionMask.target` is active/alive-gated and built from `none`, ally
  targetability, and enemy targetability.
- Ultimate action `0` is valid for active alive agents and ultimate action `1`
  remains invalid until Milestone 5.

### Step 6: Add Geometry Renderer And Manual-Control Debug Harness

Status: complete.

Target files:

- `src/marl_battlegrounds/rendering/geometry.py`
- `src/marl_battlegrounds/rendering/manual_control.py`
- `tests/test_renderer_smoke.py`
- `tests/test_manual_control_smoke.py`
- `scripts/dev/geometry_debug_renderer.py`
- `scripts/dev/run_geometry_renderer.sh`

Required behavior:

- Render map boundaries, agents, collision discs, circular pillars, rotated walls,
  and optional LOS/debug overlays.
- Provide movement-only manual control for one configured agent slot using
  `WASDQEZC` input.
- Treat no input for a timestep as `MOVE_STAY`.
- Construct joint `Action` values for the real simulator transition.
- Split and return PRNG keys around manual-control stepping.
- Use simulator state/config as read-only inputs.
- Provide a thin dev launcher for a deterministic human-inspection geometry
  scene without adding simulator semantics.
- Avoid making `scripts/dev/check.sh` depend on optional visualization packages
  unless `matplotlib` is promoted to a required dependency.

Implementation ownership:

- Codex may write renderer scaffolding and docstrings.
- The user should still understand how rendered geometry maps to simulator
  geometry before accepting it.

Acceptance:

- Renderer imports without optional dependency failures.
- `render_geometry` returns an explicit render result containing the Matplotlib
  figure and axes.
- A smoke test can construct a render result or skip cleanly when visualization
  extras are unavailable.
- Manual-control helpers import without visualization extras.
- Manual-control stepping calls `core.env.step` and returns the next key plus the
  current step outputs.
- Renderer does not affect state transitions, observations, masks, rewards, or
  tests of simulator truth.
- Manual control does not reimplement masks, collision, LOS, stun, targetability,
  or action legality.

Current reality:

- `marl_battlegrounds.rendering` imports without eagerly importing optional
  visualization dependencies.
- `render_geometry(config, state, show_agent_indices=True)` consumes
  `EnvConfig` and `EnvState` read-only and returns a `RenderResult` containing
  the Matplotlib figure and axes when the optional `viz` dependency is
  installed.
- The renderer draws map boundaries, active agent discs, active pillars, and
  active rotated walls from existing simulator contracts.
- `tests/test_renderer_smoke.py` covers import safety and optional render-result
  construction.
- `marl_battlegrounds.rendering.manual_control` maps `WASDQEZC` movement input,
  constructs fixed-slot joint actions, splits PRNG keys, and calls the real
  simulator `step`.
- `tests/test_manual_control_smoke.py` covers import safety, key mapping, action
  construction, and one-step manual-control output shape without opening a GUI.
- Manual control temporarily reserves movement keys from Matplotlib's default
  shortcut keymaps while the debug loop is open, then restores the original
  keymaps after exit.
- `scripts/dev/run_geometry_renderer.sh` is a thin launcher for
  `scripts/dev/geometry_debug_renderer.py`, which constructs a deterministic
  debug scene with an effectively unbounded max-step horizon for manual
  geometry inspection.

### Step 7: Milestone Review

Status: complete.

Acceptance:

- `scripts/dev/check.sh` passes.
- Deterministic geometry tests pass.
- Movement integration tests pass.
- LOS tests pass.
- Observation and mask tests pass.
- Renderer smoke/manual validation is complete.
- No Milestone 5 combat/class mechanics have leaked in.
- No Milestone 6 sanctuary/death/respawn behavior has leaked in.
- This progress log is updated with the final review result.

Review result:

- `scripts/dev/check.sh` passed with 254 tests, ruff format check, ruff lint,
  and pyright.
- Deterministic geometry, movement integration, LOS, observation, mask,
  renderer, and manual-control smoke coverage passed.
- Review found no combat, class, objective, full sanctuary, death, respawn,
  replay, wrapper, baseline, or curriculum behavior in the committed Milestone 4
  scope.
- Packet docs, Milestone 5 notes, and scratch files remain excluded from the
  milestone implementation commit.

## Files to Inspect and Modify

Inspect:

- `MARL_BGs_Design_Document.pdf`, sections listed in Source Anchors.
- `SKILL.md`, especially code ownership, milestone planning, implementation
  packet, test boundary, design-drift, and Git workflow rules.
- `docs/dev/milestone_3.md` for the existing milestone-plan convention.
- `src/marl_battlegrounds/core/types.py` for current data contracts.
- `src/marl_battlegrounds/core/env.py` for reset and step entry points.
- `tests/test_core_spine.py` for existing M3 contract tests.
- `pyproject.toml` for required and optional dependencies.
- `scripts/dev/check.sh` for the local quality gate.

Modify during Milestone 4:

- `docs/dev/milestone_4.md` for planning and progress.
- `src/marl_battlegrounds/core/types.py` for fixed contracts.
- `src/marl_battlegrounds/core/env.py` for reset, step, observations, and masks.
- `src/marl_battlegrounds/core/geometry.py` for pure geometry helpers.
- `src/marl_battlegrounds/rendering/__init__.py` for the import-safe optional
  rendering package surface.
- `src/marl_battlegrounds/rendering/geometry.py` for static geometry rendering
  and reusable redraw support.
- `src/marl_battlegrounds/rendering/manual_control.py` for the movement-only
  manual-control debug harness.
- `tests/test_core_spine.py` to update contract expectations.
- `tests/test_geometry.py` for low-level geometry and LOS tests.
- `tests/test_movement_integration.py` for step-level movement tests.
- `tests/test_observation_masks.py` for observation and mask tests.
- `tests/test_renderer_smoke.py` for renderer import and render-result smoke
  coverage.
- `tests/test_manual_control_smoke.py` for manual-control import, key mapping,
  action construction, and one-step smoke coverage.
- `scripts/dev/geometry_debug_renderer.py` for the deterministic manual
  geometry debug scene.
- `scripts/dev/run_geometry_renderer.sh` for the local dev launcher.

Do not modify for this milestone unless a clear defect appears:

- training code;
- baseline code;
- task registry;
- replay/export code;
- metrics code;
- wrappers or external environment adapters;
- curriculum configuration files.

## Test Strategy

Run `scripts/dev/check.sh` before each coherent commit and before opening or
updating a PR.

Core contract tests should prove:

- all constants have expected values;
- config fields store static episode settings and fixed geometry;
- state fields store slot-aligned arrays;
- observation families have fixed shapes and dtypes;
- action masks have fixed shapes and dtypes;
- reset initializes positions, radii, masks, obstacle observations, and context
  fields deterministically;
- step preserves reward and done contracts from Milestone 3.

Geometry unit tests should prove:

- boundary projection keeps disc centers inside the shrunken valid region;
- active circular pillars block disc positions;
- active rotated rectangular walls block disc positions;
- inactive obstacle slots do not affect projection;
- agent-agent overlap is reduced or resolved for active alive agents through a
  deterministic fixed-pass residual solver;
- padded/inactive agents do not create collision constraints;
- zero-distance fallback cases avoid NaNs;
- projection preserves hard static validity after movement and keeps any
  residual active-agent overlap finite, bounded, and deterministic.

Line-of-sight tests should prove:

- clear segments remain clear;
- pillars block segment LOS;
- axis-aligned walls block segment LOS;
- rotated walls block segment LOS;
- tangent or near-tangent cases are handled with explicit tolerance;
- agents do not block LOS.

Movement integration tests should prove:

- `stay` preserves position;
- cardinal movement changes one coordinate by the configured speed before
  projection;
- diagonal movement uses normalized displacement;
- boundary projection prevents leaving the map;
- obstacle projection prevents overlap after step;
- agent-agent projection reduces residual overlap after simultaneous movement
  and preserves simulator safety under fixed projection passes;
- `jax.jit(step)` still works;
- scanned rollout still works.

Observation and mask tests should prove:

- self observations include the acting slot's own spatial fields;
- allied and enemy slots use stable relation-explicit ordering;
- global map observation exposes active obstacles and zeroed inactive obstacles;
- placeholder objective slots are typed, masked, and zeroed;
- units outside observation radius are hidden or zero-filled and not targetable;
- units behind active pillars or active walls are not visible;
- dynamic unit feature rows are masked when LOS is blocked;
- LOS-blocked units are not targetable;
- inactive obstacle rows do not block dynamic-unit visibility;
- agents do not block LOS in v1;
- outside-radius units are not targetable;
- padded, inactive, and dead units are not targetable;
- `none` selection is valid;
- ultimate use remains invalid until Milestone 5.

Renderer checks should prove:

- the renderer imports cleanly;
- optional visualization dependencies do not break the core quality gate;
- a simple state/config can produce a visual artifact or cleanly skipped smoke
  result;
- rendered geometry is derived from simulator config/state.
- manual-control helpers drive `env.step` rather than reimplementing simulator
  semantics;
- Matplotlib shortcut keymaps do not steal manual movement controls while the
  debug loop is open.

## Risk Register

- Risk: writing variable-shape geometry or observation code.
  - Impact: JIT, `vmap`, and `scan` compatibility degrade quickly.
  - Mitigation: use fixed constants, padded slots, and explicit active masks.

- Risk: storing static geometry as dynamic state.
  - Impact: Markov state becomes noisy and future config/replay ownership blurs.
  - Mitigation: keep fixed map geometry in `EnvConfig`; only dynamic quantities
    go in `EnvState`.

- Risk: renderer becomes the source of geometry truth.
  - Impact: tests and simulator semantics can diverge from visualization.
  - Mitigation: renderer reads config/state only; all predicates live in core
    geometry helpers.

- Risk: hidden unit features leak current dynamic state.
  - Impact: policies can exploit current position, health, cooldowns, class
    state, or other dynamic information for units that should be hidden by LOS.
  - Mitigation: when LOS-gated visibility is false, mask the whole dynamic unit
    feature row and set visibility and targetability masks false.

- Risk: inconsistent LOS use between observation, targetability, renderer, and
  scenario tooling.
  - Impact: agents see one truth, masks enforce another, and debugging tools show
    a third.
  - Mitigation: keep LOS predicates in `core.geometry`; later observation,
    targetability, renderer overlays, and scenario validation must consume those
    helpers or documented wrappers around them.

- Risk: implementing full class target legality too early.
  - Impact: Milestone 5 design gets pre-committed before class mechanics exist.
  - Mitigation: use explicit placeholder legality and document that class
    semantics replace it in Milestone 5.

- Risk: implementing full sanctuary behavior too early.
  - Impact: Milestone 6 death/respawn/sanctuary work becomes entangled with M4.
  - Mitigation: defer re-entry, enemy-entry, carrier, combat restriction, and
    respawn semantics to Milestone 6.

- Risk: mask tests encode semantics before the user understands them.
  - Impact: the project passes tests but the apprentice does not own the core
    invariants.
  - Mitigation: Codex specifies what each test must prove; the user writes the
    semantic assertions.

- Risk: relying on Python conditionals over JAX arrays.
  - Impact: eager tests pass while JIT fails.
  - Mitigation: use `jnp.where`, masks, fixed iteration counts, `jax.lax` where
    needed, and keep JIT smoke tests active.

- Risk: unstable selection-index alignment.
  - Impact: policies cannot learn reliable unit selection.
  - Mitigation: test selection indices against allied/enemy observation slots.

- Risk: collision projection becomes physically overambitious.
  - Impact: unnecessary complexity and brittle edge cases.
  - Mitigation: implement blocked movement, hard static-geometry validity, and
    fixed-pass best-effort agent-agent residual reduction only; no momentum,
    bounce, friction, restitution, or impulse physics.

## Acceptance Criteria

Milestone 4 is complete when:

- `EnvConfig` carries fixed map and obstacle configuration.
- `EnvState` carries dynamic positions, radii, active masks, and alive masks.
- `reset` returns valid initial spatial state and structured observations.
- `step` interprets movement actions and updates positions through projection.
- Agents cannot leave the map.
- Agents cannot overlap active pillars or active walls after projection.
- Active alive agent-agent overlap is reduced by fixed projection passes; any
  residual overlap is finite, bounded, deterministic, and safe for downstream
  masks and transitions.
- LOS checks distinguish clear, pillar-blocked, and wall-blocked segments.
- Observation-radius and LOS logic together control dynamic unit visibility.
- LOS-blocked dynamic unit feature rows are masked in observer-specific
  observations.
- Static map geometry is globally exposed through padded obstacle slots.
- Basic selection masks remove padded, inactive, dead, hidden, outside-basic-
  interaction-radius, and LOS-blocked targets.
- `none` selection remains valid only for active alive acting slots.
- Ultimate use remains masked until Milestone 5.
- Rewards remain zero placeholders.
- Termination and truncation semantics remain stable.
- JIT step and scanned rollout still work.
- Minimal renderer displays the simulator's geometry.
- Manual-control debugging drives the real simulator step while leaving
  collision, LOS, masks, targetability, and future status effects owned by core
  simulator code.
- `scripts/dev/check.sh` passes.
- No combat, class, full sanctuary, death, respawn, objective, replay, baseline,
  wrapper, or curriculum behavior has leaked into this milestone.

## Review and Git Plan

Branch:

- Use `feature/m4-geometry-masks`.
- The branch must be based on updated `main` after the accepted Milestone 3
  merge.

Commit split:

1. `[FEATURE] Add geometry configuration contracts`
2. `[FEATURE] Add geometry helper entry points`
3. `[FEATURE] Integrate spatial movement into step`
4. `[FEATURE] Add visibility observations and masks`
5. `[FEATURE] Add geometry renderer and manual control harness`
6. `[DOCS] Finalize Milestone 4 documentation`

Do not commit yet when:

- core geometry logic has not been reviewed;
- tests fail;
- this file is still an active planning/progress document;
- implementation includes unacknowledged design drift;
- unrelated changes are mixed into the milestone branch.

Commit only after:

- the relevant step is coherent;
- targeted tests pass;
- `scripts/dev/check.sh` passes or the failure is understood and documented;
- Codex has reviewed the diff.

PR title when ready:

`[FEATURE] Add geometry, visibility, and mask mechanics`

PR description when ready:

```markdown
## Summary

- Add fixed map and obstacle geometry contracts.
- Add spatial movement, collision projection, line-of-sight, visibility, and
  selection-mask mechanics.
- Add deterministic geometry, movement, line-of-sight, observation, and mask
  tests plus a geometry renderer and manual-control debug harness.

## Design Anchors

- Milestone 4: Geometry, Movement, Visibility, and Masks.
- R11-R14: map geometry, line of sight, global map observation, and partial
  dynamic-unit observability.
- Sections 2.5, 2.14.4, 2.14.5, A.7, and A.8.

## Out of Scope

- Combat, classes, damage, healing, ultimates, cooldowns, death, respawn,
  full sanctuary behavior, objectives, replay, baselines, wrappers, and
  curriculum logic.

## Checks

- [ ] `scripts/dev/check.sh`
```

## Progress Log

| Step | Status | Notes |
| --- | --- | --- |
| 0. Branch and milestone plan | Complete | Branch created from updated `main`; plan drafted. |
| 1. Expand core contracts | Complete | Core contracts expanded; `scripts/dev/check.sh` passed. |
| 2. Add geometry helper entry points | Complete | Geometry helpers and low-level tests added; Checkpoint 13 reviewed; full gate passed. |
| 3. Integrate movement into `step` | Complete | Movement IDs map to per-slot deltas, `state.movement_speeds` drives displacement, projection runs through shared geometry, and non-stay JIT/scan movement tests pass. |
| 4. Implement LOS, visibility, targetability | Complete | Feature schema, shared builders, LOS-gated visibility masks, visible row filling, hidden row zeroing, basic-interaction targetability, and compiled/scan observation-mask coverage are implemented and tested. |
| 5. Build selection, movement, ultimate masks | Complete | `ActionMask.target` derives from `none`, ally targetability, and enemy targetability; movement and ultimate masks retain active/alive gating and ultimate-use unavailability. M5 owns conditional ultimate target masks. |
| 6. Add geometry renderer and manual-control harness | Complete | Static render-result API, reusable redraw path, movement-only manual-control helpers, deterministic dev launcher, and renderer/manual smoke tests added. |
| 7. Milestone review | Complete | `scripts/dev/check.sh` passed; final review found no M5/M6 behavior leakage; packet docs, Milestone 5 notes, and scratch files remain excluded from staging. |

## Current Position

Last verified quality gate:

- `scripts/dev/check.sh`

Result: passed during Step 7 closeout with 254 tests, render-result API,
import-safe optional visualization handling, manual-control smoke coverage,
geometry/movement/LOS/observation/mask coverage, ruff formatting check, ruff
linting, and pyright.

Known local Git state:

- Branch: `feature/m4-geometry-masks`.
- Milestone 4 plan is now completed documentation and may be staged with the
  final milestone commit.
- Step packet files remain untracked local working guides and must not be staged
  or committed.
- `src/marl_battlegrounds/core/chicken_scratch.py` is untracked scratch work and
  should not be staged with milestone implementation commits.
- `docs/dev/milestone_5.md` is not part of the Milestone 4 implementation
  commit.

Next implementation checkpoint:

- Commit and publish Milestone 4 closeout.

Implementation boundary:

- The user owns renderer semantics only where they define simulator truth.
- Codex owns the milestone-plan prose, review, Git guidance, and optional
  mechanical cleanup after approval.

Do not start yet:

- class/combat mechanics;
- objective mechanics;
- death, respawn, sanctuary, replay, wrapper, baseline, or curriculum behavior.
