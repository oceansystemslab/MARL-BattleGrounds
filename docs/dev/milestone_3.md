# Milestone 3: Minimal JAX Environment Spine

Status: complete locally

## Milestone Objective

Milestone 3 locks the smallest useful JAX-native simulator contract before any
game mechanics are introduced.

This milestone delivers:

- explicit core PyTree-style state and output contracts;
- functional `reset(config, key)` and `step(config, state, joint_action, key)`;
- fixed static arrays using padded internal agent slots;
- dummy observations, active-slot action masks, zero rewards, and done flags;
- deterministic tests for the minimal reset/step spine;
- smoke coverage for JIT-compatible `step`;
- smoke coverage for batched rollout over the minimal transition.

This milestone deliberately does not deliver:

- movement, collision, geometry, line of sight, or visibility;
- targeting, damage, healing, death, respawn, objectives, or events;
- replay export, logging, wrappers, task registry, baselines, or curriculum logic;
- reward semantics beyond zero placeholder rewards.

The purpose is to stabilize the core environment surface before Milestone 4 adds
spatial mechanics and before Milestone 5 adds combat.

## Source Anchors

- `MARL_BGs_Design_Document.pdf`, section 4.3.3, "Milestone 3: Minimal JAX
  Environment Spine": requires explicit PyTree state, functional reset/step,
  static padded slots, dummy observations and masks, dummy rewards and done
  flags, JIT step, batched rollout, and deterministic reset behavior.
- `MARL_BGs_Design_Document.pdf`, Appendix A.4, "Core JAX-Native Environment
  API": defines `reset(c, k)` and `step(c, st, at, kt)` as explicit transition
  functions with explicit state, rewards, termination, truncation, masks, and
  info.
- `MARL_BGs_Design_Document.pdf`, Appendix A.1-A.2: requires static JAX arrays,
  explicit PRNG keys, explicit state passing, fixed-shape padded slots, JIT,
  `vmap`, `scan`, deterministic testing, and clear separation between the
  internal core API and future compatibility wrappers.
- `SKILL.md`, "Mandatory milestone planning protocol": requires a dedicated
  milestone implementation plan before milestone implementation continues.

## Architecture Context

The core simulator spine is intentionally small:

- `marl_battlegrounds.core.types` owns the stable core data contracts.
- `marl_battlegrounds.core.env` owns functional environment entry points.
- `tests/test_core_spine.py` protects the current contracts through construction,
  reset, reward, and step tests.

The current core types are `EnvConfig`, `EnvState`, `Action`, `ActionMask`,
`Observation`, `Reward`, `DoneFlags`, and `Info`.

Important invariants:

- All internal per-agent arrays use `MAX_AGENT_SLOTS`.
- Smaller teams are represented through `active_mask`, not smaller arrays.
- `step` consumes explicit state and returns explicit next state.
- `step` must build action masks from state masks, not from `config.team_size`.
- `DoneFlags.done` is derived from `terminated OR truncated`.
- Random keys are explicit even when the current placeholder transition does not
  consume randomness.

Ownership boundaries:

- Simulator semantics stay in `core.env` and `core.types`.
- Future wrappers translate public conventions but must not own simulator
  semantics.
- Scenario loaders may eventually create valid states that did not come from
  ordinary reset, so transition logic must trust state masks.

## Dependency Order

1. Core type contracts and reset spine.
   - Complete.
   - Provides fixed slots, config, state, action, masks, observations, done
     flags, info, and deterministic reset.

2. Reward type contract.
   - Complete.
   - Provides one float32 scalar reward per internal agent slot.

3. Minimal placeholder `step`.
   - Complete locally in commit `d37f6b0`.
   - Adds the functional transition surface without movement, combat, objective,
     or reward semantics.

4. JIT smoke coverage.
   - Complete locally in commit `c1f500b`.
   - Proves the minimal transition can be compiled without hidden Python state.
   - `scripts/dev/check.sh` passed before commit.

5. Batched rollout smoke coverage.
   - Complete locally in commit `5fb6f21`.
   - Proves the minimal transition can be lifted into vectorized or scanned
     rollout structure.

6. Milestone review and PR preparation.
   - Complete locally.
   - Confirms no Milestone 4 or 5 mechanics leaked into the core spine.

## Step-by-Step Implementation Plan

### Step 0: Core Contract and Reset Spine

Complete.

Acceptance:

- Core type construction tests pass.
- `reset` returns fixed-shape state, observations, masks, and info.
- Reset masks padded slots correctly.
- Reset keeps ordinary scenario concerns out of the core reset path.

### Step 1: Reward Contract

Complete.

Acceptance:

- `Reward` exists as a NamedTuple-style core type.
- `Reward.rewards` is shape `(MAX_AGENT_SLOTS,)`.
- `Reward.rewards` is dtype `jnp.float32`.

### Step 2: Minimal Placeholder Step

Status: Complete locally in commit `d37f6b0`.

Target files:

- `src/marl_battlegrounds/core/env.py`
- `tests/test_core_spine.py`

Required `step` signature:

```python
step(
    config: EnvConfig,
    state: EnvState,
    joint_action: Action,
    key: Array,
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]
```

Required behavior:

- Increment `step_count` by exactly one.
- Preserve `agent_positions`, `team_ids`, `active_mask`, and `alive_mask`.
- Return zero `Reward.rewards` with shape `(MAX_AGENT_SLOTS,)` and dtype
  `jnp.float32`.
- Return dummy observations with reset-compatible shape and dtype.
- Return action masks gated from `next_state.active_mask`.
- Set `terminated` to scalar false.
- Set `truncated` from `next_step_count >= config.max_steps`.
- Leave `done` derived by `DoneFlags.done`.

Out of scope:

- interpreting `joint_action`;
- consuming `key`;
- movement, combat, objective, reward, event, replay, wrapper, or logging logic.

### Step 3: JIT Step Smoke

Status: Complete locally in commit `c1f500b`.

Add a narrow test proving the minimal `step` can be compiled.

Acceptance:

- The compiled call returns the same structure as ordinary `step`.
- Shapes and dtypes remain static.
- The test does not assert future mechanics.

### Step 4: Batched Rollout Smoke

Status: Complete locally in commit `5fb6f21`.

Add a narrow test proving the minimal transition can participate in batched or
scanned rollout.

Acceptance:

- The rollout executes for a small static horizon.
- The final step count matches the number of scanned steps.
- The test stays focused on transition plumbing, not environment mechanics.

### Step 5: Milestone Review

Status: Complete locally.

Review the full milestone for design drift.

Acceptance:

- `scripts/dev/check.sh` passes.
- No Milestone 4 or 5 mechanics are present.
- Public core contracts are documented by tests.
- Commit and PR prose explain that this is the minimal spine only.

Review result:

- No unacceptable design drift found.
- The implemented core spine matches the Milestone 3 contract: explicit
  `reset`/`step`, fixed padded slots, dummy observations and masks, zero
  rewards, done flags, JIT smoke coverage, and scanned rollout smoke coverage.
- No movement, collision, geometry, line-of-sight, visibility, combat, objective,
  replay, wrapper, baseline, or curriculum semantics were introduced.
- Remaining future work belongs to Milestone 4 and later mechanics.

## Files to Inspect and Modify

Inspect:

- `MARL_BGs_Design_Document.pdf` for Milestone 3 and Appendix A core API
  requirements.
- `SKILL.md` for milestone planning and code-ownership rules.
- `docs/dev/milestone_2.md` for the existing milestone documentation convention.
- `src/marl_battlegrounds/core/types.py` for data contracts.
- `src/marl_battlegrounds/core/env.py` for reset and step entry points.
- `tests/test_core_spine.py` for contract coverage.

Modify:

- `docs/dev/milestone_3.md` for milestone planning and progress.
- `src/marl_battlegrounds/core/env.py` for the minimal `step` entry point.
- `tests/test_core_spine.py` for step, JIT, and rollout smoke coverage.

Do not modify for this milestone unless a clear defect appears:

- compatibility wrappers;
- scenario loaders;
- replay or logging modules;
- baseline or training modules;
- configuration registry.

## Test Strategy

Run `scripts/dev/check.sh` before every milestone commit and before opening or
updating a PR.

Unit tests should prove:

- construction contracts for core types;
- reset fixed shapes, dtypes, and padded-slot masks;
- reward shape and dtype;
- step count increments;
- step output shapes and dtypes;
- state arrays are preserved by placeholder step;
- rewards are zero and float32;
- termination is false in placeholder step;
- truncation is horizon-based;
- done remains derived from `DoneFlags.done`;
- JIT smoke for step;
- batched or scanned rollout smoke.

The tests must not encode movement, collision, targeting, damage, objectives,
events, or future reward semantics.

## Risk Register

- Risk: recomputing step masks from `config.team_size`.
  - Impact: future scenario states could be invalidated by transition logic.
  - Mitigation: build masks from `next_state.active_mask`.

- Risk: Python control flow in step done logic.
  - Impact: JIT and vectorized rollout compatibility can break.
  - Mitigation: use JAX-compatible scalar arrays and array comparisons.

- Risk: action interpretation leaks into Milestone 3.
  - Impact: movement or combat semantics arrive before the transition contract is
    stable.
  - Mitigation: accept `joint_action` but do not interpret it.

- Risk: tests become future-mechanics tests too early.
  - Impact: brittle tests and misleading simulator guarantees.
  - Mitigation: assert only Milestone 3 contracts.

- Risk: duplicated mask construction starts to drift.
  - Impact: reset and step masks can diverge.
  - Mitigation: allow a later mechanical helper extraction after the placeholder
    step contract is reviewed.

## Acceptance Criteria

Milestone 3 is complete when:

- `reset` and `step` both exist as functional core entry points.
- All core outputs use fixed static shapes.
- Padded slots are represented by masks.
- Rewards and done flags exist in the step return path.
- `step` is JIT-compatible.
- A minimal batched or scanned rollout works.
- `scripts/dev/check.sh` passes.
- No movement, combat, objective, event, replay, wrapper, baseline, or curriculum
  semantics have been introduced.

## Review and Git Plan

Branch:

- `feature/m3-core-spine`

Commit split:

1. `[FEATURE] Add minimal reset spine`
   - Complete in commit `ebe18be`.

2. `[FEATURE] Add reward type contract`
   - Complete locally in commit `7c426ff`.

3. `[FEATURE] Add minimal step transition`
   - Adds placeholder step and its focused contract tests.
   - Complete locally in commit `d37f6b0`.

4. `[TEST] Add JIT smoke coverage for step`
   - Complete locally in commit `c1f500b`.

5. `[TEST] Add scanned rollout smoke coverage`
   - Complete locally in commit `5fb6f21`.

6. `[DOCS] Finalize Milestone 3 documentation`
   - Finalizes this milestone plan as completed documentation.
   - May include prose-only docstring/comment clarification for the placeholder
     `step` surface.
   - Commit only after the milestone is complete, so this plan records completed
     work rather than speculative planning.

`docs/dev/milestone_3.md` stayed untracked until Milestone 3 completion, per the
project coaching rule for milestone planning documents.

Do not commit yet when:

- core step logic has not been reviewed;
- tests fail;
- Milestone 4 or 5 behavior has leaked into the branch;
- unrelated changes are mixed into the diff.

PR strategy:

- Open or update a PR after the minimal step, JIT smoke, and batched-rollout smoke
  all pass local checks.
- PR title: `[FEATURE] Add minimal JAX environment spine`
- PR body should state that the branch locks the reset/step contract only and
  deliberately excludes mechanics.

## Progress Log

| Step | Status | Notes |
| --- | --- | --- |
| Core type contracts | Complete | Committed before this plan. |
| Minimal reset spine | Complete | Committed before this plan. |
| Reward type contract | Complete locally | Needs push with the Milestone 3 branch when ready. |
| Milestone 3 implementation plan | Complete locally | Ready to commit now that milestone implementation is complete. |
| Minimal placeholder step | Complete locally | Committed as `d37f6b0`; passed `uv run pytest tests/test_core_spine.py` before Step 3 edits. |
| JIT step smoke | Complete locally | Committed as `c1f500b`; `scripts/dev/check.sh` passed before commit. |
| Batched rollout smoke | Complete locally | Committed as `5fb6f21`; `scripts/dev/check.sh` passed before commit. |
| Milestone review and PR | Complete locally | Final review found no Milestone 4 or 5 behavior in the core spine. |
