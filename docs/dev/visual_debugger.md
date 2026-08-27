# Browser tools migration

The former combined “Visual Debugger and Analyzer” surface is now two explicit
products with separate launchers and authority boundaries:

| Product | Use it for | Canonical launcher |
| --- | --- | --- |
| [Combat Debugger](combat_debugger.md) | Manual live work and optional recording in the fixed 20×10 arena | `./scripts/dev/run_debug_renderer.sh` |
| [Replay Viewer](replay_viewer.md) | Existing replay artifacts, checked samples, and isolated scripted demonstrations | `./scripts/dev/run_replay_viewer.sh` |

Both products use the fixed Analysis presentation and the same native browser
renderer. They do not share command authority: the Combat Debugger can submit
manual actions, while the Replay Viewer is read-only.

## Command migration

| Previous intent | Current command |
| --- | --- |
| Open the manual live arena | `./scripts/dev/run_debug_renderer.sh` |
| Record the manual arena | `./scripts/dev/run_debug_renderer.sh --record-replay PATH` |
| Render a manual reset snapshot | `./scripts/dev/run_debug_renderer.sh --static` |
| Open a replay formerly selected with debugger `--replay` | `./scripts/dev/run_replay_viewer.sh --replay PATH` |
| Open a checked sample formerly selected with debugger `--sample-replay` | `./scripts/dev/run_replay_viewer.sh --sample-replay NAME` |
| Open a scripted demonstration formerly selected with debugger `--scenario` | `./scripts/dev/run_replay_viewer.sh --scenario NAME` |
| List replay scenarios or samples | `./scripts/dev/run_replay_viewer.sh --list-scenarios` or `./scripts/dev/run_replay_viewer.sh --list-sample-replays` |
| Render an exact replay frame | `./scripts/dev/run_replay_viewer.sh --replay PATH --static --frame-index N` |

The Combat Debugger intentionally has no scenario selector or scripted-advance
control. The Replay Viewer intentionally has no action composer, manual
submission, reset, or recording destination. Each launcher rejects options
belonging to the other product with a migration error.

## Shared boundaries

- Python remains the authority for simulator state, legality, replay
  validation, audience projection, and metric access.
- Browser layout, help, panels, animation, ranges, and the 20 visual filters
  are presentation-only.
- The replay transport uses Start/End controls, exact seeks, and document-level
  unmodified Left/Right/Space shortcuts that yield to interactive controls.
- Node.js and npm are contributor tools, not researcher runtime dependencies.
- SharedObs replay presentation follows
  [specification amendment A11](../design/specification_amendments.md#a11-sharedobs-recorded-visual-union-presentation):
  the recorded same-epoch visual union is rendering-only and is not a
  materialized learner input.

For contributor checks, see [quality_gates.md](quality_gates.md). For the
runtime/tooling split, see [dependency_policy.md](dependency_policy.md).
