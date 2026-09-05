# Browser tools migration

The former combined “Visual Debugger and Analyzer” surface is now the DevClient
and the separate Replay Viewer, with independent launchers and authority:

| Product | Use it for | Canonical launcher |
| --- | --- | --- |
| [DevClient](combat_debugger.md) | Combat Debugger plus local reusable Map and TDM Scenario authoring | `./scripts/dev/run_dev_client.sh` |
| [Replay Viewer](replay_viewer.md) | Existing replay artifacts, checked samples, and isolated scripted demonstrations | `./scripts/dev/run_replay_viewer.sh` |

Both products use the fixed Analysis presentation and the same native browser
visual language. They do not share command authority: the DevClient can author
assets and submit live actions, while the Replay Viewer is read-only.

## Command migration

| Previous intent | Current command |
| --- | --- |
| Open the developer workspace | `./scripts/dev/run_dev_client.sh` |
| Record a Combat Debugger episode | `./scripts/dev/run_dev_client.sh --record-replay PATH` |
| Render a Combat Debugger reset snapshot | `./scripts/dev/run_dev_client.sh --static` |
| Open a replay formerly selected with debugger `--replay` | `./scripts/dev/run_replay_viewer.sh --replay PATH` |
| Open a checked sample formerly selected with debugger `--sample-replay` | `./scripts/dev/run_replay_viewer.sh --sample-replay NAME` |
| Open a scripted demonstration formerly selected with debugger `--scenario` | `./scripts/dev/run_replay_viewer.sh --scenario NAME` |
| List replay scenarios or samples | `./scripts/dev/run_replay_viewer.sh --list-scenarios` or `./scripts/dev/run_replay_viewer.sh --list-sample-replays` |
| Render an exact replay frame | `./scripts/dev/run_replay_viewer.sh --replay PATH --static --frame-index N` |

The Combat Debugger lists execution-valid saved map and scenario revisions.
Scenario assets load their authored state; maps are clearly identified
deterministic default-5v5-TDM previews. Both authoring areas call the same strict
compile/revalidate loader through `Open in Debug`, and either team can use
manual control, the scripted Team Deathmatch controller, or the built-in Random
controller. Scripted TDM is the existing generic team-agnostic debugging and
regression controller; Random is diagnostic quality-control tooling. Neither is
an official baseline, Big 12 entrant, or scenario pressure controller.
Team B additionally offers **Scenario 1 Controller** under SharedObs only.
Load Scenario 1, keep Team A Manual, choose that Team B controller, and submit
turns; Reset restores the exact loaded starting state. The controller requires
a compatible short TDM snapshot and is never selectable for Team A.
Incompatible selections or loads leave the current session unchanged.
See the [Combat Debugger guide](combat_debugger.md#loading-saved-scenarios-and-map-previews)
for its roster and horizon requirements.
Future official scenario evaluations bind pressure controllers through separate
evaluation definitions while saved scenarios remain controller-independent.
DevClient use remains diagnostic. See
[A26](../design/specification_amendments.md#a26-scenario-pressure-controllers-and-behavioral-ablations)
and [A28](../design/specification_amendments.md#a28-scenario-pressure-controllers-in-the-devclient).
Save is the only way the DevClient persists asset content; it
creates durable numbered local revisions and never autosaves. Every applicable
selector exposes every latest revision in numeric-aware asset-ID order through
its native scrolling control. New Map and Scenario asset IDs use lowercase
snake case, and generated obstacle IDs use `obstacle_N`; visible content names
remain free-form. Confirmed deletion removes an unwanted saved asset. The
Replay Viewer intentionally has no authoring route, action composer, manual
submission, reset, or recording destination. The historical
`run_debug_renderer.sh` launcher remains a compatibility redirect only.

## Shared boundaries

- Python remains the authority for simulator state, legality, replay
  validation, audience projection, and metric access.
- Browser layout, help, panels, animation, and the 19 visible filter controls
  are presentation-only.
- The replay transport uses Start/End controls, exact seeks, and document-level
  unmodified Left/Right/Space shortcuts that yield to interactive controls.
- Node.js and npm are contributor tools, not researcher runtime dependencies.
- SharedObs replay presentation follows
  [specification amendment A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation):
  the recorded same-epoch visual union is rendering-only and is not a
  materialized learner input.

For contributor checks, see [quality_gates.md](quality_gates.md). For the
runtime/tooling split, see [dependency_policy.md](dependency_policy.md).
