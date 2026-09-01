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
manual control or the scripted Team Deathmatch controller. Save is the only way
the DevClient persists asset content; confirmed deletion removes an unwanted
saved asset. The Replay Viewer intentionally has no authoring route, action
composer, manual submission, reset, or recording destination. The historical
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
