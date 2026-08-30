# Dependency Policy

MARL-BattleGrounds separates researcher runtime dependencies from optional
features and contributor-only tools. Add a dependency only when the standard
library or an existing package cannot satisfy a concrete requirement.

## Python

`pyproject.toml` records dependency intent and `uv.lock` records the resolved
environment.

| Group | Purpose |
| --- | --- |
| Base | JAX environment, arrays, validation, and the two native browser products: the live Combat Debugger and read-only Replay Viewer. |
| `cuda13` | CUDA 13 JAX execution. |
| `training` | Learner, optimizer, and checkpoint tooling. |
| `interop` | Gymnasium and PettingZoo adapters. |
| `viz` | Optional Matplotlib adapters for static/headless Combat Debugger snapshots and Replay Viewer frames. |
| `dev` | Pytest, Ruff, Pyright, and pre-commit. |

The [Combat Debugger](combat_debugger.md) and
[Replay Viewer](replay_viewer.md) share base Python dependencies and the
tracked native browser renderer. The replay launcher remains import-light for
listing and existing-artifact validation; scripted scenario materialization
runs separately on CPU before the immutable bundle enters the viewer.

Both shell launchers activate `viz` only when `--static` is present. Python CI
installs `dev+viz` so supported static-painter behavior is exercised rather
than skipped.

Change `pyproject.toml` and `uv.lock` together. Use locked syncs in CI and
closeout gates. Do not move debugger, viewer, or visualization behavior into
the simulator core to avoid an optional dependency.

## Native browser runtime

Both products serve the same tracked HTML, CSS, SVG, WOFF2, and JavaScript
modules through Python's standard-library HTTP server. Product identity and
route authority remain separate: the Combat Debugger receives live manual
commands, while the Replay Viewer receives read-only artifact navigation and
authorized export/metric operations.

There is:

- no transpilation or generated application bundle;
- no runtime package manager;
- no framework, state store, WebSocket, or animation library;
- no external network asset; and
- no Node.js requirement for researchers.

The browser uses native DOM/SVG rendering, pointer coordinate projection,
presentation hover, and the Web Animations API. Python owns authorized hit
testing, command/frame validation, simulator authority, replay validation,
audience projection, and metric authorization. The SharedObs recorded visual
union remains a rendering-only contract under
[specification amendment A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation),
not a browser-reconstructed or materialized learner input.

## Frontend contributor tooling

`.node-version` pins Node 24 for contributors and CI.
`web/visual_debugger/package-lock.json` independently pins:

| Tool | Role | Runtime impact | License family |
| --- | --- | --- | --- |
| TypeScript | Strict no-emit checking of JavaScript/JSDoc | None | Apache-2.0 |
| Biome | JavaScript/CSS/HTML/JSON formatting and lint | None | MIT or Apache-2.0 |
| Playwright Test | Real-browser behavior and visual regression | None | Apache-2.0 |
| `@types/node` | Contributor type declarations | None | MIT |

Install contributor tooling with:

```bash
npm ci --prefix web/visual_debugger
npm run install:browser --prefix web/visual_debugger
```

Commit `package.json` and `package-lock.json` together when the frontend
toolchain changes. Do not hand-edit the lockfile. Node dependencies must remain
development-only unless a separately approved architecture revision
establishes a browser build/runtime need.

## Bundled font

The native browser tracks Atkinson Hyperlegible Regular and Bold WOFF2 files
for readability, deterministic screenshots, and self-contained Replay Viewer
PNG export. Exact license and provenance files live beside them:

```text
web/visual_debugger/assets/fonts/OFL.txt
web/visual_debugger/assets/fonts/PROVENANCE.md
```

Do not replace or add font/media assets without recording provenance, license,
file size, and runtime use. Runtime assets must stay inside the server's
explicit allowlist and contain no remote fetch.

## Update discipline

- Pin contributor tools and browser versions deliberately.
- Review licenses and transitive changes before updating either lockfile.
- Keep generated reports, downloaded browsers, caches, and failure artifacts
  ignored.
- Keep individual tracked visual baselines below the repository's file-size
  policy.
- Run impact-selected checks after a dependency change, followed by the
  complete closeout gate once the assembled change stops moving.
