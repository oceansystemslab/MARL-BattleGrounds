# Dependency Policy

MARL-BattleGrounds separates researcher runtime dependencies from optional
features and contributor-only tools. Add a dependency only when the standard
library or an existing package cannot satisfy a concrete requirement.

## Python

`pyproject.toml` records dependency intent and `uv.lock` records the resolved
environment.

| Group | Purpose |
| --- | --- |
| Base | JAX environment, arrays, validation, and the live loopback browser debugger. |
| `cuda13` | CUDA 13 JAX execution. |
| `training` | Learner/optimizer/checkpoint tooling. |
| `interop` | Gymnasium and PettingZoo adapters. |
| `viz` | Optional Matplotlib static/RGB/headless rendering. |
| `dev` | Pytest, Ruff, Pyright, and pre-commit. |

The live browser debugger uses base Python dependencies and a modern browser.
The shell launcher activates `viz` only for `--static`. Python CI installs
`dev+viz` so supported static painter behavior is exercised rather than
skipped.

Change `pyproject.toml` and `uv.lock` together. Use locked syncs in CI and
closeout gates. Do not add visualization or debugger behavior to the simulator
core to avoid an optional dependency.

## Browser runtime

The tracked runtime is native HTML, CSS, SVG, and JavaScript modules served by
the Python standard-library HTTP server. There is:

- no transpilation or generated bundle;
- no runtime package manager;
- no framework, state-store, WebSocket, or animation library;
- no external network asset;
- no Node.js requirement for researchers.

The browser uses native DOM/SVG rendering, pointer coordinate projection,
presentation hover, and the Web Animations API. Python owns authorized agent
hit testing; Python/Pydantic owns command/frame validation and simulator
authority.

## Frontend contributor tooling

`.node-version` pins Node 24 for contributors and CI.
`web/visual_debugger/package-lock.json` independently pins:

| Tool | Role | Runtime impact | License family |
| --- | --- | --- | --- |
| TypeScript | Strict no-emit checking of JavaScript/JSDoc | None | Apache-2.0 |
| Biome | JavaScript/CSS/HTML/JSON format and lint | None | MIT or Apache-2.0 |
| Playwright Test | Real-browser behavior and visual regression | None | Apache-2.0 |
| `@types/node` | Contributor type declarations | None | MIT |

Install with:

```bash
npm ci --prefix web/visual_debugger
npm run install:browser --prefix web/visual_debugger
```

Commit `package.json` and `package-lock.json` together when the frontend
toolchain changes. Do not hand-edit the lockfile. Node dependencies must remain
development-only unless a separately approved architecture revision establishes
a browser build/runtime need.

## Bundled font

The browser tracks Atkinson Hyperlegible Regular and Bold WOFF2 files for
readability and deterministic screenshots. The exact license and provenance
files live beside them:

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
- Run only the checks invalidated by a dependency update, followed by the
  complete closeout gate once the assembled change stops moving.
