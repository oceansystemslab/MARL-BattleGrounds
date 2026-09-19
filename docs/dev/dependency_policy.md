# Dependency Policy

Add a dependency only to meet a concrete requirement that the standard library
or an existing dependency cannot satisfy. Keep researcher runtime, optional
features and contributor tools separate. GPU JAX is the performance target;
installing extra libraries does not establish a performance result.

## Python

`pyproject.toml` declares the supported groups. `uv.lock` fixes the resolved
versions used by reproducible installs and qualification.

| Group | Purpose |
| --- | --- |
| Base | JAX/NumPy arrays, numerical analysis, configuration and schema validation; base Python also serves the native browser clients |
| `cuda13` | CUDA 13 JAX execution |
| `training` | Optional optimizer, model and checkpoint libraries; not a complete implemented learning workflow |
| `interop` | Optional Gymnasium/PettingZoo libraries; not a promise that all adapters exist |
| `viz` | Matplotlib for static/headless snapshots |
| `dev` | Pytest, Ruff, Pyright and pre-commit |

Change dependency intent and the lockfile together. Use locked installation in
CI and qualification; the check scripts assume dependencies are already prepared.
Do not move optional tool behavior into Core to avoid a dependency boundary.

DevClient and Replay Viewer use the base environment for their browser mode.
Their shell launchers select `viz` only with `--static`. Contributor Python CI
includes `dev`, `viz` and `training` so it can run the complete test inventory,
including numerical baseline checks. Browser CI includes `dev` only.
Training libraries remain optional for environment and browser users. A separate
base-only installation check proves that boundary. Replay listing and existing-file
validation remain import-light. Scripted demos materialize separately on CPU
before the immutable replay enters the Viewer; that is preparation, not a CPU
simulation performance claim.

## Native Browser Runtime

Python's standard-library server serves tracked HTML, CSS, SVG, WOFF2 and
JavaScript. There is no runtime package manager, framework or application build
bundle. Researchers do not need Node.js. Bundled assets work without a remote
asset service.

DevClient owns live debugger commands and local asset authoring. Replay Viewer
owns read-only artifact navigation plus explicitly requested export/analysis.
They share rendering code, but keep route and data authority separate. Python
owns simulator facts, validation, audience projection and metric access. The
browser formats authorized data and uses native DOM/SVG, pointer coordinates,
tooltips and Web Animations. It does not reconstruct hidden simulation truth.
See [A17](../design/specification_amendments.md#a17-sharedobs-recorded-visual-union-presentation)
for the rendering-only SharedObs visual-union boundary.

## Frontend Contributor Tools

`.node-version` records the required Node line. The independent
`web/visual_debugger/package-lock.json` fixes contributor tools:

| Tool | Use | Product Runtime Dependency |
| --- | --- | --- |
| TypeScript | Strict no-emit checking of JavaScript/JSDoc | No |
| Biome | Source formatting and lint | No |
| Playwright Test | Real Chromium and visual checks | No |
| `@types/node` | Development type declarations | No |

Install the locked tools and pinned test browser:

```bash
npm ci --prefix web/visual_debugger
npm run install:browser --prefix web/visual_debugger
```

Update `package.json` and its lockfile together; do not hand-edit resolved lock
entries. Adding a runtime build/framework requires a concrete reviewed benefit,
not merely a contributor-tool dependency. Follow the shared
[documentation standard](documentation_standard.md) when changing commands or
public setup requirements.

## Bundled Fonts and Assets

Atkinson Hyperlegible Regular/Bold WOFF2 files support readable local rendering,
repeatable screenshots and self-contained PNG export. Their license and source
record live beside the files:

```text
web/visual_debugger/assets/fonts/OFL.txt
web/visual_debugger/assets/fonts/PROVENANCE.md
```

Record source, license, size and runtime purpose before adding/replacing an
asset. Keep runtime assets in the server's explicit allowlist. Do not add a
hidden network fetch. Preserve third-party legal notices rather than rewriting
them as project prose.

## Updates and Checks

Review direct/transitive changes and licenses. Prepare both locked environments
before running affected correctness, import-isolation and public-command checks.
At qualification, run the [complete gates](quality_gates.md) for the frozen
candidate, then the separately required clean GPU gate before publication.

Keep downloaded browsers, caches, generated reports and failure artifacts
ignored. Track only intentional source/assets and approved small visual
baselines. A dependency update does not authorize regenerating expectations or
weakening tests to make the gate pass.
