# Ubuntu Setup

Use the repository lockfiles to prepare the development environment. Run these
commands from the repository root. They install dependencies; the validation
scripts later use that prepared environment without changing it.

## Prepare Python and CUDA

The project requires Python 3.14 and `uv`. The reference workstation uses Ubuntu
24.04 LTS, an NVIDIA RTX 5090 and driver 595.71.05. The lockfile pins JAX 0.10.1;
the `cuda13` extra supplies its CUDA 13 dependencies. These are recorded setup
conditions, not a promise that every driver or GPU combination is qualified.

```bash
uv sync --locked --extra cuda13 --extra dev --extra viz --extra training
scripts/doctor/host_info.sh
```

A working NVIDIA driver is required for GPU execution. `host_info.sh` prints the
selected JAX backend and devices. Review the output before sharing machine
information. A CPU fallback is not a successful GPU check.

The `training` extra adds learner/optimizer/checkpoint packages and is required
for the complete contributor test suite. Ordinary environment and browser use
does not require it. The `interop`
extra adds Gymnasium/PettingZoo dependencies; installing them does not by itself
implement every adapter. See the [dependency policy](dependency_policy.md).

## Prepare Browser Contributor Tools

Researchers can use the DevClient and Replay Viewer without Node.js. Contributors
who change browser code need the Node version in `.node-version`, npm and the
locked frontend dependencies:

```bash
npm ci --prefix web/visual_debugger
npm run install:browser --prefix web/visual_debugger
```

## Check the Checkout

```bash
scripts/dev/check.sh
scripts/dev/check_frontend.sh
scripts/dev/check_gpu.sh --allow-dirty
```

The first two commands check Python and browser correctness. The last command
is a GPU diagnostic for a working checkout. Publication needs the separate
[clean committed GPU qualification](gpu_sanity.md), and commits must follow the
[frozen-candidate gate](quality_gates.md).

GPU/JAX execution is the performance target. CPU tests still check correctness
and compatibility; their durations help balance CI jobs. Follow the shared
[Documentation Standard](documentation_standard.md)
when changing files: clear module purpose, complete function contracts, and one
file-level description only for tests.
