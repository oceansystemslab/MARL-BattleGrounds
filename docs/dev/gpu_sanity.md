# Local GPU Qualification

Normal Python, frontend, and browser validation runs on the CPU locally and on
GitHub-hosted Ubuntu runners. The repository has no GitHub Actions workflow that
targets the maintainer's physical RTX 5090. GPU qualification is deliberate,
local, and maintainer-only.

## Prepare the locked environment

Dependency installation is separate from validation:

```bash
uv sync --locked --extra cuda13 --extra dev
```

The qualification script uses the installed lockfile environment without
silently syncing or changing it.

## Formal qualification

Run the no-argument command from a clean candidate commit:

```bash
scripts/dev/check_gpu.sh
```

This is required before Codex pushes a commit, opens a PR, merges, or qualifies
a release. A dirty worktree is rejected. For development diagnosis only:

```bash
scripts/dev/check_gpu.sh --allow-dirty
```

The diagnostic result must be labeled as such and cannot qualify a release.
Any candidate change after a formal pass requires the CPU/browser pre-commit
gate and clean GPU qualification to be run again on the new bytes.

## Fail-closed CUDA contract

The script forces `JAX_PLATFORMS=cuda` and rejects conflicting platform, JIT
disablement, pytest-option, or CUDA constraint-bypass configuration. It must
fail rather than accept CPU fallback, eager execution, test collection without
execution, or a different accelerator backend.

With the pinned JAX stack, `jax.default_backend()`, the backend platform, and
`Device.platform` correctly report `"gpu"`, not `"cuda"`. Concrete CUDA
identity is proved separately by resolving the `cuda` backend/devices and
checking CUDA platform metadata. Qualification then performs JIT-compiled
2048-by-2048 matrix multiplication, synchronizes the result to defeat lazy
dispatch, and confirms that the result remains on the selected CUDA device.

Finally, the gate runs only these focused compiled-environment tests:

- `test_that_step_can_be_jit_compiled`
- `test_step_can_run_in_scanned_rollout`

The complete Python suite remains a CPU gate. A full GPU regression is not
part of routine contributor validation, publication, or release qualification.

## Security boundary

Deleting a workflow alone does not isolate a registered self-hosted runner.
The permanent boundary is stopping and uninstalling the runner service and
deregistering the runner from GitHub. This repository's validation contract
does not depend on an installed Actions listener or an automatically starting
systemd runner service.
