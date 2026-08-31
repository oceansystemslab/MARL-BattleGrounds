#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

usage() {
  cat >&2 <<'EOF'
usage: scripts/dev/check_gpu.sh [--allow-dirty | --help]

With no arguments, requires a clean worktree and performs canonical maintainer
GPU qualification. --allow-dirty is diagnostic only and cannot qualify a
commit for publication or release.
EOF
}

repository_is_clean() {
  local untracked=""

  git -C "${REPO_ROOT}" diff --quiet --exit-code -- || return 1
  git -C "${REPO_ROOT}" diff --cached --quiet --exit-code HEAD -- || return 1
  untracked="$(git -C "${REPO_ROOT}" ls-files --others --exclude-standard)" || return 1
  [[ -z "${untracked}" ]]
}

candidate_fingerprint() {
  local head_revision=""
  local tree_revision=""

  head_revision="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)" || return 1
  tree_revision="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD^{tree})" || return 1
  printf '%s:%s\n' "${head_revision}" "${tree_revision}"
}

mode=qualification
case "${1:-}" in
  "")
    ;;
  --allow-dirty)
    mode=diagnostic
    shift
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac
if (( $# != 0 )); then
  usage
  exit 2
fi

if [[ -n "${JAX_PLATFORM_NAME+x}" ]]; then
  echo "error: unset deprecated JAX_PLATFORM_NAME before GPU qualification." >&2
  exit 2
fi
if [[ -n "${JAX_XLA_BACKEND+x}" ]]; then
  echo "error: unset deprecated JAX_XLA_BACKEND before GPU qualification." >&2
  exit 2
fi
if [[ -n "${JAX_DISABLE_JIT+x}" ]]; then
  echo "error: unset JAX_DISABLE_JIT before GPU qualification." >&2
  exit 2
fi
if [[ -n "${JAX_SKIP_CUDA_CONSTRAINTS_CHECK+x}" ]]; then
  echo "error: unset JAX_SKIP_CUDA_CONSTRAINTS_CHECK before GPU qualification." >&2
  exit 2
fi
if [[ -n "${PYTEST_ADDOPTS+x}" ]]; then
  echo "error: unset PYTEST_ADDOPTS before GPU qualification." >&2
  exit 2
fi

qualification_fingerprint=""
if [[ "${mode}" == qualification ]]; then
  if ! repository_is_clean; then
    echo "error: canonical GPU qualification requires a clean worktree." >&2
    echo "Use --allow-dirty only for a diagnostic run." >&2
    exit 2
  fi
  if ! qualification_fingerprint="$(candidate_fingerprint)"; then
    echo "error: could not fingerprint the GPU qualification candidate." >&2
    exit 1
  fi
  echo "Running canonical GPU qualification for ${qualification_fingerprint}."
else
  echo "DIAGNOSTIC ONLY: this result does not qualify a commit for publication or release."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required for GPU qualification." >&2
  exit 127
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is required for GPU qualification." >&2
  exit 127
fi

export JAX_PLATFORMS=cuda
export PYTHONDONTWRITEBYTECODE=1
export UV_NO_SYNC=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd -- "${REPO_ROOT}"
nvidia-smi
uv run --no-sync python - <<'PY'
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend import backend as backend_api

default_backend_name = jax.default_backend()
registered_backends = backend_api.backends()
default_backend = backend_api.get_backend()
cuda_backend = backend_api.get_backend("cuda")
default_devices = tuple(jax.devices())
cuda_devices = tuple(jax.devices("cuda"))

print("jax", jax.__version__)
print("default backend", default_backend_name)
print("registered backends", tuple(registered_backends))
print("backend platform", default_backend.platform)
print("backend platform version", cuda_backend.platform_version)
print("devices", default_devices)

if default_backend_name != "gpu":
    raise RuntimeError("Expected JAX's CUDA backend to report the generic 'gpu' name")
if set(registered_backends) != {"cuda"}:
    raise RuntimeError("JAX must initialize exactly the concrete CUDA backend")
if default_backend.platform != "gpu" or cuda_backend.platform != "gpu":
    raise RuntimeError("Expected both selected backend handles to be GPU backends")
if default_backend is not cuda_backend:
    raise RuntimeError("The selected default backend is not the concrete CUDA backend")
if not cuda_devices or default_devices != cuda_devices:
    raise RuntimeError("The default device inventory is not exactly the CUDA inventory")
if any(device.platform != "gpu" for device in cuda_devices):
    raise RuntimeError("Every concrete CUDA device must report JAX platform 'gpu'")
if "cuda" not in str(cuda_backend.platform_version).lower():
    raise RuntimeError("The selected GPU backend does not identify itself as CUDA")


@jax.jit
def accelerator_matmul(left: jax.Array, right: jax.Array) -> jax.Array:
    return left @ right


selected_device = cuda_devices[0]
left = jax.device_put(
    jnp.ones((2048, 2048), dtype=jnp.float32),
    selected_device,
)
right = jax.device_put(
    jnp.ones((2048, 2048), dtype=jnp.float32),
    selected_device,
)
result = accelerator_matmul(left, right)
result.block_until_ready()
result_devices = tuple(result.devices())
if result.shape != (2048, 2048):
    raise RuntimeError("Synchronized CUDA matrix multiplication has the wrong shape")
if result_devices != (selected_device,):
    raise RuntimeError("Synchronized accelerator work did not remain on CUDA")
if float(result[0, 0]) != 2048.0:
    raise RuntimeError("Synchronized CUDA matrix multiplication produced a bad result")

print("result", result.shape, result.dtype, result_devices)
PY

uv run --no-sync pytest \
  tests/test_core_spine.py::test_that_step_can_be_jit_compiled \
  tests/test_core_spine.py::test_step_can_run_in_scanned_rollout \
  -q \
  --maxfail=1

if [[ "${mode}" == qualification ]]; then
  if ! repository_is_clean; then
    echo "error: the worktree changed during canonical GPU qualification." >&2
    exit 1
  fi
  if ! final_fingerprint="$(candidate_fingerprint)"; then
    echo "error: could not fingerprint the validated GPU candidate." >&2
    exit 1
  fi
  if [[ "${qualification_fingerprint}" != "${final_fingerprint}" ]]; then
    echo "error: the qualified commit changed during GPU validation." >&2
    exit 1
  fi
  echo "Canonical GPU qualification passed."
else
  echo "Diagnostic GPU check passed; this is not release qualification."
fi
