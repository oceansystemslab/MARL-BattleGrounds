#!/usr/bin/env bash

# Print host, Python, uv, NVIDIA and JAX backend information for diagnosis.
# Usage from the repository root: scripts/doctor/host_info.sh. Requires uv; its
# Python commands may prepare the project environment. Missing nvidia-smi is
# reported without failing this check, but a failed JAX initialization is fatal.
# The script prints machine/device details to stdout and writes no report file.
# Review that output before sharing it outside the project.
set -euo pipefail

echo "== Host =="
uname -a

echo
echo "== Python =="
uv run python --version

echo
echo "== uv =="
uv --version

echo
echo "== NVIDIA =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "== JAX =="
uv run python - <<'PY'
"""Print the installed JAX version and initialize its selected device backend."""

import jax

print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
PY
