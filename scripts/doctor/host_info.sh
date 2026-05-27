#!/usr/bin/env bash
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
import jax

print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
PY
