# GPU Sanity

The reference GPU stack is validated with:

nvidia-smi
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"

Expected backend: gpu.
