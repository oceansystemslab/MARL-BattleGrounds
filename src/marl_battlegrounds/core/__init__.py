"""Implement the simulator's rules with fixed-shape JAX data.

``types`` defines the shared records and array axes. ``combat`` owns class
catalogs and status strengths; ``geometry`` owns physical projection and
line of sight. ``env.reset`` and ``env.step`` build and advance one game.
``config`` checks concrete inputs on the host before numerical execution.

Researcher setup and batching live in the public environment and task
modules. Core does not own policies, training, logging, replay storage or UI."""
