"""Provide actor actions, permitted policy inputs and built-in controllers.

The actor module assembles submitted actions. The no_shared_obs and shared_obs
modules route each actor's allowed inputs to a policy. The random and reactive
controllers use those inputs; they do not change simulator rules.

Policy adapters used by evaluation live in evaluation.policy_execution.
Importing this package alone does not load the numerical controller modules.
"""
