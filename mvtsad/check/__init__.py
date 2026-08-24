"""Claim checker: deterministic scoring of structured claims against answer keys.

Serves three roles: eval harness, GRPO training reward, integrity monitor.

Import discipline: this subpackage must never import from mvtsad.gen or
mvtsad.extract. It is planned to split out as the standalone, domain-independent
mvtsad-check library, usable by anyone with injected ground truth.
"""
