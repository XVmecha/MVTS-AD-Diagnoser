"""Scene generator: randomized causal DAGs, baseline dynamics, fault injection.

Emits raw signals plus the answer key (injection record). Faults are
interventions on the generative equations; induced effects arise through
propagation and are never injected separately.
"""
