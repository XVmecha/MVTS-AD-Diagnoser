"""Evidence extractor: deterministic detection layer.

Computes threshold crossings, change-points, slopes, lagged cross-correlation
shifts, and per-channel anomaly scores. Output is an event-stream of flag
tuples. Imperfection is mandatory: false positives and false negatives are
part of the task design.
"""
