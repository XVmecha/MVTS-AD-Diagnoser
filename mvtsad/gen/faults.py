"""Fault interventions: concrete do-operations on the generative equations.

Each intervention modifies exactly one thing: an additive term on a channel's
output (drift, spike, level_shift, oscillation), the channel's noise term
(variance_change), the channel's output wholesale (stuck_at), or one incoming
edge (correlation_break). Downstream effects arise purely through propagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mvtsad.schemas import FaultClass

SIGNED_CLASSES = {FaultClass.DRIFT, FaultClass.SPIKE, FaultClass.LEVEL_SHIFT}


@dataclass
class Intervention:
    fault_class: FaultClass
    channel: int  # correlation_break: the child (dst) channel of the cut edge
    onset: int
    end: int  # exclusive
    direction: int = 0  # +1 / -1 / 0
    magnitude_z: float | None = None
    params: dict = field(default_factory=dict)
    edge_idx: int | None = None  # correlation_break only

    def additive(self, t: int, sigma: float) -> float:
        """Additive term on the channel output at time t (0 outside window)."""
        if not (self.onset <= t < self.end):
            return 0.0
        fc = self.fault_class
        if fc == FaultClass.DRIFT:
            frac = (t - self.onset + 1) / (self.end - self.onset)
            return self.direction * self.magnitude_z * sigma * frac ** self.params["ramp_power"]
        if fc in (FaultClass.SPIKE, FaultClass.LEVEL_SHIFT):
            return self.direction * self.magnitude_z * sigma
        if fc == FaultClass.OSCILLATION:
            return self.magnitude_z * sigma * math.sin(2.0 * math.pi * (t - self.onset) / self.params["period"])
        return 0.0


def sample_intervention(
    rng: np.random.Generator,
    fault_class: FaultClass,
    channel: int,
    n_timesteps: int,
    magnitude: float,
    onset: int,
    edge_idx: int | None = None,
    cut_src: int | None = None,
) -> Intervention:
    """Randomize the parameters that keep any class from having a fixed
    fingerprint: duration, ramp shape, period, persistence."""
    T = n_timesteps
    fc = fault_class
    direction = 0
    magnitude_z: float | None = float(magnitude)
    params: dict = {}

    if fc == FaultClass.DRIFT:
        dur = int(rng.uniform(0.2, 0.5) * T)
        direction = int(rng.choice([-1, 1]))
        params["ramp_power"] = float(rng.choice([0.5, 1.0, 2.0]))
    elif fc == FaultClass.SPIKE:
        dur = int(rng.integers(1, 4))
        direction = int(rng.choice([-1, 1]))
    elif fc == FaultClass.LEVEL_SHIFT:
        dur = (T - onset) if rng.random() < 0.6 else int(rng.uniform(0.2, 0.5) * T)
        direction = int(rng.choice([-1, 1]))
    elif fc == FaultClass.STUCK_AT:
        dur = int(rng.uniform(0.1, 0.4) * T)
        magnitude_z = None
    elif fc == FaultClass.VARIANCE_CHANGE:
        dur = int(rng.uniform(0.1, 0.4) * T)
        up = rng.random() < 0.7
        # The SNR dial maps onto the multiplicative factor.
        params["factor"] = 1.0 + 0.8 * magnitude if up else 1.0 / (1.0 + 0.8 * magnitude)
        direction = 1 if up else -1
        magnitude_z = None
    elif fc == FaultClass.OSCILLATION:
        dur = int(rng.uniform(0.15, 0.4) * T)
        params["period"] = float(rng.uniform(10.0, 60.0))
        direction = 0
    elif fc == FaultClass.CORRELATION_BREAK:
        dur = (T - onset) if rng.random() < 0.7 else int(rng.uniform(0.2, 0.5) * T)
        magnitude_z = None
        params["cut_src"] = int(cut_src) if cut_src is not None else -1
    else:  # pragma: no cover
        raise ValueError(f"unknown fault class {fc}")

    end = min(T, onset + max(1, dur))
    return Intervention(
        fault_class=fc,
        channel=channel,
        onset=onset,
        end=end,
        direction=direction,
        magnitude_z=magnitude_z,
        params=params,
        edge_idx=edge_idx,
    )
