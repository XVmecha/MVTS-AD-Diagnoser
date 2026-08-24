"""Simulation engine.

The engine is built for counterfactual twin runs: all stochastic streams
(baseline wander, noise) are drawn ONCE up front, then the scene is simulated
twice from the same streams, with and without interventions. The difference
between the two runs is therefore the pure causal footprint of the fault,
which is what makes the derived answer key definitionally true even under
interactions (regime switches, cut edges, feedback motifs).
"""

from __future__ import annotations

import math

import numpy as np

from mvtsad.gen.faults import Intervention
from mvtsad.gen.wiring import WiringInstance
from mvtsad.schemas import FaultClass


def draw_streams(rng: np.random.Generator, wiring: WiringInstance, n_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-draw the baseline wander and noise series for every channel.

    Returns (wander, noise), each of shape (T, N). Both are identical across
    the twin runs; variance_change scales the noise inside the faulted run.
    """
    T, N = n_timesteps, wiring.n_channels
    wander = np.zeros((T, N))
    noise = np.zeros((T, N))
    for i, ch in enumerate(wiring.channels):
        # Stationary AR(1) wander with std == wander_scale.
        innov_std = ch.wander_scale * math.sqrt(max(1e-12, 1.0 - ch.rho**2))
        innov = rng.normal(0.0, innov_std, T)
        s = rng.normal(0.0, ch.wander_scale)
        for t in range(T):
            s = ch.rho * s + innov[t]
            wander[t, i] = s
        if ch.noise == "gaussian":
            noise[:, i] = rng.normal(0.0, ch.sigma, T)
        elif ch.noise == "heavy_tailed":
            # Student-t(3) scaled to unit variance, then to sigma.
            noise[:, i] = ch.sigma * rng.standard_t(3, T) / math.sqrt(3.0)
        else:  # ar1
            phi = ch.noise_phi
            innov_std = ch.sigma * math.sqrt(max(1e-12, 1.0 - phi**2))
            e = rng.normal(0.0, innov_std, T)
            v = rng.normal(0.0, ch.sigma)
            for t in range(T):
                v = phi * v + e[t]
                noise[t, i] = v
    return wander, noise


def simulate(
    wiring: WiringInstance,
    n_timesteps: int,
    wander: np.ndarray,
    noise: np.ndarray,
    switch_time: int | None = None,
    interventions: tuple[Intervention, ...] = (),
) -> np.ndarray:
    """Run the structural equations forward. Returns signals of shape (T, N).

    Channel equation, evaluated in topological order so instant (lag-0) edges
    read the same-timestep parent value:

        x_i[t] = baseline_i + wander_i[t]
               + sum over active in-edges e: w_e(regime) * k_e(d_src) * sigma_i / sigma_src
               + noise_i[t] * variance_factor_i(t)
               + additive_fault_i(t)
        then stuck_at replacement, if active.

    where d_src = x_src - baseline_src (deviations propagate, so normal
    wander creates the coincidental correlations the extractor will flag).
    """
    T, N = n_timesteps, wiring.n_channels
    x = np.zeros((T, N))
    d = np.zeros((T, N))
    ema = np.zeros(len(wiring.edges))

    var_iv: dict[int, Intervention] = {}
    stuck_iv: dict[int, Intervention] = {}
    add_iv: dict[int, list[Intervention]] = {}
    cut_iv: dict[int, Intervention] = {}
    for iv in interventions:
        if iv.fault_class == FaultClass.VARIANCE_CHANGE:
            var_iv[iv.channel] = iv
        elif iv.fault_class == FaultClass.STUCK_AT:
            stuck_iv[iv.channel] = iv
        elif iv.fault_class == FaultClass.CORRELATION_BREAK:
            cut_iv[iv.edge_idx] = iv
        else:
            add_iv.setdefault(iv.channel, []).append(iv)

    frozen: dict[int, float] = {}
    sigmas = [ch.sigma for ch in wiring.channels]

    for t in range(T):
        regime = "b" if (switch_time is not None and t >= switch_time) else "a"

        # Lowpass edge states update from strictly past values (lag >= 1).
        for eidx, e in enumerate(wiring.edges):
            if e.kernel == "lowpass":
                src_val = d[t - e.lag, e.src] if t - e.lag >= 0 else 0.0
                ema[eidx] = (1.0 - e.alpha) * ema[eidx] + e.alpha * src_val

        for i in wiring.topo_order:
            ch = wiring.channels[i]
            contrib = 0.0
            for eidx, e in wiring.in_edges(i):
                if not e.active(regime):
                    continue
                civ = cut_iv.get(eidx)
                if civ is not None and civ.onset <= t < civ.end:
                    continue
                if e.kernel == "instant":
                    v = d[t, e.src]
                elif e.kernel == "lagged":
                    v = d[t - e.lag, e.src] if t - e.lag >= 0 else 0.0
                elif e.kernel == "lowpass":
                    v = ema[eidx]
                else:  # saturating
                    raw = d[t - e.lag, e.src] if t - e.lag >= 0 else 0.0
                    s = e.sat_scale * sigmas[e.src]
                    v = s * math.tanh(raw / s)
                contrib += e.weight(regime) * (v / sigmas[e.src]) * ch.sigma

            eps = noise[t, i]
            viv = var_iv.get(i)
            if viv is not None and viv.onset <= t < viv.end:
                eps *= viv.params["factor"]

            xi = ch.baseline + wander[t, i] + contrib + eps
            for iv in add_iv.get(i, ()):
                xi += iv.additive(t, ch.sigma)

            siv = stuck_iv.get(i)
            if siv is not None and siv.onset <= t < siv.end:
                if i not in frozen:
                    frozen[i] = xi  # value at onset, then held
                xi = frozen[i]

            x[t, i] = xi
            d[t, i] = xi - ch.baseline

    return x
