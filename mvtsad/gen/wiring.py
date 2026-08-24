"""Per-scene randomized causal structure.

A wiring instance is a sparse DAG over 8 to 16 channels with per-edge
propagation kernels, two timescale groups, per-channel noise families, and a
second regime (regime "b") in which edges may vanish, appear, or change gain.
Wiring is a pure function of (wiring_id, GEN_VERSION), so split discipline
reduces to assigning wiring_id ranges to splits.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

KERNELS = ("instant", "lagged", "lowpass", "saturating")
NOISES = ("gaussian", "heavy_tailed", "ar1")

# Seed domain tags, mixed into SeedSequence so wiring randomness and scene
# randomness never collide even if ids overlap numerically.
_WIRING_DOMAIN = 101


@dataclass
class ChannelSpec:
    baseline: float
    sigma: float  # noise scale; z units throughout the project are per-channel sigma
    group: str  # "fast" | "slow"
    rho: float  # AR(1) coefficient of the baseline wander
    wander_scale: float  # stationary std of the wander
    noise: str  # noise family, one of NOISES
    noise_phi: float  # AR coefficient when noise family is "ar1"


@dataclass
class EdgeSpec:
    src: int
    dst: int
    weight_a: float  # unitless gain in regime A (contribution is sigma-normalized)
    weight_b: float
    lag: int  # timesteps; 0 only for the instant kernel
    kernel: str
    alpha: float  # lowpass smoothing factor
    sat_scale: float  # saturating kernel scale, in src-sigma units
    active_a: bool
    active_b: bool

    def active(self, regime: str) -> bool:
        return self.active_a if regime == "a" else self.active_b

    def weight(self, regime: str) -> float:
        return self.weight_a if regime == "a" else self.weight_b


@dataclass
class WiringInstance:
    wiring_id: int
    n_channels: int
    channels: list[ChannelSpec]
    edges: list[EdgeSpec]
    topo_order: list[int]  # simulation order; every edge goes earlier -> later
    _in_edges: dict[int, list[tuple[int, EdgeSpec]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._in_edges = {i: [] for i in range(self.n_channels)}
        for idx, e in enumerate(self.edges):
            self._in_edges[e.dst].append((idx, e))

    def in_edges(self, dst: int) -> list[tuple[int, EdgeSpec]]:
        return self._in_edges[dst]

    def descendants(self, root: int, regimes: tuple[str, ...] = ("a",)) -> dict[int, int]:
        """Min graph distance from root to each reachable node, over edges
        active in ANY of the given regimes. root itself is excluded."""
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_channels)}
        for e in self.edges:
            if any(e.active(r) for r in regimes):
                adj[e.src].append(e.dst)
        depth = {root: 0}
        q: deque[int] = deque([root])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    q.append(v)
        depth.pop(root)
        return depth


def _sample_edge(rng: np.random.Generator, src: int, dst: int, channels: list[ChannelSpec]) -> EdgeSpec:
    w = float(rng.uniform(0.4, 0.9))
    if rng.random() < 0.15:
        w = -w  # counteracting coupling (feedback motif material)
    kernel = str(rng.choice(KERNELS))
    if kernel == "instant":
        lag = 0
    elif channels[dst].group == "fast":
        lag = int(rng.integers(1, 6))
    else:
        lag = int(rng.integers(4, 21))
    return EdgeSpec(
        src=src,
        dst=dst,
        weight_a=w,
        weight_b=w,
        lag=lag,
        kernel=kernel,
        alpha=float(rng.uniform(0.1, 0.5)),
        sat_scale=float(rng.uniform(1.5, 3.0)),
        active_a=True,
        active_b=True,
    )


def sample_wiring(wiring_id: int, gen_version: int = 1) -> WiringInstance:
    rng = np.random.default_rng(np.random.SeedSequence([gen_version, _WIRING_DOMAIN, wiring_id]))
    n = int(rng.integers(8, 17))
    perm = [int(c) for c in rng.permutation(n)]
    pos = {ch: k for k, ch in enumerate(perm)}

    n_fast = int(rng.integers(2, n - 1))
    fast_set = set(int(c) for c in rng.choice(n, size=n_fast, replace=False))
    channels: list[ChannelSpec] = []
    for i in range(n):
        fast = i in fast_set
        sigma = float(rng.uniform(0.5, 2.0))
        channels.append(
            ChannelSpec(
                baseline=float(rng.uniform(-10.0, 10.0)),
                sigma=sigma,
                group="fast" if fast else "slow",
                rho=float(rng.uniform(0.6, 0.85)) if fast else float(rng.uniform(0.95, 0.995)),
                wander_scale=float(rng.uniform(0.2, 0.5) * sigma) if fast else float(rng.uniform(0.3, 0.8) * sigma),
                noise=str(rng.choice(NOISES)),
                noise_phi=float(rng.uniform(0.5, 0.9)),
            )
        )

    edges: list[EdgeSpec] = []
    pairs: set[tuple[int, int]] = set()
    for ch in perm[1:]:
        preds = perm[: pos[ch]]
        k = min(len(preds), int(rng.choice([0, 1, 2], p=[0.25, 0.55, 0.20])))
        if k == 0:
            continue
        for src in rng.choice(preds, size=k, replace=False):
            src = int(src)
            edges.append(_sample_edge(rng, src, ch, channels))
            pairs.add((src, ch))

    # A wiring with zero edges makes correlation_break impossible and defeats
    # the point of the benchmark; force one edge in the rare degenerate draw.
    if not edges:
        src, dst = perm[0], perm[-1]
        edges.append(_sample_edge(rng, src, dst, channels))
        pairs.add((src, dst))

    # Regime B: edges vanish or change gain.
    for e in edges:
        if rng.random() < 0.2:
            e.active_b = False
        if rng.random() < 0.5:
            e.weight_b = e.weight_a * float(rng.uniform(0.6, 1.4))

    # Regime B: a few edges appear that regime A does not have.
    for _ in range(int(rng.integers(0, 3))):
        dst = perm[int(rng.integers(1, n))]
        preds = [p for p in perm[: pos[dst]] if (p, dst) not in pairs]
        if not preds:
            continue
        src = int(rng.choice(preds))
        e = _sample_edge(rng, src, dst, channels)
        e.active_a = False
        e.active_b = True
        edges.append(e)
        pairs.add((src, dst))

    # Feedback motif: guarantee at least one counteracting (negative) edge
    # among regime-A active edges.
    a_edges = [e for e in edges if e.active_a]
    if a_edges and not any(e.weight_a < 0 for e in a_edges):
        e = a_edges[int(rng.integers(len(a_edges)))]
        e.weight_a = -abs(e.weight_a) * 0.6
        e.weight_b = -abs(e.weight_b) * 0.6

    return WiringInstance(
        wiring_id=wiring_id,
        n_channels=n,
        channels=channels,
        edges=edges,
        topo_order=perm,
    )
