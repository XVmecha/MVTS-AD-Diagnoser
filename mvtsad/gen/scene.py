"""Scene generation: wiring + streams + fault injection + answer key.

The answer key's induced events are derived from the divergence between the
faulted run and its counterfactual clean twin (same wiring, same stochastic
streams). Divergence flows only through causal edges, so the derivation is
true by construction, including under regime switches and cut edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mvtsad.gen.faults import SIGNED_CLASSES, Intervention, sample_intervention
from mvtsad.gen.simulate import draw_streams, simulate
from mvtsad.gen.wiring import WiringInstance, sample_wiring
from mvtsad.schemas import (
    AnswerKey,
    CauseLink,
    Direction,
    FaultClass,
    InducedEvent,
    RootEvent,
)

GEN_VERSION = 1
_SCENE_DOMAIN = 202

SNR_RANGES = {"low": (1.2, 2.0), "med": (2.0, 3.5), "high": (3.5, 6.0)}

# Divergence thresholding for induced-event derivation. Deliberately below
# typical detectability: the key records subtle truths; whether an event is
# RECOVERABLE from evidence is the checker's separate computation.
_DEV_Z = 0.75
_DEV_MIN_LEN = 3
_DEV_MAX_GAP = 10


@dataclass
class DifficultyCell:
    """One cell of the stratified difficulty grid (v1 dials only)."""

    snr: str = "med"  # "low" | "med" | "high"
    depth: int = 1  # target root descendant depth: 0, 1, or 2
    regime_switch: bool = False
    onset_near_switch: bool = False  # only meaningful when regime_switch
    clean: bool = False
    fault_class: FaultClass | None = None  # None = sample uniformly

    def __post_init__(self) -> None:
        if self.onset_near_switch and not self.regime_switch:
            raise ValueError("onset_near_switch requires regime_switch")
        if self.snr not in SNR_RANGES:
            raise ValueError(f"unknown snr level {self.snr}")
        if self.depth not in (0, 1, 2):
            raise ValueError("v1 depth target must be 0, 1, or 2")


@dataclass
class Scene:
    scene_id: str
    signals: np.ndarray  # the faulted run: what the extractor sees
    clean_signals: np.ndarray  # counterfactual twin (never shown downstream)
    key: AnswerKey
    wiring: WiringInstance
    interventions: tuple[Intervention, ...] = field(default_factory=tuple)


def _runs_to_window(mask: np.ndarray, min_len: int, max_gap: int) -> tuple[int, int] | None:
    """Merge True-runs separated by gaps <= max_gap; keep runs >= min_len.
    Returns the (start, end) span from first kept run start to last kept run
    end, or None."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    runs: list[list[int]] = [[int(idx[0]), int(idx[0]) + 1]]
    for j in idx[1:]:
        if j - runs[-1][1] <= max_gap:
            runs[-1][1] = int(j) + 1
        else:
            runs.append([int(j), int(j) + 1])
    kept = [r for r in runs if r[1] - r[0] >= min_len]
    if not kept:
        return None
    return kept[0][0], kept[-1][1]


def _regimes_in_window(window: tuple[int, int], switch_time: int | None) -> tuple[str, ...]:
    if switch_time is None:
        return ("a",)
    start, end = window
    regimes = []
    if start < switch_time:
        regimes.append("a")
    if end > switch_time:
        regimes.append("b")
    return tuple(regimes)


def _sample_fault(
    rng: np.random.Generator,
    wiring: WiringInstance,
    n_timesteps: int,
    cell: DifficultyCell,
    switch_time: int | None,
) -> Intervention:
    T = n_timesteps

    # Onset: near the regime switch (the false-attribution trap) or well away
    # from it, so the two cells stay distinct.
    if cell.regime_switch and cell.onset_near_switch:
        onset = int(np.clip(switch_time + rng.integers(-20, 21), 10, T - 50))
    else:
        onset = int(rng.uniform(0.15, 0.7) * T)
        if switch_time is not None:
            for _ in range(50):
                if abs(onset - switch_time) > 50:
                    break
                onset = int(rng.uniform(0.15, 0.7) * T)

    fc = cell.fault_class or FaultClass(rng.choice([f.value for f in FaultClass]))
    regime_onset = "b" if (switch_time is not None and onset >= switch_time) else "a"
    magnitude = float(rng.uniform(*SNR_RANGES[cell.snr]))

    if fc == FaultClass.CORRELATION_BREAK:
        # Root is the child of the cut edge; prefer edges whose child matches
        # the depth target.
        active = [(idx, e) for idx, e in enumerate(wiring.edges) if e.active(regime_onset)]
        if not active:
            active = list(enumerate(wiring.edges))

        def child_depth(e) -> int:
            desc = wiring.descendants(e.dst, (regime_onset,))
            return min(max(desc.values(), default=0), 2)

        matching = [(idx, e) for idx, e in active if child_depth(e) == cell.depth]
        pool = matching or active
        eidx, edge = pool[int(rng.integers(len(pool)))]
        return sample_intervention(
            rng, fc, edge.dst, T, magnitude, onset, edge_idx=eidx, cut_src=edge.src
        )

    # Other classes: pick the root by descendant depth in the onset regime.
    def node_depth(i: int) -> int:
        desc = wiring.descendants(i, (regime_onset,))
        return min(max(desc.values(), default=0), 2)

    depths = {i: node_depth(i) for i in range(wiring.n_channels)}
    candidates = [i for i, dm in depths.items() if dm == cell.depth]
    if not candidates:
        # Relax to the closest available depth; realized depth is recorded in
        # the key's difficulty dict either way.
        best = min(set(depths.values()), key=lambda dm: abs(dm - cell.depth))
        candidates = [i for i, dm in depths.items() if dm == best]
    root = int(rng.choice(candidates))
    return sample_intervention(rng, fc, root, T, magnitude, onset)


def _derive_induced(
    wiring: WiringInstance,
    x_clean: np.ndarray,
    x_fault: np.ndarray,
    iv: Intervention,
    switch_time: int | None,
) -> list[InducedEvent]:
    root = iv.channel
    sig = np.array([ch.sigma for ch in wiring.channels])
    z = np.abs(x_fault - x_clean) / sig

    dev_windows: dict[int, tuple[int, int]] = {}
    for i in range(wiring.n_channels):
        if i == root:
            continue
        w = _runs_to_window(z[:, i] > _DEV_Z, _DEV_MIN_LEN, _DEV_MAX_GAP)
        if w is not None:
            dev_windows[i] = w

    if not dev_windows:
        return []

    regimes = _regimes_in_window((iv.onset, iv.end), switch_time)
    graph_depth = wiring.descendants(root, regimes)

    events: list[InducedEvent] = []
    for c, window in sorted(dev_windows.items()):
        causes: list[CauseLink] = []
        for _, e in wiring.in_edges(c):
            if not any(e.active(r) for r in regimes):
                continue
            p = e.src
            if p == root or p in dev_windows:
                p_start = iv.onset if p == root else dev_windows[p][0]
                if p_start <= window[0] + 2:
                    causes.append(CauseLink(parent=p, lag=e.lag))
        if not causes:
            # Relax the ordering constraint: divergence can only have arrived
            # through an edge from a deviated parent or the root.
            for _, e in wiring.in_edges(c):
                if any(e.active(r) for r in regimes) and (e.src == root or e.src in dev_windows):
                    causes.append(CauseLink(parent=e.src, lag=e.lag))
        events.append(
            InducedEvent(
                channel=c,
                window=window,
                causes=causes,
                depth=graph_depth.get(c, max(graph_depth.values(), default=0) + 1),
            )
        )
    return events


def _direction_of(iv: Intervention) -> Direction | None:
    if iv.fault_class in SIGNED_CLASSES or iv.fault_class == FaultClass.VARIANCE_CHANGE:
        return Direction.UP if iv.direction > 0 else Direction.DOWN
    return None


def generate_scene(
    scene_id: str,
    split: str,
    wiring_id: int,
    cell: DifficultyCell,
    seed: int,
) -> Scene:
    """Deterministic in (wiring_id, cell, seed, GEN_VERSION)."""
    wiring = sample_wiring(wiring_id, GEN_VERSION)
    rng = np.random.default_rng(np.random.SeedSequence([GEN_VERSION, _SCENE_DOMAIN, seed]))

    T = int(rng.integers(500, 1001))
    switch_time = int(rng.uniform(0.3, 0.7) * T) if cell.regime_switch else None
    wander, noise = draw_streams(rng, wiring, T)

    interventions: tuple[Intervention, ...] = ()
    if not cell.clean:
        interventions = (_sample_fault(rng, wiring, T, cell, switch_time),)

    x_clean = simulate(wiring, T, wander, noise, switch_time, ())
    if interventions:
        x_fault = simulate(wiring, T, wander, noise, switch_time, interventions)
        iv = interventions[0]
        induced = _derive_induced(wiring, x_clean, x_fault, iv, switch_time)
        roots = [
            RootEvent(
                channel=iv.channel,
                fault_class=iv.fault_class,
                window=(iv.onset, iv.end),
                direction=_direction_of(iv),
                magnitude_z=iv.magnitude_z,
                params=dict(iv.params),
            )
        ]
    else:
        x_fault = x_clean
        induced = []
        roots = []

    affected = {r.channel for r in roots} | {e.channel for e in induced}
    clean_channels = sorted(set(range(wiring.n_channels)) - affected)

    difficulty: dict = {
        "snr": cell.snr,
        "depth_target": cell.depth,
        "regime_switch": cell.regime_switch,
        "onset_near_switch": cell.onset_near_switch,
        "clean": cell.clean,
    }
    if roots:
        difficulty["fault_class"] = roots[0].fault_class.value
        difficulty["realized_depth"] = max((e.depth for e in induced), default=0)

    key = AnswerKey(
        scene_id=scene_id,
        split=split,
        wiring_id=wiring_id,
        n_channels=wiring.n_channels,
        n_timesteps=T,
        roots=roots,
        induced=induced,
        regime_switches=[switch_time] if switch_time is not None else [],
        clean_channels=clean_channels,
        difficulty=difficulty,
    )
    return Scene(
        scene_id=scene_id,
        signals=x_fault,
        clean_signals=x_clean,
        key=key,
        wiring=wiring,
        interventions=interventions,
    )
