"""Deterministic evidence extractor (design.md section 6).

Four statistical detectors: threshold crossings, change-points, slopes, and
lagged cross-correlation shifts. Each emits flag tuples into the evidence
bundle. The learned emitter (TOTO per-variate scores) is added later as an
optional dependency and is not required by the week-2 gate.

Calibration mimics deployed alarm systems: per-channel location/scale come
from the leading segment of the scene (the "historical normal" a plant would
have; faults onset at >= 0.15 T by construction), using median/MAD so heavy
tails do not inflate limits. Slow baseline wander therefore triggers spurious
flags, exactly as fixed limits do on real wandering channels. Flood control
follows deployed practice too: alarm rationalization, implemented as
deterministic per-channel (per-pair for xcorr) caps keeping the highest-
severity items per source, plus aggressive run merging.

Sensitivity defaults are PROVISIONAL (open design question: "representative
of deployed alarm behavior"); they may be tuned on the dev split only, never
against eval scores. Imperfection is mandated, not avoided: heavy-tailed
noise and slow baseline wander produce natural false positives, and low-SNR
faults fall below the limits, producing false negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvtsad.schemas import Direction, EvidenceBundle, EvidenceItem


@dataclass(frozen=True)
class ExtractorConfig:
    cal_frac: float = 0.15  # leading segment used as historical normal
    z_thr: float = 3.5  # sustained threshold limit (per-channel sigma units)
    z_spike: float = 3.5  # single-step limit (catches 1-3 step spikes)
    min_run: int = 3
    max_gap: int = 20  # merge nearby excursions into one alarm
    cp_halfwin: int = 25
    cp_thr: float = 2.5  # mean-shift, in sigma units
    slope_win: int = 80
    slope_thr: float = 2.5  # rise over the window, in sigma units
    xcorr_win: int = 150
    xcorr_stride: int = 100
    xcorr_maxlag: int = 20
    xcorr_couple_min: float = 0.4  # calibration |corr| to call a pair coupled
    xcorr_delta: float = 0.5  # change in |corr| that flags a shift/break
    xcorr_new: float = 0.65  # windowed |corr| that flags a new coupling
    # Alarm rationalization: per-channel (per-pair for xcorr) caps, keeping
    # the highest-severity items per source.
    cap_threshold: int = 2
    cap_changepoint: int = 3
    cap_slope: int = 2
    cap_xcorr: int = 2


def extract(signals: np.ndarray, scene_id: str, cfg: ExtractorConfig = ExtractorConfig()) -> EvidenceBundle:
    """Pure function of (signals, cfg). Never sees the answer key or the
    counterfactual clean twin."""
    T, n = signals.shape
    cal_end = max(30, int(cfg.cal_frac * T))
    loc = np.median(signals[:cal_end], axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(signals[:cal_end] - loc), axis=0), 1e-6)
    z = (signals - loc) / scale

    items: list[EvidenceItem] = []
    items += _thresholds(z, cfg)
    items += _changepoints(signals, scale, cfg)
    items += _slopes(signals, scale, cfg)
    items += _xcorr(signals, cal_end, cfg)
    return EvidenceBundle(scene_id=scene_id, n_channels=n, items=items)


def _cap(items: list[EvidenceItem], k: int) -> list[EvidenceItem]:
    """Alarm rationalization: keep the k highest-severity items per channel
    group (an item's group is its full channels list, so xcorr caps per pair).
    Deterministic: ties broken by id."""
    by_group: dict[tuple[int, ...], list[EvidenceItem]] = {}
    for it in items:
        by_group.setdefault(tuple(it.channels), []).append(it)
    kept = []
    for group in by_group.values():
        group.sort(key=lambda it: (-(it.severity or 0.0), it.id))
        kept.extend(group[:k])
    kept.sort(key=lambda it: (it.window[0], it.id))
    return kept


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Contiguous True-runs, with gaps <= max_gap merged."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs = [[int(idx[0]), int(idx[0]) + 1]]
    for j in idx[1:]:
        if j - runs[-1][1] <= max_gap:
            runs[-1][1] = int(j) + 1
        else:
            runs.append([int(j), int(j) + 1])
    return [(s, e) for s, e in runs]


def _thresholds(z: np.ndarray, cfg: ExtractorConfig) -> list[EvidenceItem]:
    items = []
    for ch in range(z.shape[1]):
        for s, e in _runs(np.abs(z[:, ch]) > cfg.z_thr, cfg.max_gap):
            seg = z[s:e, ch]
            peak = float(np.max(np.abs(seg)))
            # Short runs survive only as spikes above the higher limit.
            if e - s < cfg.min_run and peak <= cfg.z_spike:
                continue
            items.append(
                EvidenceItem(
                    id=f"thr:s{ch}@{s}",
                    source="threshold",
                    channels=[ch],
                    window=(s, e),
                    severity=peak,
                    direction=Direction.UP if seg[np.argmax(np.abs(seg))] > 0 else Direction.DOWN,
                )
            )
    return _cap(items, cfg.cap_threshold)


def _changepoints(x: np.ndarray, scale: np.ndarray, cfg: ExtractorConfig) -> list[EvidenceItem]:
    h = cfg.cp_halfwin
    T, n = x.shape
    if T < 2 * h + 1:
        return []
    cs = np.vstack([np.zeros((1, n)), np.cumsum(x, axis=0)])
    ts = np.arange(h, T - h)
    items = []
    for ch in range(n):
        m_after = (cs[ts + h, ch] - cs[ts, ch]) / h
        m_before = (cs[ts, ch] - cs[ts - h, ch]) / h
        d = (m_after - m_before) / scale[ch]
        # Greedy peak picking with an exclusion zone of h around each event.
        taken: list[int] = []
        for idx in np.argsort(-np.abs(d)):
            if abs(d[idx]) <= cfg.cp_thr:
                break
            if len(taken) >= cfg.cap_changepoint:
                break
            t = int(ts[idx])
            if all(abs(t - u) >= h for u in taken):
                taken.append(t)
                items.append(
                    EvidenceItem(
                        id=f"cp:s{ch}@{t}",
                        source="changepoint",
                        channels=[ch],
                        window=(t, t + 1),
                        severity=float(abs(d[idx])),
                        direction=Direction.UP if d[idx] > 0 else Direction.DOWN,
                    )
                )
    return items


def _slopes(x: np.ndarray, scale: np.ndarray, cfg: ExtractorConfig) -> list[EvidenceItem]:
    w = cfg.slope_win
    T, n = x.shape
    if T < w:
        return []
    k = np.arange(w) - (w - 1) / 2.0
    denom = float((k**2).sum())
    items = []
    for ch in range(n):
        # OLS slope of each length-w window starting at t, as total rise
        # over the window in sigma units.
        slope = np.convolve(x[:, ch], k[::-1], mode="valid") / denom
        rise = slope * w / scale[ch]
        for s, e in _runs(np.abs(rise) > cfg.slope_thr, cfg.max_gap):
            if e - s < cfg.min_run:
                continue
            seg = rise[s:e]
            items.append(
                EvidenceItem(
                    id=f"slope:s{ch}@{s}",
                    source="slope",
                    channels=[ch],
                    window=(s, min(T, e - 1 + w)),
                    severity=float(np.max(np.abs(seg))),
                    direction=Direction.UP if seg[np.argmax(np.abs(seg))] > 0 else Direction.DOWN,
                )
            )
    return _cap(items, cfg.cap_slope)


def _corr_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Correlation of (a[t], b[t+lag]) over the overlap, with window-global
    standardization. Positive lag = a leads b."""
    if lag >= 0:
        av, bv = a[: len(a) - lag], b[lag:]
    else:
        av, bv = a[-lag:], b[: len(b) + lag]
    if len(av) < 10:
        return 0.0
    av = av - av.mean()
    bv = bv - bv.mean()
    sa, sb = av.std(), bv.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(av @ bv / (len(av) * sa * sb))


def _best_lag(a: np.ndarray, b: np.ndarray, maxlag: int) -> tuple[int, float]:
    best_lag, best = 0, 0.0
    for lag in range(-maxlag, maxlag + 1):
        c = _corr_at(a, b, lag)
        if abs(c) > abs(best):
            best, best_lag = c, lag
    return best_lag, best


def _xcorr(x: np.ndarray, cal_end: int, cfg: ExtractorConfig) -> list[EvidenceItem]:
    T, n = x.shape
    # Calibration: which pairs are coupled, and at what lag.
    coupled: dict[tuple[int, int], tuple[int, float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            lag, c = _best_lag(x[:cal_end, i], x[:cal_end, j], cfg.xcorr_maxlag)
            if abs(c) >= cfg.xcorr_couple_min:
                coupled[(i, j)] = (lag, c)

    items = []
    for ws in range(cal_end, T - cfg.xcorr_win + 1, cfg.xcorr_stride):
        we = ws + cfg.xcorr_win
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in coupled:
                    lag, c_cal = coupled[(i, j)]
                    c_w = _corr_at(x[ws:we, i], x[ws:we, j], lag)
                    delta = abs(c_w) - abs(c_cal)
                    if abs(delta) > cfg.xcorr_delta:
                        items.append(
                            EvidenceItem(
                                id=f"xcorr:s{i}~s{j}@{lag:+d}@{ws}",
                                source="xcorr",
                                channels=[i, j],
                                window=(ws, we),
                                severity=float(abs(delta)),
                                direction=Direction.UP if delta > 0 else Direction.DOWN,
                            )
                        )
                else:
                    lag, c_w = _best_lag(x[ws:we, i], x[ws:we, j], cfg.xcorr_maxlag)
                    if abs(c_w) >= cfg.xcorr_new:
                        items.append(
                            EvidenceItem(
                                id=f"xcorr:s{i}~s{j}@{lag:+d}@{ws}",
                                source="xcorr",
                                channels=[i, j],
                                window=(ws, we),
                                severity=float(abs(c_w)),
                                direction=Direction.UP,
                            )
                        )
    return _cap(items, cfg.cap_xcorr)
