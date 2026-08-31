"""Extractor invariants: determinism, per-class evidence signatures, mandated
imperfection, and behavior on constructed signals with known structure."""

import numpy as np

from mvtsad.extract.extractor import ExtractorConfig, extract
from mvtsad.gen.scene import DifficultyCell, generate_scene
from mvtsad.schemas import Direction, FaultClass


def _overlaps(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def _evidence_for(event, bundle):
    return [
        it for it in bundle.items
        if event.channel in it.channels and _overlaps(it.window, event.window)
    ]


def scene_and_bundle(fc, seed=1, wid=30, snr="high", depth=1):
    s = generate_scene(f"x-{fc.value}-{seed}", "test", wid, DifficultyCell(snr=snr, depth=depth, fault_class=fc), seed)
    return s, extract(s.signals, s.scene_id)


def test_determinism():
    s = generate_scene("d", "test", 31, DifficultyCell(snr="med"), 5)
    a = extract(s.signals, s.scene_id)
    b = extract(s.signals, s.scene_id)
    assert a == b


def test_ids_unique_and_schema_valid():
    s, bundle = scene_and_bundle(FaultClass.DRIFT, seed=2)
    ids = [it.id for it in bundle.items]
    assert len(ids) == len(set(ids))
    assert bundle.n_channels == s.key.n_channels
    for it in bundle.items:
        assert 0 <= it.window[0] < it.window[1] <= s.key.n_timesteps


def test_drift_leaves_slope_evidence():
    for seed in (2, 3, 4):
        s, bundle = scene_and_bundle(FaultClass.DRIFT, seed=seed, wid=30 + seed)
        root = s.key.roots[0]
        ev = _evidence_for(root, bundle)
        assert any(it.source == "slope" for it in ev)
        # Slope direction matches injected drift direction.
        slopes = [it for it in ev if it.source == "slope"]
        assert any(it.direction == root.direction for it in slopes)


def test_level_shift_leaves_step_evidence():
    for seed in (1, 2, 3):
        s, bundle = scene_and_bundle(FaultClass.LEVEL_SHIFT, seed=seed, wid=40 + seed)
        ev = _evidence_for(s.key.roots[0], bundle)
        assert any(it.source in ("changepoint", "threshold") for it in ev)


def test_clean_scenes_still_produce_flags():
    """Imperfection is mandated: spurious evidence must exist on clean data,
    at a volume that fits the prompt budget."""
    counts = []
    for seed in range(6):
        s = generate_scene(f"cl{seed}", "test", 60 + seed, DifficultyCell(clean=True, regime_switch=(seed % 2 == 0)), seed)
        counts.append(len(extract(s.signals, s.scene_id).items))
    assert min(counts) > 0
    assert np.mean(counts) < 60


def test_extractor_never_sees_key():
    """Pure function of the signals: identical signals, different keys,
    identical bundles (guards against accidental key plumbing)."""
    s = generate_scene("p", "test", 33, DifficultyCell(snr="high"), 9)
    assert extract(s.signals, "a").items == extract(s.signals, "b").items


def test_xcorr_detects_constructed_coupling_break():
    """Two channels coupled at lag +5; coupling severed at t=500."""
    rng = np.random.default_rng(0)
    T = 1000
    a = rng.normal(0, 1, T)
    b = 0.2 * rng.normal(0, 1, T)
    b[5:500] += 0.9 * a[:495]  # coupled in the calibration segment and until 500
    x = np.column_stack([a, b])
    bundle = extract(x, "syn")
    breaks = [
        it for it in bundle.items
        if it.source == "xcorr" and it.channels == [0, 1] and it.direction == Direction.DOWN
    ]
    assert breaks, "no coupling-loss evidence emitted"
    assert all(it.window[0] >= 400 for it in breaks)
    assert any("@+5@" in it.id for it in breaks)


def test_xcorr_detects_new_coupling_with_lag():
    """Channels uncoupled during calibration; coupling appears at t=400."""
    rng = np.random.default_rng(1)
    T = 1000
    a = rng.normal(0, 1, T)
    b = 0.2 * rng.normal(0, 1, T)
    b[400:] += 0.9 * a[395 : T - 5]  # b[t] tracks a[t-5] from t=400
    x = np.column_stack([a, b])
    bundle = extract(x, "syn2")
    new = [it for it in bundle.items if it.source == "xcorr" and it.channels == [0, 1]]
    assert new, "no new-coupling evidence emitted"
    assert any("@+5@" in it.id for it in new)


def test_caps_bound_volume():
    cfg = ExtractorConfig()
    s = generate_scene("cap", "test", 70, DifficultyCell(snr="high", fault_class=FaultClass.VARIANCE_CHANGE), 3)
    bundle = extract(s.signals, s.scene_id, cfg)
    per = {}
    for it in bundle.items:
        per.setdefault((it.source, tuple(it.channels)), []).append(it)
    caps = {
        "threshold": cfg.cap_threshold,
        "changepoint": cfg.cap_changepoint,
        "slope": cfg.cap_slope,
        "xcorr": cfg.cap_xcorr,
    }
    for (source, _), group in per.items():
        assert len(group) <= caps[source]
