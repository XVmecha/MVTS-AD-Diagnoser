"""Generator invariants.

The load-bearing test is twin isolation: a fault may only touch the root and
its graph descendants; every other channel must be bit-identical between the
faulted run and the clean twin. That property is what makes the answer key
definitionally true.
"""

import numpy as np
import pytest

from mvtsad.gen.scene import DifficultyCell, generate_scene
from mvtsad.schemas import AnswerKey, Direction, FaultClass

CLASSES = list(FaultClass)


def make(wiring_id=7, seed=42, **cell_kwargs):
    cell = DifficultyCell(**cell_kwargs)
    return generate_scene(f"scene-{wiring_id}-{seed}", "test", wiring_id, cell, seed)


def _regimes(scene):
    return ("a", "b") if scene.key.regime_switches else ("a",)


def test_determinism():
    a = make(wiring_id=3, seed=11, snr="med", depth=1)
    b = make(wiring_id=3, seed=11, snr="med", depth=1)
    assert np.array_equal(a.signals, b.signals)
    assert a.key.model_dump() == b.key.model_dump()


def test_clean_scene():
    s = make(wiring_id=5, seed=1, clean=True)
    assert s.key.roots == [] and s.key.induced == []
    assert s.key.clean_channels == list(range(s.key.n_channels))
    assert np.array_equal(s.signals, s.clean_signals)


def test_clean_scene_with_regime_switch():
    s = make(wiring_id=5, seed=2, clean=True, regime_switch=True)
    assert s.key.roots == []
    assert len(s.key.regime_switches) == 1
    assert np.array_equal(s.signals, s.clean_signals)


@pytest.mark.parametrize("fc", CLASSES)
def test_single_root_and_twin_isolation(fc):
    """Exactly one root, and non-descendants are untouched bit-for-bit."""
    for seed in (1, 2, 3):
        s = make(wiring_id=10 + seed, seed=seed, fault_class=fc, snr="high", depth=2)
        assert len(s.key.roots) == 1
        root = s.key.roots[0].channel
        desc = set(s.wiring.descendants(root, _regimes(s)))
        untouched = sorted(set(range(s.key.n_channels)) - desc - {root})
        assert np.array_equal(s.signals[:, untouched], s.clean_signals[:, untouched])
        # Induced channels can only be descendants of the root.
        assert {e.channel for e in s.key.induced} <= desc


def test_stuck_at_freezes_output():
    s = make(wiring_id=21, seed=9, fault_class=FaultClass.STUCK_AT, snr="high")
    root = s.key.roots[0]
    start, end = root.window
    vals = s.signals[start:end, root.channel]
    assert np.all(vals == vals[0])


def test_level_shift_mean_offset():
    s = make(wiring_id=22, seed=4, fault_class=FaultClass.LEVEL_SHIFT, snr="high", depth=0)
    root = s.key.roots[0]
    start, end = root.window
    diff = s.signals[:, root.channel] - s.clean_signals[:, root.channel]
    sign = 1.0 if root.direction == Direction.UP else -1.0
    sigma = s.wiring.channels[root.channel].sigma
    expected = sign * root.magnitude_z * sigma
    assert np.allclose(diff[start:end], expected)
    assert np.allclose(diff[:start], 0.0)


def test_drift_ramps_to_magnitude():
    s = make(wiring_id=23, seed=6, fault_class=FaultClass.DRIFT, snr="high", depth=0)
    root = s.key.roots[0]
    start, end = root.window
    diff = s.signals[:, root.channel] - s.clean_signals[:, root.channel]
    sigma = s.wiring.channels[root.channel].sigma
    sign = 1.0 if root.direction == Direction.UP else -1.0
    assert np.allclose(diff[:start], 0.0)
    assert np.isclose(diff[end - 1], sign * root.magnitude_z * sigma, rtol=1e-6)
    assert abs(diff[start]) < abs(diff[end - 1])


def test_variance_change_scales_noise():
    for seed in range(20):
        s = make(wiring_id=24, seed=seed, fault_class=FaultClass.VARIANCE_CHANGE, snr="high")
        root = s.key.roots[0]
        if root.direction == Direction.UP:
            break
    assert root.direction == Direction.UP
    start, end = root.window
    diff = s.signals[:, root.channel] - s.clean_signals[:, root.channel]
    assert np.allclose(diff[:start], 0.0)
    assert np.std(diff[start:end]) > 0.5 * s.wiring.channels[root.channel].sigma


def test_correlation_break_cuts_edge():
    s = make(wiring_id=25, seed=3, fault_class=FaultClass.CORRELATION_BREAK)
    root = s.key.roots[0]
    cut_src = root.params["cut_src"]
    assert cut_src >= 0
    # The root must actually have an incoming edge from the recorded source.
    assert any(e.src == cut_src for _, e in s.wiring.in_edges(root.channel))
    # The cut changes the child's trajectory inside the window.
    start, end = root.window
    diff = s.signals[start:end, root.channel] - s.clean_signals[start:end, root.channel]
    assert np.any(diff != 0.0)


def test_induced_causes_are_real_edges():
    s = make(wiring_id=26, seed=8, fault_class=FaultClass.LEVEL_SHIFT, snr="high", depth=2)
    root = s.key.roots[0].channel
    deviated = {e.channel for e in s.key.induced} | {root}
    for ev in s.key.induced:
        assert ev.depth >= 1
        assert ev.causes, f"induced event on channel {ev.channel} has no causes"
        for link in ev.causes:
            assert link.parent in deviated
            assert any(
                e.src == link.parent and e.lag == link.lag
                for _, e in s.wiring.in_edges(ev.channel)
            )


def test_clean_channels_partition():
    s = make(wiring_id=27, seed=5, fault_class=FaultClass.DRIFT, snr="high", depth=1)
    affected = {r.channel for r in s.key.roots} | {e.channel for e in s.key.induced}
    assert set(s.key.clean_channels) | affected == set(range(s.key.n_channels))
    assert set(s.key.clean_channels) & affected == set()


def test_key_json_roundtrip():
    s = make(wiring_id=28, seed=12, fault_class=FaultClass.OSCILLATION, snr="med")
    dumped = s.key.model_dump_json()
    restored = AnswerKey.model_validate_json(dumped)
    assert restored == s.key


def test_regime_switch_alters_dynamics_only_after_switch():
    from mvtsad.gen.simulate import draw_streams, simulate
    from mvtsad.gen.wiring import sample_wiring

    # Find a wiring whose regime B actually differs from regime A.
    wiring = None
    for wid in range(1, 50):
        w = sample_wiring(wid)
        if any(e.active_a != e.active_b or e.weight_a != e.weight_b for e in w.edges):
            wiring = w
            break
    assert wiring is not None
    T, t0 = 600, 300
    rng = np.random.default_rng(0)
    wander, noise = draw_streams(rng, wiring, T)
    x_plain = simulate(wiring, T, wander, noise, switch_time=None)
    x_switch = simulate(wiring, T, wander, noise, switch_time=t0)
    assert np.array_equal(x_plain[:t0], x_switch[:t0])
    assert not np.array_equal(x_plain[t0:], x_switch[t0:])
