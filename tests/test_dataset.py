"""Dataset assembly and gold rendering.

The load-bearing test: a rendered gold completion, parsed by the checker's
own parser, must score strict accuracy 1.0, recall 1.0, grounding 1.0 against
its scene. Gold and checker are developed by the same hand, so this closes
the loop between the two definitions of correctness.
"""

import json

from mvtsad.check.checker import check_diagnosis, parse_diagnosis
from mvtsad.dataset import SplitSpec, WIRING_RANGES, build_split, grid_cells, iter_split, load_split
from mvtsad.extract.extractor import extract
from mvtsad.gen.scene import DifficultyCell, generate_scene
from mvtsad.render import render_gold, serialize_evidence
from mvtsad.schemas import AnswerKey, EvidenceBundle, FaultClass

TINY = SplitSpec(per_cell=1, clean=2)  # 27 + 2 = 29 scenes


def test_grid_is_27_cells():
    cells = grid_cells()
    assert len(cells) == 27
    assert len({cid for cid, _ in cells}) == 27


def test_split_ids_exclusive_and_class_balanced():
    rows = list(iter_split("dev", TINY))
    wids = [w for _, _, w, _, _ in rows]
    assert len(set(wids)) == len(wids)
    assert all(WIRING_RANGES["dev"] <= w < WIRING_RANGES["dev"] + 10_000 for w in wids)
    fault_classes = [c.fault_class for _, _, _, _, c in rows if not c.clean]
    counts = {fc: fault_classes.count(fc) for fc in FaultClass}
    assert min(counts.values()) >= 3  # rotation spreads all 7 classes


def test_build_split_deterministic(tmp_path):
    m1 = build_split("dev", tmp_path / "a", spec=TINY)
    m2 = build_split("dev", tmp_path / "b", spec=TINY)
    assert m1 == m2
    assert (tmp_path / "a" / "dev.jsonl").read_bytes() == (tmp_path / "b" / "dev.jsonl").read_bytes()
    assert m1["n_records"] == 29
    assert m1["root_recoverable_frac"] is not None
    records = load_split("dev", tmp_path / "a")
    assert len(records) == 29
    # Records round-trip through the schemas.
    for r in records[:3]:
        AnswerKey.model_validate(r["key"])
        EvidenceBundle.model_validate(r["evidence"])


def _scene(fc=None, seed=3, wid=90, clean=False, snr="high"):
    cell = DifficultyCell(clean=clean) if clean else DifficultyCell(snr=snr, depth=1, fault_class=fc)
    s = generate_scene(f"g-{seed}", "test", wid, cell, seed)
    return s, extract(s.signals, s.scene_id)


def test_gold_scores_perfect():
    for fc in (FaultClass.DRIFT, FaultClass.LEVEL_SHIFT, FaultClass.OSCILLATION):
        for seed in (1, 2, 3):
            s, bundle = _scene(fc=fc, seed=seed, wid=90 + seed)
            gold = render_gold(s.key, bundle, seed=seed)
            d = parse_diagnosis(gold)
            assert d is not None
            r = check_diagnosis(d, s.key, bundle)
            assert r.strict_claim_accuracy == 1.0, (fc, seed, r.verdicts)
            assert r.event_recall == 1.0
            assert r.grounding_rate == 1.0
            assert r.n_hallucinated == 0


def test_gold_clean_scene():
    s, bundle = _scene(clean=True, seed=5, wid=95)
    gold = render_gold(s.key, bundle, seed=5)
    d = parse_diagnosis(gold)
    assert d is not None and d.claims == []
    assert d.clean == list(range(s.key.n_channels))
    assert check_diagnosis(d, s.key, bundle).reward > 0.9


def test_gold_varies_but_content_stable():
    s, bundle = _scene(fc=FaultClass.DRIFT, seed=7, wid=97)
    a = render_gold(s.key, bundle, seed=1)
    b = render_gold(s.key, bundle, seed=2)
    assert a != b  # template/order variation
    da, db = parse_diagnosis(a), parse_diagnosis(b)
    assert {c.channel for c in da.claims} == {c.channel for c in db.claims}
    assert render_gold(s.key, bundle, seed=1) == a  # deterministic in seed


def test_serialize_evidence():
    s, bundle = _scene(fc=FaultClass.SPIKE, seed=9, wid=99)
    text = serialize_evidence(bundle, s.key.n_timesteps)
    assert text == serialize_evidence(bundle, s.key.n_timesteps)
    for it in bundle.items:
        assert it.id in text
    # Rough prompt-budget guard.
    assert len(text.split()) < 1200
