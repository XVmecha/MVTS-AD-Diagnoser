"""Checker semantics: matching, recoverable recall, grounding, reward, strict eval."""

import pytest

from mvtsad.check.checker import CheckConfig, check_diagnosis, parse_diagnosis
from mvtsad.schemas import (
    AnswerKey,
    CauseLink,
    Claim,
    Diagnosis,
    Direction,
    EvidenceBundle,
    EvidenceItem,
    FaultClass,
    InducedEvent,
    RootEvent,
)

CFG = CheckConfig()


def make_key(**overrides) -> AnswerKey:
    base = dict(
        scene_id="s1",
        split="test",
        wiring_id=1,
        n_channels=8,
        n_timesteps=1000,
        roots=[
            RootEvent(
                channel=4,
                fault_class=FaultClass.DRIFT,
                window=(400, 600),
                direction=Direction.UP,
                magnitude_z=3.0,
            )
        ],
        induced=[
            InducedEvent(channel=7, window=(420, 600), causes=[CauseLink(parent=4, lag=15)], depth=1)
        ],
        regime_switches=[],
        clean_channels=[0, 1, 2, 3, 5, 6],
    )
    base.update(overrides)
    return AnswerKey(**base)


def make_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        scene_id="s1",
        n_channels=8,
        items=[
            EvidenceItem(id="thr:s4@412", source="threshold", channels=[4], window=(410, 420)),
            EvidenceItem(id="slope:s4", source="slope", channels=[4], window=(400, 600)),
            EvidenceItem(id="xcorr:s4~s7", source="xcorr", channels=[4, 7], window=(420, 600)),
            # A spurious flag on a clean channel: false-alarm raw material.
            EvidenceItem(id="thr:s2@100", source="threshold", channels=[2], window=(100, 103)),
        ],
    )


def root_claim(**overrides) -> Claim:
    base = dict(
        channel=4,
        window=(400, 600),
        claim_class="drift",
        role="root",
        direction=Direction.UP,
        magnitude_z=3.0,
        evidence=["thr:s4@412", "slope:s4"],
    )
    base.update(overrides)
    return Claim(**base)


def induced_claim(**overrides) -> Claim:
    base = dict(
        channel=7,
        window=(420, 600),
        claim_class="induced",
        caused_by=4,
        lag=15,
        evidence=["xcorr:s4~s7"],
    )
    base.update(overrides)
    return Claim(**base)


def perfect() -> Diagnosis:
    return Diagnosis(claims=[root_claim(), induced_claim()], clean=[0, 1, 2, 3, 5, 6])


def test_perfect_diagnosis():
    r = check_diagnosis(perfect(), make_key(), make_evidence())
    assert r.parsed
    assert r.event_recall == 1.0
    assert r.strict_claim_accuracy == 1.0
    assert r.grounding_rate == 1.0
    assert r.n_hallucinated == 0 and r.n_grounded_but_wrong == 0
    assert r.reward > 0.9


def test_invalid_json_hard_gate():
    assert parse_diagnosis("the fault is on channel 4, trust me") is None
    r = check_diagnosis(None, make_key(), make_evidence())
    assert not r.parsed and r.reward == 0.0


def test_parse_rationale_then_json():
    text = (
        "Channel 4 shows a sustained slope after t=412 {so does 7}. "
        'Therefore: {"claims": [{"channel": 4, "window": [400, 600], '
        '"class": "drift", "role": "root", "direction": "up", "magnitude_z": 3.0, '
        '"evidence": ["thr:s4@412"]}], "clean": [0, 1, 2, 3, 5, 6, 7]}'
    )
    d = parse_diagnosis(text)
    assert d is not None and d.claims[0].claim_class == "drift"


def test_grounded_but_wrong_vs_hallucinated():
    over_reading = Claim(
        channel=2, window=(95, 110), claim_class="spike", role="root",
        direction=Direction.UP, magnitude_z=2.0, evidence=["thr:s2@100"],
    )
    inventing = Claim(
        channel=5, window=(300, 400), claim_class="drift", role="root",
        direction=Direction.UP, magnitude_z=2.0, evidence=["thr:s5@300"],
    )
    r = check_diagnosis(
        Diagnosis(claims=[over_reading, inventing], clean=[]), make_key(), make_evidence()
    )
    assert r.n_grounded_but_wrong == 1
    assert r.n_hallucinated == 1
    assert r.strict_claim_accuracy == 0.0


def test_low_iou_unmatched():
    claim = root_claim(window=(0, 100), evidence=[])
    r = check_diagnosis(Diagnosis(claims=[claim], clean=[]), make_key(), make_evidence())
    assert r.verdicts[0].matched is None


def test_wrong_magnitude_partial_credit():
    claim = root_claim(magnitude_z=10.0)  # tolerance is 0.5 * 3.0
    r = check_diagnosis(Diagnosis(claims=[claim], clean=[]), make_key(), make_evidence())
    v = r.verdicts[0]
    assert v.matched is not None and not v.magnitude_ok and not v.strict_correct
    assert 0.0 < v.partial_credit < 1.0


def test_attribution_wrong_parent_and_lag_grading():
    wrong_parent = induced_claim(caused_by=2)
    r = check_diagnosis(Diagnosis(claims=[wrong_parent], clean=[]), make_key(), make_evidence())
    v = r.verdicts[0]
    assert v.attribution_credit == 0.0 and not v.strict_correct

    lag_off = induced_claim(lag=18)  # error 3, tolerance 5
    r = check_diagnosis(Diagnosis(claims=[lag_off], clean=[]), make_key(), make_evidence())
    v = r.verdicts[0]
    assert v.attribution_ok and v.strict_correct
    assert 0.5 < v.attribution_credit < 1.0

    lag_far = induced_claim(lag=40)
    r = check_diagnosis(Diagnosis(claims=[lag_far], clean=[]), make_key(), make_evidence())
    assert not r.verdicts[0].attribution_ok


def test_unrecoverable_event_excluded_from_recall():
    key = make_key(
        induced=[
            InducedEvent(channel=7, window=(420, 600), causes=[CauseLink(parent=4, lag=15)], depth=1),
            # No evidence anywhere near channel 6: extractor blindness.
            InducedEvent(channel=6, window=(430, 600), causes=[CauseLink(parent=4, lag=20)], depth=1),
        ],
        clean_channels=[0, 1, 2, 3, 5],
    )
    r = check_diagnosis(perfect(), key, make_evidence())
    assert r.n_events == 3 and r.n_recoverable == 2
    assert r.event_recall == 1.0  # both recoverable events matched
    assert r.extractor_ceiling == pytest.approx(1 / 3)


def test_one_to_one_matching():
    dup = Diagnosis(claims=[root_claim(), root_claim(window=(410, 610))], clean=[])
    r = check_diagnosis(dup, make_key(), make_evidence())
    matched = [v for v in r.verdicts if v.matched is not None]
    assert len(matched) == 1


def test_reward_ranks_quality():
    key, ev = make_key(), make_evidence()
    full = check_diagnosis(perfect(), key, ev).reward
    partial = check_diagnosis(Diagnosis(claims=[root_claim()], clean=[]), key, ev).reward
    spam_claims = [root_claim()] + [
        Claim(channel=c, window=(100, 900), claim_class="drift", role="root",
              direction=Direction.UP, magnitude_z=2.0, evidence=[])
        for c in (0, 1, 2, 3, 5)
    ]
    spam = check_diagnosis(Diagnosis(claims=spam_claims, clean=[]), key, ev).reward
    assert full > partial > spam


def test_clean_scene_empty_diagnosis_is_perfect():
    key = make_key(roots=[], induced=[], clean_channels=list(range(8)))
    r = check_diagnosis(Diagnosis(claims=[], clean=list(range(8))), key, make_evidence())
    assert r.strict_claim_accuracy == 1.0
    assert r.reward == pytest.approx(
        CFG.w_format + CFG.w_recall + CFG.w_precision + CFG.w_attribution + CFG.w_grounding
    )


def test_integration_gold_from_generated_scene():
    """A gold diagnosis rendered straight from a generated answer key must
    score perfectly against evidence derived from that key."""
    from mvtsad.gen.scene import DifficultyCell, generate_scene

    scene = generate_scene(
        "it-1", "test", 26, DifficultyCell(snr="high", depth=2, fault_class=FaultClass.LEVEL_SHIFT), 8
    )
    key = scene.key
    items = []
    claims = []
    for i, ev in enumerate(key.roots):
        items.append(EvidenceItem(id=f"ev{i}", source="detector", channels=[ev.channel], window=ev.window))
        claims.append(
            Claim(
                channel=ev.channel, window=ev.window, claim_class=ev.fault_class.value,
                role="root", direction=ev.direction, magnitude_z=ev.magnitude_z,
                evidence=[f"ev{i}"],
            )
        )
    for j, ev in enumerate(key.induced):
        items.append(
            EvidenceItem(id=f"ind{j}", source="detector", channels=[ev.channel], window=ev.window)
        )
        claims.append(
            Claim(
                channel=ev.channel, window=ev.window, claim_class="induced",
                caused_by=ev.causes[0].parent, lag=ev.causes[0].lag, evidence=[f"ind{j}"],
            )
        )
    bundle = EvidenceBundle(scene_id=key.scene_id, n_channels=key.n_channels, items=items)
    r = check_diagnosis(Diagnosis(claims=claims, clean=key.clean_channels), key, bundle)
    assert r.strict_claim_accuracy == 1.0
    assert r.event_recall == 1.0
    assert r.n_hallucinated == 0
