"""Deterministic claim checker (design.md sections 8 and 9).

One matching pass produces everything: the graded GRPO reward, the strict
binary eval verdicts, and the diagnostic splits (grounded-but-wrong vs
hallucinated, recoverable recall vs extractor ceiling). Same input, same
score, auditable per claim.

Semantics implemented here:
- Matching gate: same channel and window IoU >= iou_min, one-to-one greedy
  by descending IoU, over all key events (roots and induced alike). Class,
  direction, magnitude, attribution are scored fields, never matching gates.
- Recall denominators use RECOVERABLE events only: events with at least one
  evidence item overlapping their channel and window. Events invisible to
  the extractor are counted in extractor_ceiling instead.
- Grounding: a citation is valid iff the id exists in the bundle, the item
  covers the claimed channel, and the item's window overlaps the claimed
  window. Unmatched claims split into grounded_but_wrong (at least one valid
  citation) and hallucinated (none).
- Strict binary verdict ("one miss, zero"): matched, class exact, direction
  and magnitude within tolerance where the event defines them, role
  consistent, attribution edge and lag within tolerance for induced claims,
  and every citation valid with at least one present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from mvtsad.schemas import AnswerKey, Diagnosis, EvidenceBundle, InducedEvent, RootEvent


@dataclass(frozen=True)
class CheckConfig:
    iou_min: float = 0.5
    lag_tol: int = 5
    magnitude_rtol: float = 0.5
    # Provisional reward weights (design section 9: rebalanced during
    # training if a term saturates).
    w_format: float = 0.1
    w_recall: float = 0.3
    w_precision: float = 0.3
    w_attribution: float = 0.15
    w_grounding: float = 0.15
    w_hallucination: float = 0.1
    w_length: float = 0.02


@dataclass
class ClaimVerdict:
    claim_idx: int
    matched: tuple[str, int] | None  # ("root"|"induced", index into key list)
    iou: float = 0.0
    class_ok: bool = False
    direction_ok: bool = False
    magnitude_ok: bool = False
    role_ok: bool = False
    attribution_credit: float = 0.0  # induced only: 0.5 edge + 0.5 lag grade
    attribution_ok: bool = False  # strict: edge correct and lag within tol
    grounding: float = 0.0  # valid citations / citations (0 if none cited)
    partial_credit: float = 0.0  # graded per-claim credit, grounding-scaled
    strict_correct: bool = False
    category: str = "unmatched"  # "matched" | "grounded_but_wrong" | "hallucinated"


@dataclass
class CheckReport:
    parsed: bool
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    n_events: int = 0
    n_recoverable: int = 0
    n_recoverable_matched: int = 0
    event_recall: float = 1.0  # vs recoverable only; vacuous = 1.0
    extractor_ceiling: float = 0.0  # fraction of events with zero evidence
    claim_precision: float = 1.0  # mean partial credit; vacuous = 1.0
    attribution_score: float = 1.0  # vs recoverable induced; vacuous = 1.0
    grounding_rate: float = 1.0  # mean per-claim grounding; vacuous = 1.0
    n_hallucinated: int = 0
    n_grounded_but_wrong: int = 0
    strict_claim_accuracy: float = 1.0  # strict-correct claims / claims; vacuous = 1.0
    reward: float = 0.0


def parse_diagnosis(text: str) -> Diagnosis | None:
    """Extract the last valid claims JSON object from model output
    (rationale-first format: prose, then JSON). None = hard gate, score 0."""
    decoder = json.JSONDecoder()
    result = None
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(text, i)
            if isinstance(obj, dict) and "claims" in obj:
                try:
                    result = Diagnosis.model_validate(obj)
                except Exception:
                    pass
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return result


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def _events(key: AnswerKey) -> list[tuple[str, RootEvent | InducedEvent]]:
    return [("root", e) for e in key.roots] + [("induced", e) for e in key.induced]


def is_recoverable(event: RootEvent | InducedEvent, evidence: EvidenceBundle) -> bool:
    return any(
        event.channel in item.channels and _overlaps(item.window, event.window)
        for item in evidence.items
    )


def _grounding(claim, evidence: EvidenceBundle) -> float:
    if not claim.evidence:
        return 0.0
    by_id = {item.id: item for item in evidence.items}
    valid = 0
    for cid in claim.evidence:
        item = by_id.get(cid)
        if item is not None and claim.channel in item.channels and _overlaps(item.window, claim.window):
            valid += 1
    return valid / len(claim.evidence)


def check_diagnosis(
    diagnosis: Diagnosis | None,
    key: AnswerKey,
    evidence: EvidenceBundle,
    cfg: CheckConfig = CheckConfig(),
) -> CheckReport:
    if diagnosis is None:
        return CheckReport(parsed=False, reward=0.0)

    events = _events(key)
    recoverable = [is_recoverable(e, evidence) for _, e in events]
    claims = diagnosis.claims

    # One-to-one greedy matching by descending IoU.
    candidates = []
    for ci, claim in enumerate(claims):
        for ei, (_, ev) in enumerate(events):
            if ev.channel == claim.channel:
                iou = _iou(claim.window, ev.window)
                if iou >= cfg.iou_min:
                    candidates.append((iou, ci, ei))
    candidates.sort(key=lambda c: -c[0])
    claim_to_event: dict[int, tuple[int, float]] = {}
    used_events: set[int] = set()
    for iou, ci, ei in candidates:
        if ci in claim_to_event or ei in used_events:
            continue
        claim_to_event[ci] = (ei, iou)
        used_events.add(ei)

    verdicts: list[ClaimVerdict] = []
    for ci, claim in enumerate(claims):
        v = ClaimVerdict(claim_idx=ci, matched=None, grounding=_grounding(claim, evidence))
        if ci in claim_to_event:
            ei, iou = claim_to_event[ci]
            kind, ev = events[ei]
            v.matched = (kind, ei if kind == "root" else ei - len(key.roots))
            v.iou = iou
            v.category = "matched"
            if kind == "root":
                v.class_ok = claim.claim_class == ev.fault_class.value
                v.direction_ok = ev.direction is None or claim.direction == ev.direction
                v.magnitude_ok = ev.magnitude_z is None or (
                    claim.magnitude_z is not None
                    and abs(claim.magnitude_z - ev.magnitude_z) <= cfg.magnitude_rtol * ev.magnitude_z
                )
                v.role_ok = claim.role == "root"
                v.attribution_ok = True  # not applicable
            else:
                v.class_ok = claim.claim_class == "induced"
                v.direction_ok = True
                v.magnitude_ok = True
                v.role_ok = claim.role in (None, "induced")
                link = next((c for c in ev.causes if c.parent == claim.caused_by), None)
                if link is not None:
                    lag_err = abs(claim.lag - link.lag) if claim.lag is not None else cfg.lag_tol + 1
                    v.attribution_credit = 0.5 + 0.5 * max(0.0, 1.0 - lag_err / cfg.lag_tol)
                    v.attribution_ok = lag_err <= cfg.lag_tol
            v.partial_credit = (
                0.4 + 0.3 * v.iou + 0.2 * v.class_ok + 0.1 * v.magnitude_ok
            ) * v.grounding
            v.strict_correct = (
                v.class_ok
                and v.direction_ok
                and v.magnitude_ok
                and v.role_ok
                and v.attribution_ok
                and v.grounding == 1.0
                and bool(claim.evidence)
            )
        else:
            v.category = "grounded_but_wrong" if v.grounding > 0 else "hallucinated"
        verdicts.append(v)

    n_events = len(events)
    n_recoverable = sum(recoverable)
    matched_event_idxs = {claim_to_event[ci][0] for ci in claim_to_event}
    n_recoverable_matched = sum(1 for ei in matched_event_idxs if recoverable[ei])

    recall = n_recoverable_matched / n_recoverable if n_recoverable else 1.0
    ceiling = (n_events - n_recoverable) / n_events if n_events else 0.0
    precision = sum(v.partial_credit for v in verdicts) / len(verdicts) if verdicts else 1.0
    grounding_rate = sum(v.grounding for v in verdicts) / len(verdicts) if verdicts else 1.0

    n_rec_induced = sum(
        1 for (kind, _), rec in zip(events, recoverable) if kind == "induced" and rec
    )
    attribution = (
        sum(v.attribution_credit for v in verdicts if v.matched and v.matched[0] == "induced")
        / n_rec_induced
        if n_rec_induced
        else 1.0
    )

    n_unmatched = sum(1 for v in verdicts if v.matched is None)
    excess = max(0, len(claims) - n_events)
    reward = (
        cfg.w_format
        + cfg.w_recall * recall
        + cfg.w_precision * precision
        + cfg.w_attribution * attribution
        + cfg.w_grounding * grounding_rate
        - cfg.w_hallucination * n_unmatched
        - cfg.w_length * excess
    )

    return CheckReport(
        parsed=True,
        verdicts=verdicts,
        n_events=n_events,
        n_recoverable=n_recoverable,
        n_recoverable_matched=n_recoverable_matched,
        event_recall=recall,
        extractor_ceiling=ceiling,
        claim_precision=precision,
        attribution_score=attribution,
        grounding_rate=grounding_rate,
        n_hallucinated=sum(1 for v in verdicts if v.category == "hallucinated"),
        n_grounded_but_wrong=sum(1 for v in verdicts if v.category == "grounded_but_wrong"),
        strict_claim_accuracy=(
            sum(v.strict_correct for v in verdicts) / len(verdicts) if verdicts else 1.0
        ),
        reward=reward,
    )
