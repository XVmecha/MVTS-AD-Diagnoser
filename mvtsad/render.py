"""Prompt serialization and gold-completion rendering (design.md section 7).

Gold completions are programmatic transformations of the answer key: no
human, no teacher LLM, ever. Anti-shortcut measures: claim order shuffled,
claim field order varied across preset JSON-legal orderings, several
rationale template variants.

Gold claims cover RECOVERABLE events only (decision logged in fyi.md,
2026-09-01): a training target that claims an event with zero evidence
teaches hallucination; the evidence-conditioned truth is the correct
supervision, and unrecoverable events are accounted in the extractor
ceiling instead.
"""

from __future__ import annotations

import json

import numpy as np

from mvtsad.check.checker import is_recoverable
from mvtsad.schemas import AnswerKey, EvidenceBundle, EvidenceItem

# JSON-legal claim field orderings; the renderer rotates among them so field
# position never carries signal.
_ROOT_ORDERS = [
    ("channel", "window", "class", "direction", "magnitude_z", "role", "evidence"),
    ("role", "channel", "class", "window", "magnitude_z", "direction", "evidence"),
    ("channel", "class", "evidence", "window", "role", "direction", "magnitude_z"),
]
_INDUCED_ORDERS = [
    ("channel", "window", "class", "caused_by", "lag", "evidence"),
    ("channel", "class", "caused_by", "window", "evidence", "lag"),
    ("caused_by", "lag", "channel", "class", "window", "evidence"),
]

_OPENINGS = [
    "Working through the evidence:",
    "Reading the flag stream:",
    "Evidence assessment:",
]


def serialize_evidence(bundle: EvidenceBundle, n_timesteps: int) -> str:
    """Compact deterministic text the diagnoser reads. One line per flag."""
    lines = [
        f"System with {bundle.n_channels} channels (s0..s{bundle.n_channels - 1}), "
        f"{n_timesteps} timesteps. Evidence flags:"
    ]
    for it in sorted(bundle.items, key=lambda it: (it.window[0], it.id)):
        chans = "~".join(f"s{c}" for c in it.channels)
        parts = [it.id, it.source, chans, f"w=[{it.window[0]},{it.window[1]})"]
        if it.severity is not None:
            parts.append(f"sev={it.severity:.1f}")
        if it.direction is not None:
            parts.append(it.direction.value)
        lines.append("  ".join(parts))
    return "\n".join(lines)


def _pertinent(bundle: EvidenceBundle, channel: int, window: tuple[int, int]) -> list[EvidenceItem]:
    hits = [
        it
        for it in bundle.items
        if channel in it.channels and min(it.window[1], window[1]) > max(it.window[0], window[0])
    ]
    hits.sort(key=lambda it: (-(it.severity or 0.0), it.id))
    return hits


def _ordered(claim: dict, order: tuple[str, ...]) -> dict:
    return {k: claim[k] for k in order if k in claim}


def render_gold(key: AnswerKey, bundle: EvidenceBundle, seed: int) -> str:
    """Rationale-first gold completion: derivation text, then claims JSON."""
    rng = np.random.default_rng(seed)
    variant = int(rng.integers(3))

    lines: list[str] = []
    claims: list[dict] = []
    claimed: set[int] = set()

    if key.regime_switches:
        t = key.regime_switches[0]
        lines.append(
            f"A coordinated shift around t={t} is consistent with an operating-mode "
            "change, not a fault; flags explained by it are discounted."
        )

    for ev in key.roots:
        if not is_recoverable(ev, bundle):
            continue
        cites = [it.id for it in _pertinent(bundle, ev.channel, ev.window)[:2]]
        claim = {
            "channel": ev.channel,
            "window": list(ev.window),
            "class": ev.fault_class.value,
            "role": "root",
            "evidence": cites,
        }
        if ev.direction is not None:
            claim["direction"] = ev.direction.value
        if ev.magnitude_z is not None:
            claim["magnitude_z"] = round(ev.magnitude_z, 1)
        claims.append(_ordered(claim, _ROOT_ORDERS[variant]))
        claimed.add(ev.channel)
        lines.append(
            f"s{ev.channel}: {', '.join(cites)} fit a {ev.fault_class.value} pattern "
            f"over [{ev.window[0]},{ev.window[1]}); no upstream driver in evidence, so root."
        )

    for ev in key.induced:
        if not is_recoverable(ev, bundle):
            continue
        link = ev.causes[int(rng.integers(len(ev.causes)))] if ev.causes else None
        cites = [it.id for it in _pertinent(bundle, ev.channel, ev.window)[:2]]
        claim = {
            "channel": ev.channel,
            "window": list(ev.window),
            "class": "induced",
            "evidence": cites,
        }
        if link is not None:
            claim["caused_by"] = link.parent
            claim["lag"] = link.lag
            lines.append(
                f"s{ev.channel}: deviation ({', '.join(cites)}) follows s{link.parent} "
                f"at lag {link.lag}: induced effect, not a second fault."
            )
        claims.append(_ordered(claim, _INDUCED_ORDERS[variant]))
        claimed.add(ev.channel)

    if not claims:
        lines.append(
            "No sustained, propagating pattern: remaining flags are isolated or "
            "explained by normal variation. No fault claimed."
        )

    clean = sorted(set(range(key.n_channels)) - claimed)
    order = rng.permutation(len(claims))
    diagnosis = {"claims": [claims[i] for i in order], "clean": clean}
    return _OPENINGS[variant] + "\n" + "\n".join(lines) + "\n\n" + json.dumps(diagnosis)
