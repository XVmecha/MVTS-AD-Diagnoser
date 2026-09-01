# fyi.md: MVTS-AD Diagnoser engineering log

Newest entries first. Design rationale lives in `docs/design.md`; this log records
implementation decisions, workarounds, and operational steps taken along the way.

## 2026-09-01 — Dataset assembly and gold rendering

**What.** `mvtsad/dataset.py`: 27-cell grid (SNR x depth x regime {none, switch,
switch_near}), fault class balanced within cells by rotation, exclusive wiring-id
ranges (sft 10k+, grpo 20k+ reserved, eval 30k+, dev 40k+), splits dev=202 /
sft=2498 (20% clean) / eval=1000. Scenes stored as JSONL (key + frozen evidence)
plus a manifest with recoverability stats; signals not stored (regenerable
deterministically; TOTO regenerates them later). `mvtsad/render.py`: compact
evidence serialization and template-rendered gold completions (rationale first,
claims JSON after; shuffled claim order, 3 field orderings, 3 rationale variants).
Build: `python -m mvtsad.dataset dev sft eval`. 7 tests (47 total).

**Why.** Last prerequisite before the week-2 gate.

**Alternatives.** Storing raw signals per split (~650 MB) was rejected; determinism
makes them free to regenerate and the JSONL stays tens of MB.

**Gotcha.** Two decisions not explicit in design.md. (1) GOLD CLAIMS COVER RECOVERABLE
EVENTS ONLY: claiming an event with zero evidence would train hallucination; the
gold clean list correspondingly includes channels whose faults left no evidence.
(2) The gold-vs-checker loop is closed by test: render_gold output parsed by
parse_diagnosis must score strict 1.0 / recall 1.0 / grounding 1.0. Any future
checker semantics change that breaks gold rendering fails tests immediately.

## 2026-08-31 — Extractor: four statistical detectors, empirically tuned defaults

**What.** `mvtsad/extract/extractor.py`: threshold crossings (sustained z > 3.5 or
single-step spike z > 3.5), change-points (windowed mean shift, half-window 25),
slopes (rolling OLS rise over 80 steps), and lagged cross-correlation shifts (coupling
baseline from the leading 15% of the scene; flags both coupling loss and new couplings
with the lag in the evidence id). Calibration location/scale are median/MAD of the
leading segment, mimicking limits set on historical normal data. 9 tests in
`tests/test_extractor.py` (40 total). ~200 ms/scene including generation.
TOTO deferred to an optional `mvtsad[toto]` extra per the 2026-08-31 decision;
the week-2 gate does not wait on it.

**Why.** Third critical-path item; the diagnoser reads only this output.

**Alternatives.** Two flood-control strategies were tried against measured flag volume
(initial defaults produced 58 items per CLEAN scene, which blows the prompt budget).
Full-scene robust calibration reduced volume but a level shift persisting to scene end
contaminates even median/MAD enough to hide itself (recoverability dropped to 0.67).
Adopted instead: leading-segment calibration (best recoverability) plus alarm
rationalization, i.e. deterministic per-channel/per-pair severity caps (threshold 2,
changepoint 3, slope 2, xcorr 2) and run merging with gap 20. Result: ~30 items per
clean scene, ~27 per faulted scene.

**Gotcha.** Three things to remember. (1) Root recoverability at the frozen defaults,
measured on 6-8 scenes per cell: 1.0 drift/level_shift/oscillation (high SNR), ~0.6-0.8
stuck_at/variance_change/correlation_break, 0.62 spike high SNR, ~0.1 spike low SNR.
Low-SNR spikes (1-2 noise-sigma single steps) are genuinely invisible; that is the
extractor ceiling by design, excluded from recall denominators. (2) Unit mismatch:
key magnitude_z is in NOISE-sigma units, evidence severity is in calibration-sigma
units (noise + wander + coupling), systematically smaller. Revisit at gold-rendering
time; magnitude claims may need the checker's rtol reconsidered. (3) Sensitivity
defaults are provisional; the sanctioned protocol is tuning on the dev split only,
never against eval scores.

## 2026-08-24 — Claim checker: one matching pass, reward and strict eval

**What.** `mvtsad/check/checker.py` (260 lines): `parse_diagnosis` (extracts the last
valid claims JSON from rationale-first output; parse failure = hard gate, reward 0),
one-to-one greedy claim-to-event matching (channel + window IoU >= 0.5), graded GRPO
reward and strict binary verdicts computed from the same match results, recoverable-only
recall with extractor ceiling reported separately, grounding validation of cited evidence
ids, grounded-but-wrong vs hallucinated split. 12 tests in `tests/test_checker.py`
including an integration test: a gold diagnosis rendered from a generated answer key
scores strict accuracy 1.0.

**Why.** The checker is the artifact everything else is judged against (eval harness,
GRPO reward, integrity monitor), and it depends only on `mvtsad/schemas.py`, so building
it before the extractor keeps the dependency order clean.

**Alternatives.** Matching could gate on fault class as well as channel + window.
Rejected: design section 8 lists class among the scored tolerances, not the matching
gates, and gating on class would zero out partial credit for
right-channel-right-window-wrong-class answers, exactly the contrast GRPO needs.

**Gotcha.** Three interpretation decisions not fully pinned by the design doc, flag for
review. (1) Strict correctness requires the claim's citations to be nonempty and all
valid (the "every field passes" reading applied to the evidence field); frontier
baselines will score lower under this than under a claims-only reading. (2) The reward's
hallucination penalty applies per UNMATCHED claim (section 9 formula), while the
hallucinated/grounded-but-wrong split (section 8) is reported but not separately
weighted. (3) The diagnosis's `clean` list is not scored in v1: clean-scene false alarms
are already captured by unmatched-claim penalties, and no design metric scores the list
itself. Attribution credit is 0.5 for the correct edge + 0.5 graded by lag error.

## 2026-08-24 — Scene generator, fault injection, twin-run answer keys

**What.** Implemented `mvtsad/schemas.py` (AnswerKey, EvidenceBundle, Diagnosis contracts),
`mvtsad/gen/wiring.py` (randomized sparse DAG sampler with 4 kernel families, 3 noise
families, two timescale groups, regime B, guaranteed feedback motif),
`mvtsad/gen/faults.py` (7 intervention classes), `mvtsad/gen/simulate.py` (structural
equation engine), `mvtsad/gen/scene.py` (difficulty cells, scene generation, induced-event
derivation). 19 tests in `tests/test_generator.py`, all passing. Benchmark: 18 ms/scene,
about 3 minutes for the full ~9k corpus.

**Why.** First item on the critical path to the week-2 gate. Answer keys must be true by
construction, which drove the central implementation choice below.

**Alternatives.** Induced events could be derived by static graph reachability from the
root. Rejected: reachability lies under regime switches, cut edges (correlation_break),
and attenuating chains. Instead the engine pre-draws all stochastic streams once and
simulates twice (with and without interventions); induced events are channels whose
trajectories diverge (|diff|/sigma > 0.75 sustained 3+ steps, gaps up to 10 closed). The
divergence can only flow through causal edges, so the key is exact under any interaction.
The test `test_single_root_and_twin_isolation` enforces the contrapositive: non-descendants
must be bit-identical between the twin runs, for every fault class.

**Gotcha.** Three sharp edges. (1) Determinism: wiring is a pure function of
(wiring_id, GEN_VERSION) via seed domain tag 101; scene randomness uses tag 202. Changing
GEN_VERSION regenerates everything; never reuse a version after publishing data.
(2) The design doc's example claims JSON uses the key "sensor", but design.md section 2
mandates "channel" throughout code and schema; schemas use "channel". (3) Even with a
single root, a deviated channel can have multiple deviated parents (multi-path DAG
propagation), so InducedEvent stores a `causes` list while the claims schema keeps scalar
`caused_by`; the checker must accept a claim matching any listed cause.

## 2026-08-24 — Ruling: single-root scenes only in v1

**What.** Andreas ruled that v1 scenes contain exactly one fault root (or none, for the
20% clean scenes). Amended `docs/design.md` sections 4.2 (question resolved), 4.3
(concurrent-roots dial removed from v1), 4.4 (3-root extrapolation cell suspended;
depth-3 chains are the candidate replacement, decided at the week-2 gate), and the
appendix.

**Why.** Concurrent roots multiply generator and checker complexity (overlapping cascades
break scalar caused_by) before the benchmark has proven it needs the difficulty. They are
held in reserve as the first escalation if the frontier baseline is not challenged by
single-root scenes.

## 2026-08-24 — Python package scaffold

**What.** Created `pyproject.toml` (package `mvtsad`, Python >= 3.11, deps numpy + pydantic,
hatchling build) and subpackages `mvtsad/gen/`, `mvtsad/extract/`, `mvtsad/check/`, plus
`.gitignore`. No functional code yet.

**Why.** Stack confirmed by Andreas (Python). Layout mirrors the architecture in
design.md §3 (simulator, extractor, checker) and the release plan in §16.

**Alternatives.** Two separate packages (`mvtsad-bench`, `mvtsad-check`) from day one,
per §16. Rejected for v1: one repo and one package is less friction; instead the split
is protected by an import rule stated in `mvtsad/check/__init__.py` (check must never
import gen or extract), so it can be extracted later without surgery.

**Gotcha.** `data/` and `adapters/` are gitignored: generated scenes and LoRA adapters
should ship as release artifacts, not commits.

## Standing operational notes

- Dev env: `.venv` (Python 3.13), created with `python3 -m venv .venv` and
  `.venv/bin/pip install -e '.[dev]'`. Run tests with `.venv/bin/pytest -q`.
- Import rule: `mvtsad.check` must never import `mvtsad.gen` or `mvtsad.extract`
  (planned split into standalone mvtsad-check). `mvtsad.schemas` is the neutral
  contract module everyone may import.
- Seed discipline: GEN_VERSION in `mvtsad/gen/scene.py` is baked into every seed.
  Bump it only deliberately; it invalidates all previously generated data.
