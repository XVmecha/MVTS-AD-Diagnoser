# MVTS-AD Diagnoser: Design Document

**Verifiable fault diagnosis with a fine-tuned LLM.**

Status: in progress. First results targeted for week 4. This document records every design decision and the reasoning behind it, so that a new contributor (human or agent) can understand not just *what* was decided but *why*. Decisions marked **[LOCKED]** are frozen for v1. Decisions marked **[STAGED]** are planned extensions that ship only after the v1 table exists. Decisions marked **[OPEN]** still need a ruling.

---

## 1. Thesis

When an anomaly detection system monitors an industrial asset, the expensive step is not detection. Detection ("something is wrong") is largely solved: threshold alarms, statistical process control, and learned detectors all raise flags within seconds. The expensive step is **diagnosis**: an engineer looking at fifteen flagged channels and determining that this is *one* fault, not fifteen. For example: a drift on channel 4, dragging channels 7 and 9 along through the plant's physical couplings, everything else noise. That step is manual, slow, and concentrated in the heads of senior engineers. Time-to-diagnosis, not time-to-detection, dominates downtime cost.

Large language models could plausibly do this step. The problem is that **nobody can currently prove one does it correctly.** Existing benchmarks for "LLM explains telemetry" score model output against ground truth that is itself inferred, either by human annotators reading real data (SenTSR-Bench) or by frontier LLMs generating labels (Time-RA). On real data, the true root cause is unknown; the annotation is an opinion. A model's causal claim ("channel 4 caused channel 7's deviation") cannot be objectively verified against an opinion.

This project inverts the problem. We **generate telemetry and break it ourselves**. Because every fault is injected by our own code, the complete causal record (which channel is the root, which deviations are downstream effects, through which couplings, at what lags) is known **by construction**, not by inference. The model must output structured, machine-checkable claims, and a deterministic Python checker scores every claim against the injection record. No LLM judges anywhere in the truth chain.

The same checker serves three roles: evaluation harness, training reward (for GRPO), and integrity monitor. That triple use is the core engineering idea of the project.

Verifiability rests on three properties, and deleting any one collapses it:

1. **Constructed ground truth.** The fault record is the generator's own log, definitionally true. (Real data kills this.)
2. **Structured claims.** The model answers in discrete typed fields, each individually comparable to the record. (Prose output kills this.)
3. **Deterministic grading.** The score is a pure function of claims vs. record; same input, same score, auditable line by line. (LLM-as-judge kills this.)

The applied goal: demonstrate that a small fine-tuned open-weight model (Qwen3-4B) can match or beat prompted frontier models (Mistral Large) at grounded diagnosis, at a fraction of the cost, deployable on-premise. This is the pattern FactoryBench demonstrated for industrial robot telemetry, where a fine-tuned 0.23B model beat GPT-4o with 400 domain samples. If instead the frontier model wins, the project's deliverable becomes the measured difficulty frontier where current models fail. Both outcomes are results; the design guarantees neither outcome is wasted (see §12, week-2 gate).

---

## 2. Terminology

We use the field's vocabulary deliberately, mapped to the classical FDD (fault detection and diagnosis) ladder:

| This project | Classical FDD term | Meaning |
|---|---|---|
| Detector / extractor | Fault **detection** | Something is wrong (produces alarms/evidence) |
| Diagnoser: channel + window claims | Fault **isolation** | Which variable, when |
| Diagnoser: class + magnitude claims | Fault **identification** | What kind, how big |
| Diagnoser: root/induced + caused_by claims | Root cause **attribution** | Causal structure among events |

The trained model is called the **diagnoser**, never the "explainer". In ML, "explainer" collides with XAI (explaining a model's internals, SHAP-style), which is precisely not this task. The task as a whole is *anomaly explanation with causal attribution*, per the emerging benchmark literature (Exathlon, AXIS).

Domain-neutral vocabulary throughout code and schema: **channel** (not sensor), **system** (not plant), **scene** (not machine run). Rationale: nothing in the method is specific to manufacturing. Faults are statistical objects, causality is lag structure. The industrial instantiation is a *skin* (config + naming), which keeps the checker and benchmark reusable in AIOps, energy, physiology, or any multivariate-telemetry domain. We *pitch* it as industrial (that is where the buyer is); we *build* it neutral.

---

## 3. Architecture

```
SIMULATOR ──raw signals──► EXTRACTOR ──evidence bundle──► DIAGNOSER ──claims──► CHECKER ──score
    │                      (detection layer:                (fine-tuned LLM)                ▲
    │                       alarms & symbolic facts)                                        │
    └────────────────── answer key (injection record) ─────────────────────────────────────┘
```

**Division of labor [LOCKED]:** numerics detect, code extracts, the model diagnoses, code verifies.

The diagnoser **never reads raw signal values.** It reads only the evidence bundle: symbolic facts produced by a deterministic extractor. This decision has four independent justifications:

1. **LLMs are poor numeric perceivers.** Tokenization mangles digit sequences; attention over long number tables is unreliable. Asking the LLM to be the instrument replays a known failure mode of the LLM-for-time-series literature.
2. **Specialized detectors already exist and win.** Anomaly detection on raw multivariate signal is a mature field (and the author's thesis domain). The LLM competing with GDN/TOTO at detection is a losing fight. The LLM doing what detectors cannot (grouping, classification, causal attribution, language) is the open lane.
3. **The extractor is a domain normalizer.** Both synthetic training scenes and real transfer datasets (TEP, SWaT) pass through the same extractor into the same evidence format. The diagnoser therefore never sees "chemistry" or "water treatment"; it sees threshold crossings and correlation shifts. This shrinks the sim-to-real gap to a distribution match over evidence patterns.
4. **Compatibility with deployed reality.** Real plants run tiered flagging systems (DCS limit alarms, SPC charts, PCA contribution plots, ML detectors). All of them reduce to the same tuple: *(source, channel(s), window, severity, direction)*. Our evidence bundle is that neutral flag format, so any real alarm system can emit into it. The integration pitch is "keep your alarms, my layer consumes their output", which is the only pitch OT culture accepts.

**Trust boundary [LOCKED]:** the diagnoser sits outside every control and safety loop. Its output is a draft diagnosis a human reviews, never an action trigger. The safety-instrumented layer of a real plant is deterministic and certified; nothing probabilistic belongs in it. This is stated here because it constrains what the output schema promises (see §7: no corrective actions, no physical root causes).

---

## 4. Data generation

### 4.1 The generative model [LOCKED]

Each **scene** is generated as follows:

- **Wiring:** a sparse random DAG over 8–16 channels. Nodes are channels; a directed edge s4→s7 means "s4's value causally influences s7's value after a lag." Each channel evolves as: own baseline dynamics + weighted, lagged contributions from parents + noise.
- **Propagation kernels** drawn per-edge from a family: {instant, lagged, low-pass smoothed, saturating}. Rationale: kernel *diversity* prevents the model from overfitting to a single generative signature (a form of shortcut learning); the families are chosen to span behaviors real physical couplings exhibit.
- **Noise** per channel from {Gaussian, heavy-tailed, AR(1)}. Noise is not cosmetic. It plays two roles: it creates coincidental correlations and spurious evidence (the raw material of false alarms), and its magnitude relative to fault size defines the SNR difficulty dial.
- **Two timescale groups** (fast channels, slow channels), mirroring real process units (fast pressure, slow temperature).
- **Two-regime switching:** the scene may contain an operating-mode change that alters the active graph (edges appear/vanish, gains change). Rationale: regime changes are the single most realistic false-positive trap. They look dramatic in evidence but are not faults, and real plant graphs are regime-dependent. A model that re-infers the graph per scene handles this; a model with a memorized graph does not.
- **A crude feedback motif:** at least one channel partially counteracts its parent (a two-line caricature of a control loop). Rationale: real propagation runs through closed-loop control that *fights* disturbances. This is the largest known structural gap between our open-loop generator and TEP, and the motif narrows it cheaply.
- Scene length 500–1000 timesteps.

**Statistical realism only [LOCKED]:** the generator aims to be *statistically* representative (sparse causal graphs, mixed timescales, mode changes, realistic noise), never *physically* faithful. Physical realism is unbounded work with no payoff here, because realism lives in the **transfer sets**: TEP is a real chemical-process simulator someone else spent years building. Simulator = statistics; transfer = realism. Every hour on simulator physics is an hour stolen from the checker.

### 4.2 Fault injection [LOCKED]

A fault is an **intervention** on the generative equations (Pearl's do-operation, made concrete). Root = where we intervened; induced = everything that deviates downstream *through the existing edges*. The induced effects are produced by propagation, never injected separately. This is exactly why the answer key is trustworthy.

Taxonomy, with what each intervention literally does:

| Class | Intervention | Real-world analogue | Evidence signature |
|---|---|---|---|
| Drift | add slowly growing offset | bearing wear, fouling | change-point + sustained slope |
| Spike | add large brief pulse | sensor glitch, shock | threshold crossing, no level shift after |
| Level shift | add constant offset from t0 | blockage, setpoint error | step change (TEP IDV-1..7 class) |
| Stuck-at | replace signal with constant | dead sensor, frozen valve | variance collapse; children decorrelate |
| Variance change | multiply noise term | loose connection, turbulence | dispersion change, no mean shift |
| Oscillation | add sinusoid | valve stiction, loop hunting | periodicity (added for TEP transfer; IDV-14 class) |
| Correlation break | cut/rewire an incoming edge | coupling failure | channels individually normal, joint behavior wrong |

Structural grouping worth keeping in mind: classes 1–3, 5, 6 *add or modulate energy*; stuck-at *replaces* the signal; correlation break *edits the graph itself*. Their evidence signatures differ fundamentally, which is what makes classification a real task. Every injection carries randomized parameters (onset, duration, magnitude relative to noise, ramp shape) so no class has a fixed fingerprint to memorize.

Non-fault event types, for completeness: **induced deviations** (labeled as effects in the key, with parent and lag) and **regime switches** (legal mode changes that must NOT be claimed as faults).

**[RESOLVED 2026-08-24]** v1 scenes contain **exactly one root** (or none, for clean scenes). Concurrent roots, and the question of whether they may mix fault classes, are staged: they enter as the first difficulty escalation at the week-2 gate (§12) if single-root difficulty fails to challenge the frontier baseline. This supersedes the earlier "≤2 roots in training" dial. Consequence: the claims schema keeps a scalar `caused_by`, and overlapping propagation cascades need no v1 handling.

### 4.3 Difficulty as a stratified parameter grid [LOCKED]

Difficulty dials in v1: propagation depth (≤2), SNR, regime-switch presence, and onset proximity of the fault to the regime switch (a fault starting near a legal mode change is the false-attribution trap available in single-root scenes). Number of simultaneous roots is fixed at 1 in v1 (ruling 2026-08-24, see §4.2); concurrent roots return as a gate-time escalation, at which point multi-event onset proximity (near-simultaneous onsets breaking "first onset = root" heuristics) returns with them.

**Stratified** generation means every grid cell is filled deliberately (e.g., ~50 eval scenes per cell), never left to random sampling. Two reasons: (1) random sampling concentrates scenes in the comfortable middle, so the model never trains on hard corners; (2) the week-2 frontier gate (§12) needs enough hard-cell scenes to locate where the baseline breaks and report the failure curve. The curve ("frontier accuracy: high on easy cells, degraded on hard cells") is itself a headline figure.

**Clean scenes [LOCKED]:** 20% of SFT scenes contain no injected fault (gold = empty claims + full clean list). This includes regime-switch-without-fault scenes. Rationale: a model trained 100% on faulty scenes learns "always claim something"; healthy-with-spurious-alarms is most of real plant life; and false-alarm rate on clean scenes is arguably the first metric an OT engineer checks. It gets a dedicated eval column.

### 4.4 Split discipline [LOCKED]

Three datasets sample the same generator: SFT (~2–3k scenes), GRPO prompts (~2–5k, staged), eval (~1k, stratified across the grid).

- **Wiring instances are exclusive per split.** Thousands of unique wirings, fresh per scene; wiring ID ranges assigned to exactly one dataset. No graph shape ever appears in two splits. Rationale: if a wiring occurs in both train and eval, eval becomes a memory test. The model "generalizes" to causal structure it has memorized, silently re-breaking the graph-in-context principle (§5). Analogue: split by patient, not by scan.
- **Wiring diversity is guaranteed within every split** (thousands of instances each, same distribution of graph statistics). Kinds shared, individuals disjoint: new problems of the same type, never the homework with numbers changed.
- **Difficulty is stratified within every split.** Difficulty is a balanced factor; topology is a held-out factor. Opposite treatments, both deliberate.
- **One extrapolation cell in eval [AMENDED 2026-08-24]:** the original cell (3 simultaneous roots vs. a training max of 2) is suspended by the single-root ruling (§4.2). Candidate replacement: depth-3 propagation chains (training max is depth 2), which keeps the "new machine / harder day" claim falsifiable with no multi-root machinery. Final choice is deferred to the week-2 gate, where difficulty is calibrated anyway.
- **TEP is sealed [LOCKED]:** run once, at the end, after the generator is frozen. No generator iteration against TEP scores (that would be test-set tuning by the back door). Knowing fault *classes* exist in chemistry (drifts, stiction) is legitimate domain knowledge; fitting generator *parameters* to TEP is leakage. SMD (and optionally SWaT, which the author knows from thesis work) serve as development-time transfer checks that we ARE allowed to look at.

### 4.5 Volume

Quantity is free (scenes generate in milliseconds); the binding constraint is grid coverage, not count. Reference points: TimeMaster warm-started SFT on ~1k samples; we budget 2–3k for SFT.

---

## 5. The central representational principle: graph in context, not in weights

**[LOCKED]** Per-scene randomized wiring exists to force one specific learning outcome. With a fixed topology across scenes, gradient descent buys the cheapest solution: memorize the graph, stop reading the correlation evidence. That produces a system-specific model, useless one machine over (and it is the documented failure mode of half the LLM-PHM literature, which fine-tunes on machine A and tests on machine A). With randomized topology, memorization is priced out: the only strategy that reduces loss is *inferring the graph from the evidence in front of you, every time*.

Consequently the **weights hold invariants** while the **context holds specifics**. Invariants: signal-statistics knowledge ("sustained slope + level shift = drift" is true on any machine), the causal-inference procedure ("onset precedence + lagged correlation implies direction"), claim discipline. Specifics: this scene's topology, channel names, later documentation. Fine-tuning is spent only on what is true everywhere.

Cheap behavioral test (staged): same fault, two scenes, opposite wiring; the caused_by claim must flip with the evidence. Perturbation probes like this (flip a lag sign, delete an xcorr item) test whether the causal layer is evidence-driven or memorized. They are this project's inheritance from the author's thesis, which showed that learned dependency graphs in GSL anomaly detectors are performance artifacts rather than reliable maps. Here, for once, the true map exists, because we drew it.

Honest boundaries of transfer, named in advance: fault classes outside the taxonomy (hence an "unknown" escape class in the schema) and timescale regimes outside the training distribution (hence randomized sampling rates and lags). What legitimately breaks on a new system is vocabulary coverage, not the procedure.

---

## 6. Extractor (detection layer)

**[LOCKED]** The extractor computes, deterministically: threshold crossings, change-points, slopes, lagged cross-correlation shifts, and per-channel anomaly scores from **one learned detector, the author's existing TOTO-based pipeline** (zero integration cost, z-score-native, and CV continuity). Output is an **event-stream** of flag tuples. Real alarm systems emit timestamped streams, not tidy summaries; matching that narrows sim-to-real by one more notch.

**Imperfection is mandatory [LOCKED]:** the extractor must produce false positives (spurious crossings on clean channels, coincidental correlations) and false negatives (weak events below sensitivity, leaving zero trace). Rationale: with a too-clean extractor, "diagnosis" degenerates into reformatting the evidence list. The reasoning task only exists if the evidence underdetermines the answer. A perfect extractor would make the headline hallucination metric a secretarial artifact.

Multi-detector ensembles are a config option, not a fork (the bundle is already multi-source: statistical extractors + one learned emitter). **[STAGED]** evidence-tier ablation: run eval with tier-1 alarms only (legacy plant) vs. full bundle. This directly answers a customer's "do I need fancy detectors first?"

Exact extractor sensitivity settings: **[OPEN]**, to be chosen during implementation with the constraint "representative of deployed alarm behavior."

---

## 7. Output schema

The diagnoser outputs a rationale followed by structured claims:

```json
{"claims": [
   {"sensor": 4, "window": [405, 600], "class": "drift", "direction": "up",
    "magnitude_z": 3.2, "role": "root", "evidence": ["thr:s4@412", "slope:s4"]},
   {"sensor": 7, "window": [425, 600], "class": "induced", "caused_by": 4,
    "lag": 15, "evidence": ["xcorr:s4~s7@+15"]}],
 "clean": [1, 2, 3, 5, 6, 8]}
```

Every field is machine-checkable. The `evidence` field requires each claim to cite items that exist in the bundle and pertain to the claimed channel and window; ungrounded claims are scorable as such (§8).

**Rationale-first, claims-after [LOCKED].** The training targets place a short evidence-citing derivation *before* the claims JSON. Reasoning: this is not post-hoc justification (which would add nothing). Autoregressively, the claim tokens are conditioned on the derivation already in context; the rationale is computation before the decision, chain-of-thought mechanics. Second argument (the deciding one): Qwen3-4B-Instruct was post-trained on reasoning-then-answer formats, so rationale-first gold steers crystallized behavior rather than fighting the model's trained prior. The rationale is **scaffolding, not deliverable**: the checker scores claims only; reported metrics are claim metrics only. Integrity requirement: the checker runs a **consistency check** (edges cited in the rationale vs. edges claimed in JSON) and the mismatch rate is reported in the repo. Unscored text that can contradict the scored answer is exactly the failure mode this project exists to fight, so we measure it even though we do not reward it. **[STAGED]** ablation: claims-only SFT vs. rationale SFT, same scenes. "Does CoT help verifiable diagnosis at 4B" is a citable finding either way.

**Deliberate scope exclusions [LOCKED]:** no physical root causes ("clogged filter") and no corrective actions. The synthetic world contains no filters; those claims are unverifiable, therefore unscored, therefore not requested from the model. Fault *class* is verifiable (we chose it at injection); physical *cause* is not. Restricting the model to claims the reward can verify is a governance feature, not a limitation. **[STAGED]** phase-2 extension: synthetic plant documentation authored from the same ground truth (the dependency graph rendered as a fake manual), which moves the boundary. Named-component attribution becomes verifiable because the doc-to-graph mapping is ours, and doc-hallucination becomes a new measurable failure mode.

**Gold completions are template-rendered [LOCKED]:** training targets are programmatic transformations of the answer key (answer key to perfect claims plus templated rationale), written by no human and no teacher LLM. This keeps the truth chain LLM-free and costs nothing. Anti-shortcut measures: claim order is **shuffled** in gold (the checker matches by content, so canonical root-first ordering would teach a positional shortcut), key order randomized where JSON-legal, several structural template variants. Slogan: the model should learn the keywords and what they mean, never the order.

---

## 8. Checker semantics

The checker parses claims (invalid JSON = score 0, hard gate) and matches claims to injected events by content: same channel, window IoU ≥ 0.5, then class/direction/magnitude within tolerances; causal edges matched with lag tolerance.

**Recall vs. recoverable events only [LOCKED].** The extractor is imperfect, so some injected events leave zero trace in the evidence; the diagnoser cannot report what it was never shown. Recall is therefore computed against **recoverable** events (≥1 evidence item overlapping the event's channel and window; formal definition ships in checker docs), and the **extractor ceiling** (fraction of injected events that produced no evidence) is reported separately. Two clean numbers instead of one contaminated one: diagnoser reasoning quality vs. detection-layer blindness. They have different fixes and different budget lines. Conflating them lets a weak detector make the fine-tune look stupid, or a sharp detector hide that the model cannot reason. (Convention borrowed from answerability in QA, SQuAD 2.0 tradition.)

**Grounded-but-wrong is not hallucinated [LOCKED].** Evidence-validity and claim-truth are independent axes (per the AIS attribution framework): a claim citing a real spurious alarm on a clean channel is *grounded but wrong* (over-reading); a claim citing nothing that exists is *hallucinated* (inventing). Reporting both separates two failure modes that require different mitigations.

---

## 9. Reward vs. eval: two functions, one codebase [LOCKED]

**Training reward (GRPO): graded composite.** GRPO samples G completions per prompt, scores each, and normalizes scores within the group (advantage = (r − mean)/std); the gradient raises the probability of above-average completions. The reward's only job is therefore to *rank the group*: granularity comes from contrast between samples, not from the scalar being descriptive. Binary scoring fails exactly when it cannot rank. On hard scenes early in training, all samples score 0, advantage collapses, the prompt teaches nothing (reward starvation). Graded scoring keeps intra-group variance alive: the almost-right sample outranks the hallucinating one.

Structure:

```
R = 0                              if JSON invalid          (hard gate)
R = w_f·format_ok
  + w_r·event_recall               (recoverable events matched)
  + w_p·claim_precision            (per-claim partial credit: channel 0.4,
                                    window-IoU 0.3, class 0.2, magnitude 0.1,
                                    scaled by grounding factor)
  + w_a·attribution_score          (causal edges, graded by lag tolerance)
  + w_g·grounding_rate             (cited evidence valid for that claim)
  − w_h·hallucination_count        (per unmatched claim)
  − w_len·claim_count_excess       (anti-spam)
```

**Published eval: strict binary claim accuracy** (a claim is correct only if every field passes; one miss, zero). You optimize on the ramp, you report on the cliff. Both functions are published in full; the paper/post states explicitly that the binary numbers are computed on the graded-trained model, so the metric never grades its own homework.

Known pathologies, pre-committed countermeasures. **Reward-term domination** (model farms one easy term, e.g. maxes grounding by making few timid claims): per-term reward curves monitored during training, weights rebalanced if a term saturates while recall stagnates; the monitoring dashboard ships in the repo. **Reward hacking at graded boundaries** (vague wide windows harvesting partial credit): anti-spam penalty, binary eval as the reported truth, and an AnomSeer-style vocabulary check (RL should shift outputs toward temporally grounded tokens). **Advantage collapse**: difficulty-stratified prompt sampling. Feed GRPO scenes where the current model scores mid-range (curriculum via the grid, which stratification gives us for free).

Method lineage, for grounding: GRPO from DeepSeekMath; rule-based verifiable rewards as default recipe from DeepSeek-R1; the RLVR name from Tulu 3; the SFT-then-GRPO-on-3–4B recipe for time series from TimeMaster (Qwen2.5-VL-3B, composite format+accuracy+insight reward, beat few-shot GPT-4o) and VeriTime (Qwen3-4B; G=4 generations optimal). Claim-level factuality scoring descends from FActScore/SAFE; the grounded-vs-correct split from AIS; event-wise evaluation caution from the point-adjust critique (Kim et al., AAAI 2022) and affiliation metrics (Huet et al., KDD 2022).

---

## 10. Models, training, compute [LOCKED]

- **Primary diagnoser: Qwen3-4B-Instruct.** Chosen for literature comparability: TimeMaster and VeriTime ran this recipe on Qwen 3–4B, so our numbers sit next to the papers we cite, and recipe bugs are distinguishable from base-model quirks. De-risking beats flag-flying for the primary arm.
- **SFT: MLX-LM LoRA, local** on a 48GB Apple-silicon machine. A 4B model in 4-bit leaves ample headroom; 2–3k scenes at ~1.5k tokens for 2–3 epochs is an overnight run. Overnight wall-clock costs zero work-hours.
- **GRPO [STAGED]: Kaggle free GPU quota + TRL GRPOTrainer + QLoRA.** RL needs G generations per prompt plus a reference model; MLX RL tooling is bleeding-edge, so this stage rents (free-tier) NVIDIA instead. The checker drops in as the reward function unchanged.
- **Budget:** approximately zero euros of training compute, at most 20 euros of API spend. The Claude subscription is a build accelerator (the 52-hour budget is mostly coding hours); it trains nothing.
- **[STAGED]** Ministral-8B SFT arm: same recipe, second base. The cross-base table ("same data, same recipe, two bases") is the enterprise model-customization story in one figure.

---

## 11. Baselines and fairness protocol [LOCKED]

- **Frontier baseline: Mistral Large** (via La Plateforme), on the *identical* evidence bundles and schema spec. Two rows: **zero-shot** (the off-the-shelf claim) and **few-shot** (2–3 worked examples in-prompt, which is what a competent engineer would actually do, and the harder baseline; reporting both is the honesty move).
- **Parse fairness:** one re-prompt on invalid JSON. The fine-tune was trained to the schema; the frontier model was not. Without this rule, the table measures formatting, not diagnosis. Parse-failure rate reported separately.
- **Prompt authorship:** the baseline prompt is written by a SOTA LLM given the full task description, schema, and one example bundle, but no access to eval scenes or checker weights (it optimizes for the task, not the test). The prompt is **frozen before any week-2 results are seen**, published verbatim, authorship disclosed. This removes both hand-tuning-the-competitor and sandbagging.
- Additional rows: base Qwen3-4B zero-shot (what fine-tuning added), SFT model, [STAGED] SFT+GRPO, [STAGED] Claude row, [STAGED] Ministral arm. The table is framed as a **cost-capability frontier**: accuracy columns next to euros per 1k diagnoses and on-prem deployability, so "similar cost open-source" is a row, not the boundary of the experiment.

---

## 12. The week-2 gate [LOCKED]

Before any training: run the frontier baseline across the stratified eval grid.

- **If Mistral Large saturates the benchmark (roughly 95%+), stop and turn the difficulty dials** (lower SNR, closer onsets, more concurrent events, more spurious evidence) until a failure curve appears. A benchmark everyone passes measures nothing, and a fine-tune would have no headroom. Frontier saturation is the project's largest single risk, and this gate converts it from a silent failure into a design iteration.
- **If the frontier struggles, the gap is the project's living space** and the curve (accuracy per difficulty cell) is a headline figure regardless of how the fine-tune performs.

Either branch produces a result. The gate is why difficulty was built as a parameter grid rather than a fixed setting.

---

## 13. Metrics

Headline: **hallucinated-claim rate** (unmatched claims per diagnosis). Supporting: event recall (vs. recoverable), strict binary claim accuracy, attribution accuracy (causal edges), **false-alarm rate on clean scenes** (the OT engineer's first question; includes regime-switch-without-fault scenes), grounding-validity rate, rationale-claims mismatch rate, extractor ceiling (reported context, not a model metric), extrapolation-cell row (3 roots), parse-failure rates. Event-wise conventions follow the affiliation/point-adjust literature to avoid inflated scores.

---

## 14. Transfer protocol [STAGED]

**Purpose of real datasets: transfer exam, not curriculum.** TEP cannot train this model: it provides fault ID + onset, not the propagation record our checker scores against; it is one fixed topology (training on it puts the graph in the weights); and it has no difficulty dial. Reconstructing claim-level keys for TEP by hand would reintroduce exactly the human-inferred ground truth this project eliminates.

**Restricted scoring on real data.** On real datasets we score only what external truth can adjudicate:

| Claim field | TEP / SWaT scoreable? | Against what |
|---|---|---|
| Isolation (root channel) | Yes | Attacked point / literature-established primary variable |
| Timing (window) | Yes | Labeled attack/fault intervals (IoU rule) |
| Class | Family-level | Fault nature from spec (step, drift, stiction as oscillation) |
| Attribution (caused_by, lags) | **No** | No propagation record exists; unverifiable on real data |
| False alarms on clean | Yes | SWaT's labeled normal-operation days |

The resulting table will therefore look asymmetric on purpose. On the synthetic benchmark, every claim field gets a score. On TEP and SWaT, only isolation, timing, and class get scores, and the attribution columns are marked "unverifiable on this data". That asymmetry is the project's argument made visible: real-world benchmarks simply do not contain the information needed to check a causal claim, and that is exactly why an injection-verified benchmark needs to exist.

We treat transfer as a result to be measured, not a property to be assumed. If the model transfers well, that is a finding. If it transfers poorly, the measured gap is also a finding, and most of the literature cannot even measure theirs. During development we sanity-check transfer on SMD and SWaT, which we are allowed to inspect. TEP stays sealed until the generator is frozen, and is run exactly once.

**[STAGED]** few-shot adaptation arm (brief fine-tune on a slice of real data, measure the gain): the customer-onboarding story in miniature.

---

## 15. Ship plan, budget, descope order

**Budget: ~52 working hours over 4 weeks** (about 13 h/week alongside employment and applications). Ship = public repo + results table + blog post. Rough allocation: simulator + checker 17h, SFT data + training 12h, baselines + eval 10h, writeup 8h, slack 5h.

**v1 (ships week 4):** generator, extractor, checker, SFT model, Mistral Large baseline (both rows), core metrics table, blog post.

**Staged extensions, in order:** (1) GRPO on Kaggle, (2) claims-only vs. rationale ablation, (3) TEP/SWaT/SMD transfer table, (4) Ministral-8B arm, (5) evidence-tier ablation, (6) Claude baseline row, (7) perturbation probes, (8) documentation-grounding phase 2.

A shipped SFT-only version with a great verifier is CV-worthy; an unshipped full version is worth nothing.

**Standing risks:** frontier saturation (gated, §12), shortcut learning (kernel diversity, shuffled gold, adversarial cells, perturbation probes), reward-term domination (monitoring dashboard), sim-to-real gap (reported, not assumed; feedback motif narrows it), 52-hour optimism (descope order pre-agreed above).

---

## 16. Release

Everything Apache-2.0 (code, data, adapters). Artifacts: `mvtsad-bench` (generator + datasets), `mvtsad-check` (checker library, the domain-independent artifact, reusable by anyone with injected ground truth in any TS domain), diagnoser adapters. Repo stays bare-bones by decision (no reproduce.sh); an eval-only `make table` from cached outputs may be added later. Blog post on GitHub Pages; canonical results table in the README.

---

## Appendix: open questions

1. ~~Mixed fault classes in multi-root scenes~~ Resolved 2026-08-24: v1 is single-root only; multi-root (mixed or not) is staged behind the week-2 gate. See §4.2.
2. Extractor sensitivity settings ("representative of deployed alarm behavior"). See §6.
3. GRPO hyperparameters beyond G=4 default (beta sweep), decided at stage time. See §9.