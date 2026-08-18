# MVTS-AD Diagnoser

Verifiable fault diagnosis with a fine-tuned LLM.

Anomaly detection on multivariate telemetry is a mostly solved problem. The expensive part comes after: fifteen channels light up and someone has to figure out whether that's fifteen problems or one problem, which channel started it, and what kind of fault it is. Today that someone is an engineer staring at trend charts.

LLMs could plausibly do this step. The uncomfortable truth is that nobody can prove one does it correctly, because on real data the true root cause is unknown. Existing benchmarks score models against human or LLM annotations, which are opinions.

This project takes a different route: I generate the telemetry and break it myself. Every fault is injected as an intervention on a randomized causal graph, so the full causal record (root, downstream effects, lags) is known by construction. The model must answer in structured claims, and a deterministic checker scores every claim against the injection record. No LLM judges anywhere. The same checker doubles as the training reward: SFT first, GRPO on top.

The interesting question: can a 4B model, fine-tuned locally for roughly zero euros, beat a prompted frontier model at grounded diagnosis? If yes, that's the result. If no, the measured difficulty frontier where current models fail is the result. The design makes sure one of the two comes out.

## Status

**In progress.** Design is frozen, code is being written. First results table targeted for week 4 (mid-September 2026).

- [x] Design document
- [ ] Scene generator + fault injection
- [ ] Evidence extractor
- [ ] Claim checker (eval harness = training reward)
- [ ] SFT (Qwen3-4B, MLX LoRA)
- [ ] Frontier baseline (Mistral Large, zero-shot + few-shot)
- [ ] Results + blog post
- [ ] Staged: GRPO, TEP/SWaT transfer, ablations

## Where things are

Everything worth reading right now is in [`docs/design.md`](docs/design.md): every design decision with its reasoning, the fault taxonomy, the reward function, the fairness protocol for baselines, and the risks I already know about. If you only read one section, read §1 (thesis) and §12 (the week-2 gate).

## Why I'm building this

I am obsessed with LLMs interacting with domain- and modality-specific models. The way I see it, the LLM is the human interface, with an interactive layer of domain-specific detection, forecasting, and planning models underneath. This project is an attempt at building exactly that architecture (domain specific detector below, general diagnoser on top), while getting hands-on with modern LLM fine-tuning: SFT and GRPO with verifiable rewards.

My background is multivariate time series and anomaly detection (MSc AI thesis on graph structure learning for anomaly detection in cyber-physical systems), which is where the detector layer and the healthy distrust of learned causal claims both come from.

## License

Apache-2.0. Everything: code, generated data, adapters.