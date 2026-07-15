# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-15

Methodology hardening based on the recent CoT-faithfulness literature.

### Added
- **Control-normalized agreement** — the paraphrase control is now used to
  normalize every change-based agreement against the model's raw instability
  (`(observed − control) / (1 − control)`), rather than being averaged into the
  score. This is the "disguised accuracy" correction: a model that flips its
  answer at any prompt jitter no longer scores as faithful. Per-step reports now
  show the control-change rate and the raw→corrected agreement.
- **Multiple-choice position-bias normalization** — option-shuffle agreement is
  discounted by `P(target letter)` measured with the options shuffled but no
  reasoning shown, so a purely letter-biased model scores zero.
- **Early-answering (Lanham truncation curve)** — a complementary, control-free
  signal: truncate the reasoning after each step, force an answer, and report the
  convergence curve plus its area-over-curve (`aoc`). Higher AOC ⇒ the answer
  relied on later reasoning. Exposed as `EarlyAnsweringResult`, rendered in the
  report, and toggled with `--no-early-answering` (off by default for `validate`).

### Changed
- Headline `faithfulness` now averages *corrected* agreement over predicts-change
  perturbations only (paraphrase controls are the normalizer, not a score). A pure
  causal-bypass model consequently scores `0.0` where the naive scheme reported a
  residual `~0.25`.

## [0.1.0] - 2026-07-15

Initial release.

### Added
- **Counterfactual intervention detector** — perturb an intermediate reasoning
  step (deletion, negation, numeric alteration, acausal replacement, option
  shuffle, and a paraphrase control), re-run the model from the corrupted prefix
  through a **k-run self-consistency harness**, and score faithfulness as the
  agreement rate between predicted and actual answer-changes (hard and soft).
- **LLM-as-judge detector** — a structured rubric over the five FINE-CoT
  unfaithfulness principles, used as a fallback for closed loops.
- **Provider-agnostic clients** — OpenAI-compatible, Anthropic, and Ollama over
  the standard library (no SDK required), plus a deterministic in-process mock,
  behind one small `LLMClient` interface with a `client_from_env()` factory.
- **Trace parser** accepting explicit steps or a single reasoning blob, with
  heuristic step splitting and multiple-choice support.
- **Reports** in JSON (full detail) and Markdown (per-step table + intervention
  breakdown), plus the correctness × faithfulness quadrant taxonomy.
- **CLI** — `cot-faithcheck run --trace trace.json` and
  `cot-faithcheck validate --dataset finecot.json`, with `--fail-under` for CI.
- **FINE-CoT** loader and validation metrics (accuracy / precision / recall / F1
  / FPR) for benchmarking the detector against expert labels.
- pytest suite with fixture traces, ruff config, and GitHub Actions CI.
