# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

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
