# Architecture

`cot-faithcheck` is a linear pipeline of small, independently testable stages.

```
trace.json
   │  parser.load_trace
   ▼
Trace ─────────────────────────────────────────────► (mode="judge")
   │  perturb.PerturbationGenerator                        │
   ▼                                                       │
[Perturbation, …]                                          │
   │  intervention.InterventionRunner  (k-run harness)     │
   ▼                                                       ▼
[InterventionResult, …]                            judge.JudgeScorer
   │  scorer.FaithfulnessScorer                            │
   ▼                                                       │
FaithfulnessReport ◄───────────────────────────────────────┘
   │  report.to_json / to_markdown
   ▼
report.json + report.md
```

## Modules

| Module | Responsibility |
|---|---|
| `types.py` | Dataclasses: `Trace`, `ReasoningStep`, `Perturbation`, `InterventionResult`, `StepScore`, `FaithfulnessReport`, and the `PerturbationKind` / `Detector` / `Quadrant` enums. |
| `errors.py` | Exception hierarchy rooted at `CotFaithcheckError`. |
| `parser.py` | Parse a trace dict / JSON / file into a `Trace`; split reasoning blobs into steps; mark the intervenable middle band. |
| `answer.py` | Extract a canonical answer token from free text and test answer equivalence (numeric, option-letter, canonical). |
| `prompts.py` | Shared prompt templates and region markers used by the runner, judge and mock. |
| `perturb.py` | `PerturbationGenerator` — build per-step perturbations (heuristic, or LLM-assisted for acausal/paraphrase). |
| `intervention.py` | `InterventionRunner` — the k-run self-consistency harness; builds corrupted prefixes, samples baselines and corrupted continuations, computes agreement. |
| `scorer.py` | `FaithfulnessScorer` — aggregate interventions into per-step and trace-level scores, quadrant, and unfaithfulness flags. |
| `judge.py` | `JudgeScorer` — the LLM-as-judge fallback with a structured rubric. |
| `pipeline.py` | `check_trace` / `check_traces` / `check_file` — wire the stages together with `mode` selection and judge fallback. |
| `report.py` | JSON and Markdown rendering (single and batch). |
| `finecot.py` | Load the FINE-CoT dataset and compute detection metrics vs. gold labels. |
| `cli.py` | `cot-faithcheck run` / `validate`. |
| `clients/` | Provider-agnostic `LLMClient` interface + OpenAI-compatible / Anthropic / Ollama / mock adapters and `client_from_env()`. |

## The client interface

The pipeline never imports a provider SDK. Every client implements:

```python
class LLMClient:
    def generate(self, messages, *, temperature, max_tokens, top_p, stop, n) -> list[str]: ...
```

`n` is the k-run sample count. Providers that support server-side `n` (OpenAI)
return all samples in one request; the rest (Anthropic, Ollama) fall back to `n`
sequential calls. This keeps the harness identical across providers.

## Design choices

- **Zero required dependencies.** The built-in clients speak HTTP via
  `urllib`; provider SDKs are optional extras. The whole library and test-suite
  run offline through `MockClient`.
- **Baseline caching.** All perturbations targeting the same step share one
  cached k-run baseline, so cost is roughly `k · (steps + perturbations)` calls,
  not `k · steps · perturbations`.
- **Everything is a dataclass** with `.to_dict()`, so the JSON report is a
  faithful, lossless record of every sampled answer — useful for offline
  re-analysis.
