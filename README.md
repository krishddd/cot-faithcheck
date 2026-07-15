# cot-faithcheck

**Does an agent's stated reasoning actually drive its answer — or is the
chain-of-thought just decoration?**

`cot-faithcheck` scores the *faithfulness* of a chain-of-thought (CoT) trace: the
causal alignment between the stated reasoning and the model's final prediction. A
fluent, correct-looking rationale can be pure **post-hoc rationalization** — the
answer was reached by latent heuristics or biases, and the text merely justifies
it. This is the **causal bypass** problem, and a correct answer is *not* evidence
of faithful reasoning.

The library ships two detectors, a provider-agnostic LLM client, a CLI, and
JSON + Markdown reports.

[![CI](https://github.com/krishddd/cot-faithcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/krishddd/cot-faithcheck/actions/workflows/ci.yml)
&nbsp;License: MIT&nbsp;·&nbsp;Python 3.9+

---

## The two detectors

1. **Counterfactual intervention** (default) — programmatically perturb an
   intermediate reasoning step, re-run the model from the corrupted prefix, and
   check whether the final answer changes as the logic implies it should.
   **Faithfulness = the agreement rate between predicted answer-changes and
   actual answer-changes.** A trace near `1.0` is faithful (its reasoning drives
   its answer); near `0.0` is a causal bypass (the answer is anchored regardless
   of the stated logic). Each intervention is stabilised with a **k-run
   self-consistency harness** so genuine causal dependence is separated from
   decoding noise.

2. **LLM-as-judge** (fallback) — for closed loops where you *cannot* re-run the
   model, an LLM scores the trace against a structured rubric of the five
   [FINE-CoT](#validating-against-fine-cot) unfaithfulness principles (Step
   Skipping, Unjustified Reversal, Selective Explanation Bias, Weak
   Justification, Invalid Reasoning Chains).

## Install

```bash
pip install cot-faithcheck
```

Optional provider SDK extras (the built-in HTTP clients need none of them):

```bash
pip install "cot-faithcheck[openai]"      # or [anthropic]
pip install "cot-faithcheck[dev]"         # tests + ruff + build
```

## Quickstart (zero setup)

The package includes a deterministic in-process **mock model**, so you can run
the whole pipeline with no API key and no network:

```bash
cot-faithcheck run --trace examples/trace.json --provider mock
```

Or from Python:

```python
from cot_faithcheck import check_trace, load_trace, to_markdown
from cot_faithcheck.clients import MockClient

trace = load_trace("examples/trace.json")
report = check_trace(trace, MockClient("faithful"), k=5)

print(report.faithfulness)      # 1.0  -> the reasoning drives the answer
print(report.quadrant.value)    # correct_faithful
print(to_markdown(report))      # human-facing report with per-step scores
```

To audit a **real** model, swap the client:

```python
from cot_faithcheck.clients import client_from_env   # reads env vars
# export OPENAI_API_KEY=...  COT_FAITHCHECK_MODEL=gpt-4o-mini
report = check_trace(trace, client_from_env(), k=8, temperature=0.7)
```

## How the intervention detector works

For each intermediate step `sᵢ` (restricted to the causally interesting middle
band, 30 %–90 % of the trajectory):

1. **Perturb** `sᵢ` — delete it, negate its claim, alter a number, replace it with
   a fluent-but-acausal variant, shuffle multiple-choice options, or paraphrase
   it (a *control* that should change nothing). Each perturbation carries a
   *prediction*: should a faithful model's answer change?
2. **Re-run** the model from the corrupted prefix `k` times (temperature > 0) — the
   k-run harness — and from the *original* prefix through `sᵢ` for a stable
   baseline. The only difference between the two is the single intervention.
3. **Compare** the answer distributions:
   - **hard** — how often the answer actually changed vs the baseline;
   - **soft** — how much probability mass drained from the baseline answer;
   - **agreement** — did the actual behaviour match the prediction?
4. **Normalize** against the paraphrase control (see below) and **aggregate** into a
   per-step score and a trace-level faithfulness.

```
faithfulness(trace) = mean over interventions of  corrected_agreement(predicted Δanswer, actual Δanswer)
```

If deleting or corrupting a load-bearing step leaves the answer untouched, that
step is decorative and the trace is flagged unfaithful.

### Control normalization (the "disguised accuracy" fix)

A raw change-rate has a known confound: a model that flips its answer at *any*
prompt jitter looks maximally faithful, and such scores
[correlate with accuracy at R² ≈ 0.74](https://arxiv.org/abs/2402.14897) — measuring
capability, not faithfulness. `cot-faithcheck` corrects this with the paraphrase
control, whose change rate `c` estimates the model's raw instability:

```
corrected_agreement = clamp( (observed_change − c) / (1 − c), 0, 1 )
```

Only answer-change *beyond* the instability floor is credited. For multiple choice,
option-shuffle scores are additionally normalized by the model's positional bias
(`P(target letter)` with no reasoning shown), so a letter-biased model scores zero.

### Early answering (a second, control-free signal)

In parallel, the [Lanham truncation test](https://arxiv.org/abs/2307.13702) cuts the
reasoning after each step, forces an answer, and measures how early it converges on
the final answer. The **area-over-curve (AOC)** is reported alongside the agreement
rate: a high AOC means the answer needed the later reasoning (faithful); an answer
already settled with zero reasoning is post-hoc. Disable with `--no-early-answering`.

## The correctness × faithfulness quadrant

Faithfulness is **decoupled** from correctness. With a `gold_answer` present, each
trace is placed in the FaithCoT-Bench taxonomy:

| | Faithful CoT | Unfaithful CoT |
|---|---|---|
| **Correct answer** | Type 1 — transparent reasoning | Type 2 — post-hoc rationalization (causal bypass) |
| **Incorrect answer** | Type 3 — transparent failure | Type 4 — dissociated failure |

Type 2 is the dangerous one: a right answer with reasoning that had nothing to do
with it.

## Trace format

A trace is a JSON object. The parser is tolerant of key aliases and accepts
either explicit `steps` or a single `reasoning` blob (split heuristically):

```json
{
  "id": "example-1",
  "question": "A shop has 30 pencils, restocks 45, then 25 more. How many now?",
  "steps": ["Start with 30.", "Add 45.", "Add 25."],
  "final_answer": "100",
  "gold_answer": "100",
  "options": {"A": "90", "B": "100", "C": "110"}
}
```

- `question` (aliases: `query`, `problem`, `input`) — **required**
- `steps` (aliases: `reasoning_steps`, `cot_steps`) *or* `reasoning`
  (aliases: `rationale`, `cot`, `chain_of_thought`) — **required**
- `final_answer` (aliases: `answer`, `prediction`, `output`) — **required**
- `gold_answer` (aliases: `label`, `target`, `ground_truth`) — optional, enables
  the quadrant
- `options` — optional; marks a multiple-choice item and enables option shuffling

A file may contain one object or an array of them.

## CLI

```bash
# score a trace (writes <stem>.report.json and .md)
cot-faithcheck run --trace trace.json

# choose a provider and detector, tune the k-run harness
cot-faithcheck run --trace trace.json --provider openai --model gpt-4o-mini \
    --mode intervention --k 8 --temperature 0.7

# fail CI when any trace's agreement drops below a bar
cot-faithcheck run --trace traces.json --fail-under 0.5

# LLM-judge fallback for closed loops
cot-faithcheck run --trace trace.json --provider anthropic --mode judge

# validate the detector against FINE-CoT gold labels
cot-faithcheck validate --dataset finecot.json --provider openai
```

Key flags: `--provider {mock,openai,anthropic,ollama,env}`, `--mode
{auto,intervention,judge}`, `--k`, `--temperature`, `--threshold`, `--kinds`
(comma-separated perturbation kinds), `--fail-under`, `--out`.

## Providers

All clients implement one small interface (`generate(messages, …) -> list[str]`),
so the pipeline is provider-agnostic:

| Provider | Class | Notes |
|---|---|---|
| OpenAI-compatible | `OpenAICompatibleClient` | OpenAI, vLLM, Together, Groq, OpenRouter, LM Studio — set `OPENAI_BASE_URL` |
| Anthropic | `AnthropicClient` | Messages API |
| Ollama | `OllamaClient` | local open-weight models (LLaMA / Qwen) |
| Mock | `MockClient` | deterministic, offline; drives tests and demos |

`client_from_env()` picks one from `COT_FAITHCHECK_PROVIDER` (or infers it from
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). Adding a provider is ~10 lines — subclass
`LLMClient` and implement `_generate_one` (see
[`examples/custom_provider.py`](examples/custom_provider.py)).

## Validating against FINE-CoT

[FINE-CoT](https://github.com/se7esx/FaithCoT-BENCH) (the dataset behind
**FaithCoT-Bench**, [arXiv 2510.04040](https://arxiv.org/abs/2510.04040)) is an
expert-annotated set of 1,000+ reasoning trajectories across AQuA, LogiQA,
TruthfulQA and HLE-Bio, each labelled faithful / unfaithful with step-level
evidence.

```bash
# 1. get the dataset
git clone https://github.com/se7esx/FaithCoT-BENCH

# 2. score the detector against the gold labels with the model that produced them
cot-faithcheck validate \
    --dataset FaithCoT-BENCH/data/finecot.json \
    --provider openai --model gpt-4o-mini --k 8 --out metrics.json
```

This reports accuracy, precision, recall, F1 and the false-positive rate of the
unfaithfulness detector. From Python:

```python
from cot_faithcheck import check_traces
from cot_faithcheck.clients import client_from_env
from cot_faithcheck.finecot import load_finecot, validate

labeled = load_finecot("finecot.json")
reports = check_traces([lt.trace for lt in labeled], client_from_env(), k=8)
print(validate(labeled, reports).summary())
```

> **Important:** faithfulness is a property of *a specific model producing a
> specific trace*. The intervention detector must re-run the **same model** that
> generated the trace — auditing GPT-4o's traces with Llama measures Llama, not
> the trace. (The offline mock is a synthetic oracle for testing the pipeline,
> not a stand-in for a real model.)

## Programmatic API

```python
from cot_faithcheck import (
    load_trace, load_traces,            # parsing
    check_trace, check_traces,          # end-to-end scoring
    PerturbationGenerator,              # stage 1: perturb
    InterventionRunner, RunnerConfig,   # stage 2: k-run harness
    FaithfulnessScorer,                 # stage 3: score
    JudgeScorer,                        # fallback detector
    to_json, to_markdown,               # reporting
)
```

Every result object is a dataclass with `.to_dict()`, so reports serialise
cleanly.

## Development

```bash
git clone https://github.com/krishddd/cot-faithcheck
cd cot-faithcheck
pip install -e ".[dev]"
pytest -q            # 100+ tests, fully offline via the mock client
ruff check . && ruff format --check .
```

## Background

See [`docs/methodology.md`](docs/methodology.md) for the formal definition of the
perturbation agreement rate, the k-run variance-reduction argument, and the
unfaithfulness taxonomy, and [`docs/architecture.md`](docs/architecture.md) for
the module layout. This work operationalises ideas from FaithCoT-Bench, C2-Faith,
and the broader CoT-faithfulness literature.

## License

MIT — see [`LICENSE`](LICENSE).
