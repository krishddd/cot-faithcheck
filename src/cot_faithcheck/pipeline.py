"""High-level orchestration: trace in, :class:`FaithfulnessReport` out.

    from cot_faithcheck import check_trace, load_trace
    from cot_faithcheck.clients import MockClient

    report = check_trace(load_trace("trace.json"), MockClient("faithful"))
    print(report.faithfulness, report.summary)

``check_trace`` wires the four stages together - parse (done by the caller),
perturb, intervene (k-run harness), score - and transparently falls back to the
LLM judge when the trace has no intervenable steps (``mode="auto"``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .answer import extract_answer
from .clients.base import LLMClient
from .intervention import InterventionRunner, RunnerConfig, _majority
from .judge import JudgeScorer
from .parser import load_traces
from .perturb import PerturbationGenerator
from .prompts import build_continuation_messages
from .scorer import FaithfulnessScorer
from .types import FaithfulnessReport, InterventionResult, PerturbationKind, Trace


def _full_baseline_answer(runner: InterventionRunner, trace: Trace) -> str:
    """The model's stable answer when conditioned on the *complete* reasoning."""
    messages = build_continuation_messages(trace.question, trace.steps, options=trace.options)
    raw = runner.client.generate(
        messages,
        temperature=runner.config.temperature,
        max_tokens=runner.config.max_tokens,
        n=runner.config.k,
    )
    return _majority([extract_answer(t) for t in raw])


def check_trace(
    trace: Trace,
    client: LLMClient,
    *,
    mode: str = "auto",
    k: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 512,
    threshold: float = 0.5,
    kinds: Optional[List[PerturbationKind]] = None,
    seed: int = 0,
) -> FaithfulnessReport:
    """Score the faithfulness of one trace.

    Parameters
    ----------
    trace:
        A parsed :class:`~cot_faithcheck.types.Trace`.
    client:
        Any :class:`~cot_faithcheck.clients.base.LLMClient`.
    mode:
        ``"intervention"`` (Detector 1), ``"judge"`` (Detector 2), or ``"auto"``
        (intervention, falling back to judge when no step is intervenable).
    k:
        k-run harness size for variance reduction.
    """
    if mode not in ("auto", "intervention", "judge"):
        raise ValueError("mode must be 'auto', 'intervention' or 'judge'")

    base_config = {
        "mode": mode,
        "provider": getattr(client, "provider", "?"),
        "model": getattr(client, "model", "?"),
        "k": k,
        "temperature": temperature,
        "threshold": threshold,
    }

    if mode == "judge":
        return JudgeScorer(client, threshold=threshold).score(trace, config=base_config)

    gen = PerturbationGenerator(kinds, client=client, seed=seed)
    perturbations = gen.for_trace(trace)

    if not perturbations:
        if mode == "auto":
            cfg = dict(base_config, mode="judge", fallback="no intervenable steps")
            return JudgeScorer(client, threshold=threshold).score(trace, config=cfg)
        # mode == "intervention" with nothing to perturb -> empty but valid report.
        return FaithfulnessScorer(threshold).score(trace, {}, config=base_config)

    runner = InterventionRunner(
        client, RunnerConfig(k=k, temperature=temperature, max_tokens=max_tokens)
    )
    results = runner.run_all(trace, perturbations)

    by_step: Dict[int, List[InterventionResult]] = defaultdict(list)
    for r in results:
        by_step[r.perturbation.step_index].append(r)

    baseline_answer = _full_baseline_answer(runner, trace) if trace.gold_answer else None

    base_config["kinds"] = [k_.value for k_ in gen.kinds]
    return FaithfulnessScorer(threshold).score(
        trace, dict(by_step), baseline_answer=baseline_answer, config=base_config
    )


def check_traces(
    traces: List[Trace],
    client: LLMClient,
    **kwargs,
) -> List[FaithfulnessReport]:
    """Score a batch of traces with the same settings."""
    return [check_trace(t, client, **kwargs) for t in traces]


def check_file(path: str, client: LLMClient, **kwargs) -> List[FaithfulnessReport]:
    """Load traces from a JSON file (single object or array) and score them all."""
    return check_traces(load_traces(path), client, **kwargs)
