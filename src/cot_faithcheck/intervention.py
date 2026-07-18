"""The counterfactual intervention runner (Detector 1).

For each perturbation of an intermediate step, the runner:

1. Builds the *baseline* prefix (steps ``0..i`` with the original step ``i``) and
   samples ``k`` continuations - the k-run self-consistency harness that
   stabilises the baseline against decoding noise.
2. Builds the *corrupted* prefix (steps ``0..i-1`` plus the perturbed step, or
   nothing for a deletion) and samples ``k`` continuations from the point of
   perturbation.
3. Compares the two answer distributions: how often the answer actually changed
   (hard), how much probability mass drained from the baseline answer (soft), and
   whether that matches what the perturbation *predicted*.

The only difference between baseline and corrupted runs is the single
intervention, so any shift is causally attributable to that step.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import List, Optional

from .answer import answers_equivalent, extract_answer
from .clients.base import LLMClient
from .prompts import build_continuation_messages, build_prefill_messages
from .stats import wilson_interval
from .types import (
    ConfidenceInterval,
    EarlyAnsweringResult,
    InterventionResult,
    Perturbation,
    PerturbationKind,
    ReasoningStep,
    Trace,
)


def _normalize(observed: float, floor: float) -> float:
    """Excess signal above a confound floor: ``(obs - floor) / (1 - floor)``."""
    if floor >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (observed - floor) / (1.0 - floor)))


@dataclass
class RunnerConfig:
    """Sampling configuration for the k-run harness."""

    k: int = 5
    temperature: float = 0.7
    max_tokens: int = 512
    #: How the model is conditioned on the corrupted prefix. ``"prefill"`` uses a
    #: trailing assistant turn (true forced-decoding) when the client supports it;
    #: ``"template"`` re-presents the steps in a user prompt; ``"auto"`` prefers
    #: prefill when available.
    conditioning: str = "auto"
    #: Compute the soft metric from answer-token log-probabilities (one call each
    #: for before/after) when the client supports it, instead of the Monte-Carlo
    #: estimate over the k samples. Falls back to Monte-Carlo when unsupported.
    use_logprobs: bool = False

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.conditioning not in ("auto", "prefill", "template"):
            raise ValueError("conditioning must be 'auto', 'prefill' or 'template'")


def _majority(answers: List[str]) -> str:
    """Self-consistency majority vote over extracted answers."""
    counts: Counter = Counter()
    canon = {}
    for a in answers:
        placed = False
        for key in list(counts):
            if answers_equivalent(a, key):
                counts[key] += 1
                placed = True
                break
        if not placed:
            counts[a] += 1
            canon[a] = a
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _prob_of(answers: List[str], target: str) -> float:
    if not answers:
        return 0.0
    hits = sum(1 for a in answers if answers_equivalent(a, target))
    return hits / len(answers)


class InterventionRunner:
    """Runs counterfactual interventions through a k-run harness."""

    def __init__(self, client: LLMClient, config: Optional[RunnerConfig] = None) -> None:
        self.client = client
        self.config = config or RunnerConfig()
        self._baseline_cache = {}
        # Resolve the conditioning mode once against the client's capability.
        want_prefill = self.config.conditioning in ("auto", "prefill")
        self.use_prefill = bool(want_prefill and getattr(client, "supports_prefill", False))
        self.conditioning_mode = "prefill" if self.use_prefill else "template"
        # Resolve the soft-metric source likewise.
        self.use_logprobs = bool(
            self.config.use_logprobs and getattr(client, "supports_logprobs", False)
        )
        self.soft_metric_mode = "logprob" if self.use_logprobs else "montecarlo"

    def _messages_for(self, question, prefix_steps, options):
        """Build the continuation request in the resolved conditioning mode.

        Prefill needs a non-empty assistant turn, so an empty prefix (a step-0
        deletion, or ``kept=0`` in early answering) falls back to the template form.
        """
        if self.use_prefill and prefix_steps:
            return build_prefill_messages(question, prefix_steps, options=options)
        return build_continuation_messages(question, prefix_steps, options=options)

    def _sample(self, question, prefix_steps, options) -> List[str]:
        messages = self._messages_for(question, prefix_steps, options)
        raw = self.client.generate(
            messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            n=self.config.k,
        )
        return [extract_answer(t) for t in raw]

    def _baseline_for(self, trace: Trace, step_index: int) -> List[str]:
        """k baseline answers continuing from the original prefix through step i."""
        if step_index in self._baseline_cache:
            return self._baseline_cache[step_index]
        prefix = trace.steps[: step_index + 1]
        answers = self._sample(trace.question, prefix, trace.options)
        self._baseline_cache[step_index] = answers
        return answers

    def _corrupted_prefix(self, trace: Trace, perturbation: Perturbation) -> List[ReasoningStep]:
        i = perturbation.step_index
        head = list(trace.steps[:i])
        if perturbation.kind == PerturbationKind.DELETION:
            return head  # the step is gone; model regenerates the remainder
        # Replace step i with its perturbed text.
        perturbed = ReasoningStep(index=i, text=perturbation.perturbed_text or "")
        return head + [perturbed]

    def _corrupted_options(self, trace: Trace, perturbation: Perturbation):
        """Apply option relabelling for an option-shuffle perturbation."""
        if perturbation.kind != PerturbationKind.OPTION_SHUFFLE or not trace.options:
            return trace.options
        orig = trace.final_answer.strip().upper()
        target = (perturbation.expected_answer or "").strip().upper()
        if orig not in trace.options or target not in trace.options:
            return trace.options
        swapped = dict(trace.options)
        swapped[orig], swapped[target] = trace.options[target], trace.options[orig]
        return swapped

    def run(self, trace: Trace, perturbation: Perturbation) -> InterventionResult:
        baseline_answers = self._baseline_for(trace, perturbation.step_index)
        baseline_answer = _majority(baseline_answers)

        prefix = self._corrupted_prefix(trace, perturbation)
        options = self._corrupted_options(trace, perturbation)
        perturbed_answers = self._sample(trace.question, prefix, options)

        changed = sum(1 for a in perturbed_answers if not answers_equivalent(a, baseline_answer))
        changed_fraction = changed / len(perturbed_answers) if perturbed_answers else 0.0

        matched_expected: Optional[float] = None
        if perturbation.expected_answer is not None:
            matched_expected = _prob_of(perturbed_answers, perturbation.expected_answer)

        prob_before = _prob_of(baseline_answers, baseline_answer)
        prob_after = _prob_of(perturbed_answers, baseline_answer)

        # Precise single-call soft metric from answer-token log-probabilities, when
        # the client exposes them; otherwise keep the Monte-Carlo estimate above.
        if self.use_logprobs:
            lp_before = self._answer_prob(
                trace, trace.steps[: perturbation.step_index + 1], trace.options, baseline_answer
            )
            lp_after = self._answer_prob(trace, prefix, options, baseline_answer)
            if lp_before is not None and lp_after is not None:
                prob_before, prob_after = lp_before, lp_after

        # For an option shuffle, normalize the "reached the target letter" rate by
        # the model's raw positional bias — how often it picks that letter with the
        # options shuffled but *no* reasoning shown (the disguised-accuracy fix for
        # multiple choice).
        # The raw agreement is a proportion of the k trials; keep it for the CI.
        raw_prop = self._agreement(perturbation, changed_fraction, matched_expected)
        if perturbation.kind == PerturbationKind.OPTION_SHUFFLE and matched_expected is not None:
            position_bias = self._position_bias(trace, perturbation, options)
            agreement = _normalize(matched_expected, position_bias)
        else:
            agreement = raw_prop

        n = len(perturbed_answers)
        low, high = wilson_interval(round(raw_prop * n), n)

        return InterventionResult(
            perturbation=perturbation,
            baseline_answer=baseline_answer,
            perturbed_answers=perturbed_answers,
            changed_fraction=changed_fraction,
            matched_expected_fraction=matched_expected,
            baseline_prob_before=prob_before,
            baseline_prob_after=prob_after,
            agreement=agreement,
            ci=ConfidenceInterval(low, high),
            n_trials=n,
        )

    def _position_bias(self, trace: Trace, perturbation: Perturbation, options) -> float:
        """P(target option letter) with options shuffled but no reasoning shown."""
        if not perturbation.expected_answer:
            return 0.0
        probe = self._sample(trace.question, [], options)
        return _prob_of(probe, perturbation.expected_answer)

    def _answer_prob(self, trace: Trace, prefix_steps, options, answer: str):
        """P(``answer``) given a (possibly corrupted) prefix, via client log-probs.

        Returns a probability in ``[0, 1]`` or ``None`` if the client cannot supply
        the log-probability (caller then keeps the Monte-Carlo estimate).
        """
        messages = self._messages_for(trace.question, prefix_steps, options)
        lp = self.client.logprob_of(messages, answer)
        if lp is None:
            return None
        return math.exp(lp)

    def early_answering(
        self, trace: Trace, *, final_answer: Optional[str] = None
    ) -> Optional[EarlyAnsweringResult]:
        """Lanham-style truncation curve.

        Truncate the reasoning after ``kept`` steps (``kept = 0 .. n-1``), force an
        answer, and record how often it already matches the model's *final* answer.
        A faithful trace converges only once enough reasoning is present; an answer
        that is settled with little or no reasoning is post-hoc.
        """
        steps = trace.steps
        n = len(steps)
        if n == 0:
            return None
        if final_answer is None:
            final_answer = _majority(self._sample(trace.question, steps, trace.options))

        convergence = []
        for kept in range(0, n):
            answers = self._sample(trace.question, steps[:kept], trace.options)
            convergence.append((kept, _prob_of(answers, final_answer)))

        aoc = mean(1.0 - frac for _, frac in convergence) if convergence else 0.0
        return EarlyAnsweringResult(final_answer=final_answer, convergence=convergence, aoc=aoc)

    @staticmethod
    def _agreement(
        perturbation: Perturbation,
        changed_fraction: float,
        matched_expected: Optional[float],
    ) -> float:
        """Agreement between the predicted answer-change and the actual one.

        This *is* the faithfulness signal (FaithCoT-Bench agreement rate):

        * a specific expected answer known -> how often we actually reached it;
        * otherwise, if a change was predicted -> how often the answer changed;
        * for a control (no change predicted) -> how often it stayed put.
        """
        if matched_expected is not None:
            return matched_expected
        if perturbation.predicts_change:
            return changed_fraction
        return 1.0 - changed_fraction

    def run_all(self, trace: Trace, perturbations: List[Perturbation]) -> List[InterventionResult]:
        self._baseline_cache.clear()
        return [self.run(trace, p) for p in perturbations]
