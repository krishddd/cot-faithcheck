"""Core data types for cot-faithcheck.

The pipeline flows through these dataclasses:

    Trace            a question + ordered ReasoningSteps + a final answer
      -> Perturbation   a corruption of one step, with a *predicted* answer effect
        -> InterventionResult   the k re-runs of the corrupted prefix
          -> StepScore    the per-step agreement rate (faithfulness of that step)
            -> FaithfulnessReport   the aggregate verdict for the whole trace

Everything is a plain ``@dataclass`` so results are trivially serialisable to the
JSON report (see :mod:`cot_faithcheck.report`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PerturbationKind(str, Enum):
    """How a reasoning step is corrupted.

    The value is the human-readable label used in reports.
    """

    DELETION = "deletion"
    NEGATION = "negation"
    NUMERIC = "numeric"
    ACAUSAL = "acausal"
    OPTION_SHUFFLE = "option_shuffle"
    PARAPHRASE = "paraphrase"  # control: meaning-preserving, predicts NO change


class Detector(str, Enum):
    """Which faithfulness detector produced a result."""

    INTERVENTION = "intervention"
    JUDGE = "judge"


class Quadrant(str, Enum):
    """FaithCoT-Bench correctness x faithfulness taxonomy (decoupled axes)."""

    CORRECT_FAITHFUL = "correct_faithful"  # Type 1: ideal
    CORRECT_UNFAITHFUL = "correct_unfaithful"  # Type 2: post-hoc rationalisation
    INCORRECT_FAITHFUL = "incorrect_faithful"  # Type 3: transparent failure
    INCORRECT_UNFAITHFUL = "incorrect_unfaithful"  # Type 4: dissociated failure
    UNKNOWN = "unknown"  # gold answer not supplied


@dataclass
class ConfidenceInterval:
    """A two-sided confidence interval on a proportion (Wilson score)."""

    low: float
    high: float
    level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {"low": self.low, "high": self.high, "level": self.level}

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass
class ReasoningStep:
    """A single intermediate step of a chain-of-thought trace."""

    index: int
    text: str
    #: ``True`` when the step lies in the causally interesting middle band
    #: (30%-90% of the trajectory, per C2-Faith) and is therefore a candidate for
    #: intervention. Set by the parser / runner.
    is_intervenable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    """A parsed chain-of-thought trajectory.

    ``question`` + ``prompt`` are what a re-runnable model is conditioned on;
    ``steps`` are the intermediate rationale; ``final_answer`` is what the trace
    claims the model concluded. ``options`` (if present) marks a multiple-choice
    item and enables option-shuffling perturbations. ``gold_answer`` is optional
    and only used to place the trace in the correctness x faithfulness quadrant.
    """

    question: str
    steps: List[ReasoningStep]
    final_answer: str
    prompt: str = ""
    options: Optional[Dict[str, str]] = None  # e.g. {"A": "...", "B": "..."}
    gold_answer: Optional[str] = None
    trace_id: str = "trace"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def intervenable_steps(self) -> List[ReasoningStep]:
        return [s for s in self.steps if s.is_intervenable]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Perturbation:
    """A concrete corruption of one reasoning step.

    ``predicts_change`` encodes what the *logical structure* implies should happen
    to the final answer once this step is corrupted. ``expected_answer``, when
    known, is the specific answer the corruption should produce (e.g. the flipped
    option for an option shuffle, or a recomputed number). The scorer measures how
    often the model's actual behaviour matches this prediction — that agreement
    rate *is* the faithfulness signal.
    """

    step_index: int
    kind: PerturbationKind
    perturbed_text: Optional[str]  # None means the step is deleted entirely
    predicts_change: bool
    expected_answer: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class InterventionResult:
    """The outcome of re-running a corrupted prefix through the k-run harness."""

    perturbation: Perturbation
    baseline_answer: str
    #: The k answers sampled after the corruption.
    perturbed_answers: List[str]
    #: Fraction of runs whose answer differs from the baseline (hard metric).
    changed_fraction: float
    #: Fraction of runs matching ``expected_answer`` (only when it is known).
    matched_expected_fraction: Optional[float]
    #: Empirical P(baseline_answer) before vs after — the soft metric.
    baseline_prob_before: float
    baseline_prob_after: float
    #: Raw agreement between predicted and actual answer-change, in [0, 1].
    agreement: float
    #: Agreement after normalising against the paraphrase-control instability
    #: baseline (the "disguised accuracy" correction). ``None`` until the scorer
    #: fills it in; equals ``agreement`` when no control is available.
    corrected_agreement: Optional[float] = None
    #: Wilson 95% CI on the raw agreement proportion over the k trials.
    ci: Optional[ConfidenceInterval] = None
    #: Number of k-run trials this result is based on.
    n_trials: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["perturbation"] = self.perturbation.to_dict()
        if self.ci is not None:
            d["ci"] = self.ci.to_dict()
        return d

    @property
    def effective_agreement(self) -> float:
        """The corrected agreement when present, else the raw agreement."""
        return self.corrected_agreement if self.corrected_agreement is not None else self.agreement


@dataclass
class StepScore:
    """Per-step faithfulness, aggregated over that step's perturbations."""

    step_index: int
    step_text: str
    faithfulness: float  # mean corrected agreement across perturbations, in [0, 1]
    soft_faithfulness: float  # mean |P(baseline) drop| across perturbations
    interventions: List[InterventionResult] = field(default_factory=list)
    #: The step's answer-change rate under the meaning-preserving paraphrase
    #: control — a measure of raw model instability used to correct the score.
    control_change_rate: float = 0.0
    #: Whether ``faithfulness`` was control-corrected (a paraphrase was available).
    corrected: bool = False
    #: How load-bearing the step is: the corrected agreement of its deletion (or
    #: the max over its perturbations if deletion was not run). A low value means
    #: the step is peripheral — corrupting it does not move the answer.
    criticality: float = 0.0
    #: Whether the step is load-bearing (``criticality >= criticality_threshold``).
    #: Only critical steps drive the trace-level verdict, so a genuinely peripheral
    #: step is not mistaken for causal bypass.
    is_critical: bool = True
    #: Present only for the LLM-judge detector.
    judge_flags: List[str] = field(default_factory=list)
    judge_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["interventions"] = [i.to_dict() for i in self.interventions]
        return d


@dataclass
class EarlyAnsweringResult:
    """Lanham-style early-answering truncation analysis.

    ``convergence`` pairs each truncation point (keep the first ``kept`` steps,
    force an answer) with the fraction of k runs that already match the model's
    *final* answer. A faithful trace only converges on its answer late; a trace
    whose answer is settled after little or no reasoning is post-hoc.

    ``aoc`` (area over the curve) is the mean, over truncation points, of
    ``1 - convergence`` — higher means the answer stayed unsettled until more
    reasoning was supplied, i.e. the reasoning mattered (more faithful).
    """

    final_answer: str
    #: (kept_steps, fraction_matching_final) in increasing ``kept`` order.
    convergence: List[Any] = field(default_factory=list)
    aoc: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "convergence": [list(p) for p in self.convergence],
            "aoc": self.aoc,
        }


@dataclass
class FaithfulnessReport:
    """The top-level verdict for one trace."""

    trace_id: str
    detector: Detector
    #: Headline agreement rate in [0, 1]; 1.0 = fully faithful, 0.0 = causal bypass.
    faithfulness: float
    soft_faithfulness: float
    is_faithful: bool
    threshold: float
    quadrant: Quadrant
    answer_correct: Optional[bool]
    step_scores: List[StepScore] = field(default_factory=list)
    #: Names of the FINE-CoT unfaithfulness principles that fired (judge or
    #: heuristics derived from the intervention results).
    unfaithfulness_flags: List[str] = field(default_factory=list)
    #: Wilson 95% CI on the raw agreement rate over critical-step interventions.
    faithfulness_ci: Optional[ConfidenceInterval] = None
    #: How many steps were load-bearing vs peripheral (intervention detector).
    n_critical_steps: int = 0
    n_peripheral_steps: int = 0
    #: Complementary Lanham early-answering analysis (intervention detector only).
    early_answering: Optional[EarlyAnsweringResult] = None
    config: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["detector"] = self.detector.value
        d["quadrant"] = self.quadrant.value
        d["step_scores"] = [s.to_dict() for s in self.step_scores]
        if self.faithfulness_ci is not None:
            d["faithfulness_ci"] = self.faithfulness_ci.to_dict()
        if self.early_answering is not None:
            d["early_answering"] = self.early_answering.to_dict()
        return d

    def weakest_step(self) -> Optional[StepScore]:
        """The least faithful step — the prime suspect for causal bypass."""
        if not self.step_scores:
            return None
        return min(self.step_scores, key=lambda s: s.faithfulness)
