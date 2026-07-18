"""cot-faithcheck: does an agent's stated reasoning actually drive its answer?

Two detectors score chain-of-thought *faithfulness* - the causal alignment
between stated reasoning and the model's prediction, as opposed to mere fluency
or correctness:

1. **Counterfactual intervention** (Detector 1): programmatically perturb an
   intermediate reasoning step, re-run the model from the corrupted prefix
   through a k-run self-consistency harness, and measure whether the final answer
   changes as the logic implies. Faithfulness = the agreement rate between
   predicted answer-changes and actual ones.
2. **LLM-as-judge** (Detector 2): a structured rubric over the five FINE-CoT
   unfaithfulness principles, used as a fallback for closed loops where the model
   cannot be re-run.

Quickstart
----------
    from cot_faithcheck import check_trace, load_trace
    from cot_faithcheck.clients import client_from_env

    report = check_trace(load_trace("trace.json"), client_from_env())
    print(report.faithfulness, report.summary)

See ``cot-faithcheck run --trace trace.json`` for the CLI.
"""

from __future__ import annotations

from .errors import (
    ClientConfigError,
    ClientError,
    CotFaithcheckError,
    NoReasoningStepsError,
    TraceParseError,
)
from .intervention import InterventionRunner, RunnerConfig
from .judge import JudgeScorer
from .parser import load_trace, load_traces, parse_trace, split_reasoning
from .perturb import PerturbationGenerator
from .pipeline import check_file, check_trace, check_traces
from .report import batch_json, batch_markdown, to_json, to_markdown
from .scorer import FaithfulnessScorer, score_step
from .types import (
    ConfidenceInterval,
    Detector,
    EarlyAnsweringResult,
    FaithfulnessReport,
    InterventionResult,
    Perturbation,
    PerturbationKind,
    Quadrant,
    ReasoningStep,
    StepScore,
    Trace,
)

__version__ = "0.4.0"

__all__ = [
    # high-level API
    "check_trace",
    "check_traces",
    "check_file",
    # parsing
    "load_trace",
    "load_traces",
    "parse_trace",
    "split_reasoning",
    # detectors / stages
    "PerturbationGenerator",
    "InterventionRunner",
    "RunnerConfig",
    "FaithfulnessScorer",
    "score_step",
    "JudgeScorer",
    # reporting
    "to_json",
    "to_markdown",
    "batch_json",
    "batch_markdown",
    # types
    "Trace",
    "ReasoningStep",
    "Perturbation",
    "PerturbationKind",
    "InterventionResult",
    "StepScore",
    "FaithfulnessReport",
    "EarlyAnsweringResult",
    "ConfidenceInterval",
    "Detector",
    "Quadrant",
    # errors
    "CotFaithcheckError",
    "TraceParseError",
    "NoReasoningStepsError",
    "ClientError",
    "ClientConfigError",
    "__version__",
]
