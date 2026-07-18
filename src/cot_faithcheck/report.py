"""Render a :class:`FaithfulnessReport` to JSON and Markdown.

The JSON form is the full, machine-readable record (every intervention and every
sampled answer). The Markdown form is a human-facing summary with a per-step
table, suitable for pasting into a PR or an eval dashboard.
"""

from __future__ import annotations

import json
from typing import List

from .types import Detector, FaithfulnessReport

_BAR_WIDTH = 10


def to_json(report: FaithfulnessReport, *, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)


def _bar(score: float) -> str:
    filled = int(round(max(0.0, min(1.0, score)) * _BAR_WIDTH))
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _truncate(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def to_markdown(report: FaithfulnessReport) -> str:
    lines: List[str] = []
    verdict = "✅ FAITHFUL" if report.is_faithful else "⚠️ UNFAITHFUL"
    lines.append(f"# CoT Faithfulness Report — `{report.trace_id}`")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}  ")
    ci_txt = ""
    if report.faithfulness_ci is not None:
        ci_txt = f" — 95% CI [{report.faithfulness_ci.low:.2f}, {report.faithfulness_ci.high:.2f}]"
    lines.append(
        f"**Faithfulness (agreement rate):** {report.faithfulness:.3f} "
        f"`{_bar(report.faithfulness)}` (threshold {report.threshold:.2f}){ci_txt}  "
    )
    if report.detector == Detector.INTERVENTION:
        soft_src = report.config.get("soft_metric", "montecarlo")
        lines.append(
            f"**Soft faithfulness (prob. mass shift):** {report.soft_faithfulness:.3f} "
            f"({soft_src})  "
        )
        cond = report.config.get("conditioning")
        if cond:
            lines.append(
                f"**Conditioning:** {cond} "
                + ("(true forced-decoding)" if cond == "prefill" else "(re-presented prefix)")
                + "  "
            )
        if report.n_critical_steps or report.n_peripheral_steps:
            lines.append(
                f"**Load-bearing steps:** {report.n_critical_steps} critical, "
                f"{report.n_peripheral_steps} peripheral (score is over critical steps)  "
            )
    if report.early_answering is not None and report.early_answering.convergence:
        lines.append(
            f"**Early-answering AOC:** {report.early_answering.aoc:.3f} "
            f"`{_bar(report.early_answering.aoc)}` (higher = answer relies on later reasoning)  "
        )
    lines.append(f"**Detector:** {report.detector.value}  ")
    lines.append(f"**Quadrant:** {report.quadrant.value}  ")
    if report.answer_correct is not None:
        lines.append(f"**Answer correct:** {report.answer_correct}  ")
    lines.append("")
    lines.append(f"> {report.summary}")
    lines.append("")

    if report.unfaithfulness_flags:
        lines.append(
            "**Unfaithfulness principles flagged:** "
            + ", ".join(f"`{f}`" for f in report.unfaithfulness_flags)
        )
        lines.append("")

    # Per-step table.
    if report.step_scores:
        lines.append("## Per-step faithfulness")
        lines.append("")
        if report.detector == Detector.INTERVENTION:
            lines.append(
                "| Step | Faithfulness | Soft | Control Δ | Load-bearing | Interventions | Reasoning |"
            )
            lines.append(
                "|-----:|:------------:|:----:|:---------:|:------------:|:-------------:|:----------|"
            )
            for s in report.step_scores:
                ctrl = f"{s.control_change_rate:.2f}" if s.corrected else "—"
                crit = (
                    f"yes ({s.criticality:.2f})"
                    if s.is_critical
                    else f"peripheral ({s.criticality:.2f})"
                )
                lines.append(
                    f"| {s.step_index} | {s.faithfulness:.2f} `{_bar(s.faithfulness)}` "
                    f"| {s.soft_faithfulness:.2f} | {ctrl} | {crit} | {len(s.interventions)} "
                    f"| {_truncate(s.step_text)} |"
                )
        else:
            lines.append("| Step | Faithfulness | Issue | Reasoning |")
            lines.append("|-----:|:------------:|:------|:----------|")
            for s in report.step_scores:
                lines.append(
                    f"| {s.step_index} | {s.faithfulness:.2f} `{_bar(s.faithfulness)}` "
                    f"| {_truncate(s.judge_rationale, 50)} | {_truncate(s.step_text)} |"
                )
        lines.append("")

    # Intervention detail.
    if report.detector == Detector.INTERVENTION and report.step_scores:
        lines.append("## Intervention detail")
        lines.append("")
        for s in report.step_scores:
            if not s.interventions:
                continue
            lines.append(f"### Step {s.step_index}: {_truncate(s.step_text, 100)}")
            lines.append("")
            lines.append(
                "| Perturbation | Predicts change | Changed | Agreement (raw→corrected) | 95% CI | Notes |"
            )
            lines.append(
                "|:-------------|:---------------:|:-------:|:-------------------------:|:------:|:------|"
            )
            for r in s.interventions:
                p = r.perturbation
                notes = []
                if r.matched_expected_fraction is not None:
                    notes.append(f"→expected {r.matched_expected_fraction:.2f}")
                notes.append(f"baseline={r.baseline_answer or '∅'}")
                if p.predicts_change:
                    agr = f"{r.agreement:.2f} → {r.effective_agreement:.2f}"
                else:
                    agr = "control"
                ci = f"[{r.ci.low:.2f}, {r.ci.high:.2f}]" if r.ci is not None else "—"
                lines.append(
                    f"| {p.kind.value} | {p.predicts_change} | {r.changed_fraction:.2f} "
                    f"| {agr} | {ci} | {'; '.join(notes)} |"
                )
            lines.append("")

    # Early-answering truncation curve.
    if report.early_answering is not None and report.early_answering.convergence:
        ea = report.early_answering
        lines.append("## Early answering (truncation curve)")
        lines.append("")
        lines.append(
            f"Convergence toward the final answer `{ea.final_answer or '∅'}` as reasoning is "
            f"revealed step by step. AOC = {ea.aoc:.3f} (higher ⇒ the answer needed the reasoning)."
        )
        lines.append("")
        lines.append("| Steps kept | P(matches final answer) | |")
        lines.append("|-----------:|:-----------------------:|:--|")
        for kept, frac in ea.convergence:
            lines.append(f"| {kept} | {frac:.2f} | `{_bar(frac)}` |")
        lines.append("")

    lines.append("---")
    cfg = ", ".join(f"{k}={v}" for k, v in report.config.items())
    lines.append(f"<sub>config: {cfg}</sub>")
    lines.append("")
    return "\n".join(lines)


def batch_markdown(reports: List[FaithfulnessReport]) -> str:
    """A combined Markdown report for a batch of traces, with a summary table."""
    if not reports:
        return "# CoT Faithfulness Report\n\n_No traces evaluated._\n"
    lines: List[str] = ["# CoT Faithfulness Report — batch", ""]
    mean_faith = sum(r.faithfulness for r in reports) / len(reports)
    n_unfaithful = sum(1 for r in reports if not r.is_faithful)
    lines.append(f"**Traces:** {len(reports)}  ")
    lines.append(f"**Mean agreement rate:** {mean_faith:.3f}  ")
    lines.append(f"**Flagged unfaithful:** {n_unfaithful}/{len(reports)}  ")
    lines.append("")
    lines.append("| Trace | Faithfulness | Verdict | Quadrant | Detector |")
    lines.append("|:------|:------------:|:-------:|:---------|:---------|")
    for r in reports:
        verdict = "faithful" if r.is_faithful else "**unfaithful**"
        lines.append(
            f"| `{r.trace_id}` | {r.faithfulness:.2f} `{_bar(r.faithfulness)}` "
            f"| {verdict} | {r.quadrant.value} | {r.detector.value} |"
        )
    lines.append("")
    for r in reports:
        lines.append(to_markdown(r))
        lines.append("")
    return "\n".join(lines)


def batch_json(reports: List[FaithfulnessReport], *, indent: int = 2) -> str:
    payload = {
        "n_traces": len(reports),
        "mean_faithfulness": (
            sum(r.faithfulness for r in reports) / len(reports) if reports else 0.0
        ),
        "n_unfaithful": sum(1 for r in reports if not r.is_faithful),
        "reports": [r.to_dict() for r in reports],
    }
    return json.dumps(payload, indent=indent, ensure_ascii=False)
