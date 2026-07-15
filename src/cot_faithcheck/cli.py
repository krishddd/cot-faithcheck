"""Command-line interface: ``cot-faithcheck run --trace trace.json``.

Commands
--------
    cot-faithcheck run --trace trace.json [--provider ...] [--mode ...]
    cot-faithcheck validate --dataset finecot.json   # score against FINE-CoT labels

``run`` accepts a single-trace or multi-trace JSON file and writes a JSON report
and a Markdown report. By default it uses the deterministic in-process mock model
(``--provider mock``) so the tool runs with zero setup; point ``--provider`` at
openai / anthropic / ollama (with the matching env vars) for a real evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .clients import (
    AnthropicClient,
    LLMClient,
    MockClient,
    OllamaClient,
    OpenAICompatibleClient,
    client_from_env,
)
from .errors import CotFaithcheckError
from .parser import load_traces
from .pipeline import check_traces
from .report import batch_json, batch_markdown, to_json, to_markdown
from .types import FaithfulnessReport, PerturbationKind


def _build_client(args: argparse.Namespace) -> LLMClient:
    provider = (args.provider or "").lower()
    if provider == "mock":
        return MockClient(args.mock_behavior, model="mock-1")
    if provider == "openai":
        return OpenAICompatibleClient(
            args.model or "gpt-4o-mini",
            base_url=args.base_url or "https://api.openai.com/v1",
        )
    if provider == "anthropic":
        return AnthropicClient(args.model or "claude-sonnet-5")
    if provider == "ollama":
        return OllamaClient(
            args.model or "llama3.1",
            base_url=args.base_url or "http://localhost:11434",
        )
    if provider in ("", "env"):
        return client_from_env()
    raise CotFaithcheckError(f"unknown provider {provider!r}")


def _parse_kinds(spec: Optional[str]) -> Optional[List[PerturbationKind]]:
    if not spec:
        return None
    kinds = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            kinds.append(PerturbationKind(token))
        except ValueError as exc:
            valid = ", ".join(k.value for k in PerturbationKind)
            raise CotFaithcheckError(
                f"unknown perturbation kind {token!r}; choose from: {valid}"
            ) from exc
    return kinds or None


def _print_console_summary(reports: List[FaithfulnessReport]) -> None:
    print("=" * 64)
    print("COT-FAITHCHECK REPORT")
    print("=" * 64)
    for r in reports:
        verdict = "FAITHFUL  " if r.is_faithful else "UNFAITHFUL"
        print(
            f"[{verdict}] {r.trace_id:<20} "
            f"agreement={r.faithfulness:.3f}  "
            f"detector={r.detector.value}  quadrant={r.quadrant.value}"
        )
        weakest = r.weakest_step()
        if weakest is not None and not r.is_faithful:
            print(
                f"             weakest step #{weakest.step_index} "
                f"(faithfulness {weakest.faithfulness:.2f})"
            )
        if r.unfaithfulness_flags:
            print(f"             flags: {', '.join(r.unfaithfulness_flags)}")
    if len(reports) > 1:
        mean_f = sum(r.faithfulness for r in reports) / len(reports)
        n_unfaithful = sum(1 for r in reports if not r.is_faithful)
        print("-" * 64)
        print(
            f"{len(reports)} traces | mean agreement {mean_f:.3f} | "
            f"{n_unfaithful} flagged unfaithful"
        )
    print("=" * 64)


def _cmd_run(args: argparse.Namespace) -> int:
    client = _build_client(args)
    traces = load_traces(args.trace)
    reports = check_traces(
        traces,
        client,
        mode=args.mode,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        threshold=args.threshold,
        kinds=_parse_kinds(args.kinds),
        seed=args.seed,
        early_answering=args.early_answering,
    )

    _print_console_summary(reports)

    single = len(reports) == 1
    json_text = to_json(reports[0]) if single else batch_json(reports)
    md_text = to_markdown(reports[0]) if single else batch_markdown(reports)

    stem = args.out or Path(args.trace).with_suffix("").name + ".report"
    json_path = Path(f"{stem}.json")
    md_path = Path(f"{stem}.md")
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    print(f"\nWrote {json_path} and {md_path}")

    if args.fail_under is not None:
        worst = min(r.faithfulness for r in reports)
        if worst < args.fail_under:
            print(
                f"\nFAIL: min agreement {worst:.3f} < --fail-under {args.fail_under:.3f}",
                file=sys.stderr,
            )
            return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .finecot import load_finecot, validate

    client = _build_client(args)
    labeled = load_finecot(args.dataset)
    if args.limit:
        labeled = labeled[: args.limit]
    reports = check_traces(
        [lt.trace for lt in labeled],
        client,
        mode=args.mode,
        k=args.k,
        temperature=args.temperature,
        threshold=args.threshold,
        seed=args.seed,
        # The binary verdict does not use early answering; skip it to save calls.
        early_answering=False,
    )
    metrics = validate(labeled, reports)
    print(metrics.summary())
    print(f"  TP={metrics.tp} FP={metrics.fp} TN={metrics.tn} FN={metrics.fn}")
    if args.out:
        import json as _json

        Path(args.out).write_text(_json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cot-faithcheck",
        description="Score whether an agent's stated reasoning actually drives its answer.",
    )
    parser.add_argument("--version", action="version", version=f"cot-faithcheck {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--provider",
            default="mock",
            help="mock | openai | anthropic | ollama | env (default: mock)",
        )
        p.add_argument("--model", default=None, help="model id for the chosen provider")
        p.add_argument("--base-url", default=None, help="override provider base URL")
        p.add_argument(
            "--mock-behavior",
            default="faithful",
            choices=["faithful", "unfaithful"],
            help="behaviour of the built-in mock model (default: faithful)",
        )
        p.add_argument("--mode", default="auto", choices=["auto", "intervention", "judge"])
        p.add_argument(
            "--no-early-answering",
            dest="early_answering",
            action="store_false",
            default=True,
            help="skip the Lanham early-answering truncation analysis",
        )
        p.add_argument("--k", type=int, default=5, help="k-run harness size (default: 5)")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--max-tokens", type=int, default=512)
        p.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="faithful iff agreement >= threshold (default: 0.5)",
        )
        p.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("run", help="score one or more traces from a JSON file")
    _add_common(run)
    run.add_argument("--trace", required=True, help="path to a trace JSON file")
    run.add_argument(
        "--kinds",
        default=None,
        help="comma-separated perturbation kinds "
        "(deletion,negation,numeric,acausal,option_shuffle,paraphrase)",
    )
    run.add_argument("--out", default=None, help="output report stem (default: <trace>.report)")
    run.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero if any trace's agreement is below this",
    )
    run.set_defaults(func=_cmd_run)

    val = sub.add_parser("validate", help="score the detector against FINE-CoT gold labels")
    _add_common(val)
    val.add_argument("--dataset", required=True, help="path to a FINE-CoT JSON/JSONL file")
    val.add_argument("--limit", type=int, default=None, help="evaluate only the first N records")
    val.add_argument("--out", default=None, help="write metrics JSON to this path")
    val.set_defaults(func=_cmd_validate)

    return parser


def _ensure_utf8_stdout() -> None:
    """Best-effort: let reports with bar glyphs/emoji print on legacy consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CotFaithcheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
