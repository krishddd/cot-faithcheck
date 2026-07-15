"""CLI behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from cot_faithcheck.cli import main


def test_cli_run_writes_reports(tmp_path, fixtures_dir):
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "--trace",
            str(fixtures_dir / "faithful_math.json"),
            "--provider",
            "mock",
            "--mock-behavior",
            "faithful",
            "--k",
            "3",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    assert payload["is_faithful"] is True
    assert Path(f"{out}.md").exists()


def test_cli_fail_under_returns_nonzero(tmp_path, fixtures_dir):
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "--trace",
            str(fixtures_dir / "faithful_math.json"),
            "--provider",
            "mock",
            "--mock-behavior",
            "unfaithful",
            "--k",
            "3",
            "--fail-under",
            "0.5",
            "--out",
            str(out),
        ]
    )
    assert code == 1


def test_cli_batch(tmp_path, fixtures_dir):
    out = tmp_path / "batch_out"
    code = main(
        [
            "run",
            "--trace",
            str(fixtures_dir / "batch.json"),
            "--provider",
            "mock",
            "--k",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    assert payload["n_traces"] == 2


def test_cli_judge_mode(tmp_path, fixtures_dir):
    out = tmp_path / "judge_out"
    code = main(
        [
            "run",
            "--trace",
            str(fixtures_dir / "faithful_math.json"),
            "--provider",
            "mock",
            "--mode",
            "judge",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    assert payload["detector"] == "judge"


def test_cli_no_early_answering(tmp_path, fixtures_dir):
    out = tmp_path / "no_ea"
    code = main(
        [
            "run",
            "--trace",
            str(fixtures_dir / "faithful_math.json"),
            "--provider",
            "mock",
            "--k",
            "2",
            "--no-early-answering",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    assert payload["early_answering"] is None


def test_cli_bad_kind_errors(tmp_path, fixtures_dir):
    code = main(
        [
            "run",
            "--trace",
            str(fixtures_dir / "faithful_math.json"),
            "--provider",
            "mock",
            "--kinds",
            "not-a-kind",
            "--out",
            str(tmp_path / "x"),
        ]
    )
    assert code == 2


def test_cli_validate(tmp_path, fixtures_dir):
    out = tmp_path / "metrics.json"
    code = main(
        [
            "validate",
            "--dataset",
            str(fixtures_dir / "finecot_sample.jsonl"),
            "--provider",
            "mock",
            "--k",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert metrics["n_labeled"] == 3
