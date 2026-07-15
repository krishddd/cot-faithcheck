"""JSON and Markdown report rendering."""

from __future__ import annotations

import json

from cot_faithcheck import check_trace, to_json, to_markdown
from cot_faithcheck.clients import MockClient
from cot_faithcheck.report import batch_json, batch_markdown


def test_json_report_roundtrips(math_trace, faithful_client):
    report = check_trace(math_trace, faithful_client, k=2, temperature=0.0)
    payload = json.loads(to_json(report))
    assert payload["trace_id"] == "aqua-faithful-1"
    assert payload["detector"] == "intervention"
    assert 0.0 <= payload["faithfulness"] <= 1.0
    assert len(payload["step_scores"]) == 3
    # Nested intervention detail survives serialisation.
    assert payload["step_scores"][0]["interventions"][0]["perturbation"]["kind"]


def test_markdown_contains_key_sections(math_trace):
    report = check_trace(
        math_trace, MockClient("unfaithful", fixed_answer="25"), k=2, temperature=0.0
    )
    md = to_markdown(report)
    assert "# CoT Faithfulness Report" in md
    assert "Per-step faithfulness" in md
    assert "Intervention detail" in md
    assert "UNFAITHFUL" in md


def test_batch_reports(fixtures_dir, faithful_client):
    from cot_faithcheck import check_file

    reports = check_file(str(fixtures_dir / "batch.json"), faithful_client, k=2, temperature=0.0)
    md = batch_markdown(reports)
    assert "batch — batch" in md.lower() or "batch" in md
    payload = json.loads(batch_json(reports))
    assert payload["n_traces"] == 2
    assert "mean_faithfulness" in payload


def test_batch_markdown_empty():
    assert "No traces" in batch_markdown([])


def test_judge_markdown_omits_intervention_detail(math_trace):
    report = check_trace(math_trace, MockClient("faithful"), mode="judge")
    md = to_markdown(report)
    assert "Intervention detail" not in md
    assert "Per-step faithfulness" in md
