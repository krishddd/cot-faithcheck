"""LLM client interface and the mock model."""

from __future__ import annotations

import pytest

from cot_faithcheck.answer import extract_answer
from cot_faithcheck.clients import (
    AnthropicClient,
    MockClient,
    OllamaClient,
    OpenAICompatibleClient,
    client_from_env,
)
from cot_faithcheck.errors import ClientConfigError
from cot_faithcheck.prompts import JUDGE_SENTINEL, build_continuation_messages
from cot_faithcheck.types import ReasoningStep


def _steps(*texts):
    return [ReasoningStep(index=i, text=t) for i, t in enumerate(texts)]


def test_mock_faithful_sums_reasoning():
    client = MockClient("faithful")
    msgs = build_continuation_messages("q", _steps("Take 10.", "Add 5."))
    out = client.generate(msgs, n=3)
    assert len(out) == 3
    assert all(extract_answer(t) == "15" for t in out)


def test_mock_faithful_reacts_to_deletion():
    client = MockClient("faithful")
    full = build_continuation_messages("q", _steps("Take 10.", "Add 5."))
    deleted = build_continuation_messages("q", _steps("Take 10."))
    assert extract_answer(client.generate(full)[0]) == "15"
    assert extract_answer(client.generate(deleted)[0]) == "10"


def test_mock_faithful_negation_flips_sign():
    client = MockClient("faithful")
    msgs = build_continuation_messages("q", _steps("It is NOT the case that: Add 8."))
    assert extract_answer(client.generate(msgs)[0]) == "-8"


def test_mock_unfaithful_ignores_reasoning():
    client = MockClient("unfaithful", fixed_answer="99")
    a = client.generate(build_continuation_messages("q", _steps("Take 10.")))[0]
    b = client.generate(build_continuation_messages("q", _steps("Take 9999.")))[0]
    assert extract_answer(a) == "99"
    assert extract_answer(b) == "99"


def test_mock_judge_request_returns_json():
    client = MockClient("unfaithful")
    msgs = [
        {"role": "system", "content": f"{JUDGE_SENTINEL} evaluate"},
        {"role": "user", "content": "trace"},
    ]
    reply = client.generate(msgs, n=1)[0]
    assert "is_faithful" in reply


def test_mock_noise_is_seeded_and_reproducible():
    a = MockClient("faithful", noise=0.5, seed=123)
    b = MockClient("faithful", noise=0.5, seed=123)
    msgs = build_continuation_messages("q", _steps("Take 1.", "Add 2."))
    assert a.generate(msgs, n=10) == b.generate(msgs, n=10)


def test_mock_custom_behavior():
    client = MockClient("custom", answer_fn=lambda messages: "7")
    assert extract_answer(client.generate([{"role": "user", "content": "x"}])[0]) == "7"


def test_mock_rejects_bad_behavior():
    with pytest.raises(ValueError):
        MockClient("nonsense")
    with pytest.raises(ValueError):
        MockClient("custom")  # missing answer_fn


def test_generate_rejects_bad_n():
    with pytest.raises(ValueError):
        MockClient("faithful").generate([{"role": "user", "content": "x"}], n=0)


def test_client_from_env_defaults_to_ollama(monkeypatch):
    for var in ("COT_FAITHCHECK_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    client = client_from_env()
    assert isinstance(client, OllamaClient)


def test_client_from_env_mock(monkeypatch):
    monkeypatch.setenv("COT_FAITHCHECK_PROVIDER", "mock")
    assert isinstance(client_from_env(), MockClient)


def test_client_from_env_unknown(monkeypatch):
    monkeypatch.setenv("COT_FAITHCHECK_PROVIDER", "does-not-exist")
    with pytest.raises(ClientConfigError):
        client_from_env()


def test_openai_client_construction():
    client = OpenAICompatibleClient("gpt-4o-mini", api_key="sk-test")
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"
    assert client._supports_native_n() is True


def test_anthropic_splits_system_message():
    client = AnthropicClient("claude-sonnet-5", api_key="x")
    system, convo = client._split_system(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert system == "be brief"
    assert convo == [{"role": "user", "content": "hi"}]
