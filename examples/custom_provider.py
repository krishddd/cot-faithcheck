"""Plug in any LLM provider by implementing the tiny LLMClient interface.

A client only has to turn chat messages into text completions. Here we wrap an
arbitrary callable (say, a company-internal inference endpoint) in a few lines.
"""

from __future__ import annotations

from typing import List

from cot_faithcheck import check_trace, load_trace
from cot_faithcheck.clients.base import GenerationConfig, LLMClient, Message


class MyProviderClient(LLMClient):
    """Adapter over an in-house ``generate(prompt) -> str`` function."""

    provider = "my-provider"

    def __init__(self, model: str = "house-model-1") -> None:
        self.model = model

    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        # Replace this with a real call to your endpoint.
        return my_house_model(prompt, temperature=config.temperature)


def my_house_model(prompt: str, temperature: float = 0.7) -> str:
    """Stand-in for a real inference endpoint (returns a canned answer)."""
    return "Reasoning continues...\nAnswer: 25"


if __name__ == "__main__":
    trace = load_trace(
        {
            "question": "Add 12, 8 and 5.",
            "steps": ["Start with 12.", "Add 8.", "Add 5."],
            "final_answer": "25",
            "gold_answer": "25",
        }
    )
    report = check_trace(trace, MyProviderClient(), k=3)
    print(report.summary)
