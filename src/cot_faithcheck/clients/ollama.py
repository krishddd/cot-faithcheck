"""Ollama client for local open-weight models.

Talks to a local Ollama server's ``/api/chat`` endpoint. Great for offline runs
against LLaMA / Qwen open-weight models — the exact families FINE-CoT uses.
"""

from __future__ import annotations

import os
from typing import List

from ..errors import ClientError
from ._http import post_json
from .base import GenerationConfig, LLMClient, Message


class OllamaClient(LLMClient):
    """Client for a local (or remote) Ollama server."""

    provider = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            },
        }
        if config.stop:
            payload["options"]["stop"] = list(config.stop)
        resp = post_json(f"{self.base_url}/api/chat", payload, timeout=self.timeout)
        message = resp.get("message") or {}
        content = message.get("content")
        if content is None:
            raise ClientError(f"no message content in response: {str(resp)[:300]}")
        return content

    @classmethod
    def from_env(cls) -> "OllamaClient":
        model = os.environ.get("COT_FAITHCHECK_MODEL", "llama3.1")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return cls(model, base_url=base_url)
