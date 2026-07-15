"""OpenAI-compatible chat-completions client.

Works with the OpenAI API and any endpoint that mirrors ``/v1/chat/completions``
(vLLM, Together, Groq, Fireworks, LM Studio, OpenRouter, ...). Only the base URL,
API key, and model change. Uses the standard library over HTTP, so the ``openai``
SDK is not required — install the ``openai`` extra only if you prefer it.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..errors import ClientConfigError, ClientError
from ._http import post_json
from .base import GenerationConfig, LLMClient, Message


class OpenAICompatibleClient(LLMClient):
    """Chat-completions client for any OpenAI-shaped endpoint."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        organization: Optional[str] = None,
        native_n: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.organization = organization
        self._native_n = native_n

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _payload(self, messages: List[Message], config: GenerationConfig, n: int) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "n": n,
        }
        if config.stop:
            payload["stop"] = list(config.stop)
        return payload

    def _request(self, messages: List[Message], config: GenerationConfig, n: int) -> List[str]:
        resp = post_json(
            f"{self.base_url}/chat/completions",
            self._payload(messages, config, n),
            headers=self._headers(),
            timeout=self.timeout,
        )
        choices = resp.get("choices")
        if not choices:
            raise ClientError(f"no choices in response: {str(resp)[:300]}")
        out = []
        for ch in choices:
            content = (ch.get("message") or {}).get("content")
            out.append(content or "")
        return out

    def _supports_native_n(self) -> bool:
        return self._native_n

    def _generate_n(self, messages: List[Message], config: GenerationConfig, n: int) -> List[str]:
        return self._request(messages, config, n)

    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        return self._request(messages, config, 1)[0]

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        model = os.environ.get("COT_FAITHCHECK_MODEL") or os.environ.get("OPENAI_MODEL")
        if not model:
            raise ClientConfigError(
                "set COT_FAITHCHECK_MODEL (or OPENAI_MODEL) to an OpenAI-compatible model id"
            )
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(model, base_url=base_url)
