"""Minimal Groq client compatible with the app's needs.

This local adapter keeps the project runnable without adding another external
dependency while still calling the Groq OpenAI-compatible chat completions API.
It exposes the small surface used by the backend:
`Groq(...).chat.completions.create(...)`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class GroqError(RuntimeError):
    pass


@dataclass(slots=True)
class _Message:
    content: str


@dataclass(slots=True)
class _Choice:
    message: _Message


@dataclass(slots=True)
class _ChatCompletionResponse:
    choices: list[_Choice]


class _ChatCompletionsAPI:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 300,
        response_format: dict[str, str] | None = None,
    ) -> _ChatCompletionResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401:
                raise GroqError(
                    "Groq authentication failed with 401 Unauthorized. Check that GROQ_API_KEY is valid and active."
                ) from exc
            if exc.code == 403:
                raise GroqError(
                    "Groq returned 403 Forbidden. Check model permissions, spend limits, billing status, and whether the selected model is enabled for your account."
                ) from exc
            raise GroqError(f"Groq API request failed: {exc.code} {error_body}") from exc
        except Exception as exc:  # pragma: no cover - network boundary
            raise GroqError(f"Groq API request failed: {exc}") from exc

        choices: list[_Choice] = []
        for choice in data.get("choices", []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            choices.append(_Choice(message=_Message(content=message.get("content", ""))))
        return _ChatCompletionResponse(choices=choices)


class _ChatAPI:
    def __init__(self, api_key: str) -> None:
        self.completions = _ChatCompletionsAPI(api_key)


class Groq:
    def __init__(self, api_key: str) -> None:
        self.chat = _ChatAPI(api_key)
