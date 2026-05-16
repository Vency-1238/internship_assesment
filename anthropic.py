"""Minimal Anthropic client compatible with the app's needs.

This local adapter keeps the project runnable in environments where installing
the official anthropic package is blocked by native build issues. It exposes the
small API surface used by the backend: `Anthropic(...).messages.create(...)`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class AnthropicError(RuntimeError):
    pass


@dataclass(slots=True)
class _TextBlock:
    type: str
    text: str


@dataclass(slots=True)
class _MessageResponse:
    content: list[_TextBlock]


class _MessagesAPI:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict[str, Any]],
        extra_headers: dict[str, str] | None = None,
    ) -> _MessageResponse:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                **(extra_headers or {}),
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401:
                raise AnthropicError(
                    "Anthropic authentication failed with 401 Unauthorized. Check that ANTHROPIC_API_KEY is a valid Claude API key, not a placeholder or revoked key."
                ) from exc
            raise AnthropicError(f"Anthropic API request failed: {exc.code} {error_body}") from exc
        except Exception as exc:  # pragma: no cover - network boundary
            raise AnthropicError(f"Anthropic API request failed: {exc}") from exc

        content_blocks = [
            _TextBlock(type=block.get("type", "text"), text=block.get("text", ""))
            for block in data.get("content", [])
            if isinstance(block, dict)
        ]
        return _MessageResponse(content=content_blocks)


class Anthropic:
    def __init__(self, api_key: str) -> None:
        self.messages = _MessagesAPI(api_key)
