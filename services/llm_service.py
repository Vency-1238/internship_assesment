"""Claude extraction service.

The service is intentionally strict: Claude must return JSON only, and the
result is validated before the API persists it. This avoids quietly storing
unstructured model output.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import anthropic
from pydantic import ValidationError

from models import StructuredQueryData, structured_data_from_json


class ExtractionServiceError(RuntimeError):
    pass


SYSTEM_PROMPT = """You are an information extraction engine for a research platform.
Return strict JSON only. Do not include markdown, commentary, or code fences.

Extract a compact research intent from the user's query using this schema:
{
  "industry": string | null,
  "entity_type": string | null,
  "region": string | null,
  "keywords": list[string]
}

Rules:
- industry should be the primary domain if present, otherwise null.
- entity_type should identify the target entity class if present, otherwise null.
- region should be a geographic region if present, otherwise null.
- keywords should contain 3 to 8 short normalized search keywords.
- Never invent facts that are not supported by the query.
- If the query is ambiguous, prefer null fields over guessing.
- Output must be valid JSON and nothing else."""


@dataclass(slots=True)
class ClaudeExtractionService:
    client: anthropic.Anthropic
    model: str

    @classmethod
    def from_environment(cls) -> "ClaudeExtractionService":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return cls(client=anthropic.Anthropic(api_key=api_key), model=model)

    async def extract_query_intent(self, query: str) -> StructuredQueryData:
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=300,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract the structured research intent from this query:\n"
                            f"{query}"
                        ),
                    }
                ],
            )
        except Exception as exc:  # pragma: no cover - network/API boundary
            raise ExtractionServiceError(f"Claude API request failed: {exc}") from exc

        content = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        if not content:
            raise ExtractionServiceError("Claude returned an empty response")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ExtractionServiceError("Claude returned invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise ExtractionServiceError("Claude response must be a JSON object")

        try:
            return structured_data_from_json(parsed)
        except ValidationError as exc:
            raise ExtractionServiceError(f"Claude JSON did not match schema: {exc}") from exc
