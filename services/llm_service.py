"""Groq extraction service.

The service is intentionally strict: the model must return JSON only, and the
result is validated before the API persists it. This avoids quietly storing
unstructured model output.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import groq
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
class GroqExtractionService:
    client: groq.Groq
    model: str

    @classmethod
    def from_environment(cls) -> "GroqExtractionService":
        api_key = os.getenv("GROQ_API_KEY", os.getenv("ANTHROPIC_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        if api_key == "sk-your-key-here" or "your-key-here" in api_key:
            raise RuntimeError(
                "GROQ_API_KEY is still a placeholder. Replace it with a real Groq API key in .env."
            )
        model = os.getenv("GROQ_MODEL", os.getenv("ANTHROPIC_MODEL", "llama-3.1-8b-instant"))
        return cls(client=groq.Groq(api_key=api_key), model=model)

    async def extract_query_intent(self, query: str) -> StructuredQueryData:
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract the structured research intent from this query:\n"
                            f"{query}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # pragma: no cover - network/API boundary
            raise ExtractionServiceError(f"Groq API request failed: {exc}") from exc

        content = response.choices[0].message.content.strip() if response.choices else ""
        if not content:
            raise ExtractionServiceError("Groq returned an empty response")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ExtractionServiceError("Groq returned invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise ExtractionServiceError("Groq response must be a JSON object")

        try:
            return structured_data_from_json(parsed)
        except ValidationError as exc:
            raise ExtractionServiceError(f"Groq JSON did not match schema: {exc}") from exc
