"""Pydantic models used by the API boundary and persistence layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StructuredQueryData(BaseModel):
    industry: str | None = None
    entity_type: str | None = None
    region: str | None = None
    keywords: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class QueryCreateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class QueryRecord(BaseModel):
    id: str
    query: str
    structured_data: StructuredQueryData
    created_at: datetime

    model_config = ConfigDict(extra="forbid")


class QueryResponse(QueryRecord):
    @classmethod
    def from_record(cls, record: QueryRecord) -> "QueryResponse":
        return cls.model_validate(record.model_dump())


class APIErrorResponse(BaseModel):
    detail: str

    model_config = ConfigDict(extra="forbid")


def structured_data_from_json(data: dict[str, Any]) -> StructuredQueryData:
    return StructuredQueryData.model_validate(data)
