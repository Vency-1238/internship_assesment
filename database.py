"""SQLite persistence for query records.

The repository keeps the database access surface small and explicit. This makes
the storage layer easy to replace with Postgres or another backend later without
touching the HTTP layer.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from models import QueryRecord, StructuredQueryData


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueryRepository:
    def __init__(self, database_path: str) -> None:
        self._database_path = database_path
        self._initialized = False

    async def initialize(self) -> None:
        if not self._initialized:
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    async def close(self) -> None:
        return None

    async def create_query(self, query: str, structured_data: StructuredQueryData) -> QueryRecord:
        return await asyncio.to_thread(self._create_query_sync, query, structured_data)

    async def get_query(self, query_id: str) -> QueryRecord | None:
        return await asyncio.to_thread(self._get_query_sync, query_id)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_sync(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL;")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    structured_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _create_query_sync(self, query: str, structured_data: StructuredQueryData) -> QueryRecord:
        query_id = str(uuid4())
        created_at = _utcnow_iso()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO queries (id, query, structured_data, created_at) VALUES (?, ?, ?, ?)",
                (query_id, query, structured_data.model_dump_json(), created_at),
            )
            connection.commit()
        return QueryRecord(
            id=query_id,
            query=query,
            structured_data=structured_data,
            created_at=datetime.fromisoformat(created_at),
        )

    def _get_query_sync(self, query_id: str) -> QueryRecord | None:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT id, query, structured_data, created_at FROM queries WHERE id = ?",
                (query_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        structured_data = StructuredQueryData.model_validate_json(row["structured_data"])
        return QueryRecord(
            id=row["id"],
            query=row["query"],
            structured_data=structured_data,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
