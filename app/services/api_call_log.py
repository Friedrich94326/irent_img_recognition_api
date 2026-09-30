"""Record every API call in the iRent ops database (``api_call_logs``).

Like :mod:`app.services.vehicle_repository`, every call opens its own short-lived connection, so
the repository is safe to use from worker threads.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS "api_call_logs" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "api_name" TEXT NOT NULL,
  "http_method" TEXT NOT NULL,
  "return_code" INTEGER NOT NULL,
  "return_msg" TEXT NOT NULL,
  "client" TEXT NOT NULL,
  "client_ip" TEXT,
  "user_agent" TEXT,
  "duration_ms" REAL NOT NULL,
  "record_time" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS "api_call_logs_api_name_record_time_idx"
  ON "api_call_logs"("api_name", "record_time");
"""


class ApiCallLogRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    def ensure_table(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_DDL)
        finally:
            conn.close()

    def record(
        self,
        api_name: str,
        http_method: str,
        return_code: int,
        return_msg: str,
        client: str,
        client_ip: str | None,
        user_agent: str | None,
        duration_ms: float,
    ) -> int:
        conn = self._connect()
        try:
            with conn:
                return conn.execute(
                    "INSERT INTO api_call_logs (api_name, http_method, return_code, return_msg, "
                    "client, client_ip, user_agent, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        api_name,
                        http_method,
                        return_code,
                        return_msg,
                        client,
                        client_ip,
                        user_agent,
                        round(duration_ms, 3),
                    ),
                ).lastrowid
        finally:
            conn.close()


def build_api_call_log(settings: Settings) -> ApiCallLogRepository | None:
    path = settings.db_path
    if path is None or not path.exists():
        logger.info("No iRent ops database available - API calls are not logged.")
        return None
    repo = ApiCallLogRepository(path)
    repo.ensure_table()
    return repo
