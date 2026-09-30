"""Every ``/api/v1`` request writes one row to ``api_call_logs`` (on a throwaway SQLite file)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.api_call_log import ApiCallLogRepository

HEALTH = "/api/v1/health"
DAMAGE = "/api/v1/damage/evaluate"


@pytest.fixture
def log_db(tmp_path: Path) -> Path:
    path = tmp_path / "log.sqlite"
    ApiCallLogRepository(path).ensure_table()
    return path


@pytest.fixture
def logged_client(client: TestClient, log_db: Path) -> TestClient:
    client.app.state.api_call_log = ApiCallLogRepository(log_db)
    return client


def _rows(path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM api_call_logs ORDER BY id").fetchall()
    finally:
        conn.close()


def test_success_is_logged(logged_client: TestClient, log_db: Path) -> None:
    assert logged_client.get(HEALTH).status_code == 200

    [row] = _rows(log_db)
    assert row["api_name"] == HEALTH
    assert row["http_method"] == "GET"
    assert row["return_code"] == 200
    assert row["return_msg"] == "OK"
    assert row["client"] == "testclient"  # no X-Client-Id: falls back to the client IP
    assert row["client_ip"] == "testclient"
    assert row["duration_ms"] >= 0
    assert row["record_time"]


def test_error_message_and_client_id(logged_client: TestClient, log_db: Path) -> None:
    resp = logged_client.post(
        DAMAGE,
        files={"file": ("a.gif", b"GIF89a", "image/gif")},
        headers={"X-Client-Id": "smoke-test"},
    )
    assert resp.status_code == 415

    [row] = _rows(log_db)
    assert row["api_name"] == DAMAGE
    assert row["return_code"] == 415
    assert row["return_msg"] == resp.json()["error"]["message"]
    assert row["client"] == "smoke-test"


def test_validation_error_is_logged(logged_client: TestClient, log_db: Path) -> None:
    assert logged_client.post(DAMAGE).status_code == 422

    [row] = _rows(log_db)
    assert row["return_code"] == 422
    assert "file" in row["return_msg"]


def test_unknown_route_logs_raw_path(logged_client: TestClient, log_db: Path) -> None:
    assert logged_client.get("/api/v1/nope").status_code == 404

    [row] = _rows(log_db)
    assert row["api_name"] == "/api/v1/nope"
    assert row["return_msg"] == "Not Found"


def test_non_api_paths_are_not_logged(logged_client: TestClient, log_db: Path) -> None:
    assert logged_client.get("/").status_code == 200
    assert _rows(log_db) == []


def test_log_failure_does_not_break_response(client: TestClient, tmp_path: Path) -> None:
    # No table in this file, so every INSERT fails.
    client.app.state.api_call_log = ApiCallLogRepository(tmp_path / "empty.sqlite")
    assert client.get(HEALTH).status_code == 200
