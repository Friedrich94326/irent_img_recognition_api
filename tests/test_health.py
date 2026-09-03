from __future__ import annotations

from fastapi.testclient import TestClient

from app.schemas.health import HealthResponse


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200

    body = HealthResponse.model_validate(resp.json())
    assert body.status == "ok"
    assert body.model_loaded is True
    assert body.model_is_mock is True
    assert body.model_name == "mock-detector"
    assert body.weights_path is None
