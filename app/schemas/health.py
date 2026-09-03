"""Schema for the health endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness plus a snapshot of the loaded detector."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "model_loaded": True,
                "model_is_mock": True,
                "model_name": "mock-detector",
                "weights_path": None,
            }
        },
    )

    status: Literal["ok"] = "ok"
    version: str
    model_loaded: bool = Field(description="Whether a detector is ready to serve requests.")
    model_is_mock: bool = Field(description="True when the built-in mock detector is active.")
    model_name: str
    weights_path: str | None = Field(
        default=None, description="Configured YOLOv8 weights path, if any."
    )
