"""Uniform error envelope used for every non-2xx response."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation.")
    field: str | None = Field(default=None, description="Offending field, for validation errors.")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "error": {
                    "code": "unsupported_media_type",
                    "message": "Unsupported content type 'image/gif'.",
                    "field": "file",
                }
            }
        },
    )

    error: ErrorDetail
