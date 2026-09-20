"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    All values can be overridden with environment variables prefixed with ``IRENT_``
    (e.g. ``IRENT_YOLO_WEIGHTS_PATH=./weights/car_damage.pt``) or via a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="IRENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "iRent Car Damage Evaluation API"
    version: str = "0.1.0"
    description: str = (
        "Upload a photo of a rental vehicle and receive detected damage, "
        "a per-class summary, and a derived overall severity."
    )

    # --- Model / inference -------------------------------------------------
    yolo_weights_path: Path | None = Field(
        default=None,
        description="Path to a YOLOv8 .pt file. If unset or missing, a mock detector is used.",
    )
    yolo_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    yolo_device: str = "cpu"

    # --- License plate OCR -------------------------------------------------
    plate_use_mock: bool = Field(
        default=False, description="Force the mock plate recognizer (skips loading EasyOCR)."
    )
    plate_use_opencv_locator: bool = Field(
        default=True,
        description="Crop plate-shaped regions with OpenCV before OCR (else OCR the full image).",
    )
    plate_min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to report text that does not match a plate format.",
    )

    # --- Upload constraints ----------------------------------------------
    max_image_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    allowed_content_types: set[str] = {"image/jpeg", "image/png", "image/webp"}

    # --- HTTP ------------------------------------------------------------
    cors_allow_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""

    return Settings()
