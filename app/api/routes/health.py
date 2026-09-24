"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_evaluator, get_plate_recognizer, get_settings, get_tire_detector
from app.config import Settings
from app.schemas.health import HealthResponse
from app.services.evaluator import DamageEvaluator
from app.services.plate_recognizer import PlateRecognizer
from app.services.tire_detector import TireDetector

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and model status")
def health(
    evaluator: DamageEvaluator = Depends(get_evaluator),
    plate_recognizer: PlateRecognizer = Depends(get_plate_recognizer),
    tire_detector: TireDetector = Depends(get_tire_detector),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.version,
        model_loaded=True,
        model_is_mock=evaluator.is_mock,
        model_name=evaluator.name,
        plate_model_name=plate_recognizer.name,
        plate_is_mock=plate_recognizer.is_mock,
        tire_model_name=tire_detector.name,
        tire_is_mock=tire_detector.is_mock,
        weights_path=str(settings.yolo_weights_path) if settings.yolo_weights_path else None,
    )
