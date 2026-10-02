"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.services.corner_classifier import CornerClassifier
from app.services.evaluator import DamageEvaluator
from app.services.plate_recognizer import PlateRecognizer
from app.services.precheck import PrecheckRepository
from app.services.tire_detector import TireDetector
from app.services.vehicle_photos import VehiclePhotoRepository
from app.services.vehicle_repository import VehicleRepository


def get_evaluator(request: Request) -> DamageEvaluator:
    """Return the process-wide detector created during application startup."""

    return request.app.state.evaluator


def get_plate_recognizer(request: Request) -> PlateRecognizer:
    """Return the process-wide plate recognizer created during application startup."""

    return request.app.state.plate_recognizer


def get_tire_detector(request: Request) -> TireDetector:
    """Return the process-wide tire detector created during application startup."""

    return request.app.state.tire_detector


def get_corner_classifier(request: Request) -> CornerClassifier | None:
    """Return the corner classifier, or ``None`` when no weights are configured."""

    return request.app.state.corner_classifier


def get_vehicle_repository(request: Request) -> VehicleRepository | None:
    """Return the iRent ops database repository, or ``None`` when no database is configured."""

    return request.app.state.vehicle_repository


def get_precheck_repository(request: Request) -> PrecheckRepository | None:
    """Return the pre-rental check repository, or ``None`` when no database is configured."""

    return request.app.state.precheck_repository


def get_vehicle_photo_repository(request: Request) -> VehiclePhotoRepository | None:
    """Return the corner photo store, or ``None`` when no database is configured."""

    return request.app.state.vehicle_photo_repository


SettingsDep = Depends(get_settings)
EvaluatorDep = Depends(get_evaluator)

__all__ = [
    "EvaluatorDep",
    "SettingsDep",
    "Settings",
    "get_corner_classifier",
    "get_evaluator",
    "get_plate_recognizer",
    "get_precheck_repository",
    "get_settings",
    "get_tire_detector",
    "get_vehicle_photo_repository",
    "get_vehicle_repository",
]
