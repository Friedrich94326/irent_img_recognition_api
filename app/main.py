"""Application factory and ASGI entrypoint (``uvicorn app.main:app``)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import Settings, get_settings
from app.core.api_logging import register_api_logging
from app.core.exceptions import register_exception_handlers
from app.services.api_call_log import build_api_call_log
from app.services.evaluator import build_evaluator
from app.services.plate_recognizer import build_plate_recognizer
from app.services.tire_detector import build_tire_detector
from app.services.vehicle_repository import build_vehicle_repository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    app.state.evaluator = build_evaluator(settings)
    logger.info(
        "Detector ready: %s (mock=%s)", app.state.evaluator.name, app.state.evaluator.is_mock
    )
    app.state.plate_recognizer = build_plate_recognizer(settings)
    logger.info("Plate recognizer ready: %s", app.state.plate_recognizer.name)
    app.state.tire_detector = build_tire_detector(settings)
    logger.info(
        "Tire detector ready: %s (mock=%s)",
        app.state.tire_detector.name,
        app.state.tire_detector.is_mock,
    )
    app.state.vehicle_repository = build_vehicle_repository(settings)
    app.state.api_call_log = build_api_call_log(settings)
    yield
    app.state.evaluator = None
    app.state.plate_recognizer = None
    app.state.tire_detector = None
    app.state.vehicle_repository = None
    app.state.api_call_log = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=settings.description,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    register_api_logging(app)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": settings.app_name, "version": settings.version, "docs": "/docs"}

    return app


app = create_app()
