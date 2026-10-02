"""Top-level API router: mounts every versioned route group under ``/api/v1``."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import damage, health, plate, precheck, tire, vehicles

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(damage.router)
api_router.include_router(plate.router)
api_router.include_router(tire.router)
api_router.include_router(precheck.router)
api_router.include_router(vehicles.router)
