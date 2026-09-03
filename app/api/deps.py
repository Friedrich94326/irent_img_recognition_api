"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.services.evaluator import DamageEvaluator


def get_evaluator(request: Request) -> DamageEvaluator:
    """Return the process-wide detector created during application startup."""

    return request.app.state.evaluator


SettingsDep = Depends(get_settings)
EvaluatorDep = Depends(get_evaluator)

__all__ = ["EvaluatorDep", "SettingsDep", "Settings", "get_evaluator", "get_settings"]
