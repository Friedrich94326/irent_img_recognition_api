"""Application error hierarchy and the handlers that render it as :class:`ErrorResponse`."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.errors import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for expected, client-facing errors.

    Subclasses set ``status_code`` and ``code``; the message is passed at raise time.
    """

    status_code: int = 400
    code: str = "bad_request"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorDetail(code=self.code, message=self.message, field=self.field)
        )


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class InvalidImageError(AppError):
    status_code = 422
    code = "invalid_image"


def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request.state.error_message = exc.message  # picked up by the API call log
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response().model_dump(),
    )


def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else None
    field = None
    if first and first.get("loc"):
        field = ".".join(str(part) for part in first["loc"] if part != "body")
    message = first["msg"] if first else "Request validation failed."
    request.state.error_message = f"{field}: {message}" if field else message
    body = ErrorResponse(
        error=ErrorDetail(code="validation_error", message=message, field=field or None)
    )
    return JSONResponse(status_code=422, content=body.model_dump())


def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    body = ErrorResponse(
        error=ErrorDetail(code="internal_error", message="An unexpected error occurred.")
    )
    return JSONResponse(status_code=500, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
