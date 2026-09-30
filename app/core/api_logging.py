"""Middleware that writes one ``api_call_logs`` row per ``/api/v1`` request."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from http import HTTPStatus

from anyio import to_thread
from fastapi import FastAPI, Request, Response

from app.services.api_call_log import ApiCallLogRepository

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
CLIENT_ID_HEADER = "X-Client-Id"


def _return_msg(request: Request, status_code: int) -> str:
    message = getattr(request.state, "error_message", None)
    if message:
        return message
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return str(status_code)


def _api_name(request: Request) -> str:
    """Full route template, e.g. ``/api/v1/damage/evaluate``; the raw path if no route matched.

    ``scope["route"]`` holds the innermost route, whose template lacks the included routers'
    prefixes, so those are taken from the leading segments of the actual path.
    """

    path = request.url.path
    template = getattr(request.scope.get("route"), "path", None)
    if not template:
        return path
    prefix_segments = path.count("/") - template.count("/")
    return "/".join(path.split("/")[: prefix_segments + 1]) + template


def register_api_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_api_call(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        repo: ApiCallLogRepository | None = getattr(request.app.state, "api_call_log", None)
        if repo is None or not request.url.path.startswith(API_PREFIX):
            return await call_next(request)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            # Unhandled errors are rendered as 500 by the outermost ServerErrorMiddleware,
            # so they surface here as an exception instead of a response.
            request.state.error_message = f"Unhandled {type(exc).__name__}: {exc}"
            await _record(request, repo, 500, started)
            raise
        await _record(request, repo, response.status_code, started)
        return response


async def _record(
    request: Request, repo: ApiCallLogRepository, status_code: int, started: float
) -> None:
    duration_ms = (time.perf_counter() - started) * 1000.0
    client_ip = request.client.host if request.client else None
    try:
        await to_thread.run_sync(
            repo.record,
            _api_name(request),
            request.method,
            status_code,
            _return_msg(request, status_code),
            request.headers.get(CLIENT_ID_HEADER) or client_ip or "unknown",
            client_ip,
            request.headers.get("user-agent"),
            duration_ms,
        )
    except Exception:  # logging must never break the API response
        logger.exception("Failed to record API call %s %s", request.method, request.url.path)
