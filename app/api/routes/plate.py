"""License-plate recognition endpoint."""

from __future__ import annotations

import time
import uuid

from anyio import to_thread
from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_plate_recognizer, get_settings
from app.config import Settings
from app.schemas.common import BoundingBox
from app.schemas.errors import ErrorResponse
from app.schemas.plate import PlateReading, PlateRecognitionResponse
from app.services.image_io import load_upload
from app.services.plate_recognizer import PlateRecognizer, RawPlate, format_plate

router = APIRouter(prefix="/plate", tags=["plate"])

_ERROR_RESPONSES = {
    413: {"model": ErrorResponse, "description": "Image exceeds the maximum allowed size"},
    415: {"model": ErrorResponse, "description": "Unsupported image content type"},
    422: {"model": ErrorResponse, "description": "Missing or unreadable image"},
}


def _to_reading(raw: RawPlate, size: tuple[int, int]) -> PlateReading:
    formatted = format_plate(raw.text)
    box = None
    if raw.xyxy:
        w, h = size
        x1, y1, x2, y2 = (
            max(0.0, min(v, limit)) for v, limit in zip(raw.xyxy, (w, h, w, h), strict=True)
        )
        if x2 > x1 and y2 > y1:
            box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return PlateReading(
        plate_number=formatted or raw.text.upper(),
        raw_text=raw.text,
        confidence=max(0.0, min(raw.confidence, 1.0)),
        valid_format=formatted is not None,
        bounding_box=box,
    )


def _merge_split_plate(raws: list[RawPlate]) -> RawPlate | None:
    """Plates are often OCR'd as two boxes ('ABC' + '1234'); try joining neighbours."""

    for a, b in zip(raws, raws[1:], strict=False):
        if format_plate(a.text) is None and format_plate(b.text) is None:
            joined = a.text + b.text
            if format_plate(joined):
                boxes = [x for x in (a.xyxy, b.xyxy) if x]
                xyxy = None
                if boxes:
                    xyxy = (
                        min(x[0] for x in boxes),
                        min(x[1] for x in boxes),
                        max(x[2] for x in boxes),
                        max(x[3] for x in boxes),
                    )
                return RawPlate(joined, min(a.confidence, b.confidence), xyxy)
    return None


@router.post(
    "/recognize",
    response_model=PlateRecognitionResponse,
    responses=_ERROR_RESPONSES,
    summary="Read the license plate number from a vehicle photo",
)
async def recognize_plate(
    file: UploadFile = File(..., description="A JPEG, PNG or WebP photo showing the plate."),
    recognizer: PlateRecognizer = Depends(get_plate_recognizer),
    settings: Settings = Depends(get_settings),
) -> PlateRecognitionResponse:
    image, metadata = await load_upload(file, settings)

    started = time.perf_counter()
    raws = await to_thread.run_sync(recognizer.read, image)
    inference_ms = (time.perf_counter() - started) * 1000.0

    merged = _merge_split_plate(raws)
    candidates = ([merged] if merged else []) + raws
    plates = [_to_reading(r, image.size) for r in candidates]
    plates = [p for p in plates if p.valid_format or p.confidence >= settings.plate_min_confidence]
    plates.sort(key=lambda p: (p.valid_format, p.confidence), reverse=True)
    best = next((p for p in plates if p.valid_format), None)

    return PlateRecognitionResponse(
        request_id=str(uuid.uuid4()),
        image=metadata,
        plates=plates,
        best_plate=best,
        model_name=recognizer.name,
        is_mock=recognizer.is_mock,
        inference_ms=round(inference_ms, 3),
    )
