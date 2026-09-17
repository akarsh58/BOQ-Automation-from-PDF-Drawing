"""FastAPI application for QS-assisted takeoff, IFC import and tender BOQ export."""

import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.boq_engine import (
    add_summary_row,
    build_boq_dataframe,
    dimensions_to_line_items,
    load_knowledge_base,
    rooms_to_line_items,
)
from app.extraction import (
    detect_rooms_walls,
    estimate_scale_from_text,
    extract_text_ocr,
    extract_room_labels,
    assign_room_labels,
    extract_text_vector,
    find_dimension_strings,
    pdf_to_images,
    pixels_to_units,
)
from app.ifc_import import import_ifc
from app.measurement import parse_takeoff, record_qa_review, validate_qa_requirements
from app.tender_export import write_tender_workbook

app = FastAPI(title="BOQ Automation API")


def _cors_origins() -> list[str]:
    configured = os.getenv("BOQ_CORS_ORIGINS", "")
    if configured.strip():
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "null",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5500",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
UPLOAD_DIR = OUTPUT_DIR / ".uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".ifc", ".ifczip"}
RENDER_DPI = 200
MAX_IMAGE_PIXELS = 60_000_000


def _safe_extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def _is_ifc(path: Path) -> bool:
    return path.suffix.lower() in {".ifc", ".ifczip"}


async def _save_upload(upload: UploadFile) -> Path:
    suffix = _safe_extension(upload.filename or "")
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a PDF, image, or IFC file.")

    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with path.open("wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="The file must be 25 MB or smaller.")
                target.write(chunk)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        LOGGER.exception("Could not read uploaded file")
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.") from exc
    finally:
        await upload.close()
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return path


def _encode_preview_image(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise HTTPException(status_code=422, detail="Could not create a preview image.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _draft_space_from_box(
    page_index: int,
    room_index: int,
    box,
    px_per_unit: float,
    name: str = "",
    confidence: float = 0.0,
) -> dict[str, Any]:
    x, y, width, height = box
    return {
        "id": f"p{page_index}-r{room_index + 1}",
        "type": "space",
        "name": name or f"Room {room_index + 1}",
        "confidence": confidence,
        "source": "2d",
        "page": page_index,
        "points": [
            [int(x), int(y)],
            [int(x + width), int(y)],
            [int(x + width), int(y + height)],
            [int(x), int(y + height)],
        ],
        "item_code": 5.1,
        "width_m": round(width / px_per_unit, 2),
        "height_m": None,
        "draft": True,
    }


def _page_data(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            images = pdf_to_images(str(path), dpi=RENDER_DPI)
            texts = extract_text_vector(str(path))
        except Exception as exc:
            LOGGER.exception("Could not read PDF")
            raise HTTPException(status_code=422, detail="Could not read the PDF.") from exc
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=422, detail="The image could not be decoded.")
        if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=422, detail="The image dimensions are too large.")
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
        texts = [""]

    if not images:
        raise HTTPException(status_code=422, detail="The drawing has no readable pages.")

    pages: list[dict[str, Any]] = []
    for page_index, image in enumerate(images):
        vector_text = texts[page_index] if page_index < len(texts) else ""
        text = vector_text
        if not text.strip():
            try:
                text = extract_text_ocr(image)
            except Exception:
                text = ""
        px_per_unit = estimate_scale_from_text(text, render_dpi=RENDER_DPI)
        boxes = detect_rooms_walls(image)
        try:
            room_labels = extract_room_labels(image)
        except Exception:
            room_labels = []
        sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        label_assignments = assign_room_labels(sorted_boxes, room_labels)
        rooms = []
        elements = []
        for room_index, box in enumerate(sorted_boxes):
            x, y, width, height = box
            assignment = label_assignments[room_index]
            room_name = assignment["name"] or f"Room {room_index + 1}"
            rooms.append(
                {
                    "id": f"p{page_index}-r{room_index + 1}",
                    "page": page_index,
                    "name": room_name,
                    "confidence": assignment["confidence"],
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                    "width_m": round(width / px_per_unit, 2),
                    "height_m": round(height / px_per_unit, 2),
                }
            )
            elements.append(
                _draft_space_from_box(
                    page_index,
                    room_index,
                    box,
                    px_per_unit,
                    room_name,
                    assignment["confidence"],
                )
            )
        pages.append(
            {
                "page": page_index,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "image": _encode_preview_image(image),
                "rooms": rooms,
                "elements": elements,
                "px_per_unit": round(px_per_unit, 4),
                "scale_calibrated": False,
                "scale_detected": bool(
                    re.search(r"\bSCALE\s*1\s*[:/]\s*\d+(?:\.\d+)?\b", text, re.IGNORECASE)
                ),
                "dimensions": find_dimension_strings(text),
            }
        )
    return pages


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be a number.") from exc
    if not minimum <= number <= maximum:
        raise HTTPException(status_code=422, detail=f"{field} must be between {minimum:g} and {maximum:g}.")
    return number


def _page_scales(raw: Optional[str], pages: list[dict[str, Any]]) -> Optional[list[float]]:
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="page_scales must be valid JSON.") from exc
    if isinstance(data, dict):
        values = [data.get(str(index)) for index in range(len(pages))]
    elif isinstance(data, list):
        values = data
    else:
        raise HTTPException(status_code=422, detail="page_scales must be a list or object.")
    if len(values) != len(pages):
        raise HTTPException(status_code=422, detail="page_scales must include one value per page.")
    return [
        _number(value, f"Page {index + 1} px_per_unit", 1, 1_000_000)
        for index, value in enumerate(values)
    ]


def _reviewed_rooms(raw: Optional[str], pages: list[dict[str, Any]]) -> Optional[list[dict[str, Any]]]:
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="reviewed_rooms must be valid JSON.") from exc
    if isinstance(data, dict):
        data = data.get("rooms")
    if not isinstance(data, list) or len(data) > 500:
        raise HTTPException(status_code=422, detail="reviewed_rooms must be a list of at most 500 rooms.")

    rooms = []
    for index, room in enumerate(data):
        if not isinstance(room, dict):
            raise HTTPException(status_code=422, detail=f"Room {index + 1} is invalid.")
        try:
            page = int(room.get("page", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Room {index + 1} page is invalid.") from exc
        if isinstance(room.get("page"), bool) or page < 0 or page >= len(pages):
            raise HTTPException(status_code=422, detail=f"Room {index + 1} references an invalid page.")
        x = _number(room.get("x", 0), f"Room {index + 1} x", 0, 100_000)
        y = _number(room.get("y", 0), f"Room {index + 1} y", 0, 100_000)
        width = _number(room.get("width", room.get("w")), f"Room {index + 1} width", 1, 100_000)
        height = _number(room.get("height", room.get("h")), f"Room {index + 1} height", 1, 100_000)
        if x + width > pages[page]["width"] or y + height > pages[page]["height"]:
            raise HTTPException(status_code=422, detail=f"Room {index + 1} must fit inside its page.")
        name = str(room.get("name", f"Room {index + 1}")).strip()[:80] or f"Room {index + 1}"
        rooms.append({"page": page, "x": x, "y": y, "width": width, "height": height, "name": name})
    return rooms


def _make_legacy_workbook(
    path: Path,
    wall_height_m: float,
    wall_thickness_m: float = 0.23,
    reviewed_rooms: Optional[list[dict[str, Any]]] = None,
    px_per_unit: Optional[float] = None,
    page_scales: Optional[list[float]] = None,
    pages: Optional[list[dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    pages = pages if pages is not None else _page_data(path)
    kb = load_knowledge_base()
    all_items: list[dict[str, Any]] = []

    if reviewed_rooms is not None:
        for room in reviewed_rooms:
            page = pages[room["page"]]
            scale = page_scales[room["page"]] if page_scales else (px_per_unit or page["px_per_unit"])
            measured = pixels_to_units((room["x"], room["y"], room["width"], room["height"]), scale)
            measured["name"] = room["name"]
            all_items.extend(rooms_to_line_items([measured], kb, wall_height_m, wall_thickness_m))
    else:
        for page in pages:
            if page["rooms"]:
                scale = page_scales[page["page"]] if page_scales else (px_per_unit or page["px_per_unit"])
                rooms = []
                for room in page["rooms"]:
                    rooms.append(
                        pixels_to_units((room["x"], room["y"], room["width"], room["height"]), scale)
                    )
                all_items.extend(rooms_to_line_items(rooms, kb, wall_height_m, wall_thickness_m))
            elif page["dimensions"]:
                all_items.extend(dimensions_to_line_items(page["dimensions"], kb))

    dataframe = add_summary_row(build_boq_dataframe(all_items))
    output_path = output_path or (OUTPUT_DIR / "generated_boq.xlsx")
    dataframe.to_excel(output_path, index=False)
    return output_path


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "BOQ Automation API is running. Use /preview and /generate for tender takeoff. See /docs",
    }


@app.post("/preview")
async def preview(file: UploadFile = File(...)):
    """Upload a drawing or IFC model and return pages and/or measured element drafts."""
    path = await _save_upload(file)
    try:
        if _is_ifc(path):
            try:
                imported = import_ifc(path)
            except RuntimeError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                LOGGER.exception("Could not read IFC")
                raise HTTPException(status_code=422, detail="Could not read the IFC model.") from exc
            return {
                "filename": file.filename,
                "kind": "ifc",
                "pages": [
                    {
                        "page": 0,
                        "width": 1000,
                        "height": 700,
                        "image": None,
                        "rooms": [],
                        "elements": imported["elements"],
                        "px_per_unit": 1,
                        "scale_calibrated": False,
                        "scale_detected": False,
                        "dimensions": [],
                    }
                ],
                "elements": imported["elements"],
                "skipped": imported.get("skipped") or [],
                "units": "m",
                "scale_method": "ifc_units",
                "schema": imported.get("schema"),
                "unit_scale_to_metres": imported.get("unit_scale_to_metres"),
                "element_count": imported.get("element_count", 0),
                "room_count": 0,
                "page_scales": [1],
            }

        pages = _page_data(path)
        elements = [el for page in pages for el in page.get("elements", [])]
        return {
            "filename": file.filename,
            "kind": "drawing",
            "pages": pages,
            "rooms": [room for page in pages for room in page["rooms"]],
            "elements": elements,
            "room_count": sum(len(page["rooms"]) for page in pages),
            "px_per_unit": pages[0]["px_per_unit"],
            "scale": pages[0]["px_per_unit"],
            "page_scales": [page["px_per_unit"] for page in pages],
            "units": "px",
        }
    finally:
        path.unlink(missing_ok=True)


async def _generate(
    file: UploadFile,
    wall_height_m: float,
    wall_thickness_m: float,
    reviewed_rooms: Optional[str],
    px_per_unit: Optional[float],
    page_scales: Optional[str] = None,
    takeoff: Optional[str] = None,
):
    wall_height = _number(wall_height_m, "wall_height_m", 0.1, 20)
    wall_thickness = _number(wall_thickness_m, "wall_thickness_m", 0.01, 2)
    scale = None if px_per_unit is None else _number(px_per_unit, "px_per_unit", 1, 1_000_000)
    path = await _save_upload(file)
    output_path = OUTPUT_DIR / f"generated_boq_{uuid.uuid4().hex}.xlsx"
    try:
        if takeoff and takeoff.strip():
            try:
                pages = None if _is_ifc(path) else _page_data(path)
                page_count = None if pages is None else len(pages)
                doc = parse_takeoff(takeoff, page_count=page_count)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if not doc.source_filename:
                doc.source_filename = file.filename or path.name
            kb = load_knowledge_base()
            try:
                write_tender_workbook(output_path, doc, kb, enforce_signoff=True)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return FileResponse(
                output_path,
                filename="tender_boq.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                background=BackgroundTask(output_path.unlink, missing_ok=True),
            )

        if _is_ifc(path):
            raise HTTPException(
                status_code=422,
                detail="IFC files require a reviewed takeoff JSON (confirm units, codes, and QS sign-off).",
            )
        pages = _page_data(path)
        rooms = _reviewed_rooms(reviewed_rooms, pages)
        scales = _page_scales(page_scales, pages)
        if rooms is None:
            raise HTTPException(
                status_code=422,
                detail="Review and submit the detected or manually traced rooms before generating a BOQ.",
            )
        if scales is None:
            raise HTTPException(
                status_code=422,
                detail="Calibrate every drawing page and submit page_scales before generating a BOQ.",
            )
        _make_legacy_workbook(path, wall_height, wall_thickness, rooms, scale, scales, pages, output_path)
        return FileResponse(
            output_path,
            filename="generated_boq.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=BackgroundTask(output_path.unlink, missing_ok=True),
        )
    except HTTPException:
        output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        LOGGER.exception("Could not generate BOQ")
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Could not generate the BOQ.") from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    reviewed_rooms: Optional[str] = Form(None),
    px_per_unit: Optional[float] = Form(None),
    page_scales: Optional[str] = Form(None),
    wall_height_m: float = Form(3.0),
    wall_thickness_m: float = Form(0.23),
    takeoff: Optional[str] = Form(None),
):
    """Generate a tender workbook from takeoff JSON, or a legacy sheet from reviewed rooms."""
    return await _generate(file, wall_height_m, wall_thickness_m, reviewed_rooms, px_per_unit, page_scales, takeoff)


@app.post("/generate-boq")
async def generate_boq(
    file: UploadFile = File(...),
    wall_height_m: float = Form(3.0),
    wall_thickness_m: float = Form(0.23),
    reviewed_rooms: Optional[str] = Form(None),
    px_per_unit: Optional[float] = Form(None),
    page_scales: Optional[str] = Form(None),
    takeoff: Optional[str] = Form(None),
):
    """Backward-compatible generation endpoint."""
    return await _generate(file, wall_height_m, wall_thickness_m, reviewed_rooms, px_per_unit, page_scales, takeoff)


@app.post("/qa-validate")
async def qa_validate(takeoff: str = Form(...)):
    """Validate QA requirements for tender-grade work."""
    try:
        doc = parse_takeoff(takeoff)
        is_valid, issues = validate_qa_requirements(doc)
        return {
            "valid": is_valid,
            "issues": issues,
            "qa_status": doc.qa_status,
            "qs_signed": doc.qs_signed,
            "measurement_standard": doc.measurement_standard,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/qa-review")
async def qa_review(
    takeoff: str = Form(...),
    reviewer: str = Form(...),
    status: str = Form(...),
    comments: Optional[str] = Form(None),
):
    """Record a QA review for the takeoff document."""
    try:
        doc = parse_takeoff(takeoff)
        comments_list = [comments] if comments else []
        updated_doc = record_qa_review(doc, reviewer, status, comments_list)
        return {
            "qa_status": updated_doc.qa_status,
            "qa_reviewer": updated_doc.qa_reviewer,
            "qa_review_date": updated_doc.qa_review_date,
            "qa_comments": updated_doc.qa_comments,
            "revision_count": len(updated_doc.revision_history),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
