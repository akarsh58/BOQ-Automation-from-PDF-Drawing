"""Measurement logic for tender takeoff: gross/wastage, opening deductions, QA."""
from __future__ import annotations

import math
from typing import Optional

from app.models import DEFAULT_ITEM_CODES, Deficiency, MeasuredElement, MeasurementLine, TakeoffDocument

IS1200_OPENING_DEDUCT_M2 = 0.1
IS1200_DEFAULT_WALL_HEIGHT = 3.0
IS1200_DEFAULT_WALL_THICKNESS = 0.23
IS1200_JOINTS_DEDUCTION_PERCENT = 2.5
IS1200_BRICKWORK_WASTAGE_PERCENT = 5.0

QA_STATUSES = {"draft", "review", "approved", "rejected"}


def _centerline_length(points) -> float:
    if not points or len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _area_from_polygon(points) -> float:
    if not points or len(points) < 3:
        return 0.0
    a = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _opening_area(el: MeasuredElement) -> float:
    w = el.width_m or 0.0
    h = el.height_m or 0.0
    if w and h:
        return w * h
    if el.polygon_m and len(el.polygon_m) >= 3:
        return _area_from_polygon(el.polygon_m)
    return 0.0


def _wall_gross(wall: MeasuredElement) -> float:
    length = _centerline_length(wall.centerline_m or [])
    if length <= 0 and wall.length_m:
        length = wall.length_m
    height = wall.height_m or IS1200_DEFAULT_WALL_HEIGHT
    thickness = wall.thickness_m or IS1200_DEFAULT_WALL_THICKNESS
    return length * height * thickness


def _should_deduct(opening: MeasuredElement, host: Optional[MeasuredElement]) -> bool:
    area = _opening_area(opening)
    if area < IS1200_OPENING_DEDUCT_M2:
        return False
    if host is None:
        return False
    host_length = _centerline_length(host.centerline_m or [])
    if host_length > 0 and host_length < 1.0:
        return False
    if host.length_m is not None and host.length_m < 1.0:
        return False
    # Opening must be near the host wall to be deducted.
    # If the opening centerline is far from the wall centerline, treat it as not attached.
    if opening.centerline_m and host.centerline_m:
        ox, oy = opening.centerline_m[0]
        hx, hy = host.centerline_m[0]
        if abs(oy - hy) > 1.0:
            return False
    return True


def _wastage(quantity: float, percent: float) -> float:
    return quantity * percent / 100.0


def _item_code_label(item_code: Optional[float], fallback: str = "") -> str:
    return f"Item {item_code}" if item_code is not None else fallback


def measure_takeoff(doc: TakeoffDocument, kb) -> tuple[list[MeasurementLine], list[Deficiency]]:
    elements = doc.elements or []
    lines: list[MeasurementLine] = []
    deficiencies: list[Deficiency] = []
    walls = [e for e in elements if e.type == "wall"]
    openings = [e for e in elements if e.type == "opening"]
    used_wall_ids = set()

    for wall in walls:
        gross = _wall_gross(wall)
        deduction = 0.0
        for opening in openings:
            if opening.host_id == wall.id and _should_deduct(opening, wall):
                deduction += _opening_area(opening)
        net = max(gross - deduction, 0.0)
        brickwork_gross = gross
        brickwork_wastage = _wastage(brickwork_gross, IS1200_BRICKWORK_WASTAGE_PERCENT)
        brickwork_qty = brickwork_gross + brickwork_wastage
        lines.append(MeasurementLine(
            ref=wall.id, element_id=wall.id, item_code=DEFAULT_ITEM_CODES["wall"],
            description="Brickwork in CM 1:6", unit="cum",
            length=wall.length_m, breadth=wall.thickness_m, depth=None,
            quantity=round(brickwork_qty, 3),
            gross_quantity=round(brickwork_gross, 3),
            wastage_quantity=round(brickwork_wastage, 3),
            wastage_percent=IS1200_BRICKWORK_WASTAGE_PERCENT,
            provisional=False, source="2d", location=wall.name,
            is1200_note=(f"Deduction for openings {'applied' if deduction > 0 else 'not deducted'}; "
                         f"joints {IS1200_JOINTS_DEDUCTION_PERCENT}%")
        ))
        plaster_qty = net * 2
        plaster_wastage = _wastage(plaster_qty, 10.0)
        lines.append(MeasurementLine(
            ref=wall.id, element_id=wall.id, item_code=DEFAULT_ITEM_CODES["plaster_internal"],
            description="Internal plaster", unit="sqm",
            length=None, breadth=None, depth=None,
            quantity=round(plaster_qty + plaster_wastage, 3),
            gross_quantity=round(plaster_qty, 3),
            wastage_quantity=round(plaster_wastage, 3),
            wastage_percent=10.0, provisional=False, source="2d", location=wall.name,
            is1200_note=f"Plaster thickness 0.012m; wastage 10%"
        ))
        used_wall_ids.add(wall.id)

    # NOTE: require_tender_gate should be called by the caller, not here
    return lines, deficiencies


def parse_takeoff(raw: dict) -> TakeoffDocument:
    elements = []
    for el in (raw.get("elements") or []):
        kind = el.get("opening_kind")
        etype = el.get("type")
        elements.append(MeasuredElement(
            id=el.get("id", ""), type=etype, name=el.get("name", ""),
            source="2d", page=el.get("page", 0),
            polygon_m=el.get("points"), centerline_m=el.get("points"),
            thickness_m=el.get("thickness_m"), height_m=el.get("height_m"),
            length_m=el.get("length_m"), width_m=el.get("width_m"),
            depth_m=el.get("depth_m"), host_id=el.get("host_id"),
            host_space_ids=el.get("host_space_ids") or [],
            shared=el.get("shared", False), opening_kind=kind,
            item_code=el.get("item_code"), ifc_type=etype,
        ))
    return TakeoffDocument(
        project_name=raw.get("project_name", ""),
        source_filename=raw.get("source_filename", ""),
        qs_name=raw.get("qs_name", ""), qs_signed=bool(raw.get("qs_signed")),
        units=raw.get("units", "m"), scale_method=raw.get("scale_method", "manual"),
        ifc_units_confirmed=bool(raw.get("ifc_units_confirmed")),
        elements=elements,
    )


def require_tender_gate(doc: TakeoffDocument) -> None:
    if not doc.qs_name:
        raise ValueError("Tender gate failed: QS name required")
    if not doc.qs_signed:
        raise ValueError("Tender gate failed: unsigned QS")
    if doc.units not in ("m", "mm", "cm", "ft"):
        raise ValueError(f"Tender gate failed: unsupported units {doc.units}")


def record_qa_review(doc: TakeoffDocument, reviewer: str, status: str) -> None:
    if status not in QA_STATUSES:
        raise ValueError(f"Invalid QA status: {status}")
    if doc.qa_status not in QA_STATUSES:
        doc.revision_history.append({"qa_status_before": "", "qa_status_after": doc.qa_status, "reviewer": reviewer, "date": doc.qa_review_date})
    else:
        doc.revision_history.append({"qa_status_before": doc.qa_status, "qa_status_after": status, "reviewer": reviewer, "date": doc.qa_review_date})
    doc.qa_status = status
    doc.qa_reviewer = reviewer
    doc.qa_review_date = doc.qa_review_date or "pending"

