"""IS 1200 measurement engine and takeoff JSON parser."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

import pandas as pd

from app.models import (
    DEFAULT_ITEM_CODES,
    ELEMENT_TYPES,
    IS1200_BRICKWORK_WASTAGE_PERCENT,
    IS1200_CONCRETE_WASTAGE_PERCENT,
    IS1200_DEDUCTION_THRESHOLD_AREA,
    IS1200_DEDUCTION_THRESHOLD_LENGTH,
    IS1200_OPENING_DEDUCT_M2,
    IS1200_PLASTER_THICKNESS_EXTERNAL,
    IS1200_PLASTER_THICKNESS_INTERNAL,
    OPENING_KINDS,
    Calibration,
    Deficiency,
    MeasuredElement,
    MeasurementLine,
    TakeoffDocument,
)

MAX_ELEMENTS = 2000


def polyline_length(points: list[tuple[float, float]]) -> float:
    if not points or len(points) < 2:
        return 0.0
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += math.hypot(end[0] - start[0], end[1] - start[1])
    return total


def polygon_area(points: list[tuple[float, float]]) -> float:
    if not points or len(points) < 3:
        return 0.0
    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    area = 0.0
    for (x1, y1), (x2, y2) in zip(closed, closed[1:]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _finite_positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _code(value: Any) -> Optional[float]:
    number = _finite_number(value)
    if number is None:
        return None
    return round(number, 4)


def _points_px_to_m(points: Any, px_per_m: float) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = _finite_number(point[0])
        y = _finite_number(point[1])
        if x is None or y is None:
            continue
        result.append((x / px_per_m, y / px_per_m))
    return result


def _points_m(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = _finite_number(point[0])
        y = _finite_number(point[1])
        if x is None or y is None:
            continue
        result.append((x, y))
    return result


def _kb_lookup(kb: pd.DataFrame, code: Optional[float]) -> tuple[str, str, Optional[float], bool]:
    if code is None:
        return "Unmapped item", "", None, True
    rows = kb[kb["item_code"] == code]
    if rows.empty:
        # CSV may store 5.1 as 5.10 or string
        rows = kb[pd.to_numeric(kb["item_code"], errors="coerce").round(4) == round(float(code), 4)]
    if rows.empty:
        return f"Item {code} (not in rate file)", "", None, True
    row = rows.iloc[0]
    return str(row["description"]), str(row["unit"]), float(row["rate_inr"]), False


def parse_takeoff(raw: Any, page_count: Optional[int] = None) -> TakeoffDocument:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("takeoff must be valid JSON") from exc
    else:
        data = raw
    if not isinstance(data, dict):
        raise ValueError("takeoff must be an object")

    units = str(data.get("units") or "px").lower()
    if units not in {"px", "m"}:
        raise ValueError("takeoff units must be 'px' or 'm'")

    elements_raw = data.get("elements")
    if not isinstance(elements_raw, list):
        raise ValueError("takeoff.elements must be a list")
    if len(elements_raw) > MAX_ELEMENTS:
        raise ValueError(f"takeoff supports at most {MAX_ELEMENTS} elements")

    page_scales = data.get("page_scales") or []
    if isinstance(page_scales, dict):
        page_scales = [page_scales.get(str(i), page_scales.get(i)) for i in range(len(page_scales))]
    calibrated_flags = data.get("scale_calibrated") or []

    calibrations: list[Calibration] = []
    if units == "px":
        if not page_scales:
            raise ValueError("2D takeoff requires page_scales")
        for index, scale in enumerate(page_scales):
            px = _finite_positive(scale)
            if px is None:
                raise ValueError(f"page_scales[{index}] must be a positive number")
            flag = False
            if isinstance(calibrated_flags, list) and index < len(calibrated_flags):
                flag = bool(calibrated_flags[index])
            elif isinstance(calibrated_flags, dict):
                flag = bool(calibrated_flags.get(str(index), calibrated_flags.get(index)))
            calibrations.append(
                Calibration(
                    page=index,
                    px_per_m=px,
                    method=str(data.get("scale_method") or "manual"),
                    calibrated=flag,
                )
            )
        if page_count is not None and len(calibrations) != page_count:
            raise ValueError("page_scales must include one value per drawing page")

    def scale_for(page: int) -> float:
        if units == "m":
            return 1.0
        if page < 0 or page >= len(calibrations):
            raise ValueError(f"Element page {page} is invalid")
        return calibrations[page].px_per_m

    elements: list[MeasuredElement] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(elements_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Element {index + 1} is invalid")
        el_type = str(item.get("type") or "").strip().lower()
        if el_type not in ELEMENT_TYPES:
            raise ValueError(f"Element {index + 1} has unknown type '{item.get('type')}'")
        el_id = str(item.get("id") or f"e{index + 1}").strip()[:80]
        if not el_id or el_id in seen_ids:
            raise ValueError(f"Element {index + 1} must have a unique id")
        seen_ids.add(el_id)
        try:
            page = int(item.get("page", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Element {el_id} page is invalid") from exc
        if page < 0:
            raise ValueError(f"Element {el_id} page is invalid")
        if page_count is not None and units == "px" and page >= page_count:
            raise ValueError(f"Element {el_id} references an invalid page")

        px_per_m = scale_for(page)
        raw_points = item.get("points") or item.get("polygon") or item.get("centerline")
        points_m = _points_m(raw_points) if units == "m" else _points_px_to_m(raw_points, px_per_m)

        opening_kind = item.get("opening_kind") or item.get("kind")
        if opening_kind:
            opening_kind = str(opening_kind).lower()
            if opening_kind not in OPENING_KINDS:
                raise ValueError(f"Element {el_id} opening_kind must be door or window")

        host_spaces = item.get("host_space_ids") or []
        if isinstance(host_spaces, str):
            host_spaces = [host_spaces]
        if not isinstance(host_spaces, list):
            host_spaces = []

        element = MeasuredElement(
            id=el_id,
            type=el_type,
            name=str(item.get("name") or el_id).strip()[:80],
            source=str(item.get("source") or ("ifc" if units == "m" and data.get("ifc_units_confirmed") else "2d")),
            page=page,
            thickness_m=_finite_positive(item.get("thickness_m")),
            height_m=_finite_positive(item.get("height_m")),
            length_m=_finite_positive(item.get("length_m")),
            width_m=_finite_positive(item.get("width_m")),
            depth_m=_finite_positive(item.get("depth_m")),
            top_rl=_finite_number(item.get("top_rl")),
            bottom_rl=_finite_number(item.get("bottom_rl")),
            host_id=str(item["host_id"]).strip() if item.get("host_id") else None,
            host_space_ids=[str(s).strip() for s in host_spaces if str(s).strip()],
            shared=_as_bool(item.get("shared")),
            opening_kind=opening_kind,
            item_code=_code(item.get("item_code")),
            ifc_type=str(item["ifc_type"]) if item.get("ifc_type") else None,
            ifc_guid=str(item["ifc_guid"]) if item.get("ifc_guid") else None,
            qto={k: v for k, v in (item.get("qto") or {}).items() if _finite_number(v) is not None},
        )
        if el_type in {"space", "slab", "footing", "excavation", "column"}:
            element.polygon_m = points_m or None
        if el_type in {"wall", "beam"}:
            element.centerline_m = points_m or None
        if el_type == "opening" and points_m:
            element.centerline_m = points_m
        elements.append(element)

    notes = data.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    return TakeoffDocument(
        project_name=str(data.get("project_name") or "").strip()[:120],
        source_filename=str(data.get("source_filename") or "").strip()[:200],
        qs_name=str(data.get("qs_name") or "").strip()[:80],
        qs_signed=bool(data.get("qs_signed")),
        units=units,
        scale_method=str(data.get("scale_method") or ("ifc_units" if units == "m" else "detected")),
        ifc_units_confirmed=bool(data.get("ifc_units_confirmed")),
        calibrations=calibrations,
        elements=elements,
        notes=[str(n) for n in notes if str(n).strip()],
    )


def require_tender_gate(doc: TakeoffDocument) -> None:
    if not doc.elements:
        raise ValueError("Add at least one measured element before generating a tender BOQ.")
    if doc.units == "px":
        if not doc.calibrations:
            raise ValueError("Calibrate drawing scale with two known points before generating.")
        if not all(cal.calibrated for cal in doc.calibrations):
            raise ValueError("Calibrate every page (two-point known length) before generating a tender BOQ.")
        if doc.scale_method not in {"two_point", "manual"} and not all(cal.calibrated for cal in doc.calibrations):
            raise ValueError("2D tender takeoff requires two-point or confirmed manual scale.")
    else:
        if not doc.ifc_units_confirmed:
            raise ValueError("Confirm IFC units (metres) before generating a tender BOQ.")
    if not doc.qs_signed:
        raise ValueError("A quantity surveyor must sign off before generating a tender BOQ.")
    if not doc.qs_name:
        raise ValueError("Enter the QS name before signing off.")


def _wall_length(wall: MeasuredElement) -> Optional[float]:
    if wall.length_m:
        return wall.length_m
    qto_len = wall.qto.get("Length") or wall.qto.get("NetLength") or wall.qto.get("GrossLength")
    if _finite_positive(qto_len):
        return float(qto_len)
    if wall.centerline_m:
        length = polyline_length(wall.centerline_m)
        return length if length > 0 else None
    return None


def _excavation_depth(el: MeasuredElement) -> Optional[float]:
    if el.depth_m:
        return el.depth_m
    if el.top_rl is not None and el.bottom_rl is not None:
        depth = abs(el.top_rl - el.bottom_rl)
        return depth if depth > 0 else None
    return None


def _opening_area(opening: MeasuredElement) -> Optional[float]:
    width = opening.width_m
    height = opening.height_m
    if width and height:
        return width * height
    if opening.qto.get("Area"):
        return float(opening.qto["Area"])
    return None


def _point_to_polyline_distance(point: tuple[float, float], points: list[tuple[float, float]]) -> float:
    if not points:
        return float("inf")
    if len(points) == 1:
        return math.hypot(point[0] - points[0][0], point[1] - points[0][1])
    distances = []
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            distances.append(math.hypot(point[0] - start[0], point[1] - start[1]))
            continue
        position = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
        position = max(0.0, min(1.0, position))
        nearest = (start[0] + position * dx, start[1] + position * dy)
        distances.append(math.hypot(point[0] - nearest[0], point[1] - nearest[1]))
    return min(distances)


def _deduct_openings(wall: MeasuredElement, openings: list[MeasuredElement]) -> tuple[float, float, list[str]]:
    """Return (area_m2 deducted, volume_m3 deducted, notes) with enhanced IS 1200 compliance."""
    area_deduct = 0.0
    volume_deduct = 0.0
    notes = []
    thickness = wall.thickness_m or 0.0
    for opening in openings:
        if opening.host_id != wall.id:
            continue
        if wall.centerline_m and opening.centerline_m:
            distance = _point_to_polyline_distance(opening.centerline_m[0], wall.centerline_m)
            tolerance = max(thickness, 0.15)
            if distance > tolerance:
                notes.append(
                    f"{opening.id} is {distance:.3f} m from host wall (>{tolerance:.3f} m); not deducted"
                )
                continue
        area = _opening_area(opening)
        if area is None:
            notes.append(f"opening {opening.id} missing width/height")
            continue
        # Enhanced IS 1200: Check both area and length thresholds
        width = opening.width_m or 0.0
        height = opening.height_m or 0.0
        max_dimension = max(width, height)
        
        if area <= IS1200_OPENING_DEDUCT_M2:
            notes.append(f"{opening.id} {area:.3f} sqm <= {IS1200_OPENING_DEDUCT_M2} sqm not deducted (IS 1200)")
            continue
        if max_dimension <= IS1200_DEDUCTION_THRESHOLD_LENGTH:
            notes.append(f"{opening.id} max dim {max_dimension:.3f} m <= {IS1200_DEDUCTION_THRESHOLD_LENGTH} m not deducted (IS 1200)")
            continue
        if area <= IS1200_DEDUCTION_THRESHOLD_AREA:
            notes.append(f"{opening.id} {area:.3f} sqm <= {IS1200_DEDUCTION_THRESHOLD_AREA} sqm not deducted (IS 1200 small opening)")
            continue
            
        area_deduct += area
        volume_deduct += area * thickness
        notes.append(f"deduct {opening.id} {area:.3f} sqm (IS 1200 compliant)")
    return area_deduct, volume_deduct, notes


def _wall_key(wall: MeasuredElement) -> Optional[tuple]:
    pts = wall.centerline_m
    if not pts or len(pts) < 2:
        return None
    a, b = pts[0], pts[-1]
    pair = (round(a[0], 2), round(a[1], 2), round(b[0], 2), round(b[1], 2))
    reversed_pair = (pair[2], pair[3], pair[0], pair[1])
    canonical = pair if pair <= reversed_pair else reversed_pair
    thickness = round(wall.thickness_m or 0, 3)
    return canonical + (thickness,)


def measure_takeoff(doc: TakeoffDocument, kb: pd.DataFrame) -> tuple[list[MeasurementLine], list[Deficiency]]:
    lines: list[MeasurementLine] = []
    deficiencies: list[Deficiency] = []
    seq = 1

    def add_line(
        element: MeasuredElement,
        code_key: str,
        quantity: float,
        unit_fallback: str,
        note: str,
        length: Optional[float] = None,
        breadth: Optional[float] = None,
        depth: Optional[float] = None,
        override_code: Optional[float] = None,
        gross_quantity: Optional[float] = None,
        wastage_percent: Optional[float] = None,
    ) -> None:
        nonlocal seq
        code = override_code if override_code is not None else DEFAULT_ITEM_CODES.get(code_key)
        description, unit, _rate, provisional = _kb_lookup(kb, code)
        if not unit:
            unit = unit_fallback
        if quantity <= 0:
            return
        lines.append(
            MeasurementLine(
                ref=f"M{seq:04d}",
                element_id=element.id,
                item_code=code,
                description=description,
                unit=unit,
                length=round(length, 3) if length is not None else None,
                breadth=round(breadth, 3) if breadth is not None else None,
                depth=round(depth, 3) if depth is not None else None,
                quantity=round(quantity, 3),
                gross_quantity=round(gross_quantity, 3) if gross_quantity is not None else None,
                wastage_quantity=(
                    round(quantity - gross_quantity, 3)
                    if gross_quantity is not None and wastage_percent is not None
                    else None
                ),
                wastage_percent=wastage_percent,
                is1200_note=note,
                provisional=provisional,
                source=element.source,
                location=element.name or element.id,
            )
        )
        seq += 1

    by_id = {el.id: el for el in doc.elements}
    walls = [el for el in doc.elements if el.type == "wall"]
    openings = [el for el in doc.elements if el.type == "opening"]
    spaces = [el for el in doc.elements if el.type == "space"]
    slabs = [el for el in doc.elements if el.type == "slab"]
    beams = [el for el in doc.elements if el.type == "beam"]
    columns = [el for el in doc.elements if el.type == "column"]
    footings = [el for el in doc.elements if el.type == "footing"]
    excavations = [el for el in doc.elements if el.type == "excavation"]

    skip_walls: set[str] = set()
    seen_keys: dict[tuple, str] = {}
    for wall in walls:
        key = _wall_key(wall)
        if key and key in seen_keys:
            skip_walls.add(wall.id)
            original = by_id[seen_keys[key]]
            original.shared = True
            deficiencies.append(
                Deficiency(wall.id, f"Duplicate of wall {original.id}; brickwork measured once (shared wall).")
            )
            continue
        if key:
            seen_keys[key] = wall.id

    for space in spaces:
        area = space.qto.get("GrossFloorArea") or space.qto.get("NetFloorArea")
        if not _finite_positive(area) and space.polygon_m:
            area = polygon_area(space.polygon_m)
        if not _finite_positive(area):
            deficiencies.append(Deficiency(space.id, "Space has no polygon or floor area."))
            continue
        add_line(
            space,
            "flooring",
            float(area),
            "sqm",
            "IS 1200 flooring: net space polygon area",
            length=None,
            breadth=None,
            depth=None,
            override_code=space.item_code or DEFAULT_ITEM_CODES["flooring"],
        )

    for wall in walls:
        if wall.id in skip_walls:
            continue
        length = _wall_length(wall)
        height = wall.height_m or wall.qto.get("Height")
        thickness = wall.thickness_m or wall.qto.get("Width")
        if not _finite_positive(length):
            deficiencies.append(Deficiency(wall.id, "Wall length missing (centerline, length_m, or Qto Length)."))
            continue
        if not _finite_positive(height):
            deficiencies.append(Deficiency(wall.id, "Wall height missing — not defaulted."))
            continue
        if not _finite_positive(thickness):
            deficiencies.append(Deficiency(wall.id, "Wall thickness missing — not defaulted."))
            continue
        length = float(length)
        height = float(height)
        thickness = float(thickness)
        gross_area = length * height
        gross_vol = length * height * thickness
        area_deduct, vol_deduct, deduct_notes = _deduct_openings(wall, openings)
        net_area = max(0.0, gross_area - area_deduct)
        net_vol = max(0.0, gross_vol - vol_deduct)
        note = "; ".join(deduct_notes) if deduct_notes else "no opening deductions"
        if wall.shared:
            note = "shared wall measured once. " + note
        # Apply IS 1200 wastage factor for brickwork
        brickwork_vol_with_wastage = net_vol * (1 + IS1200_BRICKWORK_WASTAGE_PERCENT / 100)
        add_line(
            wall,
            "wall",
            brickwork_vol_with_wastage,
            "cum",
            f"IS 1200 brickwork LxHxT with {IS1200_BRICKWORK_WASTAGE_PERCENT}% wastage; {note}",
            length=length,
            breadth=thickness,
            depth=height,
            override_code=wall.item_code or DEFAULT_ITEM_CODES["wall"],
            gross_quantity=net_vol,
            wastage_percent=IS1200_BRICKWORK_WASTAGE_PERCENT,
        )

        hosts = [sid for sid in wall.host_space_ids if sid in by_id]
        internal_faces = 2 if wall.shared or len(hosts) >= 2 else (1 if hosts else 1)
        external_faces = 0 if wall.shared or len(hosts) >= 2 else 1
        # IS 1200 plaster measured in sqm with thickness specification
        add_line(
            wall,
            "plaster_internal",
            net_area * internal_faces,
            "sqm",
            f"IS 1200 internal plaster {IS1200_PLASTER_THICKNESS_INTERNAL*1000:.0f}mm, {internal_faces} face(s); {note}",
            length=length,
            breadth=height,
            depth=IS1200_PLASTER_THICKNESS_INTERNAL,
            override_code=DEFAULT_ITEM_CODES["plaster_internal"],
        )
        add_line(
            wall,
            "paint_internal",
            net_area * internal_faces,
            "sqm",
            f"Internal paint on plastered faces; {note}",
            length=length,
            breadth=height,
            override_code=DEFAULT_ITEM_CODES["paint_internal"],
        )
        if external_faces:
            add_line(
                wall,
                "plaster_external",
                net_area * external_faces,
                "sqm",
                f"IS 1200 external plaster {IS1200_PLASTER_THICKNESS_EXTERNAL*1000:.0f}mm; {note}",
                length=length,
                breadth=height,
                depth=IS1200_PLASTER_THICKNESS_EXTERNAL,
                override_code=DEFAULT_ITEM_CODES["plaster_external"],
            )
            add_line(
                wall,
                "paint_external",
                net_area * external_faces,
                "sqm",
                f"External paint; {note}",
                length=length,
                breadth=height,
                override_code=DEFAULT_ITEM_CODES["paint_external"],
            )

    for opening in openings:
        area = _opening_area(opening)
        kind = opening.opening_kind or "door"
        if kind == "door":
            add_line(
                opening,
                "door",
                1.0,
                "each",
                "Door counted from opening schedule (not guessed)",
                length=opening.width_m,
                breadth=opening.height_m,
                override_code=opening.item_code or DEFAULT_ITEM_CODES["door"],
            )
        else:
            if not _finite_positive(area):
                deficiencies.append(Deficiency(opening.id, "Window missing width × height."))
                continue
            add_line(
                opening,
                "window",
                float(area),
                "sqm",
                "Window area from opening schedule",
                length=opening.width_m,
                breadth=opening.height_m,
                override_code=opening.item_code or DEFAULT_ITEM_CODES["window"],
            )
        if opening.host_id and opening.host_id not in by_id:
            deficiencies.append(Deficiency(opening.id, f"Host wall {opening.host_id} was not found."))
        if not opening.host_id:
            deficiencies.append(Deficiency(opening.id, "Opening has no host wall; masonry deductions were skipped."))

    footing_volume = 0.0
    for footing in footings:
        depth = footing.depth_m or footing.height_m or footing.qto.get("Height") or footing.qto.get("Depth")
        area = footing.qto.get("GrossArea")
        volume = footing.qto.get("GrossVolume") or footing.qto.get("NetVolume")
        if not _finite_positive(volume):
            if not _finite_positive(area) and footing.polygon_m:
                area = polygon_area(footing.polygon_m)
            if _finite_positive(area) and _finite_positive(depth):
                volume = float(area) * float(depth)
            elif footing.width_m and footing.length_m and _finite_positive(depth):
                volume = footing.width_m * footing.length_m * float(depth)
                area = footing.width_m * footing.length_m
        if not _finite_positive(volume):
            deficiencies.append(
                Deficiency(footing.id, "Footing volume missing (need plan × depth or IFC GrossVolume).")
            )
            continue
        volume = float(volume)
        footing_volume += volume
        # Apply IS 1200 wastage factor for concrete
        concrete_vol_with_wastage = volume * (1 + IS1200_CONCRETE_WASTAGE_PERCENT / 100)
        add_line(
            footing,
            "footing",
            concrete_vol_with_wastage,
            "cum",
            f"RCC footing LxBxD with {IS1200_CONCRETE_WASTAGE_PERCENT}% wastage (IS 1200)",
            length=footing.length_m,
            breadth=footing.width_m,
            depth=_finite_positive(depth),
            override_code=footing.item_code or DEFAULT_ITEM_CODES["footing"],
            gross_quantity=volume,
            wastage_percent=IS1200_CONCRETE_WASTAGE_PERCENT,
        )

    for column in columns:
        height = column.height_m or column.length_m or column.qto.get("Length") or column.qto.get("Height")
        volume = column.qto.get("GrossVolume") or column.qto.get("NetVolume")
        area = None
        if column.polygon_m:
            area = polygon_area(column.polygon_m)
        elif column.width_m and column.thickness_m:
            area = column.width_m * column.thickness_m
        elif column.width_m and column.length_m and not column.centerline_m:
            area = column.width_m * column.length_m
        if not _finite_positive(volume):
            if _finite_positive(area) and _finite_positive(height):
                volume = float(area) * float(height)
        if not _finite_positive(volume):
            deficiencies.append(
                Deficiency(column.id, "Column volume missing (need section × height or IFC GrossVolume).")
            )
            continue
        # Apply IS 1200 wastage factor for concrete
        concrete_vol_with_wastage = float(volume) * (1 + IS1200_CONCRETE_WASTAGE_PERCENT / 100)
        add_line(
            column,
            "column",
            concrete_vol_with_wastage,
            "cum",
            f"RCC column with {IS1200_CONCRETE_WASTAGE_PERCENT}% wastage (IS 1200)",
            length=column.width_m,
            breadth=column.thickness_m or column.length_m,
            depth=_finite_positive(height),
            override_code=column.item_code or DEFAULT_ITEM_CODES["column"],
            gross_quantity=float(volume),
            wastage_percent=IS1200_CONCRETE_WASTAGE_PERCENT,
        )

    for beam in beams:
        length = beam.length_m or (_finite_positive(polyline_length(beam.centerline_m or [])) if beam.centerline_m else None)
        if not length:
            length = beam.qto.get("Length")
        width = beam.width_m or beam.thickness_m or beam.qto.get("Width")
        depth = beam.depth_m or beam.height_m or beam.qto.get("Depth") or beam.qto.get("Height")
        volume = beam.qto.get("GrossVolume") or beam.qto.get("NetVolume")
        if not _finite_positive(volume):
            if _finite_positive(length) and _finite_positive(width) and _finite_positive(depth):
                volume = float(length) * float(width) * float(depth)
        if not _finite_positive(volume):
            deficiencies.append(
                Deficiency(beam.id, "Beam volume missing (need L×B×D or IFC GrossVolume).")
            )
            continue
        # Apply IS 1200 wastage factor for concrete
        concrete_vol_with_wastage = float(volume) * (1 + IS1200_CONCRETE_WASTAGE_PERCENT / 100)
        add_line(
            beam,
            "beam",
            concrete_vol_with_wastage,
            "cum",
            f"RCC beam LxBxD with {IS1200_CONCRETE_WASTAGE_PERCENT}% wastage (IS 1200)",
            length=_finite_positive(length),
            breadth=_finite_positive(width),
            depth=_finite_positive(depth),
            override_code=beam.item_code or DEFAULT_ITEM_CODES["beam"],
            gross_quantity=float(volume),
            wastage_percent=IS1200_CONCRETE_WASTAGE_PERCENT,
        )

    for slab in slabs:
        depth = slab.depth_m or slab.thickness_m or slab.qto.get("Depth") or slab.qto.get("Width")
        area = slab.qto.get("GrossArea") or slab.qto.get("NetArea")
        volume = slab.qto.get("GrossVolume") or slab.qto.get("NetVolume")
        if not _finite_positive(area) and slab.polygon_m:
            area = polygon_area(slab.polygon_m)
        if not _finite_positive(volume):
            if _finite_positive(area) and _finite_positive(depth):
                volume = float(area) * float(depth)
        if not _finite_positive(volume):
            deficiencies.append(Deficiency(slab.id, "Slab volume missing (need plan area × depth)."))
            continue
        # Apply IS 1200 wastage factor for concrete
        concrete_vol_with_wastage = float(volume) * (1 + IS1200_CONCRETE_WASTAGE_PERCENT / 100)
        add_line(
            slab,
            "slab",
            concrete_vol_with_wastage,
            "cum",
            f"RCC slab area x depth with {IS1200_CONCRETE_WASTAGE_PERCENT}% wastage (IS 1200)",
            length=None,
            breadth=_finite_positive(area),
            depth=_finite_positive(depth),
            override_code=slab.item_code or DEFAULT_ITEM_CODES["slab"],
            gross_quantity=float(volume),
            wastage_percent=IS1200_CONCRETE_WASTAGE_PERCENT,
        )

    excavation_volume = 0.0
    for excav in excavations:
        depth = _excavation_depth(excav)
        area = excav.qto.get("GrossArea")
        if not _finite_positive(area) and excav.polygon_m:
            area = polygon_area(excav.polygon_m)
        if not _finite_positive(area) and excav.length_m and excav.width_m:
            area = excav.length_m * excav.width_m
        if not _finite_positive(depth):
            deficiencies.append(
                Deficiency(
                    excav.id,
                    "Excavation depth missing (enter depth or top/bottom RL). Floor-plan height is not used.",
                )
            )
            continue
        if not _finite_positive(area):
            deficiencies.append(Deficiency(excav.id, "Excavation plan area missing."))
            continue
        volume = float(area) * float(depth)
        excavation_volume += volume
        add_line(
            excav,
            "excavation",
            volume,
            "cum",
            "Earthwork plan area x depth/RL difference (IS 1200); sections required for tender",
            length=excav.length_m,
            breadth=excav.width_m,
            depth=float(depth),
            override_code=excav.item_code or DEFAULT_ITEM_CODES["excavation"],
        )

    if excavation_volume > 0:
        backfill = max(0.0, excavation_volume - footing_volume)
        host = excavations[0]
        add_line(
            host,
            "backfill",
            backfill,
            "cum",
            "Backfill = excavation - footing volume"
            + (f" ({footing_volume:.3f} cum structure)" if footing_volume else " (no footings measured)"),
            depth=None,
            override_code=DEFAULT_ITEM_CODES["backfill"],
        )

    return lines, deficiencies


def validate_qa_requirements(doc: TakeoffDocument) -> tuple[bool, list[str]]:
    """Validate quality assurance requirements for tender-grade work."""
    issues = []
    
    # Check QS sign-off
    if not doc.qs_signed:
        issues.append("QS sign-off is required for tender-grade work")
    
    # Check scale calibration for 2D drawings
    if doc.units == "px":
        if not doc.calibrations:
            issues.append("Scale calibration required for 2D drawings")
        elif not all(cal.calibrated for cal in doc.calibrations):
            issues.append("All pages must be calibrated with two-point scale")
    
    # Check IFC units confirmation
    if doc.units == "m" and not doc.ifc_units_confirmed:
        issues.append("IFC units must be confirmed for tender-grade work")
    
    # Check measurement standard
    if not doc.measurement_standard:
        issues.append("Measurement standard must be specified")
    
    # Check for deficiencies
    # This would be called after measure_takeoff to check for actual deficiencies
    
    return len(issues) == 0, issues


def record_qa_review(doc: TakeoffDocument, reviewer: str, status: str, comments: list[str] = None) -> TakeoffDocument:
    """Record a QA review in the document history."""
    from datetime import datetime
    
    if comments is None:
        comments = []
    
    previous_status = doc.qa_status

    # Update current QA status
    doc.qa_status = status
    doc.qa_reviewer = reviewer
    doc.qa_review_date = datetime.now().isoformat()
    doc.qa_comments = comments
    
    # Add to revision history
    revision = {
        "timestamp": datetime.now().isoformat(),
        "reviewer": reviewer,
        "status": status,
        "comments": comments,
        "qa_status_before": previous_status,
    }
    doc.revision_history.append(revision)
    
    return doc


def abstract_from_lines(lines: list[MeasurementLine], kb: pd.DataFrame) -> list[dict[str, Any]]:
    grouped: dict[Optional[float], dict[str, Any]] = {}
    for line in lines:
        bucket = grouped.setdefault(
            line.item_code,
            {
                "item_code": line.item_code,
                "description": line.description,
                "unit": line.unit,
                "quantity": 0.0,
                "provisional": False,
            },
        )
        bucket["quantity"] = round(bucket["quantity"] + line.quantity, 3)
        bucket["provisional"] = bucket["provisional"] or line.provisional
        if line.description:
            bucket["description"] = line.description
        if line.unit:
            bucket["unit"] = line.unit
    rows = []
    for code, bucket in grouped.items():
        _desc, _unit, rate, provisional = _kb_lookup(kb, code)
        desc = bucket["description"] or _desc
        unit = bucket["unit"] or _unit
        qty = bucket["quantity"]
        is_prov = bucket["provisional"] or provisional or rate is None
        amount = round(qty * rate, 2) if rate is not None and not is_prov else None
        rows.append(
            {
                "item_code": code,
                "description": desc,
                "unit": unit,
                "quantity": qty,
                "rate_inr": rate,
                "amount_inr": amount,
                "provisional": is_prov,
            }
        )
    rows.sort(key=lambda r: (r["item_code"] is None, r["item_code"] or 0))
    return rows
