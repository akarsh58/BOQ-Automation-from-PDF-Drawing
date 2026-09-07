"""Map IFC products into MeasuredElement records (same store as 2D takeoff)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.models import DEFAULT_ITEM_CODES, MeasuredElement

MAP_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "ifc_cpwd_map.csv"


def load_ifc_map(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = path or MAP_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"IFC mapping file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"ifc_type", "item_code"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"IFC map missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    if "keyword" not in frame.columns:
        frame["keyword"] = ""
    frame["keyword"] = frame["keyword"].fillna("").astype(str).str.lower()
    frame["ifc_type"] = frame["ifc_type"].astype(str)
    frame["item_code"] = pd.to_numeric(frame["item_code"], errors="coerce")
    return frame


def _map_code(ifc_type: str, name: str, mapping: pd.DataFrame) -> Optional[float]:
    name_l = (name or "").lower()
    rows = mapping[mapping["ifc_type"].str.lower() == ifc_type.lower()]
    if rows.empty:
        return None
    with_kw = rows[rows["keyword"].str.len() > 0]
    for _, row in with_kw.iterrows():
        if row["keyword"] and row["keyword"] in name_l:
            return float(row["item_code"]) if pd.notna(row["item_code"]) else None
    blank = rows[rows["keyword"].str.len() == 0]
    if not blank.empty and pd.notna(blank.iloc[0]["item_code"]):
        return float(blank.iloc[0]["item_code"])
    if pd.notna(rows.iloc[0]["item_code"]):
        return float(rows.iloc[0]["item_code"])
    return None


def _qto_numbers(element) -> dict[str, float]:
    try:
        import ifcopenshell.util.element as element_util
    except ImportError:
        return {}
    values: dict[str, float] = {}
    try:
        qtos = element_util.get_psets(element, qtos_only=True)
    except Exception:
        return {}
    for _qto_name, props in (qtos or {}).items():
        if not isinstance(props, dict):
            continue
        for key, value in props.items():
            if key in {"id", "id_"}:
                continue
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            values[str(key)] = number
    return values


def _bbox_xyz(element) -> Optional[tuple[float, float, float]]:
    try:
        import ifcopenshell.geom as geom
    except ImportError:
        return None
    try:
        settings = geom.settings()
        shape = geom.create_shape(settings, element)
        verts = shape.geometry.verts
        if not verts:
            return None
        xs = verts[0::3]
        ys = verts[1::3]
        zs = verts[2::3]
        return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    except Exception:
        return None


def _unit_scale(model) -> float:
    try:
        import ifcopenshell.util.unit as unit_util

        return float(unit_util.calculate_unit_scale(model))
    except Exception:
        return 1.0


def _guid(element) -> str:
    return str(getattr(element, "GlobalId", "") or "")


def _name(element) -> str:
    name = getattr(element, "Name", None) or getattr(element, "LongName", None) or ""
    return str(name)[:80]


def _attr_m(element, names: tuple[str, ...], unit_scale: float) -> Optional[float]:
    for name in names:
        value = getattr(element, name, None)
        if value is None:
            continue
        try:
            number = float(value) * unit_scale
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def import_ifc(path: str | Path, mapping: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    try:
        import ifcopenshell
    except ImportError as exc:
        raise RuntimeError("ifcopenshell is not installed. pip install ifcopenshell") from exc

    model = ifcopenshell.open(str(path))
    unit_scale = _unit_scale(model)
    mapping = mapping if mapping is not None else load_ifc_map()
    elements: list[MeasuredElement] = []
    skipped: list[dict[str, str]] = []
    index = 0

    def add(el_type: str, product, **kwargs):
        nonlocal index
        index += 1
        ifc_type = product.is_a()
        name = _name(product) or f"{ifc_type}_{index}"
        qto = _qto_numbers(product)
        code = _map_code(ifc_type, name, mapping) or kwargs.pop("fallback_code", None)
        element = MeasuredElement(
            id=f"ifc-{_guid(product) or index}",
            type=el_type,
            name=name,
            source="ifc",
            page=0,
            item_code=code,
            ifc_type=ifc_type,
            ifc_guid=_guid(product) or None,
            qto=qto,
            **kwargs,
        )
        elements.append(element)

    type_map = [
        ("IfcWall", "wall", DEFAULT_ITEM_CODES["wall"]),
        ("IfcWallStandardCase", "wall", DEFAULT_ITEM_CODES["wall"]),
        ("IfcSpace", "space", DEFAULT_ITEM_CODES["flooring"]),
        ("IfcSlab", "slab", DEFAULT_ITEM_CODES["slab"]),
        ("IfcColumn", "column", DEFAULT_ITEM_CODES["column"]),
        ("IfcBeam", "beam", DEFAULT_ITEM_CODES["beam"]),
        ("IfcDoor", "opening", DEFAULT_ITEM_CODES["door"]),
        ("IfcWindow", "opening", DEFAULT_ITEM_CODES["window"]),
        ("IfcFooting", "footing", DEFAULT_ITEM_CODES["footing"]),
        ("IfcPile", "footing", DEFAULT_ITEM_CODES["footing"]),
    ]

    seen: set[int] = set()
    for ifc_class, el_type, fallback in type_map:
        try:
            products = model.by_type(ifc_class)
        except Exception:
            continue
        for product in products:
            pid = int(product.id())
            if pid in seen:
                continue
            seen.add(pid)
            extra: dict[str, Any] = {"fallback_code": fallback}
            bbox = _bbox_xyz(product)
            qto = _qto_numbers(product)
            if el_type == "wall":
                extra["length_m"] = qto.get("Length") or (bbox[0] if bbox else None)
                extra["height_m"] = qto.get("Height") or (bbox[2] if bbox else None)
                extra["thickness_m"] = qto.get("Width") or (bbox[1] if bbox else None)
            elif el_type == "opening":
                extra["opening_kind"] = "door" if product.is_a("IfcDoor") else "window"
                extra["width_m"] = _attr_m(product, ("OverallWidth",), unit_scale) or qto.get("Width")
                extra["height_m"] = _attr_m(product, ("OverallHeight",), unit_scale) or qto.get("Height")
            elif el_type == "slab":
                extra["depth_m"] = qto.get("Depth") or qto.get("Width") or (bbox[2] if bbox else None)
            elif el_type == "column":
                extra["height_m"] = qto.get("Length") or qto.get("Height") or (bbox[2] if bbox else None)
                if bbox:
                    extra["width_m"] = bbox[0]
                    extra["thickness_m"] = bbox[1]
            elif el_type == "beam":
                extra["length_m"] = qto.get("Length") or (bbox[0] if bbox else None)
                extra["width_m"] = qto.get("Width") or (bbox[1] if bbox else None)
                extra["depth_m"] = qto.get("Depth") or qto.get("Height") or (bbox[2] if bbox else None)
            elif el_type == "footing":
                extra["depth_m"] = qto.get("Height") or qto.get("Depth") or (bbox[2] if bbox else None)
                if bbox:
                    extra["length_m"] = bbox[0]
                    extra["width_m"] = bbox[1]
            try:
                add(el_type, product, **extra)
            except TypeError:
                skipped.append({"ifc_type": product.is_a(), "reason": "could not map product"})

    # Unknown civil-ish types recorded as skipped, not invented quantities.
    for extra_class in ("IfcRamp", "IfcStair", "IfcRoof", "IfcCovering"):
        try:
            for product in model.by_type(extra_class):
                skipped.append(
                    {
                        "ifc_type": product.is_a(),
                        "name": _name(product),
                        "guid": _guid(product),
                        "reason": "No CPWD mapping in this version; add to ifc_cpwd_map.csv or trace in 2D.",
                    }
                )
        except Exception:
            continue

    payload_elements = []
    for el in elements:
        payload_elements.append(
            {
                "id": el.id,
                "type": el.type,
                "name": el.name,
                "source": "ifc",
                "page": 0,
                "points": el.polygon_m or el.centerline_m or [],
                "thickness_m": el.thickness_m,
                "height_m": el.height_m,
                "length_m": el.length_m,
                "width_m": el.width_m,
                "depth_m": el.depth_m,
                "host_id": el.host_id,
                "opening_kind": el.opening_kind,
                "item_code": el.item_code,
                "ifc_type": el.ifc_type,
                "ifc_guid": el.ifc_guid,
                "qto": el.qto,
            }
        )

    return {
        "units": "m",
        "scale_method": "ifc_units",
        "ifc_units_confirmed": False,
        "unit_scale_to_metres": unit_scale,
        "schema": model.schema,
        "elements": payload_elements,
        "skipped": skipped,
        "element_count": len(payload_elements),
    }
