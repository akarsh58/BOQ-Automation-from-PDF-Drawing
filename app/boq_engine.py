"""
boq_engine.py - Converts extracted geometric/text data into a structured BOQ
using the CPWD-style knowledge base (knowledge_base/cpwd_rates.csv).
"""
import pandas as pd
import os
import math

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "cpwd_rates.csv")


def load_knowledge_base():
    if not os.path.isfile(KB_PATH):
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")
    kb = pd.read_csv(KB_PATH)
    required = {"item_code", "description", "unit", "rate_inr"}
    missing = required.difference(kb.columns)
    if missing:
        raise ValueError(f"Knowledge base is missing columns: {', '.join(sorted(missing))}")
    if kb.empty:
        raise ValueError("Knowledge base has no rates")
    kb = kb.copy()
    kb["rate_inr"] = pd.to_numeric(kb["rate_inr"], errors="coerce")
    if kb["rate_inr"].isna().any() or (kb["rate_inr"] < 0).any():
        raise ValueError("Knowledge base contains invalid rates")
    return kb


def classify_room(box_info, index):
    """
    Placeholder classifier: assigns a generic room label.
    In production, replace with an OCR-label match (e.g., text 'BEDROOM' near box)
    or a trained CV classifier.
    """
    # Reviewed rooms may carry a user-edited name. Keep the historical generic
    # label when the automatic detector did not provide one.
    name = str(box_info.get("name", "")).strip()
    return name[:80] if name else f"Room_{index + 1}"


def rooms_to_line_items(rooms, kb, wall_height_m=3.0):
    """
    Convert detected room boxes (with area/width/height) into BOQ line items:
    flooring, plastering, painting, brickwork walls.
    """
    try:
        wall_height_m = float(wall_height_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("wall_height_m must be a number") from exc
    if not math.isfinite(wall_height_m) or wall_height_m <= 0:
        raise ValueError("wall_height_m must be positive")

    def kb_row(code):
        rows = kb[kb["item_code"] == code]
        if rows.empty:
            raise ValueError(f"Knowledge base item {code} is missing")
        row = rows.iloc[0]
        return row["description"], row["unit"], float(row["rate_inr"])

    items = []
    for i, room in enumerate(rooms):
        try:
            width = float(room["width"])
            height = float(room["height"])
            area = float(room["area"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Room {i + 1} has invalid dimensions") from exc
        if not all(math.isfinite(value) and value >= 0 for value in (width, height, area)):
            raise ValueError(f"Room {i + 1} has invalid dimensions")
        perimeter = 2 * (width + height)
        wall_area = perimeter * wall_height_m
        room_name = classify_room(room, i)

        desc, unit, rate = kb_row(5.1)
        items.append({"room": room_name, "item_code": 5.1, "description": desc,
                      "unit": unit, "quantity": area, "rate_inr": rate,
                      "amount_inr": round(area * rate, 2)})

        desc, unit, rate = kb_row(4.1)
        items.append({"room": room_name, "item_code": 4.1, "description": desc,
                      "unit": unit, "quantity": wall_area, "rate_inr": rate,
                      "amount_inr": round(wall_area * rate, 2)})

        desc, unit, rate = kb_row(6.1)
        items.append({"room": room_name, "item_code": 6.1, "description": desc,
                      "unit": unit, "quantity": wall_area, "rate_inr": rate,
                      "amount_inr": round(wall_area * rate, 2)})

        desc, unit, rate = kb_row(3.1)
        wall_volume = perimeter * wall_height_m * 0.23  # 230mm brick wall thickness
        items.append({"room": room_name, "item_code": 3.1, "description": desc,
                      "unit": unit, "quantity": round(wall_volume, 2), "rate_inr": rate,
                      "amount_inr": round(wall_volume * rate, 2)})

    return items


def dimensions_to_line_items(dim_list, kb):
    """
    Fallback path: when only raw dimension text (e.g. '3000 x 4000 mm') is found
    (no clean room contours), still generate a basic flooring + painting estimate.
    """
    def to_meters(value, unit):
        unit = (unit or "mm").lower()
        factors = {"mm": 0.001, "cm": 0.01, "m": 1, "ft": 0.3048, "feet": 0.3048}
        if unit not in factors:
            raise ValueError(f"Unsupported dimension unit: {unit}")
        return float(value) * factors[unit]

    rows = kb[kb["item_code"] == 5.1]
    if rows.empty:
        raise ValueError("Knowledge base item 5.1 is missing")
    row = rows.iloc[0]
    items = []
    for i, d in enumerate(dim_list):
        try:
            w = to_meters(d["width"], d.get("width_unit", d.get("unit", "mm")))
            h = to_meters(d["height"], d.get("height_unit", d.get("unit", "mm")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Dimension {i + 1} is invalid") from exc
        if not math.isfinite(w) or not math.isfinite(h) or w <= 0 or h <= 0:
            raise ValueError(f"Dimension {i + 1} is invalid")
        area = round(w * h, 2)
        items.append({"room": f"Detected_Dim_{i+1}", "item_code": 5.1,
                      "description": row["description"], "unit": row["unit"],
                      "quantity": area, "rate_inr": float(row["rate_inr"]),
                      "amount_inr": round(area * float(row["rate_inr"]), 2)})
    return items


def build_boq_dataframe(items):
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df = df.sort_values(["room", "item_code"]).reset_index(drop=True)
    df.insert(0, "sr_no", range(1, len(df) + 1))
    return df


def add_summary_row(df):
    if df.empty:
        return df
    total = df["amount_inr"].sum()
    summary = pd.DataFrame([{
        "sr_no": "", "room": "", "item_code": "", "description": "TOTAL ESTIMATED COST",
        "unit": "", "quantity": "", "rate_inr": "", "amount_inr": round(total, 2)
    }])
    return pd.concat([df, summary], ignore_index=True)
