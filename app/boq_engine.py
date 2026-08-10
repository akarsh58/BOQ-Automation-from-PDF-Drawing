"""
boq_engine.py - Converts extracted geometric/text data into a structured BOQ
using the CPWD-style knowledge base (knowledge_base/cpwd_rates.csv).
"""
import pandas as pd
import os

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "cpwd_rates.csv")


def load_knowledge_base():
    return pd.read_csv(KB_PATH)


def classify_room(box_info, index):
    """
    Placeholder classifier: assigns a generic room label.
    In production, replace with an OCR-label match (e.g., text 'BEDROOM' near box)
    or a trained CV classifier.
    """
    return f"Room_{index + 1}"


def rooms_to_line_items(rooms, kb, wall_height_m=3.0):
    """
    Convert detected room boxes (with area/width/height) into BOQ line items:
    flooring, plastering, painting, brickwork walls.
    """
    items = []
    for i, room in enumerate(rooms):
        area = room["area"]
        perimeter = 2 * (room["width"] + room["height"])
        wall_area = perimeter * wall_height_m
        room_name = classify_room(room, i)

        def kb_row(code):
            row = kb[kb["item_code"] == code].iloc[0]
            return row["description"], row["unit"], float(row["rate_inr"])

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
    items = []
    for i, d in enumerate(dim_list):
        w = d["width"] / 1000 if d["unit"] == "mm" else d["width"]
        h = d["height"] / 1000 if d["unit"] == "mm" else d["height"]
        area = round(w * h, 2)
        row = kb[kb["item_code"] == 5.1].iloc[0]
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
