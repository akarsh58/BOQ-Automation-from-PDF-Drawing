"""Write a tender takeoff into a multi-sheet Excel workbook."""
from __future__ import annotations

import os

import pandas as pd

from app.models import Deficiency, MeasurementLine, TakeoffDocument


def _line_rows(lines: list[MeasurementLine]):
    rows = []
    for line in lines:
        rows.append({
            "ref": line.ref,
            "element_id": line.element_id,
            "item_code": line.item_code,
            "description": line.description,
            "unit": line.unit,
            "length": line.length,
            "breadth": line.breadth,
            "depth": line.depth,
            "quantity": line.quantity,
            "gross_quantity": line.gross_quantity,
            "wastage_quantity": line.wastage_quantity,
            "wastage_percent": line.wastage_percent,
            "provisional": line.provisional,
            "source": line.source,
            "location": line.location,
            "is1200_note": line.is1200_note,
        })
    return rows


def write_tender_workbook(path: str, doc: TakeoffDocument, kb, lines=None, deficiencies=None):
    """Write measurement lines, rate schedule, QA and deficiencies to Excel."""
    if lines is None:
        lines = []
    if deficiencies is None:
        deficiencies = []
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        line_frame = pd.DataFrame(_line_rows(lines))
        if not line_frame.empty:
            line_frame.insert(0, "sr_no", range(1, len(line_frame) + 1))
        line_frame.to_excel(writer, sheet_name="Measurements", index=False)

        rate_frame = kb.copy() if kb is not None else pd.DataFrame()
        if not rate_frame.empty:
            rate_frame.to_excel(writer, sheet_name="Rate_Schedule", index=False)

        qa_frame = pd.DataFrame([{
            "project_name": doc.project_name,
            "source_filename": doc.source_filename,
            "qs_name": doc.qs_name,
            "qs_signed": doc.qs_signed,
            "qa_status": doc.qa_status,
            "qa_reviewer": doc.qa_reviewer,
            "qa_review_date": doc.qa_review_date,
            "measurement_standard": doc.measurement_standard,
            "units": doc.units,
            "scale_method": doc.scale_method,
        }])
        qa_frame.to_excel(writer, sheet_name="QA", index=False)

        deficiency_frame = pd.DataFrame([
            {"element_id": d.element_id, "message": d.message}
            for d in deficiencies
        ])
        deficiency_frame.to_excel(writer, sheet_name="Deficiencies", index=False)

    return path
