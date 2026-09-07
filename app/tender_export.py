"""Tender Excel: cover, detailed taking-off, abstract, priced BOQ."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app.measurement import abstract_from_lines, measure_takeoff, require_tender_gate
from app.models import Deficiency, MeasurementLine, TakeoffDocument


HEADER_FILL = PatternFill("solid", fgColor="10264A")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TOTAL_FILL = PatternFill("solid", fgColor="EEF3FB")
PROV_FILL = PatternFill("solid", fgColor="FFF4E5")
WARN_FILL = PatternFill("solid", fgColor="FCE8E6")


def _autosize(ws, min_width=12, max_width=48):
    for column in ws.columns:
        letter = get_column_letter(column[0].column)
        length = 0
        for cell in column:
            if cell.value is None:
                continue
            length = max(length, min(len(str(cell.value)), max_width))
        ws.column_dimensions[letter].width = max(min_width, length + 2)


def _header_row(ws, headers, row=1):
    for col, title in enumerate(headers, 1):
        cell = ws.cell(row, col, title)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True)


def write_tender_workbook(
    output_path: Path,
    doc: TakeoffDocument,
    kb: pd.DataFrame,
    *,
    enforce_signoff: bool = True,
) -> Path:
    if enforce_signoff:
        require_tender_gate(doc)
    lines, deficiencies = measure_takeoff(doc, kb)
    abstract = abstract_from_lines(lines, kb)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    _write_cover(wb.active, doc, lines, abstract, deficiencies)
    _write_summary(wb.create_sheet("Executive Summary"), abstract, lines, deficiencies, kb)
    _write_detailed(wb.create_sheet("Detailed measurements"), lines)
    _write_abstract(wb.create_sheet("Abstract of quantities"), abstract)
    _write_priced(wb.create_sheet("Priced BOQ"), abstract)
    _write_category_summary(wb.create_sheet("Category Summary"), abstract, kb)
    _write_deficiencies(wb.create_sheet("Deficiencies"), deficiencies)
    _write_measurement_notes(wb.create_sheet("Measurement Notes"), lines)
    wb.save(output_path)
    return output_path


def _write_summary(ws, abstract: list[dict[str, Any]], lines, deficiencies: list[Deficiency], kb: pd.DataFrame):
    """Executive summary with key metrics and cost breakdown."""
    ws.title = "Executive Summary"
    ws["A1"] = "EXECUTIVE SUMMARY"
    ws["A1"].font = Font(size=16, bold=True, color="10264A")
    
    # Calculate key metrics
    total_items = len(abstract)
    priced_items = sum(1 for item in abstract if not item["provisional"] and item["amount_inr"] is not None)
    provisional_items = total_items - priced_items
    total_cost = sum(item["amount_inr"] or 0 for item in abstract)
    provisional_cost = sum(item["amount_inr"] or 0 for item in abstract if item["provisional"])
    confirmed_cost = total_cost - provisional_cost
    
    measurement_lines = len(lines)
    deficiency_count = len(deficiencies)
    
    # Category breakdown
    category_costs = {}
    for item in abstract:
        if item["amount_inr"] is not None:
            kb_row = kb[kb["item_code"] == item["item_code"]]
            if not kb_row.empty:
                category = kb_row.iloc[0].get("category", "Uncategorized")
                category_costs[category] = category_costs.get(category, 0) + item["amount_inr"]
    
    # Write summary metrics
    summary_data = [
        ("Total Item Codes", total_items),
        ("Priced Items", priced_items),
        ("Provisional Items", provisional_items),
        ("Measurement Lines", measurement_lines),
        ("Deficiencies Recorded", deficiency_count),
        ("Total Estimated Cost (INR)", round(total_cost, 2)),
        ("Confirmed Cost (INR)", round(confirmed_cost, 2)),
        ("Provisional Cost (INR)", round(provisional_cost, 2)),
    ]
    
    for index, (label, value) in enumerate(summary_data, 3):
        ws.cell(index, 1, label).font = Font(bold=True)
        ws.cell(index, 2, value)
    
    # Category breakdown
    if category_costs:
        start_row = len(summary_data) + 5
        ws.cell(start_row, 1, "Cost by Category").font = Font(bold=True, size=12)
        ws.cell(start_row + 1, 1, "Category").font = Font(bold=True)
        ws.cell(start_row + 1, 2, "Amount (INR)").font = Font(bold=True)
        ws.cell(start_row + 1, 3, "% of Total").font = Font(bold=True)
        
        for idx, (category, cost) in enumerate(sorted(category_costs.items(), key=lambda x: x[1], reverse=True), start_row + 2):
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            ws.cell(idx, 1, category)
            ws.cell(idx, 2, round(cost, 2))
            ws.cell(idx, 3, f"{percentage:.1f}%")
    
    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 20
    ws.column_dimensions["C"].width = 15


def _write_cover(ws, doc: TakeoffDocument, lines, abstract, deficiencies: list[Deficiency]):
    ws.title = "Cover"
    ws["A1"] = "TENDER BILL OF QUANTITIES"
    ws["A1"].font = Font(size=16, bold=True, color="10264A")
    status = "Signed" if doc.qs_signed else "Draft"
    priced_total = sum(row["amount_inr"] or 0 for row in abstract)
    rows = [
        ("Project", doc.project_name or "—"),
        ("Source file", doc.source_filename or "—"),
        ("Measurement standard", "IS 1200 (building works) mapped to project CPWD-style item codes"),
        ("Scale method", doc.scale_method),
        ("Units", "metres (IFC)" if doc.units == "m" else "pixels converted with per-page two-point / confirmed scale"),
        ("IFC units confirmed", "Yes" if doc.ifc_units_confirmed else "No / not applicable"),
        ("QS name", doc.qs_name or "—"),
        ("QS sign-off", status),
        ("Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
        ("Measurement lines", len(lines)),
        ("Priced total (INR)", round(priced_total, 2)),
        ("Deficiencies", len(deficiencies)),
    ]
    for index, (label, value) in enumerate(rows, 3):
        ws.cell(index, 1, label).font = Font(bold=True)
        ws.cell(index, 2, value)

    ws.cell(16, 1, "Assumptions and limits").font = Font(bold=True, size=12)
    assumptions = [
        "Quantities come only from traced or IFC MeasuredElements. Auto-detected rectangles are an assist, not a tender source.",
        "Shared walls are measured once for brickwork. Internal plaster uses one or two faces from host spaces / shared flag.",
        "Openings larger than 0.1 sqm are deducted from masonry and plaster (IS 1200). Smaller openings are not deducted.",
        "Wall height and thickness, slab/beam/column sizes, and excavation depth are never invented. Missing values appear on Deficiencies.",
        "Earthwork requires an excavation plan and depth or reduced levels — not a furniture floor plan.",
        "Rates are from the project knowledge_base CSV, not a live official CPWD DSR.",
        "This workbook assists a quantity surveyor. The signed QS remains the tender authority.",
        "MEP (electrical/plumbing points) is not auto-measured and must be entered as provisional/manual items.",
    ]
    assumptions.extend(doc.notes)
    for index, text in enumerate(assumptions, 17):
        ws.cell(index, 1, "•")
        ws.cell(index, 2, text)
        ws.merge_cells(start_row=index, start_column=2, end_row=index, end_column=6)

    if doc.calibrations:
        start = 17 + len(assumptions) + 2
        ws.cell(start, 1, "Page scale (px per metre)").font = Font(bold=True)
        _header_row(ws, ["page", "px_per_m", "method", "calibrated"], start + 1)
        for offset, cal in enumerate(doc.calibrations):
            ws.cell(start + 2 + offset, 1, cal.page + 1)
            ws.cell(start + 2 + offset, 2, round(cal.px_per_m, 4))
            ws.cell(start + 2 + offset, 3, cal.method)
            ws.cell(start + 2 + offset, 4, "Yes" if cal.calibrated else "No")
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 88


def _write_detailed(ws, lines: list[MeasurementLine]):
    headers = [
        "ref",
        "element_id",
        "location",
        "item_code",
        "description",
        "L",
        "B",
        "D",
        "qty",
        "unit",
        "IS1200_note",
        "provisional",
        "source",
    ]
    _header_row(ws, headers)
    for row_index, line in enumerate(lines, 2):
        values = [
            line.ref,
            line.element_id,
            line.location,
            line.item_code,
            line.description,
            line.length,
            line.breadth,
            line.depth,
            line.quantity,
            line.unit,
            line.is1200_note,
            "Yes" if line.provisional else "No",
            line.source,
        ]
        for col, value in enumerate(values, 1):
            cell = ws.cell(row_index, col, value)
            if line.provisional:
                cell.fill = PROV_FILL
    if not lines:
        ws.cell(2, 1, "No measurement lines. Check Deficiencies.")
        ws["A2"].fill = WARN_FILL
    _autosize(ws)


def _write_abstract(ws, abstract: list[dict[str, Any]]):
    headers = ["sr_no", "item_code", "description", "unit", "quantity", "provisional"]
    _header_row(ws, headers)
    for row_index, row in enumerate(abstract, 2):
        values = [
            row_index - 1,
            row["item_code"],
            row["description"],
            row["unit"],
            row["quantity"],
            "Yes" if row["provisional"] else "No",
        ]
        for col, value in enumerate(values, 1):
            cell = ws.cell(row_index, col, value)
            if row["provisional"]:
                cell.fill = PROV_FILL
    _autosize(ws)


def _write_priced(ws, abstract: list[dict[str, Any]]):
    headers = ["sr_no", "item_code", "description", "unit", "quantity", "rate_inr", "amount_inr", "status"]
    _header_row(ws, headers)
    total = 0.0
    for row_index, row in enumerate(abstract, 2):
        status = "Provisional" if row["provisional"] or row["rate_inr"] is None else "Priced"
        amount = row["amount_inr"]
        if amount is not None:
            total += amount
        values = [
            row_index - 1,
            row["item_code"],
            row["description"],
            row["unit"],
            row["quantity"],
            row["rate_inr"],
            amount,
            status,
        ]
        for col, value in enumerate(values, 1):
            cell = ws.cell(row_index, col, value)
            if status != "Priced":
                cell.fill = PROV_FILL
    total_row = len(abstract) + 2
    ws.cell(total_row, 5, "TOTAL (priced items only)")
    ws.cell(total_row, 7, round(total, 2))
    for col in range(1, 9):
        ws.cell(total_row, col).fill = TOTAL_FILL
        ws.cell(total_row, col).font = Font(bold=True)
    _autosize(ws)


def _write_deficiencies(ws, deficiencies: list[Deficiency]):
    _header_row(ws, ["element_id", "message"])
    if not deficiencies:
        ws.cell(2, 1, "—")
        ws.cell(2, 2, "No deficiencies recorded. QS must still review taking-off sheets.")
        return
    for row_index, item in enumerate(deficiencies, 2):
        ws.cell(row_index, 1, item.element_id).fill = WARN_FILL
        ws.cell(row_index, 2, item.message).fill = WARN_FILL
    _autosize(ws)


def _write_category_summary(ws, abstract: list[dict[str, Any]], kb: pd.DataFrame):
    """Summary of quantities and costs by work category."""
    ws.title = "Category Summary"
    headers = ["category", "item_count", "total_quantity", "total_amount_inr", "avg_rate", "provisional_count"]
    _header_row(ws, headers)
    
    # Group by category
    category_data = {}
    for item in abstract:
        kb_row = kb[kb["item_code"] == item["item_code"]]
        if kb_row.empty:
            category = "Uncategorized"
        else:
            category = kb_row.iloc[0].get("category", "Uncategorized")
        
        if category not in category_data:
            category_data[category] = {
                "item_count": 0,
                "total_quantity": 0.0,
                "total_amount": 0.0,
                "provisional_count": 0
            }
        
        category_data[category]["item_count"] += 1
        category_data[category]["total_quantity"] += item["quantity"]
        if item["amount_inr"] is not None:
            category_data[category]["total_amount"] += item["amount_inr"]
        if item["provisional"]:
            category_data[category]["provisional_count"] += 1
    
    # Write data
    for row_index, (category, data) in enumerate(sorted(category_data.items()), 2):
        avg_rate = data["total_amount"] / data["total_quantity"] if data["total_quantity"] > 0 else 0
        values = [
            category,
            data["item_count"],
            round(data["total_quantity"], 3),
            round(data["total_amount"], 2),
            round(avg_rate, 2),
            data["provisional_count"]
        ]
        for col, value in enumerate(values, 1):
            ws.cell(row_index, col, value)
    
    _autosize(ws)


def _write_measurement_notes(ws, lines: list[MeasurementLine]):
    """Detailed measurement notes and IS 1200 compliance information."""
    ws.title = "Measurement Notes"
    headers = ["ref", "element_id", "location", "is1200_note", "quantity", "unit", "source"]
    _header_row(ws, headers)
    
    for row_index, line in enumerate(lines, 2):
        values = [
            line.ref,
            line.element_id,
            line.location,
            line.is1200_note,
            line.quantity,
            line.unit,
            line.source
        ]
        for col, value in enumerate(values, 1):
            cell = ws.cell(row_index, col, value)
            if line.provisional:
                cell.fill = PROV_FILL
    
    if not lines:
        ws.cell(2, 1, "No measurement lines available.")
    
    _autosize(ws)


def legacy_single_sheet(df: pd.DataFrame, output_path: Path) -> Path:
    """Keep the original one-sheet demo export."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    return output_path
