"""
demo_run.py - Standalone script to test the tender takeoff pipeline without the API.

Creates a sample floor plan PDF and a synthetic IS 1200 takeoff (spaces, shared wall,
opening, RCC, excavation), then writes a multi-sheet tender workbook.

Usage:
    python demo_run.py
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))

import fitz

from app.boq_engine import load_knowledge_base
from app.measurement import measure_takeoff, parse_takeoff, require_tender_gate
from app.tender_export import write_tender_workbook


def create_sample_pdf(path="sample_input/sample_floor_plan.pdf"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    page.draw_rect(fitz.Rect(50, 50, 300, 250), color=(0, 0, 0), width=2)
    page.draw_rect(fitz.Rect(320, 50, 550, 250), color=(0, 0, 0), width=2)
    page.insert_text((60, 40), "SCALE 1:100", fontsize=10)
    page.insert_text((60, 270), "BEDROOM 3000 x 4000 mm", fontsize=8)
    page.insert_text((330, 270), "LIVING ROOM 4500 x 4000 mm", fontsize=8)
    doc.save(path)
    doc.close()
    return path


def sample_takeoff():
    """Two rooms (metres), shared party wall, door, RCC members, excavation with depth."""
    return {
        "project_name": "Demo bungalow",
        "source_filename": "sample_floor_plan.pdf",
        "qs_name": "Demo QS",
        "qs_signed": True,
        "units": "m",
        "scale_method": "ifc_units",
        "ifc_units_confirmed": True,
        "elements": [
            {
                "id": "s1",
                "type": "space",
                "name": "Bedroom",
                "page": 0,
                "points": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "item_code": 5.1,
            },
            {
                "id": "s2",
                "type": "space",
                "name": "Living",
                "page": 0,
                "points": [[4, 0], [8.5, 0], [8.5, 3], [4, 3]],
                "item_code": 5.1,
            },
            {
                "id": "w-ext-1",
                "type": "wall",
                "name": "Bedroom north",
                "points": [[0, 0], [4, 0]],
                "thickness_m": 0.23,
                "height_m": 3.0,
                "host_space_ids": ["s1"],
            },
            {
                "id": "w-shared",
                "type": "wall",
                "name": "Party wall",
                "points": [[4, 0], [4, 3]],
                "thickness_m": 0.23,
                "height_m": 3.0,
                "shared": True,
                "host_space_ids": ["s1", "s2"],
            },
            {
                "id": "w-dup",
                "type": "wall",
                "name": "Party wall duplicate",
                "points": [[4, 3], [4, 0]],
                "thickness_m": 0.23,
                "height_m": 3.0,
            },
            {
                "id": "o1",
                "type": "opening",
                "name": "Bedroom door",
                "opening_kind": "door",
                "host_id": "w-ext-1",
                "width_m": 1.0,
                "height_m": 2.1,
            },
            {
                "id": "win1",
                "type": "opening",
                "name": "Living window",
                "opening_kind": "window",
                "host_id": "w-shared",
                "width_m": 1.2,
                "height_m": 1.2,
            },
            {
                "id": "col1",
                "type": "column",
                "name": "C1",
                "width_m": 0.3,
                "thickness_m": 0.3,
                "height_m": 3.0,
            },
            {
                "id": "bm1",
                "type": "beam",
                "name": "B1",
                "points": [[0, 0], [8.5, 0]],
                "width_m": 0.23,
                "depth_m": 0.45,
            },
            {
                "id": "sl1",
                "type": "slab",
                "name": "GF slab",
                "points": [[0, 0], [8.5, 0], [8.5, 3], [0, 3]],
                "depth_m": 0.125,
            },
            {
                "id": "ft1",
                "type": "footing",
                "name": "F1",
                "points": [[0, 0], [1.2, 0], [1.2, 1.2], [0, 1.2]],
                "depth_m": 0.45,
            },
            {
                "id": "ex1",
                "type": "excavation",
                "name": "Foundation excavation",
                "points": [[0, 0], [1.5, 0], [1.5, 1.5], [0, 1.5]],
                "depth_m": 1.2,
            },
            {
                "id": "ex-bad",
                "type": "excavation",
                "name": "Missing depth (should deficiency)",
                "points": [[2, 0], [3, 0], [3, 1], [2, 1]],
            },
        ],
    }


if __name__ == "__main__":
    pdf_path = create_sample_pdf()
    print(f"Sample drawing created at: {pdf_path}")
    kb = load_knowledge_base()
    doc = parse_takeoff(sample_takeoff())
    require_tender_gate(doc)
    lines, deficiencies = measure_takeoff(doc, kb)
    print(f"Measurement lines: {len(lines)}")
    print(f"Deficiencies: {len(deficiencies)}")
    for d in deficiencies:
        print(f"  - {d.element_id}: {d.message}")
    out_path = "output/demo_generated_boq.xlsx"
    os.makedirs("output", exist_ok=True)
    write_tender_workbook(out_path, doc, kb)
    print(f"Tender BOQ generated: {out_path}")
    for line in lines:
        # Replace unicode multiplication sign with 'x' to avoid encoding issues
        note = line.is1200_note[:60].replace('×', 'x')
        print(
            f"{line.ref} {line.item_code} {line.location:20} {line.quantity:8.3f} {line.unit:4} {note}"
        )
