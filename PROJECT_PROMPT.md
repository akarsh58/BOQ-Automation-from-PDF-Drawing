# BOQ Automation Project — Complete Project Prompt & Documentation

> **Project**: BOQ (Bill of Quantities) Automation from PDF/IFC Drawings  
> **Location**: `D:\GitHub Projects\boq_automation_project`  
> **Language**: Python 3.11+  
> **Key Dependencies**: pandas, openpyxl, PyMuPDF (fitz), numpy, Pillow  
> **Measurement Standard**: IS 1200 (Indian Standard)  
> **Rate Source**: CPWD (Central Public Works Department, India)  
> **Git Remote**: https://github.com/akarsh58/BOQ-Automation-from-PDF-Drawing  
> **Branch**: master (HEAD at b3a52e0)  
> **Status**: ✅ All 4 tests passing, demo pipeline operational
## 🔄 COMPLETE DATA FLOW

```
PDF Drawing ──→ [OCR/Extraction] ──→ Room boxes + labels
                                              │
IFC Model ──────→ [ifc_import.py] ──────────────┘
                                              │
                                              ▼
                                   TakeoffDocument (models.py)
                                              │
                                              ▼
                                   measure_takeoff() ──→ MeasurementLine[]
                                              │
                                              ▼
                                   boq_engine.py ──→ Line items with CPWD rates
                                              │
                                              ▼
                                   build_boq_dataframe() + add_summary_row()
                                              │
                                              ▼
                                   write_tender_workbook() ──→ Excel (.xlsx)
```

---

## 🧮 MEASUREMENT LOGIC (IS 1200 STANDARD)

### Wall Quantity Calculation
```
brickwork_gross = wall_length × wall_height × wall_thickness
brickwork_net = brickwork_gross − sum(opening_areas)
brickwork_with_wastage = brickwork_net × (1 + 5%)
plaster_quantity = wall_net_area × 2 faces × (1 + 10% wastage)
```

### Opening Deduction Rules
- Only deduct if opening area ≥ 0.1 sqm (IS 1200 threshold)
- Only deduct if opening is on/near the host wall (centerline within 1.0m vertically)
- Small openings are NOT deducted (documented in `is1200_note`)

### IS 1200 Constants (from `models.py`)
- Default wall height: 3.0m
- Default wall thickness: 0.23m (230mm brick)
- Joint deduction: 2.5%
- Plaster thickness: 12mm (internal), 20mm (external)
- Concrete wastage: 3%
- Brickwork wastage: 5%

---

## 📊 CPWD ITEM CODES USED IN BOQ ENGINE

The `boq_engine.py` generates 4 line items per room:

| Item Code | Description | Unit | Calculation |
|-----------|-------------|------|-------------|
| 5.1 | Vitrified tile flooring | sqm | room_area |
| 4.1 | 12mm cement plaster internal | sqm | wall_net_area × 2 |
| 6.1 | Painting with acrylic emulsion | sqm | wall_net_area × 2 |
| 3.1 | Brickwork in CM 1:6 | cum | perimeter × height × thickness |
---

## 🔑 KEY FUNCTIONS REFERENCE

### `app/measurement.py`

**`measure_takeoff(doc: TakeoffDocument, kb) → (list[MeasurementLine], list[Deficiency])`**  
Main measurement engine. Iterates walls, calculates gross/net quantities, applies opening deductions, generates plaster and brickwork lines.  
**CHECK**: Verify wall length from centerline or `length_m`, opening deductions are correct, wastage percentages are 5% for brickwork and 10% for plaster.

**`parse_takeoff(raw: dict) → TakeoffDocument`**  
Converts raw dict (JSON-like) to `TakeoffDocument`. Maps element type, dimensions, host relationships.  
**CHECK**: Verify all fields from raw dict are mapped to MeasuredElement fields correctly.

**`require_tender_gate(doc: TakeoffDocument) → None`**  
Validates the tender is ready. Raises `ValueError` if: no QS name, QS not signed, unsupported units.  
**CHECK**: Ensure `qs_name`, `qs_signed`, and `units` are properly set before calling.

**`record_qa_review(doc, reviewer, status) → None`**  
Updates QA status and records revision history. Validates status against `QA_STATUSES = {"draft", "review", "approved", "rejected"}`.  
**CHECK**: Verify `revision_history` correctly captures `qa_status_before` and `qa_status_after`.

### `app/boq_engine.py`

**`load_knowledge_base() → pd.DataFrame`**  
Loads `knowledge_base/cpwd_rates.csv`, validates columns (`item_code`, `description`, `unit`, `rate_inr`), converts rates to numeric, validates no negative/missing rates.  
**CHECK**: Verify CSV path resolves correctly from `app/../knowledge_base/cpwd_rates.csv`.

**`rooms_to_line_items(rooms, kb, wall_height_m=3.0, wall_thickness_m=0.23) → list[dict]`**  
Generates 4 BOQ line items per room (flooring, plaster, paint, brickwork). Uses CPWD rates.  
**CHECK**: Verify all 4 items are generated, quantities are correct, amounts round to 2 decimals.

**`classify_room(box_info, index) → str`**  
Returns the room name from `box_info["name"]` or `"Room_{index}"` if unnamed.  
**CHECK**: Verify name is truncated to 80 chars.

**`build_boq_dataframe(items) → pd.DataFrame`**  
Creates sorted DataFrame with `sr_no` column.  
**CHECK**: Verify sorting by `["room", "item_code"]` and `sr_no` starts at 1.

**`add_summary_row(df) → pd.DataFrame`**  
Appends a "TOTAL ESTIMATED COST" summary row at the bottom.  
**CHECK**: Verify `amount_inr` in summary equals sum of all `amount_inr` values.

### `app/extraction.py`

**`assign_room_labels(boxes, labels) → list[dict]`**  
For each box, finds the highest-confidence label whose center point falls inside the box.  
**CHECK**: Verify label center `(x + width/2, y + height/2)` is within box bounds. Higher confidence wins.

### `app/tender_export.py`

**`write_tender_workbook(path, doc, kb, lines=None, deficiencies=None) → str`**  
Creates multi-sheet Excel workbook with openpyxl. Sheets: `Measurements`, `Rate_Schedule`, `QA`, `Deficiencies`.  
**CHECK**: Verify all 4 sheets are present, data aligns correctly, file is created at path.

### `app/ifc_import.py`

**`load_ifc_map(path=None) → pd.DataFrame`**  
Loads `knowledge_base/ifc_cpwd_map.csv`, validates columns.  
**CHECK**: Verify `ifc_type` and `item_code` columns exist.

**`_map_code(ifc_type, name, mapping) → Optional[float]`**  
Maps IFC type to CPWD item code. Priority: keyword match > default mapping for type > first row.  
**CHECK**: Verify keyword matching is case-insensitive and name is matched as substring.
---

## 🧪 TEST SPECIFICATIONS

### Test 1: `test_assigns_label_inside_matching_box`
- Input: 2 boxes, 2 labels (one inside box 0, one far away)
- Expected: Box 0 gets "BEDROOM" (confidence 0.91), Box 1 gets empty (confidence 0.0)

### Test 2: `test_wall_quantity_exposes_wastage_separately`
- Wall: centerline [(0,0),(4,0)], height=3, thickness=0.2
- Expected: gross=2.4, wastage=0.12, quantity=2.52

### Test 3: `test_opening_far_from_host_wall_is_not_deducted`
- Wall: [(0,0),(4,0)], opening at [(2,2)] (far vertically)
- Expected: Opening NOT deducted, note contains "not deducted"

### Test 4: `test_qa_revision_records_previous_status`
- Start with `qa_status="draft"`, call `record_qa_review(doc, "Reviewer", "approved")`
- Expected: `revision_history[0]["qa_status_before"] == "draft"`, `qa_status == "approved"`

---

## 📊 KNOWLEDGE BASE STRUCTURE

### `cpwd_rates.csv` — 82 items, 13 categories
| Category | Item Codes | Examples |
|----------|-----------|----------|
| Earthwork | 1.1–1.5 | Excavation, Backfill, Sand filling |
| Concrete | 2.1–2.10 | PCC, RCC, Shuttering, Reinforcement |
| Masonry | 3.1–3.6 | Brickwork, Partition, AAC block |
| Finishing | 4.1–4.8 | Plaster, Pointing, Painting |
| Flooring | 5.1–5.10 | Tiles, Marble, Granite, Skirting |
| Doors/Windows | 7.1–7.8 | Flush door, Aluminium window, UPVC |
| Electrical | 8.1–8.10 | Wiring points, Switch boards |
| Plumbing | 8.2, 8.8–8.10 | Pipeline, Water supply, Septic |
| Waterproofing | 9.1–9.5 | Treatment, Toilet waterproofing |
| Metal Work | 10.1–10.5 | Railings, Steel fabrication |
| Roads | 11.1–11.4 | Asphalt, Paving blocks |
| Landscaping | 12.1–12.5 | Plants, Lawn, Irrigation |

### `ifc_cpwd_map.csv` — 17 IFC type mappings
| Ifc Type | Default Item Code | Notes |
|----------|------------------|-------|
| IfcWall | 3.1 | Superstructure brickwork |
| IfcSpace | 5.1 | Floor finish |
| IfcSlab | 2.5 | RCC slab |
| IfcColumn | 2.3 | RCC column |
| IfcBeam | 2.4 | RCC beam |
| IfcDoor | 7.1 | Flush door |
| IfcWindow | 7.2 | Aluminium window |
| IfcFooting | 2.2 | RCC footing |
| IfcEarthworksCut | 1.1 | Earthwork excavation |

---

---

## 🔧 WHAT AGENTS SHOULD CHECK/EDIT

### If modifying `app/measurement.py`:
- Run `python -m pytest tests/test_measurement_accuracy.py -v` to verify wall quantities, opening deductions, and QA review still pass
- Check that `IS1200_BRICKWORK_WASTAGE_PERCENT` (5%) and plaster wastage (10%) are consistent
- Verify opening deduction threshold `IS1200_OPENING_DEDUCT_M2` (0.1 sqm) is enforced
- Check `_should_deduct()` validates opening proximity to host wall

### If modifying `app/boq_engine.py`:
- Verify `load_knowledge_base()` path resolves correctly
- Check that `rooms_to_line_items()` generates exactly 4 items per room (codes 5.1, 4.1, 6.1, 3.1)
- Verify `build_boq_dataframe()` inserts `sr_no` starting from 1
- Check `add_summary_row()` total matches sum of `amount_inr`

### If modifying `app/extraction.py`:
- Run `python -m pytest tests/test_extraction_labels.py -v`
- Verify label center calculation: `lcx = lx + lw/2`, `lcy = ly + lh/2`
- Check that higher confidence overrides lower confidence

### If modifying `app/models.py`:
- Verify `DEFAULT_ITEM_CODES` dictionary has all 35 item codes
- Check IS 1200 constants match `measurement.py` imports
- Verify `MeasuredElement` dataclass has all required fields

### If modifying `app/ifc_import.py`:
- Run `python -c "from app.ifc_import import load_ifc_map, _map_code; ..."` to verify mappings
- Check that `_map_code()` handles keyword matching correctly
- Verify `import_ifc()` returns valid element list

### If modifying `app/tender_export.py`:
- Verify `write_tender_workbook()` creates all 4 Excel sheets
- Check that `openpyxl` engine works and file is created
## ⚠️ KNOWN LIMITATIONS & NOTES

1. **`frontend/index.html`** is a binary/compressed file — NOT readable as standard HTML. Do not edit as text. It may be a built/bundled asset.
2. **`requirements.txt`** is MISSING — needs to be created with: `pandas`, `openpyxl`, `PyMuPDF` (fitz), `numpy`, `Pillow`
3. **`ifcopenshell`** is NOT installed — IFC import works for metadata but geometric calculations return empty
4. **`classify_room()`** in `boq_engine.py` is a placeholder — uses `box_info["name"]` directly. Production should use OCR-label matching
5. **Shared walls** are counted twice in the demo (w-shared and w-dup both appear) — this is a known data modeling issue
6. **The `.env` file** contains API keys — never commit this to version control
7. **`ifc_cpwd_map.csv`** has duplicate `ifc_type` entries (IfcWall appears twice) — `_map_code()` handles this correctly but the data could be cleaned