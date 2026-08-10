"""
main.py - FastAPI backend exposing a single endpoint to upload a drawing
(PDF or image) and receive back a generated BOQ Excel file.

Run with:
    uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs to try it interactively,
or use the bundled frontend (frontend/index.html).
"""
import os
import shutil
import tempfile
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.extraction import (
    pdf_to_images, extract_text_vector, extract_text_ocr,
    find_dimension_strings, detect_rooms_walls, pixels_to_units,
    estimate_scale_from_text,
)
from app.boq_engine import (
    load_knowledge_base, rooms_to_line_items, dimensions_to_line_items,
    build_boq_dataframe, add_summary_row,
)

app = FastAPI(title="BOQ Automation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "BOQ Automation API is running. See /docs"}


@app.post("/generate-boq")
async def generate_boq(file: UploadFile = File(...), wall_height_m: float = Form(3.0)):
    """
    Upload a PDF drawing. Returns a downloadable Excel BOQ.
    Pipeline: render pages -> OCR/vector text -> detect rooms -> compute quantities -> export.
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    kb = load_knowledge_base()
    all_items = []

    if suffix == ".pdf":
        images = pdf_to_images(tmp_path)
        vector_texts = extract_text_vector(tmp_path)

        for page_img, vtext in zip(images, vector_texts):
            text = vtext if vtext.strip() else extract_text_ocr(page_img)
            px_per_unit = estimate_scale_from_text(text)

            boxes = detect_rooms_walls(page_img)
            if boxes:
                rooms = [pixels_to_units(b, px_per_unit) for b in boxes]
                all_items.extend(rooms_to_line_items(rooms, kb, wall_height_m))
            else:
                dims = find_dimension_strings(text)
                all_items.extend(dimensions_to_line_items(dims, kb))
    else:
        import cv2
        img = cv2.imread(tmp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = extract_text_ocr(img_rgb)
        px_per_unit = estimate_scale_from_text(text)
        boxes = detect_rooms_walls(img_rgb)
        if boxes:
            rooms = [pixels_to_units(b, px_per_unit) for b in boxes]
            all_items.extend(rooms_to_line_items(rooms, kb, wall_height_m))
        else:
            dims = find_dimension_strings(text)
            all_items.extend(dimensions_to_line_items(dims, kb))

    df = build_boq_dataframe(all_items)
    df = add_summary_row(df)

    out_path = os.path.join(OUTPUT_DIR, "generated_boq.xlsx")
    df.to_excel(out_path, index=False)

    os.remove(tmp_path)
    return FileResponse(out_path, filename="generated_boq.xlsx",
                         media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
