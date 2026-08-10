"""
demo_run.py - Standalone script to test the pipeline WITHOUT running the API server.
Generates a synthetic sample floor plan PDF, runs it through the pipeline,
and writes a BOQ Excel file. Useful to verify your setup works end-to-end.

Usage:
    python demo_run.py
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import fitz  # PyMuPDF
from app.extraction import pdf_to_images, extract_text_vector, extract_text_ocr, detect_rooms_walls, pixels_to_units, estimate_scale_from_text
from app.boq_engine import load_knowledge_base, rooms_to_line_items, build_boq_dataframe, add_summary_row


def create_sample_pdf(path="sample_input/sample_floor_plan.pdf"):
    """Creates a very simple synthetic floor plan PDF with two rectangular rooms."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    page.draw_rect(fitz.Rect(50, 50, 300, 250), color=(0, 0, 0), width=2)   # Room 1
    page.draw_rect(fitz.Rect(320, 50, 550, 250), color=(0, 0, 0), width=2) # Room 2
    page.insert_text((60, 40), "SCALE 1:100", fontsize=10)
    page.insert_text((60, 270), "BEDROOM 3000 x 4000 mm", fontsize=8)
    page.insert_text((330, 270), "LIVING ROOM 4500 x 4000 mm", fontsize=8)
    doc.save(path)
    doc.close()
    return path


def run_pipeline(pdf_path):
    kb = load_knowledge_base()
    images = pdf_to_images(pdf_path)
    vector_texts = extract_text_vector(pdf_path)
    all_items = []

    for page_img, vtext in zip(images, vector_texts):
        text = vtext if vtext.strip() else extract_text_ocr(page_img)
        px_per_unit = estimate_scale_from_text(text, default_px_per_unit=50)
        boxes = detect_rooms_walls(page_img)
        print(f"Detected {len(boxes)} room/wall candidates on page.")
        rooms = [pixels_to_units(b, px_per_unit) for b in boxes]
        all_items.extend(rooms_to_line_items(rooms, kb, wall_height_m=3.0))

    df = build_boq_dataframe(all_items)
    df = add_summary_row(df)
    return df


if __name__ == "__main__":
    pdf_path = create_sample_pdf()
    print(f"Sample drawing created at: {pdf_path}")
    df = run_pipeline(pdf_path)
    out_path = "output/demo_generated_boq.xlsx"
    os.makedirs("output", exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"BOQ generated: {out_path}")
    print(df.to_string(index=False))
