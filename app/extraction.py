"""
extraction.py - Core module to extract text, dimensions and shapes from PDF/image drawings.
Uses pdfplumber for vector PDFs, PyMuPDF for rendering pages to images,
pytesseract for OCR on scans, and OpenCV for wall/room shape detection.
"""
import re

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image

DIM_PATTERN = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mm|m|ft|feet|cm)?\s*[x×]\s*"
    r"(\d+(?:\.\d+)?)\s*(mm|m|ft|feet|cm)?(?![\w.])",
    re.IGNORECASE,
)
NUM_PATTERN = re.compile(r"\d+(?:\.\d+)?")
MAX_PDF_PAGES = 50
MAX_RENDER_PIXELS = 60_000_000


def pdf_to_images(pdf_path, dpi=200, max_pages=MAX_PDF_PAGES):
    """Render each PDF page to a numpy image array (for OpenCV) using PyMuPDF."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    doc = fitz.open(pdf_path)
    images = []
    try:
        if len(doc) > max_pages:
            raise ValueError(f"PDFs with more than {max_pages} pages are not supported")
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            rect = page.rect
            estimated_pixels = int(rect.width * zoom) * int(rect.height * zoom)
            if estimated_pixels > MAX_RENDER_PIXELS:
                raise ValueError("A PDF page is too large to render safely")
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))
    finally:
        doc.close()
    return images


def extract_text_vector(pdf_path):
    """Extract text directly from vector PDF (fast, accurate for CAD-exported PDFs)."""
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texts.append(t)
    return texts


def extract_text_ocr(image):
    """Run Tesseract OCR on a rendered page image (for scanned drawings)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(gray)


def find_dimension_strings(text):
    """Find patterns like '3000 x 4000 mm' or '10 x 12 ft' in extracted text."""
    matches = DIM_PATTERN.findall(text)
    dims = []
    for m in matches:
        w, wu, h, hu = m
        width_unit = (wu or hu or "mm").lower()
        height_unit = (hu or wu or "mm").lower()
        unit = width_unit if width_unit == height_unit else ""
        dims.append(
            {
                "width": float(w),
                "height": float(h),
                "unit": unit or width_unit,
                "width_unit": width_unit,
                "height_unit": height_unit,
            }
        )
    return dims


def detect_rooms_walls(image):
    """
    Use OpenCV contour detection to approximate rooms/walls as rectangles.
    Returns list of bounding boxes (x, y, w, h) in pixel space, filtered by area.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = image.shape[0] * image.shape[1]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.03 * perimeter, True) if perimeter else []
        rectangular = len(approximation) >= 4
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        # Keep plausible room-sized, mostly rectangular contours and reject page
        # borders, text fragments, and extremely thin drawing artifacts.
        if (
            rectangular
            and 0.002 * img_area < area < 0.6 * img_area
            and min(w, h) >= 30
            and aspect_ratio <= 12
        ):
            boxes.append((x, y, w, h))

    # Dilation can produce several nearly identical contours for one room.
    # Prefer the larger candidate when boxes substantially overlap.
    deduplicated = []
    for candidate in sorted(boxes, key=lambda box: box[2] * box[3], reverse=True):
        cx, cy, cw, ch = candidate
        duplicate = False
        for ox, oy, ow, oh in deduplicated:
            ix = max(0, min(cx + cw, ox + ow) - max(cx, ox))
            iy = max(0, min(cy + ch, oy + oh) - max(cy, oy))
            intersection = ix * iy
            union = cw * ch + ow * oh - intersection
            contained = intersection / max(min(cw * ch, ow * oh), 1)
            if intersection / max(union, 1) >= 0.75 or contained >= 0.9:
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(candidate)
    return sorted(deduplicated, key=lambda box: (box[1], box[0]))


def pixels_to_units(box, px_per_unit, unit="m"):
    """Convert a pixel bounding box to real-world width/height using a scale factor."""
    if not np.isfinite(px_per_unit) or px_per_unit <= 0:
        raise ValueError("px_per_unit must be a positive finite number")
    x, y, w, h = box
    if w < 0 or h < 0:
        raise ValueError("box dimensions must not be negative")
    return {
        "width": round(w / px_per_unit, 2),
        "height": round(h / px_per_unit, 2),
        "unit": unit,
        "area": round((w / px_per_unit) * (h / px_per_unit), 2),
    }


def estimate_scale_from_text(text, default_px_per_unit=50, render_dpi=200, drawing_dpi=72):
    """
    Estimate pixels-per-real-world-unit (meter) from 'SCALE 1:N' text in the drawing.

    IMPORTANT: px_per_unit must be computed relative to the DPI at which the page
    was rendered to an image (render_dpi, see pdf_to_images). CAD/PDF drawings are
    authored in points (drawing_dpi=72). If render_dpi != drawing_dpi, pixel counts
    scale by (render_dpi / drawing_dpi) relative to the original drawing geometry.
    This function returns px_per_unit already adjusted for that ratio, so callers
    can directly use it with pixels_to_units() on images from pdf_to_images(dpi=render_dpi).

    Falls back to a default (also DPI-adjusted) if no scale text is found. In
    production, prefer manual calibration: let the user click two points of
    known real-world distance on the rendered image.
    """
    if render_dpi <= 0 or drawing_dpi <= 0:
        raise ValueError("DPI values must be positive")
    if default_px_per_unit <= 0:
        raise ValueError("default_px_per_unit must be positive")
    dpi_ratio = render_dpi / drawing_dpi
    m = re.search(r"\bSCALE\s*1\s*[:/]\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
    if m:
        ratio = float(m.group(1))
        if ratio <= 0:
            return default_px_per_unit * dpi_ratio
        base_px_per_unit = default_px_per_unit / (ratio / 100)
    else:
        base_px_per_unit = default_px_per_unit
    return max(5, base_px_per_unit * dpi_ratio)
