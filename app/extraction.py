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
import io

DIM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|m|ft|feet|cm)?\s*[xX]\s*(\d+(?:\.\d+)?)\s*(mm|m|ft|feet|cm)?")
NUM_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def pdf_to_images(pdf_path, dpi=200):
    """Render each PDF page to a numpy image array (for OpenCV) using PyMuPDF."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
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
        unit = wu or hu or "mm"
        dims.append({"width": float(w), "height": float(h), "unit": unit})
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
        # keep boxes that look like rooms: not too tiny, not the whole page
        if 0.002 * img_area < area < 0.6 * img_area:
            boxes.append((x, y, w, h))
    return boxes


def pixels_to_units(box, px_per_unit, unit="m"):
    """Convert a pixel bounding box to real-world width/height using a scale factor."""
    x, y, w, h = box
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
    dpi_ratio = render_dpi / drawing_dpi
    m = re.search(r"SCALE\s*1\s*[:/]\s*(\d+)", text.upper())
    if m:
        ratio = int(m.group(1))
        base_px_per_unit = default_px_per_unit / (ratio / 100)
    else:
        base_px_per_unit = default_px_per_unit
    return max(5, base_px_per_unit * dpi_ratio)
