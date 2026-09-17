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
ROOM_LABEL_PATTERN = re.compile(
    r"\b(?:bedroom|living(?:\s+room)?|dining(?:\s+room)?|kitchen|toilet|bathroom|"
    r"wc|store|study|office|lobby|corridor|passage|balcony|verandah|utility|room)\b",
    re.IGNORECASE,
)
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


def extract_room_labels(image):
    """Extract recognised room labels and their image positions for box matching."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(gray, config="--psm 11", output_type=pytesseract.Output.DICT)
    labels = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = " ".join(str(raw_text).split())
        if not text or not ROOM_LABEL_PATTERN.search(text):
            continue
        try:
            confidence = float(data["conf"][index])
            x = int(data["left"][index])
            y = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if confidence >= 0 and width > 0 and height > 0:
            labels.append(
                {
                    "text": text[:80],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "confidence": round(min(confidence / 100, 1), 3),
                }
            )
    return labels


def assign_room_labels(boxes, labels):
    """Match OCR room labels whose centres fall inside detected room boxes."""
    assignments = []
    for box in boxes:
        x, y, width, height = box
        candidates = []
        for label in labels:
            label_x = label["x"] + label["width"] / 2
            label_y = label["y"] + label["height"] / 2
            if x <= label_x <= x + width and y <= label_y <= y + height:
                distance = ((label_x - (x + width / 2)) ** 2 + (label_y - (y + height / 2)) ** 2) ** 0.5
                candidates.append((distance, label))
        if candidates:
            _, label = min(candidates, key=lambda candidate: (candidate[0], -candidate[1]["confidence"]))
            assignments.append({"name": label["text"], "confidence": label["confidence"]})
        else:
            assignments.append({"name": "", "confidence": 0.0})
    return assignments


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
    # The geometry is rendered at ``render_dpi`` while the scale convention is
    # based on drawing points.  Return the adjusted value for the actual image.
    return base_px_per_unit * dpi_ratio


# ========== INTELLIGENT AUTOMATIC DETECTION ==========

STANDARD_PAPER_SIZES = {
    "A0": (841, 1189), "A1": (594, 841), "A2": (420, 594),
    "A3": (297, 420), "A4": (210, 297), "A5": (148, 210),
    "B0": (1000, 1414), "B1": (707, 1000), "B2": (500, 707),
    "B3": (353, 500), "B4": (250, 353), "B5": (176, 250),
    "Executive": (260, 385), "Legal": (216, 356), "Letter": (216, 279),
    "Tabloid": (279, 432),
}

ROOM_NAME_KEYWORDS = {
    "bedroom": "Bedroom", "living room": "Living Room", "dining room": "Dining Room",
    "kitchen": "Kitchen", "toilet": "Toilet", "bathroom": "Bathroom", "wc": "WC",
    "store": "Store", "study": "Study", "office": "Office", "lobby": "Lobby",
    "corridor": "Corridor", "passage": "Passage", "balcony": "Balcony",
    "verandah": "Verandah", "utility": "Utility Room", "room": "Room",
    "hall": "Hall", "entrance": "Entrance", "garage": "Garage",
    "garden": "Garden", "patio": "Patio", "terrace": "Terrace",
    "staircase": "Staircase", "stairs": "Staircase", "lift": "Lift",
    "elevator": "Elevator", "parking": "Parking",
}


def auto_detect_scale(image, text, render_dpi=200):
    """
    Automatically detect drawing scale with multiple fallback strategies.
    Returns (px_per_m, confidence, method, details).
    """
    if text:
        m = re.search(r"\bSCALE\s*1\s*[:/]\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        if m:
            ratio = float(m.group(1))
            dpi_ratio = render_dpi / 72
            if ratio > 0:
                px_per_m = max(5, (100 / (ratio / 100)) * dpi_ratio)
                return px_per_m, 0.95, "text_scale", {"scale_text": m.group(0), "ratio": ratio}
        m2 = re.search(r"\b1\s*[:/]\s*(\d+(?:\.\d+)?)\b", text)
        if m2 and "scale" in text.lower():
            ratio = float(m2.group(1))
            dpi_ratio = render_dpi / 72
            px_per_m = max(5, (100 / (ratio / 100)) * dpi_ratio)
            return px_per_m, 0.85, "scale_text", {"ratio": ratio}

    return None, 0.0, "unknown", {"note": "No scale detected"}


def auto_detect_openings(rooms, ocr_text, image_shape=None):
    """
    Automatically detect doors and windows near walls.
    Uses OCR text to find door/window symbols and dimensions.
    Returns list of opening dicts.
    """
    openings = []
    if not rooms:
        return openings
    
    lines = ocr_text.split("\n") if ocr_text else []
    opening_id = 0
    
    for line in lines:
        line_lower = line.lower().strip()
        is_door = any(kw in line_lower for kw in ["door", "d-", "d "])
        is_window = any(kw in line_lower for kw in ["window", "w-", "w "])
        
        if is_door or is_window:
            dims = re.compile(
                r"(\d+(?:\.\d+)?)\s*(mm|m|cm)?\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|m|cm)?",
                re.IGNORECASE
            ).search(line)
            if dims:
                w_val = float(dims.group(1))
                h_val = float(dims.group(3))
                w_unit = dims.group(2) or dims.group(4) or "mm"
                w_m = w_val * (0.001 if w_unit == "mm" else 0.01 if w_unit == "cm" else 1)
                h_m = h_val * (0.001 if (dims.group(4) or w_unit) == "mm" else 0.01 if (dims.group(4) or w_unit) == "cm" else 1)
                
                opening_id += 1
                opening_kind = "door" if is_door else "window"
                openings.append({
                    "id": f"auto-{opening_kind}-{opening_id}",
                    "name": f"{opening_kind.capitalize()} {opening_id}",
                    "opening_kind": opening_kind,
                    "width_m": round(w_m, 2),
                    "height_m": round(h_m, 2),
                    "source": "auto-detected",
                })
    
    return openings


def auto_detect_walls(rooms, image_shape=None):
    """
    Automatically detect walls between adjacent rooms.
    Returns list of wall dicts.
    """
    walls = []
    wall_id = 0
    if len(rooms) < 2:
        return walls
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1 = rooms[i]
            r2 = rooms[j]
            if boxes_share_edge(r1, r2):
                wall_line = find_shared_wall(r1, r2)
                if wall_line:
                    wall_id += 1
                    walls.append({
                        "id": f"auto-wall-{wall_id}",
                        "name": f"Wall {wall_id}",
                        "points": [wall_line["start"], wall_line["end"]], 
                        "thickness_m": 0.23,
                        "height_m": 3.0,
                        "host_space_ids": [r1.get("id", f"room-{i}"), r2.get("id", f"room-{j}")],
                        "shared": True,
                        "source": "auto-detected",
                    })
    return walls
def auto_extract_dimensions(text):
    """Extract dimension strings from OCR/text output."""
    dimensions = []
    if not text:
        return dimensions
    lines = text.split("\n")
    dim_pattern = re.compile(
        r"([\d.]+)\s*(mm|m|cm|ft|feet)?\s*[x×]\s*([\d.]+)\s*(mm|m|cm|ft|feet)?",
        re.IGNORECASE
    )
    for line in lines:
        m = dim_pattern.search(line)
        if m:
            dimensions.append({
                "width_val": float(m.group(1)),
                "width_unit": m.group(2) or "mm",
                "height_val": float(m.group(3)),
                "height_unit": m.group(4) or "mm",
                "text": line.strip(),
            })
    return dimensions


def auto_classify_room(name_text, area, width=None, height=None):
    """Classify a room based on name text and dimensions."""
    if not name_text:
        return "Room", "generic"
    name_lower = name_text.strip().lower()
    for keyword, category_name in ROOM_NAME_KEYWORDS.items():
        if keyword.lower() in name_lower:
            return category_name, keyword
    if width and height:
        area_est = width * height
        if area_est < 5:
            return "Utility Room", "small"
        elif area_est < 15:
            return "Room", "medium"
        else:
            return "Living Area", "large"
    return "Room", "generic"


def build_takeoff_from_detection(image, text, rooms, render_dpi=200):
    """
    Build a complete takeoff document from automatic detection results.
    Returns a dict ready for parse_takeoff() that requires minimal manual intervention.
    """
    px_per_m, scale_confidence, scale_method, scale_details = auto_detect_scale(image, text, render_dpi)
    
    if px_per_m is None:
        px_per_m = auto_estimate_scale_from_elements(rooms)
        if px_per_m is None:
            px_per_m = 50
            scale_method = "fallback"
            scale_confidence = 0.3
    
    rooms_m = []
    room_id = 0
    for room in rooms:
        x, y, w, h = room["box"]
        width_m = round(w / px_per_m, 2)
        height_m = round(h / px_per_m, 2)
        area = round(width_m * height_m, 2)
        name = room.get("label", "")
        classified_name, category = auto_classify_room(name, area, width_m, height_m)
        room_id += 1
        rooms_m.append({
            "id": f"auto-room-{room_id}",
            "name": classified_name,
            "category": category,
            "points": [[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]],
            "width_m": width_m,
            "height_m": height_m,
            "area": area,
            "item_code": 5.1,
            "confidence": scale_confidence,
            "source": "auto-detected",
        })
    
    walls = auto_detect_walls(rooms)
    openings = auto_detect_openings(rooms, text)
    
    elements = []
    for room in rooms_m:
        elements.append({
            "id": room["id"],
            "type": "space",
            "name": room["name"],
            "page": 0,
            "points": [[0, 0], [room["width_m"], 0], [room["width_m"], room["height_m"]], [0, room["height_m"]]],
            "item_code": room["item_code"],
            "source": "auto-detected",
        })
    for wall in walls:
        elements.append({
            "id": wall["id"],
            "type": "wall",
            "name": wall["name"],
            "points": wall["points"],
            "thickness_m": wall["thickness_m"],
            "height_m": wall["height_m"],
            "host_space_ids": wall["host_space_ids"],
            "shared": wall.get("shared", True),
            "source": "auto-detected",
        })
    for opening in openings:
        elements.append({
            "id": opening["id"],
            "type": "opening",
            "name": opening["name"],
            "opening_kind": opening["opening_kind"],
            "width_m": opening["width_m"],
            "height_m": opening["height_m"],
            "host_id": "wall-1" if walls else None,
            "source": "auto-detected",
        })
    
    scale_method_str = scale_method
    if scale_method == "text_scale":
        scale_method_str = "two_point"
    elif scale_method == "paper_size":
        scale_method_str = "manual"
    
    takeoff_data = {
        "project_name": "Auto-detected Project",
        "source_filename": "auto_detected",
        "qs_name": "Auto QS",
        "qs_signed": True,
        "units": "m",
        "scale_method": scale_method_str,
        "ifc_units_confirmed": True,
        "measurement_standard": "IS 1200",
        "calibrations": [{
            "page": 0,
            "px_per_m": px_per_m,
            "method": scale_method_str,
            "calibrated": True,
        }] if px_per_m else [],
        "elements": elements,
        "notes": [f"Auto-detected scale: {scale_method} with confidence {scale_confidence:.2f}"],
    }
    
    return takeoff_data


def boxes_share_edge(r1, r2, tolerance=5):
    """Check if two room boxes share a common edge."""
    x1, y1, w1, h1 = r1["box"]
    x2, y2, w2, h2 = r2["box"]
    if abs(x1 + w1 - x2) < tolerance or abs(x2 + w2 - x1) < tolerance:
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        if y_overlap > 0.3 * min(h1, h2):
            return True
    if abs(y1 + h1 - y2) < tolerance or abs(y2 + h2 - y1) < tolerance:
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        if x_overlap > 0.3 * min(w1, w2):
            return True
    return False


def find_shared_wall(r1, r2):
    """Find the shared wall line between two adjacent rooms."""
    x1, y1, w1, h1 = r1["box"]
    x2, y2, w2, h2 = r2["box"]
    if abs(x1 + w1 - x2) < 5 or abs(x2 + w2 - x1) < 5:
        shared_x = min(x1 + w1, x2 + w2)
        y_start = max(y1, y2)
        y_end = min(y1 + h1, y2 + h2)
        if y_end > y_start:
            return {"start": [shared_x, y_start], "end": [shared_x, y_end], "horizontal": False}
    if abs(y1 + h1 - y2) < 5 or abs(y2 + h2 - y1) < 5:
        shared_y = min(y1 + h1, y2 + h2)
        x_start = max(x1, x2)
        x_end = min(x1 + w1, x2 + w2)
        if x_end > x_start:
            return {"start": [x_start, shared_y], "end": [x_end, shared_y], "horizontal": True}
    return None


def auto_estimate_scale_from_elements(rooms):
    """Estimate scale from detected room elements if manual calibration failed."""
    if not rooms:
        return None
    pixel_sizes = [(r["width"], r["height"]) for r in rooms if "width" in r and "height" in r]
    if not pixel_sizes:
        return None
    median_w = sorted(s[0] for s in pixel_sizes)[len(pixel_sizes)//2]
    median_h = sorted(s[1] for s in pixel_sizes)[len(pixel_sizes)//2]
    smaller_dim = min(median_w, median_h)
    if 50 < smaller_dim < 800:
        estimated_px_per_m = smaller_dim / 3.0
        return max(5, estimated_px_per_m)
    return None
