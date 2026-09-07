"""Measured takeoff elements shared by 2D tracing and IFC import."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ELEMENT_TYPES = (
    "space",
    "wall",
    "opening",
    "slab",
    "beam",
    "column",
    "footing",
    "excavation",
)

OPENING_KINDS = ("door", "window")
SOURCES = ("2d", "ifc")

# Openings smaller than this (sqm) are not deducted from masonry/plaster (IS 1200).
IS1200_OPENING_DEDUCT_M2 = 0.1

# IS 1200 measurement constants for enhanced accuracy
IS1200_DEFAULT_WALL_HEIGHT = 3.0  # metres
IS1200_DEFAULT_WALL_THICKNESS = 0.23  # 230mm brick
IS1200_JOINTS_DEDUCTION_PERCENT = 2.5  # for mortar joints in brickwork
IS1200_PLASTER_THICKNESS_INTERNAL = 0.012  # 12mm
IS1200_PLASTER_THICKNESS_EXTERNAL = 0.020  # 20mm
IS1200_CONCRETE_WASTAGE_PERCENT = 3.0  # standard wastage allowance
IS1200_BRICKWORK_WASTAGE_PERCENT = 5.0  # standard wastage allowance
IS1200_DEDUCTION_THRESHOLD_AREA = 2.0  # sqm - small openings not deducted
IS1200_DEDUCTION_THRESHOLD_LENGTH = 1.0  # m - small openings not deducted

DEFAULT_ITEM_CODES = {
    "excavation": 1.1,
    "backfill": 1.2,
    "sand_filling": 1.5,
    "pcc": 2.1,
    "footing": 2.2,
    "column": 2.3,
    "beam": 2.4,
    "slab": 2.5,
    "stairs": 2.7,
    "shuttering": 2.10,
    "reinforcement": 2.9,
    "wall": 3.1,
    "partition_wall": 3.2,
    "aac_block": 3.5,
    "plaster_internal": 4.1,
    "plaster_external": 4.2,
    "plaster_ceiling": 4.3,
    "pointing": 4.4,
    "flooring": 5.1,
    "marble_flooring": 5.3,
    "granite_flooring": 5.4,
    "skirting": 5.8,
    "paint_internal": 6.1,
    "paint_external": 6.2,
    "wall_putty": 6.4,
    "primer": 6.5,
    "door": 7.1,
    "window": 7.2,
    "upvc_window": 7.4,
    "electrical_point": 8.1,
    "plumbing_point": 8.2,
    "waterproofing": 9.1,
    "toilet_waterproofing": 9.2,
    "railing": 10.1,
    "structural_steel": 10.5,
}


@dataclass
class Calibration:
    page: int
    px_per_m: float
    method: str = "manual"
    known_length_m: Optional[float] = None
    calibrated: bool = False


@dataclass
class MeasuredElement:
    id: str
    type: str
    name: str = ""
    source: str = "2d"
    page: int = 0
    polygon_m: Optional[list[tuple[float, float]]] = None
    centerline_m: Optional[list[tuple[float, float]]] = None
    thickness_m: Optional[float] = None
    height_m: Optional[float] = None
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    top_rl: Optional[float] = None
    bottom_rl: Optional[float] = None
    host_id: Optional[str] = None
    host_space_ids: list[str] = field(default_factory=list)
    shared: bool = False
    opening_kind: Optional[str] = None
    item_code: Optional[float] = None
    ifc_type: Optional[str] = None
    ifc_guid: Optional[str] = None
    qto: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Deficiency:
    element_id: str
    message: str


@dataclass
class MeasurementLine:
    ref: str
    element_id: str
    item_code: Optional[float]
    description: str
    unit: str
    length: Optional[float]
    breadth: Optional[float]
    depth: Optional[float]
    quantity: float
    is1200_note: str
    provisional: bool = False
    source: str = "2d"
    location: str = ""


@dataclass
class TakeoffDocument:
    project_name: str = ""
    source_filename: str = ""
    qs_name: str = ""
    qs_signed: bool = False
    units: str = "px"
    scale_method: str = "detected"
    ifc_units_confirmed: bool = False
    calibrations: list[Calibration] = field(default_factory=list)
    elements: list[MeasuredElement] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Quality assurance fields
    qa_status: str = "draft"  # draft, review, approved, rejected
    qa_reviewer: str = ""
    qa_review_date: str = ""
    qa_comments: list[str] = field(default_factory=list)
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    measurement_standard: str = "IS 1200"  # Measurement standard used
