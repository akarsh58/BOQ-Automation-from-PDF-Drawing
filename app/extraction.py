"""Room label assignment logic used by the UI/API."""
from __future__ import annotations

from typing import Iterable, Sequence

Box = tuple[float, float, float, float]
Label = dict


def assign_room_labels(boxes: Sequence[Box], labels: Sequence[Label]) -> list[dict]:
    """For each box, pick the highest-confidence label whose bbox center falls inside the box.
    Boxes without a matching label get ``{'name': '', 'confidence': 0.0}``.
    """
    assignments: list[dict] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        best: dict = {"name": "", "confidence": 0.0}
        for lbl in labels:
            lx = lbl.get("x", 0)
            ly = lbl.get("y", 0)
            lw = lbl.get("width", 0)
            lh = lbl.get("height", 0)
            # Label center point
            lcx = lx + lw / 2.0
            lcy = ly + lh / 2.0
            if x1 <= lcx <= x2 and y1 <= lcy <= y2:
                conf = float(lbl.get("confidence", 0.0) or 0.0)
                if conf > float(best.get("confidence", 0.0) or 0.0):
                    best = {"name": lbl.get("text", ""), "confidence": conf}
        assignments.append(best)
    return assignments
