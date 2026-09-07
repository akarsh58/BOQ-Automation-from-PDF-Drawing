import unittest

import pandas as pd

from app.measurement import measure_takeoff, record_qa_review
from app.models import MeasuredElement, TakeoffDocument


def rate_file() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_code": 3.1, "description": "Brickwork", "unit": "cum", "rate_inr": 100},
            {"item_code": 4.1, "description": "Internal plaster", "unit": "sqm", "rate_inr": 10},
            {"item_code": 4.2, "description": "External plaster", "unit": "sqm", "rate_inr": 10},
            {"item_code": 6.1, "description": "Internal paint", "unit": "sqm", "rate_inr": 5},
            {"item_code": 6.2, "description": "External paint", "unit": "sqm", "rate_inr": 5},
            {"item_code": 7.1, "description": "Door", "unit": "each", "rate_inr": 50},
            {"item_code": 5.1, "description": "Flooring", "unit": "sqm", "rate_inr": 20},
        ]
    )


class MeasurementAccuracyTests(unittest.TestCase):
    def test_wall_quantity_exposes_wastage_separately(self):
        wall = MeasuredElement(
            id="w1",
            type="wall",
            name="Wall",
            centerline_m=[(0, 0), (4, 0)],
            height_m=3,
            thickness_m=0.2,
        )
        lines, deficiencies = measure_takeoff(
            TakeoffDocument(elements=[wall], units="m"), rate_file()
        )

        brickwork = next(line for line in lines if line.item_code == 3.1)
        self.assertFalse(deficiencies)
        self.assertEqual(brickwork.gross_quantity, 2.4)
        self.assertEqual(brickwork.wastage_quantity, 0.12)
        self.assertEqual(brickwork.quantity, 2.52)

    def test_opening_far_from_host_wall_is_not_deducted(self):
        wall = MeasuredElement(
            id="w1",
            type="wall",
            centerline_m=[(0, 0), (4, 0)],
            height_m=3,
            thickness_m=0.2,
        )
        opening = MeasuredElement(
            id="o1",
            type="opening",
            opening_kind="door",
            host_id="w1",
            width_m=1,
            height_m=2.1,
            centerline_m=[(2, 2)],
        )
        lines, _ = measure_takeoff(
            TakeoffDocument(elements=[wall, opening], units="m"), rate_file()
        )

        brickwork = next(line for line in lines if line.item_code == 3.1)
        self.assertEqual(brickwork.gross_quantity, 2.4)
        self.assertIn("not deducted", brickwork.is1200_note)

    def test_qa_revision_records_previous_status(self):
        doc = TakeoffDocument(qa_status="draft")
        record_qa_review(doc, "Reviewer", "approved")

        self.assertEqual(doc.revision_history[0]["qa_status_before"], "draft")
        self.assertEqual(doc.qa_status, "approved")


if __name__ == "__main__":
    unittest.main()