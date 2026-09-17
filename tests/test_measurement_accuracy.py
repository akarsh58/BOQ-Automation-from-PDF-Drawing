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


    def test_small_opening_is_deducted(self):
        """Small openings above 0.1 sqm should be deducted (IS 1200)."""
        wall = MeasuredElement(
            id="w1",
            type="wall",
            centerline_m=[(0, 0), (4, 0)],
            height_m=3,
            thickness_m=0.2,
        )
        opening = MeasuredElement(
            id="o-small",
            type="opening",
            opening_kind="window",
            host_id="w1",
            width_m=0.6,
            height_m=0.2,
            centerline_m=[(2, 0)],
        )
        lines, _ = measure_takeoff(
            TakeoffDocument(elements=[wall, opening], units="m"), rate_file()
        )
        brickwork = next(line for line in lines if line.item_code == 3.1)
        # 0.6 x 0.2 = 0.12 sqm > 0.1 sqm threshold → must be deducted
        self.assertAlmostEqual(2.376, brickwork.gross_quantity)

    def test_sub_01_opening_is_not_deducted(self):
        """Openings at or below 0.1 sqm remain non-deductible."""
        wall = MeasuredElement(
            id="w-small-open",
            type="wall",
            centerline_m=[(0, 0), (3, 0)],
            height_m=3,
            thickness_m=0.23,
        )
        opening = MeasuredElement(
            id="o-small-open",
            type="opening",
            opening_kind="door",
            width_m=0.30,
            height_m=0.30,
            host_id="w-small-open",
        )
        lines, _ = measure_takeoff(
            TakeoffDocument(elements=[wall, opening], units="m"), rate_file()
        )
        brickwork = next(line for line in lines if line.item_code == 3.1)
        # 0.30 x 0.30 = 0.09 sqm <= 0.1 sqm threshold → not deducted; gross = 3*3*0.23 = 2.07
        self.assertAlmostEqual(2.07, brickwork.gross_quantity)
        self.assertIn("not deducted", brickwork.is1200_note)

    def test_shared_wall_deduplication(self):
        """Duplicate shared walls are measured once with 2 plaster faces."""
        wall1 = MeasuredElement(
            id="w-shared",
            type="wall",
            centerline_m=[(0, 0), (3, 0)],
            height_m=3,
            thickness_m=0.23,
        )
        wall2 = MeasuredElement(
            id="w-dup",
            type="wall",
            centerline_m=[(0, 0), (3, 0)],
            height_m=3,
            thickness_m=0.23,
        )
        lines, _ = measure_takeoff(
            TakeoffDocument(elements=[wall1, wall2], units="m"), rate_file()
        )
        self.assertEqual(1, len([line for line in lines if line.item_code == 3.1]))
        self.assertEqual(1, len([line for line in lines if line.item_code == 4.1]))
        self.assertAlmostEqual(3 * 3 * 2, next(line.quantity for line in lines if line.item_code == 4.1))

    def test_shared_wall_with_tracing_tolerance(self):
        """Walls with slightly different traced endpoints deduplicate."""
        wall1 = MeasuredElement(
            id="w-shared-1",
            type="wall",
            centerline_m=[(0, 0), (3, 0)],
            height_m=3,
            thickness_m=0.23,
        )
        wall2 = MeasuredElement(
            id="w-shared-2",
            type="wall",
            centerline_m=[(0.01, 0.01), (3.01, 0.01)],
            height_m=3,
            thickness_m=0.23,
        )
        lines, _ = measure_takeoff(
            TakeoffDocument(elements=[wall1, wall2], units="m"), rate_file()
        )
        self.assertEqual(1, len([line for line in lines if line.item_code == 3.1]))

    def test_internal_wall_has_no_external_finishes(self):
        """Internal walls do not receive external plaster/paint charges."""
        wall = MeasuredElement(
            id="w-internal",
            type="wall",
            centerline_m=[(0, 0), (3, 0)],
            height_m=3,
            thickness_m=0.23,
            host_space_ids=["s1"],
        )
        lines, _ = measure_takeoff(
            TakeoffDocument(elements=[wall], units="m"), rate_file()
        )
        self.assertFalse(any(line.item_code == 4.2 for line in lines))
        self.assertFalse(any(line.item_code == 6.2 for line in lines))


if __name__ == "__main__":
    unittest.main()