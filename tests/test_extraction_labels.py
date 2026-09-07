import unittest

from app.extraction import assign_room_labels


class RoomLabelTests(unittest.TestCase):
    def test_assigns_label_inside_matching_box(self):
        assignments = assign_room_labels(
            [(0, 0, 100, 100), (120, 0, 100, 100)],
            [
                {"text": "BEDROOM", "x": 20, "y": 20, "width": 40, "height": 12, "confidence": 0.91},
                {"text": "KITCHEN", "x": 300, "y": 20, "width": 40, "height": 12, "confidence": 0.95},
            ],
        )

        self.assertEqual(assignments[0], {"name": "BEDROOM", "confidence": 0.91})
        self.assertEqual(assignments[1], {"name": "", "confidence": 0.0})


if __name__ == "__main__":
    unittest.main()