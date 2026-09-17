import unittest

from app.extraction import associate_opening_to_walls


class OpeningAssociationTests(unittest.TestCase):
    def setUp(self):
        self.wall = {"id": "w1", "points": [[0.0, 0.0], [4.0, 0.0]], "coordinate_system": "m"}

    def test_opening_directly_on_wall_is_associated(self):
        self.assertEqual(
            associate_opening_to_walls({"center_m": [2.0, 0.0]}, [self.wall]),
            "w1",
        )

    def test_opening_near_wall_within_tolerance_is_associated(self):
        self.assertEqual(
            associate_opening_to_walls({"center_m": [2.0, 0.1]}, [self.wall], tolerance=0.15),
            "w1",
        )

    def test_opening_far_from_wall_is_unassociated(self):
        self.assertIsNone(
            associate_opening_to_walls({"center_m": [2.0, 1.0]}, [self.wall], tolerance=0.15)
        )

    def test_opening_without_position_is_unassociated(self):
        self.assertIsNone(associate_opening_to_walls({}, [self.wall]))

    def test_pixel_metre_mismatch_does_not_associate(self):
        self.assertIsNone(
            associate_opening_to_walls({"center_px": [200, 0]}, [self.wall], tolerance=0.15)
        )

    def test_nearest_of_two_walls_is_selected(self):
        walls = [
            self.wall,
            {"id": "w2", "points": [[0.0, 0.2], [4.0, 0.2]], "coordinate_system": "m"},
        ]
        self.assertEqual(
            associate_opening_to_walls({"center_m": [2.0, 0.16]}, walls, tolerance=0.2),
            "w2",
        )

    def test_pixel_center_is_converted_before_matching(self):
        self.assertEqual(
            associate_opening_to_walls(
                {"center_px": [200.0, 8.0]},
                [self.wall],
                tolerance=0.15,
                pixels_per_m=100.0,
            ),
            "w1",
        )


if __name__ == "__main__":
    unittest.main()
