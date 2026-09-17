import unittest

from app.extraction import calculate_pixels_per_real_unit, estimate_scale_from_text


class ScaleCalculationTests(unittest.TestCase):
    def test_scale_denominators_at_200_dpi(self):
        self.assertAlmostEqual(calculate_pixels_per_real_unit(50, 200), 157.480315, places=5)
        self.assertAlmostEqual(calculate_pixels_per_real_unit(100, 200), 78.740157, places=5)
        self.assertAlmostEqual(calculate_pixels_per_real_unit(200, 200), 39.370079, places=5)

    def test_scale_is_linear_with_render_dpi(self):
        self.assertAlmostEqual(
            calculate_pixels_per_real_unit(100, 400),
            calculate_pixels_per_real_unit(100, 200) * 2,
            places=6,
        )

    def test_scale_text_variants_and_unknown(self):
        self.assertAlmostEqual(estimate_scale_from_text("Scale = 1:100", 200), 78.740157, places=5)
        self.assertAlmostEqual(estimate_scale_from_text("SCALE 1/50", 200), 157.480315, places=5)
        self.assertIsNone(estimate_scale_from_text("no scale", 200))


if __name__ == "__main__":
    unittest.main()
