"""CRS helpers."""
from __future__ import annotations

import unittest

from terrain_app import crs as crs_mod


class TestCrs(unittest.TestCase):
    def test_grid_dimensions_caps_longest_side(self) -> None:
        # Tall bbox: height should be capped at max_dimension, not width.
        bounds = (0.0, 0.0, 1000.0, 4000.0)
        w, h = crs_mod.grid_dimensions_from_bounds(bounds, 1024)
        self.assertEqual(h, 1024)
        self.assertLess(w, 1024)
        self.assertGreaterEqual(w, 64)

    def test_grid_dimensions_wide_bbox(self) -> None:
        bounds = (0.0, 0.0, 4000.0, 1000.0)
        w, h = crs_mod.grid_dimensions_from_bounds(bounds, 512)
        self.assertEqual(w, 512)
        self.assertLess(h, 512)


if __name__ == "__main__":
    unittest.main()
