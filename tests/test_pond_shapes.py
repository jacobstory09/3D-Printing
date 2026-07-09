"""Tests for pond polygon mask conversion."""
from __future__ import annotations

import unittest

import numpy as np

from terrain_app import pond_shapes


class TestPondShapes(unittest.TestCase):
    def test_mask_to_polygons_and_back(self) -> None:
        h, w = 48, 48
        mask = np.zeros((h, w), dtype=bool)
        mask[10:22, 12:28] = True
        shapes = pond_shapes.mask_to_polygons(mask)
        self.assertGreater(len(shapes), 0)
        out = pond_shapes.polygons_to_mask(shapes, h, w)
        overlap = float(np.count_nonzero(mask & out))
        self.assertGreaterEqual(overlap / float(mask.sum()), 0.85)

    def test_empty_shapes_zero_mask(self) -> None:
        h, w = 32, 32
        out = pond_shapes.polygons_to_mask([], h, w)
        self.assertFalse(out.any())

    def test_overlapping_polygons_union(self) -> None:
        h, w = 40, 40
        shapes = [
            {
                "id": "a",
                "source": "manual",
                "vertices": [[5, 5], [20, 5], [20, 20], [5, 20]],
            },
            {
                "id": "b",
                "source": "manual",
                "vertices": [[15, 15], [30, 15], [30, 30], [15, 30]],
            },
        ]
        out = pond_shapes.polygons_to_mask(shapes, h, w)
        self.assertGreater(int(np.count_nonzero(out)), 20 * 20)

    def test_validate_shapes_clips_and_drops_degenerate(self) -> None:
        h, w = 20, 20
        shapes = [
            {"id": "bad", "source": "manual", "vertices": [[0, 0], [1, 0]]},
            {
                "id": "ok",
                "source": "manual",
                "vertices": [[2, 2], [10, 2], [10, 10], [2, 10]],
            },
        ]
        out = pond_shapes.validate_shapes(shapes, h, w)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "ok")


if __name__ == "__main__":
    unittest.main()
