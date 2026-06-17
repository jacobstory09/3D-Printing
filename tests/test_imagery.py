"""Tests for imagery tile mosaic."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import mercantile
import numpy as np

from terrain_app import imagery as imagery_mod


class TestParallelMosaic(unittest.TestCase):
    def test_mosaic_assembles_all_tiles(self) -> None:
        tiles = [
            mercantile.Tile(10, 20, 12),
            mercantile.Tile(11, 20, 12),
        ]

        def fake_download(_session, _url: str) -> np.ndarray:
            return np.full((256, 256, 3), 127, dtype=np.uint8)

        with patch.object(imagery_mod, "_download_tile", side_effect=fake_download):
            canvas, bounds = imagery_mod.mosaic_xyz_to_mercator(
                tiles,
                12,
                lambda t: f"http://example/{t.z}/{t.x}/{t.y}",
                imagery_mod.make_session(),
            )
        self.assertEqual(canvas.shape, (256, 512, 3))
        self.assertEqual(int(canvas.min()), 127)
        self.assertEqual(int(canvas.max()), 127)
        self.assertEqual(len(bounds), 4)


if __name__ == "__main__":
    unittest.main()
