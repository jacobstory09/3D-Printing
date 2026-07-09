"""Pipeline lifecycle guards (export ordering, variable lifetime)."""
from __future__ import annotations

import unittest
from pathlib import Path


class TestPipelineLifecycle(unittest.TestCase):
    def test_dem_survives_until_after_ams_export(self) -> None:
        """Regression: dem must not be deleted before pond detection for AMS review."""
        src = Path(__file__).resolve().parents[1] / "terrain_app" / "pipeline.py"
        text = src.read_text(encoding="utf-8")
        prepare_dem = text.index("_prepare_pond_shapes_for_job")
        final_del_dem = text.index("del mesh_uv, texture_rgba, dem")
        self.assertLess(
            prepare_dem,
            final_del_dem,
            "pond preparation must appear before dem is deleted",
        )
        early_del = "del mesh, z_display, mask\n"
        self.assertIn(early_del, text)
        self.assertNotIn("del mesh, z_display, mask, dem", text)


if __name__ == "__main__":
    unittest.main()
