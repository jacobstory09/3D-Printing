"""Print solid: base + fill + terrain as one connected body (no Y-gap shells)."""
from __future__ import annotations

import unittest

import numpy as np
import rasterio.transform
from rasterio.features import rasterize
from shapely.geometry import box

from terrain_app import mesh as mesh_mod


def _synthetic_terrain(h: int, w: int) -> tuple:
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    dem = (0.8 * xx + 0.4 * yy + 0.2 * np.sin(xx * 6)).astype(np.float32)
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    tr = rasterio.transform.from_bounds(0, 0, 120, 120, w, h)
    poly = box(20, 20, 100, 100)
    mask = rasterize(
        [(poly, 255)], out_shape=(h, w), transform=tr, fill=0, dtype=np.uint8, all_touched=True
    )
    mesh = mesh_mod.build_mesh(dem, tex, tr, mask=mask, poly_utm=poly)
    return mesh, poly


class TestPrintSolid(unittest.TestCase):
    def test_dem_maps_to_mesh_z_and_scales_with_vertical_exaggeration(self) -> None:
        """Elevation from prepare_z must sit on vertex column 2 (Z); vex scales that span."""
        h, w = 10, 10
        rng = np.random.default_rng(0)
        dem = (rng.random((h, w)).astype(np.float32) * 5.0) + 50.0
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        tr = rasterio.transform.from_bounds(0, 0, 200, 200, w, h)
        mask = np.full((h, w), 255, dtype=np.uint8)
        m1 = mesh_mod.build_mesh(
            dem, tex, tr, vertical_exaggeration=1.0, mask=mask, poly_utm=None
        )
        m3 = mesh_mod.build_mesh(
            dem, tex, tr, vertical_exaggeration=3.0, mask=mask, poly_utm=None
        )
        z1 = np.ptp(np.asarray(m1.vertices)[:, 2])
        z3 = np.ptp(np.asarray(m3.vertices)[:, 2])
        self.assertGreater(z1, 1e-6)
        self.assertAlmostEqual(z3 / z1, 3.0, places=5)
        zprep = mesh_mod.prepare_z(dem, vertical_exaggeration=1.0, z_offset_mode="min")
        self.assertAlmostEqual(z1, float(np.ptp(zprep)), places=5)

    def test_quad_inclusion_mask_matches_mesh(self) -> None:
        h, w = 20, 20
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        dem = (0.8 * xx + 0.4 * yy + 0.2 * np.sin(xx * 6)).astype(np.float32)
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        tr = rasterio.transform.from_bounds(0, 0, 120, 120, w, h)
        poly = box(20, 20, 100, 100)
        mask = rasterize(
            [(poly, 255)], out_shape=(h, w), transform=tr, fill=0, dtype=np.uint8, all_touched=True
        )
        mesh = mesh_mod.build_mesh(dem, tex, tr, mask=mask, poly_utm=poly)
        qm = mesh_mod.quad_inclusion_mask(h, w, tr, poly, mask)
        self.assertEqual(int(qm.sum()) * 2, len(mesh.faces))

    def test_single_component_and_plate(self) -> None:
        mesh, poly = _synthetic_terrain(20, 20)
        solid, meta, surf_mm = mesh_mod.build_print_solid(
            mesh,
            poly,
            print_max_size_mm=150.0,
            base_extrusion_mm=1.0,
            center_on_bed=True,
            voxel_size_mm=None,
        )
        parts = solid.split(only_watertight=False)
        self.assertEqual(len(parts), 1, "export should be one connected solid")
        b = solid.bounds
        self.assertIsNotNone(b)
        self.assertGreaterEqual(float(b[0, 2]), -1e-3)
        self.assertTrue(meta.get("z_up"))
        self.assertEqual(int(meta.get("print_component_count", 0)), 1)
        for key in (
            "print_solid_fused",
            "print_solid_last_resort_voxel",
            "print_component_count_pre_fuse",
            "print_non_manifold_edge_count",
            "print_open_edge_count",
        ):
            self.assertIn(key, meta)
        self.assertEqual(mesh_mod.non_manifold_edge_count(solid), 0)
        self.assertEqual(int(meta["print_non_manifold_edge_count"]), 0)

    def test_3mf_roundtrip_has_no_non_manifold_edges(self) -> None:
        mesh, poly = _synthetic_terrain(14, 14)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=110.0, center_on_bed=True, voxel_size_mm=None
        )
        textured = mesh_mod.print_solid_with_satellite_uv(solid, surf)
        self.assertTrue(textured.visual.defined)
        self.assertEqual(getattr(textured.visual, "kind", None), "texture")
        raw = mesh_mod.export_print_3mf(textured)
        reloaded = mesh_mod.mesh_from_3mf_bytes(raw)
        self.assertEqual(mesh_mod.non_manifold_edge_count(reloaded), 0)


if __name__ == "__main__":
    unittest.main()
