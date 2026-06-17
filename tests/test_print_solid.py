"""Print solid: base + fill + terrain as one connected body (no Y-gap shells)."""
from __future__ import annotations

import io
import unittest
import zipfile

import numpy as np
import rasterio.transform
import trimesh
from rasterio.features import rasterize
from shapely.geometry import box

from terrain_app import mesh as mesh_mod
from terrain_app.pipeline import export_basename_from_kml_filename, export_download_filename


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

    def test_smooth_clip_polygon_stays_valid(self) -> None:
        poly = box(0, 0, 100, 100)
        out = mesh_mod.smooth_clip_polygon(poly, 5.0)
        self.assertTrue(out.is_valid and not out.is_empty)
        self.assertGreater(out.area, 0.0)

    def test_boundary_smooth_uses_polygon_for_mesh_mask_for_alpha(self) -> None:
        h, w = 20, 20
        dem = np.zeros((h, w), dtype=np.float32)
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        tr = rasterio.transform.from_bounds(0, 0, 120, 120, w, h)
        poly = box(20, 20, 100, 100)
        bounds = (0.0, 0.0, 120.0, 120.0)
        mask, clip_poly, sm = mesh_mod.resolve_boundary_clipping(
            poly, (h, w), tr, bounds, w, h, 8.0
        )
        self.assertGreater(sm, 0.0)
        self.assertIsNotNone(clip_poly)
        self.assertTrue(clip_poly.is_valid and not clip_poly.is_empty)
        self.assertTrue(np.any((mask > 0) & (mask < 255)), "alpha mask should be soft at edges")
        mesh = mesh_mod.build_mesh(dem, tex, tr, mask=mask, poly_utm=clip_poly)
        qm = mesh_mod.quad_inclusion_mask(h, w, tr, clip_poly, mask)
        self.assertGreaterEqual(len(mesh.faces), int(qm.sum()))
        # Polygon clip must not leave fringe quads that a thresholded mask would add.
        mask_hard, _, _ = mesh_mod.resolve_boundary_clipping(
            poly, (h, w), tr, bounds, w, h, 0.0
        )
        mesh_hard = mesh_mod.build_mesh(dem, tex, tr, mask=mask_hard, poly_utm=poly)
        self.assertGreater(len(mesh_hard.faces), 0)

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
        self.assertGreaterEqual(len(mesh.faces), int(qm.sum()))

    def test_polygon_clip_fewer_faces_than_whole_quads(self) -> None:
        """Boundary quads are trimmed to the polygon, not kept as full stair-step blocks."""
        from shapely.geometry import Polygon
        from shapely.prepared import prep

        h, w = 40, 40
        dem = np.zeros((h, w), dtype=np.float32)
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        tr = rasterio.transform.from_bounds(0, 0, 200, 200, w, h)
        poly = Polygon([(40, 30), (160, 35), (155, 170), (35, 165)])
        mesh = mesh_mod.build_mesh(dem, tex, tr, poly_utm=poly)
        east, north = mesh_mod._cell_center_east_north(tr, h, w)
        pp = prep(poly)
        whole_quad_faces = 0
        for i in range(h - 1):
            for j in range(w - 1):
                if mesh_mod._include_quad(i, j, east, north, pp, poly.bounds, None):
                    whole_quad_faces += 2
        self.assertLess(len(mesh.faces), whole_quad_faces)

    def test_build_print_solid_raises_on_empty_surface(self) -> None:
        empty = trimesh.Trimesh()
        poly = box(0, 0, 10, 10)
        with self.assertRaises(ValueError):
            mesh_mod.build_print_solid(empty, poly, center_on_bed=True)

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

    def test_puzzle_grid_scales_tiles_toward_print_max(self) -> None:
        """2×2 uses a larger uniform scale so each bbox tile’s long side ≈ print_max."""
        mesh, poly = _synthetic_terrain(24, 24)
        one, meta_one, _ = mesh_mod.build_print_solid(
            mesh,
            poly,
            print_max_size_mm=100.0,
            center_on_bed=True,
            voxel_size_mm=None,
            print_split_nx=1,
            print_split_nz=1,
        )
        puz, meta_puz, _ = mesh_mod.build_print_solid(
            mesh,
            poly,
            print_max_size_mm=100.0,
            center_on_bed=True,
            voxel_size_mm=None,
            print_split_nx=2,
            print_split_nz=2,
        )
        self.assertFalse(meta_one.get("print_scale_for_puzzle_grid"))
        self.assertTrue(meta_puz.get("print_scale_for_puzzle_grid"))
        self.assertAlmostEqual(float(meta_puz["scale_meters_to_print_mm"]), 2.0 * float(meta_one["scale_meters_to_print_mm"]), delta=0.05)
        b1 = one.bounds
        b2 = puz.bounds
        self.assertIsNotNone(b1)
        self.assertIsNotNone(b2)
        w1 = max(float(b1[1, 0] - b1[0, 0]), float(b1[1, 1] - b1[0, 1]))
        w2 = max(float(b2[1, 0] - b2[0, 0]), float(b2[1, 1] - b2[0, 1]))
        self.assertAlmostEqual(w1, 100.0, delta=3.0)
        self.assertGreater(w2, w1 * 1.4)
        pieces = mesh_mod.split_solid_to_xz_grid(puz, 2, 2)
        self.assertGreaterEqual(len(pieces), 1)
        for p in pieces:
            mb = p["mesh"].bounds
            self.assertIsNotNone(mb)
            ph = max(float(mb[1, 0] - mb[0, 0]), float(mb[1, 1] - mb[0, 1]))
            self.assertLessEqual(ph, 105.0)
            self.assertGreaterEqual(ph, 70.0)

    def test_3mf_embedded_texture_payload(self) -> None:
        mesh, poly = _synthetic_terrain(14, 14)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=110.0, center_on_bed=True, voxel_size_mm=None
        )
        textured = mesh_mod.print_solid_with_satellite_uv(solid, surf)
        self.assertTrue(mesh_mod.mesh_has_texture_visual_for_3mf(textured))
        raw = mesh_mod.export_textured_print_3mf(textured)
        ins = mesh_mod.inspect_3mf_texture_payload(raw)
        self.assertTrue(ins.get("ok"), msg=str(ins))
        zf = zipfile.ZipFile(io.BytesIO(raw), "r")
        try:
            self.assertIn(mesh_mod.TEXTURED_3MF_TEXTURE_PART, zf.namelist())
            self.assertIn(mesh_mod.TEXTURED_3MF_MODEL_RELS_PART, zf.namelist())
        finally:
            zf.close()
        self.assertEqual(ins.get("textured_triangle_count"), ins.get("triangle_count"))
        self.assertGreater(int(ins.get("tex2coord_count") or 0), 10)

    def test_3mf_slicer_export_is_geometry_only(self) -> None:
        mesh, poly = _synthetic_terrain(12, 12)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=100.0, center_on_bed=True, voxel_size_mm=None
        )
        textured = mesh_mod.print_solid_with_satellite_uv(solid, surf)
        raw = mesh_mod.export_print_3mf(solid)
        ins = mesh_mod.inspect_3mf_texture_payload(raw)
        self.assertFalse(ins.get("ok"))
        reloaded = mesh_mod.mesh_from_3mf_bytes(raw)
        self.assertGreater(len(reloaded.faces), 0)
        textured_ins = mesh_mod.inspect_3mf_texture_payload(
            mesh_mod.export_textured_print_3mf(textured)
        )
        self.assertTrue(textured_ins.get("ok"))

    def test_3mf_roundtrip_has_no_non_manifold_edges(self) -> None:
        mesh, poly = _synthetic_terrain(14, 14)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=110.0, center_on_bed=True, voxel_size_mm=None
        )
        textured = mesh_mod.print_solid_with_satellite_uv(solid, surf)
        self.assertTrue(textured.visual.defined)
        self.assertEqual(getattr(textured.visual, "kind", None), "texture")
        raw = mesh_mod.export_print_3mf(solid)
        reloaded = mesh_mod.mesh_from_3mf_bytes(raw)
        self.assertEqual(mesh_mod.non_manifold_edge_count(reloaded), 0)

    def test_textured_print_glb_embeds_png(self) -> None:
        import json
        import struct

        mesh, poly = _synthetic_terrain(14, 14)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=110.0, center_on_bed=True, voxel_size_mm=None
        )
        textured = mesh_mod.print_solid_with_satellite_uv(solid, surf)
        raw = mesh_mod.export_textured_print_glb(textured)
        json_len, _ = struct.unpack_from("<I4s", raw, 12)
        tree = json.loads(raw[20 : 20 + json_len].decode("utf-8").rstrip("\x20"))
        bin_len = struct.unpack_from("<I4s", raw, 20 + json_len)[0]
        bin_start = 20 + json_len + 8
        bin_chunk = raw[bin_start : bin_start + bin_len]
        self.assertEqual(len(tree.get("images", [])), 1)
        self.assertIn("TEXCOORD_0", tree["meshes"][0]["primitives"][0]["attributes"])
        self.assertTrue(b"\x89PNG" in bin_chunk)

    def test_ams_print_glb_exports_scene(self) -> None:
        import json
        import struct

        mesh, poly = _synthetic_terrain(14, 14)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=110.0, center_on_bed=True, voxel_size_mm=None
        )
        parts = [
            ({"rgb": [200, 40, 30], "part_name": "color_a"}, solid),
            ({"rgb": [30, 120, 200], "part_name": "color_b"}, solid),
        ]
        raw = mesh_mod.export_ams_print_glb(parts)
        json_len, _ = struct.unpack_from("<I4s", raw, 12)
        tree = json.loads(raw[20 : 20 + json_len].decode("utf-8").rstrip("\x20"))
        self.assertGreaterEqual(len(tree.get("meshes", [])), 1)
        self.assertGreater(len(raw), 500)


class TestExportBasename(unittest.TestCase):
    def test_stem_from_kml_filename(self) -> None:
        self.assertEqual(export_basename_from_kml_filename("My Site.kml"), "My_Site")
        self.assertEqual(export_basename_from_kml_filename(None), "terrain")

    def test_download_filename_from_meta(self) -> None:
        meta = {"export_basename": "Denver_Park"}
        self.assertEqual(export_download_filename(meta, "_print.3mf"), "Denver_Park_print.3mf")


if __name__ == "__main__":
    unittest.main()
