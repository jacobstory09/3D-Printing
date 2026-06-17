"""Tests for Bambu AMS color quantization and export."""
from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path

import numpy as np
import rasterio.transform
from rasterio.features import rasterize
from shapely.geometry import box

import trimesh

from terrain_app import ams_color
from terrain_app import mesh as mesh_mod


def _synthetic_colored_terrain(h: int, w: int) -> tuple:
    dem = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    tex[: h // 2, : w // 2] = (40, 95, 143)  # blue
    tex[: h // 2, w // 2 :] = (27, 77, 42)  # dark green
    tex[h // 2 :, : w // 2] = (90, 160, 70)  # green
    tex[h // 2 :, w // 2 :] = (130, 130, 130)  # gray
    tr = rasterio.transform.from_bounds(0, 0, 120, 120, w, h)
    poly = box(10, 10, 110, 110)
    mask = rasterize(
        [(poly, 255)], out_shape=(h, w), transform=tr, fill=0, dtype=np.uint8, all_touched=True
    )
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = tex
    rgba[:, :, 3] = mask
    mesh = mesh_mod.build_mesh(dem, tex, tr, mask=mask, poly_utm=poly)
    return mesh, poly, rgba


class TestAmsColor(unittest.TestCase):
    def test_recommend_palette_reserves_pond_slot(self) -> None:
        rgba = np.zeros((48, 48, 4), dtype=np.uint8)
        rgba[:, :, :3] = (70, 110, 45)
        rgba[:, :, 3] = 255
        rgba[18:26, 18:26, :3] = (10, 17, 11)
        palette = ams_color.recommend_ams_palette(rgba, n_colors=4)
        roles = {p.get("role") for p in palette}
        self.assertIn("pond", roles)
        self.assertIn("base", roles)
        base = next(p for p in palette if p.get("role") == "base")
        self.assertEqual(base["index"], 0)
        self.assertEqual(base["part_name"], "01_base")
        pond = next(p for p in palette if p.get("role") == "pond")
        self.assertEqual(pond["name"], "Pond")
        self.assertGreater(int(pond.get("pond_pixel_count") or 0), 0)

    def test_uniform_pasture_not_detected_conservative(self) -> None:
        h, w = 64, 64
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = (92, 118, 58)
        rgba[:, :, 3] = 255
        dem = np.linspace(138.0, 142.0, h * w, dtype=np.float64).reshape(h, w)
        pond = ams_color.detect_pond_mask(rgba, dem=dem, sensitivity="conservative")
        self.assertLess(int(np.count_nonzero(pond)), 80)

    def test_pond_coverage_cap_conservative(self) -> None:
        h, w = 80, 80
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = (70, 110, 45)
        rgba[:, :, 3] = 255
        for y in range(4, 72, 8):
            for x in range(4, 72, 8):
                rgba[y : y + 5, x : x + 5, :3] = (12, 22, 38)
        dem = np.full((h, w), 100.0, dtype=np.float64)
        pond = ams_color.detect_pond_mask(rgba, dem=dem, sensitivity="conservative")
        foot = float(h * w)
        self.assertLessEqual(float(np.count_nonzero(pond)) / foot, 0.16)

    def test_aggressive_detects_more_than_conservative_on_muddy_low(self) -> None:
        h, w = 48, 48
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = (88, 105, 62)
        rgba[:, :, 3] = 255
        rgba[16:30, 16:30, :3] = (95, 100, 78)
        dem = np.full((h, w), 120.0, dtype=np.float64)
        dem[16:30, 16:30] = 118.2
        conservative = int(np.count_nonzero(
            ams_color.detect_pond_mask(rgba, dem=dem, sensitivity="conservative")
        ))
        aggressive = int(np.count_nonzero(
            ams_color.detect_pond_mask(rgba, dem=dem, sensitivity="aggressive")
        ))
        self.assertGreaterEqual(aggressive, conservative)

    def test_clamp_ams_n_colors_caps_at_eight(self) -> None:
        self.assertEqual(ams_color.clamp_ams_n_colors(4), 4)
        self.assertEqual(ams_color.clamp_ams_n_colors(8), 8)
        self.assertEqual(ams_color.clamp_ams_n_colors(16), 8)
        self.assertEqual(ams_color.clamp_ams_n_colors(99), 8)

    def test_recommend_palette_respects_max_eight(self) -> None:
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[:, :, :3] = (70, 110, 45)
        rgba[:, :, 3] = 255
        palette = ams_color.recommend_ams_palette(rgba, n_colors=16)
        self.assertLessEqual(len(palette), ams_color.AMS_MAX_COLORS)

    def test_bright_low_elevation_lake_detected_conservative(self) -> None:
        h, w = 80, 80
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = (92, 118, 58)
        rgba[:, :, 3] = 255
        rgba[10:65, 45:72, :3] = (210, 205, 198)
        dem = np.linspace(185.0, 200.0, h * w, dtype=np.float64).reshape(h, w)
        dem[10:65, 45:72] = 178.0
        pond = ams_color.detect_pond_mask(rgba, dem=dem, sensitivity="conservative")
        lake_px = int(np.count_nonzero(pond[10:65, 45:72]))
        self.assertGreater(lake_px, 400)

    def test_recommend_palette_four_colors(self) -> None:
        rgba = np.zeros((32, 32, 4), dtype=np.uint8)
        rgba[:16, :16, :3] = (40, 95, 143)
        rgba[:16, 16:, :3] = (27, 77, 42)
        rgba[16:, :16, :3] = (90, 160, 70)
        rgba[16:, 16:, :3] = (130, 130, 130)
        rgba[:, :, 3] = 255
        palette = ams_color.recommend_ams_palette(rgba, n_colors=4)
        self.assertEqual(len(palette), 4)
        self.assertEqual(next(p for p in palette if p["role"] == "base")["index"], 0)
        land = [p for p in palette if p.get("role") == "land"]
        self.assertGreaterEqual(len(land), 2)
        base_rgb = np.array(next(p for p in palette if p["role"] == "base")["rgb"])
        for p in land:
            dist = float(np.sum((np.array(p["rgb"], dtype=np.float64) - base_rgb) ** 2))
            self.assertGreaterEqual(dist, ams_color.LAND_ACCENT_MIN_SEP_SQ * 0.5)
        for p in palette:
            self.assertTrue(str(p["hex"]).startswith("#"))
            self.assertEqual(len(p["rgb"]), 3)
            self.assertTrue(str(p["part_name"]))

    def test_render_quantized_preview_transparent_outside_footprint(self) -> None:
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[5:15, 5:15, :3] = (100, 120, 80)
        rgba[5:15, 5:15, 3] = 255
        palette = ams_color.recommend_ams_palette(rgba, n_colors=4)
        palette_rgb = np.array([p["rgb"] for p in palette], dtype=np.uint8)
        index_image = ams_color.quantize_texture_index_image(rgba, palette_rgb)
        footprint = rgba[:, :, 3] >= ams_color.MASK_ALPHA_MIN
        # Outside footprint uses index 0 (base); preview must still be transparent there.
        self.assertTrue(np.any((~footprint) & (index_image == 0)))
        preview = ams_color.render_quantized_preview(
            index_image, palette_rgb, footprint=footprint
        )
        alpha = np.array(preview)[:, :, 3]
        self.assertTrue(np.all(alpha[~footprint] == 0))
        self.assertTrue(np.all(alpha[footprint] == 255))

    def test_barycentric_uv_samples_stay_in_unit_square(self) -> None:
        """Interior UV samples must be convex combinations of corner UVs (not summed corners)."""
        corners = np.array(
            [
                [[0.1, 0.2], [0.4, 0.3], [0.2, 0.5]],
                [[0.6, 0.7], [0.8, 0.75], [0.65, 0.9]],
            ],
            dtype=np.float64,
        )
        sample_uv = np.einsum(
            "fcx,sc->fsx", corners, ams_color._FACE_LABEL_BARY_WEIGHTS
        )
        self.assertLessEqual(float(sample_uv[..., 0].max()), 0.81)
        self.assertGreaterEqual(float(sample_uv[..., 0].min()), 0.1)
        self.assertLessEqual(float(sample_uv[..., 1].max()), 0.9)
        self.assertGreaterEqual(float(sample_uv[..., 1].min()), 0.2)

    def test_label_and_split(self) -> None:
        mesh, poly, rgba = _synthetic_colored_terrain(16, 16)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=80.0, center_on_bed=True, voxel_size_mm=None
        )
        palette = ams_color.recommend_ams_palette(rgba, n_colors=4)
        palette_rgb = np.array([p["rgb"] for p in palette], dtype=np.uint8)
        index_image = ams_color.quantize_texture_index_image(rgba, palette_rgb)
        mesh_uv = mesh_mod.print_solid_with_satellite_uv(solid, surf)
        base_index = next(int(p["index"]) for p in palette if p.get("role") == "base")
        is_top = ams_color.top_face_mask(solid, surf)
        labels = ams_color.label_faces_by_palette(
            mesh_uv,
            index_image,
            is_top=is_top,
            base_index=base_index,
        )
        parts = ams_color.split_solid_by_labels(
            solid, labels, palette, process_mesh=mesh_mod.solid_process_for_export
        )
        self.assertGreaterEqual(len(parts), 2)
        self.assertEqual(len(labels), len(solid.faces))
        used = {int(x) for x in np.unique(labels)}
        self.assertGreaterEqual(len(used), 3)
        labeled = sum(len(m.faces) for _p, m in parts)
        self.assertGreater(labeled, 0)

    def test_base_and_walls_use_base_slot(self) -> None:
        mesh, poly, rgba = _synthetic_colored_terrain(16, 16)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=80.0, center_on_bed=True, voxel_size_mm=None
        )
        _palette, _parts, _index_image, face_labels = ams_color.build_ams_parts(
            solid,
            surf,
            rgba,
            print_solid_with_satellite_uv=mesh_mod.print_solid_with_satellite_uv,
            process_mesh=mesh_mod.solid_process_for_export,
            n_colors=4,
        )
        base_index = 0
        is_top = ams_color.top_face_mask(solid, surf)
        self.assertGreater(int(is_top.sum()), 0)
        self.assertGreater(int((~is_top).sum()), 0)
        self.assertTrue(np.all(face_labels[~is_top] == base_index))

    def test_walls_use_base_and_palette_has_four_slots(self) -> None:
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        rgba[:, :, :3] = (70, 110, 45)
        rgba[:, :, 3] = 255
        rgba[2:5, 2:5, :3] = (40, 95, 143)
        dem = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
        tr = rasterio.transform.from_bounds(0, 0, 120, 120, 16, 16)
        poly = box(10, 10, 110, 110)
        mask = rasterize(
            [(poly, 255)], out_shape=(16, 16), transform=tr, fill=0, dtype=np.uint8, all_touched=True
        )
        mesh = mesh_mod.build_mesh(dem, rgba[:, :, :3], tr, mask=mask, poly_utm=poly)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=80.0, center_on_bed=True, voxel_size_mm=None
        )
        palette, _index_image, face_labels = ams_color.build_ams_labels(
            solid,
            surf,
            rgba,
            print_solid_with_satellite_uv=mesh_mod.print_solid_with_satellite_uv,
            n_colors=4,
        )
        self.assertEqual(len(palette), 4)
        base_index = next(int(p["index"]) for p in palette if p.get("role") == "base")
        is_top = ams_color.top_face_mask(solid, surf)
        self.assertTrue(np.all(face_labels[~is_top] == base_index))
        used_top = {int(x) for x in np.unique(face_labels[is_top])}
        self.assertGreaterEqual(len(used_top), 2)

    def test_bambu_obj_mtl_structure(self) -> None:
        mesh, poly, rgba = _synthetic_colored_terrain(14, 14)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=90.0, center_on_bed=True, voxel_size_mm=None
        )
        zip_bytes, meta, _preview, _labels, _palette, _ams_mesh = mesh_mod.export_bambu_ams_color_package(solid, surf, rgba)
        self.assertTrue(meta.get("ok"))
        self.assertEqual(int(meta["open_edge_count"]), 0)
        self.assertEqual(int(meta["non_manifold_edge_count"]), 0)
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
        names = zf.namelist()
        self.assertIn("terrain_ams.obj", names)
        self.assertIn("terrain_ams.mtl", names)
        self.assertIn("palette.json", names)
        obj = zf.read("terrain_ams.obj").decode("utf-8")
        mtl = zf.read("terrain_ams.mtl").decode("utf-8")
        self.assertIn("usemtl 01_base", obj)
        self.assertIn("newmtl 01_base", mtl)
        self.assertIn("usemtl", obj)
        self.assertIn("newmtl", mtl)
        self.assertGreaterEqual(int(meta.get("part_count") or 0), 1)
        self.assertGreaterEqual(len(meta.get("colors") or []), 4)

    def test_ams_obj_matches_solid_topology(self) -> None:
        mesh, poly, rgba = _synthetic_colored_terrain(16, 16)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=90.0, center_on_bed=True, voxel_size_mm=None
        )
        zip_bytes, meta, _preview, _labels, _palette, _ams_mesh = mesh_mod.export_bambu_ams_color_package(
            solid, surf, rgba
        )
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
        obj_text = zf.read("terrain_ams.obj").decode("utf-8")
        verts: list[list[float]] = []
        faces: list[list[int]] = []
        for line in obj_text.splitlines():
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        combined = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        self.assertEqual(mesh_mod.open_edge_count(combined), mesh_mod.open_edge_count(solid))
        self.assertEqual(
            mesh_mod.non_manifold_edge_count(combined), mesh_mod.non_manifold_edge_count(solid)
        )
        self.assertEqual(int(meta["open_edge_count"]), mesh_mod.open_edge_count(solid))
        self.assertEqual(
            int(meta["non_manifold_edge_count"]),
            mesh_mod.non_manifold_edge_count(solid),
        )

    def test_ams_3mf_zip_structure(self) -> None:
        paths = mesh_mod.generate_bambu_ams_fixtures(
            str(Path(__file__).resolve().parent / "fixtures" / "bambu_ams")
        )
        raw = paths["colored_3mf"].read_bytes()
        zf = zipfile.ZipFile(io.BytesIO(raw), "r")
        xml = zf.read("3D/3dmodel.model").decode("utf-8")
        self.assertGreaterEqual(xml.count("<object"), 4)
        self.assertGreaterEqual(xml.count("colorgroup"), 4)
        self.assertIn("<build>", xml)

    def test_ams_export_separate_from_core_3mf(self) -> None:
        mesh, poly, rgba = _synthetic_colored_terrain(12, 12)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=70.0, center_on_bed=True, voxel_size_mm=None
        )
        core = mesh_mod.export_print_3mf(solid)
        zf = zipfile.ZipFile(io.BytesIO(core), "r")
        xml = zf.read("3D/3dmodel.model").decode("utf-8")
        self.assertNotIn("colorgroup", xml)
        ams_zip, meta, _preview, _labels, _palette, _ams_mesh = mesh_mod.export_bambu_ams_color_package(solid, surf, rgba)
        self.assertTrue(meta.get("ok"))
        self.assertNotEqual(core, ams_zip)

    def test_ams_labels_survive_voxel_repair(self) -> None:
        """Voxel-repaired print solids must still get multi-color top-surface labels."""
        mesh, poly, rgba = _synthetic_colored_terrain(20, 20)
        solid, meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=90.0, center_on_bed=True, voxel_size_mm=None
        )
        vs = float(meta["print_voxel_size_mm"])
        vox = mesh_mod._voxel_merge_filled(
            solid, vs, print_max_size_mm=float(meta["print_max_size_mm"])
        )
        vox, _dec = mesh_mod.decimate_for_ams(vox, "medium")
        is_top = ams_color.top_face_mask(vox, surf, voxel_size_mm=vs)
        self.assertGreater(int(is_top.sum()), 0)
        palette, _index_image, face_labels = ams_color.build_ams_labels(
            vox,
            surf,
            rgba,
            print_solid_with_satellite_uv=mesh_mod.print_solid_with_satellite_uv,
            voxel_size_mm=vs,
        )
        used_top = {int(x) for x in np.unique(face_labels[is_top])}
        self.assertGreaterEqual(len(used_top), 1)
        active = [p for p in palette if int(p.get("triangle_count") or 0) > 0]
        self.assertGreaterEqual(len(active), 1)
        self.assertEqual(len(face_labels), len(vox.faces))


if __name__ == "__main__":
    unittest.main()
