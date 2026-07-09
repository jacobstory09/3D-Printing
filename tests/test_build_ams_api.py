"""Deferred AMS build from pond polygons."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from terrain_app import ams_color
from terrain_app import mesh as mesh_mod
from terrain_app.pipeline import (
    POND_SHAPES_JSON,
    PRINT_CACHE_GLB,
    _load_print_cache,
    _prepare_pond_shapes_for_job,
    _save_print_cache,
    build_ams_from_job,
)


class TestBuildAmsFromJob(unittest.TestCase):
    def test_prepare_pond_shapes_writes_json(self) -> None:
        h, w = 48, 48
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = (70, 110, 45)
        rgba[:, :, 3] = 255
        rgba[18:26, 18:26, :3] = (10, 17, 11)
        dem = np.full((h, w), 100.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            ponds = _prepare_pond_shapes_for_job(
                job_dir,
                "job-1",
                rgba,
                dem,
                pond_sensitivity="conservative",
                grid_width=w,
                grid_height=h,
            )
            self.assertEqual(ponds["status"], "pending_edit")
            doc = json.loads((job_dir / POND_SHAPES_JSON).read_text())
            self.assertGreater(len(doc["shapes"]), 0)

    def test_build_ams_from_job_writes_zip(self) -> None:
        h, w = 32, 32
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = (70, 110, 45)
        rgba[:, :, 3] = 255
        rgba[:16, :16, :3] = (40, 95, 143)
        rgba[16:, 16:, :3] = (130, 130, 130)
        dem = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
        box = trimesh.creation.box(extents=(10, 10, 2))
        surf = box.copy()
        print_mesh = box.copy()

        shapes = [
            {
                "id": "p1",
                "source": "manual",
                "vertices": [[4, 4], [12, 4], [12, 12], [4, 12]],
            }
        ]

        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            Image.fromarray(rgba, mode="RGBA").save(job_dir / "texture.png")
            np.save(job_dir / "dem.npy", dem)
            _save_print_cache(job_dir, print_mesh, surf)
            meta = {
                "job_id": "test-job",
                "grid_width": w,
                "grid_height": h,
                "pond_sensitivity": "conservative",
                "ams_n_colors": 4,
                "ams_quality": "low",
                "exports": {"requested": {"print_ams": True, "print_ams_glb": False}, "built": {}},
                "print": {"print_voxel_size_mm": 2.0},
            }
            (job_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            build_ams_from_job(job_dir, shapes)

            meta_out = json.loads((job_dir / "meta.json").read_text())
            self.assertTrue(meta_out["print"]["ams"]["ok"])
            self.assertEqual(meta_out["ponds"]["status"], "exported")
            self.assertTrue((job_dir / "terrain_print_ams_obj.zip").is_file())
            self.assertTrue((job_dir / "terrain_print_ams_preview.png").is_file())
            roles = {p.get("role") for p in meta_out["print"]["ams"]["colors"]}
            self.assertIn("pond", roles)


class TestPipelineDefer(unittest.TestCase):
    def test_process_kml_defers_inline_ams_export(self) -> None:
        src = Path(__file__).resolve().parents[1] / "terrain_app" / "pipeline.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("_prepare_pond_shapes_for_job", text)
        self.assertIn("build_ams_from_job", text)
        self.assertIn("_save_print_cache", text)
        self.assertNotIn("export_bambu_ams_color_package(", text.split("def process_kml")[1])


    def test_build_ams_labels_after_glb_cache_roundtrip(self) -> None:
        """Deferred AMS path: print_cache.glb keeps UVs but drops texture image."""
        from tests.test_ams_color import _synthetic_colored_terrain

        mesh, poly, rgba = _synthetic_colored_terrain(16, 16)
        solid, _meta, surf = mesh_mod.build_print_solid(
            mesh, poly, print_max_size_mm=80.0, center_on_bed=True, voxel_size_mm=None
        )
        with tempfile.TemporaryDirectory() as td:
            job_dir = Path(td)
            _save_print_cache(job_dir, solid, surf)
            print_mesh, surf_cached = _load_print_cache(job_dir)
            mat = getattr(surf_cached.visual, "material", None)
            self.assertIsNone(getattr(mat, "image", None) if mat is not None else None)
            ams_mesh, _ = mesh_mod.decimate_for_ams(print_mesh, "low")
            _palette, _index_image, face_labels = ams_color.build_ams_labels(
                ams_mesh,
                surf_cached,
                rgba,
                print_solid_with_satellite_uv=mesh_mod.print_solid_with_satellite_uv,
                n_colors=4,
            )
            self.assertGreater(len(np.unique(face_labels)), 1)

    def test_print_solid_uv_without_material_image(self) -> None:
        solid = trimesh.creation.box(extents=(5, 5, 1))
        surf = trimesh.creation.box(extents=(5, 5, 0.5))
        surf.visual = trimesh.visual.TextureVisuals(
            uv=np.linspace(0, 1, len(surf.vertices) * 2).reshape(-1, 2)
        )
        out = mesh_mod.print_solid_with_satellite_uv(solid, surf)
        self.assertIsNotNone(out.visual)
        self.assertIsNotNone(out.visual.uv)
        self.assertEqual(len(out.visual.uv), len(out.vertices))


if __name__ == "__main__":
    unittest.main()
