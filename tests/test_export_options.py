"""Tests for export flags and AMS decimation."""
from __future__ import annotations

import unittest

import numpy as np
import trimesh

from terrain_app import mesh as mesh_mod
from terrain_app.export_options import ExportOptions, parse_export_options


class TestExportOptions(unittest.TestCase):
    def test_defaults_stl_and_ams_on(self) -> None:
        opts = ExportOptions()
        self.assertTrue(opts.print_stl)
        self.assertTrue(opts.print_ams)
        self.assertFalse(opts.preview_glb)
        self.assertTrue(opts.needs_print_solid())

    def test_parse_form_checkboxes(self) -> None:
        opts = parse_export_options(
            {
                "export_preview_glb": "1",
                "export_print_stl": "0",
                "export_print_ams": "1",
            }
        )
        self.assertTrue(opts.preview_glb)
        self.assertFalse(opts.print_stl)
        self.assertTrue(opts.print_ams)

    def test_no_print_exports_skips_solid(self) -> None:
        opts = ExportOptions(
            print_stl=False,
            print_ams=False,
            print_3mf=False,
            print_textured_glb=False,
            print_ams_glb=False,
            print_pieces=False,
        )
        self.assertFalse(opts.needs_print_solid())


class TestDecimateForAms(unittest.TestCase):
    def test_skips_when_already_small(self) -> None:
        solid = trimesh.creation.icosphere(subdivisions=2)
        out, meta = mesh_mod.decimate_for_ams(solid, "medium")
        self.assertTrue(meta["skipped"])
        self.assertEqual(len(out.faces), len(solid.faces))

    def test_reduces_large_mesh(self) -> None:
        solid = trimesh.creation.icosphere(subdivisions=6)
        n_in = len(solid.faces)
        self.assertGreater(n_in, mesh_mod.AMS_QUALITY_FACE_TARGETS["low"])
        out, meta = mesh_mod.decimate_for_ams(solid, "low")
        self.assertFalse(meta["skipped"])
        self.assertLessEqual(len(out.faces), mesh_mod.AMS_QUALITY_FACE_TARGETS["low"] + 50)
        self.assertLess(len(out.faces), n_in)


if __name__ == "__main__":
    unittest.main()
