"""End-to-end: KML → 3DEP + imagery → aligned rasters → mesh assets."""

from __future__ import annotations

import gc
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Literal

import numpy as np
import trimesh
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

from terrain_app import ams_color
from terrain_app import crs as crs_mod
from terrain_app import dem as dem_mod
from terrain_app.export_options import ExportOptions, normalize_ams_quality
from terrain_app import imagery as imagery_mod
from terrain_app import kml as kml_mod
from terrain_app import mesh as mesh_mod
from terrain_app.progress import progress_reporter, write_progress

ImageryKind = Literal["osm", "oam", "esri"]

_EXPORT_BASENAME_MAX = 120


def export_basename_from_kml_filename(filename: str | None) -> str:
    """Safe stem from uploaded KML name for download filenames (not cache paths)."""
    if filename is None or not str(filename).strip():
        return "terrain"
    stem = Path(str(filename)).stem.strip()
    if not stem:
        return "terrain"
    safe = re.sub(r"[^\w.\-]+", "_", stem, flags=re.ASCII)
    safe = safe.strip("._-")
    if not safe:
        return "terrain"
    return safe[:_EXPORT_BASENAME_MAX]


def export_download_filename(meta: Dict[str, Any], suffix: str) -> str:
    """Build a download name from job meta, e.g. ``MyPark_print.stl``."""
    base = str(meta.get("export_basename") or "terrain")
    return f"{base}{suffix}"


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def process_kml(
    kml_bytes: bytes,
    cache_root: Path,
    kml_filename: str | None = None,
    imagery: ImageryKind = "osm",
    grid_size: int = 512,
    buffer_m: float = 50.0,
    vertical_exaggeration: float = 3.0,
    print_max_size_mm: float = 200.0,
    print_base_extrusion_mm: float = 1.0,
    print_center_on_bed: bool = True,
    print_voxel_size_mm: float | None = None,
    print_split_nx: int = 1,
    print_split_nz: int = 1,
    boundary_smooth_m: float | None = None,
    pond_sensitivity: str = "conservative",
    ams_n_colors: int = 4,
    ams_quality: str = "medium",
    export_options: ExportOptions | None = None,
    job_id: str | None = None,
    report: Callable[[str, str, int], None] | None = None,
) -> str:
    grid_size = int(max(64, min(2048, grid_size)))
    job_id = job_id or str(uuid.uuid4())
    job_dir = cache_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    pond_sensitivity = ams_color.normalize_pond_sensitivity(pond_sensitivity)
    ams_n_colors = ams_color.clamp_ams_n_colors(ams_n_colors)
    ams_quality = normalize_ams_quality(ams_quality)
    exports = export_options if export_options is not None else ExportOptions()
    built: dict[str, bool] = {}
    if report is None:
        report = progress_reporter(job_dir)
    report("parse", "Reading KML boundary…", 5)
    poly_wgs = kml_mod.parse_kml_bytes(kml_bytes)
    epsg = crs_mod.working_crs_epsg(poly_wgs)
    poly_utm = crs_mod.project_polygon(poly_wgs, epsg)
    if not poly_utm.is_valid:
        poly_utm = poly_utm.buffer(0)
    bounds_utm = crs_mod.buffer_bounds_utm(poly_utm, buffer_m)
    west, south, east, north = bounds_utm
    dst_width, dst_height = crs_mod.grid_dimensions_from_bounds(bounds_utm, grid_size)

    bbox4326 = transform_bounds(f"EPSG:{epsg}", "EPSG:4326", west, south, east, north)

    session = imagery_mod.make_session()
    dem_msg = f"Downloading USGS 3DEP elevation ({dst_width}×{dst_height})…"
    report("dem", dem_msg, 15)

    def _dem_wait(elapsed_sec: int) -> None:
        report("dem", f"{dem_msg} ({elapsed_sec}s)", 15)

    src_arr, src_transform, src_crs = dem_mod.fetch_dem_raster_4326(
        bbox4326, dst_width, dst_height, session, on_wait=_dem_wait
    )
    report("dem_warp", "Projecting elevation to UTM…", 25)
    dem, dst_transform = dem_mod.warp_dem_to_utm(
        src_arr, src_transform, src_crs, epsg, bounds_utm, dst_width, dst_height
    )
    del src_arr
    gc.collect()

    imagery_labels = {"oam": "OpenAerialMap", "esri": "Esri satellite", "osm": "OpenStreetMap"}
    report(
        "imagery",
        f"Fetching {imagery_labels.get(imagery, 'imagery')} tiles…",
        40,
    )
    if imagery == "oam":
        texture = imagery_mod.fetch_texture_oam(
            bbox4326, epsg, bounds_utm, dst_width, dst_height, session
        )
    elif imagery == "esri":
        texture = imagery_mod.fetch_texture_esri(
            bbox4326, epsg, bounds_utm, dst_width, dst_height, session
        )
    else:
        texture = imagery_mod.fetch_texture_osm(
            bbox4326, epsg, bounds_utm, dst_width, dst_height, session
        )

    report("clip", "Preparing boundary mask…", 52)
    mask, clip_poly_utm, smooth_m_used = mesh_mod.resolve_boundary_clipping(
        poly_utm,
        dem.shape,
        dst_transform,
        bounds_utm,
        dst_width,
        dst_height,
        boundary_smooth_m,
    )
    texture_rgba = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.uint8)
    texture_rgba[:, :, :3] = texture
    texture_rgba[:, :, 3] = mask
    del texture

    job_dir.mkdir(parents=True, exist_ok=True)
    np.save(job_dir / "dem.npy", dem.astype(np.float32))
    z_display = mesh_mod.prepare_z(dem, vertical_exaggeration=vertical_exaggeration, z_offset_mode="min")
    np.save(job_dir / "heights_display.npy", z_display)
    Image.fromarray(texture_rgba, mode="RGBA").save(job_dir / "texture.png")

    t = dst_transform
    export_base = export_basename_from_kml_filename(kml_filename)
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "kml_filename": str(kml_filename) if kml_filename else None,
        "export_basename": export_base,
        "epsg": epsg,
        "bounds_utm": list(bounds_utm),
        "bbox4326": list(bbox4326),
        "grid_width": dst_width,
        "grid_height": dst_height,
        "buffer_m": buffer_m,
        "boundary_smooth_m": smooth_m_used,
        "vertical_exaggeration": vertical_exaggeration,
        "pond_sensitivity": pond_sensitivity,
        "ams_n_colors": ams_n_colors,
        "ams_quality": ams_quality,
        "imagery": imagery,
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "polygon_geojson_wgs84": kml_mod.polygon_geojson(poly_wgs),
    }
    finite = np.isfinite(dem)
    if finite.any():
        meta["elevation_min_m"] = float(np.nanmin(dem[finite]))
        meta["elevation_max_m"] = float(np.nanmax(dem[finite]))
        meta["full_raster_relief_m"] = float(meta["elevation_max_m"] - meta["elevation_min_m"])
    report("mesh", f"Building terrain mesh ({dst_width}×{dst_height})…", 62)
    mesh = mesh_mod.build_mesh(
        dem,
        texture_rgba[:, :, :3],
        dst_transform,
        vertical_exaggeration=vertical_exaggeration,
        z_offset_mode="min",
        mask=mask,
        poly_utm=clip_poly_utm,
    )
    mb = mesh.bounds
    if mb is not None:
        meta["clipped_surface_relief_m"] = float(mb[1, 2] - mb[0, 2])
    # DEM → prepare_z (× vertical_exaggeration) is stored on mesh vertex **Z** (index 2).
    meta["elevation_axis"] = {
        "dem_to_mesh_column": 2,
        "mesh_axis_name": "Z",
        "description": "X=UTM easting, Y=UTM northing, Z=elevation (m); print files use the same Z-up frame in mm",
    }
    z_for_stats = z_display
    inside = mask >= 128
    if np.any(inside):
        meta["masked_raster_relief_m"] = float(np.ptp(z_for_stats[inside]))
    mv = np.asarray(mesh.vertices, dtype=np.float64)
    if mv.size > 0:
        meta["mesh_vertex_z_span_m"] = float(np.ptp(mv[:, 2]))
    report("preview", "Writing optional preview exports…", 72)
    if exports.quad_mask and dem.shape[0] > 1 and dem.shape[1] > 1:
        qmask = mesh_mod.quad_inclusion_mask(
            int(dem.shape[0]), int(dem.shape[1]), dst_transform, clip_poly_utm, mask
        )
        (job_dir / "quad_mask.bin").write_bytes(np.ascontiguousarray(qmask).tobytes())
        meta["quad_mask"] = {"url": f"/api/result/{job_id}/quad_mask.bin"}
        built["quad_mask"] = True
    else:
        built["quad_mask"] = False
    if exports.preview_glb:
        glb = mesh_mod.export_glb(mesh, center_xz=True)
        (job_dir / "terrain.glb").write_bytes(glb)
        del glb
        built["preview_glb"] = True
    else:
        built["preview_glb"] = False
    if exports.preview_obj:
        zip_bytes = mesh_mod.export_obj_zip(mesh, center_xz=True)
        (job_dir / "terrain_obj.zip").write_bytes(zip_bytes)
        del zip_bytes
        built["preview_obj"] = True
    else:
        built["preview_obj"] = False
    pms = float(max(1.0, print_max_size_mm))
    pbe = float(max(0.0, print_base_extrusion_mm))
    log = logging.getLogger("terrain_app.pipeline")
    print_info: Dict[str, Any] = {}
    spx = int(max(1, print_split_nx))
    spz = int(max(1, print_split_nz))
    built["print_stl"] = False
    built["print_3mf"] = False
    built["print_textured_glb"] = False
    built["print_ams"] = False
    built["print_ams_glb"] = False
    built["print_pieces"] = False
    if exports.needs_print_solid():
        try:
            report("print", "Building watertight print solid…", 85)
            print_mesh, print_info, surf_mm = mesh_mod.build_print_solid(
                mesh,
                poly_utm,
                print_max_size_mm=pms,
                base_extrusion_mm=pbe,
                center_on_bed=print_center_on_bed,
                voxel_size_mm=print_voxel_size_mm,
                print_split_nx=spx,
                print_split_nz=spz,
            )
            del mesh, z_display, mask
            gc.collect()
            pmb = print_mesh.bounds
            if pmb is not None:
                print_info["print_vertical_span_mm"] = float(pmb[1, 2] - pmb[0, 2])
            report("print_export", "Writing print exports…", 93)
            if exports.print_stl:
                (job_dir / "terrain_print.stl").write_bytes(
                    mesh_mod.export_print_stl(print_mesh)
                )
                built["print_stl"] = True
            if exports.print_3mf:
                (job_dir / "terrain_print.3mf").write_bytes(
                    mesh_mod.export_print_3mf(print_mesh)
                )
                built["print_3mf"] = True
            mesh_uv = mesh_mod.print_solid_with_satellite_uv(print_mesh, surf_mm)
            print_info["print_textured_glb"] = False
            if exports.print_textured_glb and mesh_mod.mesh_has_texture_visual_for_3mf(mesh_uv):
                try:
                    (job_dir / "terrain_print_textured.glb").write_bytes(
                        mesh_mod.export_textured_print_glb(mesh_uv)
                    )
                    print_info["print_textured_glb"] = True
                    built["print_textured_glb"] = True
                except Exception:
                    log.exception("textured print GLB build failed")
            if exports.needs_ams():
                ams_t0 = time.time()

                def _ams_progress(msg: str) -> None:
                    elapsed = int(time.time() - ams_t0)
                    suffix = f" ({elapsed}s)" if elapsed >= 8 else ""
                    report("print_ams", f"{msg}{suffix}", 95)

                _ams_progress(f"Building {ams_n_colors}-color Bambu AMS export…")
                try:
                    ams_mesh: trimesh.Trimesh | None = None
                    ams_bytes, ams_meta, ams_preview, ams_labels, ams_palette, ams_mesh = (
                        mesh_mod.export_bambu_ams_color_package(
                            print_mesh,
                            surf_mm,
                            texture_rgba,
                            dem=dem,
                            pond_sensitivity=pond_sensitivity,
                            n_colors=ams_n_colors,
                            voxel_size_mm=print_info.get("print_voxel_size_mm"),
                            ams_quality=ams_quality,
                            on_progress=_ams_progress,
                        )
                    )
                    if exports.print_ams:
                        ams_filename = str(
                            ams_meta.get("filename") or "terrain_print_ams_obj.zip"
                        )
                        (job_dir / ams_filename).write_bytes(ams_bytes)
                        ams_preview.save(
                            job_dir / "terrain_print_ams_preview.png", format="PNG"
                        )
                        built["print_ams"] = True
                    del ams_bytes, ams_preview
                    print_info["ams"] = ams_meta
                    print_info["ams"]["print_ams_glb"] = False
                    if exports.print_ams_glb and ams_mesh is not None:
                        try:
                            (job_dir / "terrain_print_ams.glb").write_bytes(
                                mesh_mod.export_ams_print_glb_labeled(
                                    ams_mesh, ams_labels, ams_palette
                                )
                            )
                            print_info["ams"]["print_ams_glb"] = True
                            built["print_ams_glb"] = True
                        except Exception:
                            log.exception("AMS print GLB build failed")
                    del ams_labels, ams_palette, ams_mesh
                except Exception:
                    log.exception("Bambu AMS color export failed")
                    print_info["ams"] = {"ok": False, "error": "ams_export_failed"}
            del mesh_uv, texture_rgba, dem
            gc.collect()
            print_info["ok"] = True
            print_info["export_frame"] = "slicer_z_up"
            print_info["export_axes"] = (
                "X=east_mm, Y=north_mm, Z=elevation_mm (XY build plate, Z up)"
            )
            print_info["split_nx"] = spx
            print_info["split_nz"] = spz
            fsm = print_info.get("print_full_size_mm", {})
            fx = float(fsm.get("x", 0) or 0) if isinstance(fsm, dict) else 0.0
            fy = float(fsm.get("y", 0) or fsm.get("z", 0) or 0) if isinstance(fsm, dict) else 0.0
            if exports.print_pieces and spx * spz > 1 and fx > 0 and fy > 0:
                try:
                    pieces = mesh_mod.split_solid_to_xz_grid(
                        print_mesh, spx, spz, export_basename=export_base
                    )
                    if len(pieces) > 0:
                        (job_dir / "terrain_print_pieces.zip").write_bytes(
                            mesh_mod.export_print_pieces_stl_bytes(pieces)
                        )
                        built["print_pieces"] = True
                    cell_x = fx / float(spx) if spx > 0 else 0.0
                    cell_y = fy / float(spz) if spz > 0 else 0.0
                    print_info["pieces"] = {
                        "ok": bool(len(pieces) > 0),
                        "count": len(pieces),
                        "expected": spx * spz,
                        "per_piece_size_mm": {
                            "x": cell_x,
                            "y": cell_y,
                            "max_horizontal_mm_approx": max(cell_x, cell_y),
                        },
                    }
                except Exception:
                    log.exception("terrain_print_pieces.zip build failed")
                    print_info["pieces"] = {"error": "print_pieces_export_failed", "ok": False}
        except Exception:
            log.exception("terrain_print.stl build failed")
            print_info = {
                "error": "print_solid_export_failed",
                "ok": False,
                "split_nx": spx,
                "split_nz": spz,
            }
    else:
        del mesh, z_display, mask, texture_rgba, dem
        gc.collect()
        print_info = {"ok": False, "skipped": True, "reason": "no_print_exports_requested"}
    meta["exports"] = {"requested": exports.to_dict(), "built": built}
    meta["print"] = {
        "max_size_mm": pms,
        "base_extrusion_mm": pbe,
        "center_on_bed": bool(print_center_on_bed),
    }
    if print_voxel_size_mm is not None:
        meta["print"]["voxel_size_mm_request"] = float(print_voxel_size_mm)
    meta["print"].update(print_info)
    _save_json(job_dir / "meta.json", meta)
    write_progress(job_dir, status="done", step="done", message="Ready", percent=100)
    return job_id


def load_meta(cache_root: Path, job_id: str) -> Dict[str, Any]:
    p = cache_root / job_id / "meta.json"
    if not p.is_file():
        raise FileNotFoundError(job_id)
    return json.loads(p.read_text(encoding="utf-8"))
