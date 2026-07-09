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
from terrain_app import pond_shapes
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


PRINT_CACHE_GLB = "print_cache.glb"
POND_SHAPES_JSON = "pond_shapes.json"


def _save_print_cache(job_dir: Path, print_mesh: trimesh.Trimesh, surf_mm: trimesh.Trimesh) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(print_mesh, geom_name="print_mesh")
    scene.add_geometry(surf_mm, geom_name="surf_mm")
    scene.export(job_dir / PRINT_CACHE_GLB)


def _load_print_cache(job_dir: Path) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    path = job_dir / PRINT_CACHE_GLB
    if not path.is_file():
        raise FileNotFoundError("print_cache.glb")
    loaded = trimesh.load(path, force="scene")
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError("print_cache.glb is not a scene")
    # trimesh preserves geom_name on export/import
    if "print_mesh" not in loaded.geometry or "surf_mm" not in loaded.geometry:
        names = list(loaded.geometry.keys())
        if len(names) < 2:
            raise ValueError("print_cache.glb missing print_mesh or surf_mm")
        print_mesh = loaded.geometry[names[0]]
        surf_mm = loaded.geometry[names[1]]
    else:
        print_mesh = loaded.geometry["print_mesh"]
        surf_mm = loaded.geometry["surf_mm"]
    if not isinstance(print_mesh, trimesh.Trimesh) or not isinstance(surf_mm, trimesh.Trimesh):
        raise ValueError("print_cache geometries must be meshes")
    return print_mesh, surf_mm


def _prepare_pond_shapes_for_job(
    job_dir: Path,
    job_id: str,
    texture_rgba: np.ndarray,
    dem: np.ndarray,
    *,
    pond_sensitivity: str,
    grid_width: int,
    grid_height: int,
) -> dict[str, Any]:
    pond_mask_auto = ams_color.detect_pond_mask(
        texture_rgba, dem=dem, sensitivity=pond_sensitivity
    )
    shapes = pond_shapes.mask_to_polygons(pond_mask_auto)
    pond_doc: dict[str, Any] = {
        "shapes": shapes,
        "auto_count": len(shapes),
        "sensitivity": pond_sensitivity,
        "grid_width": int(grid_width),
        "grid_height": int(grid_height),
    }
    _save_json(job_dir / POND_SHAPES_JSON, pond_doc)
    return {
        "status": "pending_edit",
        "shape_count": len(shapes),
        "url": f"/api/result/{job_id}/pond_shapes.json",
    }


def build_ams_from_job(
    job_dir: Path,
    shapes: list[dict[str, Any]],
    *,
    report: Callable[[str, str, int], None] | None = None,
) -> None:
    """Build AMS exports from cached print solid and user-approved pond polygons."""
    log = logging.getLogger("terrain_app.pipeline")
    meta_path = job_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job_id = str(meta.get("job_id") or job_dir.name)
    if report is None:
        report = progress_reporter(job_dir)

    h = int(meta["grid_height"])
    w = int(meta["grid_width"])
    pond_sensitivity = ams_color.normalize_pond_sensitivity(
        str(meta.get("pond_sensitivity") or "conservative")
    )
    ams_n_colors = ams_color.clamp_ams_n_colors(int(meta.get("ams_n_colors") or 4))
    ams_quality = normalize_ams_quality(str(meta.get("ams_quality") or "medium"))
    exports_req = meta.get("exports", {}).get("requested", {})
    exports_built = meta.setdefault("exports", {}).setdefault("built", {})

    report("build_ams", "Loading cached print solid…", 10)
    print_mesh, surf_mm = _load_print_cache(job_dir)
    texture_rgba = np.array(Image.open(job_dir / "texture.png"), dtype=np.uint8)
    dem = np.load(job_dir / "dem.npy")

    footprint = pond_shapes.footprint_from_rgba(texture_rgba)
    validated = pond_shapes.validate_shapes(shapes, h, w)
    pond_mask = pond_shapes.polygons_to_mask(validated, h, w, footprint)

    pond_doc: dict[str, Any] = {
        "shapes": validated,
        "auto_count": len([s for s in validated if s.get("source") == "auto"]),
        "sensitivity": pond_sensitivity,
        "grid_width": w,
        "grid_height": h,
    }
    _save_json(job_dir / POND_SHAPES_JSON, pond_doc)

    print_info = meta.setdefault("print", {})
    ams_t0 = time.time()

    def _ams_progress(msg: str) -> None:
        elapsed = int(time.time() - ams_t0)
        suffix = f" ({elapsed}s)" if elapsed >= 8 else ""
        report("build_ams", f"{msg}{suffix}", 55)

    _ams_progress(f"Building {ams_n_colors}-color Bambu AMS export…")
    ams_mesh: trimesh.Trimesh | None = None
    ams_bytes, ams_meta, ams_preview, ams_labels, ams_palette, ams_mesh = (
        mesh_mod.export_bambu_ams_color_package(
            print_mesh,
            surf_mm,
            texture_rgba,
            dem=dem,
            pond_sensitivity=pond_sensitivity,
            pond_mask=pond_mask,
            n_colors=ams_n_colors,
            voxel_size_mm=print_info.get("print_voxel_size_mm"),
            ams_quality=ams_quality,
            on_progress=_ams_progress,
        )
    )
    if exports_req.get("print_ams", True):
        ams_filename = str(ams_meta.get("filename") or "terrain_print_ams_obj.zip")
        (job_dir / ams_filename).write_bytes(ams_bytes)
        ams_preview.save(job_dir / "terrain_print_ams_preview.png", format="PNG")
        exports_built["print_ams"] = True
    del ams_bytes, ams_preview

    print_info["ams"] = ams_meta
    print_info["ams"]["print_ams_glb"] = False
    exports_built["print_ams_glb"] = False
    if exports_req.get("print_ams_glb") and ams_mesh is not None:
        try:
            (job_dir / "terrain_print_ams.glb").write_bytes(
                mesh_mod.export_ams_print_glb_labeled(ams_mesh, ams_labels, ams_palette)
            )
            print_info["ams"]["print_ams_glb"] = True
            exports_built["print_ams_glb"] = True
        except Exception:
            log.exception("AMS print GLB build failed")
    del ams_labels, ams_palette, ams_mesh

    meta["ponds"] = {
        "status": "exported",
        "shape_count": len(validated),
        "url": f"/api/result/{job_id}/pond_shapes.json",
    }
    _save_json(meta_path, meta)
    report("build_ams", "AMS export ready", 100)


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
                clip_poly_utm,
                print_max_size_mm=pms,
                base_extrusion_mm=pbe,
                center_on_bed=print_center_on_bed,
                voxel_size_mm=print_voxel_size_mm,
                print_split_nx=spx,
                print_split_nz=spz,
            )
            # Keep dem and texture_rgba until AMS export (pond detection) finishes below.
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
                report("ponds", "Detecting ponds for map review…", 94)
                try:
                    _save_print_cache(job_dir, print_mesh, surf_mm)
                    ponds_meta = _prepare_pond_shapes_for_job(
                        job_dir,
                        job_id,
                        texture_rgba,
                        dem,
                        pond_sensitivity=pond_sensitivity,
                        grid_width=dst_width,
                        grid_height=dst_height,
                    )
                    meta["ponds"] = ponds_meta
                    print_info["ams"] = {"ok": False, "pending": True}
                except Exception:
                    log.exception("Pond preparation for AMS review failed")
                    print_info["ams"] = {"ok": False, "error": "pond_prepare_failed"}
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
    if exports.needs_ams() and meta.get("ponds", {}).get("status") == "pending_edit":
        write_progress(
            job_dir,
            status="done",
            step="ponds",
            message="Review ponds on the map, then build AMS export",
            percent=100,
        )
    else:
        write_progress(job_dir, status="done", step="done", message="Ready", percent=100)
    return job_id


def load_meta(cache_root: Path, job_id: str) -> Dict[str, Any]:
    p = cache_root / job_id / "meta.json"
    if not p.is_file():
        raise FileNotFoundError(job_id)
    return json.loads(p.read_text(encoding="utf-8"))
