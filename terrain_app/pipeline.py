"""End-to-end: KML → 3DEP + imagery → aligned rasters → mesh assets."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

from terrain_app import crs as crs_mod
from terrain_app import dem as dem_mod
from terrain_app import imagery as imagery_mod
from terrain_app import kml as kml_mod
from terrain_app import mesh as mesh_mod

ImageryKind = Literal["osm", "oam", "esri"]


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def process_kml(
    kml_bytes: bytes,
    cache_root: Path,
    imagery: ImageryKind = "osm",
    grid_size: int = 512,
    buffer_m: float = 50.0,
    vertical_exaggeration: float = 1.0,
    print_max_size_mm: float = 200.0,
    print_base_extrusion_mm: float = 1.0,
    print_center_on_bed: bool = True,
    print_voxel_size_mm: float | None = None,
    print_split_nx: int = 1,
    print_split_nz: int = 1,
) -> str:
    grid_size = int(max(64, min(2048, grid_size)))
    poly_wgs = kml_mod.parse_kml_bytes(kml_bytes)
    epsg = crs_mod.working_crs_epsg(poly_wgs)
    poly_utm = crs_mod.project_polygon(poly_wgs, epsg)
    if not poly_utm.is_valid:
        poly_utm = poly_utm.buffer(0)
    bounds_utm = crs_mod.buffer_bounds_utm(poly_utm, buffer_m)
    west, south, east, north = bounds_utm
    dst_width = grid_size
    aspect = (north - south) / max(east - west, 1e-9)
    dst_height = max(64, int(round(dst_width * aspect)))

    bbox4326 = transform_bounds(f"EPSG:{epsg}", "EPSG:4326", west, south, east, north)

    session = imagery_mod.make_session()
    src_arr, src_transform, src_crs = dem_mod.fetch_dem_raster_4326(
        bbox4326, dst_width, dst_height, session
    )
    dem, dst_transform = dem_mod.warp_dem_to_utm(
        src_arr, src_transform, src_crs, epsg, bounds_utm, dst_width, dst_height
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

    mask = rasterize(
        [(poly_utm, 255)],
        out_shape=dem.shape,
        transform=dst_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    texture_rgba = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.uint8)
    texture_rgba[:, :, :3] = texture
    texture_rgba[:, :, 3] = mask

    job_id = str(uuid.uuid4())
    job_dir = cache_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    np.save(job_dir / "dem.npy", dem.astype(np.float32))
    z_display = mesh_mod.prepare_z(dem, vertical_exaggeration=vertical_exaggeration, z_offset_mode="min")
    np.save(job_dir / "heights_display.npy", z_display)
    Image.fromarray(texture_rgba, mode="RGBA").save(job_dir / "texture.png")

    t = dst_transform
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "epsg": epsg,
        "bounds_utm": list(bounds_utm),
        "bbox4326": list(bbox4326),
        "grid_width": dst_width,
        "grid_height": dst_height,
        "buffer_m": buffer_m,
        "vertical_exaggeration": vertical_exaggeration,
        "imagery": imagery,
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "polygon_geojson_wgs84": kml_mod.polygon_geojson(poly_wgs),
    }
    finite = np.isfinite(dem)
    if finite.any():
        meta["elevation_min_m"] = float(np.nanmin(dem[finite]))
        meta["elevation_max_m"] = float(np.nanmax(dem[finite]))
        meta["full_raster_relief_m"] = float(meta["elevation_max_m"] - meta["elevation_min_m"])
    mesh = mesh_mod.build_mesh(
        dem,
        texture,
        dst_transform,
        vertical_exaggeration=vertical_exaggeration,
        z_offset_mode="min",
        mask=mask,
        poly_utm=poly_utm,
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
    z_for_stats = mesh_mod.prepare_z(
        dem, vertical_exaggeration=vertical_exaggeration, z_offset_mode="min"
    )
    inside = mask >= 128
    if np.any(inside):
        meta["masked_raster_relief_m"] = float(np.ptp(z_for_stats[inside]))
    mv = np.asarray(mesh.vertices, dtype=np.float64)
    if mv.size > 0:
        meta["mesh_vertex_z_span_m"] = float(np.ptp(mv[:, 2]))
    if dem.shape[0] > 1 and dem.shape[1] > 1:
        qmask = mesh_mod.quad_inclusion_mask(
            int(dem.shape[0]), int(dem.shape[1]), dst_transform, poly_utm, mask
        )
        (job_dir / "quad_mask.bin").write_bytes(np.ascontiguousarray(qmask).tobytes())
        meta["quad_mask"] = {"url": f"/api/result/{job_id}/quad_mask.bin"}
    glb = mesh_mod.export_glb(mesh, center_xz=True)
    (job_dir / "terrain.glb").write_bytes(glb)
    zip_bytes = mesh_mod.export_obj_zip(mesh, center_xz=True)
    (job_dir / "terrain_obj.zip").write_bytes(zip_bytes)
    pms = float(max(1.0, print_max_size_mm))
    pbe = float(max(0.0, print_base_extrusion_mm))
    log = logging.getLogger("terrain_app.pipeline")
    print_info: Dict[str, Any] = {}
    spx = int(max(1, print_split_nx))
    spz = int(max(1, print_split_nz))
    try:
        print_mesh, print_info, surf_mm = mesh_mod.build_print_solid(
            mesh,
            poly_utm,
            print_max_size_mm=pms,
            base_extrusion_mm=pbe,
            center_on_bed=print_center_on_bed,
            voxel_size_mm=print_voxel_size_mm,
        )
        pmb = print_mesh.bounds
        if pmb is not None:
            print_info["print_vertical_span_mm"] = float(pmb[1, 2] - pmb[0, 2])
        (job_dir / "terrain_print.stl").write_bytes(mesh_mod.export_print_stl(print_mesh))
        (job_dir / "terrain_print.glb").write_bytes(mesh_mod.export_print_glb(print_mesh))
        mesh_3mf = mesh_mod.print_solid_with_satellite_uv(print_mesh, surf_mm)
        raw_3mf = mesh_mod.export_print_3mf(mesh_3mf)
        (job_dir / "terrain_print.3mf").write_bytes(raw_3mf)
        print_info["print_3mf_textured"] = bool(
            mesh_3mf.visual is not None
            and mesh_3mf.visual.defined
            and getattr(mesh_3mf.visual, "kind", None) == "texture"
        )
        try:
            reloaded = mesh_mod.mesh_from_3mf_bytes(raw_3mf)
            print_info["print_3mf_roundtrip_non_manifold_edge_count"] = int(
                mesh_mod.non_manifold_edge_count(reloaded)
            )
        except Exception:
            log.exception("3MF round-trip topology check failed")
            print_info["print_3mf_roundtrip_non_manifold_edge_count"] = None
        print_info["ok"] = True
        print_info["export_frame"] = "slicer_z_up"
        print_info["export_axes"] = "X=east_mm, Y=north_mm, Z=elevation_mm (XY build plate, Z up)"
        print_info["split_nx"] = spx
        print_info["split_nz"] = spz
        fsm = print_info.get("print_full_size_mm", {})
        fx = float(fsm.get("x", 0) or 0) if isinstance(fsm, dict) else 0.0
        fy = float(fsm.get("y", 0) or fsm.get("z", 0) or 0) if isinstance(fsm, dict) else 0.0
        if spx * spz > 1 and fx > 0 and fy > 0:
            try:
                pieces = mesh_mod.split_solid_to_xz_grid(print_mesh, spx, spz)
                if len(pieces) > 0:
                    (job_dir / "terrain_print_pieces.zip").write_bytes(
                        mesh_mod.export_print_pieces_stl_bytes(pieces)
                    )
                print_info["pieces"] = {
                    "ok": bool(len(pieces) > 0),
                    "count": len(pieces),
                    "expected": spx * spz,
                    "per_piece_size_mm": {
                        "x": fx / spx,
                        "y": fy / spz,
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
    meta["print"] = {
        "max_size_mm": pms,
        "base_extrusion_mm": pbe,
        "center_on_bed": bool(print_center_on_bed),
    }
    if print_voxel_size_mm is not None:
        meta["print"]["voxel_size_mm_request"] = float(print_voxel_size_mm)
    meta["print"].update(print_info)
    _save_json(job_dir / "meta.json", meta)
    return job_id


def load_meta(cache_root: Path, job_id: str) -> Dict[str, Any]:
    p = cache_root / job_id / "meta.json"
    if not p.is_file():
        raise FileNotFoundError(job_id)
    return json.loads(p.read_text(encoding="utf-8"))
