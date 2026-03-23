"""End-to-end: KML → 3DEP + imagery → aligned rasters → mesh assets."""

from __future__ import annotations

import json
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
    mesh = mesh_mod.build_mesh(
        dem,
        texture,
        dst_transform,
        vertical_exaggeration=vertical_exaggeration,
        z_offset_mode="min",
        mask=mask,
        poly_utm=poly_utm,
    )
    glb = mesh_mod.export_glb(mesh)
    (job_dir / "terrain.glb").write_bytes(glb)
    zip_bytes = mesh_mod.export_obj_zip(mesh)
    (job_dir / "terrain_obj.zip").write_bytes(zip_bytes)
    _save_json(job_dir / "meta.json", meta)
    return job_id


def load_meta(cache_root: Path, job_id: str) -> Dict[str, Any]:
    p = cache_root / job_id / "meta.json"
    if not p.is_file():
        raise FileNotFoundError(job_id)
    return json.loads(p.read_text(encoding="utf-8"))
