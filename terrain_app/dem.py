"""Fetch USGS 3DEP elevation via National Map ImageServer."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

USGS_EXPORT = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)


def fetch_dem_raster_4326(
    bbox4326: tuple[float, float, float, float],
    width: int,
    height: int,
    session,
) -> Tuple[np.ndarray, rasterio.Affine, str]:
    minx, miny, maxx, maxy = bbox4326
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "4326",
        "size": f"{width},{height}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "f": "image",
    }
    r = session.get(USGS_EXPORT, params=params, timeout=180)
    r.raise_for_status()
    with MemoryFile(r.content) as mem:
        with mem.open() as src:
            arr = src.read(1).astype(np.float32)
            return arr, src.transform, str(src.crs) if src.crs else "EPSG:4326"


def warp_dem_to_utm(
    src_arr: np.ndarray,
    src_transform,
    src_crs: str,
    dst_epsg: int,
    dst_bounds: tuple[float, float, float, float],
    dst_width: int,
    dst_height: int,
) -> Tuple[np.ndarray, rasterio.Affine]:
    dst_crs = f"EPSG:{dst_epsg}"
    west, south, east, north = dst_bounds
    dst_transform = from_bounds(west, south, east, north, dst_width, dst_height)
    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst, dst_transform
