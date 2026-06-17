"""Fetch USGS 3DEP elevation via National Map ImageServer."""

from __future__ import annotations

import io
import logging
import threading
from typing import Callable, Tuple

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

USGS_EXPORT = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)

log = logging.getLogger(__name__)


def reproject_array(
    source: np.ndarray,
    src_transform,
    src_crs: str,
    dst_width: int,
    dst_height: int,
    dst_transform,
    dst_crs: str,
    *,
    resampling=Resampling.bilinear,
    dst_dtype: np.dtype | type = np.float32,
    dst_nodata: float | None = None,
) -> np.ndarray:
    """Warp a single-band array without NotGeoreferencedWarning from bare ndarray I/O."""
    src_arr = np.asarray(source)
    fill = np.nan if np.issubdtype(np.dtype(dst_dtype), np.floating) else 0
    if dst_nodata is not None:
        fill = dst_nodata
    dst = np.full((dst_height, dst_width), fill, dtype=dst_dtype)
    src_profile = {
        "driver": "MEM",
        "height": int(src_arr.shape[0]),
        "width": int(src_arr.shape[1]),
        "count": 1,
        "dtype": src_arr.dtype,
        "crs": src_crs,
        "transform": src_transform,
    }
    dst_profile = {
        "driver": "MEM",
        "height": int(dst_height),
        "width": int(dst_width),
        "count": 1,
        "dtype": dst_dtype,
        "crs": dst_crs,
        "transform": dst_transform,
    }
    if dst_nodata is not None:
        dst_profile["nodata"] = dst_nodata
    with MemoryFile() as mem_src:
        with mem_src.open(**src_profile) as src_ds:
            src_ds.write(src_arr, 1)
            with MemoryFile() as mem_dst:
                with mem_dst.open(**dst_profile) as dst_ds:
                    reproject(
                        source=rasterio.band(src_ds, 1),
                        destination=rasterio.band(dst_ds, 1),
                        resampling=resampling,
                    )
                    return dst_ds.read(1)


def _read_timeout_sec(width: int, height: int) -> int:
    pixels = int(max(1, width) * max(1, height))
    # ~4 MB at 1024²; allow ~2s per 100k pixels, clamped.
    return int(max(120, min(600, pixels // 50_000)))


def fetch_dem_raster_4326(
    bbox4326: tuple[float, float, float, float],
    width: int,
    height: int,
    session,
    *,
    on_wait: Callable[[int], None] | None = None,
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
    read_timeout = _read_timeout_sec(width, height)
    stop = threading.Event()

    def _heartbeat() -> None:
        if on_wait is None:
            return
        elapsed = 0
        while not stop.wait(8.0):
            elapsed += 8
            try:
                on_wait(elapsed)
            except Exception:
                log.exception("DEM download heartbeat callback failed")

    hb = threading.Thread(target=_heartbeat, daemon=True)
    if on_wait is not None:
        hb.start()
    try:
        log.info(
            "USGS 3DEP exportImage %dx%d (read timeout %ds)",
            width,
            height,
            read_timeout,
        )
        r = session.get(
            USGS_EXPORT,
            params=params,
            timeout=(30, read_timeout),
            stream=True,
        )
        r.raise_for_status()
        content = r.content
    finally:
        stop.set()
        if on_wait is not None:
            hb.join(timeout=0.2)
    if not content or len(content) < 100:
        raise ValueError(f"USGS 3DEP returned empty or tiny response ({len(content)} bytes)")
    with MemoryFile(content) as mem:
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
    dst = reproject_array(
        src_arr,
        src_transform,
        src_crs,
        dst_width,
        dst_height,
        dst_transform,
        dst_crs,
        resampling=Resampling.bilinear,
        dst_dtype=np.float32,
        dst_nodata=np.nan,
    )
    return dst, dst_transform
