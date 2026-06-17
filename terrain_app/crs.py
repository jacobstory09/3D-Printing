"""Working CRS: WGS 84 UTM zone from polygon centroid."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    base = 32600 if lat >= 0 else 32700
    return base + zone


def working_crs_epsg(poly_wgs84: Polygon) -> int:
    c = poly_wgs84.centroid
    return utm_epsg_from_lonlat(c.x, c.y)


def project_polygon(poly_wgs84: Polygon, epsg: int) -> Polygon:
    wgs = pyproj.CRS.from_epsg(4326)
    dst = pyproj.CRS.from_epsg(epsg)
    fwd = pyproj.Transformer.from_crs(wgs, dst, always_xy=True).transform
    return transform(fwd, poly_wgs84)


def buffer_bounds_utm(poly_utm: Polygon, buffer_m: float) -> tuple[float, float, float, float]:
    b = poly_utm.buffer(buffer_m).bounds
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def grid_dimensions_from_bounds(
    bounds_utm: tuple[float, float, float, float],
    max_dimension: int,
) -> tuple[int, int]:
    """
    Raster width/height (pixels) with ``max(width, height) <= max_dimension``.

    ``max_dimension`` is the UI "grid size" — longest side of the output grid.
    """
    cap = int(max(64, min(2048, max_dimension)))
    west, south, east, north = bounds_utm
    width_m = max(float(east - west), 1e-9)
    height_m = max(float(north - south), 1e-9)
    aspect = height_m / width_m
    if aspect >= 1.0:
        dst_height = cap
        dst_width = max(64, int(round(cap / aspect)))
    else:
        dst_width = cap
        dst_height = max(64, int(round(cap * aspect)))
    return dst_width, dst_height
