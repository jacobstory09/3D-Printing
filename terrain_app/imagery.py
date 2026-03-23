"""Mosaic web tiles (OSM, Esri) or warp OAM GeoTIFFs to the project UTM grid."""

from __future__ import annotations

import io
from typing import Any, Callable, Dict, List, Tuple

import mercantile
import numpy as np
import requests
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from shapely.geometry import box as shapely_box
from shapely.geometry import shape

UA = "terrain-viewer/1.0 (local flask; contact: local)"

ESRI_WORLD_IMAGERY = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def pick_zoom(minx: float, miny: float, maxx: float, maxy: float, max_tiles: int = 400) -> int:
    for z in range(18, 9, -1):
        tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zooms=[z]))
        if len(tiles) <= max_tiles:
            return z
    return 9


def _download_tile(session: requests.Session, url: str) -> np.ndarray:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def mosaic_xyz_to_mercator(
    tiles: List[mercantile.Tile],
    z: int,
    url_fn: Callable[[mercantile.Tile], str],
    session: requests.Session,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    xs = [t.x for t in tiles]
    ys = [t.y for t in tiles]
    min_tx, max_tx = min(xs), max(xs)
    min_ty, max_ty = min(ys), max(ys)
    canvas_w = (max_tx - min_tx + 1) * 256
    canvas_h = (max_ty - min_ty + 1) * 256
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    L0, _, _, N0 = mercantile.xy_bounds(mercantile.Tile(min_tx, min_ty, z))
    _, B1, R1, _ = mercantile.xy_bounds(mercantile.Tile(max_tx, max_ty, z))
    west, south, east, north = L0, B1, R1, N0

    for t in tiles:
        arr = _download_tile(session, url_fn(t))
        i = (t.x - min_tx) * 256
        j = (t.y - min_ty) * 256
        canvas[j : j + 256, i : i + 256, :] = arr

    return canvas, (west, south, east, north)


def warp_rgb_to_utm(
    rgb: np.ndarray,
    bounds3857: Tuple[float, float, float, float],
    dst_epsg: int,
    dst_bounds: Tuple[float, float, float, float],
    dst_width: int,
    dst_height: int,
) -> np.ndarray:
    west, south, east, north = bounds3857
    h, w, _ = rgb.shape
    transform_src = from_bounds(west, south, east, north, w, h)
    dst_west, dst_south, dst_east, dst_north = dst_bounds
    transform_dst = from_bounds(dst_west, dst_south, dst_east, dst_north, dst_width, dst_height)
    out = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    for b in range(3):
        reproject(
            source=rgb[:, :, b],
            destination=out[:, :, b],
            src_transform=transform_src,
            src_crs="EPSG:3857",
            dst_transform=transform_dst,
            dst_crs=f"EPSG:{dst_epsg}",
            resampling=Resampling.bilinear,
        )
    return out


def fetch_texture_esri(
    bbox4326: Tuple[float, float, float, float],
    dst_epsg: int,
    dst_bounds: Tuple[float, float, float, float],
    dst_width: int,
    dst_height: int,
    session: requests.Session,
) -> np.ndarray:
    """Esri World Imagery (satellite / aerial). Same XYZ grid as OSM."""
    minx, miny, maxx, maxy = bbox4326
    z = pick_zoom(minx, miny, maxx, maxy)
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zooms=[z]))

    def url_esri(t: mercantile.Tile) -> str:
        return ESRI_WORLD_IMAGERY.format(z=t.z, y=t.y, x=t.x)

    if not tiles:
        raise ValueError("No Esri tiles for this view")
    rgb, bounds3857 = mosaic_xyz_to_mercator(tiles, z, url_esri, session)
    return warp_rgb_to_utm(rgb, bounds3857, dst_epsg, dst_bounds, dst_width, dst_height)


def fetch_texture_osm(
    bbox4326: Tuple[float, float, float, float],
    dst_epsg: int,
    dst_bounds: Tuple[float, float, float, float],
    dst_width: int,
    dst_height: int,
    session: requests.Session,
) -> np.ndarray:
    minx, miny, maxx, maxy = bbox4326
    z = pick_zoom(minx, miny, maxx, maxy)
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zooms=[z]))

    def url_osm(t: mercantile.Tile) -> str:
        return f"https://tile.openstreetmap.org/{t.z}/{t.x}/{t.y}.png"

    if not tiles:
        raise ValueError("No OSM tiles for this view (area too large or invalid bbox)")
    rgb, bounds3857 = mosaic_xyz_to_mercator(tiles, z, url_osm, session)
    return warp_rgb_to_utm(rgb, bounds3857, dst_epsg, dst_bounds, dst_width, dst_height)


def _oam_gsd(item: Dict[str, Any]) -> float:
    props = item.get("properties") or {}
    g = item.get("gsd")
    if g is None:
        g = props.get("resolution_in_meters")
    if g is None and isinstance(props.get("resolution"), list) and props["resolution"]:
        g = props["resolution"][0]
    try:
        return float(g)
    except (TypeError, ValueError):
        return 999.0


def _oam_https_tif_url(item: Dict[str, Any]) -> str | None:
    uid = item.get("uuid")
    if isinstance(uid, str) and uid.startswith("http") and ".tif" in uid.lower():
        return uid
    return None


def warp_https_geotiff_rgb_to_utm(
    https_url: str,
    dst_epsg: int,
    dst_bounds: Tuple[float, float, float, float],
    dst_width: int,
    dst_height: int,
) -> np.ndarray | None:
    """Warp an OAM (or other) COG/GeoTIFF into the destination UTM grid."""
    vsi = https_url if https_url.startswith("/vsi") else f"/vsicurl/{https_url}"
    dst_west, dst_south, dst_east, dst_north = dst_bounds
    dst_transform = from_bounds(dst_west, dst_south, dst_east, dst_north, dst_width, dst_height)
    out = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    try:
        with rasterio.Env(GDAL_HTTP_USERAGENT=UA, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(vsi) as src:
                if not src.crs:
                    return None
                n = min(3, int(src.count))
                if n < 1:
                    return None
                tmp = np.zeros((dst_height, dst_width), dtype=np.float32)
                for k in range(3):
                    bi = min(k + 1, n)
                    tmp.fill(0)
                    reproject(
                        source=rasterio.band(src, bi),
                        destination=tmp,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=f"EPSG:{dst_epsg}",
                        resampling=Resampling.bilinear,
                    )
                    tmp = np.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
                    dt = src.dtypes[bi - 1]
                    if dt == "uint8":
                        out[:, :, k] = np.clip(tmp, 0, 255).astype(np.uint8)
                    else:
                        mx = float(np.nanmax(tmp)) if np.isfinite(tmp).any() else 1.0
                        if mx <= 1.5:
                            scaled = tmp * 255.0
                        elif mx <= 255.5:
                            scaled = tmp
                        else:
                            scaled = np.clip(tmp * (255.0 / max(mx, 1e-6)), 0, 255)
                        out[:, :, k] = np.clip(scaled, 0, 255).astype(np.uint8)
        return out
    except Exception:
        return None


def oam_pick_sources(
    bbox4326: Tuple[float, float, float, float],
    session: requests.Session,
) -> Tuple[str | None, str | None, bool]:
    """Returns (geotiff_https_url_or_none, tms_template_or_none, tms_y_flip)."""
    minx, miny, maxx, maxy = bbox4326
    r = session.get(
        "https://api.openaerialmap.org/meta",
        params={"bbox": f"{minx},{miny},{maxx},{maxy}", "limit": 40},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    results: List[Dict[str, Any]] = data.get("results") or []
    target = shapely_box(minx, miny, maxx, maxy)
    ranked: List[Dict[str, Any]] = []
    for item in results:
        gj = item.get("geojson")
        if not gj:
            continue
        try:
            geom = shape(gj)
            if not geom.intersects(target):
                continue
        except Exception:
            continue
        ranked.append(item)
    ranked.sort(key=_oam_gsd)
    if not ranked:
        ranked = list(results)

    tif_url: str | None = None
    for item in ranked:
        tif_url = _oam_https_tif_url(item)
        if tif_url:
            break

    tms_template: str | None = None
    tms_flip = True
    for item in ranked:
        props = item.get("properties") or {}
        tms = props.get("tms")
        if tms and "{z}" in tms and "{x}" in tms and "{y}" in tms:
            tms_template = tms
            tms_flip = True
            break

    return tif_url, tms_template, tms_flip


def fetch_texture_oam(
    bbox4326: Tuple[float, float, float, float],
    dst_epsg: int,
    dst_bounds: Tuple[float, float, float, float],
    dst_width: int,
    dst_height: int,
    session: requests.Session,
) -> np.ndarray:
    tif_url, tms_template, tms_flip = oam_pick_sources(bbox4326, session)
    if tif_url:
        rgb = warp_https_geotiff_rgb_to_utm(
            tif_url, dst_epsg, dst_bounds, dst_width, dst_height
        )
        if rgb is not None and np.any(rgb):
            return rgb

    if not tms_template:
        raise ValueError(
            "No usable OpenAerialMap imagery (no GeoTIFF URL and no TMS) for this bbox"
        )

    minx, miny, maxx, maxy = bbox4326
    z = pick_zoom(minx, miny, maxx, maxy)
    tiles = list(mercantile.tiles(minx, miny, maxx, maxy, zooms=[z]))

    def url_oam(t: mercantile.Tile) -> str:
        y = (2**t.z - 1) - t.y if tms_flip else t.y
        return tms_template.format(z=t.z, x=t.x, y=y)

    if not tiles:
        raise ValueError("No OAM tiles for this view")
    rgb, bounds3857 = mosaic_xyz_to_mercator(tiles, z, url_oam, session)
    return warp_rgb_to_utm(rgb, bounds3857, dst_epsg, dst_bounds, dst_width, dst_height)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s
