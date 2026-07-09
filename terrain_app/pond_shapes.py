"""Pond / lake polygon editing: mask ↔ vector shapes in texture pixel space."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import label as ndimage_label
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from skimage import measure

from terrain_app.ams_color import MASK_ALPHA_MIN

MAX_POND_SHAPES = 64
SIMPLIFY_TOLERANCE = 1.5


def _new_shape_id() -> str:
    return str(uuid.uuid4())


def mask_to_polygons(
    mask: np.ndarray,
    *,
    min_area_px: int = 12,
    simplify_tolerance: float = SIMPLIFY_TOLERANCE,
) -> list[dict[str, Any]]:
    """
    Convert a boolean pond mask to editable polygons.

    Vertices are ``[col, row]`` in texture pixel coordinates (x = column, y = row).
    """
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    labeled, n = ndimage_label(arr)
    shapes: list[dict[str, Any]] = []
    for i in range(1, int(n) + 1):
        comp = labeled == i
        if int(comp.sum()) < int(min_area_px):
            continue
        contours = measure.find_contours(comp.astype(np.float64), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        # find_contours returns (row, col); convert to (col, row).
        coords = [(float(c), float(r)) for r, c in contour]
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < float(min_area_px):
            continue
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        if poly.is_empty:
            continue
        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in geoms:
            if g.is_empty or g.area < float(min_area_px):
                continue
            ext = list(g.exterior.coords)
            if len(ext) < 4:
                continue
            vertices = [[float(x), float(y)] for x, y in ext[:-1]]
            if len(vertices) < 3:
                continue
            shapes.append(
                {
                    "id": _new_shape_id(),
                    "source": "auto",
                    "vertices": vertices,
                }
            )
    return shapes[:MAX_POND_SHAPES]


def validate_shapes(
    shapes: list[dict[str, Any]],
    h: int,
    w: int,
    *,
    max_shapes: int = MAX_POND_SHAPES,
) -> list[dict[str, Any]]:
    """Drop degenerate shapes and clip vertices to raster bounds."""
    out: list[dict[str, Any]] = []
    for raw in shapes:
        if len(out) >= max_shapes:
            break
        verts_in = raw.get("vertices")
        if not isinstance(verts_in, list) or len(verts_in) < 3:
            continue
        verts: list[list[float]] = []
        for v in verts_in:
            if not isinstance(v, (list, tuple)) or len(v) < 2:
                continue
            col = float(max(0.0, min(w - 1, float(v[0]))))
            row = float(max(0.0, min(h - 1, float(v[1]))))
            verts.append([col, row])
        if len(verts) < 3:
            continue
        poly = Polygon(verts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1.0:
            continue
        ext = list(poly.exterior.coords)
        if len(ext) < 4:
            continue
        vertices = [[float(x), float(y)] for x, y in ext[:-1]]
        if len(vertices) < 3:
            continue
        sid = str(raw.get("id") or _new_shape_id())
        source = str(raw.get("source") or "manual")
        if source not in ("auto", "manual"):
            source = "manual"
        out.append({"id": sid, "source": source, "vertices": vertices})
    return out


def polygons_to_mask(
    shapes: list[dict[str, Any]],
    h: int,
    w: int,
    footprint: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize validated pond polygons to an H×W boolean mask."""
    validated = validate_shapes(shapes, h, w)
    out = np.zeros((h, w), dtype=bool)
    if not validated:
        if footprint is not None:
            return out & np.asarray(footprint, dtype=bool)
        return out

    geoms = []
    for sh in validated:
        poly = Polygon(sh["vertices"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            geoms.append(poly)
    if not geoms:
        if footprint is not None:
            return out & np.asarray(footprint, dtype=bool)
        return out

    merged = unary_union(geoms)
    if merged.is_empty:
        if footprint is not None:
            return out & np.asarray(footprint, dtype=bool)
        return out

    # Pixel space: x = col (0..w), y = row (0..h); rasterio rows increase downward.
    transform = from_bounds(0, h, w, 0, w, h)
    if merged.geom_type == "Polygon":
        geom_list = [merged]
    elif merged.geom_type == "MultiPolygon":
        geom_list = list(merged.geoms)
    else:
        geom_list = [merged]

    for geom in geom_list:
        if geom.is_empty:
            continue
        burned = rasterize(
            [(mapping(geom), 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        out |= burned.astype(bool)

    if footprint is not None:
        out &= np.asarray(footprint, dtype=bool)
    return out


def footprint_from_rgba(texture_rgba: np.ndarray) -> np.ndarray:
    """Boolean footprint from texture alpha channel."""
    arr = np.asarray(texture_rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return np.ones(arr.shape[:2], dtype=bool)
    return arr[:, :, 3] >= MASK_ALPHA_MIN
