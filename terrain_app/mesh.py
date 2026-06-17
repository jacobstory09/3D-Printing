"""Build textured mesh and export GLB / OBJ for Blender, and a print-ready solid."""

from __future__ import annotations

import glob
import io
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio.transform
import trimesh
import trimesh.boolean
import trimesh.creation
import trimesh.repair
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely.geometry import Polygon as Polygon2D
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep


def default_boundary_smooth_m(
    bounds_utm: tuple[float, float, float, float],
    grid_width: int,
    grid_height: int,
) -> float:
    """~0.75× the larger DEM cell size (m); softens grid-aligned KML stair-steps."""
    west, south, east, north = bounds_utm
    dx = (east - west) / max(int(grid_width) - 1, 1)
    dy = (north - south) / max(int(grid_height) - 1, 1)
    return 0.75 * float(max(dx, dy))


def smooth_clip_polygon(poly: Polygon2D, smooth_m: float) -> Polygon2D:
    """Round corners via outward buffer then inward buffer (morphological smooth)."""
    d = float(smooth_m)
    if d <= 0 or poly.is_empty:
        return poly
    try:
        segs = max(16, min(48, int(12 + d * 2.0)))
        out: BaseGeometry = poly.buffer(d, join_style="round", quad_segs=segs).buffer(
            -d, join_style="round", quad_segs=segs
        )
    except Exception:
        return poly
    if out.is_empty:
        return poly
    if out.geom_type == "MultiPolygon":
        out = max(out.geoms, key=lambda g: g.area)
    if not isinstance(out, Polygon2D) or out.is_empty:
        return poly
    if not out.is_valid:
        out = out.buffer(0)
    return out


def soften_raster_mask_alpha(mask: np.ndarray, sigma_px: float = 1.25) -> np.ndarray:
    """Gaussian blur for texture alpha only; does not change mesh/print clipping."""
    if mask is None or sigma_px <= 0:
        return mask
    blurred = gaussian_filter(mask.astype(np.float64), sigma=float(sigma_px))
    return np.clip(blurred, 0.0, 255.0).astype(np.uint8)


def resolve_boundary_clipping(
    poly_utm: Polygon2D,
    dem_shape: tuple[int, int],
    transform,
    bounds_utm: tuple[float, float, float, float],
    grid_width: int,
    grid_height: int,
    boundary_smooth_m: float | None,
) -> tuple[np.ndarray, Polygon2D | None, float]:
    """
    Build mask + clip polygon for :func:`build_mesh` / :func:`quad_inclusion_mask`.

    When smoothing is on, quads clip via a rounded **polygon** (``smooth_clip_polygon``).
    The returned mask is a softened alpha channel for the texture only—not used for
    quad inclusion, which avoids fringe cells and broken print side walls.
    """
    from rasterio.features import rasterize

    h, w = int(dem_shape[0]), int(dem_shape[1])
    smooth_m = (
        default_boundary_smooth_m(bounds_utm, grid_width, grid_height)
        if boundary_smooth_m is None
        else float(boundary_smooth_m)
    )
    if smooth_m <= 0:
        mask = rasterize(
            [(poly_utm, 255)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        return mask, poly_utm, 0.0
    poly_clip = smooth_clip_polygon(poly_utm, smooth_m)
    mask = rasterize(
        [(poly_clip, 255)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    west, south, east, north = bounds_utm
    cell_m = max((east - west) / max(w - 1, 1), (north - south) / max(h - 1, 1))
    sigma_px = max(0.75, float(smooth_m) / max(cell_m, 1e-9))
    mask = soften_raster_mask_alpha(mask, sigma_px=sigma_px)
    return mask, poly_clip, smooth_m


def _cell_center_east_north(transform, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """UTM easting / northing at each DEM cell center (same frame as :func:`build_mesh`)."""
    rows, cols = np.mgrid[0:h, 0:w]
    east_t, north_t = rasterio.transform.xy(transform, rows, cols, offset="center")
    east = np.asarray(east_t, dtype=np.float64).reshape(h, w)
    north = np.asarray(north_t, dtype=np.float64).reshape(h, w)
    return east, north


def _include_quad(
    i: int,
    j: int,
    east: np.ndarray,
    north: np.ndarray,
    prepared_poly,
    poly_bounds: tuple[float, float, float, float] | None,
    mask: np.ndarray | None,
) -> bool:
    """Same inclusion rule as :func:`build_mesh` for quad ``(i, j)`` (top-left cell index)."""
    if prepared_poly is not None and poly_bounds is not None:
        es = (east[i, j], east[i + 1, j], east[i + 1, j + 1], east[i, j + 1])
        ns = (north[i, j], north[i + 1, j], north[i + 1, j + 1], north[i, j + 1])
        qminx, qmaxx = min(es), max(es)
        qminy, qmaxy = min(ns), max(ns)
        pb_w, pb_s, pb_e, pb_n = poly_bounds
        if qmaxx < pb_w or qminx > pb_e or qmaxy < pb_s or qminy > pb_n:
            return False
        try:
            ring = [
                (float(es[0]), float(ns[0])),
                (float(es[1]), float(ns[1])),
                (float(es[2]), float(ns[2])),
                (float(es[3]), float(ns[3])),
            ]
            qfoot = Polygon2D(ring)
            if not qfoot.is_valid:
                qfoot = qfoot.buffer(0)
            if qfoot.is_empty or not prepared_poly.intersects(qfoot):
                return False
        except Exception:
            return False
        return True
    if mask is not None:
        blk = mask[i : i + 2, j : j + 2]
        return bool(np.any(blk >= 128))
    return True


_CLIP_XY_ROUND = 3  # 0.001 m for UTM-scale coordinates
_CORNER_TOL_M = 1e-4


def _xy_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), _CLIP_XY_ROUND), round(float(y), _CLIP_XY_ROUND))


def _point_in_tri_2d(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12
) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    den = d00 * d11 - d01 * d01
    if abs(den) < eps:
        return False
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    return u >= -eps and v >= -eps and w >= -eps


def _height_uv_at_xy(
    x: float,
    y: float,
    i: int,
    j: int,
    east: np.ndarray,
    north: np.ndarray,
    z: np.ndarray,
    w: int,
    h: int,
) -> tuple[float, float, float]:
    """Barycentric height + UV inside DEM cell ``(i, j)`` (two-triangle split)."""
    p00 = np.array([east[i, j], north[i, j]], dtype=np.float64)
    p10 = np.array([east[i + 1, j], north[i + 1, j]], dtype=np.float64)
    p01 = np.array([east[i, j + 1], north[i, j + 1]], dtype=np.float64)
    p11 = np.array([east[i + 1, j + 1], north[i + 1, j + 1]], dtype=np.float64)
    z00, z10, z01, z11 = float(z[i, j]), float(z[i + 1, j]), float(z[i, j + 1]), float(z[i + 1, j + 1])
    u00 = (j + 0.5) / w
    v00 = 1.0 - (i + 0.5) / h
    u10 = (j + 0.5) / w
    v10 = 1.0 - (i + 1.5) / h
    u01 = (j + 1.5) / w
    v01 = 1.0 - (i + 0.5) / h
    u11 = (j + 1.5) / w
    v11 = 1.0 - (i + 1.5) / h
    p = np.array([x, y], dtype=np.float64)
    for tri, zs, us, vs in (
        ((p00, p10, p01), (z00, z10, z01), (u00, u10, u01), (v00, v10, v01)),
        ((p10, p11, p01), (z10, z11, z01), (u10, u11, u01), (v10, v11, v01)),
    ):
        a, b, c = tri
        if not _point_in_tri_2d(p, a, b, c):
            continue
        v0 = c - a
        v1 = b - a
        v2 = p - a
        d00 = float(np.dot(v0, v0))
        d01 = float(np.dot(v0, v1))
        d11 = float(np.dot(v1, v1))
        d20 = float(np.dot(v2, v0))
        d21 = float(np.dot(v2, v1))
        den = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / den
        wgt = (d00 * d21 - d01 * d20) / den
        u = 1.0 - v - wgt
        return (
            u * zs[0] + wgt * zs[1] + v * zs[2],
            u * us[0] + wgt * us[1] + v * us[2],
            u * vs[0] + wgt * vs[1] + v * vs[2],
        )
    # Fallback: nearest corner (should be rare).
    corners = ((p00, z00, u00, v00), (p10, z10, u10, v10), (p01, z01, u01, v01), (p11, z11, u11, v11))
    best = min(corners, key=lambda t: float(np.linalg.norm(p - t[0])))
    return best[1], best[2], best[3]


def _collect_polygons(geom: BaseGeometry) -> list[Polygon2D]:
    if geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom]  # type: ignore[list-item]
    if gt == "MultiPolygon":
        return [g for g in geom.geoms if not g.is_empty]
    if gt == "GeometryCollection":
        out: list[Polygon2D] = []
        for g in geom.geoms:
            out.extend(_collect_polygons(g))
        return out
    return []


def _append_clipped_quad_faces(
    i: int,
    j: int,
    clip_poly: Polygon2D,
    prepared_clip,
    east: np.ndarray,
    north: np.ndarray,
    z: np.ndarray,
    w: int,
    h: int,
    vid: np.ndarray,
    vert_cache: dict[tuple[float, float], int],
    verts: list[list[float]],
    uvs: list[list[float]],
    faces: list[list[int]],
) -> bool:
    """Triangulate ``quad ∩ clip_poly``; return True if any face was added."""
    ring = [
        (float(east[i, j]), float(north[i, j])),
        (float(east[i + 1, j]), float(north[i + 1, j])),
        (float(east[i + 1, j + 1]), float(north[i + 1, j + 1])),
        (float(east[i, j + 1]), float(north[i, j + 1])),
    ]
    qfoot = Polygon2D(ring)
    if not qfoot.is_valid:
        qfoot = qfoot.buffer(0)
    if qfoot.is_empty:
        return False
    if prepared_clip.contains(qfoot):
        v00 = int(vid[i, j])
        v10 = int(vid[i + 1, j])
        v01 = int(vid[i, j + 1])
        v11 = int(vid[i + 1, j + 1])
        faces.append([v00, v10, v01])
        faces.append([v10, v11, v01])
        return True
    inter = qfoot.intersection(clip_poly)
    if inter.is_empty or inter.area <= 1e-12:
        return False
    corner_xy = ring
    corner_vid = [int(vid[i, j]), int(vid[i + 1, j]), int(vid[i + 1, j + 1]), int(vid[i, j + 1])]
    added = False

    def resolve_vid(x: float, y: float) -> int:
        for (cx, cy), cv in zip(corner_xy, corner_vid):
            if abs(x - cx) <= _CORNER_TOL_M and abs(y - cy) <= _CORNER_TOL_M:
                return cv
        k = _xy_key(x, y)
        if k in vert_cache:
            return vert_cache[k]
        zz, uu, vv = _height_uv_at_xy(x, y, i, j, east, north, z, w, h)
        idx = len(verts)
        vert_cache[k] = idx
        verts.append([x, y, zz])
        uvs.append([uu, vv])
        return idx

    for part in _collect_polygons(inter):
        try:
            t_verts, t_faces = trimesh.creation.triangulate_polygon(part)
        except Exception:
            continue
        if len(t_verts) < 3 or len(t_faces) == 0:
            continue
        local_ids = [resolve_vid(float(p[0]), float(p[1])) for p in t_verts]
        for tri in np.asarray(t_faces, dtype=np.int64):
            faces.append([local_ids[int(tri[0])], local_ids[int(tri[1])], local_ids[int(tri[2])]])
            added = True
    return added


def quad_inclusion_mask(
    h: int,
    w: int,
    transform,
    poly_utm: Polygon2D | None,
    mask: np.ndarray | None,
) -> np.ndarray:
    """``uint8`` (h-1, w-1), 1 iff that quad is included in :func:`build_mesh` (C row-major)."""
    east, north = _cell_center_east_north(transform, h, w)
    prepared_poly = prep(poly_utm) if poly_utm is not None else None
    poly_bounds = poly_utm.bounds if poly_utm is not None else None
    out = np.zeros((h - 1, w - 1), dtype=np.uint8)
    for i in range(h - 1):
        for j in range(w - 1):
            if _include_quad(i, j, east, north, prepared_poly, poly_bounds, mask):
                out[i, j] = 1
    return out


def prepare_z(
    dem: np.ndarray,
    vertical_exaggeration: float = 1.0,
    z_offset_mode: str = "min",
) -> np.ndarray:
    z = dem.astype(np.float64)
    finite = np.isfinite(z)
    if not finite.any():
        raise ValueError("DEM has no valid elevation samples")
    fill = float(np.nanmin(z[finite]))
    z = np.where(finite, z, fill)
    if z_offset_mode == "min":
        z = z - float(np.min(z))
    elif z_offset_mode == "mean":
        z = z - float(np.mean(z))
    z = z * float(vertical_exaggeration)
    return z.astype(np.float32)


def build_mesh(
    dem: np.ndarray,
    texture_rgb: np.ndarray,
    transform,
    vertical_exaggeration: float = 1.0,
    z_offset_mode: str = "min",
    mask: np.ndarray | None = None,
    poly_utm: Polygon2D | None = None,
) -> trimesh.Trimesh:
    """Single **Z-up** frame everywhere: **X** = UTM easting, **Y** = UTM northing, **Z** = elevation (m).

    ``prepare_z`` supplies elevation; it is stored on ``vertices[:,2]``. Matches trimesh / slicer XY bed + Z.

    Quads are kept if their footprint intersects ``poly_utm`` when given, otherwise if
    any cell in the 2×2 block has ``mask >= 128``. ``mask`` also drives texture alpha.
    """
    h, w = dem.shape
    if texture_rgb.shape[0] != h or texture_rgb.shape[1] != w:
        raise ValueError("Texture size must match DEM shape")
    if mask is not None and mask.shape != (h, w):
        raise ValueError("Mask shape must match DEM shape")

    z = prepare_z(dem, vertical_exaggeration, z_offset_mode).astype(np.float64)

    verts = []
    uvs = []
    for i in range(h):
        for j in range(w):
            px, py = rasterio.transform.xy(transform, i, j, offset="center")
            # px = easting (X), py = northing (Y), z[i,j] = elevation (Z)
            verts.append([px, py, z[i, j]])
            u = (j + 0.5) / w
            v = 1.0 - (i + 0.5) / h
            uvs.append([u, v])

    verts_list: list[list[float]] = [list(v) for v in verts]
    uvs_list: list[list[float]] = [list(u) for u in uvs]
    faces: list[list[int]] = []
    vid = np.arange(h * w).reshape(h, w)
    east, north = _cell_center_east_north(transform, h, w)
    if poly_utm is not None:
        prepared_clip = prep(poly_utm)
        vert_cache: dict[tuple[float, float], int] = {}
        for i in range(h - 1):
            for j in range(w - 1):
                _append_clipped_quad_faces(
                    i,
                    j,
                    poly_utm,
                    prepared_clip,
                    east,
                    north,
                    z,
                    w,
                    h,
                    vid,
                    vert_cache,
                    verts_list,
                    uvs_list,
                    faces,
                )
    else:
        prepared_poly = None
        poly_bounds = None
        for i in range(h - 1):
            for j in range(w - 1):
                if not _include_quad(i, j, east, north, prepared_poly, poly_bounds, mask):
                    continue
                v00 = int(vid[i, j])
                v10 = int(vid[i + 1, j])
                v01 = int(vid[i, j + 1])
                v11 = int(vid[i + 1, j + 1])
                faces.append([v00, v10, v01])
                faces.append([v10, v11, v01])
    verts = np.asarray(verts_list, dtype=np.float64)
    uvs = np.asarray(uvs_list, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if mask is not None:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = texture_rgb
        rgba[:, :, 3] = mask
        img = Image.fromarray(rgba, mode="RGBA")
    else:
        img = Image.fromarray(texture_rgb, mode="RGB")
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=img)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)
    if (mask is not None or poly_utm is not None) and len(faces) > 0:
        mesh.remove_unreferenced_vertices()
    return mesh


def _center_mesh_xz(m: trimesh.Trimesh) -> None:
    """Translate so horizontal center (XY) is at origin; Z (elevation) unchanged."""
    if m.is_empty or len(m.vertices) == 0:
        return
    b = m.bounds
    if b is None:
        return
    cx = 0.5 * (float(b[0, 0]) + float(b[1, 0]))
    cy = 0.5 * (float(b[0, 1]) + float(b[1, 1]))
    m.apply_translation([-cx, -cy, 0.0])


def _gltf_tree_for_blender(tree: dict) -> dict:
    """glTF tweaks so Blender shows the satellite map reliably."""
    for mat in tree.get("materials", []):
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        pbr["metallicFactor"] = 0.0
        pbr["roughnessFactor"] = 1.0
        mat["doubleSided"] = True
        if pbr.get("baseColorTexture") is not None:
            mat["alphaMode"] = "MASK"
            mat["alphaCutoff"] = 0.08
    return tree


def _mesh_copy_for_textured_export(mesh: trimesh.Trimesh, *, center_xz: bool) -> trimesh.Trimesh:
    """Copy with full-strength diffuse so Blender/OBJ do not dim the embedded map."""
    m = mesh.copy()
    if center_xz:
        _center_mesh_xz(m)
    vis = m.visual
    if (
        vis is not None
        and vis.defined
        and getattr(vis, "kind", None) == "texture"
        and getattr(vis, "uv", None) is not None
    ):
        img = _texture_image_from_trimesh(m)
        m.visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(vis.uv, dtype=np.float64),
            image=img,
        )
    if len(m.faces) > 0:
        m.fix_normals()
    return m


def export_glb(mesh: trimesh.Trimesh, *, center_xz: bool = False) -> bytes:
    m = _mesh_copy_for_textured_export(mesh, center_xz=center_xz)
    return m.export(
        file_type="glb",
        include_normals=True,
        tree_postprocessor=_gltf_tree_for_blender,
    )


def export_obj_zip(mesh: trimesh.Trimesh, *, center_xz: bool = False) -> bytes:
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "terrain.obj")
        m = _mesh_copy_for_textured_export(mesh, center_xz=center_xz)
        m.export(path)
        mtl_path = Path(td) / "material.mtl"
        if mtl_path.is_file():
            text = mtl_path.read_text(encoding="utf-8")
            text = text.replace("0.40000000 0.40000000 0.40000000", "1.00000000 1.00000000 1.00000000")
            mtl_path.write_text(text, encoding="utf-8")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in glob.glob(os.path.join(td, "*")):
                if os.path.isfile(p):
                    zf.write(p, arcname=os.path.basename(p))
    buf.seek(0)
    return buf.read()


def _poly_utm_in_extrude_frame(poly_utm: Polygon2D) -> Polygon2D:
    """(easting, northing) in UTM and trimesh extrude 2D frame are the same (X, Y) before rotation."""
    return poly_utm


def extrude_footprint_solid_utm(
    poly_utm: Polygon2D,
    height_m: float,
) -> trimesh.Trimesh:
    """
    Prismatic block under the KML footprint in UTM metres: polygon in XY (east/north),
    extruded along -Z (down). Same Z-up frame as :func:`build_mesh`.
    """
    if not poly_utm.is_valid or poly_utm.is_empty:
        raise ValueError("extrude_footprint_solid_utm requires a non-empty KML footprint polygon")
    h = float(np.abs(float(height_m)))
    if h <= 0:
        return trimesh.Trimesh()
    poly2 = _poly_utm_in_extrude_frame(poly_utm)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    block_tm = trimesh.creation.extrude_polygon(
        poly2, height=-h, engine="earcut"  # -Z in XY ground plane
    )
    return trimesh.Trimesh(vertices=block_tm.vertices, faces=block_tm.faces, process=True)


def _horizontal_extent_utm_m(surface: trimesh.Trimesh) -> float:
    b = surface.bounds
    if b is None and len(surface.vertices) > 0:
        v = np.asarray(surface.vertices, dtype=np.float64)
        b = np.array([v.min(axis=0), v.max(axis=0)], dtype=np.float64)
    if b is None:
        return 0.0
    return float(max(b[1, 0] - b[0, 0], b[1, 1] - b[0, 1]))


def _surface_xy_bounds_m(surface: trimesh.Trimesh) -> tuple[float, float]:
    """Axis-aligned horizontal spans (m) of ``surface`` in UTM east / north."""
    b = surface.bounds
    if b is None and len(surface.vertices) > 0:
        v = np.asarray(surface.vertices, dtype=np.float64)
        b = np.array([v.min(axis=0), v.max(axis=0)], dtype=np.float64)
    if b is None:
        return 0.0, 0.0
    return float(b[1, 0] - b[0, 0]), float(b[1, 1] - b[0, 1])


def _meters_to_print_mm_scale(
    surface: trimesh.Trimesh,
    print_max_size_mm: float,
    *,
    split_nx: int = 1,
    split_nz: int = 1,
) -> float:
    """
    UTM metres → millimetres so one **print bed** constraint is met:

    - **1×1:** the terrain’s larger horizontal span equals ``print_max_size_mm`` (unchanged).
    - **Nx×Nz puzzle:** each bbox tile’s larger horizontal span equals ``print_max_size_mm``
      (uniform scale so tiles still mate). The full model is larger by ~max(Nx, Nz) in the
      worst axis vs a single-bed fit.
    """
    dx, dy = _surface_xy_bounds_m(surface)
    nxi = int(max(1, split_nx))
    nzi = int(max(1, split_nz))
    cell_long_m = max(dx / float(nxi), dy / float(nzi))
    if cell_long_m < 1e-9:
        return 1.0
    return float(print_max_size_mm) / cell_long_m


def _close_heightfield_to_floor(
    surface: trimesh.Trimesh,
    floor_z: float,
) -> trimesh.Trimesh:
    """
    Close an open XY heightfield (Z = elevation) by duplicating the triangulation to ``floor_z``
    and stitching side walls.
    """
    if surface.is_empty or len(surface.vertices) == 0 or len(surface.faces) == 0:
        return trimesh.Trimesh()
    top = surface.copy()
    v_top = np.asarray(top.vertices, dtype=np.float64)
    f_top = np.asarray(top.faces, dtype=np.int64)
    n = len(v_top)
    v_bot = v_top.copy()
    v_bot[:, 2] = float(floor_z)
    f_bot = (f_top[:, ::-1] + n).astype(np.int64)
    wall_faces: list[list[int]] = []
    ue = np.asarray(top.edges_unique, dtype=np.int64)
    fue = np.asarray(top.faces_unique_edges, dtype=np.int64)
    ec = np.bincount(fue.reshape(-1), minlength=len(ue))
    boundary = ue[ec == 1] if len(ue) > 0 else np.zeros((0, 2), dtype=np.int64)
    for e in boundary:
        i, j = int(e[0]), int(e[1])
        wall_faces.append([i, j, j + n])
        wall_faces.append([i, j + n, i + n])
    v_all = np.vstack([v_top, v_bot])
    if len(wall_faces) > 0:
        f_all = np.vstack([f_top, f_bot, np.asarray(wall_faces, dtype=np.int64)])
    else:
        f_all = np.vstack([f_top, f_bot])
    out = trimesh.Trimesh(vertices=v_all, faces=f_all, process=False)
    out = solid_process_for_export(out)
    return out


def _try_manifold_union(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    """Single watertight body via Boolean union when shells touch (after gap close)."""
    if len(parts) <= 1:
        return parts[0].copy() if parts else None
    try:
        r = trimesh.boolean.union(
            parts, engine="manifold", check_volume=False
        )
        if r is None or getattr(r, "is_empty", False) or len(r.vertices) < 3:
            return None
        r = solid_process_for_export(r)
        if len(r.split(only_watertight=False)) <= 1:
            return r
    except Exception:
        return None
    return None


# Avoid multi-gigabyte voxel grids during print-solid repair (Z-up mm mesh bounds).
_MAX_VOXEL_CELLS = 48_000_000
_VOXEL_FLOOR_MM = 0.35


def _voxel_grid_shape(mesh: trimesh.Trimesh, voxel_size_mm: float) -> np.ndarray:
    b = np.asarray(mesh.bounds, dtype=np.float64)
    if b is None or b.shape != (2, 3):
        return np.array([1, 1, 1], dtype=np.int64)
    ext = np.maximum(b[1] - b[0], 1e-6)
    return np.ceil(ext / float(voxel_size_mm)).astype(np.int64) + 1


def _clamp_voxel_size_mm(
    mesh: trimesh.Trimesh,
    voxel_size_mm: float,
    *,
    print_max_size_mm: float | None = None,
) -> float:
    """Raise voxel size until the implied grid stays within ``_MAX_VOXEL_CELLS``."""
    vs = float(max(_VOXEL_FLOOR_MM, voxel_size_mm))
    if print_max_size_mm is not None and float(print_max_size_mm) > 0:
        vs = max(vs, float(print_max_size_mm) / 80.0)
    for _ in range(12):
        cells = int(np.prod(_voxel_grid_shape(mesh, vs)))
        if cells <= _MAX_VOXEL_CELLS:
            return vs
        vs *= 1.35
    return vs


def _voxel_merge_filled(
    merged: trimesh.Trimesh,
    voxel_size_mm: float,
    *,
    print_max_size_mm: float | None = None,
) -> trimesh.Trimesh:
    """One filled voxel pass (mesh already Z-up XY + elevation)."""
    vs = _clamp_voxel_size_mm(
        merged, float(voxel_size_mm), print_max_size_mm=print_max_size_mm
    )
    vg = merged.voxelized(vs, method="ray")
    vg = vg.fill()
    out = vg.marching_cubes
    del vg
    return solid_process_for_export(out)


def _last_resort_voxel_fuse(
    solid: trimesh.Trimesh,
    voxel_size_mm: float,
    print_max_size_mm: float,
) -> trimesh.Trimesh:
    """If multiple bodies remain, concat and voxelize; retry with finer voxels if needed."""
    out = solid
    vs = _clamp_voxel_size_mm(
        solid,
        min(float(voxel_size_mm), max(_VOXEL_FLOOR_MM, float(print_max_size_mm) / 200.0)),
        print_max_size_mm=print_max_size_mm,
    )
    for _iter_i in range(3):
        parts = list(out.split(only_watertight=False))
        if len(parts) <= 1:
            return out
        union_try = _try_manifold_union(parts)
        if union_try is not None and len(union_try.split(only_watertight=False)) <= 1:
            return union_try
        merged = trimesh.util.concatenate(parts)
        merged = solid_process_for_export(merged)
        out = _voxel_merge_filled(
            merged, float(vs), print_max_size_mm=print_max_size_mm
        )
        del merged
        n_after = len(out.split(only_watertight=False))
        if n_after <= 1:
            return out
        vs = _clamp_voxel_size_mm(
            out, max(_VOXEL_FLOOR_MM, vs * 0.65), print_max_size_mm=print_max_size_mm
        )
    return out


def _fuse_stacked_components(
    solid: trimesh.Trimesh,
    voxel_size_mm: float,
    *,
    print_max_size_mm: float | None = None,
) -> tuple[trimesh.Trimesh, bool]:
    """
    If meshing produced vertically stacked disconnected shells, close vertical gaps by translation
    (no fragile XZ AABB intersection test), then voxel-merge into one body.
    """
    parts = list(solid.split(only_watertight=False))
    if len(parts) <= 1:
        return solid, False
    # Sort by bottom Z; stack so each part sits on the one below.
    parts.sort(key=lambda m: float(m.bounds[0, 2]) if m.bounds is not None else 0.0)
    b0 = parts[0].bounds
    if b0 is None:
        return solid, False
    top_z = float(b0[1, 2])
    for i in range(1, len(parts)):
        pb = parts[i].bounds
        if pb is None:
            continue
        pminz = float(pb[0, 2])
        gap = pminz - top_z
        if gap > 1e-6:
            parts[i].apply_translation([0.0, 0.0, -gap])
        b2 = parts[i].bounds
        if b2 is not None:
            top_z = max(top_z, float(b2[1, 2]))
    union_out = _try_manifold_union(parts)
    if union_out is not None:
        out = union_out
        return out, True
    merged = trimesh.util.concatenate(parts)
    merged = solid_process_for_export(merged)
    out = _voxel_merge_filled(
        merged, float(voxel_size_mm), print_max_size_mm=print_max_size_mm
    )
    del merged
    return out, True


def build_print_solid(
    surface: trimesh.Trimesh,
    poly_utm: Polygon2D,
    print_max_size_mm: float = 200.0,
    base_extrusion_mm: float = 1.0,
    center_on_bed: bool = True,
    voxel_size_mm: float | None = None,
    print_split_nx: int = 1,
    print_split_nz: int = 1,
) -> tuple[trimesh.Trimesh, dict[str, Any], trimesh.Trimesh]:
    """
    Build a **single closed print solid** in millimetres (**Z-up**: X east, Y north, Z elevation).
    Build plate is **Z = 0**; terrain sits above a 1.0 mm base. Horizontal scale is isotropic:
    with ``print_split_nx`` = ``print_split_nz`` = 1, the larger **whole-model** XY span matches
    ``print_max_size_mm``. With an ``Nx``×``Nz`` puzzle grid, the larger **per-tile** XY span
    (bounding box divided by the grid) matches ``print_max_size_mm`` so each STL can use the bed.
    ``base_extrusion_mm`` is API-only; base thickness is 1 mm.

    Returns ``(solid, meta, surf_mm)`` where ``surf_mm`` is the scaled, centered open terrain (mm)
    with texture—used to paint the watertight solid for 3MF export.
    """
    if not poly_utm.is_valid:
        poly_utm = poly_utm.buffer(0)
    if surface.is_empty or len(surface.vertices) == 0:
        raise ValueError("Cannot build print solid from empty surface mesh")
    spx = int(max(1, print_split_nx))
    spz = int(max(1, print_split_nz))
    s = _meters_to_print_mm_scale(
        surface, print_max_size_mm, split_nx=spx, split_nz=spz
    )
    # Fixed base thickness under variable fill layer.
    be = 1.0
    # Use the already-clipped mesh from build_mesh; additional centroid trimming can introduce
    # cracks in the grid that break closure when we stitch side walls.
    surf = surface.copy()
    surf.apply_scale(s)
    # Lowest terrain Z to 0, then lift by base thickness so the underside sits above the plate.
    if len(surf.vertices) > 0 and surf.bounds is not None:
        surf.apply_translation([0.0, 0.0, -float(surf.bounds[0, 2])])
    if be > 0:
        surf.apply_translation([0.0, 0.0, float(be)])

    if center_on_bed and surf.bounds is not None:
        cmin, cmax = surf.bounds[0], surf.bounds[1]
        cx, cy = 0.5 * (cmin[0] + cmax[0]), 0.5 * (cmin[1] + cmax[1])
        t_center = trimesh.transformations.translation_matrix([-cx, -cy, 0.0])
        surf.apply_transform(t_center)
    if len(surf.vertices) == 0:
        raise ValueError("Surface mesh has no geometry for print export")
    terrain_floor_z = 0.0
    terrain_solid = _close_heightfield_to_floor(surf, terrain_floor_z)
    if len(terrain_solid.vertices) == 0:
        raise ValueError("Failed to close terrain mesh into print solid")
    solid = solid_process_for_export(terrain_solid)
    vs = float(voxel_size_mm) if (voxel_size_mm is not None and float(voxel_size_mm) > 0) else 0.0
    if not solid.is_watertight or abs(float(solid.volume)) <= 1e-9:
        # Fallback path for pathological meshes: voxelize and recover a closed isosurface.
        if vs <= 0:
            vs = max(0.25, float(print_max_size_mm) / 150.0)
        solid = _voxel_merge_filled(
            terrain_solid, float(vs), print_max_size_mm=float(print_max_size_mm)
        )
    elif vs <= 0:
        # Keep metadata stable when explicit voxel size is not in use.
        vs = max(0.25, float(print_max_size_mm) / 150.0)
    n_before_fuse = len(solid.split(only_watertight=False))
    solid, _ = _fuse_stacked_components(
        solid, float(vs), print_max_size_mm=float(print_max_size_mm)
    )
    n_after_fuse = len(solid.split(only_watertight=False))
    last_resort = False
    if n_after_fuse > 1:
        solid = _last_resort_voxel_fuse(
            solid, float(vs), float(print_max_size_mm)
        )
        last_resort = True
    # Ray+marching can sit slightly below Z=0; rest on the build plate.
    sb = solid.bounds
    if sb is not None and sb[0, 2] < 0:
        solid.apply_translation([0.0, 0.0, -float(sb[0, 2])])
    n_components = len(solid.split(only_watertight=False))
    sb2 = solid.bounds
    if sb2 is not None:
        d_mm = (float(sb2[1, 0] - sb2[0, 0]), float(sb2[1, 1] - sb2[0, 1]))
        w_max_mm = max(d_mm[0], d_mm[1])
    else:
        d_mm, w_max_mm = (0.0, 0.0), 0.0
    puzzle = spx * spz > 1
    meta: dict[str, Any] = {
        "units": "mm",
        "z_up": True,
        "axes": "X=east_mm, Y=north_mm, Z=elevation_mm",
        "print_max_size_mm": float(print_max_size_mm),
        "print_split_nx": int(spx),
        "print_split_nz": int(spz),
        "print_scale_for_puzzle_grid": bool(puzzle),
        "print_per_tile_target_horizontal_mm": float(print_max_size_mm)
        if puzzle
        else None,
        "base_extrusion_mm": be,
        "scale_meters_to_print_mm": float(s),
        "print_voxel_size_mm": float(vs),
        "print_xy_extent_max_mm": float(w_max_mm),
        "print_component_count": int(n_components),
        "print_solid_fused": bool(n_before_fuse > 1 or last_resort),
        "print_solid_last_resort_voxel": last_resort,
        "print_component_count_pre_fuse": int(n_before_fuse),
        "print_full_size_mm": {
            "x": max(d_mm[0], 0.0),
            "y": max(d_mm[1], 0.0),
            "max_horizontal_mm": w_max_mm,
        },
    }
    meta["print_non_manifold_edge_count"] = int(non_manifold_edge_count(solid))
    meta["print_open_edge_count"] = int(open_edge_count(solid))
    return solid, meta, surf.copy()


def open_edge_count(mesh: trimesh.Trimesh) -> int:
    """Count manifold boundary edges (incident to exactly one face)."""
    if mesh.is_empty or len(mesh.faces) == 0:
        return 0
    fue = np.asarray(mesh.faces_unique_edges, dtype=np.int64)
    if fue.size == 0:
        return 0
    ec = np.bincount(fue.reshape(-1), minlength=len(mesh.edges_unique))
    return int(np.sum(ec == 1))


def non_manifold_edge_count(mesh: trimesh.Trimesh) -> int:
    """Count edges shared by more than two faces (slicer non-manifold)."""
    if mesh.is_empty or len(mesh.faces) == 0:
        return 0
    fue = np.asarray(mesh.faces_unique_edges, dtype=np.int64)
    if fue.size == 0:
        return 0
    ec = np.bincount(fue.reshape(-1), minlength=len(mesh.edges_unique))
    return int(np.sum(ec > 2))


def mesh_from_3mf_bytes(data: bytes) -> trimesh.Trimesh:
    """Load a single mesh from 3MF bytes (concatenate scene if needed)."""
    loaded = trimesh.load(io.BytesIO(data), file_type="3mf")
    if isinstance(loaded, trimesh.Scene):
        geom = loaded.to_geometry()
        if not isinstance(geom, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh from Scene.to_geometry(), got {type(geom)}")
        return geom
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unexpected 3MF load type: {type(loaded)}")
    return loaded


def print_solid_with_satellite_uv(
    print_solid: trimesh.Trimesh,
    surf_mm: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Attach satellite texture UVs to the print solid by nearest-neighbor in XY to ``surf_mm``."""
    out = print_solid.copy()
    vis = surf_mm.visual
    if vis is None or not vis.defined or getattr(vis, "kind", None) != "texture":
        return out
    uv_src = getattr(vis, "uv", None)
    if uv_src is None:
        return out
    uv_arr = np.asarray(uv_src, dtype=np.float64)
    if uv_arr.shape[0] != len(surf_mm.vertices):
        return out
    mat = getattr(vis, "material", None)
    img = getattr(mat, "image", None) if mat is not None else None
    if img is None:
        return out
    tree = cKDTree(np.asarray(surf_mm.vertices[:, :2], dtype=np.float64))
    q = np.asarray(out.vertices[:, :2], dtype=np.float64)
    _, idx = tree.query(q, k=1)
    idx_i = np.asarray(idx, dtype=np.intp)
    new_uv = uv_arr[idx_i]
    out.visual = trimesh.visual.TextureVisuals(uv=new_uv, image=img.copy())
    return out


def finalize_mesh_for_3mf_export(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Weld and fix winding/normals before 3MF packaging (reduces slicer non-manifold reports)."""
    t = m.copy()
    t.update_faces(t.nondegenerate_faces() & t.unique_faces())
    t.remove_unreferenced_vertices()
    if (
        t.visual is not None
        and t.visual.defined
        and getattr(t.visual, "kind", None) == "texture"
        and getattr(t.visual, "uv", None) is not None
    ):
        t.merge_vertices(merge_tex=True, merge_norm=False)
    else:
        t.merge_vertices(merge_norm=False)
    trimesh.repair.fix_winding(t)
    t.fix_normals()
    return t


def solid_process_for_export(m: trimesh.Trimesh) -> trimesh.Trimesh:
    t = m.copy()
    t.update_faces(t.nondegenerate_faces() & t.unique_faces())
    t.remove_unreferenced_vertices()
    if (not t.is_watertight) or (open_edge_count(t) > 0):
        t.fill_holes()
    t.merge_vertices(merge_norm=False)
    trimesh.repair.fix_winding(t)
    t.fix_normals()
    return t


def export_stl(solid: trimesh.Trimesh) -> bytes:
    return solid.export(file_type="stl")


def export_3mf(solid: trimesh.Trimesh) -> bytes:
    return solid.export(file_type="3mf")


# 3MF namespaces (Materials + Core; see 3MF Consortium materials extension samples).
_M3MF_NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_M3MF_NS_MAT = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
_M3MF_Q_CORE = f"{{{_M3MF_NS_CORE}}}"
_M3MF_Q_MAT = f"{{{_M3MF_NS_MAT}}}"

# Paths inside the 3MF package (OPC).
TEXTURED_3MF_TEXTURE_PART = "3D/Textures/texture.png"
TEXTURED_3MF_TEXTURE_PATH_ATTR = "/3D/Textures/texture.png"
TEXTURED_3MF_MODEL_RELS_PART = "3D/_rels/3dmodel.model.rels"
TEXTURED_3MF_MODEL_PART = "3D/3dmodel.model"
TEXTURED_3MF_TEXTURE_REL_TYPE = (
    "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dtexture"
)


def _texture_image_from_trimesh(m: trimesh.Trimesh) -> Image.Image:
    """Resolve PIL image from ``TextureVisuals`` (direct or via material)."""
    vis = m.visual
    if vis is None or not getattr(vis, "defined", False):
        raise ValueError("mesh has no visual")
    img = getattr(vis, "image", None)
    if img is None:
        mat = getattr(vis, "material", None)
        if mat is not None:
            img = getattr(mat, "image", None)
    if img is None:
        raise ValueError("mesh has no texture image")
    return img


def mesh_has_texture_visual_for_3mf(m: trimesh.Trimesh) -> bool:
    """True if ``m`` has UV + image suitable for textured 3MF export."""
    vis = m.visual
    if vis is None or not vis.defined:
        return False
    if getattr(vis, "kind", None) != "texture":
        return False
    uvs = getattr(vis, "uv", None)
    if uvs is None:
        return False
    try:
        uv_arr = np.asarray(uvs, dtype=np.float64)
        if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
            return False
        if uv_arr.shape[0] != len(m.vertices):
            return False
        _texture_image_from_trimesh(m)
    except Exception:
        return False
    return True


def _fmt_coord(x: float) -> str:
    return format(float(x), ".10g")


def export_textured_print_3mf_zip(m: trimesh.Trimesh) -> bytes:
    """
    Write a 3MF package with embedded PNG and per-vertex UVs (3MF Materials extension).

    Vertex and triangle indices follow 3MF consortium examples (0-based ``v1``/``p1``).
    """
    if m.is_empty or len(m.vertices) == 0 or len(m.faces) == 0:
        raise ValueError("empty mesh")
    v = np.asarray(m.vertices, dtype=np.float64)
    f = np.asarray(m.faces, dtype=np.int64)
    uv = np.asarray(m.visual.uv, dtype=np.float64)  # type: ignore[union-attr]
    if uv.shape != (len(v), 2):
        raise ValueError("UV count must match vertex count after finalize for textured 3MF")
    img = _texture_image_from_trimesh(m)
    png_io = io.BytesIO()
    img.save(png_io, format="PNG")
    png_bytes = png_io.getvalue()

    texture2d_id = 1
    texgroup_id = 2
    object_id = 3

    text = io.StringIO()
    text.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    text.write(
        "<model unit=\"millimeter\" xml:lang=\"en-US\" "
        f'xmlns="{_M3MF_NS_CORE}" xmlns:m="{_M3MF_NS_MAT}">\n'
    )
    text.write("<resources>\n")
    text.write(
        f'<m:texture2d id="{texture2d_id}" path="{TEXTURED_3MF_TEXTURE_PATH_ATTR}" '
        'contenttype="image/png" tilestyleu="wrap" tilestylev="wrap" />\n'
    )
    text.write(f'<m:texture2dgroup id="{texgroup_id}" texid="{texture2d_id}">\n')
    for i in range(len(uv)):
        u, wv = float(uv[i, 0]), float(uv[i, 1])
        text.write(f'<m:tex2coord u="{_fmt_coord(u)}" v="{_fmt_coord(wv)}" />\n')
    text.write("</m:texture2dgroup>\n")
    text.write(f'<object id="{object_id}" type="model">\n<mesh>\n<vertices>\n')
    for i in range(len(v)):
        x, y, z = float(v[i, 0]), float(v[i, 1]), float(v[i, 2])
        text.write(
            f'<vertex x="{_fmt_coord(x)}" y="{_fmt_coord(y)}" z="{_fmt_coord(z)}" />\n'
        )
    text.write("</vertices>\n<triangles>\n")
    for fi in range(len(f)):
        a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
        text.write(
            f'<triangle v1="{a}" v2="{b}" v3="{c}" '
            f'pid="{texgroup_id}" p1="{a}" p2="{b}" p3="{c}" />\n'
        )
    text.write(
        "</triangles>\n</mesh>\n</object>\n</resources>\n"
        "<build>\n"
        f'<item objectid="{object_id}" transform="1 0 0 0 1 0 0 0 1 0 0 0" />\n'
        "</build>\n</model>\n"
    )
    model_bytes = text.getvalue().encode("utf-8")

    ct = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        '  <Default Extension="png" ContentType="image/png"/>\n'
        '  <Default Extension="texture" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodeltexture"/>\n'
        "</Types>\n"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" '
        f'Target="/{TEXTURED_3MF_MODEL_PART}" Id="rel0"/>\n'
        "</Relationships>\n"
    )
    model_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        f'  <Relationship Type="{TEXTURED_3MF_TEXTURE_REL_TYPE}" '
        f'Target="{TEXTURED_3MF_TEXTURE_PATH_ATTR}" Id="relTex"/>\n'
        "</Relationships>\n"
    )

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct.encode("utf-8"))
        zf.writestr("_rels/.rels", rels.encode("utf-8"))
        zf.writestr(TEXTURED_3MF_MODEL_RELS_PART, model_rels.encode("utf-8"))
        zf.writestr(TEXTURED_3MF_MODEL_PART, model_bytes)
        zf.writestr(TEXTURED_3MF_TEXTURE_PART, png_bytes)
    return out.getvalue()


def inspect_3mf_texture_payload(raw_3mf: bytes) -> dict[str, Any]:
    """
    Verify that a 3MF ZIP embeds a PNG and Materials texture markup.

    Use this for job metadata: ``print_3mf_textured`` should reflect on-disk payload,
    not only in-memory ``TextureVisuals``.
    """
    out: dict[str, Any] = {
        "ok": False,
        "texture_png_present": False,
        "texture_path_zip": TEXTURED_3MF_TEXTURE_PART,
        "texture_path_attr_expected": TEXTURED_3MF_TEXTURE_PATH_ATTR,
        "texture_size_px": None,
        "model_part": TEXTURED_3MF_MODEL_PART,
        "model_has_texture2d": False,
        "model_has_texture2dgroup": False,
        "model_has_tex2coord": False,
        "texture2d_count": 0,
        "tex2coord_count": 0,
        "triangle_count": 0,
        "textured_triangle_count": 0,
        "error": None,
    }
    zf: zipfile.ZipFile | None = None
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_3mf), "r")
    except zipfile.BadZipFile as e:
        out["error"] = f"bad_zip: {e}"
        return out
    try:
        if TEXTURED_3MF_TEXTURE_PART not in zf.namelist():
            out["error"] = f"missing:{TEXTURED_3MF_TEXTURE_PART}"
            return out
        out["texture_png_present"] = True
        out["model_rels_present"] = TEXTURED_3MF_MODEL_RELS_PART in zf.namelist()
        with zf.open(TEXTURED_3MF_TEXTURE_PART) as fp:
            im = Image.open(fp)
            im.load()
            out["texture_size_px"] = [int(im.width), int(im.height)]

        if TEXTURED_3MF_MODEL_PART not in zf.namelist():
            out["error"] = f"missing:{TEXTURED_3MF_MODEL_PART}"
            return out
        xml_data = zf.read(TEXTURED_3MF_MODEL_PART)
        root = ET.fromstring(xml_data)

        n_tex2d = 0
        n_tex2coord = 0
        n_tri = 0
        n_tex_tri = 0
        for el in root.iter():
            tag = el.tag
            if tag == _M3MF_Q_MAT + "texture2d":
                n_tex2d += 1
            elif tag == _M3MF_Q_MAT + "tex2coord":
                n_tex2coord += 1
            elif tag == _M3MF_Q_CORE + "triangle":
                n_tri += 1
                pid = el.get("pid")
                if pid is not None and el.get("p1") is not None:
                    p2 = el.get("p2")
                    p3 = el.get("p3")
                    if p2 is not None and p3 is not None:
                        n_tex_tri += 1

        out["model_has_texture2d"] = n_tex2d > 0
        out["model_has_texture2dgroup"] = any(
            el.tag == _M3MF_Q_MAT + "texture2dgroup" for el in root.iter()
        )
        out["model_has_tex2coord"] = n_tex2coord > 0
        out["texture2d_count"] = n_tex2d
        out["tex2coord_count"] = n_tex2coord
        out["triangle_count"] = n_tri
        out["textured_triangle_count"] = n_tex_tri

        out["ok"] = bool(
            out["texture_png_present"]
            and out.get("model_rels_present")
            and out["model_has_texture2d"]
            and out["model_has_texture2dgroup"]
            and out["model_has_tex2coord"]
            and n_tex_tri == n_tri
            and n_tri > 0
        )
    except Exception as e:
        out["error"] = f"inspect_failed: {e}"
        out["ok"] = False
    finally:
        if zf is not None:
            try:
                zf.close()
            except Exception:
                pass
    return out


def orient_print_mesh_for_slicer(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Print mesh is already X east, Y north, Z elevation (mm); no axis permutation for slicers."""
    return m.copy()


def export_print_stl(solid: trimesh.Trimesh) -> bytes:
    """Watertight print solid for FDM slicers (Z-up, XY bed)."""
    return orient_print_mesh_for_slicer(solid).export(file_type="stl")


def export_print_3mf(solid: trimesh.Trimesh) -> bytes:
    """
    Slicer-safe core 3MF (geometry only, same solid as STL).

    Many slicers reject or choke on the Materials-extension textured package; use
    :func:`export_textured_print_3mf_zip` only when a target app explicitly supports it.
    """
    m = finalize_mesh_for_3mf_export(orient_print_mesh_for_slicer(solid))
    return m.export(file_type="3mf")


def export_textured_print_3mf(solid: trimesh.Trimesh) -> bytes:
    """Materials-extension 3MF with embedded PNG + UVs (not all slicers can open this)."""
    m = finalize_mesh_for_3mf_export(orient_print_mesh_for_slicer(solid))
    if not mesh_has_texture_visual_for_3mf(m):
        raise ValueError("mesh has no UV texture for textured 3MF export")
    return export_textured_print_3mf_zip(m)


def export_print_glb(solid: trimesh.Trimesh) -> bytes:
    """Geometry-only print solid as GLB (same watertight body as STL)."""
    return orient_print_mesh_for_slicer(solid).export(file_type="glb")


def export_textured_print_glb(solid: trimesh.Trimesh) -> bytes:
    """Watertight print solid (mm, Z-up) with embedded satellite UV texture."""
    m = orient_print_mesh_for_slicer(solid)
    if not mesh_has_texture_visual_for_3mf(m):
        raise ValueError("mesh has no UV texture for textured print GLB export")
    return export_glb(m, center_xz=False)


def export_ams_print_glb_labeled(
    solid: trimesh.Trimesh,
    face_labels: np.ndarray,
    palette: list[dict[str, Any]],
) -> bytes:
    """Print solid with AMS vertex colors from face labels (no per-color submesh copies)."""
    if solid.is_empty or len(solid.faces) == 0:
        raise ValueError("no solid geometry for AMS GLB export")
    labels = np.asarray(face_labels, dtype=np.intp)
    if len(labels) != len(solid.faces):
        raise ValueError("face_labels length must match solid face count")
    m = orient_print_mesh_for_slicer(solid)
    index_to_rgb: dict[int, tuple[int, int, int]] = {}
    max_idx = 0
    for ent in palette:
        idx = int(ent["index"])
        rgb = ent["rgb"]
        index_to_rgb[idx] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        max_idx = max(max_idx, idx)
    table = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for idx, rgb in index_to_rgb.items():
        table[idx] = rgb
    faces = np.asarray(m.faces, dtype=np.int64)
    face_rgb = table[np.clip(labels, 0, max_idx)]
    vcol = np.zeros((len(m.vertices), 4), dtype=np.uint8)
    vcol[:, 3] = 255
    vcol[faces[:, 0], :3] = face_rgb
    vcol[faces[:, 1], :3] = face_rgb
    vcol[faces[:, 2], :3] = face_rgb
    m.visual = trimesh.visual.ColorVisuals(vertex_colors=vcol)
    return export_glb(m, center_xz=False)


def export_ams_print_glb(
    parts: list[tuple[dict[str, Any], trimesh.Trimesh]],
) -> bytes:
    """Print solid split into AMS color regions as a multi-mesh GLB (vertex colors)."""
    if not parts:
        raise ValueError("no AMS parts to export")
    scene = trimesh.Scene()
    for meta, mesh in parts:
        if mesh.is_empty or len(mesh.vertices) == 0:
            continue
        m = orient_print_mesh_for_slicer(mesh)
        r, g, b = meta["rgb"]
        rgba = np.tile([int(r), int(g), int(b), 255], (len(m.vertices), 1)).astype(np.uint8)
        m.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
        name = str(meta.get("part_name") or meta.get("name") or "part")
        scene.add_geometry(m, geom_name=name)
    if not scene.geometry:
        raise ValueError("no AMS parts to export")
    return scene.export(file_type="glb")


def split_solid_to_xz_grid(
    solid: trimesh.Trimesh,
    nx: int,
    nz: int,
    export_basename: str = "terrain",
) -> list[dict[str, Any]]:
    """
    Split a closed mesh in **Z-up mm** (X east, Y north, Z elevation) into an ``nx``×``nz`` grid
    in the **XY** plane; each piece spans the full **Z** extent of the solid. Parameter names
    ``nx`` / ``nz`` are kept for the API: ``nx`` = columns along **X**, ``nz`` = rows along **Y**
    (northing).
    """
    nxi = int(max(1, nx))
    nzi = int(max(1, nz))
    base = (export_basename or "terrain").strip() or "terrain"
    if solid.is_empty or len(solid.vertices) == 0:
        return []
    b = solid.bounds
    if b is None:
        return []
    pad = 0.02
    b = np.array(b, dtype=np.float64)
    b[0, 0] -= pad
    b[1, 0] += pad
    b[0, 1] -= pad
    b[1, 1] += pad
    b[0, 2] -= pad
    b[1, 2] += pad
    out: list[dict[str, Any]] = []
    total = nxi * nzi
    z0f, z1f = float(b[0, 2]), float(b[1, 2])
    for iy in range(nzi):
        y0 = b[0, 1] + (b[1, 1] - b[0, 1]) * (iy / float(nzi))
        y1 = b[0, 1] + (b[1, 1] - b[0, 1]) * ((iy + 1) / float(nzi))
        for ix in range(nxi):
            x0 = b[0, 0] + (b[1, 0] - b[0, 0]) * (ix / float(nxi))
            x1 = b[0, 0] + (b[1, 0] - b[0, 0]) * ((ix + 1) / float(nxi))
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            cz = 0.5 * (z0f + z1f)
            ex = max((x1 - x0), 1e-6)
            ey = max((y1 - y0), 1e-6)
            ez = max((z1f - z0f), 1e-6)
            box = trimesh.creation.box(
                extents=(ex, ey, ez),
                transform=trimesh.transformations.translation_matrix((float(cx), float(cy), float(cz))),
            )
            part = boolean_intersect_safe(solid, box, engine="manifold")
            if part is None or part.is_empty or len(part.vertices) < 3:
                continue
            part = solid_process_for_export(part)
            if part.is_empty or len(part.vertices) < 3:
                continue
            k = 1 + ix + iy * nxi
            out.append(
                {
                    "id": k,
                    "ix": ix,
                    "iz": iy,
                    "row": iy + 1,
                    "col": ix + 1,
                    "n_columns": nxi,
                    "n_rows": nzi,
                    "index_of_total": k,
                    "total_pieces": total,
                    "filename": f"{base}_print_r{iy + 1:02d}c{ix + 1:02d}.stl",
                    "mesh": part,
                }
            )
    return out


def boolean_intersect_safe(
    a: trimesh.Trimesh, b: trimesh.Trimesh, engine: str = "manifold"
) -> trimesh.Trimesh | None:
    try:
        r = trimesh.boolean.intersection(
            [a, b], engine=engine, check_volume=False
        )
    except Exception:
        return None
    if r is None or (hasattr(r, "is_empty") and r.is_empty) or len(r.vertices) < 3:
        return None
    return r


def export_print_pieces_stl_bytes(pieces: list[dict[str, Any]]) -> bytes:
    """Zip of one STL per non-empty cell: ``terrain_print_r{row}c{col}.stl`` (1-based, Y then X)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pieces:
            m = p["mesh"]
            name = str(p.get("filename") or f"piece_{p.get('id', 0)}.stl")
            zf.writestr(name, export_print_stl(m))
    buf.seek(0)
    return buf.read()


def _rgb_to_mtl_components(rgb: list[int] | tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return (r / 255.0, g / 255.0, b / 255.0)


def _hex_rgba_for_3mf(rgb: list[int] | tuple[int, int, int]) -> str:
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return f"#{r:02X}{g:02X}{b:02X}FF"


def export_bambu_ams_obj_zip(
    parts: list[tuple[dict[str, Any], trimesh.Trimesh]],
) -> bytes:
    """
    ZIP with one OBJ (multiple ``o`` / ``usemtl`` groups) + MTL diffuse colors for Bambu Studio.

    Legacy path for fixture meshes built as separate bodies. Prefer
    :func:`export_bambu_ams_obj_zip_labeled` for face-labeled splits of one print solid.
    """
    if not parts:
        raise ValueError("no AMS parts to export")
    mtl_lines = ["# Bambu AMS filament colors\n"]
    obj_lines = ["# Terrain print solid — 4-color AMS regions\n", "mtllib terrain_ams.mtl\n"]
    v_offset = 0
    for meta, mesh in parts:
        mat = str(meta.get("material_name") or meta.get("part_name") or "color")
        r, g, b = _rgb_to_mtl_components(meta["rgb"])
        mtl_lines.append(f"newmtl {mat}\n")
        mtl_lines.append(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
        mtl_lines.append(f"Ka {r:.6f} {g:.6f} {b:.6f}\n")
        mtl_lines.append("Ks 0.200000 0.200000 0.200000\n")
        mtl_lines.append("Ns 10.000000\n\n")
        obj_lines.append(f"o {mat}\n")
        obj_lines.append(f"usemtl {mat}\n")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        for i in range(len(verts)):
            x, y, z = float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2])
            obj_lines.append(f"v {_fmt_coord(x)} {_fmt_coord(y)} {_fmt_coord(z)}\n")
        for fi in range(len(faces)):
            a = int(faces[fi, 0]) + v_offset + 1
            b = int(faces[fi, 1]) + v_offset + 1
            c = int(faces[fi, 2]) + v_offset + 1
            obj_lines.append(f"f {a} {b} {c}\n")
        v_offset += len(verts)
    readme = (
        "Bambu Studio import\n"
        "-------------------\n"
        "1. File → Import → terrain_ams.obj (or drag this ZIP's OBJ onto the plate).\n"
        "2. If asked to load as one object with multiple parts, choose Yes.\n"
        "3. In the Objects panel, assign each part/material to the closest AMS filament.\n"
        "4. See palette.json for recommended color names and hex values.\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("terrain_ams.mtl", "".join(mtl_lines).encode("utf-8"))
        zf.writestr("terrain_ams.obj", "".join(obj_lines).encode("utf-8"))
        zf.writestr("README.txt", readme.encode("utf-8"))
    buf.seek(0)
    return buf.read()


def export_bambu_ams_obj_zip_labeled(
    solid: trimesh.Trimesh,
    face_labels: np.ndarray,
    palette: list[dict[str, Any]],
) -> bytes:
    """
    AMS OBJ/MTL ZIP using one shared vertex pool (same topology as ``solid``).

    Face-labeled regions reference global vertex indices so color boundaries do not
    duplicate geometry or introduce spurious open / non-manifold edges.
    """
    if solid.is_empty or len(solid.vertices) == 0 or len(solid.faces) == 0:
        raise ValueError("no solid geometry for AMS OBJ export")
    if len(face_labels) != len(solid.faces):
        raise ValueError("face_labels length must match solid face count")
    verts = np.asarray(solid.vertices, dtype=np.float64)
    faces = np.asarray(solid.faces, dtype=np.int64)
    index_to_meta = {int(p["index"]): p for p in palette}
    mtl_lines = ["# Bambu AMS filament colors\n"]
    obj_lines = [
        "# Terrain print solid — AMS regions (shared vertices)\n",
        "mtllib terrain_ams.mtl\n",
    ]
    obj_lines.extend(
        f"v {_fmt_coord(float(v[0]))} {_fmt_coord(float(v[1]))} {_fmt_coord(float(v[2]))}\n"
        for v in verts
    )
    active_parts = 0
    for idx in sorted(index_to_meta.keys()):
        meta = index_to_meta[idx]
        face_idx = np.where(np.asarray(face_labels, dtype=np.intp) == idx)[0]
        if face_idx.size == 0:
            continue
        mat = str(meta.get("material_name") or meta.get("part_name") or "color")
        r, g, b = _rgb_to_mtl_components(meta["rgb"])
        mtl_lines.append(f"newmtl {mat}\n")
        mtl_lines.append(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
        mtl_lines.append(f"Ka {r:.6f} {g:.6f} {b:.6f}\n")
        mtl_lines.append("Ks 0.200000 0.200000 0.200000\n")
        mtl_lines.append("Ns 10.000000\n\n")
        obj_lines.append(f"o {mat}\n")
        obj_lines.append(f"usemtl {mat}\n")
        for fi in face_idx:
            a = int(faces[int(fi), 0]) + 1
            b = int(faces[int(fi), 1]) + 1
            c = int(faces[int(fi), 2]) + 1
            obj_lines.append(f"f {a} {b} {c}\n")
        active_parts += 1
    if active_parts == 0:
        raise ValueError("no AMS face groups to export")
    readme = (
        "Bambu Studio import\n"
        "-------------------\n"
        "1. File → Import → terrain_ams.obj (or drag this ZIP's OBJ onto the plate).\n"
        "2. If asked to load as one object with multiple parts, choose Yes.\n"
        "3. In the Objects panel, assign each part/material to the closest AMS filament.\n"
        "4. See palette.json for recommended color names and hex values.\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("terrain_ams.mtl", "".join(mtl_lines).encode("utf-8"))
        zf.writestr("terrain_ams.obj", "".join(obj_lines).encode("utf-8"))
        zf.writestr("README.txt", readme.encode("utf-8"))
    buf.seek(0)
    return buf.read()


def export_ams_multicolor_3mf_zip(
    parts: list[tuple[dict[str, Any], trimesh.Trimesh]],
) -> bytes:
    """Experimental multi-object 3MF with Materials ``colorgroup`` per AMS part."""
    if not parts:
        raise ValueError("no AMS parts to export")
    text = io.StringIO()
    text.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    text.write(
        "<model unit=\"millimeter\" xml:lang=\"en-US\" "
        f'xmlns="{_M3MF_NS_CORE}" xmlns:m="{_M3MF_NS_MAT}">\n'
    )
    text.write("<resources>\n")
    object_ids: list[int] = []
    next_id = 1
    for meta, mesh in parts:
        if mesh.is_empty or len(mesh.vertices) == 0:
            continue
        cg_id = next_id
        next_id += 1
        obj_id = next_id
        next_id += 1
        object_ids.append(obj_id)
        hex_c = _hex_rgba_for_3mf(meta["rgb"])
        part_name = str(meta.get("part_name") or meta.get("name") or f"part_{obj_id}")
        text.write(f'<m:colorgroup id="{cg_id}">\n')
        text.write(f'<m:color color="{hex_c}" />\n')
        text.write("</m:colorgroup>\n")
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int64)
        text.write(
            f'<object id="{obj_id}" name="{part_name}" pid="{cg_id}" type="model">\n<mesh>\n<vertices>\n'
        )
        for i in range(len(v)):
            x, y, z = float(v[i, 0]), float(v[i, 1]), float(v[i, 2])
            text.write(
                f'<vertex x="{_fmt_coord(x)}" y="{_fmt_coord(y)}" z="{_fmt_coord(z)}" />\n'
            )
        text.write("</vertices>\n<triangles>\n")
        for fi in range(len(f)):
            a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
            text.write(f'<triangle v1="{a}" v2="{b}" v3="{c}" />\n')
        text.write("</triangles>\n</mesh>\n</object>\n")
    text.write("</resources>\n<build>\n")
    for oid in object_ids:
        text.write(
            f'<item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 0 0 0" />\n'
        )
    text.write("</build>\n</model>\n")
    model_bytes = text.getvalue().encode("utf-8")
    ct = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        "</Types>\n"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" '
        f'Target="/{TEXTURED_3MF_MODEL_PART}" Id="rel0"/>\n'
        "</Relationships>\n"
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct.encode("utf-8"))
        zf.writestr("_rels/.rels", rels.encode("utf-8"))
        zf.writestr(TEXTURED_3MF_MODEL_PART, model_bytes)
    return out.getvalue()


AMS_QUALITY_FACE_TARGETS: dict[str, int] = {
    "high": 400_000,
    "medium": 200_000,
    "low": 80_000,
}


def decimate_for_ams(
    solid: trimesh.Trimesh,
    quality: str,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Return a decimated copy of ``solid`` for AMS labeling/export (STL stays full-res)."""
    from terrain_app.export_options import normalize_ams_quality

    import logging

    q = normalize_ams_quality(quality)
    target = int(AMS_QUALITY_FACE_TARGETS[q])
    n = len(solid.faces)
    meta: dict[str, Any] = {
        "quality": q,
        "target_faces": target,
        "faces_in": n,
        "faces_out": n,
        "skipped": True,
    }
    if n <= target:
        return solid.copy(), meta
    try:
        out = solid.copy()
        out = out.simplify_quadric_decimation(face_count=target)
        out.remove_unreferenced_vertices()
        meta["skipped"] = False
        meta["faces_out"] = int(len(out.faces))
        return out, meta
    except Exception:
        logging.getLogger("terrain_app.mesh").warning(
            "AMS decimation failed; using full mesh", exc_info=True
        )
        meta["error"] = "decimation_failed"
        return solid.copy(), meta


def export_bambu_ams_color_package(
    print_mesh: trimesh.Trimesh,
    surf_mm: trimesh.Trimesh,
    texture_rgba: np.ndarray,
    *,
    n_colors: int = 4,
    mesh_uv: trimesh.Trimesh | None = None,
    dem: np.ndarray | None = None,
    pond_sensitivity: str | None = "conservative",
    voxel_size_mm: float | None = None,
    ams_quality: str = "medium",
    on_progress: Callable[[str], None] | None = None,
) -> tuple[bytes, dict[str, Any], Image.Image, np.ndarray, list[dict[str, Any]], trimesh.Trimesh]:
    """
    Quantize satellite imagery, label the print solid, and package for Bambu Studio.

    Primary export is OBJ/MTL ZIP (best Bambu color import compatibility).
    Returns ``(zip_bytes, meta, preview, face_labels, palette)`` for optional GLB packaging.
    Labeling uses a decimated copy of ``print_mesh``; STL exports use the full solid.
    """
    from terrain_app import ams_color

    n_colors = ams_color.clamp_ams_n_colors(n_colors)
    if on_progress:
        on_progress("Simplifying mesh for AMS color labeling…")
    ams_mesh, decimation_meta = decimate_for_ams(print_mesh, ams_quality)
    ams_mesh_uv = print_solid_with_satellite_uv(ams_mesh, surf_mm)
    palette, index_image, face_labels = ams_color.build_ams_labels(
        ams_mesh,
        surf_mm,
        texture_rgba,
        print_solid_with_satellite_uv=print_solid_with_satellite_uv,
        n_colors=n_colors,
        mesh_uv=ams_mesh_uv,
        dem=dem,
        pond_sensitivity=pond_sensitivity,
        voxel_size_mm=voxel_size_mm,
        on_progress=on_progress,
    )
    active_count = sum(1 for p in palette if int(p.get("triangle_count") or 0) > 0)
    if active_count == 0:
        raise ValueError("AMS export produced no mesh parts")
    pal_rgb = np.array([p["rgb"] for p in palette], dtype=np.uint8)
    if texture_rgba.shape[2] >= 4:
        footprint = np.asarray(texture_rgba[:, :, 3], dtype=np.uint8) >= ams_color.MASK_ALPHA_MIN
    else:
        footprint = None
    preview = ams_color.render_quantized_preview(index_image, pal_rgb, footprint=footprint)
    if on_progress:
        on_progress("Writing Bambu AMS OBJ package…")
    zip_bytes = export_bambu_ams_obj_zip_labeled(ams_mesh, face_labels, palette)
    import json

    palette_json = json.dumps(
        {
            "colors": palette,
            "import_hint": (
                "In Bambu Studio: import terrain_ams.obj, accept multi-part assembly if "
                "prompted, then assign each material/part to the closest AMS filament. "
                "01_base is the predominant color on walls, underside, and matching top "
                "terrain; other parts are accent top-surface colors only."
            ),
        },
        indent=2,
    )
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("palette.json", palette_json.encode("utf-8"))
    zip_bytes = buf.getvalue()
    meta: dict[str, Any] = {
        "ok": True,
        "format": "obj_mtl_zip",
        "filename": "terrain_print_ams_obj.zip",
        "download_suffix": "_print_ams_obj.zip",
        "n_colors": len(palette),
        "max_colors": ams_color.AMS_MAX_COLORS,
        "part_count": active_count,
        "color_embedded": True,
        "open_edge_count": int(open_edge_count(print_mesh)),
        "non_manifold_edge_count": int(non_manifold_edge_count(ams_mesh)),
        "ams_decimation": decimation_meta,
        "colors": palette,
        "import_hint": (
            "In Bambu Studio: import terrain_ams.obj from the ZIP, keep parts aligned as one "
            "assembly if prompted, then assign each material region to the recommended AMS "
            "filament colors shown below. Use 01_base for walls, underside, and the "
            "dominant top terrain color; map accent top colors to other AMS slots."
        ),
    }
    return zip_bytes, meta, preview, face_labels, palette, ams_mesh


def generate_bambu_ams_fixtures(out_dir: str | Path) -> dict[str, Path]:
    """
    Write tiny multi-color fixtures for manual Bambu Studio import testing.

    Creates colored 3MF, OBJ/MTL ZIP, and multi-part OBJ ZIP under ``out_dir``.
    """
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    palette = [
        {
            "index": 0,
            "rgb": [40, 95, 143],
            "hex": "#285F8F",
            "name": "Water Blue",
            "part_name": "01_water_blue",
            "material_name": "01_water_blue",
        },
        {
            "index": 1,
            "rgb": [27, 77, 42],
            "hex": "#1B4D2A",
            "name": "Forest Green",
            "part_name": "02_forest_green",
            "material_name": "02_forest_green",
        },
        {
            "index": 2,
            "rgb": [90, 160, 70],
            "hex": "#5AA046",
            "name": "Green",
            "part_name": "03_green",
            "material_name": "03_green",
        },
        {
            "index": 3,
            "rgb": [130, 130, 130],
            "hex": "#828282",
            "name": "Gray",
            "part_name": "04_gray",
            "material_name": "04_gray",
        },
    ]
    parts: list[tuple[dict[str, Any], trimesh.Trimesh]] = []
    offsets = [(0, 0), (12, 0), (0, 12), (12, 12)]
    for meta, (ox, oy) in zip(palette, offsets):
        box = trimesh.creation.box(extents=(10, 10, 3))
        box.apply_translation((ox, oy, 1.5))
        parts.append((meta, solid_process_for_export(box)))
    paths: dict[str, Path] = {}
    p3mf = out / "fixture_ams_colored.3mf"
    p3mf.write_bytes(export_ams_multicolor_3mf_zip(parts))
    paths["colored_3mf"] = p3mf
    pobj = out / "fixture_ams_obj.zip"
    obj_zip = export_bambu_ams_obj_zip(parts)
    buf = io.BytesIO(obj_zip)
    with zipfile.ZipFile(buf, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("palette.json", json.dumps({"colors": palette}, indent=2).encode("utf-8"))
    pobj.write_bytes(buf.getvalue())
    paths["obj_mtl_zip"] = pobj
    pparts = out / "fixture_ams_parts.zip"
    pbuf = io.BytesIO()
    with zipfile.ZipFile(pbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for meta, mesh in parts:
            name = f"{meta['part_name']}.obj"
            ob = io.BytesIO()
            mesh.export(file_obj=ob, file_type="obj")
            zf.writestr(name, ob.getvalue())
        zf.writestr("palette.json", json.dumps({"colors": palette}, indent=2).encode("utf-8"))
    pparts.write_bytes(pbuf.getvalue())
    paths["parts_zip"] = pparts
    return paths

