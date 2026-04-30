"""Build textured mesh and export GLB / OBJ for Blender, and a print-ready solid."""

from __future__ import annotations

import glob
import io
import json
import os
import tempfile
import time
import zipfile
from typing import Any

import numpy as np
import rasterio.transform
import trimesh
import trimesh.boolean
import trimesh.repair
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import Polygon as Polygon2D
from shapely.prepared import prep

# region agent log
# Use package-local log (`.cursor/...` may be unwritable in some sandboxes; copy to .cursor for IDE ingest if needed).
_AGENT_DEBUG_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), "debug-0a1df2.log"))


def _agent_debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    try:
        os.makedirs(os.path.dirname(_AGENT_DEBUG_LOG), exist_ok=True)
        payload = {
            "sessionId": "0a1df2",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as _df:
            _df.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError:
        pass


# endregion


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

    Quads are kept only if their ground footprint intersects ``poly_utm`` when given.
    ``mask`` drives the texture alpha channel (RGBA).
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

    verts = np.asarray(verts, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    faces = []
    vid = np.arange(h * w).reshape(h, w)
    prepared_poly = prep(poly_utm) if poly_utm is not None else None
    poly_bounds = poly_utm.bounds if poly_utm is not None else None
    east, north = _cell_center_east_north(transform, h, w)
    for i in range(h - 1):
        for j in range(w - 1):
            if not _include_quad(i, j, east, north, prepared_poly, poly_bounds, mask):
                continue
            v00 = vid[i, j]
            v10 = vid[i + 1, j]
            v01 = vid[i, j + 1]
            v11 = vid[i + 1, j + 1]
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])
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


def export_glb(mesh: trimesh.Trimesh, *, center_xz: bool = False) -> bytes:
    if center_xz:
        m = mesh.copy()
        _center_mesh_xz(m)
        return m.export(file_type="glb")
    return mesh.export(file_type="glb")


def export_obj_zip(mesh: trimesh.Trimesh, *, center_xz: bool = False) -> bytes:
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "terrain.obj")
        if center_xz:
            m = mesh.copy()
            _center_mesh_xz(m)
            m.export(path)
        else:
            mesh.export(path)
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


def _meters_to_print_mm_scale(surface: trimesh.Trimesh, print_max_size_mm: float) -> float:
    """Meters in scene → millimetre units so the terrain’s max horizontal (XY) span equals ``print_max_size_mm``."""
    ext = _horizontal_extent_utm_m(surface)
    if ext < 1e-9:
        return 1.0
    return float(print_max_size_mm) / ext


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


def _voxel_merge_filled(merged: trimesh.Trimesh, voxel_size_mm: float) -> trimesh.Trimesh:
    """One filled voxel pass (mesh already Z-up XY + elevation)."""
    cz = merged.copy()
    vg = cz.voxelized(float(voxel_size_mm), method="ray")
    vg = vg.fill()
    out = vg.marching_cubes
    return solid_process_for_export(out)


def _last_resort_voxel_fuse(
    solid: trimesh.Trimesh,
    voxel_size_mm: float,
    print_max_size_mm: float,
) -> trimesh.Trimesh:
    """If multiple bodies remain, concat and voxelize; retry with finer voxels if needed."""
    out = solid
    vs = min(float(voxel_size_mm), max(0.15, float(print_max_size_mm) / 200.0))
    # region agent log
    _agent_debug_log(
        hypothesis_id="C",
        location="mesh.py:_last_resort_voxel_fuse:entry",
        message="last_resort_start",
        data={
            "parts_in": len(list(solid.split(only_watertight=False))),
            "vs0": round(vs, 6),
            "print_max": float(print_max_size_mm),
        },
    )
    # endregion
    for iter_i in range(4):
        parts = list(out.split(only_watertight=False))
        if len(parts) <= 1:
            # region agent log
            _agent_debug_log(
                hypothesis_id="C",
                location="mesh.py:_last_resort_voxel_fuse:exit_early",
                message="last_resort_single",
                data={"iter": iter_i, "parts": len(parts)},
            )
            # endregion
            return out
        union_try = _try_manifold_union(parts)
        if union_try is not None and len(union_try.split(only_watertight=False)) <= 1:
            # region agent log
            _agent_debug_log(
                hypothesis_id="E",
                location="mesh.py:_last_resort_voxel_fuse:manifold_union",
                message="last_resort_union_ok",
                data={"iter": iter_i},
            )
            # endregion
            return union_try
        merged = trimesh.util.concatenate(parts)
        merged = solid_process_for_export(merged)
        out = _voxel_merge_filled(merged, float(vs))
        n_after = len(out.split(only_watertight=False))
        # region agent log
        _agent_debug_log(
            hypothesis_id="C",
            location="mesh.py:_last_resort_voxel_fuse:iter",
            message="last_resort_after_voxel",
            data={"iter": iter_i, "vs": round(vs, 6), "parts_after": n_after},
        )
        # endregion
        if n_after <= 1:
            return out
        vs = max(0.1, vs * 0.5)
    # region agent log
    _agent_debug_log(
        hypothesis_id="C",
        location="mesh.py:_last_resort_voxel_fuse:exhausted",
        message="last_resort_still_multi",
        data={"parts_final": len(list(out.split(only_watertight=False)))},
    )
    # endregion
    return out


def _fuse_stacked_components(
    solid: trimesh.Trimesh,
    voxel_size_mm: float,
) -> tuple[trimesh.Trimesh, bool]:
    """
    If meshing produced vertically stacked disconnected shells, close vertical gaps by translation
    (no fragile XZ AABB intersection test), then voxel-merge into one body.
    """
    parts = list(solid.split(only_watertight=False))
    # region agent log
    _agent_debug_log(
        hypothesis_id="B",
        location="mesh.py:_fuse_stacked_components:entry",
        message="fuse_entry",
        data={"parts_in": len(parts), "vs": round(float(voxel_size_mm), 6)},
    )
    # endregion
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
        # region agent log
        _agent_debug_log(
            hypothesis_id="E",
            location="mesh.py:_fuse_stacked_components:manifold_union",
            message="fuse_manifold_union_ok",
            data={"parts_out": len(list(out.split(only_watertight=False)))},
        )
        # endregion
        return out, True
    merged = trimesh.util.concatenate(parts)
    merged = solid_process_for_export(merged)
    out = _voxel_merge_filled(merged, float(voxel_size_mm))
    # region agent log
    _agent_debug_log(
        hypothesis_id="B",
        location="mesh.py:_fuse_stacked_components:after_voxel",
        message="fuse_after_voxel_merge",
        data={"parts_out": len(list(out.split(only_watertight=False)))},
    )
    # endregion
    return out, True


def build_print_solid(
    surface: trimesh.Trimesh,
    poly_utm: Polygon2D,
    print_max_size_mm: float = 200.0,
    base_extrusion_mm: float = 1.0,
    center_on_bed: bool = True,
    voxel_size_mm: float | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any], trimesh.Trimesh]:
    """
    Build a **single closed print solid** in millimetres (**Z-up**: X east, Y north, Z elevation).
    Build plate is **Z = 0**; terrain sits above a 1.0 mm base. Horizontal scale fits the longer
    **XY** span to ``print_max_size_mm``. ``base_extrusion_mm`` is API-only; base thickness is 1 mm.

    Returns ``(solid, meta, surf_mm)`` where ``surf_mm`` is the scaled, centered open terrain (mm)
    with texture—used to paint the watertight solid for 3MF export.
    """
    if not poly_utm.is_valid:
        poly_utm = poly_utm.buffer(0)
    if surface.is_empty or len(surface.vertices) == 0:
        raise ValueError("Cannot build print solid from empty surface mesh")
    s = _meters_to_print_mm_scale(surface, print_max_size_mm)
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

    if center_on_bed:
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
        cz = terrain_solid.copy()
        vg = cz.voxelized(float(vs), method="ray")
        vg = vg.fill()
        solid = vg.marching_cubes
        solid = solid_process_for_export(solid)
    elif vs <= 0:
        # Keep metadata stable when explicit voxel size is not in use.
        vs = max(0.25, float(print_max_size_mm) / 150.0)
    n_before_fuse = len(solid.split(only_watertight=False))
    # region agent log
    _agent_debug_log(
        hypothesis_id="A",
        location="mesh.py:build_print_solid:before_fuse",
        message="pre_fuse_state",
        data={
            "n_before_fuse": n_before_fuse,
            "vs": round(float(vs), 6),
            "watertight": bool(solid.is_watertight),
            "used_voxel_fallback": bool(not solid.is_watertight or abs(float(solid.volume)) <= 1e-9),
        },
    )
    # endregion
    solid, _ = _fuse_stacked_components(solid, float(vs))
    n_after_fuse = len(solid.split(only_watertight=False))
    # region agent log
    _agent_debug_log(
        hypothesis_id="B",
        location="mesh.py:build_print_solid:after_fuse",
        message="post_fuse_state",
        data={"n_after_fuse": n_after_fuse},
    )
    # endregion
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
    # region agent log
    _agent_debug_log(
        hypothesis_id="D",
        location="mesh.py:build_print_solid:after_plate_snap",
        message="final_components",
        data={
            "n_components_final": int(n_components),
            "last_resort": last_resort,
        },
    )
    # endregion
    sb2 = solid.bounds
    if sb2 is not None:
        d_mm = (float(sb2[1, 0] - sb2[0, 0]), float(sb2[1, 1] - sb2[0, 1]))
        w_max_mm = max(d_mm[0], d_mm[1])
    else:
        d_mm, w_max_mm = (0.0, 0.0), 0.0
    ext0 = _horizontal_extent_utm_m(surface)
    w_target = (float(ext0) * s) if ext0 > 0 else 0.0
    meta: dict[str, Any] = {
        "units": "mm",
        "z_up": True,
        "axes": "X=east_mm, Y=north_mm, Z=elevation_mm",
        "print_max_size_mm": float(print_max_size_mm),
        "base_extrusion_mm": be,
        "scale_meters_to_print_mm": float(s),
        "print_voxel_size_mm": float(vs),
        "print_xy_extent_max_mm": float(min(w_target, print_max_size_mm) if w_target > 0 else 0.0),
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


def orient_print_mesh_for_slicer(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Print mesh is already X east, Y north, Z elevation (mm); no axis permutation for slicers."""
    return m.copy()


def export_print_stl(solid: trimesh.Trimesh) -> bytes:
    """Watertight print solid for FDM slicers (Z-up, XY bed)."""
    return orient_print_mesh_for_slicer(solid).export(file_type="stl")


def export_print_3mf(solid: trimesh.Trimesh) -> bytes:
    m = finalize_mesh_for_3mf_export(orient_print_mesh_for_slicer(solid))
    return m.export(file_type="3mf")


def export_print_glb(solid: trimesh.Trimesh) -> bytes:
    return orient_print_mesh_for_slicer(solid).export(file_type="glb")


def split_solid_to_xz_grid(
    solid: trimesh.Trimesh,
    nx: int,
    nz: int,
) -> list[dict[str, Any]]:
    """
    Split a closed mesh in **Z-up mm** (X east, Y north, Z elevation) into an ``nx``×``nz`` grid
    in the **XY** plane; each piece spans the full **Z** extent of the solid. Parameter names
    ``nx`` / ``nz`` are kept for the API: ``nx`` = columns along **X**, ``nz`` = rows along **Y**
    (northing).
    """
    nxi = int(max(1, nx))
    nzi = int(max(1, nz))
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
                    "filename": f"terrain_print_r{iy + 1:02d}c{ix + 1:02d}.stl",
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
