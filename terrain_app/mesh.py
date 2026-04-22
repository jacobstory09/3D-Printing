"""Build textured mesh and export GLB / OBJ for Blender, and a print-ready solid."""

from __future__ import annotations

import glob
import io
import os
import tempfile
import zipfile
from typing import Any

import numpy as np
import rasterio.transform
import trimesh
import trimesh.boolean
from PIL import Image
from shapely.geometry import Polygon as Polygon2D
from shapely.prepared import prep


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
    """Vertices match the web viewer (Three.js Y-up): X=easting, Y=elevation, Z=northing.

    Quads are kept only if their **ground footprint** (UTM east/north) intersects ``poly_utm``
    when given (same boundary as the KML). A cheap bbox test skips most exterior quads.

    ``mask`` still drives the texture alpha channel (RGBA) to match the web viewer.
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
            # py = UTM northing, px = easting — same frame as viewer.js buildGeometry (x,z ground + y up)
            verts.append([px, z[i, j], py])
            u = (j + 0.5) / w
            v = 1.0 - (i + 0.5) / h
            uvs.append([u, v])

    verts = np.asarray(verts, dtype=np.float64)
    uvs = np.asarray(uvs, dtype=np.float64)
    faces = []
    vid = np.arange(h * w).reshape(h, w)
    prepared_poly = prep(poly_utm) if poly_utm is not None else None
    poly_bounds = poly_utm.bounds if poly_utm is not None else None
    for i in range(h - 1):
        for j in range(w - 1):
            if prepared_poly is not None and poly_bounds is not None:
                a, b, c, d = vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]
                es = (verts[a, 0], verts[b, 0], verts[c, 0], verts[d, 0])
                ns = (verts[a, 2], verts[b, 2], verts[c, 2], verts[d, 2])
                qminx, qmaxx = min(es), max(es)
                qminy, qmaxy = min(ns), max(ns)
                pb_w, pb_s, pb_e, pb_n = poly_bounds
                if qmaxx < pb_w or qminx > pb_e or qmaxy < pb_s or qminy > pb_n:
                    continue
                try:
                    ring = [(float(es[0]), float(ns[0])), (float(es[1]), float(ns[1])), (float(es[2]), float(ns[2])), (float(es[3]), float(ns[3]))]
                    qfoot = Polygon2D(ring)
                    if not qfoot.is_valid:
                        qfoot = qfoot.buffer(0)
                    if qfoot.is_empty or not prepared_poly.intersects(qfoot):
                        continue
                except Exception:
                    continue
            elif mask is not None:
                blk = mask[i : i + 2, j : j + 2]
                if not np.any(blk >= 128):
                    continue
            v00 = vid[i, j]
            v10 = vid[i + 1, j]
            v01 = vid[i, j + 1]
            v11 = vid[i + 1, j + 1]
            # Winding reversed vs old Z-up layout so normals stay outward after Y-up remap
            faces.append([v00, v01, v10])
            faces.append([v10, v01, v11])
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


def export_glb(mesh: trimesh.Trimesh) -> bytes:
    return mesh.export(file_type="glb")


def export_obj_zip(mesh: trimesh.Trimesh) -> bytes:
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "terrain.obj")
        mesh.export(path)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in glob.glob(os.path.join(td, "*")):
                if os.path.isfile(p):
                    zf.write(p, arcname=os.path.basename(p))
    buf.seek(0)
    return buf.read()


# --- Y-up (X east, Y up, Z north) to trimesh’s Z-up (ground XY, +Z extrude / +Z ray tests) ---

# World = (X_east, Y_elev, Z_north) → trimesh: (X_t, Y_t, Z_t) = (X_w, Z_w, Y_w)
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _poly_utm_in_extrude_frame(poly_utm: Polygon2D) -> Polygon2D:
    """(easting, northing) in UTM and trimesh extrude 2D frame are the same (X, Y) before rotation."""
    return poly_utm


def extrude_footprint_solid_utm(
    poly_utm: Polygon2D,
    height_m: float,
) -> trimesh.Trimesh:
    """
    Prismatic block under the KML footprint: extrude along -Y in world (down),
    in UTM units (1 m). Result is in Y-up world coords (X east, Y up, Z north).
    """
    if not poly_utm.is_valid or poly_utm.is_empty:
        raise ValueError("print_base_extrusion_m requires a non-empty KML footprint polygon")
    h = float(np.abs(float(height_m)))
    if h <= 0:
        return trimesh.Trimesh()
    poly2 = _poly_utm_in_extrude_frame(poly_utm)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    # trimesh extrude along -Z, then R maps that to -Y (down) in our frame
    block_tm = trimesh.creation.extrude_polygon(
        poly2, height=-h, engine="earcut"  # -Z in creation frame
    )
    out = trimesh.Trimesh(vertices=block_tm.vertices, faces=block_tm.faces, process=True)
    out.apply_transform(_Y_UP_TO_Z_UP)
    return out


def _horizontal_extent_utm_m(surface: trimesh.Trimesh) -> float:
    b = surface.bounds
    if b is None and len(surface.vertices) > 0:
        v = np.asarray(surface.vertices, dtype=np.float64)
        b = np.array([v.min(axis=0), v.max(axis=0)], dtype=np.float64)
    if b is None:
        return 0.0
    return float(max(b[1, 0] - b[0, 0], b[1, 2] - b[0, 2]))


def _meters_to_print_mm_scale(surface: trimesh.Trimesh, print_max_size_mm: float) -> float:
    """Meters in scene → millimetre units so the terrain’s max (X/Z) span equals ``print_max_size_mm``."""
    ext = _horizontal_extent_utm_m(surface)
    if ext < 1e-9:
        return 1.0
    return float(print_max_size_mm) / ext


def build_print_solid(
    surface: trimesh.Trimesh,
    poly_utm: Polygon2D,
    print_max_size_mm: float = 200.0,
    base_extrusion_mm: float = 2.0,
    center_on_bed: bool = True,
    voxel_size_mm: float | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """
    Fuse a prismatic **KML-footprint** base and the terrain skin into a **watertight** print mesh in
    **millimetres**, Y up, build plate at Y=0. Horizontal scale is chosen from the **surface** so the
    longer ground edge equals ``print_max_size_mm``. The prismatic block under the footprint is
    ``base_extrusion_mm`` tall. Open terrain plus base are merged via **voxelization and marching
    cubes** (trimesh uses scikit-image), since boolean CSG requires closed volumes and the
    heightfield is not a volume.
    """
    if not poly_utm.is_valid:
        poly_utm = poly_utm.buffer(0)
    if surface.is_empty or len(surface.vertices) == 0:
        raise ValueError("Cannot build print solid from empty surface mesh")
    s = _meters_to_print_mm_scale(surface, print_max_size_mm)
    # In metres: need depth d such that d * s = base_extrusion_mm
    be = float(max(0.0, base_extrusion_mm))
    depth_m = (be / s) if s > 1e-12 and be > 0 else 0.0
    if depth_m > 0:
        base_utm = extrude_footprint_solid_utm(poly_utm, depth_m)
    else:
        base_utm = trimesh.Trimesh()

    surf = surface.copy()
    if len(base_utm.vertices) > 0:
        bms = base_utm.copy()
        bms.apply_scale(s)
    else:
        bms = trimesh.Trimesh()
    surf.apply_scale(s)

    lo_y = 1e9
    if len(bms.vertices) > 0:
        lo_y = min(lo_y, float(bms.bounds[0, 1]))
    if len(surf.vertices) > 0:
        lo_y = min(lo_y, float(surf.bounds[0, 1]))
    t_up = trimesh.transformations.translation_matrix([0.0, -lo_y, 0.0])
    if len(bms.vertices) > 0:
        bms.apply_transform(t_up)
    if len(surf.vertices) > 0:
        surf.apply_transform(t_up)
    if center_on_bed:
        comb = bms + surf
        cmin, cmax = comb.bounds[0], comb.bounds[1]
        cx, cz = 0.5 * (cmin[0] + cmax[0]), 0.5 * (cmin[2] + cmax[2])
        t_center = trimesh.transformations.translation_matrix([-cx, 0.0, -cz])
        if len(bms.vertices) > 0:
            bms.apply_transform(t_center)
        surf.apply_transform(t_center)
    if len(surf.vertices) == 0:
        raise ValueError("Surface mesh has no geometry for print export")
    if len(bms.vertices) == 0:
        combined = surf
    else:
        combined = bms + surf
    # The terrain is an open heightfield; boolean union requires closed volumes, so we voxelize
    # and isosurface to a single solid. Ray voxelization (default) assumes +Z; rotate Y-up → Z-up first.
    vs = float(voxel_size_mm) if (voxel_size_mm is not None and float(voxel_size_mm) > 0) else 0.0
    if vs <= 0:
        vs = max(0.25, float(print_max_size_mm) / 150.0)
    cz = combined.copy()
    cz.apply_transform(_Y_UP_TO_Z_UP)
    vg = cz.voxelized(float(vs), method="ray")
    solid = vg.marching_cubes
    t_inv = trimesh.transformations.inverse_matrix(_Y_UP_TO_Z_UP)
    solid.apply_transform(t_inv)
    solid = solid_process_for_export(solid)
    # Ray+marching can sit half a voxel below Y=0; rest the solid on the build plate.
    sb = solid.bounds
    if sb is not None and sb[0, 1] < 0:
        solid.apply_translation([0.0, -float(sb[0, 1]), 0.0])
    sb2 = solid.bounds
    if sb2 is not None:
        d_mm = (float(sb2[1, 0] - sb2[0, 0]), float(sb2[1, 2] - sb2[0, 2]))
        w_max_mm = max(d_mm[0], d_mm[1])
    else:
        d_mm, w_max_mm = (0.0, 0.0), 0.0
    ext0 = _horizontal_extent_utm_m(surface)
    w_target = (float(ext0) * s) if ext0 > 0 else 0.0
    meta: dict[str, Any] = {
        "units": "mm",
        "y_up": True,
        "print_max_size_mm": float(print_max_size_mm),
        "base_extrusion_mm": be,
        "scale_meters_to_print_mm": float(s),
        "print_voxel_size_mm": float(vs),
        "print_xy_extent_max_mm": float(min(w_target, print_max_size_mm) if w_target > 0 else 0.0),
        "print_full_size_mm": {
            "x": max(d_mm[0], 0.0),
            "z": max(d_mm[1], 0.0),
            "max_horizontal_mm": w_max_mm,
        },
    }
    return solid, meta


def solid_process_for_export(m: trimesh.Trimesh) -> trimesh.Trimesh:
    t = m.copy()
    t.update_faces(t.nondegenerate_faces() & t.unique_faces())
    t.remove_unreferenced_vertices()
    t.fill_holes()
    t.merge_vertices(merge_norm=False)
    return t


def export_stl(solid: trimesh.Trimesh) -> bytes:
    return solid.export(file_type="stl")


def split_solid_to_xz_grid(
    solid: trimesh.Trimesh,
    nx: int,
    nz: int,
) -> list[dict[str, Any]]:
    """
    Split a closed mesh in **Y-up mm** with axis-aligned XZ cells (``nx`` by ``nz`` in plan).
    Each cell is a puzzle piece: clip the solid to the XZ box spanning full Y (keeps a flat
    bottom). Returns dicts with ``id`` (1-based row-major: row ``iz`` south→north, col ``ix``),
    ``ix`` (0..nx-1), ``iz`` (0..nz-1), and ``mesh``.
    """
    nxi = int(max(1, nx))
    nzi = int(max(1, nz))
    if solid.is_empty or len(solid.vertices) == 0:
        return []
    b = solid.bounds
    if b is None:
        return []
    # Tiny inflation so coplanar boundary faces are included in intersection.
    pad = 0.02
    b = np.array(b, dtype=np.float64)
    b[0, 0] -= pad
    b[1, 0] += pad
    b[0, 2] -= pad
    b[1, 2] += pad
    b[0, 1] -= pad
    b[1, 1] += pad
    out: list[dict[str, Any]] = []
    total = nxi * nzi
    for iz in range(nzi):
        z0 = b[0, 2] + (b[1, 2] - b[0, 2]) * (iz / float(nzi))
        z1 = b[0, 2] + (b[1, 2] - b[0, 2]) * ((iz + 1) / float(nzi))
        for ix in range(nxi):
            x0 = b[0, 0] + (b[1, 0] - b[0, 0]) * (ix / float(nxi))
            x1 = b[0, 0] + (b[1, 0] - b[0, 0]) * ((ix + 1) / float(nxi))
            cx, cy, cz = 0.5 * (x0 + x1), 0.5 * (b[0, 1] + b[1, 1]), 0.5 * (z0 + z1)
            ex, ey, ez = max((x1 - x0), 1e-6), max((b[1, 1] - b[0, 1]), 1e-6), max((z1 - z0), 1e-6)
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
            k = 1 + ix + iz * nxi
            out.append(
                {
                    "id": k,
                    "ix": ix,
                    "iz": iz,
                    "row": iz + 1,
                    "col": ix + 1,
                    "n_columns": nxi,
                    "n_rows": nzi,
                    "index_of_total": k,
                    "total_pieces": total,
                    "filename": f"terrain_print_r{iz + 1:02d}c{ix + 1:02d}.stl",
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
    """Zip of one STL per non-empty cell: ``terrain_print_r{row}c{col}.stl`` (1-based, Z then X)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pieces:
            m = p["mesh"]
            name = str(p.get("filename") or f"piece_{p.get('id', 0)}.stl")
            zf.writestr(name, export_stl(m))
    buf.seek(0)
    return buf.read()
