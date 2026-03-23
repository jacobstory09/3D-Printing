"""Build textured mesh and export GLB / OBJ for Blender."""

from __future__ import annotations

import glob
import io
import os
import tempfile
import zipfile

import numpy as np
import rasterio.transform
import trimesh
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
