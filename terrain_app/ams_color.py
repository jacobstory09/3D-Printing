"""Quantize satellite imagery to AMS filament colors and split print solids by region."""

from __future__ import annotations

import colorsys
import re
from typing import Any, Callable

import numpy as np
import trimesh
import trimesh.proximity
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import label as ndimage_label
from scipy.stats import mode as scipy_mode
from skimage.color import lab2rgb, rgb2lab

# Bambu AMS practical limit (dual AMS / merge workflow); hard cap for exports.
AMS_MAX_COLORS = 8
# Minimum perceptual separation (ΔE in CIELAB) between base and land accent swatches.
LAND_ACCENT_MIN_SEP = 8.0
LAND_ACCENT_MIN_SEP_SQ = LAND_ACCENT_MIN_SEP * LAND_ACCENT_MIN_SEP
# Squared Lab distance for folding near-base top colors into the base slot (~10 ΔE).
MERGE_INDEX_SLOP_LAB_SQ = 100.0
LAND_ACCENT_MIN_PIXELS = 32
# Mask threshold matches mesh clipping alpha semantics.
MASK_ALPHA_MIN = 128
POND_MIN_PIXELS = 12
POND_MIN_BLOB_PIXELS = 12
POND_MAX_BLOB_PIXELS = 600  # default floor; scaled up per footprint in detect_pond_mask
POND_MAX_BLOB_FOOTPRINT_FRAC = 0.42
POND_MIN_FILL = 0.18
POND_LARGE_BLOB_PIXELS = 2000
POND_LARGE_MIN_FILL = 0.08
POND_MAX_ASPECT = 10.0
POND_UNIFORM_STD_MAX = 10.0
POND_MUDDY_SAT_MAX = 0.36
# Recommended AMS filament when water is detected (imagery is often muddy tan, not blue).
POND_FILAMENT_RGB = (45, 100, 135)
POND_SENSITIVITY_CHOICES = ("conservative", "balanced", "aggressive")
POND_SENSITIVITY_PARAMS: dict[str, dict[str, float | bool]] = {
    "conservative": {
        "elev_percentile": 20,
        "elev_margin_m": 0.5,
        "max_coverage_frac": 0.15,
        "ring_drop_m": 0.5,
        "require_ring_for_assisted": True,
    },
    "balanced": {
        "elev_percentile": 25,
        "elev_margin_m": 0.75,
        "max_coverage_frac": 0.20,
        "ring_drop_m": 0.35,
        "require_ring_for_assisted": True,
    },
    "aggressive": {
        "elev_percentile": 35,
        "elev_margin_m": 1.0,
        "max_coverage_frac": 0.28,
        "ring_drop_m": 0.25,
        "require_ring_for_assisted": False,
    },
}
# Face centroid within this distance (mm) of the open top surface counts as "top".
TOP_FACE_PROXIMITY_MM = 0.75
# Upward-facing normal Z component required for a face to count as printable top.
TOP_FACE_NORMAL_Z_MIN = 0.35
# Barycentric interior UV sample weights for majority-vote face labeling.
_FACE_LABEL_BARY_WEIGHTS = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=np.float64,
)


def _rgb_uint8_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 (…, 3) to CIELAB float64 (…, 3)."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    scale = 255.0 if arr.max() > 1.0 else 1.0
    return rgb2lab(arr / scale)


def _lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB float64 (N, 3) to sRGB uint8."""
    lab_arr = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    if lab_arr.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rgb = lab2rgb(lab_arr)
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _refine_palette_lab_means(pixels: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """Recompute median-cut seeds as Lab-space means of their assigned pixels."""
    px = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    pal = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)
    if px.size == 0 or pal.size == 0:
        return pal
    px_lab = _rgb_uint8_to_lab(px)
    pal_lab = _rgb_uint8_to_lab(pal)
    labels = np.argmin(
        np.sum((px_lab[:, None, :] - pal_lab[None, :, :]) ** 2, axis=2), axis=1
    )
    refined_lab = np.zeros_like(pal_lab)
    for i in range(len(pal)):
        sel = labels == i
        refined_lab[i] = px_lab[sel].mean(axis=0) if sel.any() else pal_lab[i]
    return _lab_to_rgb_uint8(refined_lab)


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
    return s.strip("_") or "color"


def _rgb_to_filament_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if s < 0.12:
        return "Light Gray" if v > 0.62 else "Gray"
    if 0.50 <= h <= 0.72:
        return "Water Blue" if v < 0.55 else "Blue"
    if 0.20 <= h < 0.50:
        return "Forest Green" if v < 0.42 else "Green"
    if 0.05 <= h < 0.20:
        return "Brown" if v < 0.45 else "Tan"
    if h < 0.05 or h > 0.92:
        return "Rust" if v < 0.5 else "Red"
    return "Green"


def _hex_rgb(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def normalize_pond_sensitivity(value: str | None) -> str:
    s = (value or "conservative").lower().strip()
    if s not in POND_SENSITIVITY_CHOICES:
        return "conservative"
    return s


def clamp_ams_n_colors(n_colors: int | float | None) -> int:
    """Clamp requested AMS filament slots to the Bambu-safe range."""
    try:
        n = int(n_colors if n_colors is not None else 4)
    except (TypeError, ValueError):
        n = 4
    return int(max(2, min(AMS_MAX_COLORS, n)))


def _pond_sensitivity_params(sensitivity: str | None) -> dict[str, float | bool]:
    return dict(POND_SENSITIVITY_PARAMS[normalize_pond_sensitivity(sensitivity)])


def _blob_looks_like_water(
    med_rgb: np.ndarray,
    median_brightness: float,
    *,
    strict_muddy: bool = False,
) -> bool:
    """Heuristic for pond/open-water blobs in aerial/satellite RGB."""
    r, g, b = (int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2]))
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if 0.38 <= h <= 0.75 and s >= 0.05 and v >= 0.08:
        return True
    if median_brightness < 36:
        return True
    if v < 0.30 and b >= r - 6 and (int(g) + int(b)) > int(r) + 6:
        return True
    # Muddy / tannic open water in Esri (similar brightness to pasture, low saturation).
    if s < POND_MUDDY_SAT_MAX and 0.22 <= v <= 0.85:
        if strict_muddy:
            return b >= r - 4 and (int(g) + int(b)) > int(r) + 8
        return True
    return False


def _max_pond_blob_pixels(footprint: np.ndarray) -> int:
    n = int(np.count_nonzero(footprint))
    return int(max(POND_MAX_BLOB_PIXELS, n * POND_MAX_BLOB_FOOTPRINT_FRAC))


def _local_brightness_std(bright: np.ndarray, footprint: np.ndarray, size: int = 9) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    b = np.where(footprint, bright, 0.0)
    m = uniform_filter(b, size=size)
    m2 = uniform_filter(b * b, size=size)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _muddy_water_seed(rgb: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Uniform, low-saturation regions — catches tannic lakes that match pasture color."""
    bright = rgb.max(axis=2).astype(np.float64)
    mx = rgb.max(axis=2).astype(np.float64)
    mn = rgb.min(axis=2).astype(np.float64)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    local_std = _local_brightness_std(bright, footprint)
    return (
        footprint
        & (local_std <= POND_UNIFORM_STD_MAX)
        & (sat <= POND_MUDDY_SAT_MAX)
        & (bright >= 40.0)
        & (bright <= 215.0)
    )


def _dem_depression_seed(dem: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Flat local elevation lows — water surfaces in gentle terrain."""
    from scipy.ndimage import minimum_filter, uniform_filter

    if dem.shape != footprint.shape:
        return np.zeros(footprint.shape, dtype=bool)
    valid = footprint & np.isfinite(dem)
    if not valid.any():
        return np.zeros(footprint.shape, dtype=bool)
    fill = float(np.nanmedian(dem[valid]))
    z = np.where(valid, dem.astype(np.float64), fill)
    local_min = minimum_filter(z, size=21)
    depression = valid & ((z - local_min) <= 0.75)
    zm = uniform_filter(z, size=15)
    zm2 = uniform_filter(z * z, size=15)
    z_std = np.sqrt(np.maximum(zm2 - zm * zm, 0.0))
    return depression & (z_std <= 0.45)


def _footprint_elevation_low_mask(
    dem: np.ndarray,
    footprint: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Pixels at or below a footprint-wide elevation percentile."""
    dem_arr = np.asarray(dem, dtype=np.float64)
    valid = footprint & np.isfinite(dem_arr)
    out = np.zeros(footprint.shape, dtype=bool)
    if not valid.any():
        return out
    cut = float(np.percentile(dem_arr[valid], percentile))
    return valid & (dem_arr <= cut)


def _blob_uniform_bright_water(med_rgb: np.ndarray, median_brightness: float) -> bool:
    """Bright, uniform tan/gray playa or sunlit muddy lake in Esri imagery."""
    r, g, b = (int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2]))
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if median_brightness < 118.0 or v < 0.52:
        return False
    if s >= POND_MUDDY_SAT_MAX:
        return False
    # Neutral/cool bright surface — not reddish tilled field or green pasture.
    if r > g + 28 and r > b + 18:
        return False
    if g > r + 35:
        return False
    return abs(r - g) <= 32 and abs(g - b) <= 32


def _blob_ring_depression_ok(
    comp: np.ndarray,
    dem: np.ndarray,
    footprint: np.ndarray,
    *,
    min_drop_m: float,
) -> bool:
    """Blob median elevation must sit below a thin outer ring."""
    from scipy.ndimage import binary_dilation

    if min_drop_m <= 0:
        return True
    dilated = binary_dilation(comp, iterations=2)
    ring = dilated & ~comp & footprint
    if not ring.any():
        return True
    dem_arr = np.asarray(dem, dtype=np.float64)
    blob_vals = dem_arr[comp & np.isfinite(dem_arr)]
    ring_vals = dem_arr[ring & np.isfinite(dem_arr)]
    if blob_vals.size == 0 or ring_vals.size == 0:
        return True
    return float(np.median(ring_vals) - np.median(blob_vals)) >= float(min_drop_m)


def _apply_pond_coverage_cap(
    accepted: list[tuple[np.ndarray, float]],
    *,
    max_pixels: int,
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if not accepted or max_pixels <= 0:
        return out
    accepted.sort(key=lambda item: item[1], reverse=True)
    used = 0
    for comp, _score in accepted:
        sz = int(comp.sum())
        if used + sz > max_pixels:
            continue
        out |= comp
        used += sz
    return out


def detect_pond_mask(
    texture_rgba: np.ndarray,
    dem: np.ndarray | None = None,
    *,
    sensitivity: str | None = "conservative",
) -> np.ndarray:
    """
    Pond / open-water pixels inside the footprint mask.

    Tier A seeds (dark / teal imagery) are accepted with standard checks. Tier B
    (muddy imagery plus DEM depression) requires footprint-low elevation and, in
    conservative modes, a perimeter ring depression test. A sensitivity preset
    caps total pond coverage to limit pasture false positives.
    """
    params = _pond_sensitivity_params(sensitivity)
    elev_pct = float(params["elev_percentile"])
    elev_margin_m = float(params["elev_margin_m"])
    max_coverage_frac = float(params["max_coverage_frac"])
    ring_drop_m = float(params["ring_drop_m"])
    require_ring_for_assisted = bool(params["require_ring_for_assisted"])

    arr = np.asarray(texture_rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("texture_rgba must be H×W×3 or H×W×4")
    rgb = arr[:, :, :3]
    if arr.shape[2] >= 4:
        footprint = arr[:, :, 3] >= MASK_ALPHA_MIN
    else:
        footprint = np.ones(rgb.shape[:2], dtype=bool)

    bright = rgb.max(axis=2).astype(np.float64)
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    max_blob = _max_pond_blob_pixels(footprint)

    if footprint.any():
        p5 = float(np.percentile(bright[footprint], 5))
        dark_cutoff = min(p5 + 3.0, 45.0)
        dark_seed = footprint & (bright <= dark_cutoff)
    else:
        dark_seed = np.zeros(footprint.shape, dtype=bool)
    open_seed = footprint & (b > r + 3) & (g > r + 2) & (bright >= 20) & (bright <= 100)
    high_conf_seed = dark_seed | open_seed
    muddy = _muddy_water_seed(rgb, footprint)
    seed = high_conf_seed.copy()
    foot_elev_low: np.ndarray | None = None
    if dem is not None:
        dem_arr = np.asarray(dem, dtype=np.float64)
        dem_dep = _dem_depression_seed(dem_arr, footprint)
        seed |= dem_dep & muddy
        # Seed bright uniform lows even when local 21×21 depression is shallow.
        seed_pct = min(95.0, elev_pct + 10.0)
        foot_elev_low = _footprint_elevation_low_mask(dem_arr, footprint, seed_pct)
        seed |= muddy & foot_elev_low

    low_elev_cut: float | None = None
    if dem is not None and footprint.any():
        dem_valid = np.asarray(dem, dtype=np.float64)
        elev = dem_valid[footprint & np.isfinite(dem_valid)]
        if elev.size > 0:
            low_elev_cut = float(np.percentile(elev, elev_pct))

    foot_px = int(np.count_nonzero(footprint))
    max_pond_px = int(max(1, round(foot_px * max_coverage_frac))) if foot_px > 0 else 0
    accepted: list[tuple[np.ndarray, float]] = []
    labeled, n = ndimage_label(seed)
    for i in range(1, int(n) + 1):
        comp = labeled == i
        sz = int(comp.sum())
        if sz < POND_MIN_BLOB_PIXELS or sz > max_blob:
            continue
        ys, xs = np.where(comp)
        bh = int(ys.max() - ys.min() + 1)
        bw = int(xs.max() - xs.min() + 1)
        min_fill = POND_LARGE_MIN_FILL if sz >= POND_LARGE_BLOB_PIXELS else POND_MIN_FILL
        if sz / max(bh * bw, 1) < min_fill:
            continue
        if max(bh, bw) / max(min(bh, bw), 1) > POND_MAX_ASPECT:
            continue
        med = np.median(rgb[comp], axis=0)
        mb = float(np.median(bright[comp]))
        med_elev: float | None = None
        if low_elev_cut is not None and dem is not None:
            med_elev = float(np.median(np.asarray(dem, dtype=np.float64)[comp]))
        elev_low_blob = med_elev is not None and low_elev_cut is not None and med_elev <= low_elev_cut
        if (
            foot_px > 0
            and sz > foot_px * 0.08
            and mb > 118.0
            and not elev_low_blob
        ):
            continue

        high_conf = bool(np.any(high_conf_seed & comp))
        assisted = not high_conf

        if med_elev is not None and low_elev_cut is not None:
            if med_elev > low_elev_cut + elev_margin_m:
                continue
            if assisted and med_elev > low_elev_cut + elev_margin_m * 0.35:
                continue

        ring_ok = True
        if dem is not None:
            ring_min_drop = ring_drop_m if (assisted or require_ring_for_assisted) else 0.0
            if elev_low_blob and sz >= POND_LARGE_BLOB_PIXELS:
                ring_min_drop = min(ring_min_drop, 0.2)
            elif elev_low_blob:
                ring_min_drop = min(ring_min_drop, 0.3)
            ring_ok = _blob_ring_depression_ok(
                comp,
                np.asarray(dem, dtype=np.float64),
                footprint,
                min_drop_m=ring_min_drop,
            )
            if assisted and require_ring_for_assisted and not ring_ok and not elev_low_blob:
                continue

        water_ok = _blob_looks_like_water(
            med, mb, strict_muddy=assisted and not elev_low_blob
        )
        if not water_ok and assisted and elev_low_blob:
            water_ok = _blob_uniform_bright_water(med, mb)
        if not water_ok:
            continue

        score = 1.0
        if high_conf:
            score += 3.0
        if elev_low_blob:
            score += 2.0
        if sz >= POND_LARGE_BLOB_PIXELS:
            score += 1.5
        if med_elev is not None and low_elev_cut is not None:
            score += max(0.0, (low_elev_cut + elev_margin_m - med_elev) / max(elev_margin_m, 0.1))
        if ring_ok:
            score += 0.75
        accepted.append((comp, score))

    out = _apply_pond_coverage_cap(
        accepted, max_pixels=max_pond_px, shape=footprint.shape
    )
    return out & footprint


def _land_palette_distinct_from_base(
    base_rgb: np.ndarray,
    land_pixels: np.ndarray,
    n_land: int,
    *,
    min_sep_sq: float = LAND_ACCENT_MIN_SEP_SQ,
) -> np.ndarray:
    """
    Pick ``n_land`` accent swatches from land pixels via k-means++ in Lab space.

    Pixels that are already base-like in Lab are excluded so homogeneous grass
    still yields distinct filament slots for paths, trees, etc.
    """
    n = int(max(0, n_land))
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    px = np.asarray(land_pixels, dtype=np.uint8).reshape(-1, 3)
    if px.size == 0:
        return _mediancut_palette_rgb(land_pixels, n)
    base_lab = _rgb_uint8_to_lab(base_rgb.reshape(1, 3))[0]
    pool_lab = _rgb_uint8_to_lab(px)
    d_base = np.sum((pool_lab - base_lab) ** 2, axis=1)
    accent_lab = pool_lab[d_base >= float(min_sep_sq)]
    if len(accent_lab) < max(LAND_ACCENT_MIN_PIXELS, n * 8):
        accent_lab = pool_lab
    if len(accent_lab) < n:
        return _mediancut_palette_rgb(land_pixels, n)
    try:
        centroids, _ = kmeans2(accent_lab, n, minit="++", iter=20)
    except (ValueError, FloatingPointError):
        return _mediancut_palette_rgb(land_pixels, n)
    if not np.all(np.isfinite(centroids)):
        return _mediancut_palette_rgb(land_pixels, n)
    return _lab_to_rgb_uint8(centroids)


def _mediancut_palette_rgb(pixels: np.ndarray, n_colors: int) -> np.ndarray:
    """Cluster flattened RGB pixels into ``n_colors`` swatches via PIL median-cut."""
    n = int(max(1, n_colors))
    px = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    if px.size == 0:
        raise ValueError("no pixels to quantize")
    n_px = len(px)
    side = int(max(4, np.ceil(np.sqrt(n_px))))
    flat_count = side * side
    tiled = np.tile(px, (1 + flat_count // max(1, n_px), 1))[:flat_count]
    img_arr = tiled.reshape(side, side, 3)
    qimg = Image.fromarray(img_arr, mode="RGB").quantize(
        colors=n, method=Image.Quantize.MEDIANCUT
    )
    palette_img = qimg.getpalette()
    if palette_img is None:
        raise ValueError("quantize produced no palette")
    got = len(palette_img) // 3
    if got <= 0:
        raise ValueError("quantize produced no palette")
    base = np.array(palette_img[: got * 3], dtype=np.uint8).reshape(got, 3)
    if got >= n:
        base = base[:n]
    else:
        # Uniform tiles can yield fewer swatches than requested; pad by repeating last.
        while base.shape[0] < n:
            base = np.vstack([base, base[-1:]])
        base = base[:n]
    return _refine_palette_lab_means(px, base)


def _predominant_rgb(pixels: np.ndarray) -> np.ndarray:
    """Dominant masked color (one median-cut swatch) for base + walls."""
    return _mediancut_palette_rgb(pixels, 1)[0]


def _assign_palette_indices(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep base at index 0; pond next; then sort other top colors by texture coverage."""
    base = [e for e in entries if e.get("role") == "base"]
    pond = [e for e in entries if e.get("role") == "pond"]
    tops = [e for e in entries if e.get("role") not in ("base", "pond")]
    if len(base) != 1:
        raise ValueError("palette must include exactly one base entry")
    tops.sort(key=lambda e: (-float(e.get("coverage_pct") or 0.0), int(e.get("index", 0))))
    ordered = base + pond + tops
    for rank, ent in enumerate(ordered, start=1):
        ent["index"] = rank - 1
        slug = _slugify(str(ent.get("name") or "color"))
        if ent.get("role") == "base":
            slug = "base"
        elif ent.get("role") == "pond":
            slug = "pond"
        ent["part_name"] = f"{rank:02d}_{slug}"
        ent["material_name"] = ent["part_name"]
    return ordered


def _palette_entries_from_rgb(
    pal_flat: np.ndarray,
    pixels: np.ndarray,
    *,
    pond_slot: int | None = None,
    base_slot: int | None = None,
) -> list[dict[str, Any]]:
    n = int(pal_flat.shape[0])
    px_lab = _rgb_uint8_to_lab(pixels)
    pal_lab = _rgb_uint8_to_lab(pal_flat)
    d2 = np.sum((px_lab[:, None, :] - pal_lab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)
    total = float(len(labels))
    counts = np.bincount(labels, minlength=n).astype(np.float64)

    entries: list[dict[str, Any]] = []
    for i in range(n):
        rgb_t = (int(pal_flat[i, 0]), int(pal_flat[i, 1]), int(pal_flat[i, 2]))
        if base_slot is not None and i == base_slot:
            role = "base"
            name = "Base"
        elif pond_slot is not None and i == pond_slot:
            role = "pond"
            name = "Pond"
        else:
            role = "land"
            name = _rgb_to_filament_name(rgb_t)
        entries.append(
            {
                "index": i,
                "rgb": list(rgb_t),
                "hex": _hex_rgb(rgb_t),
                "name": name,
                "role": role,
                "part_name": "",
                "coverage_pct": round(100.0 * float(counts[i]) / total, 2) if total > 0 else 0.0,
            }
        )

    if base_slot is not None:
        return _assign_palette_indices(entries)
    entries.sort(key=lambda e: (-float(e["coverage_pct"]), int(e["index"])))
    for rank, ent in enumerate(entries, start=1):
        ent["index"] = rank - 1
        slug = _slugify(str(ent["name"]))
        if ent.get("role") == "pond":
            slug = "pond"
        ent["part_name"] = f"{rank:02d}_{slug}"
        ent["material_name"] = ent["part_name"]
    return entries


def recommend_ams_palette(
    texture_rgba: np.ndarray,
    n_colors: int = 4,
    *,
    reserve_base: bool = True,
    dem: np.ndarray | None = None,
    pond_sensitivity: str | None = "conservative",
) -> list[dict[str, Any]]:
    """
    Cluster masked satellite RGB into ``n_colors`` filament recommendations.

    When ``reserve_base`` is true (default for print AMS), index 0 is the predominant
    masked color. It labels the solid base + side walls and any top surface that matches
    that color; remaining slots cover other top-surface colors only.
    """
    n = clamp_ams_n_colors(n_colors)
    arr = np.asarray(texture_rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("texture_rgba must be H×W×3 or H×W×4")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    rgb = arr[:, :, :3]
    if arr.shape[2] >= 4:
        mask = arr[:, :, 3] >= MASK_ALPHA_MIN
    else:
        mask = np.ones((h, w), dtype=bool)

    pixels = rgb[mask].reshape(-1, 3)
    if pixels.size == 0:
        pixels = rgb.reshape(-1, 3)

    if not reserve_base:
        pond_mask = detect_pond_mask(arr, dem=dem, sensitivity=pond_sensitivity)
        pond_count = int(np.count_nonzero(pond_mask & mask))
        reserve_pond = pond_count >= POND_MIN_PIXELS and n >= 2
        if reserve_pond:
            land_mask = mask & ~pond_mask
            land_pixels = rgb[land_mask].reshape(-1, 3)
            if land_pixels.size == 0:
                land_pixels = pixels
            pond_imagery = np.median(rgb[pond_mask], axis=0).round().astype(np.uint8)
            land_palette = _mediancut_palette_rgb(land_pixels, n - 1)
            pal_flat = np.vstack([land_palette, pond_imagery.reshape(1, 3)])
            entries = _palette_entries_from_rgb(pal_flat, pixels, pond_slot=int(n - 1))
            for ent in entries:
                if ent.get("role") == "pond":
                    ent["pond_pixel_count"] = pond_count
                    ent["imagery_rgb"] = list(map(int, pond_imagery))
                    ent["filament_rgb"] = list(POND_FILAMENT_RGB)
            return entries
        pal_flat = _mediancut_palette_rgb(pixels, n)
        return _palette_entries_from_rgb(pal_flat, pixels)

    top_n = n - 1
    base_rgb = _predominant_rgb(pixels)
    pond_mask = detect_pond_mask(arr, dem=dem, sensitivity=pond_sensitivity)
    pond_count = int(np.count_nonzero(pond_mask & mask))
    reserve_pond = pond_count >= POND_MIN_PIXELS and top_n >= 2

    if reserve_pond:
        land_mask = mask & ~pond_mask
        land_pixels = rgb[land_mask].reshape(-1, 3)
        if land_pixels.size == 0:
            land_pixels = pixels
        pond_imagery = np.median(rgb[pond_mask], axis=0).round().astype(np.uint8)
        land_palette = _land_palette_distinct_from_base(base_rgb, land_pixels, top_n - 1)
        top_pal = np.vstack([land_palette, pond_imagery.reshape(1, 3)])
        pond_slot = int(top_n)
    else:
        top_pal = _land_palette_distinct_from_base(base_rgb, pixels, top_n)
        pond_slot = None

    pal_flat = np.vstack([base_rgb.reshape(1, 3), top_pal])
    entries = _palette_entries_from_rgb(
        pal_flat, pixels, pond_slot=pond_slot, base_slot=0
    )
    if reserve_pond:
        for ent in entries:
            if ent.get("role") == "pond":
                ent["pond_pixel_count"] = pond_count
                ent["imagery_rgb"] = list(map(int, pond_imagery))
                ent["filament_rgb"] = list(POND_FILAMENT_RGB)
                foot = float(np.count_nonzero(mask))
                if foot > 0:
                    ent["coverage_pct"] = round(100.0 * float(pond_count) / foot, 2)
    return entries


def quantize_texture_index_image(
    texture_rgba: np.ndarray,
    palette_rgb: np.ndarray,
    *,
    merge_index: int | None = None,
    merge_slop: float = MERGE_INDEX_SLOP_LAB_SQ,
    force_index_mask: tuple[int, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Return H×W uint8 label image (0..K-1) by nearest palette color in CIELAB.

    When ``merge_index`` is set, labels whose nearest swatch is within ``merge_slop``
    (squared Lab ΔE) of that palette entry are remapped to ``merge_index``. This
    folds duplicate base-like top colors into the base slot for fewer filament changes.
    """
    arr = np.asarray(texture_rgba, dtype=np.uint8)
    pal = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)
    if pal.shape[0] == 0:
        raise ValueError("empty palette")
    rgb = arr[:, :, :3]
    h, w = rgb.shape[:2]
    flat_lab = _rgb_uint8_to_lab(rgb.reshape(-1, 3))
    pal_lab = _rgb_uint8_to_lab(pal)
    d2 = np.sum((flat_lab[:, None, :] - pal_lab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1).astype(np.intp)
    if merge_index is not None and 0 <= int(merge_index) < pal.shape[0]:
        base_lab = pal_lab[int(merge_index)]
        chosen_lab = pal_lab[labels]
        dup = np.sum((chosen_lab - base_lab) ** 2, axis=1) <= float(merge_slop)
        labels = np.where(dup, int(merge_index), labels)
    if force_index_mask is not None:
        force_idx, force_mask = force_index_mask
        fm = np.asarray(force_mask, dtype=bool).reshape(-1)
        if fm.shape[0] == labels.shape[0]:
            labels = labels.copy()
            labels[fm] = int(force_idx)
    out = labels.astype(np.uint8).reshape(h, w)
    if arr.shape[2] >= 4:
        out[arr[:, :, 3] < MASK_ALPHA_MIN] = 0
    return out


def render_quantized_preview(
    index_image: np.ndarray,
    palette_rgb: np.ndarray,
    *,
    footprint: np.ndarray | None = None,
) -> Image.Image:
    """RGBA preview of the quantized color map.

    ``footprint`` is a boolean H×W mask (``True`` inside the KML boundary). Masked-out
    pixels are transparent even when their palette index is 0 (base color).
    """
    pal = np.asarray(palette_rgb, dtype=np.uint8).reshape(-1, 3)
    idx = np.asarray(index_image, dtype=np.intp)
    rgb = pal[np.clip(idx, 0, len(pal) - 1)]
    if footprint is not None:
        inside = np.asarray(footprint, dtype=bool)
        if inside.shape != idx.shape:
            raise ValueError("footprint shape must match index_image")
        alpha = np.where(inside, 255, 0).astype(np.uint8)
    else:
        alpha = np.full(idx.shape, 255, dtype=np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def _sample_index_at_uv(index_image: np.ndarray, u: float, v: float) -> int:
    h, w = index_image.shape
    if w <= 0 or h <= 0:
        return 0
    # UV convention matches build_mesh: v=0 at north (top row).
    x = int(np.clip(u * w, 0, w - 1))
    y = int(np.clip((1.0 - v) * h, 0, h - 1))
    return int(index_image[y, x])


def _interpolate_uv_on_surf(surf_mm: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """UV coordinates for 3D points via closest-point barycentric interp on ``surf_mm``."""
    vis = surf_mm.visual
    if vis is None or not vis.defined or getattr(vis, "kind", None) != "texture":
        raise ValueError("surf_mm needs texture UVs")
    uv_arr = np.asarray(vis.uv, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    closest, _dist, tri_ids = trimesh.proximity.closest_point(surf_mm, pts)
    tri_ids = np.asarray(tri_ids, dtype=np.intp)
    faces = np.asarray(surf_mm.faces, dtype=np.int64)[tri_ids]
    tris = np.asarray(surf_mm.triangles, dtype=np.float64)[tri_ids]
    bary = trimesh.triangles.points_to_barycentric(tris, closest)
    uv_corners = uv_arr[faces]
    return np.einsum("ncf,nc->nf", uv_corners, bary)


def _top_face_proximity_tol_mm(
    *,
    tol_mm: float | None = None,
    voxel_size_mm: float | None = None,
) -> float:
    """Distance threshold for matching print-solid faces to the open terrain surface."""
    if tol_mm is not None:
        return float(tol_mm)
    vs = float(voxel_size_mm) if voxel_size_mm is not None and float(voxel_size_mm) > 0 else 0.0
    if vs > 0:
        return max(TOP_FACE_PROXIMITY_MM, vs * 1.5)
    return float(TOP_FACE_PROXIMITY_MM)


def top_face_mask(
    print_solid: trimesh.Trimesh,
    surf_mm: trimesh.Trimesh,
    *,
    tol_mm: float | None = None,
    voxel_size_mm: float | None = None,
) -> np.ndarray:
    """
    True for triangles on the open terrain top (not base cap or side walls).

    Uses upward-facing face normals (fast). Proximity to ``surf_mm`` is only used
    when normals yield no top faces (e.g. badly oriented voxel repair).
    """
    n_faces = len(print_solid.faces)
    if n_faces == 0 or print_solid.is_empty:
        return np.zeros(n_faces, dtype=bool)
    fn = np.asarray(print_solid.face_normals, dtype=np.float64)
    normal_top = fn[:, 2] > TOP_FACE_NORMAL_Z_MIN
    if int(normal_top.sum()) > 0:
        return normal_top
    if surf_mm.is_empty or len(surf_mm.faces) == 0:
        return normal_top
    centroids = np.asarray(print_solid.triangles_center, dtype=np.float64)
    eff_tol = _top_face_proximity_tol_mm(tol_mm=tol_mm, voxel_size_mm=voxel_size_mm)
    _closest, distances, tri_ids = trimesh.proximity.closest_point(surf_mm, centroids)
    surf_normals = np.asarray(surf_mm.face_normals, dtype=np.float64)[
        np.asarray(tri_ids, dtype=np.intp)
    ]
    return (np.asarray(distances, dtype=np.float64) <= eff_tol) & (
        surf_normals[:, 2] > TOP_FACE_NORMAL_Z_MIN
    )


def label_faces_by_palette(
    mesh_uv: trimesh.Trimesh,
    index_image: np.ndarray,
    *,
    is_top: np.ndarray | None = None,
    base_index: int = 0,
    local_to_global: dict[int, int] | None = None,
    surf_mm: trimesh.Trimesh | None = None,
) -> np.ndarray:
    """Assign palette indices; non-top faces use ``base_index`` when ``is_top`` is set."""
    if mesh_uv.is_empty or len(mesh_uv.faces) == 0:
        return np.zeros(0, dtype=np.intp)
    faces = np.asarray(mesh_uv.faces, dtype=np.int64)
    n_faces = len(faces)
    h, w = index_image.shape

    if is_top is not None:
        labels = np.full(n_faces, int(base_index), dtype=np.intp)
        label_faces = np.flatnonzero(is_top)
        if label_faces.size == 0:
            return labels
    else:
        labels = None
        label_faces = np.arange(n_faces, dtype=np.intp)

    subset = faces[label_faces]
    vis = mesh_uv.visual
    if (
        vis is not None
        and vis.defined
        and getattr(vis, "kind", None) == "texture"
        and getattr(vis, "uv", None) is not None
    ):
        uv = np.asarray(vis.uv, dtype=np.float64)
        uv_corners = uv[subset]
        sample_uv = np.einsum("fcx,sc->fsx", uv_corners, _FACE_LABEL_BARY_WEIGHTS)
    else:
        out = np.full(n_faces, int(base_index), dtype=np.intp)
        return out
    su = sample_uv[..., 0]
    sv = sample_uv[..., 1]
    x = np.clip((su * w).astype(np.intp), 0, max(w - 1, 0))
    y = np.clip(((1.0 - sv) * h).astype(np.intp), 0, max(h - 1, 0))
    sample_labels = index_image[y, x].astype(np.intp)
    face_labels = scipy_mode(sample_labels, axis=1, keepdims=False).mode.astype(np.intp)
    if local_to_global is not None:
        max_local = max(int(k) for k in local_to_global)
        table = np.arange(max_local + 1, dtype=np.intp)
        for local, global_idx in local_to_global.items():
            table[int(local)] = int(global_idx)
        face_labels = table[face_labels]
    if labels is not None:
        labels[label_faces] = face_labels
        return labels
    return face_labels


def split_solid_by_labels(
    solid: trimesh.Trimesh,
    face_labels: np.ndarray,
    palette: list[dict[str, Any]],
    *,
    process_mesh=None,
) -> list[tuple[dict[str, Any], trimesh.Trimesh]]:
    """
    Split ``solid`` into one mesh per palette index with non-zero faces.

    ``process_mesh`` is optional (e.g. :func:`terrain_app.mesh.solid_process_for_export`).
    """
    if len(face_labels) != len(solid.faces):
        raise ValueError("face_labels length must match solid face count")
    index_to_meta = {int(p["index"]): p for p in palette}
    parts: list[tuple[dict[str, Any], trimesh.Trimesh]] = []
    for idx in sorted(index_to_meta.keys()):
        meta = index_to_meta[idx]
        face_idx = np.where(face_labels == idx)[0]
        if face_idx.size == 0:
            continue
        sub = solid.submesh([face_idx], append=True)
        if sub.is_empty or len(sub.faces) == 0:
            continue
        if process_mesh is not None:
            sub = process_mesh(sub)
        parts.append((meta, sub))
    return parts


def build_ams_labels(
    print_mesh: trimesh.Trimesh,
    surf_mm: trimesh.Trimesh,
    texture_rgba: np.ndarray,
    *,
    print_solid_with_satellite_uv,
    n_colors: int = 4,
    mesh_uv: trimesh.Trimesh | None = None,
    dem: np.ndarray | None = None,
    pond_sensitivity: str | None = "conservative",
    voxel_size_mm: float | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Quantize imagery and label print-solid faces without splitting geometry.

    Returns (palette, index_image, face_labels).
    """
    n_colors = clamp_ams_n_colors(n_colors)
    arr = np.asarray(texture_rgba, dtype=np.uint8)
    if arr.shape[2] >= 4:
        footprint = arr[:, :, 3] >= MASK_ALPHA_MIN
    else:
        footprint = np.ones(arr.shape[:2], dtype=bool)
    if on_progress:
        on_progress("Detecting water and building color palette…")
    pond_mask = detect_pond_mask(arr, dem=dem, sensitivity=pond_sensitivity)
    palette = recommend_ams_palette(
        texture_rgba, n_colors=n_colors, reserve_base=True, dem=dem, pond_sensitivity=pond_sensitivity
    )
    base_index = next(int(p["index"]) for p in palette if p.get("role") == "base")
    palette_rgb = np.array([p["rgb"] for p in palette], dtype=np.uint8)
    force_pond: tuple[int, np.ndarray] | None = None
    pond_entry = next((p for p in palette if p.get("role") == "pond"), None)
    if pond_entry is not None and pond_mask.any():
        force_pond = (int(pond_entry["index"]), pond_mask)
    index_image = quantize_texture_index_image(
        texture_rgba,
        palette_rgb,
        force_index_mask=force_pond,
        merge_index=base_index,
        merge_slop=MERGE_INDEX_SLOP_LAB_SQ,
    )
    if mesh_uv is None:
        mesh_uv = print_solid_with_satellite_uv(print_mesh, surf_mm)
    if on_progress:
        on_progress("Labeling mesh faces by color region…")
    is_top = top_face_mask(print_mesh, surf_mm, voxel_size_mm=voxel_size_mm)
    face_labels = label_faces_by_palette(
        mesh_uv,
        index_image,
        is_top=is_top,
        base_index=base_index,
    )
    n_slots = max((int(p["index"]) for p in palette), default=-1) + 1
    tri_counts = np.bincount(
        np.asarray(face_labels, dtype=np.intp), minlength=max(n_slots, 1)
    )
    for ent in palette:
        ent["triangle_count"] = int(tri_counts[int(ent["index"])])
    return palette, index_image, face_labels


def build_ams_parts(
    print_mesh: trimesh.Trimesh,
    surf_mm: trimesh.Trimesh,
    texture_rgba: np.ndarray,
    *,
    print_solid_with_satellite_uv,
    process_mesh=None,
    n_colors: int = 4,
    mesh_uv: trimesh.Trimesh | None = None,
    voxel_size_mm: float | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], trimesh.Trimesh]], np.ndarray, np.ndarray]:
    """
    Full AMS labeling pipeline.

    Returns (palette, parts, index_image, face_labels).
    """
    palette, index_image, face_labels = build_ams_labels(
        print_mesh,
        surf_mm,
        texture_rgba,
        print_solid_with_satellite_uv=print_solid_with_satellite_uv,
        n_colors=n_colors,
        mesh_uv=mesh_uv,
        voxel_size_mm=voxel_size_mm,
    )
    parts = split_solid_by_labels(
        print_mesh, face_labels, palette, process_mesh=process_mesh
    )
    return palette, parts, index_image, face_labels
