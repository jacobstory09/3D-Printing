# Terrain viewer (Flask + 3DEP + OSM / OpenAerialMap)

Upload a KML boundary (US only), fetch **USGS 3DEP** elevation and **OpenStreetMap** or **OpenAerialMap** imagery, preview in 3D, and export **GLB** / **OBJ+MTL** for Blender.

## Working CRS

All rasters and exports use **WGS 84 / UTM** for the zone containing the polygon centroid (meters on the ground).

## Attribution

- Elevation: [USGS 3D Elevation Program (3DEP)](https://www.usgs.gov/3d-elevation-program)
- **Esri World Imagery** (optional source): Esri, Maxar, Earthstar Geographics, and the GIS User Community — see [Esri terms](https://www.esri.com/en-us/legal/terms/data).
- OSM tiles: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright)
- OpenAerialMap: imagery retains the license shown in API results (often CC-BY)

## Setup

```bash
cd "/path/to/3D Printing"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:5000

Processed jobs are cached under `instance/cache/` (Flask’s auto-detected instance folder at the project root). A background task removes old jobs automatically (defaults: **1 h** max runtime per job, **24 h** cache retention, **2 h** without progress updates for orphaned jobs). Override with environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TERRAIN_JOB_MAX_RUNTIME_SEC` | `3600` | Abort a running job after this many seconds |
| `TERRAIN_JOB_CACHE_TTL_SEC` | `86400` | Delete finished or failed jobs after this age |
| `TERRAIN_JOB_STALE_RUNNING_SEC` | `7200` | Delete stuck `running` jobs with no progress updates |
| `TERRAIN_JOB_CLEANUP_INTERVAL_SEC` | `900` | How often the server scans the cache |

Timed-out jobs keep `progress.json` (with an error) but drop large artifacts immediately; finished jobs are removed entirely once they expire.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/process` | `multipart/form-data`: `kml` file; optional `imagery=osm\|oam\|esri`, `grid_size` (64–2048, default 512), `buffer_m` (meters), `vertical_exaggeration`, export checkboxes (`export_print_stl`, `export_print_ams`, …), `ams_quality` (`high`\|`medium`\|`low`) |
| GET | `/api/result/<id>` | JSON metadata + viewer payload |
| GET | `/api/result/<id>/texture.png` | RGB texture |
| GET | `/api/result/<id>/export.glb` | GLB with UV-mapped texture |
| GET | `/api/result/<id>/export.zip` | OBJ + MTL + texture PNG |
| GET | `/api/result/<id>/export.stl` | Print solid, geometry only (for slicers) |
| GET | `/api/result/<id>/export.3mf` | Same print solid as STL (geometry only) |
| GET | `/api/result/<id>/export_print_textured.glb` | Print solid + embedded satellite texture (Blender) |
| GET | `/api/result/<id>/export_print_ams.glb` | Print solid + 4 AMS color regions (Blender) |
| GET | `/api/result/<id>/export_print_ams` | 4-color Bambu AMS OBJ/MTL ZIP (`terrain_ams.obj` + palette) |

### Export vs preview

Exports are built from the **same** `dem.npy`, transform, texture, `vertical_exaggeration`, and `prepare_z` logic as the viewer—there is no separate sample mesh. Vertices use **Z-up**: **X = UTM easting**, **Y = UTM northing**, **Z = display elevation** (min removed × exaggeration), in metres for terrain GLB/OBJ and millimetres for print files.

**Three print-base exports (same watertight solid, different color):**

| File | Color | Use for |
|------|-------|---------|
| `*_print.stl` (or `*_print.3mf`) | None — geometry only | Bambu Studio, PrusaSlicer, etc. |
| `*_print_textured.glb` | Full satellite map (UV texture) | Blender preview of printable model |
| `*_print_ams.glb` | 4 solid filament colors | Blender preview of AMS color plan |

You do **not** need both STL and a geometry-only GLB — they are the same solid. STL/3MF is for slicing; the GLB variants are for viewing color in Blender.

**Blender (4.x) — preview surface (no print base):**

- Use **`…_obj.zip`** or **`….glb`** from the preview links.
- **GLB:** File → Import → **glTF 2.0 (.glb/.gltf)**. Switch the viewport to **Material Preview** or **Rendered** (top-right sphere icon). **Solid** shading is gray only — that does not mean the texture is missing.
- **OBJ:** Extract the **whole** ZIP into one folder (`terrain.obj`, `material.mtl`, `material_0.png`), then File → Import → **Wavefront (.obj)** from that folder. If `material_0.png` is missing, the map will not load.
- KML edges use texture **alpha**; glTF exports set **Alpha Clip** (~0.08). In Shader Editor you can tune **Alpha Cutoff** on the Principled BSDF if edges look wrong.
- Keep import **scale 1** and **rotation 0°** for a 1:1 match with the app (Z-up, XY ground). Do not apply 0.001 scale unless you intend to rescale.

**Blender — print solid with base:**

- **Full satellite:** import `*_print_textured.glb` (Material Preview / Rendered).
- **4-color AMS plan:** import `*_print_ams.glb` (vertex colors; Material Preview).
- **`01_base`** is the predominant terrain color on walls, the underside, and matching top areas; accent colors cover the rest of the top surface.

**KML clipping:** The web viewer triangulates the **same quads** as **exports (GLB + OBJ)**: only faces whose **UTM ground quad** intersects the KML polygon (Shapely). Texture **RGBA** still uses the raster mask for alpha. Large jobs (e.g. 1000×700+) may take longer while culling faces. Use **Alpha Clip** in Blender if needed for texture edges.

**Print / slicing:** `*_print.stl` and `*_print.3mf` are **geometry-only** (no embedded color) so slicers open them reliably.

**Bambu AMS color (`*_print_ams_obj.zip`):** Quantizes the satellite image to **4 filament colors** — one predominant swatch (`01_base`) shared by walls, underside, and matching top terrain; three accent top colors — splits the print solid into matching material regions, and packages **`terrain_ams.obj`** + **`terrain_ams.mtl`** + **`palette.json`** in a ZIP. After processing, the web UI shows recommended filament names, hex swatches, and coverage %. In **Bambu Studio**: import `terrain_ams.obj` from the ZIP → choose **Yes** if asked to load as one object with multiple parts → assign **`01_base`** to your dominant filament and map accent top colors to other AMS slots (fewer filament changes than treating base and dominant top as separate colors). Optional test fixtures: run `mesh_mod.generate_bambu_ams_fixtures("tests/fixtures/bambu_ams")` for tiny colored 3MF/OBJ samples.

## Notes

- Large `grid_size` and wide areas increase USGS and tile download time.
- OSM tile usage must follow [OSMF tile policy](https://operations.osmfoundation.org/policies/tiles/) (reasonable traffic, proper User-Agent).

## Known issues
- Navigating in the terrain viewer: use **Fit view** / **Top** if the camera gets lost after load.

**Boundary smoothing:** The **Boundary smooth** control offers **Auto** (~0.75× the DEM cell size in metres), or fixed **5 / 10 / 20** metre rounding. Mesh and print solids use a **smoothed polygon** and cut each DEM quad along that outline (not whole stair-step blocks). The texture uses a soft alpha fade at the edge. Finer **grid** resolution also reduces remaining edge faceting.
