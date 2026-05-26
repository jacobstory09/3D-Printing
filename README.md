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

Processed jobs are cached under `instance/cache/` (Flask’s auto-detected instance folder at the project root).

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/process` | `multipart/form-data`: `kml` file; optional `imagery=osm\|oam\|esri`, `grid_size` (64–2048), `buffer_m` (meters), `vertical_exaggeration` |
| GET | `/api/result/<id>` | JSON metadata + viewer payload |
| GET | `/api/result/<id>/texture.png` | RGB texture |
| GET | `/api/result/<id>/export.glb` | GLB with UV-mapped texture |
| GET | `/api/result/<id>/export.zip` | OBJ + MTL + texture PNG |
| GET | `/api/result/<id>/export.3mf` | Watertight **print** solid in 3MF; map texture embedded as `3D/Textures/texture.png` + UVs |
| GET | `/api/result/<id>/export.stl` | Same print solid as STL |
| GET | `/api/result/<id>/export_print.glb` | Same print solid as GLB (no satellite texture) |

### Export vs preview

Exports are built from the **same** `dem.npy`, transform, texture, `vertical_exaggeration`, and `prepare_z` logic as the viewer—there is no separate sample mesh. Vertices use **Z-up**: **X = UTM easting**, **Y = UTM northing**, **Z = display elevation** (min removed × exaggeration), in metres for terrain GLB/OBJ and millimetres for print files.

**Blender (4.x):**

- Prefer **`terrain.glb`**: File → Import → glTF 2.0. Use **Material Preview** or **Rendered** viewport shading; **Solid** shows only gray geometry (no satellite), which matches a shaded DEM, not “missing” data.
- **`terrain_obj.zip`**: Extract **all** files to one folder (`terrain.obj`, `material.mtl`, `material_0.png`). Import the `.obj` from that folder so `map_Kd material_0.png` resolves. The ZIP is a normal archive (OBJ + MTL + PNG); it is generated on each build from the same mesh as the GLB.
- Vertex coordinates are **large** (real UTM metres, e.g. hundreds of thousands). Avoid ad‑hoc **0.001 scale** or **90° rotations** unless you intend to change orientation. For a 1:1 match with the app, keep rotation **0°** and scale **1** on import; the ground plane is **XY** and **Z** is up.

**KML clipping:** The web viewer triangulates the **same quads** as **exports (GLB + OBJ)**: only faces whose **UTM ground quad** intersects the KML polygon (Shapely). Texture **RGBA** still uses the raster mask for alpha. Large jobs (e.g. 1000×700+) may take longer while culling faces. Use **Alpha Clip** in Blender if needed for texture edges.

**Print / slicing (`terrain_print.*`):** `terrain_print.3mf` is a normal 3MF ZIP: geometry in `3D/3dmodel.model`, imagery in **`3D/Textures/texture.png`**, aligned via the same UV mapping as the preview surface (nearest XY sample from the scaled terrain). In `meta.json`, **`print.print_3mf_textured`** is `true` only when an on-disk check (`print.print_3mf_texture_inspection`) confirms that PNG + texture markup are present—**not** when a website or slicer chooses to preview gray plastic. Many quoting UIs ignore diffuse textures; use Blender, a 3MF-capable viewer, or unzip the file to verify the map is inside the package.

## Notes

- Large `grid_size` and wide areas increase USGS and tile download time.
- OSM tile usage must follow [OSMF tile policy](https://operations.osmfoundation.org/policies/tiles/) (reasonable traffic, proper User-Agent).

## Known issues
- Navigating in the terrain viewer is clunky. Click and drag to rotate works well, zoom in/out is broken

**Boundary smoothing:** The **Boundary smooth** control offers **Auto** (~0.75× the DEM cell size in metres), or fixed **5 / 10 / 20** metre rounding. Mesh and print solids use a **smoothed polygon** and cut each DEM quad along that outline (not whole stair-step blocks). The texture uses a soft alpha fade at the edge. Finer **grid** resolution also reduces remaining edge faceting.
