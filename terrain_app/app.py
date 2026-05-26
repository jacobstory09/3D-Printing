"""Flask application."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from terrain_app.pipeline import export_download_filename, load_meta, process_kml


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    instance = Path(app.instance_path)
    cache_root = instance / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/process")
    def api_process():
        if "kml" not in request.files:
            return jsonify({"error": "Missing kml file"}), 400
        f = request.files["kml"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        data = f.read()
        if not data:
            return jsonify({"error": "Empty file"}), 400
        imagery = (request.form.get("imagery") or "osm").lower()
        if imagery not in ("osm", "oam", "esri"):
            imagery = "osm"
        try:
            grid_size = int(request.form.get("grid_size") or 1024)
        except ValueError:
            grid_size = 1024
        try:
            buffer_m = float(request.form.get("buffer_m") or 50)
        except ValueError:
            buffer_m = 50.0
        try:
            vz = float(request.form.get("vertical_exaggeration") or 3)
        except ValueError:
            vz = 1.0
        vz = max(0.01, vz)
        try:
            print_max = float(request.form.get("print_max_size_mm") or 200)
        except ValueError:
            print_max = 200.0
        try:
            print_base = float(request.form.get("print_base_extrusion_mm") or 1)
        except ValueError:
            print_base = 1.0
        vox = request.form.get("print_voxel_size_mm")
        try:
            if vox is None or str(vox).strip() == "":
                print_voxel = None
            else:
                pv = float(vox)
                print_voxel = pv if pv > 0 else None
        except ValueError:
            print_voxel = None
        try:
            split_nx = int(request.form.get("print_split_nx") or 1)
        except ValueError:
            split_nx = 1
        try:
            split_nz = int(request.form.get("print_split_nz") or 1)
        except ValueError:
            split_nz = 1
        bsm_raw = request.form.get("boundary_smooth_m")
        boundary_smooth: float | None = None
        if bsm_raw is not None and str(bsm_raw).strip().lower() not in ("", "auto"):
            try:
                boundary_smooth = float(bsm_raw)
            except ValueError:
                boundary_smooth = None
        try:
            job_id = process_kml(
                data,
                cache_root,
                kml_filename=f.filename,
                imagery=imagery,  # type: ignore[arg-type]
                grid_size=grid_size,
                buffer_m=buffer_m,
                vertical_exaggeration=vz,
                print_max_size_mm=print_max,
                print_base_extrusion_mm=print_base,
                print_voxel_size_mm=print_voxel,
                print_split_nx=split_nx,
                print_split_nz=split_nz,
                boundary_smooth_m=boundary_smooth,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("process failed")
            return jsonify({"error": str(e)}), 500
        meta = load_meta(cache_root, job_id)
        return jsonify({"ok": True, "meta": meta})

    @app.get("/api/result/<job_id>")
    def api_result(job_id: str):
        try:
            meta = load_meta(cache_root, job_id)
        except FileNotFoundError:
            return jsonify({"error": "Unknown job"}), 404
        dem_path = cache_root / job_id / "heights_display.npy"
        dem = np.load(dem_path, mmap_mode="r")
        heights_b64 = None
        try:
            raw = np.asarray(dem, dtype=np.float32).tobytes()
            heights_b64 = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass
        out = {
            "meta": meta,
            "textures": {
                "rgba": f"/api/result/{job_id}/texture.png",
            },
            "heights": {
                "url": f"/api/result/{job_id}/heights.bin",
                "width": meta["grid_width"],
                "height": meta["grid_height"],
                "encoding": "float32-le",
            },
        }
        if heights_b64 and len(heights_b64) < 6_000_000:
            out["heights_inline_base64"] = heights_b64
        return jsonify(out)

    @app.get("/api/result/<job_id>/texture.png")
    def api_texture(job_id: str):
        p = cache_root / job_id / "texture.png"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(p, mimetype="image/png")

    @app.get("/api/result/<job_id>/heights.bin")
    def api_heights(job_id: str):
        p = cache_root / job_id / "heights_display.npy"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        arr = np.load(p)
        return Response(arr.astype(np.float32).tobytes(), mimetype="application/octet-stream")

    @app.get("/api/result/<job_id>/quad_mask.bin")
    def api_quad_mask(job_id: str):
        """Quad inclusion (h-1)×(w-1) uint8 row-major, same rules as print mesh / build_mesh."""
        p = cache_root / job_id / "quad_mask.bin"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(p, mimetype="application/octet-stream")

    @app.get("/api/result/<job_id>/export.glb")
    def api_glb(job_id: str):
        p = cache_root / job_id / "terrain.glb"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        meta = load_meta(cache_root, job_id)
        return send_file(
            p,
            mimetype="model/gltf-binary",
            as_attachment=True,
            download_name=export_download_filename(meta, ".glb"),
        )

    @app.get("/api/result/<job_id>/export.zip")
    def api_zip(job_id: str):
        p = cache_root / job_id / "terrain_obj.zip"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(
            p,
            mimetype="application/zip",
            as_attachment=True,
            download_name=export_download_filename(load_meta(cache_root, job_id), "_obj.zip"),
        )

    @app.get("/api/result/<job_id>/export.stl")
    def api_stl(job_id: str):
        p = cache_root / job_id / "terrain_print.stl"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(
            p,
            mimetype="model/stl",
            as_attachment=True,
            download_name=export_download_filename(load_meta(cache_root, job_id), "_print.stl"),
        )

    @app.get("/api/result/<job_id>/export_print.glb")
    def api_print_glb(job_id: str):
        """Same watertight print solid as STL/3MF, as glTF binary (no satellite texture)."""
        p = cache_root / job_id / "terrain_print.glb"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(
            p,
            mimetype="model/gltf-binary",
            as_attachment=True,
            download_name=export_download_filename(load_meta(cache_root, job_id), "_print.glb"),
        )

    @app.get("/api/result/<job_id>/export.3mf")
    def api_3mf(job_id: str):
        p = cache_root / job_id / "terrain_print.3mf"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(
            p,
            mimetype="model/3mf",
            as_attachment=True,
            download_name=export_download_filename(load_meta(cache_root, job_id), "_print.3mf"),
        )

    @app.get("/api/result/<job_id>/export_print_pieces.zip")
    def api_print_pieces_zip(job_id: str):
        p = cache_root / job_id / "terrain_print_pieces.zip"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(
            p,
            mimetype="application/zip",
            as_attachment=True,
            download_name=export_download_filename(
                load_meta(cache_root, job_id), "_print_pieces.zip"
            ),
        )

    return app
