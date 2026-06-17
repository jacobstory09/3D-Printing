"""Flask application."""

from __future__ import annotations

import base64
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from terrain_app import ams_color
from terrain_app.export_options import normalize_ams_quality, parse_export_options
from terrain_app.job_cleanup import (
    JobTimeoutError,
    cleanup_expired_jobs,
    job_settings,
    prune_job_artifacts,
    start_cleanup_scheduler,
)
from terrain_app.pipeline import export_download_filename, load_meta, process_kml
from terrain_app.progress import load_progress, progress_reporter_with_timeout, write_progress

_PROCESS_SEMAPHORE = threading.Semaphore(1)


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
    job_cfg = job_settings()
    active_jobs: set[str] = set()
    active_jobs_lock = threading.Lock()

    def _active_job_ids() -> frozenset[str]:
        with active_jobs_lock:
            return frozenset(active_jobs)

    start_cleanup_scheduler(cache_root, job_cfg, _active_job_ids, app.logger)

    @app.get("/")
    def index():
        return render_template("index.html")

    def _parse_process_form() -> tuple[dict[str, Any], str | None]:
        """Parse multipart form; return (kwargs for process_kml, error message)."""
        if "kml" not in request.files:
            return {}, "Missing kml file"
        f = request.files["kml"]
        if not f.filename:
            return {}, "Empty filename"
        data = f.read()
        if not data:
            return {}, "Empty file"
        imagery = (request.form.get("imagery") or "osm").lower()
        if imagery not in ("osm", "oam", "esri"):
            imagery = "osm"
        try:
            grid_size = int(request.form.get("grid_size") or 512)
        except ValueError:
            grid_size = 512
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
        pond_sensitivity = ams_color.normalize_pond_sensitivity(
            request.form.get("pond_sensitivity")
        )
        try:
            ams_n_colors = ams_color.clamp_ams_n_colors(
                int(request.form.get("ams_n_colors") or 4)
            )
        except (TypeError, ValueError):
            ams_n_colors = 4
        ams_quality = normalize_ams_quality(request.form.get("ams_quality"))
        export_options = parse_export_options(request.form)
        return {
            "kml_bytes": data,
            "kml_filename": f.filename,
            "imagery": imagery,
            "grid_size": grid_size,
            "buffer_m": buffer_m,
            "vertical_exaggeration": vz,
            "print_max_size_mm": print_max,
            "print_base_extrusion_mm": print_base,
            "print_voxel_size_mm": print_voxel,
            "print_split_nx": split_nx,
            "print_split_nz": split_nz,
            "boundary_smooth_m": boundary_smooth,
            "pond_sensitivity": pond_sensitivity,
            "ams_n_colors": ams_n_colors,
            "ams_quality": ams_quality,
            "export_options": export_options,
        }, None

    @app.post("/api/process")
    def api_process():
        params, err = _parse_process_form()
        if err:
            return jsonify({"error": err}), 400
        cleanup_expired_jobs(cache_root, settings=job_cfg, active_job_ids=_active_job_ids())
        job_id = str(uuid.uuid4())
        job_dir = cache_root / job_id
        started_at = time.time()
        write_progress(
            job_dir,
            status="running",
            step="queued",
            message="Starting…",
            percent=0,
            started_at=started_at,
        )

        def worker() -> None:
            with active_jobs_lock:
                active_jobs.add(job_id)
            try:
                with _PROCESS_SEMAPHORE:
                    report = progress_reporter_with_timeout(
                        job_dir,
                        max_runtime_sec=job_cfg.max_runtime_sec,
                        started_at=started_at,
                    )
                    try:
                        process_kml(
                            params["kml_bytes"],
                            cache_root,
                            kml_filename=params["kml_filename"],
                            imagery=params["imagery"],  # type: ignore[arg-type]
                            grid_size=params["grid_size"],
                            buffer_m=params["buffer_m"],
                            vertical_exaggeration=params["vertical_exaggeration"],
                            print_max_size_mm=params["print_max_size_mm"],
                            print_base_extrusion_mm=params["print_base_extrusion_mm"],
                            print_voxel_size_mm=params["print_voxel_size_mm"],
                            print_split_nx=params["print_split_nx"],
                            print_split_nz=params["print_split_nz"],
                            boundary_smooth_m=params["boundary_smooth_m"],
                            pond_sensitivity=params["pond_sensitivity"],
                            ams_n_colors=params["ams_n_colors"],
                            ams_quality=params["ams_quality"],
                            export_options=params["export_options"],
                            job_id=job_id,
                            report=report,
                        )
                    except JobTimeoutError as e:
                        write_progress(
                            job_dir,
                            status="error",
                            step="timeout",
                            message=str(e),
                            percent=100,
                            error=str(e),
                            started_at=started_at,
                        )
                        prune_job_artifacts(job_dir)
                    except ValueError as e:
                        write_progress(
                            job_dir,
                            status="error",
                            step="error",
                            message=str(e),
                            percent=100,
                            error=str(e),
                            started_at=started_at,
                        )
                    except Exception as e:
                        app.logger.exception("process failed")
                        write_progress(
                            job_dir,
                            status="error",
                            step="error",
                            message="Processing failed",
                            percent=100,
                            error=str(e),
                            started_at=started_at,
                        )
            finally:
                with active_jobs_lock:
                    active_jobs.discard(job_id)

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"ok": True, "job_id": job_id}), 202

    @app.get("/api/process/<job_id>/progress")
    def api_process_progress(job_id: str):
        try:
            progress = load_progress(cache_root, job_id)
        except FileNotFoundError:
            return jsonify({"error": "Unknown job"}), 404
        return jsonify(progress)

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

    @app.get("/api/result/<job_id>/preview.glb")
    def api_preview_glb(job_id: str):
        """Same terrain.glb as export, served inline for the Three.js viewer."""
        p = cache_root / job_id / "terrain.glb"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(p, mimetype="model/gltf-binary")

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

    @app.get("/api/result/<job_id>/export_print_textured.glb")
    def api_print_textured_glb(job_id: str):
        """Watertight print solid (mm, Z-up) with embedded satellite texture."""
        p = cache_root / job_id / "terrain_print_textured.glb"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        meta = load_meta(cache_root, job_id)
        return send_file(
            p,
            mimetype="model/gltf-binary",
            as_attachment=True,
            download_name=export_download_filename(meta, "_print_textured.glb"),
        )

    @app.get("/api/result/<job_id>/export_print_ams.glb")
    def api_print_ams_glb(job_id: str):
        """Watertight print solid split into 4 AMS color regions (vertex colors)."""
        p = cache_root / job_id / "terrain_print_ams.glb"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        meta = load_meta(cache_root, job_id)
        return send_file(
            p,
            mimetype="model/gltf-binary",
            as_attachment=True,
            download_name=export_download_filename(meta, "_print_ams.glb"),
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

    @app.get("/api/result/<job_id>/export_print_ams")
    def api_print_ams(job_id: str):
        meta = load_meta(cache_root, job_id)
        ams = (meta.get("print") or {}).get("ams") or {}
        filename = str(ams.get("filename") or "terrain_print_ams_obj.zip")
        suffix = str(ams.get("download_suffix") or "_print_ams_obj.zip")
        p = cache_root / job_id / filename
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(
            p,
            mimetype="application/zip",
            as_attachment=True,
            download_name=export_download_filename(meta, suffix),
        )

    @app.get("/api/result/<job_id>/ams_preview.png")
    def api_ams_preview(job_id: str):
        p = cache_root / job_id / "terrain_print_ams_preview.png"
        if not p.is_file():
            return jsonify({"error": "Not found"}), 404
        return send_file(p, mimetype="image/png")

    return app
