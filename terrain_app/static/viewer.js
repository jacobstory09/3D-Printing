const statusEl = document.getElementById("status");
const progressWrap = document.getElementById("progress-wrap");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");
const exportsEl = document.getElementById("exports");
const lnkGlb = document.getElementById("lnk-glb");
const lnkZip = document.getElementById("lnk-zip");
const lnkStl = document.getElementById("lnk-stl");
const lnkPrintTexturedGlb = document.getElementById("lnk-print-textured-glb");
const lnkPrintAmsGlb = document.getElementById("lnk-print-ams-glb");
const lnk3mf = document.getElementById("lnk-3mf");
const lnkAmsColor = document.getElementById("lnk-ams-color");
const lnkPiecesZip = document.getElementById("lnk-pieces-zip");
const amsColorsEl = document.getElementById("ams-colors");
const amsColorListEl = document.getElementById("ams-color-list");
const viewport = document.getElementById("viewport");
const pondEditorEl = document.getElementById("pond-editor");
const pondShowEl = document.getElementById("pond-show");
const pondFinishBtn = document.getElementById("pond-finish");
const pondCancelBtn = document.getElementById("pond-cancel");
const buildAmsBtn = document.getElementById("build-ams");

let canvas;
let ctx;
let resizeObserver;
/** Loaded RGBA satellite image (alpha = KML footprint). */
let mapImage = null;
/** Screen-space pan/zoom: image drawn at (offsetX, offsetY) with uniform scale. */
let view = { scale: 1, offsetX: 0, offsetY: 0 };
let drag = null;
let currentJobId = null;
/** @type {Array<{id: string, source: string, vertices: number[][]}>} */
let pondShapes = [];
let pondMode = "navigate";
/** @type {number[][]} */
let draftVertices = [];
let showPonds = true;

function initCanvas() {
  if (canvas) return;
  canvas = document.createElement("canvas");
  canvas.setAttribute("aria-label", "Property map preview");
  viewport.appendChild(canvas);
  ctx = canvas.getContext("2d");
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("dblclick", onPointerDblClick);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  resizeObserver = new ResizeObserver(onResize);
  resizeObserver.observe(viewport);
  window.addEventListener("resize", onResize);
  onResize();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load map image"));
    img.src = url;
  });
}

function viewportSize() {
  return {
    w: Math.max(viewport.clientWidth, 1),
    h: Math.max(viewport.clientHeight, 1),
  };
}

function fitView() {
  if (!mapImage) return;
  const { w, h } = viewportSize();
  const pad = 0.04;
  const scaleX = (w * (1 - 2 * pad)) / mapImage.width;
  const scaleY = (h * (1 - 2 * pad)) / mapImage.height;
  view.scale = Math.min(scaleX, scaleY);
  view.offsetX = (w - mapImage.width * view.scale) / 2;
  view.offsetY = (h - mapImage.height * view.scale) / 2;
  draw();
}

function screenToTexture(sx, sy) {
  return {
    col: (sx - view.offsetX) / view.scale,
    row: (sy - view.offsetY) / view.scale,
  };
}

function textureToScreen(col, row) {
  return {
    sx: view.offsetX + col * view.scale,
    sy: view.offsetY + row * view.scale,
  };
}

function newShapeId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `pond-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function pointInPolygon(col, row, vertices) {
  let inside = false;
  for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
    const xi = vertices[i][0];
    const yi = vertices[i][1];
    const xj = vertices[j][0];
    const yj = vertices[j][1];
    const intersect =
      yi > row !== yj > row &&
      col < ((xj - xi) * (row - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function drawPondOverlays() {
  if (!ctx || !mapImage || !showPonds) return;
  const drawPoly = (vertices, fill, stroke, lineWidth = 2) => {
    if (!vertices || vertices.length < 2) return;
    ctx.beginPath();
    const p0 = textureToScreen(vertices[0][0], vertices[0][1]);
    ctx.moveTo(p0.sx, p0.sy);
    for (let i = 1; i < vertices.length; i++) {
      const p = textureToScreen(vertices[i][0], vertices[i][1]);
      ctx.lineTo(p.sx, p.sy);
    }
    ctx.closePath();
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  };
  for (const sh of pondShapes) {
    drawPoly(sh.vertices, "rgba(45, 120, 180, 0.35)", "rgba(70, 160, 220, 0.9)");
  }
  if (draftVertices.length > 0) {
    ctx.beginPath();
    const d0 = textureToScreen(draftVertices[0][0], draftVertices[0][1]);
    ctx.moveTo(d0.sx, d0.sy);
    for (let i = 1; i < draftVertices.length; i++) {
      const p = textureToScreen(draftVertices[i][0], draftVertices[i][1]);
      ctx.lineTo(p.sx, p.sy);
    }
    ctx.strokeStyle = "rgba(255, 220, 80, 0.95)";
    ctx.lineWidth = 2;
    ctx.stroke();
    for (const v of draftVertices) {
      const p = textureToScreen(v[0], v[1]);
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 220, 80, 0.95)";
      ctx.fill();
    }
  }
}

function draw() {
  if (!ctx || !canvas) return;
  const { w, h } = viewportSize();
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0f1218";
  ctx.fillRect(0, 0, w, h);
  if (!mapImage) return;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(
    mapImage,
    view.offsetX,
    view.offsetY,
    mapImage.width * view.scale,
    mapImage.height * view.scale
  );
  drawPondOverlays();
}

function updatePondModeUi() {
  const nav = document.getElementById("pond-mode-nav");
  const drawBtn = document.getElementById("pond-mode-draw");
  const delBtn = document.getElementById("pond-mode-delete");
  nav?.classList.toggle("is-active", pondMode === "navigate");
  drawBtn?.classList.toggle("is-active", pondMode === "draw");
  delBtn?.classList.toggle("is-active", pondMode === "delete");
  if (pondFinishBtn) pondFinishBtn.hidden = pondMode !== "draw" || draftVertices.length < 3;
  if (pondCancelBtn) pondCancelBtn.hidden = pondMode !== "draw" || draftVertices.length === 0;
  if (canvas) {
    canvas.style.cursor =
      pondMode === "navigate" ? "grab" : pondMode === "draw" ? "crosshair" : "pointer";
  }
}

function setPondMode(mode) {
  pondMode = mode;
  if (mode !== "draw") draftVertices = [];
  updatePondModeUi();
  draw();
}

function finishDraftPolygon() {
  if (draftVertices.length < 3) return;
  pondShapes.push({
    id: newShapeId(),
    source: "manual",
    vertices: draftVertices.map((v) => [v[0], v[1]]),
  });
  draftVertices = [];
  setPondMode("navigate");
}

function deletePondAt(col, row) {
  for (let i = pondShapes.length - 1; i >= 0; i--) {
    if (pointInPolygon(col, row, pondShapes[i].vertices)) {
      pondShapes.splice(i, 1);
      draw();
      return true;
    }
  }
  return false;
}

function canvasPoint(ev) {
  const rect = canvas.getBoundingClientRect();
  return { sx: ev.clientX - rect.left, sy: ev.clientY - rect.top };
}

function onResize() {
  if (!canvas || !viewport) return;
  const { w, h } = viewportSize();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (mapImage) {
    const cx = w / 2;
    const cy = h / 2;
    const wx = (cx - view.offsetX) / view.scale;
    const wy = (cy - view.offsetY) / view.scale;
    view.offsetX = cx - wx * view.scale;
    view.offsetY = cy - wy * view.scale;
  }
  draw();
}

function zoomAt(factor, sx, sy) {
  const wx = (sx - view.offsetX) / view.scale;
  const wy = (sy - view.offsetY) / view.scale;
  view.scale = Math.min(Math.max(view.scale * factor, 0.02), 200);
  view.offsetX = sx - wx * view.scale;
  view.offsetY = sy - wy * view.scale;
  draw();
}

function onWheel(ev) {
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left;
  const sy = ev.clientY - rect.top;
  const factor = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
  zoomAt(factor, sx, sy);
}

function onPointerDown(ev) {
  if (ev.button !== 0) return;
  const { sx, sy } = canvasPoint(ev);
  if (pondMode === "draw") {
    const { col, row } = screenToTexture(sx, sy);
    draftVertices.push([col, row]);
    updatePondModeUi();
    draw();
    return;
  }
  if (pondMode === "delete") {
    const { col, row } = screenToTexture(sx, sy);
    deletePondAt(col, row);
    return;
  }
  canvas.setPointerCapture(ev.pointerId);
  canvas.style.cursor = "grabbing";
  drag = { x: ev.clientX, y: ev.clientY, ox: view.offsetX, oy: view.offsetY };
}

function onPointerDblClick(ev) {
  if (pondMode !== "draw" || draftVertices.length < 3) return;
  ev.preventDefault();
  finishDraftPolygon();
}

function onPointerMove(ev) {
  if (!drag) return;
  view.offsetX = drag.ox + (ev.clientX - drag.x);
  view.offsetY = drag.oy + (ev.clientY - drag.y);
  draw();
}

function onPointerUp(ev) {
  if (!drag) return;
  canvas.releasePointerCapture(ev.pointerId);
  canvas.style.cursor = "grab";
  drag = null;
}

async function loadMapPreview(textureUrl) {
  initCanvas();
  mapImage = await loadImage(textureUrl);
  fitView();
}

document.getElementById("print-split-preset")?.addEventListener("change", (ev) => {
  const wrap = document.getElementById("print-split-custom");
  if (!wrap) return;
  const isCustom = ev.target.value === "custom";
  wrap.classList.toggle("is-visible", isCustom);
  wrap.hidden = !isCustom;
});

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function setProgress(message, percent) {
  if (!progressWrap || !progressBar || !progressLabel) return;
  progressWrap.hidden = false;
  const pct = Math.max(0, Math.min(100, Number(percent) || 0));
  progressBar.value = pct;
  progressLabel.textContent = pct > 0 ? `${message} (${pct}%)` : message;
}

function hideProgress() {
  if (progressWrap) progressWrap.hidden = true;
}

async function readJson(resp) {
  const text = await resp.text();
  if (!text.trim()) {
    throw new Error(
      resp.ok
        ? "Server returned an empty response (is the dev server still running?)"
        : `Request failed (${resp.status} ${resp.statusText})`
    );
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Invalid JSON from server (${resp.status})`);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForJob(jobId) {
  let emptyRetries = 0;
  while (true) {
    let r;
    try {
      r = await fetch(`/api/process/${jobId}/progress`);
    } catch {
      throw new Error("Lost connection to server while waiting for the job");
    }
    let p;
    try {
      p = await readJson(r);
      emptyRetries = 0;
    } catch (err) {
      // Dev-server reload or a race while progress.json is being written.
      if (emptyRetries < 8 && /empty response|Invalid JSON/i.test(String(err.message))) {
        emptyRetries += 1;
        await sleep(500);
        continue;
      }
      throw err;
    }
    if (!r.ok) {
      if (r.status === 404) throw new Error("Job expired or was removed from the server");
      throw new Error(p.error || r.statusText);
    }
    setProgress(p.message || "Working…", p.percent);
    if (p.status === "done") return jobId;
    if (p.status === "error") throw new Error(p.error || p.message || "Processing failed");
    await sleep(1000);
  }
}

function exportDownloadName(meta, suffix) {
  const base = meta.export_basename || "terrain";
  return `${base}${suffix}`;
}

function exportBuilt(meta, key) {
  const built = meta.exports && meta.exports.built;
  if (built && Object.prototype.hasOwnProperty.call(built, key)) {
    return Boolean(built[key]);
  }
  if (key === "preview_glb" || key === "preview_obj") return false;
  if (key === "print_stl" || key === "print_ams") {
    return Boolean(meta.print && meta.print.ok);
  }
  if (key === "print_textured_glb") return Boolean(meta.print && meta.print.print_textured_glb);
  if (key === "print_ams_glb") {
    const ams = meta.print && meta.print.ams;
    return Boolean(ams && ams.print_ams_glb);
  }
  if (key === "print_3mf") return Boolean(meta.print && meta.print.ok);
  if (key === "print_pieces") {
    const pz = meta.print && meta.print.pieces;
    return Boolean(pz && pz.ok && pz.count > 0);
  }
  return false;
}

function appendExportFlags(fd) {
  const pairs = [
    ["export_preview_glb", "export-preview-glb"],
    ["export_preview_obj", "export-preview-obj"],
    ["export_quad_mask", "export-quad-mask"],
    ["export_print_stl", "export-print-stl"],
    ["export_print_3mf", "export-print-3mf"],
    ["export_print_textured_glb", "export-print-textured-glb"],
    ["export_print_ams", "export-print-ams"],
    ["export_print_ams_glb", "export-print-ams-glb"],
    ["export_print_pieces", "export-print-pieces"],
  ];
  for (const [name, id] of pairs) {
    const el = document.getElementById(id);
    if (el && el.checked) fd.append(name, "1");
  }
}

function setExportLink(anchor, href, downloadName) {
  if (!anchor) return;
  anchor.href = href;
  if (downloadName) anchor.download = downloadName;
  else anchor.removeAttribute("download");
}

function swatchHexForColor(c) {
  if (c.role === "pond" && Array.isArray(c.filament_rgb) && c.filament_rgb.length >= 3) {
    return (
      "#" +
      c.filament_rgb
        .map((x) => Math.max(0, Math.min(255, Number(x) | 0)).toString(16).padStart(2, "0"))
        .join("")
    );
  }
  return c.hex || "#888";
}

function renderAmsColors(ams) {
  if (!amsColorListEl || !amsColorsEl) return;
  if (!ams || !ams.ok || !Array.isArray(ams.colors) || ams.colors.length === 0) {
    amsColorsEl.hidden = true;
    amsColorListEl.innerHTML = "";
    return;
  }
  amsColorListEl.innerHTML = "";
  for (const c of ams.colors) {
    const hex = swatchHexForColor(c);
    const row = document.createElement("div");
    row.className = "ams-color-row";
    const swatch = document.createElement("span");
    swatch.className = "ams-swatch";
    swatch.style.background = hex;
    swatch.title = hex;
    const label = document.createElement("div");
    label.innerHTML = `<strong>${c.name || c.part_name || "Color"}</strong><div class="ams-color-meta">${c.part_name || ""} · ${hex} · ${c.coverage_pct != null ? `${c.coverage_pct}%` : ""}</div>`;
    const tris = document.createElement("div");
    tris.className = "ams-color-meta";
    tris.textContent = c.triangle_count != null ? `${c.triangle_count} tris` : "";
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(tris);
    amsColorListEl.appendChild(row);
  }
  amsColorsEl.hidden = false;
}

function pondsPending(meta) {
  const ponds = meta && meta.ponds;
  return Boolean(ponds && ponds.status === "pending_edit");
}

async function loadPondShapes(jobId) {
  const r = await fetch(`/api/result/${jobId}/pond_shapes.json`);
  const doc = await readJson(r);
  if (!r.ok) throw new Error(doc.error || "Failed to load pond shapes");
  pondShapes = Array.isArray(doc.shapes) ? doc.shapes : [];
  draw();
}

function setupPondEditor(jobId, meta) {
  currentJobId = jobId;
  if (!pondsPending(meta)) {
    if (pondEditorEl) pondEditorEl.hidden = true;
    return;
  }
  if (pondEditorEl) pondEditorEl.hidden = false;
  setPondMode("navigate");
  loadPondShapes(jobId).catch((e) => setStatus(String(e.message || e)));
}

function wireExportLinks(jobId, meta) {
  const jobBuilt = (key) => exportBuilt(meta, key);
  if (jobBuilt("preview_glb")) {
    setExportLink(lnkGlb, `/api/result/${jobId}/export.glb`, exportDownloadName(meta, ".glb"));
    lnkGlb?.removeAttribute("aria-disabled");
  } else {
    setExportLink(lnkGlb, "#", null);
    lnkGlb?.setAttribute("aria-disabled", "true");
  }
  if (jobBuilt("preview_obj")) {
    setExportLink(lnkZip, `/api/result/${jobId}/export.zip`, exportDownloadName(meta, "_obj.zip"));
    lnkZip?.removeAttribute("aria-disabled");
  } else {
    setExportLink(lnkZip, "#", null);
    lnkZip?.setAttribute("aria-disabled", "true");
  }
  if (meta.print && meta.print.ok) {
    if (jobBuilt("print_stl")) {
      setExportLink(lnkStl, `/api/result/${jobId}/export.stl`, exportDownloadName(meta, "_print.stl"));
      lnkStl?.removeAttribute("aria-disabled");
    } else {
      setExportLink(lnkStl, "#", null);
      lnkStl?.setAttribute("aria-disabled", "true");
    }
    if (jobBuilt("print_textured_glb")) {
      setExportLink(
        lnkPrintTexturedGlb,
        `/api/result/${jobId}/export_print_textured.glb`,
        exportDownloadName(meta, "_print_textured.glb")
      );
      lnkPrintTexturedGlb?.removeAttribute("aria-disabled");
    } else {
      setExportLink(lnkPrintTexturedGlb, "#", null);
      lnkPrintTexturedGlb?.setAttribute("aria-disabled", "true");
    }
    if (jobBuilt("print_3mf")) {
      setExportLink(lnk3mf, `/api/result/${jobId}/export.3mf`, exportDownloadName(meta, "_print.3mf"));
      lnk3mf?.removeAttribute("aria-disabled");
    } else {
      setExportLink(lnk3mf, "#", null);
      lnk3mf?.setAttribute("aria-disabled", "true");
    }
    const ams = meta.print && meta.print.ams;
    if (jobBuilt("print_ams") && ams && ams.ok) {
      const amsSuffix = ams.download_suffix || "_print_ams_obj.zip";
      setExportLink(
        lnkAmsColor,
        `/api/result/${jobId}/export_print_ams`,
        exportDownloadName(meta, amsSuffix)
      );
      lnkAmsColor?.removeAttribute("aria-disabled");
      renderAmsColors(ams);
    } else {
      setExportLink(lnkAmsColor, "#", null);
      lnkAmsColor?.setAttribute("aria-disabled", "true");
      if (ams && ams.ok) renderAmsColors(ams);
      else renderAmsColors(null);
    }
    if (jobBuilt("print_ams_glb") && ams && ams.print_ams_glb) {
      setExportLink(
        lnkPrintAmsGlb,
        `/api/result/${jobId}/export_print_ams.glb`,
        exportDownloadName(meta, "_print_ams.glb")
      );
      lnkPrintAmsGlb?.removeAttribute("aria-disabled");
    } else {
      setExportLink(lnkPrintAmsGlb, "#", null);
      lnkPrintAmsGlb?.setAttribute("aria-disabled", "true");
    }
  } else {
    setExportLink(lnkStl, "#", null);
    lnkStl?.setAttribute("aria-disabled", "true");
    setExportLink(lnkPrintTexturedGlb, "#", null);
    lnkPrintTexturedGlb?.setAttribute("aria-disabled", "true");
    setExportLink(lnkPrintAmsGlb, "#", null);
    lnkPrintAmsGlb?.setAttribute("aria-disabled", "true");
    setExportLink(lnk3mf, "#", null);
    lnk3mf?.setAttribute("aria-disabled", "true");
    setExportLink(lnkAmsColor, "#", null);
    lnkAmsColor?.setAttribute("aria-disabled", "true");
    renderAmsColors(null);
  }
  const pz = meta.print && meta.print.pieces;
  if (jobBuilt("print_pieces") && pz && pz.ok && pz.count > 0) {
    setExportLink(
      lnkPiecesZip,
      `/api/result/${jobId}/export_print_pieces.zip`,
      exportDownloadName(meta, "_print_pieces.zip")
    );
    lnkPiecesZip?.removeAttribute("aria-disabled");
  } else {
    setExportLink(lnkPiecesZip, "#", null);
    lnkPiecesZip?.setAttribute("aria-disabled", "true");
  }
  exportsEl.hidden = false;
}

function buildStatusText(meta, jobId) {
  const pz = meta.print && meta.print.pieces;
  const splitDesc =
    meta.print && (meta.print.split_nx > 1 || meta.print.split_nz > 1)
      ? `Puzzle: ${meta.print.split_nx}×${meta.print.split_nz} (each tile scaled to ~print max mm on ground; monolithic STL is full assembly)\n`
      : "";
  const piecesLine =
    pz && pz.count > 0
      ? `Puzzle STLs: ${pz.count} file(s), ~${
          pz.per_piece_size_mm
            ? (
                pz.per_piece_size_mm.max_horizontal_mm_approx != null
                  ? Number(pz.per_piece_size_mm.max_horizontal_mm_approx)
                  : Math.max(pz.per_piece_size_mm.x, pz.per_piece_size_mm.y ?? 0)
              ).toFixed(1)
            : "?"
        } mm max on ground per tile\n`
      : meta.print && (meta.print.split_nx > 1 || meta.print.split_nz > 1)
        ? `Puzzle STLs: none (split failed or empty)\n`
        : "";
  const printLine =
    meta.print && meta.print.ok
      ? `Print STL: ${meta.print.print_max_size_mm ?? meta.print.max_size_mm} mm max, base ${meta.print.base_extrusion_mm} mm, voxel ${
          meta.print.voxel_size_mm_request != null
            ? meta.print.voxel_size_mm_request
            : meta.print.print_voxel_size_mm != null
              ? `auto (~${Number(meta.print.print_voxel_size_mm).toFixed(2)} mm)`
              : "auto"
        }${
          meta.print.print_vertical_span_mm != null
            ? `, model height ~${Number(meta.print.print_vertical_span_mm).toFixed(2)} mm (slicer Z)`
            : ""
        }\n`
      : meta.print && (meta.print.error || !meta.print.ok)
        ? `Print STL: unavailable\n`
        : "";
  const multiBodyWarn =
    meta.print && meta.print.ok && meta.print.print_component_count > 1
      ? `Warning: print mesh has ${meta.print.print_component_count} separate bodies (expected one)\n`
      : "";
  const fullRel = meta.full_raster_relief_m;
  const clipRel = meta.clipped_surface_relief_m;
  const reliefHint =
    fullRel != null && clipRel != null && fullRel > 1 && clipRel < 0.05 * fullRel
      ? `Note: KML footprint is much flatter than the full download tile; preview matches print.\n`
      : "";
  const vex = meta.vertical_exaggeration != null ? Number(meta.vertical_exaggeration) : 1;
  const meshZSpan =
    meta.mesh_vertex_z_span_m != null
      ? meta.mesh_vertex_z_span_m
      : meta.mesh_vertex_y_span_m != null
        ? meta.mesh_vertex_y_span_m
        : null;
  const meshZLine =
    meshZSpan != null
      ? `Terrain mesh: DEM → vertex Z span ${Number(meshZSpan).toFixed(2)} m (×${vex} vex); same Z in print STL (mm after scaling).\n`
      : "";
  const maskR = meta.masked_raster_relief_m;
  const reliefMismatch =
    maskR != null &&
    meshZSpan != null &&
    maskR > 0.5 &&
    meshZSpan < 0.05 * maskR
      ? `Warning: footprint DEM varies ~${Number(maskR).toFixed(2)} m but triangulated mesh only ~${Number(meshZSpan).toFixed(2)} m—possible polygon vs grid mismatch.\n`
      : "";
  const pondLine = pondsPending(meta)
    ? `Ponds: review ${meta.ponds.shape_count ?? 0} suggested shape(s) on the map, then Build AMS export.\n`
    : "";
  return (
    `EPSG:${meta.epsg} · ${meta.grid_width}×${meta.grid_height}\n` +
    (meta.elevation_min_m != null
      ? `Elev ${meta.elevation_min_m.toFixed(1)}–${meta.elevation_max_m.toFixed(1)} m (raw DEM)\n`
      : "") +
    meshZLine +
    reliefMismatch +
    reliefHint +
    pondLine +
    splitDesc +
    piecesLine +
    printLine +
    multiBodyWarn +
    `Job ${jobId}`
  );
}

document.getElementById("view-reset")?.addEventListener("click", fitView);

document.getElementById("pond-mode-nav")?.addEventListener("click", () => setPondMode("navigate"));
document.getElementById("pond-mode-draw")?.addEventListener("click", () => setPondMode("draw"));
document.getElementById("pond-mode-delete")?.addEventListener("click", () => setPondMode("delete"));
pondFinishBtn?.addEventListener("click", () => finishDraftPolygon());
pondCancelBtn?.addEventListener("click", () => {
  draftVertices = [];
  setPondMode("navigate");
});
pondShowEl?.addEventListener("change", () => {
  showPonds = Boolean(pondShowEl.checked);
  draw();
});

buildAmsBtn?.addEventListener("click", async () => {
  if (!currentJobId) return;
  if (buildAmsBtn.disabled) return;
  buildAmsBtn.disabled = true;
  setProgress("Building AMS export…", 5);
  try {
    const resp = await fetch(`/api/result/${currentJobId}/build_ams`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shapes: pondShapes }),
    });
    const body = await readJson(resp);
    if (!resp.ok) throw new Error(body.error || resp.statusText);
    await waitForJob(currentJobId);
    hideProgress();
    const rr = await fetch(`/api/result/${currentJobId}`);
    const rj = await readJson(rr);
    if (!rr.ok) throw new Error(rj.error || rr.statusText);
    const meta = rj.meta;
    wireExportLinks(currentJobId, meta);
    if (pondEditorEl) pondEditorEl.hidden = true;
    setStatus(buildStatusText(meta, currentJobId));
  } catch (e) {
    console.error(e);
    hideProgress();
    setStatus(String(e.message || e));
  } finally {
    buildAmsBtn.disabled = false;
  }
});

document.getElementById("run").addEventListener("click", async () => {
  const fileInput = document.getElementById("kml");
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Choose a KML file first.");
    return;
  }
  const fd = new FormData();
  fd.append("kml", file);
  fd.append("imagery", document.getElementById("imagery").value);
  fd.append("grid_size", document.getElementById("grid").value);
  fd.append("buffer_m", document.getElementById("buffer").value);
  fd.append("boundary_smooth_m", document.getElementById("boundary-smooth").value);
  fd.append("vertical_exaggeration", document.getElementById("vex").value);
  fd.append("pond_sensitivity", document.getElementById("pond-sensitivity").value);
  fd.append("ams_n_colors", document.getElementById("ams-n-colors").value);
  fd.append("ams_quality", document.getElementById("ams-quality").value);
  appendExportFlags(fd);
  fd.append("print_max_size_mm", document.getElementById("print-max").value);
  fd.append("print_base_extrusion_mm", document.getElementById("print-base").value);
  const voxIn = document.getElementById("print-voxel").value.trim();
  const voxNum = voxIn === "" ? NaN : Number(voxIn);
  if (voxIn && voxIn.toLowerCase() !== "auto" && voxNum > 0) {
    fd.append("print_voxel_size_mm", voxIn);
  }
  const preset = document.getElementById("print-split-preset").value;
  let spx = 1;
  let spz = 1;
  if (preset === "2x2") {
    spx = 2;
    spz = 2;
  } else if (preset === "3x3") {
    spx = 3;
    spz = 3;
  } else if (preset === "4x4") {
    spx = 4;
    spz = 4;
  } else if (preset === "custom") {
    spx = Math.min(12, Math.max(1, parseInt(document.getElementById("print-split-nx").value, 10) || 1));
    spz = Math.min(12, Math.max(1, parseInt(document.getElementById("print-split-nz").value, 10) || 1));
  }
  fd.append("print_split_nx", String(spx));
  fd.append("print_split_nz", String(spz));
  setStatus("");
  setProgress("Uploading…", 0);
  exportsEl.hidden = true;
  document.getElementById("run").disabled = true;
  try {
    const pr = await fetch("/api/process", { method: "POST", body: fd });
    const pj = await readJson(pr);
    if (!pr.ok) throw new Error(pj.error || pr.statusText);
    const jobId = pj.job_id;
    if (!jobId) throw new Error("No job id returned");
    await waitForJob(jobId);
    hideProgress();
    const rr = await fetch(`/api/result/${jobId}`);
    const rj = await readJson(rr);
    if (!rr.ok) throw new Error(rj.error || rr.statusText);
    const meta = rj.meta;
    await loadMapPreview(rj.textures.rgba);
    wireExportLinks(jobId, meta);
    setupPondEditor(jobId, meta);
    setStatus(buildStatusText(meta, jobId));
  } catch (e) {
    console.error(e);
    hideProgress();
    setStatus(String(e.message || e));
  } finally {
    document.getElementById("run").disabled = false;
  }
});
