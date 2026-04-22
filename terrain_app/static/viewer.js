import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const statusEl = document.getElementById("status");
const exportsEl = document.getElementById("exports");
const lnkGlb = document.getElementById("lnk-glb");
const lnkZip = document.getElementById("lnk-zip");
const lnkStl = document.getElementById("lnk-stl");
const viewport = document.getElementById("viewport");

let renderer;
let scene;
let camera;
let controls;
let meshGroup;

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function decodeBase64Float32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

async function fetchHeights(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error("Failed to load heights");
  const ab = await r.arrayBuffer();
  return new Float32Array(ab);
}

function buildGeometry(meta, heights) {
  const w = meta.grid_width;
  const h = meta.grid_height;
  const [a, b, c, d, e, f] = meta.transform;
  if (heights.length !== w * h) {
    throw new Error(`Height count ${heights.length} != ${w * h}`);
  }
  const pos = new Float32Array(w * h * 3);
  const uv = new Float32Array(w * h * 2);
  let p = 0;
  let t = 0;
  for (let i = 0; i < h; i++) {
    for (let j = 0; j < w; j++) {
      const x = a * j + b * i + c;
      const z = d * j + e * i + f;
      const y = heights[i * w + j];
      pos[p++] = x;
      pos[p++] = y;
      pos[p++] = z;
      uv[t++] = (j + 0.5) / w;
      uv[t++] = 1.0 - (i + 0.5) / h;
    }
  }
  const idx = [];
  const vid = (i, j) => i * w + j;
  for (let i = 0; i < h - 1; i++) {
    for (let j = 0; j < w - 1; j++) {
      const v00 = vid(i, j);
      const v10 = vid(i + 1, j);
      const v01 = vid(i, j + 1);
      const v11 = vid(i + 1, j + 1);
      idx.push(v00, v10, v01, v10, v11, v01);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

function initThree() {
  if (renderer) return;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1218);
  camera = new THREE.PerspectiveCamera(50, 1, 0.5, 1e9);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  viewport.appendChild(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const amb = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(amb);
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(1, 2, 1);
  scene.add(dir);
  meshGroup = new THREE.Group();
  scene.add(meshGroup);
  window.addEventListener("resize", onResize);
  onResize();
  animate();
}

function onResize() {
  if (!renderer) return;
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);
  if (!renderer) return;
  controls.update();
  renderer.render(scene, camera);
}

function clearMesh() {
  if (!meshGroup) return;
  while (meshGroup.children.length) {
    const o = meshGroup.children.pop();
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (o.material.map) o.material.map.dispose();
      o.material.dispose();
    }
  }
}

async function loadScene(meta, heights, textureUrl) {
  initThree();
  clearMesh();
  const geo = buildGeometry(meta, heights);
  const tex = await new THREE.TextureLoader().loadAsync(textureUrl);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.flipY = false;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const mat = new THREE.MeshStandardMaterial({
    map: tex,
    transparent: true,
    alphaTest: 0.05,
    metalness: 0.05,
    roughness: 0.85,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  meshGroup.add(mesh);
  geo.computeBoundingSphere();
  const bs = geo.boundingSphere;
  const center = bs.center.clone();
  const r = Math.max(bs.radius, 50);
  camera.position.set(center.x + r * 1.2, center.y + r * 0.9, center.z + r * 1.2);
  controls.target.copy(center);
  controls.update();
}

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
  fd.append("vertical_exaggeration", document.getElementById("vex").value);
  fd.append("print_max_size_mm", document.getElementById("print-max").value);
  fd.append("print_base_extrusion_mm", document.getElementById("print-base").value);
  const voxIn = document.getElementById("print-voxel").value.trim();
  if (voxIn) fd.append("print_voxel_size_mm", voxIn);
  setStatus("Fetching 3DEP and imagery…");
  exportsEl.hidden = true;
  document.getElementById("run").disabled = true;
  try {
    const pr = await fetch("/api/process", { method: "POST", body: fd });
    const pj = await pr.json();
    if (!pr.ok) throw new Error(pj.error || pr.statusText);
    const jobId = pj.meta.job_id;
    const rr = await fetch(`/api/result/${jobId}`);
    const rj = await rr.json();
    if (!rr.ok) throw new Error(rj.error || rr.statusText);
    const meta = rj.meta;
    let heights;
    if (rj.heights_inline_base64) {
      heights = decodeBase64Float32(rj.heights_inline_base64);
    } else {
      heights = await fetchHeights(rj.heights.url);
    }
    const texUrl = rj.textures.rgba;
    await loadScene(meta, heights, texUrl);
    lnkGlb.href = `/api/result/${jobId}/export.glb`;
    lnkZip.href = `/api/result/${jobId}/export.zip`;
    if (meta.print && meta.print.ok) {
      lnkStl.href = `/api/result/${jobId}/export.stl`;
      lnkStl.removeAttribute("aria-disabled");
    } else {
      lnkStl.href = "#";
      lnkStl.setAttribute("aria-disabled", "true");
    }
    exportsEl.hidden = false;
    const printLine =
      meta.print && meta.print.ok
        ? `Print STL: ${meta.print.print_max_size_mm ?? meta.print.max_size_mm} mm max, base ${meta.print.base_extrusion_mm} mm, voxel ${meta.print.print_voxel_size_mm != null ? meta.print.print_voxel_size_mm : "auto"}\n`
        : meta.print && (meta.print.error || !meta.print.ok)
          ? `Print STL: unavailable\n`
          : "";
    setStatus(
      `EPSG:${meta.epsg} · ${meta.grid_width}×${meta.grid_height}\n` +
        (meta.elevation_min_m != null
          ? `Elev ${meta.elevation_min_m.toFixed(1)}–${meta.elevation_max_m.toFixed(1)} m (raw DEM)\n`
          : "") +
        printLine +
        `Job ${jobId}`,
    );
  } catch (e) {
    console.error(e);
    setStatus(String(e.message || e));
  } finally {
    document.getElementById("run").disabled = false;
  }
});
