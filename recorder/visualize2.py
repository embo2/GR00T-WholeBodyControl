"""HTTP visualizer for a stream2.py session directory.

Each per-modality dataset is opened with granular.DatasetReader. The page
shows:
- N random RGB frames from orin_camera_rgb/ (JPEG bytes served as-is).
- N random pico_xrt/ body_joints overlaid as a Three.js wireframe.
- A "shuffle" button to re-randomize.

Layout produced by stream2.py:
    <session_dir>/
        orin_camera_rgb/   {orin_camera_rgb: 'rgb',  metadata: 'tree'}
        pico_xrt/          {pico_xrt: 'tree',        metadata: 'tree'}
        unitree_robot/     {unitree_robot: 'tree',   metadata: 'tree'}
        extras/            {extras: 'tree',          metadata: 'tree'}

Usage:
    cd recorder
    uv run python visualize2.py --session data/session_YYYYMMDD_HHMMSS
    # SSH-forward:  ssh -L 8080:localhost:8080 user@orin
    # then open http://localhost:8080/ on your laptop
"""

import argparse
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import granular
import numpy as np


SMPL_PARENTS = [
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]


def make_decoders():
    decoders = dict(granular.decoders)
    decoders['rgb'] = lambda x: x  # raw JPEG bytes pass-through
    return decoders


HTML_PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>recorder visualize2</title>
<style>
body { margin: 0; background: #111; color: #ddd; font-family: ui-monospace, monospace; }
.bar { display: flex; gap: 12px; padding: 10px 14px; align-items: center; border-bottom: 1px solid #333; }
button { background: #222; color: #ddd; border: 1px solid #444; padding: 6px 12px; cursor: pointer; }
input[type=number] { width: 64px; }
.row { display: flex; padding: 12px; gap: 12px; }
.panel { background: #1a1a1a; border: 1px solid #333; padding: 10px; }
#grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
#grid figure { margin: 0; background: #000; border: 1px solid #222; }
#grid img { display: block; width: 100%; height: auto; }
#grid figcaption { font-size: 10px; color: #888; padding: 2px 4px; }
canvas { display: block; }
#status { color: #888; font-size: 12px; }
</style>

<div class="bar">
  <button id="shuffle">⟳ shuffle</button>
  N <input id="n" type="number" min="1" max="50" value="10">
  <span id="status"></span>
</div>

<div class="row">
  <div class="panel" style="flex: 7">
    <div id="grid"></div>
  </div>
  <div class="panel" style="flex: 5">
    <canvas id="three" width="640" height="640"></canvas>
  </div>
</div>

<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const SMPL_PARENTS = __SMPL_PARENTS__;

const status = document.getElementById('status');
const grid = document.getElementById('grid');
const nInput = document.getElementById('n');

// ---- three.js scene ----
const canvas = document.getElementById('three');
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setSize(640, 640);
renderer.setClearColor(0x111111);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 50);
camera.position.set(2.5, 1.6, 2.5);
camera.lookAt(0, 1.0, 0);
const controls = new THREE.OrbitControls(camera, canvas);
controls.target.set(0, 1.0, 0);
controls.update();
scene.add(new THREE.GridHelper(4, 8, 0x444444, 0x222222));
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.5); dir.position.set(2, 4, 1); scene.add(dir);

// Reusable groups so shuffle can clear & redraw without leaks.
let bodyGroup = new THREE.Group();
scene.add(bodyGroup);

function clearBodies() {
  scene.remove(bodyGroup);
  bodyGroup.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
  bodyGroup = new THREE.Group();
  scene.add(bodyGroup);
}

function colorFor(i, n) {
  const h = (i / Math.max(n, 1)) * 360;
  return new THREE.Color(`hsl(${h}, 80%, 60%)`);
}

function addSkeleton(joints, color) {
  const sphereGeom = new THREE.SphereGeometry(0.018, 8, 8);
  const sphereMat = new THREE.MeshLambertMaterial({color});
  for (let i = 0; i < SMPL_PARENTS.length; i++) {
    const j = joints[i];
    const m = new THREE.Mesh(sphereGeom, sphereMat);
    m.position.set(j[0], j[1], j[2]);
    bodyGroup.add(m);
  }
  const positions = [];
  for (let i = 0; i < SMPL_PARENTS.length; i++) {
    const par = SMPL_PARENTS[i];
    if (par < 0) continue;
    const a = joints[i], b = joints[par];
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.7}));
  bodyGroup.add(lines);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ---- shuffle / load ----
async function shuffle() {
  const n = Math.max(1, parseInt(nInput.value || 10));
  status.textContent = 'loading…';
  const info = await (await fetch(`/api/random?n=${n}`)).json();

  // Render image grid
  grid.innerHTML = '';
  if (info.rgb_count === 0) {
    const msg = document.createElement('div');
    msg.style.cssText = 'grid-column: 1 / -1; padding: 24px; color: #c66;';
    msg.textContent = `no orin_camera_rgb data — check the session dir on the server. (rgb_count=0)`;
    grid.appendChild(msg);
  }
  for (const idx of info.rgb_indices) {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = `/api/rgb/${idx}?t=${Date.now()}`;
    img.onerror = () => {
      img.alt = `rgb[${idx}] failed to load`;
      console.error(`/api/rgb/${idx} failed`);
    };
    const cap = document.createElement('figcaption');
    cap.textContent = `rgb[${idx}]`;
    fig.appendChild(img); fig.appendChild(cap);
    grid.appendChild(fig);
  }

  // Render skeletons
  clearBodies();
  if (info.body_indices.length) {
    const bodies = await (await fetch(
      `/api/bodies?indices=${info.body_indices.join(',')}`)).json();
    bodies.forEach((joints, k) => {
      addSkeleton(joints, colorFor(k, bodies.length));
    });
  }

  status.textContent =
    `rgb: ${info.rgb_count} total (showing ${info.rgb_indices.length}) · ` +
    `body: ${info.body_count} total (showing ${info.body_indices.length})`;
}

document.getElementById('shuffle').onclick = shuffle;
nInput.addEventListener('change', shuffle);
shuffle();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    rgb_reader = None
    body_reader = None

    def log_message(self, fmt, *args):
        # Concise per-request log so empty-grid issues are debuggable.
        print(f"[visualize2] {self.address_string()} - {fmt % args}", flush=True)

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        try:
            path = self.path.split("?")[0]
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            if path == "/" or path == "/index.html":
                page = HTML_PAGE.replace("__SMPL_PARENTS__", json.dumps(SMPL_PARENTS))
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/random":
                n = 10
                for kv in qs.split("&"):
                    if kv.startswith("n="):
                        try:
                            n = max(1, int(kv[2:]))
                        except ValueError:
                            pass
                rgb_total = len(self.rgb_reader) if self.rgb_reader else 0
                body_total = len(self.body_reader) if self.body_reader else 0
                rgb_idx = sorted(random.sample(range(rgb_total), min(n, rgb_total))) \
                    if rgb_total else []
                body_idx = sorted(random.sample(range(body_total), min(n, body_total))) \
                    if body_total else []
                self._send_json({
                    "rgb_count": rgb_total, "body_count": body_total,
                    "rgb_indices": rgb_idx, "body_indices": body_idx,
                })
                return
            if path.startswith("/api/rgb/"):
                idx = int(path[len("/api/rgb/"):])
                if not self.rgb_reader:
                    self._send(404, b"no rgb dataset", "text/plain")
                    return
                jpeg = self.rgb_reader[idx]['orin_camera_rgb']
                if not isinstance(jpeg, (bytes, bytearray)):
                    jpeg = bytes(jpeg)
                self._send(200, jpeg, "image/jpeg")
                return
            if path == "/api/bodies":
                if not self.body_reader:
                    self._send_json([])
                    return
                indices = []
                for kv in qs.split("&"):
                    if kv.startswith("indices="):
                        for tok in kv[len("indices="):].split(","):
                            if tok:
                                try:
                                    indices.append(int(tok))
                                except ValueError:
                                    pass
                out = []
                for i in indices:
                    body = self.body_reader[i]['pico_xrt']['body_joints']
                    out.append(np.asarray(body).astype(float).tolist())
                self._send_json(out)
                return
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._send(500, f"error: {e!r}".encode(), "text/plain")
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True,
                   help="path to a stream2 session directory")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0",
                   help="bind host. Use 127.0.0.1 if you'll only access via SSH-forward.")
    args = p.parse_args()

    decoders = make_decoders()
    rgb_dir = os.path.join(args.session, "orin_camera_rgb")
    body_dir = os.path.join(args.session, "pico_xrt")

    rgb_reader = (granular.DatasetReader(rgb_dir, decoders)
                  if os.path.isdir(rgb_dir) else None)
    body_reader = (granular.DatasetReader(body_dir, decoders)
                   if os.path.isdir(body_dir) else None)
    Handler.rgb_reader = rgb_reader
    Handler.body_reader = body_reader

    rgb_n = len(rgb_reader) if rgb_reader else 0
    body_n = len(body_reader) if body_reader else 0
    print(f"[visualize2] session={args.session}", flush=True)
    print(f"[visualize2] orin_camera_rgb: "
          f"{'(missing)' if rgb_reader is None else f'{rgb_n} frames'}", flush=True)
    print(f"[visualize2] pico_xrt:        "
          f"{'(missing)' if body_reader is None else f'{body_n} frames'}", flush=True)
    if rgb_reader is None and body_reader is None:
        print("[visualize2] NOTE: neither dataset found. Did you point --session "
              "at a specific session_<timestamp>/ subdir, not the parent data/?",
              flush=True)
    elif rgb_n == 0 or body_n == 0:
        print("[visualize2] NOTE: at least one dataset has 0 frames. Make sure "
              "the recorder ran with running=True (R3/is_first event arrived).",
              flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[visualize2] serving on http://{args.host}:{args.port}/", flush=True)
    print(f"[visualize2] SSH forward:  "
          f"ssh -L {args.port}:localhost:{args.port} <user>@<orin>", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if rgb_reader is not None:
            rgb_reader.close()
        if body_reader is not None:
            body_reader.close()


if __name__ == "__main__":
    main()
