"""Visualize a recorder dataset.

Hosts a single-page HTTP UI on `--port` (default 8080). SSH-forward with:
    ssh -L 8080:localhost:8080 user@orin
…then open http://localhost:8080/ in a local browser.

The UI plays back one chunk at a time:
- the orin_camera RGB stream as an HTML5 mp4 video
- the pico_xrt SMPL body joints as a Three.js wireframe synced to the video
- the extras flags (is_first / is_last / reward) as on-timeline markers

Dataset is loaded with granular.DatasetReader (github.com/danijar/granular).
"""

import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import granular
import imageio.v2 as imageio
import numpy as np


# SMPL kinematic tree (parent indices for each of 24 joints).
SMPL_PARENTS = [
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]
SMPL_NAMES = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee",
    "spine2", "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot",
    "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hand", "R_hand",
]


def encode_mp4(frames, fps):
    """Re-encode (T, H, W, 3) uint8 BGR frames to an mp4 in-memory."""
    buf = io.BytesIO()
    with imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                            pixelformat="yuv420p", quality=8) as w:
        for f in frames:
            # camera publishes BGR; imageio expects RGB
            w.append_data(np.ascontiguousarray(f[..., ::-1]))
    return buf.getvalue()


def chunk_fps(metadata, key):
    """Estimate fps for a stream from its 'created' (ns) timestamps."""
    ts = metadata.get(key, {}).get("created")
    if ts is None or len(ts) < 2:
        return 30.0
    deltas_ns = np.diff(np.asarray(ts).astype(np.int64))
    if (deltas_ns > 0).any():
        return float(1e9 / np.median(deltas_ns[deltas_ns > 0]))
    return 30.0


def to_serializable(x):
    """Make numpy arrays / scalars JSON-serializable."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    return x


HTML_PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>recorder visualizer</title>
<style>
body { margin: 0; background: #111; color: #ddd; font-family: ui-monospace, monospace; }
.row { display: flex; gap: 12px; padding: 12px; }
.panel { background: #1a1a1a; border: 1px solid #333; padding: 8px; }
video { width: 640px; height: 480px; background: #000; }
canvas { display: block; }
.bar { display: flex; gap: 8px; align-items: center; padding: 8px; }
button { background: #222; color: #ddd; border: 1px solid #444; padding: 6px 10px; cursor: pointer; }
input[type=number] { width: 64px; }
.tag { padding: 2px 6px; border-radius: 3px; font-size: 11px; }
.running { background: #2a5; color: #000; }
.paused  { background: #a52; color: #fff; }
.reward  { background: #aa3; color: #000; }
#extras { margin-top: 6px; font-size: 12px; }
#timeline { width: 100%; height: 24px; background: #000; position: relative; }
#timeline .mark { position: absolute; top: 0; bottom: 0; width: 2px; }
#timeline .first { background: #2a5; }
#timeline .last  { background: #a52; }
#timeline .rew   { background: #aa3; }
#status { font-size: 12px; color: #888; }
</style>

<div class="bar">
  <button id="prev">◀ prev</button>
  chunk <input id="idx" type="number" min="0" value="0">
  / <span id="count">?</span>
  <button id="next">next ▶</button>
  <span id="status"></span>
</div>

<div class="row">
  <div class="panel">
    <video id="vid" controls muted playsinline></video>
    <div id="extras"></div>
    <div id="timeline"></div>
  </div>
  <div class="panel">
    <canvas id="three" width="640" height="480"></canvas>
  </div>
</div>

<script src="https://unpkg.com/three@0.155.0/build/three.min.js"></script>
<script>
const SMPL_PARENTS = __SMPL_PARENTS__;
let chunkCount = 0;
let chunkData = null;     // {body_joints, body_ts, video_ts, extras}

const vid = document.getElementById('vid');
const idxInput = document.getElementById('idx');
const countSpan = document.getElementById('count');
const status = document.getElementById('status');
const extrasEl = document.getElementById('extras');
const timeline = document.getElementById('timeline');

// ---- three.js skeleton ----
const canvas = document.getElementById('three');
const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
renderer.setSize(640, 480);
renderer.setClearColor(0x111111);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 640/480, 0.05, 50);
camera.position.set(2.5, 1.6, 2.5);
camera.lookAt(0, 1.0, 0);

scene.add(new THREE.GridHelper(4, 8, 0x444444, 0x222222));
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.5); dir.position.set(2, 4, 1); scene.add(dir);

const jointGeom = new THREE.SphereGeometry(0.025, 8, 8);
const jointMat = new THREE.MeshLambertMaterial({color: 0x66ccff});
const joints = [];
for (let i = 0; i < SMPL_PARENTS.length; i++) {
  const m = new THREE.Mesh(jointGeom, jointMat);
  scene.add(m); joints.push(m);
}
const bonePositions = new Float32Array(SMPL_PARENTS.length * 2 * 3);
const boneGeom = new THREE.BufferGeometry();
boneGeom.setAttribute('position', new THREE.BufferAttribute(bonePositions, 3));
const boneLines = new THREE.LineSegments(boneGeom, new THREE.LineBasicMaterial({color: 0xffffff}));
scene.add(boneLines);

function setSkeleton(jointsArr) {
  // jointsArr: [24][7] = (x, y, z, qx, qy, qz, qw). We use only positions.
  for (let i = 0; i < SMPL_PARENTS.length; i++) {
    const j = jointsArr[i];
    if (!j) continue;
    joints[i].position.set(j[0], j[1], j[2]);
  }
  let p = 0;
  for (let i = 0; i < SMPL_PARENTS.length; i++) {
    const par = SMPL_PARENTS[i];
    if (par < 0) continue;
    const a = jointsArr[i], b = jointsArr[par];
    bonePositions[p++] = a[0]; bonePositions[p++] = a[1]; bonePositions[p++] = a[2];
    bonePositions[p++] = b[0]; bonePositions[p++] = b[1]; bonePositions[p++] = b[2];
  }
  boneGeom.setDrawRange(0, p / 3);
  boneGeom.attributes.position.needsUpdate = true;
}
function render() { renderer.render(scene, camera); requestAnimationFrame(render); }
render();

// ---- timing sync ----
function nearestBodyIdx(videoTimeS) {
  if (!chunkData || !chunkData.body_ts || chunkData.body_ts.length === 0) return null;
  const tns = BigInt(Math.round(videoTimeS * 1e9)) + chunkData.video_ts0;
  let lo = 0, hi = chunkData.body_ts.length - 1, best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const tmid = BigInt(chunkData.body_ts[mid]);
    if (tmid < tns) { best = mid; lo = mid + 1; }
    else if (tmid > tns) { hi = mid - 1; }
    else { return mid; }
  }
  return best;
}

vid.addEventListener('timeupdate', () => {
  if (!chunkData) return;
  const i = nearestBodyIdx(vid.currentTime);
  if (i != null) setSkeleton(chunkData.body_joints[i]);
});

// ---- timeline overlays ----
function renderTimeline() {
  timeline.innerHTML = '';
  if (!chunkData || !chunkData.video_ts || chunkData.video_ts.length === 0) return;
  const t0 = BigInt(chunkData.video_ts[0]);
  const tN = BigInt(chunkData.video_ts[chunkData.video_ts.length - 1]);
  const span = Number(tN - t0) / 1e9;
  if (span <= 0) return;
  function add(cls, ts) {
    const off = Number(BigInt(ts) - t0) / 1e9 / span;
    if (off < 0 || off > 1) return;
    const m = document.createElement('div');
    m.className = 'mark ' + cls;
    m.style.left = (off * 100).toFixed(2) + '%';
    timeline.appendChild(m);
  }
  (chunkData.extras || []).forEach((e, i) => {
    const ts = chunkData.extras_ts[i];
    if (e.is_first) add('first', ts);
    if (e.is_last)  add('last',  ts);
    if (e.reward >= 1.0) add('rew', ts);
  });
}
function renderExtrasSummary() {
  if (!chunkData) { extrasEl.textContent = ''; return; }
  const ex = chunkData.extras || [];
  const firsts = ex.filter(e => e.is_first).length;
  const lasts = ex.filter(e => e.is_last).length;
  const rewards = ex.filter(e => e.reward >= 1.0).length;
  extrasEl.innerHTML =
    `<span class="tag running">first: ${firsts}</span>` +
    `<span class="tag paused">last: ${lasts}</span>` +
    `<span class="tag reward">reward: ${rewards}</span>`;
}

// ---- loading ----
async function loadChunk(i) {
  status.textContent = 'loading chunk ' + i + '...';
  const res = await fetch('/api/chunk/' + i + '/data');
  if (!res.ok) { status.textContent = 'load failed: ' + res.status; return; }
  const meta = await res.json();
  chunkData = {
    body_joints: meta.body_joints || [],
    body_ts: (meta.body_ts || []).map(String),
    video_ts: (meta.video_ts || []).map(String),
    video_ts0: BigInt(meta.video_ts && meta.video_ts.length ? meta.video_ts[0] : 0),
    extras: meta.extras || [],
    extras_ts: (meta.extras_ts || []).map(String),
  };
  vid.src = '/api/chunk/' + i + '/rgb.mp4?t=' + Date.now();
  vid.load();
  renderTimeline();
  renderExtrasSummary();
  status.textContent =
    `chunk ${i}: video ${chunkData.video_ts.length}f, ` +
    `body ${chunkData.body_joints.length}f, ` +
    `extras ${chunkData.extras.length}`;
}

document.getElementById('prev').onclick = () => {
  const i = Math.max(0, parseInt(idxInput.value) - 1);
  idxInput.value = i; loadChunk(i);
};
document.getElementById('next').onclick = () => {
  const i = Math.min(chunkCount - 1, parseInt(idxInput.value) + 1);
  idxInput.value = i; loadChunk(i);
};
idxInput.addEventListener('change', () => loadChunk(parseInt(idxInput.value)));

(async function init() {
  const info = await (await fetch('/api/info')).json();
  chunkCount = info.count; countSpan.textContent = String(chunkCount);
  if (chunkCount > 0) loadChunk(0);
  else status.textContent = 'empty dataset';
})();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    reader = None  # set by main()

    def log_message(self, fmt, *args):
        pass  # quiet

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self._send(status, body, "application/json")

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/?"):
                page = HTML_PAGE.replace("__SMPL_PARENTS__", json.dumps(SMPL_PARENTS))
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/info":
                self._send_json({"count": len(self.reader), "spec": dict(self.reader.spec)})
                return
            if self.path.startswith("/api/chunk/"):
                parts = self.path.split("/")
                # /api/chunk/{i}/{kind}
                idx = int(parts[3])
                kind = parts[4].split("?")[0]
                self._handle_chunk(idx, kind)
                return
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._send(500, f"error: {e!r}".encode(), "text/plain")
            raise

    def _handle_chunk(self, idx, kind):
        if kind == "rgb.mp4":
            datapoint = self.reader[idx]
            md = datapoint.get("metadata", {})
            frames = datapoint.get("orin_camera_rgb")
            if frames is None or len(frames) == 0:
                self._send(404, b"no rgb in chunk", "text/plain")
                return
            fps = chunk_fps(md, "orin_camera_rgb")
            mp4 = encode_mp4(frames, fps=fps)
            self._send(200, mp4, "video/mp4")
            return

        if kind == "data":
            datapoint = self.reader[idx]
            md = datapoint.get("metadata", {})
            video_ts = list(map(int, np.asarray(
                md.get("orin_camera_rgb", {}).get("created", [])).astype(np.int64).tolist()))
            body = datapoint.get("pico_xrt")
            body_ts = list(map(int, np.asarray(
                md.get("pico_xrt", {}).get("created", [])).astype(np.int64).tolist()))
            body_joints = []
            if body is not None and "body_joints" in body:
                bj = np.asarray(body["body_joints"])
                body_joints = bj.astype(np.float32).tolist()
            extras_arr = []
            extras_ts = list(map(int, np.asarray(
                md.get("extras", {}).get("created", [])).astype(np.int64).tolist()))
            ex = datapoint.get("extras")
            if ex is not None:
                n = len(extras_ts) or len(ex.get("is_first", []))
                ifs = list(map(bool, np.asarray(ex.get("is_first", [])).astype(bool).tolist()))
                ils = list(map(bool, np.asarray(ex.get("is_last", [])).astype(bool).tolist()))
                rwd = list(map(float, np.asarray(ex.get("reward", [])).astype(float).tolist()))
                for i in range(n):
                    extras_arr.append({
                        "is_first": ifs[i] if i < len(ifs) else False,
                        "is_last": ils[i] if i < len(ils) else False,
                        "reward":  rwd[i] if i < len(rwd) else 0.0,
                    })
            self._send_json({
                "video_ts": video_ts,
                "body_ts": body_ts,
                "body_joints": body_joints,
                "extras": extras_arr,
                "extras_ts": extras_ts,
            })
            return

        self._send(404, b"unknown kind", "text/plain")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="path to the granular dataset directory")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0",
                   help="bind host. Use 127.0.0.1 if you'll only access via SSH-forward.")
    args = p.parse_args()

    with granular.DatasetReader(args.data, granular.decoders) as reader:
        Handler.reader = reader
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"[visualize] dataset={args.data}  chunks={len(reader)}", flush=True)
        print(f"[visualize] serving on http://{args.host}:{args.port}/", flush=True)
        print(f"[visualize] SSH forward example: "
              f"ssh -L {args.port}:localhost:{args.port} <user>@<orin>", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
