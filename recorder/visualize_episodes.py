"""Episode-level visualizer for a stream2.py session directory.

Splits the recording into episodes by walking `extras/` for `is_first` →
next `is_last` pairs. For each episode:
  - serves an mp4 of `orin_camera_rgb/` frames whose `created` timestamp
    falls within the episode's [first, last] window
  - serves the `extras` time series (reward, is_first, is_last) for the
    same window, plotted as three stacked sparklines.

Layout produced by stream2.py (one datapoint per frame, per modality):
    <session_dir>/
        orin_camera_rgb/   {orin_camera_rgb: 'rgb', metadata: 'tree'}  # raw JPEG
        extras/            {extras: 'tree',        metadata: 'tree'}

Usage:
    cd recorder
    uv run python visualize_episodes.py --session data/session_YYYYMMDD_HHMMSS
    # SSH-forward:  ssh -L 8080:localhost:8080 user@orin
    # then open http://localhost:8080/

References:
    granular dataset format: github.com/danijar/granular
"""

import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import granular
import imageio.v2 as imageio
import numpy as np


def make_decoders():
    decoders = dict(granular.decoders)
    decoders['rgb'] = lambda x: x  # raw JPEG bytes pass-through
    return decoders


def scan_extras(extras_reader):
    """Walk every extras datapoint, return parallel arrays."""
    n = len(extras_reader)
    created = np.empty(n, dtype=np.int64)
    is_first = np.zeros(n, dtype=bool)
    is_last = np.zeros(n, dtype=bool)
    reward = np.zeros(n, dtype=np.float64)
    reward_put_in = np.zeros(n, dtype=np.float64)
    reward_take_out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        dp = extras_reader[i]
        e = dp['extras']
        created[i] = int(dp['metadata']['created'])
        is_first[i] = bool(e.get('is_first', False))
        is_last[i] = bool(e.get('is_last', False))
        reward[i] = float(e.get('reward', 0.0))
        reward_put_in[i] = float(e.get('reward_put_in', 0.0))
        reward_take_out[i] = float(e.get('reward_take_out', 0.0))
    return dict(
        created=created, is_first=is_first, is_last=is_last,
        reward=reward, reward_put_in=reward_put_in,
        reward_take_out=reward_take_out,
    )


def find_episodes(extras):
    """Pair each is_first with the next is_last. Return list of (i_start, i_end).

    Open episodes (is_first with no closing is_last) are skipped — we only
    return complete pairs so the video has a defined endpoint.
    """
    n = len(extras['is_first'])
    eps = []
    i = 0
    while i < n:
        if extras['is_first'][i]:
            j = i + 1
            while j < n and not extras['is_last'][j]:
                # ignore stray is_first re-fires before the close
                j += 1
            if j < n:
                eps.append((i, j))
                i = j + 1
            else:
                # open-ended; drop it
                break
        else:
            i += 1
    return eps


def scan_rgb_timestamps(rgb_reader):
    """Read just the metadata stream, skipping JPEG bytes, for fast filtering."""
    n = len(rgb_reader)
    ts = np.empty(n, dtype=np.int64)
    for i in range(n):
        ts[i] = int(rgb_reader[i, ('metadata',)]['metadata']['created'])
    return ts


def encode_episode_mp4(rgb_reader, frame_indices, fps):
    """Decode JPEG bytes for each index and write an in-memory mp4."""
    if not len(frame_indices):
        return b""
    buf = io.BytesIO()
    with imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                            pixelformat="yuv420p", quality=8) as w:
        for idx in frame_indices:
            jpeg = rgb_reader[int(idx), ('orin_camera_rgb',)]['orin_camera_rgb']
            frame = imageio.imread(io.BytesIO(jpeg))  # H,W,3 RGB
            w.append_data(np.ascontiguousarray(frame))
    return buf.getvalue()


HTML_PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>recorder episode visualizer</title>
<style>
body { margin: 0; background: #111; color: #ddd; font-family: ui-monospace, monospace; }
.bar { display: flex; gap: 12px; padding: 10px 14px; align-items: center; border-bottom: 1px solid #333; }
button { background: #222; color: #ddd; border: 1px solid #444; padding: 6px 12px; cursor: pointer; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
input[type=number] { width: 64px; background: #222; color: #ddd; border: 1px solid #444; padding: 4px; }
.row { display: flex; padding: 12px; gap: 12px; flex-wrap: wrap; }
.panel { background: #1a1a1a; border: 1px solid #333; padding: 10px; }
video { width: 640px; height: 480px; background: #000; display: block; }
canvas.ts { width: 640px; height: 80px; display: block; background: #000; }
.label { font-size: 11px; color: #888; padding: 4px 0 2px; }
#status { color: #888; font-size: 12px; }
.kpis { display: flex; gap: 16px; font-size: 12px; color: #aaa; padding: 6px 0; }
.kpis b { color: #fff; }
</style>

<div class="bar">
  <button id="prev">◀ prev</button>
  episode <input id="idx" type="number" min="0" value="0">
  / <span id="count">?</span>
  <button id="next">next ▶</button>
  <span id="status"></span>
</div>

<div class="row">
  <div class="panel">
    <video id="vid" controls muted playsinline></video>
    <div class="kpis" id="kpis"></div>
  </div>
  <div class="panel">
    <div class="label">reward</div>
    <canvas class="ts" id="ts_reward"  width="640" height="80"></canvas>
    <div class="label">is_first</div>
    <canvas class="ts" id="ts_first"   width="640" height="80"></canvas>
    <div class="label">is_last</div>
    <canvas class="ts" id="ts_last"    width="640" height="80"></canvas>
  </div>
</div>

<script>
let episodeCount = 0;
let episode = null;  // {duration_s, ts_rel, reward, is_first, is_last, ...}

const vid = document.getElementById('vid');
const idxInput = document.getElementById('idx');
const countSpan = document.getElementById('count');
const status = document.getElementById('status');
const kpisEl = document.getElementById('kpis');

function drawTrace(canvas, ts, ys, color, mode) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#333';
  ctx.beginPath();
  ctx.moveTo(0, H - 1); ctx.lineTo(W, H - 1);
  ctx.moveTo(0, 0); ctx.lineTo(0, H);
  ctx.stroke();
  if (!ts || !ts.length) return;
  const t0 = ts[0], tN = ts[ts.length - 1];
  const span = Math.max(1e-6, tN - t0);
  let ymin = Infinity, ymax = -Infinity;
  for (const v of ys) { if (v < ymin) ymin = v; if (v > ymax) ymax = v; }
  if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
  if (ymax - ymin < 1e-9) { ymax = ymin + 1; }
  const xs = (i) => ((ts[i] - t0) / span) * (W - 2) + 1;
  const ysc = (v) => H - 2 - ((v - ymin) / (ymax - ymin)) * (H - 4);

  if (mode === 'bool') {
    ctx.fillStyle = color;
    for (let i = 0; i < ys.length; i++) {
      if (ys[i]) {
        const x = xs(i);
        ctx.fillRect(x - 1, 2, 2, H - 4);
      }
    }
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xs(0), ysc(ys[0]));
    for (let i = 1; i < ys.length; i++) ctx.lineTo(xs(i), ysc(ys[i]));
    ctx.stroke();
  }

  ctx.fillStyle = '#888'; ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(ymin.toFixed(2), 4, H - 4);
  ctx.fillText(ymax.toFixed(2), 4, 12);
}

function renderEpisode() {
  if (!episode) return;
  drawTrace(document.getElementById('ts_reward'),
            episode.ts_rel, episode.reward, '#dd5', 'line');
  drawTrace(document.getElementById('ts_first'),
            episode.ts_rel, episode.is_first, '#5d5', 'bool');
  drawTrace(document.getElementById('ts_last'),
            episode.ts_rel, episode.is_last, '#d55', 'bool');
  const rewardCount = episode.reward.filter(r => r >= 1.0).length;
  kpisEl.innerHTML =
    `frames: <b>${episode.frame_count}</b>` +
    ` &middot; duration: <b>${episode.duration_s.toFixed(2)}s</b>` +
    ` &middot; extras: <b>${episode.ts_rel.length}</b>` +
    ` &middot; reward&ge;1 ticks: <b>${rewardCount}</b>` +
    ` &middot; max reward: <b>${Math.max(...episode.reward, 0).toFixed(2)}</b>`;
}

async function loadEpisode(i) {
  status.textContent = `loading episode ${i}...`;
  const res = await fetch('/api/episode/' + i + '/data');
  if (!res.ok) { status.textContent = 'load failed: ' + res.status; return; }
  episode = await res.json();
  vid.src = '/api/episode/' + i + '/video.mp4?t=' + Date.now();
  vid.load();
  renderEpisode();
  document.getElementById('prev').disabled = (i <= 0);
  document.getElementById('next').disabled = (i >= episodeCount - 1);
  status.textContent = `episode ${i} of ${episodeCount - 1}`;
}

document.getElementById('prev').onclick = () => {
  const i = Math.max(0, parseInt(idxInput.value) - 1);
  idxInput.value = i; loadEpisode(i);
};
document.getElementById('next').onclick = () => {
  const i = Math.min(episodeCount - 1, parseInt(idxInput.value) + 1);
  idxInput.value = i; loadEpisode(i);
};
idxInput.addEventListener('change', () => loadEpisode(parseInt(idxInput.value)));

(async function init() {
  const info = await (await fetch('/api/info')).json();
  episodeCount = info.episode_count;
  countSpan.textContent = String(episodeCount - 1);
  document.title = `episodes: ${episodeCount} — ${info.session}`;
  if (episodeCount > 0) loadEpisode(0);
  else status.textContent = 'no complete episodes in this session';
})();
</script>
"""


class State:
    extras_reader = None
    rgb_reader = None
    extras_data = None    # dict from scan_extras
    rgb_ts = None         # int64[N]
    episodes = None       # list[(i_start, i_end)]
    session_name = ''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

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
                self._send(200, HTML_PAGE.encode("utf-8"),
                           "text/html; charset=utf-8")
                return
            if self.path == "/api/info":
                self._send_json({
                    "session": State.session_name,
                    "episode_count": len(State.episodes),
                })
                return
            if self.path.startswith("/api/episode/"):
                parts = self.path.split("/")
                idx = int(parts[3])
                kind = parts[4].split("?")[0]
                self._handle_episode(idx, kind)
                return
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._send(500, f"error: {e!r}".encode(), "text/plain")
            raise

    def _handle_episode(self, idx, kind):
        if not (0 <= idx < len(State.episodes)):
            self._send(404, b"episode out of range", "text/plain")
            return
        i_start, i_end = State.episodes[idx]
        ex = State.extras_data
        t_start = int(ex['created'][i_start])
        t_end = int(ex['created'][i_end])

        if kind == "data":
            ts_ns = ex['created'][i_start:i_end + 1].astype(np.int64)
            ts_rel = (ts_ns - ts_ns[0]).astype(np.float64) / 1e9
            mask_rgb = (State.rgb_ts >= t_start) & (State.rgb_ts <= t_end)
            n_frames = int(mask_rgb.sum())
            self._send_json({
                "duration_s": float((t_end - t_start) / 1e9),
                "frame_count": n_frames,
                "ts_rel": ts_rel.tolist(),
                "reward": ex['reward'][i_start:i_end + 1].tolist(),
                "reward_put_in":  ex['reward_put_in'][i_start:i_end + 1].tolist(),
                "reward_take_out": ex['reward_take_out'][i_start:i_end + 1].tolist(),
                "is_first": ex['is_first'][i_start:i_end + 1].astype(int).tolist(),
                "is_last":  ex['is_last'][i_start:i_end + 1].astype(int).tolist(),
            })
            return

        if kind == "video.mp4":
            mask = (State.rgb_ts >= t_start) & (State.rgb_ts <= t_end)
            frame_indices = np.where(mask)[0]
            if len(frame_indices) >= 2:
                deltas = np.diff(State.rgb_ts[frame_indices].astype(np.int64))
                pos = deltas[deltas > 0]
                fps = float(1e9 / np.median(pos)) if len(pos) else 30.0
            else:
                fps = 30.0
            mp4 = encode_episode_mp4(State.rgb_reader, frame_indices, fps)
            if not mp4:
                self._send(404, b"no frames in episode window", "text/plain")
                return
            self._send(200, mp4, "video/mp4")
            return

        self._send(404, b"unknown kind", "text/plain")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True,
                   help="path to a stream2.py session_* directory")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0",
                   help="bind host. Use 127.0.0.1 if only via SSH-forward.")
    args = p.parse_args()

    decoders = make_decoders()
    extras_path = os.path.join(args.session, "extras")
    rgb_path = os.path.join(args.session, "orin_camera_rgb")

    with granular.DatasetReader(extras_path, decoders) as ex_reader, \
         granular.DatasetReader(rgb_path, decoders) as rgb_reader:
        State.extras_reader = ex_reader
        State.rgb_reader = rgb_reader
        State.session_name = os.path.basename(os.path.normpath(args.session))

        print(f"[viz] scanning extras ({len(ex_reader)} datapoints)...", flush=True)
        State.extras_data = scan_extras(ex_reader)
        print(f"[viz] scanning rgb timestamps ({len(rgb_reader)} datapoints)...",
              flush=True)
        State.rgb_ts = scan_rgb_timestamps(rgb_reader)
        State.episodes = find_episodes(State.extras_data)
        print(f"[viz] {len(State.episodes)} complete episodes "
              f"(is_first → is_last) found in {State.session_name}", flush=True)
        for i, (a, b) in enumerate(State.episodes):
            dur = (State.extras_data['created'][b]
                   - State.extras_data['created'][a]) / 1e9
            print(f"  ep {i}: extras [{a}, {b}]  duration {dur:.2f}s", flush=True)

        if not State.episodes:
            print("[viz] no complete episodes — nothing to serve", flush=True)
            return

        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"[viz] serving on http://{args.host}:{args.port}/", flush=True)
        print(f"[viz] SSH forward: ssh -L {args.port}:localhost:{args.port} "
              f"<user>@<orin>", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
