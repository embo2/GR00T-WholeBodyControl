"""Standalone-per-sensor recorder, per-frame writes.

Each sensor in the `sensors` list runs in its own process. The worker
opens one granular DatasetWriter per stream-key (so per-frame appends
work even when modalities yield at different rates), and writes one
datapoint per yield — no chunking.

The 'rgb' spec type means "raw bytes pass-through" — the encoder is
identity, so OrinCamera's JPEG bytes go straight to disk without
re-encoding.

The Extras worker is the only one that mutates the running flag and
drives the headset HUD.

Run:
    cd recorder
    uv run python stream2.py --out data --fps 30
"""

import argparse
import os
import threading
import time

import granular
import granular.formats
import numpy as np
import portal
import zmq

from brainco_hand import BrainCoHand
from extras import Extras
from livox_lidar import LivoxLidar
from orin_camera import OrinCamera
from pico_xrt import PicoXrt
from unitree_robot import UnitreeRobot


# granular.formats.encode_tree falls through to msgpack.packb on np.generic
# scalars (np.int64 / np.float64 / np.bool_ / ...), which can't serialize
# them. Walk the structure first and convert any numpy scalar to its
# Python equivalent via .item(); ndarrays still take the special path
# inside the original encode_tree because they're not np.generic.
_orig_encode_tree = granular.formats.encode_tree


def _to_native(xs):
    if isinstance(xs, (list, tuple)):
        return [_to_native(x) for x in xs]
    if isinstance(xs, dict):
        return {k: _to_native(v) for k, v in xs.items()}
    if isinstance(xs, np.generic) and not isinstance(xs, np.ndarray):
        return xs.item()
    return xs


def _encode_tree_safe(value):
    return _orig_encode_tree(_to_native(value))


granular.formats.encode_tree = _encode_tree_safe
granular.encoders['tree'] = _encode_tree_safe


HUD_HOST = "localhost"
HUD_PORT = 5570


class HudStatusPublisher:
    """ZMQ PUB connected to the OrinVideoSender hud SUB (port 5570).

    Sends "RUNNING" or "PAUSED" frames. A 1 Hz heartbeat re-sends the last
    state so the C++ HUD recovers from ZMQ slow-joiner drops.
    """

    def __init__(self, host=HUD_HOST, port=HUD_PORT, heartbeat_s=1.0):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self._running = False
        self._stop = threading.Event()
        threading.Thread(target=self._heartbeat,
                         args=(heartbeat_s,), daemon=True).start()

    def publish(self, running):
        self._running = bool(running)
        self._send()

    def reward(self):
        # One-shot: tells the C++ HUD to flash a green border. The flash
        # duration is owned by the C++ side (currently 1000 ms).
        try:
            self._sock.send(b"REWARD", flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _send(self):
        msg = b"RUNNING" if self._running else b"PAUSED"
        try:
            self._sock.send(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _heartbeat(self, period):
        while not self._stop.wait(period):
            self._send()

    def close(self):
        self._stop.set()
        self._sock.close(0)
        self._ctx.term()


def make_encoders():
    """granular encoder map, with 'rgb' = identity (raw bytes pass-through)
    and an identity-style decoder so reads come back as the same bytes."""
    encoders = dict(granular.encoders)
    encoders['rgb'] = lambda x: x  # already bytes, ship as-is
    return encoders


def worker(sensor, running_event, stopping_event, session_dir, fps):
    print(f'[{sensor.name}] init...', flush=True)
    sensor.init()
    print(f'[{sensor.name}] connected. (target fps={fps})', flush=True)

    encoders = make_encoders()

    writers = {}
    for key, dtype in sensor.spec.items():
        sub_spec = {key: dtype, 'metadata': 'tree'}
        out = os.path.join(session_dir, key)
        writers[key] = granular.DatasetWriter(out, sub_spec, encoders)
        print(f'[{sensor.name}] dataset {out}/  spec={sub_spec}', flush=True)

    # Extras drives the running flag and HUD; everyone else just reads it.
    is_extras = (sensor.name == 'extras')
    hud = HudStatusPublisher() if is_extras else None
    if hud:
        hud.publish(running_event.is_set())

    period = 1.0 / fps if fps and fps > 0 else 0.0
    last_submit = 0.0

    try:
        for item in sensor.stream():
            if stopping_event.is_set():
                break

            # Every shipped sensor yields {key: {'metadata': ..., 'data': ...}}
            # with one or more keys per yield (camera yields one key at a time).
            assert isinstance(item, dict), (sensor.name, type(item))

            for key, payload in item.items():
                metadata = payload['metadata']
                data = payload['data']

                # Extras handles the running edge transitions before the gate.
                if is_extras and isinstance(data, dict):
                    if data.get('is_first') and not running_event.is_set():
                        running_event.set()
                        if hud:
                            hud.publish(True)
                        print(f'[{sensor.name}] recording RUNNING (is_first)',
                              flush=True)
                    # Log reward edges regardless of running state — useful
                    # to confirm the both-grips combo is actually firing.
                    # Also tell the headset HUD to flash green for ~1s.
                    if float(data.get('reward', 0.0)) >= 1.0:
                        print(f'[{sensor.name}] reward=1.0', flush=True)
                        if hud:
                            hud.reward()

                if not running_event.is_set():
                    continue

                # is_last frames must always land in the dataset, even if
                # the throttle would otherwise drop this tick.
                is_last = bool(is_extras and isinstance(data, dict)
                               and data.get('is_last'))

                if period > 0 and not is_last:
                    now = time.monotonic()
                    if now - last_submit < period:
                        continue
                    last_submit = now

                writers[key].append({key: data, 'metadata': metadata})

                if is_last:
                    running_event.clear()
                    if hud:
                        hud.publish(False)
                    print(f'[{sensor.name}] recording PAUSED (is_last)',
                          flush=True)
    finally:
        for w in writers.values():
            w.close()
        if hud:
            hud.close()
        print(f'[{sensor.name}] writers closed; index flushed.', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data', help='base output directory')
    p.add_argument('--fps', type=float, default=30.0,
                   help='per-sensor throttle target. 0 disables throttling.')
    args = p.parse_args()

    sensors = [
        OrinCamera(),
        UnitreeRobot(interface='enP8p1s0'),
        PicoXrt(),
        Extras(),
        # LivoxLidar(),
        # BrainCoHand(),
    ]

    session_dir = os.path.join(args.out, time.strftime('session_%Y%m%d_%H%M%S'))
    os.makedirs(session_dir, exist_ok=True)
    print(f'[main] writing to {session_dir}/', flush=True)

    running_event = portal.context.mp.Event()
    stopping_event = portal.context.mp.Event()

    procs = []
    for sensor in sensors:
        proc = portal.Process(
            worker, sensor, running_event, stopping_event,
            session_dir, args.fps, start=True,
        )
        procs.append(proc)
    print(f'[main] spawned {len(procs)} sensor processes', flush=True)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('[main] shutting down...', flush=True)
        stopping_event.set()
        for proc in procs:
            proc.join(timeout=15.0)
        print('[main] done', flush=True)


if __name__ == '__main__':
    main()
