import argparse
import collections
import threading
import time
import queue
import concurrent.futures

import elements
import numpy as np
import granular
import portal
import zmq

from brainco_hand import BrainCoHand
from extras import Extras
from livox_lidar import LivoxLidar
from orin_camera import OrinCamera
from pico_xrt import PicoXrt
from unitree_robot import UnitreeRobot

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


def worker(obj, addr, stopping, fps=30.0):
    print(f"[{obj.name}] init...", flush=True)
    obj.init()
    print(f"[{obj.name}] connected.  (target fps={fps})", flush=True)
    client = portal.Client(addr, name=f'{obj.name}Client')

    pending = queue.Queue()
    def _background():
      while True:
        data = pending.get()
        if data is None:
          break
        client.submit(data)
    thread = portal.Thread(_background, start=True)

    # Drop frames that arrive faster than the target rate. Source iteration
    # still happens at native rate (so the underlying socket/queue keeps
    # draining), we just don't enqueue everything for portal.submit.
    stream = obj.stream()
    period = 1.0 / fps if fps and fps > 0 else 0.0
    last_submit = 0.0
    while not stopping.is_set():
      data = next(stream)
      if period <= 0.0:
        pending.put(data)
        continue
      now = time.monotonic()
      if now - last_submit >= period:
        pending.put(data)
        last_submit = now
    pending.put(None)
    thread.join()


def encode(chunk, spec):
    fn = lambda *xs: np.stack(xs, 0)
    datapoint = {}
    metadata = {}
    for k, v in chunk.items():
      v = elements.tree.map(fn, *v)
      metadata[k] = v['metadata']
      datapoint[k] = v['data']
    datapoint['metadata'] = metadata
    encoders = granular.encoders
    return {k: encoders[spec[k]](v) for k, v in datapoint.items()}


def write_worker(pending, spec, stopping):
  pool = concurrent.futures.ThreadPoolExecutor(32)
  writer = granular.DatasetWriter('data', spec, None)
  futures = collections.deque()
  try:
    while not stopping.is_set():
      while True:
        try:
          chunk = pending.get(timeout=0.1)
        except queue.Empty:
          break
        print('received keys', chunk.keys())
        keys = set(chunk.keys())
        needed = set(spec.keys()) - {'metadata'}
        if keys != needed:
          print('missing keys, skipping...', needed - keys)
          break
        future = pool.submit(encode, chunk, spec)
        futures.append(future)
      while len(futures) > 0 and futures[0].done():
        future = futures.popleft()
        writer.append(future.result())
        print('wrote chunk')
        print(pending.qsize())
  finally:
    writer.close()
  print('[writer] closed; index flushed.', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--addr", type=str, default='localhost:6000')
    p.add_argument("--fps", type=float, default=30.0,
                   help="per-sensor throttle target. 0 disables throttling.")
    args = p.parse_args()

    sensors = [
        #OrinCamera(),
        UnitreeRobot(interface='enP8p1s0'),
        PicoXrt(),
        Extras(),
        # LivoxLidar(),
        # BrainCoHand(),
    ]

    spec = {}
    for sensor in sensors:
      assert all(k not in spec for k in sensor.spec)
      spec.update(sensor.spec)
    assert 'metadata' not in spec
    spec['metadata'] = 'tree'

    sensor_stopping = portal.context.mp.Event()
    procs = []
    for sensor in sensors:
        proc = portal.Process(
            worker, sensor, args.addr, sensor_stopping, args.fps, start=True
        )
        procs.append(proc)

    lock = threading.Lock()
    port = int(args.addr.rsplit(':', 1)[1])
    chunk = collections.defaultdict(list)

    running = False
    hud = HudStatusPublisher()
    hud.publish(running)
    print(f"[main] hud status publisher → tcp://{HUD_HOST}:{HUD_PORT} "
          f"(initial={'RUNNING' if running else 'PAUSED'})", flush=True)

    def submit(data):
      nonlocal running
      if 'extras' in data and data['extras']['data']['is_first']:
        running = True
        hud.publish(True)
        print("[main] recorder RUNNING (extras is_first)", flush=True)
      if not running:
        return
      with lock:
        for k, v in data.items():
          chunk[k].append(v)
      if 'extras' in data and data['extras']['data']['is_last']:
        running = False
        hud.publish(False)
        print("[main] recorder PAUSED (extras is_last)", flush=True)

    server = portal.Server(port)
    server.bind('submit', submit)
    server.start(block=False)

    pending = queue.Queue()
    write_stopping = threading.Event()
    thread = portal.Thread(write_worker, pending, spec, write_stopping, start=True)

    interval = 2.0
    try:
      while True:
        time.sleep(interval)
        with lock:
          if running:
            pending.put(chunk)
          chunk = collections.defaultdict(list)
    except KeyboardInterrupt:
        print("[main] shutting down...", flush=True)
        sensor_stopping.set()
        for proc in procs:
            proc.join()
        server.close()
        write_stopping.set()
        thread.join()
        hud.close()


if __name__ == "__main__":
    main()
