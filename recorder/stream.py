import argparse
import collections
import signal
import threading
import time
import queue

import elements
import numpy as np
import portal
import zmq

from brainco_hand import BrainCoHand
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

# XRT
# export CMAKE_PREFIX_PATH="$(uv run python -m pybind11 --cmakedir)"
# uv pip install pybind11 cmake setuptools
# uv pip install --no-build-isolation -e ../external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64


def worker(obj, addr):
    print(f"[{obj.name}] init...", flush=True)
    obj.init()
    print(f"[{obj.name}] connected.", flush=True)
    client = portal.Client(addr, name=f'{obj.name}Client')

    pending = queue.Queue()
    def _background():
      while True:
        client.submit(pending.get())
    background = portal.Thread(_background, start=True)

    for data in obj.stream():
      pending.put(data)

    background.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--addr", type=str, default='localhost:6000')
    args = p.parse_args()

    sensors = [
        OrinCamera(),
        UnitreeRobot(interface='enP8p1s0'),
        PicoXrt(),
        # LivoxLidar(),
        # BrainCoHand(),
    ]

    procs = []
    for sensor in sensors:
        proc = portal.Process(worker, sensor, args.addr, start=True)
        procs.append(proc)

    lock = threading.Lock()
    port = int(args.addr.rsplit(':', 1)[1])
    chunk = collections.defaultdict(list)

    running = False
    hud = HudStatusPublisher()
    hud.publish(running)
    print(f"[main] hud status publisher → tcp://{HUD_HOST}:{HUD_PORT} "
          f"(initial={'RUNNING' if running else 'PAUSED'})", flush=True)

    def start():
      nonlocal running
      running = True
      hud.publish(running)
      print("[main] recorder RUNNING", flush=True)

    def pause():
      nonlocal running
      running = False
      hud.publish(running)
      print("[main] recorder PAUSED", flush=True)

    def submit(data):
      if not running:
        return
      with lock:
        for k, v in data.items():
          chunk[k].append(v)

    server = portal.Server(port)
    server.bind('submit', submit)
    server.bind('start', start)
    server.bind('pause', pause)
    server.start(block=False)

    interval = 2.0
    try:
      while True:
        time.sleep(interval)
        with lock:
          print({k: len(v) / interval for k, v in chunk.items()})
          chunk.clear()
    finally:
        print("[main] shutting down...", flush=True)
        for proc in procs:
            proc.kill()
        for proc in procs:
            proc.join(timeout=2.0)
        server.close()
        hud.close()


if __name__ == "__main__":
    main()
