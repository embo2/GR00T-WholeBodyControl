import argparse
import collections
import signal
import threading
import time
import queue

import elements
import numpy as np
import portal

from brainco_hand import BrainCoHand
from livox_lidar import LivoxLidar
from orin_camera import OrinCamera
from pico_xrt import PicoXrt
from unitree_robot import UnitreeRobot

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
    def submit(data):
      with lock:
        for k, v in data.items():
          chunk[k].append(v)
    server = portal.Server(port)
    server.bind('submit', submit)
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


if __name__ == "__main__":
    main()
