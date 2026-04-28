#!/usr/bin/env python3
"""
Standalone data stream monitor for G1 teleop.

Subscribes to all available data channels and prints received modalities.
No data is saved — this is for verifying connectivity and inspecting streams.

Channels:
  ZMQ  tcp://<orin>:5558  topic "ego_view"       RGB JPEG frames
  ZMQ  tcp://<orin>:5563  topic "depth"           Depth Z16 frames
  ZMQ  tcp://<pico>:5556  topic "pose"            Pico teleop data
  DDS  rt/lowstate                                Robot joint state (500 Hz)
  DDS  rt/secondary_imu                           Torso IMU
  DDS  rt/dex3/left/state                         Left hand state
  DDS  rt/dex3/right/state                        Right hand state

Usage:
  python stream_monitor.py
  python stream_monitor.py --orin-host 192.168.123.164 --pico-host localhost
  python stream_monitor.py --no-dds   # skip DDS (if unitree SDK not available)
"""

import argparse
import json
import signal
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import zmq

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_

    HAS_DDS = True
except ImportError:
    HAS_DDS = False


HEADER_SIZE = 1024
DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": bool,
}


class StreamStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = defaultdict(int)
        self.bytes = defaultdict(int)
        self.last_time = defaultdict(float)

    def record(self, modality: str, nbytes: int):
        now = time.time()
        with self.lock:
            self.counts[modality] += 1
            self.bytes[modality] += nbytes
            self.last_time[modality] = now

    def report(self):
        with self.lock:
            now = time.time()
            entries = []
            for mod in sorted(self.counts.keys()):
                age = now - self.last_time[mod]
                status = "LIVE" if age < 2.0 else f"stale ({age:.0f}s)"
                entries.append(
                    f"  {mod:<30s}  frames={self.counts[mod]:<8d}  "
                    f"bytes={self.bytes[mod]:<12d}  [{status}]"
                )
            return "\n".join(entries) if entries else "  (no data received yet)"

    def reset_counts(self):
        with self.lock:
            self.counts.clear()
            self.bytes.clear()


def unpack_pose_message(msg: bytes) -> dict:
    """Unpack a pose ZMQ message into a dict of numpy arrays."""
    topic_prefix = b"pose"
    if not msg.startswith(topic_prefix):
        return {}
    payload = msg[len(topic_prefix):]
    header_raw = payload[:HEADER_SIZE].rstrip(b"\x00")
    header = json.loads(header_raw)
    binary = payload[HEADER_SIZE:]

    data = {}
    offset = 0
    for field in header.get("fields", []):
        dt = DTYPE_MAP.get(field["dtype"], np.float32)
        shape = field["shape"]
        count = 1
        for s in shape:
            count *= s
        nbytes = count * np.dtype(dt).itemsize
        data[field["name"]] = np.frombuffer(
            binary[offset : offset + nbytes], dtype=dt
        ).reshape(shape)
        offset += nbytes
    return data


def probe_zmq_channel(address: str, topic: bytes, timeout_ms: int = 2000) -> bool:
    """Check if a ZMQ PUB socket is reachable and sending data."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(address)
    sock.setsockopt(zmq.SUBSCRIBE, topic)
    try:
        sock.recv()
        return True
    except zmq.Again:
        return False
    finally:
        sock.close()
        ctx.term()


def zmq_ego_rgb_thread(address: str, stats: StreamStats, stop: threading.Event):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.connect(address)
    sock.setsockopt(zmq.SUBSCRIBE, b"ego_view")

    while not stop.is_set():
        try:
            _topic = sock.recv()
            ts_bytes = sock.recv()
            jpeg_bytes = sock.recv()
            total = len(_topic) + len(ts_bytes) + len(jpeg_bytes)
            stats.record("rgb/ego_view", total)
        except zmq.Again:
            continue
    sock.close()
    ctx.term()


def zmq_depth_thread(address: str, stats: StreamStats, stop: threading.Event):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.connect(address)
    sock.setsockopt(zmq.SUBSCRIBE, b"depth")

    while not stop.is_set():
        try:
            _topic = sock.recv()
            ts_bytes = sock.recv()
            raw = sock.recv()
            total = len(_topic) + len(ts_bytes) + len(raw)
            stats.record("depth/z16", total)
        except zmq.Again:
            continue
    sock.close()
    ctx.term()


def zmq_pico_thread(address: str, stats: StreamStats, stop: threading.Event):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.connect(address)
    sock.setsockopt(zmq.SUBSCRIBE, b"pose")

    while not stop.is_set():
        try:
            msg = sock.recv()
            data = unpack_pose_message(msg)
            for key, arr in data.items():
                stats.record(f"pico/{key}", arr.nbytes)
        except zmq.Again:
            continue
    sock.close()
    ctx.term()


def dds_thread(network_interface: str, stats: StreamStats, stop: threading.Event):
    if not HAS_DDS:
        return

    ChannelFactoryInitialize(0, network_interface)

    low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    imu_torso_sub = ChannelSubscriber("rt/secondary_imu", IMUState_)

    def on_low_state(msg):
        stats.record("robot/lowstate", sys.getsizeof(msg))

    def on_imu_torso(msg):
        stats.record("robot/imu_torso", sys.getsizeof(msg))

    low_state_sub.Init(on_low_state, 1)
    imu_torso_sub.Init(on_imu_torso, 1)

    try:
        from unitree_sdk2py.idl.hg.msg.dds_ import HandState_

        left_hand_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        right_hand_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)

        def on_left_hand(msg):
            stats.record("robot/left_hand", sys.getsizeof(msg))

        def on_right_hand(msg):
            stats.record("robot/right_hand", sys.getsizeof(msg))

        left_hand_sub.Init(on_left_hand, 1)
        right_hand_sub.Init(on_right_hand, 1)
    except ImportError:
        pass

    while not stop.is_set():
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Monitor teleop data streams")
    parser.add_argument("--orin-host", default="localhost",
                        help="XRoboToolkit Orin hostname/IP (default: localhost)")
    parser.add_argument("--pico-host", default="localhost",
                        help="Pico manager hostname/IP (default: localhost)")
    parser.add_argument("--rgb-port", type=int, default=5558)
    parser.add_argument("--depth-port", type=int, default=5563)
    parser.add_argument("--pico-port", type=int, default=5556)
    parser.add_argument("--dds-interface", default="",
                        help="Network interface for DDS (e.g. enP8p1s0). Empty = default.")
    parser.add_argument("--probe-timeout", type=int, default=3000,
                        help="Timeout in ms when probing channels (default: 3000)")
    args = parser.parse_args()

    rgb_addr = f"tcp://{args.orin_host}:{args.rgb_port}"
    depth_addr = f"tcp://{args.orin_host}:{args.depth_port}"
    pico_addr = f"tcp://{args.pico_host}:{args.pico_port}"

    print("=" * 60)
    print("  Teleop Stream Monitor")
    print("=" * 60)

    # --- Probe channels ---
    print("\nProbing channels...\n")

    channels = [
        ("RGB (ego_view)", rgb_addr, b"ego_view"),
        ("Depth (z16)", depth_addr, b"depth"),
        ("Pico (pose)", pico_addr, b"pose"),
    ]

    available = {}
    for name, addr, topic in channels:
        ok = probe_zmq_channel(addr, topic, timeout_ms=args.probe_timeout)
        status = "OK" if ok else "NOT FOUND"
        available[name] = ok
        print(f"  {name:<25s} {addr:<35s} [{status}]")

    if not HAS_DDS:
        print(f"  {'DDS (proprioception)':<25s} {'rt/lowstate, rt/secondary_imu':<35s} [skipped (unitree_sdk2py not installed)]")
        available["DDS"] = False
    else:
        print(f"  {'DDS (proprioception)':<25s} {'rt/lowstate, rt/secondary_imu':<35s} [will attempt]")
        available["DDS"] = True

    active_count = sum(1 for v in available.values() if v)
    if active_count == 0:
        print("\nNo channels available. Make sure the teleop processes are running.")
        print("Continuing anyway — will wait for data...\n")
    else:
        print(f"\n{active_count} channel(s) detected. Starting subscribers...\n")

    # --- Start subscriber threads ---
    stats = StreamStats()
    stop = threading.Event()
    threads = []

    t = threading.Thread(target=zmq_ego_rgb_thread, args=(rgb_addr, stats, stop), daemon=True)
    t.start()
    threads.append(t)

    t = threading.Thread(target=zmq_depth_thread, args=(depth_addr, stats, stop), daemon=True)
    t.start()
    threads.append(t)

    t = threading.Thread(target=zmq_pico_thread, args=(pico_addr, stats, stop), daemon=True)
    t.start()
    threads.append(t)

    if available.get("DDS"):
        t = threading.Thread(
            target=dds_thread, args=(args.dds_interface, stats, stop), daemon=True
        )
        t.start()
        threads.append(t)

    # --- Print loop ---
    def handle_sigint(sig, frame):
        stop.set()

    signal.signal(signal.SIGINT, handle_sigint)

    print("Listening... (Ctrl+C to stop)\n")
    try:
        while not stop.is_set():
            time.sleep(2.0)
            print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            print(stats.report())
    except KeyboardInterrupt:
        pass

    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    print("\nStopped.")


if __name__ == "__main__":
    main()
