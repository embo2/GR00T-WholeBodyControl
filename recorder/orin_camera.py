import struct
import time

import cv2
import numpy as np
import zmq


class OrinCamera:
    def __init__(self, name="orin_camera", host="localhost",
                 rgb_port=5558, depth_port=5563,
                 rgb_topic=b"ego_view", depth_topic=b"depth",
                 depth_shape=(480, 640)):
        self.name = name
        self.host = host
        self.rgb_port = rgb_port
        self.depth_port = depth_port
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.depth_shape = depth_shape
        self._ctx = None
        self._rgb = None
        self._depth = None
        self._poller = None

    def _make_sub(self, port, topic):
        s = self._ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 100)
        s.connect(f"tcp://{self.host}:{port}")
        s.setsockopt(zmq.SUBSCRIBE, topic)
        return s

    def init(self):
        self._ctx = zmq.Context()
        self._rgb = self._make_sub(self.rgb_port, self.rgb_topic)
        self._depth = self._make_sub(self.depth_port, self.depth_topic)
        self._poller = zmq.Poller()
        self._poller.register(self._rgb, zmq.POLLIN)
        self._poller.register(self._depth, zmq.POLLIN)
        rgb_seen = depth_seen = False
        while not (rgb_seen and depth_seen):
            events = dict(self._poller.poll(timeout=1000))
            if self._rgb in events:
                rgb_seen = True
            if self._depth in events:
                depth_seen = True

    def stream(self):
        rgb_key = f"{self.name}_rgb"
        depth_key = f"{self.name}_depth"
        last_ts_rgb = -1
        last_ts_depth = -1
        while True:
            events = dict(self._poller.poll(timeout=1000))
            if self._rgb in events:
                self._rgb.recv()
                ts_ns = struct.unpack("<Q", self._rgb.recv())[0]
                jpeg = self._rgb.recv()
                if ts_ns != last_ts_rgb:
                    last_ts_rgb = ts_ns
                    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                       cv2.IMREAD_COLOR)
                    yield {rgb_key: {
                        "timestamp": np.int64(time.time_ns()),
                        "created": np.int64(ts_ns),
                        "frame": bgr,
                    }}
            if self._depth in events:
                self._depth.recv()
                ts_ns = struct.unpack("<Q", self._depth.recv())[0]
                raw = self._depth.recv()
                if ts_ns != last_ts_depth:
                    last_ts_depth = ts_ns
                    z16 = np.frombuffer(raw, dtype=np.uint16).reshape(self.depth_shape)
                    yield {depth_key: {
                        "timestamp": np.int64(time.time_ns()),
                        "created": np.int64(ts_ns),
                        "frame": z16,
                    }}


if __name__ == "__main__":
    obj = OrinCamera()
    print(f"[{obj.name}] init...", flush=True)
    obj.init()
    print(f"[{obj.name}] connected. (Ctrl+C to stop)", flush=True)
    counts = {}
    starts = {}
    try:
        for d in obj.stream():
            for key, msg in d.items():
                counts[key] = counts.get(key, 0) + 1
                starts.setdefault(key, time.monotonic())
                elapsed = time.monotonic() - starts[key]
                if elapsed >= 2.0:
                    shape = msg["frame"].shape
                    dtype = msg["frame"].dtype
                    print(f"[{key}] {counts[key]} msgs in {elapsed:.2f}s = "
                          f"{counts[key] / elapsed:.1f} msgs/s  "
                          f"frame={shape} {dtype}", flush=True)
                    counts[key] = 0
                    starts[key] = time.monotonic()
    except KeyboardInterrupt:
        pass
