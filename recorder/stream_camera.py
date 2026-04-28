#!/usr/bin/env python3
import argparse
import struct
import time

import cv2
import numpy as np
import zmq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orin-host", default="localhost")
    p.add_argument("--rgb-port", type=int, default=5558)
    p.add_argument("--depth-port", type=int, default=5563)
    p.add_argument("--depth-shape", default="480x640", help="HxW of the Z16 depth frame")
    p.add_argument("--rate", type=float, default=10.0)
    args = p.parse_args()

    h, w = (int(x) for x in args.depth_shape.split("x"))

    ctx = zmq.Context()

    def make_sub(port, topic):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.RCVTIMEO, 100)
        s.connect(f"tcp://{args.orin_host}:{port}")
        s.setsockopt(zmq.SUBSCRIBE, topic)
        return s

    rgb = make_sub(args.rgb_port, b"ego_view")
    depth = make_sub(args.depth_port, b"depth")

    period = 1.0 / max(args.rate, 0.1)
    try:
        while True:
            print(f"\n=== {time.strftime('%H:%M:%S')} ===")
            try:
                rgb.recv()
                ts = struct.unpack("<Q", rgb.recv())[0]
                jpeg = rgb.recv()
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                print(f"rgb   t={ts/1e9:.3f}  shape={img.shape}  jpeg={len(jpeg)}B")
            except zmq.Again:
                print("rgb   (no data)")

            try:
                depth.recv()
                ts = struct.unpack("<Q", depth.recv())[0]
                raw = depth.recv()
                d = np.frombuffer(raw, np.uint16).reshape(h, w)
                print(f"depth t={ts/1e9:.3f}  shape={d.shape}  min={d.min()} max={d.max()} mean={d.mean():.0f}")
            except zmq.Again:
                print("depth (no data)")

            time.sleep(period)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
