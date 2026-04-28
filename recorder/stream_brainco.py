#!/usr/bin/env python3
import argparse
import time

import msgpack
import zmq

TOPIC = b"touch"


def fmt_fingers(fingers):
    if not fingers:
        return "(empty)"
    return "  ".join(
        f"{f['finger'][:3]}:p={f['proximity']:>4d} F={f['force']:>4d}{'*' if f['contact'] else ' '}"
        for f in fingers
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5559)
    p.add_argument("--rate", type=float, default=5.0)
    args = p.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.setsockopt(zmq.SUBSCRIBE, TOPIC)

    period = 1.0 / max(args.rate, 0.1)
    try:
        while True:
            print(f"\n=== {time.strftime('%H:%M:%S')} ===")
            try:
                raw = sock.recv()
                payload = raw[len(TOPIC):] if raw.startswith(TOPIC) else raw
                msg = msgpack.unpackb(payload, raw=False)
                age = (time.time() - msg["ts"]) * 1000
                print(f"hand   age={age:.0f}ms  sdk_read={msg.get('sdk_read_ms', 0):.1f}ms  "
                      f"grip_write={msg.get('grip_write_ms', 0):.1f}ms")
                lg = msg.get("left_grip", -1.0)
                rg = msg.get("right_grip", -1.0)
                print(f"  grip   L={lg:+.2f}  R={rg:+.2f}")
                print(f"  joints L={msg.get('left_joints', []) or '(none)'}")
                print(f"  joints R={msg.get('right_joints', []) or '(none)'}")
                print(f"  touch  L {fmt_fingers(msg.get('left', []))}")
                print(f"  touch  R {fmt_fingers(msg.get('right', []))}")
            except zmq.Again:
                print("touch  (no data)")
            time.sleep(period)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
