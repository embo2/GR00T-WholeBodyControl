"""Cycle the left BrainCo hand closed/open every 2s for 30s via zmq_hand_bridge.

Sends continuously at 20Hz so a single dropped ZMQ frame doesn't miss an edge
(the bridge's SUB uses zmq.CONFLATE=1 and will only consume the latest value).

Requires the bridge to be running:
    python -m brainco_hand.zmq_hand_bridge [...]
"""

import time

import msgpack
import zmq

DURATION_S = 30.0
HALF_PERIOD_S = 2.0
SEND_PERIOD_S = 0.05  # 20Hz


def main() -> None:
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.connect("tcp://localhost:5560")
    time.sleep(0.5)  # PUB slow-joiner handshake

    start = time.monotonic()
    next_toggle = start + HALF_PERIOD_S
    grip = 1.0
    print(f"[{0.0:5.1f}s] left grip -> {grip}")
    while True:
        now = time.monotonic()
        elapsed = now - start
        if elapsed >= DURATION_S:
            break
        if now >= next_toggle:
            grip = 0.0 if grip == 1.0 else 1.0
            next_toggle += HALF_PERIOD_S
            print(f"[{elapsed:5.1f}s] left grip -> {grip}")
        sock.send(b"hand_cmd" + msgpack.packb({"side": "left", "grip": grip}))
        time.sleep(SEND_PERIOD_S)

    # Final explicit open — sent a few times for reliability.
    for _ in range(5):
        sock.send(b"hand_cmd" + msgpack.packb({"side": "left", "grip": 0.0}))
        time.sleep(SEND_PERIOD_S)
    print("done — left hand commanded open")
    sock.close()


if __name__ == "__main__":
    main()
