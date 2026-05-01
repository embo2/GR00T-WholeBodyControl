import time

import msgpack
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.connect("tcp://localhost:5566")
time.sleep(0.2)

RATE_HZ = 100
HOLD_S = 1.0
PERIOD = 1.0 / RATE_HZ
TICKS = int(HOLD_S * RATE_HZ)

for i in range(5):
    print(f"cycle {i+1}/5: close")
    for _ in range(TICKS):
        sock.send(b"hand_control" + msgpack.packb({"left": 1.0, "right": 1.0}))
        time.sleep(PERIOD)

    print(f"cycle {i+1}/5: open")
    for _ in range(TICKS):
        sock.send(b"hand_control" + msgpack.packb({"left": 0.0, "right": 0.0}))
        time.sleep(PERIOD)

sock.close()
ctx.term()
