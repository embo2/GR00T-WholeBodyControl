import time

import msgpack
import numpy as np
import zmq

f64 = np.float64


class Extras:
    TOPIC = b"extra"

    def __init__(self, name="extras", host="localhost", port=5572):
        self.name = name
        self.host = host
        self.port = port
        self._ctx = None
        self._sock = None
        self._poller = None

    @property
    def spec(self):
        return {self.name: 'tree'}

    def init(self):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 100)
        self._sock.connect(f"tcp://{self.host}:{self.port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, self.TOPIC)
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        # Block until we see at least one frame so callers know the
        # publisher is alive (matches OrinCamera/PicoXrt behaviour).
        while not dict(self._poller.poll(timeout=1000)):
            pass

    def stream(self):
        topic_len = len(self.TOPIC)
        while True:
            events = dict(self._poller.poll(timeout=1000))
            if self._sock not in events:
                continue
            raw = self._sock.recv()
            payload = raw[topic_len:] if raw.startswith(self.TOPIC) else raw
            msg = msgpack.unpackb(payload, raw=False)
            metadata = {
                "timestamp": np.int64(time.time_ns()),
                "created": np.int64(msg["created"]),
            }
            data = {
                "is_first": bool(msg["is_first"]),
                "is_last": bool(msg["is_last"]),
                "reward": f64(msg["reward"]),
                "reward_put_in": f64(msg.get("reward_put_in", 0.0)),
                "reward_take_out": f64(msg.get("reward_take_out", 0.0)),
            }
            yield {self.name: {'metadata': metadata, 'data': data}}


if __name__ == "__main__":
    obj = Extras()
    obj.init()
    print(f"[{obj.name}] connected. (Ctrl+C to stop)", flush=True)
    count = 0
    start = time.perf_counter()
    for d in obj.stream():
        count += 1
        elapsed = time.perf_counter()
        fps = count / (elapsed - start)
        print(f'{fps:.2f} msgs/s')
