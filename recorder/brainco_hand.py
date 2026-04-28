import time

import msgpack
import zmq


class BrainCoHand:
    def __init__(self, name="brainco_hand", host="localhost",
                 port=5559, topic=b"touch"):
        self.name = name
        self.host = host
        self.port = port
        self.topic = topic
        self._sock = None
        self._poller = None

    def init(self):
        ctx = zmq.Context()
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 100)
        self._sock.connect(f"tcp://{self.host}:{self.port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        while not dict(self._poller.poll(timeout=1000)):
            pass

    def stream(self):
        while True:
            events = dict(self._poller.poll(timeout=1000))
            if self._sock not in events:
                continue
            raw = self._sock.recv()
            payload = raw[len(self.topic):] if raw.startswith(self.topic) else raw
            msg = msgpack.unpackb(payload, raw=False)
            msg["t_realtime"] = time.time()
            msg["t_monotonic"] = time.monotonic()
            yield {self.name: msg}


if __name__ == "__main__":
    obj = BrainCoHand()
    print(f"[{obj.name}] init...", flush=True)
    obj.init()
    print(f"[{obj.name}] connected. (Ctrl+C to stop)", flush=True)
    counts = {}
    starts = {}
    try:
        for d in obj.stream():
            for key, _ in d.items():
                counts[key] = counts.get(key, 0) + 1
                starts.setdefault(key, time.monotonic())
                elapsed = time.monotonic() - starts[key]
                if elapsed >= 2.0:
                    print(f"[{key}] {counts[key]} msgs in {elapsed:.2f}s = "
                          f"{counts[key] / elapsed:.1f} msgs/s", flush=True)
                    counts[key] = 0
                    starts[key] = time.monotonic()
    except KeyboardInterrupt:
        pass
