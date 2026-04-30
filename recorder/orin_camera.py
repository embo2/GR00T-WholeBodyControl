import struct
import time

import numpy as np
import zmq


class OrinCamera:
    def __init__(self, name="orin_camera", host="localhost",
                 rgb_port=5558, depth_port=5563,
                 rgb_topic=b"ego_view", depth_topic=b"depth",
                 rgb_shape=(480, 640), depth_shape=(480, 640)):
        self.name = name
        self.host = host
        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape
        self.host = host
        self.rgb_port = rgb_port
        self.rgb_topic = rgb_topic
        self.depth_port = depth_port
        self.depth_topic = depth_topic
        self.initialized = False

    def init(self):
        self.ctx = zmq.Context()
        rgb_addr = f'{self.host}:{self.rgb_port}'
        depth_addr = f'{self.host}:{self.depth_port}'
        self.rgb = self._make_sub(self.ctx, rgb_addr, self.rgb_topic)
        self.depth = self._make_sub(self.ctx, depth_addr, self.depth_topic)
        self.poller = zmq.Poller()
        self.poller.register(self.rgb, zmq.POLLIN)
        self.poller.register(self.depth, zmq.POLLIN)
        rgb_seen, depth_seen = False, False
        while not (rgb_seen and depth_seen):
            events = dict(self.poller.poll(timeout=1000))
            if self.rgb in events:
                rgb_seen = True
            if self.depth in events:
                depth_seen = True
        self.initialized = True

    @property
    def spec(self):
      #return {f'{self.name}_rgb': 'mp4', f'{self.name}_depth': 'mp4'}
        return {f'{self.name}_rgb': 'mp4'}

    def stream(self):
        assert self.initialized
        rgb_key = f"{self.name}_rgb"
        depth_key = f"{self.name}_depth"
        last_created_rgb = -1
        last_created_depth = -1
        while True:
            events = dict(self.poller.poll(timeout=1000))
            if self.rgb in events:
                self.rgb.recv()
                created = struct.unpack("<Q", self.rgb.recv())[0]
                raw = self.rgb.recv()
                if created != last_created_rgb:
                    last_created_rgb = created
                    image = np.frombuffer(raw, dtype=np.uint8)
                    image = image.reshape((*self.rgb_shape, 3))
                    image = np.ascontiguousarray(image[..., ::-1])
                    metadata = {
                        "timestamp": np.int64(time.time_ns()),
                        "created": np.int64(created),
                    }
                    yield {rgb_key: {'metadata': metadata, 'data': image}}

            # if self.depth in events:
            #     self.depth.recv()
            #     created = struct.unpack("<Q", self.depth.recv())[0]
            #     raw = self.depth.recv()
            #     if created != last_created_depth:
            #         last_created_depth = created
            #         image = np.frombuffer(raw, dtype=np.uint16)
            #         image = image.reshape(self.depth_shape)
            #         metadata = {
            #             "timestamp": np.int64(time.time_ns()),
            #             "created": np.int64(created),
            #         }
            #         yield {depth_key: {'metadata': metadata, 'data': image}}

    def _make_sub(self, ctx, addr, topic):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 100)
        s.connect(f"tcp://{addr}")
        s.setsockopt(zmq.SUBSCRIBE, topic)
        return s


if __name__ == "__main__":
    obj = OrinCamera()
    obj.init()
    print(f"[{obj.name}] connected. (Ctrl+C to stop)", flush=True)
    count = 0
    start = time.perf_counter()
    for d in obj.stream():
        count += 1
        elapsed = time.perf_counter()
        fps = count / (elapsed - start)
        print(f'{fps:.2f} msgs/s')
