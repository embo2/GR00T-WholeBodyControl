import queue as _queue
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

NUM_MOTORS = 29


class UnitreeRobot:
    def __init__(self, name="unitree_robot", interface="",
                 topic="rt/lowstate", domain_id=0,
                 queue_max=10000, sub_queue_len=100):
        self.name = name
        self.interface = interface
        self.topic = topic
        self.domain_id = domain_id
        self.queue_max = queue_max
        self.sub_queue_len = sub_queue_len
        self._q = None
        self._sub = None

    def _on_lowstate(self, msg):
        motors = msg.motor_state[:NUM_MOTORS]
        payload = {
            "timestamp": np.int64(time.time_ns()),
            "mode_pr": np.int64(msg.mode_pr),
            "mode_machine": np.int64(msg.mode_machine),
            "q": np.array([m.q for m in motors], np.float64),
            "dq": np.array([m.dq for m in motors], np.float64),
            "tau": np.array([m.tau_est for m in motors], np.float64),
            "imu_quat": np.array(msg.imu_state.quaternion, np.float64),
            "imu_rpy": np.array(msg.imu_state.rpy, np.float64),
            "imu_gyro": np.array(msg.imu_state.gyroscope, np.float64),
            "imu_accel": np.array(msg.imu_state.accelerometer, np.float64),
        }
        try:
            self._q.put_nowait(payload)
        except _queue.Full:
            pass

    def init(self):
        self._q = _queue.Queue(maxsize=self.queue_max)
        ChannelFactoryInitialize(self.domain_id, self.interface)
        self._sub = ChannelSubscriber(self.topic, LowState_)
        self._sub.Init(self._on_lowstate, self.sub_queue_len)
        while self._q.empty():
            time.sleep(0.01)

    def stream(self):
        while True:
            yield {self.name: self._q.get()}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--interface", default="enP8p1s0",
                   help="DDS network interface (default: enP8p1s0, the G1 robot LAN on this Orin)")
    args = p.parse_args()
    obj = UnitreeRobot(interface=args.interface)
    print(f"[{obj.name}] init... (interface={args.interface!r})", flush=True)
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
