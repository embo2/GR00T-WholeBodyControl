import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

NUM_MOTORS = 29


class UnitreeRobot:
    def __init__(self, name="unitree_robot", interface="",
                 topic="rt/lowstate", domain_id=0):
        assert interface
        self.name = name
        self.interface = interface
        self.domain_id = domain_id
        self.topic = topic

    def init(self):
        ChannelFactoryInitialize(self.domain_id, self.interface)
        self.sub = ChannelSubscriber(self.topic, LowState_)
        self.sub.Init(None, 0)

    @property
    def spec(self):
      return {self.name: 'tree'}

    def stream(self):
        while True:
            msg = self.sub.Read()
            if msg is None:
              continue
            yield self._process(msg)

    def _process(self, msg):
        motors = msg.motor_state[:NUM_MOTORS]
        metadata = {
            "timestamp": np.int64(time.time_ns()),
            "created": np.int64(time.time_ns()),
        }
        data = {
            "mode_pr": np.int64(msg.mode_pr),
            "mode_machine": np.int64(msg.mode_machine),
            "tick": np.uint32(msg.tick),
            "q": np.array([m.q for m in motors], np.float64),
            "dq": np.array([m.dq for m in motors], np.float64),
            "ddq": np.array([m.ddq for m in motors], np.float64),
            "tau": np.array([m.tau_est for m in motors], np.float64),
            "vol": np.array([m.vol for m in motors], np.float64),
            "motor_mode": np.array([m.mode for m in motors], np.uint8),
            "motor_state_flags": np.array([m.motorstate for m in motors], np.uint32),
            "temp_coil": np.array([m.temperature[0] for m in motors], np.int16),
            "temp_case": np.array([m.temperature[1] for m in motors], np.int16),
            "imu_quat": np.array(msg.imu_state.quaternion, np.float64),
            "imu_rpy": np.array(msg.imu_state.rpy, np.float64),
            "imu_gyro": np.array(msg.imu_state.gyroscope, np.float64),
            "imu_accel": np.array(msg.imu_state.accelerometer, np.float64),
            "imu_temp": np.int16(msg.imu_state.temperature),
        }
        return {self.name: {'metadata': metadata, 'data': data}}



if __name__ == "__main__":
    interface = 'enP8p1s0'
    obj = UnitreeRobot(interface=interface)
    obj.init()
    print(f"[{obj.name}] connected. (Ctrl+C to stop)", flush=True)
    count = 0
    start = time.perf_counter()
    for d in obj.stream():
        count += 1
        elapsed = time.perf_counter()
        fps = count / (elapsed - start)
        print(f'{fps:.2f} msgs/s')
