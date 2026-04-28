#!/usr/bin/env python3
import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

NUM_MOTORS = 29


def fmt(values, prec=3):
    return "[" + ", ".join(f"{v:+.{prec}f}" for v in values) + "]"


def print_lowstate(ls):
    m = ls.motor_state[:NUM_MOTORS]
    print(f"\n=== {time.strftime('%H:%M:%S')}  tick={ls.tick} ===")
    print(f"  q   = {fmt([x.q for x in m])}")
    print(f"  dq  = {fmt([x.dq for x in m])}")
    print(f"  tau = {fmt([x.tau_est for x in m], prec=2)}")
    print(f"  pelvis quat={fmt(ls.imu_state.quaternion)} rpy={fmt(ls.imu_state.rpy)}")
    print(f"  pelvis gyro={fmt(ls.imu_state.gyroscope)} accel={fmt(ls.imu_state.accelerometer)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interface", default="")
    p.add_argument("--rate", type=float, default=2.0)
    args = p.parse_args()

    ChannelFactoryInitialize(0, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(None, 0)

    period = 1.0 / max(args.rate, 0.1)
    last = None
    try:
        while True:
            ls = sub.Read() or last
            if ls is not None:
                print_lowstate(ls)
                last = ls
            else:
                print("(no data yet)")
            time.sleep(period)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
