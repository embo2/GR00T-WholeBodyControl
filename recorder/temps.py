"""Print per-part temperatures from G1 LowState (rt/lowstate).

Run:
    uv run python temps.py --interface enP8p1s0
"""

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# G1 29-DoF joint order (matches LowState motor_state[0..28]).
G1_JOINTS = [
    "L_HIP_PITCH", "L_HIP_ROLL", "L_HIP_YAW", "L_KNEE",
    "L_ANKLE_PITCH", "L_ANKLE_ROLL",
    "R_HIP_PITCH", "R_HIP_ROLL", "R_HIP_YAW", "R_KNEE",
    "R_ANKLE_PITCH", "R_ANKLE_ROLL",
    "WAIST_YAW", "WAIST_ROLL", "WAIST_PITCH",
    "L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW",
    "L_ELBOW", "L_WRIST_ROLL", "L_WRIST_PITCH", "L_WRIST_YAW",
    "R_SHOULDER_PITCH", "R_SHOULDER_ROLL", "R_SHOULDER_YAW",
    "R_ELBOW", "R_WRIST_ROLL", "R_WRIST_PITCH", "R_WRIST_YAW",
]
NUM_MOTORS = len(G1_JOINTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface", default="enP8p1s0")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--rate", type=float, default=1.0, help="print Hz")
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(None, 0)
    print(f"subscribed to rt/lowstate on {args.interface}", flush=True)

    period = 1.0 / args.rate
    while True:
        msg = sub.Read()
        if msg is None:
            time.sleep(0.005)
            continue

        print("\033[2J\033[H", end="")  # clear screen
        print(f"=== G1 temperatures  t={time.strftime('%H:%M:%S')} ===")
        print(f"{'joint':<22}{'coil':>6}{'case':>6}")
        for i in range(NUM_MOTORS):
            t = msg.motor_state[i].temperature  # [coil, case], int16 °C
            print(f"{G1_JOINTS[i]:<22}{t[0]:>6}{t[1]:>6}")

        print(f"\nimu temp: {msg.imu_state.temperature}")

        time.sleep(period)


if __name__ == "__main__":
    main()
