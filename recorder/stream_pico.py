#!/usr/bin/env python3
import argparse
import time

import xrobotoolkit_sdk as xrt

SMPL_JOINTS = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee",
    "spine2", "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot",
    "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hand", "R_hand",
]

def fmt(values, prec=3):
    return "[" + ", ".join(f"{float(v):+.{prec}f}" for v in values) + "]"


def is_real(joint):
    pos_mag = sum(v * v for v in joint[0:3]) ** 0.5
    quat_drift = abs(1.0 - abs(joint[6]))
    return pos_mag > 1e-4 or quat_drift > 1e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=2.0)
    args = p.parse_args()

    xrt.init()
    print("XRT init complete. Waiting for body data...")
    while not xrt.is_body_data_available():
        time.sleep(0.05)
    print("Body data available. Streaming (Ctrl+C to stop).")

    period = 1.0 / max(args.rate, 0.1)
    last_stamp_ns = 0
    try:
        while True:
            stamp = xrt.get_time_stamp_ns()
            body = xrt.get_body_joints_pose()
            joints = [list(body[i]) for i in range(24)]
            lt, rt = xrt.get_left_trigger(), xrt.get_right_trigger()
            lg, rg = xrt.get_left_grip(), xrt.get_right_grip()
            la, ra = xrt.get_left_axis(), xrt.get_right_axis()
            a = xrt.get_A_button()
            b = xrt.get_B_button()
            x_btn = xrt.get_X_button()
            y_btn = xrt.get_Y_button()
            menu = xrt.get_left_menu_button()

            dt_ms = (stamp - last_stamp_ns) / 1e6 if last_stamp_ns else 0
            last_stamp_ns = stamp

            print(f"\n=== {time.strftime('%H:%M:%S')}  dt={dt_ms:.1f}ms ===")
            print(f"  trig L={lt:.2f} R={rt:.2f}  grip L={lg:.2f} R={rg:.2f}")
            print(f"  axis L={fmt(la, 2)}  R={fmt(ra, 2)}")
            print(f"  btn  A={int(a)} B={int(b)} X={int(x_btn)} Y={int(y_btn)} menu={int(menu)}")
            real_count = sum(1 for j in joints if is_real(j))
            print(f"  body  {real_count}/24 joints have real data")
            for i, name in enumerate(SMPL_JOINTS):
                j = joints[i]
                flag = "*" if is_real(j) else " "
                print(f" {flag} {name:<10s} pos={fmt(j[0:3])}  quat(xyzw)={fmt(j[3:7])}")
            time.sleep(period)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
