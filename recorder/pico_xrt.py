import time
import numpy as np

SMPL_JOINTS = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee",
    "spine2", "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot",
    "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hand", "R_hand",
]


f64 = np.float64


class PicoXrt:
    def __init__(self, name="pico_xrt"):
        self.name = name
        self._xrt = None

    def init(self):
        import xrobotoolkit_sdk as xrt
        self._xrt = xrt
        xrt.init()
        while not xrt.is_body_data_available():
            time.sleep(0.05)

    def stream(self):
        xrt = self._xrt
        last_ts_ns = 0
        while True:
            ts_ns = xrt.get_time_stamp_ns()
            if ts_ns == last_ts_ns:
                time.sleep(1e-4)
                continue
            last_ts_ns = ts_ns
            body = xrt.get_body_joints_pose()
            yield {self.name: {
                "timestamp": np.int64(time.time_ns()),
                "created": np.int64(ts_ns),
                "body_joints": np.array([list(body[i]) for i in range(24)], f64),
                "left_trigger": f64(xrt.get_left_trigger()),
                "right_trigger": f64(xrt.get_right_trigger()),
                "left_grip": f64(xrt.get_left_grip()),
                "right_grip": f64(xrt.get_right_grip()),
                "left_axis": np.array(list(xrt.get_left_axis()), f64),
                "right_axis": np.array(list(xrt.get_right_axis()), f64),
                "buttons": {
                    "A": bool(xrt.get_A_button()),
                    "B": bool(xrt.get_B_button()),
                    "X": bool(xrt.get_X_button()),
                    "Y": bool(xrt.get_Y_button()),
                    "menu": bool(xrt.get_left_menu_button()),
                },
            }}


if __name__ == "__main__":
    obj = PicoXrt()
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
