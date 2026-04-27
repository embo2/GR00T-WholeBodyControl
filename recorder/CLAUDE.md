# Recorder — Teleop Data Collection

Standalone Python data collection layer that subscribes to existing teleop streams without modifying any teleop code.

## Data Sources & How to Access

### Pico Teleop (ZMQ)
- **Source**: `gear_sonic/scripts/pico_manager_thread_server.py` — `run_once()` at line ~1287
- **Port**: 5556 (ZMQ PUB), topic `"pose"`
- **Format**: `[topic_prefix][1024-byte JSON header][binary fields]` — use `pack_pose_message` format from `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`
- **Contains**:
  - `vr_position` (9 floats) — left wrist xyz, right wrist xyz, head xyz
  - `vr_orientation` (12 floats) — 3 quaternions wxyz
  - `left_trigger`, `right_trigger` (float 0-1) — controller triggers
  - `left_grip`, `right_grip` (float 0-1) — controller grips
  - `left_hand_joints`, `right_hand_joints` (7 floats each) — Dex3 joints after IK retargeting from trigger/grip
  - `joint_pos` (29 floats) — SMPL-retargeted wrist joint positions
  - `smpl_pose`, `smpl_joints`, `body_quat_w` — raw SMPL body tracking
  - `heading_increment` — yaw from right joystick
- **Retargeting**: Pico body tracking → SMPL → VR 3-point calibration + G1 wrist IK happens in `pico_manager_thread_server.py`. The C++ deploy binary receives already-solved joint targets.

### RGB Frames (ZMQ)
- **Source**: `XRoboToolkit-Orin-Video-Sender/main_realsense_tcp.cpp` — `ego_zmq` namespace
- **Port**: 5558 (ZMQ PUB), topic `"ego_view"`
- **Format**: multipart `[topic_bytes, u64_le_timestamp_ns, jpeg_bytes]` (quality 85)
- **Rate**: ~30 FPS

### Depth Frames (ZMQ)
- **Source**: same C++ binary — `depth_zmq` namespace
- **Port**: 5563 (ZMQ PUB), topic `"depth"`
- **Format**: multipart `[topic_bytes, u64_le_timestamp_ns, raw_z16_bytes]` (640x480 uint16)
- **Rate**: ~30 FPS

### Robot Proprioception (Unitree DDS)
- **Source**: Unitree G1 hardware, read in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
- **Primary read function**: `GatherRobotStateToLogger()` at line ~2806
- **DDS topics**:
  - `rt/lowstate` (LowState_) — 500 Hz: joint q/dq/tau_est (29 motors), pelvis IMU (quat, gyro, accel), motor temps, errors
  - `rt/secondary_imu` (IMUState_) — torso IMU (quat, gyro, accel)
  - `rt/dex3/left/state`, `rt/dex3/right/state` — Dex3 hand joint positions/velocities (7 DOF each)
- **Python access**: subscribe via `unitree_sdk2py` ChannelSubscriber to the same DDS topics

### Video Timing (ZMQ, optional)
- **Port**: 5571 (ZMQ PUB), topic `"video_timing"`
- **Format**: multipart `[topic_bytes, 36-byte packed struct]` — per-frame pipeline latency

## Architecture

All ports are hardcoded `static const` in the C++ sources. ZMQ PUB/SUB is one-to-many — multiple subscribers get copies without affecting existing consumers.

The deploy pipeline (`gear_sonic_deploy/deploy.sh`) is pure C++. The pico manager (`gear_sonic/scripts/pico_manager_thread_server.py`) is Python. Both run independently. This recorder subscribes to both without coupling to either.

The RealSense camera can only be opened by one process. During teleop the C++ binary owns it, so use ZMQ. When C++ is not running, `pyrealsense2` can read directly.

## Key Insertion Points (if modifying teleop code directly)

- **Teleop inputs (Python)**: `pico_manager_thread_server.py:1459` — `numpy_data` dict right before `pack_pose_message`
- **Robot state (C++)**: `g1_deploy_onnx_ref.cpp:2940` — end of `GatherRobotStateToLogger()`
- **Teleop inputs consumed by policy (C++)**: `g1_deploy_onnx_ref.cpp:2972` — end of `GatherInputInterfaceData()`
- **RGB capture (C++)**: `main_realsense_tcp.cpp:760` — after `poll_for_frames`
- **Depth capture (C++)**: `main_realsense_tcp.cpp:842` — after aligned depth extraction
