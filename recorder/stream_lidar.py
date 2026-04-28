#!/usr/bin/env python3
import argparse
import socket
import struct
import threading
import time

MCAST_GROUP = "224.1.1.164"
HEADER_SIZE = 36
POINT_SIZE = 14
IMU_SIZE = 24


def make_socket(host_ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass
    s.bind(("", port))
    mreq = struct.pack("4s4s", socket.inet_aton(MCAST_GROUP), socket.inet_aton(host_ip))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    s.settimeout(1.0)
    return s


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.packets = 0
        self.points = 0
        self.last_xyz = None
        self.imu_count = 0
        self.imu = None


def pc_loop(host_ip, state, stop):
    s = make_socket(host_ip, 56301)
    while not stop.is_set():
        try:
            data, _ = s.recvfrom(65535)
        except socket.timeout:
            continue
        if len(data) < HEADER_SIZE + POINT_SIZE:
            continue
        try:
            _, _, _, dot_num, _, _, dt, _ = struct.unpack_from("<BHHHHBBb", data, 0)
        except struct.error:
            continue
        if dt != 1:
            continue
        npts = 0
        last = None
        for i in range(dot_num):
            off = HEADER_SIZE + i * POINT_SIZE
            if off + POINT_SIZE > len(data):
                break
            x, y, z, _r, _tag = struct.unpack_from("<iiiBB", data, off)
            if x == 0 and y == 0 and z == 0:
                continue
            npts += 1
            last = (x * 0.001, y * 0.001, z * 0.001)
        with state.lock:
            state.packets += 1
            state.points += npts
            if last is not None:
                state.last_xyz = last


def imu_loop(host_ip, state, stop):
    s = make_socket(host_ip, 56401)
    while not stop.is_set():
        try:
            data, _ = s.recvfrom(65535)
        except socket.timeout:
            continue
        if len(data) < HEADER_SIZE + IMU_SIZE:
            continue
        try:
            _, _, _, _, _, _, dt, _ = struct.unpack_from("<BHHHHBBb", data, 0)
        except struct.error:
            continue
        if dt != 0:
            continue
        try:
            gx, gy, gz, ax, ay, az = struct.unpack_from("<6f", data, HEADER_SIZE)
        except struct.error:
            continue
        with state.lock:
            state.imu = (gx, gy, gz, ax, ay, az)
            state.imu_count += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host-ip", default="192.168.123.164")
    p.add_argument("--rate", type=float, default=2.0)
    args = p.parse_args()

    state = State()
    stop = threading.Event()
    threading.Thread(target=pc_loop, args=(args.host_ip, state, stop), daemon=True).start()
    threading.Thread(target=imu_loop, args=(args.host_ip, state, stop), daemon=True).start()

    period = 1.0 / max(args.rate, 0.1)
    last_pkts = last_pts = last_ic = 0
    last_t = time.time()
    try:
        while True:
            time.sleep(period)
            with state.lock:
                pkts, pts, last_xyz = state.packets, state.points, state.last_xyz
                ic, imu = state.imu_count, state.imu
            now = time.time()
            dt = max(now - last_t, 1e-6)
            print(f"\n=== {time.strftime('%H:%M:%S')} ===")
            xyz = "(none)" if last_xyz is None else f"({last_xyz[0]:+.2f}, {last_xyz[1]:+.2f}, {last_xyz[2]:+.2f})"
            print(f"pc    pkt/s={(pkts-last_pkts)/dt:6.0f}  pts/s={(pts-last_pts)/dt:7.0f}  last_xyz={xyz}")
            if imu is None:
                print("imu   (no data)")
            else:
                print(f"imu   hz={(ic-last_ic)/dt:6.0f}  "
                      f"gyro=({imu[0]:+.3f}, {imu[1]:+.3f}, {imu[2]:+.3f})  "
                      f"accel=({imu[3]:+.3f}, {imu[4]:+.3f}, {imu[5]:+.3f})")
            last_pkts, last_pts, last_ic, last_t = pkts, pts, ic, now
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
