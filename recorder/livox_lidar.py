import select
import socket
import struct
import time


class LivoxLidar:
    def __init__(self, name="livox_lidar",
                 host_ip="192.168.123.164",
                 mcast_group="224.1.1.164",
                 pc_port=56301, imu_port=56401,
                 rcvbuf_bytes=4 * 1024 * 1024):
        self.name = name
        self.host_ip = host_ip
        self.mcast_group = mcast_group
        self.pc_port = pc_port
        self.imu_port = imu_port
        self.rcvbuf_bytes = rcvbuf_bytes
        self._pc_sock = None
        self._imu_sock = None

    def _make_socket(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        s.bind(("", port))
        mreq = struct.pack("4s4s",
                           socket.inet_aton(self.mcast_group),
                           socket.inet_aton(self.host_ip))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_bytes)
        return s

    def init(self):
        self._pc_sock = self._make_socket(self.pc_port)
        self._imu_sock = self._make_socket(self.imu_port)
        pc_seen = imu_seen = False
        while not (pc_seen and imu_seen):
            rlist, _, _ = select.select([self._pc_sock, self._imu_sock], [], [], 1.0)
            if self._pc_sock in rlist:
                pc_seen = True
            if self._imu_sock in rlist:
                imu_seen = True

    def stream(self):
        pc_key = f"{self.name}_pc"
        imu_key = f"{self.name}_imu"
        while True:
            rlist, _, _ = select.select([self._pc_sock, self._imu_sock], [], [], 1.0)
            for sock in rlist:
                try:
                    data, _ = sock.recvfrom(65535)
                except (BlockingIOError, InterruptedError):
                    continue
                key = pc_key if sock is self._pc_sock else imu_key
                yield {key: {
                    "t_realtime": time.time(),
                    "t_monotonic": time.monotonic(),
                    "raw": bytes(data),
                }}


if __name__ == "__main__":
    obj = LivoxLidar()
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
