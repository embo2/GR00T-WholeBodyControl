#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <zmq.h>
#include <msgpack.h>

typedef struct {
    uint8_t hardware_type;
    uint8_t sku_type;
    const char *serial_number;
    const char *firmware_version;
} DeviceInfo;

typedef void* (*open_fn)(const char *port, uint32_t baudrate);
typedef void  (*close_fn)(void *handle);
typedef void  (*set_pos_fn)(void *handle, uint8_t slave, const uint16_t *pos, size_t n);
typedef DeviceInfo* (*get_info_fn)(void *handle, uint8_t slave);
typedef void  (*free_info_fn)(DeviceInfo *info);
typedef void  (*log_fn)(uint8_t level);

static int init_hand(get_info_fn get_info, free_info_fn free_info,
                     void *handle, uint8_t slave, const char *name) {
    for (int attempt = 1; attempt <= 10; attempt++) {
        DeviceInfo *info = get_info(handle, slave);
        int ok = info && info->serial_number && info->serial_number[0] != '\0';
        if (info) {
            fprintf(stderr, "[%s] attempt %d: hw=%u sku=%u sn=%s fw=%s -> %s\n",
                name, attempt, info->hardware_type, info->sku_type,
                info->serial_number ? info->serial_number : "(null)",
                info->firmware_version ? info->firmware_version : "(null)",
                ok ? "OK" : "RETRY");
            free_info(info);
        } else {
            fprintf(stderr, "[%s] attempt %d: NULL -> RETRY\n", name, attempt);
        }
        if (ok) return 1;
        usleep(150000);  // 150ms between attempts
    }
    fprintf(stderr, "[%s] FAILED to init after retries\n", name);
    return 0;
}

static const char *SDK_PATH = "./libbc_stark_sdk.so";

static const char *TOPIC = "hand_control";

static volatile sig_atomic_t running = 1;
static void on_signal(int sig) { (void)sig; running = 0; }

static double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

static int extract_double(msgpack_object *o, double *out) {
    if (o->type == MSGPACK_OBJECT_FLOAT32 || o->type == MSGPACK_OBJECT_FLOAT64) {
        *out = o->via.f64; return 1;
    }
    if (o->type == MSGPACK_OBJECT_POSITIVE_INTEGER) { *out = (double)o->via.u64; return 1; }
    if (o->type == MSGPACK_OBJECT_NEGATIVE_INTEGER) { *out = (double)o->via.i64; return 1; }
    return 0;
}

int main(int argc, char **argv) {
    int port = 5566;
    int opt;
    while ((opt = getopt(argc, argv, "P:h")) != -1) {
        switch (opt) {
            case 'P': port = atoi(optarg); break;
            case 'h':
            default:
                fprintf(stderr, "usage: %s [-P port]   default: 5566\n", argv[0]);
                return opt == 'h' ? 0 : 1;
        }
    }

    void *sdk = dlopen(SDK_PATH, RTLD_NOW);
    if (!sdk) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    open_fn      modbus_open  = (open_fn)     dlsym(sdk, "modbus_open");
    close_fn     modbus_close = (close_fn)    dlsym(sdk, "modbus_close");
    set_pos_fn   set_pos      = (set_pos_fn)  dlsym(sdk, "stark_set_finger_positions");
    get_info_fn  get_info     = (get_info_fn) dlsym(sdk, "stark_get_device_info");
    free_info_fn free_info    = (free_info_fn)dlsym(sdk, "free_device_info");
    log_fn       init_log     = (log_fn)      dlsym(sdk, "init_logging");
    if (!modbus_open || !modbus_close || !set_pos || !get_info || !free_info) {
        fprintf(stderr, "dlsym: missing symbol\n"); return 1;
    }
    if (init_log) init_log(0);

    void *left  = modbus_open("/dev/left_hand",  460800);
    void *right = modbus_open("/dev/right_hand", 460800);
    if (!left || !right) { fprintf(stderr, "modbus_open failed\n"); return 1; }

    if (!init_hand(get_info, free_info, left,  0x7e, "left") ||
        !init_hand(get_info, free_info, right, 0x7f, "right")) {
        fprintf(stderr, "hand init failed; refusing to start to avoid silent no-op writes\n");
        return 1;
    }

    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) { dup2(devnull, 2); close(devnull); }

    void *ctx = zmq_ctx_new();
    void *sub = zmq_socket(ctx, ZMQ_SUB);
    int conflate = 1, rcvtimeo = 10, linger = 0;
    zmq_setsockopt(sub, ZMQ_CONFLATE, &conflate, sizeof(conflate));
    zmq_setsockopt(sub, ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
    zmq_setsockopt(sub, ZMQ_LINGER,   &linger,   sizeof(linger));

    char endpoint[64];
    snprintf(endpoint, sizeof(endpoint), "tcp://*:%d", port);
    if (zmq_bind(sub, endpoint) != 0) {
        fprintf(stderr, "zmq_bind %s failed: %s\n", endpoint, zmq_strerror(zmq_errno()));
        return 1;
    }
    zmq_setsockopt(sub, ZMQ_SUBSCRIBE, TOPIC, strlen(TOPIC));

    printf("hand_server: listening on %s topic=\"%s\"\n", endpoint, TOPIC);
    fflush(stdout);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    char buf[1024];
    msgpack_unpacked unpacked;
    msgpack_unpacked_init(&unpacked);
    size_t topic_len = strlen(TOPIC);

    const long long SEND_PERIOD_NS = 100000000LL;
    struct timespec ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_now);
    long long last_send_ns = (long long)ts_now.tv_sec * 1000000000LL + ts_now.tv_nsec;
    double pending_l = -1.0, pending_r = -1.0;
    int has_pending = 0;
    long long recv_count = 0, drop_count = 0, send_count = 0;

    while (running) {
        int n = zmq_recv(sub, buf, sizeof(buf), 0);
        if (n > 0 && (size_t)n >= topic_len && memcmp(buf, TOPIC, topic_len) == 0) {
            size_t off = 0;
            msgpack_unpack_return r = msgpack_unpack_next(&unpacked,
                buf + topic_len, (size_t)n - topic_len, &off);
            if (r == MSGPACK_UNPACK_SUCCESS && unpacked.data.type == MSGPACK_OBJECT_MAP) {
                msgpack_object obj = unpacked.data;
                for (uint32_t i = 0; i < obj.via.map.size; i++) {
                    msgpack_object_kv *kv = &obj.via.map.ptr[i];
                    if (kv->key.type != MSGPACK_OBJECT_STR) continue;
                    const char *k = kv->key.via.str.ptr;
                    uint32_t klen = kv->key.via.str.size;
                    double v;
                    if (!extract_double(&kv->val, &v)) continue;
                    if (klen == 4 && memcmp(k, "left", 4) == 0)       pending_l = v;
                    else if (klen == 5 && memcmp(k, "right", 5) == 0) pending_r = v;
                }
                if (has_pending) drop_count++;
                has_pending = 1;
                recv_count++;
            }
        } else if (n < 0 && zmq_errno() != EAGAIN && zmq_errno() != EINTR) {
            break;
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        long long now_ns = (long long)ts_now.tv_sec * 1000000000LL + ts_now.tv_nsec;
        if (has_pending && now_ns - last_send_ns >= SEND_PERIOD_NS) {
            if (pending_l >= 0) {
                uint16_t pv = (uint16_t)(clamp01(pending_l) * 1000.0 + 0.5);
                uint16_t pos[6] = {pv,pv,pv,pv,pv,pv};
                set_pos(left, 0x7e, pos, 6);
            }
            if (pending_r >= 0) {
                uint16_t pv = (uint16_t)(clamp01(pending_r) * 1000.0 + 0.5);
                uint16_t pos[6] = {pv,pv,pv,pv,pv,pv};
                set_pos(right, 0x7f, pos, 6);
            }
            send_count++;
            printf("send #%lld L=%.3f R=%.3f (rx=%lld drop=%lld)\n",
                send_count,
                pending_l >= 0 ? clamp01(pending_l) : -1.0,
                pending_r >= 0 ? clamp01(pending_r) : -1.0,
                recv_count, drop_count);
            fflush(stdout);
            last_send_ns = now_ns;
            has_pending = 0;
            pending_l = pending_r = -1.0;
        }
    }

    msgpack_unpacked_destroy(&unpacked);

    uint16_t opened[6] = {0,0,0,0,0,0};
    set_pos(left,  0x7e, opened, 6);
    set_pos(right, 0x7f, opened, 6);

    zmq_close(sub);
    zmq_ctx_term(ctx);
    modbus_close(left);
    modbus_close(right);
    dlclose(sdk);
    return 0;
}
