// Minimal OAK-D → NVENC → PICO TCP streamer.
//
// Strips out everything in main.cpp that isn't on the headset video path:
// no HUD, no ego_zmq publisher to the laptop recorder, no per-frame timing
// publisher, no preview, no send-only mode. What remains:
//
//   OAK-D ColorCamera (DepthAI)
//     └─ capture thread → shared cv::Mat
//                            │
//                            ▼
//   PICO TCP listen :13579 → OPEN_CAMERA(width,height,bitrate,h264|h265,target_ip,target_port)
//                            │
//                            ▼
//   GStreamer pipeline: appsrc → videoconvert → nvvidconv → nvv4l2h264enc → appsink
//                                                                              │
//                            on_new_sample → 4-byte length-prefix → TCPClient ─┘
//                                                                       │
//                                                                       ▼
//                                                                 PICO at target_ip:target_port

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <opencv2/opencv.hpp>
#include <depthai/depthai.hpp>

#include "network_helper.hpp"

// ---------------------------------------------------------------------------
// PICO wire protocol parsers (4-byte little-endian framing)
// ---------------------------------------------------------------------------

struct CameraRequest {
    int32_t width = 0, height = 0, fps = 0, bitrate = 0;
    int32_t enableMvHevc = 0, renderMode = 0, port = 0;
    std::string ip;
};

static int32_t read_le_i32(const uint8_t *p) {
    return (int32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}

static std::string read_short_string(const std::vector<uint8_t> &d, size_t &off) {
    uint8_t n = d[off++];
    std::string s(reinterpret_cast<const char *>(&d[off]), n);
    off += n;
    return s;
}

// PICO OPEN_CAMERA payload: 0xCA 0xFE 0x01, 28-byte fixed fields, then two
// length-prefixed strings (camera name, ip).
static CameraRequest parse_camera_request(const std::vector<uint8_t> &data) {
    if (data.size() < 31 || data[0] != 0xCA || data[1] != 0xFE || data[2] != 1)
        throw std::invalid_argument("bad CameraRequest header");
    CameraRequest r;
    r.width        = read_le_i32(&data[3]);
    r.height       = read_le_i32(&data[7]);
    r.fps          = read_le_i32(&data[11]);
    r.bitrate      = read_le_i32(&data[15]);
    r.enableMvHevc = read_le_i32(&data[19]);
    r.renderMode   = read_le_i32(&data[23]);
    r.port         = read_le_i32(&data[27]);
    size_t off = 31;
    read_short_string(data, off);  // camera name (unused)
    r.ip = read_short_string(data, off);
    return r;
}

struct NetworkCommand {
    std::string command;
    std::vector<uint8_t> data;
};

// PICO outer framing: int32 cmd_len, cmd_bytes, int32 data_len, data_bytes.
static NetworkCommand parse_network_command(const std::vector<uint8_t> &buf) {
    if (buf.size() < 8) throw std::invalid_argument("buffer too small");
    int32_t cmd_len = read_le_i32(&buf[0]);
    NetworkCommand n;
    if (cmd_len > 0) {
        n.command.assign(reinterpret_cast<const char *>(&buf[4]), cmd_len);
        size_t z = n.command.find('\0');
        if (z != std::string::npos) n.command.resize(z);
    }
    int32_t data_len = read_le_i32(&buf[4 + cmd_len]);
    if (data_len > 0) n.data.assign(buf.begin() + 8 + cmd_len, buf.begin() + 8 + cmd_len + data_len);
    return n;
}

// ---------------------------------------------------------------------------
// OAK-D capture thread
// ---------------------------------------------------------------------------

namespace oakd {
static cv::Mat g_frame;
static std::mutex g_mtx;
static std::atomic<int> g_seq{0};
static std::atomic<bool> g_running{false};

static cv::Mat take_frame() {
    std::lock_guard<std::mutex> lock(g_mtx);
    cv::Mat out;
    std::swap(out, g_frame);
    return out;
}

static void capture_loop() {
    using namespace dai;
    Pipeline p;
    auto cam = p.create<node::ColorCamera>();
    cam->setBoardSocket(CameraBoardSocket::CAM_A);
    cam->setResolution(node::ColorCamera::Properties::SensorResolution::THE_800_P);
    cam->setIspScale(1, 2);  // 1280x800 -> 640x400
    cam->setColorOrder(node::ColorCamera::Properties::ColorOrder::BGR);
    cam->setInterleaved(false);
    cam->setFps(60);
    auto x = p.create<node::XLinkOut>();
    x->setStreamName("rgb");
    cam->isp.link(x->input);

    std::unique_ptr<Device> dev;
    try {
        dev = std::make_unique<Device>(p);
        std::cout << "[oakd] pipeline started (USB=" << dev->getUsbSpeed() << ")" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "[oakd] start failed: " << e.what() << std::endl;
        return;
    }
    auto q = dev->getOutputQueue("rgb", 4, false);
    while (g_running.load()) {
        auto f = q->tryGet<ImgFrame>();
        if (!f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        cv::Mat bgr = f->getCvFrame();
        {
            std::lock_guard<std::mutex> lock(g_mtx);
            g_frame = bgr.clone();
        }
        g_seq.fetch_add(1);
    }
    std::cout << "[oakd] stopped" << std::endl;
}
}  // namespace oakd

// ---------------------------------------------------------------------------
// Streaming session: TCP client back to PICO + GStreamer NVENC pipeline
// ---------------------------------------------------------------------------

static std::atomic<bool> stop_requested{false};
static std::atomic<bool> streaming_active{false};
static std::unique_ptr<TCPClient> sender;

static GstFlowReturn on_new_sample(GstAppSink *sink, gpointer) {
    GstSample *sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_ERROR;
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        if (sender && sender->isConnected() && map.data && map.size > 0) {
            try {
                std::vector<uint8_t> packet(4 + map.size);
                packet[0] = (map.size >> 24) & 0xFF;
                packet[1] = (map.size >> 16) & 0xFF;
                packet[2] = (map.size >> 8)  & 0xFF;
                packet[3] = (map.size)       & 0xFF;
                std::copy(map.data, map.data + map.size, packet.begin() + 4);
                sender->sendData(packet);
            } catch (const std::exception &e) {
                std::cerr << "[send] " << e.what() << std::endl;
                streaming_active.store(false);
            }
        }
        gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static void streaming_session(const CameraRequest &cfg) {
    std::cout << "[stream] connect back to " << cfg.ip << ":" << cfg.port << std::endl;
    try {
        sender = std::make_unique<TCPClient>(cfg.ip, cfg.port);
        sender->connect();
    } catch (const TCPException &e) {
        std::cerr << "[stream] connect failed: " << e.what() << std::endl;
        sender.reset();
        return;
    }

    int out_w = cfg.width  > 0 ? cfg.width  : 1280;
    int out_h = cfg.height > 0 ? cfg.height : 720;
    int fps   = 60;
    int br    = cfg.bitrate > 0 ? cfg.bitrate : 4000000;

    // PICO requests a wide stereo frame (>= 2:1). We have one camera, so we
    // duplicate the same view into both eyes side-by-side.
    int eye_w = out_w, eye_h = out_h;
    bool stereo = (out_w > out_h * 2 - 10);
    if (stereo) { eye_w = out_w / 2; eye_h = out_h; }

    std::string enc    = cfg.enableMvHevc ? "nvv4l2h265enc" : "nvv4l2h264enc";
    std::string parser = cfg.enableMvHevc ? "h265parse"     : "h264parse";

    std::string pipeline_str =
        "appsrc name=mysource is-live=true format=time "
        "caps=video/x-raw,format=BGR,width=" + std::to_string(out_w) +
        ",height=" + std::to_string(out_h) +
        ",framerate=" + std::to_string(fps) + "/1 ! "
        "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        + enc + " maxperf-enable=1 insert-sps-pps=true idrinterval=30 "
        "bitrate=" + std::to_string(br) + " preset-level=UltraFastPreset ! "
        + parser + " ! appsink name=mysink emit-signals=true sync=false";

    std::cout << "[stream] pipeline: " << pipeline_str << std::endl;
    GError *err = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &err);
    if (!pipeline) {
        std::cerr << "[stream] gst error: " << err->message << std::endl;
        g_clear_error(&err);
        sender.reset();
        return;
    }
    GstElement *appsrc  = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    cv::Mat frame, sbs;
    int frame_id = 0;
    int last_seq = 0;
    streaming_active.store(true);
    std::cout << "[stream] running (out=" << out_w << "x" << out_h
              << " eye=" << eye_w << "x" << eye_h << ")" << std::endl;

    while (streaming_active.load() && !stop_requested.load()) {
        int seq = oakd::g_seq.load();
        if (seq == last_seq) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_seq = seq;
        frame = oakd::take_frame();
        if (frame.empty()) continue;

        if (frame.cols != eye_w || frame.rows != eye_h)
            cv::resize(frame, frame, cv::Size(eye_w, eye_h));

        cv::Mat *out = &frame;
        if (stereo) {
            sbs = cv::Mat(out_h, out_w, frame.type());
            frame.copyTo(sbs(cv::Rect(0, 0, eye_w, eye_h)));
            frame.copyTo(sbs(cv::Rect(eye_w, 0, eye_w, eye_h)));
            out = &sbs;
        }

        size_t bytes = out->total() * out->elemSize();
        GstBuffer *buf = gst_buffer_new_allocate(nullptr, bytes, nullptr);
        GstMapInfo map;
        gst_buffer_map(buf, &map, GST_MAP_WRITE);
        memcpy(map.data, out->data, bytes);
        gst_buffer_unmap(buf, &map);
        GST_BUFFER_PTS(buf)      = gst_util_uint64_scale(frame_id, GST_SECOND, fps);
        GST_BUFFER_DURATION(buf) = gst_util_uint64_scale(1, GST_SECOND, fps);
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), buf);
        frame_id++;
    }

    std::cout << "[stream] tearing down" << std::endl;
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    if (sender && sender->isConnected()) sender->disconnect();
    sender.reset();
}

// ---------------------------------------------------------------------------
// PICO TCP listener
// ---------------------------------------------------------------------------

static std::unique_ptr<std::thread> session_thread;

static void on_command(const std::string &raw) {
    std::vector<uint8_t> bin(raw.begin(), raw.end());
    if (bin.size() < 4) return;
    uint32_t body_len = (uint32_t)bin[0] << 24 | (uint32_t)bin[1] << 16 |
                        (uint32_t)bin[2] << 8  | (uint32_t)bin[3];
    if (4 + body_len > bin.size()) return;
    std::vector<uint8_t> payload(bin.begin() + 4, bin.begin() + 4 + body_len);

    try {
        NetworkCommand n = parse_network_command(payload);
        std::cout << "[pico] cmd: " << n.command << std::endl;
        if (n.command == "OPEN_CAMERA") {
            CameraRequest cfg = parse_camera_request(n.data);
            std::cout << "[pico] open " << cfg.width << "x" << cfg.height
                      << " " << cfg.bitrate << "bps -> " << cfg.ip << ":" << cfg.port
                      << " (h" << (cfg.enableMvHevc ? "265" : "264") << ")" << std::endl;
            streaming_active.store(false);
            if (session_thread && session_thread->joinable()) session_thread->join();
            session_thread = std::make_unique<std::thread>(streaming_session, cfg);
        } else if (n.command == "CLOSE_CAMERA") {
            streaming_active.store(false);
        }
    } catch (const std::exception &e) {
        std::cerr << "[pico] parse: " << e.what() << std::endl;
    }
}

static void on_disconnect() {
    std::cout << "[pico] disconnect" << std::endl;
    streaming_active.store(false);
}

static void handle_sigint(int) {
    stop_requested.store(true);
    streaming_active.store(false);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    signal(SIGINT, handle_sigint);

    std::string listen_addr = "0.0.0.0:13579";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--listen" && i + 1 < argc) listen_addr = argv[++i];
        else if (a == "--help") {
            std::cout << "usage: " << argv[0] << " [--listen IP:PORT]\n"
                      << "  default: --listen 0.0.0.0:13579\n";
            return 0;
        }
    }

    oakd::g_running.store(true);
    std::thread oakd_thread(oakd::capture_loop);

    std::cout << "[main] listening " << listen_addr << std::endl;
    std::unique_ptr<TCPServer> server;
    while (!stop_requested.load()) {
        try {
            server = std::make_unique<TCPServer>(listen_addr);
            server->setDataCallback(on_command);
            server->setDisconnectCallback(on_disconnect);
            server->start();
            while (!stop_requested.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            server->stop();
            server.reset();
        } catch (const std::exception &e) {
            std::cerr << "[main] listen err: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    streaming_active.store(false);
    if (session_thread && session_thread->joinable()) session_thread->join();
    oakd::g_running.store(false);
    if (oakd_thread.joinable()) oakd_thread.join();
    std::cout << "[main] exit" << std::endl;
    return 0;
}
