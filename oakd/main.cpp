#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <glib-unix.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <openssl/md5.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <zmq.h>
#include <depthai/depthai.hpp>

#include "network_helper.hpp"

// ---------------------------------------------------------------------------
// HUD overlay state (reward flash + desk height)
// ---------------------------------------------------------------------------
// Receives commands from the laptop PICO streamer on ZMQ SUB port 5570.
// Messages are single-part strings: "REWARD" or "DESK:27.3"
// The overlay is drawn on the PICO video AFTER the ego_zmq publish, so the
// recorded dataset frames stay clean.
namespace hud {

static std::mutex g_mtx;
static std::chrono::steady_clock::time_point g_reward_until;
static std::string g_desk_height_text;
static void *g_ctx = nullptr;
static void *g_sub = nullptr;
static std::atomic<bool> g_running{false};
// Recorder pause state. When true, draw() paints a red border to indicate
// the recorder is NOT appending/saving. Toggled via "PAUSED" / "RUNNING"
// messages on the same SUB socket. Default is true (red border) so the
// operator sees the not-recording state until the recorder explicitly
// confirms RUNNING — fails safe if the recorder is down or starts late.
static std::atomic<bool> g_recorder_paused{true};

static void init(int port = 5570) {
  g_ctx = zmq_ctx_new();
  g_sub = zmq_socket(g_ctx, ZMQ_SUB);
  int linger = 0, rcvhwm = 4, rcvtimeo = 50;
  zmq_setsockopt(g_sub, ZMQ_LINGER, &linger, sizeof(linger));
  zmq_setsockopt(g_sub, ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
  zmq_setsockopt(g_sub, ZMQ_RCVTIMEO, &rcvtimeo, sizeof(rcvtimeo));
  zmq_setsockopt(g_sub, ZMQ_SUBSCRIBE, "", 0);
  std::string ep = "tcp://0.0.0.0:" + std::to_string(port);
  if (zmq_bind(g_sub, ep.c_str()) == 0) {
    std::cout << "[hud] Overlay SUB bound on " << ep << std::endl;
  } else {
    std::cerr << "[hud] bind failed: " << zmq_strerror(zmq_errno()) << std::endl;
  }
  g_running.store(true);
}

static void poll_thread() {
  char buf[256];
  while (g_running.load()) {
    int rc = zmq_recv(g_sub, buf, sizeof(buf) - 1, 0);
    if (rc <= 0) continue;
    buf[rc] = '\0';
    std::string msg(buf, rc);
    std::lock_guard<std::mutex> lock(g_mtx);
    if (msg == "REWARD") {
      g_reward_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    } else if (msg.substr(0, 5) == "DESK:") {
      g_desk_height_text = msg.substr(5);
    } else if (msg == "PAUSED") {
      g_recorder_paused.store(true);
      std::cout << "[hud] recorder PAUSED" << std::endl;
    } else if (msg == "RUNNING") {
      g_recorder_paused.store(false);
      std::cout << "[hud] recorder RUNNING" << std::endl;
    }
  }
}

static void draw(cv::Mat &frame) {
  std::lock_guard<std::mutex> lock(g_mtx);
  auto now = std::chrono::steady_clock::now();

  // Red border while the recorder is paused (not running).
  if (g_recorder_paused.load()) {
    int t = 24; // thicker than reward flash so it's unmistakable
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols - 1, frame.rows - 1),
                  cv::Scalar(0, 0, 255), t);
  }

  // Green border flash for reward
  if (now < g_reward_until) {
    int t = 8; // border thickness
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols - 1, frame.rows - 1),
                  cv::Scalar(0, 255, 0), t);
  }

  // Desk height text in bottom-right
  if (!g_desk_height_text.empty()) {
    std::string txt = "Desk: " + g_desk_height_text + "\"";
    int baseline = 0;
    double scale = 0.7;
    int thickness = 2;
    cv::Size sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    cv::Point org(frame.cols - sz.width - 12, frame.rows - 12);
    // Dark background for readability
    cv::rectangle(frame, cv::Point(org.x - 4, org.y - sz.height - 4),
                  cv::Point(org.x + sz.width + 4, org.y + baseline + 4),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, txt, org, cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(255, 255, 255), thickness);
  }
}

static void shutdown() {
  g_running.store(false);
  if (g_sub) { zmq_close(g_sub); g_sub = nullptr; }
  if (g_ctx) { zmq_ctx_term(g_ctx); g_ctx = nullptr; }
}

} // namespace hud

// ---------------------------------------------------------------------------
// ZMQ ego-view publisher
// ---------------------------------------------------------------------------
// Tees JPEG-compressed RGB frames to a PUB socket (bind to *:5558, topic
// "ego_view") so the laptop-side teleop recorder can log them as
// observation.images.ego_view without needing a second camera capture path.
// This runs alongside the existing NVENC H.264 -> PICO path; we publish the
// raw cv::Mat BEFORE it is fed into the GStreamer NVENC appsrc, so the headset
// stream is unaffected. Frame format:
//   multipart: [ topic_bytes, u64_le timestamp_ns, jpeg_bytes ]
// JPEG quality 85, encoded on the Jetson CPU via cv::imencode(".jpg", ...).
// No backpressure: we use ZMQ_PUB with a small send HWM so slow subscribers
// drop frames instead of blocking the capture loop.
namespace ego_zmq {

static void *g_ctx = nullptr;
static void *g_pub = nullptr;
static std::mutex g_mtx;
static const char *kTopic = "ego_view";
static const int kBindPort = 5558;
static int g_publish_count = 0;

// Async worker: single-slot queue. Capture loop writes, worker encodes+sends.
static cv::Mat g_pending_frame;
static uint64_t g_pending_capture_ts_ns{0};  // system_clock ns at camera capture
static std::mutex g_queue_mtx;
static std::condition_variable g_queue_cv;
static std::atomic<bool> g_running{false};

static bool init_pub() {
  std::lock_guard<std::mutex> lock(g_mtx);
  if (g_pub) return true;
  g_ctx = zmq_ctx_new();
  if (!g_ctx) {
    std::cerr << "[ego_zmq] zmq_ctx_new failed" << std::endl;
    return false;
  }
  g_pub = zmq_socket(g_ctx, ZMQ_PUB);
  if (!g_pub) {
    std::cerr << "[ego_zmq] zmq_socket failed" << std::endl;
    zmq_ctx_term(g_ctx);
    g_ctx = nullptr;
    return false;
  }
  int hwm = 4, linger = 0;
  zmq_setsockopt(g_pub, ZMQ_SNDHWM, &hwm, sizeof(hwm));
  zmq_setsockopt(g_pub, ZMQ_LINGER, &linger, sizeof(linger));
  std::string endpoint = std::string("tcp://*:") + std::to_string(kBindPort);
  if (zmq_bind(g_pub, endpoint.c_str()) != 0) {
    std::cerr << "[ego_zmq] bind " << endpoint << " failed: "
              << zmq_strerror(zmq_errno()) << std::endl;
    zmq_close(g_pub); g_pub = nullptr; zmq_ctx_term(g_ctx); g_ctx = nullptr;
    return false;
  }
  g_running.store(true);
  std::cout << "[ego_zmq] PUB bound on " << endpoint
            << " (topic=\"" << kTopic << "\", jpeg q=85, async)" << std::endl;
  return true;
}

// Worker thread: waits for a frame, JPEG-encodes, and publishes.
// Wire format: multipart [topic, u64_le timestamp_ns, jpeg_bytes]
// JPEG quality 85, encoded on the Jetson CPU via cv::imencode(".jpg", ...).
// Timestamp is the camera capture time (set by publish_bgr), not encode time.
static void worker_thread() {
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
  while (g_running.load()) {
    cv::Mat frame;
    uint64_t capture_ts_ns = 0;
    {
      std::unique_lock<std::mutex> lock(g_queue_mtx);
      g_queue_cv.wait_for(lock, std::chrono::milliseconds(100),
                          []{ return !g_pending_frame.empty() || !g_running.load(); });
      if (g_pending_frame.empty()) continue;
      frame = std::move(g_pending_frame);
      capture_ts_ns = g_pending_capture_ts_ns;
      g_pending_frame = cv::Mat();  // clear slot
    }
    if (!g_pub || frame.empty()) continue;
    std::vector<uint8_t> jpeg;
    try { if (!cv::imencode(".jpg", frame, jpeg, params)) continue; } catch (...) { continue; }
    uint8_t ts_le[8];
    for (int i = 0; i < 8; ++i) ts_le[i] = static_cast<uint8_t>((capture_ts_ns >> (8*i)) & 0xFF);
    if (zmq_send(g_pub, kTopic, std::strlen(kTopic), ZMQ_SNDMORE | ZMQ_DONTWAIT) < 0) continue;
    if (zmq_send(g_pub, ts_le, 8, ZMQ_SNDMORE | ZMQ_DONTWAIT) < 0) continue;
    zmq_send(g_pub, jpeg.data(), jpeg.size(), ZMQ_DONTWAIT);
    if ((++g_publish_count % 300) == 1) {
      std::cout << "[ego_zmq] published frame #" << g_publish_count
                << " (jpeg=" << jpeg.size() << "B, " << frame.cols << "x" << frame.rows << ")"
                << std::endl;
    }
  }
}

// Called from capture loop — queues a clone with capture timestamp (drops if worker is busy).
// capture_ts_ns: system_clock nanoseconds at frame capture (0 = stamp now).
static void publish_bgr(const cv::Mat &bgr, uint64_t capture_ts_ns = 0) {
  if (!g_pub || bgr.empty() || !g_running.load()) return;
  if (!capture_ts_ns) {
    capture_ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  }
  {
    std::lock_guard<std::mutex> lock(g_queue_mtx);
    g_pending_frame = bgr.clone();  // overwrites previous if worker hasn't consumed it
    g_pending_capture_ts_ns = capture_ts_ns;
  }
  g_queue_cv.notify_one();
}

static void shutdown_pub() {
  g_running.store(false);
  g_queue_cv.notify_all();
  std::lock_guard<std::mutex> lock(g_mtx);
  if (g_pub) { zmq_close(g_pub); g_pub = nullptr; }
  if (g_ctx) { zmq_ctx_term(g_ctx); g_ctx = nullptr; }
}

}  // namespace ego_zmq

// ---------------------------------------------------------------------------
// ZMQ video-pipeline timing publisher (port 5571, topic "video_timing")
// ---------------------------------------------------------------------------
// Publishes a compact fixed-size struct every frame so the laptop-side
// recorder can capture per-phase video-pipeline latency into the dataset.
// Format: multipart [topic_bytes, VideoTimingFrame struct (36 bytes LE)]
namespace video_timing_zmq {

struct __attribute__((packed)) VideoTimingFrame {
  double ts;          // system_clock seconds at capture
  int32_t recv_us;    // frame receive / decode
  int32_t tee_us;     // JPEG encode + ZMQ pub to recorder
  int32_t resize_us;  // cv::resize
  int32_t hud_us;     // HUD overlay draw
  int32_t enc_us;     // GStreamer/NVENC encode + appsrc push
  int32_t total_us;   // total per-frame pipeline time
};  // 32 bytes packed, little-endian on ARM64

static_assert(sizeof(VideoTimingFrame) == 32, "VideoTimingFrame must be 32 bytes");

static void *g_ctx = nullptr;
static void *g_pub = nullptr;
static const char *kTopic = "video_timing";
static const int kBindPort = 5571;

static bool init_pub() {
  g_ctx = zmq_ctx_new();
  if (!g_ctx) {
    std::cerr << "[video_timing_zmq] zmq_ctx_new failed" << std::endl;
    return false;
  }
  g_pub = zmq_socket(g_ctx, ZMQ_PUB);
  if (!g_pub) {
    std::cerr << "[video_timing_zmq] zmq_socket failed" << std::endl;
    zmq_ctx_term(g_ctx); g_ctx = nullptr;
    return false;
  }
  int hwm = 4, linger = 0;
  zmq_setsockopt(g_pub, ZMQ_SNDHWM, &hwm, sizeof(hwm));
  zmq_setsockopt(g_pub, ZMQ_LINGER, &linger, sizeof(linger));
  std::string ep = std::string("tcp://*:") + std::to_string(kBindPort);
  if (zmq_bind(g_pub, ep.c_str()) != 0) {
    std::cerr << "[video_timing_zmq] bind " << ep << " failed: "
              << zmq_strerror(zmq_errno()) << std::endl;
    zmq_close(g_pub); g_pub = nullptr; zmq_ctx_term(g_ctx); g_ctx = nullptr;
    return false;
  }
  std::cout << "[video_timing_zmq] PUB bound on " << ep << std::endl;
  return true;
}

static void publish_timing(const VideoTimingFrame &frame) {
  if (!g_pub) return;
  if (zmq_send(g_pub, kTopic, std::strlen(kTopic), ZMQ_SNDMORE | ZMQ_DONTWAIT) < 0) return;
  zmq_send(g_pub, &frame, sizeof(frame), ZMQ_DONTWAIT);
}

static void shutdown_pub() {
  if (g_pub) { zmq_close(g_pub); g_pub = nullptr; }
  if (g_ctx) { zmq_ctx_term(g_ctx); g_ctx = nullptr; }
}

}  // namespace video_timing_zmq

// Network Protocol Structures (same as ZED version)
struct CameraRequestData {
  int width;  int height;  int fps;  int bitrate;
  int enableMvHevc;  int renderMode;  int port;
  std::string camera;  std::string ip;
  CameraRequestData() : width(0), height(0), fps(0), bitrate(0),
    enableMvHevc(0), renderMode(0), port(0) {}
};

struct NetworkDataProtocol {
  std::string command;  int length;  std::vector<uint8_t> data;
  NetworkDataProtocol() : length(0) {}
  NetworkDataProtocol(const std::string &cmd, const std::vector<uint8_t> &d)
      : command(cmd), data(d), length(d.size()) {}
};

class CameraRequestDeserializer {
public:
  static CameraRequestData deserialize(const std::vector<uint8_t> &data) {
    if (data.size() < 10) throw std::invalid_argument("Data too small");
    size_t offset = 0;
    if (data[offset] != 0xCA || data[offset + 1] != 0xFE)
      throw std::invalid_argument("Invalid magic bytes");
    offset += 2;
    uint8_t version = data[offset++];
    if (version != 1) throw std::invalid_argument("Unsupported version");
    CameraRequestData result;
    if (offset + 28 > data.size()) throw std::invalid_argument("Too small");
    result.width = readInt32(data, offset);
    result.height = readInt32(data, offset + 4);
    result.fps = readInt32(data, offset + 8);
    result.bitrate = readInt32(data, offset + 12);
    result.enableMvHevc = readInt32(data, offset + 16);
    result.renderMode = readInt32(data, offset + 20);
    result.port = readInt32(data, offset + 24);
    offset += 28;
    result.camera = readCompactString(data, offset);
    result.ip = readCompactString(data, offset);
    return result;
  }
private:
  static int32_t readInt32(const std::vector<uint8_t> &data, size_t offset) {
    return static_cast<int32_t>((data[offset]) | (data[offset+1]<<8) |
      (data[offset+2]<<16) | (data[offset+3]<<24));
  }
  static std::string readCompactString(const std::vector<uint8_t> &data, size_t &offset) {
    uint8_t length = data[offset++];
    if (length == 0) return std::string();
    std::string result(reinterpret_cast<const char*>(&data[offset]), length);
    offset += length;
    return result;
  }
};

class NetworkDataProtocolDeserializer {
public:
  static NetworkDataProtocol deserialize(const std::vector<uint8_t> &buffer) {
    if (buffer.size() < 8) throw std::invalid_argument("Buffer too small");
    size_t offset = 0;
    int32_t cmdLen = readInt32(buffer, offset); offset += 4;
    std::string command;
    if (cmdLen > 0) {
      command = std::string(reinterpret_cast<const char*>(&buffer[offset]), cmdLen);
      size_t n = command.find('\0');
      if (n != std::string::npos) command = command.substr(0, n);
    }
    offset += cmdLen;
    int32_t dataLen = readInt32(buffer, offset); offset += 4;
    std::vector<uint8_t> data;
    if (dataLen > 0) data.assign(buffer.begin()+offset, buffer.begin()+offset+dataLen);
    return NetworkDataProtocol(command, data);
  }
private:
  static int32_t readInt32(const std::vector<uint8_t> &data, size_t offset) {
    return static_cast<int32_t>((data[offset]) | (data[offset+1]<<8) |
      (data[offset+2]<<16) | (data[offset+3]<<24));
  }
};

// Global state
CameraRequestData current_camera_config;
std::atomic<bool> stop_requested{false};
std::atomic<bool> streaming_active{false};
std::atomic<bool> encoding_enabled{false};
std::atomic<bool> send_enabled{false};
std::atomic<bool> preview_enabled{false};

std::unique_ptr<std::thread> listen_thread;
std::unique_ptr<std::thread> streaming_thread;
std::mutex config_mutex;
std::condition_variable streaming_cv;
std::mutex streaming_mutex;

std::unique_ptr<TCPClient> sender_ptr;
std::unique_ptr<TCPServer> server_ptr;
std::string send_to_server = "";
int send_to_port = 0;
int max_eye_width = 0;  // 0 = use PICO's requested size; >0 = cap per-eye width (e.g. 480)

// OAK-D capture thread — single global owner of the camera, publishes to
// ego_zmq. Streaming thread reads frames from g_frame.
// Uses DepthAI: ColorCamera (CAM_A) only — RGB 640x400 BGR8 output.
namespace rs2_capture {
  static cv::Mat g_frame;
  static std::mutex g_mtx;
  static std::atomic<bool> g_running{false};
  static std::atomic<int> g_frame_seq{0};

  static cv::Mat get_frame() {
    std::lock_guard<std::mutex> lock(g_mtx);
    cv::Mat out;
    std::swap(out, g_frame);  // zero-copy transfer, g_frame becomes empty
    return out;
  }

  static void capture_thread() {
    using namespace dai;

    Pipeline pipeline;

    auto cam = pipeline.create<node::ColorCamera>();
    cam->setBoardSocket(CameraBoardSocket::CAM_A);
    // OAK-D Pro W OV9782 variant: all sockets are OV9782 (1280x800 max).
    cam->setResolution(node::ColorCamera::Properties::SensorResolution::THE_800_P);
    cam->setIspScale(1, 2);  // 1280x800 -> 640x400
    cam->setColorOrder(node::ColorCamera::Properties::ColorOrder::BGR);
    cam->setInterleaved(false);
    cam->setFps(60);

    auto xRgb = pipeline.create<node::XLinkOut>();
    xRgb->setStreamName("rgb");
    cam->isp.link(xRgb->input);

    std::unique_ptr<Device> device;
    try {
      device = std::make_unique<Device>(pipeline);
      std::cout << "[oakd_capture] Pipeline started (RGB 640x400 @60, USB="
                << device->getUsbSpeed() << ")" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[oakd_capture] Failed to start: " << e.what() << std::endl;
      return;
    }

    auto qRgb = device->getOutputQueue("rgb", 4, false);

    while (g_running.load() && !stop_requested.load()) {
      auto rgbFrame = qRgb->tryGet<ImgFrame>();
      if (!rgbFrame) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      uint64_t capture_ts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

      cv::Mat bgr = rgbFrame->getCvFrame();
      ego_zmq::publish_bgr(bgr, capture_ts_ns);
      {
        std::lock_guard<std::mutex> lock(g_mtx);
        g_frame = bgr.clone();
      }
      g_frame_seq.fetch_add(1);
    }

    std::cout << "[oakd_capture] Stopped" << std::endl;
  }
}  // namespace rs2_capture

template<typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

bool initialize_sender() {
  int retry = 10;
  while (retry > 0 && !sender_ptr && !stop_requested.load()) {
    try {
      sender_ptr = std::unique_ptr<TCPClient>(new TCPClient(send_to_server, send_to_port));
      std::cout << "Connecting to " << send_to_server << ":" << send_to_port << std::endl;
      sender_ptr->connect();
      return true;
    } catch (const TCPException &e) {
      std::cerr << "Connect failed: " << e.what() << std::endl;
      sender_ptr = nullptr;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    retry--;
  }
  return false;
}

// Forward declarations
void handleOpenCamera(const std::vector<uint8_t> &data);
void handleCloseCamera(const std::vector<uint8_t> &data);
void startStreamingThread();
void stopStreamingThread();
void streamingThreadFunction();
void listenThreadFunction(const std::string &listen_address);

void onDataCallback(const std::string &command) {
  std::vector<uint8_t> binaryData(command.begin(), command.end());
  if (binaryData.size() < 4) return;

  uint32_t bodyLength = (static_cast<uint32_t>(binaryData[0]) << 24) |
    (static_cast<uint32_t>(binaryData[1]) << 16) |
    (static_cast<uint32_t>(binaryData[2]) << 8) |
    static_cast<uint32_t>(binaryData[3]);

  if (4 + bodyLength > binaryData.size()) return;
  std::vector<uint8_t> protocolData(binaryData.begin()+4, binaryData.begin()+4+bodyLength);

  try {
    NetworkDataProtocol protocol = NetworkDataProtocolDeserializer::deserialize(protocolData);
    std::cout << "Command: '" << protocol.command << "'" << std::endl;
    if (protocol.command == "OPEN_CAMERA") handleOpenCamera(protocol.data);
    else if (protocol.command == "CLOSE_CAMERA") handleCloseCamera(protocol.data);
  } catch (const std::exception &e) {
    std::cout << "Parse error: " << e.what() << std::endl;
  }
}

void onDisconnectCallback() {
  std::cout << "Client disconnected" << std::endl;
  stopStreamingThread();
}

void listenThreadFunction(const std::string &listen_address) {
  std::cout << "Listening on " << listen_address << std::endl;
  while (!stop_requested.load()) {
    try {
      server_ptr = make_unique_helper<TCPServer>(listen_address);
      server_ptr->setDataCallback(onDataCallback);
      server_ptr->setDisconnectCallback(onDisconnectCallback);
      server_ptr->start();
      while (!stop_requested.load() && server_ptr)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (server_ptr) { server_ptr->stop(); server_ptr = nullptr; }
      if (!stop_requested.load()) std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (const std::exception &e) {
      std::cerr << "Listen error: " << e.what() << std::endl;
      if (!stop_requested.load()) std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }
}

void handle_sigint(int) {
  stop_requested.store(true);
  stopStreamingThread();
  if (server_ptr) { server_ptr->stop(); server_ptr = nullptr; }
  streaming_cv.notify_all();
}

GstFlowReturn on_new_sample(GstAppSink *sink, gpointer) {
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample) return GST_FLOW_ERROR;
  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    if (send_enabled.load() && sender_ptr && sender_ptr->isConnected() && map.data && map.size > 0) {
      try {
        std::vector<uint8_t> packet(4 + map.size);
        packet[0] = (map.size >> 24) & 0xFF;
        packet[1] = (map.size >> 16) & 0xFF;
        packet[2] = (map.size >> 8) & 0xFF;
        packet[3] = (map.size) & 0xFF;
        std::copy(map.data, map.data + map.size, packet.begin() + 4);
        sender_ptr->sendData(packet);
      } catch (const std::exception &e) {
        std::cerr << "[on_new_sample] TCP send error: " << e.what() << std::endl;
        streaming_active.store(false);
      } catch (...) {
        std::cerr << "[on_new_sample] TCP send error (unknown)" << std::endl;
        streaming_active.store(false);
      }
    }
    gst_buffer_unmap(buffer, &map);
  }
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

void handleOpenCamera(const std::vector<uint8_t> &data) {
  std::cout << "OPEN_CAMERA received" << std::endl;
  try {
    CameraRequestData config = CameraRequestDeserializer::deserialize(data);
    std::cout << "Request: " << config.width << "x" << config.height
              << " @ " << config.fps << "fps, " << config.bitrate << "bps"
              << ", type=" << config.camera
              << ", target=" << config.ip << ":" << config.port << std::endl;

    // Accept any camera type — we'll serve RealSense frames
    {
      std::lock_guard<std::mutex> lock(config_mutex);
      current_camera_config = config;
    }
    send_to_server = config.ip;
    send_to_port = config.port;
    startStreamingThread();
  } catch (const std::exception &e) {
    std::cerr << "Parse error: " << e.what() << std::endl;
  }
}

void handleCloseCamera(const std::vector<uint8_t> &data) {
  std::cout << "CLOSE_CAMERA received" << std::endl;
  stopStreamingThread();
}

void startStreamingThread() {
  std::lock_guard<std::mutex> lock(streaming_mutex);
  if (streaming_thread && streaming_thread->joinable()) return;
  streaming_active.store(true);
  streaming_thread = make_unique_helper<std::thread>(streamingThreadFunction);
}

void stopStreamingThread() {
  std::lock_guard<std::mutex> lock(streaming_mutex);
  streaming_active.store(false);
  encoding_enabled.store(false);
  send_enabled.store(false);
  if (sender_ptr && sender_ptr->isConnected()) sender_ptr->disconnect();
  sender_ptr = nullptr;
  if (streaming_thread && streaming_thread->joinable()) {
    streaming_cv.notify_all();
    streaming_thread->join();
    streaming_thread = nullptr;
  }
}

void streamingThreadFunction() {
  std::cout << "Streaming thread started" << std::endl;
  try {
    if (!initialize_sender()) {
      std::cerr << "Failed to connect, stopping" << std::endl;
      return;
    }
    encoding_enabled.store(true);
    send_enabled.store(true);

    CameraRequestData config;
    { std::lock_guard<std::mutex> lock(config_mutex); config = current_camera_config; }

    // Use requested dimensions or defaults
    int cap_w = (config.width > 0) ? config.width : 1280;
    int cap_h = (config.height > 0) ? config.height : 720;
    int cap_fps = 60;
    int bitrate = (config.bitrate > 0) ? config.bitrate : 4000000;

    // The PICO requests a wide stereo frame (e.g. 2560x720 or 2160x810).
    // We have a mono RealSense. Detect stereo by aspect ratio > 2:1 and
    // duplicate the image side-by-side for both eyes.
    int out_w = cap_w;
    int out_h = cap_h;
    bool duplicate_stereo = false;
    if (cap_w > cap_h * 2 - 10) {
      // Stereo request — capture at half width (one eye), duplicate later
      cap_w = out_w / 2;
      cap_h = out_h;
      duplicate_stereo = true;
      // Optionally cap per-eye resolution for lower latency
      if (max_eye_width > 0 && cap_w > max_eye_width) {
        double scale = (double)max_eye_width / cap_w;
        cap_w = max_eye_width;
        cap_h = (int)(cap_h * scale);
        // Round to even for encoder
        cap_h = (cap_h + 1) & ~1;
        out_w = cap_w * 2;
        out_h = cap_h;
      }
      std::cout << "Stereo mode: per-eye " << cap_w << "x" << cap_h
                << ", output " << out_w << "x" << out_h << std::endl;
    }

    std::cout << "Reading frames from OAK-D capture thread" << std::endl;

    // Build GStreamer pipeline
    std::string encoder = config.enableMvHevc ? "nvv4l2h265enc" : "nvv4l2h264enc";
    std::string parser = config.enableMvHevc ? "h265parse" : "h264parse";

    std::string pipeline_str =
      "appsrc name=mysource is-live=true format=time "
      "caps=video/x-raw,format=BGR,width=" + std::to_string(out_w) +
      ",height=" + std::to_string(out_h) +
      ",framerate=" + std::to_string(cap_fps) + "/1 ! "
      "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
      "tee name=t "
      "t. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
      + encoder + " maxperf-enable=1 insert-sps-pps=true "
      "idrinterval=30 bitrate=" + std::to_string(bitrate) +
      " preset-level=UltraFastPreset ! " +
      parser + " ! appsink name=mysink emit-signals=true sync=false ";

    if (preview_enabled.load()) {
      pipeline_str += "t. ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! nvvidconv ! videoconvert ! autovideosink sync=false ";
    }

    std::cout << "Pipeline: " << pipeline_str << std::endl;

    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
      std::cerr << "Pipeline error: " << error->message << std::endl;
      g_clear_error(&error);
      return;
    }

    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    cv::Mat frame, stereo_frame;
    int frame_id = 0;

    // Timing accumulators (microseconds)
    double t_recv_sum = 0, t_tee_sum = 0, t_resize_sum = 0;
    double t_hud_sum = 0, t_enc_sum = 0, t_total_sum = 0;
    int t_count = 0;
    const int T_REPORT_EVERY = 100;
    auto us_now = []() {
      return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    std::cout << "Streaming OAK-D→PICO..."
              << " (capture=" << cap_w << "x" << cap_h << " output=" << out_w << "x" << out_h << ")" << std::endl;
    while (streaming_active.load() && !stop_requested.load()) {
      long t0 = us_now();

      bool got_frame = false;
      static int last_seq = 0;
      int seq = rs2_capture::g_frame_seq.load();
      if (seq != last_seq) {
        frame = rs2_capture::get_frame();
        got_frame = !frame.empty();
        last_seq = seq;
      }
      if (!got_frame) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      long t1 = us_now();  // after recv/decode

      // ego_zmq publishing is handled by the OAK-D capture thread itself.
      long t2 = us_now();

      // Resize to match PICO per-eye dimensions if needed
      if (frame.cols != cap_w || frame.rows != cap_h) {
        cv::resize(frame, frame, cv::Size(cap_w, cap_h));
      }
      long t3 = us_now();  // after resize

      cv::Mat *out_frame = &frame;

      // If stereo mode, simply duplicate the mono RGB into both eyes side-by-side.
      // (Depth-based right-eye synthesis was removed — only the RGB ego-view
      // path remains, and the PICO gets two copies of the same view.)
      if (duplicate_stereo) {
        cv::Mat left_eye;
        if (frame.cols != cap_w || frame.rows != cap_h)
          cv::resize(frame, left_eye, cv::Size(cap_w, cap_h));
        else
          left_eye = frame;

        hud::draw(left_eye);
        stereo_frame = cv::Mat(out_h, out_w, frame.type());
        left_eye.copyTo(stereo_frame(cv::Rect(0, 0, cap_w, cap_h)));
        left_eye.copyTo(stereo_frame(cv::Rect(cap_w, 0, cap_w, cap_h)));
        out_frame = &stereo_frame;
      } else {
        // Non-stereo mode: draw HUD directly
        hud::draw(frame);
      }
      long t4 = us_now();  // after HUD/duplication

      if (encoding_enabled.load()) {
        size_t buf_size = out_frame->total() * out_frame->elemSize();
        GstBuffer *buffer = gst_buffer_new_allocate(nullptr, buf_size, nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, out_frame->data, buf_size);
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frame_id, GST_SECOND, cap_fps);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, cap_fps);
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
        frame_id++;
      }
      long t5 = us_now();  // after encode push

      // Publish per-frame timing on ZMQ for the recorder
      {
        video_timing_zmq::VideoTimingFrame tf;
        tf.ts = std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count();
        tf.recv_us   = static_cast<int32_t>(t1 - t0);
        tf.tee_us    = static_cast<int32_t>(t2 - t1);
        tf.resize_us = static_cast<int32_t>(t3 - t2);
        tf.hud_us    = static_cast<int32_t>(t4 - t3);
        tf.enc_us    = static_cast<int32_t>(t5 - t4);
        tf.total_us  = static_cast<int32_t>(t5 - t0);
        video_timing_zmq::publish_timing(tf);
      }

      // Accumulate timings
      t_recv_sum += (t1 - t0); t_tee_sum += (t2 - t1); t_resize_sum += (t3 - t2);
      t_hud_sum += (t4 - t3); t_enc_sum += (t5 - t4);
      t_total_sum += (t5 - t0);
      if (++t_count >= T_REPORT_EVERY) {
        double n = t_count;
        std::cout << "[timing] avg over " << t_count << " frames (us): "
                  << "recv=" << (int)(t_recv_sum/n)
                  << " tee=" << (int)(t_tee_sum/n)
                  << " resize=" << (int)(t_resize_sum/n)
                  << " hud=" << (int)(t_hud_sum/n)
                  << " enc=" << (int)(t_enc_sum/n)
                  << " TOTAL=" << (int)(t_total_sum/n)
                  << " (fps=" << std::fixed << std::setprecision(1) << (1e6 * n / t_total_sum) << ")"
                  << std::endl;
        t_recv_sum = t_tee_sum = t_resize_sum = t_hud_sum = t_enc_sum = t_total_sum = 0;
        t_count = 0;
      }
    }

    std::cout << "Stopping stream..." << std::endl;
    gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsrc);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
  } catch (const std::exception &e) {
    std::cerr << "Streaming error: " << e.what() << std::endl;
  }
  std::cout << "Streaming thread finished" << std::endl;
}

int main(int argc, char *argv[]) {
  gst_init(&argc, &argv);
  signal(SIGINT, handle_sigint);

  // Bind the ZMQ ego-view publisher once at startup. This is independent of
  // the PICO TCP listener so the laptop recorder can connect and receive
  // frames as soon as any streaming session is active, regardless of whether
  // the headset has sent OPEN_CAMERA yet. If binding fails (port in use),
  // we log and continue — the H.264 -> PICO path must not be blocked.
  if (!ego_zmq::init_pub()) {
    std::cerr << "[ego_zmq] failed to initialize PUB socket; continuing "
                 "without laptop ego-view tee" << std::endl;
  }
  std::thread ego_worker(ego_zmq::worker_thread);

  if (!video_timing_zmq::init_pub()) {
    std::cerr << "[video_timing_zmq] failed to initialize PUB socket; continuing "
                 "without video timing publisher" << std::endl;
  }

  // Start the OAK-D capture thread (sole owner of the camera).
  rs2_capture::g_running.store(true);
  std::thread rs2_thread(rs2_capture::capture_thread);

  // Start HUD overlay receiver (port 5570)
  hud::init();
  std::thread hud_thread(hud::poll_thread);

  bool listen_enabled = false;
  std::string listen_address = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--preview") preview_enabled.store(true);
    else if (arg == "--listen" && i+1 < argc) { listen_enabled = true; listen_address = argv[++i]; }
    else if (arg == "--send") { send_enabled.store(true); }
    else if (arg == "--server" && i+1 < argc) { send_to_server = argv[++i]; }
    else if (arg == "--port" && i+1 < argc) { send_to_port = std::stoi(argv[++i]); }
    else if (arg == "--max-eye-width" && i+1 < argc) { max_eye_width = std::stoi(argv[++i]); }
    else if (arg == "--help") {
      std::cout << "OrinVideoSender (OAK-D) for XRoboToolkit\n"
                << "  --listen ADDR      Listen for PICO commands (IP:PORT)\n"
                << "  --max-eye-width N  Cap per-eye width for PICO stream (e.g. 480)\n"
                << "  --preview          Show local preview\n"
                << "  --send             Direct send mode\n"
                << "  --server IP        Target IP\n"
                << "  --port PORT        Target port\n";
      return 0;
    }
  }

  if (!listen_enabled && !send_enabled.load()) {
    std::cerr << "Need --listen or --send. Use --help." << std::endl;
    return -1;
  }

  if (listen_enabled) {
    std::cout << "OrinVideoSender listening on " << listen_address << std::endl;
    listen_thread = make_unique_helper<std::thread>(listenThreadFunction, listen_address);
    while (!stop_requested.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (listen_thread && listen_thread->joinable()) listen_thread->join();
  } else {
    std::lock_guard<std::mutex> lock(config_mutex);
    current_camera_config.width = 1280;
    current_camera_config.height = 720;
    current_camera_config.fps = 30;
    current_camera_config.bitrate = 4000000;
    current_camera_config.ip = send_to_server;
    current_camera_config.port = send_to_port;
    startStreamingThread();
    while (!stop_requested.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  stopStreamingThread();
  rs2_capture::g_running.store(false);
  if (rs2_thread.joinable()) rs2_thread.join();
  ego_zmq::shutdown_pub();
  if (ego_worker.joinable()) ego_worker.join();
  video_timing_zmq::shutdown_pub();
  hud::shutdown();
  if (hud_thread.joinable()) hud_thread.join();
  std::cout << "Exiting." << std::endl;
  return 0;
}
