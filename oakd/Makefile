###############################################################################
#
# Standalone Makefile for OrinVideoSender
# Self-contained build configuration
# By liuchuan.yu@bytedance.com
#
###############################################################################

# Compiler settings
CXX = g++
APP := OrinVideoSender

SRCS := main.cpp

OBJS := $(SRCS:.cpp=.o)

# Include paths
CPPFLAGS := -std=c++17 -I/usr/local/include -I/usr/local/include/depthai-shared/3rdparty -I/usr/local/lib/cmake/depthai/dependencies/include \
	-I/usr/include/opencv4 \
	$(shell pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 glib-2.0 2>/dev/null || echo "") \
	$(shell pkg-config --cflags libzmq 2>/dev/null || echo "")

# Compiler flags
CXXFLAGS := -Wall -Wextra -O2 -g

# FFmpeg flags
CXXFLAGS += $(shell pkg-config --cflags libavcodec libavformat libavutil libswscale libavdevice 2>/dev/null || echo "")

# Library paths and libraries
LDFLAGS :=

# FFmpeg libraries (must come first to avoid conflicts)
LDFLAGS += $(shell pkg-config --libs libavcodec libavformat libavutil libswscale libavdevice 2>/dev/null || echo "-lavcodec -lavformat -lavutil -lswscale -lavdevice")

# Core libraries
LDFLAGS += \
	-lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs \
	-lssl -lcrypto \
	-lpthread \
	-lstdc++

# GStreamer libraries
LDFLAGS += $(shell pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 glib-2.0 2>/dev/null || echo "-lgstreamer-1.0 -lgstapp-1.0 -lglib-2.0")

# ZMQ library
LDFLAGS += $(shell pkg-config --libs libzmq 2>/dev/null || echo "-lzmq")
LDFLAGS += -L/usr/local/lib -ldepthai-core -ldepthai-opencv

all: $(APP) streaming_only

debug: CXXFLAGS += -DDEBUG -g3 -O0
debug: $(APP)

%.o: %.cpp
	@echo "Compiling: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(APP): $(OBJS)
	@echo "Linking: $@"
	$(CXX) -o $@ $(OBJS) $(LDFLAGS)

streaming_only: streaming_only.cpp network_helper.hpp
	@echo "Linking: $@"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ streaming_only.cpp $(LDFLAGS)

clean:
	rm -rf $(APP) $(OBJS) streaming_only

install: $(APP)
	@echo "Installing $(APP)..."
	install -D $(APP) /usr/local/bin/$(APP)

.PHONY: all debug clean install
