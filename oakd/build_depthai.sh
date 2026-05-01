#!/usr/bin/env bash
# Build and install depthai-core (C++ SDK for Luxonis OAK-D cameras).
# Run once on the Orin before `make`-ing OrinVideoSender.
set -euo pipefail

DEPTHAI_DIR="${DEPTHAI_DIR:-$HOME/depthai-core}"
DEPTHAI_TAG="${DEPTHAI_TAG:-v2.29.0}"   # v2 API — v3 reorganized XLinkOut/getOutputQueue
JOBS="${JOBS:-$(nproc)}"

echo "[build_depthai] Installing build dependencies..."
sudo apt update
sudo apt install -y git cmake build-essential libusb-1.0-0-dev

if [ ! -d "$DEPTHAI_DIR" ]; then
    echo "[build_depthai] Cloning depthai-core $DEPTHAI_TAG into $DEPTHAI_DIR..."
    git clone --branch "$DEPTHAI_TAG" --recursive --depth 1 https://github.com/luxonis/depthai-core.git "$DEPTHAI_DIR"
else
    echo "[build_depthai] $DEPTHAI_DIR exists, checking out $DEPTHAI_TAG..."
    git -C "$DEPTHAI_DIR" fetch --tags --depth 1 origin "$DEPTHAI_TAG"
    git -C "$DEPTHAI_DIR" checkout "$DEPTHAI_TAG"
    git -C "$DEPTHAI_DIR" submodule update --init --recursive
fi

cd "$DEPTHAI_DIR"

echo "[build_depthai] Configuring CMake..."
cmake -S . -B build \
    -D BUILD_SHARED_LIBS=ON \
    -D DEPTHAI_BUILD_EXAMPLES=OFF \
    -D DEPTHAI_BUILD_TESTS=OFF \
    -D DEPTHAI_OPENCV_SUPPORT=ON \
    -D DEPTHAI_ENABLE_CURL=OFF \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D CMAKE_BUILD_TYPE=Release

echo "[build_depthai] Building with $JOBS jobs (this takes ~15 min on Orin)..."
cmake --build build --parallel "$JOBS"

echo "[build_depthai] Installing to /usr/local..."
sudo cmake --install build
sudo ldconfig

echo "[build_depthai] Installing udev rule for OAK USB device..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/80-movidius.rules > /dev/null
sudo udevadm control --reload
sudo udevadm trigger

echo "[build_depthai] Verifying install..."
for f in \
    /usr/local/include/depthai/depthai.hpp \
    /usr/local/lib/libdepthai-core.so \
    /usr/local/lib/libdepthai-opencv.so
do
    if [ -e "$f" ]; then
        echo "  OK  $f"
    else
        echo "  MISSING  $f"
        exit 1
    fi
done

echo "[build_depthai] Done. Unplug and replug the OAK-D once, then run 'make' in this directory."
