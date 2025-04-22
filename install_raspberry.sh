#!/bin/bash

# Exit on error
set -e

echo "#######################################"
echo "# Face Expression Recognition System  #"
echo "# Raspberry Pi 5 Installer            #"
echo "#######################################"

# Check if running as root (sudo)
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo privileges"
  echo "Example: sudo ./install_raspberry.sh"
  exit 1
fi

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
  echo "This script is intended for Raspberry Pi only"
  exit 1
fi

echo "Installing dependencies for Raspberry Pi..."

# Update package list
sudo apt-get update

# Install required packages
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    python3-pyqt5 \
    libqt4-test

# Enable Raspberry Pi camera interface
echo "Enabling Raspberry Pi camera..."
if ! grep -q "start_x=1" /boot/config.txt; then
  echo "Updating /boot/config.txt to enable camera..."
  # Backup config
  cp /boot/config.txt /boot/config.txt.bak
  # Add camera config if not present
  echo "start_x=1" >> /boot/config.txt
  echo "gpu_mem=128" >> /boot/config.txt
fi

# Check OpenCV installation
echo "Checking OpenCV installation..."
if pkg-config --exists opencv4; then
  OPENCV_VERSION=$(pkg-config --modversion opencv4)
  echo "OpenCV $OPENCV_VERSION is installed"
else
  echo "OpenCV 4 is not installed. Installing now..."
  sudo apt-get install -y libopencv-dev python3-opencv
  
  # Check again
  if ! pkg-config --exists opencv4; then
    echo "Warning: Could not install OpenCV 4 via apt."
    echo "Would you like to build OpenCV from source? This will take a long time (y/n)"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ]; then
      # User said yes
      echo "Running the OpenCV setup script..."
      chmod +x setup_opencv.sh
      sudo ./setup_opencv.sh
    else
      echo "OpenCV installation skipped. The program may not work correctly."
    fi
  fi
fi

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=armv8-a -mtune=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard"

# Build with all available cores
make -j$(nproc)

# Copy the model file
if [ ! -f "emotion_ferplus_v2.onnx" ]; then
  echo "Copying emotion model file..."
  mkdir -p models
  cp ../models/emotion_ferplus_v2.onnx .
fi

echo "Installation complete!"
echo "To run the program, use: ./face_mood_detection --camera=pi --scale=0.4"
echo ""
echo "Note: You may need to reboot for camera changes to take effect:"
echo "sudo reboot" 