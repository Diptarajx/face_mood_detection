#!/bin/bash

# Exit on error
set -e

echo "Building Face and Mood Detection program..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Check OpenCV installation
if ! pkg-config --exists opencv4; then
    echo "ERROR: OpenCV 4 not found!"
    echo "Please run ./setup_opencv.sh first or install OpenCV 4 using:"
    echo "sudo apt-get install libopencv-dev"
    exit 1
fi

# Get OpenCV version and flags
OPENCV_VERSION=$(pkg-config --modversion opencv4)
echo "Found OpenCV version: $OPENCV_VERSION"

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
echo "Building application..."
make -j$(nproc)

# Copy the model file if needed
if [ ! -f "emotion_ferplus_v2.onnx" ]; then
    echo "Copying emotion model file..."
    cp ../models/emotion_ferplus_v2.onnx .
fi

echo "Build complete! Run the program with:"
echo "./face_mood_detection --debug --scale 0.5 --verbose" 