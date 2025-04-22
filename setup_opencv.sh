#!/bin/bash

# Exit on error
set -e

echo "Starting OpenCV with Face module installation for Ubuntu"

# Update and install dependencies
echo "Updating system and installing dependencies..."
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libfontconfig1-dev libcairo2-dev
sudo apt-get install -y libgdk-pixbuf2.0-dev libpango1.0-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y python3-dev
# Ubuntu-specific packages
sudo apt-get install -y libeigen3-dev
sudo apt-get install -y libopenblas-dev
sudo apt-get install -y libcanberra-gtk-module

# Create a directory for OpenCV build
mkdir -p ~/opencv_build
cd ~/opencv_build

# Clone OpenCV and OpenCV contrib repositories
echo "Cloning OpenCV repositories..."
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Checkout a stable version (4.5.5 is a good balance between features and stability)
cd opencv
git checkout 4.5.5
cd ../opencv_contrib
git checkout 4.5.5
cd ..

# Create build directory
mkdir -p opencv/build
cd opencv/build

# Check for CUDA support
if [ -x "$(command -v nvcc)" ]; then
    echo "CUDA found, enabling GPU acceleration..."
    CUDA_OPTIONS="-D WITH_CUDA=ON \
                 -D OPENCV_DNN_CUDA=ON \
                 -D ENABLE_FAST_MATH=1 \
                 -D CUDA_FAST_MATH=1 \
                 -D WITH_CUBLAS=1"
else
    echo "CUDA not found, building without GPU acceleration..."
    CUDA_OPTIONS="-D WITH_CUDA=OFF"
fi

# Configure OpenCV build with face module
echo "Configuring OpenCV build with face module..."
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D ENABLE_NEON=ON \
      -D BUILD_TESTS=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D BUILD_opencv_dnn=ON \
      $CUDA_OPTIONS ..

# Determine number of CPU cores for parallel build
NUM_CORES=$(nproc)
echo "Building with $NUM_CORES cores..."

# Build OpenCV
echo "Building OpenCV (this will take a while)..."
make -j"$NUM_CORES"

# Install OpenCV
echo "Installing OpenCV..."
sudo make install
sudo ldconfig

echo "OpenCV with face module has been installed successfully!"
echo "You can now build the face and mood detection program." 