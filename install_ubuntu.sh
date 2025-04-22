#!/bin/bash
set -e

echo "##################################"
echo "# Face Mood Detection Installer  #"
echo "# for Ubuntu Linux               #"
echo "##################################"

# Check if running as root (sudo)
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script with sudo privileges"
  echo "Example: sudo ./install_ubuntu.sh"
  exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y build-essential cmake pkg-config git
apt-get install -y libopencv-dev python3-opencv
apt-get install -y libgtk-3-dev
apt-get install -y libeigen3-dev
apt-get install -y libopenblas-dev

# Check OpenCV installation
echo "Checking OpenCV installation..."
if pkg-config --exists opencv4; then
  OPENCV_VERSION=$(pkg-config --modversion opencv4)
  echo "OpenCV $OPENCV_VERSION is installed"
else
  echo "OpenCV 4 is not installed. Installing now..."
  apt-get install -y libopencv-dev python3-opencv
  
  # Check again
  if ! pkg-config --exists opencv4; then
    echo "Warning: Could not install OpenCV 4 via apt. Would you like to build from source? (y/n)"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ]; then
      # User said yes
      echo "Running the OpenCV setup script..."
      chmod +x setup_opencv.sh
      ./setup_opencv.sh
    else
      echo "OpenCV installation skipped. The program may not work correctly."
    fi
  fi
fi

# Create build directory and compile
echo "Building the application..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# Copy the model file
if [ ! -f "emotion_ferplus_v2.onnx" ]; then
  echo "Copying emotion model file..."
  cp ../models/emotion_ferplus_v2.onnx .
fi

echo "Installation complete!"
echo "Run the application with: ./build/face_mood_detection --debug --scale 0.5 --verbose"
echo ""
echo "Note: If you encounter a camera access error, make sure your user has access to the camera device:"
echo "sudo usermod -a -G video $USER"
echo "Then log out and log back in for the changes to take effect." 