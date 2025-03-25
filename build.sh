#!/bin/bash

# Exit on error
set -e

echo "Building Face and Mood Detection program..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make

echo "Build complete! Run the program with: ./face_mood_detection" 