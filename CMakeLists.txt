cmake_minimum_required(VERSION 3.10)
project(FaceMoodDetection)

# Set C++ standard to C++17 for filesystem
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(face_mood_detection face_mood_detection.cpp)

# Link libraries
target_link_libraries(face_mood_detection ${OpenCV_LIBS})

# Copy model files to build directory
file(COPY models/emotion_ferplus.onnx DESTINATION ${CMAKE_BINARY_DIR})

# Installation
install(TARGETS face_mood_detection DESTINATION bin)
install(FILES models/emotion_ferplus.onnx DESTINATION bin) 