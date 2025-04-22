cmake_minimum_required(VERSION 3.10)
project(face_mood_detection)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Raspberry Pi specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    add_compile_options(-O3 -march=armv8-a -mtune=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard)
    add_definitions(-DUSE_NEON)
endif()

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Add executable
add_executable(face_mood_detection face_mood_detection.cpp)

# Link libraries
target_link_libraries(face_mood_detection
    ${OpenCV_LIBS}
    Threads::Threads
)

# Set output directory
set_target_properties(face_mood_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Install target
install(TARGETS face_mood_detection
    RUNTIME DESTINATION bin
) 