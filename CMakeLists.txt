cmake_minimum_required(VERSION 3.16)
project(HandGestureDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)
find_package(SDL2_mixer REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    ${SDL2_MIXER_INCLUDE_DIRS}
)

# MediaPipe dependencies (Assuming you built MediaPipe separately and include it)
include_directories(${PROJECT_SOURCE_DIR}/mediapipe_include)
link_directories(${PROJECT_SOURCE_DIR}/mediapipe_lib)

add_executable(HandGesture main.cpp)

target_link_libraries(HandGesture
    ${OpenCV_LIBS}
    SDL2::SDL2
    SDL2_mixer::SDL2_mixer
    libmediapipe_gpu.so
    libmediapipe_framework.so
    libabsl_synchronization.so
    libprotobuf.so
)
