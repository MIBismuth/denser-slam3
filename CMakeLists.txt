cmake_minimum_required(VERSION 3.10)
project(orbimp)
add_definitions(-DUSE_MONOTONIC)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-deprecated-declarations -D_GLIBCXX_USE_CXX11_ABI=1")
# Specify the path to your ORB_SLAM3 installation
set(ORB_SLAM3_DIR $ENV{ORB_SLAM3_ROOT_DIR})  # Default path, adjust as needed
# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
# Include directories
include_directories(
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2
    ${ORB_SLAM3_DIR}/Thirdparty/g2o
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
# Add source files
set(SOURCES
    src/main.cpp
    src/SharedMemoryInterface.cpp
)

# Add executable
add_executable(orbimp ${SOURCES})

# Link against ORB_SLAM3 library
target_link_libraries(orbimp
    ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    -lboost_system
    -lcrypto
)
