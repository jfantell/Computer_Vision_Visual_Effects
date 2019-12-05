cmake_minimum_required(VERSION 3.15)
project(PointCloud2Mesh)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

set(CMAKE_CXX_STANDARD 14)
find_library(REALSENSE2 realsense2 REQUIRED)
find_package(PCL 1.3 REQUIRED)

# Per PCL documentation
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})

set(LIBRARIES ${PCL_LIBRARIES} ${REALSENSE2})

add_executable(cloud2mesh src/cloud2mesh.cpp src/pcl_wrapper.cpp include/pcl_wrapper.h)
target_link_libraries(cloud2mesh ${LIBRARIES})
