cmake_minimum_required(VERSION 3.19)
project(HW6)

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(src)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})