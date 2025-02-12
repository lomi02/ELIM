cmake_minimum_required(VERSION 3.28.3)
project("Algoritmi Esame")

set(CMAKE_CXX_STANDARD 26)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(AlgoritmiEsame Blackboard.cpp)

target_link_libraries(AlgoritmiEsame ${OpenCV_LIBS})