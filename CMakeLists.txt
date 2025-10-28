cmake_minimum_required(VERSION 3.28.3)
project("Algoritmi Esame")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(AlgoritmiEsame src/Split&Merge.cpp)

target_link_libraries(AlgoritmiEsame ${OpenCV_LIBS})