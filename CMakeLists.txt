cmake_minimum_required (VERSION 3.02)
project (RayTracing)
# Uncomment for whichever compiler you use
# set(CMAKE_CC_COMPILER "g++-8")
# set(CMAKE_CC_COMPILER "g++")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-g -O3 -Wall -MMD -MP -fopenmp -std=c++11")
include_directories($(RayTracing_SOURCE_DIR)include)

add_executable(rtSerial src/serial/Main.cpp src/Math.cpp src/serial/RayTracer.cpp src/Output.cpp)
add_executable(rtOpenMP src/openmp/Main.cpp src/Math.cpp src/openmp/RayTracer.cpp src/Output.cpp)
# Need to compile using nvcc
# add_executable(rtCuda src/cuda/Main.cpp src/cuda/cudaRT.cpp src/Math.cpp src/Output.cpp)
