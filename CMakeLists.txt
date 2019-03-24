cmake_minimum_required (VERSION 3.02)
project (RayTracing)
# Uncomment for whichever compiler you use
# set(CMAKE_CC_COMPILER "g++-8")
# set(CMAKE_CC_COMPILER "g++")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CC_FLAGS "-g -O3 -Wall -c -MMD -MP")
include_directories($(RayTracing_SOURCE_DIR)/include)

add_executable(rtSerial src/Main.cpp src/Math.cpp src/RayTracer.cpp)

