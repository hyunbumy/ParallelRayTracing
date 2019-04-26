cmake_minimum_required (VERSION 2.6)
project (RayTracing)

# Uncomment for whichever compiler you use

# set(CMAKE_CC_COMPILER "g++")

include_directories($(RayTracing_SOURCE_DIR)include)

#------------------------------------------------------------
#-----------------Create Binary for MPI----------------------
#------------------------------------------------------------

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(rtMPI src/mpi/Main.cpp src/Resources.cpp src/mpi/RayTracer.cpp src/Output.cpp src/Scene.cpp src/Camera.cpp)
target_link_libraries(rtMPI ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(rtMPI PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(rtMPI PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

#------------------------------------------------------------
#---------------Create Binary for Serial---------------------
#------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-g -O3 -Wall -MMD -MP -std=c++11")

add_executable(rtSerial src/serial/Main.cpp src/Resources.cpp src/serial/RayTracer.cpp src/Output.cpp src/Scene.cpp src/Camera.cpp)
target_link_libraries(rtMPI m)

#------------------------------------------------------------
#---------------Create Binary for OpenMP---------------------
#------------------------------------------------------------

set(CMAKE_CXX_FLAGS "-g -O3 -Wall -MMD -MP -fopenmp -std=c++11")
add_executable(rtOpenMP src/openmp/Main.cpp src/Resources.cpp src/openmp/RayTracer.cpp src/Output.cpp src/Scene.cpp src/Camera.cpp)

# Need to compile using nvcc
# add_executable(rtCuda src/cuda/Main.cpp src/cuda/cudaRT.cpp src/Math.cpp src/Output.cpp)
